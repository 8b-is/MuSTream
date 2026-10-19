"""Chord recognition dataset: random audio crops with per-window chord labels.

Labels are windowed at a fixed rate (majority chord per window, "N" if empty).
A pitch shift is drawn per item but applied later by chord_augment.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import time

import mir_eval
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

# Imported both bare and as a package, so put this directory on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chord_augment import draw_pitch_shift                    # noqa: E402
import pickle

NO_CHORD = "N"

# --- multi-dataset ----
# Fold a 7th-degree extension into the quality ("G:maj(b7)" -> "G:7"); other
# extensions and bass inversions are dropped.
_SEVENTH_QUALITY = {
    'maj': {None: 'maj', 'b7': '7', '7': 'maj7'},
    'min': {None: 'min', 'b7': 'min7', '7': 'minmaj7'},
    'dim': {None: 'dim', 'b7': 'hdim7', 'bb7': 'dim7'},
    'aug': {None: 'aug'},
    'sus2': {None: 'sus2'},
    'sus4': {None: 'sus4'},
}

# --- POP909 ---
_ALREADY_FUSED_QUALITIES = {'maj7', 'min7', '7', 'dim7', 'hdim7', 'minmaj7'}
_SHARP_ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Qualities not in the vocabulary, folded onto their nearest neighbour.
# Keep in sync with chord_vocab.QUALITIES.
_COLLAPSE_QUALITY = {
    'minmaj7': 'min',
    'sus2': 'maj',
}


def reduce_chord_label(chord_label: str) -> str:
    """Collapse a chord label to a root:quality class of chord_vocab (or NO_CHORD)."""
    if chord_label == mir_eval.chord.NO_CHORD or 'X' in chord_label:
        return NO_CHORD
    root, quality, scale_degrees, _bass = mir_eval.chord.split(chord_label)
    root = _SHARP_ROOTS[mir_eval.chord.pitch_class_to_semitone(root)]

    if quality in _ALREADY_FUSED_QUALITIES:
        return f'{root}:{_COLLAPSE_QUALITY.get(quality, quality)}'

    quality_map = _SEVENTH_QUALITY.get(quality)
    if quality_map is None:
        return NO_CHORD  # unsupported base triad (e.g. bare '1'/'5' power chords)
    seventh = next((d for d in ('b7', '7', 'bb7') if d in scale_degrees), None)
    reduced = quality_map.get(seventh, quality_map[None])
    return f'{root}:{_COLLAPSE_QUALITY.get(reduced, reduced)}'

# MERT's CNN uses valid convolutions: N*24000 samples -> N*75 - 1 frames.
_MERT_CONV_KERNELS = (10, 3, 3, 3, 3, 2, 2)  # from config.json
_MERT_CONV_STRIDES = (5,  2, 2, 2, 2, 2, 2)


def mert_output_len(n_samples: int, kernels=_MERT_CONV_KERNELS, strides=_MERT_CONV_STRIDES) -> int:
    """Frames a valid-convolution CNN stack produces for n_samples input."""
    L = n_samples
    for k, s in zip(kernels, strides):
        L = (L - k) // s + 1
    return L


def mert_input_for_output(n_frames: int, kernels=_MERT_CONV_KERNELS, strides=_MERT_CONV_STRIDES) -> int:
    """Minimum input samples for a valid-convolution CNN stack to produce exactly n_frames."""
    L = n_frames
    for k, s in zip(reversed(kernels), reversed(strides)):
        L = (L - 1) * s + k
    return L


def kernels_strides_for_backbone(backbone: str):
    """Conv (kernels, strides) of the backbone: MERT's, or small_mert's for 'small_bidirectional'."""
    if backbone == "small_bidirectional":
        # Repo root appended (not prepended) so it doesn't shadow this task's modules.
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from modeling.mert.small_mert import _CONV_KERNEL, _CONV_STRIDE
        return _CONV_KERNEL, _CONV_STRIDE
    return _MERT_CONV_KERNELS, _MERT_CONV_STRIDES


def read_label_file(path: Path) -> List[Tuple[float, float, str]]:
    """Return list of (begin, end, chord) from a .lab file."""
    entries = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts:
                begin, end, chord = parts
                entries.append((float(begin), float(end), chord))
    return entries


def preprocess_labels(
    entries: List[Tuple[float, float, str]],
    total_duration: float,
    window: float,
) -> List[Tuple[float, float, str]]:
    """
    Split (begin, end, chord) entries into fixed-size windows.

    For each window [t, t+window], the chord with the longest overlap wins.
    Returns a list of (win_start, win_end, chord).
    """
    win_starts = np.arange(0, total_duration, window).round(10).tolist()

    windowed = []
    for ws in win_starts:
        we = ws + window
        best_chord = NO_CHORD
        best_overlap = 0.0
        for begin, end, chord in entries:
            if begin > we or end < ws:
                continue

            overlap = max(0.0, min(we, end) - max(ws, begin))
            if overlap > best_overlap:
                best_overlap = overlap
                best_chord = chord
        windowed.append((ws, we, best_chord))

    return windowed


def load_audio(path: Path, sr: int) -> torch.Tensor:
    """Load an audio file as mono, resampled to sr. Returns [1, n_samples]."""
    waveform, orig_sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    return waveform


# Encodings that can be seeked sample-exactly (others are decoded whole).
_SAMPLE_EXACT_SEEK_ENCODINGS = frozenset({'PCM_S', 'PCM_U', 'PCM_F', 'FLAC'})


def _slice_crop(waveform: torch.Tensor, crop_start_samples: int, crop_samples: int) -> torch.Tensor:
    """Return exactly crop_samples from crop_start_samples, zero-padded at the end."""
    crop = waveform[:, crop_start_samples : crop_start_samples + crop_samples]
    missing = crop_samples - crop.shape[1]
    if missing > 0:
        crop = torch.nn.functional.pad(crop, (0, missing))
    return crop


def load_audio_crop(
    path: Path,
    sr: int,
    crop_start_samples: int,
    crop_samples: int,
) -> torch.Tensor:
    """Load [1, crop_samples] mono audio at sr from crop_start_samples, zero-padded.

    Reads only the crop when the file is already at sr in a seekable PCM/FLAC
    format; otherwise decodes and resamples the whole file.
    """
    info = torchaudio.info(path)
    if info.sample_rate != sr or info.encoding not in _SAMPLE_EXACT_SEEK_ENCODINGS:
        return _slice_crop(load_audio(path, sr), crop_start_samples, crop_samples)

    # Past EOF this returns an empty tensor, which _slice_crop pads.
    waveform, _ = torchaudio.load(
        path, frame_offset=crop_start_samples, num_frames=crop_samples)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return _slice_crop(waveform, 0, crop_samples)


def pick_crop_index(
    windowed: List[Tuple[float, float, str]],
    n_windows: int,
    start_idx: int = None,
) -> int:
    """Start window of a crop with n_windows complete labels (random unless start_idx is set)."""
    n_valid = len(windowed) - n_windows + 1
    if n_valid <= 0:
        return 0
    if start_idx is None or start_idx >= n_valid:
        return torch.randint(n_valid, (1,)).item()
    return start_idx


def crop_labels(
    windowed: List[Tuple[float, float, str]],
    idx: int,
    n_windows: int,
    chord2idx: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return n_windows labels from window idx and a validity mask (NO_CHORD padding is masked)."""
    label_slice = windowed[idx : idx + n_windows]
    n_real = len(label_slice)
    chords = [chord for _, _, chord in label_slice]
    chords += [NO_CHORD] * (n_windows - n_real)

    label_tensor = torch.tensor([chord2idx[c] for c in chords], dtype=torch.long)

    # Mask: True for real label windows, False for zero-padded positions
    mask = torch.zeros(n_windows, dtype=torch.bool)
    mask[:n_real] = True
    return label_tensor, mask


def crop_audio_and_labels(
    waveform: torch.Tensor,
    windowed: List[Tuple[float, float, str]],
    n_windows: int,
    crop_samples: int,
    sr: int,
    chord2idx: dict,
    start_idx: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop a loaded waveform at a label-window boundary; return (audio, labels, mask)."""
    idx = pick_crop_index(windowed, n_windows, start_idx)
    crop_start_sec = windowed[idx][0] if windowed else 0.0
    audio_crop = _slice_crop(waveform, int(crop_start_sec * sr), crop_samples)
    label_tensor, mask = crop_labels(windowed, idx, n_windows, chord2idx)
    return audio_crop, label_tensor, mask


class MultiDatasetChordDataset(Dataset):
    """Chord dataset over a split directory of record pickles.

    Args:
        split_dir:      directory with {split}.pkl, a list of dicts with
                        'audio_path' and 'chord_label_path' (.lab)
        vocab_file:     (chord2idx, idx2chord) pickle
        split:          "train", "val" or "test"
        window:         label window size in seconds
        sr:             target sample rate
        crop_duration:  crop length in seconds
        pitch_shifts:   train only: semitone shifts to draw from, or None
    """

    def __init__(
        self,
        split_dir: Path,
        vocab_file: Path,
        split: str = "train",
        window: float = 1 / 15,
        sr: int = 24000,
        crop_duration: float = 30.0,
        crop_samples: int = None,
        start_idx=None,
        pitch_shifts=None,
    ):
        assert split in ("train", "val", "test"), f"split must be 'train', 'val' or 'test', got {split!r}"
        self.split = split
        self.sr = sr
        self.crop_samples = crop_samples if crop_samples is not None else int(crop_duration * sr)
        self.start_idx = start_idx
        self.window = window
        self.n_windows = round(crop_duration / window)
        # Augment the training split only.
        self.pitch_shifts = tuple(pitch_shifts) if (pitch_shifts and split == "train") else None

        self.records = pickle.load(open(Path(split_dir) / f"{split}.pkl", 'rb'))
        self.chord2idx, self.idx2chord = pickle.load(open(Path(vocab_file), 'rb'))
        self.build_dataset()

    def build_dataset(self):
        self.windowed_labels = []
        self.reduced_labels = []
        self.ori_labels = []
        for record in tqdm(self.records):
            entries = read_label_file(record['chord_label_path'])
            reduced_entries = [(b, e, reduce_chord_label(c)) for b, e, c in entries]
            total_duration = reduced_entries[-1][1] if reduced_entries else 0.0
            windowed = preprocess_labels(reduced_entries, total_duration, self.window)
            self.windowed_labels.append(windowed)
            self.reduced_labels.append(reduced_entries)
            self.ori_labels.append(entries)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (audio [1, crop_samples], labels [n_windows], mask [n_windows], n_steps).

        labels are un-transposed; n_steps is the drawn shift (0 when off).
        """
        record = self.records[idx]
        windowed = self.windowed_labels[idx]

        # Pick the crop from the labels first so only that crop is read.
        window_idx = pick_crop_index(windowed, self.n_windows, self.start_idx)
        crop_start_sec = windowed[window_idx][0] if windowed else 0.0
        audio_crop = load_audio_crop(
            record['audio_path'], self.sr, int(crop_start_sec * self.sr), self.crop_samples,
        )
        label_tensor, mask = crop_labels(
            windowed, window_idx, self.n_windows, self.chord2idx)
        return audio_crop, label_tensor, mask, draw_pitch_shift(self.pitch_shifts)


if __name__ == "__main__":
    import sys

    split_dir = sys.argv[1] if len(sys.argv) > 1 else 'data_splits/splits'
    vocab_file = sys.argv[2] if len(sys.argv) > 2 else 'data_splits/dictionary_121.pkl'

    print(f"--- {split_dir} (vocab {vocab_file}) ---")
    dataset = MultiDatasetChordDataset(
        split_dir=split_dir, split='train', vocab_file=vocab_file)
    for i, (audio, label, mask, n_steps) in enumerate(dataset):
        print(f"audio={tuple(audio.shape)} labels={tuple(label.shape)} "
              f"valid={mask.sum().item()}/{mask.shape[0]} shift={n_steps:+d}")
        if i >= 4:
            break

    # Check every reduced label in every split is in the vocabulary.
    print("\nchecking reduce_chord_label against the vocabulary...")
    from collections import Counter
    seen = Counter()
    for split in ('train', 'val', 'test'):
        try:
            records = pickle.load(open(Path(split_dir) / f'{split}.pkl', 'rb'))
        except FileNotFoundError:
            print(f"  {split}: absent, skipped")
            continue
        unknown = Counter()
        for record in tqdm(records, desc=f'  {split}', leave=False):
            for _b, _e, chord in read_label_file(record['chord_label_path']):
                reduced = reduce_chord_label(chord)
                seen[reduced] += 1
                if reduced not in dataset.chord2idx:
                    unknown[reduced] += 1
        status = 'OK' if not unknown else f'{len(unknown)} UNKNOWN: {dict(unknown)}'
        print(f"  {split}: {len(records)} songs, {status}")
    print(f"  {len(seen)} distinct reduced labels, all in a {len(dataset.chord2idx)}-class vocabulary")

    # Train draws a shift per item; val/test never do.
    augmented = MultiDatasetChordDataset(
        split_dir=split_dir, split='train', vocab_file=vocab_file,
        pitch_shifts=tuple(range(-4, 8)))
    drawn = {augmented[0][3] for _ in range(60)}
    print(f"\ntrain augmented: drew {len(drawn)} distinct shifts in "
          f"[{min(drawn):+d}, {max(drawn):+d}] over 60 reads of one item")
    val = MultiDatasetChordDataset(
        split_dir=split_dir, split='val', vocab_file=vocab_file,
        pitch_shifts=tuple(range(-4, 8)))
    assert val.pitch_shifts is None and val[0][3] == 0
    print("val ignores pitch_shifts: OK")
