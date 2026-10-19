"""Beat/downbeat dataset: raw audio crops + per-frame targets (train) or whole tracks + event times (test)."""

import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from data_utils import load_txt, save_txt, quantize_events

# MERT's conv feature-extractor kernels/strides (valid convolutions).
_MERT_CONV_KERNELS = (10, 3, 3, 3, 3, 2, 2)
_MERT_CONV_STRIDES = (5, 2, 2, 2, 2, 2, 2)


def mert_output_len(n_samples: int, kernels=_MERT_CONV_KERNELS, strides=_MERT_CONV_STRIDES) -> int:
    """Number of output frames the conv stack produces for n_samples input."""
    L = n_samples
    for k, s in zip(kernels, strides):
        L = (L - k) // s + 1
    return L


def mert_input_for_output(n_frames: int, kernels=_MERT_CONV_KERNELS, strides=_MERT_CONV_STRIDES) -> int:
    """Minimum input samples for the conv stack to produce exactly n_frames."""
    L = n_frames
    for k, s in zip(reversed(kernels), reversed(strides)):
        L = (L - 1) * s + k
    return L


def kernels_strides_for_backbone(backbone: str):
    """(kernels, strides) of the conv stack used by `backbone` (MERT's, or the 26M student's)."""
    if backbone == "small_bidirectional":
        # Append mustream/ (for modeling/) without shadowing this task's modules.
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from modeling.mert.small_mert import _CONV_KERNEL, _CONV_STRIDE
        return _CONV_KERNEL, _CONV_STRIDE
    return _MERT_CONV_KERNELS, _MERT_CONV_STRIDES


def _make_splits(track_ids, split_dir: Path, train_ratio: float, val_ratio: float, seed: int):
    split_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(track_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }
    for name, split_ids in splits.items():
        save_txt(split_dir / f"{name}.txt", split_ids)


class BeatDataset(Dataset):
    """Beat/downbeat dataset.

    Args:
        audio_dir:       directory of `<track_id>.<audio_ext>` files (ids may contain '/')
        annotation_dir:  directory of `<track_id>/beat.lab` (`time  bar_position`; 1 = downbeat)
        split_dir:       directory with `{split}.txt` id lists; auto-built from
                          train_ratio/val_ratio/split_seed if missing (single corpus only)
        split:           'train' | 'val' | 'test'
        mode:            'train' -> random crop + frame targets; 'test' -> whole track +
                          event times in seconds (use batch_size=1)
        sr:              sample rate fed to MERT
        fps:             probe frame rate = sample_rate / hop_size / pool_factor
        pool_factor:     temporal pooling factor (must match extract_mert_features)
        crop_samples:    crop length in samples (train mode); use mert_input_for_output()
        target_widen:    widen each positive frame by +/- this many frames
        audio_ext:       extension, or a list tried in order per track
        kernels/strides: conv stack of the backbone (see kernels_strides_for_backbone)
    """

    def __init__(
        self,
        audio_dir: Union[str, Path],
        annotation_dir: Union[str, Path],
        split_dir: Union[str, Path],
        split: str = "train",
        mode: str = "train",
        sr: int = 24000,
        fps: float = 75.0,
        pool_factor: int = 1,
        crop_samples: Optional[int] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        split_seed: int = 42,
        target_widen: int = 1,
        audio_ext: Union[str, Sequence[str]] = "wav",
        kernels=_MERT_CONV_KERNELS,
        strides=_MERT_CONV_STRIDES,
    ):
        assert split in ("train", "val", "test")
        assert mode in ("train", "test")
        self.audio_dir = Path(audio_dir)
        self.annotation_dir = Path(annotation_dir)
        self.mode = mode
        self.sr = sr
        self.fps = fps
        self.pool_factor = pool_factor
        self.crop_samples = crop_samples
        self.target_widen = target_widen
        self.audio_exts = (audio_ext,) if isinstance(audio_ext, str) else tuple(audio_ext)
        # Conv stack crop_samples was sized for (see kernels_strides_for_backbone).
        self._kernels = kernels
        self._strides = strides

        if mode == "train" and crop_samples is None:
            raise ValueError("crop_samples is required when mode='train'")

        split_dir = Path(split_dir)
        split_file = split_dir / f"{split}.txt"
        if not split_file.exists():
            all_ids = [
                p.name for p in self.annotation_dir.iterdir()
                if p.is_dir() and self._find_audio(p.name) is not None
            ]
            if not all_ids:
                raise FileNotFoundError(
                    f"{split_file} does not exist and no <track>/beat.lab + "
                    f"<track>.{{{','.join(self.audio_exts)}}} pair was found directly under "
                    f"{self.annotation_dir}. Create {{train,val,test}}.txt in {split_dir} "
                    "(one track id per line; see the README's Data section)."
                )
            _make_splits(all_ids, split_dir, train_ratio, val_ratio, split_seed)
        self.file_ids = load_txt(split_file)

        self.build_dataset()

    def read_beat_label(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse a beat.lab file into (beat_times, downbeat_times), both in seconds."""
        try:
            arr = np.loadtxt(path, dtype=float, comments=None)
        except FileNotFoundError:
            return np.array([]), np.array([])
        if arr.size == 0:
            return np.array([]), np.array([])
        if arr.ndim == 1:
            # only beat times, no bar-position column -> no downbeat info
            return arr, np.array([])
        beats = arr[:, 0]
        downbeats = arr[arr[:, 1] == 1][:, 0]
        return beats, downbeats

    def _find_audio(self, fid: str) -> Optional[Path]:
        """First existing `audio_dir/<fid>.<ext>` over self.audio_exts, or None."""
        for ext in self.audio_exts:
            path = self.audio_dir / f"{fid}.{ext}"
            if path.exists():
                return path
        return None

    def build_dataset(self):
        # Resolve audio paths once; missing audio fails here, not mid-epoch.
        self.beats, self.downbeats, self.audio_paths = [], [], {}
        missing = []
        for fid in self.file_ids:
            beats, downbeats = self.read_beat_label(self.annotation_dir / fid / "beat.lab")
            self.beats.append(beats)
            self.downbeats.append(downbeats)
            path = self._find_audio(fid)
            if path is None:
                missing.append(fid)
            else:
                self.audio_paths[fid] = path
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} track(s) in the split have no audio under {self.audio_dir} "
                f"with extension in {self.audio_exts}: {', '.join(missing[:5])}"
                + (" ..." if len(missing) > 5 else "")
            )

    def __len__(self) -> int:
        return len(self.file_ids)

    def _load_audio(self, fid: str) -> torch.Tensor:
        waveform, orig_sr = torchaudio.load(self.audio_paths[fid])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, self.sr)
        return waveform

    def __getitem__(self, idx: int):
        fid = self.file_ids[idx]
        waveform = self._load_audio(fid)
        total_samples = waveform.shape[1]

        if self.mode == "test":
            # Whole, uncropped track + raw (unquantized) event times in seconds.
            return waveform, self.beats[idx], self.downbeats[idx], fid

        # mode == "train": fixed-length random crop + quantized per-frame targets.
        if total_samples > self.crop_samples:
            start_sample = torch.randint(total_samples - self.crop_samples + 1, (1,)).item()
        else:
            start_sample = 0
        end_sample = start_sample + self.crop_samples

        if end_sample > total_samples:
            audio_crop = torch.nn.functional.pad(
                waveform[:, start_sample:], (0, end_sample - total_samples)
            )
            n_valid_samples = total_samples - start_sample
        else:
            audio_crop = waveform[:, start_sample:end_sample]
            n_valid_samples = self.crop_samples

        T = mert_output_len(self.crop_samples, kernels=self._kernels, strides=self._strides) // self.pool_factor
        T_valid = 0
        if n_valid_samples > 0:
            raw_valid_frames = max(0, mert_output_len(n_valid_samples, kernels=self._kernels, strides=self._strides))
            T_valid = min(T, raw_valid_frames // self.pool_factor)

        start_sec = start_sample / self.sr
        end_sec = end_sample / self.sr
        beats, downbeats = self.beats[idx], self.downbeats[idx]
        beats_crop = beats[(beats >= start_sec) & (beats < end_sec)] - start_sec
        downbeats_crop = downbeats[(downbeats >= start_sec) & (downbeats < end_sec)] - start_sec

        beat_seq = quantize_events(beats_crop, self.fps, T, widen=self.target_widen)
        downbeat_seq = quantize_events(downbeats_crop, self.fps, T, widen=self.target_widen)

        mask = torch.zeros(T, dtype=torch.bool)
        mask[:T_valid] = True
        downbeat_available = torch.tensor(len(downbeats) > 0, dtype=torch.bool)

        return (
            audio_crop,
            torch.from_numpy(beat_seq),
            torch.from_numpy(downbeat_seq),
            mask,
            downbeat_available,
        )


if __name__ == "__main__":
    crop_samples = mert_input_for_output(30 * 75)
    dataset = BeatDataset(
        audio_dir="path/to/audio",
        annotation_dir="path/to/annotations",
        split_dir="data_splits/splits",
        split="train",
        crop_samples=crop_samples,
    )
    print(f"{len(dataset)} tracks")
    audio, beat, downbeat, mask, downbeat_available = dataset[0]
    print(
        "audio", audio.shape,
        "\nbeat", beat.shape, beat.sum().item(), "positive frames",
        "\ndownbeat", downbeat.shape, downbeat.sum().item(), "positive frames", downbeat[:300],
        "\nvalid frames", mask.sum().item(), "/", mask.shape[0],
        "\ndownbeat_available", downbeat_available.item(),
    )
