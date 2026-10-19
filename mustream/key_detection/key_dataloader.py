"""
Key-detection dataset, audio loading, and the crop/window geometry shared by
each train_* script and its evaluator. Returns raw waveforms; the backbone runs
in the training loop.

Self-test: python key_dataloader.py [split_dir] [audio_dir] [annotation_dir]
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from key_augment import draw_pitch_shift, resolve_pitch_shifts
from key_utils import key_name_to_index, load_txt, read_key_annotation

# MERT's conv feature extractor (valid convolutions).
_MERT_CONV_KERNELS = (10, 3, 3, 3, 3, 2, 2)
_MERT_CONV_STRIDES = (5, 2, 2, 2, 2, 2, 2)


def mert_output_len(n_samples: int, kernels=_MERT_CONV_KERNELS, strides=_MERT_CONV_STRIDES) -> int:
    """Output frames of a valid-convolution stack for n_samples input."""
    L = n_samples
    for k, s in zip(kernels, strides):
        L = (L - k) // s + 1
    return L


def mert_input_for_output(n_frames: int, kernels=_MERT_CONV_KERNELS, strides=_MERT_CONV_STRIDES) -> int:
    """Minimum input samples for a valid-convolution stack to produce n_frames."""
    L = n_frames
    for k, s in zip(reversed(kernels), reversed(strides)):
        L = (L - 1) * s + k
    return L


def kernels_strides_for_backbone(backbone: str):
    """Conv (kernels, strides) of the given backbone ('small_bidirectional' differs from MERT)."""
    if backbone == "small_bidirectional":
        # Repo root appended (not prepended) so it never shadows this task's modules.
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from modeling.mert.small_mert import _CONV_KERNEL, _CONV_STRIDE
        return _CONV_KERNEL, _CONV_STRIDE
    return _MERT_CONV_KERNELS, _MERT_CONV_STRIDES


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: Path, sr: int) -> torch.Tensor:
    """Load an audio file as mono, resampled to sr. Returns [1, n_samples]."""
    waveform, orig_sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    return waveform


# Encodings that support sample-exact seeking.
_SAMPLE_EXACT_SEEK_ENCODINGS = frozenset({'PCM_S', 'PCM_U', 'PCM_F', 'FLAC'})


def _slice_crop(waveform: torch.Tensor, crop_start_samples: int, crop_samples: int) -> torch.Tensor:
    """Exactly crop_samples from crop_start_samples, zero-padded at the end."""
    crop = waveform[:, crop_start_samples : crop_start_samples + crop_samples]
    missing = crop_samples - crop.shape[1]
    if missing > 0:
        crop = torch.nn.functional.pad(crop, (0, missing))
    return crop


def load_audio_crop(path: Path, sr: int, crop_start_samples: int, crop_samples: int) -> torch.Tensor:
    """Mono crop at sr, [1, crop_samples], zero-padded if the file ends first.

    Seeks directly when the file is already at sr (e.g. 24 kHz wav); otherwise
    decodes and resamples the whole file, then crops.
    """
    info = torchaudio.info(path)
    if info.sample_rate != sr or info.encoding not in _SAMPLE_EXACT_SEEK_ENCODINGS:
        return _slice_crop(load_audio(path, sr), crop_start_samples, crop_samples)

    # Past EOF returns empty; _slice_crop zero-pads.
    waveform, _ = torchaudio.load(path, frame_offset=crop_start_samples, num_frames=crop_samples)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return _slice_crop(waveform, 0, crop_samples)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class KeyDataset(Dataset):
    """
    Args:
        audio_dir:       directory of `<track_id>.<audio_ext>` audio files
        annotation_dir:  directory of `<track_id>.key` annotations (one global key);
                         track ids may contain '/' (e.g. `<corpus>/<stem>`)
        split_dir:       directory holding `{split}.txt` track-id lists
        split:           'train' | 'val' | 'test'
        mode:            'train' -> a crop + key + drawn pitch shift
                         'test'  -> the whole track + key + track id (batch_size=1)
        sr:              audio sample rate fed to the backbone (24000 for MERT)
        crop_samples:    exact crop length in samples (mode='train'); use the geometry's value
        crop_duration:   fallback crop length in seconds if crop_samples is None
        crop_mode:       'random' (training) or 'center' (fixed, for val)
        crop_align:      round crop starts down to a multiple of this (hop_size * pool_factor); 1 = off
        max_duration:    mode='test' only -- cap the returned excerpt at this many
                         seconds (None = whole file)
        audio_ext:       audio extension, or candidates tried in order per track
        pitch_shifts:    mode='train' only -- semitone shifts to draw from, or None.
                         Applied later by key_augment.apply_pitch_shift_batch.
    """

    SPLITS = ("train", "val", "test")
    MODES = ("train", "test")
    CROP_MODES = ("random", "center")

    def __init__(
        self,
        audio_dir: Union[str, Path],
        annotation_dir: Union[str, Path],
        split_dir: Union[str, Path],
        split: str = "train",
        mode: str = "train",
        sr: int = 24000,
        crop_samples: Optional[int] = None,
        crop_duration: float = 10.0,
        crop_mode: str = "random",
        crop_align: int = 1,
        max_duration: Optional[float] = None,
        audio_ext: Union[str, Sequence[str]] = "wav",
        pitch_shifts: Optional[Sequence[int]] = None,
    ):
        assert split in self.SPLITS, f"split must be one of {self.SPLITS}, got {split!r}"
        assert mode in self.MODES, f"mode must be one of {self.MODES}, got {mode!r}"
        assert crop_mode in self.CROP_MODES, f"crop_mode must be one of {self.CROP_MODES}, got {crop_mode!r}"

        self.audio_dir = Path(audio_dir)
        self.annotation_dir = Path(annotation_dir)
        self.split = split
        self.mode = mode
        self.sr = sr
        self.crop_samples = crop_samples if crop_samples is not None else int(crop_duration * sr)
        self.crop_mode = crop_mode
        self.crop_align = max(1, int(crop_align))
        self.max_samples = None if max_duration is None else int(max_duration * sr)
        self.audio_exts = (audio_ext,) if isinstance(audio_ext, str) else tuple(audio_ext)
        # Augmentation only in mode='train'.
        self.pitch_shifts = tuple(pitch_shifts) if (pitch_shifts and mode == "train") else None

        split_file = Path(split_dir) / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"no split file {split_file} -- create {{train,val,test}}.txt there "
                "(one track id per line; see the README's Data section)"
            )

        # Read every label once; unparsable keys or missing audio raise.
        self.track_ids, self.labels, self.audio_paths, self.n_samples = [], [], [], []
        for track_id in load_txt(split_file):
            key_name = read_key_annotation(self.annotation_dir / f"{track_id}.key")
            if key_name is None:
                raise ValueError(
                    f"{track_id}: annotation is not a single unambiguous key -- remove it "
                    f"from {split_file}"
                )
            audio_path = self._find_audio(track_id)
            if audio_path is None:
                raise FileNotFoundError(
                    f"{track_id}: no audio file under {self.audio_dir} with extension "
                    f"{self.audio_exts} -- rebuild {split_file} against the root you are training on"
                )
            info = torchaudio.info(audio_path)
            self.track_ids.append(track_id)
            self.labels.append(key_name_to_index(key_name))
            self.audio_paths.append(audio_path)
            # Track length at self.sr (from the header).
            self.n_samples.append(int(round(info.num_frames * self.sr / info.sample_rate)))

    def _find_audio(self, track_id: str) -> Optional[Path]:
        for ext in self.audio_exts:
            path = self.audio_dir / f"{track_id}.{ext}"
            if path.exists():
                return path
        return None

    def __len__(self) -> int:
        return len(self.track_ids)

    def _crop_start(self, idx: int) -> int:
        """Crop start in samples (random or center), aligned down to crop_align."""
        span = self.n_samples[idx] - self.crop_samples
        if span <= 0:
            return 0
        if self.crop_mode == "center":
            start = span // 2
        else:
            start = int(torch.randint(span + 1, (1,)).item())
        return (start // self.crop_align) * self.crop_align

    def __getitem__(self, idx: int):
        """
        mode='train':  (audio [1, crop_samples], key_index, n_steps)
        mode='test':   (audio [1, n_samples], key_index, track_id)

        key_index is the untransposed key; n_steps is the drawn pitch shift (0 if off).
        """
        label = self.labels[idx]
        if self.mode == "train":
            audio = load_audio_crop(
                self.audio_paths[idx], self.sr, self._crop_start(idx), self.crop_samples)
            return audio, label, draw_pitch_shift(self.pitch_shifts)

        audio = load_audio(self.audio_paths[idx], self.sr)
        if self.max_samples is not None:
            audio = audio[:, : self.max_samples]
        return audio, label, self.track_ids[idx]


# ---------------------------------------------------------------------------
# Crop / window geometry
# ---------------------------------------------------------------------------
# Sample/frame arithmetic per setup, shared by training and evaluation.


@dataclass(frozen=True)
class CropGeometry:
    """Fields shared by every setup."""
    sample_rate: int
    hop_size: int
    pool_factor: int
    label_rate: float     # pooled frames per second
    n_frames: int         # pooled frames the head sees per training crop
    crop_samples: int
    crop_align: int

    @property
    def frame_duration(self) -> float:
        """Seconds of audio per pooled frame."""
        return self.hop_size * self.pool_factor / self.sample_rate


@dataclass(frozen=True)
class OfflineGeometry(CropGeometry):
    """Whole crop through the backbone in one pass."""

    def describe(self) -> str:
        return (
            f"Crop       : {self.n_frames / self.label_rate:g}s ({self.crop_samples} samples, "
            f"{self.n_frames} pooled frames at {self.label_rate:g} fps), one backbone pass"
        )


@dataclass(frozen=True)
class ChunkedGeometry(CropGeometry):
    n_frames_per_chunk: int
    chunk_samples: int    # backbone input length for one chunk
    chunk_stride: int     # frame-aligned spacing between chunk starts
    n_chunks: int         # chunks in one training crop

    def describe(self) -> str:
        return (
            f"Chunk      : {self.n_frames_per_chunk / self.label_rate:g}s "
            f"({self.chunk_samples} samples, {self.n_frames_per_chunk} frames)\n"
            f"Chunks/crop: {self.n_chunks} (covering {self.n_chunks * self.n_frames_per_chunk}/"
            f"{self.n_frames} pooled frames of the {self.n_frames / self.label_rate:g}s crop)"
        )


@dataclass(frozen=True)
class SlidingGeometry(CropGeometry):
    n_frames_per_hop: int
    n_frames_per_context: int
    hop_samples: int
    context_samples: int
    pad_left: int         # zeros before the first hop (the buffer starts empty)
    n_hops: int

    def describe(self) -> str:
        return (
            f"Hop        : {self.n_frames_per_hop / self.label_rate:g}s "
            f"({self.hop_samples} samples, {self.n_frames_per_hop} frames)\n"
            f"Context    : {self.n_frames_per_context / self.label_rate:g}s "
            f"({self.context_samples} samples, {self.n_frames_per_context} frames)\n"
            f"Hops/crop  : {self.n_hops} (covering {self.n_hops * self.n_frames_per_hop}/"
            f"{self.n_frames} pooled frames of the {self.n_frames / self.label_rate:g}s crop)"
        )


@dataclass(frozen=True)
class StreamingGeometry(CropGeometry):
    target_raw_frames: int   # raw (pre-pool) frames the backbone is trimmed to

    def describe(self) -> str:
        return (
            f"Crop       : {self.n_frames / self.label_rate:g}s ({self.crop_samples} samples, "
            f"{self.target_raw_frames} raw frames, {self.n_frames} pooled frames at "
            f"{self.label_rate:g} fps)"
        )


def _base_geometry(config: dict) -> Tuple[int, int, int, float, int]:
    sample_rate = config["sample_rate"]
    hop_size = config["hop_size"]
    pool_factor = config["pool_factor"]
    label_rate = sample_rate / hop_size / pool_factor
    n_frames = round(config["crop_duration"] * label_rate)
    return sample_rate, hop_size, pool_factor, label_rate, n_frames


def _frames_from_duration(duration: float, label_rate: float, name: str, floor: bool) -> int:
    """Pooled frames in `duration` seconds; warns if not a whole number."""
    exact = duration * label_rate
    frames = int(exact) if floor else round(exact)
    if frames < 1:
        raise ValueError(
            f"{name}={duration}s is too short: need at least {1 / label_rate:.4f}s "
            f"for one pooled frame at this label rate ({label_rate:g} fps)")
    if abs(exact - round(exact)) > 0.01:
        print(f"Warning: {name}={duration}s x label_rate={label_rate:g} Hz = {exact:.2f} "
              f"(not an integer); using {frames} frames and dropping the remainder.")
    return frames


def offline_geometry(config: dict) -> OfflineGeometry:
    """Geometry for train_offline.py / infer.OfflineKeyExtractor."""
    sample_rate, hop_size, pool_factor, label_rate, n_frames = _base_geometry(config)
    kernels, strides = kernels_strides_for_backbone(config.get("backbone", "pretrained"))
    crop_samples = mert_input_for_output(n_frames * pool_factor, kernels=kernels, strides=strides)
    assert mert_output_len(crop_samples, kernels=kernels, strides=strides) == n_frames * pool_factor
    return OfflineGeometry(
        sample_rate=sample_rate, hop_size=hop_size, pool_factor=pool_factor,
        label_rate=label_rate, n_frames=n_frames,
        crop_samples=crop_samples, crop_align=hop_size * pool_factor,
    )


def chunked_geometry(config: dict) -> ChunkedGeometry:
    """Geometry for train_chunked.py / infer.ChunkedKeyExtractor."""
    sample_rate, hop_size, pool_factor, label_rate, n_frames = _base_geometry(config)
    kernels, strides = kernels_strides_for_backbone(config.get("backbone", "pretrained"))

    n_frames_per_chunk = _frames_from_duration(
        config["chunk_duration"], label_rate, "chunk_duration", floor=True)
    chunk_samples = mert_input_for_output(
        n_frames_per_chunk * pool_factor, kernels=kernels, strides=strides)
    assert mert_output_len(chunk_samples, kernels=kernels, strides=strides) == \
        n_frames_per_chunk * pool_factor
    chunk_stride = n_frames_per_chunk * pool_factor * hop_size
    n_chunks = max(1, n_frames // n_frames_per_chunk)
    return ChunkedGeometry(
        sample_rate=sample_rate, hop_size=hop_size, pool_factor=pool_factor,
        label_rate=label_rate, n_frames=n_frames,
        crop_samples=(n_chunks - 1) * chunk_stride + chunk_samples,
        crop_align=hop_size * pool_factor,
        n_frames_per_chunk=n_frames_per_chunk, chunk_samples=chunk_samples,
        chunk_stride=chunk_stride, n_chunks=n_chunks,
    )


def sliding_geometry(config: dict) -> SlidingGeometry:
    """Geometry for train_sliding.py / infer.SlidingKeyExtractor."""
    sample_rate, hop_size, pool_factor, label_rate, n_frames = _base_geometry(config)
    kernels, strides = kernels_strides_for_backbone(config.get("backbone", "pretrained"))

    hop_duration = config["hop_duration"]
    context_duration = config["context_duration"]
    if context_duration < hop_duration:
        raise ValueError(
            f"context_duration={context_duration}s must be >= hop_duration={hop_duration}s")

    n_frames_per_hop = _frames_from_duration(hop_duration, label_rate, "hop_duration", floor=False)
    n_frames_per_context = _frames_from_duration(
        context_duration, label_rate, "context_duration", floor=False)
    # Sizes chosen so windows and hops land exactly on pooled frame boundaries.
    context_samples = mert_input_for_output(
        n_frames_per_context * pool_factor, kernels=kernels, strides=strides)
    assert mert_output_len(context_samples, kernels=kernels, strides=strides) == \
        n_frames_per_context * pool_factor
    hop_samples = n_frames_per_hop * pool_factor * hop_size
    n_hops = max(1, n_frames // n_frames_per_hop)
    return SlidingGeometry(
        sample_rate=sample_rate, hop_size=hop_size, pool_factor=pool_factor,
        label_rate=label_rate, n_frames=n_frames,
        crop_samples=n_hops * hop_samples, crop_align=hop_size * pool_factor,
        n_frames_per_hop=n_frames_per_hop, n_frames_per_context=n_frames_per_context,
        hop_samples=hop_samples, context_samples=context_samples,
        pad_left=context_samples - hop_samples, n_hops=n_hops,
    )


def streaming_geometry(config: dict) -> StreamingGeometry:
    """Geometry for train_streaming.py / infer_streaming.py (crop_samples is a multiple of hop_size)."""
    sample_rate, hop_size, pool_factor, label_rate, n_frames = _base_geometry(config)
    target_raw_frames = n_frames * pool_factor
    return StreamingGeometry(
        sample_rate=sample_rate, hop_size=hop_size, pool_factor=pool_factor,
        label_rate=label_rate, n_frames=n_frames,
        crop_samples=target_raw_frames * hop_size, crop_align=hop_size * pool_factor,
        target_raw_frames=target_raw_frames,
    )


def streaming_readout_schedule(
    n_frames: int,
    pool_factor: int,
    lookahead_block: int = 1,
    total_raw_frames: Optional[int] = None,
):
    """Pooled frames to read out from a streaming backbone, and the raw frames each has consumed.

    lookahead_block == 1 (strictly causal): every frame, frame i has consumed
    (i+1)*pool_factor raw frames. lookahead_block = C > 1 (block-causal): one
    readout per C-frame block, stamped at the block's end; incomplete trailing
    blocks beyond total_raw_frames are dropped.

    Returns (indices, consumed_raw_frames), both 1-D int arrays.
    """
    assert n_frames > 0, n_frames
    C = max(1, int(lookahead_block))

    idx = np.arange(n_frames)
    last_raw = (idx + 1) * pool_factor - 1        # last raw frame pooled into frame i
    block_end = (last_raw // C + 1) * C           # raw frames consumed, block-causal

    if total_raw_frames is not None:
        complete = block_end <= total_raw_frames
        if complete.any():
            idx, block_end = idx[complete], block_end[complete]
        else:
            idx, block_end = idx[-1:], np.array([total_raw_frames])

    # One readout per distinct block end.
    keep = np.flatnonzero(np.diff(block_end, append=block_end[-1] + 1) > 0)
    return idx[keep], block_end[keep]


def build_datasets(config: dict, geometry: CropGeometry):
    """(train, val) KeyDatasets; val uses a fixed center crop and no augmentation."""
    kwargs = dict(
        audio_dir=config["audio_dir"],
        annotation_dir=config["annotation_dir"],
        split_dir=config["split_dir"],
        sr=config["sample_rate"],
        crop_samples=geometry.crop_samples,
        crop_align=geometry.crop_align,
        audio_ext=config.get("audio_ext", "wav"),
    )
    train_dataset = KeyDataset(
        split="train", mode="train", crop_mode="random",
        pitch_shifts=resolve_pitch_shifts(config), **kwargs)
    val_dataset = KeyDataset(split="val", mode="train", crop_mode="center", **kwargs)
    return train_dataset, val_dataset


def build_eval_dataset(config: dict, split: str = "test") -> "KeyDataset":
    """Whole-track (mode='test') dataset for evaluation."""
    return KeyDataset(
        audio_dir=config["audio_dir"],
        annotation_dir=config["annotation_dir"],
        split_dir=config["split_dir"],
        split=split,
        mode="test",
        sr=config["sample_rate"],
        max_duration=config.get("max_duration"),
        audio_ext=config.get("audio_ext", "wav"),
    )


if __name__ == "__main__":
    # Readout-schedule self-test, then a smoke test over the real splits.
    import sys as _sys

    # Strictly causal (C=1).
    idx, consumed = streaming_readout_schedule(6, pool_factor=5, lookahead_block=1)
    assert idx.tolist() == [0, 1, 2, 3, 4, 5], idx
    assert consumed.tolist() == [5, 10, 15, 20, 25, 30], consumed

    # Block-causal C=15, pool_factor 5: one readout per block.
    idx, consumed = streaming_readout_schedule(9, pool_factor=5, lookahead_block=15)
    assert idx.tolist() == [2, 5, 8], idx
    assert consumed.tolist() == [15, 30, 45], consumed

    # C not a multiple of pool_factor; incomplete trailing block dropped.
    idx, consumed = streaming_readout_schedule(4, pool_factor=5, lookahead_block=8)
    assert idx.tolist() == [0, 2, 3], f"{idx} {consumed}"
    assert consumed.tolist() == [8, 16, 24], consumed
    idx, consumed = streaming_readout_schedule(
        4, pool_factor=5, lookahead_block=8, total_raw_frames=20)
    assert consumed.tolist() == [8, 16], consumed           # the block ending at 24 > 20 is dropped
    assert idx.tolist() == [0, 2], idx

    # A track shorter than one block keeps its last frame, charged the real audio.
    idx, consumed = streaming_readout_schedule(
        1, pool_factor=5, lookahead_block=15, total_raw_frames=5)
    assert idx.tolist() == [0] and consumed.tolist() == [5], (idx, consumed)

    # Times must be strictly increasing.
    for C in (1, 3, 8, 15, 38):
        _, consumed = streaming_readout_schedule(1800, 5, C, total_raw_frames=9000)
        assert np.all(np.diff(consumed) > 0), C
        assert consumed[-1] <= 9000, (C, consumed[-1])
    print("streaming_readout_schedule self-test OK")

    split_dir = _sys.argv[1] if len(_sys.argv) > 1 else "data_splits/splits"
    audio_dir = _sys.argv[2] if len(_sys.argv) > 2 else "path/to/audio"
    annotation_dir = _sys.argv[3] if len(_sys.argv) > 3 else "path/to/annotations"

    from key_utils import KEY_CLASSES

    kernels, strides = kernels_strides_for_backbone("pretrained")
    crop_samples = mert_input_for_output(750, kernels=kernels, strides=strides)   # 10 s at 75 fps
    for split in ("train", "val", "test"):
        dataset = KeyDataset(
            audio_dir=audio_dir, annotation_dir=annotation_dir, split_dir=split_dir,
            split=split, mode="train", crop_samples=crop_samples, crop_align=320 * 5,
            crop_mode="random" if split == "train" else "center",
        )
        audio, label, n_steps = dataset[0]
        print(f"{split:5s} n={len(dataset):4d}  crop={tuple(audio.shape)}  "
              f"label={label} ({KEY_CLASSES[label]})  shift={n_steps:+d}  {dataset.track_ids[0]}")
        assert audio.shape == (1, crop_samples)
        assert mert_output_len(crop_samples, kernels=kernels, strides=strides) == 750

    # Train split draws a pitch shift per item.
    augmented = KeyDataset(
        audio_dir=audio_dir, annotation_dir=annotation_dir, split_dir=split_dir,
        split="train", mode="train", crop_samples=crop_samples, crop_align=320 * 5,
        pitch_shifts=tuple(range(-4, 8)),
    )
    drawn = {augmented[0][2] for _ in range(60)}
    assert drawn <= set(range(-4, 8)) and len(drawn) > 1, drawn
    print(f"train augmented: drew {len(drawn)} distinct shifts in "
          f"[{min(drawn):+d}, {max(drawn):+d}] over 60 reads")

    test_dataset = KeyDataset(
        audio_dir=audio_dir, annotation_dir=annotation_dir, split_dir=split_dir,
        split="test", mode="test",
    )
    audio, label, track_id = test_dataset[0]
    print(f"test  whole track: {tuple(audio.shape)} = {audio.shape[1] / 24000:.1f}s  "
          f"{track_id}  {KEY_CLASSES[label]}")
    print("key_dataloader smoke test OK")
