"""Audio dataloader for MERT distillation: random fixed-length crops of each
track in a split, read from the HDF5 cache built by
data_processing/collect_audio_wav.py (or from the raw audio files).
"""
import hashlib
import os
import pickle
import random
import h5py
import numpy as np

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader


def _h5_shard_index(rel_path: str, num_shards: int) -> int:
    """Same md5-based shard assignment as data_processing/collect_audio_wav.py."""
    digest = hashlib.md5(rel_path.encode('utf-8')).hexdigest()
    return int(digest, 16) % num_shards


class AudioDataset(Dataset):
    """Dataset that reads a random n_samples-long crop of each track listed in a
    pre-split pkl file, from the sharded HDF5 cache or the audio files.

    Parameters
    ----------
    split_path      : path to train/val/test.pkl (list of relative audio paths)
    audio_root      : root the pkl paths are relative to (raw-audio loading only)
    audio_info_path : optional audio_info.pkl from collect_audio_info.py; saves a
                      torchaudio.info() call per item when loading raw audio
    h5_root         : prefix of the sharded HDF5 audio cache. If given, audio is
                      loaded via load_audio_h5 instead of load_audio_segment.
    h5_shards       : number of shards the HDF5 cache is split across
    duration        : approximate crop length in seconds (see n_samples)
    sample_rate     : sample rate of the returned audio
    hop_length      : MERT frame hop in samples; crop starts are aligned to it
    n_fft           : window size used to round the crop length: n_samples is
                      (n_frames - 1) * hop_length + n_fft with n_frames rounded
                      up to a multiple of 5 (240192 samples for 10 s at 24 kHz)
    """

    def __init__(
        self,
        split_path: str,
        audio_root: str = None,
        audio_info_path: str = None,
        h5_root: str = None,
        h5_shards: int = 256,
        duration: float = 10.0,
        sample_rate: int = 24000,
        hop_length: int = 320,
        n_fft: int = 512,
    ):
        self.files = pickle.load(open(split_path, 'rb'))
        self.duration = duration
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        # Crop length: n_frames rounded up to a multiple of 5
        n_frames = (int(duration * sample_rate) - n_fft) // hop_length + 1
        n_frames = ((n_frames + 4) // 5) * 5
        self.n_samples = (n_frames - 1) * hop_length + n_fft

        if audio_root is None and h5_root is None:
            raise ValueError('[error] Please provide root for audio or h5 files')
        self.load_method = 'h5' if h5_root is not None else 'audio'

        # load from audio files
        self.audio_root = audio_root
        self.file_info = {}
        self.resamplers = {}
        if audio_info_path is not None:
            self.file_info = pickle.load(open(audio_info_path, 'rb'))
            
            src_srs = {info[1] for info in self.file_info.values() if info[1] != sample_rate}
            self.resamplers = {
                sr: torchaudio.transforms.Resample(sr, sample_rate)
                for sr in src_srs
            }

        # load from h5
        self.h5_root = h5_root
        self.h5_shards = h5_shards
        
        # h5py file handles, opened lazily per worker process and cached by
        # shard index — a handle opened before the DataLoader forks its
        # workers is not safe to share across processes.
        self._h5_handles = {}

    def load_audio_segment(self, path: str, cached_info=None):
        """Load only the needed crop from disk using frame_offset.

        Works for any source SR. The crop start is chosen in target-SR space
        (aligned to hop_length), then converted to source-SR frame indices for
        the torchaudio.load call. A small buffer of source frames is added so
        the resampler filter has enough context; the result is trimmed afterward.
        """
        if cached_info is not None:
            num_frames_src, src_sr = cached_info
        else:
            info = torchaudio.info(path)
            num_frames_src, src_sr = info.num_frames, info.sample_rate

        # Estimate total duration in target-SR samples to pick a crop start
        total_target = int(num_frames_src * self.sample_rate / src_sr)
        if total_target > self.n_samples:
            max_start = total_target - self.n_samples
            max_start_aligned = (max_start // self.hop_length) * self.hop_length
            start = random.randrange(0, max_start_aligned + self.hop_length, self.hop_length)
            start = min(start, max_start)
        else:
            start = 0

        if src_sr == self.sample_rate:
            waveform, _ = torchaudio.load(
                path, frame_offset=start, num_frames=self.n_samples, channels_first=True
            )
        else:
            start_src = int(start * src_sr / self.sample_rate)
            n_src = -(-self.n_samples * src_sr // self.sample_rate)  # ceiling division
            n_src = min(n_src + 512, num_frames_src - start_src)
            waveform, _ = torchaudio.load(
                path, frame_offset=start_src, num_frames=n_src, channels_first=True
            )
            resampler = self.resamplers.get(src_sr)
            if resampler is None:
                resampler = torchaudio.transforms.Resample(src_sr, self.sample_rate)
                self.resamplers[src_sr] = resampler
            waveform = resampler(waveform)
            waveform = waveform[:, :self.n_samples]  # trim resampler overshoot

        segment = waveform.mean(dim=0)  # (T,)
        if segment.shape[0] < self.n_samples:
            segment = F.pad(segment, (0, self.n_samples - segment.shape[0]))
        return segment, start

    def load_audio_h5(self, rel_path: str):
        """Load a random segment directly from the sharded HDF5 audio cache
        at self.h5_root, keyed by rel_path. Waveforms there are already mono
        int16 at self.sample_rate, so no resampling is needed.
        """
        shard_idx = _h5_shard_index(rel_path, self.h5_shards)
        h5_file = self._h5_handles.get(shard_idx)
        if h5_file is None:
            h5_file = h5py.File(f"{self.h5_root}.{shard_idx:04d}.h5", 'r')
            self._h5_handles[shard_idx] = h5_file

        data = h5_file[rel_path]
        num_samples_src = data.shape[0]

        if num_samples_src > self.n_samples:
            max_start = num_samples_src - self.n_samples
            max_start_aligned = (max_start // self.hop_length) * self.hop_length
            start = random.randrange(0, max_start_aligned + self.hop_length, self.hop_length)
            start = min(start, max_start)
        else:
            start = 0

        waveform_i16 = data[start:start + self.n_samples]
        segment = torch.from_numpy(waveform_i16.astype(np.float32) / 32768.0)
        if segment.shape[0] < self.n_samples:
            segment = F.pad(segment, (0, self.n_samples - segment.shape[0]))

        return segment, start

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        rel_path  = self.files[idx]

        if self.load_method == 'h5':
            segment, start = self.load_audio_h5(rel_path)
        else:
            full_path = os.path.join(self.audio_root, rel_path)
            segment, start = self.load_audio_segment(full_path, self.file_info.get(rel_path))

        return {'waveform': segment}   # (n_samples,) float32 in [-1, 1]


def get_audio_dataloader(
    split_path: str,
    audio_root: str = 'path/to/MTG-Jamendo',
    audio_info_path: str = None,
    h5_root: str = None,
    h5_shards: int = 256,
    duration: float = 10.0,
    batch_size: int = 8,
    num_workers: int = 8,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    dataset = AudioDataset(
        split_path=split_path,
        audio_root=audio_root,
        audio_info_path=audio_info_path,
        h5_root=h5_root,
        h5_shards=h5_shards,
        duration=duration,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


if __name__ == "__main__":
    audio_data = get_audio_dataloader(
        h5_root='path/to/MTG-Jamendo-h5/MTG-Jamendo',
        split_path='data_splits/mtg_jamendo/train.pkl',
        batch_size=1, shuffle=False)
    for batch in audio_data:
        spect = batch['waveform']
        print(spect.shape)
        torchaudio.save("output.wav", spect, sample_rate=24000)
        break
