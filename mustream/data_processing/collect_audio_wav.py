"""Resample every audio file under a directory to mono int16 at a target sample
rate and save them into sharded HDF5 files. Re-running resumes where it stopped.

Usage (from mustream/):
    python data_processing/collect_audio_wav.py
    python data_processing/collect_audio_wav.py --root path/to/MTG-Jamendo \\
                                --out  path/to/MTG-Jamendo-h5/MTG-Jamendo \\
                                --sr   24000 --shards 256 --workers 8
"""

import argparse
import glob
import hashlib
import itertools
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import h5py
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

ROOT    = 'path/to/MTG-Jamendo'
OUT     = 'path/to/MTG-Jamendo-h5/MTG-Jamendo'
SR      = 24000
SHARDS  = 256


def _shard_index(rel_path: str, num_shards: int) -> int:
    """Deterministic shard assignment, stable across reruns regardless of file-list order."""
    digest = hashlib.md5(rel_path.encode('utf-8')).hexdigest()
    return int(digest, 16) % num_shards


def _shard_path(out_prefix: str, shard_idx: int, num_shards: int) -> str:
    width = max(len(str(num_shards - 1)), 4)
    return f"{out_prefix}.{shard_idx:0{width}d}.h5"


def _init_worker():
    """Pin each worker to a single CPU thread. Without this, torchaudio's ops
    (e.g. resample) each spawn their own OpenMP/MKL thread pool, so N worker
    processes oversubscribe the machine by N times and contend with each other."""
    torch.set_num_threads(1)


def _load_audio(args):
    """Worker: load, resample to mono, quantize to int16. Returns (rel_path, array, None) or (rel_path, None, err)."""
    full_path, root, target_sr = args
    rel_path = os.path.relpath(full_path, root)
    try:
        waveform, sr = torchaudio.load(full_path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        waveform = waveform.mean(dim=0)  # mix to mono, shape: (T,)
        waveform_i16 = np.clip(waveform.numpy() * 32768.0, -32768, 32767).astype(np.int16)
        return rel_path, waveform_i16, None
    except Exception as e:
        return rel_path, None, str(e)


def _collect_existing_keys(out_prefix, num_shards):
    """Return the set of all dataset paths (not groups) already present across shard files."""
    keys = set()
    for shard_path in glob.glob(f"{out_prefix}.*.h5"):
        with h5py.File(shard_path, 'r') as f:
            f.visititems(lambda name, obj: keys.add(name) if isinstance(obj, h5py.Dataset) else None)
    return keys


def collect(root: str, out: str, target_sr: int, num_workers: int, num_shards: int, limit: int = None):
    all_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in AUDIO_EXTS:
                all_files.append(os.path.join(dirpath, fname))
    all_files.sort()
    print(f"Found {len(all_files)} audio files under {root}")

    if limit is not None:
        all_files = all_files[:limit]

    existing = _collect_existing_keys(out, num_shards)
    if existing:
        print(f"Resuming: {len(existing)} datasets already present across shards")

    tasks = [
        (f, root, target_sr)
        for f in tqdm(all_files)
        if os.path.relpath(f, root) not in existing
    ]
    print(f"To process: {len(tasks)} files  |  workers: {num_workers}  |  "
          f"target SR: {target_sr} Hz  |  shards: {num_shards}")

    if not tasks:
        print("Nothing to do.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)

    errors = []
    written = 0

    # Workers decode+quantize audio; the main process writes to HDF5 serially per shard
    # (h5py file handles are not thread/fork-safe), one handle per shard kept open for the run.
    #
    # Tasks are submitted in a bounded sliding window rather than all at once: for large
    # datasets, eagerly building a futures dict for every task blocks (and holds memory)
    # before the progress bar can show anything, which looks like a hang.
    shard_handles = {}
    task_iter = iter(tasks)
    max_in_flight = num_workers * 4
    try:
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker) as executor:
            pending = {executor.submit(_load_audio, t) for t in itertools.islice(task_iter, max_in_flight)}
            with tqdm(total=len(tasks), desc='Collecting audio') as pbar:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        rel_path, data, err = future.result()
                        if err is not None:
                            errors.append((rel_path, err))
                        else:
                            shard_idx = _shard_index(rel_path, num_shards)
                            if shard_idx not in shard_handles:
                                shard_handles[shard_idx] = h5py.File(
                                    _shard_path(out, shard_idx, num_shards), 'a')
                            shard_handles[shard_idx].create_dataset(rel_path, data=data)
                            written += 1
                        pbar.update(1)
                    pending |= {executor.submit(_load_audio, t)
                                for t in itertools.islice(task_iter, len(done))}
    finally:
        for hf in shard_handles.values():
            hf.close()

    print(f"\nWrote {written} datasets across {len(shard_handles)} shards")
    if errors:
        print(f"Skipped {len(errors)} files due to errors:")
        for rel, msg in errors[:10]:
            print(f"  {rel}: {msg}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root',    default=ROOT,
                        help='Audio root directory')
    parser.add_argument('--out',     default=OUT,
                        help="Output path prefix; shard files are '<out>.NNNN.h5' (appended to if they exist)")
    parser.add_argument('--sr',      type=int, default=SR,
                        help='Target sample rate in Hz (default: %(default)s)')
    parser.add_argument('--workers', type=int, default=min(os.cpu_count(), 16),
                        help='Worker processes for audio decoding (default: %(default)s)')
    parser.add_argument('--shards',  type=int, default=SHARDS,
                        help='Number of HDF5 shard files to split output across (default: %(default)s)')
    parser.add_argument('--limit',   type=int, default=None,
                        help='Process at most this many files (default: no limit)')
    args = parser.parse_args()
    collect(args.root, args.out, args.sr, args.workers, args.shards, args.limit)
