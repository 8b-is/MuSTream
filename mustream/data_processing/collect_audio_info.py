"""Scan an audio directory and save {rel_path: (num_frames, sample_rate)} for
every file torchaudio can read.

Usage (from mustream/):
    python data_processing/collect_audio_info.py
    python data_processing/collect_audio_info.py --root path/to/MTG-Jamendo --out data_splits/mtg_jamendo/audio_info.pkl --workers 16
"""

import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import torchaudio
from tqdm import tqdm

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

ROOT    = 'path/to/MTG-Jamendo'
OUT     = 'data_splits/mtg_jamendo/audio_info.pkl'


def _get_info(args):
    """Worker: return (rel_path, (num_frames, sample_rate), None) or (rel_path, None, err_str)."""
    full_path, root = args
    rel_path = os.path.relpath(full_path, root)
    try:
        info = torchaudio.info(full_path)
        return rel_path, (info.num_frames, info.sample_rate), None
    except Exception as e:
        return rel_path, None, str(e)


def collect(root: str, out: str, num_workers: int):
    # Gather all audio file paths first so tqdm shows total count
    all_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in AUDIO_EXTS:
                all_files.append(os.path.join(dirpath, fname))
    all_files.sort()
    print(f"Found {len(all_files)} audio files under {root}")
    print(f"Using {num_workers} worker processes")

    info_dict = {}   # rel_path -> (num_frames, sample_rate)
    errors    = []

    tasks = [(f, root) for f in all_files]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_get_info, t): t for t in tasks}
        with tqdm(total=len(all_files), desc='Reading audio info') as pbar:
            for future in as_completed(futures):
                rel_path, info_tuple, err = future.result()
                if err is not None:
                    errors.append((rel_path, err))
                else:
                    info_dict[rel_path] = info_tuple
                pbar.update(1)

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(info_dict, f)

    print(f"\nSaved {len(info_dict)} entries to {out}")
    if errors:
        print(f"Skipped {len(errors)} files due to errors:")
        for rel, msg in errors[:10]:
            print(f"  {rel}: {msg}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',    default=ROOT,
                        help='Audio root directory (paths in output are relative to this)')
    parser.add_argument('--out',     default=OUT,
                        help='Output pkl path')
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='Number of worker processes (default: all CPU cores)')
    args = parser.parse_args()
    collect(args.root, args.out, args.workers)
