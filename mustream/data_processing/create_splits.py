"""Split the readable audio files under a directory (those listed in
audio_info.pkl) into train/val/test by song, and save train.pkl, val.pkl and
test.pkl as lists of relative paths.

Usage (from mustream/):
    python data_processing/create_splits.py
    python data_processing/create_splits.py --root path/to/MTG-Jamendo --out data_splits/mtg_jamendo \\
                                            --val-frac 0.05 --test-frac 0.05 --seed 42
"""

import argparse
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}

ROOT      = 'path/to/MTG-Jamendo'
OUT       = 'data_splits/mtg_jamendo'
VAL_FRAC  = 0.05
TEST_FRAC = 0.05
SEED      = 42


def song_key(rel_path: str) -> str:
    """Group key for a file: its folder if it sits 2+ levels below the root
    (stems of one song), otherwise the file itself (a full track)."""
    parent = os.path.dirname(rel_path)
    if os.path.dirname(parent):
        return parent
    return os.path.splitext(rel_path)[0]


def create_splits(root: str, out: str, audio_info_path: str,
                  val_frac: float, test_frac: float, seed: int):
    with open(audio_info_path, 'rb') as f:
        audio_info = pickle.load(f)
    print(f"Loaded {audio_info_path}: {len(audio_info)} entries")

    # Group readable audio files by song
    groups = defaultdict(list)
    n_unavailable = 0
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if Path(fname).suffix.lower() not in AUDIO_EXTS:
                continue
            rel = os.path.relpath(os.path.join(dirpath, fname), root)
            if rel not in audio_info:
                n_unavailable += 1
                continue
            groups[song_key(rel)].append(rel)

    print(f"Audio files kept              : {sum(len(v) for v in groups.values())}")
    print(f"Skipped (not in audio_info)   : {n_unavailable}")

    # Shuffle and split at the song level
    song_keys = sorted(groups)
    print(f"Unique songs                  : {len(song_keys)}")
    random.seed(seed)
    random.shuffle(song_keys)

    n = len(song_keys)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    splits = {
        'train': song_keys[:n_train],
        'val':   song_keys[n_train:n_train + n_val],
        'test':  song_keys[n_train + n_val:],
    }

    os.makedirs(out, exist_ok=True)
    for name, keys in splits.items():
        files = [f for k in keys for f in sorted(groups[k])]
        out_path = os.path.join(out, f'{name}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(files, f)
        print(f"Saved {out_path}: {len(keys)} songs, {len(files)} files")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',       default=ROOT,
                        help='Audio root directory (paths in the splits are relative to this)')
    parser.add_argument('--out',        default=OUT,
                        help='Output directory for train/val/test.pkl')
    parser.add_argument('--audio-info', default=None,
                        help='audio_info.pkl from collect_audio_info.py (default: <out>/audio_info.pkl)')
    parser.add_argument('--val-frac',   type=float, default=VAL_FRAC,
                        help='Fraction of songs for validation')
    parser.add_argument('--test-frac',  type=float, default=TEST_FRAC,
                        help='Fraction of songs for test')
    parser.add_argument('--seed',       type=int, default=SEED,
                        help='Shuffle seed')
    args = parser.parse_args()
    create_splits(args.root, args.out, args.audio_info or os.path.join(args.out, 'audio_info.pkl'),
                  args.val_frac, args.test_frac, args.seed)
