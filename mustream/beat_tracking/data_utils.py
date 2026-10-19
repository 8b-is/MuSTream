import random

import numpy as np
import torch


def save_txt(filepath, data):
    with open(filepath, "w", encoding="utf8") as f:
        for item in data:
            f.write(f"{item}\n")


def load_txt(filepath):
    with open(filepath, encoding="utf8") as f:
        return [line.strip() for line in f if line.strip()]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def quantize_events(times: np.ndarray, fps: float, length: int, widen: int = 0) -> np.ndarray:
    """Binary per-frame targets of size `length`: 1 at round(t*fps) +/- `widen` frames for each event t."""
    seq = np.zeros(length, dtype=np.float32)
    if len(times) == 0:
        return seq
    idx = np.round(np.asarray(times) * fps).astype(int)
    for offset in range(-widen, widen + 1):
        shifted = idx + offset
        shifted = shifted[(shifted >= 0) & (shifted < length)]
        seq[shifted] = 1.0
    return seq
