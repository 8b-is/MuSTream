"""
Pitch-shift augmentation for key detection: shifting audio by n semitones
transposes its key label by n (same mode).

The shift is drawn per item in KeyDataset and applied (audio + label together)
to the whole batch on the GPU by apply_pitch_shift_batch.

Self-test: python key_augment.py
"""

import os
import sys
from typing import Optional, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Repo root appended (not prepended) so this task's models.py wins.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from audio_augment import augment_batch          # noqa: E402  (mustream/)
from key_utils import transpose_key_index         # noqa: E402

# Default inclusive shift range (12 semitones = every pitch class reachable).
DEFAULT_PITCH_SHIFT_RANGE = (-4, 7)


def resolve_pitch_shifts(config: dict) -> Optional[Tuple[int, ...]]:
    """Semitone shifts to draw from, or None if config['augment_pitch'] is off.

    config['pitch_shift_range'] is an inclusive [lo, hi] (default [-4, 7]).
    """
    if not config.get("augment_pitch", False):
        return None
    lo, hi = config.get("pitch_shift_range") or DEFAULT_PITCH_SHIFT_RANGE
    lo, hi = int(lo), int(hi)
    if hi < lo:
        raise ValueError(f"pitch_shift_range must be [lo, hi] with hi >= lo, got [{lo}, {hi}]")
    return tuple(range(lo, hi + 1))


def draw_pitch_shift(shifts: Optional[Sequence[int]]) -> int:
    """One semitone shift drawn uniformly from `shifts` (0 when it is None)."""
    if not shifts:
        return 0
    return int(shifts[torch.randint(len(shifts), (1,)).item()])


def apply_pitch_shift_batch(
    audio: torch.Tensor,          # [B, 1, T]
    labels: torch.Tensor,         # [B] key class indices
    n_steps: Optional[torch.Tensor],   # [B] semitones, or None
    out_samples: Optional[int] = None,
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pitch-shift each row by its own n_steps and transpose its key label to match.

    `audio` should already be on the GPU. None / all-zero n_steps is a no-op.
    `chunk_size` bounds peak memory of the batched shift.
    Returns (audio [B, 1, out_samples], labels [B]).
    """
    out_samples = audio.shape[-1] if out_samples is None else out_samples
    if n_steps is None:
        return audio[..., :out_samples], labels

    n_steps = n_steps.to(device=audio.device, dtype=torch.long)
    if not bool((n_steps != 0).any()):
        return audio[..., :out_samples], labels

    shifted = augment_batch(
        audio[:, 0, :], n_steps=n_steps, out_samples=out_samples, chunk_size=chunk_size)
    return shifted.unsqueeze(1), transpose_key_index(labels, n_steps.to(labels.device))


def describe(shifts: Optional[Sequence[int]]) -> str:
    """One line for a training script's startup banner."""
    if not shifts:
        return "Pitch augment           : off"
    return (f"Pitch augment           : {len(shifts)} shifts, "
            f"{min(shifts):+d}..{max(shifts):+d} semitones (label transposed to match)")


if __name__ == "__main__":
    # Self-test (no checkpoint or audio needed).
    import math

    from key_utils import KEY_CLASSES, KEY_TO_INDEX, N_KEY_CLASSES

    # --- label side --------------------------------------------------------
    assert KEY_CLASSES[transpose_key_index(KEY_TO_INDEX['C major'], 7)] == 'G major'
    assert KEY_CLASSES[transpose_key_index(KEY_TO_INDEX['C major'], -4)] == 'G# major'
    assert KEY_CLASSES[transpose_key_index(KEY_TO_INDEX['A# minor'], 3)] == 'C# minor'
    assert KEY_CLASSES[transpose_key_index(KEY_TO_INDEX['B minor'], 1)] == 'C minor'
    # A shift of 12 (or 0) is the identity, and the mode never changes.
    for idx in range(N_KEY_CLASSES):
        assert transpose_key_index(idx, 0) == idx
        assert transpose_key_index(idx, 12) == idx
        assert transpose_key_index(idx, 5) // 12 == idx // 12
    # ... elementwise on a tensor, negative shifts included
    labels = torch.arange(N_KEY_CLASSES)
    steps = torch.tensor([-4, -1, 0, 3, 7] * 4 + [5, 5, 5, 5])
    moved = transpose_key_index(labels, steps)
    assert torch.equal(transpose_key_index(moved, -steps), labels)
    print("label transposition OK")

    # --- config resolution -------------------------------------------------
    assert resolve_pitch_shifts({}) is None
    assert resolve_pitch_shifts({"augment_pitch": False, "pitch_shift_range": [-4, 7]}) is None
    shifts = resolve_pitch_shifts({"augment_pitch": True})
    assert shifts == tuple(range(-4, 8)) and len(shifts) == 12, shifts
    assert resolve_pitch_shifts({"augment_pitch": True, "pitch_shift_range": [-2, 2]}) == (-2, -1, 0, 1, 2)
    try:
        resolve_pitch_shifts({"augment_pitch": True, "pitch_shift_range": [3, 1]})
        raise AssertionError("expected a ValueError for an inverted range")
    except ValueError:
        pass
    assert all(draw_pitch_shift(shifts) in shifts for _ in range(200))
    assert draw_pitch_shift(None) == 0
    print("config resolution OK")

    # --- audio side --------------------------------------------------------
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else
                          "cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    B = len(shifts)
    n_steps = torch.tensor(shifts, device=device)

    # Output length is preserved at both crop geometries.
    for crop_samples, name in ((240080, "chunked/sliding 10 s crop"),
                               (240000, "streaming 10 s crop")):
        audio = torch.randn(B, 1, crop_samples, device=device)
        labels = torch.arange(B, device=device) % N_KEY_CLASSES
        out, out_labels = apply_pitch_shift_batch(audio, labels, n_steps)
        assert out.shape == (B, 1, crop_samples), out.shape
        assert torch.isfinite(out).all()
        assert torch.equal(out_labels, transpose_key_index(labels, n_steps))
        print(f"{name}: {tuple(audio.shape)} -> {tuple(out.shape)} OK")

    # A 440 Hz tone lands within a few cents of 440 * 2**(n/12).
    sr, seconds = 24000, 2
    t = torch.arange(sr * seconds, device=device, dtype=torch.float32) / sr
    tone = torch.sin(2 * math.pi * 440 * t)[None, None].repeat(B, 1, 1)
    out, _ = apply_pitch_shift_batch(
        tone, torch.zeros(B, dtype=torch.long, device=device), n_steps)
    window = torch.hann_window(sr * seconds, device=device)
    peak_bin = torch.fft.rfft(out[:, 0] * window).abs().argmax(-1).float()
    peak_hz = peak_bin * sr / (sr * seconds)
    worst = 0.0
    for i, n in enumerate(shifts):
        expected = 440.0 * 2 ** (n / 12)
        cents = 1200 * math.log2(float(peak_hz[i]) / expected)
        worst = max(worst, abs(cents))
        assert abs(cents) < 5.0, (n, expected, float(peak_hz[i]), cents)
    print(f"pitch accuracy OK (worst {worst:.2f} cents over {shifts[0]:+d}..{shifts[-1]:+d})")

    # No-op paths return the input untouched.
    audio = torch.randn(4, 1, 24000, device=device)
    labels = torch.arange(4, device=device)
    for zero in (None, torch.zeros(4, dtype=torch.long, device=device)):
        out, out_labels = apply_pitch_shift_batch(audio, labels, zero)
        assert torch.equal(out, audio) and torch.equal(out_labels, labels)
    print("no-op paths OK")
    print("key_augment self-test OK")
