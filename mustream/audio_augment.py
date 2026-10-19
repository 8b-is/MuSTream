"""
Batched, GPU-side pitch-shift augmentation for waveform batches.

``augment_batch`` shifts every row of a (B, T) batch by its own number of
semitones in one pass (phase vocoder + resample, as in
``torchaudio.functional.pitch_shift``), returning the same length it was given.
Used by chord_recognition/chord_augment.py and key_detection/key_augment.py.
"""

import math
from fractions import Fraction
from typing import Optional, Tuple

import torch
from torchaudio.functional.functional import (
    _apply_sinc_resample_kernel,
    _get_sinc_resample_kernel,
)

# torchaudio.functional.pitch_shift's own STFT defaults, kept identical so this
# agrees with it.
N_FFT = 512
HOP = N_FFT // 4
BINS_PER_OCTAVE = 12

# Largest denominator allowed when rounding a resampling ratio to a rational,
# so the sinc kernels stay small and cacheable. 64 keeps every shift in
# [-3, +3] semitones within 0.9 cents of equal temperament.
RATIO_MAX_DENOMINATOR = 64

TWO_PI = 2 * math.pi


def semitone_ratio(n_steps: int, bins_per_octave: int = BINS_PER_OCTAVE) -> Tuple[int, int, float]:
    """Rational approximation of the length multiplier ``2**(-n/bins)`` that a
    pitch shift of ``n_steps`` applies. Returns (orig_freq, new_freq, ratio)
    where ``ratio == new_freq / orig_freq`` -- resampling from ``orig_freq`` to
    ``new_freq`` scales length by ``ratio`` and pitch by ``1 / ratio``."""
    exact = 2.0 ** (-float(n_steps) / bins_per_octave)
    frac = Fraction(exact).limit_denominator(RATIO_MAX_DENOMINATOR)
    return frac.denominator, frac.numerator, float(frac)


class _KernelCache:
    """Sinc resampling kernels and Hann windows, built once per
    (ratio, device, dtype) and reused for the rest of training."""

    def __init__(self):
        self._resample = {}
        self._window = {}

    def window(self, device, dtype) -> torch.Tensor:
        key = (device, dtype)
        win = self._window.get(key)
        if win is None:
            win = torch.hann_window(N_FFT, device=device, dtype=dtype)
            self._window[key] = win
        return win

    def resample(self, waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
        key = (orig_freq, new_freq, waveform.device, waveform.dtype)
        entry = self._resample.get(key)
        gcd = math.gcd(orig_freq, new_freq)
        if entry is None:
            entry = _get_sinc_resample_kernel(
                orig_freq, new_freq, gcd,
                lowpass_filter_width=6, rolloff=0.99,
                resampling_method="sinc_interp_hann", beta=None,
                device=waveform.device, dtype=waveform.dtype,
            )
            self._resample[key] = entry
        kernel, width = entry
        return _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)


_CACHE = _KernelCache()


def _batched_phase_vocoder(
    spec: torch.Tensor,
    rates: torch.Tensor,
    phase_advance: torch.Tensor,
    n_out: int,
) -> torch.Tensor:
    """``torchaudio.functional.phase_vocoder`` with a per-sample rate.

    spec  : (B, F, T) complex
    rates : (B,) float -- output frame k of row b reads input frame k * rate[b]
    n_out : one grid length for the whole batch; reads past a row's end land in
            zero padding, which the caller crops away.

    Returns (B, F, n_out) complex. The phase is accumulated in float64 and
    reduced mod 2*pi, so it does not drift over long clips.
    """
    B, F, T = spec.shape
    real_dtype = spec.real.dtype

    # (B, n_out) read positions, in float64 so the grid stays exact.
    time_steps = torch.arange(n_out, device=spec.device, dtype=torch.float64) * rates[:, None].double()
    alphas = (time_steps % 1.0).to(real_dtype)[:, None, :]                   # (B, 1, n_out)

    phase_0 = spec[..., :1].angle()                                          # (B, F, 1)
    spec = torch.nn.functional.pad(spec, [0, 2])                             # zero tail to read into
    mag_all, ang_all = spec.abs(), spec.angle()

    idx_0 = time_steps.long().clamp_(max=T + 1)[:, None, :].expand(B, F, n_out)
    idx_1 = (time_steps + 1).long().clamp_(max=T + 1)[:, None, :].expand(B, F, n_out)

    angle_0 = torch.gather(ang_all, 2, idx_0)
    angle_1 = torch.gather(ang_all, 2, idx_1)
    norm_0 = torch.gather(mag_all, 2, idx_0)
    norm_1 = torch.gather(mag_all, 2, idx_1)

    # Per-frame phase increment, unwrapped against the expected advance.
    phase = angle_1 - angle_0 - phase_advance
    phase = phase - TWO_PI * torch.round(phase / TWO_PI)
    phase = phase + phase_advance
    phase = torch.cat([phase_0, phase[..., :-1]], dim=-1)

    phase_acc = torch.cumsum(phase.double(), -1).remainder_(TWO_PI).to(real_dtype)
    mag = alphas * norm_1 + (1 - alphas) * norm_0
    return torch.polar(mag, phase_acc)


def _augment_chunk(
    work: torch.Tensor,
    steps_list: list,
    n_steps: torch.Tensor,
    out_samples: int,
    ratio_by_step: dict,
) -> torch.Tensor:
    """The DSP itself, on a slice of the batch small enough to be cheap in
    memory. See augment_batch for what the arguments mean."""
    device = work.device
    q = torch.tensor([ratio_by_step[s][2] for s in steps_list],
                     device=device, dtype=torch.float32)  # (B,) phase-vocoder rate

    # Samples each row needs out of the iSTFT to survive its own resample with
    # out_samples left over; the largest requirement sets the shared grid.
    need = int(math.ceil(out_samples / float(q.min())))
    n_out_frames = -(-need // HOP) + 2

    window = _CACHE.window(device, work.dtype)
    spec = torch.stft(
        work, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=window,
        center=True, pad_mode="reflect", normalized=False, onesided=True,
        return_complex=True,
    )
    phase_advance = torch.linspace(
        0, math.pi * HOP, spec.shape[-2], device=device, dtype=work.dtype
    )[..., None]
    spec = _batched_phase_vocoder(spec, q, phase_advance, n_out_frames)
    vocoded = torch.istft(
        spec, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=window,
        length=(n_out_frames - 1) * HOP,
    )
    del spec

    # One resample per distinct semitone count.
    out = vocoded.new_empty(work.shape[0], out_samples)
    for step in set(steps_list):
        orig_freq, new_freq, _ = ratio_by_step[step]
        rows = (n_steps == step).nonzero(as_tuple=True)[0]
        piece = vocoded.index_select(0, rows)
        if step != 0:
            piece = _CACHE.resample(piece, orig_freq, new_freq)
        assert piece.shape[-1] >= out_samples, (piece.shape, out_samples, step)
        out.index_copy_(0, rows, piece[:, :out_samples])
    return out


def augment_batch(
    waveform: torch.Tensor,
    n_steps: Optional[torch.Tensor] = None,
    out_samples: Optional[int] = None,
    bins_per_octave: int = BINS_PER_OCTAVE,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Pitch-shift every row of a batch independently.

    waveform    : (B, T_in), on any device
    n_steps     : (B,) integer semitones, or None for no pitch shift. Each
                  distinct value costs one cached resampling kernel.
    out_samples : length of the result, cropped from the front; defaults to T_in
                  and must not exceed it.
    chunk_size  : how many rows to run through the vocoder at once; lower it if
                  the training step is memory-tight.

    Returns (B, out_samples) on the input's device and dtype. Runs under
    no_grad with autocast disabled.
    """
    assert waveform.dim() == 2, waveform.shape
    B, T_in = waveform.shape
    out_samples = T_in if out_samples is None else out_samples
    assert T_in >= out_samples, (T_in, out_samples)

    if n_steps is None:
        n_steps = torch.zeros(B, dtype=torch.long, device=waveform.device)
    n_steps = n_steps.to(device=waveform.device, dtype=torch.long)

    # Unshifted rows are copied, never round-tripped through the STFT, so they
    # come back untouched.
    steps_list = n_steps.tolist()
    is_identity = n_steps == 0
    if bool(is_identity.all()):
        return waveform[:, :out_samples]

    # Rounded ratios (see RATIO_MAX_DENOMINATOR). The vocoder must use the same
    # rounded value the resampler will actually apply, otherwise the two stages
    # disagree and the output length drifts from out_samples.
    ratio_by_step = {s: semitone_ratio(s, bins_per_octave) for s in set(steps_list)}

    device = waveform.device
    chunk = max(1, chunk_size)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        out = torch.empty(B, out_samples, device=device, dtype=torch.float32)

        idle = is_identity.nonzero(as_tuple=True)[0]
        if idle.numel():
            out.index_copy_(0, idle, waveform.index_select(0, idle)[:, :out_samples].float())

        active = (~is_identity).nonzero(as_tuple=True)[0]
        for i in range(0, active.numel(), chunk):
            rows = active[i:i + chunk]
            out.index_copy_(0, rows, _augment_chunk(
                waveform.index_select(0, rows).float(),
                [steps_list[j] for j in rows.tolist()],
                n_steps.index_select(0, rows),
                out_samples,
                ratio_by_step,
            ))

    return out.to(waveform.dtype)
