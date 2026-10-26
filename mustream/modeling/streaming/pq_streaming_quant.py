"""Standalone standard-library references for quantization and beat timing.

These helpers are exercised by unit tests; they are not wired into MuSTream's
backbone, ONNX export or browser inference path. No model accuracy, speed, CPU
cache residency or memory-footprint improvement has been established here.

* BFP-64 quantizes 64 values to integer mantissas and a shared exponent. The
  hypothetical packed payload is 65 bytes versus 256 bytes for FP32 (256/65).
  Actual storage uses Python objects; no binary serializer is implemented.
* Rotary phases use integer modulo-2^32 arithmetic after frequency quantization.
  Q15 lookup values and a 4096-entry table approximate sine and cosine. Integer
  phase arithmetic does not eliminate quantization error over long sessions.
* The ternary linear reference keeps unpacked trits, integer scales and bit masks.
  Its byte estimate assumes five trits per byte plus four bytes per row scale;
  it excludes bias, bit masks and Python overhead. Forward evaluation uses the
  trit lists, additions and scale multiplication, not a popcount matrix kernel.
* The beat oscillator rounds its phase increment down to Q16. Reported BPM is the
  reduced rational for the configured period, not a measured or exact emitted
  tempo. The just-intonation ratios are constants, not a key-detection model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


def reduce_rational_pq(num: int, den: int) -> Tuple[int, int]:
    """Return exact GCD-reduced (p, q) with q > 0."""
    if den == 0:
        raise ZeroDivisionError("rational denominator must not be zero")
    if den < 0:
        num, den = -num, -den
    g = math.gcd(abs(num), den)
    return (num // g, den // g)


def format_pq(num: int, den: int) -> str:
    p, q = reduce_rational_pq(num, den)
    return f"{p}/{q}"


def isqrt_u128(n: int) -> int:
    """Exact integer floor square root floor(sqrt(n))."""
    if n < 2:
        return max(0, n)
    return math.isqrt(n)


# ---------------------------------------------------------------------------
# 1. BFP-64 reference ring cache (hypothetical payload ratio 256/65)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bfp64Block:
    """Python representation of 64 mantissas and a shared power-of-two exponent."""

    mantissas: Tuple[int, ...]
    exponent: int

    @classmethod
    def quantize(cls, values: Sequence[float]) -> "Bfp64Block":
        if len(values) != 64:
            raise ValueError(f"Expected 64 values, got {len(values)}")
        if not all(math.isfinite(v) for v in values):
            raise ValueError("BFP values must be finite")
        max_abs = max((abs(v) for v in values), default=0.0)
        if max_abs == 0.0:
            return cls(mantissas=(0,) * 64, exponent=0)
        raw_exp = math.ceil(math.log2(max(max_abs / 127.0, 2.0**-126)))
        exp_i8 = max(-126, min(126, int(raw_exp)))
        inv_scale = math.ldexp(1.0, -exp_i8)
        mants = tuple(
            max(-127, min(127, int(round(v * inv_scale)))) for v in values
        )
        return cls(mantissas=mants, exponent=exp_i8)

    def dequantize(self) -> List[float]:
        scale = math.ldexp(1.0, self.exponent)
        return [m * scale for m in self.mantissas]


@dataclass
class KVCacheResultPq:
    keys: List[List[float]]    # [cap + T, D]
    values: List[List[float]]  # [cap + T, D]
    positions: List[int]       # [cap + T]


class RingKVCacheBfp64Pq:
    """
    Single-head reference ring cache using BFP-64 Python objects.

    Returns cached rows before writing the incoming chunk, followed by unquantized
    incoming rows. Stored rows are quantized, so later reads are approximate.
    Byte properties describe ideal packed payloads, not this object's footprint.
    """

    BLOCK_SIZE = 64

    def __init__(self, capacity: int, dim_per_head: int = 64):
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if isinstance(dim_per_head, bool) or not isinstance(dim_per_head, int):
            raise TypeError("dim_per_head must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if dim_per_head <= 0 or dim_per_head % self.BLOCK_SIZE:
            raise ValueError("dim_per_head must be a positive multiple of 64")
        self.capacity = int(capacity)
        self.dim_per_head = int(dim_per_head)
        self.num_blocks = self.dim_per_head // self.BLOCK_SIZE
        zero_blk = Bfp64Block((0,) * 64, 0)
        self.k_ring: List[List[Bfp64Block]] = [
            [zero_blk for _ in range(self.num_blocks)] for _ in range(self.capacity)
        ]
        self.v_ring: List[List[Bfp64Block]] = [
            [zero_blk for _ in range(self.num_blocks)] for _ in range(self.capacity)
        ]
        self.end_offset: int = 0

    @property
    def bfp64_bytes(self) -> int:
        """Hypothetical packed K/V payload bytes, excluding Python overhead."""
        return 2 * self.capacity * self.num_blocks * 65

    @property
    def fp32_bytes(self) -> int:
        return 2 * self.capacity * self.dim_per_head * 4

    def compression_ratio_pq(self) -> Tuple[int, int]:
        return reduce_rational_pq(self.fp32_bytes, self.bfp64_bytes)

    @staticmethod
    def quantize_vec(vec: Sequence[float]) -> List[Bfp64Block]:
        if len(vec) == 0 or len(vec) % 64:
            raise ValueError("vector length must be a positive multiple of 64")
        nb = len(vec) // 64
        return [Bfp64Block.quantize(vec[b * 64 : (b + 1) * 64]) for b in range(nb)]

    @staticmethod
    def dequantize_vec(blocks: Sequence[Bfp64Block]) -> List[float]:
        out: List[float] = []
        for blk in blocks:
            out.extend(blk.dequantize())
        return out

    def reset(self) -> None:
        self.end_offset = 0

    def complete(
        self, k_chunk: Sequence[Sequence[float]], v_chunk: Sequence[Sequence[float]]
    ) -> KVCacheResultPq:
        T = len(k_chunk)
        if T != len(v_chunk) or not 0 < T <= self.capacity:
            raise ValueError("key/value chunks must have equal lengths within cache capacity")
        if any(len(row) != self.dim_per_head for row in k_chunk):
            raise ValueError("key row width must equal dim_per_head")
        if any(len(row) != self.dim_per_head for row in v_chunk):
            raise ValueError("value row width must equal dim_per_head")
        # Prepare every row first: a bad later row must leave the cache intact.
        k_quantized = [self.quantize_vec(row) for row in k_chunk]
        v_quantized = [self.quantize_vec(row) for row in v_chunk]
        cap = self.capacity
        end = self.end_offset
        last_written = end - 1
        end_index = last_written % cap

        old_positions: List[int] = []
        for slot in range(cap):
            if slot >= end:
                old_positions.append(-1)
            else:
                delta_slot = slot - end_index
                pos = (
                    last_written + delta_slot
                    if delta_slot <= 0
                    else last_written + delta_slot - cap
                )
                old_positions.append(pos)

        new_positions = [end + t for t in range(T)]
        k_old = [self.dequantize_vec(self.k_ring[s]) for s in range(cap)]
        v_old = [self.dequantize_vec(self.v_ring[s]) for s in range(cap)]

        keys = k_old + [list(row) for row in k_chunk]
        values = v_old + [list(row) for row in v_chunk]
        positions = old_positions + new_positions

        for t in range(T):
            write_slot = (end + t) % cap
            self.k_ring[write_slot] = k_quantized[t]
            self.v_ring[write_slot] = v_quantized[t]
        self.end_offset += T

        return KVCacheResultPq(keys=keys, values=values, positions=positions)


# ---------------------------------------------------------------------------
# 2. Quantized modulo-2^32 rotary embedding reference
# ---------------------------------------------------------------------------


class RotaryEmbeddingPq:
    """
    Approximate rotary embedding with quantized frequency and lookup values.
    The phase multiplication itself uses exact modulo-2^32 integer arithmetic:
        phi_{t, d} = ((t + offset) * omega_u32[d]) mod 2^32
    Rounding the frequency introduces error that can accumulate with the offset.
    """

    RING_MOD = 1 << 32
    LUT_BITS = 12
    LUT_SIZE = 1 << LUT_BITS  # 4096-entry Q15 LUT
    Q15_ONE = 32768

    def __init__(self, dim: int, interleave: bool = True, max_period: int = 10000):
        assert dim % 2 == 0
        self.dim = int(dim)
        self.interleave = bool(interleave)
        self.max_period = int(max_period)

        half_d = self.dim // 2
        self.omega_u32: List[int] = []
        for d in range(half_d):
            freq_rad = 1.0 / (float(self.max_period) ** (2.0 * d / self.dim))
            step_u32 = int(round((freq_rad / (2.0 * math.pi)) * self.RING_MOD)) & 0xFFFFFFFF
            self.omega_u32.append(step_u32)

        self.cos_lut_q15: List[int] = [
            int(round(math.cos(2.0 * math.pi * i / self.LUT_SIZE) * self.Q15_ONE))
            for i in range(self.LUT_SIZE)
        ]
        self.sin_lut_q15: List[int] = [
            int(round(math.sin(2.0 * math.pi * i / self.LUT_SIZE) * self.Q15_ONE))
            for i in range(self.LUT_SIZE)
        ]

    def rotate_frame(self, vec: Sequence[float], abs_step: int) -> List[float]:
        assert len(vec) == self.dim
        half_d = self.dim // 2
        inv_q15 = 1.0 / float(self.Q15_ONE)
        out = [0.0] * self.dim
        for d in range(half_d):
            phase_u32 = (abs_step * self.omega_u32[d]) & 0xFFFFFFFF
            lut_idx = phase_u32 >> (32 - self.LUT_BITS)
            rotr = self.cos_lut_q15[lut_idx] * inv_q15
            roti = self.sin_lut_q15[lut_idx] * inv_q15
            if self.interleave:
                vr = vec[2 * d]
                vi = vec[2 * d + 1]
                out[2 * d] = vr * rotr - vi * roti
                out[2 * d + 1] = vr * roti + vi * rotr
            else:
                vr = vec[d]
                vi = vec[half_d + d]
                out[d] = vr * rotr - vi * roti
                out[half_d + d] = vr * roti + vi * rotr
        return out


# ---------------------------------------------------------------------------
# 3. Balanced-ternary linear reference with Q16 scales
# ---------------------------------------------------------------------------


class QuantumTernaryLinearPq:
    """
    Balanced-ternary {-1, 0, +1} linear projection with per-row Q16 scales.

    Trit lists and bit masks coexist in memory. The five-trit byte count is an
    estimate only; there is no packed-weight serializer or packed forward kernel.
    """

    Q16_SHIFT = 16
    Q16_ONE = 1 << Q16_SHIFT

    def __init__(
        self,
        weight: Sequence[Sequence[float]],
        bias: Optional[Sequence[float]] = None,
    ):
        self.out_features = len(weight)
        self.in_features = len(weight[0]) if self.out_features > 0 else 0
        if self.out_features == 0 or self.in_features == 0:
            raise ValueError("weight must be a nonempty rectangular matrix")
        if any(len(row) != self.in_features for row in weight):
            raise ValueError("weight rows must have equal widths")
        if not all(math.isfinite(v) for row in weight for v in row):
            raise ValueError("weights must be finite")
        if bias is not None:
            if len(bias) != self.out_features:
                raise ValueError("bias length must equal out_features")
            if not all(math.isfinite(v) for v in bias):
                raise ValueError("bias values must be finite")
        self.num_words = (self.in_features + 63) // 64

        self.trits: List[List[int]] = []
        self.scale_q16: List[int] = []
        self.pos_masks: List[List[int]] = []
        self.neg_masks: List[List[int]] = []
        self.row_popcounts: List[int] = []

        for r in range(self.out_features):
            row = weight[r]
            gamma = max(sum(abs(v) for v in row) / max(len(row), 1), 1e-8)
            t_row = [max(-1, min(1, int(round(v / gamma)))) for v in row]
            self.trits.append(t_row)
            self.scale_q16.append(int(round(gamma * self.Q16_ONE)))

            p_words: List[int] = []
            n_words: List[int] = []
            pop = 0
            for w_idx in range(self.num_words):
                c0 = w_idx * 64
                c1 = min(self.in_features, c0 + 64)
                pos_u64 = 0
                neg_u64 = 0
                for b_idx, t_val in enumerate(t_row[c0:c1]):
                    if t_val > 0:
                        pos_u64 |= 1 << b_idx
                    elif t_val < 0:
                        neg_u64 |= 1 << b_idx
                p_words.append(pos_u64)
                n_words.append(neg_u64)
                pop += (pos_u64 | neg_u64).bit_count()
            self.pos_masks.append(p_words)
            self.neg_masks.append(n_words)
            self.row_popcounts.append(pop)

        if bias is not None:
            self.bias_q16: Optional[List[int]] = [
                int(round(b * self.Q16_ONE)) for b in bias
            ]
        else:
            self.bias_q16 = None

    def packed_5trit_bytes(self) -> int:
        """Ideal weight/scale payload estimate; excludes bias, masks and overhead."""
        total_trits = self.out_features * self.in_features
        return (total_trits + 4) // 5 + self.out_features * 4

    def fp32_bytes(self) -> int:
        return self.out_features * self.in_features * 4

    def compression_ratio_pq(self) -> Tuple[int, int]:
        return reduce_rational_pq(self.fp32_bytes(), self.packed_5trit_bytes())

    def forward_vec(self, x: Sequence[float]) -> List[float]:
        if len(x) != self.in_features:
            raise ValueError("input length must equal in_features")
        if not all(math.isfinite(v) for v in x):
            raise ValueError("input values must be finite")
        x_q16 = [int(round(v * self.Q16_ONE)) for v in x]
        inv_q16 = 1.0 / float(self.Q16_ONE)
        out: List[float] = []
        for r in range(self.out_features):
            t_row = self.trits[r]
            acc = 0
            for x_val, t_val in zip(x_q16, t_row):
                if t_val == 1:
                    acc += x_val
                elif t_val == -1:
                    acc -= x_val
            val_q16 = (acc * self.scale_q16[r]) >> self.Q16_SHIFT
            if self.bias_q16 is not None:
                val_q16 += self.bias_q16[r]
            out.append(val_q16 * inv_q16)
        return out


# ---------------------------------------------------------------------------
# 4. Q16 beat oscillator and just-intonation reference ratios
# ---------------------------------------------------------------------------


JUST_INTONATION_RATIOS_PQ: Tuple[Tuple[int, int], ...] = (
    (1, 1),    # Unison (C)
    (16, 15),  # Minor 2nd (C#)
    (9, 8),    # Major 2nd (D)
    (6, 5),    # Minor 3rd (Eb)
    (5, 4),    # Major 3rd (E)
    (4, 3),    # Perfect 4th (F)
    (45, 32),  # Tritone (F#)
    (3, 2),    # Perfect 5th (G)
    (8, 5),    # Minor 6th (Ab)
    (5, 3),    # Major 6th (A)
    (9, 5),    # Minor 7th (Bb)
    (15, 8),   # Major 7th (B)
)


@dataclass
class BeatPllStatePq:
    """Q16 beat oscillator with rational nominal BPM and a bar-position counter."""

    fps_num: int = 75
    fps_den: int = 1
    period_frames_num: int = 37
    period_frames_den: int = 1
    phase_q16: int = 0
    bar_beat_idx: int = 0  # Position of the next emitted beat within the bar.
    beats_per_bar: int = 4

    def step_frame(self, beat_active: bool, downbeat_active: bool) -> Tuple[bool, bool, Tuple[int, int]]:
        """
        Advance 1 frame on the Q16 phase ring [0, 65536) and return:
        (emit_beat, emit_downbeat, nominal_bpm_pq).
        """
        omega_q16 = (65536 * self.period_frames_den) // max(self.period_frames_num, 1)
        self.phase_q16 = (self.phase_q16 + omega_q16) & 0xFFFF

        if beat_active:
            err = self.phase_q16 if self.phase_q16 < 32768 else self.phase_q16 - 65536
            self.phase_q16 = (self.phase_q16 - (err >> 2)) & 0xFFFF

        emit_beat = beat_active or (self.phase_q16 < omega_q16)
        emit_downbeat = False
        if emit_beat:
            if downbeat_active:
                self.bar_beat_idx = 0
                emit_downbeat = True
            else:
                emit_downbeat = self.bar_beat_idx == 0
            self.bar_beat_idx = (self.bar_beat_idx + 1) % self.beats_per_bar

        bpm_pq = reduce_rational_pq(
            60 * self.fps_num * self.period_frames_den,
            self.fps_den * self.period_frames_num,
        )
        return emit_beat, emit_downbeat, bpm_pq
