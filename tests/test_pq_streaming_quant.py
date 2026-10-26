"""Functional checks for standalone references, not model/performance benchmarks.

Byte-count checks cover theoretical payload estimates, not actual Python memory.
"""

import importlib.util
import math
import os
import random
import unittest

_MOD_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "mustream",
    "modeling",
    "streaming",
    "pq_streaming_quant.py",
)
import sys

_SPEC = importlib.util.spec_from_file_location("pq_streaming_quant", _MOD_PATH)
pq_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["pq_streaming_quant"] = pq_mod
assert _SPEC.loader is not None
_SPEC.loader.exec_module(pq_mod)

BeatPllStatePq = pq_mod.BeatPllStatePq
Bfp64Block = pq_mod.Bfp64Block
JUST_INTONATION_RATIOS_PQ = pq_mod.JUST_INTONATION_RATIOS_PQ
QuantumTernaryLinearPq = pq_mod.QuantumTernaryLinearPq
RingKVCacheBfp64Pq = pq_mod.RingKVCacheBfp64Pq
RotaryEmbeddingPq = pq_mod.RotaryEmbeddingPq
format_pq = pq_mod.format_pq
reduce_rational_pq = pq_mod.reduce_rational_pq


def _l2_norm(vec):
    return math.sqrt(sum(v * v for v in vec))


class TestYamahaMuSTreamPq(unittest.TestCase):
    def test_bfp64_ring_kv_cache_compression_and_accuracy(self):
        rng = random.Random(42)
        cap, D = 64, 64
        bfp_cache = RingKVCacheBfp64Pq(capacity=cap, dim_per_head=D)
        self.assertEqual(bfp_cache.compression_ratio_pq(), (256, 65))

        for step in range(4):
            k_chunk = [[rng.gauss(0.0, 1.0) for _ in range(D)] for _ in range(16)]
            v_chunk = [[rng.gauss(0.0, 1.0) for _ in range(D)] for _ in range(16)]
            res = bfp_cache.complete(k_chunk, v_chunk)
            self.assertEqual(len(res.keys), cap + 16)
            # Check reconstruction error on this seeded sample, not a worst-case bound.
            for row in k_chunk:
                rec = bfp_cache.dequantize_vec(bfp_cache.quantize_vec(row))
                err = _l2_norm([a - b for a, b in zip(row, rec)]) / _l2_norm(row)
                self.assertLess(err, 0.012)
            if step == 3:
                valid_pos = [p for p in res.positions if p >= 0]
                self.assertEqual(len(valid_pos), 3 * 16 + 16)

    def test_rotary_embedding_pq_norm_preservation_at_huge_offset(self):
        rng = random.Random(7)
        D = 64
        vec = [rng.gauss(0.0, 1.0) for _ in range(D)]
        rope_pq = RotaryEmbeddingPq(dim=D, interleave=True)
        # Simulate 24 hours of continuous concert streaming at 75 fps: offset = 6,480,000 frames
        rot = rope_pq.rotate_frame(vec, abs_step=6_480_000)
        norm_ratio = _l2_norm(rot) / _l2_norm(vec)
        self.assertAlmostEqual(norm_ratio, 1.0, places=3)

    def test_quantum_ternary_linear_pq_compression_and_popcount(self):
        rng = random.Random(11)
        in_d, out_d = 256, 128
        w = [[rng.gauss(0.0, 0.05) for _ in range(in_d)] for _ in range(out_d)]
        b = [rng.gauss(0.0, 0.01) for _ in range(out_d)]
        q_lin = QuantumTernaryLinearPq(w, b)
        p, q = q_lin.compression_ratio_pq()
        self.assertGreater(p / q, 18.5)
        for r in range(out_d):
            expected_l2_sq = sum(t * t for t in q_lin.trits[r])
            self.assertEqual(q_lin.row_popcounts[r], expected_l2_sq)
        x = [rng.gauss(0.0, 1.0) for _ in range(in_d)]
        y = q_lin.forward_vec(x)
        self.assertEqual(len(y), out_d)

    def test_laser_phoenix_beat_pll_pq(self):
        pll = BeatPllStatePq(fps_num=75, fps_den=1, period_frames_num=375, period_frames_den=10)
        beats = 0
        downbeats = 0
        for frame in range(300):
            acoustic_beat = (frame % 38) == 0
            acoustic_db = (frame % 152) == 0
            eb, edb, bpm_pq = pll.step_frame(acoustic_beat, acoustic_db)
            if eb:
                beats += 1
            if edb:
                downbeats += 1
            self.assertEqual(bpm_pq, (120, 1))
        self.assertGreaterEqual(beats, 8)
        self.assertGreaterEqual(downbeats, 2)
        self.assertEqual(len(JUST_INTONATION_RATIOS_PQ), 12)


if __name__ == "__main__":
    unittest.main()
