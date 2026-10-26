"""Boundary regressions for the standalone quantization helpers (no models)."""

import copy
import unittest

from test_pq_streaming_quant import pq_mod


class TestPqInputBounds(unittest.TestCase):
    def test_zero_denominator_is_not_a_number(self):
        for numerator in (0, 7, -7):
            with self.subTest(numerator=numerator), self.assertRaises(ZeroDivisionError):
                pq_mod.reduce_rational_pq(numerator, 0)
        self.assertEqual(pq_mod.reduce_rational_pq(6, -8), (-3, 4))
        self.assertEqual(pq_mod.reduce_rational_pq(0, 8), (0, 1))

    def test_cache_dimensions_are_positive_integers(self):
        for capacity in (0, -1):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                pq_mod.RingKVCacheBfp64Pq(capacity)
        for capacity in (1.5, True):
            with self.subTest(capacity=capacity), self.assertRaises(TypeError):
                pq_mod.RingKVCacheBfp64Pq(capacity)
        for dim in (0, -64, 63, 65):
            with self.subTest(dim=dim), self.assertRaises(ValueError):
                pq_mod.RingKVCacheBfp64Pq(2, dim)
        for dim in (64.0, True):
            with self.subTest(dim=dim), self.assertRaises(TypeError):
                pq_mod.RingKVCacheBfp64Pq(2, dim)

    def test_quantization_rejects_incomplete_blocks(self):
        for length in (0, 1, 63, 65, 127):
            with self.subTest(length=length), self.assertRaises(ValueError):
                pq_mod.RingKVCacheBfp64Pq.quantize_vec([1.0] * length)
        with self.assertRaises(ValueError):
            pq_mod.Bfp64Block.quantize([1.0] * 63)
        blocks = pq_mod.RingKVCacheBfp64Pq.quantize_vec([1.0] * 128)
        self.assertEqual(pq_mod.RingKVCacheBfp64Pq.dequantize_vec(blocks), [1.0] * 128)

    def test_rejected_cache_update_is_atomic(self):
        cache = pq_mod.RingKVCacheBfp64Pq(3)
        cache.complete([[2.0] * 64], [[3.0] * 64])
        before = copy.deepcopy((cache.k_ring, cache.v_ring, cache.end_offset))
        row = [1.0] * 64
        invalid_chunks = [([], []), ([row], []), ([row] * 4, [row] * 4)]
        for length in (0, 63, 65, 128):
            invalid_chunks.extend([
                ([row, [1.0] * length], [row, row]),
                ([row, row], [row, [1.0] * length]),
            ])
        for bad in (float('nan'), float('inf'), -float('inf')):
            invalid_chunks.extend([
                ([row, [bad] * 64], [row, row]),
                ([row, row], [row, [bad] * 64]),
            ])
        for keys, values in invalid_chunks:
            with self.subTest(keys=keys, values=values):
                with self.assertRaises(ValueError):
                    cache.complete(keys, values)
                self.assertEqual((cache.k_ring, cache.v_ring, cache.end_offset), before)
        result = cache.complete([row], [row])
        self.assertEqual(result.keys[0], [2.0] * 64)
        self.assertEqual(result.values[0], [3.0] * 64)
        self.assertEqual(result.positions, [0, -1, -1, 1])

    def test_linear_requires_rectangular_nonempty_weights(self):
        for weights in ([], [[]], [[1.0], [1.0, 2.0]]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                pq_mod.QuantumTernaryLinearPq(weights)

    def test_linear_requires_matching_bias(self):
        for bias in ([], [0.0, 0.0]):
            with self.subTest(bias=bias), self.assertRaises(ValueError):
                pq_mod.QuantumTernaryLinearPq([[1.0, 1.0]], bias)

    def test_linear_rejects_nonfinite_values(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    pq_mod.QuantumTernaryLinearPq([[bad]])
                with self.assertRaises(ValueError):
                    pq_mod.QuantumTernaryLinearPq([[1.0]], [bad])
                with self.assertRaises(ValueError):
                    pq_mod.QuantumTernaryLinearPq([[1.0]]).forward_vec([bad])

    def test_linear_requires_exact_input_width(self):
        linear = pq_mod.QuantumTernaryLinearPq([[1.0, -1.0]], [0.5])
        for x in ([], [1.0], [1.0, 2.0, 3.0]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                linear.forward_vec(x)
        self.assertEqual(linear.forward_vec([2.0, 1.0]), [1.5])


if __name__ == '__main__':
    unittest.main()
