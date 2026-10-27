"""Tiny CPU checks against the actual MuSTream streaming implementation."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mustream"))
import torch
from modeling.moshi.modules.transformer import RingKVCache, StreamingTransformer, create_sin_embedding

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class TransformerRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def model(self, positional_embedding="rope"):
        return StreamingTransformer(d_model=8, num_heads=2, num_layers=2,
                                    dim_feedforward=16, causal=True, context=4,
                                    positional_embedding=positional_embedding).eval()

    def test_two_dimensional_embedding_is_finite_and_correct(self):
        positions = torch.tensor([0, 1, 2, 100]).reshape(1, -1, 1)
        for dtype in (torch.float32, torch.float64):
            actual = create_sin_embedding(positions, 2, dtype=dtype)
            p = positions.to(dtype)
            torch.testing.assert_close(actual, torch.cat((p.cos(), p.sin()), dim=-1))
            self.assertTrue(torch.isfinite(actual).all().item())

    @torch.no_grad()
    def test_chunked_matches_full_after_ring_wrap(self):
        for position in ("sin", "rope", "sin_rope"):
            model = self.model(position)
            x = torch.randn(2, 12, 8)
            expected = model(x)
            with model.streaming(2):
                actual = torch.cat([model(chunk) for chunk in x.split(2, dim=1)], dim=1)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            self.assertTrue(all(state is None for state in model.get_streaming_state().values()))

    @torch.no_grad()
    def test_partial_reset_matches_two_independent_streams(self):
        for position in ("sin", "rope", "sin_rope"):
            model = self.model(position)
            fresh, continuing = copy.deepcopy(model), copy.deepcopy(model)
            prefix, suffix = torch.randn(2, 6, 8), torch.randn(2, 2, 8)
            with model.streaming(2), fresh.streaming(1), continuing.streaming(1):
                for chunk in prefix.split(2, dim=1):
                    model(chunk)
                    continuing(chunk[1:2])
                model.reset_streaming(torch.tensor([True, False]))
                states = model.get_streaming_state()
                self.assertEqual(states[""].offsets.tolist(), [0, 6])
                for name, state in states.items():
                    if name.endswith("self_attn"):
                        self.assertEqual(state.offset.tolist(), [0, 6])
                        self.assertEqual(state.kv_cache.end_offset.tolist(), [0, 6])
                        self.assertEqual(state.offset_cpu, 6)
                actual = model(suffix)
                expected = torch.cat((fresh(suffix[:1]), continuing(suffix[1:])))
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    @torch.no_grad()
    def test_full_reset_matches_new_stream(self):
        model = self.model()
        fresh = copy.deepcopy(model)
        x, suffix = torch.randn(2, 4, 8), torch.randn(2, 2, 8)
        with model.streaming(2), fresh.streaming(2):
            model(x)
            model.reset_streaming()
            for name, state in model.get_streaming_state().items():
                if name.endswith("self_attn"):
                    self.assertEqual(state.offset_cpu, 0)
                    self.assertEqual(state.offset.tolist(), [0, 0])
            torch.testing.assert_close(model(suffix), fresh(suffix), atol=1e-6, rtol=1e-5)

    @torch.no_grad()
    def test_paused_stream_matches_independent_streams(self):
        for position in ("sin", "rope", "sin_rope"):
            for skipped_length, resumed_length in ((1, 1), (2, 1), (4, 2)):
                with self.subTest(position=position, skipped=skipped_length, resumed=resumed_length):
                    model = self.model(position)
                    paused, active = copy.deepcopy(model), copy.deepcopy(model)
                    prefix = torch.randn(2, 8, 8)
                    skipped = torch.randn(2, skipped_length, 8)
                    suffix = torch.randn(2, resumed_length, 8)
                    with model.streaming(2), paused.streaming(1), active.streaming(1):
                        for chunk in prefix.split(2, dim=1):
                            model(chunk)
                            paused(chunk[:1])
                            active(chunk[1:])
                        model.set_exec_mask(torch.tensor([False, True]))
                        for _ in range(2):
                            model(skipped)
                            active(skipped[1:])
                        model.set_exec_mask(torch.tensor([True, True]))
                        actual = model(suffix)
                        expected = torch.cat((paused(suffix[:1]), active(suffix[1:])))
                        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    @torch.no_grad()
    def test_cache_preserves_paused_entries(self):
        for length in (1, 2, 4):
            with self.subTest(chunk_length=length):
                cache = RingKVCache(2, 2, 4, 4, device=torch.device("cpu"), dtype=torch.float32)
                cache.complete(torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 4), torch.tensor([True, True]))
                before = cache.cache.clone()
                keys, values = torch.randn(2, 2, length, 4), torch.randn(2, 2, length, 4)
                cache.complete(keys, values, torch.tensor([False, True]))
                torch.testing.assert_close(cache.cache[:, 0], before[:, 0], rtol=0, atol=0)
                torch.testing.assert_close(cache.cache[0, 1, :, :length], keys[1], rtol=0, atol=0)
                torch.testing.assert_close(cache.cache[1, 1, :, :length], values[1], rtol=0, atol=0)
                self.assertEqual(cache.end_offset.tolist(), [4, 4 + length])
                before = cache.cache.clone()
                offsets = cache.end_offset.clone()
                cache.complete(keys, values, torch.tensor([False, False]))
                torch.testing.assert_close(cache.cache, before, rtol=0, atol=0)
                torch.testing.assert_close(cache.end_offset, offsets, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
