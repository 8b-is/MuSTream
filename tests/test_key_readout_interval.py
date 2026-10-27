"""Readout timing checks with synthetic embeddings, without model downloads."""
import ast
from pathlib import Path
from typing import Optional, Tuple
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / 'mustream/key_detection'


def load_definition(filename, name, namespace):
    """Use the real definitions without importing unrelated audio/model loaders."""
    tree = ast.parse((ROOT / filename).read_text())
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
    assert len(selected) == 1
    exec(compile(ast.Module(body=selected, type_ignores=[]),
                 str(ROOT / filename), 'exec'), namespace)
    return namespace[name]


NAMESPACE = dict(np=np, torch=torch, Optional=Optional, Tuple=Tuple,
                 _BaseKeyExtractor=object)
load_definition('key_dataloader.py', 'streaming_readout_schedule', NAMESPACE)
Extractor = load_definition('infer_streaming.py', 'StreamingKeyExtractor', NAMESPACE)


class KeyReadoutIntervalTests(unittest.TestCase):
    def readouts(self, raw_frames, interval=1., block=15, single_pass=False):
        extractor = Extractor.__new__(Extractor)
        extractor.hop_size, extractor.sample_rate, extractor.pool_factor = 320, 24000, 5
        extractor.lookahead_block = block
        extractor.readout_stride = None if interval is None else max(1, round(interval * 15))
        extractor.use_single_pass = single_pass
        extractor.model, extractor.device, extractor.stream_chunk_frames = None, 'cpu', 15
        embeddings = torch.zeros(1, 1, raw_frames // 5, 2)
        NAMESPACE['extract_streaming_features'] = lambda *args: embeddings
        NAMESPACE['extract_streaming_features_streaming'] = lambda *args: embeddings
        times, indices, actual_embeddings = extractor._running_features(torch.zeros(1, raw_frames * 320))
        self.assertIs(actual_embeddings, embeddings)
        self.assertEqual(len(times), len(indices))
        self.assertTrue(np.all(np.diff(times) > 0))
        return times, indices

    def test_block_causal_interval_is_seconds_not_blocks(self):
        times, indices = self.readouts(300)
        np.testing.assert_allclose(times, [1, 2, 3, 4])
        np.testing.assert_array_equal(indices, [14, 29, 44, 59])

    def test_single_pass_uses_the_same_timestamp_schedule(self):
        times, _ = self.readouts(300, single_pass=True)
        np.testing.assert_allclose(times, [1, 2, 3, 4])

    def test_strictly_causal_schedule_is_preserved(self):
        times, indices = self.readouts(300, block=1)
        np.testing.assert_allclose(times, [1, 2, 3, 4])
        np.testing.assert_array_equal(indices, [14, 29, 44, 59])

    def test_final_readout_survives_thinning(self):
        times, _ = self.readouts(240)
        np.testing.assert_allclose(times, [1, 2, 3, 3.2])

    def test_interval_before_available_block_uses_next_readout(self):
        times, _ = self.readouts(165, block=8)
        np.testing.assert_allclose(times, np.array([80, 152, 160]) / 75)

    def test_short_track_keeps_its_only_readout(self):
        times, indices = self.readouts(5)
        np.testing.assert_allclose(times, [5 / 75])
        np.testing.assert_array_equal(indices, [0])

    def test_default_and_sub_block_intervals_keep_every_block(self):
        for interval in (None, 1 / 15, 2 / 15):
            with self.subTest(interval=interval):
                times, _ = self.readouts(75, interval=interval)
                np.testing.assert_allclose(times, [0.2, 0.4, 0.6, 0.8, 1.0])


if __name__ == '__main__':
    unittest.main()
