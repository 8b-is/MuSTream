"""Beat/bar alignment regressions for the standalone fixed-point oscillator."""

import unittest

from test_pq_streaming_quant import BeatPllStatePq


class TestBeatPllBarProgression(unittest.TestCase):
    def test_separated_acoustic_downbeats_do_not_duplicate(self):
        for meter in (1, 3, 4, 7):
            with self.subTest(meter=meter):
                pll = BeatPllStatePq(period_frames_num=32, beats_per_bar=meter)
                events = []
                for frame in range(32 * meter * 3):
                    beat, downbeat, bpm = pll.step_frame(
                        frame % 32 == 0, frame % (32 * meter) == 0
                    )
                    if beat:
                        events.append((frame, downbeat))
                    self.assertEqual(bpm, (1125, 8))
                self.assertEqual(events, [
                    (frame, frame % (32 * meter) == 0)
                    for frame in range(0, 32 * meter * 3, 32)
                ])

    def test_single_acoustic_anchor_preserves_following_bars(self):
        pll = BeatPllStatePq(period_frames_num=32)
        downbeats = []
        for frame in range(384):
            _, downbeat, _ = pll.step_frame(frame % 32 == 0, frame == 0)
            if downbeat:
                downbeats.append(frame)
        self.assertEqual(downbeats, [0, 128, 256])

    def test_acoustic_downbeat_reanchors_midbar(self):
        pll = BeatPllStatePq(period_frames_num=32)
        events = []
        for frame in range(256):
            beat, downbeat, _ = pll.step_frame(frame % 32 == 0, frame == 64)
            if beat:
                events.append((frame, downbeat))
        self.assertEqual(events, [
            (0, True), (32, False), (64, True), (96, False),
            (128, False), (160, False), (192, True), (224, False),
        ])

    def test_free_running_oscillator_keeps_meter(self):
        pll = BeatPllStatePq(period_frames_num=32)
        events = []
        for frame in range(256):
            beat, downbeat, _ = pll.step_frame(False, False)
            if beat:
                events.append((frame, downbeat))
        self.assertEqual(events, [
            (31, True), (63, False), (95, False), (127, False),
            (159, True), (191, False), (223, False), (255, False),
        ])


if __name__ == '__main__':
    unittest.main()
