"""
Key-detection evaluation for train_streaming.py checkpoints (causal streaming MERT).

Feature extraction modes:
  - streaming (default): one persistent .streaming() session fed chunk by chunk;
    stream_chunk_duration / stream_chunk_frames only affect speed.
  - single_pass: true -- the whole track in one forward call (diagnostic;
    memory is quadratic in length).
Use backbone_positional_embedding 'rope': each track runs as one long session.

Usage (from key_detection/):
  python infer_streaming.py <config.yaml> <checkpoint.pt> [--split val] [--out run.json] [--curves run.npz]
  python infer_streaming.py <config.yaml> <checkpoint.pt> --audio a.wav [...]  # write <out-dir>/<name>.key per file
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (
    build_classifier, extract_streaming_features, extract_streaming_features_streaming,
    load_streaming_backbone,
)
from key_dataloader import streaming_geometry, streaming_readout_schedule
import yaml

from key_utils import KEY_CLASSES, N_KEY_CLASSES
from infer import _BaseKeyExtractor, build_arg_parser, evaluate_main

import warnings
warnings.filterwarnings("ignore")


class StreamingKeyExtractor(_BaseKeyExtractor):
    """Streaming backbone + trained head (from train_streaming.py); call predict() / predict_running() per track."""

    def __init__(self, config: dict, checkpoint_path):
        # Not _BaseKeyExtractor.__init__: the streaming backbone is built separately.
        self.config = config
        self.sample_rate: int = config["sample_rate"]
        self.hop_size: int = config["hop_size"]
        self.pool_factor: int = config["pool_factor"]
        self.device = torch.device(config.get("device", "cpu"))
        max_duration = config.get("max_duration")
        self.max_samples = None if max_duration is None else int(max_duration * self.sample_rate)

        self.geom = streaming_geometry(config)
        self.frame_duration = self.geom.frame_duration

        self.backbone_name = config.get("backbone", "streaming")
        assert self.backbone_name in ("streaming", "small_streaming"), (
            f"invalid backbone {self.backbone_name!r}: only 'streaming' and 'small_streaming' "
            "(modeling/streaming/streaming_mert.py) are supported here"
        )

        positional_embedding = config.get("backbone_positional_embedding", "rope")
        if positional_embedding == "sin":
            print(
                "WARNING: backbone_positional_embedding='sin' is an ABSOLUTE scheme, and this "
                f"evaluation runs one uninterrupted session over each whole track. Training only "
                f"covered offsets up to crop_duration ({config['crop_duration']}s), so most of a "
                "2-minute excerpt sits outside the range these weights were trained for. Use a "
                "'rope' backbone for whole-track key evaluation."
            )

        # Feature extraction mode: streaming (default) or single_pass.
        self.use_single_pass = bool(config.get("single_pass", False))
        if not self.use_single_pass:
            if config.get("stream_chunk_duration") is not None:
                self.stream_chunk_frames = (
                    round(config["stream_chunk_duration"] * self.geom.label_rate) * self.pool_factor)
            else:
                self.stream_chunk_frames = int(config.get("stream_chunk_frames", self.pool_factor))
            assert self.stream_chunk_frames >= 1

        # Optional: keep one readout per readout_interval seconds (curve size only).
        interval = config.get("readout_interval")
        self.readout_stride = None if interval is None else max(
            1, int(round(interval * self.geom.label_rate)))

        mert_align_checkpoint = config.get("mert_align_checkpoint")
        print(f"Loading {self.backbone_name} backbone ({config['MERT_model']})...")
        if mert_align_checkpoint is not None:
            print(f"  (overriding with distilled weights from {mert_align_checkpoint})")
        self.model = load_streaming_backbone(
            config["MERT_model"], self.device, config.get("streaming_context", 1000),
            mert_align_checkpoint=mert_align_checkpoint,
            positional_embedding=positional_embedding,
            backbone=self.backbone_name,
            num_attention_heads=config.get("student_num_attention_heads", 8),
            expect_emb_dim=config["emb_dim"],
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.model_type = config.get("model_type", "LSTM")
        self.classifier = build_classifier(self.model_type, N_KEY_CLASSES, config).to(self.device)
        self.classifier.load_state_dict(ckpt["model_state"])
        self.classifier.eval()
        # Block size in raw frames, taken from the checkpoint (1 = strictly causal).
        self.lookahead_block = int(getattr(self.model, "lookahead_block", 1))

        mode = ("single full pass" if self.use_single_pass
                else f"streaming ({self.stream_chunk_frames} raw frames/step)")
        print(f"Loaded {self.model_type} head from epoch {ckpt.get('epoch', '?')}")
        print(f"Backbone={self.backbone_name} [{mode}], pool_factor={self.pool_factor}, "
              f"MERT-layer={config['mert_layer']}, frame={self.frame_duration:.4f}s")
        if self.lookahead_block > 1:
            # One readout per block, stamped at the block's end.
            print(f"Backbone is BLOCK-CAUSAL: lookahead_block={self.lookahead_block} raw frames "
                  f"({self.lookahead_block * self.hop_size / self.sample_rate:.3f}s max future per "
                  f"frame) -> one readout per block, every "
                  f"{self.lookahead_block * self.hop_size / self.sample_rate:.3f}s")

    def _running_features(
        self, waveform: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        # Trim to a whole number of pooled frames.
        raw_frames = waveform.shape[1] // self.hop_size
        target_raw_frames = (raw_frames // self.pool_factor) * self.pool_factor
        assert target_raw_frames > 0, "track is shorter than one pooled frame"
        audio = waveform[:, : target_raw_frames * self.hop_size].unsqueeze(0)  # [1, 1, T]

        if self.use_single_pass:
            embeddings = extract_streaming_features(
                self.model, audio, self.pool_factor, target_raw_frames, self.device)
        else:
            embeddings = extract_streaming_features_streaming(
                self.model, audio, self.pool_factor, target_raw_frames, self.device,
                self.hop_size, self.stream_chunk_frames)
        n_frames = embeddings.shape[2]

        # Readout frames and the audio each has consumed (one per block if block-causal).
        frame_idx, consumed_raw = streaming_readout_schedule(
            n_frames, self.pool_factor, self.lookahead_block, target_raw_frames)
        if self.readout_stride is not None and self.readout_stride > 1:
            # Thin the schedule, always keeping the last (whole-track) readout.
            keep = np.arange(self.readout_stride - 1, len(frame_idx), self.readout_stride)
            if keep.size == 0 or keep[-1] != len(frame_idx) - 1:
                keep = np.append(keep, len(frame_idx) - 1)
            frame_idx, consumed_raw = frame_idx[keep], consumed_raw[keep]

        times = consumed_raw * (self.hop_size / self.sample_rate)
        return times, frame_idx, embeddings


def make_extractor(config: dict, checkpoint_path):
    """Extractor factory for evaluate_main."""
    return StreamingKeyExtractor(config, checkpoint_path)


def predict_main(config_path: str, checkpoint_path: str, audio_paths: list, out_dir: str) -> None:
    """Estimate the key of user-provided audio files.

    Writes <out_dir>/<name>.key (final key) and <name>.key_running.tsv (time, running key).
    """
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    extractor = StreamingKeyExtractor(config, checkpoint_path)
    os.makedirs(out_dir, exist_ok=True)
    for audio_path in audio_paths:
        times, logits = extractor.predict_running(audio_path)
        keys = [KEY_CLASSES[i] for i in logits.argmax(dim=-1).tolist()]
        stem = Path(out_dir) / Path(audio_path).stem
        with open(f"{stem}.key", "w") as f:
            f.write(keys[-1] + "\n")
        with open(f"{stem}.key_running.tsv", "w") as f:
            for t, k in zip(times, keys):
                f.write(f"{t:.3f}\t{k}\n")
        changes = [(t, k) for i, (t, k) in enumerate(zip(times, keys)) if i == 0 or k != keys[i - 1]]
        print(f"{audio_path}: {keys[-1]}  (running estimate: "
              + ", ".join(f"{t:.1f}s {k}" for t, k in changes) + f") -> {stem}.key")


def main():
    parser = build_arg_parser(
        "Evaluate a streaming key-detection checkpoint (whole-track MIREX weighted score plus "
        "the convergence profile), or estimate the key of audio files with --audio."
    )
    parser.add_argument("--audio", nargs="+", default=None,
                        help="audio file(s) to process instead of evaluating on --split")
    parser.add_argument("--out-dir", default="predictions", help="where --audio writes .key files")
    args = parser.parse_args()
    if args.audio:
        predict_main(args.config, args.checkpoint, args.audio, args.out_dir)
    else:
        evaluate_main(args, extractor_factory=make_extractor)


if __name__ == "__main__":
    main()
