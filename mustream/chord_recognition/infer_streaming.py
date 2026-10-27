"""Evaluate train_streaming.py checkpoints on the causal streaming MERT.

Default mode feeds each song chunk by chunk through one KV-cached streaming
session. The backbone cache is bounded, but evaluation retains the whole
waveform and full-track features, so total memory grows with song length.
single_pass: true runs the whole song in one forward call (diagnostic; can OOM).

Usage (from chord_recognition/):
  python infer_streaming.py <config.yaml> <checkpoint.pt>                      # mir_eval scores on config['eval_split']
  python infer_streaming.py <config.yaml> <checkpoint.pt> --audio a.wav [...]  # write <out-dir>/<name>.lab per file
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

from models import (
    backbone_lookahead_block, build_chord_postprocessor, build_classifier,
    emission_block_for_config,
    extract_streaming_features, extract_streaming_features_streaming, load_streaming_backbone,
)
from infer import (
    _ChordDecodeMixin, _load_audio, _merge_frames, _write_lab, evaluate_and_report,
)
import pickle

import warnings
warnings.filterwarnings("ignore")


class StreamingChordExtractor(_ChordDecodeMixin):
    """Streaming backbone + trained probe; call predict() per audio file.

    Args:
        config:           MERT_streaming.yaml-style config dict
        checkpoint_path:  .pt checkpoint saved by train_streaming.py
    """

    def __init__(self, config: dict, checkpoint_path: str | Path):
        self.sample_rate: int = config["sample_rate"]
        self.pool_factor: int = config["pool_factor"]
        self.hop_size: int = config["hop_size"]
        self.frame_duration: float = config["hop_size"] * config["pool_factor"] / config["sample_rate"]
        self.device = torch.device(config.get("device", "cpu"))
        self.backbone_name = config.get("backbone", "streaming")
        assert self.backbone_name in ("streaming", "small_streaming"), (
            f"invalid backbone {self.backbone_name!r}: only 'streaming' and 'small_streaming' "
            "(modeling/streaming/streaming_mert.py) are supported here"
        )

        # Default: genuine streaming; single_pass: true = one forward call.
        label_rate = config["sample_rate"] / config["hop_size"] / config["pool_factor"]
        self.use_single_pass = bool(config.get("single_pass", False))

        if not self.use_single_pass:
            # Chunk size in raw frames (default: one pooled frame); speed only.
            if config.get("stream_chunk_duration") is not None:
                self.stream_chunk_frames = round(config["stream_chunk_duration"] * label_rate) * config["pool_factor"]
            else:
                self.stream_chunk_frames = int(config.get("stream_chunk_frames", config["pool_factor"]))
            assert self.stream_chunk_frames >= 1

        _, self.idx2chord = pickle.load(open(config["vocab_path"], 'rb'))

        mert_align_checkpoint = config.get("mert_align_checkpoint")

        # Backbone lookahead block (read from the distillation checkpoint) sets
        # the emission burst size used by block-synchronous decoding.
        raw_block = backbone_lookahead_block(mert_align_checkpoint)
        self.emission_block = emission_block_for_config(config)
        self.postproc = build_chord_postprocessor(
            config, len(self.idx2chord), self.device, self.emission_block)
        print(f"Loading {self.backbone_name} backbone ({config['MERT_model']})...")
        if mert_align_checkpoint is not None:
            print(f"  (overriding with fine-tuned weights from {mert_align_checkpoint})")
        self.model = load_streaming_backbone(
            config["MERT_model"], self.device,
            config.get("streaming_context", 1000), mert_align_checkpoint=mert_align_checkpoint,
            positional_embedding=config.get("backbone_positional_embedding", "rope"),
            backbone=self.backbone_name,
            num_attention_heads=config.get("student_num_attention_heads", 8),
            expect_emb_dim=config["emb_dim"],
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device)

        model_type = config.get("model_type", "LSTM")

        # Probes are strictly causal; refuse checkpoints trained with a probe lookahead.
        ckpt_config = ckpt.get("config") or {}
        if float(ckpt_config.get("lookahead") or 0.0) > 0:
            raise ValueError(
                f"{checkpoint_path} was trained with a probe lookahead "
                f"({ckpt_config['lookahead']}s), which is no longer supported"
            )

        self.classifier = build_classifier(model_type, len(self.idx2chord), config).to(self.device)
        self.classifier.load_state_dict(ckpt["model_state"])
        self.classifier.eval()
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
        mode = "single full pass" if self.use_single_pass else f"streaming ({self.stream_chunk_frames} raw frames/step)"
        print(
            f"Backbone={self.backbone_name} [{mode}], pool-factor={config['pool_factor']}, "
            f"MERT-layer={config['mert_layer']}"
        )
        print(
            f"  backbone lookahead_block={raw_block} raw frames "
            f"({raw_block / (config['sample_rate'] / config['hop_size']) * 1000:.0f} ms) "
            f"= {self.emission_block} pooled frame(s) per emission burst"
        )
        print(f"  {self.postproc.describe()}")

    def extract_features(self, audio_path: str | Path):
        """Run the backbone only; return (embeddings [1, L, n_frames, D], n_frames)."""
        waveform = _load_audio(Path(audio_path), self.sample_rate)
        total_samples = waveform.shape[1]

        raw_frames = total_samples // self.hop_size
        target_raw_frames = (raw_frames // self.pool_factor) * self.pool_factor
        usable_samples = target_raw_frames * self.hop_size
        audio = waveform[:, :usable_samples].unsqueeze(0)  # [1, 1, usable_samples]

        n_frames = target_raw_frames // self.pool_factor

        with torch.no_grad():
            if self.use_single_pass:
                embeddings = extract_streaming_features(
                    self.model, audio, self.pool_factor,
                    target_raw_frames, self.device,
                )  # [1, L, n_frames, D]
            else:
                embeddings = extract_streaming_features_streaming(
                    self.model, audio, self.pool_factor,
                    target_raw_frames, self.device, self.hop_size, self.stream_chunk_frames,
                )  # [1, L, n_frames, D]
        return embeddings, n_frames

    def logits_from_features(self, embeddings, n_frames, classifier=None) -> torch.Tensor:
        """Per-frame logits [T_r, vocab_size] from extract_features() output, optionally via another probe."""
        classifier = self.classifier if classifier is None else classifier
        with torch.no_grad():
            logits = classifier(embeddings)                  # [1, n_frames, V]
        assert logits.shape[1] == n_frames, (logits.shape, n_frames)
        return logits[0]                                     # [T_r, vocab_size]

    def predict_logits(self, audio_path: str | Path) -> torch.Tensor:
        """Per-frame logits [T_r, vocab_size] for one track, backbone + probe."""
        embeddings, n_frames = self.extract_features(audio_path)
        return self.logits_from_features(embeddings, n_frames)

    def predict_segments(self, audio_path: str | Path) -> list[tuple[float, float, str]]:
        """Return (start, end, chord) segments, decoded via config['decoder'] (default argmax)."""
        return self.decode_segments(self.predict_logits(audio_path))

    def predict(self, audio_path: str | Path, out_path: str | Path | None = None) -> Path:
        """Write a .lab file for audio_path (default: next to it) and return its path."""
        audio_path = Path(audio_path)
        out_path = Path(out_path) if out_path is not None else audio_path.with_suffix(".lab")

        print(f"Processing {audio_path} ...")
        segments = self.predict_segments(audio_path)
        _write_lab(segments, out_path)
        print(f"Saved {len(segments)} chord segments → {out_path}")
        return out_path


# ---------------------------------------------------------------------------
# Evaluation / CLI entry point
# ---------------------------------------------------------------------------

def evaluate_main(config_path: str, checkpoint_path: str) -> None:
    """Score a train_streaming.py checkpoint on config['eval_split'] (same report as infer.py)."""
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    extractor = StreamingChordExtractor(config, checkpoint_path)
    evaluate_and_report(extractor, config, checkpoint_path, 'infer_streaming.py')


def predict_main(config_path: str, checkpoint_path: str, audio_paths: list, out_dir: str) -> None:
    """Transcribe user-provided audio files to <out_dir>/<name>.lab (start, end, chord)."""
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    extractor = StreamingChordExtractor(config, checkpoint_path)
    os.makedirs(out_dir, exist_ok=True)
    for audio_path in audio_paths:
        extractor.predict(audio_path, Path(out_dir) / (Path(audio_path).stem + ".lab"))


def main():
    parser = argparse.ArgumentParser(description="Streaming chord recognition: evaluate a checkpoint, "
                                                 "or transcribe audio files with --audio.")
    parser.add_argument("config", help="config YAML (the run's own / the released config.yaml)")
    parser.add_argument("checkpoint", help="probe checkpoint (best.pt)")
    parser.add_argument("--audio", nargs="+", default=None,
                        help="audio file(s) to transcribe instead of evaluating on config['eval_split']")
    parser.add_argument("--out-dir", default="predictions", help="where --audio writes .lab files")
    args = parser.parse_args()
    if args.audio:
        predict_main(args.config, args.checkpoint, args.audio, args.out_dir)
    else:
        evaluate_main(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
