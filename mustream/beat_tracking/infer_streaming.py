"""
Inference and mir_eval evaluation for train_streaming.py checkpoints (causal streaming MERT).

Modes: chunk-by-chunk backbone streaming (default; one bounded KV-cache session)
or single_pass: true (whole track in one call; may OOM). Evaluation retains the
whole waveform and full-track features, so total memory grows with track length.

Usage (from beat_tracking/):
  python infer_streaming.py <config.yaml> <checkpoint.pt>                      # mir_eval scores on the test split
  python infer_streaming.py <config.yaml> <checkpoint.pt> --audio a.wav [...]  # write <out-dir>/<name>.beats per file
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio
import yaml
from tqdm import tqdm
import mir_eval

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from postprocessor import (  # noqa: E402  (beat_tracking/postprocessor.py)
    build_postprocessor, postproc_run_suffix,
)

from beat_dataloader import BeatDataset
from models import (
    build_probe, extract_streaming_features,
    extract_streaming_features_streaming, load_streaming_backbone,
)
from infer import precision_recall_f_measure, report

import warnings
warnings.filterwarnings("ignore")


class StreamingBeatExtractor:
    """Causal streaming backbone + probe for train_streaming.py checkpoints."""

    def __init__(self, config: dict, checkpoint_path: str):
        self.sample_rate: int = config["sample_rate"]
        self.pool_factor: int = config["pool_factor"]
        self.hop_size: int = config["hop_size"]
        fps = config["sample_rate"] / config["hop_size"] / config["pool_factor"]
        self.device = torch.device(config.get("device", "cpu"))
        self.backbone_name = config.get("backbone", "streaming")
        assert self.backbone_name in ("streaming", "small_streaming"), (
            f"invalid backbone {self.backbone_name!r}: only 'streaming' and 'small_streaming' "
            "(modeling/streaming/streaming_mert.py) are supported here"
        )

        # Streaming (default) or single full pass; chunk size is only a speed knob.
        label_rate = config["sample_rate"] / config["hop_size"] / config["pool_factor"]
        self.use_single_pass = bool(config.get("single_pass", False))

        if not self.use_single_pass:
            if config.get("stream_chunk_duration") is not None:
                self.stream_chunk_frames = round(config["stream_chunk_duration"] * label_rate) * config["pool_factor"]
            else:
                self.stream_chunk_frames = int(config.get("stream_chunk_frames", config["pool_factor"]))
            assert self.stream_chunk_frames >= 1

        self.postproc = build_postprocessor(config, fps)

        mert_align_checkpoint = config.get("mert_align_checkpoint")
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

        self.probe = build_probe(model_type, config).to(self.device)
        self.probe.load_state_dict(ckpt["model_state"])
        self.probe.eval()
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
        mode = "single full pass" if self.use_single_pass else f"streaming ({self.stream_chunk_frames} raw frames/step)"
        print(f"Backbone={self.backbone_name} [{mode}], pool-factor={config['pool_factor']}")

    def predict(self, waveform: torch.Tensor):
        """[1, T_samples] whole track -> (beat_times, downbeat_times)."""
        logits = self.logits(*self.features(waveform))
        return self.postproc(logits[:, 0], logits[:, 1])

    def features(self, waveform: torch.Tensor):
        """[1, T_samples] whole track -> ([1, L, n_frames, D] embeddings, n_frames)."""
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

    def logits(self, embeddings: torch.Tensor, n_frames: int) -> torch.Tensor:
        """([1, L, n_frames, D], n_frames) -> [n_frames, 2] logits on the CPU."""
        with torch.no_grad():
            out = self.probe(embeddings)[0]  # [n_frames, 2]
        assert out.shape[0] == n_frames, (out.shape, n_frames)
        return out.detach().cpu()


# ---------------------------------------------------------------------------
# Evaluation / CLI entry point
# ---------------------------------------------------------------------------

def evaluate_main(config_path: str, checkpoint_path: str) -> None:
    """Evaluate a checkpoint on the test split with mir_eval (pooled and per dataset); logs to wandb if use_wandb."""
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    fps = config["sample_rate"] / config["hop_size"] / config["pool_factor"]

    test_dataset = BeatDataset(
        audio_dir=config["audio_dir"],
        annotation_dir=config["annotation_dir"],
        split_dir=config["split_dir"],
        split="test",
        mode="test",
        sr=config["sample_rate"],
        fps=fps,
        pool_factor=config["pool_factor"],
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        split_seed=config["split_seed"],
        audio_ext=config.get("audio_ext", "wav"),
    )
    print(f"Test tracks: {len(test_dataset)}")

    extractor = StreamingBeatExtractor(config, checkpoint_path)

    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        import wandb
        run_name = os.path.basename(os.path.dirname(checkpoint_path))
        run_name += postproc_run_suffix(config)
        wandb.init(
            entity=config.get("wandb_entity"),
            project=config.get("wandb_project", "beat-tracking"),
            name=f"eval-{run_name}",
            job_type="eval",
        )

    beat_scores, downbeat_scores = [], []
    beat_pr_scores, downbeat_pr_scores = [], []
    # Per-dataset grouping from `<Dataset>/<track>` ids (None if no '/').
    beat_groups, downbeat_groups = [], []
    for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
        waveform, beats_gt, downbeats_gt, fid = test_dataset[idx]
        group = fid.split("/", 1)[0] if "/" in fid else None

        beat_pred, downbeat_pred = extractor.predict(waveform)

        if len(beats_gt) > 0:
            beat_scores.append(mir_eval.beat.evaluate(beats_gt, beat_pred))
            beat_pr_scores.append(precision_recall_f_measure(beats_gt, beat_pred))
            beat_groups.append(group)
        if len(downbeats_gt) > 0:
            downbeat_scores.append(mir_eval.beat.evaluate(downbeats_gt, downbeat_pred))
            downbeat_pr_scores.append(precision_recall_f_measure(downbeats_gt, downbeat_pred))
            downbeat_groups.append(group)

    per_dataset = {}
    datasets = sorted({g for g in beat_groups + downbeat_groups if g is not None})
    if len(datasets) > 1:
        print("\nPer-dataset:")
        for dataset in datasets:
            for name, scores, pr, groups in (
                ("Beat", beat_scores, beat_pr_scores, beat_groups),
                ("Downbeat", downbeat_scores, downbeat_pr_scores, downbeat_groups),
            ):
                keep = [i for i, g in enumerate(groups) if g == dataset]
                if not keep:
                    continue
                summary = report(f"  {dataset} {name}", [scores[i] for i in keep],
                                 [pr[i] for i in keep])
                if summary is not None:
                    slug = dataset.replace(" ", "_")
                    per_dataset.update(
                        {f"test/{slug}/{name.lower()}_{k}": v for k, v in summary.items()})
        print("\nPooled:")

    beat_summary = report("Beat", beat_scores, beat_pr_scores)
    downbeat_summary = report("Downbeat", downbeat_scores, downbeat_pr_scores)

    if use_wandb:
        log = dict(per_dataset)
        if beat_summary is not None:
            log.update({f"test/beat_{k}": v for k, v in beat_summary.items()})
        if downbeat_summary is not None:
            log.update({f"test/downbeat_{k}": v for k, v in downbeat_summary.items()})
        if log:
            wandb.log(log)
        wandb.finish()


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    """Mono audio resampled to sample_rate, [1, T]."""
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform


def predict_main(config_path: str, checkpoint_path: str, audio_paths: list, out_dir: str) -> None:
    """Track beats in user-provided audio files; writes <out_dir>/<name>.beats (time, is_downbeat)."""
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    extractor = StreamingBeatExtractor(config, checkpoint_path)
    os.makedirs(out_dir, exist_ok=True)
    for audio_path in audio_paths:
        beats, downbeats = extractor.predict(load_audio(audio_path, config["sample_rate"]))
        downbeat_set = {round(float(t), 4) for t in downbeats}
        times = sorted({round(float(t), 4) for t in beats} | downbeat_set)
        out_path = Path(out_dir) / (Path(audio_path).stem + ".beats")
        with open(out_path, "w") as f:
            for t in times:
                f.write(f"{t:.4f}\t{int(t in downbeat_set)}\n")
        print(f"{audio_path}: {len(beats)} beats, {len(downbeats)} downbeats -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Streaming beat tracking: evaluate a checkpoint, "
                                                 "or track beats in audio files with --audio.")
    parser.add_argument("config", help="config YAML (the run's own / the released config.yaml)")
    parser.add_argument("checkpoint", help="probe checkpoint (best.pt)")
    parser.add_argument("--audio", nargs="+", default=None,
                        help="audio file(s) to process instead of evaluating on the test split")
    parser.add_argument("--out-dir", default="predictions", help="where --audio writes .beats files")
    args = parser.parse_args()
    if args.audio:
        predict_main(args.config, args.checkpoint, args.audio, args.out_dir)
    else:
        evaluate_main(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
