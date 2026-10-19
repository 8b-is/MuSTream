"""
Inference and mir_eval evaluation for offline, chunked and sliding-window
checkpoints (train_offline.py / train_chunked.py / train_sliding.py).

Usage (from beat_tracking/):
  python infer.py <config.yaml> <checkpoint.pt>

As a module:
  extractor = make_extractor(config, "logs/<run>/best.pt")
  beat_pred, downbeat_pred = extractor.predict(waveform)
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import mir_eval

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from postprocessor import (  # noqa: E402  (beat_tracking/postprocessor.py)
    build_postprocessor, postproc_run_suffix,
)

from beat_dataloader import BeatDataset, mert_input_for_output, mert_output_len, kernels_strides_for_backbone
from models import (
    build_probe, extract_mert_features, load_backbone_and_processor, lookahead_frames,
)

import warnings
warnings.filterwarnings("ignore")


def precision_recall_f_measure(reference_beats, estimated_beats, f_measure_threshold=0.07):
    """Precision/recall/F-measure for beat events, with mir_eval.beat.f_measure's trimming and threshold."""
    reference_beats = mir_eval.beat.trim_beats(reference_beats)
    estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
    if estimated_beats.size == 0 or reference_beats.size == 0:
        return 0.0, 0.0, 0.0
    matching = mir_eval.util.match_events(reference_beats, estimated_beats, f_measure_threshold)
    precision = float(len(matching)) / len(estimated_beats)
    recall = float(len(matching)) / len(reference_beats)
    return precision, recall, mir_eval.util.f_measure(precision, recall)


def report(name, scores, pr_scores):
    if not scores:
        print(f"{name}: no ground truth available in test split")
        return None
    fmeasure = np.mean([s["F-measure"] for s in scores])
    cmlc = np.mean([s["Correct Metric Level Continuous"] for s in scores])
    cmlt = np.mean([s["Correct Metric Level Total"] for s in scores])
    amlc = np.mean([s["Any Metric Level Continuous"] for s in scores])
    amlt = np.mean([s["Any Metric Level Total"] for s in scores])
    precision = np.mean([p for p, _, _ in pr_scores])
    recall = np.mean([r for _, r, _ in pr_scores])
    f1 = np.mean([f for _, _, f in pr_scores])
    print(
        f"{name} (n={len(scores)}) — F-measure: {fmeasure:.4f}  "
        f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}  "
        f"CMLc: {cmlc:.4f}  CMLt: {cmlt:.4f}  AMLc: {amlc:.4f}  AMLt: {amlt:.4f}"
    )
    return {
        "f_measure": fmeasure,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cmlc": cmlc,
        "cmlt": cmlt,
        "amlc": amlc,
        "amlt": amlt,
    }


def _extract_chunked_features(mert, processor, waveform, config, chunk_samples, chunk_stride, kernels, strides, device):
    """Chunked MERT features for a whole track [1, T_samples] -> [1, L, T_valid, D] (padding trimmed)."""
    total_samples = waveform.shape[1]
    n_chunks = max(1, -(-total_samples // chunk_stride))  # ceil div
    pad_len = (n_chunks - 1) * chunk_stride + chunk_samples - total_samples
    audio_padded = F.pad(waveform, (0, pad_len)) if pad_len > 0 else waveform
    windows = audio_padded.unfold(1, chunk_samples, chunk_stride)  # [1, n_chunks, chunk_samples]
    audio_flat = windows[0].unsqueeze(1).to(device)                # [n_chunks, 1, chunk_samples]

    emb_flat = extract_mert_features(
        mert, processor, audio_flat,
        sample_rate=config["sample_rate"],
        pool_factor=config["pool_factor"],
        device=device,
    )  # [n_chunks, n_layers, T_c, D]

    n_layers, T_c, D = emb_flat.shape[1:]
    emb = emb_flat.permute(1, 0, 2, 3).reshape(n_layers, n_chunks * T_c, D).unsqueeze(0)

    valid_frames = max(0, mert_output_len(total_samples, kernels=kernels, strides=strides) // config["pool_factor"])
    valid_frames = min(valid_frames, emb.shape[2])
    return emb[:, :, :valid_frames, :]


def _extract_sliding_features(mert, processor, waveform, config, hop_samples, context_samples,
                               n_labels_per_hop, pad_left, kernels, strides, device, mert_batch_size=None,
                               lookahead=0):
    """Sliding-window MERT features for a whole track [1, T_samples] -> [1, L, T_valid, D], as in train_sliding.py.

    lookahead is in label frames; the caller shrinks pad_left by the same amount.
    """
    total_samples = waveform.shape[1]
    frame_samples = hop_samples // n_labels_per_hop      # samples per pooled label frame
    n_hops = max(1, -(-total_samples // hop_samples))  # ceil div
    right_pad = n_hops * hop_samples - total_samples + lookahead * frame_samples
    padded = F.pad(waveform, (pad_left, right_pad))

    windows = padded.unsqueeze(0).unfold(2, context_samples, hop_samples)  # [1, 1, n_hops, context_samples]
    windows_flat = windows.reshape(n_hops, 1, context_samples)

    batch_size = mert_batch_size or n_hops
    hop_embeddings = []
    for start in range(0, n_hops, batch_size):
        sub = windows_flat[start:start + batch_size].to(device)
        emb_sub = extract_mert_features(
            mert, processor, sub,
            sample_rate=config["sample_rate"], pool_factor=config["pool_factor"], device=device,
        )
        hop_end = emb_sub.shape[2] - lookahead
        # .contiguous() so the full-window output can be freed.
        hop_embeddings.append(emb_sub[:, :, hop_end - n_labels_per_hop : hop_end, :].contiguous())
    emb_flat = torch.cat(hop_embeddings, dim=0)   # [n_hops, n_layers, n_labels_per_hop, D]

    n_layers, _, D = emb_flat.shape[1:]
    emb = emb_flat.permute(1, 0, 2, 3).reshape(n_layers, n_hops * n_labels_per_hop, D).unsqueeze(0)

    valid_frames = max(0, mert_output_len(total_samples, kernels=kernels, strides=strides) // config["pool_factor"])
    valid_frames = min(valid_frames, emb.shape[2])
    return emb[:, :, :valid_frames, :]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BeatExtractor:
    """Backbone + probe for offline or chunked checkpoints (chunked if config has chunk_duration)."""

    def __init__(self, config: dict, checkpoint_path: str):
        self.sample_rate: int = config["sample_rate"]
        self.pool_factor: int = config["pool_factor"]
        self.device = torch.device(config.get("device", "cpu"))
        self.config = config

        fps = config["sample_rate"] / config["hop_size"] / config["pool_factor"]
        backbone = config.get("backbone", "pretrained")
        self.kernels, self.strides = kernels_strides_for_backbone(backbone)

        self.chunk_duration = config.get("chunk_duration")
        if self.chunk_duration is not None:
            n_labels_per_chunk = int(self.chunk_duration * fps)
            self.chunk_samples = mert_input_for_output(
                n_labels_per_chunk * config["pool_factor"], kernels=self.kernels, strides=self.strides
            )
            self.chunk_stride = n_labels_per_chunk * config["pool_factor"] * config["hop_size"]

        self.postproc = build_postprocessor(config, fps)

        print(f"Loading {backbone} backbone (MERT_model={config['MERT_model']})...")
        self.mert, self.processor = load_backbone_and_processor(config, self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model_type = config.get("model_type", "LSTM")
        self.probe = build_probe(model_type, config).to(self.device)
        self.probe.load_state_dict(ckpt["model_state"])
        self.probe.eval()
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    def features(self, waveform: torch.Tensor) -> torch.Tensor:
        """[1, T_samples] whole track -> [1, L, T, D] backbone embeddings."""
        with torch.no_grad():
            if self.chunk_duration is not None:
                return _extract_chunked_features(
                    self.mert, self.processor, waveform, self.config,
                    self.chunk_samples, self.chunk_stride, self.kernels, self.strides, self.device,
                )
            audio = waveform.unsqueeze(0).to(self.device)  # [1, 1, T_samples]
            return extract_mert_features(
                self.mert, self.processor, audio,
                sample_rate=self.sample_rate, pool_factor=self.pool_factor, device=self.device,
            )

    def logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """[1, L, T, D] embeddings -> [T, 2] per-frame beat/downbeat logits on the CPU."""
        with torch.no_grad():
            return self.probe(embeddings)[0].detach().cpu()

    def predict(self, waveform: torch.Tensor):
        """[1, T_samples] whole track -> (beat_times, downbeat_times)."""
        logits = self.logits(self.features(waveform))
        return self.postproc(logits[:, 0], logits[:, 1])


class SlidingBeatExtractor:
    """Backbone + probe for sliding-window checkpoints (train_sliding.py)."""

    def __init__(self, config: dict, checkpoint_path: str):
        self.sample_rate: int = config["sample_rate"]
        self.pool_factor: int = config["pool_factor"]
        self.device = torch.device(config.get("device", "cpu"))
        self.config = config
        self.mert_batch_size = config.get("mert_batch_size")

        fps = config["sample_rate"] / config["hop_size"] / config["pool_factor"]
        hop_duration = config["hop_duration"]
        context_duration = config["context_duration"]
        n_labels_per_hop = round(hop_duration * fps)
        n_labels_per_context = round(context_duration * fps)

        backbone = config.get("backbone", "pretrained")
        self.kernels, self.strides = kernels_strides_for_backbone(backbone)
        self.n_labels_per_hop = n_labels_per_hop
        self.hop_samples = n_labels_per_hop * config["pool_factor"] * config["hop_size"]
        self.context_samples = mert_input_for_output(
            n_labels_per_context * config["pool_factor"], kernels=self.kernels, strides=self.strides
        )
        # Window lookahead in label frames (as in train_sliding.py).
        self.lookahead = lookahead_frames(config)
        if self.lookahead > n_labels_per_context - n_labels_per_hop:
            raise ValueError(
                f"lookahead={config.get('lookahead')}s ({self.lookahead} frames) leaves no room for "
                f"the hop in a {context_duration}s window -- see train_sliding.py"
            )
        self.pad_left = (
            self.context_samples - self.hop_samples
            - self.lookahead * config["pool_factor"] * config["hop_size"]
        )

        self.postproc = build_postprocessor(config, fps)

        print(f"Loading {backbone} backbone (MERT_model={config['MERT_model']})...")
        self.mert, self.processor = load_backbone_and_processor(config, self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # The eval lookahead must match the one the checkpoint was trained with.
        ckpt_config = ckpt.get("config")
        if ckpt_config is not None and "lookahead" in ckpt_config:
            trained_lookahead = lookahead_frames(ckpt_config)
            if trained_lookahead != self.lookahead:
                raise ValueError(
                    f"lookahead mismatch: {checkpoint_path} was trained with "
                    f"{trained_lookahead} frames of window lookahead, but this config's "
                    f"lookahead works out to {self.lookahead} frames "
                    "-- evaluate with the run's own logs/<run>/config.yaml"
                )

        model_type = config.get("model_type", "LSTM")
        self.probe = build_probe(model_type, config).to(self.device)
        self.probe.load_state_dict(ckpt["model_state"])
        self.probe.eval()
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
        print(
            f"Context={context_duration}s, hop={hop_duration}s, "
            f"lookahead={self.lookahead / fps:.4g}s ({self.lookahead} frames), "
            f"pool-factor={config['pool_factor']}"
        )

    def features(self, waveform: torch.Tensor) -> torch.Tensor:
        """[1, T_samples] whole track -> [1, L, T, D] sliding-window embeddings."""
        with torch.no_grad():
            return _extract_sliding_features(
                self.mert, self.processor, waveform, self.config,
                self.hop_samples, self.context_samples, self.n_labels_per_hop, self.pad_left,
                self.kernels, self.strides, self.device, mert_batch_size=self.mert_batch_size,
                lookahead=self.lookahead,
            )

    def logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """[1, L, T, D] embeddings -> [T, 2] per-frame beat/downbeat logits on the CPU."""
        with torch.no_grad():
            return self.probe(embeddings)[0].detach().cpu()

    def predict(self, waveform: torch.Tensor):
        """[1, T_samples] whole track -> (beat_times, downbeat_times)."""
        logits = self.logits(self.features(waveform))
        return self.postproc(logits[:, 0], logits[:, 1])


def _is_sliding_config(config: dict) -> bool:
    """True for a train_sliding.py config (has hop_duration and context_duration)."""
    return "hop_duration" in config and "context_duration" in config


def make_extractor(config: dict, checkpoint_path: str):
    """SlidingBeatExtractor for sliding-window configs, else BeatExtractor."""
    if _is_sliding_config(config):
        return SlidingBeatExtractor(config, checkpoint_path)
    return BeatExtractor(config, checkpoint_path)


# ---------------------------------------------------------------------------
# Evaluation (mir_eval scoring against ground-truth beats/downbeats)
# ---------------------------------------------------------------------------

def evaluate_main(config_path: str, checkpoint_path: str) -> None:
    """Evaluate a checkpoint on the test split with mir_eval; prints Beat/Downbeat summary lines (and logs to wandb if use_wandb)."""
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

    extractor = make_extractor(config, checkpoint_path)

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
    for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
        waveform, beats_gt, downbeats_gt, fid = test_dataset[idx]

        beat_pred, downbeat_pred = extractor.predict(waveform)

        if len(beats_gt) > 0:
            beat_scores.append(mir_eval.beat.evaluate(beats_gt, beat_pred))
            beat_pr_scores.append(precision_recall_f_measure(beats_gt, beat_pred))
        if len(downbeats_gt) > 0:
            downbeat_scores.append(mir_eval.beat.evaluate(downbeats_gt, downbeat_pred))
            downbeat_pr_scores.append(precision_recall_f_measure(downbeats_gt, downbeat_pred))

    beat_summary = report("Beat", beat_scores, beat_pr_scores)
    downbeat_summary = report("Downbeat", downbeat_scores, downbeat_pr_scores)

    if use_wandb:
        log = {}
        if beat_summary is not None:
            log.update({f"test/beat_{k}": v for k, v in beat_summary.items()})
        if downbeat_summary is not None:
            log.update({f"test/downbeat_{k}": v for k, v in downbeat_summary.items()})
        if log:
            wandb.log(log)
        wandb.finish()


def main():
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    evaluate_main(config_path, checkpoint_path)


if __name__ == "__main__":
    main()
