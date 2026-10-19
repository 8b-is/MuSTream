"""
Key-detection evaluation for train_offline.py / train_chunked.py /
train_sliding.py checkpoints (extractor picked from the config), plus the
evaluation driver shared with infer_streaming.py.

Reports the whole-track MIREX weighted score and the convergence profile
(running prediction vs. seconds of audio consumed).

Usage (from key_detection/):
  python infer.py <config.yaml> <checkpoint.pt> [--split val] [--out run.json] [--curves run.npz]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_classifier, extract_MERT_features, load_backbone_and_processor
from key_dataloader import build_eval_dataset, chunked_geometry, offline_geometry, sliding_geometry
from key_utils import (
    DEFAULT_REPORT_GRID, KEY_CLASSES, N_KEY_CLASSES, aggregate_convergence,
    convergence_metrics, curve_at_grid, format_summary, json_safe,
)

import warnings
warnings.filterwarnings("ignore")


def _load_audio(path: Path, target_sr: int, max_samples: Optional[int] = None) -> torch.Tensor:
    """Mono audio at target_sr, [1, T], optionally truncated to max_samples."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if max_samples is not None:
        waveform = waveform[:, :max_samples]
    return waveform


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def apply_head(classifier: torch.nn.Module, embeddings: torch.Tensor,
               readout_idx: np.ndarray) -> torch.Tensor:
    """Run the head over [1, L, T, D] embeddings and return logits at readout_idx, [n_readouts, n_classes]."""
    running = classifier(embeddings, running=True)[0]
    idx = torch.as_tensor(readout_idx, device=running.device, dtype=torch.long)
    return running[idx]


class _BaseKeyExtractor:
    """Shared backbone/head loading and readout API.

    Subclasses implement _running_features(waveform) -> (times, readout_idx, embeddings).
    """

    def __init__(self, config: dict, checkpoint_path):
        self.config = config
        self.sample_rate: int = config["sample_rate"]
        self.pool_factor: int = config["pool_factor"]
        self.device = torch.device(config.get("device", "cpu"))
        self.mert_batch_size = config.get("mert_batch_size")
        max_duration = config.get("max_duration")
        self.max_samples = None if max_duration is None else int(max_duration * self.sample_rate)

        print(f"Loading {config.get('backbone', 'pretrained')} backbone "
              f"(MERT_model={config['MERT_model']})...")
        self.mert, self.processor = load_backbone_and_processor(config, self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model_type = config.get("model_type", "LSTM")
        self.classifier = build_classifier(self.model_type, N_KEY_CLASSES, config).to(self.device)
        self.classifier.load_state_dict(ckpt["model_state"])
        self.classifier.eval()
        print(f"Loaded {self.model_type} head from epoch {ckpt.get('epoch', '?')}")

    # -- subclass hook ------------------------------------------------------
    def _running_features(
        self, waveform: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """(times, readout_idx, embeddings [1, L, T, D]); times[i] = seconds of audio consumed."""
        raise NotImplementedError

    def _running_logits(self, waveform: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        times, readout_idx, embeddings = self._running_features(waveform)
        return times, apply_head(self.classifier, embeddings, readout_idx)

    def _extract_in_batches(self, windows_flat: torch.Tensor) -> torch.Tensor:
        """extract_MERT_features over [N, 1, samples], in sub-batches of mert_batch_size."""
        batch = self.mert_batch_size or windows_flat.shape[0]
        parts = [
            extract_MERT_features(
                self.mert, self.processor, windows_flat[i : i + batch].to(self.device),
                pool_factor=self.pool_factor)
            for i in range(0, windows_flat.shape[0], batch)
        ]
        return torch.cat(parts, dim=0)

    # -- public API ---------------------------------------------------------
    def running_features(self, audio_path) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """(times, readout_idx, embeddings) for one track; pass to apply_head."""
        waveform = _load_audio(Path(audio_path), self.sample_rate, self.max_samples)
        with torch.no_grad():
            return self._running_features(waveform)

    def predict_running(self, audio_path) -> Tuple[np.ndarray, torch.Tensor]:
        """(times, logits): logits[i] is the prediction after times[i] seconds; the last is whole-track."""
        waveform = _load_audio(Path(audio_path), self.sample_rate, self.max_samples)
        with torch.no_grad():
            return self._running_logits(waveform)

    def predict_logits(self, audio_path) -> torch.Tensor:
        """Whole-track key logits, [N_KEY_CLASSES] -- the last running readout."""
        _times, logits = self.predict_running(audio_path)
        return logits[-1]

    def predict(self, audio_path) -> int:
        """Whole-track key class index."""
        return int(self.predict_logits(audio_path).argmax().item())

    def predict_key(self, audio_path) -> str:
        """Whole-track key as a name, e.g. 'A# minor'."""
        return KEY_CLASSES[self.predict(audio_path)]


class OfflineKeyExtractor(_BaseKeyExtractor):
    """train_offline.py: whole track in one pass; one readout at the end.

    Set max_duration if a long track does not fit in memory.
    """

    def __init__(self, config: dict, checkpoint_path):
        super().__init__(config, checkpoint_path)
        self.geom = offline_geometry(config)
        print(f"Offline: whole track in one pass, pool_factor={self.pool_factor}, "
              f"MERT-layer={config['mert_layer']}")

    def _running_features(
        self, waveform: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        total = waveform.shape[1]
        if total < self.geom.crop_samples:
            # Pad short tracks; the readout time uses the real length.
            waveform = F.pad(waveform, (0, self.geom.crop_samples - total))
        emb = self._extract_in_batches(waveform.unsqueeze(1))          # [1, L, T, D]
        ends = np.array([total / self.sample_rate])
        idx = np.array([emb.shape[2] - 1])
        return ends, idx, emb


class ChunkedKeyExtractor(_BaseKeyExtractor):
    """train_chunked.py: independent chunk_duration chunks; one readout per chunk, at its end.

    A trailing partial chunk is dropped.
    """

    def __init__(self, config: dict, checkpoint_path):
        super().__init__(config, checkpoint_path)
        self.geom = chunked_geometry(config)
        print(f"Chunked: chunk={self.geom.n_frames_per_chunk / self.geom.label_rate:g}s "
              f"({self.geom.chunk_samples} samples), pool_factor={self.pool_factor}, "
              f"MERT-layer={config['mert_layer']}")

    def _running_features(
        self, waveform: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        geom = self.geom
        total = waveform.shape[1]
        if total < geom.chunk_samples:
            # Pad short tracks; the readout time uses the real length.
            n_chunks = 1
            waveform = F.pad(waveform, (0, geom.chunk_samples - total))
            ends = np.array([total / self.sample_rate])
        else:
            n_chunks = (total - geom.chunk_samples) // geom.chunk_stride + 1
            ends = np.minimum(
                np.arange(1, n_chunks + 1) * geom.chunk_stride, total) / self.sample_rate

        windows = waveform.unfold(1, geom.chunk_samples, geom.chunk_stride)[0, :n_chunks]
        emb_flat = self._extract_in_batches(windows.unsqueeze(1))     # [n_chunks, L, T_c, D]
        n_layers, T_c, D = emb_flat.shape[1:]
        emb = emb_flat.permute(1, 0, 2, 3).reshape(1, n_layers, n_chunks * T_c, D)

        # Read out at the last frame of each chunk.
        idx = np.arange(1, n_chunks + 1) * T_c - 1
        return ends, idx, emb


class SlidingKeyExtractor(_BaseKeyExtractor):
    """train_sliding.py: context_duration windows every hop_duration; one readout per hop, at its end."""

    def __init__(self, config: dict, checkpoint_path):
        super().__init__(config, checkpoint_path)
        self.geom = sliding_geometry(config)
        print(f"Sliding: context={self.geom.n_frames_per_context / self.geom.label_rate:g}s, "
              f"hop={self.geom.n_frames_per_hop / self.geom.label_rate:g}s, "
              f"pool_factor={self.pool_factor}, MERT-layer={config['mert_layer']}")

    def _running_features(
        self, waveform: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        geom = self.geom
        total = waveform.shape[1]
        n_hops = max(1, total // geom.hop_samples)
        ends = np.minimum(np.arange(1, n_hops + 1) * geom.hop_samples, total) / self.sample_rate

        # Left pad = empty buffer at stream start; right pad only for tracks shorter than one hop.
        right_pad = max(0, n_hops * geom.hop_samples - total)
        padded = F.pad(waveform, (geom.pad_left, right_pad))
        windows = padded.unfold(1, geom.context_samples, geom.hop_samples)[0, :n_hops]
        emb_flat = self._extract_in_batches(windows.unsqueeze(1))     # [n_hops, L, T_ctx, D]
        n_layers, T_ctx, D = emb_flat.shape[1:]

        emb_hop = emb_flat[:, :, T_ctx - geom.n_frames_per_hop :, :]
        emb = emb_hop.permute(1, 0, 2, 3).reshape(
            1, n_layers, n_hops * geom.n_frames_per_hop, D)

        idx = np.arange(1, n_hops + 1) * geom.n_frames_per_hop - 1
        return ends, idx, emb


def _is_sliding_config(config: dict) -> bool:
    """True for a train_sliding.py config."""
    return "hop_duration" in config and "context_duration" in config


def make_extractor(config: dict, checkpoint_path):
    """Pick the extractor from the config: hop_duration + context_duration -> sliding,
    chunk_duration -> chunked, otherwise offline."""
    if _is_sliding_config(config):
        return SlidingKeyExtractor(config, checkpoint_path)
    if "chunk_duration" in config:
        return ChunkedKeyExtractor(config, checkpoint_path)
    return OfflineKeyExtractor(config, checkpoint_path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class ScoreAccumulator:
    """Accumulates per-track running curves into (summary, records, curves)."""

    def __init__(self, settle_tol: float = 0.0, grid: Sequence[float] = DEFAULT_REPORT_GRID):
        self.settle_tol = settle_tol
        self.grid = grid
        self.per_track: List[dict] = []
        self.grid_scores: List[np.ndarray] = []
        self.grid_correct: List[np.ndarray] = []
        self.records: List[dict] = []
        self.curves: Dict[str, np.ndarray] = {}

    def add(self, track_id: str, reference: int, times: np.ndarray,
            preds: np.ndarray) -> None:
        metrics = convergence_metrics(times, preds, reference, settle_tol=self.settle_tol)
        scores_t, correct_t = curve_at_grid(times, preds, reference, self.grid)
        self.per_track.append(metrics)
        self.grid_scores.append(scores_t)
        self.grid_correct.append(correct_t)
        self.curves[track_id] = np.stack([np.asarray(times, dtype=np.float64),
                                          preds.astype(np.float64)])
        self.records.append({
            "track_id": track_id,
            "reference": KEY_CLASSES[reference],
            "prediction": KEY_CLASSES[int(preds[-1])],
            "n_readouts": int(len(preds)),
            "metrics": metrics,
            "grid_weighted_score": list(scores_t),
        })

    def finish(self) -> Tuple[Dict[str, object], List[dict], Dict[str, np.ndarray]]:
        summary = aggregate_convergence(
            self.per_track, self.grid_scores, self.grid_correct, self.grid)
        return summary, self.records, self.curves


def evaluate_extractor(
    extractor,
    dataset,
    settle_tol: float = 0.0,
    grid: Sequence[float] = DEFAULT_REPORT_GRID,
    desc: str = "eval",
) -> Tuple[Dict[str, object], List[dict], Dict[str, np.ndarray]]:
    """Run an extractor over every track of a KeyDataset.

    Returns (summary, per_track_records, curves {track_id: [2, n_readouts] (times, predictions)}).
    """
    accum = ScoreAccumulator(settle_tol=settle_tol, grid=grid)
    for i in tqdm(range(len(dataset)), desc=desc, unit="track"):
        times, logits = extractor.predict_running(dataset.audio_paths[i])
        accum.add(dataset.track_ids[i], dataset.labels[i], times,
                  logits.argmax(dim=-1).cpu().numpy())
    return accum.finish()


def write_results(
    out_path,
    summary: Dict[str, object],
    records: List[dict],
    config: dict,
    checkpoint_path: str,
    split: str,
) -> None:
    """Summary + per-track records as JSON, for later comparison/plotting."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        # NaN (undefined) -> null.
        json.dump(json_safe({
            "checkpoint": str(checkpoint_path),
            "split": split,
            "config": config,
            "key_classes": KEY_CLASSES,
            "summary": summary,
            "tracks": records,
        }), f, indent=2, allow_nan=False)
    print(f"Wrote per-track results -> {out_path}")


def write_curves(out_path, curves: Dict[str, np.ndarray]) -> None:
    """Running curves as .npz: per track a [2, n_readouts] array (times, predicted class); '/' in ids -> '__'."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **{k.replace("/", "__"): v for k, v in curves.items()})
    print(f"Wrote {len(curves)} running curves -> {out_path}")


def evaluate_main(args, extractor_factory=make_extractor) -> Dict[str, object]:
    """Evaluate a checkpoint on one split and print the summary (shared with infer_streaming.py)."""
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    if args.max_duration is not None:
        config["max_duration"] = args.max_duration

    extractor = extractor_factory(config, args.checkpoint)
    dataset = build_eval_dataset(config, args.split)
    print(f"Evaluating {len(dataset)} tracks of the {args.split} split "
          f"({config['split_dir']})")

    summary, records, curves = evaluate_extractor(
        extractor, dataset, settle_tol=args.settle_tol, desc=f"{args.split}")
    print()
    print(format_summary(summary, name=f"Key[{args.split}]"))

    if args.out:
        write_results(args.out, summary, records, config, args.checkpoint, args.split)
    if args.curves:
        write_curves(args.curves, curves)

    if config.get("use_wandb", False):
        import wandb
        run_name = os.path.basename(os.path.dirname(str(args.checkpoint)))
        wandb.init(
            entity=config.get("wandb_entity"),
            project=config.get("wandb_project", "key-detection"),
            name=f"eval-{run_name}",
            job_type="eval",
        )
        wandb.log({f"{args.split}/{k}": v for k, v in summary.items()
                   if isinstance(v, (int, float))})
        wandb.finish()
    return summary


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """The CLI shared by infer.py and infer_streaming.py."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="training/eval config YAML (use the run's own logs/<run>/config.yaml)")
    parser.add_argument("checkpoint", help="checkpoint .pt written by the matching train_* script")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"),
                        help="split to evaluate (default: %(default)s)")
    parser.add_argument("--settle-tol", type=float, default=0.0,
                        help="fraction of the remaining track (by duration) that may be wrong and "
                             "still count as settled (default: %(default)s = strict)")
    parser.add_argument("--max-duration", type=float, default=None,
                        help="cap each track at this many seconds (overrides config['max_duration'])")
    parser.add_argument("--out", default=None, help="write summary + per-track metrics to this JSON")
    parser.add_argument("--curves", default=None,
                        help="write the raw running curves to this .npz, for plotting")
    return parser


def main():
    args = build_arg_parser(
        "Evaluate an offline/chunked/sliding key-detection checkpoint: whole-track MIREX "
        "weighted score plus the convergence profile."
    ).parse_args()
    evaluate_main(args)


if __name__ == "__main__":
    main()
