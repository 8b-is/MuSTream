"""Chord inference and mir_eval evaluation for train_offline.py / train_chunked.py /
train_sliding.py checkpoints.

Usage (from chord_recognition/):
  python infer.py <config.yaml> <checkpoint.pt>   # mir_eval scores on config['eval_split']

As a module:
  extractor = make_extractor(config, "logs/<run>/best.pt")   # picks the right extractor
  extractor.predict("audio.wav", "output.lab")
"""

import sys, os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torchaudio
import yaml
from transformers import Wav2Vec2FeatureExtractor

import mir_eval

from models import (
    build_chord_postprocessor, build_classifier, emission_block_for_config,
    extract_MERT_features, load_backbone_and_processor, lookahead_frames,
)
from chord_dataloader import (
    kernels_strides_for_backbone, mert_input_for_output, mert_output_len, read_label_file, reduce_chord_label,
)
import pickle

import warnings
warnings.filterwarnings("ignore")


def _load_audio(path: Path, target_sr: int) -> torch.Tensor:
    """Load audio file, convert to mono, resample → [1, T_samples]."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def _extract_in_batches(
    mert: torch.nn.Module,
    processor: Wav2Vec2FeatureExtractor,
    audio_flat: torch.Tensor,      # [N, 1, window_samples]
    sample_rate: int,
    pool_factor: int,
    device: torch.device,
    batch_size: int | None,
) -> torch.Tensor:
    """Run extract_MERT_features on N windows, in sub-batches of batch_size if set."""
    if batch_size is None or audio_flat.shape[0] <= batch_size:
        return extract_MERT_features(mert, processor, audio_flat.to(device), pool_factor)
    chunks = []
    for start in range(0, audio_flat.shape[0], batch_size):
        sub = audio_flat[start : start + batch_size].to(device)
        chunks.append(extract_MERT_features(mert, processor, sub, pool_factor))
    return torch.cat(chunks, dim=0)


def _classify_features(classifier, emb, valid_frames, return_logits=False):
    """Apply a probe to features, trimmed to valid_frames; return logits or argmax labels."""
    logits = classifier(emb)                       # [1, T, vocab_size]
    if return_logits:
        return logits[0][:valid_frames]            # [T_valid, vocab_size]
    preds = logits.argmax(dim=-1)[0].tolist()
    return preds[:min(valid_frames, len(preds))]


def _extract_frames_features(
    waveform: torch.Tensor,
    mert: torch.nn.Module,
    processor: Wav2Vec2FeatureExtractor,
    sample_rate: int,
    pool_factor: int,
    chunk_samples: int,
    chunk_stride: int,
    device: torch.device,
    kernels,
    strides,
    mert_batch_size: int | None = None,
):
    """Encode the waveform as chunk_samples windows at chunk_stride; return (emb, valid_frames).

    kernels/strides must match the running backbone (kernels_strides_for_backbone).
    """
    total_samples = waveform.shape[1]
    n_chunks = max(1, -(-total_samples // chunk_stride))  # ceil div
    pad_len = (n_chunks - 1) * chunk_stride + chunk_samples - total_samples
    audio_padded = torch.nn.functional.pad(waveform, (0, pad_len)) if pad_len > 0 else waveform
    windows = audio_padded.unfold(1, chunk_samples, chunk_stride)  # [1, n_chunks, chunk_samples]
    audio_flat = windows[0].unsqueeze(1)                           # [n_chunks, 1, chunk_samples]

    emb_flat = _extract_in_batches(
        mert, processor, audio_flat, sample_rate, pool_factor, device, mert_batch_size,
    )  # [n_chunks, n_layers, T_c, D]

    n_layers, T_c, D = emb_flat.shape[1:]
    emb = emb_flat.permute(1, 0, 2, 3).reshape(n_layers, n_chunks * T_c, D).unsqueeze(0)

    valid_frames = max(0, mert_output_len(total_samples, kernels=kernels, strides=strides) // pool_factor)
    return emb, valid_frames


def _extract_frames_features_sliding(
    waveform: torch.Tensor,
    mert: torch.nn.Module,
    processor: Wav2Vec2FeatureExtractor,
    sample_rate: int,
    pool_factor: int,
    hop_samples: int,
    context_samples: int,
    n_labels_per_hop: int,
    pad_left: int,
    device: torch.device,
    kernels,
    strides,
    mert_batch_size: int | None = None,
    lookahead: int = 0,
):
    """Sliding-window encoding of a whole track, as in train_sliding.py; return (emb, valid_frames).

    lookahead is in pooled label frames. kernels/strides must match the running backbone.
    """
    total_samples = waveform.shape[1]
    frame_samples = hop_samples // n_labels_per_hop      # samples per pooled label frame
    lookahead_samples = lookahead * frame_samples
    n_hops = max(1, -(-total_samples // hop_samples))  # ceil div
    right_pad = n_hops * hop_samples - total_samples + lookahead_samples
    padded = torch.nn.functional.pad(waveform, (pad_left, right_pad))

    windows = padded.unsqueeze(0).unfold(2, context_samples, hop_samples)  # [1, 1, n_hops, context_samples]
    windows_flat = windows.reshape(n_hops, 1, context_samples)

    batch_size = mert_batch_size or n_hops
    hop_embeddings = []
    for start in range(0, n_hops, batch_size):
        sub = windows_flat[start : start + batch_size].to(device)
        emb_sub = extract_MERT_features(mert, processor, sub, pool_factor)
        hop_end = emb_sub.shape[2] - lookahead      # == -n_labels_per_hop slice when lookahead == 0
        hop_embeddings.append(emb_sub[:, :, hop_end - n_labels_per_hop : hop_end, :])
    emb_flat = torch.cat(hop_embeddings, dim=0)   # [n_hops, n_layers, n_labels_per_hop, D]

    n_layers, _, D = emb_flat.shape[1:]
    emb = emb_flat.permute(1, 0, 2, 3).reshape(n_layers, n_hops * n_labels_per_hop, D).unsqueeze(0)

    valid_frames = max(0, mert_output_len(total_samples, kernels=kernels, strides=strides) // pool_factor)
    return emb, valid_frames


def _merge_frames(
    frame_preds: list[int],
    idx2chord: dict,
    frame_duration: float,
) -> list[tuple[float, float, str]]:
    """Collapse consecutive identical frame labels into (start, end, chord) segments."""
    if not frame_preds:
        return []

    segments = []
    current_label = frame_preds[0]
    seg_start = 0

    for i, label in enumerate(frame_preds[1:], start=1):
        if label != current_label:
            segments.append((seg_start * frame_duration, i * frame_duration, idx2chord[current_label]))
            current_label = label
            seg_start = i

    segments.append((seg_start * frame_duration, len(frame_preds) * frame_duration, idx2chord[current_label]))
    return segments


def _write_lab(segments: list[tuple[float, float, str]], out_path: Path) -> None:
    with open(out_path, "w") as f:
        for start, end, chord in segments:
            f.write(f"{start:.6f}\t{end:.6f}\t{chord}\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class _ChordDecodeMixin:
    """Shared logits -> (start, end, chord) segments step; host sets postproc, idx2chord, frame_duration."""

    def decode_segments(self, logits) -> list[tuple[float, float, str]]:
        """[T, vocab] logits -> segments, through config['decoder']."""
        return _merge_frames(self.postproc.decode(logits), self.idx2chord, self.frame_duration)


class ChordExtractor(_ChordDecodeMixin):
    """Full-crop / chunked extractor: MERT + trained probe; call predict() per audio file.

    Args:
        config:           training config dict
        checkpoint_path:  .pt checkpoint from train_offline.py or train_chunked.py
    """

    def __init__(self, config: dict | Path, checkpoint_path: str | Path):
        self.sample_rate: int   = config["sample_rate"]
        self.pool_factor: int   = config["pool_factor"]
        self.frame_duration: float = config["hop_size"] * config["pool_factor"] / config["sample_rate"]
        self.device = torch.device(config.get("device", "cpu"))

        # Window = chunk_duration if set (chunked), else crop_duration.
        label_rate = config["sample_rate"] / config["hop_size"] / config["pool_factor"]
        if config.get("chunk_duration") is not None:
            n_labels_per_window = int(config["chunk_duration"] * label_rate)
        else:
            n_labels_per_window = round(config["crop_duration"] * label_rate)
        # Conv geometry of the running backbone, as in training.
        self.kernels, self.strides = kernels_strides_for_backbone(config.get("backbone", "pretrained"))
        self.chunk_samples = mert_input_for_output(
            n_labels_per_window * config["pool_factor"], kernels=self.kernels, strides=self.strides
        )
        assert mert_output_len(
            self.chunk_samples, kernels=self.kernels, strides=self.strides
        ) == n_labels_per_window * config["pool_factor"]

        self.chunk_stride = n_labels_per_window * config["pool_factor"] * config["hop_size"]
        self.mert_batch_size = config.get("mert_batch_size")

        # Pooled frames emitted per forward call (for decoder.block: auto); None when offline.
        self.emission_block = emission_block_for_config(config)

        _, self.idx2chord = pickle.load(open(config["vocab_path"], 'rb'))
        self.postproc = build_chord_postprocessor(
            config, len(self.idx2chord), self.device, self.emission_block)

        print(f"Loading {config.get('backbone', 'pretrained')} backbone (MERT_model={config['MERT_model']})...")
        self.mert, self.processor = load_backbone_and_processor(config, self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model_type = config.get("model_type", "LSTM")
        self.classifier = build_classifier(model_type, len(self.idx2chord), config).to(self.device)
        self.classifier.load_state_dict(ckpt["model_state"])
        self.classifier.eval()
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
        print("Pool-factor={}, MERT-layer={}".format(config["pool_factor"], config["mert_layer"]))
        print(f"  {self.postproc.describe()}")

    def extract_features(self, audio_path: str | Path):
        """Run the backbone only: audio_path -> (embeddings, valid_frames)."""
        waveform = _load_audio(Path(audio_path), self.sample_rate)
        with torch.no_grad():
            return _extract_frames_features(
                waveform, self.mert, self.processor,
                self.sample_rate, self.pool_factor, self.chunk_samples, self.chunk_stride,
                self.device, self.kernels, self.strides, mert_batch_size=self.mert_batch_size,
            )

    def logits_from_features(self, emb, valid_frames, classifier=None) -> torch.Tensor:
        """Per-frame logits [T_valid, vocab_size] from extracted features, optionally via another probe."""
        with torch.no_grad():
            return _classify_features(
                self.classifier if classifier is None else classifier,
                emb, valid_frames, return_logits=True)

    def _run(self, audio_path: str | Path, return_logits: bool):
        emb, valid_frames = self.extract_features(audio_path)
        with torch.no_grad():
            return _classify_features(self.classifier, emb, valid_frames, return_logits)

    def predict_logits(self, audio_path: str | Path) -> torch.Tensor:
        """Per-frame logits [T_valid, vocab_size] for one track."""
        return self._run(audio_path, return_logits=True)

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


class SlidingChordExtractor(_ChordDecodeMixin):
    """Sliding-window extractor (as in train_sliding.py); call predict() per audio file.

    Args:
        config:           train_sliding.py config dict
        checkpoint_path:  .pt checkpoint from train_sliding.py
    """

    def __init__(self, config: dict | Path, checkpoint_path: str | Path):
        self.sample_rate: int = config["sample_rate"]
        self.pool_factor: int = config["pool_factor"]
        self.frame_duration: float = config["hop_size"] * config["pool_factor"] / config["sample_rate"]
        self.device = torch.device(config.get("device", "cpu"))

        label_rate = config["sample_rate"] / config["hop_size"] / config["pool_factor"]
        hop_duration = config["hop_duration"]
        context_duration = config["context_duration"]
        n_labels_per_hop = round(hop_duration * label_rate)
        n_labels_per_context = round(context_duration * label_rate)

        self.kernels, self.strides = kernels_strides_for_backbone(config.get("backbone", "pretrained"))
        self.n_labels_per_hop = n_labels_per_hop
        self.hop_samples = n_labels_per_hop * config["pool_factor"] * config["hop_size"]
        self.context_samples = mert_input_for_output(
            n_labels_per_context * config["pool_factor"], kernels=self.kernels, strides=self.strides
        )
        # Window lookahead in label frames, as in train_sliding.py.
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
        assert mert_output_len(
            self.context_samples, kernels=self.kernels, strides=self.strides
        ) == n_labels_per_context * config["pool_factor"]
        self.mert_batch_size = config.get("mert_batch_size")

        # One hop of frames is emitted per MERT pass (for decoder.block: auto).
        self.emission_block = emission_block_for_config(config)

        _, self.idx2chord = pickle.load(open(config["vocab_path"], 'rb'))
        self.postproc = build_chord_postprocessor(
            config, len(self.idx2chord), self.device, self.emission_block)

        print(f"Loading {config.get('backbone', 'pretrained')} backbone (MERT_model={config['MERT_model']})...")
        self.mert, self.processor = load_backbone_and_processor(config, self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Lookahead must match the checkpoint's saved config.
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
        self.classifier = build_classifier(model_type, len(self.idx2chord), config).to(self.device)
        self.classifier.load_state_dict(ckpt["model_state"])
        self.classifier.eval()
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
        print(
            f"Context={context_duration}s, hop={hop_duration}s, "
            f"lookahead={self.lookahead / label_rate:.4g}s ({self.lookahead} frames), "
            f"pool-factor={config['pool_factor']}, MERT-layer={config['mert_layer']}"
        )
        print(f"  {self.postproc.describe()}")

    def extract_features(self, audio_path: str | Path):
        """Run the backbone only (sliding-window): audio_path -> (embeddings, valid_frames)."""
        waveform = _load_audio(Path(audio_path), self.sample_rate)
        with torch.no_grad():
            return _extract_frames_features_sliding(
                waveform, self.mert, self.processor,
                self.sample_rate, self.pool_factor,
                self.hop_samples, self.context_samples, self.n_labels_per_hop, self.pad_left,
                self.device, self.kernels, self.strides,
                mert_batch_size=self.mert_batch_size,
                lookahead=self.lookahead,
            )

    def logits_from_features(self, emb, valid_frames, classifier=None) -> torch.Tensor:
        """Per-frame logits [T_valid, vocab_size] from extracted features, optionally via another probe."""
        with torch.no_grad():
            return _classify_features(
                self.classifier if classifier is None else classifier,
                emb, valid_frames, return_logits=True)

    def _run(self, audio_path: str | Path, return_logits: bool):
        emb, valid_frames = self.extract_features(audio_path)
        with torch.no_grad():
            return _classify_features(self.classifier, emb, valid_frames, return_logits)

    def predict_logits(self, audio_path: str | Path) -> torch.Tensor:
        """Per-frame logits [T_valid, vocab_size] for one track."""
        return self._run(audio_path, return_logits=True)

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


def _is_sliding_config(config: dict) -> bool:
    """True for a train_sliding.py config (has hop_duration and context_duration)."""
    return "hop_duration" in config and "context_duration" in config


def make_extractor(config: dict, checkpoint_path: str | Path):
    """Return SlidingChordExtractor for sliding configs, else ChordExtractor."""
    if _is_sliding_config(config):
        return SlidingChordExtractor(config, checkpoint_path)
    return ChordExtractor(config, checkpoint_path)


# ---------------------------------------------------------------------------
# Evaluation (mir_eval scoring against ground-truth labels)
# ---------------------------------------------------------------------------

# mir_eval chord comparators computed (no *_inv: the vocabulary has no inversions).
CHORD_METRICS = ('majmin', 'sevenths', 'tetrads', 'mirex', 'root', 'triads')

# Headline metrics; the rest are diagnostics.
REPORTED_METRICS = ('majmin', 'sevenths', 'tetrads', 'mirex')


def score_segments(ref_intervals, ref_labels, est_segments, metrics=CHORD_METRICS):
    """Score one song's (start, end, chord) segments against RAW reference labels.

    Reference frames a comparator rejects (-1) are excluded from that metric.
    Returns (scores, scored_durations, total_duration), or None if nothing to score.
    """
    if not est_segments or len(ref_labels) == 0:
        return None
    est_intervals = np.array([[start, end] for start, end, _ in est_segments])
    est_labels = [chord for _, _, chord in est_segments]

    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals, est_labels,
        ref_intervals.min(), ref_intervals.max(),
        mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD,
    )
    intervals, merged_ref_labels, merged_est_labels = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels,
    )
    durations = mir_eval.util.intervals_to_durations(intervals)

    scores, scored = {}, {}
    for metric in metrics:
        comparisons = getattr(mir_eval.chord, metric)(merged_ref_labels, merged_est_labels)
        keep = comparisons >= 0
        scored[metric] = float(durations[keep].sum())
        if keep.any() and scored[metric] > 0:
            scores[metric] = float(
                (comparisons[keep] * durations[keep]).sum() / scored[metric])
    return scores, scored, float(durations.sum())


class ChordScoreAccumulator:
    """Accumulate MACRO chord scores: per-song weighted accuracy, averaged with equal weight per song."""

    def __init__(self, metrics=CHORD_METRICS):
        self.metrics = tuple(metrics)
        self._per_song = {m: [] for m in self.metrics}
        self._scored = {m: 0.0 for m in self.metrics}
        self.total_duration = 0.0
        self.n_songs = 0
        self.n_skipped = 0

    def add(self, ref_intervals, ref_labels, est_segments) -> bool:
        """Score one song; return False (counted as skipped) if it can't be scored."""
        try:
            result = score_segments(ref_intervals, ref_labels, est_segments, self.metrics)
        except Exception as exc:      # an unparseable annotation must not kill a sweep
            print(f"[skip] {type(exc).__name__}: {exc}")
            self.n_skipped += 1
            return False
        if result is None:
            self.n_skipped += 1
            return False
        scores, scored, duration = result
        self.n_songs += 1
        self.total_duration += duration
        for metric, value in scores.items():
            self._per_song[metric].append(value)
        for metric, seconds in scored.items():
            self._scored[metric] += seconds
        return True

    def skip(self) -> None:
        self.n_skipped += 1

    def finish(self) -> dict:
        """Return {'n_songs', 'n_skipped', 'total_duration', metric -> {'macro', 'sd', 'n', 'coverage'}}."""
        summary = {'n_songs': self.n_songs, 'n_skipped': self.n_skipped,
                   'total_duration': self.total_duration}
        for metric in self.metrics:
            values = self._per_song[metric]
            summary[metric] = {
                'macro': float(np.mean(values)) if values else float('nan'),
                'sd': float(np.std(values)) if values else float('nan'),
                'n': len(values),
                'coverage': (self._scored[metric] / self.total_duration)
                            if self.total_duration else float('nan'),
            }
        return summary


def reference_for(record):
    """Return raw (ref_intervals, ref_labels) for one record, or None if unannotated."""
    entries = read_label_file(record['chord_label_path'])
    if not entries:
        return None
    return (np.array([[begin, end] for begin, end, _ in entries]),
            [chord for _, _, chord in entries])


def evaluate_multi_dataset(extractor, records, metrics=CHORD_METRICS):
    """Predict and score every record; return the macro summary dict."""
    accum = ChordScoreAccumulator(metrics)
    for record in tqdm(records):
        reference = reference_for(record)
        if reference is None:
            accum.skip()
            continue
        ref_intervals, ref_labels = reference
        est_segments = extractor.predict_segments(record['audio_path'])
        accum.add(ref_intervals, ref_labels, est_segments)
    return accum.finish()


def format_report(summary, metrics=REPORTED_METRICS) -> str:
    """Machine-parseable summary line of `name=value` pairs."""
    pairs = ' '.join(f'{m}={summary[m]["macro"]:.4f}' for m in metrics)
    return f'Chord (n={summary["n_songs"]}) — {pairs}'


def format_detail(summary, metrics=CHORD_METRICS) -> str:
    """Per-metric macro +- sd, song count and duration coverage."""
    lines = [f'Chord detail (MACRO: mean over songs, equal weight per song; '
             f'{summary["n_songs"]} songs, {summary["n_skipped"]} skipped, '
             f'{summary["total_duration"] / 3600:.2f} h of reference)']
    for metric in metrics:
        stats = summary[metric]
        star = '*' if metric in REPORTED_METRICS else ' '
        lines.append(f'  {star} {metric:<9} {stats["macro"]:.4f} +- {stats["sd"]:.4f}'
                     f'   n={stats["n"]:<5} coverage={stats["coverage"]:.1%}')
    lines.append('  (* = reported; the rest are diagnostics. No *_inv metric is '
                 'listed: this vocabulary has no inversions, so they are capped '
                 'at ~0.85 by construction.)')
    return '\n'.join(lines)


def wandb_metrics(summary, split, metrics=CHORD_METRICS) -> dict:
    """Flat dict for wandb.log."""
    payload = {f'{split}/n': summary['n_songs'],
               f'{split}/n_skipped': summary['n_skipped']}
    for metric in metrics:
        payload[f'{split}/{metric}'] = summary[metric]['macro']
        payload[f'{split}/{metric}_sd'] = summary[metric]['sd']
        payload[f'{split}/{metric}_coverage'] = summary[metric]['coverage']
    return payload


def resolve_eval_split(config) -> str:
    """Return config['eval_split'] ('train', 'val' or 'test'; default 'val')."""
    split = config.get('eval_split', 'val')
    if split not in ('train', 'val', 'test'):
        raise ValueError(f"eval_split must be 'train', 'val' or 'test', got {split!r}")
    return split


def evaluate_and_report(extractor, config, checkpoint_path, script_label='infer.py'):
    """Score `extractor` on config['eval_split'], print the report, and log to wandb if enabled."""
    split = resolve_eval_split(config)
    records = pickle.load(open(os.path.join(config['split_dir'], f'{split}.pkl'), 'rb'))
    if not records:
        print(f"Chord (n=0) — {split}.pkl of {config['split_dir']} is empty")
        return None
    print(f"[{script_label}] scoring {len(records)} songs from "
          f"{config['split_dir']}/{split}.pkl")

    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        import wandb
        run_name = os.path.basename(os.path.dirname(checkpoint_path))
        wandb.init(
            entity=config.get("wandb_entity"),
            project=config.get("wandb_project", "chord-recognition"),
            name=f"eval-{run_name}",
            job_type="eval",
        )

    summary = evaluate_multi_dataset(extractor, records)
    if not summary['n_songs']:
        print(f"Chord (n=0) — nothing scored ({summary['n_skipped']} skipped)")
        if use_wandb:
            wandb.finish()
        return summary

    print(format_report(summary))
    print(format_detail(summary))

    if use_wandb:
        wandb.log(wandb_metrics(summary, split))
        wandb.finish()
    return summary


def evaluate_main(config_path: str, checkpoint_path: str) -> None:
    """Score a train_offline/chunked/sliding checkpoint on config['eval_split']."""
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    extractor = make_extractor(config, checkpoint_path)
    evaluate_and_report(extractor, config, checkpoint_path, 'infer.py')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python infer.py <config.yaml> <checkpoint.pt>")
    evaluate_main(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
