"""
Compare beat post-processors on one checkpoint's test-split logits: minimal,
offline dbn and dbn_blocksync (per --blocks), with accuracy and measured per-beat
emission latency. Logits are computed once and can be cached with --cache.

Usage (from beat_tracking/):
  python sweep_postproc.py logs/<run>/config.yaml logs/<run>/best.pt [--cache logits.npz]
  python sweep_postproc.py <config> <ckpt> --blocks 1,auto,30
"""

import argparse
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import mir_eval
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from beat_dataloader import BeatDataset
from infer import precision_recall_f_measure
from models import emission_block_for_config
from postprocessor import DBN_TRANSITION_LAMBDA, DecoderBank

import warnings
warnings.filterwarnings("ignore")

# 'auto' = the backbone's own emission block; 1 = frame-by-frame decoding.
DEFAULT_BLOCKS = "1,auto"

# Per-worker DecoderBank, built once (DBN state spaces are slow to construct).
_WORKER = {}


def _init_worker(fps, specs, deploy_block, transition_lambda):
    _WORKER["bank"] = DecoderBank(specs, fps, deploy_block,
                                  transition_lambda=transition_lambda)


def _decode_track(task):
    """Run every postprocessor on one track; returns {method: (beats, downbeats, emit_times)}."""
    beat_logits, downbeat_logits, oracle_meter, oracle_labels = task
    bank = _WORKER["bank"]
    out = bank(beat_logits, downbeat_logits)
    if oracle_meter:
        # Same decoding with the bar length pinned to the offline DBN's choice (not streamable).
        pinned = bank.best_meter(beat_logits, downbeat_logits)
        for label, value in bank(beat_logits, downbeat_logits, hmm_index=pinned,
                                 labels=oracle_labels).items():
            out[f"oracle_{label}"] = value
    return out


def extract_logits(config, checkpoint_path, dataset, cache_path, num_tracks):
    """Per-frame [beat, downbeat] logits for every test track (any checkpoint type), optionally cached."""
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        if len(cached["fids"]) >= num_tracks:
            print(f"Loaded cached logits for {len(cached['fids'])} tracks from {cache_path}")
            return list(cached["logits"])[:num_tracks], list(cached["fids"])[:num_tracks]
        print(f"{cache_path} only holds {len(cached['fids'])}/{num_tracks} tracks, re-extracting")

    if config.get("backbone", "").endswith("streaming"):
        from infer_streaming import StreamingBeatExtractor
        extractor = StreamingBeatExtractor(config, checkpoint_path)
    else:
        from infer import make_extractor
        extractor = make_extractor(config, checkpoint_path)
    extractor.postproc = lambda beat, downbeat: (
        beat.numpy().astype(np.float32), downbeat.numpy().astype(np.float32)
    )

    logits, fids = [], []
    for idx in tqdm(range(num_tracks), desc="Model forward"):
        waveform, _, _, fid = dataset[idx]
        beat_logits, downbeat_logits = extractor.predict(waveform)
        logits.append(np.stack([beat_logits, downbeat_logits]))
        fids.append(fid)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        # Fill an object array explicitly so equal-length tracks don't collapse to one array.
        boxed = np.empty(len(logits), dtype=object)
        for i, track in enumerate(logits):
            boxed[i] = track
        np.savez_compressed(cache_path, logits=boxed, fids=np.array(fids))
        print(f"Cached logits to {cache_path}")
    return logits, fids


def score(per_track, key, ground_truth):
    """Mean mir_eval + P/R/F for one method."""
    rows = {"beat": [], "downbeat": []}
    for track, (beats_gt, downbeats_gt) in zip(per_track, ground_truth):
        beat_pred, downbeat_pred, _emit = track[key]
        for name, gt, pred in (
            ("beat", beats_gt, beat_pred), ("downbeat", downbeats_gt, downbeat_pred)
        ):
            if len(gt) == 0:
                continue
            scores = mir_eval.beat.evaluate(gt, np.asarray(pred))
            precision, recall, f1 = precision_recall_f_measure(gt, np.asarray(pred))
            rows[name].append((
                scores["F-measure"], scores["Correct Metric Level Total"],
                scores["Any Metric Level Total"], precision, recall, f1,
            ))
    return {
        name: dict(zip(("f_measure", "cmlt", "amlt", "precision", "recall", "f1"),
                       np.mean(vals, axis=0)))
        for name, vals in rows.items() if vals
    }


def latency(per_track, key):
    """Mean and 95th-percentile emission latency (announce time - beat time, seconds)."""
    values = []
    for track in per_track:
        beats, _downbeats, emit = track[key]
        values.append(np.asarray(emit, dtype=np.float64) - np.asarray(beats, dtype=np.float64))
    values = np.concatenate(values) if values else np.zeros(0)
    if not len(values):
        return float("nan"), float("nan")
    return float(values.mean()), float(np.percentile(values, 95))


def parse_blocks(spec, auto_block):
    """'1,auto,30' -> sorted unique block sizes; 'auto' = the backbone's block (error for offline)."""
    blocks = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "auto":
            if not auto_block:
                raise SystemExit(
                    "--blocks includes 'auto', but this config's backbone emits no block "
                    "per forward call (full-crop/offline) -- name the blocks explicitly, "
                    "or drop 'auto'")
            blocks.add(int(auto_block))
        else:
            blocks.add(int(item))
    return sorted(blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--blocks", default=DEFAULT_BLOCKS,
                        help="comma-separated block-synchronous block sizes in FRAMES; "
                             "'auto' is the backbone's own emission block "
                             f"(default: {DEFAULT_BLOCKS})")
    parser.add_argument("--cache", default=None,
                        help="path to an .npz of cached logits (written if absent)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="only use the first N test tracks (smoke tests)")
    parser.add_argument("--transition-lambda", type=float, default=DBN_TRANSITION_LAMBDA,
                        help="the DBN's tempo-change penalty; its optimum can differ "
                             "between block-synchronous and whole-track decoding "
                             f"(default: {DBN_TRANSITION_LAMBDA:g})")
    parser.add_argument("--oracle-meter", action="store_true",
                        help="also report block-synchronous decoding with the bar length pinned to "
                             "the offline DBN's choice -- not streamable, but it attributes "
                             "the remaining gap to meter selection")
    parser.add_argument("--csv", default=None, help="also write the result table here")
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    fps = config["sample_rate"] / config["hop_size"] / config["pool_factor"]
    # Frames the backbone emits per forward call; latencies are measured against it.
    auto_block = emission_block_for_config(config)
    blocks = parse_blocks(args.blocks, auto_block)
    deploy_block = auto_block or 1

    dataset = BeatDataset(
        audio_dir=config["audio_dir"], annotation_dir=config["annotation_dir"],
        split_dir=config["split_dir"], split="test", mode="test",
        sr=config["sample_rate"], fps=fps, pool_factor=config["pool_factor"],
        train_ratio=config["train_ratio"], val_ratio=config["val_ratio"],
        split_seed=config["split_seed"], audio_ext=config.get("audio_ext", "wav"),
    )
    num_tracks = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    print(f"Test tracks: {num_tracks}  |  fps: {fps:g}")
    print(f"Blocks (frames -> s): " + ", ".join(f"{b}->{b / fps:.3g}" for b in blocks))
    print(f"Backbone emission block: "
          + (f"{auto_block} frames ({auto_block / fps:.3g} s)" if auto_block
             else "none (full-crop backbone; latency measured per frame)"))

    logits, _ = extract_logits(config, args.checkpoint, dataset, args.cache, num_tracks)
    ground_truth = [(dataset[i][1], dataset[i][2]) for i in range(num_tracks)]
    # nothing below this point needs the GPU
    torch.cuda.empty_cache()

    # (label, kind, param) per row, in the order the table prints them.
    specs = ([("minimal", "minimal", 0), ("dbn", "offline", 0)]
             + [(f"block{b}", "blocksync", b) for b in blocks])
    oracle_labels = [label for label, kind, _p in specs if kind == "blocksync"]
    tasks = [(track[0], track[1], args.oracle_meter, oracle_labels) for track in logits]
    # 'spawn' (not 'fork') to avoid deadlocks after running CUDA in the parent.
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker,
                             initargs=(fps, specs, deploy_block,
                                       args.transition_lambda)) as pool:
        per_track = list(tqdm(pool.map(_decode_track, tasks), total=len(tasks),
                              desc="Decoding"))

    methods = (
        [("minimal", "minimal"), ("dbn (offline)", "dbn")]
        + [(f"dbn_blocksync blk={b}" + (" (backbone's own)" if b == auto_block else ""),
            f"block{b}") for b in blocks]
    )
    if args.oracle_meter:
        methods += [
            (f"  ^ blk={b}, meter pinned", f"oracle_block{b}") for b in blocks
        ]

    results = {label: score(per_track, key, ground_truth)
               for label, key in methods}

    offline = results["dbn (offline)"]
    minimal = results["minimal"]
    latencies = {label: latency(per_track, key) for label, key in methods}
    header = (f"{'postprocessor':<32}{'lat ms':>8}{'p95':>7}{'beat F':>9}{'beat CMLt':>11}"
              f"{'beat AMLt':>11}{'db F':>9}{'db CMLt':>10}{'db AMLt':>10}"
              f"{'Δbeat/min':>10}{'Δdb/min':>9}{'Δbeat/off':>10}")
    print("\n" + header)
    print("-" * len(header))
    rows = []
    for label, _ in methods:
        r = results[label]
        beat, downbeat = r["beat"], r["downbeat"]
        lat_mean, lat_p95 = latencies[label]
        d_beat_min = beat["f_measure"] - minimal["beat"]["f_measure"]
        d_downbeat_min = downbeat["f_measure"] - minimal["downbeat"]["f_measure"]
        d_beat_off = beat["f_measure"] - offline["beat"]["f_measure"]
        d_downbeat_off = downbeat["f_measure"] - offline["downbeat"]["f_measure"]
        print(f"{label:<32}{lat_mean * 1000:>8.0f}{lat_p95 * 1000:>7.0f}"
              f"{beat['f_measure']:>9.4f}{beat['cmlt']:>11.4f}{beat['amlt']:>11.4f}"
              f"{downbeat['f_measure']:>9.4f}{downbeat['cmlt']:>10.4f}{downbeat['amlt']:>10.4f}"
              f"{d_beat_min:>+10.4f}{d_downbeat_min:>+9.4f}{d_beat_off:>+10.4f}")
        rows.append([label, lat_mean, lat_p95,
                     beat["f_measure"], beat["cmlt"], beat["amlt"],
                     downbeat["f_measure"], downbeat["cmlt"], downbeat["amlt"],
                     d_beat_min, d_downbeat_min, d_beat_off, d_downbeat_off])
    print("-" * len(header))
    print("lat = mean emission latency per beat: when it could be ANNOUNCED (the "
          "backbone hands\n      frames over "
          f"{deploy_block} at a time = {deploy_block / fps:.3g} s) minus when it happens.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["postprocessor", "latency_mean_s", "latency_p95_s",
                             "beat_f", "beat_cmlt", "beat_amlt",
                             "downbeat_f", "downbeat_cmlt", "downbeat_amlt",
                             "delta_beat_f_vs_minimal", "delta_downbeat_f_vs_minimal",
                             "delta_beat_f_vs_offline", "delta_downbeat_f_vs_offline"])
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
