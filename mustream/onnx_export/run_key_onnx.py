#!/usr/bin/env python
"""
Streaming key detection from exported ONNX graphs (no PyTorch): audio in, a
running key estimate out, one chunk at a time through a streaming backbone graph
and a streaming key probe graph.

    python onnx_export/run_key_onnx.py song.wav \
        --backbone <backbone.streaming.onnx> --probe <key_probe.streaming.onnx> \
        [--live] [--curve out_dir/]

Needs only its sibling onnx_stream.py plus numpy, onnxruntime, soundfile, soxr.
The probe emits running logits (row t = estimate from audio 0..t); one estimate
is read per chunk (its last row). Reports the final key, when it was first
reached / settled, and how often the estimate changed.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# onnx_stream.py is looked up beside this file so the script runs from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from onnx_stream import (
        OnnxStreamer,
        backbone_features,
        check_pair,
        describe_pair,
        report_timing,
        stream_windows,
    )
except ImportError as exc:                            # pragma: no cover
    sys.exit(f"{exc}: onnx_stream.py must sit beside run_key_onnx.py")


def transcribe(audio_path, backbone, probe, geom, live=False, live_every=1):
    """One track -> (predictions, times, per-chunk ms); predictions[i] is the key
    reported after times[i] seconds of audio. Resets both graphs first."""
    backbone.reset()
    probe.reset()

    spec = probe.spec
    names = spec.get("key_labels")
    samples_per_block = spec["label_frames"] * spec["pool_factor"] * spec["hop_size"]
    block_duration = samples_per_block / spec["sample_rate"]

    preds, ms, n_real = [], [], 0
    for chunk, real in stream_windows(audio_path, geom):
        n_real += real
        t0 = time.perf_counter()
        features = backbone_features(backbone, chunk)
        (logits,) = probe(features=features)
        ms.append((time.perf_counter() - t0) * 1000)

        # One readout per chunk: with a block-causal backbone every frame of the
        # chunk has seen audio to the chunk's end, so only the last row is an
        # honest estimate at its timestamp (as in key_detection/infer_streaming.py).
        pred = int(logits[0, -1].argmax(-1))
        if len(preds) % live_every == 0 and live:
            t = (len(preds) + 1) * block_duration
            shown = names[pred] if names else pred
            print(f"  {t:7.2f}s  {shown:<10} [{ms[-1]:5.1f} ms]", flush=True)
        preds.append(pred)

    # Drop chunks produced only by the zero-padded tail.
    preds = preds[:n_real // samples_per_block]
    # Time = audio consumed at that readout (the chunk's right edge).
    times = (np.arange(len(preds)) + 1) * block_duration
    return preds, times, np.array(ms)


def summarize(preds, times, names):
    """Ground-truth-free convergence summary: final key, first/settle time, flips."""
    if not preds:
        return None
    final = preds[-1]
    arr = np.asarray(preds)
    # Settled = the last time the estimate switched to the final key.
    settled_idx = len(arr) - 1
    while settled_idx > 0 and arr[settled_idx - 1] == final:
        settled_idx -= 1
    first_idx = int(np.flatnonzero(arr == final)[0])
    flips = int(np.count_nonzero(np.diff(arr)))
    span = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    return {
        "final": final,
        "final_name": names[final] if names else str(final),
        "t_first": float(times[first_idx]),
        "t_settle": float(times[settled_idx]),
        "n_flips": flips,
        "flips_per_min": (flips / span * 60.0) if span > 0 else 0.0,
        "duration": float(times[-1]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Streaming key detection from exported ONNX graphs. No PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", nargs="+", help="audio file(s); any format libsndfile reads")
    parser.add_argument("--backbone", required=True, help="a *.streaming.onnx backbone graph")
    parser.add_argument("--probe", required=True, help="a *.streaming.onnx key probe graph")
    parser.add_argument("--live", action="store_true",
                        help="print the running estimate as it is revised")
    parser.add_argument("--live-every", type=int, default=5,
                        help="--live only: print every Nth chunk (1 = every 200 ms)")
    parser.add_argument("--curve", default=None,
                        help="directory to write the full running curve into, as TSV "
                             "(seconds, key) -- the input to a convergence plot")
    parser.add_argument("--threads", type=int, default=None,
                        help="onnxruntime intra-op threads (default: the runtime's choice)")
    parser.add_argument("--providers", default=None,
                        help="comma-separated onnxruntime execution providers, e.g. "
                             "CoreMLExecutionProvider,CPUExecutionProvider")
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()] if args.providers else None

    backbone = OnnxStreamer(args.backbone, providers, args.threads)
    probe = OnnxStreamer(args.probe, providers, args.threads)
    geom = check_pair(backbone, probe, "key_probe")

    p = probe.spec
    names = p.get("key_labels")
    chunk_ms = describe_pair(backbone, probe, geom, args.backbone, args.probe,
                             extra=f", {p['n_classes']} classes, running readout")
    if p.get("train_readout") == "final":
        print("[pipe]   train_readout=final: the run supervised only each crop's last frame, "
              "so early readouts are unsupervised prefixes")

    if args.curve:
        Path(args.curve).mkdir(parents=True, exist_ok=True)

    for path in args.audio:
        print(f"\n[pipe] {path}")
        preds, times, ms = transcribe(path, backbone, probe, geom,
                                      live=args.live, live_every=max(1, args.live_every))
        if not preds:
            print("[pipe]   no audio decoded")
            continue

        # One readout per chunk, so the rate here is the chunk rate.
        report_timing(ms, chunk_ms, len(preds), p["label_rate"] / p["label_frames"])
        s = summarize(preds, times, names)
        print(f"[pipe]   key: {s['final_name']}")
        print(f"[pipe]   first said it at {s['t_first']:.1f}s, settled on it at "
              f"{s['t_settle']:.1f}s of {s['duration']:.1f}s")
        print(f"[pipe]   {s['n_flips']} changes of mind ({s['flips_per_min']:.1f}/min)")

        if args.curve:
            out = Path(args.curve) / (Path(path).stem + ".key.tsv")
            with open(out, "w") as f:
                for t, k in zip(times, preds):
                    f.write(f"{t:.4f}\t{names[k] if names else k}\n")
            print(f"[pipe]   -> {out}")


if __name__ == "__main__":
    main()
