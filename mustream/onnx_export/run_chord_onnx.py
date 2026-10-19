#!/usr/bin/env python
"""
Streaming chord recognition from exported ONNX graphs (no PyTorch): audio in,
chord labels out, one chunk at a time through a streaming backbone graph and a
streaming chord probe graph.

    python onnx_export/run_chord_onnx.py song.wav \
        --backbone <backbone.streaming.onnx> --probe <chord_probe.streaming.onnx> \
        [--live] [--lab out_dir/]

Needs only its sibling onnx_stream.py plus numpy, onnxruntime, soundfile, soxr.
Per chunk (e.g. 0.2 s): backbone -> [1, L, chunk_frames, D] features -> probe ->
label_frames chord labels (the graph's `tags` with a blocksync decoder, else
per-frame argmax of the logits). This is the Python reference for the
streaming-demo/ browser pipeline.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# onnx_stream.py is looked up beside this file so the script runs from any cwd.
# Names are re-exported for callers that import them from here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from onnx_stream import (                         # noqa: F401
        Geometry,
        OnnxStreamer,
        backbone_features,
        describe_pair,
        report_timing,
        stream_chunks,
        stream_windows,
    )
    from onnx_stream import check_pair as _check_pair
except ImportError as exc:                            # pragma: no cover
    sys.exit(f"{exc}: onnx_stream.py must sit beside run_chord_onnx.py")


def check_pair(backbone, probe, _unused=None):
    """onnx_stream.check_pair for a chord probe (the optional third argument is ignored)."""
    return _check_pair(backbone, probe, "chord_probe")


# ---------------------------------------------------------------------------
# labels -> segments
# ---------------------------------------------------------------------------

def merge_frames(labels, idx2chord, frame_duration):
    """Collapse consecutive identical frame labels into (start, end, chord) segments."""
    if not labels:
        return []
    segments, current, start = [], labels[0], 0
    for i, label in enumerate(labels[1:], start=1):
        if label != current:
            segments.append((start * frame_duration, i * frame_duration, idx2chord[current]))
            current, start = label, i
    segments.append((start * frame_duration, len(labels) * frame_duration, idx2chord[current]))
    return segments


def write_lab(segments, out_path):
    """Write segments as a tab-separated .lab file."""
    with open(out_path, "w") as f:
        for start, end, chord in segments:
            f.write(f"{start:.6f}\t{end:.6f}\t{chord}\n")


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

def transcribe(audio_path, backbone, probe, geom, live=False, argmax_fallback=True):
    """One track -> (labels, per-chunk ms, frame_duration, idx2chord). Resets
    both graphs first."""
    backbone.reset()
    probe.reset()

    spec = probe.spec
    labels_out = spec["tags_output"] is not None
    idx2chord = spec.get("chord_labels")
    frame_duration = spec["pool_factor"] * spec["hop_size"] / spec["sample_rate"]
    samples_per_label = spec["pool_factor"] * spec["hop_size"]
    labels, times, n_real = [], [], 0
    for chunk, real in stream_windows(audio_path, geom):
        n_real += real
        t0 = time.perf_counter()
        features = backbone_features(backbone, chunk)
        outs = probe(features=features)
        if labels_out:
            # tags_valid marks which tags are final (all ones for blocksync).
            chunk_labels = [int(t) for t, ok in zip(outs[1][0], outs[2][0]) if ok]
        else:
            chunk_labels = [int(i) for i in outs[0][0].argmax(-1)] if argmax_fallback else []
        times.append((time.perf_counter() - t0) * 1000)

        if live and chunk_labels:
            t = len(labels) * frame_duration
            names = " ".join(idx2chord[i] if idx2chord else str(i) for i in chunk_labels)
            print(f"  {t:7.2f}s  {names}   [{times[-1]:5.1f} ms]", flush=True)
        labels += chunk_labels

    # Drop labels produced only by the zero-padded tail.
    labels = labels[:n_real // samples_per_label]
    return labels, np.array(times), frame_duration, idx2chord


def main():
    parser = argparse.ArgumentParser(
        description="Streaming chord recognition from exported ONNX graphs. No PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", nargs="+", help="audio file(s); any format libsndfile reads")
    parser.add_argument("--backbone", required=True, help="a *.streaming.onnx backbone graph")
    parser.add_argument("--probe", required=True, help="a *.streaming.onnx chord probe graph")
    parser.add_argument("--lab", default=None, help="directory to write .lab files into")
    parser.add_argument("--live", action="store_true",
                        help="print each chunk's labels as soon as it is decoded")
    parser.add_argument("--threads", type=int, default=None,
                        help="onnxruntime intra-op threads (default: the runtime's choice)")
    parser.add_argument("--providers", default=None,
                        help="comma-separated onnxruntime execution providers, e.g. "
                             "CoreMLExecutionProvider,CPUExecutionProvider")
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()] if args.providers else None

    backbone = OnnxStreamer(args.backbone, providers, args.threads)
    probe = OnnxStreamer(args.probe, providers, args.threads)
    geom = check_pair(backbone, probe)

    p = probe.spec
    extra = (f", decoder={p['decoder']}"
             + (f" (lambda={p['self_bias']:g} nats)" if p["decoder"] != "none" else "")
             + f", vocab {p['vocab_size']}")
    chunk_ms = describe_pair(backbone, probe, geom, args.backbone, args.probe, extra=extra)
    if p["decoder"] == "none":
        print("[pipe]   no CRF in the graph -- labels are per-frame argmax of the logits")

    if args.lab:
        os.makedirs(args.lab, exist_ok=True)

    for path in args.audio:
        print(f"\n[pipe] {path}")
        labels, times, frame_duration, idx2chord = transcribe(
            path, backbone, probe, geom, live=args.live)
        if not labels:
            print("[pipe]   no audio decoded")
            continue

        report_timing(times, chunk_ms, len(labels), 1.0 / frame_duration)

        if idx2chord:
            segments = merge_frames(labels, idx2chord, frame_duration)
            print(f"[pipe]   {len(segments)} segments; first few:")
            for start, end, chord in segments[:5]:
                print(f"           {start:7.2f} - {end:7.2f}  {chord}")
            if args.lab:
                out = Path(args.lab) / (Path(path).stem + ".lab")
                write_lab(segments, out)
                print(f"[pipe]   -> {out}")
        else:
            print("[pipe]   the probe spec carries no chord names; labels are indices only")


if __name__ == "__main__":
    main()
