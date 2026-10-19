#!/usr/bin/env python
"""
Streaming beat/downbeat tracking from exported ONNX graphs: audio in, beat times
out, one chunk at a time through a streaming backbone graph and a streaming beat
probe graph.

    python onnx_export/run_beat_onnx.py song.wav \
        --backbone <backbone.streaming.onnx> --probe <beat_probe.streaming.onnx> \
        [--postproc minimal|dbn_blocksync|dbn] [--live] [--beats out_dir/]

Per chunk: backbone -> features -> probe -> label_frames (beat, downbeat) logits.
The probe graph has no decoder; the logits are turned into times here:

  --postproc minimal        (default) numpy peak picker equal to
                            beat_tracking/postprocessor.py's `minimal`; needs
                            only numpy, onnxruntime, soundfile, soxr.
  --postproc dbn_blocksync  beat_tracking/postprocessor.py's streaming DBN
                            (needs torch, madmom and this repo).
  --postproc dbn            the offline DBN (same dependencies).
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
    sys.exit(f"{exc}: onnx_stream.py must sit beside run_beat_onnx.py")

# postprocessor.py's `minimal` rule: a local maximum over +/- 3 frames whose
# activation is over 0.5, i.e. whose logit is over 0 (no sigmoid needed).
PEAK_WINDOW = 7
PEAK_HALF = PEAK_WINDOW // 2
DEDUP_WIDTH = 1


def pick_peaks(logits, fps):
    """[T] logits for one channel -> peak times in seconds (local maxima over
    PEAK_WINDOW frames with logit > 0; adjacent peaks merged to their mean)."""
    T = logits.shape[0]
    if T == 0:
        return np.zeros(0)
    # Centred sliding max; -inf padding so edge frames see a short window.
    padded = np.pad(logits, PEAK_HALF, mode="constant", constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, PEAK_WINDOW)
    is_peak = (logits >= windows.max(axis=-1)) & (logits > 0.0)

    frames = np.flatnonzero(is_peak)
    if frames.size == 0:
        return np.zeros(0)
    # Merge peaks no more than DEDUP_WIDTH apart into their running mean.
    merged, p, c = [], float(frames[0]), 1
    for q in frames[1:]:
        if q - p <= DEDUP_WIDTH:
            c += 1
            p += (q - p) / c
        else:
            merged.append(p)
            p, c = float(q), 1
    merged.append(p)
    return np.array(merged) / fps


def snap_downbeats(beats, downbeats):
    """Move every downbeat onto the nearest detected beat, then drop duplicates."""
    if beats.size == 0 or downbeats.size == 0:
        return downbeats
    snapped = beats[np.argmin(np.abs(beats[None, :] - downbeats[:, None]), axis=1)]
    return np.unique(snapped)


def decode_with_repo(activations, fps, postproc_type, dbn_block):
    """Decode [T, 2] activations with beat_tracking/postprocessor.py's
    Postprocessor (imported lazily: needs torch, madmom and this repo)."""
    import torch

    # this file lives in onnx_export/, one level below mustream/
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "beat_tracking"))
    from postprocessor import Postprocessor            # noqa: E402

    pp = Postprocessor(type=postproc_type, fps=fps, dbn_block=dbn_block)
    beat = torch.from_numpy(activations[:, 0].astype(np.float32))
    downbeat = torch.from_numpy(activations[:, 1].astype(np.float32))
    beats, downbeats = pp(beat, downbeat)
    return np.asarray(beats), np.asarray(downbeats)


def transcribe(audio_path, backbone, probe, geom, live=False):
    """One track -> (activations [T, 2], per-chunk ms). Resets both graphs first."""
    backbone.reset()
    probe.reset()

    spec = probe.spec
    fps = spec["label_rate"]
    samples_per_frame = spec["pool_factor"] * spec["hop_size"]

    chunks, ms, n_real = [], [], 0
    for chunk, real in stream_windows(audio_path, geom):
        n_real += real
        t0 = time.perf_counter()
        features = backbone_features(backbone, chunk)
        (logits,) = probe(features=features)
        ms.append((time.perf_counter() - t0) * 1000)
        chunks.append(logits[0])

        if live:
            # Peaks picked from this chunk alone (the final pass re-picks over
            # the whole curve, so boundary peaks can differ).
            t0_chunk = (len(chunks) - 1) * logits.shape[1] / fps
            hits = pick_peaks(logits[0][:, 0], fps)
            if hits.size:
                print(f"  {t0_chunk:7.2f}s  beat x{hits.size}   [{ms[-1]:5.1f} ms]", flush=True)

    activations = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2))
    # Drop frames produced only by the zero-padded tail.
    return activations[:n_real // samples_per_frame], np.array(ms)


def write_beats(beats, downbeats, out_path):
    """One line per beat: time, then `1` for a downbeat or `x` otherwise."""
    marks = {round(float(t), 4): "1" for t in downbeats}
    with open(out_path, "w") as f:
        for t in sorted(set(np.round(np.concatenate([beats, downbeats]), 4))):
            f.write(f"{t:.4f}\t{marks.get(t, 'x')}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Streaming beat/downbeat tracking from exported ONNX graphs. No PyTorch "
                    "unless --postproc asks for the DBN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", nargs="+", help="audio file(s); any format libsndfile reads")
    parser.add_argument("--backbone", required=True, help="a *.streaming.onnx backbone graph")
    parser.add_argument("--probe", required=True, help="a *.streaming.onnx beat probe graph")
    parser.add_argument("--postproc",
                        choices=["minimal", "dbn_blocksync", "dbn"],
                        default="minimal",
                        help="minimal: the inline peak picker, no extra dependencies, 40 ms "
                             "lookahead. The dbn_* options need madmom and this repo; "
                             "dbn_blocksync is the streaming one, plain dbn is OFFLINE")
    parser.add_argument("--beats", default=None, help="directory to write .beats files into")
    parser.add_argument("--activations", default=None,
                        help="directory to write the raw [T, 2] activations into, as .npy -- "
                             "for running a decoder separately")
    parser.add_argument("--live", action="store_true",
                        help="print each chunk's own peaks the moment it is decoded")
    parser.add_argument("--threads", type=int, default=None,
                        help="onnxruntime intra-op threads (default: the runtime's choice)")
    parser.add_argument("--providers", default=None,
                        help="comma-separated onnxruntime execution providers, e.g. "
                             "CoreMLExecutionProvider,CPUExecutionProvider")
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()] if args.providers else None

    backbone = OnnxStreamer(args.backbone, providers, args.threads)
    probe = OnnxStreamer(args.probe, providers, args.threads)
    geom = check_pair(backbone, probe, "beat_probe")

    p = probe.spec
    fps = p["label_rate"]
    chunk_ms = describe_pair(backbone, probe, geom, args.backbone, args.probe,
                             extra=f", channels {p['channels']}, {p['activation']}s")
    if args.postproc == "minimal":
        print(f"[pipe]   postproc: minimal peak picking (+/-{PEAK_HALF} frames = "
              f"{PEAK_HALF / fps * 1000:.0f} ms lookahead, threshold 0.5)")
    else:
        block = p.get("emission_block")
        print(f"[pipe]   postproc: {args.postproc} via beat_tracking/postprocessor.py "
              f"(block={block}) -- needs madmom and this repo")
        if args.postproc == "dbn":
            print("[pipe]   NOTE: plain 'dbn' is offline -- nothing is emitted until the track ends")

    for d in (args.beats, args.activations):
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)

    for path in args.audio:
        print(f"\n[pipe] {path}")
        activations, ms = transcribe(path, backbone, probe, geom, live=args.live)
        if not activations.size:
            print("[pipe]   no audio decoded")
            continue

        report_timing(ms, chunk_ms, len(activations), fps)

        if args.postproc == "minimal":
            beats = pick_peaks(activations[:, 0], fps)
            downbeats = snap_downbeats(beats, pick_peaks(activations[:, 1], fps))
        else:
            beats, downbeats = decode_with_repo(
                activations, fps, args.postproc, p.get("emission_block"))

        duration = len(activations) / fps
        tempo = 60.0 / np.median(np.diff(beats)) if beats.size > 1 else float("nan")
        print(f"[pipe]   {beats.size} beats, {downbeats.size} downbeats over "
              f"{duration:.1f}s (median tempo {tempo:.1f} BPM)")
        if beats.size:
            print("[pipe]   first few: "
                  + " ".join(f"{t:.2f}" for t in beats[:8]))

        if args.beats:
            out = Path(args.beats) / (Path(path).stem + ".beats")
            write_beats(beats, downbeats, out)
            print(f"[pipe]   -> {out}")
        if args.activations:
            out = Path(args.activations) / (Path(path).stem + ".npy")
            np.save(out, activations)
            print(f"[pipe]   -> {out}")


if __name__ == "__main__":
    main()
