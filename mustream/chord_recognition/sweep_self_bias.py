#!/usr/bin/env python
"""Sweep the self-bias CRF's lambda against transcript stability (flips/min, blips)
and accuracy, using exported ONNX backbone + chord probe pairs.

Emissions are computed once per pair and re-decoded in numpy for every lambda.
By default the numpy decode is checked against the graph's own tags at its
exported lambda (--no-verify skips); --check-torch also checks it against
models.decode_block_synchronous.

    python chord_recognition/sweep_self_bias.py \
        --pair 26M=<backbone.streaming.onnx>,<probe.streaming.onnx> \
        --pair 330M=<backbone>,<probe> \
        --assignment <split_assignment.json> --dataset <name> --split test --examples 4
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "chord_recognition"))

from onnx_export.run_chord_onnx import (                                  # noqa: E402
    OnnxStreamer, backbone_features, check_pair, stream_windows,
)

NO_CHORD = "N"


_reduce_impl = None


def _reduce_chord_label(label):
    """chord_dataloader.reduce_chord_label, with a mir_eval-only fallback when torch is absent."""
    global _reduce_impl
    if _reduce_impl is not None:
        return _reduce_impl(label)
    try:
        from chord_dataloader import reduce_chord_label as impl
    except ImportError:
        import mir_eval
        _SEVENTH = {'maj': {None: 'maj', 'b7': '7', '7': 'maj7'},
                    'min': {None: 'min', 'b7': 'min7', '7': 'minmaj7'},
                    'dim': {None: 'dim', 'b7': 'hdim7', 'bb7': 'dim7'},
                    'aug': {None: 'aug'}, 'sus2': {None: 'sus2'}, 'sus4': {None: 'sus4'}}
        _FUSED = {'maj7', 'min7', '7', 'dim7', 'hdim7', 'minmaj7'}
        _SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        _COLLAPSE = {'minmaj7': 'min', 'sus2': 'maj'}

        def impl(chord_label):
            if chord_label == mir_eval.chord.NO_CHORD or 'X' in chord_label:
                return NO_CHORD
            root, quality, degrees, _bass = mir_eval.chord.split(chord_label)
            root = _SHARP[mir_eval.chord.pitch_class_to_semitone(root)]
            if quality in _FUSED:
                return f'{root}:{_COLLAPSE.get(quality, quality)}'
            qmap = _SEVENTH.get(quality)
            if qmap is None:
                return NO_CHORD
            seventh = next((d for d in ('b7', '7', 'bb7') if d in degrees), None)
            reduced = qmap.get(seventh, qmap[None])
            return f'{root}:{_COLLAPSE.get(reduced, reduced)}'
    _reduce_impl = impl
    return _reduce_impl(label)


# ---------------------------------------------------------------------------
# the decoder
# ---------------------------------------------------------------------------

def decode_blocksync(emissions, biases, block):
    """Numpy block-synchronous self-bias Viterbi: [T, V] emissions -> [K, T] tags, one row per lambda.

    Emissions may be logits or log-probs (same result).
    """
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    T, V = emissions.shape
    K = len(biases)
    b = np.asarray(biases, dtype=np.float64)[:, None]
    if np.any(b < 0):
        raise ValueError("a negative self-bias penalises STAYING; lambda >= 0")
    idx = np.arange(V)[None, :]
    out = np.empty((K, T), dtype=np.int64)
    score = None
    for start in range(0, T, block):
        n = min(block, T - start)
        bps = []
        for i in range(n):
            e = emissions[start + i].astype(np.float64)[None, :]
            if score is None:
                score = np.repeat(e, K, axis=0)              # start_transitions = 0
            else:
                gmax = score.max(axis=1, keepdims=True)
                gidx = score.argmax(axis=1)[:, None]         # first max, as torch.max does
                cand = score + b
                # argmax over previous tag u of score[u] + lambda*(u == v); ties -> lower index.
                take = (cand > gmax) | ((cand == gmax) & (idx < gidx))
                bps.append(np.where(take, idx, gidx))
                score = np.maximum(cand, gmax) + e
            score -= score.max(axis=1, keepdims=True)        # anti-drift, argmax-neutral
        tag = score.argmax(axis=1)                           # newest frame, no hindsight
        tags = [tag]
        for bp in reversed(bps[len(bps) - (n - 1):] if n > 1 else []):
            tag = bp[np.arange(K), tag]
            tags.append(tag)
        out[:, start:start + n] = np.stack(tags[::-1], axis=1)
    return out


def check_against_torch(biases, block, vocab_size=121, T=97, seed=0):
    """Assert decode_blocksync matches models.decode_block_synchronous at every lambda."""
    try:
        import torch
        from models import build_self_bias_crf, decode_block_synchronous
    except ImportError as exc:
        raise SystemExit(f"--check-torch needs torch and chord_recognition/models.py ({exc}); "
                         "the sweep itself runs without either, so drop the flag to proceed")

    rng = np.random.default_rng(seed)
    emis = rng.normal(size=(T, vocab_size)).astype(np.float32) * 3.0
    mine = decode_blocksync(emis, biases, block)
    t = torch.from_numpy(emis).unsqueeze(0)
    for k, lam in enumerate(biases):
        crf = build_self_bias_crf(vocab_size, float(lam))
        ref = decode_block_synchronous(t, crf, block)[0].numpy()
        bad = int((ref != mine[k]).sum())
        assert bad == 0, (f"numpy decode != models.decode_block_synchronous at "
                          f"lambda={lam}: {bad}/{T} frames differ")
    print(f"[check] numpy decode matches models.decode_block_synchronous at all "
          f"{len(biases)} lambdas (block={block}, V={vocab_size}, T={T})")


# ---------------------------------------------------------------------------
# labels, segments, metrics
# ---------------------------------------------------------------------------

def read_lab(path):
    entries = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts:
                entries.append((float(parts[0]), float(parts[1]), parts[2]))
    return entries


def gt_frames(lab_path, chord2idx, frame_duration, n_frames):
    """Ground-truth labels on the probe's frame grid (longest-overlap chord per frame)."""
    entries = [(b, e, _reduce_chord_label(c)) for b, e, c in read_lab(lab_path)]
    labels = np.full(n_frames, chord2idx[NO_CHORD], dtype=np.int64)
    for i in range(n_frames):
        ws, we = i * frame_duration, (i + 1) * frame_duration
        best, best_ov = NO_CHORD, 0.0
        for b, e, c in entries:
            if b > we or e < ws:
                continue
            ov = max(0.0, min(we, e) - max(ws, b))
            if ov > best_ov:
                best_ov, best = ov, c
        labels[i] = chord2idx[best]
    return labels


def segments(a):
    """(start_frame[], length[]) runs of equal label."""
    e = np.flatnonzero(np.diff(a)) + 1
    s = np.concatenate(([0], e))
    t = np.concatenate((e, [len(a)]))
    return s, t - s


def count_blips(a, starts, lens, frame_duration):
    """Count sub-0.5 s A -> B -> A excursions."""
    if len(lens) < 3:
        return 0
    lab = a[starts]
    short = lens * frame_duration < 0.5
    return int(np.count_nonzero(short[1:-1] & (lab[:-2] == lab[2:])))


def comparison_tables(vocab):
    """[V, V] mir_eval comparator score tables (-1 = reference not scored)."""
    import mir_eval
    V = len(vocab)
    ref = np.repeat(vocab, V).tolist()
    est = list(vocab) * V
    return {name: np.asarray(fn(ref, est), dtype=np.float64).reshape(V, V)
            for name, fn in (("root", mir_eval.chord.root),
                             ("thirds", mir_eval.chord.thirds),
                             ("majmin", mir_eval.chord.majmin),
                             ("mirex", mir_eval.chord.mirex))}


# ---------------------------------------------------------------------------
# emissions
# ---------------------------------------------------------------------------

def track_emissions(audio_path, backbone, probe, geom):
    """Per-frame (emissions, graph tags) for one track."""
    backbone.reset()
    probe.reset()
    spec = probe.spec
    samples_per_label = spec["pool_factor"] * spec["hop_size"]
    emis, tags, n_real = [], [], 0
    for window, real in stream_windows(audio_path, geom):
        n_real += real
        outs = probe(features=backbone_features(backbone, window, geom))
        emis.append(np.asarray(outs[0][0], dtype=np.float32))
        if spec["tags_output"] is not None:
            tags.extend(int(t) for t, ok in zip(outs[1][0], outs[2][0]) if ok)
    n = n_real // samples_per_label
    return np.concatenate(emis, axis=0)[:n], np.asarray(tags[:n], dtype=np.int64)


def load_emissions(name, backbone_path, probe_path, tracks, audio_of, cache_dir, threads):
    """Compute (or load from cache) one pair's emissions for every track."""
    cache = Path(cache_dir) / f"emissions_{name}.npz" if cache_dir else None
    backbone = OnnxStreamer(str(backbone_path), None, threads)
    probe = OnnxStreamer(str(probe_path), None, threads)
    geom = check_pair(backbone, probe, None)
    spec = probe.spec
    meta = {
        "vocab": spec["chord_labels"],
        "block": spec["emission_block"] or spec["label_frames"],
        "self_bias": spec["self_bias"],
        "frame_duration": spec["pool_factor"] * spec["hop_size"] / spec["sample_rate"],
        "probe": Path(probe_path).name,
        "n_params": spec["n_params"],
        "backbone_params": backbone.spec["n_params"],
    }
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if set(f"emis::{t}" for t in tracks) <= set(z.files):
            print(f"[{name}] reusing {cache}")
            return {t: (z[f"emis::{t}"], z[f"tags::{t}"]) for t in tracks}, meta

    print(f"[{name}] {meta['backbone_params']/1e6:.1f}M backbone + {meta['n_params']/1e6:.2f}M "
          f"probe, block={meta['block']}, exported self_bias={meta['self_bias']}")
    store, out = {}, {}
    for t in tracks:
        e, g = track_emissions(str(audio_of(t)), backbone, probe, geom)
        out[t] = (e, g)
        store[f"emis::{t}"] = e          # keep float32
        store[f"tags::{t}"] = g
        print(f"    {t[:60]:<60} {e.shape[0]:5d} frames", flush=True)
    if cache:
        os.makedirs(cache.parent, exist_ok=True)
        np.savez_compressed(cache, **store)
        print(f"[{name}] cached -> {cache}")
    return out, meta


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def sweep_pair(name, emissions, meta, tracks, lab_of, biases, verify):
    vocab = meta["vocab"]
    fd = meta["frame_duration"]
    block = meta["block"]
    chord2idx = {c: i for i, c in enumerate(vocab)}
    tabs = comparison_tables(vocab)

    acc = [dict(flips=0, nseg=0, lens=[], correct=0, blips=0,
                score={k: 0.0 for k in tabs}, counted={k: 0 for k in tabs})
           for _ in biases]
    gt_acc = dict(flips=0, nseg=0, lens=[], blips=0)
    n_frames = 0

    for t in tracks:
        emis, graph_tags = emissions[t]
        gt = gt_frames(str(lab_of(t)), chord2idx, fd, emis.shape[0])
        dec = decode_blocksync(emis, biases, block)

        if verify and len(graph_tags):
            if meta["self_bias"] not in biases:
                raise SystemExit(
                    f"--verify needs the graph's own lambda ({meta['self_bias']}) in --biases; "
                    f"add it or pass --no-verify")
            k = biases.index(meta["self_bias"])
            bad = int((dec[k] != graph_tags).sum())
            if bad:
                raise SystemExit(f"{name}/{t}: numpy decode != the ONNX graph at "
                                 f"lambda={meta['self_bias']}: {bad}/{len(graph_tags)} frames")

        n_frames += len(gt)
        gs, gl = segments(gt)
        gt_acc["flips"] += int(np.count_nonzero(np.diff(gt)))
        gt_acc["nseg"] += len(gl)
        gt_acc["lens"].append(gl * fd)
        gt_acc["blips"] += count_blips(gt, gs, gl, fd)

        for k in range(len(biases)):
            p, s = dec[k], acc[k]
            st, ln = segments(p)
            s["flips"] += int(np.count_nonzero(np.diff(p)))
            s["nseg"] += len(ln)
            s["lens"].append(ln * fd)
            s["blips"] += count_blips(p, st, ln, fd)
            s["correct"] += int((p == gt).sum())
            for cname, tab in tabs.items():
                v = tab[gt, p]
                s["score"][cname] += float(v[v >= 0].sum())
                s["counted"][cname] += int((v >= 0).sum())

    if verify:
        print(f"[{name}] verified: the numpy decode reproduces the ONNX graph exactly at "
              f"lambda={meta['self_bias']} on all {len(tracks)} tracks")

    dur = n_frames * fd
    rows = []
    for k, lam in enumerate(biases):
        s = acc[k]
        lens = np.concatenate(s["lens"])
        rows.append(dict(self_bias=lam, flips=s["flips"], flips_per_min=s["flips"] / dur * 60,
                         nseg=s["nseg"], median_seg=float(np.median(lens)),
                         frac_short=float((lens < 0.5).mean()), blips=s["blips"],
                         exact=s["correct"] / n_frames,
                         **{c: s["score"][c] / max(s["counted"][c], 1) for c in tabs}))
    glens = np.concatenate(gt_acc["lens"])
    gt_row = dict(flips=gt_acc["flips"], flips_per_min=gt_acc["flips"] / dur * 60,
                  nseg=gt_acc["nseg"], median_seg=float(np.median(glens)),
                  frac_short=float((glens < 0.5).mean()), blips=gt_acc["blips"],
                  duration_min=dur / 60)
    return rows, gt_row


def print_table(name, rows, gt_row, n_tracks, exported_bias):
    print(f"\n=== {name} :: self-bias sweep ({n_tracks} tracks, "
          f"{gt_row['duration_min']:.1f} min) ===")
    print(f"{'lambda':>7}{'flips':>8}{'flips/min':>11}{'vs GT':>7}{'segs':>7}{'med seg':>9}"
          f"{'<0.5s':>7}{'blips':>7}{'exact':>8}{'mirex':>8}{'majmin':>8}{'thirds':>8}{'root':>7}")
    print("-" * 103)
    for r in rows:
        mark = "*" if r["self_bias"] == exported_bias else " "
        print(f"{r['self_bias']:>6g}{mark}{r['flips']:>7d}{r['flips_per_min']:>11.1f}"
              f"{r['flips_per_min'] / gt_row['flips_per_min']:>6.2f}x{r['nseg']:>7d}"
              f"{r['median_seg']:>8.2f}s{100 * r['frac_short']:>6.0f}%{r['blips']:>7d}"
              f"{100 * r['exact']:>7.1f}%{100 * r['mirex']:>7.1f}%{100 * r['majmin']:>7.1f}%"
              f"{100 * r['thirds']:>7.1f}%{100 * r['root']:>6.1f}%")
    print(f"{'GT':>7}{gt_row['flips']:>8d}{gt_row['flips_per_min']:>11.1f}{1.0:>6.2f}x"
          f"{gt_row['nseg']:>7d}{gt_row['median_seg']:>8.2f}s{100 * gt_row['frac_short']:>6.0f}%"
          f"{gt_row['blips']:>7d}")
    print("  * = the lambda this probe was exported with; flips = frame-to-frame label "
          "changes; blips = A->B->A excursions under 0.5 s")


def print_tradeoff(name, rows, base_bias):
    base = next((r for r in rows if r["self_bias"] == base_bias), None)
    if base is None:
        return
    print(f"\n--- {name} :: cost of raising lambda above {base_bias:g} ---")
    print(f"{'lambda':>7}{'d flips/min':>13}{'d flips':>9}{'d blips':>9}{'d mirex':>9}"
          f"{'flips/min per mirex point':>27}")
    for r in rows:
        if r["self_bias"] <= base_bias:
            continue
        df = r["flips_per_min"] - base["flips_per_min"]
        dm = (r["mirex"] - base["mirex"]) * 100
        rate = f"{-df / -dm:>27.1f}" if dm < -1e-9 else f"{'free':>27}"
        print(f"{r['self_bias']:>7g}{df:>13.1f}{100 * df / base['flips_per_min']:>8.0f}%"
              f"{100 * (r['blips'] - base['blips']) / max(base['blips'], 1):>8.0f}%{dm:>9.1f}{rate}")


def print_examples(name, emissions, meta, tracks, lab_of, biases, n_examples, window):
    """Print the excerpts with the most excess flicker relative to the reference."""
    vocab = meta["vocab"]
    fd = meta["frame_duration"]
    block = meta["block"]
    c2i = {c: i for i, c in enumerate(vocab)}
    W = int(round(window / fd))
    best = []
    for t in tracks:
        emis, _ = emissions[t]
        gt = gt_frames(str(lab_of(t)), c2i, fd, emis.shape[0])
        dec = decode_blocksync(emis, biases, block)
        for st in range(0, len(gt) - W, W):
            gf = int(np.count_nonzero(np.diff(gt[st:st + W])))
            pf = int(np.count_nonzero(np.diff(dec[0][st:st + W])))
            best.append((pf - 2 * gf, t, st, dec[:, st:st + W], gt[st:st + W]))
    best.sort(reverse=True, key=lambda x: x[0])

    seen = set()
    shown = 0
    for _, t, st, dec, gt in best:
        if t in seen or shown >= n_examples:
            continue
        seen.add(t)
        shown += 1
        print(f"\n--- {name}  {t}  {st * fd:.0f}-{st * fd + window:.0f}s ---")
        for label, arr in [(f"lam={b:g}", dec[k]) for k, b in enumerate(biases)] + [("GT", gt)]:
            s, ln = segments(np.asarray(arr))
            flips = int(np.count_nonzero(np.diff(arr)))
            body = " ".join(f"{vocab[arr[i]]}({L * fd:.2f})" for i, L in zip(s, ln))
            print(f"  {label:<8} [{flips:>2} flips] {body}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Sweep the self-bias CRF's lambda against transcript stability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pair", action="append", required=True, metavar="NAME=BACKBONE,PROBE",
                    help="a named (backbone, chord probe) pair of *.streaming.onnx graphs; "
                         "repeat to put several models in one table")
    ap.add_argument("--dataset", required=True, help="dataset name in the split assignment")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--assignment", required=True,
                    help="split assignment JSON, relative to mustream/")
    ap.add_argument("--audio-root", default="data/audio")
    ap.add_argument("--annotation-root", default="data/annotation")
    ap.add_argument("--audio-ext", default=".wav")
    ap.add_argument("--biases", default="0,0.5,1,1.5,2,2.5,3,4,5,6,8,10,12,16,24,32,48,64")
    ap.add_argument("--tracks", type=int, default=None, help="only the first N tracks")
    ap.add_argument("--cache", default=None,
                    help="directory to cache emissions in; re-sweeping is then free")
    ap.add_argument("--examples", type=int, default=0, help="excerpts to print per pair")
    ap.add_argument("--example-window", type=float, default=12.0, help="excerpt length, seconds")
    ap.add_argument("--example-biases", default=None,
                    help="lambdas to show in excerpts (default: the exported one, 3, and 8)")
    ap.add_argument("--tradeoff-from", type=float, default=None,
                    help="also print the cost of raising lambda above this (default: exported)")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip checking the numpy decode against the graph's own tags")
    ap.add_argument("--check-torch", action="store_true",
                    help="also check the numpy decode against models.decode_block_synchronous")
    ap.add_argument("--threads", type=int, default=None, help="onnxruntime intra-op threads")
    ap.add_argument("--json", default=None, help="write the sweep here")
    args = ap.parse_args()

    biases = [float(x) for x in args.biases.split(",")]
    assign = json.load(open(REPO / args.assignment))["assignment"]
    tracks = sorted(k.split("\t")[1] for k, v in assign.items()
                    if v == args.split and k.split("\t")[0] == args.dataset)
    if args.tracks:
        tracks = tracks[:args.tracks]
    if not tracks:
        raise SystemExit(f"no {args.split} tracks for dataset {args.dataset!r} in {args.assignment}")

    audio_of = lambda t: REPO / args.audio_root / args.dataset / f"{t}{args.audio_ext}"
    lab_of = lambda t: REPO / args.annotation_root / args.dataset / t / "chord.lab"
    missing = [t for t in tracks if not audio_of(t).exists() or not lab_of(t).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(tracks)} {args.dataset} {args.split} tracks "
                         f"lack audio or chord.lab here, e.g. {missing[0]}")
    print(f"[data] {args.dataset} {args.split}: {len(tracks)} tracks")

    out = {"biases": biases, "dataset": args.dataset, "split": args.split, "pairs": {}}
    for spec in args.pair:
        name, _, paths = spec.partition("=")
        bpath, _, ppath = paths.partition(",")
        if not name or not bpath or not ppath:
            raise SystemExit(f"--pair wants NAME=BACKBONE.onnx,PROBE.onnx; got {spec!r}")
        emissions, meta = load_emissions(name, bpath, ppath, tracks, audio_of,
                                         args.cache, args.threads)
        if args.check_torch:
            check_against_torch(biases, meta["block"], len(meta["vocab"]))
        rows, gt_row = sweep_pair(name, emissions, meta, tracks, lab_of, biases, args.verify)
        print_table(name, rows, gt_row, len(tracks), meta["self_bias"])
        print_tradeoff(name, rows, args.tradeoff_from
                       if args.tradeoff_from is not None else meta["self_bias"])
        if args.examples:
            ex = ([float(x) for x in args.example_biases.split(",")] if args.example_biases
                  else sorted({meta["self_bias"] or 1.0, 3.0, 8.0}))
            print_examples(name, emissions, meta, tracks, lab_of, ex,
                           args.examples, args.example_window)
        out["pairs"][name] = {"rows": rows, "gt": gt_row, "meta":
                              {k: v for k, v in meta.items() if k != "vocab"}}

    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
