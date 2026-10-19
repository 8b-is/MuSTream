"""CRF post-processing sweep for chord checkpoints: offline vs. block-synchronous
Viterbi, plus non-CRF smoothing baselines, on cached emissions.

Transition sources: --self-bias LAMBDA (transitions = LAMBDA * I, in nats) and/or
--bigram-transitions (log P(next | prev) counted on the training labels).
Baselines: --median-widths (centred), --causal-widths and --ema-alphas (zero lookahead).

Columns: agree (vs. that CRF's offline decode), wacc (MIREX weighted accuracy),
segs / seg_len, bdry (signed boundary offset, + = late), flick (short segments).

Usage (from chord_recognition/):
  python sweep_crf.py --self-test
  python sweep_crf.py config/MERT_streaming.yaml logs/<run>/best.pt \
      --self-bias 0.5,1,2,4,8 --block-sync auto
"""

import argparse
import os
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from models import (
    LinearChainCRF,
    StreamingViterbi,
    backbone_lookahead_block,
    build_self_bias_crf,
    decode_block_synchronous,
    emission_block_for_config,
)
from infer import _merge_frames, score_segments
from chord_dataloader import preprocess_labels, read_label_file, reduce_chord_label
import pickle

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CRF construction
# ---------------------------------------------------------------------------

def count_label_bigrams(
    records: list,
    chord2idx: dict,
    num_tags: int,
    frame_duration: float,
    cache_path: Path,
    recount: bool = False,
):
    """Count frame-level label bigrams at the model's label rate (cached to cache_path).

    Returns (bigram [V, V], first [V], last [V]).
    """
    if cache_path.exists() and not recount:
        z = np.load(cache_path)
        if (z["num_tags"] == num_tags and np.isclose(z["frame_duration"], frame_duration)
                and z["n_records"] == len(records)):
            print(f"Loaded bigram counts from {cache_path} "
                  f"({int(z['n_records'])} tracks, {int(z['bigram'].sum()):,} transitions)")
            return z["bigram"], z["first"], z["last"]
        print(f"{cache_path} does not match this config -- recounting")

    bigram = np.zeros((num_tags, num_tags), dtype=np.int64)
    first = np.zeros(num_tags, dtype=np.int64)
    last = np.zeros(num_tags, dtype=np.int64)

    for record in tqdm(records, desc="counting bigrams", unit="track"):
        entries = read_label_file(record["chord_label_path"])
        reduced = [(b, e, reduce_chord_label(c)) for b, e, c in entries]
        if not reduced:
            continue
        total_duration = reduced[-1][1]
        windowed = preprocess_labels(reduced, total_duration, frame_duration)
        seq = np.array([chord2idx[c] for _, _, c in windowed], dtype=np.int64)
        if seq.size == 0:
            continue
        first[seq[0]] += 1
        last[seq[-1]] += 1
        if seq.size > 1:
            np.add.at(bigram, (seq[:-1], seq[1:]), 1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, bigram=bigram, first=first, last=last,
             num_tags=num_tags, frame_duration=frame_duration, n_records=len(records))
    print(f"Counted {bigram.sum():,} frame transitions over {len(records)} tracks "
          f"-> cached in {cache_path}")
    return bigram, first, last


def build_bigram_crf(
    bigram: np.ndarray,
    first: np.ndarray,
    last: np.ndarray,
    alpha: float,
    device: torch.device,
) -> LinearChainCRF:
    """CRF with add-alpha-smoothed log bigram transitions (and start/end) from label counts."""
    num_tags = bigram.shape[0]
    trans = np.log((bigram + alpha) / (bigram.sum(axis=1, keepdims=True) + alpha * num_tags))
    start = np.log((first + alpha) / (first.sum() + alpha * num_tags))
    end = np.log((last + alpha) / (last.sum() + alpha * num_tags))

    crf = LinearChainCRF(num_tags).to(device)
    with torch.no_grad():
        crf.transitions.copy_(torch.as_tensor(trans, dtype=crf.transitions.dtype))
        crf.start_transitions.copy_(torch.as_tensor(start, dtype=crf.start_transitions.dtype))
        crf.end_transitions.copy_(torch.as_tensor(end, dtype=crf.end_transitions.dtype))

    # Report the implied switch penalty (comparable to --self-bias).
    observed = bigram.sum(axis=1) > 0
    diag = np.diag(trans)[observed]
    off = trans.copy()
    np.fill_diagonal(off, -np.inf)
    best_off = off[observed].max(axis=1)
    p_stay = np.exp(diag)
    print(
        f"bigram CRF: mean log P(stay)={diag.mean():.3f}, "
        f"mean best log P(switch)={best_off.mean():.3f} "
        f"-> effective switch penalty ~{(diag - best_off).mean():.2f} nats "
        f"(compare --self-bias)\n"
        f"            mean P(stay)={p_stay.mean():.4f} -> implied mean chord duration "
        f"~{(1.0 / (1.0 - p_stay)).mean():.1f} frames, "
        f"{len(diag)} of {num_tags} tags observed"
    )
    return crf.eval()


# ---------------------------------------------------------------------------
# Self-test: correctness of StreamingViterbi against LinearChainCRF.decode
# ---------------------------------------------------------------------------

def self_test(device: torch.device = torch.device("cpu")) -> None:
    """Check the streaming decoders and filters on synthetic emissions (no data needed)."""
    torch.manual_seed(0)
    V, B, T = 12, 3, 40
    crf = LinearChainCRF(V).to(device)
    with torch.no_grad():   # spread the transitions out so paths really do compete
        crf.transitions.uniform_(-2.0, 2.0)
    emissions = torch.randn(B, T, V, device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    offline = crf.decode(emissions, mask)                       # [B, T]

    # 1. lag=None must *be* the offline decoder, not merely agree with it.
    dec = StreamingViterbi(crf, lag=None)
    assert torch.equal(dec.decode_sequence(emissions), offline), "lag=None != crf.decode()"

    # 2. lag >= T routes every frame through flush() -> exact offline decode.
    for lag in (T, T + 7):
        out = StreamingViterbi(crf, lag=lag).decode_sequence(emissions)
        assert torch.equal(out, offline), f"lag={lag} (>= T={T}) should be exact, got a mismatch"

    # 3. Every lag must return exactly T frames, in one-per-step steady state.
    for lag in (0, 1, 2, 5, 10, 20):
        out = StreamingViterbi(crf, lag=lag).decode_sequence(emissions)
        assert out.shape == (B, T), f"lag={lag} produced {tuple(out.shape)}, expected {(B, T)}"

    # 4. Agreement with offline should be non-trivial and improve with lag.
    agree = [
        (StreamingViterbi(crf, lag=lag).decode_sequence(emissions) == offline).float().mean().item()
        for lag in (0, 1, 2, 5, 10, 20, T)
    ]
    assert agree[-1] == 1.0
    assert agree[-2] >= agree[0], f"agreement should not degrade with lag: {agree}"

    # 5. A stream shorter than the lag must still emit all its frames (via flush).
    short = emissions[:, :3]
    out = StreamingViterbi(crf, lag=20).decode_sequence(short)
    assert out.shape == (B, 3)
    assert torch.equal(out, crf.decode(short, mask[:, :3])), "short-stream tail mismatch"

    # 6. Running scores stay bounded on a long stream.
    long_em = torch.randn(1, 4000, V, device=device) * 5.0
    dec = StreamingViterbi(crf, lag=8)
    for t in range(long_em.shape[1]):
        dec.step(long_em[:, t])
    assert dec.score.abs().max().item() < 1e3, "running scores drifted -- normalisation broken"

    # 7. use_viterbi=False is plain per-frame argmax, zero lag.
    out = StreamingViterbi(None, use_viterbi=False).decode_sequence(emissions)
    assert torch.equal(out, emissions.argmax(dim=-1)), "use_viterbi=False != argmax"

    # 8. Window baselines: width<=1 is a no-op, length preserved, blips absorbed.
    labels = [3] * 10 + [7] + [3] * 10
    assert mode_filter(labels, 1) == labels, "width=1 must be a no-op"
    assert len(mode_filter(labels, 5)) == len(labels)
    assert mode_filter(labels, 5) == [3] * 21, "mode filter failed to absorb a 1-frame blip"
    assert mode_filter([3, 3, 7, 7], 3)[2] == 7, "tie should keep the centre frame's own label"

    em = torch.randn(25, V, device=device)
    assert boxcar_filter(em, 1) == em.argmax(dim=-1).tolist(), "width=1 must be a no-op"
    assert len(boxcar_filter(em, 7)) == 25
    flat = em[:1].expand(25, V).contiguous()
    assert boxcar_filter(flat, 9) == flat.argmax(dim=-1).tolist(), "boxcar altered a flat signal"

    # 9. Block-synchronous: block=1 == lag=0; block >= T == offline (without end_transitions).
    lag0 = StreamingViterbi(crf, lag=0).decode_sequence(emissions)
    assert torch.equal(decode_block_synchronous(emissions, crf, 1), lag0), \
        "blocksync=1 != lag=0"
    crf_noend = LinearChainCRF(V).to(device)
    crf_noend.load_state_dict(crf.state_dict())
    with torch.no_grad():
        crf_noend.end_transitions.zero_()
    offline_noend = crf_noend.decode(emissions, mask)
    for block in (T, T + 13):
        assert torch.equal(decode_block_synchronous(emissions, crf_noend, block), offline_noend), \
            f"blocksync={block} (>= T={T}) should be the offline decode"
    for block in (2, 3, 5, 6, 7):
        out = decode_block_synchronous(emissions, crf, block)
        assert out.shape == (B, T), f"blocksync={block} produced {tuple(out.shape)}"
    assert decode_block_synchronous(emissions[:, :7], crf, 3).shape == (B, 7)
    bs_agree = [
        (decode_block_synchronous(emissions, crf, b) == offline).float().mean().item()
        for b in (1, 2, 3, 6, 12)
    ]

    # 10. Causal filters: identity settings are no-ops, length preserved, never revise the past.
    assert causal_mode_filter(labels, 1) == labels, "causal mode width=1 must be a no-op"
    assert len(causal_mode_filter(labels, 5)) == len(labels)
    assert causal_boxcar_filter(em, 1) == em.argmax(dim=-1).tolist(), "causal boxcar w=1 no-op"
    assert len(causal_boxcar_filter(em, 7)) == 25
    assert ema_filter(em, 1.0) == em.argmax(dim=-1).tolist(), "ema alpha=1 must be a no-op"
    assert len(ema_filter(em, 0.3)) == 25
    em2 = em.clone()
    em2[-1] += 100.0
    for fn in (lambda x: causal_boxcar_filter(x, 7),
               lambda x: ema_filter(x, 0.3),
               lambda x: causal_mode_filter(x.argmax(dim=-1).tolist(), 5)):
        assert fn(em)[:-1] == fn(em2)[:-1], "a causal filter changed a past output"
    assert causal_boxcar_filter(flat, 9) == flat.argmax(dim=-1).tolist()
    assert ema_filter(flat, 0.3) == flat.argmax(dim=-1).tolist()

    # 11. Boundary / flicker diagnostics on hand-computed cases.
    ref_iv = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    perfect = [(0.0, 1.0, "C"), (1.0, 2.0, "G"), (2.0, 3.0, "C")]
    assert np.allclose(boundary_offsets(ref_iv, perfect), 0.0), "perfect estimate must be 0 offset"
    late = [(0.0, 1.2, "C"), (1.2, 2.2, "G"), (2.2, 3.0, "C")]
    assert np.allclose(boundary_offsets(ref_iv, late), 0.2), "late estimate must read +0.2"
    assert boundary_offsets(ref_iv, [(0.0, 3.0, "C")]) is None, "one segment has no boundaries"
    assert short_segment_fraction([(0.0, 0.1, "C"), (0.1, 0.2, "G")], 0.1) == 1.0
    assert short_segment_fraction([(0.0, 5.0, "C")], 0.1) == 0.0

    print("self-test passed:")
    print("  lag=None is bit-identical to LinearChainCRF.decode()")
    print(f"  lag >= T is bit-identical to LinearChainCRF.decode()  (T={T})")
    print("  use_viterbi=False is bit-identical to per-frame argmax")
    print("  all lags emit exactly T frames; short streams flush correctly")
    print("  mode/boxcar filters are no-ops at width=1 and length-preserving")
    print("  blocksync=1 is bit-identical to lag=0; blocksync >= T is the offline decode")
    print("  causal filters are no-ops at their identity setting and never revise the past")
    print("  boundary/flicker diagnostics agree with hand-computed cases")
    print("  agreement vs. offline by lag (0,1,2,5,10,20,T): "
          + ", ".join(f"{a:.3f}" for a in agree))
    print("  agreement vs. offline by blocksync (1,2,3,6,12): "
          + ", ".join(f"{a:.3f}" for a in bs_agree))


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def decode_one(emissions: torch.Tensor, crf, use_viterbi: bool, lag) -> list[int]:
    """emissions [T, V] -> decoded frame labels, as a plain list."""
    dec = StreamingViterbi(crf, lag=lag, use_viterbi=use_viterbi)
    return dec.decode_sequence(emissions.unsqueeze(0))[0].tolist()


# ---------------------------------------------------------------------------
# Non-CRF smoothing baselines (centred window: width//2 frames of lookahead)
# ---------------------------------------------------------------------------

def mode_filter(labels: list[int], width: int) -> list[int]:
    """Centred majority vote over `width` labels (ties keep the centre label)."""
    if width <= 1:
        return list(labels)
    half = width // 2
    n = len(labels)
    out = []
    for t in range(n):
        window = labels[max(0, t - half) : min(n, t + half + 1)]
        counts = Counter(window)
        best = max(counts.values())
        out.append(labels[t] if counts[labels[t]] == best
                   else next(k for k, v in counts.items() if v == best))
    return out


def boxcar_filter(emissions: torch.Tensor, width: int) -> list[int]:
    """Centred mean of the emissions over `width` frames, then argmax."""
    if width <= 1:
        return emissions.argmax(dim=-1).tolist()
    T = emissions.shape[0]
    half = width // 2
    csum = torch.cat([torch.zeros_like(emissions[:1]), emissions.cumsum(dim=0)], dim=0)
    idx = torch.arange(T, device=emissions.device)
    lo = (idx - half).clamp(min=0)
    hi = (idx + half + 1).clamp(max=T)
    totals = csum[hi] - csum[lo]                                  # [T, V]
    return (totals / (hi - lo).unsqueeze(1)).argmax(dim=-1).tolist()


# ---------------------------------------------------------------------------
# Zero-lookahead (causal) smoothers
# ---------------------------------------------------------------------------

def causal_boxcar_filter(emissions: torch.Tensor, width: int) -> list[int]:
    """Trailing mean of the last `width` emission frames, then argmax."""
    if width <= 1:
        return emissions.argmax(dim=-1).tolist()
    T = emissions.shape[0]
    csum = torch.cat([torch.zeros_like(emissions[:1]), emissions.cumsum(dim=0)], dim=0)
    idx = torch.arange(T, device=emissions.device)
    lo = (idx - width + 1).clamp(min=0)
    totals = csum[idx + 1] - csum[lo]                             # [T, V]
    return (totals / (idx + 1 - lo).unsqueeze(1)).argmax(dim=-1).tolist()


def causal_mode_filter(labels: list[int], width: int) -> list[int]:
    """Trailing majority vote over the last `width` labels (ties keep frame t's label)."""
    if width <= 1:
        return list(labels)
    out = []
    for t in range(len(labels)):
        window = labels[max(0, t - width + 1) : t + 1]
        counts = Counter(window)
        best = max(counts.values())
        out.append(labels[t] if counts[labels[t]] == best
                   else next(k for k, v in counts.items() if v == best))
    return out


def ema_filter(emissions: torch.Tensor, alpha: float) -> list[int]:
    """Exponential moving average of the emissions (s_t = (1-alpha) s_{t-1} + alpha e_t), then argmax.

    alpha in (0, 1]; 1 = plain argmax.
    """
    if alpha >= 1.0:
        return emissions.argmax(dim=-1).tolist()
    em = emissions.detach().cpu().numpy()
    s = em[0].astype(np.float64, copy=True)
    out = [int(s.argmax())]
    for t in range(1, em.shape[0]):
        s *= (1.0 - alpha)
        s += alpha * em[t]
        out.append(int(s.argmax()))
    return out


# ---------------------------------------------------------------------------
# Extra diagnostics
# ---------------------------------------------------------------------------

def boundary_offsets(ref_intervals: np.ndarray, est_segments: list) -> np.ndarray | None:
    """Signed seconds from each reference chord change to the nearest estimated one (+ = late)."""
    if len(ref_intervals) < 2 or len(est_segments) < 2:
        return None
    ref_b = np.asarray([iv[0] for iv in ref_intervals[1:]], dtype=float)
    est_b = np.asarray([s[0] for s in est_segments[1:]], dtype=float)
    if ref_b.size == 0 or est_b.size == 0:
        return None
    pos = np.searchsorted(est_b, ref_b)
    lo = np.clip(pos - 1, 0, est_b.size - 1)
    hi = np.clip(pos, 0, est_b.size - 1)
    cand = np.stack([est_b[lo], est_b[hi]], axis=1) - ref_b[:, None]   # [n, 2]
    return cand[np.arange(ref_b.size), np.argmin(np.abs(cand), axis=1)]


def short_segment_fraction(est_segments: list, frame_duration: float,
                           max_frames: int = 2) -> float:
    """Fraction of estimated segments shorter than `max_frames` frames."""
    if not est_segments:
        return float("nan")
    thr = max_frames * frame_duration - 1e-9
    return float(np.mean([(e - b) < thr for b, e, _ in est_segments]))


def run_sweep(
    config_path: str,
    checkpoint_path: str,
    self_biases: list,
    use_bigram: bool,
    bigram_alpha: float,
    bigram_split: str,
    recount: bool,
    median_widths: list,
    causal_widths: list,
    ema_alphas: list,
    block_syncs: list,
    split: str,
    limit,
    emission_mode: str,
    dump_per_track: str | None = None,
) -> None:
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    device = torch.device(config.get("device", "cpu"))
    frame_duration = config["hop_size"] * config["pool_factor"] / config["sample_rate"]

    # Backbone lookahead block (raw frames), read from the distillation checkpoint.
    raw_block = backbone_lookahead_block(config.get("mert_align_checkpoint"))
    # Pooled frames per emission burst for any backbone type ('auto').
    auto_block = emission_block_for_config(config)
    block_syncs = [b if b != "auto" else auto_block for b in block_syncs]
    if any(b is None for b in block_syncs):
        raise SystemExit(
            "--block-sync auto: this config's backbone has no emission block (a full-crop "
            "offline model) -- there is no free hindsight to spend, so use the offline rows "
            "(which are printed anyway) or pass an explicit block size."
        )
    block_syncs = [int(b) for b in block_syncs]

    chord2idx, idx2chord = pickle.load(open(config["vocab_path"], 'rb'))
    num_tags = len(idx2chord)

    # All transition sources, built once.
    crfs = []
    if use_bigram:
        bigram_records = pickle.load(open(os.path.join(config["split_dir"], f"{bigram_split}.pkl"), 'rb'))
        cache = Path(config["split_dir"]) / (
            f"bigram_counts_{bigram_split}_{round(1 / frame_duration)}Hz_{num_tags}tags.npz"
        )
        counts = count_label_bigrams(
            bigram_records, chord2idx, num_tags, frame_duration, cache, recount,
        )
        crfs.append(("bigram", build_bigram_crf(*counts, alpha=bigram_alpha, device=device)))
    crfs += [(f"self_bias={b:g}", build_self_bias_crf(num_tags, b, device)) for b in self_biases]
    if not crfs and not (median_widths or causal_widths or ema_alphas):
        raise SystemExit("nothing to sweep: pass --bigram-transitions, "
                         "--median-widths, --causal-widths, --ema-alphas and/or a "
                         "non-empty --self-bias")

    # Works with any extractor exposing predict_logits (streaming, sliding, chunked, offline).
    if config.get("backbone") in ("streaming", "small_streaming"):
        from infer_streaming import StreamingChordExtractor
        extractor = StreamingChordExtractor(config, checkpoint_path)
    else:
        from infer import make_extractor      # dispatches sliding vs. chunked/full
        extractor = make_extractor(config, checkpoint_path)

    records = pickle.load(open(os.path.join(config["split_dir"], f"{split}.pkl"), 'rb'))
    if limit is not None:
        records = records[:limit]
    print(f"\nSweeping {len(records)} {split} tracks, "
          f"label rate {1 / frame_duration:.2f} Hz ({frame_duration * 1000:.1f} ms/frame)")
    print(f"Emissions: {emission_mode}")
    if auto_block:
        print(f"Emission block: {auto_block} pooled label frame(s) "
              f"({auto_block * frame_duration * 1000:.0f} ms; backbone lookahead_block="
              f"{raw_block} raw frames) -> block-synchronous decoding gets up to "
              f"{auto_block - 1} frames of hindsight for free\n")
    else:
        print("Emission block: none (full-crop offline backbone -- no free hindsight)\n")

    # Each setting: (label, ref_key for `agree` or None, lag_frames or None = whole
    # track, decode_fn: emissions [T, V] -> list[int]).
    settings = [("argmax", None, 0, lambda em: em.argmax(dim=-1).tolist())]
    for name, crf in crfs:
        settings.append((f"{name} offline", name, None,
                         partial(decode_one, crf=crf, use_viterbi=True, lag=None)))
        # Block-synchronous: zero added latency.
        for b in block_syncs:
            settings.append((f"{name} blocksync={b}", name, 0,
                             lambda em, crf=crf, b=b:
                                 decode_block_synchronous(em.unsqueeze(0), crf, b)[0].tolist()))
    # Centred window filters: pay width//2 frames of lookahead.
    for w in median_widths:
        settings.append((f"mode w={w}", None, w // 2,
                         lambda em, w=w: mode_filter(em.argmax(dim=-1).tolist(), w)))
        settings.append((f"boxcar w={w}", None, w // 2,
                         lambda em, w=w: boxcar_filter(em, w)))
    # Trailing window filters and the IIR: zero lookahead, zero added latency.
    for w in causal_widths:
        settings.append((f"cmode w={w}", None, 0,
                         lambda em, w=w: causal_mode_filter(em.argmax(dim=-1).tolist(), w)))
        settings.append((f"cboxcar w={w}", None, 0,
                         lambda em, w=w: causal_boxcar_filter(em, w)))
    for a in ema_alphas:
        settings.append((f"ema a={a:g}", None, 0,
                         lambda em, a=a: ema_filter(em, a)))

    per_track_rows: list = []
    stats = {label: {"agree": [], "wacc": [], "segs": [], "seg_len": [],
                     "bdry": [], "abdry": [], "flick": []}
             for label, *_ in settings}

    for record in tqdm(records, desc="tracks", unit="track"):
        entries = read_label_file(record["chord_label_path"])
        # Raw reference labels; mir_eval does its own reduction.
        if not entries:
            continue
        ref_intervals = np.array([[b, e] for b, e, _ in entries])
        ref_labels = [c for _, _, c in entries]

        logits = extractor.predict_logits(record["audio_path"]).to(device)   # [T, V]
        emissions = F.log_softmax(logits, dim=-1) if emission_mode == "logprobs" else logits

        # Offline decode per CRF, the reference for its `agree` column.
        offline_ref = {
            name: decode_one(emissions, crf, True, None) for name, crf in crfs
        }

        for label, ref_key, _lag, decode_fn in settings:
            preds = decode_fn(emissions)
            segments = _merge_frames(preds, idx2chord, frame_duration)

            if ref_key is not None:
                ref = offline_ref[ref_key]
                stats[label]["agree"].append(
                    float(np.mean([p == r for p, r in zip(preds, ref)]))
                )

            # wacc column = MIREX weighted accuracy.
            result = score_segments(ref_intervals, ref_labels, segments)
            if result is not None and "mirex" in result[0]:
                stats[label]["wacc"].append(result[0]["mirex"])
            stats[label]["segs"].append(len(segments))
            stats[label]["seg_len"].append(len(preds) * frame_duration / max(1, len(segments)))

            offsets = boundary_offsets(ref_intervals, segments)
            if offsets is not None and offsets.size:
                stats[label]["bdry"].append(float(offsets.mean()))
                stats[label]["abdry"].append(float(np.abs(offsets).mean()))
            stats[label]["flick"].append(short_segment_fraction(segments, frame_duration))

            if dump_per_track is not None:
                per_track_rows.append({
                    "track": os.path.basename(record["audio_path"]),
                    "decoder": label,
                    "wacc": (result[0]["mirex"]
                             if result is not None and "mirex" in result[0] else float("nan")),
                    "segs": len(segments),
                    "flick": short_segment_fraction(segments, frame_duration),
                    "bdry": (float(offsets.mean())
                             if offsets is not None and offsets.size else float("nan")),
                })

    # ---------------------------------------------------------------- report
    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    base = mean(stats["argmax"]["wacc"])
    print(f"\n{'decoder':<26} {'+latency':>9} {'agree':>8} {'wacc':>8} {'d_wacc':>8} "
          f"{'segs':>7} {'seg_len':>8} {'bdry':>8} {'|bdry|':>7} {'flick':>7}")
    print("-" * 104)
    for label, _ref_key, lag, _fn in settings:
        s = stats[label]
        latency = "whole trk" if lag is None else f"{lag * frame_duration:.3f}s"
        agree = mean(s["agree"])
        agree_str = "-" if np.isnan(agree) else f"{agree * 100:.2f}%"
        w = mean(s["wacc"])
        print(
            f"{label:<26} {latency:>9} {agree_str:>8} "
            f"{w:8.4f} {w - base:+8.4f} {mean(s['segs']):7.1f} {mean(s['seg_len']):7.2f}s "
            f"{mean(s['bdry']):+8.3f} {mean(s['abdry']):7.3f} {mean(s['flick']) * 100:6.1f}%"
        )
    print(f"\nn={len(stats[settings[0][0]]['wacc'])} tracks scored")
    print("+latency = ADDED latency on top of the backbone's own, in seconds. blocksync rows "
          "read 0.000s\n           because their hindsight comes from frames the block-causal "
          "backbone had already\n           computed -- see the lookahead_block line above.")
    print("d_wacc   = wacc minus the argmax row: what this post-processing actually bought.")
    print("agree    = frame agreement with that CRF's own offline decode (its 'offline' row is "
          "its own\n           reference, so 100%); '-' where there is no offline/streaming gap "
          "-- argmax and the\n           window/IIR filters are exact at the latency shown.")
    print("bdry     = mean SIGNED seconds from each reference chord change to the nearest "
          "estimated one\n           (+ = late). |bdry| is its absolute counterpart. A causal "
          "smoother buys accuracy by\n           hesitating, which shows up here and not in "
          "wacc alone.")
    print("flick    = fraction of estimated segments shorter than 2 frames.")

    if dump_per_track is not None:
        import csv
        with open(dump_per_track, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["track", "decoder", "wacc", "segs", "flick", "bdry"])
            w.writeheader()
            w.writerows(per_track_rows)
        print(f"\nPer-track scores -> {dump_per_track} ({len(per_track_rows)} rows). Every "
              f"decoder saw the\nsame emissions for the same track, so differences are PAIRED "
              f"-- test them that way\nrather than against the +-sd of the means.")


def main():
    parser = argparse.ArgumentParser(
        description="Block-synchronous vs. offline CRF decoding sweep for chord probes."
    )
    parser.add_argument("config", nargs="?", help="path to a MERT_streaming.yaml-style config")
    parser.add_argument("checkpoint", nargs="?", help="path to a train_streaming.py .pt checkpoint")
    parser.add_argument("--self-test", action="store_true",
                        help="verify StreamingViterbi against LinearChainCRF.decode() on "
                             "synthetic emissions and exit (no config/checkpoint needed)")
    parser.add_argument("--self-bias", default="2.0",
                        help="comma-separated self-transition bonuses in nats; one CRF per "
                             "value (default: %(default)s). Pass '' to omit this source.")
    parser.add_argument("--bigram-transitions", action="store_true",
                        help="also sweep a CRF whose transitions are the maximum-likelihood "
                             "log P(next | prev) counted from the training labels at the label "
                             "rate -- no training required, and unlike --self-bias it carries a "
                             "real progression prior, not just 'chords persist'")
    parser.add_argument("--bigram-alpha", type=float, default=0.1,
                        help="add-alpha smoothing for the bigram counts, keeping unobserved "
                             "transitions finite instead of -inf (default: %(default)s)")
    parser.add_argument("--bigram-split", default="train",
                        help="split to count bigrams on (default: %(default)s -- counting on "
                             "the split you evaluate would leak)")
    parser.add_argument("--recount", action="store_true",
                        help="ignore the cached bigram counts and recount from the label files")
    parser.add_argument("--median-widths", default="",
                        help="comma-separated centred-window widths in frames for the two "
                             "non-Viterbi smoothing baselines: a majority-vote ('mode', the "
                             "categorical median filter) over the argmax labels, and a boxcar "
                             "mean over the emissions. Each needs width//2 frames of lookahead, "
                             "so they add width//2 frames of latency.")
    parser.add_argument("--causal-widths", default="",
                        help="comma-separated TRAILING window widths in frames -- the "
                             "zero-lookahead counterparts of --median-widths ('cmode' majority "
                             "vote over labels, 'cboxcar' mean over emissions). These add no "
                             "latency at all, so they are the baselines a zero-latency CRF has "
                             "to beat.")
    parser.add_argument("--ema-alphas", default="",
                        help="comma-separated one-pole IIR coefficients for a zero-lookahead "
                             "exponential smoother over the emissions (s_t = (1-a) s_{t-1} + "
                             "a e_t). Effective averaging length ~1/a frames; a=1 is plain "
                             "argmax. O(V) state -- the cheapest smoother on the list.")
    parser.add_argument("--block-sync", default="",
                        help="comma-separated block sizes in POOLED label frames for "
                             "block-synchronous Viterbi decoding, or 'auto' to read the "
                             "backbone's own lookahead_block out of the mert_align_checkpoint. "
                             "Decodes every frame of a block at block completion using the "
                             "whole block -- free hindsight on a block-causal backbone, since "
                             "those frames were computed simultaneously. Zero added latency.")
    parser.add_argument("--dump-per-track", default=None, metavar="CSV",
                        help="also write per-track, per-decoder scores to this CSV. Decoders "
                             "share one backbone pass per track, so the rows are PAIRED and a "
                             "paired test on the differences is far more sensitive than "
                             "comparing the means' error bars.")
    parser.add_argument("--split", default="val", help="split to sweep (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only sweep the first N tracks of the split")
    parser.add_argument("--emission-mode", default="logprobs", choices=["logprobs", "logits"],
                        help="feed the CRF log-softmaxed probe outputs (default, makes "
                             "--self-bias interpretable in nats) or raw logits")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.config or not args.checkpoint:
        parser.error("config and checkpoint are required unless --self-test is given")

    run_sweep(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        self_biases=[float(x) for x in args.self_bias.split(",") if x.strip()],
        use_bigram=args.bigram_transitions,
        bigram_alpha=args.bigram_alpha,
        bigram_split=args.bigram_split,
        recount=args.recount,
        median_widths=[int(x) for x in args.median_widths.split(",") if x.strip()],
        causal_widths=[int(x) for x in args.causal_widths.split(",") if x.strip()],
        ema_alphas=[float(x) for x in args.ema_alphas.split(",") if x.strip()],
        block_syncs=[x.strip() for x in args.block_sync.split(",") if x.strip()],
        split=args.split,
        limit=args.limit,
        emission_mode=args.emission_mode,
        dump_per_track=args.dump_per_track,
    )


if __name__ == "__main__":
    main()
