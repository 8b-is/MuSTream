"""
Key vocabulary (24 classes), `.key` annotation parsing, the MIREX weighted
score, and the convergence metrics.

Self-test: python key_utils.py
"""

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import mir_eval
import numpy as np

# Canonical pitch-class spelling (sharps).
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MODES = ['major', 'minor']

# Class index = pitch_class + 12 * mode_index (0-11 major, 12-23 minor).
KEY_CLASSES: List[str] = [
    f'{pc} {mode}' for mode in MODES for pc in PITCH_CLASSES
]
N_KEY_CLASSES = len(KEY_CLASSES)
KEY_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(KEY_CLASSES)}

# Accepted root spellings (sharps and flats).
_ROOT_TO_PITCH_CLASS = {
    'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3, 'e': 4, 'f': 5,
    'f#': 6, 'gb': 6, 'g': 7, 'g#': 8, 'ab': 8, 'a': 9, 'a#': 10, 'bb': 10,
    'b': 11,
}


def save_txt(filepath, data) -> None:
    with open(filepath, "w", encoding="utf8") as f:
        for item in data:
            f.write(f"{item}\n")


def load_txt(filepath) -> List[str]:
    with open(filepath, encoding="utf8") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Vocabulary / annotation parsing
# ---------------------------------------------------------------------------

def normalize_key_name(key: str) -> Optional[str]:
    """Canonical '<pitch class> <mode>' name, or None if `key` is not exactly one major/minor key.

        >>> normalize_key_name('Bb minor'), normalize_key_name('a# minor')
        ('A# minor', 'A# minor')
    """
    if not key:
        return None
    parts = key.strip().lower().split()
    if len(parts) != 2:
        return None
    root, mode = parts
    if mode not in MODES or root not in _ROOT_TO_PITCH_CLASS:
        return None
    return f'{PITCH_CLASSES[_ROOT_TO_PITCH_CLASS[root]]} {mode}'


def key_name_to_index(key: str) -> Optional[int]:
    """Class index for a key string in any accepted spelling, else None."""
    canonical = normalize_key_name(key)
    return None if canonical is None else KEY_TO_INDEX[canonical]


def transpose_key_index(index, n_steps):
    """Key class of `index` transposed up by `n_steps` semitones (mode unchanged).

    Works elementwise on ints, numpy arrays and torch tensors.
    """
    mode = index // 12                        # 0 = major, 1 = minor
    return mode * 12 + (index % 12 + n_steps) % 12


def read_key_annotation(path) -> Optional[str]:
    """Canonical key from a `.key` annotation file (first tab field), or None if ambiguous."""
    with open(path) as f:
        line = f.readline()
    return normalize_key_name(line.split('\t')[0])


def read_key_confidence(path, default: int = 2) -> int:
    """MTG annotator confidence (2 clear, 1 doubtful, 0 no key); `default` if absent."""
    with open(path) as f:
        fields = f.readline().split('\t')
    if len(fields) < 2 or not fields[1].strip():
        return default
    try:
        return int(fields[1].strip())
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# MIREX weighted score
# ---------------------------------------------------------------------------

def _build_score_matrix() -> np.ndarray:
    """24x24 table of mir_eval.key.weighted_score, indexed [reference, estimate]."""
    matrix = np.zeros((N_KEY_CLASSES, N_KEY_CLASSES), dtype=np.float64)
    for ref, ref_name in enumerate(KEY_CLASSES):
        for est, est_name in enumerate(KEY_CLASSES):
            matrix[ref, est] = mir_eval.key.weighted_score(ref_name, est_name)
    return matrix


# Built once at import.
MIREX_SCORE_MATRIX = _build_score_matrix()


def weighted_score(reference: int, estimate: int) -> float:
    """MIREX weighted score for one (reference, estimate) class-index pair."""
    return float(MIREX_SCORE_MATRIX[reference, estimate])


def weighted_scores(reference: int, estimates: Sequence[int]) -> np.ndarray:
    """MIREX weighted score of each estimate against one reference."""
    return MIREX_SCORE_MATRIX[reference, np.asarray(estimates, dtype=int)]


def weighted_score_sum(references: Sequence[int], estimates: Sequence[int]) -> float:
    """Total MIREX weighted score over a batch of (reference, estimate) pairs."""
    references = np.asarray(references, dtype=int)
    estimates = np.asarray(estimates, dtype=int)
    return float(MIREX_SCORE_MATRIX[references, estimates].sum())


# ---------------------------------------------------------------------------
# Convergence metrics
# ---------------------------------------------------------------------------
# A running curve is (times, predictions): predictions[i] is the key estimate
# after times[i] seconds of audio (strictly increasing). Per track:
#   w_final / correct_final  weighted score / exactness of the last readout
#   taws                     time-averaged weighted score (trapezoid rule)
#   t_first_correct          first time the prediction is exactly right (NaN if never)
#   t_settle                 first time after which it stays right; NaN if the
#                            final prediction is wrong. settle_tol allows that
#                            fraction (by duration) of the remainder to be wrong.
#   n_flips / flip_rate      prediction changes, total and per minute


def convergence_metrics(
    times: Sequence[float],
    predictions: Sequence[int],
    reference: int,
    settle_tol: float = 0.0,
) -> Dict[str, float]:
    """Per-track convergence metrics for one running curve (see block comment above).

    Args:
        times:       seconds of audio consumed at each readout, strictly increasing
        predictions: key class index at each readout
        reference:   ground-truth key class index
        settle_tol:  fraction of the remainder (by duration) that may be wrong; 0 = strict
    """
    times = np.asarray(times, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=int)
    assert times.shape == predictions.shape, (times.shape, predictions.shape)
    assert times.size > 0, "empty prediction curve"
    assert np.all(np.diff(times) > 0), "times must be strictly increasing"
    assert 0.0 <= settle_tol < 1.0, f"settle_tol must be in [0, 1), got {settle_tol}"

    scores = weighted_scores(reference, predictions)
    correct = predictions == reference

    # Trapezoid weights per readout; a single readout gets weight 1.
    if times.size == 1:
        weights = np.ones(1)
    else:
        edges = np.concatenate(([times[0]], 0.5 * (times[:-1] + times[1:]), [times[-1]]))
        weights = np.diff(edges)

    correct_idx = np.flatnonzero(correct)
    t_first_correct = float(times[correct_idx[0]]) if correct_idx.size else float('nan')

    # Settling time: earliest readout whose remainder is (near-)all correct.
    t_settle = float('nan')
    if correct[-1]:
        wrong_weight = np.cumsum((~correct * weights)[::-1])[::-1]   # suffix sums
        total_weight = np.cumsum(weights[::-1])[::-1]
        within = (wrong_weight <= settle_tol * total_weight) & correct
        settle_idx = np.flatnonzero(within)
        if settle_idx.size:
            t_settle = float(times[settle_idx[0]])

    n_flips = int(np.count_nonzero(np.diff(predictions)))
    span = float(times[-1] - times[0])

    return {
        "w_final": float(scores[-1]),
        "correct_final": float(correct[-1]),
        "taws": float(np.sum(scores * weights) / np.sum(weights)),
        "t_first_correct": t_first_correct,
        "t_settle": t_settle,
        "n_flips": float(n_flips),
        "flip_rate": float(n_flips / span * 60.0) if span > 0 else 0.0,
        "duration": float(times[-1]),
    }


def prediction_at(times: Sequence[float], predictions: Sequence[int], t: float) -> Optional[int]:
    """Prediction in effect after t seconds of audio, or None if no readout yet."""
    times = np.asarray(times, dtype=np.float64)
    idx = int(np.searchsorted(times, t, side='right')) - 1
    return None if idx < 0 else int(predictions[idx])


def curve_at_grid(
    times: Sequence[float],
    predictions: Sequence[int],
    reference: int,
    grid: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """(weighted_score, correct) at each grid time; NaN past the track end or before the first readout."""
    scores = np.full(len(grid), np.nan)
    correct = np.full(len(grid), np.nan)
    t_end = float(np.asarray(times)[-1])
    for i, t in enumerate(grid):
        if t > t_end:
            continue
        pred = prediction_at(times, predictions, t)
        if pred is None:
            continue
        scores[i] = weighted_score(reference, pred)
        correct[i] = float(pred == reference)
    return scores, correct


# Prefix durations (s) at which W(t) / Acc(t) are reported.
DEFAULT_REPORT_GRID = (1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0)


def aggregate_convergence(
    per_track: Sequence[Dict[str, float]],
    grid_scores: Optional[Sequence[np.ndarray]] = None,
    grid_correct: Optional[Sequence[np.ndarray]] = None,
    grid: Sequence[float] = DEFAULT_REPORT_GRID,
) -> Dict[str, object]:
    """Pool per-track metrics (and optional curve_at_grid samples) into a run summary.

    t_first_correct / t_settle are averaged only over tracks where they are
    defined; p_correct_ever / p_settle give the coverage.
    """
    def _mean(key: str) -> float:
        values = np.array([m[key] for m in per_track], dtype=np.float64)
        values = values[~np.isnan(values)]
        return float(values.mean()) if values.size else float('nan')

    def _median(key: str) -> float:
        values = np.array([m[key] for m in per_track], dtype=np.float64)
        values = values[~np.isnan(values)]
        return float(np.median(values)) if values.size else float('nan')

    n = len(per_track)
    first_correct = np.array([m["t_first_correct"] for m in per_track], dtype=np.float64)
    settle = np.array([m["t_settle"] for m in per_track], dtype=np.float64)

    summary: Dict[str, object] = {
        "n": n,
        "weighted_score": _mean("w_final"),
        "accuracy": _mean("correct_final"),
        "taws": _mean("taws"),
        "t_first_correct_mean": _mean("t_first_correct"),
        "t_first_correct_median": _median("t_first_correct"),
        "p_correct_ever": float(np.count_nonzero(~np.isnan(first_correct)) / n) if n else float('nan'),
        "t_settle_mean": _mean("t_settle"),
        "t_settle_median": _median("t_settle"),
        "p_settle": float(np.count_nonzero(~np.isnan(settle)) / n) if n else float('nan'),
        "flip_rate": _mean("flip_rate"),
    }

    def _nanmean_columns(rows) -> list:
        """Column means ignoring NaN (all-NaN columns stay NaN, silently)."""
        stacked = np.stack([np.asarray(r, dtype=np.float64) for r in rows])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(stacked, axis=0).tolist(), np.count_nonzero(
                ~np.isnan(stacked), axis=0).tolist()

    if grid_scores is not None:
        summary["grid"] = list(grid)
        summary["grid_weighted_score"], summary["grid_n"] = _nanmean_columns(grid_scores)
    if grid_correct is not None:
        summary["grid_accuracy"], _ = _nanmean_columns(grid_correct)
    return summary


def json_safe(value):
    """Replace NaN with None recursively, for strict JSON output."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return json_safe(value.item())
    return value


def format_summary(summary: Dict[str, object], name: str = "Key") -> str:
    """Human-readable report; the first line is the single-line global metric."""
    def fmt(value, suffix: str = "", digits: int = 2) -> str:
        """Format a number; NaN/None prints as 'n/a'."""
        return "n/a" if value is None or np.isnan(value) else f"{value:.{digits}f}{suffix}"

    lines = [
        f"{name} (n={summary['n']}) — weighted_score: {summary['weighted_score']:.4f}  "
        f"accuracy: {summary['accuracy']:.4f}  TAWS: {summary['taws']:.4f}",
        f"  t_first_correct: mean {fmt(summary['t_first_correct_mean'], 's')}  "
        f"median {fmt(summary['t_first_correct_median'], 's')}  "
        f"(ever correct on {summary['p_correct_ever'] * 100:.1f}% of tracks)",
        f"  t_settle:        mean {fmt(summary['t_settle_mean'], 's')}  "
        f"median {fmt(summary['t_settle_median'], 's')}  "
        f"(settles on {summary['p_settle'] * 100:.1f}% of tracks)",
        f"  flip_rate:       {fmt(summary['flip_rate'])} changes/min",
    ]
    if "grid_weighted_score" in summary:
        grid = summary["grid"]
        lines.append("  W(t): " + "  ".join(
            f"{t:g}s={fmt(w, digits=3)}" for t, w in zip(grid, summary["grid_weighted_score"])
        ))
    if "grid_accuracy" in summary:
        grid = summary["grid"]
        lines.append("  Acc(t): " + "  ".join(
            f"{t:g}s={fmt(a, digits=3)}" for t, a in zip(grid, summary["grid_accuracy"])
        ))
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-test.
    assert len(KEY_CLASSES) == 24 and KEY_CLASSES[0] == 'C major' and KEY_CLASSES[12] == 'C minor'
    assert normalize_key_name('Bb minor') == normalize_key_name('a# minor') == 'A# minor'
    assert normalize_key_name('C major') == 'C major'
    for ambiguous in ('-', '', 'g minor/major', 'a minor / d major', 'a-a# minor', 'x'):
        assert normalize_key_name(ambiguous) is None, ambiguous

    c_major, g_major, a_minor, c_minor, d_major = (
        KEY_TO_INDEX[k] for k in ('C major', 'G major', 'A minor', 'C minor', 'D major'))
    assert weighted_score(c_major, c_major) == 1.0
    assert weighted_score(c_major, g_major) == 0.5    # perfect fifth up
    assert weighted_score(c_major, a_minor) == 0.3    # relative minor
    assert weighted_score(a_minor, c_major) == 0.3    # relative major
    assert weighted_score(c_major, c_minor) == 0.2    # parallel minor
    assert weighted_score(c_major, d_major) == 0.0

    # Wrong for the first half, right for the second: settles at the halfway
    # point, TAWS ~ 0.5, one flip.
    times = np.arange(1, 11, dtype=float)
    preds = [d_major] * 5 + [c_major] * 5
    m = convergence_metrics(times, preds, c_major)
    assert m["correct_final"] == 1.0 and m["w_final"] == 1.0
    assert m["t_first_correct"] == 6.0 and m["t_settle"] == 6.0
    assert abs(m["taws"] - 0.5) < 0.02, m["taws"]
    assert m["n_flips"] == 1.0

    # Right early, then one blip, then right again: t_first_correct stays early,
    # strict t_settle moves past the blip, and a tolerance recovers it.
    preds = [c_major] * 3 + [g_major] + [c_major] * 6
    m = convergence_metrics(times, preds, c_major)
    assert m["t_first_correct"] == 1.0 and m["t_settle"] == 5.0, m
    m_tol = convergence_metrics(times, preds, c_major, settle_tol=0.2)
    assert m_tol["t_settle"] == 1.0, m_tol

    # Never right: both times undefined, and the relative-minor guess still
    # earns partial credit.
    m = convergence_metrics(times, [a_minor] * 10, c_major)
    assert np.isnan(m["t_first_correct"]) and np.isnan(m["t_settle"])
    assert abs(m["taws"] - 0.3) < 1e-9 and m["correct_final"] == 0.0

    # Grid sampling holds the last readout at or before t, and NaNs past the end.
    scores, correct = curve_at_grid(times, [d_major] * 5 + [c_major] * 5, c_major, [0.5, 3.0, 8.0, 20.0])
    assert np.isnan(scores[0]) and scores[1] == 0.0 and scores[2] == 1.0 and np.isnan(scores[3])
    assert np.isnan(correct[0]) and correct[1] == 0.0 and correct[2] == 1.0

    print(format_summary(aggregate_convergence(
        [convergence_metrics(times, [d_major] * 5 + [c_major] * 5, c_major),
         convergence_metrics(times, [a_minor] * 10, c_major)],
        grid=(1.0, 5.0, 10.0),
        grid_scores=[curve_at_grid(times, [d_major] * 5 + [c_major] * 5, c_major, (1.0, 5.0, 10.0))[0],
                     curve_at_grid(times, [a_minor] * 10, c_major, (1.0, 5.0, 10.0))[0]],
    )))
    print("key_utils self-test OK")
