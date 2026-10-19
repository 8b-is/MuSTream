"""The 121-class chord vocabulary: 10 qualities x 12 roots + N, closed under transposition.

Index layout: quality_index * 12 + root_index, with N last.

Usage:
  python chord_vocab.py                      # self-test
  python -m chord_vocab --write <path.pkl>    # write a dictionary pickle
"""

import pickle
from typing import Dict, Optional, Sequence, Tuple

import torch

NO_CHORD = "N"

# Sharp spelling, ascending from C; must match chord_dataloader._SHARP_ROOTS.
ROOTS: Tuple[str, ...] = ('C', 'C#', 'D', 'D#', 'E', 'F',
                          'F#', 'G', 'G#', 'A', 'A#', 'B')

# 10 qualities x 12 roots + N = 121 classes.
QUALITIES: Tuple[str, ...] = ('maj', 'min', '7', 'min7', 'maj7',
                              'sus4', 'dim', 'dim7', 'hdim7', 'aug')


def build_vocab(qualities: Sequence[str] = QUALITIES) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return (chord2idx, idx2chord) over qualities x ROOTS + NO_CHORD (quality-major, N last)."""
    idx2chord: Dict[int, str] = {}
    for q_i, quality in enumerate(qualities):
        for r_i, root in enumerate(ROOTS):
            idx2chord[q_i * 12 + r_i] = f'{root}:{quality}'
    idx2chord[12 * len(qualities)] = NO_CHORD
    chord2idx = {c: i for i, c in idx2chord.items()}
    return chord2idx, idx2chord


def split_label(label: str) -> Tuple[Optional[str], Optional[str]]:
    """('C#', 'min7') for 'C#:min7'; (None, None) for NO_CHORD."""
    if label == NO_CHORD:
        return None, None
    root, quality = label.split(':', 1)
    return root, quality


def transposed_label(label: str, n_steps: int) -> str:
    """`label` moved up n_steps semitones. NO_CHORD is fixed."""
    root, quality = split_label(label)
    if root is None:
        return NO_CHORD
    return f'{ROOTS[(ROOTS.index(root) + n_steps) % 12]}:{quality}'


def build_transpose_table(idx2chord: Dict[int, str]) -> torch.Tensor:
    """[V, 12] table: table[i, n] = class i transposed up n semitones.

    Raises if the vocabulary is not closed under transposition.
    """
    V = len(idx2chord)
    if set(idx2chord) != set(range(V)):
        raise ValueError(
            f"idx2chord must be indexed by exactly 0..{V - 1} to be usable as a gather "
            f"table; got {V} entries with keys outside that range"
        )
    chord2idx = {c: i for i, c in idx2chord.items()}
    table = torch.zeros(V, 12, dtype=torch.long)
    missing = set()
    for i in range(V):
        label = idx2chord[i]
        for n in range(12):
            moved = transposed_label(label, n)
            if moved not in chord2idx:
                missing.add(moved)
                continue
            table[i, n] = chord2idx[moved]
    if missing:
        raise ValueError(
            f"vocabulary is not closed under transposition: {len(missing)} label(s) "
            f"reachable by transposing a class in it are themselves absent, e.g. "
            f"{sorted(missing)[:6]}. Pitch-shift augmentation needs every quality at "
            f"all 12 roots -- rebuild the dictionary with chord_vocab.build_vocab() "
            f"(see this module's docstring for why the legacy 137-class one fails this)."
        )
    return table


def transpose_chord_index(
    labels: torch.Tensor,        # [...] class indices, any shape
    n_steps: torch.Tensor,       # broadcastable to labels' shape
    table: torch.Tensor,         # [V, 12] from build_transpose_table
) -> torch.Tensor:
    """Transpose class indices by n_steps semitones, elementwise (negative shifts allowed)."""
    table = table.to(labels.device)
    n = (n_steps % 12).to(device=labels.device, dtype=torch.long)
    return table[labels, n.expand_as(labels)]


def describe(idx2chord: Dict[int, str]) -> str:
    """One line for a training script's startup banner."""
    qualities = []
    for label in idx2chord.values():
        _root, quality = split_label(label)
        if quality is not None and quality not in qualities:
            qualities.append(quality)
    return (f"Vocabulary              : {len(idx2chord)} classes = "
            f"{len(qualities)} qualities x 12 roots + N ({', '.join(qualities)})")


def _write(path: str, qualities: Sequence[str] = QUALITIES) -> None:
    chord2idx, idx2chord = build_vocab(qualities)
    build_transpose_table(idx2chord)          # refuse to write a non-closed vocab
    with open(path, 'wb') as f:
        pickle.dump((chord2idx, idx2chord), f)
    print(f"Wrote {len(chord2idx)} classes -> {path}")
    print(describe(idx2chord))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--write', metavar='PATH',
                        help='write the (chord2idx, idx2chord) pickle here')
    parser.add_argument('--qualities', nargs='*', default=None,
                        help=f'override the quality list (default: {" ".join(QUALITIES)})')
    args = parser.parse_args()
    qualities = tuple(args.qualities) if args.qualities else QUALITIES

    if args.write:
        _write(args.write, qualities)
        raise SystemExit(0)

    # --- self-test ---------------------------------------------------------
    chord2idx, idx2chord = build_vocab()
    assert len(chord2idx) == 121, len(chord2idx)
    assert len(chord2idx) == len(idx2chord) == 12 * len(QUALITIES) + 1
    assert idx2chord[120] == NO_CHORD
    assert idx2chord[0] == 'C:maj' and idx2chord[11] == 'B:maj' and idx2chord[12] == 'C:min'
    print(describe(idx2chord))

    table = build_transpose_table(idx2chord)
    assert table.shape == (121, 12)
    # label arithmetic
    assert idx2chord[int(table[chord2idx['C:maj'], 7])] == 'G:maj'
    assert idx2chord[int(table[chord2idx['A#:min7'], 3])] == 'C#:min7'
    assert idx2chord[int(table[chord2idx['B:dim7'], 1])] == 'C:dim7'
    # N is fixed, quality never changes, 0 and 12 are the identity
    for i in range(121):
        assert int(table[i, 0]) == i
        _r, q = split_label(idx2chord[i])
        for n in range(12):
            _r2, q2 = split_label(idx2chord[int(table[i, n])])
            assert q2 == q, (idx2chord[i], n, idx2chord[int(table[i, n])])
    assert all(int(table[chord2idx[NO_CHORD], n]) == chord2idx[NO_CHORD] for n in range(12))
    print("transposition table OK")

    # per-frame labels with a per-item shift, and round-tripping
    labels = torch.randint(121, (4, 7))
    steps = torch.tensor([-4, 0, 3, 7]).unsqueeze(1)
    moved = transpose_chord_index(labels, steps, table)
    assert torch.equal(transpose_chord_index(moved, -steps, table), labels)
    print("per-frame transposition OK")

    # A non-closed vocabulary (a quality at only some roots) is refused.
    legacy_like = [f'{r}:maj' for r in ROOTS] + ['C:sus2', 'C#:sus2', 'D:sus2'] + [NO_CHORD]
    partial = dict(enumerate(legacy_like))
    try:
        build_transpose_table(partial)
        raise AssertionError("expected a ValueError for a non-closed vocabulary")
    except ValueError as exc:
        assert 'not closed under transposition' in str(exc), exc
    # ... and so is a dictionary that is not indexed 0..V-1
    holey = {i: c for i, c in idx2chord.items() if i != 65}
    try:
        build_transpose_table(holey)
        raise AssertionError("expected a ValueError for a non-contiguous dictionary")
    except ValueError as exc:
        assert 'indexed by exactly' in str(exc), exc
    print("closure + contiguity checks OK")
    print("chord_vocab self-test OK")
