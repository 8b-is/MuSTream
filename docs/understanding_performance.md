# Music-understanding performance: internal vs. MTG-Jamendo backbones

The results reported in the paper come from MuSTream backbones distilled on an
internal music collection with release restrictions. To make the pipeline fully
reproducible and to investigate the robustness of our method, we distilled a second
set of backbones on the public [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) dataset, using the same
recipe (see [`mustream/README.md`](../mustream/README.md), sections 1–2). The
released checkpoints are this MTG-Jamendo set.

This page compares the two sets on the downstream understanding tasks.

## Setup

- **Only the distillation data differs.** Architecture, distillation settings,
  probes (chord, beat / downbeat, key) and evaluation data are the same for both sets.
- **The internal-data results are copied from the paper.** See the paper for
  details on the MuSTream models, teacher configurations ($\ell$, $p$) and metrics.
- **MERT-26M** is the bidirectional 26M student from the capacity-distillation stage
  and the teacher of MuSTream-26M. Since it is not causal, it is evaluated offline
  and in two inference modes that approximate streaming:
  - offline: 10 s windows;
  - chunked: independent 0.2 s chunks;
  - sliding: a 5.0 s window advanced by 0.2 s.

## Results

### Internal data (paper, 26.6k hours)

#### MuSTream (streaming)

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Inference mode</th>
      <th colspan="3">Chord Recognition</th>
      <th colspan="3">Beat Tracking</th>
      <th colspan="3">Downbeat Tracking</th>
      <th colspan="2">Key Detection</th>
    </tr>
    <tr>
      <th>MajMin</th>
      <th>Sevenths</th>
      <th>MIREX</th>
      <th>F1</th>
      <th>CMLt</th>
      <th>AMLt</th>
      <th>F1</th>
      <th>CMLt</th>
      <th>AMLt</th>
      <th>Weighted</th>
      <th>TAWS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MuSTream-330M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right"><b>0.731</b></td>
      <td align="right"><b>0.613</b></td>
      <td align="right"><b>0.745</b></td>
      <td align="right">0.790</td>
      <td align="right">0.621</td>
      <td align="right">0.759</td>
      <td align="right">0.618</td>
      <td align="right">0.509</td>
      <td align="right">0.681</td>
      <td align="right"><b>0.650</b></td>
      <td align="right"><b>0.645</b></td>
    </tr>
    <tr>
      <td>MuSTream-330M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.722</td>
      <td align="right">0.604</td>
      <td align="right">0.736</td>
      <td align="right"><b>0.798</b></td>
      <td align="right"><b>0.638</b></td>
      <td align="right"><b>0.772</b></td>
      <td align="right"><b>0.638</b></td>
      <td align="right"><b>0.531</b></td>
      <td align="right"><b>0.700</b></td>
      <td align="right">0.588</td>
      <td align="right">0.601</td>
    </tr>
    <tr>
      <td>MuSTream-95M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right">0.715</td>
      <td align="right">0.606</td>
      <td align="right">0.723</td>
      <td align="right">0.757</td>
      <td align="right">0.571</td>
      <td align="right">0.721</td>
      <td align="right">0.543</td>
      <td align="right">0.442</td>
      <td align="right">0.630</td>
      <td align="right">0.604</td>
      <td align="right">0.607</td>
    </tr>
    <tr>
      <td>MuSTream-95M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.713</td>
      <td align="right">0.600</td>
      <td align="right">0.722</td>
      <td align="right">0.771</td>
      <td align="right">0.595</td>
      <td align="right">0.741</td>
      <td align="right">0.557</td>
      <td align="right">0.454</td>
      <td align="right">0.643</td>
      <td align="right">0.596</td>
      <td align="right">0.601</td>
    </tr>
    <tr>
      <td>MuSTream-26M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right">0.686</td>
      <td align="right">0.579</td>
      <td align="right">0.697</td>
      <td align="right">0.732</td>
      <td align="right">0.529</td>
      <td align="right">0.696</td>
      <td align="right">0.463</td>
      <td align="right">0.360</td>
      <td align="right">0.577</td>
      <td align="right">0.593</td>
      <td align="right">0.592</td>
    </tr>
    <tr>
      <td>MuSTream-26M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.687</td>
      <td align="right">0.581</td>
      <td align="right">0.697</td>
      <td align="right">0.751</td>
      <td align="right">0.565</td>
      <td align="right">0.718</td>
      <td align="right">0.490</td>
      <td align="right">0.387</td>
      <td align="right">0.590</td>
      <td align="right">0.598</td>
      <td align="right">0.596</td>
    </tr>
  </tbody>
</table>

#### MERT-26M (bidirectional, distilled from MERT-95M)

<table>
  <thead>
    <tr>
      <th rowspan="2">Inference mode</th>
      <th colspan="3">Chord Recognition</th>
      <th colspan="3">Beat Tracking</th>
      <th colspan="3">Downbeat Tracking</th>
      <th colspan="2">Key Detection</th>
    </tr>
    <tr>
      <th>MajMin</th>
      <th>Sevenths</th>
      <th>MIREX</th>
      <th>F1</th>
      <th>CMLt</th>
      <th>AMLt</th>
      <th>F1</th>
      <th>CMLt</th>
      <th>AMLt</th>
      <th>Weighted</th>
      <th>TAWS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>offline (10 s)</td>
      <td align="right"><b>0.720</b></td>
      <td align="right"><b>0.616</b></td>
      <td align="right"><b>0.726</b></td>
      <td align="right"><b>0.790</b></td>
      <td align="right"><b>0.652</b></td>
      <td align="right"><b>0.827</b></td>
      <td align="right"><b>0.579</b></td>
      <td align="right"><b>0.521</b></td>
      <td align="right"><b>0.778</b></td>
      <td align="right">0.632</td>
      <td align="right"><b>0.630</b></td>
    </tr>
    <tr>
      <td>chunked (0.2 s)</td>
      <td align="right">0.701</td>
      <td align="right">0.595</td>
      <td align="right">0.708</td>
      <td align="right">0.693</td>
      <td align="right">0.463</td>
      <td align="right">0.649</td>
      <td align="right">0.426</td>
      <td align="right">0.312</td>
      <td align="right">0.536</td>
      <td align="right">0.610</td>
      <td align="right">0.611</td>
    </tr>
    <tr>
      <td>sliding (5.0 s, 0.2 s)</td>
      <td align="right">0.703</td>
      <td align="right">0.596</td>
      <td align="right">0.712</td>
      <td align="right">0.739</td>
      <td align="right">0.530</td>
      <td align="right">0.701</td>
      <td align="right">0.469</td>
      <td align="right">0.367</td>
      <td align="right">0.581</td>
      <td align="right"><b>0.635</b></td>
      <td align="right">0.625</td>
    </tr>
  </tbody>
</table>

### MTG-Jamendo (public data, released checkpoints, 3.8k hours)

#### MuSTream (streaming)

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Inference mode</th>
      <th colspan="3">Chord Recognition</th>
      <th colspan="3">Beat Tracking</th>
      <th colspan="3">Downbeat Tracking</th>
      <th colspan="2">Key Detection</th>
    </tr>
    <tr>
      <th>MajMin</th>
      <th>Sevenths</th>
      <th>MIREX</th>
      <th>F1</th>
      <th>CMLt</th>
      <th>AMLt</th>
      <th>F1</th>
      <th>CMLt</th>
      <th>AMLt</th>
      <th>Weighted</th>
      <th>TAWS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MuSTream-330M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right"><b>0.737</b></td>
      <td align="right"><b>0.623</b></td>
      <td align="right"><b>0.747</b></td>
      <td align="right">0.776</td>
      <td align="right">0.593</td>
      <td align="right">0.736</td>
      <td align="right">0.614</td>
      <td align="right">0.500</td>
      <td align="right">0.678</td>
      <td align="right"><b>0.657</b></td>
      <td align="right"><b>0.645</b></td>
    </tr>
    <tr>
      <td>MuSTream-330M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.728</td>
      <td align="right">0.612</td>
      <td align="right">0.742</td>
      <td align="right"><b>0.793</b></td>
      <td align="right">0.623</td>
      <td align="right"><b>0.766</b></td>
      <td align="right"><b>0.635</b></td>
      <td align="right"><b>0.523</b></td>
      <td align="right"><b>0.704</b></td>
      <td align="right">0.590</td>
      <td align="right">0.603</td>
    </tr>
    <tr>
      <td>MuSTream-95M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right">0.705</td>
      <td align="right">0.589</td>
      <td align="right">0.714</td>
      <td align="right">0.773</td>
      <td align="right">0.608</td>
      <td align="right">0.734</td>
      <td align="right">0.585</td>
      <td align="right">0.490</td>
      <td align="right">0.655</td>
      <td align="right">0.599</td>
      <td align="right">0.600</td>
    </tr>
    <tr>
      <td>MuSTream-95M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.704</td>
      <td align="right">0.586</td>
      <td align="right">0.713</td>
      <td align="right">0.789</td>
      <td align="right"><b>0.639</b></td>
      <td align="right">0.759</td>
      <td align="right">0.598</td>
      <td align="right">0.506</td>
      <td align="right">0.670</td>
      <td align="right">0.620</td>
      <td align="right">0.626</td>
    </tr>
    <tr>
      <td>MuSTream-26M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right">0.678</td>
      <td align="right">0.568</td>
      <td align="right">0.689</td>
      <td align="right">0.732</td>
      <td align="right">0.529</td>
      <td align="right">0.701</td>
      <td align="right">0.463</td>
      <td align="right">0.357</td>
      <td align="right">0.576</td>
      <td align="right">0.622</td>
      <td align="right">0.613</td>
    </tr>
    <tr>
      <td>MuSTream-26M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.671</td>
      <td align="right">0.562</td>
      <td align="right">0.682</td>
      <td align="right">0.735</td>
      <td align="right">0.538</td>
      <td align="right">0.700</td>
      <td align="right">0.491</td>
      <td align="right">0.384</td>
      <td align="right">0.588</td>
      <td align="right">0.624</td>
      <td align="right">0.620</td>
    </tr>
  </tbody>
</table>

#### MERT-26M (bidirectional, distilled from MERT-95M)

<table>
  <thead>
    <tr>
      <th rowspan="2">Inference mode</th>
      <th colspan="3">Chord Recognition</th>
      <th colspan="3">Beat Tracking</th>
      <th colspan="3">Downbeat Tracking</th>
      <th colspan="2">Key Detection</th>
    </tr>
    <tr>
      <th>MajMin</th>
      <th>Sevenths</th>
      <th>MIREX</th>
      <th>F1</th>
      <th>CMLt</th>
      <th>AMLt</th>
      <th>F1</th>
      <th>CMLt</th>
      <th>AMLt</th>
      <th>Weighted</th>
      <th>TAWS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>offline (10 s)</td>
      <td align="right"><b>0.723</b></td>
      <td align="right"><b>0.614</b></td>
      <td align="right"><b>0.731</b></td>
      <td align="right"><b>0.775</b></td>
      <td align="right"><b>0.617</b></td>
      <td align="right"><b>0.826</b></td>
      <td align="right"><b>0.569</b></td>
      <td align="right"><b>0.509</b></td>
      <td align="right"><b>0.781</b></td>
      <td align="right"><b>0.651</b></td>
      <td align="right"><b>0.644</b></td>
    </tr>
    <tr>
      <td>chunked (0.2 s)</td>
      <td align="right">0.698</td>
      <td align="right">0.588</td>
      <td align="right">0.707</td>
      <td align="right">0.675</td>
      <td align="right">0.415</td>
      <td align="right">0.636</td>
      <td align="right">0.410</td>
      <td align="right">0.300</td>
      <td align="right">0.528</td>
      <td align="right">0.645</td>
      <td align="right">0.630</td>
    </tr>
    <tr>
      <td>sliding (5.0 s, 0.2 s)</td>
      <td align="right">0.701</td>
      <td align="right">0.594</td>
      <td align="right">0.712</td>
      <td align="right">0.736</td>
      <td align="right">0.527</td>
      <td align="right">0.698</td>
      <td align="right">0.503</td>
      <td align="right">0.398</td>
      <td align="right">0.600</td>
      <td align="right">0.632</td>
      <td align="right">0.630</td>
    </tr>
  </tbody>
</table>

## Observations

- **Overall, the two sets perform about the same, showing that our distillation method is robust to the training data source and size.** Most differences are within ±0.02, and neither set is better across the board.
- **The trends in the paper still hold with MTG-Jamendo:**
  - Larger backbones are better on chord and downbeat.
  - The teacher that sees 0.2 s past each block ($\ell = 15$, $p = 15$) improves beat and downbeat tracking at every size.
  - With MERT-26M, offline inference is best, followed by sliding, then chunked.
  - The streaming MuSTream-26M matches or beats MERT-26M in chunked mode on beat and downbeat, and comes close to it in sliding mode.
