# Key detection: full results

In the paper, key detection behaves less consistently across streaming configurations than chord recognition and beat tracking. This page gives the full key-detection results behind that statement, for the models trained on internal data (the same models as the paper's main table). All results are on the 604-track GiantSteps Key test set and are averaged over three probe seeds.

## Metrics

**Whole-track metrics** (higher is better):
  - *Weighted*: MIREX weighted score of the whole-track prediction (1 for the correct key, partial credit for fifth, relative and parallel errors);
  - *Accuracy*: share of tracks whose whole-track prediction is exactly correct;
  - *TAWS*: time-averaged weighted score over the whole running estimate.

**Convergence metrics**, computed per track from the running estimate:
  - *First correct*: time of the first exactly correct estimate. *Ever correct* is the share of tracks that have one.
  - *Settle*: time after which the estimate stays correct, allowing up to 5% of the remaining audio to be wrong. It is defined only for tracks whose final estimate is correct, so the share of settled tracks equals Accuracy.
  - First correct and Settle times are medians over the tracks where they are defined (lower is better).
  - *Flips / min*: how often the estimate changes, per minute of audio.
  - $W(t)$: weighted score of the running estimate after $t$ seconds of audio.

## Whole-track results

The best value in each column is in bold.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Inference mode</th>
      <th>Weighted</th>
      <th>Accuracy</th>
      <th>TAWS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">MERT-330M</td>
      <td>Chunked</td>
      <td align="right">0.618</td>
      <td align="right">0.554</td>
      <td align="right">0.628</td>
    </tr>
    <tr>
      <td>Sliding</td>
      <td align="right">0.624</td>
      <td align="right">0.563</td>
      <td align="right">0.627</td>
    </tr>
    <tr>
      <td rowspan="3">MuSTream-330M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right"><b>0.650</b></td>
      <td align="right"><b>0.591</b></td>
      <td align="right"><b>0.645</b></td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.588</td>
      <td align="right">0.512</td>
      <td align="right">0.601</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 30, <i>p</i> = 0</td>
      <td align="right">0.602</td>
      <td align="right">0.538</td>
      <td align="right">0.619</td>
    </tr>
    <tr>
      <td rowspan="2">MERT-95M</td>
      <td>Chunked</td>
      <td align="right">0.617</td>
      <td align="right">0.551</td>
      <td align="right">0.618</td>
    </tr>
    <tr>
      <td>Sliding</td>
      <td align="right">0.618</td>
      <td align="right">0.557</td>
      <td align="right">0.617</td>
    </tr>
    <tr>
      <td rowspan="3">MuSTream-95M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right">0.604</td>
      <td align="right">0.540</td>
      <td align="right">0.607</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.596</td>
      <td align="right">0.530</td>
      <td align="right">0.601</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 30, <i>p</i> = 0</td>
      <td align="right">0.629</td>
      <td align="right">0.567</td>
      <td align="right">0.625</td>
    </tr>
    <tr>
      <td rowspan="2">MERT-26M</td>
      <td>Chunked</td>
      <td align="right">0.610</td>
      <td align="right">0.542</td>
      <td align="right">0.611</td>
    </tr>
    <tr>
      <td>Sliding</td>
      <td align="right">0.635</td>
      <td align="right">0.570</td>
      <td align="right">0.625</td>
    </tr>
    <tr>
      <td rowspan="3">MuSTream-26M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right">0.593</td>
      <td align="right">0.520</td>
      <td align="right">0.592</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.598</td>
      <td align="right">0.527</td>
      <td align="right">0.596</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 30, <i>p</i> = 0</td>
      <td align="right">0.602</td>
      <td align="right">0.533</td>
      <td align="right">0.597</td>
    </tr>
  </tbody>
</table>

## Convergence

How fast the running estimate becomes correct, and how stable it is. The fastest times and the highest *Ever correct* are in bold. *Readout every* is how often a model updates its estimate. *Flips / min* grows with the readout rate, so compare it only between models with the same readout interval.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Inference mode</th>
      <th>Readout every</th>
      <th>First correct, median (s)</th>
      <th>Ever correct</th>
      <th>Settle, median (s)</th>
      <th>Flips / min</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">MERT-330M</td>
      <td>Chunked</td>
      <td align="right">0.2 s</td>
      <td align="right"><b>0.58</b></td>
      <td align="right">0.836</td>
      <td align="right">1.39</td>
      <td align="right">12.9</td>
    </tr>
    <tr>
      <td>Sliding</td>
      <td align="right">1 s</td>
      <td align="right">1.63</td>
      <td align="right">0.804</td>
      <td align="right">2.25</td>
      <td align="right">3.4</td>
    </tr>
    <tr>
      <td rowspan="3">MuSTream-330M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right">0.2 s</td>
      <td align="right">0.72</td>
      <td align="right">0.835</td>
      <td align="right"><b>1.35</b></td>
      <td align="right">10.3</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.2 s</td>
      <td align="right">0.87</td>
      <td align="right">0.831</td>
      <td align="right">2.13</td>
      <td align="right">13.9</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 30, <i>p</i> = 0</td>
      <td align="right">0.4 s</td>
      <td align="right">0.93</td>
      <td align="right">0.822</td>
      <td align="right">1.80</td>
      <td align="right">7.0</td>
    </tr>
    <tr>
      <td rowspan="2">MERT-95M</td>
      <td>Chunked</td>
      <td align="right">0.2 s</td>
      <td align="right">0.66</td>
      <td align="right">0.846</td>
      <td align="right">2.87</td>
      <td align="right">14.8</td>
    </tr>
    <tr>
      <td>Sliding</td>
      <td align="right">1 s</td>
      <td align="right">2.00</td>
      <td align="right">0.816</td>
      <td align="right">3.63</td>
      <td align="right">4.2</td>
    </tr>
    <tr>
      <td rowspan="3">MuSTream-95M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right">0.2 s</td>
      <td align="right">0.65</td>
      <td align="right">0.852</td>
      <td align="right">2.93</td>
      <td align="right">17.8</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.2 s</td>
      <td align="right">0.73</td>
      <td align="right">0.840</td>
      <td align="right">2.81</td>
      <td align="right">15.6</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 30, <i>p</i> = 0</td>
      <td align="right">0.4 s</td>
      <td align="right">1.17</td>
      <td align="right">0.845</td>
      <td align="right">3.28</td>
      <td align="right">10.0</td>
    </tr>
    <tr>
      <td rowspan="2">MERT-26M</td>
      <td>Chunked</td>
      <td align="right">0.2 s</td>
      <td align="right">0.77</td>
      <td align="right">0.857</td>
      <td align="right">3.58</td>
      <td align="right">20.0</td>
    </tr>
    <tr>
      <td>Sliding</td>
      <td align="right">1 s</td>
      <td align="right">2.00</td>
      <td align="right">0.836</td>
      <td align="right">5.33</td>
      <td align="right">5.5</td>
    </tr>
    <tr>
      <td rowspan="3">MuSTream-26M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right">0.2 s</td>
      <td align="right">0.92</td>
      <td align="right"><b>0.860</b></td>
      <td align="right">6.57</td>
      <td align="right">24.3</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.2 s</td>
      <td align="right">0.80</td>
      <td align="right">0.847</td>
      <td align="right">4.64</td>
      <td align="right">18.4</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 30, <i>p</i> = 0</td>
      <td align="right">0.4 s</td>
      <td align="right">1.33</td>
      <td align="right">0.854</td>
      <td align="right">11.23</td>
      <td align="right">15.6</td>
    </tr>
  </tbody>
</table>

## Running estimate over time

Weighted score $W(t)$ after $t$ seconds of audio. At each $t$, only tracks at least $t$ seconds long are counted (all 604 tracks up to 30 s, 573 at 120 s).

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Inference mode</th>
      <th>1 s</th>
      <th>2 s</th>
      <th>3 s</th>
      <th>5 s</th>
      <th>7.5 s</th>
      <th>10 s</th>
      <th>15 s</th>
      <th>20 s</th>
      <th>30 s</th>
      <th>45 s</th>
      <th>60 s</th>
      <th>90 s</th>
      <th>120 s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">MERT-330M</td>
      <td>Chunked</td>
      <td align="right"><b>0.479</b></td>
      <td align="right"><b>0.537</b></td>
      <td align="right"><b>0.561</b></td>
      <td align="right">0.596</td>
      <td align="right">0.628</td>
      <td align="right">0.629</td>
      <td align="right">0.636</td>
      <td align="right">0.631</td>
      <td align="right">0.637</td>
      <td align="right">0.642</td>
      <td align="right">0.639</td>
      <td align="right">0.630</td>
      <td align="right">0.615</td>
    </tr>
    <tr>
      <td>Sliding</td>
      <td align="right">0.449</td>
      <td align="right">0.520</td>
      <td align="right">0.553</td>
      <td align="right"><b>0.605</b></td>
      <td align="right"><b>0.632</b></td>
      <td align="right"><b>0.634</b></td>
      <td align="right">0.635</td>
      <td align="right">0.631</td>
      <td align="right">0.627</td>
      <td align="right">0.635</td>
      <td align="right">0.634</td>
      <td align="right">0.630</td>
      <td align="right">0.623</td>
    </tr>
    <tr>
      <td rowspan="3">MuSTream-330M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right">0.439</td>
      <td align="right">0.513</td>
      <td align="right">0.543</td>
      <td align="right">0.590</td>
      <td align="right">0.627</td>
      <td align="right">0.630</td>
      <td align="right">0.636</td>
      <td align="right"><b>0.643</b></td>
      <td align="right"><b>0.647</b></td>
      <td align="right"><b>0.655</b></td>
      <td align="right"><b>0.654</b></td>
      <td align="right"><b>0.657</b></td>
      <td align="right"><b>0.650</b></td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.426</td>
      <td align="right">0.499</td>
      <td align="right">0.530</td>
      <td align="right">0.573</td>
      <td align="right">0.619</td>
      <td align="right">0.624</td>
      <td align="right">0.625</td>
      <td align="right">0.619</td>
      <td align="right">0.612</td>
      <td align="right">0.611</td>
      <td align="right">0.608</td>
      <td align="right">0.599</td>
      <td align="right">0.588</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 30, <i>p</i> = 0</td>
      <td align="right">0.441</td>
      <td align="right">0.520</td>
      <td align="right">0.550</td>
      <td align="right">0.590</td>
      <td align="right">0.629</td>
      <td align="right">0.628</td>
      <td align="right"><b>0.637</b></td>
      <td align="right">0.635</td>
      <td align="right">0.631</td>
      <td align="right">0.632</td>
      <td align="right">0.623</td>
      <td align="right">0.618</td>
      <td align="right">0.600</td>
    </tr>
    <tr>
      <td rowspan="2">MERT-95M</td>
      <td>Chunked</td>
      <td align="right">0.465</td>
      <td align="right">0.511</td>
      <td align="right">0.543</td>
      <td align="right">0.582</td>
      <td align="right">0.607</td>
      <td align="right">0.610</td>
      <td align="right">0.613</td>
      <td align="right">0.610</td>
      <td align="right">0.620</td>
      <td align="right">0.626</td>
      <td align="right">0.621</td>
      <td align="right">0.624</td>
      <td align="right">0.613</td>
    </tr>
    <tr>
      <td>Sliding</td>
      <td align="right">0.427</td>
      <td align="right">0.496</td>
      <td align="right">0.536</td>
      <td align="right">0.583</td>
      <td align="right">0.612</td>
      <td align="right">0.614</td>
      <td align="right">0.623</td>
      <td align="right">0.622</td>
      <td align="right">0.623</td>
      <td align="right">0.627</td>
      <td align="right">0.621</td>
      <td align="right">0.627</td>
      <td align="right">0.620</td>
    </tr>
    <tr>
      <td rowspan="3">MuSTream-95M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right">0.443</td>
      <td align="right">0.511</td>
      <td align="right">0.534</td>
      <td align="right">0.577</td>
      <td align="right">0.614</td>
      <td align="right">0.621</td>
      <td align="right">0.619</td>
      <td align="right">0.622</td>
      <td align="right">0.616</td>
      <td align="right">0.622</td>
      <td align="right">0.613</td>
      <td align="right">0.607</td>
      <td align="right">0.603</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.447</td>
      <td align="right">0.519</td>
      <td align="right">0.547</td>
      <td align="right">0.585</td>
      <td align="right">0.616</td>
      <td align="right">0.618</td>
      <td align="right">0.618</td>
      <td align="right">0.616</td>
      <td align="right">0.611</td>
      <td align="right">0.610</td>
      <td align="right">0.601</td>
      <td align="right">0.598</td>
      <td align="right">0.591</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 30, <i>p</i> = 0</td>
      <td align="right">0.419</td>
      <td align="right">0.502</td>
      <td align="right">0.529</td>
      <td align="right">0.575</td>
      <td align="right">0.614</td>
      <td align="right">0.621</td>
      <td align="right">0.617</td>
      <td align="right">0.620</td>
      <td align="right">0.625</td>
      <td align="right">0.640</td>
      <td align="right">0.634</td>
      <td align="right">0.633</td>
      <td align="right">0.628</td>
    </tr>
    <tr>
      <td rowspan="2">MERT-26M</td>
      <td>Chunked</td>
      <td align="right">0.450</td>
      <td align="right">0.511</td>
      <td align="right">0.547</td>
      <td align="right">0.578</td>
      <td align="right">0.601</td>
      <td align="right">0.609</td>
      <td align="right">0.611</td>
      <td align="right">0.609</td>
      <td align="right">0.611</td>
      <td align="right">0.620</td>
      <td align="right">0.618</td>
      <td align="right">0.619</td>
      <td align="right">0.606</td>
    </tr>
    <tr>
      <td>Sliding</td>
      <td align="right">0.432</td>
      <td align="right">0.499</td>
      <td align="right">0.536</td>
      <td align="right">0.578</td>
      <td align="right">0.605</td>
      <td align="right">0.614</td>
      <td align="right">0.619</td>
      <td align="right">0.614</td>
      <td align="right">0.615</td>
      <td align="right">0.633</td>
      <td align="right">0.628</td>
      <td align="right">0.637</td>
      <td align="right">0.634</td>
    </tr>
    <tr>
      <td rowspan="3">MuSTream-26M</td>
      <td><i>ℓ</i> = 15, <i>p</i> = 0</td>
      <td align="right">0.415</td>
      <td align="right">0.498</td>
      <td align="right">0.519</td>
      <td align="right">0.565</td>
      <td align="right">0.594</td>
      <td align="right">0.605</td>
      <td align="right">0.598</td>
      <td align="right">0.601</td>
      <td align="right">0.596</td>
      <td align="right">0.605</td>
      <td align="right">0.599</td>
      <td align="right">0.599</td>
      <td align="right">0.589</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 15, <i>p</i> = 15</td>
      <td align="right">0.431</td>
      <td align="right">0.491</td>
      <td align="right">0.518</td>
      <td align="right">0.560</td>
      <td align="right">0.588</td>
      <td align="right">0.594</td>
      <td align="right">0.599</td>
      <td align="right">0.592</td>
      <td align="right">0.599</td>
      <td align="right">0.606</td>
      <td align="right">0.608</td>
      <td align="right">0.594</td>
      <td align="right">0.597</td>
    </tr>
    <tr>
      <td><i>ℓ</i> = 30, <i>p</i> = 0</td>
      <td align="right">0.381</td>
      <td align="right">0.481</td>
      <td align="right">0.504</td>
      <td align="right">0.555</td>
      <td align="right">0.586</td>
      <td align="right">0.597</td>
      <td align="right">0.600</td>
      <td align="right">0.596</td>
      <td align="right">0.595</td>
      <td align="right">0.606</td>
      <td align="right">0.604</td>
      <td align="right">0.605</td>
      <td align="right">0.601</td>
    </tr>
  </tbody>
</table>

## Observations

- **The best streaming configuration changes with model size.** Among the MuSTream models, ($\ell = 15$, $p = 0$) is best at 330M, but ($\ell = 30$, $p = 0$) is best at 95M and 26M. Seeing past the supervised block ($p = 15$) lowers the weighted score by 0.062 at 330M and 0.008 at 95M, but raises it by 0.005 at 26M. For beat and downbeat tracking, $p = 15$ helps at every size.
- **Convergence also varies across configurations.** The fastest-settling MuSTream configuration is ($\ell = 15$, $p = 0$) at 330M (1.35 s) but ($\ell = 15$, $p = 15$) at 95M (2.81 s) and 26M (4.64 s). Among models that update every 0.2 s, $p = 15$ increases flips at 330M (10.3 → 13.9 per minute) but reduces them at 95M (17.8 → 15.6) and 26M (24.3 → 18.4).
- **Streaming models find the key quickly but do not always keep it.** Models that update every 0.2 s are first correct after a median of 0.6–0.9 s, versus 1.6–2.0 s for sliding. Across all models, 80–86% of tracks are correct at some point, but only 51–59% end on the correct key.