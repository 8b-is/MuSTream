/**
 * The bar-pointer DBN, block-synchronously, in the browser.
 *
 * This is the accurate beat postprocessor -- Böck et al.'s bar-pointer HMM, the one
 * madmom implements and beat_tracking/postprocessor.py wraps -- ported from
 * `run_beat_onnx.py --postproc dbn_blocksync`. Peak picking (beatPostproc.js) has no
 * tempo model and no bar model; this does, which is why it holds a tempo through a weak
 * bar instead of emitting an extra beat, and why it can say WHICH beat of the bar a beat
 * is rather than only that a downbeat happened.
 *
 * WHY IT CAN BE STREAMED AT ALL. madmom's decoder is offline in three separate ways: it
 * runs Viterbi over the whole activation sequence and backtracks from the final state,
 * it chooses between the 3- and 4-beats-per-bar HMMs by comparing whole-track log
 * probabilities, and it trims the activations to the first/last frame above a threshold.
 * Block-synchronous decoding keeps the models and the forward recursion exactly, and
 * changes only when the backtrack happens: the recursion runs unbroken across blocks, and
 * at each block boundary it backtracks over just that block. The meter is chosen the same
 * way but causally -- at every boundary the HMM with the higher accumulated log
 * probability so far owns the committed output -- and the thresholding is dropped, since
 * it is not causal and only ever trimmed leading and trailing silence.
 *
 * WHY IT IS FREE HERE. The backbone is block-causal: it does not hand over frame t when
 * t's audio arrives, it hands over t's whole 15-frame block at one instant. Decoding
 * those frames one at a time would throw away information already in memory, because the
 * oldest frame of the block could have had the entire block as hindsight at no cost --
 * none of them could have been emitted any earlier anyway. So the oldest frame of a block
 * gets 14 frames of hindsight, the newest gets 0, the mean is 7 (93 ms at 75 fps), and
 * the ADDED latency over per-frame decoding is exactly zero.
 *
 * What remains is the wait every postprocessor here pays, `minimal` included: a beat is
 * placed at the highest-activation frame of a maximal run of in-beat-range states, so it
 * cannot be announced until that run has closed -- typically 2-3 frames at 120 BPM.
 *
 * Everything in this file is arithmetic over typed arrays: no dependency, and nothing
 * that needs a graph. scripts/verify-dbn.mjs checks the state space, the transition
 * model and the observation model against madmom's own arrays element by element, and
 * the decoded beat times against run_beat_onnx.py --postproc dbn_blocksync.
 */

// madmom's DBNDownBeatTrackingProcessor defaults, as beat_tracking/postprocessor.py
// passes them (DBN_BEATS_PER_BAR, DBN_MIN_BPM, DBN_MAX_BPM, DBN_TRANSITION_LAMBDA, and
// the processor's own num_tempi/observation_lambda).
export const DBN_DEFAULTS = {
  beatsPerBar: [3, 4],
  minBpm: 55.0,
  maxBpm: 215.0,
  transitionLambda: 100,
  numTempi: 60,
  observationLambda: 16,
};

/**
 * numpy's rint: round half to EVEN, not half away from zero.
 *
 * Math.round(2.5) is 3 and np.round(2.5) is 2, and the interval grid below is built by
 * rounding a log-spaced ramp -- so a single tie decided the other way would shift a
 * tempo state and every transition into it.
 */
function rint(x) {
  const r = Math.round(x);
  // Math.round rounds .5 up; the only disagreement is an exact .5, where numpy wants
  // the even neighbour.
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/**
 * The tempo grid: madmom's BeatStateSpace intervals, in frames per beat.
 *
 * With num_tempi set (60 by default, and postprocessor.py does not override it) madmom
 * takes the LOG-spaced branch, and it approaches the requested count from below: space
 * `n` intervals logarithmically, round them, drop duplicates, and if that left fewer than
 * asked for, try again with n+1. Reproduced exactly, iteration and all, because the grid
 * decides the state space.
 */
function beatIntervals(minInterval, maxInterval, numTempi) {
  const linear = [];
  for (let i = rint(minInterval); i <= rint(maxInterval); i++) linear.push(i);
  if (!(numTempi != null && numTempi < linear.length)) return linear;

  const lo = Math.log2(minInterval);
  const hi = Math.log2(maxInterval);
  let intervals = [];
  for (let n = numTempi; intervals.length < numTempi; n++) {
    // np.logspace(lo, hi, n, base=2) is 2 ** np.linspace(lo, hi, n), and numpy's
    // linspace is start + k * step with the endpoint written back exactly.
    const step = n > 1 ? (hi - lo) / (n - 1) : 0;
    const seen = new Set();
    for (let k = 0; k < n; k++) seen.add(rint(2 ** (k === n - 1 ? hi : lo + k * step)));
    intervals = [...seen].sort((a, b) => a - b);
  }
  return intervals;
}

/**
 * numpy's pairwise summation, for one contiguous row.
 *
 * A left-to-right sum of the same 60 numbers lands a few ulps away, and those ulps
 * survive into every transition log probability -- enough to make an element-by-element
 * comparison against madmom's arrays fail while proving nothing. numpy sums eight partial
 * accumulators and combines them as a balanced tree (see pairwise_sum in numpy's
 * loops.c.src; the tree only recurses above 128 elements, which a tempo row never
 * reaches), so this reproduces it.
 */
function pairwiseSum(a) {
  const n = a.length;
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res += a[i];
    return res;
  }
  const r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
  let i = 8;
  for (; i < n - (n % 8); i += 8)
    for (let j = 0; j < 8; j++) r[j] += a[i + j];
  let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
  for (; i < n; i++) res += a[i];
  return res;
}

/**
 * madmom's exponential tempo-change distribution, one row per source interval.
 *
 * exp(-lambda * |to/from - 1|), everything at or below np.spacing(1) zeroed BEFORE the
 * rows are normalised, which is what keeps a 21-frame beat from transitioning straight to
 * an 82-frame one and what holds a state's incoming-transition block to 42 sources rather
 * than 60.
 */
function tempoTransitions(intervals, transitionLambda) {
  const n = intervals.length;
  const THRESHOLD = 2 ** -52;             // np.spacing(1)
  const rows = [];
  for (let from = 0; from < n; from++) {
    const row = new Float64Array(n);
    for (let to = 0; to < n; to++) {
      const p = Math.exp(-transitionLambda * Math.abs(intervals[to] / intervals[from] - 1));
      row[to] = p <= THRESHOLD ? 0 : p;
    }
    const sum = pairwiseSum(row);
    for (let to = 0; to < n; to++) row[to] /= sum;
    rows.push(row);
  }
  return rows;
}

/**
 * One of madmom's bar HMMs: the state space, the sparse transition model and the
 * observation model, in the layout the Viterbi recursion below reads.
 *
 * `states` / `pointers` / `logProbs` are madmom's CSR-by-target layout: the incoming
 * transitions of target state s are the entries pointers[s]..pointers[s+1], each naming
 * its source state and log probability. Sources are ascending within a block, which is
 * what makes the recursion's tie-breaking match madmom's (both keep the lowest-indexed
 * source on a tie).
 */
export function buildBarHmm({
  beatsPerBar,
  fps,
  minBpm = DBN_DEFAULTS.minBpm,
  maxBpm = DBN_DEFAULTS.maxBpm,
  transitionLambda = DBN_DEFAULTS.transitionLambda,
  numTempi = DBN_DEFAULTS.numTempi,
  observationLambda = DBN_DEFAULTS.observationLambda,
}) {
  const intervals = beatIntervals((60 * fps) / maxBpm, (60 * fps) / minBpm, numTempi);
  const numIntervals = intervals.length;
  const perBeat = intervals.reduce((n, i) => n + i, 0);
  const numStates = perBeat * beatsPerBar;

  // BeatStateSpace, stacked `beatsPerBar` times with the beat counter added to the
  // position (BarStateSpace). Positions are k * (1/i), the way numpy's
  // linspace(0, 1, i, endpoint=False) computes them.
  const statePositions = new Float64Array(numStates);
  const stateIntervals = new Int32Array(numStates);
  const firstStates = [];                 // firstStates[beat][intervalIndex]
  const lastStates = [];
  let at = 0;
  for (let beat = 0; beat < beatsPerBar; beat++) {
    const first = new Int32Array(numIntervals);
    const last = new Int32Array(numIntervals);
    for (let v = 0; v < numIntervals; v++) {
      const i = intervals[v];
      first[v] = at;
      const stepSize = 1 / i;
      for (let k = 0; k < i; k++) {
        statePositions[at] = k * stepSize + beat;
        stateIntervals[at] = i;
        at++;
      }
      last[v] = at - 1;
    }
    firstStates.push(first);
    lastStates.push(last);
  }

  // BarTransitionModel. Two kinds of transition, and no state receives both: every state
  // that is not the first of a beat is entered from its predecessor at probability 1,
  // and the first states of a beat are entered from the last states of the previous beat
  // under the tempo-change distribution.
  const isFirst = new Uint8Array(numStates);
  for (const first of firstStates) for (const s of first) isFirst[s] = 1;

  const tempo = tempoTransitions(intervals, transitionLambda);
  // Collect per target state, so the CSR comes out sorted by (target, source) -- which
  // is what scipy's csr_matrix hands madmom's make_sparse.
  const incoming = Array.from({ length: numStates }, () => []);
  for (let s = 0; s < numStates; s++)
    if (!isFirst[s]) incoming[s].push([s - 1, 1.0]);
  for (let beat = 0; beat < beatsPerBar; beat++) {
    const to = firstStates[beat];
    // beat 0 is entered from the LAST beat of the bar: madmom indexes
    // last_states[beat - 1], and -1 wraps.
    const from = lastStates[(beat - 1 + beatsPerBar) % beatsPerBar];
    for (let f = 0; f < numIntervals; f++)
      for (let t = 0; t < numIntervals; t++)
        if (tempo[f][t] !== 0) incoming[to[t]].push([from[f], tempo[f][t]]);
  }

  const numTransitions = incoming.reduce((n, list) => n + list.length, 0);
  const states = new Int32Array(numTransitions);
  const pointers = new Int32Array(numStates + 1);
  const logProbs = new Float64Array(numTransitions);
  let k = 0;
  for (let s = 0; s < numStates; s++) {
    pointers[s] = k;
    incoming[s].sort((a, b) => a[0] - b[0]);
    for (const [source, prob] of incoming[s]) {
      states[k] = source;
      logProbs[k] = Math.log(prob);
      k++;
    }
  }
  pointers[numStates] = k;

  // RNNDownBeatTrackingObservationModel: which column of the 3-column log density a
  // state reads. 0 = no beat, 1 = beat, 2 = downbeat, where "in beat range" is the first
  // 1/observation_lambda of a beat and the downbeat range is that of beat 0 only.
  const border = 1 / observationLambda;
  const omPointers = new Uint8Array(numStates);
  for (let s = 0; s < numStates; s++) {
    if (statePositions[s] % 1 < border) omPointers[s] = 1;
    if (statePositions[s] < border) omPointers[s] = 2;
  }

  return {
    beatsPerBar, numStates, numTransitions, intervals,
    statePositions, stateIntervals, states, pointers, logProbs, omPointers,
    observationLambda,
  };
}

/**
 * The Viterbi forward recursion over one bar HMM, stepped one frame at a time, keeping a
 * ring of the last `ringSize` frames of backpointers.
 *
 * madmom keeps the whole [T, numStates] backpointer table and backtracks once from the
 * final state, which is exactly what makes it offline. Keeping only a ring is what lets a
 * caller backtrack a bounded number of steps at every block boundary instead; the
 * recursion itself is unchanged, including the tie-break towards the lowest-indexed
 * source state.
 *
 * logDelta is renormalised to max 0 every frame so a long track cannot underflow, and the
 * accumulated shift lives in logProb -- which is therefore the running best-path log
 * probability, directly comparable between the 3- and 4-beat HMMs. That comparison is the
 * causal form of madmom's single whole-track argmax over HMMs.
 */
class BarViterbi {
  constructor(hmm, ringSize) {
    this.hmm = hmm;
    this.ringSize = ringSize;
    // np.log(hmm.initial_distribution), which madmom leaves uniform.
    this.logDelta = new Float64Array(hmm.numStates).fill(-Math.log(hmm.numStates));
    this.next = new Float64Array(hmm.numStates);
    this.logProb = 0;
    // One flat Int32Array rather than an array of arrays: this is written numStates times
    // per frame, and ringSize * numStates is 15 * 12152 for the 4-beat HMM.
    this.ring = new Int32Array(ringSize * hmm.numStates);
  }

  step(logDensity, ringIndex) {
    const { numStates, states, pointers, logProbs, omPointers } = this.hmm;
    const logDelta = this.logDelta;
    const next = this.next;
    const ring = this.ring;
    const base = ringIndex * numStates;
    let shift = -Infinity;
    for (let s = 0; s < numStates; s++) {
      let bestVal = -Infinity;
      let bestPrev = 0;
      // Strict `>` keeps the FIRST maximum, and sources are ascending within a block, so
      // a tie goes to the lowest-indexed source -- madmom's behaviour.
      for (let k = pointers[s], end = pointers[s + 1]; k < end; k++) {
        const v = logDelta[states[k]] + logProbs[k];
        if (v > bestVal) { bestVal = v; bestPrev = states[k]; }
      }
      ring[base + s] = bestPrev;
      const value = bestVal + logDensity[omPointers[s]];
      next[s] = value;
      if (value > shift) shift = value;
    }
    for (let s = 0; s < numStates; s++) logDelta[s] = next[s] - shift;
    this.logProb += shift;
  }

  /** The best state at the current frame. */
  argmax() {
    const logDelta = this.logDelta;
    let best = 0;
    for (let s = 1; s < logDelta.length; s++) if (logDelta[s] > logDelta[best]) best = s;
    return best;
  }

  /** Where the path that is in `state` at frame `frame` was at frame - 1. */
  previous(state, frame) {
    const slot = ((frame % this.ringSize) + this.ringSize) % this.ringSize;
    return this.ring[slot * this.hmm.numStates + state];
  }
}

/**
 * Block-synchronous DBN decoding of a live activation stream.
 *
 * Feed it the probe's activations one block at a time; it returns the beats that block
 * settled, each with its position in the bar. The forward recursion runs unbroken across
 * blocks, so the tempo and phase history is the full causal one, and only the backtrack
 * is bounded.
 */
export class BlockSyncDbn {
  /**
   * @param fps               label rate of the activations (75 for the beat probe)
   * @param block             frames the backbone hands over per forward call (15)
   * @param beatsPerBar       null (default) weighs DBN_DEFAULTS.beatsPerBar causally, as
   *                          run_beat_onnx.py does; a number pins the bar length, which
   *                          halves the work and removes the one genuinely unstable part
   *                          of causal decoding -- the meter can otherwise flip for tens
   *                          of seconds. An ARRAY sets the candidates to weigh, so a
   *                          deployment that expects 5/4 or 2/4 can say so: madmom builds
   *                          a bar HMM for any length, and nothing here assumes two.
   *                          Every extra candidate is another full HMM stepped per frame.
   * @param transitionLambda  the DBN's tempo-change penalty.
   */
  constructor({ fps, block, beatsPerBar = null,
                transitionLambda = DBN_DEFAULTS.transitionLambda,
                minBpm = DBN_DEFAULTS.minBpm, maxBpm = DBN_DEFAULTS.maxBpm } = {}) {
    this.fps = fps;
    this.block = Math.max(1, block | 0);
    this.bars = beatsPerBar == null ? DBN_DEFAULTS.beatsPerBar
              : Array.isArray(beatsPerBar) ? [...beatsPerBar] : [beatsPerBar];
    this.hmms = this.bars.map((n) =>
      buildBarHmm({ beatsPerBar: n, fps, minBpm, maxBpm, transitionLambda }));
    this.logDensity = new Float64Array(3);
    // The block's activations, kept so a run's argmax can be taken over the frames the
    // backtrack commits. Ring of exactly one block, so this is O(block), not O(track).
    this.actBeat = new Float64Array(this.block);
    this.actDown = new Float64Array(this.block);
    this.reset();
  }

  reset() {
    this.decoders = this.hmms.map((h) => new BarViterbi(h, this.block));
    this.frame = 0;            // frames consumed so far
    this.blockStart = 0;       // first frame of the block being accumulated
    // The open run of in-beat-range states, carried across blocks: a beat sits at the
    // highest-activation frame of its run and cannot be announced until the run closes.
    this.run = null;
    this.started = false;
    this.prevInBeat = false;
    this.prevOwner = -1;
    this.beats = [];
    this.downbeats = [];
    this.meter = null;         // beats per bar of the HMM currently winning
  }

  /**
   * madmom's 2-column activation, from the probe's logits.
   *
   * The sigmoid and the epsilon clipping are Postprocessor.postp_dbn's (0 and 1 break the
   * DBN), and the two columns are combined_activation's: column 0 is
   * beat-but-not-downbeat, column 1 is downbeat.
   */
  static activation(beatLogit, downbeatLogit) {
    const EPS = 1e-5;
    const beat = (1 / (1 + Math.exp(-beatLogit))) * (1 - EPS) + EPS / 2;
    const down = (1 / (1 + Math.exp(-downbeatLogit))) * (1 - EPS) + EPS / 2;
    return [Math.max(beat - down, EPS / 2), down];
  }

  /**
   * One block of logits in, the beats it settled out.
   *
   * A short block is legitimate and is closed as it stands, with the hindsight it has --
   * which is what the last chunk of a track gets once the zero-padded tail is trimmed.
   */
  push(beatLogits, downbeatLogits) {
    const emitted = [];
    for (let i = 0; i < beatLogits.length; i++) {
      const [a0, a1] = BlockSyncDbn.activation(beatLogits[i], downbeatLogits[i]);
      const slot = this.frame % this.block;
      this.actBeat[slot] = a0;
      this.actDown[slot] = a1;
      // RNNDownBeatTrackingObservationModel.log_densities: no-beat, beat, downbeat.
      this.logDensity[0] = Math.log((1 - (a0 + a1)) / (DBN_DEFAULTS.observationLambda - 1));
      this.logDensity[1] = Math.log(a0);
      this.logDensity[2] = Math.log(a1);
      for (const decoder of this.decoders) decoder.step(this.logDensity, slot);
      this.frame++;
      if (this.frame - this.blockStart === this.block) this.#closeBlock(emitted);
    }
    return this.#collect(emitted);
  }

  /** End of track: close the block still accumulating, however short. */
  flush() {
    const emitted = [];
    if (this.frame > this.blockStart) this.#closeBlock(emitted);
    // A run still open when the stream ends is a beat too: _beats_from_path takes the
    // last run's end as the final frame rather than dropping it.
    if (this.run) { this.#emit(emitted, this.run); this.run = null; }
    return this.#collect(emitted);
  }

  /**
   * Commit the frames of the block that just closed.
   *
   * The winner is the HMM with the higher accumulated log probability SO FAR -- the causal
   * form of madmom's whole-track argmax -- and the backtrack walks from its best state at
   * the newest frame back over just this block.
   */
  #closeBlock(emitted) {
    const last = this.frame - 1;
    const span = last - this.blockStart + 1;
    let best = 0;
    for (let d = 1; d < this.decoders.length; d++)
      if (this.decoders[d].logProb > this.decoders[best].logProb) best = d;
    const decoder = this.decoders[best];
    const hmm = this.hmms[best];
    this.meter = hmm.beatsPerBar;

    // Walk back once, filling newest-first, then commit oldest-first so the runs the
    // beats come from are chronological.
    const path = new Int32Array(span);
    let state = decoder.argmax();
    path[span - 1] = state;
    for (let j = 1; j < span; j++) {
      state = decoder.previous(state, last - j + 1);
      path[span - 1 - j] = state;
    }
    this.blockStart = this.frame;
    for (let i = 0; i < span; i++)
      this.#commit(emitted, last - span + 1 + i, path[i], best, hmm);
  }

  /** One committed frame: extend or close the open run of in-beat-range states. */
  #commit(emitted, frame, state, owner, hmm) {
    const inBeat = hmm.omPointers[state] >= 1;
    const position = hmm.statePositions[state];
    // _beats_from_path's run boundaries: where in-beat-range changes, and where the
    // winning HMM changes (the two index different state spaces).
    const boundary = !this.started || inBeat !== this.prevInBeat || owner !== this.prevOwner;
    if (boundary && this.run) { this.#emit(emitted, this.run); this.run = null; }
    this.started = true;
    this.prevInBeat = inBeat;
    this.prevOwner = owner;
    if (!inBeat) return;

    const slot = frame % this.block;
    const a0 = this.actBeat[slot];
    const a1 = this.actDown[slot];
    if (!this.run) this.run = { frame, best: -Infinity, position };
    // madmom takes the argmax over the run's flattened 2-column slice, i.e. over BOTH
    // activations, the first maximum winning. Tracked incrementally, so no activation
    // history beyond the block has to be kept.
    if (a0 > this.run.best) { this.run.best = a0; this.run.frame = frame; this.run.position = position; }
    if (a1 > this.run.best) { this.run.best = a1; this.run.frame = frame; this.run.position = position; }
  }

  #emit(emitted, run) {
    emitted.push({
      time: run.frame / this.fps,
      frame: run.frame,
      // The DBN knows where in the bar a beat sits, which peak picking cannot: 1 is the
      // downbeat.
      beatNumber: Math.floor(run.position) + 1,
    });
  }

  #collect(emitted) {
    const beats = emitted.map((b) => b.time);
    const downbeats = emitted.filter((b) => b.beatNumber === 1).map((b) => b.time);
    this.beats.push(...beats);
    this.downbeats.push(...downbeats);
    return { beats, downbeats, events: emitted };
  }

  /** Tempo from the last few inter-beat intervals, in BPM. */
  tempo(window = 9) {
    const beats = this.beats.slice(-(window + 1));
    if (beats.length < 2) return null;
    const gaps = beats.slice(1).map((t, i) => t - beats[i]).sort((a, b) => a - b);
    const mid = gaps.length >> 1;
    const m = gaps.length % 2 ? gaps[mid] : (gaps[mid - 1] + gaps[mid]) / 2;
    return m > 0 ? 60 / m : null;
  }

  totals() {
    return { beats: this.beats.length, downbeats: this.downbeats.length };
  }
}
