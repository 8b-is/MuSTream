/**
 * Chord emissions -> chord labels, one chunk at a time, through the self-bias
 * CRF -- with lambda chosen HERE rather than baked into the graph.
 *
 * This is models.decode_block_synchronous driving models.StreamingViterbi with
 * models.build_self_bias_crf(V, lambda), ported the way beatDbn.js ports the
 * bar-pointer DBN: the graph emits activations, the host decides what to make
 * of them.
 *
 * WHY THE HOST DECODES AT ALL, when the probe graph carries its own CRF and
 * returns `tags` ready to display.
 *
 * lambda is a Viterbi parameter, and export_chord_probe_onnx.py freezes it into
 * the graph (`--self-bias`, default 1). One nat was tuned for ACCURACY, and it
 * is the accuracy optimum -- chord_recognition/sweep_self_bias.py measures the
 * curve turning over right there. But it is not the stability optimum, and the
 * transcript it produces changes its mind about 48 times a minute against
 * ground truth's 30, with a third of its segments shorter than half a second.
 * That is what a listener sees as the display flickering.
 *
 * Raising lambda buys stability from accuracy at a rate that is steep at first
 * and then flattens (Beatles chord test split, 18 tracks, both students):
 *
 *     lambda    flips/min    segments < 0.5s    blips    MIREX
 *        1         47.8            36%           270     70.4%   <- exported
 *        3         37.8            24%           131     69.3%   <- DEFAULT here
 *        8         30.2            15%            72     65.4%
 *      (GT)        30.1             5%             2         -
 *
 * lambda = 3 halves the blips -- sub-0.5 s excursions that return to the chord
 * they left, the flicker a listener reads as the display being broken -- for
 * about one point of MIREX. Past ~6 the exchange rate collapses, and no lambda
 * reaches ground truth's SHAPE, because with only `block` frames of hindsight a
 * large switch penalty defers a decision rather than resolving it. That needs
 * lookahead, not a bigger prior; see chord_recognition/models.py's note that the
 * optimal lambda scales with the hindsight available.
 *
 * Decoding here rather than re-exporting also means lambda is a runtime choice:
 * MusicEngine.configure({chord: {selfBias}}) changes it without refetching a
 * ~5 MB graph, and `selfBias: null` hands the decision back to the graph's own
 * baked-in CRF, which is what scripts/verify.mjs uses to check this file against
 * run_chord_onnx.py.
 *
 * COST. transitions = lambda * I, so the [V, V] max over the previous tag
 * collapses: every off-diagonal entry is 0, leaving
 * max(best-score-anywhere, score[v] + lambda). A frame is therefore O(V), and a
 * chunk is 3 x 121 -- a rounding error beside the graphs that produced the
 * emissions.
 *
 * PRE-SOFTMAX INPUT IS FINE, and this file relies on it: the graph's `logits`
 * output is raw, while its internal CRF scores log-softmaxed emissions
 * (`emission_mode: "logprobs"`). log_softmax subtracts logsumexp, a per-frame
 * CONSTANT, and adding a constant to every class of one frame shifts every path
 * score through that frame equally -- it moves neither the argmax nor any
 * backpointer. So the raw logits decode identically, and lambda keeps its
 * meaning in nats. That equivalence is not assumed: sweep_self_bias.py feeds the
 * graph's raw logits to the same recursion and gets the graph's own tags back,
 * frame for frame, on every track it is run over.
 */

/**
 * The demo's lambda, in nats. See the table above for what it costs and buys;
 * chord_recognition/sweep_self_bias.py is what produced it and will reproduce it.
 */
export const DEFAULT_CHORD_SELF_BIAS = 3;

/** The graph's own baked-in decode, i.e. "use the `tags` output, not this file". */
export const GRAPH_SELF_BIAS = null;

export class BlockSyncChordCrf {
  /**
   * @param vocabSize  chord classes (121 for the declared vocabulary)
   * @param block      label frames that arrive at one instant -- the probe's
   *                   `emission_block`. The oldest frame of a block gets
   *                   block-1 frames of hindsight and the newest gets none,
   *                   which is free on a block-causal backbone because neither
   *                   could have been emitted any earlier.
   * @param selfBias   lambda, in nats. 0 reduces this to per-frame argmax.
   */
  constructor({ vocabSize, block, selfBias = DEFAULT_CHORD_SELF_BIAS }) {
    if (!(block >= 1)) throw new Error(`block must be >= 1, got ${block}`);
    if (!(selfBias >= 0)) throw new Error(`a negative self-bias penalises STAYING; ` +
                                          `lambda >= 0, got ${selfBias}`);
    this.vocabSize = vocabSize;
    this.block = block;
    this.selfBias = selfBias;
    // float64 throughout: the reference recursion is numpy's, and a near-tied
    // chord boundary is decided by differences small enough that float32
    // storage alone moves about one frame in 40k.
    this.score = new Float64Array(vocabSize);
    this.next = new Float64Array(vocabSize);
    this.bps = Array.from({ length: block }, () => new Int32Array(vocabSize));
    this.reset();
  }

  /** Drop the carried recursion -- a new track. */
  reset() {
    this.started = false;
    this.score.fill(0);
  }

  /**
   * One block of emissions in, its tags out.
   *
   * @param logits row-major [n, vocabSize], as the graph returns them
   * @param n      frames in this block; must be <= block, since a backtrack
   *               can only reach as far as the ring of backpointers kept
   * @returns Int32Array[n] tags for those frames, final on return
   */
  push(logits, n = this.block) {
    if (n > this.block)
      throw new Error(`${n} frames is more than one block (${this.block}); the backtrack ` +
                      `would need backpointers that were never kept`);
    const V = this.vocabSize;
    const lambda = this.selfBias;
    let kept = 0;                       // backpointer rows written this block

    for (let i = 0; i < n; i++) {
      const off = i * V;
      if (!this.started) {
        // start_transitions = 0, so frame 0 is its own emission -- and it gets
        // no backpointer, which is why the first block backtracks n-1 steps
        // like every other one.
        for (let v = 0; v < V; v++) this.score[v] = logits[off + v];
        this.started = true;
      } else {
        // The best previous tag, ties to the lower index -- torch.max's rule,
        // which the reference implementation inherits.
        let gmax = -Infinity, gidx = 0;
        for (let v = 0; v < V; v++) {
          if (this.score[v] > gmax) { gmax = this.score[v]; gidx = v; }
        }
        const bp = this.bps[kept++];
        for (let v = 0; v < V; v++) {
          const cand = this.score[v] + lambda;          // stay on v
          const stay = cand > gmax || (cand === gmax && v < gidx);
          bp[v] = stay ? v : gidx;
          this.next[v] = (cand > gmax ? cand : gmax) + logits[off + v];
        }
        const swap = this.score; this.score = this.next; this.next = swap;
      }
      // Subtracting a per-frame constant leaves the argmax and every
      // backpointer untouched, and stops an open-ended stream's scores from
      // drifting out of range. StreamingViterbi.step does the same.
      let max = -Infinity;
      for (let v = 0; v < V; v++) if (this.score[v] > max) max = this.score[v];
      for (let v = 0; v < V; v++) this.score[v] -= max;
    }

    // Newest frame first: it has no hindsight, exactly as lag=0 would give it.
    // Then walk this block's transitions back. end_transitions are NOT applied
    // -- the stream has not ended.
    const tags = new Int32Array(n);
    let tag = 0, best = -Infinity;
    for (let v = 0; v < V; v++) if (this.score[v] > best) { best = this.score[v]; tag = v; }
    tags[n - 1] = tag;
    for (let i = n - 2, row = kept - 1; i >= 0; i--, row--) {
      tag = this.bps[row][tag];
      tags[i] = tag;
    }
    return tags;
  }
}
