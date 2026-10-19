/**
 * Beat activations -> beat and downbeat times, one chunk at a time.
 *
 * This is run_beat_onnx.py's `--postproc minimal` -- the peak picker that file
 * verified against beat_tracking/postprocessor.py's own `minimal` to the last
 * bit -- rewritten to run incrementally, because the browser has to emit a beat
 * while the track is still playing rather than after it ends.
 *
 * WHY NOT THE DBN. beat_tracking's accurate postprocessor is madmom's
 * bar-pointer HMM: ~2000 states decoded by Viterbi, and host code that would
 * not survive ONNX tracing even in principle. It knows about tempo continuity
 * and bar structure, which peak picking does not, so it is more accurate --
 * but it cannot be shipped in a graph, and porting it is a different project
 * from this one. What is here is the rule the deployable runner defaults to.
 *
 * The rule, per channel: keep a frame that is the maximum of the 7-frame window
 * centred on it and whose activation is above 0.5 -- i.e. whose LOGIT is above
 * 0, which is why no sigmoid appears in this file -- then merge runs of peaks
 * no more than one frame apart into their mean position. Then move every
 * downbeat onto the nearest beat, because a downbeat IS a beat and the two
 * channels are picked independently.
 *
 * LATENCY, all of it bounded and none of it growing with the track:
 *   a beat      frame t is judged once t+3 exists (the window's right half) and
 *               its merge group closes once a frame too far to join it has been
 *               judged, so ~5 frames = 67 ms at 75 fps.
 *   a downbeat  the above, plus SNAP_LOOKAHEAD frames of beats to snap onto,
 *               so ~11 frames = 147 ms.
 * Both are inside the 200 ms chunk the pipeline already costs.
 */

// beat_tracking/postprocessor.py's `minimal` rule as constants: a local maximum
// over +/- 3 frames (70 ms at 75 fps), kept when the logit is over 0.
export const PEAK_WINDOW = 7;
export const PEAK_HALF = PEAK_WINDOW >> 1;
export const DEDUP_WIDTH = 1;

// How far past a downbeat the beat channel must have been judged before the
// downbeat is snapped onto a beat. The offline snap sees every beat in the
// track; a streaming one has to settle for every beat within this window, which
// at 75 fps is 107 ms -- comfortably more than the couple of frames the two
// channels ever disagree by.
export const SNAP_LOOKAHEAD = 8;

/** Median of an array, which is what the tempo of a beat sequence is taken to
 *  be (a mean would be dragged by every missed or doubled beat). */
export function median(values) {
  if (!values.length) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * One channel's activations in, peak positions (in fractional frames) out, as
 * soon as each is decided.
 *
 * Streaming, but not approximate: a frame is judged only once every frame its
 * window reaches is in hand, and a merge group is closed only once a frame too
 * far to join it has been judged, so the sequence this emits over a track is
 * identical to what pick_peaks returns for the whole activation curve at once.
 */
export class PeakPicker {
  constructor() {
    this.buf = [];        // logits; buf[i] is frame `base + i`
    this.base = 0;
    this.next = 0;        // the next frame to judge
    this.group = null;    // the merge group still open: {pos, count}
  }

  /** The newest frame whose peak/not-peak is settled. */
  get decidedThrough() {
    return this.next - 1;
  }

  push(values) {
    for (let i = 0; i < values.length; i++) this.buf.push(values[i]);
    return this.#judge(false);
  }

  /** End of the channel: judge the last frames, whose windows run off the end
   *  into the -inf padding np.pad gives them, and close the open group. */
  flush() {
    return this.#judge(true);
  }

  /** Frame i's logit, or -Infinity where there is no such frame -- which is
   *  what np.pad(logits, 3, constant_values=-inf) does at both edges, and what
   *  torch's max_pool1d does with padding too. */
  #at(i) {
    const j = i - this.base;
    return j >= 0 && j < this.buf.length ? this.buf[j] : -Infinity;
  }

  #judge(final) {
    const out = [];
    const newest = this.base + this.buf.length - 1;
    // Frame t needs frames up to t+3 before it can be called a local maximum;
    // at the end of the stream there are none, and -inf is the honest stand-in.
    const limit = final ? newest : newest - PEAK_HALF;

    for (; this.next <= limit; this.next++) {
      const t = this.next;
      let max = -Infinity;
      for (let i = t - PEAK_HALF; i <= t + PEAK_HALF; i++) {
        const v = this.#at(i);
        if (v > max) max = v;
      }
      const isPeak = this.#at(t) >= max && this.#at(t) > 0;

      // deduplicate_peaks, as a running group: a group closes as soon as a
      // frame further than DEDUP_WIDTH from its running mean has been judged,
      // peak or not, since every later frame is further still.
      if (this.group && t - this.group.pos > DEDUP_WIDTH) {
        out.push(this.group.pos);
        this.group = null;
      }
      if (isPeak) {
        if (this.group) {
          this.group.count += 1;
          this.group.pos += (t - this.group.pos) / this.group.count;
        } else {
          this.group = { pos: t, count: 1 };
        }
      }
    }

    if (final && this.group) {
      out.push(this.group.pos);
      this.group = null;
    }
    // Nothing before here can be reached by a future window, so it can go: the
    // buffer stays O(window), not O(track).
    const keep = this.next - PEAK_HALF - 1 - this.base;
    if (keep > 4 * PEAK_WINDOW) {
      this.buf = this.buf.slice(keep);
      this.base += keep;
    }
    return out;
  }
}

/**
 * Both channels, plus the downbeat snap and the running tempo: the whole of
 * run_beat_onnx.py's minimal postprocessor, incrementally.
 */
export class BeatTracker {
  constructor(fps, { tempoWindow = 9 } = {}) {
    this.fps = fps;
    this.tempoWindow = tempoWindow;
    this.beatPicker = new PeakPicker();
    this.downbeatPicker = new PeakPicker();
    this.beats = [];              // every beat time so far, seconds
    this.downbeats = [];          // every snapped downbeat time so far
    this.pending = [];            // downbeat frames waiting for beats to snap to
  }

  /** One chunk of both channels' logits. Returns only what this chunk settled. */
  push(beatLogits, downbeatLogits) {
    const beats = this.beatPicker.push(beatLogits).map((f) => f / this.fps);
    this.beats.push(...beats);
    for (const f of this.downbeatPicker.push(downbeatLogits)) this.pending.push(f);
    return { beats, downbeats: this.#snapPending(false) };
  }

  /** End of track. */
  flush() {
    const beats = this.beatPicker.flush().map((f) => f / this.fps);
    this.beats.push(...beats);
    for (const f of this.downbeatPicker.flush()) this.pending.push(f);
    return { beats, downbeats: this.#snapPending(true) };
  }

  /**
   * postp_minimal's last step: move every downbeat onto the nearest detected
   * beat, then drop duplicates.
   *
   * Not cosmetic. The two channels are picked independently, so a downbeat can
   * land a frame or two off the beat it belongs to -- musically wrong, and
   * enough to miss a mir_eval tolerance window. On one track it moved 8 of 31
   * downbeats by exactly one frame.
   */
  #snapPending(final) {
    const out = [];
    while (this.pending.length) {
      const frame = this.pending[0];
      if (!final && this.beatPicker.decidedThrough - frame < SNAP_LOOKAHEAD) break;
      this.pending.shift();
      const t = this.#snap(frame / this.fps);
      // np.unique: two downbeats snapped onto one beat are one downbeat. The
      // snapped times are non-decreasing, so checking the last is enough.
      if (this.downbeats[this.downbeats.length - 1] === t) continue;
      this.downbeats.push(t);
      out.push(t);
    }
    return out;
  }

  #snap(t) {
    if (!this.beats.length) return t;      // no beats to snap to: leave it alone
    let best = this.beats[0];
    for (const b of this.beats) if (Math.abs(b - t) < Math.abs(best - t)) best = b;
    return best;
  }

  /** Tempo from the last few inter-beat intervals, in BPM. Local rather than
   *  whole-track, so it follows a track that changes tempo. */
  tempo(window = this.tempoWindow) {
    const beats = this.beats.slice(-(window + 1));
    if (beats.length < 2) return null;
    const gaps = beats.slice(1).map((t, i) => t - beats[i]);
    const m = median(gaps);
    return m > 0 ? 60 / m : null;
  }

  /**
   * How many beats the model put between consecutive downbeats, as a median.
   *
   * Deliberately not called "beats per bar": without the DBN's bar model
   * nothing here knows a bar's length a priori, nothing can say whether a given
   * beat is the 2nd or the 3rd of its bar, and this probe's downbeat channel
   * often fires every half bar (2 on a 4/4 track). It is a measurement of the
   * two channels' relationship, not a claim about the time signature.
   */
  downbeatPeriod() {
    if (this.downbeats.length < 2) return null;
    const counts = [];
    for (let i = 1; i < this.downbeats.length; i++) {
      const from = this.downbeats[i - 1];
      const to = this.downbeats[i];
      counts.push(this.beats.filter((t) => t >= from && t < to).length);
    }
    const m = Math.round(median(counts));
    return m > 0 ? m : null;
  }

  totals() {
    return { beats: this.beats.length, downbeats: this.downbeats.length };
  }
}
