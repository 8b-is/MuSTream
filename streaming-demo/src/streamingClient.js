/**
 * Main-thread side of the real pipelines: decode audio the way the models
 * expect it, talk to the inference worker, and turn what comes back into
 * something a listener can read.
 */

/** Promise-based client for infer.worker.js. */
export class InferenceStream {
  /** `model` picks the student when the graphs are remote (see modelSource.js). */
  constructor(model = null) {
    this.model = model;
    this.worker = new Worker(new URL("./infer.worker.js", import.meta.url), { type: "module" });
    this.pending = new Map();
    this.seq = 0;
    this.onProgress = null;
    this.info = null;
    this.requested = new Set();
    this.loading = null;
    this.worker.onmessage = ({ data }) => {
      if (data.type === "progress") {
        this.onProgress?.(data);
        return;
      }
      const entry = this.pending.get(data.seq);
      if (!entry) return;
      this.pending.delete(data.seq);
      if (data.type === "error") entry.reject(new Error(data.message));
      else entry.resolve(data);
    };
    this.worker.onerror = (e) => {
      const err = new Error(e.message || "inference worker failed");
      for (const { reject } of this.pending.values()) reject(err);
      this.pending.clear();
    };
  }

  #send(msg, transfer = []) {
    const seq = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(seq, { resolve, reject });
      this.worker.postMessage({ ...msg, seq }, transfer);
    });
  }

  /**
   * Bring up the graphs for these tasks, and keep whatever is already up.
   *
   * Idempotent and additive: repeated or concurrent callers share one load, and
   * asking for a task on top of the ones already loaded fetches only what it
   * adds. Which is the whole point -- Chord then All tasks must not re-download
   * a backbone that is already compiled in the worker.
   */
  load(tasks, onProgress, config) {
    this.onProgress = onProgress;
    const missing = tasks.filter((t) => !this.requested.has(t));
    if (!missing.length && this.loading) return this.loading;
    for (const t of missing) this.requested.add(t);
    // A previous failure must not poison a retry, and a failed attempt must
    // give its tasks back so the next call actually re-asks for them.
    const wanted = [...this.requested];
    this.loading = (this.loading ?? Promise.resolve())
      .catch(() => {})
      .then(() => this.#send({ type: "load", tasks: wanted, config, model: this.model }))
      .then((res) => (this.info = res.info))
      .catch((err) => {
        for (const t of missing) this.requested.delete(t);
        this.loading = null;
        throw err;
      });
    return this.loading;
  }

  /** Which tasks the worker actually has a head for, once loaded. */
  get tasks() {
    return this.info?.tasks ?? [];
  }

  /**
   * Change a task's decoder settings -- {beat: {postproc, beatsPerBar}} -- without
   * reloading any graph. Resets the affected head, so call it between tracks.
   */
  configure(config) {
    this.loading = (this.loading ?? Promise.resolve())
      .then(() => this.#send({ type: "configure", config }))
      .then((res) => (this.info = res.info));
    return this.loading;
  }

  /** Start a fresh track: every graph's state is per-track by construction. */
  reset() {
    return this.#send({ type: "reset" });
  }

  /**
   * One chunk in, every loaded task's output for it out. `audio` is transferred,
   * not copied; `real` is how many of its samples came from the file rather
   * than from zero-padding, so each head can end its output where the audio does.
   */
  pushChunk(audio, real = audio.length) {
    return this.#send({ type: "chunk", audio, real }, [audio.buffer]);
  }

  /** End of track: whatever a postprocessor was still holding (beat's peak
   *  picker keeps the last few frames until their windows are complete). */
  finish() {
    return this.#send({ type: "finish" });
  }

  close() {
    this.worker.terminate();
  }
}

/**
 * Decode an uploaded file to mono float32 at the models' sample rate.
 *
 * DECODE IN A CONTEXT THAT ALREADY RUNS AT THE MODEL'S RATE. decodeAudioData
 * resamples to the rate of the context it is called on, so decoding in a plain
 * `new AudioContext()` resamples the file to the output device's 44.1 or 48 kHz
 * first, and getting to 24 kHz then costs a SECOND resample. That round trip is
 * not cosmetic: measured against ffmpeg's single soxr pass on a 24 kHz file it
 * moved the waveform in the third decimal (RMS 0.14097 vs 0.14118), which is
 * enough to flip a borderline key estimate. An OfflineAudioContext at
 * `sampleRate` does it in one pass, which is what run_chord_onnx.py does.
 */
export async function decodeToModelRate(file, sampleRate) {
  const bytes = await file.arrayBuffer();
  let decoded;
  try {
    decoded = await new OfflineAudioContext(1, 1, sampleRate).decodeAudioData(bytes);
  } catch (err) {
    // Not every engine implements decodeAudioData on an offline context. Two
    // resamples beat refusing the file.
    const ctx = new AudioContext();
    try {
      decoded = await ctx.decodeAudioData(bytes);
    } finally {
      ctx.close();
    }
  }
  if (decoded.sampleRate === sampleRate && decoded.numberOfChannels === 1)
    return { samples: decoded.getChannelData(0), duration: decoded.duration };

  // Down-mix (and resample, if the fallback above was taken). Connecting a
  // multi-channel buffer to a 1-channel destination averages the channels,
  // which is what stream_chunks' block.mean(axis=1) does.
  const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * sampleRate), sampleRate);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return { samples: rendered.getChannelData(0), duration: decoded.duration };
}

const QUALITIES = {
  maj: "", min: "m", 7: "7", min7: "m7", maj7: "maj7",
  sus4: "sus4", dim: "dim", dim7: "dim7", hdim7: "m7♭5", aug: "aug",
};

/** "A#:min7" -> "A♯m7", "N" -> "N.C." (the vocabulary's no-chord symbol). */
export function displayChord(label) {
  if (!label || label === "N") return "N.C.";
  const [root, quality = "maj"] = label.split(":");
  const pretty = root.replace("#", "♯");
  return pretty + (QUALITIES[quality] ?? `:${quality}`);
}

/** "C# minor" -> {root:"C♯", mode:"minor"}, so the UI can size the two apart. */
export function displayKey(label) {
  if (!label) return { root: "—", mode: "" };
  const [root, mode = ""] = label.split(" ");
  return { root: root.replace("#", "♯"), mode };
}

/**
 * Collapse the 15 label frames per second into the segments a listener hears,
 * in place, as they arrive -- the streaming form of infer.py's _merge_frames.
 * Returns a new array only when it changes, so React re-renders once per label.
 */
export function appendLabel(segments, { label, confidence, time }, frameDuration) {
  const last = segments[segments.length - 1];
  if (last && last.label === label) {
    const frames = last.frames + 1;
    // Mean over the segment's frames, not the best of them: a segment's
    // confidence should not be the one frame that happened to be certain.
    const merged = { ...last, end: time + frameDuration, frames,
                     confidence: last.confidence + (confidence - last.confidence) / frames };
    return [...segments.slice(0, -1), merged];
  }
  return [...segments, { label, confidence, start: time, end: time + frameDuration, frames: 1 }];
}

/**
 * What a key estimate can say about itself without knowing the answer: the
 * ground-truth-free half of key_detection/key_utils.py's convergence_metrics,
 * and the same summary run_key_onnx.py prints.
 *
 * `history` is the running estimate, one entry per 0.2 s block, oldest first.
 * t_settle is when the model last ARRIVED at its current answer and never left
 * again -- the honest "settled" time, not merely the first time it said it.
 */
export function keyConvergence(history) {
  if (!history.length) return null;
  const final = history[history.length - 1].index;
  let settled = history.length - 1;
  while (settled > 0 && history[settled - 1].index === final) settled -= 1;
  const first = history.findIndex((h) => h.index === final);
  let flips = 0;
  for (let i = 1; i < history.length; i++) if (history[i].index !== history[i - 1].index) flips += 1;
  const span = history[history.length - 1].time - history[0].time;
  return {
    label: history[history.length - 1].label,
    confidence: history[history.length - 1].confidence,
    tFirst: history[first].time,
    tSettle: history[settled].time,
    flips,
    flipsPerMin: span > 0 ? (flips / span) * 60 : 0,
    duration: history[history.length - 1].time,
  };
}
