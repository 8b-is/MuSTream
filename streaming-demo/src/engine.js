/**
 * Real streaming music understanding in the browser: the protocol of
 * run_chord_onnx.py / run_beat_onnx.py / run_key_onnx.py (and their shared
 * onnx_stream.py), ported to onnxruntime-web.
 *
 *   0.2 s of audio (4800 samples @ 24 kHz)
 *     -> backbone graph  -> [1, 13, 15, D] features + updated KV/conv state
 *     -> chord probe     -> 3 chord labels   + updated LSTM/CRF state
 *     -> beat probe      -> 15 (beat, downbeat) logits + updated LSTM state
 *     -> key probe       -> 3 running key estimates    + updated LSTM state
 *
 * Every graph carries fixed-size state, so cost per chunk is O(1) in track
 * length and latency is one chunk (200 ms) plus compute. Nothing here is
 * specific to one distilled student: all geometry is read from the specs.
 *
 * TWO BACKBONES, NOT ONE. The chord and key probes were distilled against the
 * `r0` streaming backbone; the beat probe against `r15`. They are different
 * weights, so a probe run on the other one produces tensors that line up
 * perfectly and numbers that are quietly wrong -- which is why checkPair
 * compares the provenance both graphs record, and why a task's backbone is
 * named per task rather than assumed. Tasks that name the SAME backbone share
 * one session and one backbone run per chunk: in "all tasks" mode that is two
 * backbone runs and three probe runs, not three of each.
 *
 * Kept free of any browser-only import so the same code can be driven from
 * Node for verification (see scripts/verify.mjs).
 */
import * as ort from "onnxruntime-web";
import { BeatTracker } from "./beatPostproc.js";
import { BlockSyncDbn } from "./beatDbn.js";
import { BlockSyncChordCrf, DEFAULT_CHORD_SELF_BIAS } from "./chordCrf.js";

const ARRAYS = {
  float32: Float32Array,
  float16: Uint16Array,
  int64: BigInt64Array,
  int32: Int32Array,
  bool: Uint8Array,
};

/** The probe graph each task must be handed, so pointing a head at another
 *  task's probe fails here rather than where its output is interpreted. */
const PROBE_GRAPH = { chord: "chord_probe", beat: "beat_probe", key: "key_probe" };

function zeroTensor({ shape, dtype }) {
  const Ctor = ARRAYS[dtype];
  if (!Ctor) throw new Error(`unsupported state dtype: ${dtype}`);
  return new ort.Tensor(dtype, new Ctor(shape.reduce((a, b) => a * b, 1)), shape);
}

/** argmax of one row of a flat [rows, width] buffer. */
function argmax(data, offset, width) {
  let best = 0;
  for (let c = 1; c < width; c++) if (data[offset + c] > data[offset + best]) best = c;
  return best;
}

/**
 * One row's posterior for one class.
 *
 * The graphs' `logits` outputs are pre-softmax (the chord probe's
 * `emission_mode: "logprobs"` describes what it hands its own CRF, not what it
 * returns), so a stable softmax is required; it is a no-op on a row that is
 * already normalised.
 */
function posterior(data, offset, width, index) {
  let max = -Infinity;
  for (let c = 0; c < width; c++) if (data[offset + c] > max) max = data[offset + c];
  let sum = 0;
  for (let c = 0; c < width; c++) sum += Math.exp(data[offset + c] - max);
  return Math.exp(data[offset + index] - max) / sum;
}

/**
 * One exported graph plus its carried state.
 *
 * Every graph these pipelines use -- backbone, chord probe, beat probe, key
 * probe -- uses the same protocol: the spec lists `state` entries as
 * {name, output, shape, dtype}, each zero-initialised once per session and then
 * fed its own previous output. So one class drives all of them. That symmetry
 * is onnx_stream.py's, deliberately kept: one loop to port, not four.
 */
export class OnnxStreamer {
  constructor(session, spec) {
    this.session = session;
    this.spec = spec;
    this.stateEntries = spec.state;
    const stateOutputs = new Set(this.stateEntries.map((e) => e.output));
    // Everything the graph returns that is not fed back in is a result.
    this.valueNames = session.outputNames.filter((n) => !stateOutputs.has(n));
    this.outputNames = [...this.valueNames, ...this.stateEntries.map((e) => e.output)];
    this.reset();
  }

  static async create(model, spec, sessionOptions) {
    const session = await ort.InferenceSession.create(model, sessionOptions);
    try {
      return new OnnxStreamer(session, spec);
    } catch (err) {
      // The runtime session exists even if invalid metadata prevents wrapping it.
      try { await session.release(); } catch {}
      throw err;
    }
  }

  /** Start a fresh track. Carrying any graph's state across tracks would let
   *  one track's history bleed into the next. */
  reset() {
    this.state = {};
    for (const entry of this.stateEntries) this.state[entry.name] = zeroTensor(entry);
  }

  async run(inputs) {
    const outs = await this.session.run({ ...inputs, ...this.state }, this.outputNames);
    for (const entry of this.stateEntries) this.state[entry.name] = outs[entry.output];
    return outs;
  }

  stateBytes() {
    return Object.values(this.state).reduce((n, t) => n + t.data.byteLength, 0);
  }
}

/**
 * Refuse a backbone and probe that do not belong together, before any audio is
 * read -- onnx_stream.py's check_pair. Every one of these mismatches would
 * otherwise run happily and produce output that means nothing: a probe reads
 * whatever tensor it is handed.
 *
 * Returns the provenance warning, if any, rather than throwing on it: a
 * re-export or a moved path is legitimate, so the caller decides. Swapping
 * backbones between the r0 and r15 families is not, and this is the only check
 * that can see it.
 */
export function checkPair(task, backbone, probe) {
  const b = backbone.spec;
  const p = probe.spec;
  const graph = PROBE_GRAPH[task];
  if (!graph) throw new Error(`unknown task: ${task}`);
  if (p.graph !== graph || p.mode !== "streaming")
    throw new Error(`the probe graph is ${p.graph}/${p.mode}, expected a streaming ${graph}`);
  if (b.mode !== "streaming")
    throw new Error(
      `this demo needs a streaming backbone graph (export_mert_onnx.py); got a ${b.mode} graph`
    );
  if (b.sample_rate !== p.sample_rate)
    throw new Error(`sample rate mismatch: backbone ${b.sample_rate}, probe ${p.sample_rate}`);

  const want = p.features_input.shape;
  const got = b.feature_shape;
  if (got[0] !== want[0] || got[1] !== want[1] || got[3] !== want[3])
    throw new Error(
      `shape mismatch: the backbone emits [${got}], the probe expects [${want}]. These were ` +
        "built for different backbones (check the layer count L and the hidden size)."
    );
  if (got[2] !== want[2])
    throw new Error(`chunk mismatch: backbone ${got[2]} frames, probe wants ${want[2]}`);

  // The probe was trained on features from ONE distillation checkpoint.
  const trainedOn = p.mert_align_checkpoint;
  const builtFrom = b.source_checkpoint;
  if (trainedOn && builtFrom && trainedOn !== builtFrom)
    return `the ${task} probe was trained against ${trainedOn} but its backbone graph was ` +
           `exported from ${builtFrom}; the features will be subtly wrong unless these are ` +
           "the same weights";
  return null;
}

/**
 * A probe graph plus whatever turns its output into something to show.
 *
 * The three heads share the driving (one probe run per chunk, state threaded by
 * OnnxStreamer) and differ only in `decode`, which is exactly the split the
 * three run_*_onnx.py runners make around onnx_stream.py.
 */
class Head {
  constructor(task, probe, backboneAlias) {
    this.task = task;
    this.probe = probe;
    this.backboneAlias = backboneAlias;
    const p = probe.spec;
    this.featuresInput = p.features_input.name;
    this.labelFrames = p.label_frames;
    this.samplesPerFrame = p.pool_factor * p.hop_size;
    this.frameDuration = this.samplesPerFrame / p.sample_rate;
    this.labelRate = p.label_rate;
    this.paramsM = p.n_params / 1e6;
    this.frames = 0;
    // NOT this.reset() -- a subclass cannot set the fields its own reset() needs until
    // super() has returned, so each concrete head resets itself last.
  }

  reset() {
    this.probe.reset();
    this.frames = 0;          // label frames emitted so far, padding included
  }

  /** How many label frames of a track are real once the zero-padded tail is
   *  dropped -- without this a track always gains up to one chunk of output
   *  invented from silence. This is run_*_onnx.py's n_real // samples_per_frame. */
  validFrames(realSamples) {
    return Math.floor(realSamples / this.samplesPerFrame);
  }

  async run(features, realSamples) {
    const outs = await this.probe.run({ [this.featuresInput]: features });
    const result = this.decode(outs, this.validFrames(realSamples));
    this.frames += this.labelFrames;
    return result;
  }

  /** End of track: whatever the postprocessor was still holding. */
  finish() {
    return null;
  }

  describe() {
    const p = this.probe.spec;
    return {
      task: this.task,
      params: this.paramsM,
      labelsPerChunk: this.labelFrames,
      labelRate: this.labelRate,
      frameDuration: this.frameDuration,
      decoder: p.decoder,
      backbone: this.backboneAlias,
    };
  }
}

/**
 * Chord: 3 frame labels per chunk, final on arrival (blocksync CRF).
 *
 * The CRF runs HERE, not in the graph, so its self-bias lambda is a runtime
 * choice -- see chordCrf.js for what lambda does and why the default is not the
 * value the graph was exported with. `selfBias: null` gives the graph's own
 * `tags` output back, which is the mode that must agree with run_chord_onnx.py.
 *
 * Either way the probe graph's internal CRF still runs and still carries its
 * `crf_score` state; when this head decodes, that output is simply not read.
 */
class ChordHead extends Head {
  constructor(task, probe, backboneAlias, config) {
    super(task, probe, backboneAlias);
    this.labels = probe.spec.chord_labels || null;
    this.hasTags = Boolean(probe.spec.tags_output);
    this.vocabSize = probe.spec.vocab_size;
    // The frames the backbone hands over per forward call -- the hindsight a
    // block-synchronous backtrack can spend for free.
    this.block = probe.spec.emission_block || this.labelFrames;
    this.graphSelfBias = probe.spec.self_bias ?? null;
    this.selfBias = DEFAULT_CHORD_SELF_BIAS;
    this.configure(config);
  }

  /**
   * Choose lambda, in nats.
   *
   * `null` defers to the CRF compiled into the graph (whatever
   * export_chord_probe_onnx.py --self-bias was given). A number decodes here
   * instead. Nothing reaches the graphs, so switching is free -- but it does
   * reset the decoder's recursion, so it belongs between tracks.
   */
  configure({ selfBias } = {}) {
    if (selfBias !== undefined) {
      const lambda = selfBias == null ? null : Number(selfBias);
      if (lambda != null && !(lambda >= 0))
        throw new Error(`chord selfBias must be null or a number >= 0, got ${selfBias}`);
      if (lambda !== this.selfBias) this.crf = null;
      this.selfBias = lambda;
    }
    if (this.selfBias != null && !this.probe.spec.logits_output)
      throw new Error("this chord probe graph returns no logits, so lambda cannot be " +
                      "chosen here; configure {chord: {selfBias: null}} to use its own CRF");
    if (this.selfBias == null && !this.hasTags)
      throw new Error("this chord probe graph carries no CRF, so there are no tags to " +
                      "defer to; configure {chord: {selfBias: <number>}}");
    this.reset();
  }

  reset() {
    super.reset();
    if (this.selfBias == null) { this.crf = null; return; }
    this.crf ??= new BlockSyncChordCrf({
      vocabSize: this.vocabSize, block: this.block, selfBias: this.selfBias,
    });
    this.crf.reset();
  }

  decode(outs, validFrames) {
    const logits = outs.logits?.data;
    const out = [];
    const emit = (index, i) => {
      const frame = this.frames + i;
      if (frame >= validFrames) return;                       // zero-padded tail
      out.push({
        index,
        label: this.labels ? this.labels[index] : String(index),
        confidence: logits ? posterior(logits, i * this.vocabSize, this.vocabSize, index) : null,
        time: frame * this.frameDuration,
      });
    };
    if (this.crf) {
      // Every frame goes into the recursion, including any the zero-padded tail
      // produced: run_chord_onnx.py decodes the whole chunk and trims after, and
      // emit() does the trimming. Withholding them here would change the decode
      // of the frames that ARE kept.
      const tags = this.crf.push(logits, this.labelFrames);
      for (let i = 0; i < tags.length; i++) emit(tags[i], i);
    } else if (this.hasTags) {
      const tags = outs.tags.data;
      const valid = outs.tags_valid.data;   // all-ones for blocksync; a fixed-lag
      for (let i = 0; i < tags.length; i++) {  // probe withholds frames here
        if (Number(valid[i])) emit(Number(tags[i]), i);
      }
    } else {
      for (let i = 0; i < this.labelFrames; i++)
        emit(argmax(logits, i * this.vocabSize, this.vocabSize), i);
    }
    return { labels: out };
  }

  describe() {
    return {
      ...super.describe(),
      vocab: this.vocabSize,
      block: this.block,
      // What is actually decoding, and what the graph would have done alone.
      selfBias: this.selfBias,
      graphSelfBias: this.graphSelfBias,
      decodedBy: this.crf ? "host" : "graph",
    };
  }
}

/**
 * Beat: 15 (beat, downbeat) logits per chunk at 75 fps, turned into times by one of two
 * postprocessors -- and the choice is the whole of the difference between them, since the
 * activations are identical either way.
 *
 *   dbn_blocksync  The bar-pointer DBN (beatDbn.js), decoded block-synchronously. It
 *                  knows about tempo continuity and bar structure, so it holds a tempo
 *                  through a weak bar instead of inventing a beat, and it can say WHICH
 *                  beat of the bar each beat is. Zero added latency on a block-causal
 *                  backbone: the 15 frames arrive at one instant, so decoding them
 *                  together is free. run_beat_onnx.py's `dbn_blocksync`, and the
 *                  deployment default its docstring names.
 *
 *   minimal        The peak picker (beatPostproc.js). No tempo model and no bar model,
 *                  but a tenth of the work and nothing to build at startup.
 *                  run_beat_onnx.py --postproc minimal.
 *
 * What is NOT here is madmom's OFFLINE dbn: it backtracks from the final state, so it
 * cannot emit a beat until the track ends.
 */
class BeatHead extends Head {
  constructor(task, probe, backboneAlias, config) {
    super(task, probe, backboneAlias);
    this.channels = probe.spec.channels || ["beat", "downbeat"];
    this.nChannels = probe.spec.n_channels ?? this.channels.length;
    // The frames the backbone hands over per forward call. The probe records it, and it
    // is exactly what makes block-synchronous decoding cost nothing.
    this.block = probe.spec.emission_block || this.labelFrames;
    this.postproc = DEFAULT_BEAT_POSTPROC;
    this.beatsPerBar = null;
    this.configure(config);
  }

  /**
   * Choose the postprocessor, and the bar length.
   *
   * `beatsPerBar` null decides 3-vs-4 causally at every block, which is what
   * run_beat_onnx.py does and what a system with no prior knowledge has to do; the choice
   * can flip for tens of seconds. Pinning it removes that instability and halves the
   * decode, and is what a deployment that knows the time signature should do.
   */
  configure({ postproc, beatsPerBar } = {}) {
    if (postproc !== undefined)
      this.postproc = postproc === "dbn_blocksync" ? "dbn_blocksync" : "minimal";
    if (beatsPerBar !== undefined) {
      const pinned = beatsPerBar == null ? null : Number(beatsPerBar);
      if (pinned !== this.beatsPerBar) this.dbn = null;      // a different state space
      this.beatsPerBar = pinned;
    }
    this.reset();
  }

  reset() {
    super.reset();
    if (this.postproc === "dbn_blocksync") {
      // Built once and reused: the state space and transition model are ~10 ms of
      // arithmetic, and reset() runs per track.
      this.dbn ??= new BlockSyncDbn({
        fps: this.labelRate, block: this.block, beatsPerBar: this.beatsPerBar,
      });
      this.dbn.reset();
      this.tracker = this.dbn;
    } else {
      this.tracker = new BeatTracker(this.labelRate);
    }
  }

  decode(outs, validFrames) {
    const logits = outs.logits.data;                 // [1, 15, 2], channel-last
    // Feed the postprocessor only the frames the audio paid for: run_beat_onnx.py trims
    // the activations before decoding, so silence at the end of a track must not be
    // allowed to invent a beat.
    const n = Math.max(0, Math.min(this.labelFrames, validFrames - this.frames));
    const beatLogits = new Float32Array(n);
    const downLogits = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      beatLogits[i] = logits[i * this.nChannels];
      downLogits[i] = logits[i * this.nChannels + 1];
    }
    return this.#report(this.tracker.push(beatLogits, downLogits), {
      // The raw curve, for drawing: `activation: logit`, so a probability is a sigmoid
      // away and the peak picker's threshold of 0 is a probability of 0.5.
      beat: beatLogits, downbeat: downLogits, frame: this.frames,
    });
  }

  finish() {
    const out = this.tracker.flush();
    return out.beats.length || out.downbeats.length ? this.#report(out) : null;
  }

  #report(out, activations) {
    return {
      beats: out.beats,
      downbeats: out.downbeats,
      // One shape for both postprocessors. Only the DBN knows a beat's position in the
      // bar; the peak picker's downbeats are a separate channel snapped onto beats, so
      // its events carry no number and the UI reads `downbeats` instead.
      events: out.events ?? out.beats.map((time) => ({ time, beatNumber: null })),
      activations,
      tempo: this.tracker.tempo(),
      // The DBN chooses the meter and can state it; the peak picker can only report how
      // many beats it OBSERVED between downbeats.
      meter: this.postproc === "dbn_blocksync" ? this.tracker.meter : null,
      downbeatPeriod: this.postproc === "minimal" ? this.tracker.downbeatPeriod() : null,
      totals: this.tracker.totals(),
    };
  }

  describe() {
    return {
      ...super.describe(),
      channels: this.channels,
      postproc: this.postproc,
      block: this.block,
      beatsPerBar: this.beatsPerBar,
    };
  }
}

/**
 * Key: a GLOBAL label, so what streams is a running estimate rather than a
 * label per frame.
 *
 * ONE READOUT PER BLOCK, not per frame. The backbone is block-causal with
 * lookahead_block == the chunk, so every frame in a chunk has already seen
 * audio to the chunk's end; only the last row is an honest prefix estimate at
 * its own timestamp. key_detection/infer_streaming.py and run_key_onnx.py make
 * the same choice, which is what makes their numbers comparable with these.
 */
class KeyHead extends Head {
  constructor(task, probe, backboneAlias) {
    super(task, probe, backboneAlias);
    this.labels = probe.spec.key_labels || null;
    this.nClasses = probe.spec.n_classes;
    this.samplesPerBlock = this.labelFrames * this.samplesPerFrame;
    this.blockDuration = this.samplesPerBlock / probe.spec.sample_rate;
    this.trainReadout = probe.spec.train_readout;
    this.reset();
  }

  reset() {
    super.reset();
    this.blocks = 0;
  }

  decode(outs, validFrames) {
    const logits = outs.logits.data;                       // [1, 3, 24], running
    const block = this.blocks++;
    // preds[:n_real // samples_per_block]: a block the padding invented is not
    // an estimate of anything.
    if ((block + 1) * this.labelFrames > validFrames) return { estimate: null };
    const row = (this.labelFrames - 1) * this.nClasses;    // the newest readout
    const index = argmax(logits, row, this.nClasses);
    return {
      estimate: {
        index,
        label: this.labels ? this.labels[index] : String(index),
        confidence: posterior(logits, row, this.nClasses, index),
        // Time is audio CONSUMED at this readout: the block saw everything up
        // to its own right edge.
        time: (block + 1) * this.blockDuration,
      },
    };
  }

  describe() {
    return {
      ...super.describe(),
      classes: this.nClasses,
      blockDuration: this.blockDuration,
      trainReadout: this.trainReadout,
    };
  }
}

const HEADS = { chord: ChordHead, beat: BeatHead, key: KeyHead };

/**
 * The DBN, not peak picking, unless asked otherwise.
 *
 * It is what beat_tracking/postprocessor.py calls the deployment default for a
 * block-causal backbone, it costs nothing in latency here, and it is the only one of the
 * two that can say where in the bar a beat falls. It is not uniformly better on beat
 * times -- see the README -- which is why the other one stays selectable.
 */
const DEFAULT_BEAT_POSTPROC = "dbn_blocksync";

/**
 * One or more tasks over one audio stream: the backbones they need, the probes
 * that read them, and the state all of it carries.
 */
export class MusicEngine {
  constructor(sessionOptions, config = {}) {
    this.sessionOptions = sessionOptions;
    // Per-task decoder settings, kept here so a task loaded later is built with the
    // choice already made (the beat head's postprocessor, for instance).
    this.config = config;
    this.backbones = new Map();   // alias -> OnnxStreamer
    this.heads = new Map();       // task  -> Head
    this.warnings = [];
    this.sampleRate = null;
    this.chunkSamples = null;
    this.realSamples = 0;
  }

  /**
   * Bring up the graphs for some tasks. Called again to add tasks later, which
   * is what lets moving from Chord to All tasks fetch only what is missing --
   * a backbone already up is reused, not re-fetched or re-compiled.
   *
   * `graphs`: {alias: {model, spec}} for every backbone and probe named below.
   * `tasks`:  {task: {backbone: alias, probe: alias}}.
   */
  async addTasks({ graphs, tasks, config }) {
    // Stage the whole addition. Publishing a head before geometry/decoder
    // validation made a rejected task look loaded on retry, bypassing the check.
    const nextConfig = config ? { ...this.config, ...config } : this.config;
    const backbones = new Map(this.backbones);
    const heads = new Map(this.heads);
    const warnings = [...this.warnings];
    let sampleRate = this.sampleRate;
    let chunkSamples = this.chunkSamples;
    const additions = Object.entries(tasks).filter(([task]) => !heads.has(task));
    const created = [];
    const create = async (graph) => {
      const streamer = await OnnxStreamer.create(graph.model, graph.spec, this.sessionOptions);
      created.push(streamer);
      return streamer;
    };
    try {
      for (const alias of new Set(additions.map(([, pair]) => pair.backbone))) {
        if (backbones.has(alias)) continue;
        if (!graphs[alias]) throw new Error(`no graph was supplied for backbone "${alias}"`);
        backbones.set(alias, await create(graphs[alias]));
      }
      for (const [task, pair] of additions) {
        const backbone = backbones.get(pair.backbone);
        if (!backbone) throw new Error(`${task}: no backbone graph "${pair.backbone}" was loaded`);
        const probe = await create(graphs[pair.probe]);
        const warning = checkPair(task, backbone, probe);
        if (warning) warnings.push(warning);
        const Ctor = HEADS[task];
        heads.set(task, new Ctor(task, probe, pair.backbone, nextConfig[task]));

        const b = backbone.spec;
        if (sampleRate === null) {
          sampleRate = b.sample_rate;
          chunkSamples = b.chunk_samples;
        } else if (sampleRate !== b.sample_rate || chunkSamples !== b.chunk_samples) {
          throw new Error("the tasks' backbones disagree on the chunk geometry; they cannot " +
                          "share one audio stream");
        }
      }
    } catch (err) {
      // Only this attempt's sessions belong to us; keep working tasks intact.
      // Cleanup failures must not hide the cause or skip another release.
      await Promise.allSettled(created.map(async (g) => g.session.release()));
      throw err;
    }
    this.config = nextConfig;
    this.backbones = backbones;
    this.heads = heads;
    this.warnings = warnings;
    this.sampleRate = sampleRate;
    this.chunkSamples = chunkSamples;
    this.reset();
    return this.describe();
  }

  /**
   * Change a task's decoder settings -- {beat: {postproc, beatsPerBar}} -- without
   * touching the graphs.
   *
   * The graphs are the expensive part and none of this reaches them: the beat probe emits
   * the same activations whichever postprocessor reads them, so switching decoder is free
   * apart from building the DBN's state space once. It does reset the head's state, so
   * this belongs between tracks, not mid-track.
   */
  configure(config = {}) {
    // Build replacement heads with fresh carried state, sharing only the
    // compiled sessions. No current decoder or probe is reset until every
    // requested setting has constructed successfully.
    const heads = new Map(this.heads);
    for (const [task, settings] of Object.entries(config)) {
      const head = heads.get(task);
      if (!head?.configure) continue;
      const previous = task === "chord"
        ? { selfBias: head.selfBias }
        : { postproc: head.postproc, beatsPerBar: head.beatsPerBar };
      const changes = Object.fromEntries(Object.entries(settings === undefined ? {} : settings)
        .filter(([, value]) => value !== undefined));
      const probe = new OnnxStreamer(head.probe.session, head.probe.spec);
      heads.set(task, new HEADS[task](task, probe, head.backboneAlias,
        { ...previous, ...changes }));
    }
    this.heads = heads;
    this.config = { ...this.config, ...config };
    return this.describe();
  }

  reset() {
    for (const b of this.backbones.values()) b.reset();
    for (const h of this.heads.values()) h.reset();
    this.realSamples = 0;
  }

  /**
   * One chunk in, every loaded task's output for it out. `audio` must be
   * exactly chunkSamples float32 mono samples at sampleRate; `realSamples` is
   * how many of them came from the file rather than from the caller's
   * zero-padding, which is what lets each head end its output where the audio
   * does.
   *
   * The backbone is block-causal with lookahead_block == the chunk, so frame t
   * already sees to the end of its own block: a chunk's output is final the
   * moment the chunk is processed.
   */
  async pushChunk(audio, realSamples = audio.length) {
    if (audio.length !== this.chunkSamples)
      throw new Error(`expected ${this.chunkSamples} samples, got ${audio.length}`);
    this.realSamples += realSamples;
    const t0 = performance.now();

    // One backbone run per distinct backbone, however many probes read it: in
    // "all tasks" mode chord and key share the r0 features.
    const features = new Map();
    const audioTensor = new ort.Tensor("float32", audio, [1, audio.length]);
    for (const [alias, backbone] of this.backbones) {
      const outs = await backbone.run({ [backbone.spec.audio_input.name]: audioTensor });
      features.set(alias, outs[backbone.valueNames[0]]);
    }

    const result = { ms: 0, taskMs: {} };
    for (const [task, head] of this.heads) {
      const t1 = performance.now();
      result[task] = await head.run(features.get(head.backboneAlias), this.realSamples);
      result.taskMs[task] = performance.now() - t1;
    }
    result.ms = performance.now() - t0;
    return result;
  }

  /** End of track: flush whatever any postprocessor is still holding. */
  finish() {
    const result = {};
    for (const [task, head] of this.heads) {
      const tail = head.finish();
      if (tail) result[task] = tail;
    }
    return result;
  }

  describe() {
    const backbones = {};
    let backboneParams = 0;
    let stateBytes = 0;
    for (const [alias, b] of this.backbones) {
      backbones[alias] = {
        params: b.spec.n_params / 1e6,
        context: b.spec.context,
        stateBytes: b.stateBytes(),
        source: b.spec.source_checkpoint,
      };
      backboneParams += b.spec.n_params / 1e6;
      stateBytes += b.stateBytes();
    }
    const heads = {};
    for (const [task, h] of this.heads) {
      heads[task] = { ...h.describe(), stateBytes: h.probe.stateBytes() };
      stateBytes += h.probe.stateBytes();
    }
    return {
      tasks: [...this.heads.keys()],
      backbones,
      heads,
      warnings: this.warnings,
      sampleRate: this.sampleRate,
      chunkSamples: this.chunkSamples,
      chunkMs: (this.chunkSamples / this.sampleRate) * 1000,
      params: backboneParams + Object.values(heads).reduce((n, h) => n + h.params, 0),
      backboneParams,
      stateBytes,
    };
  }
}

/** Build an engine from {alias: {model, spec}} graphs and a task->pair map,
 *  where `model` is a URL or an ArrayBuffer/Uint8Array of the .onnx graph. */
export async function createEngine({ graphs, tasks, sessionOptions, config }) {
  const engine = new MusicEngine(sessionOptions, config);
  await engine.addTasks({ graphs, tasks });
  return engine;
}
