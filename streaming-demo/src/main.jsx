import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity, AudioWaveform, BarChart3, CheckCircle2, CircleStop,
  FileAudio, Gauge, KeyRound, Music2, Pause, Play, Radio, Sparkles, Upload
} from "lucide-react";
import {
  InferenceStream, decodeToModelRate, displayChord, displayKey, appendLabel, keyConvergence
} from "./streamingClient.js";
import { FIX_HINT, MODEL_REPO, fetchManifest, pickModel } from "./modelSource.js";
import "./styles.css";
// Bundled with the page (dev: served from ../docs, see vite.config.js), so the
// sample button works the same locally and on the hosted Space.
import sampleUrl from "../../docs/test.wav?url";

// `tasks` is which heads a page runs. All three are exported, and all three
// stream through the same path: a graph whose spec lists its `state`, driven one
// chunk per hop. "All tasks" runs them over one audio stream -- two backbones
// (chord and key share r0, beat needs r15) and three probes per chunk.
const PAGES = {
  all: { label: "All tasks", icon: BarChart3, tasks: ["chord", "beat", "key"],
         desc: "Run chord, beat, and key understanding together, over one stream." },
  chord: { label: "Chord recognition", icon: Music2, tasks: ["chord"],
           desc: "Identify the current harmonic chord in real time." },
  beat: { label: "Beat tracking", icon: Activity, tasks: ["beat"],
          desc: "Track beat and downbeat positions, and tempo, as the audio plays." },
  key: { label: "Key detection", icon: KeyRound, tasks: ["key"],
         desc: "Estimate the global musical key, revised as more is heard." },
};

const TASK_LABEL = { chord: "chord recognition", beat: "beat tracking", key: "key detection" };
const noHead = (task) =>
  MODEL_REPO
    ? `No head for ${TASK_LABEL[task]} in ${MODEL_REPO} for this model.`
    : `No head exported for ${TASK_LABEL[task]} in this checkpoint — run \`npm run link-models\` ` +
      "against a tree that has one.";

// Bar lengths the meter can be PINNED to, in beats, split by how much of the beat
// probe's training data is actually in them (measured over its training tracks
// that carry downbeats, weighted by duration).
//
// The DBN builds a bar HMM for any length, so this list is free to be wider than the
// pair chosen causally -- and it must be, because those are different jobs. A candidate
// has to WIN a likelihood comparison against the others, and measurement says every
// addition to {3,4} makes the common case worse: adding 2 alone drops 8 known-meter
// tracks from 5/8 correct to 2/8, because a 2-beat bar explains a 4-beat one and wins by
// 34-143 nats. A pin has nothing to compete against, so it can only help.
//
// Note these are BEATS PER BAR, not time signatures: nothing in the model encodes a note
// value. 3/8 is 3 beats, and 6/8 is 2 beats felt in two or 6 felt in six.
const BAR_LENGTHS = [
  { beats: 2, share: "12% of training", common: true },
  { beats: 3, share: "11%", common: true },
  { beats: 4, share: "67%", common: true },
  { beats: 5, share: "1.7%", common: false },
  { beats: 6, share: "1.5%", common: false },
  { beats: 7, share: "3.6%", common: false },
  { beats: 8, share: "2.8%", common: false },
  { beats: 12, share: "0.5%", common: false },
];
const wellTrained = (n) => BAR_LENGTHS.some((b) => b.beats === n && b.common);

// Pixels per second on the beat strip: ~13 s of history before the playhead in
// a 900 px panel, which is three or four bars at most tempos.
const PPS = 55;
const PLAYHEAD = 0.78;      // where "now" sits across the strip

// One session for the whole page: a backbone is ~100 MB to fetch and compile,
// so moving between Home and a task must not pay for it twice, and moving
// between tasks must pay only for what it adds.
let sharedStream = null;
const stream = () => (sharedStream ||= new InferenceStream(pickModel()));

function formatTime(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function emptyBeat() {
  return { events: [], downbeats: [], tempo: null, meter: null, downbeatPeriod: null,
           totals: { beats: 0, downbeats: 0 } };
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[sorted.length >> 1];
}

/**
 * Keep a horizontally scrolling strip pinned to its newest item.
 *
 * Both strips grow to the right as the track plays, and without this the newest
 * chord or key -- the one the page is about -- scrolls out of sight while the
 * user watches the oldest.
 */
function useStickyScroll(newest) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [newest]);
  return ref;
}

/** Which heads the checkpoint actually has, so Home can say so before any
 *  graph is fetched. The manifest is a few hundred bytes. */
function useAvailableTasks() {
  const [available, setAvailable] = useState(null);
  useEffect(() => {
    let alive = true;
    fetchManifest(pickModel())
      .then((m) => alive && setAvailable({ tasks: Object.keys(m.tasks), model: m.model }))
      .catch(() => alive && setAvailable(null));
    return () => { alive = false; };
  }, []);
  return available;
}

function Home({ onChoose }) {
  const available = useAvailableTasks();
  return (
    <main className="page">
      <section className="shell home-shell">
        <header className="home-hero">
          <div className="eyebrow"><AudioWaveform size={15} /> MUSIC UNDERSTANDING</div>
          <h1>Real-time Music Understanding</h1>
          <p>
            Stream audio through a music understanding model and visualize
            harmonic and rhythmic predictions as they arrive.
          </p>
        </header>

        <div className="task-grid">
          {Object.entries(PAGES).map(([id, page]) => {
            const Icon = page.icon;
            const absent = available ? page.tasks.filter((t) => !available.tasks.includes(t)) : [];
            return (
              <button key={id} className={`task-card ${id === "all" ? "featured" : ""}`} onClick={() => onChoose(id)}>
                <div className="task-icon"><Icon size={22} /></div>
                <div className="task-title">
                  {page.label}
                  {absent.length > 0 && <span className="badge">no head yet</span>}
                </div>
                <div className="task-desc">{page.desc}</div>
                <div className="task-arrow">Open →</div>
              </button>
            );
          })}
        </div>

        <div className="home-note">
          <span className="dot live-dot" />
          {available
            ? `Chord, beat and key all run the real streaming ${available.model} graphs in your browser via onnxruntime-web.` +
              (MODEL_REPO ? " Nothing is uploaded: your audio stays on this device." : "")
            : `No graphs found — ${FIX_HINT}.`}
        </div>
      </section>
    </main>
  );
}

/**
 * Beats on a moving ruler, each tick at the time it was detected at, with the
 * playhead where the audio actually is.
 *
 * Positioning ticks by their own timestamp rather than flashing them on arrival
 * is what makes this honest: a beat is known one chunk plus the postprocessor's own
 * wait after it sounded, so its tick appears just behind the playhead, and where it
 * sits is where the model says the beat was.
 *
 * The layer is moved by one style write per frame, so adding a tick re-renders
 * and scrolling does not.
 */
function BeatStrip({ audioRef, events, downbeats }) {
  const layerRef = useRef(null);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const el = audioRef.current;
      if (el && layerRef.current)
        layerRef.current.style.transform = `translate3d(${-el.currentTime * PPS}px,0,0)`;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [audioRef]);

  // A downbeat time IS one of the beat times -- the DBN numbers the beat it decoded, and
  // the peak picker snaps its downbeats onto beats -- so matching by value is exact.
  const isDownbeat = new Set(downbeats);
  return (
    <div className="strip-view">
      <div className="strip-layer" ref={layerRef} style={{ left: `${PLAYHEAD * 100}%` }}>
        {events.map(({ time, number }) => (
          <span key={time} className={`strip-tick ${isDownbeat.has(time) ? "downbeat" : ""}`}
                style={{ left: `${time * PPS}px` }}>
            {/* Only the DBN knows where in the bar a beat sits; peak picking leaves this
                blank rather than guessing. */}
            {number != null && <i>{number}</i>}
          </span>
        ))}
      </div>
      <div className="strip-playhead" style={{ left: `${PLAYHEAD * 100}%` }} />
    </div>
  );
}

function Demo({ page, onHome }) {
  const audioRef = useRef(null);
  const pumpRef = useRef(null);
  const infoRef = useRef(null);        // model geometry, once the graphs are up
  const samplesRef = useRef(null);     // decoded mono audio at the model's rate
  const decodedRef = useRef({ file: null, rate: 0, samples: null });
  const cursorRef = useRef(0);         // next chunk to feed
  const busyRef = useRef(false);       // one chunk in flight; state is sequential
  const endedRef = useRef(false);
  const finishedRef = useRef(false);   // the tail flush is only owed once
  const timesRef = useRef([]);

  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [playing, setPlaying] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [position, setPosition] = useState(0);
  const [duration, setDuration] = useState(0);
  const [status, setStatus] = useState("Ready");
  const [autoStart, setAutoStart] = useState(false);   // stream the sample once loaded
  // "loading", not "idle": the effect below always runs, so an idle default
  // would have the status pill claim Ready for a frame before the fetch starts.
  const [engine, setEngine] = useState({ status: "loading", progress: 0, bytes: 0, info: null, error: null });
  const [latency, setLatency] = useState({ compute: 0, chunkMs: 200 });

  const [currentChord, setCurrentChord] = useState({ label: null, confidence: 0 });
  const [segments, setSegments] = useState([]);
  const [beat, setBeat] = useState(emptyBeat());
  // Which beat postprocessor is running. Both read the SAME activations -- the choice
  // only changes what turns them into times -- so switching costs no inference.
  const [beatConfig, setBeatConfig] = useState({ postproc: "dbn_blocksync", beatsPerBar: null });
  const [keyState, setKeyState] = useState({ history: [], runs: [] });

  // Both strips follow their newest entry; the last segment's end moves with
  // every label, so it is what tells them something arrived.
  const timelineRef = useStickyScroll(segments[segments.length - 1]?.end);
  const keyRunsRef = useStickyScroll(keyState.runs[keyState.runs.length - 1]?.end);

  const wanted = PAGES[page].tasks;
  const ready = engine.status === "ready";
  // What is actually running: what this page asked for, minus any head the
  // checkpoint does not have.
  const live = engine.info ? wanted.filter((t) => engine.info.tasks.includes(t)) : [];
  const absent = engine.info ? wanted.filter((t) => !engine.info.tasks.includes(t)) : [];
  const runs = (task) => wanted.includes(task);

  // Bring up the graphs this page needs as soon as it is opened. Additive: the
  // worker keeps what is already compiled and fetches only the rest.
  useEffect(() => {
    let alive = true;
    const s = stream();
    const settle = (info) => {
      infoRef.current = info;
      setEngine({ status: "ready", progress: 1, bytes: 0, info, error: null });
      setLatency((l) => ({ ...l, chunkMs: info.chunkMs }));
      if (info.heads.beat)
        setBeatConfig({ postproc: info.heads.beat.postproc,
                        beatsPerBar: info.heads.beat.beatsPerBar ?? null });
      for (const w of info.warnings || []) console.warn(`[engine] ${w}`);
    };
    // Already up from an earlier page: no progress bar, no refetch.
    if (s.info && wanted.every((t) => s.info.tasks.includes(t) || (s.info.missing || []).includes(t))) {
      settle(s.info);
      return () => { alive = false; };
    }
    setEngine((e) => ({ ...e, status: "loading", error: null }));
    s.load(wanted, ({ loaded, total, compiling }) => {
      if (alive) setEngine((e) => ({ ...e, status: compiling ? "compiling" : "loading",
                                     progress: loaded / total, bytes: total }));
    }, { beat: beatConfig })
      .then((info) => alive && settle(info))
      .catch((err) => alive && setEngine((e) => ({ ...e, status: "error", error: err.message })));
    return () => { alive = false; };
  }, [page]);

  useEffect(() => () => {
    clearInterval(pumpRef.current);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
  }, [audioUrl]);

  function resetStreams() {
    setSegments([]);
    setCurrentChord({ label: null, confidence: 0 });
    setBeat(emptyBeat());
    setKeyState({ history: [], runs: [] });
    timesRef.current = [];
    cursorRef.current = 0;
    endedRef.current = false;
    finishedRef.current = false;
  }

  /** Decode once per (file, rate); the models want mono at their own rate. */
  async function ensureDecoded(target, rate) {
    const cached = decodedRef.current;
    if (cached.file === target && cached.rate === rate) return cached.samples;
    const { samples } = await decodeToModelRate(target, rate);
    decodedRef.current = { file: target, rate, samples };
    return samples;
  }

  function chooseFile(e) {
    const selected = e.target.files?.[0];
    if (selected) loadFile(selected);
  }

  /** The bundled sample: load it like an upload, then stream it as soon as the
   *  graphs are up (the effect below), so one click is the whole demo. */
  async function playSample() {
    setStatus("Fetching the sample");
    try {
      const res = await fetch(sampleUrl);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      if (await loadFile(new File([blob], "test.wav", { type: blob.type || "audio/wav" })))
        setAutoStart(true);
    } catch (err) {
      setStatus("Could not fetch the sample");
      console.error(err);
    }
  }

  useEffect(() => {
    if (!autoStart || !ready || !file) return;
    setAutoStart(false);
    // Safari may refuse play() this far from the click; the file stays loaded.
    startDemo().catch((err) => {
      setStatus("Sample loaded: press Start streaming");
      console.warn(err);
    });
  }, [autoStart, ready, file]);

  /** Returns whether the file decoded. */
  async function loadFile(selected) {
    clearInterval(pumpRef.current);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setFile(selected);
    setAudioUrl(URL.createObjectURL(selected));
    setPosition(0);
    setDuration(0);
    setPlaying(false);
    setStreaming(false);
    resetStreams();
    samplesRef.current = null;

    setStatus("Decoding audio");
    try {
      // 24 kHz is what every graph in ckpt_onnx/ uses; the real rate replaces
      // it on the first play if a future export differs.
      samplesRef.current = await ensureDecoded(selected, infoRef.current?.sampleRate ?? 24000);
      setStatus("Ready");
      return true;
    } catch (err) {
      setStatus("Could not decode this file");
      console.error(err);
      return false;
    }
  }

  /** Fold one chunk's output into the screen. */
  function absorb(out, geometry) {
    if (out.chord?.labels?.length) {
      const labels = out.chord.labels;
      setSegments((prev) =>
        labels.reduce((acc, l) => appendLabel(acc, l, geometry.chordFrame), prev).slice(-64));
      setCurrentChord(labels[labels.length - 1]);
    }
    if (out.beat && (out.beat.events.length || out.beat.downbeats.length)) {
      setBeat((prev) => ({
        events: [...prev.events,
                 ...out.beat.events.map((e) => ({ time: e.time, number: e.beatNumber }))].slice(-512),
        downbeats: [...prev.downbeats, ...out.beat.downbeats].slice(-128),
        tempo: out.beat.tempo ?? prev.tempo,
        meter: out.beat.meter ?? prev.meter,
        downbeatPeriod: out.beat.downbeatPeriod ?? prev.downbeatPeriod,
        totals: out.beat.totals ?? prev.totals,
      }));
    }
    if (out.key?.estimate) {
      const est = out.key.estimate;
      const block = geometry.keyBlock;
      setKeyState((prev) => ({
        history: [...prev.history, est],
        // The estimate is stamped with the audio it has CONSUMED, so the run it
        // starts began one block earlier.
        runs: appendLabel(prev.runs, { ...est, time: est.time - block }, block).slice(-32),
      }));
    }
  }

  /**
   * Feed every chunk whose audio the listener has already heard.
   *
   * A chunk is 0.2 s and is fed only once playback has passed its end, so no
   * model ever sees further into the track than the listener does: what is on
   * screen is the pipeline's genuine one-chunk latency, not a replay.
   */
  const pump = useCallback(async () => {
    const info = infoRef.current;
    const samples = samplesRef.current;
    const el = audioRef.current;
    if (busyRef.current || !info || !samples || !el || !info.tasks.length) return;
    busyRef.current = true;
    try {
      const { chunkSamples, sampleRate } = info;
      const chunkSeconds = chunkSamples / sampleRate;
      const totalChunks = Math.ceil(samples.length / chunkSamples);
      const geometry = {
        chordFrame: info.heads.chord?.frameDuration,
        keyBlock: info.heads.key?.blockDuration,
      };

      while (cursorRef.current < totalChunks &&
             ((cursorRef.current + 1) * chunkSeconds <= el.currentTime || endedRef.current)) {
        const start = cursorRef.current * chunkSamples;
        // Copy: the buffer is transferred to the worker, and a subarray would
        // take the whole decoded track with it.
        let audio = samples.slice(start, start + chunkSamples);
        const real = audio.length;
        if (real < chunkSamples) {
          const padded = new Float32Array(chunkSamples);
          padded.set(audio);
          audio = padded;
        }
        // `real` is what lets each head drop the output the zero-padded tail
        // invented, so every transcript ends where the audio does.
        const out = await stream().pushChunk(audio, real);
        cursorRef.current += 1;
        absorb(out, geometry);
        timesRef.current = [...timesRef.current.slice(-49), out.ms];
        setLatency({ compute: median(timesRef.current), chunkMs: info.chunkMs });
      }

      if (endedRef.current && cursorRef.current >= totalChunks && !finishedRef.current) {
        finishedRef.current = true;
        // Beat's peak picker holds the last few frames until their windows are
        // complete; this is where those beats come out.
        absorb(await stream().finish(), geometry);
        clearInterval(pumpRef.current);
        pumpRef.current = null;
        setStreaming(false);
        setPlaying(false);
        setStatus("Finished");
      }
    } catch (err) {
      clearInterval(pumpRef.current);
      setStreaming(false);
      setStatus("Inference failed");
      console.error(err);
    } finally {
      busyRef.current = false;
    }
  }, []);

  async function startDemo() {
    const el = audioRef.current;
    if (!el || !file || !ready) return;

    setStatus("Decoding audio");
    try {
      samplesRef.current = await ensureDecoded(file, infoRef.current.sampleRate);
    } catch (err) {
      setStatus("Could not decode this file");
      console.error(err);
      return;
    }

    resetStreams();
    if (live.length) await stream().reset();   // per-track state, by construction
    el.currentTime = 0;
    await el.play();
    setPlaying(true);
    setStreaming(true);
    setStatus(live.length ? "Streaming" : "Playing (no head to run)");

    clearInterval(pumpRef.current);
    if (live.length) pumpRef.current = setInterval(pump, 50);
  }

  /**
   * Switch beat postprocessor (or pin the meter).
   *
   * No graph is touched: the probe emits the same activations either way, so this is a
   * decoder swap. It does reset the beat head's state, which is why the control is
   * disabled while streaming rather than applied mid-track.
   */
  async function changeBeatConfig(next) {
    setBeatConfig(next);
    setBeat(emptyBeat());
    try {
      const info = await stream().configure({ beat: next });
      infoRef.current = info;
      setEngine((e) => ({ ...e, info }));
    } catch (err) {
      setStatus("Could not change the decoder");
      console.error(err);
    }
  }

  function stopDemo() {
    audioRef.current?.pause();
    clearInterval(pumpRef.current);
    pumpRef.current = null;
    setPlaying(false);
    setStreaming(false);
    setStatus("Stopped");
  }

  function togglePlay() {
    if (!audioRef.current || !file) return;
    if (playing) {
      audioRef.current.pause();
      setPlaying(false);
      setStatus(streaming ? "Paused" : "Ready");
      return;
    }
    if (!streaming) startDemo();
    else {
      audioRef.current.play();
      setPlaying(true);
      setStatus("Streaming");
    }
  }

  const progress = duration ? Math.min(100, (position / duration) * 100) : 0;
  const PageIcon = PAGES[page].icon;
  const chordLabel = currentChord.label ? displayChord(currentChord.label) : "—";
  const rtf = latency.chunkMs ? latency.compute / latency.chunkMs : 0;
  const convergence = keyConvergence(keyState.history);
  const keyNow = displayKey(convergence?.label);
  // What THIS page runs, not everything the worker has compiled: opening beat
  // after chord leaves the chord backbone loaded, and counting it here would
  // overstate what the page in front of you costs.
  const liveBackbones = engine.info ? [...new Set(live.map((t) => engine.info.heads[t].backbone))] : [];
  const liveState = engine.info
    ? liveBackbones.reduce((n, a) => n + engine.info.backbones[a].stateBytes, 0) +
      live.reduce((n, t) => n + engine.info.heads[t].stateBytes, 0)
    : 0;
  const lastWasDownbeat = beat.events.length > 0 &&
    beat.downbeats[beat.downbeats.length - 1] === beat.events[beat.events.length - 1].time;
  const pill = engine.status === "loading" ? `Loading model ${Math.round(engine.progress * 100)}%`
    : engine.status === "compiling" ? "Compiling graphs"
    : engine.status === "error" ? "Model failed" : status;

  return (
    <main className="page">
      <section className="shell">
        <header className="header">
          <div className="header-left">
            <button className="back-button" onClick={onHome}>← Home</button>
            <div className="eyebrow"><PageIcon size={15} /> {PAGES[page].label.toUpperCase()}</div>
            <h1>{PAGES[page].label}</h1>
            <p>{PAGES[page].desc}</p>
          </div>
          <div className={`status-pill ${streaming ? "live" : ""}`}>
            <span className="dot" /> {pill}
          </div>
        </header>

        {engine.status !== "ready" && (
          <section className="panel model-panel">
            {engine.status === "error"
              ? <div className="empty padded">Could not load the graphs: {engine.error} — {FIX_HINT}, then reload.</div>
              : <>
                  <div className="panel-head">
                    <span>{engine.status === "compiling" ? "Compiling the graphs" : "Fetching the graphs"}</span>
                    <span className="small-muted">
                      {engine.bytes ? `~${Math.round(engine.bytes / 1e6)} MB` : "…"}, cached after the first load
                    </span>
                  </div>
                  <div className="progress-wrap model-progress">
                    <div className="progress"><div style={{ width: `${Math.round(engine.progress * 100)}%` }} /></div>
                  </div>
                </>}
          </section>
        )}

        {absent.length > 0 && (
          <section className="panel"><div className="empty padded">{absent.map(noHead).join(" ")}</div></section>
        )}

        <div className={`metrics-grid ${wanted.length > 2 ? "three" : wanted.length === 2 ? "two" : "one"}`}>
          {runs("chord") && (
            <section className="metric-card hero-metric">
              <div className="metric-kicker">CURRENT CHORD</div>
              <div className="metric-value chord">{chordLabel}</div>
              <div className="confidence">
                <Gauge size={15} />
                {currentChord.confidence
                  ? `${Math.round(currentChord.confidence * 100)}% confidence`
                  : "Awaiting prediction"}
              </div>
            </section>
          )}
          {runs("beat") && (
            <section className="metric-card">
              <div className="metric-kicker">TEMPO</div>
              <div className="metric-value">
                {beat.tempo ? beat.tempo.toFixed(0) : "—"}
                {beat.tempo ? <small> BPM</small> : null}
              </div>
              <div className="confidence">
                {/* Re-mounted on every beat, so the CSS animation restarts: one
                    flash per detected beat, brighter on a downbeat. */}
                <span key={beat.totals.beats}
                      className={`pulse-dot ${lastWasDownbeat ? "strong" : ""}`} />
                {beat.totals.beats
                  ? `${beat.totals.beats} beats · ${beat.totals.downbeats} downbeats` +
                    (beat.meter ? ` · ${beat.meter}/4` : "")
                  : "Awaiting beats"}
              </div>
            </section>
          )}
          {runs("key") && (
            <section className="metric-card">
              <div className="metric-kicker">ESTIMATED KEY</div>
              <div className="metric-value">
                {keyNow.root}{keyNow.mode ? <small> {keyNow.mode}</small> : null}
              </div>
              <div className="confidence">
                <Gauge size={15} />
                {convergence
                  ? `${Math.round(convergence.confidence * 100)}% · settled at ${convergence.tSettle.toFixed(1)}s`
                  : "Awaiting prediction"}
              </div>
            </section>
          )}
        </div>

        {runs("chord") && (
          <section className="panel">
            <div className="panel-head">
              <span>Recognition timeline</span>
              <span className="small-muted">
                {streaming ? "live" : "idle"} · {engine.info?.heads.chord
                  ? `${engine.info.heads.chord.labelsPerChunk} labels every ${engine.info.chunkMs.toFixed(0)} ms, ` +
                    `${engine.info.heads.chord.decoder} CRF, λ=` +
                    // Which lambda is actually decoding: the host's, or the one
                    // frozen into the graph when selfBias is null.
                    `${engine.info.heads.chord.selfBias ?? engine.info.heads.chord.graphSelfBias}`
                  : "200 ms chunks"}
              </span>
            </div>
            {segments.length > 0
              ? <div className="timeline" ref={timelineRef}>
                  {segments.map((s, i) => (
                    <div className={`event ${i === segments.length - 1 ? "current" : ""}`} key={`${s.start}-${i}`}>
                      <strong>{displayChord(s.label)}</strong>
                      <span>{formatTime(s.start)} · {(s.end - s.start).toFixed(1)}s · {Math.round(s.confidence * 100)}%</span>
                    </div>
                  ))}
                </div>
              : <div className="empty padded">Start streaming to see chords appear as the audio plays.</div>}
          </section>
        )}

        {runs("beat") && (
          <section className="panel">
            <div className="panel-head">
              <span>Beat timeline</span>
              <span className="panel-controls">
                <span className="control">
                  <span className="control-label">Decoder</span>
                  <select value={beatConfig.postproc} disabled={streaming}
                          onChange={(e) => changeBeatConfig({ ...beatConfig, postproc: e.target.value })}>
                    <option value="dbn_blocksync">bar-pointer DBN</option>
                    <option value="minimal">peak picking</option>
                  </select>
                </span>
                {beatConfig.postproc === "dbn_blocksync" && (
                  <span className="control">
                    <span className="control-label">Meter</span>
                    <select value={beatConfig.beatsPerBar ?? ""} disabled={streaming}
                            onChange={(e) => changeBeatConfig({
                              ...beatConfig,
                              beatsPerBar: e.target.value ? Number(e.target.value) : null })}>
                    {/* `causal` weighs the two candidates the probe was trained on and
                        picks between them per block. Every other option PINS one bar
                        length -- a single HMM, nothing to lose a comparison to -- which
                        is why lengths outside the trained pair are offered here and not
                        as candidates: pinning cannot destabilise anything, adding a
                        competitor can. */}
                    <option value="">causal (3 or 4)</option>
                    <optgroup label="well represented in training">
                      {BAR_LENGTHS.filter((b) => b.common).map((b) => (
                        <option key={b.beats} value={b.beats}>{b.beats} beats</option>
                      ))}
                    </optgroup>
                    <optgroup label="thinly represented — see note below">
                      {BAR_LENGTHS.filter((b) => !b.common).map((b) => (
                        <option key={b.beats} value={b.beats}>{b.beats} beats</option>
                      ))}
                    </optgroup>
                    </select>
                  </span>
                )}
              </span>
            </div>
            {beat.events.length > 0
              ? <BeatStrip audioRef={audioRef} events={beat.events} downbeats={beat.downbeats} />
              : <div className="empty padded">Start streaming to see beats land as the audio plays.</div>}
            <div className="panel-foot">
              <p>
                Tall ticks are downbeats.{" "}
                {beatConfig.postproc === "dbn_blocksync"
                  ? "The number under a tick is the beat's place in its bar."
                  : "Peak picking gives no bar numbers."}
                {" "}Stop streaming to change the settings above.
              </p>
              <p>
                <strong>Decoder</strong> — <em>bar-pointer DBN</em> follows tempo and bars, and
                numbers every beat. <em>peak picking</em> is simpler: no numbers, more stray beats.
              </p>
              {beatConfig.postproc === "dbn_blocksync" && (
                <p>
                  <strong>Meter</strong> — <em>causal</em> lets the model pick 3 or 4 beats per
                  bar, and it may flip early on. Pin it if you know it. Counted in beats, so 3/8
                  is 3 and 6/8 is 2 or 6.
                  {beatConfig.beatsPerBar != null && !wellTrained(beatConfig.beatsPerBar) &&
                    ` ${beatConfig.beatsPerBar} beats is only ` +
                    `${BAR_LENGTHS.find((b) => b.beats === beatConfig.beatsPerBar)?.share} of ` +
                    "training — expect weaker downbeats."}
                </p>
              )}
            </div>
          </section>
        )}

        {runs("key") && (
          <section className="panel key-panel">
            <div className="panel-head">
              <span>Key estimate</span>
              <span className="small-muted">
                {engine.info?.heads.key
                  ? `one readout every ${(engine.info.heads.key.blockDuration * 1000).toFixed(0)} ms, ${engine.info.heads.key.classes} classes`
                  : "running readout"}
              </span>
            </div>
            {keyState.runs.length > 0
              ? <>
                  <div className="key-history" ref={keyRunsRef}>
                    {keyState.runs.map((r, i) => {
                      const k = displayKey(r.label);
                      return (
                        <div className={`key-badge ${i === keyState.runs.length - 1 ? "current" : ""}`}
                             key={`${r.start}-${i}`}>
                          <strong>{k.root} <span>{k.mode}</span></strong>
                          <span>{formatTime(r.start)} · {(r.end - r.start).toFixed(1)}s</span>
                        </div>
                      );
                    })}
                  </div>
                  <div className="panel-foot">
                    Key is a GLOBAL label, so what streams is a running estimate: the readout at
                    time t is the answer from audio 0..t. First said <strong>{convergence.label}</strong>
                    {" "}at {convergence.tFirst.toFixed(1)}s, settled on it at {convergence.tSettle.toFixed(1)}s,
                    {" "}{convergence.flips} changes of mind ({convergence.flipsPerMin.toFixed(1)}/min).
                  </div>
                </>
              : <div className="empty padded">Start streaming to watch the estimate converge.</div>}
          </section>
        )}

        <section className="player panel">
          <div className="file-row">
            <div className="file-icon"><FileAudio size={19} /></div>
            <div className="file-meta">
              <strong>{file ? file.name : "No audio selected"}</strong>
              <span>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : "Upload a WAV, MP3, or other browser-supported audio file"}</span>
            </div>
            <button className="button secondary" disabled={streaming || autoStart} onClick={playSample}
                    title="Stream the bundled sample track (docs/test.wav)">
              <Sparkles size={16} /> Try a sample
            </button>
            <label className="button secondary"><Upload size={16} /> Upload<input type="file" accept="audio/*" onChange={chooseFile} hidden /></label>
          </div>

          <div className="progress-wrap">
            <div className="progress"><div style={{ width: `${progress}%` }} /></div>
            <div className="time-row"><span>{formatTime(position)}</span><span>{formatTime(duration)}</span></div>
          </div>

          <div className="controls">
            <button className="button secondary" disabled={!file || !ready} onClick={togglePlay}>
              {playing ? <Pause size={17} /> : <Play size={17} />} {playing ? "Pause" : "Play"}
            </button>
            <button className="button primary" disabled={!file || !ready} onClick={startDemo}>
              <Radio size={17} /> Start streaming
            </button>
            <button className="button danger" disabled={!streaming} onClick={stopDemo}>
              <CircleStop size={17} /> Stop
            </button>
          </div>

          <audio
            ref={audioRef}
            src={audioUrl}
            onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
            onTimeUpdate={(e) => setPosition(e.currentTarget.currentTime)}
            onEnded={() => {
              endedRef.current = true;
              if (live.length) pump();
              else { setPlaying(false); setStreaming(false); setStatus("Finished"); }
            }}
          />
        </section>

        <footer className="footer">
          <div><CheckCircle2 size={15} /> Model <strong>{engine.info
            ? `${engine.info.model} streaming · ${live.join(", ") || "no head"}`
            : "loading"}</strong></div>
          <div><Activity size={15} /> Graphs <strong>{engine.info
            ? `${liveBackbones.length} backbone + ${live.length} probe · ${(liveState / 1e6).toFixed(0)} MB state`
            : "—"}</strong></div>
          <div><Gauge size={15} /> Latency <strong>{latency.compute
            ? `${latency.compute.toFixed(0)} ms compute + ${latency.chunkMs.toFixed(0)} ms chunk · RTF ${rtf.toFixed(2)}`
            : "—"}</strong></div>
          {engine.info?.threads > 1 && <div><AudioWaveform size={15} /> wasm <strong>{engine.info.threads} threads</strong></div>}
        </footer>
      </section>
    </main>
  );
}

function App() {
  const [page, setPage] = useState("home");
  return page === "home"
    ? <Home onChoose={setPage} />
    : <Demo page={page} onHome={() => setPage("home")} />;
}

createRoot(document.getElementById("root")).render(<App />);
