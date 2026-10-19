/**
 * Drive src/engine.js from Node, so the browser's inference path can be checked
 * against real audio -- and against run_chord_onnx.py / run_beat_onnx.py /
 * run_key_onnx.py -- without a browser. Same engine, same graphs, same chunk
 * protocol as the demo page; only the audio source differs (ffmpeg here, Web
 * Audio there).
 *
 *   node scripts/verify.mjs <audio> [--tasks chord,beat,key] [--chunks N] [--json]
 *                           [--lab chord.lab] [--beats beat.lab] [--key C.key]
 *                           [--beat-postproc dbn_blocksync|minimal] [--beat-meter 3|4]
 *                           [--chord-self-bias <nats>|graph]
 *
 * `--chord-self-bias graph` is the one to use when checking against
 * run_chord_onnx.py: the demo's chord default decodes in the host at a HIGHER
 * lambda than the graphs carry (src/chordCrf.js says why), so without it the two
 * are being asked to agree about different decoders.
 *
 * With a ground truth for a task it scores it the way that task is scored:
 * frame accuracy for chord, F-measure at 70 ms for beat, exact match for key.
 */
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { createEngine } from "../src/engine.js";
import { keyConvergence } from "../src/streamingClient.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MODELS = path.join(HERE, "..", "public", "models");

const argv = process.argv.slice(2);
const flag = (name, fallback = null) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? fallback : argv[i + 1];
};
const has = (name) => argv.includes(`--${name}`);
const VALUED = new Set(["tasks", "chunks", "lab", "beats", "key", "beat-postproc", "beat-meter",
                        "chord-self-bias"]);
const positional = argv.filter((a, i) => !a.startsWith("--") &&
                                         !(argv[i - 1]?.startsWith("--") && VALUED.has(argv[i - 1].slice(2))));
const audioPath = positional[0];
if (!audioPath) {
  console.error("usage: node scripts/verify.mjs <audio> [--tasks chord,beat,key] [--chunks N] " +
                "[--lab chord.lab] [--beats beat.lab] [--key file.key] [--json]");
  process.exit(2);
}
const limit = Number(flag("chunks", Infinity));

const manifest = JSON.parse(readFileSync(path.join(MODELS, "models.json"), "utf8"));
const wanted = (flag("tasks") || Object.keys(manifest.tasks).join(",")).split(",").filter(Boolean);
const tasks = {};
for (const task of wanted) {
  if (!manifest.tasks[task]) {
    console.error(`[pipe] no exported head for ${task}; the manifest has ` +
                  `${Object.keys(manifest.tasks).join(", ")}`);
    process.exit(2);
  }
  tasks[task] = manifest.tasks[task];
}

const graphs = {};
for (const pair of Object.values(tasks))
  for (const alias of [pair.backbone, pair.probe])
    graphs[alias] ??= {
      model: readFileSync(path.join(MODELS, manifest.graphs[alias].file)),
      spec: JSON.parse(readFileSync(path.join(MODELS, `${alias}.json`), "utf8")),
    };

function decode(file, sampleRate) {
  const out = spawnSync("ffmpeg", ["-v", "error", "-i", file, "-f", "f32le",
                                   "-ac", "1", "-ar", String(sampleRate), "-"],
                        { maxBuffer: 1 << 30, encoding: "buffer" });
  if (out.status !== 0) throw new Error(`ffmpeg failed: ${out.stderr}`);
  const b = out.stdout;
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}

const beatMeter = flag("beat-meter");
// The demo decodes chords in the host at its own lambda (see src/chordCrf.js),
// which is deliberately NOT the lambda the graph was exported with. Comparing
// this file against run_chord_onnx.py therefore has to say `--chord-self-bias
// graph`, which hands the decode back to the graph's own CRF; any number
// decodes here instead.
const chordBias = flag("chord-self-bias");
const engine = await createEngine({
  graphs, tasks,
  config: { beat: { postproc: flag("beat-postproc") ?? undefined,
                    beatsPerBar: beatMeter ? Number(beatMeter) : null },
            chord: { selfBias: chordBias == null ? undefined
                              : chordBias === "graph" ? null : Number(chordBias) } },
});
const info = engine.describe();
console.log(`[pipe] ${manifest.model}: ${info.tasks.join(", ")} over ` +
            `${Object.keys(info.backbones).length} backbone(s), ` +
            `${info.params.toFixed(1)}M params total`);
for (const [alias, b] of Object.entries(info.backbones))
  console.log(`[pipe]   ${alias}: ${(b.params).toFixed(1)}M params, context ${b.context} frames`);
for (const [task, h] of Object.entries(info.heads))
  console.log(`[pipe]   ${task}: ${h.params.toFixed(2)}M params, ${h.labelsPerChunk} frames/chunk ` +
              `at ${h.labelRate} fps on ${h.backbone}` +
              (h.decoder && h.decoder !== "none" ? `, decoder=${h.decoder}` : "") +
              (h.postproc ? `, postproc=${h.postproc}` : "") +
              (h.postproc === "dbn_blocksync"
                ? ` (block ${h.block}, meter ${h.beatsPerBar ?? "causal"})` : ""));
console.log(`[pipe] carried state ${(info.stateBytes / 1e6).toFixed(0)} MB (O(1) per chunk)`);
for (const w of info.warnings) console.warn(`[warn] ${w}`);

const audio = decode(audioPath, info.sampleRate);
console.log(`[pipe] ${audioPath}: ${(audio.length / info.sampleRate).toFixed(1)}s at ${info.sampleRate} Hz`);

const chunk = info.chunkSamples;
const nChunks = Math.min(limit, Math.ceil(audio.length / chunk));
const chords = [];
const beats = [];
const downbeats = [];
const keyHistory = [];
const times = [];
let tempo = null;
let downbeatPeriod = null;
let meter = null;
const beatNumbers = [];

for (let i = 0; i < nChunks; i++) {
  const slice = audio.subarray(i * chunk, Math.min((i + 1) * chunk, audio.length));
  const padded = slice.length === chunk ? slice : Object.assign(new Float32Array(chunk), slice);
  const out = await engine.pushChunk(padded, slice.length);
  times.push(out.ms);
  if (out.chord) for (const l of out.chord.labels) chords.push(l);
  if (out.beat) {
    beats.push(...out.beat.beats);
    downbeats.push(...out.beat.downbeats);
    tempo = out.beat.tempo ?? tempo;
    downbeatPeriod = out.beat.downbeatPeriod ?? downbeatPeriod;
    meter = out.beat.meter ?? meter;
    for (const e of out.beat.events) beatNumbers.push(e.beatNumber);
  }
  if (out.key?.estimate) keyHistory.push(out.key.estimate);
}
const tail = engine.finish();
if (tail.beat) {
  beats.push(...tail.beat.beats);
  downbeats.push(...tail.beat.downbeats);
  for (const e of tail.beat.events) beatNumbers.push(e.beatNumber);
}

times.sort((a, b) => a - b);
const median = times[times.length >> 1];
const seconds = Math.min(audio.length, nChunks * chunk) / info.sampleRate;
console.log(`[pipe] ${seconds.toFixed(1)}s of audio; per chunk median ${median.toFixed(1)} ms, ` +
            `p90 ${times[Math.floor(times.length * 0.9)].toFixed(1)} ms, ` +
            `budget ${info.chunkMs.toFixed(0)} ms -> RTF ${(median / info.chunkMs).toFixed(2)}`);

function mergeFrames(labels, frameDuration) {
  const segments = [];
  for (const [i, l] of labels.entries()) {
    const last = segments[segments.length - 1];
    if (last && last.label === l) last.end = (i + 1) * frameDuration;
    else segments.push({ start: i * frameDuration, end: (i + 1) * frameDuration, label: l });
  }
  return segments;
}

const frameDuration = info.heads.chord?.frameDuration;
const segments = info.heads.chord ? mergeFrames(chords.map((c) => c.label), frameDuration) : [];

if (has("json")) {
  console.log(JSON.stringify({
    model: manifest.model,
    frameDuration,
    chords: chords.map((c) => c.label),
    segments,
    beats,
    downbeats,
    beatNumbers,
    tempo,
    meter,
    key: keyConvergence(keyHistory),
    keyCurve: keyHistory.map((k) => [Number(k.time.toFixed(4)), k.label]),
  }));
  process.exit(0);
}

if (info.heads.chord) {
  const c = info.heads.chord;
  console.log(`[chord] decoded by the ${c.decodedBy} at self_bias=` +
              `${c.selfBias ?? c.graphSelfBias} nats (graph carries ${c.graphSelfBias})`);
  console.log(`[chord] ${chords.length} labels, ${segments.length} segments; first few:`);
  for (const s of segments.slice(0, 6))
    console.log(`         ${s.start.toFixed(2).padStart(7)} - ${s.end.toFixed(2).padStart(7)}  ${s.label}`);
}
if (info.heads.beat) {
  console.log(`[beat]  ${beats.length} beats, ${downbeats.length} downbeats ` +
              `(median tempo ${tempo ? tempo.toFixed(1) : "n/a"} BPM` +
              (meter ? `, meter ${meter}/4` : `, downbeat every ${downbeatPeriod ?? "?"} beats`) + ")");
  // The DBN numbers every beat within its bar; the peak picker cannot, and prints times.
  console.log(`         first few: ` + (meter
    ? beats.slice(0, 8).map((t, i) => `${t.toFixed(2)}/${beatNumbers[i]}`).join(" ")
    : beats.slice(0, 8).map((t) => t.toFixed(2)).join(" ")));
}
if (info.heads.key) {
  const k = keyConvergence(keyHistory);
  if (k) {
    console.log(`[key]   ${k.label} (${(k.confidence * 100).toFixed(0)}% confident)`);
    console.log(`         first said it at ${k.tFirst.toFixed(1)}s, settled at ${k.tSettle.toFixed(1)}s ` +
                `of ${k.duration.toFixed(1)}s; ${k.flips} changes of mind ` +
                `(${k.flipsPerMin.toFixed(1)}/min)`);
  }
}

// ---------------------------------------------------------------------------
// scoring, when a ground truth is given
// ---------------------------------------------------------------------------

if (flag("lab") && info.heads.chord) {
  const truth = readFileSync(flag("lab"), "utf8").trim().split("\n").map((line) => {
    const [start, end, label] = line.trim().split(/\s+/);
    return { start: Number(start), end: Number(end), label };
  });
  let hit = 0, root = 0, scored = 0;
  for (const [i, c] of chords.entries()) {
    const t = (i + 0.5) * frameDuration;
    const ref = truth.find((s) => t >= s.start && t < s.end);
    if (!ref) continue;
    scored++;
    if (ref.label === c.label) hit++;
    if (ref.label.split(":")[0] === c.label.split(":")[0]) root++;
  }
  console.log(`[eval]  chord vs ${path.basename(flag("lab"))}: exact ${(100 * hit / scored).toFixed(1)}%, ` +
              `root ${(100 * root / scored).toFixed(1)}% over ${scored} frames`);
}

if (flag("beats") && info.heads.beat) {
  // F-measure at the standard 70 ms tolerance, greedily matched one-to-one --
  // the same window mir_eval.beat.f_measure uses.
  const ref = readFileSync(flag("beats"), "utf8").trim().split("\n")
    .map((l) => Number(l.trim().split(/\s+/)[0]))
    .filter((t) => Number.isFinite(t) && t <= seconds);
  const TOL = 0.07;
  const used = new Set();
  let tp = 0;
  for (const t of beats) {
    let best = -1, bestErr = Infinity;
    for (const [i, r] of ref.entries()) {
      const err = Math.abs(r - t);
      if (!used.has(i) && err <= TOL && err < bestErr) { best = i; bestErr = err; }
    }
    if (best >= 0) { used.add(best); tp++; }
  }
  const precision = beats.length ? tp / beats.length : 0;
  const recall = ref.length ? tp / ref.length : 0;
  const f = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  console.log(`[eval]  beat vs ${path.basename(flag("beats"))}: F ${(100 * f).toFixed(1)}% ` +
              `(P ${(100 * precision).toFixed(1)}%, R ${(100 * recall).toFixed(1)}%) ` +
              `over ${ref.length} reference beats at ${TOL * 1000} ms`);
}

if (flag("key") && info.heads.key) {
  const ref = readFileSync(flag("key"), "utf8").trim().split("\n")[0].trim();
  const got = keyHistory[keyHistory.length - 1]?.label;
  console.log(`[eval]  key vs ${path.basename(flag("key"))}: ${got === ref ? "correct" : "wrong"} ` +
              `(said ${got}, annotated ${ref})`);
}
