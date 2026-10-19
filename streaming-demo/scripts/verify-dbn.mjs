/**
 * Check src/beatDbn.js against madmom -- the thing it is a port OF -- rather than against
 * its own idea of what it should do.
 *
 *   node scripts/verify-dbn.mjs <activations.npy> [--ref beats.json] [--hmm dir]
 *                               [--block 15] [--meter 3|4]
 *
 * The activations are what `run_beat_onnx.py --activations DIR` writes: a [T, 2] float32
 * .npy of raw beat/downbeat LOGITS. With no other argument this just decodes them and
 * reports, which is useful on its own. The two comparisons need artifacts from a Python
 * environment that has madmom, and both are worth regenerating whenever this file or
 * postprocessor.py changes:
 *
 * --hmm DIR   compares the state space, the sparse transition model and the observation
 *             model, element by element, against madmom's own arrays. Export them with:
 *
 *   from madmom.features.downbeats import DBNDownBeatTrackingProcessor
 *   import numpy as np
 *   p = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0,
 *                                    fps=75, transition_lambda=100)
 *   for i, h in enumerate(p.hmms):
 *       tm = h.transition_model
 *       np.save(f"{DIR}/h{i}_states.npy", tm.states.astype(np.int64))
 *       np.save(f"{DIR}/h{i}_pointers.npy", tm.pointers.astype(np.int64))
 *       np.save(f"{DIR}/h{i}_logprob.npy", tm.log_probabilities.astype(np.float64))
 *       np.save(f"{DIR}/h{i}_ompointers.npy", h.observation_model.pointers.astype(np.int64))
 *       np.save(f"{DIR}/h{i}_positions.npy", tm.state_space.state_positions.astype(np.float64))
 *       np.save(f"{DIR}/h{i}_intervals.npy", tm.state_space.state_intervals.astype(np.int64))
 *
 *             Everything must match exactly EXCEPT the transition log probabilities,
 *             which are allowed a few ulps: they come from exp(), and V8's exp differs
 *             from numpy's by up to 1 ulp on these inputs. Measured, that is <= 1e-14 in
 *             log space, and it changes no decoded beat on any track tried -- Viterbi
 *             would have to be choosing between two paths within 1e-14 of each other.
 *
 * --ref FILE  compares the decoded beat times AND bar positions against the repo's own
 *             decode_blocksync. Produce it with beat_tracking/postprocessor.py:
 *
 *   tracker = FixedLagDBNDownBeatTracker(beats_per_bar=[3, 4], min_bpm=55.0,
 *                                        max_bpm=215.0, fps=75, transition_lambda=100)
 *   act = combined_activation(sigmoid(logits[:, 0]) * (1 - 1e-5) + 1e-5 / 2,
 *                             sigmoid(logits[:, 1]) * (1 - 1e-5) + 1e-5 / 2)
 *   out = tracker.decode_blocksync(act, BLOCK, hmm_index=None)   # or [3,4].index(meter)
 *   json.dump({"beats": out[:, 0].tolist(), "numbers": out[:, 1].astype(int).tolist()}, f)
 *
 *             This is the comparison that matters, and it is exact: same beats, same
 *             count, same number within the bar.
 */
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { BlockSyncDbn, buildBarHmm } from "../src/beatDbn.js";

const argv = process.argv.slice(2);
const flag = (name, fallback = null) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? fallback : argv[i + 1];
};
const VALUED = new Set(["ref", "hmm", "block", "meter", "fps"]);
const positional = argv.filter((a, i) => !a.startsWith("--") &&
                                         !(argv[i - 1]?.startsWith("--") && VALUED.has(argv[i - 1].slice(2))));
const actPath = positional[0];
if (!actPath) {
  console.error("usage: node scripts/verify-dbn.mjs <activations.npy> [--ref beats.json] " +
                "[--hmm dir] [--block 15] [--meter 3|4]");
  process.exit(2);
}
const fps = Number(flag("fps", 75));
const block = Number(flag("block", 15));
const meter = flag("meter") ? Number(flag("meter")) : null;

/** Just enough .npy to read the 1-D and 2-D arrays this script is handed. */
function readNpy(file) {
  const b = readFileSync(file);
  if (b.subarray(0, 6).toString("latin1") !== "\x93NUMPY") throw new Error(`${file}: not a .npy`);
  const headerLen = b.readUInt16LE(8);
  const header = b.subarray(10, 10 + headerLen).toString("latin1");
  const descr = header.match(/'descr': '([^']+)'/)[1];
  const shape = header.match(/'shape': \(([^)]*)\)/)[1]
    .split(",").map((x) => x.trim()).filter(Boolean).map(Number);
  const body = b.subarray(10 + headerLen);
  const of = { "<f4": Float32Array, "<f8": Float64Array, "<i8": BigInt64Array, "<i4": Int32Array }[descr];
  if (!of) throw new Error(`${file}: unsupported dtype ${descr}`);
  return { shape, data: new of(body.buffer, body.byteOffset, body.byteLength / of.BYTES_PER_ELEMENT) };
}

let failures = 0;

// ---------------------------------------------------------------------------
// the models
// ---------------------------------------------------------------------------

const hmmDir = flag("hmm");
if (hmmDir) {
  // The log probabilities inherit exp()'s last bit; everything else is integer or exact
  // arithmetic and must match bit for bit.
  const TOLERANCE = { log_probs: 1e-13 };
  console.log(`[hmm] against madmom's arrays in ${hmmDir}`);
  for (const [i, beatsPerBar] of [[0, 3], [1, 4]]) {
    const hmm = buildBarHmm({ beatsPerBar, fps });
    console.log(`[hmm] ${beatsPerBar} beats/bar: ${hmm.numStates} states, ` +
                `${hmm.numTransitions} transitions, ${hmm.intervals.length} tempi ` +
                `${hmm.intervals[0]}..${hmm.intervals[hmm.intervals.length - 1]} frames`);
    for (const [label, mine, file] of [
      ["states", hmm.states, "states"],
      ["pointers", hmm.pointers, "pointers"],
      ["log_probs", hmm.logProbs, "logprob"],
      ["om_pointers", hmm.omPointers, "ompointers"],
      ["positions", hmm.statePositions, "positions"],
      ["intervals", hmm.stateIntervals, "intervals"],
    ]) {
      const p = path.join(hmmDir, `h${i}_${file}.npy`);
      if (!existsSync(p)) { console.log(`        ${label.padEnd(12)} SKIPPED (no ${path.basename(p)})`); continue; }
      const ref = readNpy(p).data;
      const tol = TOLERANCE[label] ?? 0;
      if (mine.length !== ref.length) {
        console.log(`        ${label.padEnd(12)} LENGTH ${mine.length} vs ${ref.length}`);
        failures++;
        continue;
      }
      let worst = 0;
      for (let j = 0; j < ref.length; j++) {
        const r = typeof ref[j] === "bigint" ? Number(ref[j]) : ref[j];
        const d = Math.abs(mine[j] - r);
        if (d > worst) worst = d;
      }
      const ok = worst <= tol;
      if (!ok) failures++;
      console.log(`        ${label.padEnd(12)} n=${String(ref.length).padStart(6)}  ` +
                  (worst === 0 ? "identical"
                   : ok ? `within tolerance (max ${worst.toExponential(2)}, allowed ${tol.toExponential(0)})`
                        : `DIFFERS by ${worst.toExponential(3)}`));
    }
  }
}

// ---------------------------------------------------------------------------
// the decode
// ---------------------------------------------------------------------------

const { shape, data } = readNpy(actPath);
const T = shape[0];
if (shape[1] !== 2) throw new Error(`expected [T, 2] activations, got [${shape}]`);
console.log(`[dbn] ${path.basename(actPath)}: ${T} frames at ${fps} fps ` +
            `(${(T / fps).toFixed(1)}s), block ${block}, meter ${meter ?? "causal"}`);

const dbn = new BlockSyncDbn({ fps, block, beatsPerBar: meter });
const events = [];
const perBlock = [];
// Fed exactly as the browser feeds it: one block per forward call, nothing held back.
for (let s = 0; s < T; s += block) {
  const n = Math.min(block, T - s);
  const beat = new Float32Array(n);
  const down = new Float32Array(n);
  for (let i = 0; i < n; i++) { beat[i] = data[(s + i) * 2]; down[i] = data[(s + i) * 2 + 1]; }
  const t0 = performance.now();
  events.push(...dbn.push(beat, down).events);
  perBlock.push(performance.now() - t0);
}
events.push(...dbn.flush().events);

perBlock.sort((a, b) => a - b);
const budget = (block / fps) * 1000;
const downbeats = events.filter((e) => e.beatNumber === 1);
console.log(`[dbn] ${events.length} beats, ${downbeats.length} downbeats, meter ${dbn.meter}/4, ` +
            `median tempo ${dbn.tempo(999)?.toFixed(1) ?? "n/a"} BPM`);
console.log(`[dbn] per block median ${perBlock[perBlock.length >> 1].toFixed(2)} ms, ` +
            `p90 ${perBlock[Math.floor(perBlock.length * 0.9)].toFixed(2)} ms, ` +
            `budget ${budget.toFixed(0)} ms -> RTF ${(perBlock[perBlock.length >> 1] / budget).toFixed(3)}`);
console.log(`[dbn] first few: ${events.slice(0, 8).map((e) => `${e.time.toFixed(3)}/${e.beatNumber}`).join(" ")}`);

const refPath = flag("ref");
if (refPath) {
  const ref = JSON.parse(readFileSync(refPath, "utf8"));
  const sameCount = events.length === ref.beats.length;
  let worstTime = 0;
  let numbersOk = true;
  if (sameCount) {
    for (let i = 0; i < events.length; i++) {
      worstTime = Math.max(worstTime, Math.abs(events[i].time - ref.beats[i]));
      if (events[i].beatNumber !== ref.numbers[i]) numbersOk = false;
    }
  }
  const ok = sameCount && worstTime === 0 && numbersOk;
  if (!ok) failures++;
  console.log(`[ref] vs ${path.basename(refPath)}: ${ref.beats.length} beats — ` +
              (ok ? "IDENTICAL, times and bar positions both"
                  : `${sameCount ? "" : `COUNT ${events.length} vs ${ref.beats.length}; `}` +
                    `${worstTime ? `max time diff ${worstTime.toExponential(2)}; ` : ""}` +
                    `${numbersOk ? "" : "bar positions differ"}`));
}

process.exit(failures ? 1 : 0);
