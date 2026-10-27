import assert from "node:assert/strict";
import { copyFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { test } from "node:test";

// Exercise the real engine and decoders with a tiny session double. No models,
// browser, package installation or WebAssembly compilation are needed.
async function fixture(t) {
  const dir = mkdtempSync(path.join(os.tmpdir(), "mustream-engine-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  writeFileSync(path.join(dir, "package.json"), '{"type":"module"}');
  for (const file of ["engine.js", "beatPostproc.js", "beatDbn.js", "chordCrf.js"])
    copyFileSync(new URL(file, import.meta.url), path.join(dir, file));
  const pkg = path.join(dir, "node_modules", "onnxruntime-web");
  mkdirSync(pkg, { recursive: true });
  writeFileSync(path.join(pkg, "package.json"), '{"type":"module","exports":"./index.js"}');
  writeFileSync(path.join(pkg, "index.js"), `
    export const sessions = [];
    export class Tensor {
      constructor(type, data, dims) { Object.assign(this, {type, data, dims}); }
    }
    export const InferenceSession = { async create(model) {
      if (model.fail) throw new Error("compile failed");
      const session = { model, outputNames: ["value", "cache_out"], released: 0,
        async release() { this.released++; if (model.releaseFail) throw new Error("release failed"); }
      };
      sessions.push(session);
      return session;
    }};
  `);
  const { MusicEngine } = await import(pathToFileURL(path.join(dir, "engine.js")));
  const { sessions } = await import(pathToFileURL(path.join(pkg, "index.js")));
  return { engine: new MusicEngine({}), sessions };
}

function graph(task = "backbone", rate = 24000) {
  const spec = {
    mode: "streaming", sample_rate: rate, chunk_samples: rate / 5,
    state: [{ name: "cache", output: "cache_out", shape: [1], dtype: "float32" }],
    n_params: 100, context: 1, feature_shape: [1, 1, 3, 2],
    audio_input: { name: "audio" }, features_input: { name: "features", shape: [1, 1, 3, 2] },
    label_frames: 3, pool_factor: 10, hop_size: 160, label_rate: 15,
    graph: task + "_probe", n_classes: 2, vocab_size: 2,
    tags_output: "tags", logits_output: "logits",
  };
  return { model: {}, spec };
}

async function loadKey(engine) {
  await engine.addTasks({ graphs: { shared: graph(), key: graph("key") },
    tasks: { key: { backbone: "shared", probe: "key" } } });
  engine.realSamples = 4800;
  engine.backbones.get("shared").state.cache.data[0] = 17;
  engine.heads.get("key").frames = 3;
}

function snapshot(engine) {
  return structuredClone({ info: engine.describe(), config: engine.config,
    realSamples: engine.realSamples,
    cache: [...engine.backbones].map(([name, b]) => [name, [...b.state.cache.data]]),
    frames: [...engine.heads].map(([name, h]) => [name, h.frames]) });
}

test("geometry rejection preserves existing tasks and cannot be bypassed by retry", async t => {
  const { engine, sessions } = await fixture(t);
  await loadKey(engine);
  const before = snapshot(engine);
  const request = { graphs: { other: graph("backbone", 48000), chord: graph("chord", 48000) },
    tasks: { chord: { backbone: "other", probe: "chord" } } };
  await assert.rejects(engine.addTasks(request), /chunk geometry/);
  assert.deepEqual(snapshot(engine), before);
  await assert.rejects(engine.addTasks(request), /chunk geometry/);
  assert.deepEqual(snapshot(engine), before);
  assert.ok(sessions.slice(2).every(s => s.released === 1));
  assert.ok(sessions.slice(0, 2).every(s => s.released === 0));
});

test("late probe compilation failure rolls back a whole batch and permits retry", async t => {
  const { engine, sessions } = await fixture(t);
  const before = snapshot(engine);
  const graphs = { shared: graph(), key: graph("key"), chord: graph("chord") };
  graphs.chord.model.fail = true;
  const request = { graphs, tasks: { key: { backbone: "shared", probe: "key" },
    chord: { backbone: "shared", probe: "chord" } }, config: { chord: { selfBias: 2 } } };
  await assert.rejects(engine.addTasks(request), /compile failed/);
  assert.deepEqual(snapshot(engine), before);
  assert.ok(sessions.every(s => s.released === 1));
  graphs.chord.model.fail = false;
  await engine.addTasks(request);
  assert.deepEqual(engine.describe().tasks, ["key", "chord"]);
  assert.equal(engine.backbones.size, 1);
});

test("decoder construction failure leaves config, warnings and working state intact", async t => {
  const { engine, sessions } = await fixture(t);
  await loadKey(engine);
  engine.backbones.get("shared").spec.source_checkpoint = "original";
  const before = snapshot(engine);
  const chord = graph("chord");
  chord.spec.mert_align_checkpoint = "different";
  await assert.rejects(engine.addTasks({ graphs: { chord },
    tasks: { chord: { backbone: "shared", probe: "chord" } },
    config: { chord: { selfBias: -1 } } }), /selfBias/);
  assert.deepEqual(snapshot(engine), before);
  assert.equal(sessions[2].released, 1);
  assert.equal(sessions[0].released, 0);
});

test("a session is released if its state metadata cannot construct a streamer", async t => {
  const { engine, sessions } = await fixture(t);
  const backbone = graph();
  backbone.spec.state[0].dtype = "invalid";
  await assert.rejects(engine.addTasks({ graphs: { shared: backbone, key: graph("key") },
    tasks: { key: { backbone: "shared", probe: "key" } } }), /unsupported state dtype/);
  assert.equal(engine.backbones.size, 0);
  assert.equal(sessions.length, 1);
  assert.equal(sessions[0].released, 1);
});

test("cleanup failure does not mask the original error or skip other releases", async t => {
  const { engine, sessions } = await fixture(t);
  const graphs = { shared: graph(), key: graph("key"), chord: graph("chord") };
  graphs.shared.model.releaseFail = true;
  graphs.chord.spec.graph = "wrong_probe";
  await assert.rejects(engine.addTasks({ graphs, tasks: {
    key: { backbone: "shared", probe: "key" }, chord: { backbone: "shared", probe: "chord" },
  } }), /expected a streaming chord_probe/);
  assert.ok(sessions.every(s => s.released === 1));
  assert.equal(engine.heads.size, 0);
});

test("successful additions share backbones, reset tracks, and remain idempotent", async t => {
  const { engine, sessions } = await fixture(t);
  await loadKey(engine);
  const shared = engine.backbones.get("shared");
  const request = { graphs: { chord: graph("chord") },
    tasks: { chord: { backbone: "shared", probe: "chord" } } };
  await engine.addTasks(request);
  assert.equal(engine.backbones.get("shared"), shared);
  assert.deepEqual(engine.describe().tasks, ["key", "chord"]);
  assert.equal(engine.realSamples, 0);
  assert.equal(shared.state.cache.data[0], 0);
  assert.equal(engine.heads.get("key").frames, 0);
  await engine.addTasks(request);
  assert.equal(sessions.length, 3);
  assert.ok(sessions.every(s => s.released === 0));
});

async function loadDecoders(engine, logitsOnly = false) {
  const chord = graph("chord");
  if (logitsOnly) delete chord.spec.tags_output;
  await engine.addTasks({ graphs: { shared: graph(), chord, beat: graph("beat") },
    tasks: { chord: { backbone: "shared", probe: "chord" }, beat: { backbone: "shared", probe: "beat" } },
    config: { chord: { selfBias: 2 }, beat: { postproc: "minimal", beatsPerBar: 3 } } });
  for (const head of engine.heads.values()) {
    head.frames = 9;
    head.probe.state.cache.data[0] = 11;
  }
}

test("rejected chord mode preserves decoder, settings and probe state", async t => {
  const { engine } = await fixture(t);
  await loadDecoders(engine, true);
  const before = snapshot(engine);
  const head = engine.heads.get("chord"), crf = head.crf, state = head.probe.state;
  assert.throws(() => engine.configure({ chord: { selfBias: null } }), /no CRF/);
  assert.deepEqual(snapshot(engine), before);
  assert.equal(engine.heads.get("chord"), head);
  assert.equal(head.crf, crf);
  assert.equal(head.probe.state, state);
  assert.equal(state.cache.data[0], 11);
  engine.configure({ chord: { selfBias: 3 } });
  assert.equal(engine.describe().heads.chord.selfBias, 3);
});

test("late configuration rejection cannot reset an earlier head", async t => {
  const { engine } = await fixture(t);
  await loadDecoders(engine);
  const before = snapshot(engine);
  const beat = engine.heads.get("beat"), tracker = beat.tracker, state = beat.probe.state;
  assert.throws(() => engine.configure({ beat: { beatsPerBar: 4 }, chord: { selfBias: -1 } }), /selfBias/);
  assert.deepEqual(snapshot(engine), before);
  assert.equal(engine.heads.get("beat"), beat);
  assert.equal(beat.tracker, tracker);
  assert.equal(beat.probe.state, state);
  assert.equal(state.cache.data[0], 11);
});

test("successful partial settings preserve other choices and share compiled sessions", async t => {
  const { engine, sessions } = await fixture(t);
  await loadDecoders(engine);
  const chord = engine.heads.get("chord"), beat = engine.heads.get("beat");
  const count = sessions.length;
  engine.configure({ beat: { beatsPerBar: 4 } });
  const updated = engine.heads.get("beat");
  assert.equal(updated.postproc, "minimal");
  assert.equal(updated.beatsPerBar, 4);
  assert.equal(updated.frames, 0);
  assert.equal(updated.probe.state.cache.data[0], 0);
  assert.equal(updated.probe.session, beat.probe.session);
  assert.equal(engine.heads.get("chord"), chord);
  assert.equal(chord.frames, 9);
  assert.equal(sessions.length, count);
  assert.ok(sessions.every(s => s.released === 0));
  engine.configure({ beat: { postproc: undefined, beatsPerBar: undefined } });
  assert.equal(engine.describe().heads.beat.postproc, "minimal");
  assert.equal(engine.describe().heads.beat.beatsPerBar, 4);
  assert.throws(() => engine.configure({ beat: null }), TypeError);
  assert.equal(engine.describe().heads.beat.beatsPerBar, 4);
  engine.configure({ chord: { selfBias: null } });
  assert.equal(engine.describe().heads.chord.decodedBy, "graph");
  engine.configure({ chord: { selfBias: 2 } });
  assert.equal(engine.describe().heads.chord.decodedBy, "host");
});
