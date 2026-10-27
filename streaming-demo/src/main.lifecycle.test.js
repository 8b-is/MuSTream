import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";

// Exercise the actual Demo controller with deterministic hooks and deferred I/O.
// This does not render JSX or validate browser audio/model execution.
const source = readFileSync(new URL("./main.jsx", import.meta.url), "utf8");
const begin = source.indexOf("function Demo({ page, onHome }) {");
const end = source.indexOf("  const progress = duration", begin);
assert.ok(begin >= 0 && end > begin, "Demo controller boundary changed; review fixture");
const controller = source.slice(begin, end) + `
  return { loadFile, playSample, startDemo, stopDemo, pump,
    file, status, playing, streaming, autoStart, segments,
    samplesRef, decodedRef, cursorRef, busyRef, endedRef, pumpRef, audioRef };
}`;
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
};
const flush = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };

function fixture() {
  const hooks = [], effects = [], timers = new Map(), decodes = [], chunks = [], resets = [], finishes = [];
  let slot = 0, timerId = 0, view;
  const info = { tasks: ["chord"], sampleRate: 20, chunkSamples: 4, chunkMs: 200,
    heads: { chord: { frameDuration: 0.2 } }, warnings: [] };
  const audio = { currentTime: 0, paused: true, pause() { this.paused = true; },
    async play() { this.paused = false; } };
  const client = { info,
    pushChunk(audio, real) { const d = deferred(); chunks.push({ ...d, audio, real }); return d.promise; },
    reset() { const d = deferred(); resets.push(d); return d.promise; },
    finish() { const d = deferred(); finishes.push(d); return d.promise; },
  };
  const context = vm.createContext({
    useRef(value) { const i = slot++; return hooks[i] ??= { current: value }; },
    useState(value) { const i = slot++; hooks[i] ??= { value }; return [hooks[i].value,
      next => { hooks[i].value = typeof next === "function" ? next(hooks[i].value) : next; }]; },
    useEffect(fn, deps) { const i = slot++; const old = hooks[i];
      if (!old || deps.some((d, n) => !Object.is(d, old.deps[n])))
        effects.push(() => { old?.cleanup?.(); hooks[i] = { deps, cleanup: fn() }; }); },
    useCallback: fn => fn, useStickyScroll() {},
    emptyBeat: () => ({}), PAGES: { chord: { tasks: ["chord"] } },
    stream: () => client,
    decodeToModelRate(file, rate) { const d = deferred(); decodes.push({ ...d, file, rate }); return d.promise; },
    URL: { createObjectURL: f => `blob:${f.name}`, revokeObjectURL() {} },
    setInterval(fn) { timers.set(++timerId, fn); return timerId; },
    clearInterval(id) { timers.delete(id); },
    sampleUrl: "/sample.wav", fetch: async () => { throw new Error("unexpected fetch"); },
    File: class { constructor(parts, name) { this.name = name; } },
    appendLabel: (prev, label) => [...prev, label], median: () => 1,
    console: { error() {}, warn() {} }, Float32Array,
  });
  vm.runInContext(controller, context);
  const render = () => { slot = 0; view = context.Demo({ page: "chord", onHome() {} });
    view.audioRef.current = audio; while (effects.length) effects.shift()(); return view; };
  render(); render();
  return { render, audio, client, context, decodes, chunks, resets, finishes, timers,
    unmount() { for (const h of hooks) h?.cleanup?.(); },
    async load(name = "A") { const promise = render().loadFile({ name });
      decodes.at(-1).resolve({ samples: new Float32Array(8) }); await promise; return render(); },
    async start() { const promise = render().startDemo(); await flush();
      resets.at(-1).resolve({}); await promise; return render(); },
  };
}
const output = label => ({ ms: 1, chord: { labels: [{ label }] } });

test("latest selection keeps its decoded samples and cache after an older success", async () => {
  const f = fixture(); const a = f.render().loadFile({ name: "A" });
  const b = f.render().loadFile({ name: "B" });
  const samples = new Float32Array([2]); f.decodes[1].resolve({ samples }); await b;
  f.decodes[0].resolve({ samples: new Float32Array([1]) }); await a;
  const v = f.render(); assert.equal(v.file.name, "B"); assert.equal(v.samplesRef.current, samples);
  assert.equal(v.decodedRef.current.file.name, "B"); assert.equal(v.status, "Ready");
});

test("old decode failure cannot replace the current success status", async () => {
  const f = fixture(); const a = f.render().loadFile({ name: "A" });
  const b = f.render().loadFile({ name: "B" });
  f.decodes[1].resolve({ samples: new Float32Array(8) }); await b;
  f.decodes[0].reject(new Error("old file failed")); await a;
  assert.equal(f.render().status, "Ready");
});

for (const action of ["replace", "stop", "unmount"]) {
  test(`in-flight chunk cannot change output or cursor after ${action}`, async () => {
    const f = fixture(); await f.load(); await f.start(); f.audio.currentTime = 0.2;
    const pumping = f.render().pump();
    if (action === "replace") await f.load("B");
    else if (action === "stop") f.render().stopDemo();
    else f.unmount();
    f.chunks[0].resolve(output("old")); await pumping;
    const v = f.render(); assert.equal(v.cursorRef.current, 0); assert.equal(v.segments.length, 0);
    assert.equal(f.timers.size, 0); assert.equal(v.busyRef.current, false);
  });
}

test("old chunk failure does not clear the new run's timer or status", async () => {
  const f = fixture(); await f.load(); await f.start(); f.audio.currentTime = 0.2;
  const pumping = f.render().pump(); await f.load("B"); await f.start();
  f.chunks[0].reject(new Error("old inference failed")); await pumping;
  assert.equal(f.render().status, "Streaming"); assert.equal(f.timers.size, 1);
  f.audio.currentTime = 0.2;
  const current = f.render().pump(); f.chunks[1].resolve(output("new")); await current;
  assert.equal(f.render().segments[0].label, "new");
});

test("old finish completion cannot finish a replacement run", async () => {
  const f = fixture(); await f.load(); await f.start();
  const v = f.render(); v.cursorRef.current = 2; v.endedRef.current = true;
  const pumping = v.pump(); await f.load("B"); await f.start();
  f.finishes[0].resolve(output("old tail")); await pumping;
  assert.equal(f.render().status, "Streaming"); assert.equal(f.render().segments.length, 0);
  assert.equal(f.timers.size, 1);
});

test("cancelled start after reset cannot play or schedule inference", async () => {
  const f = fixture(); await f.load(); const starting = f.render().startDemo(); await flush();
  f.render().stopDemo(); f.resets[0].resolve({}); await starting;
  assert.equal(f.audio.paused, true); assert.equal(f.timers.size, 0);
  assert.equal(f.render().status, "Stopped");
});

test("late play completion after stop cannot revive UI or the interval", async () => {
  const f = fixture(); await f.load(); const play = deferred(); f.audio.play = () => play.promise;
  const starting = f.render().startDemo(); await flush(); f.resets[0].resolve({}); await flush();
  f.render().stopDemo(); play.resolve(); await starting;
  assert.equal(f.render().playing, false); assert.equal(f.render().status, "Stopped");
  assert.equal(f.timers.size, 0);
});

test("repeated Start keeps only the latest reset continuation", async () => {
  const f = fixture(); await f.load(); const a = f.render().startDemo(); await flush();
  const b = f.render().startDemo(); await flush();
  f.resets[0].resolve({}); await a; assert.equal(f.timers.size, 0);
  f.resets[1].resolve({}); await b; assert.equal(f.timers.size, 1);
});

test("late sample fetch cannot replace a user upload", async () => {
  const f = fixture(); const fetched = deferred(); f.context.fetch = () => fetched.promise;
  const sample = f.render().playSample(); await f.load("upload");
  fetched.resolve({ ok: true, blob: async () => ({}) }); await sample;
  assert.equal(f.render().file.name, "upload"); assert.equal(f.render().autoStart, false);
  assert.equal(f.decodes.length, 1);
});

test("sample blob completion after an upload cannot auto-start the sample", async () => {
  const f = fixture(); const blob = deferred();
  f.context.fetch = async () => ({ ok: true, blob: () => blob.promise });
  const sample = f.render().playSample(); await flush(); await f.load("upload");
  blob.resolve({}); await sample;
  assert.equal(f.render().file.name, "upload"); assert.equal(f.render().autoStart, false);
});

test("current sample still loads and starts after decoding", async () => {
  const f = fixture(); f.context.fetch = async () => ({ ok: true, blob: async () => ({}) });
  const sample = f.render().playSample(); await flush();
  f.decodes[0].resolve({ samples: new Float32Array(8) }); await sample;
  f.render(); await flush(); f.resets[0].resolve({}); await flush();
  assert.equal(f.render().file.name, "test.wav"); assert.equal(f.render().status, "Streaming");
  assert.equal(f.timers.size, 1);
});

test("a stale scheduled callback cannot feed a replacement track", async () => {
  const f = fixture(); await f.load(); await f.start(); const oldCallback = [...f.timers.values()][0];
  await f.load("B"); await f.start(); f.audio.currentTime = 0.2;
  await oldCallback(); assert.equal(f.chunks.length, 0);
});

test("a pending start decode cannot overwrite a newer selection", async () => {
  const f = fixture(); await f.load(); f.render().decodedRef.current = {};
  const starting = f.render().startDemo(); await f.load("B");
  f.decodes[1].resolve({ samples: new Float32Array([1]) }); await starting;
  assert.equal(f.render().file.name, "B"); assert.equal(f.render().samplesRef.current.length, 8);
  assert.equal(f.resets.length, 0);
});

test("unmount cancels a pending decode", async () => {
  const f = fixture(); const loading = f.render().loadFile({ name: "A" }); f.unmount();
  f.decodes[0].resolve({ samples: new Float32Array(8) }); await loading;
  assert.equal(f.render().samplesRef.current, null);
});

test("normal chunks and tail still finish the current run", async () => {
  const f = fixture(); await f.load(); await f.start(); const v = f.render(); v.endedRef.current = true;
  const pumping = v.pump(); f.chunks[0].resolve(output("one")); await flush();
  f.chunks[1].resolve(output("two")); await flush(); f.finishes[0].resolve(output("tail")); await pumping;
  const done = f.render(); assert.equal(done.cursorRef.current, 2); assert.equal(done.segments.length, 3);
  assert.equal(done.status, "Finished"); assert.equal(f.timers.size, 0);
});

test("current reset and play failures are handled and allow retry", async () => {
  const f = fixture(); await f.load(); const badReset = f.render().startDemo(); await flush();
  f.resets[0].reject(new Error("reset failed")); await badReset;
  assert.equal(f.render().status, "Could not reset the stream");
  f.audio.play = async () => { throw new Error("autoplay refused"); };
  const badPlay = f.render().startDemo(); await flush(); f.resets[1].resolve({}); await badPlay;
  assert.equal(f.render().status, "Audio loaded: press Play to try again");
  assert.equal(f.timers.size, 0);
});
