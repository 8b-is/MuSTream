/**
 * The inference thread: onnxruntime-web plus the graphs the chosen tasks need,
 * driven one 0.2 s chunk at a time.
 *
 * It lives in a worker for two reasons -- a backbone is ~100 MB to fetch and
 * compile, and each chunk is a few tens of ms of wasm that would otherwise land
 * on the frame the UI is trying to paint.
 *
 * Protocol (all messages carry the caller's `seq` back):
 *   in   {type:"load", tasks, config, model}
 *                                      -> {type:"progress"} * n, {type:"ready", info}
 *        {type:"configure", config}    -> {type:"ready", info}
 *        {type:"reset"}                -> {type:"ok"}
 *        {type:"chunk", audio, real}   -> {type:"result", ms, chord?, beat?, key?}
 *        {type:"finish"}               -> {type:"result", ...}  the tail flush
 *   any failure comes back as {type:"error", message}.
 *
 * `load` is incremental and idempotent: asking for tasks already up is free,
 * and asking for a new one fetches only the graphs it adds -- moving from Chord
 * to All tasks pays for the beat backbone and two probes, not for everything
 * again.
 *
 * Requests are serialised through one promise chain: every graph carries state
 * that is only meaningful if chunks are fed in order, so two overlapping run()
 * calls would corrupt the session rather than merely race.
 */
import * as ort from "onnxruntime-web";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import mjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import { MusicEngine } from "./engine.js";
import { FIX_HINT, fetchManifest } from "./modelSource.js";

// Both halves of the wasm backend must be named explicitly: ORT would otherwise
// resolve them against its own script URL, which is a hashed asset after a
// build, and the bundled loader fails outright ("K is not a function") if it is
// given the .wasm without its .mjs.
ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl };
// Threads need SharedArrayBuffer, i.e. the COOP/COEP headers vite.config.js
// sets. Without them ORT would throw on spawn, so ask for what we can have.
ort.env.wasm.numThreads = globalThis.crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 1)
  : 1;

// Absolute: fetch() in a worker resolves relative URLs against the worker
// script, which lives in /src (dev) or /assets (build), not the site root.
const MODELS = "/models";       // served from public/, see scripts/link-models.mjs
const graphUrl = (file) => (/^https?:/.test(file) ? file : `${MODELS}/${file}`);
const CACHE = "mustream-graphs-v1";
const SESSION = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };

let manifest = null;
let engine = null;
// Decoder settings, remembered here as well as in the engine so a task loaded later is
// built with the choice already made rather than defaulting and being reconfigured.
let config = {};
const loaded = new Set();       // graph aliases already compiled into `engine`
let chain = Promise.resolve();

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());

  const chunks = [];
  let loadedBytes = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loadedBytes += value.length;
    onProgress(loadedBytes, total);
  }
  const out = new Uint8Array(loadedBytes);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  return out;
}

/**
 * A remote graph, from the Cache API when this browser has fetched it before.
 *
 * The HTTP cache cannot do this for Hugging Face: every download is answered
 * with a freshly signed CDN redirect, so the bytes never come back under the
 * same URL twice and a returning visitor would pay the full download again.
 * Local graphs are same-origin and left to the HTTP cache.
 */
async function fetchGraph(url, onProgress) {
  const remote = /^https?:/.test(url) && globalThis.caches;
  const cache = remote ? await caches.open(CACHE).catch(() => null) : null;
  const hit = await cache?.match(url).catch(() => null);
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    onProgress(bytes.length, bytes.length);
    return bytes;
  }
  const bytes = await fetchWithProgress(url, onProgress);
  // A Blob copies now, so the graph handed to ORT is not shared with the cache
  // write; a full quota only means fetching again next visit.
  cache?.put(url, new Response(new Blob([bytes]))).catch(() => {});
  return bytes;
}

async function getManifest(model) {
  manifest ||= await fetchManifest(model);
  return manifest;
}

/**
 * Bring up whatever the requested tasks still need.
 *
 * The manifest is the authority on which backbone a task's probe belongs to,
 * because that pairing is not guessable: chord and key read the r0 backbone,
 * beat reads r15. Tasks naming the same backbone resolve to one alias here and
 * therefore to one fetch, one compile and one run per chunk.
 */
async function load(wanted, wantedConfig, model) {
  if (wantedConfig) config = { ...config, ...wantedConfig };
  const m = await getManifest(model);
  const tasks = {};
  for (const task of wanted) {
    if (!m.tasks[task]) continue;               // no head exported for it
    if (engine?.heads.has(task)) continue;      // already up
    tasks[task] = m.tasks[task];
  }
  if (!Object.keys(tasks).length) {
    if (engine && wantedConfig) engine.configure(wantedConfig);
    if (!engine) throw new Error(
      `no exported head for ${wanted.join(", ")} -- the ${m.model} graphs have ` +
      `heads for ${Object.keys(m.tasks).join(", ") || "nothing"} (${FIX_HINT})`);
    return info();
  }

  // Fetch each graph once, with one progress bar over the bytes still missing.
  const needed = [];
  for (const pair of Object.values(tasks))
    for (const alias of [pair.backbone, pair.probe])
      if (!loaded.has(alias) && !needed.includes(alias)) needed.push(alias);

  const got = Object.fromEntries(needed.map((a) => [a, 0]));
  const totals = Object.fromEntries(needed.map((a) => [a, m.graphs[a].bytes || 0]));
  const report = (alias) => (bytes, total) => {
    got[alias] = bytes;
    totals[alias] = total;
    const sum = (o) => Object.values(o).reduce((n, v) => n + v, 0);
    self.postMessage({ type: "progress", loaded: sum(got), total: sum(totals) });
  };

  const graphs = {};
  await Promise.all(needed.map(async (alias) => {
    const [model, spec] = await Promise.all([
      fetchGraph(graphUrl(m.graphs[alias].file), report(alias)),
      // A remote manifest carries its specs; a linked one points at them.
      m.graphs[alias].spec ?? fetch(`${MODELS}/${alias}.json`).then((r) => {
        if (!r.ok) throw new Error(`${alias}.json: HTTP ${r.status}`);
        return r.json();
      }),
    ]);
    graphs[alias] = { model, spec };
  }));
  self.postMessage({ type: "progress", loaded: 1, total: 1, compiling: true });

  engine ||= new MusicEngine(SESSION, config);
  await engine.addTasks({ graphs, tasks, config });
  for (const alias of needed) loaded.add(alias);
  return info();
}

/** The parts of `info` that come from the session rather than the engine. */
function describeSession() {
  return {
    model: manifest.model,
    available: Object.keys(manifest.tasks),
    missing: manifest.missing || [],
    threads: ort.env.wasm.numThreads,
    bytes: Object.fromEntries([...loaded].map((a) => [a, manifest.graphs[a].bytes])),
  };
}

function info() {
  return { ...engine.describe(), ...describeSession() };
}

async function handle(msg) {
  switch (msg.type) {
    case "load":
      return { type: "ready", info: await load(msg.tasks, msg.config, msg.model) };
    case "configure":
      config = { ...config, ...msg.config };
      return { type: "ready", info: { ...engine.configure(msg.config), ...describeSession() } };
    case "reset":
      engine.reset();
      return { type: "ok" };
    case "chunk":
      return { type: "result", ...(await engine.pushChunk(msg.audio, msg.real)) };
    case "finish":
      return { type: "result", ...engine.finish() };
    default:
      throw new Error(`unknown message: ${msg.type}`);
  }
}

self.onmessage = ({ data }) => {
  chain = chain.then(async () => {
    try {
      self.postMessage({ seq: data.seq, ...(await handle(data)) });
    } catch (err) {
      self.postMessage({ seq: data.seq, type: "error", message: String(err?.message || err) });
    }
  });
};
