/**
 * Expose one exported model's graphs to the dev server under stable short
 * names, plus a manifest saying which backbone each task's probe belongs to, so
 * the app fetches `models/chord-probe.onnx` whichever model is chosen.
 *
 *   node scripts/link-models.mjs            keep whatever is linked (default 26M)
 *   node scripts/link-models.mjs 330M       switch to the 330M student
 *   node scripts/link-models.mjs --list     show the models available
 *
 * `npm run dev` runs this with no argument, so it never clobbers a choice you
 * made. The .onnx files live in the repo's gitignored ckpt/ tree (up to 1.2 GB
 * each) and are symlinked rather than copied; vite serves public/ as-is.
 *
 * TWO BACKBONES PER MODEL, and that is the whole reason this script writes a
 * manifest instead of just linking files. The chord and key probes were
 * distilled against the `r0` streaming backbone, the beat probe against `r15`
 * -- different weights, so a probe run on the other family produces tensors
 * that line up perfectly and numbers that are quietly wrong. Both are linked,
 * chord and key are pointed at the one they share (so the browser fetches and
 * compiles it once, not twice), and the pairing is CHECKED here rather than
 * trusted: every probe records the distillation checkpoint it was trained on
 * and every backbone records the one it was exported from, so a mis-pairing is
 * a link-time error instead of a silently wrong demo.
 */
import { existsSync, lstatSync, mkdirSync, readFileSync, readlinkSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEMO = path.join(HERE, "..");
const CKPT = path.join(DEMO, "..", "mustream", "ckpt", "onnx");   // see mustream/README.md
const OUT = path.join(DEMO, "public", "models");
const B = (size, family) => path.join(CKPT, "backbone", `streaming_${size}_${family}.streaming`);
const P = (task, size, family) => path.join(CKPT, task, `streaming_${size}_${family}.streaming`);

// Streaming pairs only: an incremental backbone is what this demo drives.
// Chord and key run on the r0 backbone, beat on r15; every probe is paired with
// the backbone it was trained on (checked below).
const PAIRING = { chord: "r0", beat: "r15", key: "r0" };
const MODELS = Object.fromEntries(["26M", "95M", "330M"].map((size) => [size, {
  backbones: { r0: B(size, "r0"), r15: B(size, "r15") },
  tasks: Object.fromEntries(Object.entries(PAIRING).map(([task, family]) =>
    [task, { backbone: family, probe: P(task, size, family) }])),
}]));
const DEFAULT = "26M";
const MANIFEST = path.join(OUT, "models.json");

const alias = { r0: "backbone-r0", r15: "backbone-r15" };
const probeAlias = (task) => `${task}-probe`;
const spec = (base) => JSON.parse(readFileSync(base + ".json", "utf8"));

/** Which model the current links point at, if any. */
function active() {
  try {
    return JSON.parse(readFileSync(MANIFEST, "utf8")).model ?? null;
  } catch {
    return null;
  }
}

const arg = process.argv[2];
if (arg === "--list" || arg === "-l") {
  const current = active();
  for (const [name, model] of Object.entries(MODELS)) {
    const have = Object.values(model.backbones).every((b) => existsSync(b + ".onnx"));
    const bytes = have
      ? Object.values(model.backbones).reduce((n, b) => n + statSync(b + ".onnx").size, 0)
      : 0;
    const tasks = Object.entries(model.tasks)
      .filter(([, t]) => existsSync(t.probe + ".onnx"))
      .map(([task]) => task);
    console.log(`  ${name.padEnd(6)} ${(have ? `${(bytes / 1e6).toFixed(0)} MB` : "not exported").padStart(12)}` +
                `  ${tasks.join(",").padEnd(16)} ${name === current ? "<- linked" : ""}`);
  }
  process.exit(0);
}
if (arg && !MODELS[arg]) {
  console.error(`[models] unknown model "${arg}"; try one of: ${Object.keys(MODELS).join(", ")}`);
  process.exit(1);
}

const current = active();
const choice = arg || (current && MODELS[current] ? current : DEFAULT);
if (!arg && current === choice && existsSync(MANIFEST)) {
  // Still verify the links resolve: a moved or deleted ckpt/ leaves the
  // manifest describing graphs that 404 at load time, which is a far more
  // confusing failure than this one.
  const stale = Object.values(JSON.parse(readFileSync(MANIFEST, "utf8")).graphs)
    .filter((g) => !existsSync(path.join(OUT, g.file)));
  if (!stale.length) {
    console.log(`[models] ${choice} already linked (npm run link-models <name> to switch)`);
    process.exit(0);
  }
}

const model = MODELS[choice];
mkdirSync(OUT, { recursive: true });

function link(base, name) {
  for (const ext of [".onnx", ".json"]) {
    const src = base + ext;
    const dst = path.join(OUT, name + ext);
    if (!existsSync(src)) return false;
    try { lstatSync(dst); rmSync(dst, { force: true }); } catch {}
    symlinkSync(path.relative(OUT, src), dst);
  }
  return true;
}

const graphs = {};
const tasks = {};
const missing = [];

for (const [family, base] of Object.entries(model.backbones)) {
  if (!link(base, alias[family])) {
    console.error(`[models] missing backbone: ${path.relative(DEMO, base)}.onnx`);
    console.error("[models] export it first (export_mert_onnx.py)");
    process.exit(1);
  }
  const s = spec(base);
  graphs[alias[family]] = {
    file: `${alias[family]}.onnx`,
    bytes: statSync(base + ".onnx").size,
    params: s.n_params,
    source: s.source_checkpoint,
  };
  console.log(`[models] ${choice}: ${alias[family]} -> ${path.basename(base)}.onnx ` +
              `(${(statSync(base + ".onnx").size / 1e6).toFixed(1)} MB)`);
}

for (const [task, pair] of Object.entries(model.tasks)) {
  const name = probeAlias(task);
  if (!link(pair.probe, name)) {
    // A task whose head has not been exported is left out of the manifest, and
    // the UI says so rather than showing invented numbers.
    missing.push(task);
    for (const ext of [".onnx", ".json"]) {
      try { lstatSync(path.join(OUT, name + ext)); rmSync(path.join(OUT, name + ext), { force: true }); } catch {}
    }
    console.warn(`[models] ${task}: no probe exported (${path.relative(DEMO, pair.probe)}.onnx)`);
    continue;
  }
  const s = spec(pair.probe);
  const backbone = graphs[alias[pair.backbone]];
  // The check no shape comparison can make: a probe reads whatever tensor it is
  // handed, so the only thing that catches the r0/r15 families being crossed is
  // the provenance both graphs record.
  if (s.mert_align_checkpoint && backbone.source && s.mert_align_checkpoint !== backbone.source) {
    console.error(`[models] ${task}: probe was trained against\n[models]   ${s.mert_align_checkpoint}`);
    console.error(`[models] but ${alias[pair.backbone]} was exported from\n[models]   ${backbone.source}`);
    console.error("[models] these are different weights -- fix the pairing in this script");
    process.exit(1);
  }
  graphs[name] = {
    file: `${name}.onnx`,
    bytes: statSync(pair.probe + ".onnx").size,
    params: s.n_params,
    source: s.mert_align_checkpoint,
  };
  tasks[task] = { backbone: alias[pair.backbone], probe: name };
  console.log(`[models] ${choice}: ${name} -> ${path.basename(pair.probe)}.onnx ` +
              `(${(statSync(pair.probe + ".onnx").size / 1e6).toFixed(1)} MB, on ${alias[pair.backbone]})`);
}

// Only the backbones some linked task actually names: linking an unused one
// would have the browser fetch a graph nothing reads.
for (const name of Object.keys(graphs)) {
  if (name.startsWith("backbone-") && !Object.values(tasks).some((t) => t.backbone === name))
    delete graphs[name];
}

writeFileSync(MANIFEST, JSON.stringify({ model: choice, tasks, graphs, missing }, null, 2) + "\n");
console.log(`[models] -> ${path.relative(DEMO, MANIFEST)}: ${Object.keys(tasks).join(", ")}` +
            (missing.length ? ` (no head for ${missing.join(", ")})` : ""));
