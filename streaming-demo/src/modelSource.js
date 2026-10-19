/**
 * Where the graphs come from, and the manifest describing them.
 *
 * Two sources, chosen at build time:
 *
 *   local   (default)  `npm run dev`: scripts/link-models.mjs links one model's
 *                      graphs into public/models and writes models.json there.
 *   remote             VITE_MODEL_REPO=<org>/<repo>: the graphs are fetched from
 *                      that Hugging Face repo (laid out as its onnx/ tree, the
 *                      same tree `hf download` puts in mustream/ckpt),
 *                      so the built site is static and needs no checkpoint at
 *                      all -- see `npm run build:space`. `?model=95M` on the page
 *                      URL picks the student; the default is 26M.
 *
 * Both produce the same manifest, {model, tasks, graphs, missing}; a remote
 * graph's `file` is an absolute URL and its `spec` is already fetched.
 */

export const MODEL_REPO = import.meta.env.VITE_MODEL_REPO || "";
// Pin to a commit hash for a release: graphs are cached by URL in the browser,
// so a moving `main` could serve a returning visitor last month's weights.
const REVISION = import.meta.env.VITE_MODEL_REVISION || "main";

export const SIZES = ["26M", "95M", "330M"];
const DEFAULT_SIZE = "26M";

// Which backbone family each probe was distilled against -- the same pairing as
// PAIRING in scripts/link-models.mjs, which has the full story. engine.js
// re-checks it against the provenance both graphs record.
const PAIRING = { chord: "r0", beat: "r15", key: "r0" };
const alias = (family) => `backbone-${family}`;

/** The student this page should run: `?model=` when remote, whatever is linked otherwise. */
export function pickModel(search = globalThis.location?.search ?? "") {
  if (!MODEL_REPO) return null;
  const asked = new URLSearchParams(search).get("model");
  return SIZES.find((s) => s.toLowerCase() === asked?.toLowerCase()) ?? DEFAULT_SIZE;
}

/** What to tell someone whose graphs did not load. */
export const FIX_HINT = MODEL_REPO
  ? `check that huggingface.co/${MODEL_REPO} is reachable`
  : "run `npm run link-models`";

export async function fetchManifest(model) {
  if (MODEL_REPO) return remoteManifest(model || DEFAULT_SIZE);
  // Absolute: a worker resolves relative URLs against its own script, not the site root.
  const res = await fetch("/models/models.json");
  if (!res.ok) throw new Error(`models.json: HTTP ${res.status} -- ${FIX_HINT}`);
  return res.json();
}

async function remoteManifest(model) {
  const base = `https://huggingface.co/${MODEL_REPO}/resolve/${REVISION}/onnx`;
  const stem = (dir, family) => `${base}/${dir}/streaming_${model}_${family}.streaming`;
  // A 404 is a head this repo does not ship; anything else is a real failure.
  const spec = async (url) => {
    const res = await fetch(url);
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    return res.json();
  };

  const graphs = {};
  const tasks = {};
  const missing = [];
  const families = [...new Set(Object.values(PAIRING))];
  const [backbones, probes] = await Promise.all([
    Promise.all(families.map((f) => spec(stem("backbone", f) + ".json"))),
    Promise.all(Object.entries(PAIRING).map(([task, f]) => spec(stem(task, f) + ".json"))),
  ]);

  families.forEach((family, i) => {
    if (!backbones[i]) throw new Error(`${MODEL_REPO} has no ${model} ${family} backbone`);
    graphs[alias(family)] = {
      file: stem("backbone", family) + ".onnx",
      spec: backbones[i],
      bytes: 0,                      // learnt from content-length as it downloads
      params: backbones[i].n_params,
      source: backbones[i].source_checkpoint,
    };
  });
  Object.entries(PAIRING).forEach(([task, family], i) => {
    if (!probes[i]) { missing.push(task); return; }
    const name = `${task}-probe`;
    graphs[name] = {
      file: stem(task, family) + ".onnx",
      spec: probes[i],
      bytes: 0,
      params: probes[i].n_params,
      source: probes[i].mert_align_checkpoint,
    };
    tasks[task] = { backbone: alias(family), probe: name };
  });
  return { model, tasks, graphs, missing };
}
