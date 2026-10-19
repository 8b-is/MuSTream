/**
 * Build the demo as a static site that fetches its graphs from Hugging Face,
 * ready to upload as a static Hugging Face Space:
 *
 *   VITE_MODEL_REPO=yamaha-research/MuSTream npm run build:space
 *   hf upload yamaha-research/MuSTream dist . --repo-type space
 *
 * The page is then at https://yamaha-research-mustream.static.hf.space (link
 * that, not huggingface.co/spaces/..., whose iframe wrapper is not cross-origin
 * isolated and so runs the wasm backend on one thread). Optional: VITE_MODEL_REVISION
 * pins the model repo to a commit, see src/modelSource.js.
 *
 * Nothing here needs a checkpoint, so it runs on any machine with Node.
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { build } from "vite";

const DEMO = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const repo = process.env.VITE_MODEL_REPO;
if (!repo || !/^[\w.-]+\/[\w.-]+$/.test(repo)) {
  console.error("[space] set VITE_MODEL_REPO=<org>/<model-repo>, the repo holding onnx/");
  process.exit(1);
}

await build({ root: DEMO, mode: "production" });

// The Space's config. custom_headers is how a static Space gets the COOP/COEP
// pair vite.config.js sets locally; the graphs come from huggingface.co with
// CORS headers, which is all require-corp asks of a fetch().
writeFileSync(path.join(DEMO, "dist", "README.md"), `---
title: MuSTream Streaming Demo
emoji: 🎵
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
custom_headers:
  cross-origin-opener-policy: same-origin
  cross-origin-embedder-policy: require-corp
  cross-origin-resource-policy: cross-origin
---

Real-time chord recognition, beat tracking and key detection, running in your
browser with onnxruntime-web. Graphs: https://huggingface.co/${repo}
`);
console.log(`[space] dist/ is ready for a static Space (graphs from ${repo})`);
