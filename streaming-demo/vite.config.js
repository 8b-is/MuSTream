import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// COOP/COEP make SharedArrayBuffer available, which is what lets
// onnxruntime-web run its wasm backend multi-threaded. Without these the
// worker falls back to one thread (see infer.worker.js). A hosted build sets
// them through the host instead (scripts/build-space.mjs).
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig(({ mode }) => {
  // With the graphs fetched from Hugging Face (src/modelSource.js), public/
  // holds nothing the build needs -- only links to local graphs, which would
  // otherwise be copied into dist/ at hundreds of MB each.
  const remote = Boolean(loadEnv(mode, process.cwd(), "VITE_").VITE_MODEL_REPO);
  return {
    plugins: [react()],
    // ../docs holds the sample track main.jsx bundles; the dev server only
    // serves outside the demo's own folder when told to.
    server: { headers: isolation, fs: { allow: [".", "../docs"] } },
    preview: { headers: isolation },
    worker: { format: "es" },
    build: { copyPublicDir: !remote },
    // The wasm binary is fetched at runtime from the URL the worker sets, not
    // pre-bundled; leaving ORT out of dep optimisation keeps that resolution intact.
    optimizeDeps: { exclude: ["onnxruntime-web"] },
  };
});
