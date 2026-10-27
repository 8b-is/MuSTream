import assert from "node:assert/strict";
import { copyFileSync, existsSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, readlinkSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";

function fixture(t) {
  const root = mkdtempSync(path.join(os.tmpdir(), "mustream-links-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const demo = path.join(root, "streaming-demo");
  const script = path.join(demo, "scripts", "link-models.mjs");
  const out = path.join(demo, "public", "models");
  mkdirSync(path.dirname(script), { recursive: true });
  copyFileSync(new URL("./link-models.mjs", import.meta.url), script);
  const base = (size, task, family) => path.join(root, "mustream", "ckpt", "onnx", task, `streaming_${size}_${family}.streaming`);
  for (const size of ["26M", "95M"]) {
    for (const [task, family] of [["backbone", "r0"], ["backbone", "r15"], ["chord", "r0"], ["beat", "r15"], ["key", "r0"]]) {
      const file = base(size, task, family);
      mkdirSync(path.dirname(file), { recursive: true });
      writeFileSync(file + ".onnx", `synthetic ${size} ${task} ${family}`);
      writeFileSync(file + ".json", JSON.stringify({
        n_params: 1,
        [task === "backbone" ? "source_checkpoint" : "mert_align_checkpoint"]: `${size}-${family}`,
      }));
    }
  }
  const run = (...args) => spawnSync(process.execPath, [script, ...args], { encoding: "utf8", timeout: 10000 });
  const snapshot = () => Object.fromEntries(readdirSync(out).sort().map(name => {
    const file = path.join(out, name);
    return [name, { link: lstatSync(file).isSymbolicLink() ? readlinkSync(file) : null, content: readFileSync(file, "utf8") }];
  }));
  const manifest = () => JSON.parse(readFileSync(path.join(out, "models.json"), "utf8"));
  assert.equal(run("26M").status, 0);
  return { run, base, snapshot, manifest, out };
}

for (const [label, task, family, ext, change] of [
  ["missing second backbone", "backbone", "r15", ".onnx", "remove"],
  ["missing backbone metadata", "backbone", "r0", ".json", "remove"],
  ["malformed backbone metadata", "backbone", "r15", ".json", "invalid"],
  ["malformed last probe metadata", "key", "r0", ".json", "invalid"],
  ["null metadata", "key", "r0", ".json", "null"],
  ["array metadata", "key", "r0", ".json", "array"],
  ["mismatched last probe provenance", "key", "r0", ".json", "mismatch"],
  ["non-file graph", "backbone", "r15", ".onnx", "directory"],
]) {
  test(`rejects ${label} without changing the working model`, t => {
    const f = fixture(t);
    const before = f.snapshot();
    const file = f.base("95M", task, family) + ext;
    if (change === "remove" || change === "directory") rmSync(file);
    if (change === "directory") mkdirSync(file);
    if (change === "invalid") writeFileSync(file, "{bad json");
    if (change === "null") writeFileSync(file, "null");
    if (change === "array") writeFileSync(file, "[]");
    if (change === "mismatch") writeFileSync(file, JSON.stringify({ mert_align_checkpoint: "wrong-family" }));
    const result = f.run("95M");
    assert.equal(result.status, 1, result.stderr);
    assert.deepEqual(f.snapshot(), before);
  });
}

test("a rejected first selection creates no output directory or aliases", t => {
  const f = fixture(t);
  rmSync(f.out, { recursive: true });
  writeFileSync(f.base("95M", "key", "r0") + ".json", "{bad json");
  assert.equal(f.run("95M").status, 1);
  assert.equal(existsSync(f.out), false);
});

test("defers removal of an absent optional probe until all validation succeeds", t => {
  const f = fixture(t);
  const before = f.snapshot();
  rmSync(f.base("95M", "chord", "r0") + ".json");
  writeFileSync(f.base("95M", "key", "r0") + ".json", "{bad json");
  assert.equal(f.run("95M").status, 1);
  assert.deepEqual(f.snapshot(), before);
});

test("switches valid models and preserves the selection on a no-argument run", t => {
  const f = fixture(t);
  const result = f.run("95M");
  assert.equal(result.status, 0, result.stderr);
  assert.equal(f.manifest().model, "95M");
  assert.deepEqual(f.manifest().missing, []);
  for (const [name, entry] of Object.entries(f.snapshot())) {
    if (name === "models.json") continue;
    assert.match(entry.link, /streaming_95M_/);
  }
  const before = f.snapshot();
  assert.equal(f.run().status, 0);
  assert.deepEqual(f.snapshot(), before);
});

for (const ext of [".onnx", ".json"]) {
  test(`omits an optional probe missing ${ext} and removes both old aliases`, t => {
    const f = fixture(t);
    rmSync(f.base("95M", "beat", "r15") + ext);
    const result = f.run("95M");
    assert.equal(result.status, 0, result.stderr);
    const manifest = f.manifest();
    assert.equal(manifest.model, "95M");
    assert.deepEqual(manifest.missing, ["beat"]);
    assert.equal(manifest.tasks.beat, undefined);
    assert.equal(manifest.graphs["backbone-r15"], undefined);
    assert.equal(f.snapshot()["beat-probe.onnx"], undefined);
    assert.equal(f.snapshot()["beat-probe.json"], undefined);
  });
}

test("listing and unknown model requests preserve the working model", t => {
  const f = fixture(t);
  const before = f.snapshot();
  assert.equal(f.run("--list").status, 0);
  assert.equal(f.run("unknown").status, 1);
  assert.deepEqual(f.snapshot(), before);
});
