import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { test } from "node:test";

// Run the actual serialized worker handler; replace only browser/model I/O.
function fixture() {
  const configs = [], replies = [];
  let failAdd = false;
  class Engine {
    heads = new Map();
    config = {};
    async addTasks({ tasks, config }) {
      configs.push(structuredClone(config));
      if (failAdd) { failAdd = false; throw new Error("compile failed"); }
      this.config = config;
      for (const task of Object.keys(tasks)) this.heads.set(task, {});
    }
    configure(config) {
      if (config?.chord?.selfBias < 0) throw new Error("invalid bias");
      this.config = { ...this.config, ...config };
      return this.describe();
    }
    describe() { return { tasks: [...this.heads.keys()] }; }
  }
  const manifest = { model: "synthetic", tasks: {
    chord: { backbone: "shared", probe: "chord" }, beat: { backbone: "shared", probe: "beat" }
  }, graphs: Object.fromEntries(["shared", "chord", "beat"].map(x => [x, { file: x, bytes: 1, spec: {} }])) };
  const context = vm.createContext({ MusicEngine: Engine, ort: { env: { wasm: {} } },
    wasmUrl: "synthetic.wasm", mjsUrl: "synthetic.mjs", FIX_HINT: "synthetic",
    fetchManifest: async () => manifest, Uint8Array,
    fetch: async () => ({ ok: true, headers: { get: () => null }, arrayBuffer: async () => new ArrayBuffer(1) }),
    self: { postMessage: data => replies.push(data) } });
  const source = readFileSync(new URL("./infer.worker.js", import.meta.url), "utf8")
    .replace(/^import .*;$/gm, "");
  vm.runInContext(source, context);
  let seq = 0;
  return { configs, failNextAdd: () => { failAdd = true; }, async send(message) {
    const id = ++seq;
    context.self.onmessage({ data: { ...message, seq: id } });
    await vm.runInContext("chain", context);
    return replies.find(x => x.seq === id);
  } };
}

test("rejected configure before load cannot poison future settings", async () => {
  const f = fixture();
  assert.equal((await f.send({ type: "configure", config: { chord: { selfBias: -1 } } })).type, "error");
  assert.equal((await f.send({ type: "load", tasks: ["chord"] })).type, "ready");
  assert.deepEqual(f.configs[0], {});
});

test("rejected configure and same-task load preserve retained settings", async () => {
  const f = fixture();
  await f.send({ type: "load", tasks: ["chord"], config: { chord: { selfBias: 2 } } });
  for (const type of ["configure", "load"])
    assert.equal((await f.send({ type, tasks: ["chord"], config: { chord: { selfBias: -1 } } })).type, "error");
  assert.equal((await f.send({ type: "load", tasks: ["beat"] })).type, "ready");
  assert.deepEqual(f.configs.at(-1), { chord: { selfBias: 2 } });
});

test("rejected task additions do not retain their decoder settings", async () => {
  const f = fixture();
  await f.send({ type: "load", tasks: ["chord"], config: { chord: { selfBias: 2 } } });
  f.failNextAdd();
  assert.equal((await f.send({ type: "load", tasks: ["beat"], config: { chord: { selfBias: 5 } } })).type, "error");
  assert.equal((await f.send({ type: "load", tasks: ["beat"] })).type, "ready");
  assert.deepEqual(f.configs.at(-1), { chord: { selfBias: 2 } });
});

test("successful configure is retained for later task additions", async () => {
  const f = fixture();
  await f.send({ type: "load", tasks: ["chord"] });
  assert.equal((await f.send({ type: "configure", config: { beat: { postproc: "minimal" } } })).type, "ready");
  await f.send({ type: "load", tasks: ["beat"] });
  assert.deepEqual(f.configs.at(-1), { beat: { postproc: "minimal" } });
});
