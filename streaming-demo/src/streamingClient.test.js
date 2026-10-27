import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import { test } from "node:test";
import { InferenceStream } from "./streamingClient.js";

function fixture(t) {
  const original = globalThis.Worker;
  class FakeWorker {
    messages = [];
    terminated = 0;
    postMessage(message, transfer) {
      if (this.sendError) throw this.sendError;
      this.messages.push({ message, transfer });
    }
    terminate() { this.terminated++; }
    reply(data) { this.onmessage?.({ data }); }
    fail(message) { this.onerror?.({ message }); }
  }
  globalThis.Worker = FakeWorker;
  t.after(() => {
    if (original === undefined) delete globalThis.Worker;
    else globalThis.Worker = original;
  });
  const stream = new InferenceStream();
  return { stream, worker: stream.worker };
}

// Bounded settlement checks turn a dangling promise into a useful assertion,
// without sleeping or keeping a test worker alive indefinitely.
async function outcome(promise) {
  let result = { status: "pending" };
  promise.then(value => { result = { status: "fulfilled", value }; },
    error => { result = { status: "rejected", error }; });
  await setImmediate();
  return result;
}

test("close rejects pending requests and is idempotent", async t => {
  const { stream, worker } = fixture(t);
  const reset = stream.reset();
  const finish = stream.finish();
  const results = [outcome(reset), outcome(finish)];
  stream.close();
  stream.close();
  for (const result of await Promise.all(results)) assert.equal(result.status, "rejected");
  assert.equal(stream.pending.size, 0);
  assert.equal(worker.terminated, 1);
  assert.equal((await outcome(stream.reset())).status, "rejected");
  assert.equal(worker.messages.length, 2);
});

test("worker failure rejects current and queued loads and later retries", async t => {
  const { stream, worker } = fixture(t);
  const first = stream.load(["chord"]);
  const queued = stream.load(["beat"]);
  first.catch(() => {});
  queued.catch(() => {});
  await setImmediate();
  worker.fail("worker import failed");
  for (const result of await Promise.all([outcome(first), outcome(queued)]))
    assert.equal(result.status, "rejected");
  assert.equal(stream.pending.size, 0);
  assert.equal(worker.terminated, 1);
  const retry = await outcome(stream.load(["chord"]));
  assert.equal(retry.status, "rejected");
  assert.match(retry.error.message, /worker import failed/);
  assert.equal(worker.messages.length, 1);
});

test("unreadable worker responses settle outstanding requests", async t => {
  const { stream, worker } = fixture(t);
  const result = outcome(stream.reset());
  worker.onmessageerror();
  assert.equal((await result).status, "rejected");
  assert.equal(stream.pending.size, 0);
  assert.equal((await outcome(stream.finish())).status, "rejected");
});

test("close between a ready reply and its continuation cannot restore stale info", async t => {
  const { stream, worker } = fixture(t);
  const loading = stream.load(["chord"]);
  loading.catch(() => {});
  await setImmediate();
  worker.reply({ seq: 1, type: "ready", info: { tasks: ["chord"] } });
  stream.close();
  assert.equal((await outcome(loading)).status, "rejected");
  assert.equal(stream.info, null);
});

test("close before a queued load sends prevents worker traffic", async t => {
  const { stream, worker } = fixture(t);
  const result = outcome(stream.load(["chord"]));
  stream.close();
  assert.equal((await result).status, "rejected");
  assert.equal(worker.messages.length, 0);
});

test("postMessage failure rejects and removes its request without poisoning the worker", async t => {
  const { stream, worker } = fixture(t);
  worker.sendError = new Error("cannot transfer buffer");
  const result = await outcome(stream.pushChunk(new Float32Array(4)));
  assert.equal(result.status, "rejected");
  assert.equal(stream.pending.size, 0);
  worker.sendError = null;
  const retry = stream.reset();
  worker.reply({ seq: worker.messages[0].message.seq, type: "ok" });
  assert.equal((await retry).type, "ok");
});

test("normal replies remain correlated, and request errors permit a retry", async t => {
  const { stream, worker } = fixture(t);
  const first = outcome(stream.reset());
  const second = stream.finish();
  worker.reply({ seq: 2, type: "result", chord: [] });
  worker.reply({ seq: 1, type: "error", message: "no engine yet" });
  assert.equal((await second).type, "result");
  assert.equal((await first).status, "rejected");
  assert.equal(stream.pending.size, 0);
  const retry = stream.reset();
  worker.reply({ seq: 3, type: "ok" });
  assert.equal((await retry).type, "ok");
  assert.equal(worker.terminated, 0);
});

test("closed streams ignore late progress and ready messages", async t => {
  const { stream, worker } = fixture(t);
  let progress = 0;
  const result = outcome(stream.load(["chord"], () => progress++));
  await setImmediate();
  stream.info = { tasks: ["chord"] };
  stream.close();
  worker.reply({ type: "progress", loaded: 1, total: 2 });
  worker.reply({ seq: 1, type: "ready", info: { tasks: ["beat"] } });
  await result;
  await setImmediate();
  assert.equal(progress, 0);
  assert.equal(stream.info, null);
  assert.equal(stream.onProgress, null);
});
