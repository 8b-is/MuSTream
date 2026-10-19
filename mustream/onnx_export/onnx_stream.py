#!/usr/bin/env python
"""
Shared runtime for the ONNX runners (run_chord_onnx.py, run_beat_onnx.py,
run_key_onnx.py): drive an exported streaming graph with its carried state,
check that a backbone/probe pair belongs together, and cut audio into the
fixed-size chunks the streaming backbone takes.

No PyTorch and nothing else from this repository:

    pip install onnxruntime numpy soundfile soxr
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:                                   # pragma: no cover
    sys.exit("onnxruntime is required: pip install onnxruntime")


# ---------------------------------------------------------------------------
# driving one exported graph
# ---------------------------------------------------------------------------

class OnnxStreamer:
    """
    One exported streaming graph plus its carried state (backbone or probe).

    The spec's `state` entries ({name, output, shape, dtype}) are zero-initialised
    by reset() and each call feeds the previous `output` back into `name`.
    Calling returns the graph's non-state outputs, in graph order.
    """

    def __init__(self, onnx_path, providers=None, intra_threads=None):
        onnx_path = str(onnx_path)
        opts = ort.SessionOptions()
        if intra_threads:
            opts.intra_op_num_threads = intra_threads
            opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path, opts, providers=providers or ["CPUExecutionProvider"])

        sidecar = os.path.splitext(onnx_path)[0] + ".json"
        meta = self.session.get_modelmeta().custom_metadata_map.get("mert_onnx_spec")
        if meta is not None:
            self.spec = json.loads(meta)
        elif os.path.isfile(sidecar):
            with open(sidecar) as f:
                self.spec = json.load(f)
        else:
            raise FileNotFoundError(f"no embedded spec in {onnx_path} and no {sidecar}")

        self.state_entries = self.spec["state"]
        state_outputs = {e["output"] for e in self.state_entries}
        self.value_names = [o.name for o in self.session.get_outputs()
                            if o.name not in state_outputs]
        self.output_names = self.value_names + [e["output"] for e in self.state_entries]
        self.reset()

    def reset(self):
        """Start a fresh track (zero all state)."""
        self.state = {e["name"]: np.zeros(e["shape"], dtype=np.dtype(e["dtype"]))
                      for e in self.state_entries}

    def __call__(self, **inputs):
        outs = self.session.run(self.output_names, {**inputs, **self.state})
        for entry, value in zip(self.state_entries, outs[len(self.value_names):]):
            self.state[entry["name"]] = value
        return outs[:len(self.value_names)]

    def state_bytes(self):
        return sum(v.nbytes for v in self.state.values())


class Geometry:
    """
    Chunking for a streaming backbone/probe pair, derived from the two specs.

    hop            samples fed to the backbone per call (= one probe chunk)
    probe_frames   raw backbone frames the probe takes per call
    label_frames   probe output frames per call
    """

    def __init__(self, backbone, probe):
        b, p = backbone.spec, probe.spec
        self.sample_rate = b["sample_rate"]
        self.probe_frames = p["chunk_frames"]
        self.label_frames = p["label_frames"]
        self.hop = self.label_frames * p["pool_factor"] * p["hop_size"]

    def describe(self):
        return (f"chunk {self.hop} samples ({self.hop / self.sample_rate * 1000:.0f} ms), "
                f"{self.label_frames} labels/chunk")


def check_pair(backbone, probe, graph):
    """
    Refuse a backbone/probe pair that does not belong together, before any audio
    is read. `graph` is the expected probe kind: 'chord_probe', 'beat_probe' or
    'key_probe'. Returns the pair's Geometry.
    """
    b, p = backbone.spec, probe.spec
    if b.get("mode") != "streaming":
        raise SystemExit(f"the backbone graph is {b.get('mode')!r}, expected a streaming backbone")
    if p.get("graph") != graph or p.get("mode") != "streaming":
        raise SystemExit(
            f"the probe graph is {p.get('graph')!r}/{p.get('mode')!r}, "
            f"expected a streaming {graph}")
    if b["sample_rate"] != p["sample_rate"]:
        raise SystemExit(
            f"sample rate mismatch: backbone {b['sample_rate']}, probe {p['sample_rate']}")

    want = list(p["features_input"]["shape"])
    got = list(b["feature_shape"])
    if got[:2] != want[:2] or got[3] != want[3]:
        raise SystemExit(
            f"shape mismatch: the backbone emits {got}, the probe expects {want} "
            "(check the layer count L and the hidden size).")
    if got[2] != want[2]:
        raise SystemExit(f"chunk mismatch: backbone {got[2]} frames, probe wants {want[2]}")

    # A probe only matches the backbone checkpoint it was trained on; the shapes
    # cannot tell, so compare the recorded provenance.
    trained_on = p.get("mert_align_checkpoint")
    built_from = b.get("source_checkpoint")
    if trained_on and built_from and os.path.abspath(trained_on) != os.path.abspath(built_from):
        print(f"[warn] this probe was trained against\n"
              f"[warn]   {trained_on}\n"
              f"[warn] but the backbone graph was exported from\n"
              f"[warn]   {built_from}\n"
              f"[warn] the features will be subtly wrong unless these are the same weights")

    if b["lookahead_block"] != p["chunk_frames"]:
        print(f"[warn] the backbone's lookahead_block is {b['lookahead_block']} frames but "
              f"the chunk is {p['chunk_frames']}; a larger chunk only adds latency")

    geom = Geometry(backbone, probe)
    if geom.hop != b["chunk_samples"]:
        raise SystemExit(
            f"chunk mismatch: backbone takes {b['chunk_samples']} samples, probe chunk is {geom.hop}")
    return geom


# ---------------------------------------------------------------------------
# audio -> fixed-size chunks at the model's sample rate
# ---------------------------------------------------------------------------

def stream_chunks(path, target_sr, chunk_samples, read_block=65536):
    """
    Yield (chunk, n_real): `chunk_samples` float32 mono samples at `target_sr`,
    and how many of them came from the file (the last chunk is zero-padded).
    Reads and resamples incrementally, so memory is bounded by the chunk.
    """
    import soundfile as sf

    info = sf.info(str(path))
    resampler = None
    if info.samplerate != target_sr:
        import soxr
        resampler = soxr.ResampleStream(info.samplerate, target_sr, 1, dtype="float32")

    buf = np.zeros(0, dtype=np.float32)
    for block in sf.blocks(str(path), blocksize=read_block, dtype="float32", always_2d=True):
        mono = block.mean(axis=1).astype(np.float32, copy=False)
        if resampler is not None:
            mono = resampler.resample_chunk(mono)
        if mono.size:
            buf = np.concatenate([buf, mono])
        while buf.size >= chunk_samples:
            yield buf[:chunk_samples], chunk_samples
            buf = buf[chunk_samples:]

    if resampler is not None:
        tail = resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
        if tail.size:
            buf = np.concatenate([buf, tail])
            while buf.size >= chunk_samples:
                yield buf[:chunk_samples], chunk_samples
                buf = buf[chunk_samples:]
    if buf.size:
        yield np.pad(buf, (0, chunk_samples - buf.size)), int(buf.size)


def stream_windows(path, geom):
    """Yield (chunk, n_real), one backbone input per call. An empty file still
    yields one all-zero chunk."""
    any_chunk = False
    for chunk, n_real in stream_chunks(path, geom.sample_rate, geom.hop):
        any_chunk = True
        yield chunk, n_real
    if not any_chunk:
        yield np.zeros(geom.hop, dtype=np.float32), 0


def backbone_features(backbone, chunk, geom=None):
    """One audio chunk -> the probe's [1, L, chunk_frames, D] input."""
    (features,) = backbone(audio=chunk[None, :].astype(np.float32))
    return features


def describe_pair(backbone, probe, geom, backbone_path, probe_path, extra=""):
    """Print the pair banner shared by the runners; returns the chunk length in ms."""
    b, p = backbone.spec, probe.spec
    chunk_ms = geom.hop / geom.sample_rate * 1000
    print(f"[pipe] backbone {Path(backbone_path).name}")
    print(f"[pipe]   {b['n_params'] / 1e6:.1f}M params, {geom.describe()}")
    print(f"[pipe]   context {b['context']} frames, carried state "
          f"{backbone.state_bytes() / 1e6:.0f} MB")
    print(f"[pipe] probe    {Path(probe_path).name}")
    print(f"[pipe]   {p['n_params'] / 1e6:.2f}M params, {p['label_frames']} frames/chunk "
          f"({p['label_rate']:.0f} fps){extra}")
    return chunk_ms


def report_timing(times, chunk_ms, n_frames, label_rate):
    """Print throughput and per-chunk latency."""
    duration = n_frames / label_rate
    total = times.sum() / 1000
    print(f"[pipe]   {n_frames} frames over {duration:.1f}s in {total:.1f}s "
          f"({duration / total:.1f}x real time)")
    print(f"[pipe]   per chunk: median {np.median(times):.1f} ms, p90 "
          f"{np.percentile(times, 90):.1f} ms, budget {chunk_ms:.0f} ms "
          f"-> RTF {np.median(times) / chunk_ms:.2f}")
