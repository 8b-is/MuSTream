#!/usr/bin/env python
"""
Export a trained key LSTM probe to a streaming ONNX graph that runs on top of an
exported streaming backbone.

    python onnx_export/export_key_probe_onnx.py ckpt/pytorch/key/streaming_26M_r0/best.pt \
        --out out/streaming_26M_r0.streaming.onnx

The positional argument is a probe .pt or its run directory (best.pt is used).
The run config comes from the checkpoint, the config.yaml beside it, or --config.

Graph:  features [B, L, chunk_frames, D] (the backbone's hidden_states), h0, c0
     -> pool by pool_factor -> layer selection -> LSTM -> per-frame linear head
     -> RUNNING logits [B, label_frames, 24] (row t = estimate from audio 0..t),
        h_out, c_out

The newest row is the current key estimate; argmax happens in host code (see
run_key_onnx.py). --check (default on) compares the streamed logits with the
PyTorch head's forward(x, running=True), plus exact per-frame argmax agreement.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn

# Siblings import from HERE; the probe packages from mustream/.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "key_detection"))

from key_detection.models import (                                   # noqa: E402
    build_classifier,
    select_mert_layer,
)
from key_detection.key_utils import KEY_CLASSES                      # noqa: E402
from export_mert_onnx import (                                       # noqa: E402
    DEFAULT_OPSET,
    _export_onnx,
    _report,
    _save_spec,
)
from export_chord_probe_onnx import (                                # noqa: E402
    checkpoint_id,
    default_n_backbone_layers,
    pool_features,
    read_probe_config,
    resolve_probe_path,
)


# ---------------------------------------------------------------------------
# rebuilding the probe
# ---------------------------------------------------------------------------

def infer_head_arch(model_state: dict) -> dict:
    """LSTM input size, hidden size, depth and class count, read off the weights."""
    if "lstm.weight_ih_l0" not in model_state:
        raise RuntimeError(
            "no 'lstm.weight_ih_l0' in this checkpoint -- it is not a KeyLSTM probe "
            f"(keys start with: {sorted(model_state)[:4]}). Only model_type: LSTM is supported.")
    prefix = "lstm.weight_ih_l"
    layers = sorted(int(k[len(prefix):]) for k in model_state if k.startswith(prefix))
    ih = model_state["lstm.weight_ih_l0"]
    head = model_state["head.weight"]
    return {
        "lstm_input_size": int(ih.shape[1]),
        "lstm_hidden_dim": int(ih.shape[0] // 4),
        "lstm_num_layers": len(layers),
        "n_classes": int(head.shape[0]),
    }


def build_head_from_checkpoint(checkpoint_path: str, config_path: str = None):
    """A trained key probe -> (KeyLSTM, metadata dict, config)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    config = read_probe_config(ckpt, checkpoint_path, config_path)
    arch = infer_head_arch(model_state)

    model_type = config.get("model_type", "LSTM")
    if model_type != "LSTM":
        raise RuntimeError(
            f"config says model_type: {model_type!r}; this exporter covers the LSTM head only.")

    head = build_classifier(model_type, arch["n_classes"], config)
    head.load_state_dict(model_state, strict=True)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    emb_dim = arch["lstm_input_size"]
    if emb_dim != int(config["emb_dim"]):
        raise RuntimeError(
            f"config emb_dim {config['emb_dim']} disagrees with the weights' {emb_dim}")

    meta = {
        "source_checkpoint": checkpoint_id(checkpoint_path),
        "epoch": ckpt.get("epoch"),
        "model_type": model_type,
        "n_classes": arch["n_classes"],
        "emb_dim": emb_dim,
        "n_backbone_layers": None,      # filled in at export: [B, L, T, D]'s L
        "mert_layer": config["mert_layer"],
        "pool_factor": int(config["pool_factor"]),
        "sample_rate": int(config["sample_rate"]),
        "hop_size": int(config["hop_size"]),
        "lstm_hidden_dim": arch["lstm_hidden_dim"],
        "lstm_num_layers": arch["lstm_num_layers"],
        "backbone": config.get("backbone", "pretrained"),
        "mert_align_checkpoint": (os.path.basename(config["mert_align_checkpoint"])
                                  if config.get("mert_align_checkpoint") else None),
        # 'final': only each training crop's last frame was supervised.
        "train_readout": config.get("train_readout", "final"),
        "crop_duration": config.get("crop_duration"),
        "n_params": sum(p.numel() for p in head.parameters()),
    }
    meta["label_rate"] = meta["sample_rate"] / meta["hop_size"] / meta["pool_factor"]
    meta["key_labels"] = _read_labels(arch["n_classes"], checkpoint_path)
    return head, meta, config


def _read_labels(n_classes, checkpoint_path):
    """Index -> key name table (key_utils.KEY_CLASSES), or None on a size mismatch."""
    if len(KEY_CLASSES) != n_classes:
        print(f"[probe] WARNING: key_utils.KEY_CLASSES has {len(KEY_CLASSES)} entries but "
              f"{os.path.basename(checkpoint_path)} was trained with {n_classes}; "
              "omitting the names")
        return None
    return list(KEY_CLASSES)


# ---------------------------------------------------------------------------
# the traceable graph
# ---------------------------------------------------------------------------

class StreamingKeyProbeONNX(nn.Module):
    """models.KeyLSTM's running forward with the LSTM state passed in and out
    explicitly (KeyLSTM keeps it in a Python attribute). --check compares
    against the real forward(x, running=True)."""

    def __init__(self, head, meta):
        super().__init__()
        self.head = head
        self.pool_factor = meta["pool_factor"]

    def forward(self, features, h0, c0):
        x = pool_features(features, self.pool_factor)          # [B, L, T, D]
        x = select_mert_layer(x, self.head.mert_layer)         # [B, T, D]
        out, (h, c) = self.head.lstm(x, (h0, c0))
        return self.head.head(out), h, c                       # [B, T, n_classes]


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def _state_entries(meta, batch_size):
    """State tensors: zero-initialize once per track, feed `output` back into `name`."""
    L, H = meta["lstm_num_layers"], meta["lstm_hidden_dim"]
    return [
        {"name": "h0", "output": "h_out", "shape": [L, batch_size, H], "dtype": "float32"},
        {"name": "c0", "output": "c_out", "shape": [L, batch_size, H], "dtype": "float32"},
    ]


def export_streaming(head, meta, out_path, chunk_frames, batch_size, n_layers, opset,
                     dynamic_batch=False, embed_metadata=True):
    """Export the streaming probe graph and write its spec. Returns the spec."""
    pool = meta["pool_factor"]
    if chunk_frames % pool:
        raise ValueError(
            f"--chunk-frames {chunk_frames} must be a multiple of pool_factor {pool}")
    label_frames = chunk_frames // pool
    D, C = meta["emb_dim"], meta["n_classes"]
    L, H = meta["lstm_num_layers"], meta["lstm_hidden_dim"]

    args = (torch.zeros(batch_size, n_layers, chunk_frames, D),
            torch.zeros(L, batch_size, H), torch.zeros(L, batch_size, H))
    input_names = ["features", "h0", "c0"]
    output_names = ["logits", "h_out", "c_out"]

    dynamic_axes = None
    if dynamic_batch:
        # Batch is axis 0 everywhere except the LSTM's [L, B, H] state.
        dynamic_axes = {n: {0: "batch"} for n in input_names + output_names}
        for n in ("h0", "c0", "h_out", "c_out"):
            dynamic_axes[n] = {1: "batch"}

    wrapper = StreamingKeyProbeONNX(head, meta).eval()
    external_data = _export_onnx(
        wrapper, args, out_path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True,
    )

    spec = {
        "graph": "key_probe",
        "external_data": external_data,
        "decoder": "none",              # argmax in host code
        "n_backbone_layers": n_layers,
        "batch_size": batch_size,
        "dynamic_batch": dynamic_batch,
        "opset": opset,
        **meta,
        "mode": "streaming",
        "chunk_frames": chunk_frames,
        "label_frames": label_frames,
        "features_input": {"name": "features",
                           "shape": [batch_size, n_layers, chunk_frames, D],
                           "dtype": "float32"},
        "logits_output": {"name": "logits", "shape": [batch_size, label_frames, C],
                          "running": True},
        "state": _state_entries(meta, batch_size),
    }
    json_path = _save_spec(out_path, spec, embed_metadata, external_data)
    print(f"[export] streaming graph -> {out_path}")
    print(f"[export] spec            -> {json_path}")
    return spec


# ---------------------------------------------------------------------------
# runtime helper (used by --check)
# ---------------------------------------------------------------------------

class KeyProbeOnnxStreamer:
    """
    Drives an exported streaming key probe:

        probe = KeyProbeOnnxStreamer("key_probe.streaming.onnx")
        logits = probe(backbone_features)     # [B, label_frames, n_classes], running

    Call reset() between tracks.
    """

    def __init__(self, onnx_path, providers=None, batch_size=None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            onnx_path, providers=providers or ["CPUExecutionProvider"])
        with open(os.path.splitext(onnx_path)[0] + ".json") as f:
            self.spec = json.load(f)
        if self.spec.get("graph") != "key_probe" or self.spec["mode"] != "streaming":
            raise ValueError(f"{onnx_path} is not a streaming key-probe graph")

        self.batch_size = batch_size or self.spec["batch_size"]
        if self.batch_size != self.spec["batch_size"] and not self.spec["dynamic_batch"]:
            raise ValueError(
                f"this graph was exported with a fixed batch of {self.spec['batch_size']}")
        self.chunk_frames = self.spec["chunk_frames"]
        self.label_frames = self.spec["label_frames"]
        self.output_names = ["logits"] + [s["output"] for s in self.spec["state"]]
        self.reset()

    def reset(self):
        """Start a fresh track: zero LSTM state."""
        self.state = {}
        for e in self.spec["state"]:
            shape = list(e["shape"])
            shape[1] = self.batch_size
            self.state[e["name"]] = np.zeros(shape, dtype=np.dtype(e["dtype"]))

    def __call__(self, features):
        """features: float32 [B, L, chunk_frames, D] -> running logits
        [B, label_frames, n_classes]."""
        features = np.ascontiguousarray(features, dtype=np.float32)
        if features.shape[-2] != self.chunk_frames:
            raise ValueError(
                f"expected {self.chunk_frames} raw frames per chunk, got {features.shape[-2]}")
        outs = self.session.run(self.output_names, {"features": features, **self.state})
        for entry, value in zip(self.spec["state"], outs[1:]):
            self.state[entry["name"]] = value
        return outs[0]


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def _reference_logits(head, features, pool_factor):
    """The real head's whole-sequence running forward on pooled features."""
    with torch.no_grad():
        return head(pool_features(features, pool_factor), running=True).numpy()


def _report_argmax(got, ref, label):
    """Exact per-frame argmax agreement."""
    g, r = got.argmax(-1), ref.argmax(-1)
    agree = int((g == r).sum())
    total = int(r.size)
    print(f"[check] {label:<52} {agree}/{total} exact  "
          f"{'ok' if agree == total else 'FAILED'}")
    if agree != total:
        first = tuple(np.argwhere(g != r)[0])
        raise AssertionError(
            f"{total - agree} of {total} per-frame key argmaxes differ from the PyTorch head "
            f"(first at {first})")


def check_streaming(head, meta, onnx_path, spec, n_chunks, tol, batch_size=None):
    """Feed random features chunk by chunk and compare with the PyTorch head's
    whole-sequence running forward."""
    batch_size = batch_size or spec["batch_size"]
    chunk_frames, label_frames = spec["chunk_frames"], spec["label_frames"]
    n_layers, D = spec["n_backbone_layers"], meta["emb_dim"]
    frames = n_chunks * chunk_frames
    torch.manual_seed(0)
    features = torch.randn(batch_size, n_layers, frames, D)

    print(f"[check] reference: PyTorch head over {frames} raw frames "
          f"({n_chunks * label_frames} label frames, "
          f"{frames / meta['label_rate'] / meta['pool_factor']:.1f}s, batch={batch_size})")
    ref = _reference_logits(head, features, meta["pool_factor"])

    streamer = KeyProbeOnnxStreamer(onnx_path, batch_size=batch_size)
    f_np = features.numpy()
    logits = np.concatenate(
        [streamer(f_np[:, :, c * chunk_frames:(c + 1) * chunk_frames]) for c in range(n_chunks)],
        axis=1)
    _report(f"ONNX streaming ({n_chunks} x {label_frames} label frames) vs whole sequence",
            logits, ref, tol)
    _report_argmax(logits, ref, "ONNX streaming per-frame key vs PyTorch")

    names = meta.get("key_labels")
    for b in range(batch_size):
        final_onnx, final_ref = int(logits[b, -1].argmax()), int(ref[b, -1].argmax())
        shown = names[final_ref] if names else final_ref
        print(f"[check]   batch {b}: final key {shown!r} "
              f"{'ok' if final_onnx == final_ref else 'FAILED'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export a trained key LSTM probe to a streaming ONNX graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint",
                        help="a trained probe .pt, or the run directory holding it (uses best.pt)")
    parser.add_argument("--config", default=None,
                        help="run config (default: the one in the checkpoint, else config.yaml "
                             "beside it)")
    # Accepted for older command lines; streaming is the only mode.
    parser.add_argument("--mode", choices=["streaming"], default="streaming",
                        help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None,
                        help="output .onnx path (default: <checkpoint>.streaming.onnx)")
    parser.add_argument("--chunk-frames", type=int, default=None,
                        help="RAW backbone frames per call (default: 15 = 200 ms)")
    parser.add_argument("--n-backbone-layers", type=int, default=None,
                        help="L in the backbone's [B, L, T, D] output (default: from the run's "
                             "MERT_model)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="leave the batch axis dynamic")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--no-check", action="store_true",
                        help="skip the comparison against the PyTorch head")
    parser.add_argument("--check-chunks", type=int, default=40,
                        help="chunks fed by the check")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="max logits difference vs PyTorch, relative to the reference's max |x| "
                             "(argmaxes are compared exactly)")
    parser.add_argument("--no-embed-metadata", action="store_true",
                        help="write only the .json sidecar")
    args = parser.parse_args()

    checkpoint_path = resolve_probe_path(args.checkpoint)
    print(f"[export] loading {checkpoint_path}")
    head, meta, config = build_head_from_checkpoint(checkpoint_path, args.config)
    print(f"[export] key LSTM probe: {meta['n_params'] / 1e6:.2f}M params, "
          f"emb_dim={meta['emb_dim']}, hidden={meta['lstm_hidden_dim']}x{meta['lstm_num_layers']}, "
          f"classes={meta['n_classes']}, mert_layer={meta['mert_layer']}, "
          f"pool_factor={meta['pool_factor']}, backbone={meta['backbone']}, "
          f"train_readout={meta['train_readout']}")

    n_layers = args.n_backbone_layers or default_n_backbone_layers(config, meta["emb_dim"])
    print(f"[export] backbone layer axis L={n_layers}")
    meta["n_backbone_layers"] = n_layers

    chunk_frames = args.chunk_frames or 15     # the streaming backbones' lookahead block

    out_path = args.out or f"{os.path.splitext(checkpoint_path)[0]}.streaming.onnx"
    spec = export_streaming(head, meta, out_path, chunk_frames, args.batch_size,
                            n_layers, args.opset, args.dynamic_batch,
                            not args.no_embed_metadata)
    print(f"[export] chunk = {chunk_frames} raw frames -> {spec['label_frames']} label "
          f"frames = {spec['label_frames'] / meta['label_rate'] * 1000:.0f} ms")
    if not args.no_check:
        check_streaming(head, meta, out_path, spec, args.check_chunks, args.tolerance)
        if args.dynamic_batch:
            check_streaming(head, meta, out_path, spec, args.check_chunks, args.tolerance,
                            batch_size=args.batch_size + 1)

    print("[export] done")


if __name__ == "__main__":
    main()
