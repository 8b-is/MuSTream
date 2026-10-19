#!/usr/bin/env python
"""
Export a trained chord LSTM probe (optionally with its self-bias CRF decoder)
to a streaming ONNX graph that runs on top of an exported streaming backbone.

    python onnx_export/export_chord_probe_onnx.py ckpt/pytorch/chord/streaming_26M_r0/best.pt \
        --decoder blocksync --out out/streaming_26M_r0.streaming.onnx

The positional argument is a probe .pt or its run directory (best.pt is used).
The run config comes from the checkpoint, the config.yaml beside it, or --config.

Graph:  features [B, L, chunk_frames, D] (the backbone's hidden_states), h0, c0
        [, crf_score, crf_started]
     -> pool by pool_factor -> probe forward -> logits [B, label_frames, V]
        [, tags, tags_valid] (decoder) + updated state (h_out, c_out [, crf_*_out])

--decoder:
  none        logits only; argmax in host code.
  blocksync   Viterbi over each chunk's label frames (no added latency on a
              block-causal backbone).

--self-bias is the CRF's only parameter (transitions = self_bias * I), in nats
on log-softmaxed emissions unless --emission-mode logits. --check (default on)
compares streamed logits with the PyTorch head and tags with
models.ChordPostProcessor exactly.
"""
import argparse
import glob
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

import yaml                                                        # noqa: E402

from chord_recognition.models import (                             # noqa: E402
    ChordPostProcessor,
    build_classifier,
    build_self_bias_crf,
    emission_block_for_config,
)
from export_mert_onnx import (                                     # noqa: E402
    DEFAULT_OPSET,
    _export_onnx,
    _report,
    _save_spec,
)

DECODERS = ("none", "blocksync")
CONFIG_FILENAME = "config.yaml"


# ---------------------------------------------------------------------------
# rebuilding the probe (also used by the beat and key exporters)
# ---------------------------------------------------------------------------

def checkpoint_id(path):
    """A checkpoint's name without local directories: '<parent>/<file>'."""
    path = os.path.normpath(path)
    return os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))


def resolve_backbone_path(config: dict, task: str) -> dict:
    """Copy of config with a relative mert_align_checkpoint resolved from mustream/<task>/."""
    path = config.get("mert_align_checkpoint")
    if path and not os.path.isabs(path):
        config = dict(config, mert_align_checkpoint=os.path.normpath(os.path.join(REPO_ROOT, task, path)))
    return config


def resolve_probe_path(path: str) -> str:
    """A run directory or a .pt file -> the .pt file (best.pt, else the newest)."""
    if not os.path.isdir(path):
        return path
    best = os.path.join(path, "best.pt")
    if os.path.isfile(best):
        return best
    candidates = glob.glob(os.path.join(path, "*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no *.pt in {path}")
    newest = max(candidates, key=os.path.getmtime)
    if len(candidates) > 1:
        print(f"[probe] no best.pt in {path}; using the newest of {len(candidates)} "
              f"checkpoints: {os.path.basename(newest)}")
    return newest


def read_probe_config(ckpt: dict, checkpoint_path: str, explicit: str = None) -> dict:
    """The run config: --config, else the one stored in the checkpoint, else
    config.yaml beside it."""
    if explicit:
        with open(explicit) as f:
            return yaml.safe_load(f)
    if isinstance(ckpt.get("config"), dict):
        return ckpt["config"]
    sibling = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), CONFIG_FILENAME)
    if os.path.isfile(sibling):
        with open(sibling) as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(
        f"{checkpoint_path} carries no 'config' and there is no {CONFIG_FILENAME} beside it. "
        "Pass --config: mert_layer, emb_dim and pool_factor cannot be read from the weights.")


def infer_head_arch(model_state: dict) -> dict:
    """LSTM input size, hidden size, depth and vocabulary size, read off the weights."""
    if "lstm.weight_ih_l0" not in model_state:
        raise RuntimeError(
            "no 'lstm.weight_ih_l0' in this checkpoint -- it is not an LSTM probe "
            f"(keys start with: {sorted(model_state)[:4]}). Only model_type: LSTM is supported.")
    prefix = "lstm.weight_ih_l"
    layers = sorted(int(k[len(prefix):]) for k in model_state if k.startswith(prefix))
    ih = model_state["lstm.weight_ih_l0"]
    out = model_state["out.weight"]
    return {
        "lstm_input_size": int(ih.shape[1]),
        "lstm_hidden_dim": int(ih.shape[0] // 4),
        "lstm_num_layers": len(layers),
        "vocab_size": int(out.shape[0]),
    }


def build_head_from_checkpoint(checkpoint_path: str, config_path: str = None):
    """A trained chord probe -> (ChordLSTM, metadata dict, config)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    config = read_probe_config(ckpt, checkpoint_path, config_path)
    arch = infer_head_arch(model_state)

    model_type = config.get("model_type", "LSTM")
    if model_type != "LSTM":
        raise RuntimeError(
            f"config says model_type: {model_type!r}; this exporter covers the LSTM head only.")

    head = build_classifier(model_type, arch["vocab_size"], config)
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
        "vocab_size": arch["vocab_size"],
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
        "emission_block": emission_block_for_config(resolve_backbone_path(config, "chord_recognition")),
        "n_params": sum(p.numel() for p in head.parameters()),
    }
    meta["label_rate"] = meta["sample_rate"] / meta["hop_size"] / meta["pool_factor"]
    meta["chord_labels"] = _read_labels(config, meta["vocab_size"], checkpoint_path)
    return head, meta, config


def _read_labels(config, vocab_size, checkpoint_path):
    """Index -> chord name table from the run's vocab_path, or None if unavailable."""
    path = config.get("vocab_path")
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, "chord_recognition", path)
    if not os.path.isfile(path):
        print(f"[probe] vocab_path {path} not found; the spec will carry no chord names")
        return None
    import pickle
    with open(path, "rb") as f:
        _, idx2chord = pickle.load(f)
    labels = [idx2chord[i] for i in range(len(idx2chord))]
    if len(labels) != vocab_size:
        print(f"[probe] WARNING: {os.path.basename(path)} has {len(labels)} labels but "
              f"{os.path.basename(checkpoint_path)} was trained with {vocab_size}; "
              "omitting the names")
        return None
    return labels


def default_n_backbone_layers(config, emb_dim):
    """L in the backbone's [B, L, T, D] output, from the run's MERT_model."""
    return {"m-a-p/MERT-v1-330M": 25, "m-a-p/MERT-v1-95M": 13}.get(
        config.get("MERT_model"), 13 if emb_dim < 1024 else 25)


# ---------------------------------------------------------------------------
# the CRF decoder as a traceable module
# ---------------------------------------------------------------------------

class BlockSyncViterbi(nn.Module):
    """
    models.decode_block_synchronous for one chunk: advance the Viterbi recursion
    over the chunk's T frames, then backtrack from the newest frame through this
    chunk's backpointers. State is the running score plus a `started` flag.
    """

    def __init__(self, crf):
        super().__init__()
        self.register_buffer("transitions", crf.transitions.detach().clone())
        self.register_buffer("start_transitions", crf.start_transitions.detach().clone())

    def _advance(self, score, emission, started):
        """One frame of the recursion -> (new score, backpointer)."""
        scores = score.unsqueeze(2) + self.transitions.unsqueeze(0)      # [B, V, V]
        best, prev = scores.max(dim=1)                                   # [B, V]
        cont = best + emission
        first = self.start_transitions.unsqueeze(0) + emission
        new = torch.where(started.view(-1, 1) > 0, cont, first)
        new = new - new.max(dim=1, keepdim=True).values                  # keep in float range
        return new, prev

    def forward(self, emissions, score, started):
        T = emissions.shape[1]
        bps = []
        for t in range(T):
            score, prev = self._advance(score, emissions[:, t], started)
            if t > 0:
                bps.append(prev)
            started = torch.ones_like(started)

        tag = score.argmax(dim=1)
        tags = [tag]
        for bp in reversed(bps):
            tag = torch.gather(bp, 1, tag.unsqueeze(1)).squeeze(1)
            tags.append(tag)
        tags.reverse()
        tags = torch.stack(tags, dim=1)                                  # [B, T]
        return tags, torch.ones_like(tags, dtype=torch.int64), score, started


def build_decoder(method: str, vocab_size: int, self_bias: float):
    """--decoder -> the traceable module, or None for 'none'."""
    if method == "none":
        return None
    if method == "blocksync":
        return BlockSyncViterbi(build_self_bias_crf(vocab_size, self_bias)).eval()
    raise ValueError(f"unknown decoder {method!r}, expected one of {DECODERS}")


# ---------------------------------------------------------------------------
# the traceable graph
# ---------------------------------------------------------------------------

def pool_features(features, pool_factor):
    """[B, L, T, D] -> [B, L, T/pool, D], non-overlapping mean over time."""
    if pool_factor == 1:
        return features
    B, L, T, D = features.shape
    return features.reshape(B, L, T // pool_factor, pool_factor, D).mean(3)


def _emissions(logits, emission_mode):
    """What the CRF scores: log-softmax ('logprobs') or raw logits."""
    return torch.log_softmax(logits, dim=-1) if emission_mode == "logprobs" else logits


class StreamingChordProbeONNX(nn.Module):
    """One chunk of backbone features plus state in; logits (and tags) plus the
    updated state out."""

    def __init__(self, head, meta, decoder, emission_mode):
        super().__init__()
        self.head = head
        self.decoder = decoder
        self.pool_factor = meta["pool_factor"]
        self.emission_mode = emission_mode

    def forward(self, features, h0, c0, *crf_state):
        x = pool_features(features, self.pool_factor)
        logits, (h, c) = self.head(x, state=(h0, c0), return_state=True)
        if self.decoder is None:
            return logits, h, c
        emissions = _emissions(logits, self.emission_mode)
        score, started = crf_state
        tags, valid, score, started = self.decoder(emissions, score, started)
        return logits, tags, valid, h, c, score, started


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def _state_entries(meta, batch_size, decoder):
    """State tensors: zero-initialize once per track, feed `output` back into `name`."""
    L, H, V = meta["lstm_num_layers"], meta["lstm_hidden_dim"], meta["vocab_size"]
    state = [
        {"name": "h0", "output": "h_out", "shape": [L, batch_size, H], "dtype": "float32"},
        {"name": "c0", "output": "c_out", "shape": [L, batch_size, H], "dtype": "float32"},
    ]
    if decoder is None:
        return state
    state.append({"name": "crf_score", "output": "crf_score_out",
                  "shape": [batch_size, V], "dtype": "float32"})
    state.append({"name": "crf_started", "output": "crf_started_out",
                  "shape": [batch_size], "dtype": "int64"})
    return state


def export_streaming(head, meta, out_path, chunk_frames, batch_size, n_layers, opset,
                     decoder, decoder_name, emission_mode, self_bias,
                     dynamic_batch=False, embed_metadata=True):
    """Export the streaming probe graph and write its spec. Returns the spec."""
    pool = meta["pool_factor"]
    if chunk_frames % pool:
        raise ValueError(
            f"--chunk-frames {chunk_frames} must be a multiple of pool_factor {pool}")
    label_frames = chunk_frames // pool

    D, V = meta["emb_dim"], meta["vocab_size"]
    L, H = meta["lstm_num_layers"], meta["lstm_hidden_dim"]
    features = torch.zeros(batch_size, n_layers, chunk_frames, D)
    h0 = torch.zeros(L, batch_size, H)
    c0 = torch.zeros(L, batch_size, H)

    input_names = ["features", "h0", "c0"]
    output_names = ["logits"]
    args = [features, h0, c0]
    if decoder is not None:
        args.append(torch.zeros(batch_size, V))
        input_names.append("crf_score")
        output_names += ["tags", "tags_valid"]
        args.append(torch.zeros(batch_size, dtype=torch.int64))
        input_names.append("crf_started")
    output_names += ["h_out", "c_out"]
    if decoder is not None:
        output_names += ["crf_score_out", "crf_started_out"]

    dynamic_axes = None
    if dynamic_batch:
        # Batch is axis 0 everywhere except the LSTM's [L, B, H] state.
        dynamic_axes = {n: {0: "batch"} for n in input_names + output_names}
        for n in ("h0", "c0", "h_out", "c_out"):
            dynamic_axes[n] = {1: "batch"}

    wrapper = StreamingChordProbeONNX(head, meta, decoder, emission_mode).eval()
    external_data = _export_onnx(
        wrapper, tuple(args), out_path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True,
    )

    spec = {
        "graph": "chord_probe",
        "mode": "streaming",
        "external_data": external_data,
        "chunk_frames": chunk_frames,
        "label_frames": label_frames,
        "decoder": decoder_name,
        "self_bias": self_bias if decoder is not None else None,
        "emission_mode": emission_mode if decoder is not None else None,
        "n_backbone_layers": n_layers,
        "batch_size": batch_size,
        "dynamic_batch": dynamic_batch,
        "opset": opset,
        "features_input": {"name": "features",
                           "shape": [batch_size, n_layers, chunk_frames, meta["emb_dim"]],
                           "dtype": "float32"},
        "logits_output": {"name": "logits", "shape": [batch_size, label_frames, V]},
        "tags_output": (None if decoder is None else
                        {"name": "tags", "shape": [batch_size, label_frames], "dtype": "int64",
                         "valid": "tags_valid"}),
        "state": _state_entries(meta, batch_size, decoder),
        **meta,
    }
    json_path = _save_spec(out_path, spec, embed_metadata, external_data)
    print(f"[export] streaming graph -> {out_path}")
    print(f"[export] spec            -> {json_path}")
    return spec


# ---------------------------------------------------------------------------
# runtime helper (used by --check)
# ---------------------------------------------------------------------------

class ChordProbeOnnxStreamer:
    """
    Drives an exported streaming chord probe:

        probe = ChordProbeOnnxStreamer("probe.streaming.onnx")
        logits, tags, valid = probe(backbone_features)   # tags/valid None without a decoder

    Call reset() between tracks.
    """

    def __init__(self, onnx_path, providers=None, batch_size=None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            onnx_path, providers=providers or ["CPUExecutionProvider"])
        with open(os.path.splitext(onnx_path)[0] + ".json") as f:
            self.spec = json.load(f)
        if self.spec.get("graph") != "chord_probe" or self.spec["mode"] != "streaming":
            raise ValueError(f"{onnx_path} is not a streaming chord-probe graph")

        self.batch_size = batch_size or self.spec["batch_size"]
        if self.batch_size != self.spec["batch_size"] and not self.spec["dynamic_batch"]:
            raise ValueError(
                f"this graph was exported with a fixed batch of {self.spec['batch_size']}")
        self.chunk_frames = self.spec["chunk_frames"]
        self.label_frames = self.spec["label_frames"]
        self.has_decoder = self.spec["tags_output"] is not None
        self.output_names = (["logits"] + (["tags", "tags_valid"] if self.has_decoder else [])
                             + [s["output"] for s in self.spec["state"]])
        self.reset()

    def _state_shape(self, entry):
        shape = list(entry["shape"])
        shape[1 if entry["name"] in ("h0", "c0") else 0] = self.batch_size
        return shape

    def reset(self):
        """Start a fresh track: zero LSTM state and Viterbi score."""
        self.state = {e["name"]: np.zeros(self._state_shape(e), dtype=np.dtype(e["dtype"]))
                      for e in self.spec["state"]}

    def __call__(self, features):
        """features: float32 [B, L, chunk_frames, D] -> (logits, tags, tags_valid)."""
        features = np.ascontiguousarray(features, dtype=np.float32)
        if features.shape[-2] != self.chunk_frames:
            raise ValueError(
                f"expected {self.chunk_frames} raw frames per chunk, got {features.shape[-2]}")
        outs = self.session.run(self.output_names, {"features": features, **self.state})
        n_out = 3 if self.has_decoder else 1
        for entry, value in zip(self.spec["state"], outs[n_out:]):
            self.state[entry["name"]] = value
        return (outs[0], outs[1], outs[2]) if self.has_decoder else (outs[0], None, None)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def _reference_logits(head, features, pool_factor):
    with torch.no_grad():
        return head(pool_features(features, pool_factor)).numpy()


def _reference_tags(logits, meta, decoder_name, self_bias, emission_mode, block):
    """models.ChordPostProcessor's decode of one track's [T, V] logits."""
    pp = ChordPostProcessor(
        num_tags=meta["vocab_size"], self_bias=self_bias, method=decoder_name,
        block=block, emission_mode=emission_mode,
    )
    return np.array(pp.decode(torch.from_numpy(logits)), dtype=np.int64)


def _compare_tags(got, ref, decoder, logits):
    """Exact tag comparison; also prints label-switch counts so a trivially
    constant sequence is visible."""
    argmax_switches = int((np.diff(logits.argmax(-1)) != 0).sum())
    switches = int((np.diff(ref) != 0).sum())
    agree = int((got == ref).sum())
    print(f"[check] {'ONNX ' + decoder + ' tags vs ChordPostProcessor':<46} "
          f"{agree}/{len(ref)} exact  {'ok' if agree == len(ref) else 'FAILED'}")
    print(f"[check]   (reference: {len(np.unique(ref))} distinct labels, {switches} switches "
          f"vs {argmax_switches} for argmax)")
    if agree != len(ref):
        raise AssertionError(
            f"{decoder}: {len(ref) - agree} of {len(ref)} labels differ from "
            "models.ChordPostProcessor's decode of the same logits")


def check_streaming(head, meta, onnx_path, spec, n_chunks, tol, self_bias, batch_size=None):
    """Feed random features chunk by chunk; compare logits with the PyTorch
    head's whole-sequence forward and tags with ChordPostProcessor."""
    batch_size = batch_size or spec["batch_size"]
    chunk_frames, label_frames = spec["chunk_frames"], spec["label_frames"]
    n_layers, D = spec["n_backbone_layers"], meta["emb_dim"]
    frames = n_chunks * chunk_frames
    torch.manual_seed(0)
    features = torch.randn(batch_size, n_layers, frames, D)

    print(f"[check] reference: PyTorch head over {frames} raw frames "
          f"({n_chunks * label_frames} label frames, "
          f"{frames / meta['label_rate'] / meta['pool_factor']:.1f}s, batch={batch_size})")
    ref_logits = _reference_logits(head, features, meta["pool_factor"])

    streamer = ChordProbeOnnxStreamer(onnx_path, batch_size=batch_size)
    f_np = features.numpy()
    outs = [streamer(f_np[:, :, c * chunk_frames:(c + 1) * chunk_frames]) for c in range(n_chunks)]
    logits = np.concatenate([o[0] for o in outs], axis=1)
    _report(f"ONNX streaming ({n_chunks} x {label_frames} label frames) vs whole sequence",
            logits, ref_logits, tol)

    if spec["tags_output"] is not None:
        tags = np.concatenate([o[1] for o in outs], axis=1)
        decoder = spec["decoder"]
        for b in range(batch_size):
            ref = _reference_tags(ref_logits[b], meta, decoder, self_bias,
                                  spec["emission_mode"], block=label_frames)
            _compare_tags(tags[b], ref, decoder, ref_logits[b])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export a trained chord LSTM probe (optionally with its CRF decoder) to a "
                    "streaming ONNX graph.",
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
    parser.add_argument("--decoder", choices=list(DECODERS), default="none",
                        help="none = logits only (argmax in host code); "
                             "blocksync = Viterbi over each chunk, no added latency")
    parser.add_argument("--self-bias", type=float, default=1.0,
                        help="self-bias CRF lambda, in nats on log-softmaxed emissions")
    parser.add_argument("--emission-mode", choices=["logprobs", "logits"], default="logprobs",
                        help="what the CRF scores")
    parser.add_argument("--chunk-frames", type=int, default=None,
                        help="RAW backbone frames per call (default: emission block x pool_factor)")
    parser.add_argument("--n-backbone-layers", type=int, default=None,
                        help="L in the backbone's [B, L, T, D] output (default: from the run's "
                             "MERT_model)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="leave the batch axis dynamic")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--no-check", action="store_true",
                        help="skip the comparison against PyTorch and ChordPostProcessor")
    parser.add_argument("--check-chunks", type=int, default=40,
                        help="chunks fed by the check")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="max logits difference vs PyTorch, relative to the reference's max |x| "
                             "(tags are compared exactly)")
    parser.add_argument("--no-embed-metadata", action="store_true",
                        help="write only the .json sidecar")
    args = parser.parse_args()

    checkpoint_path = resolve_probe_path(args.checkpoint)
    print(f"[export] loading {checkpoint_path}")
    head, meta, config = build_head_from_checkpoint(checkpoint_path, args.config)
    print(f"[export] chord LSTM probe: {meta['n_params'] / 1e6:.2f}M params, "
          f"emb_dim={meta['emb_dim']}, hidden={meta['lstm_hidden_dim']}x{meta['lstm_num_layers']}, "
          f"vocab={meta['vocab_size']}, mert_layer={meta['mert_layer']}, "
          f"pool_factor={meta['pool_factor']}, backbone={meta['backbone']}")

    n_layers = args.n_backbone_layers or default_n_backbone_layers(config, meta["emb_dim"])
    print(f"[export] backbone layer axis L={n_layers}")

    block = meta["emission_block"] or 3
    chunk_frames = args.chunk_frames or block * meta["pool_factor"]
    decoder = build_decoder(args.decoder, meta["vocab_size"], args.self_bias)
    if decoder is not None:
        pp = ChordPostProcessor(meta["vocab_size"], args.self_bias, args.decoder,
                                block=chunk_frames // meta["pool_factor"],
                                emission_mode=args.emission_mode)
        print(f"[export] {pp.describe()}")
    meta["n_backbone_layers"] = n_layers

    out_path = args.out or f"{os.path.splitext(checkpoint_path)[0]}.streaming.onnx"
    spec = export_streaming(
        head, meta, out_path, chunk_frames, args.batch_size, n_layers, args.opset,
        decoder, args.decoder, args.emission_mode, args.self_bias,
        args.dynamic_batch, not args.no_embed_metadata)
    print(f"[export] chunk = {chunk_frames} raw frames -> {spec['label_frames']} label "
          f"frames = {spec['label_frames'] / meta['label_rate'] * 1000:.0f} ms")
    if not args.no_check:
        check_streaming(head, meta, out_path, spec, args.check_chunks, args.tolerance,
                        args.self_bias)
        if args.dynamic_batch:
            check_streaming(head, meta, out_path, spec, args.check_chunks, args.tolerance,
                            args.self_bias, batch_size=args.batch_size + 1)

    print("[export] done")


if __name__ == "__main__":
    main()
