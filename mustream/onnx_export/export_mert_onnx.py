#!/usr/bin/env python
"""
Export a streaming MERT backbone (a distilled causal student) to a streaming
ONNX graph: one audio chunk plus the carried state in, that chunk's features
plus the updated state out.

    python onnx_export/export_mert_onnx.py ckpt/pytorch/backbone/streaming_26M_r0.ckpt \
        --out out/streaming_26M_r0.streaming.onnx

The positional argument is a .ckpt or the run directory holding one. The
architecture, positional scheme, lookahead block and context are read off the
checkpoint (plus its args.yml when present).

Graph inputs:  audio [B, chunk_samples], offset [B], k_cache / v_cache
               [L, B, H, context, Dh], conv_state_i [B, C, pad]
Graph outputs: hidden_states [B, L+1, chunk_frames, D] (or `hidden` with
               --hidden-states final), offset_out, k_cache_out, v_cache_out,
               conv_state_i_out

Zero-initialize every state once per track and feed each `*_out` back into the
matching input. Shapes are fixed (only --dynamic-batch frees the batch axis).
A .json spec describing inputs, outputs and state is written beside the .onnx
and embedded in its metadata. --check (default on) compares the streamed ONNX
output against the PyTorch full pass over a sequence long enough to wrap the
KV cache.
"""
import argparse
import glob
import json
import math
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np
import torch
from torch import nn

# Siblings import from HERE; modeling/ and the probe packages from mustream/.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from modeling.load_checkpoint import (                         # noqa: E402
    align_base_model,
    align_lookahead_block,
    align_positional_embedding,
    align_student_state_dict,
    infer_small_streaming_arch,
    is_streaming_layout,
    load_align_checkpoint,
    read_align_run_args,
)
from modeling.mert.small_mert import (                         # noqa: E402
    build_small_mert_config,
    resolve_base_config,
)
from modeling.moshi.modules.transformer import create_sin_embedding  # noqa: E402
from modeling.streaming.streaming_mert import FullStreamingMERTModel  # noqa: E402

DEFAULT_OPSET = 17
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_HOP_LENGTH = 320
SPEC_METADATA_KEY = "mert_onnx_spec"

# Static-shape asserts in moshi's rope / sin embedding trip this tracer warning;
# shapes are fixed by design and --check verifies the result.
warnings.filterwarnings(
    "ignore", message=".*Converting a tensor to a Python boolean.*",
    category=torch.jit.TracerWarning,
)


# ---------------------------------------------------------------------------
# rebuilding the model
# ---------------------------------------------------------------------------

def resolve_checkpoint_path(path: str) -> str:
    """A run directory or a .ckpt file -> the .ckpt file (newest if several)."""
    if not os.path.isdir(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is not a checkpoint or a run directory")
        return path
    candidates = glob.glob(os.path.join(path, "*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"no *.ckpt in {path}")
    newest = max(candidates, key=os.path.getmtime)
    if len(candidates) > 1:
        print(f"[export] {len(candidates)} checkpoints in {path}, using the newest: {os.path.basename(newest)}")
    return newest


def processor_do_normalize(base_model: str) -> bool:
    """Whether the base MERT's published feature extractor normalizes the
    waveform. Recorded in the spec only; the streaming path never normalizes."""
    try:
        from transformers import Wav2Vec2FeatureExtractor
        proc = Wav2Vec2FeatureExtractor.from_pretrained(base_model, trust_remote_code=True)
        return bool(proc.do_normalize)
    except Exception as exc:          # a missing processor must not fail an export
        print(f"[export] could not read {base_model}'s processor ({exc}); "
              "recording do_normalize=False")
        return False


def build_streaming_student(checkpoint_path: str, context: int = None,
                            num_attention_heads: int = None):
    """A streaming-student .ckpt -> (FullStreamingMERTModel, metadata dict).
    Weights load strict=True, so a wrong architecture guess fails loudly."""
    ckpt = load_align_checkpoint(checkpoint_path)
    state_dict = align_student_state_dict(ckpt, checkpoint_path)
    if not is_streaming_layout(state_dict):
        raise ValueError(f"{checkpoint_path} does not hold a streaming student; only streaming "
                         "backbones can be exported")

    run_args = read_align_run_args(checkpoint_path)
    base_model = align_base_model(ckpt, checkpoint_path)
    if not base_model:
        raise RuntimeError(
            f"{checkpoint_path} records no `mert_model` and neither does its args.yml; "
            "the base MERT config (pre- vs post-norm) cannot be inferred from the weights.")
    base_config = resolve_base_config(base_model)

    arch = infer_small_streaming_arch(state_dict)
    is_full_width = (
        arch["hidden_size"] == base_config.hidden_size
        and arch["intermediate_size"] == base_config.intermediate_size
        and arch["num_hidden_layers"] == base_config.num_hidden_layers
    )
    if is_full_width:
        # 'streaming' student: the base MERT's own width, conv stack and heads.
        config = base_config
        if num_attention_heads is not None and num_attention_heads != config.num_attention_heads:
            raise ValueError(
                f"--num-attention-heads {num_attention_heads} contradicts {base_model}'s own "
                f"{config.num_attention_heads} heads; a full-width streaming student keeps them.")
    else:
        # 'small_streaming' student. Heads are not recorded in any weight shape.
        heads = num_attention_heads or int(run_args.get("student_num_attention_heads", 8))
        config = build_small_mert_config(
            base_config,
            hidden_size=arch["hidden_size"],
            intermediate_size=arch["intermediate_size"],
            num_attention_heads=heads,
            num_hidden_layers=arch["num_hidden_layers"],
        )

    trained_context = int(ckpt.get("streaming_context") or run_args.get("streaming_context", 1000))
    if context is None:
        context = trained_context
    elif context != trained_context:
        print(f"[export] WARNING: --context {context} differs from the trained streaming_context "
              f"{trained_context}; re-evaluate the downstream probes before trusting this export.")
    positional_embedding = align_positional_embedding(ckpt)
    lookahead_block = align_lookahead_block(ckpt)

    model = FullStreamingMERTModel(
        config,
        context=context,
        positional_embedding=positional_embedding,
        lookahead_block=lookahead_block,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    meta = {
        "source_checkpoint": os.path.basename(checkpoint_path),
        "source_kind": "align_checkpoint",
        "streaming": True,
        "base_mert_model": base_model,
        "student_size": "full_width" if is_full_width else "small",
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_hidden_state_layers": config.num_hidden_layers + 1,
        "do_stable_layer_norm": bool(config.do_stable_layer_norm),
        "context": context,
        "positional_embedding": positional_embedding,
        "lookahead_block": lookahead_block,
        "sample_rate": int(run_args.get("sample_rate", DEFAULT_SAMPLE_RATE)),
        "hop_length": int(run_args.get("hop_length", DEFAULT_HOP_LENGTH)),
        "do_normalize": processor_do_normalize(base_model),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    return model, meta


# ---------------------------------------------------------------------------
# traceable forward pass with explicit streaming state
# ---------------------------------------------------------------------------

def conv_state_specs(model: FullStreamingMERTModel):
    """(conv_layer_index, in_channels, left_context_samples) for every conv layer
    that needs a left-context buffer (kernel > stride)."""
    specs = []
    for i, conv in enumerate(model.feature_extractor.conv_layers):
        pad = conv._effective_kernel_size - conv._stride
        if pad > 0:
            specs.append((i, conv.conv.conv.in_channels, pad))
    return specs


def total_conv_stride(model: FullStreamingMERTModel) -> int:
    """Samples per output frame."""
    stride = 1
    for conv in model.feature_extractor.conv_layers:
        stride *= conv._stride
    return stride


def _attention_bias(offset, n_query, n_cached, context, lookahead_block, dtype, device):
    """Additive attention mask [B, 1, T, S] for one chunk, in absolute frame
    positions: cached keys at offset-n_cached..offset-1 (negative = not yet
    written), the chunk's keys at offset..offset+T-1. Block-causal with
    `lookahead_block`, bounded to `context` frames of past."""
    ar_q = torch.arange(n_query, device=device, dtype=torch.long)
    pos_q = offset.view(-1, 1) + ar_q                                   # [B, T]
    if n_cached > 0:
        ar_c = torch.arange(n_cached, device=device, dtype=torch.long)
        pos_k = torch.cat([offset.view(-1, 1) - n_cached + ar_c, pos_q], dim=1)
    else:
        pos_k = pos_q                                                   # [B, S]
    delta = pos_q.unsqueeze(-1) - pos_k.unsqueeze(1)                    # [B, T, S]

    if lookahead_block > 1:
        right = (lookahead_block - 1) - (pos_q % lookahead_block)
        valid = delta >= -right.unsqueeze(-1)
    else:
        valid = delta >= 0
    valid = valid & (pos_k >= 0).unsqueeze(1)
    if context is not None:
        valid = valid & (delta < context)

    zero = torch.zeros((), dtype=dtype, device=device)
    neg = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device)
    return torch.where(valid.unsqueeze(1), zero, neg)                   # [B, 1, T, S]


def _self_attention(attn, x, k_cache, v_cache, offset, lookahead_block, capacity):
    """Streaming self-attention with the KV cache passed in/out explicitly (FIFO
    of `capacity` frames). Returns (output, new_k, new_v); caches are None for
    the stateless full pass."""
    assert attn.kv_repeat == 1 and not attn.cross_attention and attn.mult == 1
    T = x.shape[1]
    H = attn.num_heads
    Dh = attn.embed_dim // H

    qkv = attn.in_projs[0](x)                                           # [B, T, 3*C]
    qkv = qkv.view(-1, T, 3, H, Dh).permute(2, 0, 3, 1, 4)              # [3, B, H, T, Dh]
    q, k, v = qkv[0], qkv[1], qkv[2]
    if attn.rope is not None:
        q, k = attn.rope(q, k, offset, time_before_heads=False)         # cache holds rotated keys

    if k_cache is None:
        keys, values, n_cached = k, v, 0
        new_k = new_v = None
    else:
        n_cached = capacity
        keys = torch.cat([k_cache, k], dim=2)                           # [B, H, cap+T, Dh]
        values = torch.cat([v_cache, v], dim=2)
        new_k, new_v = keys[:, :, T:], values[:, :, T:]

    bias = _attention_bias(
        offset, T, n_cached, attn.context, lookahead_block, x.dtype, x.device,
    )
    scores = torch.matmul(q, keys.transpose(-1, -2)) / math.sqrt(Dh) + bias
    out = torch.matmul(torch.softmax(scores, dim=-1), values)           # [B, H, T, Dh]
    out = out.transpose(1, 2).reshape(-1, T, H * Dh)
    return attn.out_projs[0](out), new_k, new_v


def _transformer_layer(layer, x, k_cache, v_cache, offset, lookahead_block, capacity, pre_norm):
    """One transformer layer, pre-norm or post-norm."""
    if pre_norm:
        update, new_k, new_v = _self_attention(
            layer.self_attn, layer.norm1(x), k_cache, v_cache, offset, lookahead_block, capacity)
        x = x + layer.layer_scale_1(update)
        update = layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
        x = x + layer.layer_scale_2(update)
    else:
        update, new_k, new_v = _self_attention(
            layer.self_attn, x, k_cache, v_cache, offset, lookahead_block, capacity)
        x = layer.norm1(x + layer.layer_scale_1(update))
        update = layer.linear2(layer.activation(layer.linear1(x)))
        x = layer.norm2(x + layer.layer_scale_2(update))
    return x, new_k, new_v


def _streaming_core(model, audio, offset, k_cache, v_cache, conv_states, output_hidden_states,
                    capacity=0):
    """FullStreamingMERTModel.forward with explicit state. With k_cache/v_cache/
    conv_states None it is the stateless full pass.

    Returns (features, new_conv_states, new_k, new_v); features is [B, L, T, D]
    when output_hidden_states else [B, T, D]."""
    specs = conv_state_specs(model)
    by_layer = {layer_id: idx for idx, (layer_id, _, _) in enumerate(specs)}

    h = audio.unsqueeze(1)                                              # [B, 1, samples]
    new_conv_states = []
    for i, conv in enumerate(model.feature_extractor.conv_layers):
        if i in by_layer:
            pad = specs[by_layer[i]][2]
            if conv_states is None:
                left = torch.zeros(
                    h.shape[0], h.shape[1], pad, dtype=h.dtype, device=h.device)
            else:
                left = conv_states[by_layer[i]]
            h = torch.cat([left, h], dim=-1)
            new_conv_states.append(h[..., -pad:])
        h = model.feature_extractor.activation(conv.conv(h))

    h = model.feature_projection(h.transpose(1, 2))                     # [B, T, D]
    if not model.config.do_stable_layer_norm:
        h = model.final_layer_norm(h)                                   # post-norm: before the stack

    transformer = model.transformer
    T = h.shape[1]
    if transformer.positional_embedding in {"sin", "sin_rope"}:
        positions = offset.view(-1, 1, 1) + torch.arange(T, device=h.device).view(1, -1, 1)
        h = h + transformer.positional_scale * create_sin_embedding(
            positions, h.shape[-1], max_period=transformer.max_period, dtype=h.dtype)

    hidden_states = []
    new_k, new_v = [], []
    pre_norm = bool(model.config.do_stable_layer_norm)
    for i, layer in enumerate(transformer.layers):
        if output_hidden_states:
            hidden_states.append(h)
        h, k_i, v_i = _transformer_layer(
            layer, h,
            None if k_cache is None else k_cache[i],
            None if v_cache is None else v_cache[i],
            offset, model.lookahead_block, capacity, pre_norm,
        )
        if k_cache is not None:
            new_k.append(k_i)
            new_v.append(v_i)

    if model.config.do_stable_layer_norm:
        h = model.final_layer_norm(h)                                   # pre-norm: after the stack
    if output_hidden_states:
        hidden_states.append(h)
        features = torch.stack(hidden_states, dim=1)                    # [B, L, T, D]
    else:
        features = h

    stacked_k = torch.stack(new_k, dim=0) if k_cache is not None else None
    stacked_v = torch.stack(new_v, dim=0) if v_cache is not None else None
    return features, new_conv_states, stacked_k, stacked_v


class StreamingMERTONNX(nn.Module):
    """One streaming step: (audio chunk, state) -> (features, new state)."""

    def __init__(self, model: FullStreamingMERTModel, output_hidden_states: bool = True):
        super().__init__()
        self.model = model
        self.output_hidden_states = output_hidden_states
        self.specs = conv_state_specs(model)
        self.capacity = model.transformer.layers[0].self_attn.context
        self.hop = total_conv_stride(model)

    def forward(self, audio, offset, k_cache, v_cache, *conv_states):
        features, new_conv_states, new_k, new_v = _streaming_core(
            self.model, audio, offset, k_cache, v_cache, list(conv_states),
            self.output_hidden_states, self.capacity,
        )
        new_offset = offset + audio.shape[-1] // self.hop
        return (features, new_offset, new_k, new_v, *new_conv_states)


class FullPassMERTONNX(nn.Module):
    """Stateless whole-sequence pass through the explicit-state rewrite; used by
    --check to verify the rewrite against the PyTorch model."""

    def __init__(self, model: FullStreamingMERTModel, output_hidden_states: bool = True):
        super().__init__()
        self.model = model
        self.output_hidden_states = output_hidden_states

    def forward(self, audio):
        offset = torch.zeros(audio.shape[0], dtype=torch.long, device=audio.device)
        features, _, _, _ = _streaming_core(
            self.model, audio, offset, None, None, None, self.output_hidden_states)
        return features


# ---------------------------------------------------------------------------
# export (shared with the probe exporters)
# ---------------------------------------------------------------------------

def _save_spec(onnx_path, spec, embed_metadata: bool, external_data: bool = False):
    """Write <name>.json beside the graph and, if asked, embed it in the ONNX
    metadata under SPEC_METADATA_KEY."""
    json_path = os.path.splitext(onnx_path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(spec, f, indent=2)
    if embed_metadata:
        import onnx
        # Leave external tensors on disk so the proto stays under 2 GB.
        proto = onnx.load(onnx_path, load_external_data=not external_data)
        entry = proto.metadata_props.add()
        entry.key = SPEC_METADATA_KEY
        entry.value = json.dumps(spec)
        onnx.save(proto, onnx_path)
    return json_path


def _export_onnx(wrapper, inputs, out_path, **export_kwargs) -> bool:
    """torch.onnx.export into out_path. If the model exceeds protobuf's 2 GB
    limit, the weights are consolidated into a single <name>.onnx.data beside
    it. Returns whether that external-data file was written."""
    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=".onnx_export_", dir=out_dir)
    try:
        tmp_path = os.path.join(tmp_dir, os.path.basename(out_path))
        with torch.no_grad():
            torch.onnx.export(wrapper, inputs, tmp_path, **export_kwargs)

        spilled = [f for f in os.listdir(tmp_dir) if f != os.path.basename(out_path)]
        if not spilled:
            shutil.move(tmp_path, out_path)
            return False

        import onnx
        data_name = os.path.basename(out_path) + ".data"
        data_path = os.path.join(out_dir, data_name)
        if os.path.exists(data_path):
            os.remove(data_path)        # onnx.save appends
        onnx.save(
            onnx.load(tmp_path), out_path,
            save_as_external_data=True, all_tensors_to_one_file=True,
            location=data_name, size_threshold=1024, convert_attribute=True,
        )
        print(f"[export] over 2 GB: weights written to {data_name}, which must travel with the .onnx")
        return True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _feature_shape(meta, batch_size, frames, output_hidden_states):
    if output_hidden_states:
        return [batch_size, meta["num_hidden_state_layers"], frames, meta["hidden_size"]]
    return [batch_size, frames, meta["hidden_size"]]


def export_streaming(
    model, meta, out_path, chunk_frames, batch_size, opset, output_hidden_states,
    dynamic_batch=False, embed_metadata=True,
):
    """Export the streaming graph and write its spec. Returns the spec."""
    hop = total_conv_stride(model)
    context = meta["context"]
    block = meta["lookahead_block"]
    if chunk_frames % block:
        raise ValueError(
            f"--chunk-frames {chunk_frames} must be a multiple of the model's lookahead_block {block}")
    if chunk_frames > context:
        raise ValueError(
            f"--chunk-frames {chunk_frames} exceeds the attention context (KV cache size) {context}")

    H = meta["num_attention_heads"]
    Dh = meta["hidden_size"] // H
    L = meta["num_hidden_layers"]
    specs = conv_state_specs(model)

    audio = torch.zeros(batch_size, chunk_frames * hop)
    offset = torch.zeros(batch_size, dtype=torch.long)
    k_cache = torch.zeros(L, batch_size, H, context, Dh)
    v_cache = torch.zeros(L, batch_size, H, context, Dh)
    conv_states = [torch.zeros(batch_size, ch, length) for _, ch, length in specs]

    conv_names = [f"conv_state_{i}" for i in range(len(specs))]
    input_names = ["audio", "offset", "k_cache", "v_cache"] + conv_names
    feature_name = "hidden_states" if output_hidden_states else "hidden"
    output_names = [feature_name, "offset_out", "k_cache_out", "v_cache_out"] + \
                   [f"{n}_out" for n in conv_names]

    dynamic_axes = None
    if dynamic_batch:
        # Batch is axis 0 everywhere except the [L, B, ...] KV caches.
        dynamic_axes = {n: {0: "batch"} for n in input_names + output_names}
        for n in ("k_cache", "v_cache", "k_cache_out", "v_cache_out"):
            dynamic_axes[n] = {1: "batch"}

    wrapper = StreamingMERTONNX(model, output_hidden_states).eval()
    external_data = _export_onnx(
        wrapper, (audio, offset, k_cache, v_cache, *conv_states), out_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    spec = {
        "mode": "streaming",
        "external_data": external_data,
        "chunk_frames": chunk_frames,
        "chunk_samples": chunk_frames * hop,
        "output_hidden_states": output_hidden_states,
        "feature_output": feature_name,
        "feature_shape": _feature_shape(meta, batch_size, chunk_frames, output_hidden_states),
        "batch_size": batch_size,
        "dynamic_batch": dynamic_batch,
        "opset": opset,
        "audio_input": {"name": "audio", "shape": [batch_size, chunk_frames * hop], "dtype": "float32"},
        # Zero-initialize each once per track; feed `output` back into `name`.
        "state": (
            [{"name": "offset", "output": "offset_out", "shape": [batch_size], "dtype": "int64"},
             {"name": "k_cache", "output": "k_cache_out",
              "shape": [L, batch_size, H, context, Dh], "dtype": "float32"},
             {"name": "v_cache", "output": "v_cache_out",
              "shape": [L, batch_size, H, context, Dh], "dtype": "float32"}]
            + [{"name": n, "output": f"{n}_out", "shape": [batch_size, ch, length],
                "dtype": "float32", "conv_layer": layer_id}
               for n, (layer_id, ch, length) in zip(conv_names, specs)]
        ),
        **meta,
    }
    json_path = _save_spec(out_path, spec, embed_metadata, external_data)
    print(f"[export] streaming graph -> {out_path}")
    print(f"[export] spec            -> {json_path}")
    return spec


# ---------------------------------------------------------------------------
# runtime helper (used by --check)
# ---------------------------------------------------------------------------

class MERTOnnxStreamer:
    """
    Drives an exported streaming backbone graph:

        streamer = MERTOnnxStreamer("mert.streaming.onnx")
        for chunk in chunks_of(audio, streamer.chunk_samples):
            feats = streamer(chunk)          # [B, L, chunk_frames, D]

    Call reset() between independent tracks.
    """

    def __init__(self, onnx_path, providers=None, batch_size=None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            onnx_path, providers=providers or ["CPUExecutionProvider"])
        spec_json = self.session.get_modelmeta().custom_metadata_map.get(SPEC_METADATA_KEY)
        if spec_json is None:
            with open(os.path.splitext(onnx_path)[0] + ".json") as f:
                spec_json = f.read()
        self.spec = json.loads(spec_json)
        if self.spec["mode"] != "streaming":
            raise ValueError(f"{onnx_path} is a '{self.spec['mode']}' graph, not a streaming one")

        self.batch_size = batch_size or self.spec["batch_size"]
        if self.batch_size != self.spec["batch_size"] and not self.spec["dynamic_batch"]:
            raise ValueError(
                f"this graph was exported with a fixed batch of {self.spec['batch_size']}")
        self.chunk_frames = self.spec["chunk_frames"]
        self.chunk_samples = self.spec["chunk_samples"]
        self.sample_rate = self.spec["sample_rate"]
        self.output_names = [self.spec["feature_output"]] + [s["output"] for s in self.spec["state"]]
        self.reset()

    def _state_shape(self, entry):
        shape = list(entry["shape"])
        # batch is axis 0 everywhere except the [L, B, H, cap, Dh] KV caches
        axis = 1 if entry["name"] in ("k_cache", "v_cache") else 0
        shape[axis] = self.batch_size
        return shape

    def reset(self):
        """Start a fresh track: zero state, offset 0."""
        self.state = {
            entry["name"]: np.zeros(self._state_shape(entry), dtype=np.dtype(entry["dtype"]))
            for entry in self.spec["state"]
        }

    def __call__(self, audio_chunk):
        """audio_chunk: float32 [B, chunk_samples] -> features for those frames."""
        audio = np.ascontiguousarray(audio_chunk, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None]
        if audio.shape[-1] != self.chunk_samples:
            raise ValueError(
                f"expected {self.chunk_samples} samples per chunk, got {audio.shape[-1]}; "
                "zero-pad the last chunk")
        outputs = self.session.run(self.output_names, {"audio": audio, **self.state})
        for entry, value in zip(self.spec["state"], outputs[1:]):
            self.state[entry["name"]] = value
        return outputs[0]


# ---------------------------------------------------------------------------
# verification (shared with the probe exporters)
# ---------------------------------------------------------------------------

def _report(label, a, b, tol):
    """Print and assert max|a - b| / max|b| < tol (relative to the reference's
    scale, since deep-layer activations are large)."""
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError(f"{label}: tolerance must be positive and finite, got {tol}")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape mismatch: actual {a.shape}, reference {b.shape}")
    if a.size == 0:
        raise AssertionError(f"{label}: cannot compare empty arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f"{label}: actual and reference arrays must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        diff = float(np.abs(a - b).max())
    scale = max(float(np.abs(b).max()), 1e-9)
    rel = diff / scale
    passed = np.isfinite(rel) and rel < tol
    status = "ok" if passed else "FAILED"
    print(f"[check] {label:<46} rel diff = {rel:.2e}  "
          f"(abs {diff:.3e}, ref max |x| = {scale:.1f})  {status}")
    if not passed:
        raise AssertionError(
            f"{label}: relative diff {rel:.3e} exceeds tolerance {tol:.1e} "
            f"(abs {diff:.3e} against a reference of scale {scale:.3f})")
    return rel


def reference_features(model, audio, output_hidden_states):
    """The PyTorch model's own whole-sequence forward."""
    with torch.no_grad():
        hidden, all_hidden = model(audio, output_hidden_states=output_hidden_states)
    if output_hidden_states:
        return torch.stack(all_hidden, dim=1).numpy()
    return hidden.numpy()


def check_streaming(model, meta, onnx_path, spec, check_frames, tol, batch_size=None):
    """Stream random audio through the ONNX graph chunk by chunk and compare with
    the PyTorch full pass; also checks that reset() reproduces the first run."""
    batch_size = batch_size or spec["batch_size"]
    chunk_frames = spec["chunk_frames"]
    hop = total_conv_stride(model)
    n_chunks = max(1, -(-check_frames // chunk_frames))
    frames = n_chunks * chunk_frames
    torch.manual_seed(0)
    audio = torch.randn(batch_size, frames * hop)

    print(f"[check] reference: PyTorch full pass over {frames} frames "
          f"({frames * hop / meta['sample_rate']:.1f}s, context={meta['context']} frames, "
          f"batch={batch_size})")
    ref = reference_features(model, audio, spec["output_hidden_states"])

    wrapper = FullPassMERTONNX(model, spec["output_hidden_states"]).eval()
    with torch.no_grad():
        _report("explicit-state rewrite vs PyTorch full pass", wrapper(audio).numpy(), ref, tol)

    streamer = MERTOnnxStreamer(onnx_path, batch_size=batch_size)
    audio_np = audio.numpy()
    outs = [streamer(audio_np[:, c * chunk_frames * hop:(c + 1) * chunk_frames * hop])
            for c in range(n_chunks)]
    axis = 2 if spec["output_hidden_states"] else 1
    streamed = np.concatenate(outs, axis=axis)
    _report(f"ONNX streaming ({n_chunks} x {chunk_frames} frames) vs full pass", streamed, ref, tol)

    streamer.reset()
    again = np.concatenate(
        [streamer(audio_np[:, c * chunk_frames * hop:(c + 1) * chunk_frames * hop])
         for c in range(n_chunks)], axis=axis)
    _report("ONNX streaming after reset() vs first session", again, streamed, 1e-12)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export a streaming MERT backbone (.ckpt) to a streaming ONNX graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint",
                        help="streaming backbone checkpoint (.ckpt) or the run directory holding it")
    # Accepted for older command lines; streaming is the only mode.
    parser.add_argument("--mode", choices=["streaming"], default="streaming",
                        help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None,
                        help="output .onnx path (default: <checkpoint>.streaming.onnx)")
    parser.add_argument("--chunk-frames", type=int, default=None,
                        help="frames per call; a multiple of the lookahead_block "
                             "(default: the lookahead_block)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="leave the batch axis dynamic")
    parser.add_argument("--hidden-states", choices=["all", "final"], default="all",
                        help="all: [B, L, T, D] every layer (what the probes consume); "
                             "final: [B, T, D] only")
    parser.add_argument("--context", type=int, default=None,
                        help="attention context / KV-cache size in frames "
                             "(default: the trained streaming_context)")
    parser.add_argument("--num-attention-heads", type=int, default=None,
                        help="only for a small student whose args.yml is missing (default: 8)")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--no-check", action="store_true",
                        help="skip the numeric comparison against PyTorch")
    parser.add_argument("--check-frames", type=int, default=None,
                        help="sequence length used by the check (default: context + 4 chunks)")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="max difference vs PyTorch, relative to the reference's max |x|")
    parser.add_argument("--no-embed-metadata", action="store_true",
                        help="write only the .json sidecar, do not embed it in the .onnx")
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    print(f"[export] loading {checkpoint_path}")
    model, meta = build_streaming_student(
        checkpoint_path, context=args.context, num_attention_heads=args.num_attention_heads)
    print(f"[export] {meta['student_size']} streaming student: {meta['n_params'] / 1e6:.1f}M params, "
          f"hidden_size={meta['hidden_size']}, layers={meta['num_hidden_layers']}, "
          f"heads={meta['num_attention_heads']}, base={meta['base_mert_model']}")
    print(f"[export] positional_embedding={meta['positional_embedding']}, "
          f"context={meta['context']} frames, lookahead_block={meta['lookahead_block']} frames")
    if meta["positional_embedding"] in ("sin", "sin_rope"):
        print("[export] WARNING: absolute positional embedding; features drift out of "
              "distribution past the frame offsets seen in training. Prefer a 'rope' student.")

    output_hidden_states = args.hidden_states == "all"
    chunk_frames = args.chunk_frames or max(meta["lookahead_block"], 1)
    out_path = args.out or f"{os.path.splitext(checkpoint_path)[0]}.streaming.onnx"
    spec = export_streaming(
        model, meta, out_path, chunk_frames, args.batch_size, args.opset,
        output_hidden_states, args.dynamic_batch, not args.no_embed_metadata)
    state_mb = sum(int(np.prod(s["shape"])) * (8 if s["dtype"] == "int64" else 4)
                   for s in spec["state"]) / 1e6
    print(f"[export] chunk = {chunk_frames} frames = "
          f"{spec['chunk_samples'] / meta['sample_rate'] * 1000:.0f} ms of audio, "
          f"state = {state_mb:.1f} MB")
    if not args.no_check:
        check_frames = args.check_frames or (meta["context"] + 4 * chunk_frames)
        check_streaming(model, meta, out_path, spec, check_frames, args.tolerance)
        if args.dynamic_batch:
            check_streaming(model, meta, out_path, spec, check_frames, args.tolerance,
                            batch_size=args.batch_size + 1)

    print("[export] done")


if __name__ == "__main__":
    main()
