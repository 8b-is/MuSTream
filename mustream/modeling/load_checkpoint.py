"""
Load train_mert_causal_distill.py checkpoints: read the student weights and run
metadata, and rebuild distilled small students (bidirectional or streaming).
"""
import os
from typing import Optional, Union

import torch
import yaml

from .init_streaming import load_remapped
from .mert.configuration_MERT import MERTConfig
from .mert.small_mert import (
    WRAPPER_PREFIX,
    SmallBidirectionalMERTModel,
    build_small_mert_config,
    resolve_base_config,
)
from .streaming.streaming_mert import STREAMING_LAYER_PREFIX, FullStreamingMERTModel

STUDENT_PREFIX = "model."
COMPILE_PREFIX = "_orig_mod."
ARGS_FILENAME = "args.yml"


class _ArgsLoader(yaml.SafeLoader):
    """YAML SafeLoader that also accepts !!python/tuple."""


_ArgsLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def load_align_checkpoint(checkpoint_path: str) -> dict:
    """Load a Lightning checkpoint on CPU."""
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def align_student_state_dict(ckpt: dict, checkpoint_path: str = "") -> dict:
    """Extract the student's state dict from a Lightning checkpoint."""
    lit_state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    student_state_dict = {}
    for key, value in lit_state_dict.items():
        if not key.startswith(STUDENT_PREFIX):
            continue
        key = key[len(STUDENT_PREFIX):]
        if key.startswith(COMPILE_PREFIX):
            key = key[len(COMPILE_PREFIX):]
        student_state_dict[key] = value

    if not student_state_dict:
        raise RuntimeError(
            f"no '{STUDENT_PREFIX}*' keys found in {checkpoint_path or 'this checkpoint'} "
            "-- is it a train_mert_causal_distill.py checkpoint?"
        )
    return student_state_dict


def load_align_student_state_dict(checkpoint_path: str) -> dict:
    """Load a checkpoint and return only the student's state dict."""
    return align_student_state_dict(load_align_checkpoint(checkpoint_path), checkpoint_path)


def align_positional_embedding(ckpt: dict) -> str:
    """Positional embedding the streaming student was trained with (untagged checkpoints predate rope)."""
    return ckpt.get("student_positional_embedding", "sin")


def align_lookahead_block(ckpt: dict) -> int:
    """Lookahead block size the streaming student was trained with (1 = strictly causal)."""
    return int(ckpt.get("student_lookahead_block", 1))


def read_align_run_args(checkpoint_path: str) -> dict:
    """Read the run's args.yml next to the checkpoint; {} if unavailable."""
    args_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), ARGS_FILENAME)
    if not os.path.isfile(args_path):
        return {}
    try:
        with open(args_path) as f:
            args = yaml.load(f, Loader=_ArgsLoader)
    except Exception as exc:      # a malformed record must not break a load
        print(f"[load_checkpoint] could not read {args_path}: {exc}")
        return {}
    return args if isinstance(args, dict) else {}


def align_base_model(ckpt: dict, checkpoint_path: str):
    """Published MERT the run was based on, or None if unrecorded."""
    tagged = ckpt.get("mert_model")
    if tagged:
        return tagged
    return read_align_run_args(checkpoint_path).get("mert_model")


# ---------------------------------------------------------------------------
# Rebuilding trained students
# ---------------------------------------------------------------------------

def small_mert_hf_state_dict(student_state_dict: dict) -> dict:
    """Strip the 'mert.' prefix from a small bidirectional student's state dict."""
    stripped = {
        key[len(WRAPPER_PREFIX):]: value
        for key, value in student_state_dict.items()
        if key.startswith(WRAPPER_PREFIX)
    }
    if len(stripped) != len(student_state_dict):
        others = sorted(k for k in student_state_dict if not k.startswith(WRAPPER_PREFIX))
        raise RuntimeError(
            f"expected every key to start with {WRAPPER_PREFIX!r} (a "
            f"student_model: small_bidirectional checkpoint), but found {len(others)} that "
            f"do not, e.g. {others[:5]} -- wrong checkpoint?"
        )
    return stripped


def infer_small_mert_arch(hf_state_dict: dict) -> dict:
    """Infer hidden_size, intermediate_size and num_hidden_layers from bidirectional weights."""
    layer_ids = {
        int(key.split(".")[2]) for key in hf_state_dict if key.startswith("encoder.layers.")
    }
    if not layer_ids:
        raise RuntimeError("no 'encoder.layers.*' keys found -- not a bidirectional MERT state dict?")
    q_proj = hf_state_dict["encoder.layers.0.attention.q_proj.weight"]
    intermediate = hf_state_dict["encoder.layers.0.feed_forward.intermediate_dense.weight"]
    return {
        "hidden_size": q_proj.shape[0],
        "intermediate_size": intermediate.shape[0],
        "num_hidden_layers": max(layer_ids) + 1,
    }


def check_base_model(recorded: Optional[str], base_config: Union[str, MERTConfig], source: str) -> None:
    """Raise if the base MERT differs from the one the checkpoint was trained with."""
    if recorded is None or not isinstance(base_config, str) or recorded == base_config:
        return
    raise RuntimeError(
        f"{source} was distilled from {recorded!r}, but this run's mert_model is "
        f"{base_config!r}. mert_model supplies the config a rebuilt small model inherits -- "
        "most consequentially do_stable_layer_norm (MERT-v1-330M is pre-norm, MERT-v1-95M "
        "post-norm), which no parameter shape records, so the mismatch would load every "
        f"weight and then compute the wrong function. Set mert_model: {recorded!r}."
    )


def check_expected_arch(arch: dict, expected: Optional[dict], source: str) -> None:
    """Raise if the inferred architecture differs from the configured one."""
    if not expected:
        return
    mismatched = {
        key: (value, arch.get(key))
        for key, value in expected.items()
        if value is not None and arch.get(key) != value
    }
    if mismatched:
        detail = ", ".join(f"{k}: config says {want}, checkpoint has {got}" for k, (want, got) in mismatched.items())
        raise RuntimeError(f"{source} does not match the configured architecture -- {detail}")


def load_small_bidirectional_from_align_checkpoint(
    checkpoint_path: str,
    base_config: Union[str, MERTConfig],
    num_attention_heads: int = 8,
    expected_arch: Optional[dict] = None,
) -> SmallBidirectionalMERTModel:
    """Rebuild a distilled small bidirectional student (e.g. as a teacher)."""
    ckpt = load_align_checkpoint(checkpoint_path)
    student_state_dict = align_student_state_dict(ckpt, checkpoint_path)
    check_base_model(align_base_model(ckpt, checkpoint_path), base_config, checkpoint_path)
    hf_state_dict = small_mert_hf_state_dict(student_state_dict)
    arch = infer_small_mert_arch(hf_state_dict)
    check_expected_arch(arch, expected_arch, checkpoint_path)

    config = build_small_mert_config(
        resolve_base_config(base_config),
        hidden_size=arch["hidden_size"],
        intermediate_size=arch["intermediate_size"],
        num_attention_heads=num_attention_heads,
        num_hidden_layers=arch["num_hidden_layers"],
    )
    model = SmallBidirectionalMERTModel(config)
    model.load_state_dict(student_state_dict, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"loaded small bidirectional MERT from {checkpoint_path}: {n_params / 1e6:.1f}M params, "
        f"hidden_size={arch['hidden_size']}, intermediate_size={arch['intermediate_size']}, "
        f"num_hidden_layers={arch['num_hidden_layers']}, num_attention_heads={num_attention_heads}"
    )
    model.eval()
    return model


def is_streaming_layout(state_dict) -> bool:
    """Whether a state dict is in FullStreamingMERTModel's layout."""
    return any(key.startswith(STREAMING_LAYER_PREFIX) for key in state_dict)


def infer_small_streaming_arch(state_dict) -> dict:
    """Infer hidden_size, intermediate_size and num_hidden_layers from streaming weights."""
    layer_ids = {
        int(key.split(".")[2]) for key in state_dict if key.startswith(STREAMING_LAYER_PREFIX)
    }
    if not layer_ids:
        raise RuntimeError(
            f"no '{STREAMING_LAYER_PREFIX}*' keys found -- not a streaming student's state dict?"
        )
    return {
        "hidden_size": state_dict["feature_projection.projection.weight"].shape[0],
        "intermediate_size": state_dict[f"{STREAMING_LAYER_PREFIX}0.linear1.weight"].shape[0],
        "num_hidden_layers": max(layer_ids) + 1,
    }


def check_positional_embedding(trained_with: str, requested: str, source: str) -> None:
    """Raise if the positional embedding differs from the one used in training."""
    if trained_with == requested:
        return
    raise RuntimeError(
        f"{source} was distilled with positional_embedding={trained_with!r} but the backbone "
        f"is being built with {requested!r}. These are different functions of the same weights "
        f"(see modeling/streaming/streaming_mert.py); set backbone_positional_embedding: {trained_with!r} "
        f"in the config, or distill a new checkpoint with "
        f"student_positional_embedding: {requested!r}."
    )


def load_small_streaming_mert_from_align_checkpoint(
    checkpoint_path: str,
    base_config,
    num_attention_heads: int = 8,
    context: int = 1000,
    positional_embedding: str = "rope",
    expected_arch=None,
    lookahead_block: int = 1,
):
    """
    Build a small streaming student from a checkpoint:
      - small bidirectional checkpoint: remapped as an initialization;
      - small streaming checkpoint: loaded as-is.
    """
    ckpt = load_align_checkpoint(checkpoint_path)
    student_state_dict = align_student_state_dict(ckpt, checkpoint_path)
    check_base_model(align_base_model(ckpt, checkpoint_path), base_config, checkpoint_path)

    already_streaming = is_streaming_layout(student_state_dict)
    if already_streaming:
        check_positional_embedding(
            align_positional_embedding(ckpt), positional_embedding, checkpoint_path,
        )
        # Only a finished streaming student has a lookahead block to match.
        trained_block = align_lookahead_block(ckpt)
        if trained_block != lookahead_block:
            raise RuntimeError(
                f"{checkpoint_path} was distilled with student_lookahead_block="
                f"{trained_block} but is being rebuilt with {lookahead_block}. The mask "
                "is not recorded in any weight, so this would load every parameter and "
                f"then compute a different function; use {trained_block}."
            )
        arch = infer_small_streaming_arch(student_state_dict)
    else:
        student_state_dict = small_mert_hf_state_dict(student_state_dict)
        arch = infer_small_mert_arch(student_state_dict)
    check_expected_arch(arch, expected_arch, checkpoint_path)

    config = build_small_mert_config(
        resolve_base_config(base_config),
        hidden_size=arch["hidden_size"],
        intermediate_size=arch["intermediate_size"],
        num_attention_heads=num_attention_heads,
        num_hidden_layers=arch["num_hidden_layers"],
    )
    model = FullStreamingMERTModel(
        config, context=context, positional_embedding=positional_embedding,
        lookahead_block=lookahead_block,
    )
    if already_streaming:
        model.load_state_dict(student_state_dict, strict=True)
    else:
        load_remapped(model, student_state_dict, config)
    n_params = sum(p.numel() for p in model.parameters())
    source = "loaded from" if already_streaming else "initialized from"
    print(
        f"small streaming student {source} {checkpoint_path}: {n_params / 1e6:.1f}M params, "
        f"hidden_size={arch['hidden_size']}, intermediate_size={arch['intermediate_size']}, "
        f"num_hidden_layers={arch['num_hidden_layers']}, num_attention_heads={num_attention_heads}, "
        f"positional_embedding={positional_embedding}, context={context} frames"
    )
    model.eval()
    return model


def load_streaming_mert(
    checkpoint_path: str,
    num_attention_heads: int = 8,
    context: Optional[int] = None,
):
    """
    Rebuild a streaming MERT backbone (any size) from a causal-distillation checkpoint,
    using the settings recorded in it: base MERT, positional embedding, lookahead block
    and streaming context. Returns the model in eval mode on the CPU.

    num_attention_heads: 26M student only (not recorded in the weights; 8 for the released one).
    context:             KV-cache window in frames; default = the trained streaming_context.
    """
    ckpt = load_align_checkpoint(checkpoint_path)
    state_dict = align_student_state_dict(ckpt, checkpoint_path)
    if not is_streaming_layout(state_dict):
        raise RuntimeError(f"{checkpoint_path} does not hold a streaming backbone")
    base_model = align_base_model(ckpt, checkpoint_path)
    if not base_model:
        raise RuntimeError(f"{checkpoint_path} records no base MERT (`mert_model`)")
    if context is None:
        context = int(ckpt.get("streaming_context")
                      or read_align_run_args(checkpoint_path).get("streaming_context", 750))
    positional_embedding = align_positional_embedding(ckpt)
    lookahead_block = align_lookahead_block(ckpt)

    base_config = MERTConfig.from_pretrained(base_model)
    arch = infer_small_streaming_arch(state_dict)
    is_full_width = (arch["hidden_size"] == base_config.hidden_size
                     and arch["num_hidden_layers"] == base_config.num_hidden_layers)
    if is_full_width:
        config = base_config
    else:
        config = build_small_mert_config(
            base_config,
            hidden_size=arch["hidden_size"],
            intermediate_size=arch["intermediate_size"],
            num_attention_heads=num_attention_heads,
            num_hidden_layers=arch["num_hidden_layers"],
        )
    model = FullStreamingMERTModel(
        config, context=context, positional_embedding=positional_embedding,
        lookahead_block=lookahead_block,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
