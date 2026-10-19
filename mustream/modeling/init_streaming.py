"""
Initialize the streaming backbone (streaming/streaming_mert.py) from a
bidirectional MERT's weights, or build a small streaming student.
"""
import torch

from .load_mert import MODEL_NAME, load_raw_state_dict
from .mert.configuration_MERT import MERTConfig
from .mert.small_mert import build_small_mert_config, resolve_base_config
from .streaming.streaming_mert import FullStreamingMERTModel


def remap_feature_extractor_keys(state_dict, num_layers):
    """Remap conv weights into StreamingConv1d's layout; drop the GroupNorm and any extra layers."""
    remapped = {}
    dropped = []
    for key, value in state_dict.items():
        if not key.startswith("feature_extractor.conv_layers."):
            remapped[key] = value
            continue
        parts = key.split(".")
        layer_id = int(parts[2])
        if layer_id >= num_layers:
            dropped.append(key)
            continue
        if parts[3] == "layer_norm":
            dropped.append(key)
            continue
        assert parts[3] == "conv" and parts[4] in ("weight", "bias"), key
        new_key = f"feature_extractor.conv_layers.{layer_id}.conv.conv.{parts[4]}"
        remapped[new_key] = value
    return remapped, dropped


def _pack_qkv(state_dict, layer_id):
    """Concatenate q/k/v weights and biases into moshi's single in_proj."""
    prefix = f"encoder.layers.{layer_id}.attention"
    q_w = state_dict[f"{prefix}.q_proj.weight"]
    k_w = state_dict[f"{prefix}.k_proj.weight"]
    v_w = state_dict[f"{prefix}.v_proj.weight"]
    q_b = state_dict[f"{prefix}.q_proj.bias"]
    k_b = state_dict[f"{prefix}.k_proj.bias"]
    v_b = state_dict[f"{prefix}.v_proj.bias"]
    return torch.cat([q_w, k_w, v_w], dim=0), torch.cat([q_b, k_b, v_b], dim=0)


def remap_transformer_layer_keys(state_dict, num_layers):
    """Remap HF encoder layers into the streaming transformer's layout."""
    remapped = {}
    remapped["final_layer_norm.weight"] = state_dict["encoder.layer_norm.weight"]
    remapped["final_layer_norm.bias"] = state_dict["encoder.layer_norm.bias"]

    for i in range(num_layers):
        src = f"encoder.layers.{i}"
        dst = f"transformer.layers.{i}"

        in_proj_w, in_proj_b = _pack_qkv(state_dict, i)
        remapped[f"{dst}.self_attn.in_projs.0.weight"] = in_proj_w
        remapped[f"{dst}.self_attn.in_projs.0.bias"] = in_proj_b
        remapped[f"{dst}.self_attn.out_projs.0.weight"] = state_dict[f"{src}.attention.out_proj.weight"]
        remapped[f"{dst}.self_attn.out_projs.0.bias"] = state_dict[f"{src}.attention.out_proj.bias"]

        remapped[f"{dst}.linear1.weight"] = state_dict[f"{src}.feed_forward.intermediate_dense.weight"]
        remapped[f"{dst}.linear1.bias"] = state_dict[f"{src}.feed_forward.intermediate_dense.bias"]
        remapped[f"{dst}.linear2.weight"] = state_dict[f"{src}.feed_forward.output_dense.weight"]
        remapped[f"{dst}.linear2.bias"] = state_dict[f"{src}.feed_forward.output_dense.bias"]

        remapped[f"{dst}.norm1.weight"] = state_dict[f"{src}.layer_norm.weight"]
        remapped[f"{dst}.norm1.bias"] = state_dict[f"{src}.layer_norm.bias"]
        remapped[f"{dst}.norm2.weight"] = state_dict[f"{src}.final_layer_norm.weight"]
        remapped[f"{dst}.norm2.bias"] = state_dict[f"{src}.final_layer_norm.bias"]

    return remapped


def remap_bidirectional_state_dict(state_dict, config: MERTConfig):
    """Remap an HF-layout MERT state dict to FullStreamingMERTModel; also return dropped keys."""
    conv_remapped, dropped_conv_keys = remap_feature_extractor_keys(state_dict, config.num_feat_extract_layers)
    feature_extractor_keys = {k: v for k, v in conv_remapped.items() if k.startswith("feature_extractor.")}
    feature_projection_keys = {k: v for k, v in conv_remapped.items() if k.startswith("feature_projection.")}
    transformer_keys = remap_transformer_layer_keys(state_dict, config.num_hidden_layers)

    dropped_other = [
        k for k in state_dict
        if k.startswith("encoder.pos_conv_embed.") or k == "masked_spec_embed"
    ]
    remapped = {**feature_extractor_keys, **feature_projection_keys, **transformer_keys}
    return remapped, dropped_conv_keys, dropped_other


def load_remapped(model: FullStreamingMERTModel, state_dict, config: MERTConfig, verbose: bool = True):
    """Load remapped bidirectional weights into `model`, failing on any missing or unexpected key."""
    remapped, dropped_conv_keys, dropped_other = remap_bidirectional_state_dict(state_dict, config)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if verbose:
        print(f"dropped (expected, GroupNorm affine): {dropped_conv_keys}")
        print(f"dropped (expected, obsolete under this architecture): {dropped_other}")
    if missing or unexpected:
        raise RuntimeError(f"unexpected state_dict mismatch: missing={missing}, unexpected={unexpected}")
    return model


def load_full_streaming_mert(
    mert_model_name: str = MODEL_NAME,
    context: int = 1000,
    positional_embedding: str = "rope",
    lookahead_block: int = 1,
):
    """Build the streaming model initialized from a published MERT."""
    config = MERTConfig.from_pretrained(mert_model_name)
    model = FullStreamingMERTModel(
        config, context=context, positional_embedding=positional_embedding,
        lookahead_block=lookahead_block,
    )
    load_remapped(model, load_raw_state_dict(mert_model_name), config)
    model.eval()
    return model


def build_small_streaming_mert(
    base_config,
    hidden_size: int = 432,
    intermediate_size: int = 1392,
    num_attention_heads: int = 8,
    num_hidden_layers=None,
    context: int = 1000,
    positional_embedding: str = "rope",
    lookahead_block: int = 1,
):
    """Build a randomly-initialized small streaming student.

    `base_config` must be the MERT family the student is paired with (it sets pre- vs post-norm).
    """
    config = build_small_mert_config(
        resolve_base_config(base_config),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
    )
    return FullStreamingMERTModel(
        config, context=context, positional_embedding=positional_embedding,
        lookahead_block=lookahead_block,
    )
