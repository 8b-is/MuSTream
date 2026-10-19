"""
Load the pretrained MERT (the distillation teacher) from the local source files,
fixing the legacy weight_norm key names so the positional conv weights load.
"""
import torch
from transformers.utils import cached_file

from .mert.configuration_MERT import MERTConfig
from .mert.modeling_MERT import MERTModel

MODEL_NAME = "m-a-p/MERT-v1-330M"

_LEGACY_TO_PARAMETRIZED_KEYS = {
    "encoder.pos_conv_embed.conv.weight_g": "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "encoder.pos_conv_embed.conv.weight_v": "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
}


def _remap_legacy_weight_norm_keys(state_dict, model_state_dict):
    """Rename legacy weight_g/weight_v keys to the parametrized names if needed."""
    if "encoder.pos_conv_embed.conv.parametrizations.weight.original0" not in model_state_dict:
        return state_dict  # installed torch still uses the legacy weight_norm API
    return {_LEGACY_TO_PARAMETRIZED_KEYS.get(k, k): v for k, v in state_dict.items()}


def load_raw_state_dict(model_name: str = MODEL_NAME):
    """Load the checkpoint's HF-layout state dict."""
    weights_path = cached_file(model_name, "pytorch_model.bin")
    return torch.load(weights_path, map_location="cpu", weights_only=True)


def load_mert(model_name: str = MODEL_NAME) -> MERTModel:
    """Build and load the pretrained MERT in eval mode."""
    config = MERTConfig.from_pretrained(model_name)
    model = MERTModel(config)

    state_dict = _remap_legacy_weight_norm_keys(load_raw_state_dict(model_name), model.state_dict())

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"unexpected state_dict mismatch: missing={missing}, unexpected={unexpected}")

    model.eval()
    return model


if __name__ == "__main__":
    model = load_mert()
    print(model.config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {MODEL_NAME}: {n_params / 1e6:.1f}M params")

    dummy_input = torch.randn(1, 16000)
    with torch.no_grad():
        out = model(dummy_input, output_hidden_states=True)
    print(f"last_hidden_state shape: {out.last_hidden_state.shape}")
    print(f"num hidden_states layers: {len(out.hidden_states)}")
