"""
Small bidirectional MERT student for lit_module/mert_causal_distill.py.

Thin-and-deep student in the style of FitHuBERT (https://arxiv.org/abs/2207.00555),
using STaRHuBERT-L's configuration (https://arxiv.org/abs/2312.09040).
Trained from random init by distilling from a full-size MERT teacher.
"""
from typing import Optional, Union

import torch
from torch import nn

from .configuration_MERT import MERTConfig
from .modeling_MERT import MERTModel


MODEL_NAME = "m-a-p/MERT-v1-95M"

# Checkpoint key prefix of the wrapped MERTModel.
WRAPPER_PREFIX = "mert."


# STaRHuBERT-L conv feature extractor (9 layers, total stride 320 like MERT).
_CONV_KERNEL = (10, 1, 3, 3, 3, 3, 1, 2, 2)
_CONV_STRIDE = (5, 1, 2, 2, 2, 2, 1, 2, 2)
_CONV_DIM_PREFIX = (128, 256, 256, 256, 256, 256)  # last three layers use hidden_size


def build_small_mert_config(
    teacher_config: MERTConfig,
    hidden_size: int = 432,
    intermediate_size: int = 1392,
    num_attention_heads: int = 8,
    num_hidden_layers: Optional[int] = None,
) -> MERTConfig:
    """Build a smaller student MERTConfig from the teacher's config."""
    num_hidden_layers = num_hidden_layers if num_hidden_layers is not None else teacher_config.num_hidden_layers
    cfg_dict = teacher_config.to_dict()
    cfg_dict.update(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        conv_dim=list(_CONV_DIM_PREFIX) + [hidden_size] * 3,
        conv_kernel=list(_CONV_KERNEL),
        conv_stride=list(_CONV_STRIDE),
    )
    return MERTConfig(**cfg_dict)


def resolve_base_config(base_config: Union[str, MERTConfig]) -> MERTConfig:
    """Load a MERTConfig from a model name, or pass one through unchanged."""
    if isinstance(base_config, MERTConfig):
        return base_config
    return MERTConfig.from_pretrained(base_config)


class SmallBidirectionalMERTModel(nn.Module):
    """Small MERT student returning (last_hidden_state, all_hidden_states)."""

    def __init__(self, config: MERTConfig):
        super().__init__()
        self.config = config
        self.mert = MERTModel(config)

    def forward(self, input_values: torch.Tensor, output_hidden_states: bool = False):
        outputs = self.mert(input_values, output_hidden_states=output_hidden_states, return_dict=False)
        last_hidden_state = outputs[0]
        all_hidden_states = outputs[1] if output_hidden_states else None
        return last_hidden_state, all_hidden_states


def build_small_mert(
    teacher_model_name: str = MODEL_NAME,
    hidden_size: int = 432,
    intermediate_size: int = 1392,
    num_attention_heads: int = 8,
    num_hidden_layers: Optional[int] = None,
) -> SmallBidirectionalMERTModel:
    """Build a randomly-initialized small MERT student."""
    teacher_config = MERTConfig.from_pretrained(teacher_model_name)
    config = build_small_mert_config(
        teacher_config,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
    )
    model = SmallBidirectionalMERTModel(config)
    model.train()
    return model


if __name__ == "__main__":
    model = build_small_mert()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"small bidirectional MERT student: {n_params / 1e6:.1f}M params")
    print(model.config)

    dummy_input = torch.randn(1, 16000)
    with torch.no_grad():
        last_hidden_state, hidden_states = model(dummy_input, output_hidden_states=True)
    print(f"last_hidden_state shape: {last_hidden_state.shape}")
    print(f"num hidden_states layers: {len(hidden_states)}")
    assert len(hidden_states) == model.config.num_hidden_layers + 1
