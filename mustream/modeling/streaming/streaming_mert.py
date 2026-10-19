"""
Streaming MERT architecture (MERT-95M or MERT-330M) built on moshi's streaming
modules: a causal conv feature extractor and a KV-cached streaming transformer for
incremental, frame-by-frame inference.

Pretrained weights are loaded by modeling/init_streaming.py; trained students
are rebuilt by modeling/load_checkpoint.py.
"""
import torch
from torch import nn
from transformers.activations import ACT2FN

from ..mert.configuration_MERT import MERTConfig
from ..mert.modeling_MERT import MERTFeatureProjection
from ..moshi.modules.conv import StreamingConv1d
from ..moshi.modules.streaming import StreamingContainer
from ..moshi.modules.transformer import (
    StreamingMultiheadAttention,
    StreamingTransformerLayer,
    StreamingTransformer,
    apply_weights_per_step,
    set_attention_lookahead_block,
)



class CausalHubertFeatureEncoder(StreamingContainer):
    """Causal conv feature extractor (MERT's conv stack on StreamingConv1d)."""

    def __init__(self, config: MERTConfig):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            StreamingConv1d(
                in_channels=1 if i == 0 else config.conv_dim[i - 1],
                out_channels=config.conv_dim[i],
                kernel_size=config.conv_kernel[i],
                stride=config.conv_stride[i],
                bias=config.conv_bias,
                causal=True,
                norm="none",
            )
            for i in range(config.num_feat_extract_layers)
        ])
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, input_values):
        hidden_states = input_values[:, None]
        for conv in self.conv_layers:
            hidden_states = self.activation(conv(hidden_states))
        return hidden_states


class BiasedStreamingMultiheadAttention(StreamingMultiheadAttention):
    """StreamingMultiheadAttention with biased projections, to load MERT's biases."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_kv = self.num_heads // self.kv_repeat
        kv_dim = (self.embed_dim // self.num_heads) * num_kv
        out_dim = self.embed_dim + 2 * kv_dim
        device = self.in_projs[0].weight.device
        dtype = self.in_projs[0].weight.dtype
        self.in_projs = nn.ModuleList([
            nn.Linear(self.embed_dim, out_dim, bias=True, device=device, dtype=dtype)
            for _ in range(self.mult)
        ])
        self.out_projs = nn.ModuleList([
            nn.Linear(self.embed_dim, self.embed_dim, bias=True, device=device, dtype=dtype)
            for _ in range(self.mult)
        ])


class BiasedStreamingTransformerLayer(StreamingTransformerLayer):
    """Pre-norm streaming transformer layer with biases, matching MERT-330M."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.skip_self_attn
        assert self.linear1 is not None and self.linear2 is not None, "expected gating='none'"
        device = self.linear1.weight.device
        dtype = self.linear1.weight.dtype
        d_model = self.linear1.in_features
        dim_feedforward = self.linear1.out_features
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=True, device=device, dtype=dtype)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=True, device=device, dtype=dtype)

        attn = self.self_attn
        self.self_attn = BiasedStreamingMultiheadAttention(
            embed_dim=attn.embed_dim,
            num_heads=attn.num_heads,
            causal=attn.causal,
            context=attn.context,
            rope=attn.rope,
            kv_repeat=attn.kv_repeat,
            weights_per_step=attn.weights_per_step,
            weights_per_step_schedule=attn.weights_per_step_schedule,
            device=device,
            dtype=dtype,
        )


class PostNormBiasedStreamingTransformerLayer(BiasedStreamingTransformerLayer):
    """Post-norm streaming transformer layer with biases, matching MERT-95M."""

    def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip_self_attn:
            return x
        update = self.self_attn(x, x, x)
        x = x.to(update) + self.layer_scale_1(update)
        return self.norm1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        state = self._streaming_state
        offset = 0
        if state is not None:
            offset = state.offset_cpu
        if self.gating is None:
            assert self.linear1 is not None
            assert self.linear2 is not None
            update = self.linear2(self.activation(self.linear1(x)))
        else:
            if self.weights_per_step:
                assert isinstance(self.gating, nn.ModuleList)
                update = apply_weights_per_step(self.gating, self.weights_per_step_schedule, x, offset)
            else:
                update = self.gating(x)
        x = x.to(update) + self.layer_scale_2(update)
        return self.norm2(x)


class FullStreamingMERTModel(nn.Module):
    """
    Streaming MERT model returning (last_hidden_state, all_hidden_states).

    context:              attention window in frames (1000 frames ~ 13.3s at 75 fps).
    positional_embedding: 'rope' (default, relative) keeps long sessions in distribution;
                          must match the scheme a loaded checkpoint was trained with.
    lookahead_block:      >1 lets each frame attend to the rest of its block.
    """

    POSITIONAL_EMBEDDINGS = ("sin", "rope", "sin_rope", "none")

    def __init__(
        self,
        config: MERTConfig,
        context: int = 1000,
        positional_embedding: str = "rope",
        lookahead_block: int = 1,
    ):
        super().__init__()
        if positional_embedding not in self.POSITIONAL_EMBEDDINGS:
            raise ValueError(
                f"invalid positional_embedding {positional_embedding!r}, "
                f"expected one of {self.POSITIONAL_EMBEDDINGS}"
            )
        assert lookahead_block >= 1, lookahead_block
        self.config = config
        self.positional_embedding = positional_embedding
        self.lookahead_block = lookahead_block
        self.feature_extractor = CausalHubertFeatureEncoder(config)
        self.feature_projection = MERTFeatureProjection(config)
        layer_class = (
            BiasedStreamingTransformerLayer
            if config.do_stable_layer_norm
            else PostNormBiasedStreamingTransformerLayer
        )
        self.transformer = StreamingTransformer(
            d_model=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
            dim_feedforward=config.intermediate_size,
            causal=True,
            context=context,
            positional_embedding=positional_embedding,
            layer_class=layer_class,
        )
        if lookahead_block > 1:
            # Frames attend ahead to the end of their block (lookahead_block - 1 frames).
            set_attention_lookahead_block(self.transformer, lookahead_block)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_values, output_hidden_states=False):
        extract_features = self.feature_extractor(input_values).transpose(1, 2)
        hidden_states = self.feature_projection(extract_features)
        if not self.config.do_stable_layer_norm:
            # Post-norm (MERT-95M): encoder layer norm goes before the layer stack.
            hidden_states = self.final_layer_norm(hidden_states)
        if output_hidden_states:
            hidden_states, all_hidden_states = self.transformer(hidden_states, output_hidden_states=True)
        else:
            hidden_states = self.transformer(hidden_states)
            all_hidden_states = None
        if self.config.do_stable_layer_norm:
            hidden_states = self.final_layer_norm(hidden_states)  # pre-norm: after the stack
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return hidden_states, all_hidden_states


# State-dict prefix that marks a streaming checkpoint (see load_checkpoint.is_streaming_layout).
STREAMING_LAYER_PREFIX = "transformer.layers."
