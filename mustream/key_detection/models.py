"""
Key-detection probe heads and frozen-backbone loading / feature extraction.

Every head is prefix-causal with two readouts:
    logits  = head(x)                  # [B, n_classes]     segment prediction
    running = head(x, running=True)    # [B, T, n_classes]  running[:, t] uses frames 0..t
and head(x) == head(x, running=True)[:, -1].

  KeyMLP          cumulative-mean pooling -> MLP
  KeyLSTM         unidirectional LSTM -> per-frame linear head
  KeyTransformer  causal StreamingTransformer -> per-frame linear head

Self-test: python models.py
"""

import os
import sys
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from modeling.moshi.modules.transformer import StreamingTransformer


def select_mert_layer(x: torch.Tensor, mert_layer: Union[int, str]) -> torch.Tensor:
    """[B, n_layers, T, D] -> [B, T, D]."""
    n_layers = x.shape[1]

    if mert_layer == 'average':      # mean over all layers
        return x.mean(dim=1)
    elif mert_layer == 'half':       # upper half
        return x[:, n_layers // 2:, :, :].mean(dim=1)
    elif mert_layer == 'low':
        return x[:, : n_layers // 3, :, :].mean(dim=1)
    elif mert_layer == 'middle':
        return x[:, n_layers // 3 : 2 * n_layers // 3, :, :].mean(dim=1)
    elif mert_layer == 'high':
        return x[:, 2 * n_layers // 3 :, :, :].mean(dim=1)
    elif mert_layer == 'top':
        return x[:, -4:, :, :].mean(dim=1)
    elif isinstance(mert_layer, int):
        return x[:, mert_layer, :, :]
    else:
        raise ValueError('invalid mert layer {}'.format(mert_layer))


# ---------------------------------------------------------------------------
# Probe heads
# ---------------------------------------------------------------------------

class _KeyHead(nn.Module):
    """Shared front end and readout/streaming contract; subclasses implement _running()."""

    def __init__(
        self,
        mert_layer: Union[int, str] = 'average',
    ):
        super().__init__()
        self.mert_layer = mert_layer

    # -- subclass hooks -----------------------------------------------------
    def _running(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _enter_streaming(self, batch_size: int) -> None:
        pass

    def _exit_streaming(self) -> None:
        pass

    # -- public API ---------------------------------------------------------
    def forward(self, x: torch.Tensor, running: bool = False) -> torch.Tensor:
        """x: [B, n_layers, T, D] -> [B, n_classes], or [B, T, n_classes] if running=True."""
        features = select_mert_layer(x, self.mert_layer)                     # [B, T, D]
        logits = self._running(features)                                     # [B, T, n_classes]
        return logits if running else logits[:, -1]

    def running_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward(x, running=True)."""
        return self.forward(x, running=True)

    @contextmanager
    def streaming(self, batch_size: int):
        """Carry head state across forward calls, so chunk-by-chunk input matches one full pass."""
        self._enter_streaming(batch_size)
        try:
            yield self
        finally:
            self._exit_streaming()


class KeyMLP(_KeyHead):
    """Cumulative-mean pooling -> MLP -> key logits. hidden_dim=None gives a linear probe."""

    def __init__(
        self,
        n_classes: int,
        emb_dim: int = 1024,
        hidden_dim: Optional[int] = 512,
        mert_layer: Union[int, str] = 'average',
    ):
        super().__init__(mert_layer)
        if hidden_dim is None:
            self.head = nn.Linear(emb_dim, n_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_classes),
            )
        self._sum: Optional[torch.Tensor] = None
        self._count = 0
        self._session = False

    def _enter_streaming(self, batch_size: int) -> None:
        self._sum, self._count, self._session = None, 0, True

    def _exit_streaming(self) -> None:
        self._sum, self._count, self._session = None, 0, False

    def _running(self, features: torch.Tensor) -> torch.Tensor:
        T = features.shape[1]
        cumsum = features.cumsum(dim=1)                                       # [B, T, D]
        offset = 0
        if self._session:
            if self._sum is not None:
                cumsum = cumsum + self._sum.unsqueeze(1)
            offset = self._count
            self._sum = cumsum[:, -1].detach()
            self._count = offset + T
        counts = torch.arange(
            offset + 1, offset + T + 1, device=features.device, dtype=features.dtype)
        pooled = cumsum / counts.view(1, T, 1)                                # [B, T, D]
        return self.head(pooled)                                              # [B, T, n_classes]


class KeyLSTM(_KeyHead):
    """Unidirectional LSTM -> per-frame linear head -> key logits."""

    def __init__(
        self,
        n_classes: int,
        emb_dim: int = 1024,
        mert_layer: Union[int, str] = 'average',
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__(mert_layer)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, n_classes)
        self._state = None
        self._session = False

    def _enter_streaming(self, batch_size: int) -> None:
        self._state, self._session = None, True

    def _exit_streaming(self) -> None:
        self._state, self._session = None, False

    def _running(self, features: torch.Tensor) -> torch.Tensor:
        out, state = self.lstm(features, self._state if self._session else None)
        if self._session:
            self._state = tuple(s.detach() for s in state)
        return self.head(out)                                                 # [B, T, n_classes]


class KeyTransformer(_KeyHead):
    """Causal StreamingTransformer -> per-frame linear head -> key logits.

    context: attention lookback in frames (None = unbounded). Must be finite at
    training time to use streaming().
    """

    def __init__(
        self,
        n_classes: int,
        emb_dim: int = 1024,
        mert_layer: Union[int, str] = 'average',
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        context: Optional[int] = None,
        positional_embedding: str = "rope",
    ):
        super().__init__(mert_layer)
        self.input_proj = nn.Identity() if d_model == emb_dim else nn.Linear(emb_dim, d_model)
        self.context = context
        self.transformer = StreamingTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            causal=True,
            context=context,
            positional_embedding=positional_embedding,
        )
        self.head = nn.Linear(d_model, n_classes)
        self._streaming_cm = None

    def _enter_streaming(self, batch_size: int) -> None:
        if self.context is None:
            raise RuntimeError(
                "KeyTransformer.streaming() needs a finite `context`: moshi sizes its "
                "RingKVCache from it and cannot infer a capacity otherwise (see "
                "moshi/modules/transformer.py). Set transformer_context in the config to a "
                "frame count and retrain -- a context=None checkpoint computes unbounded "
                "within-segment attention, which is a different function from the "
                "ring-cached bounded attention streaming would give it."
            )
        self._streaming_cm = self.transformer.streaming(batch_size)
        self._streaming_cm.__enter__()

    def _exit_streaming(self) -> None:
        if self._streaming_cm is not None:
            self._streaming_cm.__exit__(None, None, None)
            self._streaming_cm = None

    def _running(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features)                                         # [B, T, d_model]
        x = self.transformer(x)                                               # [B, T, d_model]
        return self.head(x)                                                   # [B, T, n_classes]


def build_classifier(model_type: str, n_classes: int, config: dict) -> nn.Module:
    """Build KeyMLP / KeyLSTM / KeyTransformer from a config ('MLP' | 'LSTM' | 'Transformer')."""
    shared = dict(
        n_classes=n_classes,
        emb_dim=config["emb_dim"],
        mert_layer=config["mert_layer"],
    )
    if model_type == "MLP":
        return KeyMLP(hidden_dim=config.get("hidden_dim", 512), **shared)
    elif model_type == "LSTM":
        return KeyLSTM(
            hidden_dim=config.get("lstm_hidden_dim", 256),
            num_layers=config.get("lstm_num_layers", 2),
            dropout=config.get("lstm_dropout", 0.0),
            **shared,
        )
    elif model_type == "Transformer":
        return KeyTransformer(
            d_model=config.get("transformer_d_model", 256),
            num_heads=config.get("transformer_num_heads", 4),
            num_layers=config.get("transformer_num_layers", 4),
            dim_feedforward=config.get("transformer_dim_feedforward", 1024),
            context=config.get("transformer_context"),
            positional_embedding=config.get("transformer_positional_embedding", "rope"),
            **shared,
        )
    else:
        raise ValueError(f"invalid model_type {model_type!r}, expected 'MLP', 'LSTM', or 'Transformer'")


# ---------------------------------------------------------------------------
# Training readout
# ---------------------------------------------------------------------------

TRAIN_READOUTS = ("final", "running")


def resolve_train_readout(config: dict) -> str:
    """Validated config['train_readout']: 'final' (default, CE on the segment logit)
    or 'running' (mean CE over every prefix readout)."""
    readout = config.get("train_readout") or "final"
    if readout not in TRAIN_READOUTS:
        raise ValueError(
            f"invalid train_readout {readout!r}, expected one of {TRAIN_READOUTS}")
    return readout


def readout_loss(
    classifier: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    train_readout: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(loss, segment_logits [B, n_classes]) for one batch under the given readout."""
    if train_readout == "final":
        logits = classifier(embeddings)                                       # [B, n_classes]
        return criterion(logits, labels), logits

    running = classifier(embeddings, running=True)                            # [B, T, n_classes]
    B, T, n_classes = running.shape
    loss = criterion(
        running.reshape(B * T, n_classes),
        labels.unsqueeze(1).expand(B, T).reshape(B * T),
    )
    return loss, running[:, -1]


# ---------------------------------------------------------------------------
# Offline (bidirectional) backbones
# ---------------------------------------------------------------------------

def extract_MERT_features(
    mert: nn.Module,
    processor: Wav2Vec2FeatureExtractor,
    audio: torch.Tensor,
    pool_factor: int = 5,
) -> torch.Tensor:
    """Frozen bidirectional backbone features, [B, L, T // pool_factor, D].

    Works for HF MERT and SmallBidirectionalMERTModel; `audio` must be on the backbone's device.
    """
    B = audio.shape[0]
    input_values = audio[:, 0, :]   # [B, T_samples]
    if processor.do_normalize:
        mean = input_values.mean(dim=-1, keepdim=True)
        var = input_values.var(dim=-1, unbiased=False, keepdim=True)
        input_values = (input_values - mean) / torch.sqrt(var + 1e-7)

    with torch.no_grad():
        outputs = mert(input_values=input_values, output_hidden_states=True)
    hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs[1]

    hidden = torch.stack(hidden_states).permute(1, 0, 2, 3)   # [B, L, T, D]
    _, n_layers, _, hidden_dim = hidden.shape

    T = hidden.shape[2]
    T_trim = (T // pool_factor) * pool_factor
    hidden = hidden[:, :, :T_trim, :]
    T_reduced = T_trim // pool_factor
    hidden = hidden.reshape(B, n_layers, T_reduced, pool_factor, hidden_dim).mean(3)
    return hidden                                              # [B, L, T_reduced, D]


def load_student_state_dict_from_align_checkpoint(
    checkpoint_path: str,
    expect_positional_embedding: Optional[str] = None,
) -> dict:
    """Student backbone weights from a train_mert_causal_distill.py Lightning checkpoint.

    expect_positional_embedding: if set, raise when the checkpoint's recorded scheme differs.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if expect_positional_embedding is not None:
        trained_with = ckpt.get("student_positional_embedding", "sin")
        if trained_with != expect_positional_embedding:
            raise RuntimeError(
                f"{checkpoint_path} was distilled with positional_embedding={trained_with!r} "
                f"but the backbone is being built with {expect_positional_embedding!r}. These are "
                "different functions of the same weights (see modeling/streaming/streaming_mert.py); set "
                f"backbone_positional_embedding: {trained_with!r} in the config, or distill a new "
                f"checkpoint with student_positional_embedding: {expect_positional_embedding!r}."
            )
    lit_state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    student_state_dict = {}
    for key, value in lit_state_dict.items():
        if not key.startswith("model."):
            continue
        key = key[len("model."):]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        student_state_dict[key] = value

    if not student_state_dict:
        raise RuntimeError(f"no 'model.*' keys found in {checkpoint_path} -- wrong checkpoint?")
    return student_state_dict


def backbone_lookahead_block(checkpoint_path):
    """Block-causal block size (raw frames) recorded in the checkpoint; 1 = strictly causal."""
    if not checkpoint_path:
        return 1
    try:
        # mmap: only one small key is needed.
        ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    except (RuntimeError, ValueError, TypeError):
        ckpt = torch.load(checkpoint_path, map_location="cpu")   # pre-zipfile checkpoint
    return int(ckpt.get("student_lookahead_block", 1))


def check_streaming_context(checkpoint_path: Optional[str], streaming_context: int) -> None:
    """Warn if streaming_context differs from the distillation run's (read from args.yml next to the checkpoint)."""
    if not checkpoint_path:
        return
    args_path = os.path.join(os.path.dirname(str(checkpoint_path)), "args.yml")
    if not os.path.exists(args_path):
        return
    try:
        import yaml
        with open(args_path) as f:
            args = yaml.load(f, Loader=yaml.FullLoader) or {}
    except Exception:
        return
    trained = args.get("streaming_context")
    if trained is not None and int(trained) != int(streaming_context):
        print(
            f"WARNING: {checkpoint_path} was distilled with streaming_context={int(trained)} "
            f"raw frames but is being evaluated with {int(streaming_context)}. The KV window is "
            "not recorded in the checkpoint, so this loads cleanly and then computes a different "
            f"function than the weights were trained for. Set streaming_context: {int(trained)} "
            "unless the difference is the experiment."
        )


def load_backbone_and_processor(config: dict, device: torch.device):
    """Frozen bidirectional backbone + HF feature extractor.

    config['backbone']: 'pretrained' (any m-a-p/MERT-* checkpoint) or
    'small_bidirectional' (26M student; needs mert_align_checkpoint).
    """
    backbone = config.get("backbone", "pretrained")
    mert_model_name = config["MERT_model"]
    processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_model_name, trust_remote_code=True)

    if backbone == "pretrained":
        mert = AutoModel.from_pretrained(mert_model_name, trust_remote_code=True)
    elif backbone == "small_bidirectional":
        from modeling.mert.configuration_MERT import MERTConfig
        from modeling.mert.small_mert import SmallBidirectionalMERTModel, build_small_mert_config

        teacher_config = MERTConfig.from_pretrained(mert_model_name)
        student_config = build_small_mert_config(
            teacher_config,
            hidden_size=config.get("student_hidden_size", 432),
            intermediate_size=config.get("student_intermediate_size", 1392),
            num_attention_heads=config.get("student_num_attention_heads", 8),
            num_hidden_layers=config.get("student_num_hidden_layers"),
        )
        mert = SmallBidirectionalMERTModel(student_config)
        checkpoint_path = config["mert_align_checkpoint"]
        student_state_dict = load_student_state_dict_from_align_checkpoint(checkpoint_path)
        missing, unexpected = mert.load_state_dict(student_state_dict, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        print(f"Loaded distilled small_bidirectional weights from {checkpoint_path}")
    else:
        raise ValueError(
            f"invalid backbone {backbone!r}, expected 'pretrained' or 'small_bidirectional'")

    _check_emb_dim(mert, config.get("emb_dim"), backbone)
    mert = mert.to(device).eval()
    for p in mert.parameters():
        p.requires_grad_(False)
    return mert, processor


def _check_emb_dim(model: nn.Module, expect_emb_dim: Optional[int], backbone: str) -> None:
    """Raise unless emb_dim equals the backbone's hidden_size (1024 / 768 / 432)."""
    if expect_emb_dim is None or expect_emb_dim == model.config.hidden_size:
        return
    raise ValueError(
        f"emb_dim {expect_emb_dim} does not match the {backbone} backbone's hidden_size "
        f"{model.config.hidden_size} -- set emb_dim to the backbone's own width"
    )


# ---------------------------------------------------------------------------
# Streaming (causal) backbones
# ---------------------------------------------------------------------------

def _load_small_streaming_backbone(
    mert_model_name: str,
    device: torch.device,
    streaming_context: int,
    mert_align_checkpoint: Optional[str],
    positional_embedding: str,
    num_attention_heads: int,
):
    """26M causal streaming student, rebuilt from its (required) mert_align_checkpoint."""
    if not mert_align_checkpoint:
        raise ValueError(
            "backbone: small_streaming needs mert_align_checkpoint -- the small AR student "
            "exists only as a train_mert_causal_distill.py distillation product (student_model: "
            "small_streaming), there are no published weights at that size to fall back on"
        )
    from modeling.load_checkpoint import load_small_streaming_mert_from_align_checkpoint

    model = load_small_streaming_mert_from_align_checkpoint(
        mert_align_checkpoint,
        base_config=mert_model_name,
        num_attention_heads=num_attention_heads,
        context=streaming_context,
        positional_embedding=positional_embedding,
        lookahead_block=backbone_lookahead_block(mert_align_checkpoint),
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_streaming_backbone(
    mert_model_name: str,
    device: torch.device,
    streaming_context: int,
    mert_align_checkpoint: Optional[str] = None,
    positional_embedding: str = "rope",
    backbone: str = "streaming",
    num_attention_heads: int = 8,
    expect_emb_dim: Optional[int] = None,
):
    """Frozen causal streaming MERT (modeling/streaming/streaming_mert.py); takes raw waveforms.

    backbone: 'streaming' -- MERT-330M/95M remapped from mert_model_name, optionally
              overridden by mert_align_checkpoint (a train_mert_causal_distill.py checkpoint);
              'small_streaming' -- 26M student, mert_align_checkpoint required.
    streaming_context: KV-cache window in raw frames (75 fps); must match the checkpoint's.
    positional_embedding: must match the checkpoint ('rope' recommended).
    """
    from modeling.init_streaming import remap_feature_extractor_keys, remap_transformer_layer_keys
    from modeling.load_mert import load_raw_state_dict
    from modeling.mert.configuration_MERT import MERTConfig
    from modeling.streaming.streaming_mert import FullStreamingMERTModel

    check_streaming_context(mert_align_checkpoint, streaming_context)

    if backbone == "small_streaming":
        model = _load_small_streaming_backbone(
            mert_model_name, device, streaming_context, mert_align_checkpoint,
            positional_embedding, num_attention_heads,
        )
        _check_emb_dim(model, expect_emb_dim, backbone)
        return model
    if backbone != "streaming":
        raise ValueError(
            f"invalid backbone {backbone!r}, expected 'streaming' or 'small_streaming'")

    mert_config = MERTConfig.from_pretrained(mert_model_name)
    # Block size always comes from the checkpoint, never the config.
    lookahead_block = backbone_lookahead_block(mert_align_checkpoint)
    model = FullStreamingMERTModel(
        mert_config, context=streaming_context, positional_embedding=positional_embedding,
        lookahead_block=lookahead_block,
    )
    raw_state_dict = load_raw_state_dict(mert_model_name)
    conv_remapped, _ = remap_feature_extractor_keys(raw_state_dict, mert_config.num_feat_extract_layers)
    feature_extractor_keys = {k: v for k, v in conv_remapped.items() if k.startswith("feature_extractor.")}
    feature_projection_keys = {k: v for k, v in conv_remapped.items() if k.startswith("feature_projection.")}
    transformer_keys = remap_transformer_layer_keys(raw_state_dict, mert_config.num_hidden_layers)
    missing, unexpected = model.load_state_dict(
        {**feature_extractor_keys, **feature_projection_keys, **transformer_keys}, strict=False
    )
    assert not missing and not unexpected, (missing, unexpected)
    if mert_align_checkpoint is not None:
        student_state_dict = load_student_state_dict_from_align_checkpoint(
            mert_align_checkpoint, expect_positional_embedding=positional_embedding,
        )
        missing, unexpected = model.load_state_dict(student_state_dict, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        print(f"Loaded distilled streaming MERT weights from {mert_align_checkpoint}")
    print(f"Streaming backbone: positional_embedding={positional_embedding}, "
          f"context={streaming_context} frames, lookahead_block={lookahead_block}")
    _check_emb_dim(model, expect_emb_dim, backbone)
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_streaming_features(
    model: nn.Module,
    audio: torch.Tensor,         # [B, 1, T_samples]
    pool_factor: int,
    target_raw_frames: int,      # exact raw (pre-pool) frame count to trim to before pooling
    device: torch.device,
) -> torch.Tensor:
    """Whole input through the streaming backbone in one call -> [B, L, target_raw_frames // pool_factor, D].

    Memory is quadratic in length; see extract_streaming_features_streaming for long tracks.
    """
    waveform = audio[:, 0, :].to(device)  # [B, T_samples]
    with torch.no_grad():
        _, hidden_states = model(waveform, output_hidden_states=True)
        hidden = torch.stack(hidden_states, dim=1)  # [B, L, T_raw, D]

    B, n_layers, T_raw, D = hidden.shape
    assert T_raw >= target_raw_frames, (
        f"the streaming backbone produced only {T_raw} raw frames, need >= {target_raw_frames} "
        "-- the crop wasn't long enough for its conv stack"
    )
    hidden = hidden[:, :, :target_raw_frames, :]
    T_reduced = target_raw_frames // pool_factor
    hidden = hidden.reshape(B, n_layers, T_reduced, pool_factor, D).mean(3)
    return hidden


def extract_streaming_features_streaming(
    model: nn.Module,
    audio: torch.Tensor,         # [B, 1, T_samples]
    pool_factor: int,
    target_raw_frames: int,
    device: torch.device,
    hop_size: int,
    chunk_raw_frames: int,
) -> torch.Tensor:
    """Chunk-by-chunk extraction in one persistent .streaming() session (O(context) memory).

    chunk_raw_frames only affects speed; it must be a multiple of the backbone's
    lookahead_block.
    """
    waveform = audio[:, 0, :].to(device)  # [B, T_samples]
    B, total_samples = waveform.shape
    # Block-causal backbones need block-aligned chunks.
    _block = getattr(model, "lookahead_block", 1)
    assert _block == 1 or chunk_raw_frames % _block == 0, (
        f"chunk_raw_frames={chunk_raw_frames} is not a multiple of the backbone's "
        f"lookahead_block={_block}; block-aligned chunks are required"
    )
    chunk_samples = chunk_raw_frames * hop_size
    n_chunks = -(-target_raw_frames // chunk_raw_frames)  # ceil div
    pad_len = n_chunks * chunk_samples - total_samples
    padded = torch.nn.functional.pad(waveform, (0, pad_len)) if pad_len > 0 else waveform

    hidden_chunks = []
    with torch.no_grad(), model.feature_extractor.streaming(B), model.transformer.streaming(B):
        for c in range(n_chunks):
            chunk = padded[:, c * chunk_samples : (c + 1) * chunk_samples]
            _, hs = model(chunk, output_hidden_states=True)
            hidden_chunks.append(torch.stack(hs, dim=1))              # [B, L, chunk_raw_frames, D]
    hidden = torch.cat(hidden_chunks, dim=2)

    n_layers, D = hidden.shape[1], hidden.shape[3]
    hidden = hidden[:, :, :target_raw_frames, :]
    T_reduced = target_raw_frames // pool_factor
    hidden = hidden.reshape(B, n_layers, T_reduced, pool_factor, D).mean(3)
    return hidden


if __name__ == "__main__":
    # Self-test (no checkpoint, audio or GPU): last readout == segment prediction,
    # prefix causality, and chunked streaming == one full pass.
    torch.manual_seed(0)
    B, L, T, D, V = 2, 13, 30, 64, 24
    CHUNK = 3
    x = torch.randn(B, L, T, D)

    configs = [
        ("MLP", dict(model_type="MLP", hidden_dim=32)),
        ("MLP-linear", dict(model_type="MLP", hidden_dim=None)),
        ("LSTM", dict(model_type="LSTM", lstm_hidden_dim=32, lstm_num_layers=2)),
        ("Transformer", dict(model_type="Transformer", transformer_d_model=32,
                             transformer_num_heads=4, transformer_num_layers=2,
                             transformer_dim_feedforward=64, transformer_context=T)),
    ]
    for name, overrides in configs:
        config = dict(emb_dim=D, mert_layer='average', **overrides)
        head = build_classifier(config["model_type"], V, config).eval()

        with torch.no_grad():
            segment = head(x)
            running = head(x, running=True)
            assert segment.shape == (B, V), segment.shape
            assert running.shape == (B, T, V), running.shape
            err_last = (segment - running[:, -1]).abs().max().item()
            assert err_last < 1e-5, (name, err_last)

            # prefix causality: readouts over a truncated input must match
            prefix = head(x[:, :, : T // 2], running=True)
            err_prefix = (prefix - running[:, : T // 2]).abs().max().item()
            assert err_prefix < 1e-5, (name, err_prefix)

            # chunked streaming session vs one full pass
            parts = []
            with head.streaming(B):
                for i in range(0, T, CHUNK):
                    parts.append(head(x[:, :, i : i + CHUNK], running=True))
            chunked = torch.cat(parts, dim=1)
            err_stream = (chunked - running).abs().max().item()
            assert err_stream < 1e-4, (name, err_stream)

            # without a session each chunk restarts (expected to differ)
            no_session = torch.cat(
                [head(x[:, :, i : i + CHUNK], running=True) for i in range(0, T, CHUNK)], dim=1)
            err_no_session = (no_session - running).abs().max().item()

        n_params = sum(p.numel() for p in head.parameters())
        print(f"{name:12s} params={n_params:8,d}  last-readout={err_last:.2e}  "
              f"prefix={err_prefix:.2e}  streamed={err_stream:.2e}  "
              f"(no session: {err_no_session:.2e})")

    # train_readout: both branches return the same segment logits.
    config = dict(emb_dim=D, mert_layer='average', model_type="MLP", hidden_dim=32)
    head = build_classifier("MLP", V, config).eval()
    labels = torch.randint(V, (B,))
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        loss_final, logits_final = readout_loss(head, x, labels, criterion, "final")
        loss_running, logits_running = readout_loss(head, x, labels, criterion, "running")
    assert logits_final.shape == logits_running.shape == (B, V)
    assert (logits_final - logits_running).abs().max().item() < 1e-5
    assert loss_final.item() != loss_running.item()
    print(f"readout       final loss={loss_final.item():.4f}  running loss={loss_running.item():.4f}")
    print("models self-test OK")
