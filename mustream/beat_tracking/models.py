"""Frozen backbones (MERT, 26M students, causal streaming MERT), probe heads, loss and metrics for beat tracking."""

import os
import sys
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _REPO_ROOT)
from modeling.moshi.modules.transformer import StreamingTransformer


def load_mert(model_name: str, device: torch.device):
    """Load MERT + its feature extractor, frozen and in eval mode."""
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    mert = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    for p in mert.parameters():
        p.requires_grad_(False)
    return mert, processor


def load_student_state_dict_from_align_checkpoint(checkpoint_path: str) -> dict:
    """Student backbone weights (`model.*` keys) from a distillation (Lightning) checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
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


def load_backbone_and_processor(config: dict, device: torch.device):
    """(frozen backbone, feature extractor) for config['backbone']: 'pretrained' (HF MERT)
    or 'small_bidirectional' (26M student; needs a train_mert_size_distill.py mert_align_checkpoint)."""
    backbone = config.get("backbone", "pretrained")
    mert_model_name = config["MERT_model"]

    if backbone == "pretrained":
        return load_mert(mert_model_name, device)

    if backbone == "small_bidirectional":
        from modeling.mert.configuration_MERT import MERTConfig
        from modeling.mert.small_mert import SmallBidirectionalMERTModel, build_small_mert_config

        processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_model_name, trust_remote_code=True)
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

        mert = mert.to(device).eval()
        for p in mert.parameters():
            p.requires_grad_(False)
        return mert, processor

    raise ValueError(f"invalid backbone {backbone!r}, expected 'pretrained' or 'small_bidirectional'")


def backbone_lookahead_block(checkpoint_path):
    """Block-causal lookahead block (raw frames) recorded in a distillation checkpoint; 1 if none."""
    if not checkpoint_path:
        return 1
    try:
        # mmap: only one small key is needed from a large checkpoint.
        ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    except (RuntimeError, ValueError, TypeError):
        ckpt = torch.load(checkpoint_path, map_location="cpu")   # pre-zipfile checkpoint
    return int(ckpt.get("student_lookahead_block", 1))


def emission_block_for_config(config: dict):
    """Label frames the backbone emits per forward call (block for dbn_blocksync), or None.

    streaming/small_streaming: the checkpoint's lookahead block / pool_factor;
    sliding: one hop; chunked: one chunk; offline: None.
    """
    pool_factor = config["pool_factor"]
    label_rate = config["sample_rate"] / config["hop_size"] / pool_factor
    if config.get("backbone", "pretrained") in ("streaming", "small_streaming"):
        raw = backbone_lookahead_block(config.get("mert_align_checkpoint"))
        return max(1, raw // pool_factor)
    if config.get("hop_duration") is not None:
        return max(1, round(config["hop_duration"] * label_rate))
    if config.get("chunk_duration") is not None:
        return max(1, round(config["chunk_duration"] * label_rate))
    return None


def extract_mert_features(
    mert: nn.Module,
    processor: Wav2Vec2FeatureExtractor,
    audio: torch.Tensor,  # [B, 1, T_samples]
    sample_rate: int,
    pool_factor: int = 1,
    device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    """All-layer hidden states [B, n_layers, T_raw // pool_factor, D] of a frozen MERT or 26M student (no grad)."""
    B = audio.shape[0]
    input_values = audio[:, 0, :]   # [B, T_samples]
    if processor.do_normalize:
        mean = input_values.mean(dim=-1, keepdim=True)
        var = input_values.var(dim=-1, unbiased=False, keepdim=True)
        input_values = (input_values - mean) / torch.sqrt(var + 1e-7)

    with torch.no_grad():
        outputs = mert(input_values=input_values, output_hidden_states=True)
    hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs[1]

    hidden = torch.stack(hidden_states).permute(1, 0, 2, 3)  # [B, n_layers, T_raw, D]
    if pool_factor > 1:
        _, n_layers, T_raw, D = hidden.shape
        T_trim = (T_raw // pool_factor) * pool_factor
        hidden = hidden[:, :, :T_trim, :]
        hidden = hidden.reshape(B, n_layers, T_trim // pool_factor, pool_factor, D).mean(3)
    return hidden


def select_mert_layer(x: torch.Tensor, mert_layer: Union[int, str]) -> torch.Tensor:
    """[B, n_layers, T, D] -> [B, T, D], averaging over all layers or picking one."""
    if mert_layer == "average":
        return x.mean(dim=1)
    if isinstance(mert_layer, int):
        return x[:, mert_layer, :, :]
    raise ValueError(f"invalid mert_layer {mert_layer!r}")


class LinearProbe(nn.Module):
    """Per-frame probe: [B, n_layers, T, D] -> [B, T, 2] (beat, downbeat) logits.

    hidden_dim=None -> single linear layer; int -> one ReLU hidden layer.
    """

    def __init__(
        self,
        emb_dim: int = 1024,
        hidden_dim: Optional[int] = None,
        mert_layer: Union[int, str] = "average",
    ):
        super().__init__()
        self.mert_layer = mert_layer
        if hidden_dim is None:
            self.head = nn.Linear(emb_dim, 2)
        else:
            self.head = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = select_mert_layer(x, self.mert_layer)  # [B, T, D]
        return self.head(x)  # [B, T, 2]


class TransformerProbe(nn.Module):
    """Causal Transformer probe: [B, n_layers, T, D] -> [B, T, 2] logits.

    context: max attention lookback in frames (None = unbounded). Use 'rope' positional embedding.
    """

    def __init__(
        self,
        emb_dim: int = 1024,
        mert_layer: Union[int, str] = "average",
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        context: Optional[int] = None,
        positional_embedding: str = "rope",
    ):
        super().__init__()
        self.mert_layer = mert_layer
        self.input_proj = (
            nn.Identity() if d_model == emb_dim else nn.Linear(emb_dim, d_model)
        )
        self.transformer = StreamingTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            causal=True,
            context=context,
            positional_embedding=positional_embedding,
        )
        self.head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = select_mert_layer(x, self.mert_layer)  # [B, T, D]
        x = self.input_proj(x)  # [B, T, d_model]
        x = self.transformer(x)  # [B, T, d_model]
        return self.head(x)  # [B, T, 2]


class LSTMProbe(nn.Module):
    """Unidirectional (causal) LSTM probe: [B, n_layers, T, D] -> [B, T, 2] logits."""

    def __init__(
        self,
        emb_dim: int = 1024,
        mert_layer: Union[int, str] = "average",
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mert_layer = mert_layer
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = select_mert_layer(x, self.mert_layer)  # [B, T, D]
        x, _ = self.lstm(x)  # [B, T, hidden_dim]
        return self.head(x)  # [B, T, 2]


def build_probe(model_type: str, config: dict) -> nn.Module:
    """Build the probe for model_type 'MLP' | 'Transformer' | 'LSTM' from config."""
    if model_type == "MLP":
        return LinearProbe(
            emb_dim=config["emb_dim"],
            hidden_dim=config.get("hidden_dim"),
            mert_layer=config["mert_layer"],
        )
    elif model_type == "Transformer":
        return TransformerProbe(
            emb_dim=config["emb_dim"],
            mert_layer=config["mert_layer"],
            d_model=config.get("transformer_d_model", 256),
            num_heads=config.get("transformer_num_heads", 4),
            num_layers=config.get("transformer_num_layers", 4),
            dim_feedforward=config.get("transformer_dim_feedforward", 1024),
            context=config.get("transformer_context"),
            positional_embedding=config.get("transformer_positional_embedding", "rope"),
        )
    elif model_type == "LSTM":
        return LSTMProbe(
            emb_dim=config["emb_dim"],
            mert_layer=config["mert_layer"],
            hidden_dim=config.get("lstm_hidden_dim", 256),
            num_layers=config.get("lstm_num_layers", 2),
            dropout=config.get("lstm_dropout", 0.0),
        )
    else:
        raise ValueError(f"invalid model_type {model_type!r}, expected 'MLP', 'Transformer', or 'LSTM'")


# ---------------------------------------------------------------------------
# Sliding-window lookahead
# ---------------------------------------------------------------------------

def lookahead_frames(config: dict) -> int:
    """Sliding-window config['lookahead'] in seconds -> k label frames (rounded; 0 = causal)."""
    duration = float(config.get("lookahead", 0.0) or 0.0)
    if duration < 0:
        raise ValueError(f"lookahead must be >= 0 seconds, got {duration}")

    label_rate = config["sample_rate"] / config["hop_size"] / config["pool_factor"]
    lookahead = round(duration * label_rate)
    if duration > 0 and lookahead == 0:
        raise ValueError(
            f"lookahead={duration}s rounds to 0 frames at {label_rate:.4g} fps "
            f"(one label frame is {1 / label_rate:.4f}s) -- ask for at least half a frame, "
            "or set 0 to disable lookahead explicitly"
        )
    return lookahead


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------

def compute_loss(logits, beat_seq, downbeat_seq, mask, downbeat_available, config):
    """Masked pos-weighted BCE for beat + downbeat; tracks without downbeat labels skip the downbeat term.

    logits: [B, T, 2]; beat_seq/downbeat_seq: [B, T]; mask: [B, T] valid frames;
    downbeat_available: [B].
    """
    device = logits.device
    beat_logits, downbeat_logits = logits[..., 0], logits[..., 1]

    beat_loss_fn = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(config["pos_weight_beat"], device=device),
    )
    downbeat_loss_fn = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(config["pos_weight_downbeat"], device=device),
    )

    beat_mask = mask.float()
    beat_loss = (beat_loss_fn(beat_logits, beat_seq) * beat_mask).sum() / beat_mask.sum().clamp(min=1)

    downbeat_mask = mask.float() * downbeat_available.float().unsqueeze(1)
    downbeat_loss = (downbeat_loss_fn(downbeat_logits, downbeat_seq) * downbeat_mask).sum() / downbeat_mask.sum().clamp(min=1)

    return beat_loss + downbeat_loss, beat_loss.detach(), downbeat_loss.detach()


def frame_stats(logits, beat_seq, downbeat_seq, mask, downbeat_available):
    """Per-frame (correct, valid, (tp, fp, fn)) counts for beat and downbeat, for epoch-level micro-averages."""
    with torch.no_grad():
        beat_pred = (logits[..., 0] > 0).float()
        downbeat_pred = (logits[..., 1] > 0).float()
        m = mask.float()
        dm = m * downbeat_available.float().unsqueeze(1)

        def stats(pred, target, w):
            tp = (pred * target * w).sum()
            fp = (pred * (1 - target) * w).sum()
            fn = ((1 - pred) * target * w).sum()
            valid = w.sum()
            correct = valid - fp - fn
            return correct.item(), valid.item(), (tp.item(), fp.item(), fn.item())

        beat_correct, beat_valid, beat_tpfpfn = stats(beat_pred, beat_seq, m)
        downbeat_correct, downbeat_valid, downbeat_tpfpfn = stats(downbeat_pred, downbeat_seq, dm)
    return (beat_correct, beat_valid, beat_tpfpfn), (downbeat_correct, downbeat_valid, downbeat_tpfpfn)


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# Name of the val_selection_score metric (for logs).
BEST_CHECKPOINT_METRIC = "mean val frame F1 (beat, downbeat)"


def val_selection_score(stats: dict) -> float:
    """best.pt selection score: mean of beat and downbeat val frame F1."""
    return 0.5 * (stats["beat_f1"] + stats["downbeat_f1"])


# ---------------------------------------------------------------------------
# Causal streaming backbone loading + feature extraction
# ---------------------------------------------------------------------------

def _student_state_dict_from_align_checkpoint(
    checkpoint_path: str,
    expect_positional_embedding: Optional[str] = None,
) -> dict:
    """Streaming student weights from a train_mert_causal_distill.py checkpoint.

    Raises if the checkpoint's positional embedding differs from expect_positional_embedding.
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


def _check_emb_dim(model: nn.Module, expect_emb_dim: Optional[int], backbone: str) -> None:
    """Raise if config emb_dim != the backbone's hidden_size (1024 / 768 / 432)."""
    if expect_emb_dim is None or expect_emb_dim == model.config.hidden_size:
        return
    raise ValueError(
        f"emb_dim: {expect_emb_dim} does not match the {backbone} backbone's hidden_size "
        f"{model.config.hidden_size} -- set emb_dim: {model.config.hidden_size} in the config"
    )


def _load_small_streaming_backbone(
    mert_model_name: str,
    device: torch.device,
    streaming_context: int,
    mert_align_checkpoint: Optional[str],
    positional_embedding: str,
    num_attention_heads: int,
):
    """Frozen 26M causal student rebuilt from its (required) mert_align_checkpoint."""
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
    """Frozen, eval-mode causal streaming MERT backbone.

    backbone='streaming': published MERT (mert_model_name) remapped into the streaming
    architecture, optionally overridden by mert_align_checkpoint (causal distillation).
    backbone='small_streaming': the 26M causal student; mert_align_checkpoint required,
    mert_model_name must be the family it was distilled from.
    positional_embedding must match the checkpoint (use 'rope').
    """
    from modeling.init_streaming import remap_feature_extractor_keys, remap_transformer_layer_keys
    from modeling.load_mert import load_raw_state_dict
    from modeling.mert.configuration_MERT import MERTConfig
    from modeling.streaming.streaming_mert import FullStreamingMERTModel

    if backbone == "small_streaming":
        model = _load_small_streaming_backbone(
            mert_model_name, device, streaming_context, mert_align_checkpoint,
            positional_embedding, num_attention_heads,
        )
        _check_emb_dim(model, expect_emb_dim, backbone)
        return model
    if backbone != "streaming":
        raise ValueError(
            f"invalid backbone {backbone!r}, expected 'streaming' or 'small_streaming'"
        )

    mert_config = MERTConfig.from_pretrained(mert_model_name)
    # Lookahead block always comes from the checkpoint.
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
        student_state_dict = _student_state_dict_from_align_checkpoint(
            mert_align_checkpoint, expect_positional_embedding=positional_embedding,
        )
        missing, unexpected = model.load_state_dict(student_state_dict, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        print(f"Loaded fine-tuned streaming MERT weights from {mert_align_checkpoint}")
    print(f"Streaming backbone: positional_embedding={positional_embedding}, context={streaming_context} frames")
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
    """Single forward pass -> [B, L, target_raw_frames // pool_factor, D] (memory quadratic in length)."""
    waveform = audio[:, 0, :].to(device)  # [B, T_samples]
    with torch.no_grad():
        _, hidden_states = model(waveform, output_hidden_states=True)
        hidden = torch.stack(hidden_states, dim=1)  # [B, L, T_raw, D]

    B, n_layers, T_raw, D = hidden.shape
    assert T_raw >= target_raw_frames, (
        f"the streaming backbone produced only {T_raw} raw frames, need >= {target_raw_frames} "
        "-- crop_samples wasn't long enough for its conv stack"
    )
    hidden = hidden[:, :, :target_raw_frames, :]
    T_reduced = target_raw_frames // pool_factor
    hidden = hidden.reshape(B, n_layers, T_reduced, pool_factor, D).mean(3)
    return hidden


def extract_streaming_features_streaming(
    model: nn.Module,
    audio: torch.Tensor,         # [B, 1, T_samples], already trimmed to an exact multiple of pool_factor*hop_size
    pool_factor: int,
    target_raw_frames: int,
    device: torch.device,
    hop_size: int,
    chunk_raw_frames: int,
) -> torch.Tensor:
    """Incremental extraction: feed chunks of chunk_raw_frames into one .streaming() session.

    The backbone KV cache is O(context), but this helper retains the whole
    waveform and all hidden chunks before returning pooled full-track features.
    Total memory grows with track length, even at a fixed chunk size. Results are
    independent of valid chunk size; chunk_raw_frames must be a multiple of the
    backbone's lookahead_block.
    """
    waveform = audio[:, 0, :].to(device)  # [B, T_samples]
    B, total_samples = waveform.shape
    # Chunks must align to the block grid of a block-causal backbone.
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
    hidden = torch.cat(hidden_chunks, dim=2)                          # [B, L, n_chunks*chunk_raw_frames, D]

    n_layers, D = hidden.shape[1], hidden.shape[3]
    hidden = hidden[:, :, :target_raw_frames, :]
    T_reduced = target_raw_frames // pool_factor
    hidden = hidden.reshape(B, n_layers, T_reduced, pool_factor, D).mean(3)
    return hidden
