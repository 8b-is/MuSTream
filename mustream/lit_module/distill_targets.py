"""
Local-interval teacher targets for lit_module/mert_causal_distill.py.

Instead of running the bidirectional teacher over the whole crop, the teacher
only sees a local interval of audio around the supervised frames, so it has
limited future context during distillation: at most W - 1 + r frames past any
target frame (see below), rather than the rest of the crop. The student is
therefore matched against what a causal / block-causal model can (nearly) see.

Interval geometry, in raw 75 fps frames, for an interval ending at frame e:
    window_frames    (L) context up to the block's end, [e - L, e).
    supervise_frames (W) frames at the interval's right edge kept as targets,
                     [e - W, e).
    predict_frames   (r) future frames past the block the teacher also sees,
                     [e, e + r); 0 makes the target exactly reachable for a
                     block-causal student.

Interval ends lie on a multiple-of-W grid so they coincide with the student's
block boundaries. The first L - W frames of a crop cannot end a full interval;
sample_prefix_end / prefix_teacher_hidden supervise them with the teacher
truncated to [0, end + r).

Intervals are cut at the transformer input, not the waveform, because MERT's
first conv layer uses a length-dependent GroupNorm; the conv frontend runs once
over the whole crop.
"""
import torch


def inner_mert(model):
    """Return the HF MERTModel, unwrapping a SmallBidirectionalMERTModel."""
    return model if hasattr(model, "feature_extractor") else model.mert


@torch.no_grad()
def pre_encoder_features(model, waveform: torch.Tensor) -> torch.Tensor:
    """(B, N samples) -> (B, T, D): MERTModel.forward up to the encoder."""
    m = inner_mert(model)
    x = m.feature_extractor(waveform).transpose(1, 2)
    if m.config.feature_extractor_cqt:
        cqt = m.feature_extractor_cqt(waveform).transpose(1, 2)[:, : x.shape[1], :]
        x = torch.cat([x, cqt], 2)
    x = m.feature_projection(x)
    return m._mask_hidden_states(x, mask_time_indices=None)   # no-op in eval


@torch.no_grad()
def encoder_hidden_states(model, x: torch.Tensor) -> torch.Tensor:
    """(B, T, D) -> (B, L, T, D), the same hidden-state tuple MERTModel returns
    (pre-transformer input first, one per layer after)."""
    out = inner_mert(model).encoder(
        x, attention_mask=None, output_attentions=False,
        output_hidden_states=True, return_dict=True,
    )
    return torch.stack(out.hidden_states, dim=1)


def sample_window_ends(
    batch_size: int, n_windows: int, window_frames: int, supervise_frames: int,
    n_frames: int, device, generator=None, predict_frames: int = 0,
) -> torch.Tensor:
    """(B, K) exclusive end frames on a multiple-of-`supervise_frames` grid,
    drawn without replacement per item. Each end leaves a full interval behind
    (end >= window_frames) and room for the teacher-only tail
    (end + predict_frames <= n_frames)."""
    lo = -(-window_frames // supervise_frames)                    # ceil, first usable grid point
    hi = (n_frames - predict_frames) // supervise_frames          # last that fits with the tail
    n_grid = hi - lo + 1
    if n_grid < 1:
        raise ValueError(
            f"no interval fits: window_frames={window_frames}, supervise_frames="
            f"{supervise_frames} and predict_frames={predict_frames} need a crop of at "
            f"least {lo * supervise_frames + predict_frames} frames, but it has {n_frames}"
        )
    k = min(n_windows, n_grid)
    # argsort of uniform noise == a per-row randperm, without a Python loop.
    noise = torch.rand(batch_size, n_grid, device=device, generator=generator)
    picks = noise.argsort(dim=1)[:, :k]                 # (B, k) distinct grid indices
    return (picks + lo) * supervise_frames              # (B, k) frame indices


def sample_prefix_end(window_frames: int, supervise_frames: int, step: int) -> int:
    """One end from {W, 2W, ..., window_frames - W}, the region
    sample_window_ends cannot reach. Log-uniform to favour the earliest frames;
    seeded by `step` so every DDP rank draws the same end."""
    n_grid = window_frames // supervise_frames - 1
    assert n_grid >= 1, (window_frames, supervise_frames)
    u = torch.rand((), generator=torch.Generator().manual_seed(int(step))).item()
    return max(1, min(n_grid, round(n_grid ** u))) * supervise_frames


def gather_windows(x: torch.Tensor, ends: torch.Tensor, length: int) -> torch.Tensor:
    """x (B, T, D), ends (B, K) -> (B, K, length, D): the `length` frames ending
    just before each end."""
    B, K = ends.shape
    D = x.shape[-1]
    idx = ends[..., None] - length + torch.arange(length, device=x.device)   # (B, K, length)
    flat = idx.reshape(B, K * length, 1).expand(-1, -1, D)
    return x.gather(1, flat).view(B, K, length, D)


def gather_frames_per_layer(x: torch.Tensor, ends: torch.Tensor, length: int) -> torch.Tensor:
    """x (B, L, T, D), ends (B, K) -> (B, L, K*length, D): gather_windows
    applied to every hidden-state layer. Selects the student frames that match
    the teacher targets."""
    B, L, _, D = x.shape
    K = ends.shape[1]
    idx = ends[..., None] - length + torch.arange(length, device=x.device)   # (B, K, length)
    flat = idx.reshape(B, 1, K * length, 1).expand(B, L, K * length, D)
    return x.gather(2, flat)


@torch.no_grad()
def windowed_teacher_hidden(
    teacher, pre: torch.Tensor, ends: torch.Tensor,
    window_frames: int, supervise_frames: int, predict_frames: int = 0,
) -> torch.Tensor:
    """
    pre (B, T, D) pre-encoder features, ends (B, K)
        -> (B, L, K*supervise_frames, D) teacher targets.

    For an end e the teacher reads [e - window_frames, e + predict_frames) and
    the supervised block is [e - supervise_frames, e). All intervals have the
    same length, so they run through the encoder as one batch.
    """
    B, K = ends.shape
    L, W = window_frames, supervise_frames
    # Intervals end at e + r, so gathering length L + r covers [e - L, e + r).
    windows = gather_windows(pre, ends + predict_frames, L + predict_frames)
    hidden = encoder_hidden_states(teacher, windows.flatten(0, 1))   # (B*K, L_h, L+r, D)
    # Absolute frame e - L + k sits at interval index k, so the block [e-W, e)
    # is at [L-W, L) -- NOT the interval's tail once r > 0.
    hidden = hidden[:, :, L - W:L, :]                                # (B*K, L_h, W, D)
    n_layers, D = hidden.shape[1], hidden.shape[3]
    # (B, K, L, W, D) -> (B, L, K, W, D) -> (B, L, K*W, D), so the time axis is
    # ordered the same way gather_frames_per_layer orders the student's.
    return (
        hidden.view(B, K, n_layers, supervise_frames, D)
        .permute(0, 2, 1, 3, 4)
        .reshape(B, n_layers, K * supervise_frames, D)
    )


@torch.no_grad()
def prefix_teacher_hidden(
    teacher, pre: torch.Tensor, end: int, supervise_frames: int,
    predict_frames: int = 0,
) -> torch.Tensor:
    """
    pre (B, T, D) -> (B, L, supervise_frames, D): target for a block near the
    stream start. The teacher reads [0, end + predict_frames) and the block is
    [end - supervise_frames, end); `end` is shared by the whole batch.
    """
    assert supervise_frames <= end, (supervise_frames, end)
    assert end + predict_frames <= pre.shape[1], (end, predict_frames, pre.shape)
    hidden = encoder_hidden_states(teacher, pre[:, : end + predict_frames])
    return hidden[:, :, end - supervise_frames:end, :]                   # (B, L_h, W, D)
