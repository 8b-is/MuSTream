"""Chord probe heads (MLP / LSTM / Transformer), backbone loaders, feature extraction
and decoding post-processors."""

import os
import sys
from collections import deque
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from transformers import AutoModel, Wav2Vec2FeatureExtractor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from modeling.moshi.modules.transformer import StreamingTransformer


def select_mert_layer(x: torch.Tensor, mert_layer: Union[int, str]) -> torch.Tensor:
    """[B, n_layers, T, D] -> [B, T, D]."""
    n_layers = x.shape[1]

    if mert_layer == 'average':  # mean over all layers → [B, T, D]
        return x.mean(dim=1)
    elif mert_layer == 'half':   # 13-25
        return x[:, n_layers//2:, :, :].mean(dim=1)
    elif mert_layer == 'low':  # 1-8
        return x[:, : n_layers//3, :, :].mean(dim=1)
    elif mert_layer == 'middle':  # 9-16
        return x[:, n_layers//3 : 2*n_layers//3, :, :].mean(dim=1)
    elif mert_layer == 'high':  # 17-25
        return x[:, 2*n_layers//3 :, :, :].mean(dim=1)
    elif mert_layer == 'top':  # 22-25
        return x[:, -4:, :, :].mean(dim=1)
    elif isinstance(mert_layer, int):
        return x[:, mert_layer, :, :]
    else:
        raise ValueError('invalid mert layer {}'.format(mert_layer))


class ChordClassifier(nn.Module):
    """Per-frame MLP probe: [B, n_layers, T, emb_dim] -> [B, T, vocab_size]."""

    def __init__(
            self,
            vocab_size: int,
            hidden_dim: Optional[int] = 512,
            emb_dim: int = 768,
            mert_layer: Union[int, str] = 'average',
            ):
        super().__init__()
        if hidden_dim is None:
            self.mlp = nn.Linear(emb_dim, vocab_size)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, vocab_size),
            )
        self.mert_layer = mert_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = select_mert_layer(x, self.mert_layer)               # → [B, T, D]
        return self.mlp(x)           # → [B, T, vocab_size]


class ChordTransformer(nn.Module):
    """Causal Transformer probe: [B, n_layers, T, emb_dim] -> [B, T, vocab_size].

    context: max attention lookback in frames (None = unbounded within a crop).
    Must be finite at training time for streaming() to work.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 768,
        mert_layer: Union[int, str] = 'average',
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
        self.out = nn.Linear(d_model, vocab_size)

    @contextmanager
    def streaming(self, batch_size: int):
        """Context manager that carries the KV cache across forward() calls (chunked inference)."""
        if self.context is None:
            raise RuntimeError(
                "ChordTransformer.streaming() needs a finite `context`: moshi sizes its "
                "RingKVCache from it and cannot infer a capacity otherwise (see "
                "moshi/modules/transformer.py). Set transformer_context in the config to a "
                "frame count and retrain -- a context=None checkpoint computes unbounded "
                "within-crop attention, which is a different function from the ring-cached "
                "bounded attention streaming would give it."
            )
        with self.transformer.streaming(batch_size):
            yield self

    def reset_streaming(self) -> None:
        """Clear the KV cache mid-session."""
        self.transformer.reset_streaming()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = select_mert_layer(x, self.mert_layer)                 # → [B, T, D]
        x = self.input_proj(x)     # → [B, T, d_model]
        x = self.transformer(x)    # → [B, T, d_model]
        return self.out(x)         # → [B, T, vocab_size]


class ChordLSTM(nn.Module):
    """Unidirectional LSTM probe: [B, n_layers, T, emb_dim] -> [B, T, vocab_size]."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 768,
        mert_layer: Union[int, str] = 'average',
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
        self.out = nn.Linear(hidden_dim, vocab_size)
        self._streaming_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._is_streaming = False

    @contextmanager
    def streaming(self, batch_size: int):
        """Context manager that carries (h, c) across forward() calls (batch_size is unused)."""
        del batch_size
        prev_streaming, prev_state = self._is_streaming, self._streaming_state
        self._is_streaming, self._streaming_state = True, None
        try:
            yield self
        finally:
            self._is_streaming, self._streaming_state = prev_streaming, prev_state

    def reset_streaming(self) -> None:
        """Drop (h, c) mid-session."""
        self._streaming_state = None

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x:            [B, n_layers, T, emb_dim]
            state:        optional initial (h, c); None = zeros, or the carried
                          state inside a streaming() session
            return_state: also return the final (h, c)
        """
        x = select_mert_layer(x, self.mert_layer)                 # → [B, T, D]
        if state is None and self._is_streaming:
            state = self._streaming_state
        x, new_state = self.lstm(x, state)   # state=None → zeros, as nn.LSTM's default
        if self._is_streaming:
            self._streaming_state = (new_state[0].detach(), new_state[1].detach())
        logits = self.out(x)                 # → [B, T, vocab_size]
        return (logits, new_state) if return_state else logits


def build_classifier(model_type: str, vocab_size: int, config: dict) -> nn.Module:
    """Build the probe for model_type ('MLP', 'Transformer' or 'LSTM') from a config dict."""
    if model_type == "MLP":
        return ChordClassifier(
            vocab_size=vocab_size,
            hidden_dim=config["hidden_dim"],
            emb_dim=config["emb_dim"],
            mert_layer=config["mert_layer"],
        )
    elif model_type == "Transformer":
        return ChordTransformer(
            vocab_size=vocab_size,
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
        return ChordLSTM(
            vocab_size=vocab_size,
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
    """Sliding-window config['lookahead'] in seconds (default 0 = causal) -> whole pooled label frames k."""
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


class LinearChainCRF(nn.Module):
    """Linear-chain CRF: forward() = mean NLL, decode() = Viterbi. transitions[i, j] scores i -> j."""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        emissions: torch.Tensor,   # [B, T, num_tags]
        tags: torch.Tensor,        # [B, T]  ground-truth indices
        mask: torch.Tensor,        # [B, T]  bool, True = valid frame
    ) -> torch.Tensor:
        """Returns mean negative log-likelihood (scalar)."""
        score = self._sequence_score(emissions, tags, mask)
        log_Z = self._log_partition(emissions, mask)
        return -(score - log_Z).mean()

    def decode(
        self,
        emissions: torch.Tensor,   # [B, T, num_tags]
        mask: torch.Tensor,        # [B, T]  bool
    ) -> torch.Tensor:             # [B, T]  predicted tag indices
        return self._viterbi_decode(emissions, mask)

    def streaming_decoder(
        self,
        lag: Optional[int] = None,
        use_viterbi: bool = True,
    ) -> "StreamingViterbi":
        """Convenience factory -- see StreamingViterbi for what lag/use_viterbi mean."""
        return StreamingViterbi(self, lag=lag, use_viterbi=use_viterbi)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sequence_score(self, emissions, tags, mask):
        """Gold-sequence score for each item in the batch. Returns [B]."""
        B, T, _ = emissions.shape

        # Score for first frame
        score = (
            self.start_transitions[tags[:, 0]]
            + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        )

        # Accumulate emission + transition scores for frames 1..T-1
        for t in range(1, T):
            emit = emissions[:, t].gather(1, tags[:, t:t+1]).squeeze(1)   # [B]
            trans = self.transitions[tags[:, t - 1], tags[:, t]]          # [B]
            score = score + (emit + trans) * mask[:, t].float()

        # End transition from the last *valid* tag
        last_idx = mask.long().sum(1).clamp(min=1) - 1                    # [B]
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)      # [B]
        score = score + self.end_transitions[last_tags]
        return score

    def _log_partition(self, emissions, mask):
        """Log-partition (normaliser) Z via forward algorithm. Returns [B]."""
        B, T, _ = emissions.shape

        # α at t=0:  [B, num_tags]
        alpha = self.start_transitions + emissions[:, 0]

        for t in range(1, T):
            # [B, num_tags, 1] + [1, num_tags, num_tags] + [B, 1, num_tags]
            # scores[b, i, j] = alpha[b,i] + trans[i,j] + emit[b,j]
            scores = (
                alpha.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, t].unsqueeze(1)
            )                                                              # [B, V, V]
            new_alpha = torch.logsumexp(scores, dim=1)                    # [B, V]
            # Keep previous alpha for padded positions
            alpha = torch.where(mask[:, t:t+1], new_alpha, alpha)

        return torch.logsumexp(alpha + self.end_transitions, dim=1)       # [B]

    def _viterbi_decode(self, emissions, mask):
        """Viterbi decoding. Returns [B, T] best-path tag indices."""
        B, T, _ = emissions.shape

        viterbi = self.start_transitions + emissions[:, 0]                # [B, V]
        backpointers = []

        for t in range(1, T):
            # scores[b, i, j] = viterbi[b,i] + trans[i,j]
            scores = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0) # [B, V, V]
            best_scores, best_prev = scores.max(dim=1)                    # [B, V]
            new_viterbi = best_scores + emissions[:, t]
            viterbi = torch.where(mask[:, t:t+1], new_viterbi, viterbi)
            backpointers.append(best_prev)                                # [B, V]

        viterbi = viterbi + self.end_transitions
        best_tags = viterbi.argmax(dim=1)                                 # [B]

        # Back-track through stored best-previous-tag pointers
        paths = [best_tags]
        for bp in reversed(backpointers):
            best_tags = bp.gather(1, best_tags.unsqueeze(1)).squeeze(1)
            paths.append(best_tags)
        paths.reverse()
        return torch.stack(paths, dim=1)                                  # [B, T]


class StreamingViterbi:
    """Incremental Viterbi over a LinearChainCRF's emissions.

    use_viterbi=False: per-frame argmax (crf may be None).
    lag=None: offline, full decode at flush().
    lag=k: emits frame t-k at each step (used internally by decode_block_synchronous).

    Usage: tag = dec.step(emission) per frame ([B] or None), then dec.flush().
    """

    def __init__(
        self,
        crf: Optional[LinearChainCRF],
        lag: Optional[int] = None,
        use_viterbi: bool = True,
    ):
        if use_viterbi and crf is None:
            raise ValueError("use_viterbi=True requires a LinearChainCRF, got crf=None")
        if lag is not None and lag < 0:
            raise ValueError(f"lag must be None or >= 0, got {lag}")
        self.crf = crf
        self.lag = lag
        self.use_viterbi = use_viterbi
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop all state -- start a fresh sequence."""
        self.score: Optional[torch.Tensor] = None   # [B, V] running Viterbi scores
        # Holds the last `lag` backpointers (lag=0 -> greedy).
        self.bps: deque = deque(maxlen=self.lag if self.lag is not None else None)
        self.buffer: list = []                      # offline mode only
        self.t = -1                                 # index of last stepped frame
        self.n_emitted = 0

    def step(self, emission: torch.Tensor) -> Optional[torch.Tensor]:
        """Feed one frame's [B, V] emissions; return [B] tags for frame t - lag, or None."""
        self.t += 1

        if not self.use_viterbi:
            self.n_emitted += 1
            return emission.argmax(dim=-1)

        if self.lag is None:                        # offline: just accumulate
            self.buffer.append(emission)
            return None

        if self.score is None:
            self.score = self.crf.start_transitions + emission           # frame 0
        else:
            # scores[b, i, j] = score[b, i] + trans[i, j]; max over previous tag i.
            scores = self.score.unsqueeze(2) + self.crf.transitions.unsqueeze(0)
            best, prev = scores.max(dim=1)                                # [B, V]
            self.score = best + emission
            self.bps.append(prev)
        # Renormalise to keep scores in float range on long streams.
        self.score = self.score - self.score.max(dim=1, keepdim=True).values

        if self.t < self.lag:
            return None                                                  # warming up
        tag = self.score.argmax(dim=1)                                    # best state now
        for bp in reversed(self.bps):                                     # back `lag` steps
            tag = bp.gather(1, tag.unsqueeze(1)).squeeze(1)
        self.n_emitted += 1
        return tag

    def flush(self) -> torch.Tensor:
        """End of stream: return [B, k] tags for all frames not yet emitted (applies end_transitions)."""
        if self.t < 0 or not self.use_viterbi:
            return torch.empty(0, 0, dtype=torch.long)

        if self.lag is None:                        # offline: the real decoder
            emissions = torch.stack(self.buffer, dim=1)                  # [B, T, V]
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
            return self.crf.decode(emissions, mask)                       # [B, T]

        score = self.score + self.crf.end_transitions
        tag = score.argmax(dim=1)
        tail = [tag]
        for bp in reversed(self.bps):
            tag = bp.gather(1, tag.unsqueeze(1)).squeeze(1)
            tail.append(tag)
        tail.reverse()                              # frames (t - len(bps)) .. t
        # Skip the oldest frame if step() already emitted it.
        drop = 1 if self.n_emitted > 0 else 0
        tail = tail[drop:]
        if not tail:                    # lag=0: step() finalises every frame
            return torch.empty(self.score.shape[0], 0, dtype=torch.long,
                               device=self.score.device)
        return torch.stack(tail, dim=1)                                   # [B, k]

    def emit_block(self, n: int) -> torch.Tensor:
        """Finalise the last n stepped frames together; return [B, n] tags. Needs lag >= n-1."""
        if not self.use_viterbi:
            raise RuntimeError("emit_block() needs the Viterbi recursion; use_viterbi=False "
                               "finalises every frame in step() already")
        if self.lag is None:
            raise RuntimeError("emit_block() needs a finite lag -- construct with "
                               "lag=block-1, not lag=None (offline buffers instead of stepping)")
        if n <= 0:
            raise ValueError(f"n must be >= 1, got {n}")
        if self.score is None:
            raise RuntimeError("emit_block() before any step()")
        if n - 1 > len(self.bps):
            raise RuntimeError(
                f"emit_block({n}) needs {n - 1} backpointers but the ring holds "
                f"{len(self.bps)} (lag={self.lag}); build with lag >= n-1"
            )
        tag = self.score.argmax(dim=1)                   # newest frame, no hindsight
        tags = [tag]
        # Walk back through THIS block's transitions only -- the newest n-1.
        for bp in reversed(list(self.bps)[len(self.bps) - (n - 1):]):
            tag = bp.gather(1, tag.unsqueeze(1)).squeeze(1)
            tags.append(tag)
        tags.reverse()
        self.n_emitted += n
        return torch.stack(tags, dim=1)                  # [B, n]

    def decode_sequence(self, emissions: torch.Tensor) -> torch.Tensor:
        """Decode [B, T, V] emissions by stepping every frame then flushing; return [B, T] tags."""
        self.reset()
        T = emissions.shape[1]
        out = [t.unsqueeze(1) for t in (self.step(emissions[:, i]) for i in range(T))
               if t is not None]
        tail = self.flush()
        if tail.numel():
            out.append(tail)
        decoded = torch.cat(out, dim=1) if out else emissions.new_empty(
            emissions.shape[0], 0, dtype=torch.long)
        assert decoded.shape[1] == T, (
            f"decoder emitted {decoded.shape[1]} frames for a {T}-frame sequence"
        )
        return decoded


def decode_block_synchronous(
    emissions: torch.Tensor,
    crf: LinearChainCRF,
    block: int,
) -> torch.Tensor:
    """Viterbi-decode [B, T, V] emissions block by block (backtrack within each block); return [B, T].

    No added latency on a backbone that emits `block` frames per forward call.
    """
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    dec = StreamingViterbi(crf, lag=max(block - 1, 0))
    T = emissions.shape[1]
    out = []
    for start in range(0, T, block):
        n = min(block, T - start)
        for i in range(n):
            dec.step(emissions[:, start + i])   # advance the recursion; ignore its emission
        out.append(dec.emit_block(n))
    decoded = torch.cat(out, dim=1)
    assert decoded.shape[1] == T, (decoded.shape, T)
    return decoded


def build_self_bias_crf(
    num_tags: int,
    self_bias: float,
    device: Optional[torch.device] = None,
) -> LinearChainCRF:
    """Untrained CRF with transitions = self_bias * I (a switch penalty; in nats on log-probs)."""
    crf = LinearChainCRF(num_tags)
    with torch.no_grad():
        crf.transitions.zero_()
        crf.transitions.fill_diagonal_(self_bias)
        crf.start_transitions.zero_()
        crf.end_transitions.zero_()
    if device is not None:
        crf = crf.to(device)
    return crf.eval()


# ---------------------------------------------------------------------------
# Chord post-processing: self-bias CRF Viterbi at a chosen latency
# ---------------------------------------------------------------------------


_PP_METHODS = ("none", "blocksync", "offline")


def emission_block_for_config(config: dict):
    """Pooled frames the backbone emits per forward call (streaming: from the checkpoint's
    lookahead block; sliding: one hop; chunked: one chunk; offline: None)."""
    label_rate = config["sample_rate"] / config["hop_size"] / config["pool_factor"]
    if config.get("backbone", "pretrained") in ("streaming", "small_streaming"):
        raw = backbone_lookahead_block(config.get("mert_align_checkpoint"))
        return max(1, raw // config["pool_factor"])
    if config.get("hop_duration") is not None:
        return max(1, round(config["hop_duration"] * label_rate))
    if config.get("chunk_duration") is not None:
        return max(1, round(config["chunk_duration"] * label_rate))
    return None


class ChordPostProcessor:
    """Per-frame logits -> labels via a self-bias CRF.

    method: 'none' (argmax), 'blocksync' (no added latency) or 'offline'
    (whole track, not causal).
    """

    def __init__(self, num_tags: int, self_bias: float, method: str = "blocksync",
                 block: Optional[int] = None,
                 emission_mode: str = "logprobs",
                 device: Optional[torch.device] = None):
        if method not in _PP_METHODS:
            raise ValueError(f"method must be one of {_PP_METHODS}, got {method!r}")
        if emission_mode not in ("logprobs", "logits"):
            raise ValueError(f"emission_mode must be 'logprobs' or 'logits', got {emission_mode!r}")
        if method == "blocksync":
            if not block or block < 1:
                raise ValueError(
                    f"method='blocksync' needs block >= 1, got {block!r}. Pass block='auto' in "
                    "the config to take it from the backbone's own emission granularity."
                )

        self.method = method
        self.self_bias = float(self_bias)
        self.block = block
        self.emission_mode = emission_mode
        self.crf = (None if method == "none"
                    else build_self_bias_crf(num_tags, self_bias, device))

    def describe(self) -> str:
        if self.method == "none":
            return "post-processing: none (per-frame argmax)"
        where = {"blocksync": f"block-synchronous, block={self.block} frames (ZERO added latency)",
                 "offline": "whole-track Viterbi (NOT causal)"}[self.method]
        return (f"post-processing: self-bias CRF lambda={self.self_bias:g} nats on "
                f"{self.emission_mode}, {where}")

    def decode(self, logits: torch.Tensor) -> list:
        """Decode one track's logits [T, V] into T frame labels."""
        if self.method == "none":
            return logits.argmax(dim=-1).tolist()
        emissions = (F.log_softmax(logits, dim=-1)
                     if self.emission_mode == "logprobs" else logits)
        crf = self.crf.to(emissions.device)
        if self.method == "blocksync":
            return decode_block_synchronous(emissions.unsqueeze(0), crf, self.block)[0].tolist()
        # offline
        mask = torch.ones(1, emissions.shape[0], dtype=torch.bool, device=emissions.device)
        return crf.decode(emissions.unsqueeze(0), mask)[0].tolist()


def build_chord_postprocessor(config: dict, num_tags: int,
                              device: Optional[torch.device] = None,
                              emission_block: Optional[int] = None) -> ChordPostProcessor:
    """Build a ChordPostProcessor from config['decoder'] (absent = per-frame argmax), e.g.

        decoder:
          method:        blocksync   # none | blocksync | offline
          self_bias:     1.0
          block:         auto        # blocksync only; 'auto' = emission_block
          emission_mode: logprobs    # or logits
    """
    spec = config.get("decoder") or {}
    method = spec.get("method", "none")
    block = spec.get("block", "auto")
    if block == "auto":
        block = emission_block
        if method == "blocksync" and not block:
            raise ValueError(
                "decoder.block is 'auto' but this extractor did not report an emission "
                "granularity (a whole-track/offline backbone has none) -- set an explicit "
                "block, or use method: offline"
            )
    return ChordPostProcessor(
        num_tags=num_tags,
        self_bias=spec.get("self_bias", 1.0),
        method=method,
        block=block,
        emission_mode=spec.get("emission_mode", "logprobs"),
        device=device,
    )


def extract_MERT_features(
    mert: nn.Module,
    processor: Wav2Vec2FeatureExtractor,
    audio: torch.Tensor,
    pool_factor: int = 5
) -> torch.Tensor:
    """Frozen MERT (HF or small student) features, time-pooled: [B, L, T // pool_factor, D]."""
    B = audio.shape[0]
    input_values = audio[:, 0, :]   # [B, T_samples]
    if processor.do_normalize:
        mean = input_values.mean(dim=-1, keepdim=True)
        var = input_values.var(dim=-1, unbiased=False, keepdim=True)
        input_values = (input_values - mean) / torch.sqrt(var + 1e-7)

    with torch.no_grad():
        outputs = mert(input_values=input_values, output_hidden_states=True)
    hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs[1]

    # stack: [L, B, T, D] → permute → [B, L, T, D]
    hidden = torch.stack(hidden_states).permute(1, 0, 2, 3)
    _, n_layers, _, hidden_dim = hidden.shape

    # Temporal pooling: trim to a multiple of pool_factor then reshape+mean
    T = hidden.shape[2]
    T_trim = (T // pool_factor) * pool_factor
    hidden = hidden[:, :, :T_trim, :]                                     # [B, L, T_trim, D]
    T_reduced = T_trim // pool_factor
    hidden = hidden.reshape(B, n_layers, T_reduced, pool_factor, hidden_dim).mean(3)
    return hidden                                                          # [B, L, T_reduced, D]


def load_student_state_dict_from_align_checkpoint(
    checkpoint_path: str,
    expect_positional_embedding: Optional[str] = None,
) -> dict:
    """Extract the student backbone weights from a distillation (Lightning) checkpoint.

    expect_positional_embedding: if set, raise unless the checkpoint was distilled with it.
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
    """Lookahead block (raw frames) recorded in a distillation checkpoint; 1 if none."""
    if not checkpoint_path:
        return 1
    try:
        # mmap avoids loading the whole (large) checkpoint for one key.
        ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    except (RuntimeError, ValueError, TypeError):
        ckpt = torch.load(checkpoint_path, map_location="cpu")   # pre-zipfile checkpoint
    return int(ckpt.get("student_lookahead_block", 1))


def load_backbone_and_processor(config: dict, device: torch.device):
    """Load the frozen 'pretrained' or 'small_bidirectional' backbone and its feature extractor."""
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
        raise ValueError(f"invalid backbone {backbone!r}, expected 'pretrained' or 'small_bidirectional'")

    mert = mert.to(device).eval()
    for p in mert.parameters():
        p.requires_grad_(False)
    return mert, processor


def _check_emb_dim(model: nn.Module, expect_emb_dim: Optional[int], backbone: str) -> None:
    """Raise unless emb_dim equals the backbone's hidden_size (1024 / 768 / 432)."""
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
    """Load the frozen 26M causal student ('small_streaming'); mert_align_checkpoint is required."""
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
    """Load the frozen causal streaming MERT (takes raw waveforms; no feature extractor).

    Args:
        mert_model_name:       base MERT ('m-a-p/MERT-v1-330M' or '-95M')
        streaming_context:     KV-cache window in raw frames
        mert_align_checkpoint: optional train_mert_causal_distill.py checkpoint to load
                               (required for 'small_streaming')
        positional_embedding:  must match the checkpoint; use 'rope'
        backbone:              'streaming' or 'small_streaming'
        num_attention_heads:   'small_streaming' only; must match the distillation run
        expect_emb_dim:        if set, checked against the backbone's hidden_size
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
    # Lookahead block comes from the checkpoint, never the config.
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
    """Single forward pass over the whole input -> [B, L, target_raw_frames // pool_factor, D]."""
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
    audio: torch.Tensor,         # [B, 1, T_samples], already trimmed to an exact multiple of pool_factor*hop_size
    pool_factor: int,
    target_raw_frames: int,
    device: torch.device,
    hop_size: int,
    chunk_raw_frames: int,
) -> torch.Tensor:
    """Incremental extraction: feed chunks of chunk_raw_frames through one streaming session.

    The backbone KV cache is bounded by its context, but this helper retains the
    whole waveform and all hidden chunks before returning pooled full-track
    features. Total memory grows with track length, even at a fixed chunk size.
    Results do not depend on valid chunk size; chunk_raw_frames must be a
    multiple of the backbone's lookahead_block.
    """
    waveform = audio[:, 0, :].to(device)  # [B, T_samples]
    B, total_samples = waveform.shape
    # Chunks must align with the backbone's lookahead block.
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


if __name__ == "__main__":
    # Self-test (no data needed): chunked streaming sessions of the LSTM /
    # Transformer probes must match a whole-sequence pass.  python models.py
    torch.manual_seed(0)
    B, L, T, D, V = 2, 13, 30, 64, 17
    CHUNK = 3                       # 3 label frames = 0.2 s at a 15 Hz label rate
    x = torch.randn(B, L, T, D)

    def chunked(head, use_session: bool) -> torch.Tensor:
        parts = []
        if use_session:
            with head.streaming(B):
                for i in range(0, T, CHUNK):
                    parts.append(head(x[:, :, i : i + CHUNK]))
        else:
            for i in range(0, T, CHUNK):
                parts.append(head(x[:, :, i : i + CHUNK]))
        return torch.cat(parts, dim=1)

    with torch.no_grad():
        lstm = ChordLSTM(V, emb_dim=D, hidden_dim=32, num_layers=2).eval()
        full = lstm(x)
        assert full.shape == (B, T, V)
        # return_state must not perturb the logits any existing caller gets
        assert torch.equal(full, lstm(x, return_state=True)[0])

        streamed = chunked(lstm, use_session=True)
        stateless = chunked(lstm, use_session=False)     # no carried state
        # explicit state threading, for callers not using the session
        state, parts = None, []
        for i in range(0, T, CHUNK):
            y, state = lstm(x[:, :, i : i + CHUNK], state=state, return_state=True)
            parts.append(y)
        manual = torch.cat(parts, dim=1)

        print(f"LSTM        session vs full pass : {(streamed - full).abs().max():.3e}")
        print(f"LSTM  explicit state vs full pass : {(manual - full).abs().max():.3e}")
        print(f"LSTM      no state vs full pass : {(stateless - full).abs().max():.3e}  "
              "<- pre-fix; must be large")
        assert (streamed - full).abs().max() < 1e-5, "LSTM streaming session != full pass"
        assert (manual - full).abs().max() < 1e-5, "LSTM explicit state != full pass"
        assert (stateless - full).abs().max() > 1e-3, (
            "chunking without carried state should differ from a full pass -- if it does "
            "not, this test is not exercising what it thinks it is"
        )
        assert lstm._streaming_state is None and not lstm._is_streaming, "session leaked state"

        # reset_streaming mid-session == starting that window from scratch
        with lstm.streaming(B):
            lstm(x[:, :, :10])
            lstm.reset_streaming()
            after_reset = lstm(x[:, :, 10:20])
        assert (after_reset - lstm(x[:, :, 10:20])).abs().max() < 1e-5, "reset_streaming failed"

        # ChordTransformer: context=None cannot stream; a finite context matches the full pass.
        tf_none = ChordTransformer(V, emb_dim=D, d_model=32, num_heads=4, num_layers=2,
                                  dim_feedforward=64, context=None).eval()
        assert tf_none(x).shape == (B, T, V)         # non-streaming use unaffected
        try:
            with tf_none.streaming(B):
                pass
            raise AssertionError("context=None should refuse to stream")
        except RuntimeError as e:
            assert "context" in str(e)

        tf = ChordTransformer(V, emb_dim=D, d_model=32, num_heads=4, num_layers=2,
                             dim_feedforward=64, context=64).eval()
        full = tf(x)
        streamed = chunked(tf, use_session=True)
        stateless = chunked(tf, use_session=False)   # empty KV cache every chunk
        print(f"Transformer session vs full pass : {(streamed - full).abs().max():.3e}")
        print(f"Transformer no cache vs full pass : {(stateless - full).abs().max():.3e}  "
              "<- pre-fix; must be large")
        assert (streamed - full).abs().max() < 1e-5, "Transformer session != full pass"
        assert (stateless - full).abs().max() > 1e-3

    print("\nprobe streaming self-test passed: chunked sessions reproduce the "
          "whole-sequence pass for both stateful heads")
