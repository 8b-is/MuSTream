from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


DBN_BEATS_PER_BAR = [3, 4]
DBN_MIN_BPM = 55.0
DBN_MAX_BPM = 215.0
DBN_TRANSITION_LAMBDA = 100
POSTPROC_TYPES = ("minimal", "dbn", "dbn_blocksync")


class Postprocessor:
    """Turn frame-wise beat/downbeat logits (1D or batched 2D) into event times in seconds.

    Types:
        minimal:       peak-picking; streamable with 3 frames of lookahead.
        dbn:           madmom DBN (Böck et al.), offline whole-track Viterbi.
        dbn_blocksync: same DBN, decoded per block of `dbn_block` frames the backbone
                       emits per forward call; streamable with no added latency.
    Args:
        type (str): "minimal", "dbn" or "dbn_blocksync".
        fps (int): frame rate of the predictions.
        dbn_block (int): block size in frames ("dbn_blocksync" only).
        dbn_beats_per_bar (int): pin the bar length to 3 or 4 ("dbn_blocksync" only);
            None decides it causally.
        dbn_transition_lambda (float): DBN tempo-change penalty.
    """

    def __init__(self, type: str = "minimal", fps: int = 50,
                 dbn_beats_per_bar: int | None = None, dbn_block: int | None = None,
                 dbn_transition_lambda: float = DBN_TRANSITION_LAMBDA):
        assert type in POSTPROC_TYPES, f"invalid postproc_type {type!r}, expected one of {POSTPROC_TYPES}"
        self.type = type
        self.fps = fps
        self.dbn_block = dbn_block
        dbn_kwargs = dict(
            beats_per_bar=DBN_BEATS_PER_BAR,
            min_bpm=DBN_MIN_BPM,
            max_bpm=DBN_MAX_BPM,
            fps=self.fps,
            transition_lambda=dbn_transition_lambda,
        )
        if type == "dbn":
            from madmom.features.downbeats import DBNDownBeatTrackingProcessor

            self.dbn = DBNDownBeatTrackingProcessor(**dbn_kwargs)
        elif type == "dbn_blocksync":
            if not dbn_block or int(dbn_block) < 1:
                raise ValueError(
                    f"postproc_type 'dbn_blocksync' requires dbn_block >= 1 frames, got "
                    f"{dbn_block!r}. Set `dbn_block: auto` to take it from the backbone's "
                    "own emission granularity (models.emission_block_for_config)."
                )
            self.dbn_block = int(dbn_block)
            # None = choose the bar length causally; an int pins it.
            self.dbn_hmm_index = (
                None if dbn_beats_per_bar is None
                else DBN_BEATS_PER_BAR.index(int(dbn_beats_per_bar))
            )
            self.dbn = BlockSyncDBNDownBeatTracker(**dbn_kwargs)

    def __call__(
        self,
        beat: torch.Tensor,
        downbeat: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Beat/downbeat logits ([T] or [B, T], optional padding mask) -> event times in seconds (per item if batched)."""
        batched = False if beat.ndim == 1 else True
        if padding_mask is None:
            padding_mask = torch.ones_like(beat, dtype=torch.bool)

        # if beat and downbeat are 1D tensors, add a batch dimension
        if not batched:
            beat = beat.unsqueeze(0)
            downbeat = downbeat.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)

        if self.type == "minimal":
            postp_beat, postp_downbeat = self.postp_minimal(
                beat, downbeat, padding_mask
            )
        elif self.type == "dbn":
            postp_beat, postp_downbeat = self.postp_dbn(beat, downbeat, padding_mask)
        elif self.type == "dbn_blocksync":
            postp_beat, postp_downbeat = self.postp_dbn(
                beat, downbeat, padding_mask, item_fn=self._postp_dbn_blocksync_item
            )
        else:
            raise ValueError("Invalid postprocessing type")

        # remove the batch dimension if it was added
        if not batched:
            postp_beat = postp_beat[0]
            postp_downbeat = postp_downbeat[0]

        # update the model prediction dict
        return postp_beat, postp_downbeat

    def postp_minimal(self, beat, downbeat, padding_mask):
        # concatenate beat and downbeat in the same tensor of shape (B, T, 2)
        packed_pred = rearrange(
            [beat, downbeat], "c b t -> b t c", b=beat.shape[0], t=beat.shape[1], c=2
        )
        # set padded elements to -1000 (= probability zero even in float64) so they don't influence the maxpool
        pred_logits = packed_pred.masked_fill(~padding_mask.unsqueeze(-1), -1000)
        # reshape to (2*B, T) to apply max pooling
        pred_logits = rearrange(pred_logits, "b t c -> (c b) t")
        # pick maxima within +/- 70ms
        pred_peaks = pred_logits.masked_fill(
            pred_logits != F.max_pool1d(pred_logits, 7, 1, 3), -1000
        )
        # keep maxima with over 0.5 probability (logit > 0)
        pred_peaks = pred_peaks > 0
        #  rearrange back to two tensors of shape (B, T)
        beat_peaks, downbeat_peaks = rearrange(
            pred_peaks, "(c b) t -> c b t", b=beat.shape[0], t=beat.shape[1], c=2
        )
        # run the piecewise operations
        with ThreadPoolExecutor() as executor:
            postp_beat, postp_downbeat = zip(
                *executor.map(
                    self._postp_minimal_item, beat_peaks, downbeat_peaks, padding_mask
                )
            )
        return postp_beat, postp_downbeat

    def _postp_minimal_item(self, padded_beat_peaks, padded_downbeat_peaks, mask):
        """Function to compute the operations that must be computed piece by piece, and cannot be done in batch."""
        # unpad the predictions by truncating the padding positions
        beat_peaks = padded_beat_peaks[mask]
        downbeat_peaks = padded_downbeat_peaks[mask]
        # pass from a boolean array to a list of times in frames.
        beat_frame = torch.nonzero(beat_peaks).cpu().numpy()[:, 0]
        downbeat_frame = torch.nonzero(downbeat_peaks).cpu().numpy()[:, 0]
        # remove adjacent peaks
        beat_frame = deduplicate_peaks(beat_frame)
        downbeat_frame = deduplicate_peaks(downbeat_frame)
        # convert from frame to seconds
        beat_time = beat_frame / self.fps
        downbeat_time = downbeat_frame / self.fps
        # move the downbeat to the nearest beat
        if (
            len(beat_time) > 0
        ):  # skip if there are no beats, like in the first training steps
            for i, d_time in enumerate(downbeat_time):
                beat_idx = np.argmin(np.abs(beat_time - d_time))
                downbeat_time[i] = beat_time[beat_idx]
        # remove duplicate downbeat times (if some db were moved to the same position)
        downbeat_time = np.unique(downbeat_time)
        return beat_time, downbeat_time

    def postp_dbn(self, beat, downbeat, padding_mask, item_fn=None):
        beat_prob = beat.double().sigmoid()
        downbeat_prob = downbeat.double().sigmoid()
        # limit lower and upper bound, since 0 and 1 create problems in the DBN
        epsilon = 1e-5
        beat_prob = beat_prob * (1 - epsilon) + epsilon / 2
        downbeat_prob = downbeat_prob * (1 - epsilon) + epsilon / 2
        with ThreadPoolExecutor() as executor:
            postp_beat, postp_downbeat = zip(
                *executor.map(
                    item_fn or self._postp_dbn_item, beat_prob, downbeat_prob, padding_mask
                )
            )
        return postp_beat, postp_downbeat

    def _postp_dbn_item(self, padded_beat_prob, padded_downbeat_prob, mask):
        """Function to compute the operations that must be computed piece by piece, and cannot be done in batch."""
        combined_act = combined_activation(
            padded_beat_prob[mask], padded_downbeat_prob[mask]
        )
        # run the DBN
        dbn_out = self.dbn(combined_act)
        postp_beat = dbn_out[:, 0]
        postp_downbeat = dbn_out[dbn_out[:, 1] == 1][:, 0]
        return postp_beat, postp_downbeat

    def _postp_dbn_blocksync_item(self, padded_beat_prob, padded_downbeat_prob, mask):
        """Same as _postp_dbn_item, but decoded block-synchronously -- streamable at
        ZERO latency beyond the block the backbone already delivers."""
        combined_act = combined_activation(
            padded_beat_prob[mask], padded_downbeat_prob[mask]
        )
        dbn_out = self.dbn.decode_blocksync(
            combined_act, self.dbn_block, hmm_index=self.dbn_hmm_index)
        postp_beat = dbn_out[:, 0]
        postp_downbeat = dbn_out[dbn_out[:, 1] == 1][:, 0]
        return postp_beat, postp_downbeat


def resolve_dbn_block(config: dict) -> int:
    """Block size (frames) for dbn_blocksync: config['dbn_block'], or the backbone's own if 'auto'.

    Raises for offline backbones, which emit no blocks.
    """
    block = config.get("dbn_block", "auto")
    if block not in (None, "auto"):
        return int(block)
    # Deferred import so the self-test doesn't need transformers/moshi.
    from models import emission_block_for_config

    block = emission_block_for_config(config)
    if not block:
        raise ValueError(
            "postproc_type 'dbn_blocksync' with dbn_block: auto, but this config's backbone "
            "emits no block of frames per forward call (a full-crop/offline backbone reads "
            "the whole track at once) -- set an explicit dbn_block, or use postproc_type: dbn"
        )
    return block


def build_postprocessor(config: dict, fps: float) -> Postprocessor:
    """Build the Postprocessor from config's postproc_type / dbn_* keys."""
    postproc_type = config.get("postproc_type", "minimal")
    return Postprocessor(
        type=postproc_type,
        fps=fps,
        dbn_beats_per_bar=config.get("dbn_beats_per_bar"),
        dbn_block=resolve_dbn_block(config) if postproc_type == "dbn_blocksync" else None,
        dbn_transition_lambda=config.get("dbn_transition_lambda", DBN_TRANSITION_LAMBDA),
    )


def postproc_run_suffix(config: dict) -> str:
    """Run-name suffix identifying the postprocessor settings."""
    postproc_type = config.get("postproc_type", "minimal")
    lambda_suffix = ""
    transition_lambda = config.get("dbn_transition_lambda", DBN_TRANSITION_LAMBDA)
    if postproc_type != "minimal" and transition_lambda != DBN_TRANSITION_LAMBDA:
        lambda_suffix = f"_tl{transition_lambda:g}"
    if postproc_type == "dbn":
        return "_dbn" + lambda_suffix
    if postproc_type == "dbn_blocksync":
        suffix = f"_dbnblk{resolve_dbn_block(config)}"
    else:
        return ""
    if config.get("dbn_beats_per_bar"):
        suffix += f"_bpb{config['dbn_beats_per_bar']}"
    return suffix + lambda_suffix


def emission_time(ready_frames, block: int, fps: float):
    """Seconds at which a decision needing frames up to `ready_frames` can be
    announced, when the backbone delivers frames in blocks of `block`."""
    ready = np.asarray(ready_frames, dtype=np.float64)
    return (np.floor(ready / block) + 1) * block / float(fps)


def combined_activation(beat_prob, downbeat_prob) -> np.ndarray:
    """2-column DBN activation [beat-but-not-downbeat, downbeat] from per-frame probabilities."""
    epsilon = 1e-5
    beat_prob = np.asarray(beat_prob.cpu() if hasattr(beat_prob, "cpu") else beat_prob)
    downbeat_prob = np.asarray(
        downbeat_prob.cpu() if hasattr(downbeat_prob, "cpu") else downbeat_prob
    )
    return np.vstack(
        (np.maximum(beat_prob - downbeat_prob, epsilon / 2), downbeat_prob)
    ).T


class _BarHMMDecoder:
    """Frame-by-frame Viterbi forward recursion over one madmom bar HMM (same as
    madmom's viterbi), keeping a ring buffer of the last `ring_size` frames of backpointers.

    Args:
        hmm: a madmom HiddenMarkovModel from DBNDownBeatTrackingProcessor.hmms
        ring_size: frames of backpointers to keep (= the largest block)
    """

    def __init__(self, hmm, ring_size: int):
        tm = hmm.transition_model
        self.prev_states = tm.states.astype(np.int64)
        self.pointers = tm.pointers.astype(np.int64)
        self.log_trans = tm.log_probabilities.astype(np.float64)
        self.num_states = int(tm.num_states)
        self.seg_len = np.diff(self.pointers)
        assert self.seg_len.min() >= 1, "every state must have an incoming transition"
        self.seg_id = np.repeat(np.arange(self.num_states), self.seg_len)
        self.om_pointers = hmm.observation_model.pointers.astype(np.int64)
        self.state_positions = tm.state_space.state_positions
        # log_delta is renormalised to max 0 each frame; log_prob accumulates the shift
        # (= running best-path log probability, comparable across HMMs).
        self.log_delta = np.log(hmm.initial_distribution)
        self.log_prob = 0.0
        self.ring = np.empty((ring_size, self.num_states), dtype=np.int32)
        self.ring_size = ring_size

    def step(self, log_density_row: np.ndarray, ring_idx: int) -> None:
        """Advance one frame; writes this frame's backpointers into ring[ring_idx]."""
        cand = self.log_delta[self.prev_states] + self.log_trans
        best = np.maximum.reduceat(cand, self.pointers[:-1])
        # Per-state argmax over incoming transitions; ties go to the lowest index (as madmom).
        hit = np.flatnonzero(cand >= np.repeat(best, self.seg_len))
        arg = self.pointers[:-1].copy()
        arg[self.seg_id[hit][::-1]] = hit[::-1]
        self.ring[ring_idx] = self.prev_states[arg]
        delta = best + log_density_row[self.om_pointers]
        shift = float(delta.max())
        self.log_delta = delta - shift
        self.log_prob += shift


class BlockSyncDBNDownBeatTracker:
    """Block-synchronous (streamable) decoding of madmom's downbeat DBN.

    The Viterbi forward recursion runs frame by frame; at each block boundary the
    block's frames are committed by backtracking from the current best state. The
    3-vs-4 meter is chosen causally from the accumulated log probability; no
    threshold trimming.

    Args are DBNDownBeatTrackingProcessor's, so the HMMs match the offline path.
    """

    def __init__(
        self,
        beats_per_bar=(3, 4),
        min_bpm: float = DBN_MIN_BPM,
        max_bpm: float = DBN_MAX_BPM,
        fps: int = 50,
        transition_lambda: float = DBN_TRANSITION_LAMBDA,
        **kwargs,
    ):
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor

        self.fps = fps
        self._proc = DBNDownBeatTrackingProcessor(
            beats_per_bar=list(beats_per_bar),
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            fps=fps,
            transition_lambda=transition_lambda,
            **kwargs,
        )
        self.hmms = self._proc.hmms

    def best_meter(self, activations: np.ndarray) -> int:
        """Index of the bar-length HMM the offline DBN would pick (not causal)."""
        return int(np.argmax([hmm.viterbi(activations)[1] for hmm in self.hmms]))

    def decode_blocksync(self, activations: np.ndarray, block: int,
                         hmm_index=None, return_ready: bool = False) -> np.ndarray:
        """Block-synchronous decoding: commit each block of `block` frames when it lands.

        The forward recursion runs across blocks; at each boundary the backtrack covers
        just that block. block >= num_frames equals the offline DBN.

        Args:
            activations: (num_frames, 2), as built by combined_activation().
            block: block size in frames.
            hmm_index: pin the bar length to self.hmms[hmm_index]; None = choose causally.
            return_ready: also return, per beat, the frame that completed its decision.
        """
        return self.decode_blocks(activations, [block], hmm_index=hmm_index,
                                  return_ready=return_ready)[int(block)]

    def decode_blocks(self, activations: np.ndarray, blocks, hmm_index=None,
                      return_ready: bool = False) -> dict:
        """decode_blocksync() for several block sizes from one forward pass.

        Returns:
            {block: (num_beats, 2) array of [time in seconds, beat number]}
        """
        blocks = sorted({int(b) for b in blocks})
        assert blocks and blocks[0] >= 1, "blocks must be positive frame counts"
        num_frames = len(activations)

        # One ring of the largest block serves every size.
        ring_size = blocks[-1]
        hmm_ids = list(range(len(self.hmms))) if hmm_index is None else [hmm_index]
        decoders = [_BarHMMDecoder(self.hmms[i], ring_size) for i in hmm_ids]
        log_densities = [
            self.hmms[i].observation_model.log_densities(activations) for i in hmm_ids
        ]

        paths = {b: np.full(num_frames, -1, dtype=np.int64) for b in blocks}
        owners = {b: np.zeros(num_frames, dtype=np.int8) for b in blocks}
        # First frame of the block each size is currently accumulating.
        starts = {b: 0 for b in blocks}

        for frame in range(num_frames):
            ring_idx = frame % ring_size
            for decoder, log_density in zip(decoders, log_densities):
                decoder.step(log_density[frame], ring_idx)
            # Blocks close on schedule; the last frame closes any open (partial) block.
            last = frame == num_frames - 1
            closing = [b for b in blocks if last or (frame + 1) % b == 0]
            if not closing:
                continue
            # Causal meter choice: HMM with the best path so far.
            best = int(np.argmax([decoder.log_prob for decoder in decoders]))
            winner = hmm_ids[best]
            decoder = decoders[best]
            top = int(np.argmax(decoder.log_delta))
            for b in closing:
                state = top
                paths[b][frame] = state
                owners[b][frame] = winner
                for j in range(1, frame - starts[b] + 1):
                    state = int(decoder.ring[(frame - j + 1) % ring_size][state])
                    paths[b][frame - j] = state
                    owners[b][frame - j] = winner
                starts[b] = frame + 1

        return {
            b: self._beats_from_path(activations, paths[b], owners[b],
                                     return_ready=return_ready)
            for b in blocks
        }

    def _beats_from_path(self, activations, path, owner,
                         return_ready: bool = False) -> np.ndarray:
        """Decoded state path -> (time, beat number) pairs, as madmom's correct=True peak-picking.

        With return_ready=True, also returns per beat the first frame after its state run.
        """
        num_frames = len(path)
        if num_frames == 0:
            return (np.empty((0, 2)), np.empty(0, dtype=np.int64)) if return_ready \
                else np.empty((0, 2))
        assert (path >= 0).all(), "every frame must be committed"

        in_beat_range = np.empty(num_frames, dtype=bool)
        positions = np.empty(num_frames, dtype=np.float64)
        for idx, hmm in enumerate(self.hmms):
            sel = owner == idx
            if not sel.any():
                continue
            states = path[sel]
            in_beat_range[sel] = hmm.observation_model.pointers[states] >= 1
            positions[sel] = hmm.transition_model.state_space.state_positions[states]
        beat_numbers = positions.astype(int) + 1

        boundary = np.ones(num_frames, dtype=bool)
        boundary[1:] = (in_beat_range[1:] != in_beat_range[:-1]) | (
            owner[1:] != owner[:-1]
        )
        starts = np.flatnonzero(boundary)
        ends = np.r_[starts[1:], num_frames]
        runs = [(left, right) for left, right in zip(starts, ends) if in_beat_range[left]]
        beats = [
            # argmax over both activation columns (as madmom), hence //2.
            int(np.argmax(activations[left:right])) // 2 + left
            for left, right in runs
        ]
        if not beats:
            return (np.empty((0, 2)), np.empty(0, dtype=np.int64)) if return_ready \
                else np.empty((0, 2))
        beats = np.asarray(beats, dtype=np.int64)
        out = np.vstack((beats / float(self.fps), beat_numbers[beats])).T
        if not return_ready:
            return out
        ready = np.asarray([min(right, num_frames - 1) for _left, right in runs],
                           dtype=np.int64)
        return out, ready


class DecoderBank:
    """Several postprocessors applied per track, each also returning per-beat emission times.

    Args:
        specs: [(label, kind, param)], kind one of 'minimal', 'offline' or
            'blocksync' (param = block in frames).
        fps: frame rate of the logits.
        deploy_block: frames the backbone emits per forward call; None = offline
            backbone (everything is announced at the end of the track).
        transition_lambda: DBN tempo-change penalty.
        beats_per_bar: pin the bar length (3 or 4) for causal rows; None = causal choice.
    """

    def __init__(self, specs, fps: float, deploy_block: int,
                 transition_lambda: float = DBN_TRANSITION_LAMBDA,
                 beats_per_bar: int | None = None):
        self.specs = [(label, kind, param) for label, kind, param in specs]
        self.fps = fps
        self.deploy_block = None if deploy_block is None else max(1, int(deploy_block))
        self.blocks = sorted({int(p) for _l, k, p in self.specs if k == "blocksync"})
        self.hmm_index = (None if beats_per_bar is None
                          else DBN_BEATS_PER_BAR.index(int(beats_per_bar)))
        dbn_kwargs = dict(beats_per_bar=DBN_BEATS_PER_BAR, min_bpm=DBN_MIN_BPM,
                          max_bpm=DBN_MAX_BPM, fps=fps,
                          transition_lambda=transition_lambda)
        self.minimal = (Postprocessor(type="minimal", fps=fps)
                        if any(k == "minimal" for _l, k, _p in self.specs) else None)
        self.offline = None
        if any(k == "offline" for _l, k, _p in self.specs):
            from madmom.features.downbeats import DBNDownBeatTrackingProcessor

            self.offline = DBNDownBeatTrackingProcessor(**dbn_kwargs)
        self.tracker = BlockSyncDBNDownBeatTracker(**dbn_kwargs) if self.blocks else None

    def activation(self, beat_logits, downbeat_logits) -> np.ndarray:
        """Logits -> 2-column DBN activation (same clipping as Postprocessor.postp_dbn)."""
        epsilon = 1e-5
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))
        return combined_activation(sigmoid(beat_logits) * (1 - epsilon) + epsilon / 2,
                                   sigmoid(downbeat_logits) * (1 - epsilon) + epsilon / 2)

    def best_meter(self, beat_logits, downbeat_logits) -> int:
        """Bar-length HMM index the offline DBN picks (not causal)."""
        return self.tracker.best_meter(self.activation(beat_logits, downbeat_logits))

    def __call__(self, beat_logits, downbeat_logits, hmm_index="default", labels=None):
        """Decode one track; returns {label: (beat times, downbeat times, emission times)}.

        hmm_index: 'default' = the bank's beats_per_bar; None or an index overrides it.
        labels: restrict to these spec labels.
        """
        if hmm_index == "default":
            hmm_index = self.hmm_index
        wanted = [s for s in self.specs if labels is None or s[0] in labels]
        num_frames = len(np.asarray(beat_logits))
        out = {}

        if any(k in ("blocksync", "offline") for _l, k, _p in wanted):
            act = self.activation(beat_logits, downbeat_logits)

        blocks = sorted({p for _l, k, p in wanted if k == "blocksync"})
        block_out = (self.tracker.decode_blocks(act, blocks, hmm_index=hmm_index,
                                                return_ready=True) if blocks else {})

        # Offline backbone: everything is announced at the end of the track.
        whole_track = self.deploy_block is None

        for label, kind, param in wanted:
            if kind == "minimal":
                beats, downbeats = self.minimal(
                    torch.as_tensor(np.asarray(beat_logits, dtype=np.float64)),
                    torch.as_tensor(np.asarray(downbeat_logits, dtype=np.float64)))
                # minimal needs frames up to t+3.
                ready = np.round(np.asarray(beats) * self.fps).astype(np.int64) + 3
                emit = (np.full(len(beats), num_frames / self.fps) if whole_track
                        else emission_time(np.minimum(ready, num_frames - 1),
                                           self.deploy_block, self.fps))
            elif kind == "offline":
                dbn_out = self.offline(act)
                beats, downbeats = dbn_out[:, 0], dbn_out[dbn_out[:, 1] == 1][:, 0]
                emit = np.full(len(beats), num_frames / self.fps)
            else:
                dbn_out, ready = block_out[int(param)]
                beats, downbeats = dbn_out[:, 0], dbn_out[dbn_out[:, 1] == 1][:, 0]
                if whole_track:
                    emit = np.full(len(beats), num_frames / self.fps)
                else:
                    # Effective block is at least the backbone's own.
                    emit = emission_time(ready, max(int(param), self.deploy_block), self.fps)
            out[label] = (np.asarray(beats, dtype=np.float64),
                          np.asarray(downbeats, dtype=np.float64),
                          np.asarray(emit, dtype=np.float64))
        return out


def test_blocksync_viterbi(num_frames: int = 900, fps: int = 75, seed: int = 0) -> None:
    """Self-test of the block-synchronous decoder against madmom. Run `python postprocessor.py`."""
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor

    rng = np.random.RandomState(seed)
    beat_prob = np.full(num_frames, 0.02)
    downbeat_prob = np.full(num_frames, 0.01)
    for i, frame in enumerate(range(7, num_frames, 34)):  # ~132 bpm
        beat_prob[frame] = 0.9
        if i % 4 == 0:
            downbeat_prob[frame] = 0.8
    beat_prob = np.clip(beat_prob + rng.rand(num_frames) * 0.03, 1e-5, 0.98)
    downbeat_prob = np.clip(downbeat_prob + rng.rand(num_frames) * 0.03, 1e-5, 0.98)
    act = combined_activation(beat_prob, downbeat_prob)

    tracker = BlockSyncDBNDownBeatTracker(
        beats_per_bar=DBN_BEATS_PER_BAR, min_bpm=DBN_MIN_BPM, max_bpm=DBN_MAX_BPM,
        fps=fps, transition_lambda=DBN_TRANSITION_LAMBDA,
    )
    offline = DBNDownBeatTrackingProcessor(
        beats_per_bar=DBN_BEATS_PER_BAR, min_bpm=DBN_MIN_BPM, max_bpm=DBN_MAX_BPM,
        fps=fps, transition_lambda=DBN_TRANSITION_LAMBDA, threshold=0,
    )(act)
    # --- block-synchronous decoding ---------------------------------------
    # block >= T == offline.
    assert np.allclose(tracker.decode_blocksync(act, num_frames), offline), \
        "blocksync block>=T must reproduce madmom's offline path"
    print(f"OK: blocksync block={num_frames} reproduces madmom's offline path ({len(offline)} beats)")

    block_list = [1, 5, 15, 30, 75]
    together = tracker.decode_blocks(act, block_list)
    for block in block_list:
        assert np.allclose(together[block], tracker.decode_blocksync(act, block)), block
    print(f"OK: decode_blocks({block_list}) matches one decode_blocksync per block")

    for block in (5, 15, 30, 75):
        beats, ready = tracker.decode_blocksync(act, block, return_ready=True)
        matched = sum(bool(np.any(np.abs(offline[:, 0] - t) <= 0.07)) for t in beats[:, 0])
        latency = emission_time(ready, block, fps) - beats[:, 0]
        print(f"  block={block:>3} ({block / fps * 1000:>5.0f} ms): "
              f"{matched}/{len(beats)} beats within 70 ms of offline, "
              f"emission latency mean {latency.mean() * 1000:.0f} ms / "
              f"max {latency.max() * 1000:.0f} ms")

    # minimal's latency, for reference.
    beat_logits = torch.from_numpy(np.log(beat_prob / (1 - beat_prob)))
    downbeat_logits = torch.from_numpy(np.log(downbeat_prob / (1 - downbeat_prob)))
    minimal_beats = Postprocessor(type="minimal", fps=fps)(beat_logits, downbeat_logits)[0]
    for block in (15,):
        ready = np.round(np.asarray(minimal_beats) * fps).astype(np.int64) + 3
        latency = emission_time(ready, block, fps) - np.asarray(minimal_beats)
        print(f"  minimal, block={block}: emission latency mean "
              f"{latency.mean() * 1000:.0f} ms / max {latency.max() * 1000:.0f} ms")

    # --- the three entry points must agree -------------------------------------
    # Postprocessor, DecoderBank and decode_blocksync must agree.
    block = 15
    via_postproc = Postprocessor(type="dbn_blocksync", fps=fps, dbn_block=block)(
        beat_logits, downbeat_logits)
    bank = DecoderBank([("minimal", "minimal", 0), ("blk", "blocksync", block),
                        ("off", "offline", 0)], fps=fps, deploy_block=block)
    via_bank = bank(beat_logits.numpy(), downbeat_logits.numpy())
    assert np.allclose(via_postproc[0], via_bank["blk"][0]), "Postprocessor != DecoderBank"
    assert np.allclose(via_postproc[1], via_bank["blk"][1]), "downbeats disagree"
    assert np.allclose(via_bank["blk"][0], tracker.decode_blocksync(act, block)[:, 0])
    assert np.allclose(via_bank["minimal"][0], minimal_beats)
    assert np.allclose(via_bank["off"][0], offline[:, 0])
    print("OK: Postprocessor, DecoderBank and decode_blocksync agree on the same track")


def deduplicate_peaks(peaks, width=1) -> np.ndarray:
    """Merge runs of peak frame indices at most `width` apart into their mean."""
    result = []
    peaks = map(int, peaks)  # ensure we get ordinary Python int objects
    try:
        p = next(peaks)
    except StopIteration:
        return np.array(result)
    c = 1
    for p2 in peaks:
        if p2 - p <= width:
            c += 1
            p += (p2 - p) / c  # update mean
        else:
            result.append(p)
            p = p2
            c = 1
    result.append(p)
    return np.array(result)

if __name__ == "__main__":
    test_blocksync_viterbi()
