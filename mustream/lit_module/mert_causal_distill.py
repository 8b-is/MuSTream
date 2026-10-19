"""
Lightning module for causal MERT distillation: trains a streaming
(KV-cache, block-causal) MERT student to match a frozen bidirectional MERT
teacher with a cosine + L2 loss.

Students:
  'streaming'        MERT's own width, initialized from the pretrained MERT.
  'small_streaming'  small width (modeling/mert/small_mert.py), random init or
                     warm-started from a small bidirectional teacher checkpoint.

The teacher is the published `mert_model`, or a small bidirectional student from
train_mert_size_distill.py (teacher_checkpoint). When the widths differ,
per-layer linear projections map the student into the teacher's hidden size.
Small bidirectional students are trained by lit_module/mert_size_distill.py.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor

import argbind

from base_trainer import BaseLightningModel
from lr_scheduler import LinearWarmupCosineDecay
from dataloader import get_audio_dataloader
from evaluate_mert_distill import MERTChordEvaluator, MERTBeatEvaluator

bind = argbind.bind

# Causal students only; small bidirectional ones train in mert_size_distill.py.
STUDENT_MODELS = ("streaming", "small_streaming")

AdamW = bind(torch.optim.AdamW)
LinearWarmupCosineDecay = bind(LinearWarmupCosineDecay)
get_dataloader = bind(get_audio_dataloader, without_prefix=True)
MERTChordEvaluator = bind(MERTChordEvaluator)
MERTBeatEvaluator = bind(MERTBeatEvaluator)


def _select_distill_layer(hidden: torch.Tensor, distill_layer: Union[int, str]) -> torch.Tensor:
    """[B, L, T, D] -> [B, T, D], except 'all' which leaves the layer axis in
    place ([B, L, T, D]) so compute_losses can compare every layer
    side-by-side instead of first reducing across layers."""
    if distill_layer == "average":
        return hidden.mean(dim=1)
    if distill_layer == "final":
        return hidden[:, -1, :, :]
    if distill_layer == "all":
        return hidden
    if isinstance(distill_layer, int):
        return hidden[:, distill_layer, :, :]
    raise ValueError(f"invalid distill_layer {distill_layer!r}")


@bind(without_prefix=True)
class LitMERTCausalDistill(BaseLightningModel):
    def __init__(
        self,
        compile: bool = True,
        train_split: str = "",
        valid_split: str = "",
        sample_rate: int = 24000,
        hop_length: int = 320,
        mert_model: str = "m-a-p/MERT-v1-330M",
        teacher_checkpoint: str = "",       # distill from a small student's own checkpoint instead of mert_model
        teacher_num_attention_heads: Optional[int] = None,  # only used with teacher_checkpoint; defaults to student_num_attention_heads
        distill_layer: Union[int, str] = "average",
        distill_intermediate_weight: float = 1.0,
        distill_final_weight: float = 1.0,
        student_model: str = "streaming",  # 'streaming' | 'small_streaming'
        streaming_context: int = 1000,
        student_positional_embedding: str = "rope",  # 'rope' | 'sin'
        student_hidden_size: int = 432,             # only used when student_model == 'small_streaming'
        student_intermediate_size: int = 1392,      # only used when student_model == 'small_streaming'
        student_num_attention_heads: int = 8,       # only used when student_model == 'small_streaming'
        student_num_hidden_layers: Optional[int] = None,  # only used when student_model == 'small_streaming'; defaults to the teacher's own depth
        student_init_from_teacher: bool = True,  # only used when student_model == 'small_streaming' and teacher_checkpoint is set
        student_lookahead_block: int = 1,   # block-causal right context, in frames; 1 = strictly causal
        teacher_window_frames: int = 0,     # local-interval teacher length in frames; 0 = full-crop teacher
        teacher_supervise_frames: int = 15, # frames kept at each interval's right edge
        teacher_windows_per_crop: int = 4,  # intervals drawn per crop per step
        teacher_predict_frames: int = 0,    # frames the teacher sees past the block; 0 = reachable target
        teacher_prefix_windows: bool = False,  # also supervise one block at the stream start (training only)
        cosine_loss_weight: float = 1.0,
        l2_loss_weight: float = 0.1,
    ):
        """
        Trains a streaming MERT student against a frozen bidirectional teacher.

        Model:
          mert_model:          published MERT; the default teacher, and the source
                               of the shared config and waveform normalization.
          teacher_checkpoint:  use a small bidirectional student checkpoint as the
                               teacher; teacher_num_attention_heads defaults to
                               student_num_attention_heads.
          student_model:       'streaming' | 'small_streaming'.
          student_hidden_size / student_intermediate_size /
          student_num_attention_heads / student_num_hidden_layers:
                               small_streaming architecture (STaRHuBERT-L defaults;
                               depth defaults to the teacher's).
          student_init_from_teacher:
                               small_streaming with teacher_checkpoint only:
                               warm-start from the teacher's weights (the
                               architecture must then match the teacher's).
          streaming_context:   KV-cache attention window in frames.
          student_positional_embedding:
                               'rope' | 'sin'; stored in the checkpoint.
          student_lookahead_block:
                               block size C; each frame sees up to C-1 future
                               frames (1 = strictly causal). Stored in the checkpoint.

        Local-interval teacher (see lit_module/distill_targets.py): the teacher
        only sees a local interval, so its future context is limited to at most
        teacher_supervise_frames - 1 + teacher_predict_frames frames.
          teacher_window_frames:    interval length in frames up to the block's
                                    end; 0 = full-crop teacher.
          teacher_supervise_frames: frames supervised at each interval's right edge;
                                    must be a multiple of student_lookahead_block.
          teacher_windows_per_crop: intervals drawn per crop per step.
          teacher_predict_frames:   future frames the teacher sees past each block
                                    that the student cannot (0 = reachable target).
          teacher_prefix_windows:   training only: also supervise one block at the
                                    start of the stream.

        Loss:
          distill_layer:       'average' | 'final' | 'all' | int layer index.
          distill_intermediate_weight / distill_final_weight:
                               for 'all': weight of layers 0..-2 vs. the final layer.
          cosine_loss_weight / l2_loss_weight: weights of the two loss terms.
        """
        super(LitMERTCausalDistill, self).__init__()
        assert student_model in STUDENT_MODELS, (student_model, STUDENT_MODELS)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.distill_layer = distill_layer
        self.distill_intermediate_weight = distill_intermediate_weight
        self.distill_final_weight = distill_final_weight
        self.student_model_name = student_model
        self.mert_model_name = mert_model
        self.student_positional_embedding = student_positional_embedding
        # Block-causal right context, a property of moshi's attention mask.
        assert student_lookahead_block >= 1, student_lookahead_block
        self.student_lookahead_block = student_lookahead_block

        assert teacher_window_frames >= 0, teacher_window_frames
        assert teacher_supervise_frames >= 1, teacher_supervise_frames
        assert teacher_windows_per_crop >= 1, teacher_windows_per_crop
        assert teacher_predict_frames >= 0, teacher_predict_frames
        if teacher_predict_frames and not teacher_window_frames:
            raise ValueError(
                f"teacher_predict_frames={teacher_predict_frames} only applies to the "
                "local-interval teacher; set teacher_window_frames > 0 too"
            )
        if teacher_prefix_windows and not teacher_window_frames:
            raise ValueError(
                "teacher_prefix_windows only applies to the local-interval teacher; set "
                "teacher_window_frames > 0 too. The full-crop teacher already supervises "
                "every frame of the crop, the first ones included."
            )
        if teacher_prefix_windows and teacher_window_frames // teacher_supervise_frames < 2:
            raise ValueError(
                f"teacher_prefix_windows has no grid point to draw: "
                f"teacher_window_frames={teacher_window_frames} is under two "
                f"teacher_supervise_frames={teacher_supervise_frames} steps, so the only end "
                "below the first usable one would be the interval itself"
            )
        if teacher_window_frames:
            if teacher_supervise_frames > teacher_window_frames:
                raise ValueError(
                    f"teacher_supervise_frames={teacher_supervise_frames} exceeds the interval "
                    f"itself (teacher_window_frames={teacher_window_frames})"
                )
            # Interval ends sit on a multiple-of-W grid; for a block-causal
            # student those ends must also be block boundaries, or the two
            # would disagree about how much future each frame is allowed.
            if student_lookahead_block > 1 and teacher_supervise_frames % student_lookahead_block:
                raise ValueError(
                    f"teacher_supervise_frames={teacher_supervise_frames} must be a multiple of "
                    f"student_lookahead_block={student_lookahead_block} so interval ends land on "
                    "block boundaries"
                )
            print(
                f"local-interval teacher: {teacher_windows_per_crop} intervals/crop x "
                f"{teacher_window_frames} frames ({teacher_window_frames / 75:.2f}s), "
                f"supervising the last {teacher_supervise_frames} "
                f"({teacher_supervise_frames / 75:.2f}s) of each -> "
                f"{teacher_windows_per_crop * teacher_supervise_frames} frames/crop; "
                f"teacher future context {teacher_predict_frames}.."
                f"{teacher_predict_frames + teacher_supervise_frames - 1} frames, student block "
                f"{student_lookahead_block}"
            )
            if teacher_predict_frames:
                print(
                    f"  teacher also reads {teacher_predict_frames} frames "
                    f"({teacher_predict_frames / 75:.2f}s) past each block that the student "
                    f"cannot see -- the target is deliberately unreachable by that much"
                )
            if teacher_prefix_windows:
                last_prefix_end = teacher_window_frames - teacher_supervise_frames
                print(
                    f"  + 1 prefix block/step, end drawn log-uniformly from "
                    f"{teacher_supervise_frames}..{last_prefix_end} with the teacher truncated to "
                    f"[0, end+{teacher_predict_frames}) -- so the first {last_prefix_end} frames "
                    f"({last_prefix_end / 75:.2f}s) of a stream get gradient too. Training only; "
                    f"the val loss stays the {teacher_windows_per_crop} full intervals"
                )
        self.teacher_window_frames = teacher_window_frames
        self.teacher_supervise_frames = teacher_supervise_frames
        self.teacher_windows_per_crop = teacher_windows_per_crop
        self.teacher_predict_frames = teacher_predict_frames
        self.teacher_prefix_windows = teacher_prefix_windows
        self.cosine_loss_weight = cosine_loss_weight
        self.l2_loss_weight = l2_loss_weight

        # Create dataloaders
        train_dataloader = get_dataloader(split_path=train_split, shuffle=True)
        val_dataloader = get_dataloader(split_path=valid_split, shuffle=False)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        print("\nTraining Dataloader Length:")
        print(len(train_dataloader))
        print("\nValid Dataloader Length:")
        print(len(val_dataloader))

        # Trim crops to a multiple of hop_length: the causal conv stack needs an
        # input length divisible by every layer's stride.
        n_samples = train_dataloader.dataset.n_samples
        self.waveform_trim_samples = (n_samples // hop_length) * hop_length
        assert self.waveform_trim_samples > 0

        # Waveform normalization expected by mert_model's preprocessor.
        self.do_normalize = Wav2Vec2FeatureExtractor.from_pretrained(
            mert_model, trust_remote_code=True
        ).do_normalize

        # --- teacher: frozen bidirectional MERT (published or small checkpoint) ---
        if teacher_checkpoint:
            from modeling.load_checkpoint import load_small_bidirectional_from_align_checkpoint

            self.mert_teacher = load_small_bidirectional_from_align_checkpoint(
                teacher_checkpoint,
                base_config=mert_model,
                num_attention_heads=(
                    teacher_num_attention_heads
                    if teacher_num_attention_heads is not None
                    else student_num_attention_heads
                ),
            )
        else:
            from modeling.load_mert import load_mert
            self.mert_teacher = load_mert(mert_model)
        for p in self.mert_teacher.parameters():
            p.requires_grad_(False)
        self.mert_teacher.eval()

        # --- student: trainable streaming or small streaming MERT ----------
        if student_model == "streaming":
            from modeling.init_streaming import load_full_streaming_mert

            print(f"streaming student: positional_embedding={student_positional_embedding}, "
                  f"context={streaming_context} frames, "
                  f"lookahead_block={student_lookahead_block} frames")
            self.model = load_full_streaming_mert(
                mert_model, context=streaming_context,
                positional_embedding=student_positional_embedding,
                lookahead_block=student_lookahead_block,
            )
        else:  # "small_streaming"
            from modeling.init_streaming import build_small_streaming_mert
            from modeling.load_checkpoint import load_small_streaming_mert_from_align_checkpoint

            if teacher_checkpoint and student_init_from_teacher:
                # Warm start from the teacher's weights; student_* values are
                # checked against the checkpoint.
                self.model = load_small_streaming_mert_from_align_checkpoint(
                    teacher_checkpoint,
                    base_config=mert_model,
                    num_attention_heads=student_num_attention_heads,
                    context=streaming_context,
                    positional_embedding=student_positional_embedding,
                    lookahead_block=student_lookahead_block,
                    expected_arch={
                        "hidden_size": student_hidden_size,
                        "intermediate_size": student_intermediate_size,
                        "num_hidden_layers": student_num_hidden_layers,
                    },
                )
            else:
                print(f"small streaming student (random init): "
                      f"positional_embedding={student_positional_embedding}, "
                      f"context={streaming_context} frames, "
                      f"lookahead_block={student_lookahead_block} frames")
                self.model = build_small_streaming_mert(
                    mert_model,
                    hidden_size=student_hidden_size,
                    intermediate_size=student_intermediate_size,
                    num_attention_heads=student_num_attention_heads,
                    num_hidden_layers=(
                        student_num_hidden_layers
                        if student_num_hidden_layers is not None
                        else self.mert_teacher.config.num_hidden_layers
                    ),
                    context=streaming_context,
                    positional_embedding=student_positional_embedding,
                    lookahead_block=student_lookahead_block,
                )
        self.model.train()

        # distill_layer='all' pairs hidden states by index, so depths must match.
        if distill_layer == "all":
            assert self.model.config.num_hidden_layers == self.mert_teacher.config.num_hidden_layers, (
                f"distill_layer='all' needs matching depths, but the student has "
                f"{self.model.config.num_hidden_layers} layers and the teacher "
                f"{self.mert_teacher.config.num_hidden_layers}"
            )

        # Per-layer projections to the teacher's width (only when the widths
        # differ), trained jointly with the student.
        self.student_hidden_size = self.model.config.hidden_size
        teacher_hidden_size = self.mert_teacher.config.hidden_size
        if self.student_hidden_size != teacher_hidden_size:
            num_distill_layers = self.model.config.num_hidden_layers + 1
            self.distill_projections = nn.ModuleList([
                nn.Linear(self.student_hidden_size, teacher_hidden_size)
                for _ in range(num_distill_layers)
            ])
        else:
            self.distill_projections = None

        if compile:
            self.model = torch.compile(self.model)

        # reduction='none' in both cases -- compute_losses does its own
        # (optionally layer-weighted) averaging.
        self.cossim = nn.CosineEmbeddingLoss(reduction="none")
        self.l2 = nn.MSELoss(reduction="none")

        # Probes read the student's own (unprojected) hidden states.
        self.chord_evaluator = MERTChordEvaluator(emb_dim=self.student_hidden_size)
        self.beat_evaluator = MERTBeatEvaluator(emb_dim=self.student_hidden_size)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Store settings the weights cannot reveal: positional embedding,
        lookahead block, base MERT and teacher_predict_frames."""
        checkpoint["student_positional_embedding"] = self.student_positional_embedding
        checkpoint["student_lookahead_block"] = self.student_lookahead_block
        checkpoint["mert_model"] = self.mert_model_name
        checkpoint["teacher_predict_frames"] = self.teacher_predict_frames

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Refuse to resume with a different teacher_predict_frames, lookahead
        block or positional embedding than the checkpoint was trained with."""
        trained_r = checkpoint.get("teacher_predict_frames", 0)
        if trained_r != self.teacher_predict_frames:
            raise RuntimeError(
                f"this checkpoint was trained against a target with "
                f"teacher_predict_frames={trained_r}, but the current config says "
                f"{self.teacher_predict_frames}. Resuming would switch objectives mid-run; "
                f"set it back to {trained_r}, or start a fresh run (new save_dir)."
            )

        trained_block = checkpoint.get("student_lookahead_block", 1)
        if trained_block != self.student_lookahead_block:
            raise RuntimeError(
                f"this checkpoint was trained with student_lookahead_block={trained_block} "
                f"(each frame saw up to {trained_block - 1} frames ahead), but the current "
                f"config says {self.student_lookahead_block}. Resuming would keep weights "
                "trained under one attention mask and continue them under another; set the "
                f"config back to {trained_block} to continue this run, or start a fresh run "
                f"(new save_dir) to train at {self.student_lookahead_block}."
            )

        trained_with = checkpoint.get("student_positional_embedding", "sin")
        if trained_with != self.student_positional_embedding:
            raise RuntimeError(
                f"this checkpoint was trained with student_positional_embedding="
                f"{trained_with!r}, but the current config says "
                f"{self.student_positional_embedding!r}. Resuming would keep the weights while "
                "changing the function they are used in; set the config back to "
                f"{trained_with!r} to continue this run, or start a fresh run (new save_dir) "
                f"to train a {self.student_positional_embedding!r} student."
            )

    def _maybe_normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Zero-mean/unit-variance waveform normalization, if mert_model uses it."""
        if not self.do_normalize:
            return waveform
        mean = waveform.mean(dim=-1, keepdim=True)
        var = waveform.var(dim=-1, unbiased=False, keepdim=True)
        return (waveform - mean) / torch.sqrt(var + 1e-7)

    def get_teacher_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform (B, T_samples) -> teacher features after layer selection.
        Handles both HF output objects and this repo's tuple outputs."""
        self.mert_teacher.eval()
        waveform = self._maybe_normalize(waveform)
        with torch.no_grad():
            outputs = self.mert_teacher(waveform, output_hidden_states=True)
            hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs[1]
            hidden = torch.stack(hidden_states, dim=1)  # [B, L, T, D]
        return _select_distill_layer(hidden, self.distill_layer)

    def get_student_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform (B, T_samples) -> student features projected to the
        teacher's width, after layer selection."""
        return _select_distill_layer(
            self._project_student(self._student_hidden(waveform)), self.distill_layer
        )

    def _student_hidden(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform (B, T_samples) -> (B, L, T, D_student), unprojected."""
        waveform = self._maybe_normalize(waveform)
        _, hidden_states = self.model(waveform, output_hidden_states=True)
        return torch.stack(hidden_states, dim=1)

    def _project_student(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, L, T, D_student) -> (B, L, T, D_teacher); identity when the widths match."""
        if self.distill_projections is None:
            return hidden
        assert hidden.size(1) == len(self.distill_projections), \
            (hidden.size(1), len(self.distill_projections))
        return torch.stack(
            [proj(hidden[:, i]) for i, proj in enumerate(self.distill_projections)], dim=1
        )

    def get_inputs(self, batch):
        # Teacher and student see the same crop, trimmed to a multiple of
        # hop_length (see waveform_trim_samples).
        waveform = batch["waveform"][:, :self.waveform_trim_samples]

        if self.teacher_window_frames:
            return self._windowed_inputs(waveform)

        teacher_feat = self.get_teacher_features(waveform)   # [B, (L,) T_teacher, D]
        student_feat = self.get_student_features(waveform)   # [B, (L,) T_student, D]

        # The two conv stacks can give different frame counts; trim to the
        # common length (both start at sample 0).
        T_common = min(teacher_feat.shape[-2], student_feat.shape[-2])
        return student_feat[..., :T_common, :], teacher_feat[..., :T_common, :]

    def _windowed_inputs(self, waveform: torch.Tensor):
        """Local-interval teacher inputs (teacher_window_frames > 0): the teacher's
        encoder runs on K local intervals and the same frames are gathered from the
        student. Both come back as [B, (L,) K*W, D], plus one prefix block in
        training when teacher_prefix_windows is set."""
        from lit_module.distill_targets import (
            gather_frames_per_layer, pre_encoder_features, prefix_teacher_hidden,
            sample_prefix_end, sample_window_ends, windowed_teacher_hidden,
        )

        self.mert_teacher.eval()
        normalized = self._maybe_normalize(waveform)
        with torch.no_grad():
            pre = pre_encoder_features(self.mert_teacher, normalized)     # [B, T, D]

        student_hidden = self._student_hidden(waveform)                   # [B, L, T_s, D]
        # An interval may only end where BOTH streams have the frame.
        n_frames = min(pre.shape[1], student_hidden.shape[2])
        ends = sample_window_ends(
            pre.shape[0], self.teacher_windows_per_crop, self.teacher_window_frames,
            self.teacher_supervise_frames, n_frames, pre.device,
            predict_frames=self.teacher_predict_frames,
        )                                                                 # [B, K]

        with torch.no_grad():
            teacher_hidden = windowed_teacher_hidden(
                self.mert_teacher, pre, ends,
                self.teacher_window_frames, self.teacher_supervise_frames,
                predict_frames=self.teacher_predict_frames,
            )                                                             # [B, L, K*W, D]
        student_sel = gather_frames_per_layer(
            student_hidden, ends, self.teacher_supervise_frames,
        )                                                                 # [B, L, K*W, D]

        if self.teacher_prefix_windows and self.training:
            # One extra block from the start of the stream (training only).
            e0 = sample_prefix_end(
                self.teacher_window_frames, self.teacher_supervise_frames, self.global_step,
            )
            with torch.no_grad():
                teacher_prefix = prefix_teacher_hidden(
                    self.mert_teacher, pre, e0, self.teacher_supervise_frames,
                    predict_frames=self.teacher_predict_frames,
                )                                                         # [B, L, W, D]
            student_prefix = gather_frames_per_layer(
                student_hidden, torch.full_like(ends[:, :1], e0), self.teacher_supervise_frames,
            )                                                             # [B, L, W, D]
            teacher_hidden = torch.cat([teacher_hidden, teacher_prefix], dim=2)
            student_sel = torch.cat([student_sel, student_prefix], dim=2)

        student_sel = self._project_student(student_sel)

        return (
            _select_distill_layer(student_sel, self.distill_layer),
            _select_distill_layer(teacher_hidden, self.distill_layer),
        )

    def _layer_loss_weights(self, num_layers: int, device, dtype) -> torch.Tensor:
        """[L] layer weights for distill_layer == 'all': the final layer gets
        distill_final_weight, every other layer distill_intermediate_weight."""
        weights = torch.full((num_layers,), self.distill_intermediate_weight, device=device, dtype=dtype)
        weights[-1] = self.distill_final_weight
        return weights

    def compute_losses(self, batch):
        student_feat, teacher_feat = self.get_inputs(batch)
        pred = student_feat.reshape(-1, student_feat.size(-1))
        target = teacher_feat.reshape(-1, teacher_feat.size(-1))

        # Per-row losses so distill_layer == 'all' can weight rows by layer.
        cosine_per_row = self.cossim(pred, target, torch.ones(pred.size(0), device=pred.device))  # [N]
        l2_per_row = self.l2(pred, target).mean(dim=-1)  # [N]

        if self.distill_layer == "all":
            B, L, T, _ = student_feat.shape
            layer_weights = self._layer_loss_weights(L, pred.device, pred.dtype)
            row_weights = layer_weights.view(1, L, 1).expand(B, L, T).reshape(-1)
            denom = row_weights.sum()
            cosine_loss = (cosine_per_row * row_weights).sum() / denom
            l2_loss = (l2_per_row * row_weights).sum() / denom
        else:
            cosine_loss = cosine_per_row.mean()
            l2_loss = l2_per_row.mean()

        total_loss = (
            self.cosine_loss_weight * cosine_loss
            + self.l2_loss_weight * l2_loss
        )
        return total_loss, cosine_loss, l2_loss, student_feat.size(0)

    def training_step(self, batch, batch_idx):
        total_loss, cosine_loss, l2_loss, batch_size = self.compute_losses(batch)
        self._log_dict(
            {
                "train/cosine_loss": cosine_loss,
                "train/l2_loss": l2_loss,
                "train/total_loss": total_loss,
            },
            batch_size=batch_size,
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, cosine_loss, l2_loss, batch_size = self.compute_losses(batch)
        self._log_dict(
            {
                "val/cosine_loss": cosine_loss,
                "val/l2_loss": l2_loss,
                "val/total_loss": total_loss,
            },
            batch_size=batch_size,
        )
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Run the chord / beat probe evaluations on the student every
        eval_interval steps (rank 0 only)."""
        if not self.trainer.is_global_zero:
            return

        device = next(self.parameters()).device

        if (
            self.chord_evaluator.is_enabled()
            and self.global_step != 0
            and self.global_step % self.chord_evaluator.eval_interval == 0
        ):
            metrics = self.chord_evaluator.run(
                model=self.model,
                student_model_name=self.student_model_name,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                device=device,
                do_normalize=self.do_normalize,
            )
            for logger in self.trainer.loggers:
                logger.log_metrics({k: float(v) for k, v in metrics.items()}, step=self.global_step)

        if (
            self.beat_evaluator.is_enabled()
            and self.global_step != 0
            and self.global_step % self.beat_evaluator.eval_interval == 0
        ):
            beat_metrics = self.beat_evaluator.run(
                model=self.model,
                student_model_name=self.student_model_name,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                device=device,
                do_normalize=self.do_normalize,
            )
            for logger in self.trainer.loggers:
                logger.log_metrics({k: float(v) for k, v in beat_metrics.items()}, step=self.global_step)

    def configure_optimizers(self):
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()))
        scheduler = LinearWarmupCosineDecay(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
