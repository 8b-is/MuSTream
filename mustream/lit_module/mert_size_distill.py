"""
Lightning module for size distillation: trains a small bidirectional MERT
student (modeling/mert/small_mert.py) to match a frozen full-size bidirectional
MERT teacher with a cosine + L2 loss on per-layer projected hidden states.

The projections (self.distill_projections) only serve the loss and are not part
of the student. Streaming students are trained by lit_module/mert_causal_distill.py.
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

# Student name passed to the probe evaluators.
STUDENT_MODEL = "small_bidirectional"

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
class LitMERTSizeDistill(BaseLightningModel):
    def __init__(
        self,
        compile: bool = True,
        train_split: str = "",
        valid_split: str = "",
        sample_rate: int = 24000,
        hop_length: int = 320,
        mert_model: str = "m-a-p/MERT-v1-95M",
        teacher_checkpoint: str = "",       # distill from an earlier small student instead of mert_model
        teacher_num_attention_heads: Optional[int] = None,  # only used with teacher_checkpoint; defaults to student_num_attention_heads
        student_hidden_size: int = 432,
        student_intermediate_size: int = 1392,
        student_num_attention_heads: int = 8,
        student_num_hidden_layers: Optional[int] = None,  # defaults to the teacher's own depth
        distill_layer: Union[int, str] = "all",
        distill_intermediate_weight: float = 1.0,
        distill_final_weight: float = 1.0,
        cosine_loss_weight: float = 1.0,
        l2_loss_weight: float = 0.1,
    ):
        """
        Trains a small bidirectional MERT student against a frozen bidirectional teacher.

          mert_model:          published MERT; the default teacher, and the source
                               of the shared config and waveform normalization.
          teacher_checkpoint:  use an earlier small student checkpoint as the
                               teacher; teacher_num_attention_heads defaults to
                               student_num_attention_heads.
          student_hidden_size / student_intermediate_size /
          student_num_attention_heads / student_num_hidden_layers:
                               student architecture (STaRHuBERT-L defaults; depth
                               defaults to the teacher's).
          distill_layer:       'average' | 'final' | 'all' | int layer index.
          distill_intermediate_weight / distill_final_weight:
                               for 'all': weight of layers 0..-2 vs. the final layer.
          cosine_loss_weight / l2_loss_weight: weights of the two loss terms.
        """
        super(LitMERTSizeDistill, self).__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.distill_layer = distill_layer
        self.distill_intermediate_weight = distill_intermediate_weight
        self.distill_final_weight = distill_final_weight
        self.student_model_name = STUDENT_MODEL
        self.mert_model_name = mert_model
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

        # Trim crops to a whole number of frames (a multiple of hop_length).
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

        # --- student: small bidirectional MERT, random init ---
        from modeling.mert.small_mert import SmallBidirectionalMERTModel, build_small_mert_config

        small_config = build_small_mert_config(
            self.mert_teacher.config,
            hidden_size=student_hidden_size,
            intermediate_size=student_intermediate_size,
            num_attention_heads=student_num_attention_heads,
            num_hidden_layers=student_num_hidden_layers,
        )
        self.model = SmallBidirectionalMERTModel(small_config)
        self.model.train()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"small bidirectional student: {n_params / 1e6:.1f}M params, "
            f"hidden_size={small_config.hidden_size}, "
            f"intermediate_size={small_config.intermediate_size}, "
            f"num_hidden_layers={small_config.num_hidden_layers}, "
            f"num_attention_heads={small_config.num_attention_heads}"
        )

        # distill_layer='all' pairs hidden states by index, so depths must match.
        if distill_layer == "all":
            assert self.model.config.num_hidden_layers == self.mert_teacher.config.num_hidden_layers, (
                "distill_layer='all' pairs hidden states layer by layer, so it needs matching depths, but "
                f"the student has {self.model.config.num_hidden_layers} layers and the teacher "
                f"{self.mert_teacher.config.num_hidden_layers} -- leave student_num_hidden_layers "
                "unset to inherit the teacher's own depth"
            )

        if not any((cosine_loss_weight, l2_loss_weight)):
            raise ValueError(
                "both cosine_loss_weight and l2_loss_weight are 0.0, "
                "so nothing would drive the student. Set at least one."
            )

        # Per-layer projections to the teacher's width, trained jointly with the
        # student and discarded afterwards.
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
        """Store the base MERT, which sets pre- vs post-norm when the student
        is rebuilt (not recorded in the weights)."""
        checkpoint["mert_model"] = self.mert_model_name

    def _maybe_normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Zero-mean/unit-variance waveform normalization, if mert_model uses it."""
        if not self.do_normalize:
            return waveform
        mean = waveform.mean(dim=-1, keepdim=True)
        var = waveform.var(dim=-1, unbiased=False, keepdim=True)
        return (waveform - mean) / torch.sqrt(var + 1e-7)

    def get_teacher_hidden(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform (B, T_samples) -> (B, L, T, D_teacher), frozen. Handles both
        HF output objects and this repo's tuple outputs."""
        self.mert_teacher.eval()
        waveform = self._maybe_normalize(waveform)
        with torch.no_grad():
            outputs = self.mert_teacher(waveform, output_hidden_states=True)
            hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs[1]
            return torch.stack(hidden_states, dim=1)  # [B, L, T, D]

    def get_student_hidden(self, waveform: torch.Tensor) -> torch.Tensor:
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
        """batch -> (student_hidden, teacher_hidden), both [B, L, T, D] over the
        same frames -- unprojected; _feature_losses projects the student."""
        # Teacher and student see the same crop, trimmed to a multiple of
        # hop_length (see waveform_trim_samples).
        waveform = batch["waveform"][:, :self.waveform_trim_samples]

        teacher_hidden = self.get_teacher_hidden(waveform)   # [B, L, T_teacher, D_t]
        student_hidden = self.get_student_hidden(waveform)   # [B, L, T_student, D_s]

        # The two conv stacks can differ by a frame; trim to the common length.
        T_common = min(teacher_hidden.shape[2], student_hidden.shape[2])
        return student_hidden[:, :, :T_common, :], teacher_hidden[:, :, :T_common, :]

    def _layer_loss_weights(self, num_layers: int, device, dtype) -> torch.Tensor:
        """[L] layer weights for distill_layer == 'all': the final layer gets
        distill_final_weight, every other layer distill_intermediate_weight."""
        weights = torch.full((num_layers,), self.distill_intermediate_weight, device=device, dtype=dtype)
        weights[-1] = self.distill_final_weight
        return weights

    def _feature_losses(self, student_hidden, teacher_hidden):
        """Cosine + L2 between the projected student and the teacher over the
        selected layer(s). Returns (weighted_total, {name: scalar})."""
        student_feat = _select_distill_layer(self._project_student(student_hidden), self.distill_layer)
        teacher_feat = _select_distill_layer(teacher_hidden, self.distill_layer)

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

        total = (
            self.cosine_loss_weight * cosine_loss
            + self.l2_loss_weight * l2_loss
        )
        return total, {
            "cosine_loss": cosine_loss.detach(),
            "l2_loss": l2_loss.detach(),
        }

    def compute_losses(self, batch):
        """-> (total_loss, {name: scalar for logging}, batch_size).

        total_loss is the training objective, and is what
        {train,val}/total_loss and hence checkpoint_metric watch.
        """
        student_hidden, teacher_hidden = self.get_inputs(batch)
        total_loss, logs = self._feature_losses(student_hidden, teacher_hidden)
        logs["total_loss"] = total_loss
        return total_loss, logs, student_hidden.size(0)

    def training_step(self, batch, batch_idx):
        total_loss, logs, batch_size = self.compute_losses(batch)
        self._log_dict({f"train/{k}": v for k, v in logs.items()}, batch_size=batch_size)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, logs, batch_size = self.compute_losses(batch)
        self._log_dict({f"val/{k}": v for k, v in logs.items()}, batch_size=batch_size)
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
