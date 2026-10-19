"""
Periodic chord / beat probe evaluation during MERT distillation
(lit_module/mert_size_distill.py, lit_module/mert_causal_distill.py).

Every `eval_interval` training steps, a fresh probe is trained on the frozen
student's hidden states with the same probe, loss and metrics as
chord_recognition/train_offline.py and beat_tracking/train_offline.py, and its
validation scores are logged. Both evaluators are off unless `enable: true` and
their data paths are set (see config/mert_*_distill.yml).
"""

import os
import pickle
import sys
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# beat_tracking/ modules use bare imports (e.g. `from data_utils import ...`).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "beat_tracking"))

from chord_recognition.models import ChordClassifier
from chord_recognition.chord_dataloader import MultiDatasetChordDataset

from beat_tracking.models import LinearProbe, compute_loss, frame_stats, precision_recall_f1
from beat_tracking.beat_dataloader import BeatDataset


def extract_student_features(
    model: nn.Module,
    student_model_name: str,
    audio: torch.Tensor,      # [B, 1, T_samples]
    pool_factor: int = 1,
    do_normalize: bool = False,
) -> torch.Tensor:
    """Frozen student forward -> [B, L, T // pool_factor, D] hidden states (mean-pooled in time).

    student_model_name is unused: every student returns (last_hidden_state, all_hidden_states).
    do_normalize applies the zero-mean / unit-variance waveform normalization of mert_model.
    """
    B = audio.shape[0]
    waveform = audio[:, 0, :]  # [B, T_samples]
    if do_normalize:
        mean = waveform.mean(dim=-1, keepdim=True)
        var = waveform.var(dim=-1, unbiased=False, keepdim=True)
        waveform = (waveform - mean) / torch.sqrt(var + 1e-7)
    with torch.no_grad():
        _, hidden_states = model(waveform, output_hidden_states=True)
        hidden = torch.stack(hidden_states, dim=1)   # [B, L, T, D]

    _, n_layers, T, D = hidden.shape
    if pool_factor > 1:
        T_trim = (T // pool_factor) * pool_factor
        hidden = hidden[:, :, :T_trim, :]
        hidden = hidden.reshape(B, n_layers, T_trim // pool_factor, pool_factor, D).mean(3)
    return hidden


def _crop_samples_for_student(crop_duration: float, sample_rate: int, hop_length: int, pool_factor: int) -> int:
    """Crop length in samples: a whole number of pooled frames (a multiple of hop_length,
    as the streaming students' causal convs require)."""
    n_windows = round(crop_duration * (sample_rate / hop_length / pool_factor))
    return n_windows * pool_factor * hop_length


class MERTChordEvaluator:
    """Chord probe (ChordClassifier + cross-entropy) on the student's features.

    Config keys (MERTChordEvaluator.*): enable, eval_interval (steps), split_dir
    ({train,val}.pkl as in chord_recognition/), vocab_file, crop_duration (s),
    pool_factor, emb_dim (set to the student's hidden size by the Lightning module),
    mert_layer, hidden_dim, probe_epochs, probe_lr, batch_size, num_workers.
    """

    def __init__(
        self,
        enable: bool = False,
        eval_interval: int = 2000,
        split_dir: str = 'chord_recognition/data_splits/splits',
        vocab_file: str = 'chord_recognition/data_splits/dictionary_121.pkl',
        crop_duration: float = 10.0,
        pool_factor: int = 5,
        emb_dim: int = 1024,
        mert_layer: Union[int, str] = 'average',
        hidden_dim: Optional[int] = 512,
        probe_epochs: int = 10,
        probe_lr: float = 1e-3,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        self.enable = enable
        self.eval_interval = eval_interval
        self.split_dir = split_dir
        self.vocab_file = vocab_file
        self.crop_duration = crop_duration
        self.pool_factor = pool_factor
        self.emb_dim = emb_dim
        self.mert_layer = mert_layer
        self.hidden_dim = hidden_dim
        self.probe_epochs = probe_epochs
        self.probe_lr = probe_lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataloaders = None
        self.vocab = None

    def is_enabled(self) -> bool:
        return self.enable and bool(self.split_dir and self.vocab_file)

    def ensure_dataloaders(self, sample_rate: int, hop_length: int):
        """Build the train/val loaders once, on first use."""
        if self.dataloaders is not None:
            return
        window_size = 1.0 / (sample_rate / hop_length / self.pool_factor)
        crop_samples = _crop_samples_for_student(self.crop_duration, sample_rate, hop_length, self.pool_factor)

        with open(self.vocab_file, "rb") as f:
            self.vocab = pickle.load(f)[0]
        print(f"[MERTChordEval] vocabulary size: {len(self.vocab)}")

        shared = dict(
            split_dir=self.split_dir, vocab_file=self.vocab_file,
            sr=sample_rate, window=window_size,
            crop_samples=crop_samples, crop_duration=self.crop_duration,
        )
        train_ds = MultiDatasetChordDataset(split="train", **shared)
        val_ds = MultiDatasetChordDataset(split="val", **shared)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
        self.dataloaders = (train_loader, val_loader)

    def run_probe_epoch(
        self,
        loader: DataLoader,
        model: nn.Module,
        student_model_name: str,
        classifier: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        train: bool,
        do_normalize: bool = False,
    ) -> tuple:
        """One probe epoch -> (mean loss per sample, frame accuracy)."""
        classifier.train(train)
        total_loss = 0.0
        total_correct = 0
        total_frames = 0
        n_samples = 0

        # The 4th item is the pitch shift (always 0 here: no augmentation).
        for audio, labels, _mask, _n_steps in tqdm(loader, desc="[MERTChordEval]", leave=False, unit="batch"):
            audio = audio.to(device)
            labels = labels.to(device)

            embeddings = extract_student_features(
                model, student_model_name, audio, pool_factor=self.pool_factor,
                do_normalize=do_normalize,
            )  # [B, L, T_r, D]
            T_reduced = embeddings.shape[2]
            frame_labels = labels[:, :T_reduced]
            logits = classifier(embeddings)  # [B, T_r, V]

            B = logits.shape[0]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), frame_labels.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=-1)
            total_correct += (preds == frame_labels).sum().item()
            total_frames += B * T_reduced
            total_loss += loss.item() * B
            n_samples += B

        return total_loss / max(n_samples, 1), total_correct / max(total_frames, 1)

    def run(
        self,
        model: nn.Module,
        student_model_name: str,
        sample_rate: int,
        hop_length: int,
        device: torch.device,
        do_normalize: bool = False,
    ) -> dict:
        """Train a fresh probe for probe_epochs on the student's features; return train/val metrics."""
        self.ensure_dataloaders(sample_rate, hop_length)
        train_loader, val_loader = self.dataloaders

        classifier = ChordClassifier(
            vocab_size=len(self.vocab),
            hidden_dim=self.hidden_dim,
            emb_dim=self.emb_dim,
            mert_layer=self.mert_layer,
        ).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.probe_lr)
        criterion = nn.CrossEntropyLoss()

        was_training = model.training
        model.eval()
        print(f"[MERTChordEval] training probe ({self.probe_epochs} epochs) …")
        tr_loss = tr_acc = float("nan")
        for _ in range(self.probe_epochs):
            tr_loss, tr_acc = self.run_probe_epoch(
                train_loader, model, student_model_name, classifier, optimizer, criterion, device, train=True,
                do_normalize=do_normalize,
            )
        val_loss, val_acc = self.run_probe_epoch(
            val_loader, model, student_model_name, classifier, optimizer, criterion, device, train=False,
            do_normalize=do_normalize,
        )
        if was_training:
            model.train()

        print(f"[MERTChordEval] probe_train_acc={tr_acc:.4f}  probe_val_acc={val_acc:.4f}")
        return {
            "chord_eval/probe_train_loss": tr_loss,
            "chord_eval/probe_train_acc": tr_acc,
            "chord_eval/probe_val_loss": val_loss,
            "chord_eval/probe_val_acc": val_acc,
        }


class MERTBeatEvaluator:
    """Beat/downbeat probe (LinearProbe + pos-weighted BCE) on the student's features.

    Config keys (MERTBeatEvaluator.*): enable, eval_interval (steps), audio_dir,
    annotation_dir, split_dir (layout as in beat_tracking/), train_ratio / val_ratio /
    split_seed (only to auto-build a missing split), crop_duration (s), target_widen,
    pool_factor, emb_dim (set by the Lightning module), mert_layer, hidden_dim,
    pos_weight_beat, pos_weight_downbeat, probe_epochs, probe_lr, batch_size,
    num_workers, audio_ext.
    """

    def __init__(
        self,
        enable: bool = False,
        eval_interval: int = 2000,
        audio_dir: str = '',
        annotation_dir: str = '',
        split_dir: str = 'beat_tracking/data_splits/splits',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        split_seed: int = 42,
        crop_duration: float = 10.0,
        target_widen: int = 1,
        pool_factor: int = 1,
        emb_dim: int = 1024,
        mert_layer: Union[int, str] = 'average',
        hidden_dim: Optional[int] = 512,
        pos_weight_beat: float = 1.0,
        pos_weight_downbeat: float = 3.0,
        probe_epochs: int = 10,
        probe_lr: float = 1e-3,
        batch_size: int = 8,
        num_workers: int = 4,
        audio_ext: str = "wav",
    ):
        self.enable = enable
        self.eval_interval = eval_interval
        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.split_dir = split_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        self.crop_duration = crop_duration
        self.target_widen = target_widen
        self.pool_factor = pool_factor
        self.emb_dim = emb_dim
        self.mert_layer = mert_layer
        self.hidden_dim = hidden_dim
        self.pos_weights = {"pos_weight_beat": pos_weight_beat, "pos_weight_downbeat": pos_weight_downbeat}
        self.probe_epochs = probe_epochs
        self.probe_lr = probe_lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.audio_ext = audio_ext
        self.dataloaders = None

    def is_enabled(self) -> bool:
        return self.enable and bool(self.audio_dir and self.annotation_dir)

    def ensure_dataloaders(self, sample_rate: int, hop_length: int):
        """Build the train/val loaders once, on first use."""
        if self.dataloaders is not None:
            return
        fps = sample_rate / hop_length / self.pool_factor
        crop_samples = _crop_samples_for_student(self.crop_duration, sample_rate, hop_length, self.pool_factor)

        shared = dict(
            audio_dir=self.audio_dir, annotation_dir=self.annotation_dir, split_dir=self.split_dir,
            sr=sample_rate, fps=fps, pool_factor=self.pool_factor, crop_samples=crop_samples,
            train_ratio=self.train_ratio, val_ratio=self.val_ratio, split_seed=self.split_seed,
            target_widen=self.target_widen, audio_ext=self.audio_ext,
        )
        train_ds = BeatDataset(split="train", mode="train", **shared)
        val_ds = BeatDataset(split="val", mode="train", **shared)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
        self.dataloaders = (train_loader, val_loader)

    def run_probe_epoch(
        self,
        loader: DataLoader,
        model: nn.Module,
        student_model_name: str,
        probe: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        train: bool,
        do_normalize: bool = False,
    ) -> dict:
        """One probe epoch -> loss plus frame-level beat/downbeat accuracy, precision, recall and F1."""
        probe.train(train)
        total_loss = 0.0
        n_batches = 0
        beat_correct = beat_valid = 0.0
        beat_tp = beat_fp = beat_fn = 0.0
        downbeat_correct = downbeat_valid = 0.0
        downbeat_tp = downbeat_fp = downbeat_fn = 0.0

        for audio, beat_seq, downbeat_seq, mask, downbeat_available in tqdm(loader, desc="[MERTBeatEval]", leave=False, unit="batch"):
            audio = audio.to(device)
            beat_seq = beat_seq.to(device)
            downbeat_seq = downbeat_seq.to(device)
            mask = mask.to(device)
            downbeat_available = downbeat_available.to(device)

            embeddings = extract_student_features(
                model, student_model_name, audio, pool_factor=self.pool_factor,
                do_normalize=do_normalize,
            )  # [B, L, T, D]
            logits = probe(embeddings)  # [B, T, 2]

            T = min(logits.shape[1], beat_seq.shape[1])
            logits_use = logits[:, :T]
            beat_seq_use = beat_seq[:, :T]
            downbeat_seq_use = downbeat_seq[:, :T]
            mask_use = mask[:, :T]

            loss, _, _ = compute_loss(
                logits_use, beat_seq_use, downbeat_seq_use, mask_use, downbeat_available, self.pos_weights,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            (b_correct, b_valid, (b_tp, b_fp, b_fn)), (d_correct, d_valid, (d_tp, d_fp, d_fn)) = frame_stats(
                logits_use, beat_seq_use, downbeat_seq_use, mask_use, downbeat_available,
            )
            beat_correct += b_correct; beat_valid += b_valid
            beat_tp += b_tp; beat_fp += b_fp; beat_fn += b_fn
            downbeat_correct += d_correct; downbeat_valid += d_valid
            downbeat_tp += d_tp; downbeat_fp += d_fp; downbeat_fn += d_fn

            total_loss += loss.item()
            n_batches += 1

        beat_precision, beat_recall, beat_f1 = precision_recall_f1(beat_tp, beat_fp, beat_fn)
        downbeat_precision, downbeat_recall, downbeat_f1 = precision_recall_f1(downbeat_tp, downbeat_fp, downbeat_fn)
        return {
            "loss": total_loss / max(n_batches, 1),
            "beat_acc": beat_correct / max(beat_valid, 1),
            "downbeat_acc": downbeat_correct / max(downbeat_valid, 1),
            "beat_precision": beat_precision,
            "beat_recall": beat_recall,
            "beat_f1": beat_f1,
            "downbeat_precision": downbeat_precision,
            "downbeat_recall": downbeat_recall,
            "downbeat_f1": downbeat_f1,
        }

    def run(
        self,
        model: nn.Module,
        student_model_name: str,
        sample_rate: int,
        hop_length: int,
        device: torch.device,
        do_normalize: bool = False,
    ) -> dict:
        """Train a fresh probe for probe_epochs on the student's features; return train/val metrics."""
        self.ensure_dataloaders(sample_rate, hop_length)
        train_loader, val_loader = self.dataloaders

        probe = LinearProbe(
            emb_dim=self.emb_dim, hidden_dim=self.hidden_dim, mert_layer=self.mert_layer,
        ).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.probe_lr)

        was_training = model.training
        model.eval()
        print(f"[MERTBeatEval] training probe ({self.probe_epochs} epochs) …")
        tr_stats = {"loss": float("nan")}
        for _ in range(self.probe_epochs):
            tr_stats = self.run_probe_epoch(
                train_loader, model, student_model_name, probe, optimizer, device, train=True,
                do_normalize=do_normalize,
            )
        val_stats = self.run_probe_epoch(
            val_loader, model, student_model_name, probe, optimizer, device, train=False,
            do_normalize=do_normalize,
        )
        if was_training:
            model.train()

        print(
            f"[MERTBeatEval] val_beat_acc={val_stats['beat_acc']:.4f}  "
            f"val_downbeat_acc={val_stats['downbeat_acc']:.4f}  "
            f"val_beat_f1={val_stats['beat_f1']:.4f}  val_downbeat_f1={val_stats['downbeat_f1']:.4f}"
        )
        return {
            "beat_eval/probe_train_loss": tr_stats["loss"],
            "beat_eval/probe_val_loss": val_stats["loss"],
            "beat_eval/probe_val_beat_acc": val_stats["beat_acc"],
            "beat_eval/probe_val_downbeat_acc": val_stats["downbeat_acc"],
            "beat_eval/beat_f_measure": val_stats["beat_f1"],
            "beat_eval/downbeat_f_measure": val_stats["downbeat_f1"],
        }
