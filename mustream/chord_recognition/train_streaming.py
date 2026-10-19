"""Train a chord probe on a frozen causal streaming MERT (KV-cache).

config['backbone']: 'streaming' (MERT-330M/95M remapped into the causal
architecture) or 'small_streaming' (26M causal student; needs
mert_align_checkpoint). The backbone is causal, so the whole crop goes through
in one forward pass. Only the probe is trained and saved.

Usage (from chord_recognition/):
  python train_streaming.py config/MERT_streaming.yaml <run_name>
"""

import os
import shutil
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from models import (
    build_classifier, extract_streaming_features, load_streaming_backbone,
)
from chord_augment import (
    apply_pitch_shift_batch, describe as describe_augment, resolve_pitch_shifts,
)
from chord_dataloader import MultiDatasetChordDataset
from chord_vocab import build_transpose_table, describe as describe_vocab
import pickle

config_path = sys.argv[1]
run_name = sys.argv[2]
config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

sample_rate = config["sample_rate"]
pool_factor = config["pool_factor"]
backbone_name = config.get("backbone", "streaming")
STREAMING_BACKBONES = ("streaming", "small_streaming")
assert backbone_name in STREAMING_BACKBONES, (
    f"invalid backbone {backbone_name!r}: only {STREAMING_BACKBONES} "
    "(modeling/streaming/streaming_mert.py) are supported here"
)


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------

def run_epoch(
    loader: DataLoader,
    model: nn.Module,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    target_raw_frames: int,
    train: bool,
    desc: str = "",
    pitch_shifts=None,        # training only: allowed semitone shifts
    transpose_table=None,     # [V, 12] chord_vocab table; required with pitch_shifts
) -> Tuple[float, float]:
    classifier.train(train)
    total_loss = 0.0
    total_correct = 0
    total_frames = 0

    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for audio, labels, _mask, n_steps in bar:
        labels = labels.to(device)

        if train and pitch_shifts:
            # Batched GPU pitch shift; transposes audio and labels together.
            audio, labels = apply_pitch_shift_batch(
                audio.to(device), labels, n_steps, transpose_table)

        embeddings = extract_streaming_features(
            model, audio,
            pool_factor=pool_factor,
            target_raw_frames=target_raw_frames, device=device,
        )  # [B, L, T_r, D]
        logits = classifier(embeddings)  # [B, T_r, V]
        T_reduced = logits.shape[1]
        frame_labels = labels[:, :T_reduced]

        B = logits.shape[0]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), frame_labels.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=-1)
        batch_correct = (preds == frame_labels).sum().item()
        batch_frames = B * T_reduced
        total_correct += batch_correct
        total_frames += batch_frames
        total_loss += loss.item() * B

        bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_correct / batch_frames:.4f}")

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_frames
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device(config["device"])
    save_dir = os.path.join("logs", run_name)
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(save_dir, "config.yaml"))
    torch.manual_seed(config.get("seed", 42))

    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            entity=config.get("wandb_entity"),
            project=config.get("wandb_project", "chord-recognition"),
            name=run_name,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    # Crop length is a multiple of hop_size (required by the causal conv stack).
    model_type = config.get("model_type", "LSTM")
    n_windows = round(config["crop_duration"] * (sample_rate / config["hop_size"] / pool_factor))
    target_raw_frames = n_windows * pool_factor
    mert_crop_samples = target_raw_frames * config["hop_size"]
    window_size = 1 / (sample_rate / config["hop_size"] / pool_factor)

    chord2idx, idx2chord = pickle.load(open(config["vocab_path"], 'rb'))
    vocab = chord2idx
    # Built up front so a vocabulary not closed under transposition fails early.
    pitch_shifts = resolve_pitch_shifts(config)
    transpose_table = build_transpose_table(idx2chord) if pitch_shifts else None
    print(describe_vocab(idx2chord))
    print(f"Backbone: {backbone_name}")
    print("Loading datasets (sr=24000 to match MERT)...")
    dataset_kwargs = dict(
        split_dir=config["split_dir"],
        vocab_file=config["vocab_path"],
        sr=sample_rate,
        window=window_size,
        crop_samples=mert_crop_samples,
        crop_duration=config["crop_duration"],
    )
    train_dataset = MultiDatasetChordDataset(
        split="train", pitch_shifts=pitch_shifts, **dataset_kwargs)
    val_dataset = MultiDatasetChordDataset(split="val", **dataset_kwargs)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True,
    )

    mert_align_checkpoint = config.get("mert_align_checkpoint")
    print(f"Loading {backbone_name} backbone ({config['MERT_model']})...")
    if mert_align_checkpoint is not None:
        print(f"  (overriding with fine-tuned weights from {mert_align_checkpoint})")
    model = load_streaming_backbone(
        config["MERT_model"], device, config.get("streaming_context", 1000),
        mert_align_checkpoint=mert_align_checkpoint,
        positional_embedding=config.get("backbone_positional_embedding", "rope"),
        backbone=backbone_name,
        num_attention_heads=config.get("student_num_attention_heads", 8),
        expect_emb_dim=config["emb_dim"],
    )

    classifier = build_classifier(model_type, len(vocab), config).to(device)

    lr = float(config["lr"])
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in classifier.parameters())
    print(f"\nClassifier params: {n_params:,}  (lr={lr:g})")
    print("Backbone:          frozen (probe-only training)")
    print(describe_augment(pitch_shifts))
    print(f"Seed: {config.get('seed', 42)}")
    print(f"Device: {device}  |  train={len(train_dataset)}  val={len(val_dataset)}\n")

    best_val_acc = -1.0
    epoch_bar = tqdm(range(1, config["epochs"] + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        tr_loss, tr_acc = run_epoch(
            train_loader, model, classifier, optimizer, criterion, device,
            target_raw_frames, train=True, desc=f"Epoch {epoch}/{config['epochs']} [train]",
            pitch_shifts=pitch_shifts, transpose_table=transpose_table,
        )
        val_loss, val_acc = run_epoch(
            val_loader, model, classifier, optimizer, criterion, device,
            target_raw_frames, train=False, desc=f"Epoch {epoch}/{config['epochs']} [val]  ",
        )

        epoch_bar.set_postfix(
            tr_loss=f"{tr_loss:.4f}", tr_acc=f"{tr_acc:.4f}",
            val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}",
        )
        tqdm.write(
            f"Epoch {epoch:>3}/{config['epochs']}  "
            f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if use_wandb:
            wandb.log({"epoch": epoch, "train/acc": tr_acc, "train/loss": tr_loss, "val/acc": val_acc, "val/loss": val_loss})

        ckpt = {"epoch": epoch, "model_state": classifier.state_dict(), "config": config}
        # best.pt is selected on val frame accuracy over crops, not the mir_eval metric.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))
        if not epoch % config["ckpt_interval"]:
            torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch:03d}_{val_acc:.3f}.pt"))

    print(f"\nBest val frame accuracy: {best_val_acc:.4f}  ->  {save_dir}/best.pt")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
