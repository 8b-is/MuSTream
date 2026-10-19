"""
Causal streaming key-detection training on a frozen streaming MERT
(modeling/streaming/streaming_mert.py). The backbone is causal by construction,
so the whole crop goes through in one forward pass. Only the head is trained.

config['backbone']: 'streaming' (MERT-330M/95M, set MERT_model/emb_dim) or
'small_streaming' (26M distilled student; mert_align_checkpoint required).

Usage (from key_detection/):
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (
    build_classifier, extract_streaming_features, load_streaming_backbone,
    readout_loss, resolve_train_readout,
)
from key_dataloader import build_datasets, streaming_geometry
from key_augment import apply_pitch_shift_batch, describe as describe_augment, resolve_pitch_shifts
from key_utils import N_KEY_CLASSES, weighted_score_sum


# ---------------------------------------------------------------------------
# Config / geometry
# ---------------------------------------------------------------------------
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

# Shared with infer_streaming.py.
geom = streaming_geometry(config)
target_raw_frames = geom.target_raw_frames


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
    train: bool,
    train_readout: str,
    desc: str = "",
    pitch_shifts=None,
) -> Tuple[float, float, float]:
    """Run one train/val epoch; returns (avg_loss, accuracy, weighted_score)."""
    classifier.train(train)
    total_loss = 0.0
    total_correct = 0
    total_weighted = 0.0
    total_items = 0

    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for audio, labels, n_steps in bar:
        labels = labels.to(device)
        B = audio.shape[0]

        if train and pitch_shifts:
            # Batched GPU op, so move audio to the device first.
            audio, labels = apply_pitch_shift_batch(audio.to(device), labels, n_steps)

        embeddings = extract_streaming_features(
            model, audio, pool_factor=pool_factor,
            target_raw_frames=target_raw_frames, device=device,
        )  # [B, L, n_frames, D]

        with torch.set_grad_enabled(train):
            loss, logits = readout_loss(classifier, embeddings, labels, criterion, train_readout)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_weighted += weighted_score_sum(labels.cpu().numpy(), preds.cpu().numpy())
        total_loss += loss.item() * B
        total_items += B

        bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{total_correct / total_items:.4f}")

    return total_loss / total_items, total_correct / total_items, total_weighted / total_items


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
            project=config.get("wandb_project", "key-detection"),
            name=run_name,
            config=config,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    print(f"Backbone: {backbone_name}")
    print(geom.describe())
    print("Loading datasets (sr={} to match MERT)...".format(sample_rate))
    train_dataset, val_dataset = build_datasets(config, geom)
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
        print(f"  (overriding with distilled weights from {mert_align_checkpoint})")
    model = load_streaming_backbone(
        config["MERT_model"], device, config.get("streaming_context", 1000),
        mert_align_checkpoint=mert_align_checkpoint,
        positional_embedding=config.get("backbone_positional_embedding", "rope"),
        backbone=backbone_name,
        num_attention_heads=config.get("student_num_attention_heads", 8),
        expect_emb_dim=config["emb_dim"],
    )

    # Block-causal backbone (the MuSTream backbones): warn if the crop is not a whole
    # number of blocks.
    lookahead_block = int(getattr(model, "lookahead_block", 1))
    if lookahead_block > 1:
        block_seconds = lookahead_block * config["hop_size"] / sample_rate
        print(f"Backbone is BLOCK-CAUSAL: lookahead_block={lookahead_block} raw frames "
              f"({block_seconds:.3f}s of in-block future at most)")
        if target_raw_frames % lookahead_block:
            print(f"  WARNING: the {target_raw_frames}-raw-frame crop is not a whole number of "
                  f"{lookahead_block}-frame blocks; its last partial block sees the crop end as "
                  f"its future. Pick a crop_duration that is a multiple of {block_seconds:.4g}s "
                  "to avoid it.")

    model_type = config.get("model_type", "LSTM")
    train_readout = resolve_train_readout(config)
    classifier = build_classifier(model_type, N_KEY_CLASSES, config).to(device)

    lr = float(config["lr"])
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\nHead ({model_type}) params : {sum(p.numel() for p in classifier.parameters()):,}  (lr={lr:g})")
    print("Backbone                 : frozen (probe-only training)")
    pitch_shifts = resolve_pitch_shifts(config)
    print(
        f"Readout                  : {train_readout}\n"
        f"{describe_augment(pitch_shifts)}\n"
        f"Device                   : {device}\n"
        f"Train / val crops        : {len(train_dataset)} / {len(val_dataset)}\n"
    )

    best_val_weighted = -1.0
    epochs = config["epochs"]
    epoch_bar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        tr_loss, tr_acc, tr_w = run_epoch(
            train_loader, model, classifier, optimizer, criterion, device,
            train=True, train_readout=train_readout, desc=f"Epoch {epoch}/{epochs} [train]",
            pitch_shifts=pitch_shifts)
        val_loss, val_acc, val_w = run_epoch(
            val_loader, model, classifier, optimizer, criterion, device,
            train=False, train_readout=train_readout, desc=f"Epoch {epoch}/{epochs} [val]  ")

        epoch_bar.set_postfix(tr_acc=f"{tr_acc:.4f}", val_acc=f"{val_acc:.4f}", val_w=f"{val_w:.4f}")
        tqdm.write(
            f"Epoch {epoch:>3}/{epochs}  tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} tr_w={tr_w:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_w={val_w:.4f}"
        )
        if use_wandb:
            wandb.log({"epoch": epoch,
                       "train/loss": tr_loss, "train/acc": tr_acc, "train/weighted_score": tr_w,
                       "val/loss": val_loss, "val/acc": val_acc, "val/weighted_score": val_w})

        # best.pt by val weighted score, plus a periodic snapshot every ckpt_interval epochs.
        ckpt = {"epoch": epoch, "model_state": classifier.state_dict(),
                "val_weighted": val_w, "config": config}
        if val_w > best_val_weighted:
            best_val_weighted = val_w
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))
        if config.get("ckpt_interval") and not epoch % config["ckpt_interval"]:
            torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch:03d}_{val_w:.3f}.pt"))

    print(f"\nBest val weighted score: {best_val_weighted:.4f}  ->  {save_dir}/best.pt")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
