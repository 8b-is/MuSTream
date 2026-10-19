"""
Sliding-window key-detection training: a frozen bidirectional MERT is made causal
by re-encoding the last `context_duration` seconds at every `hop_duration` hop and
keeping only the newest hop's frames. Only the head is trained.

Usage (from key_detection/):
  python train_sliding.py config/MERT_sliding.yaml <run_name>
"""

import os
import shutil
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (
    build_classifier, extract_MERT_features, load_backbone_and_processor,
    readout_loss, resolve_train_readout,
)
from key_dataloader import build_datasets, sliding_geometry
from key_augment import apply_pitch_shift_batch, describe as describe_augment, resolve_pitch_shifts
from key_utils import N_KEY_CLASSES, weighted_score_sum


# ---------------------------------------------------------------------------
# Config / geometry
# ---------------------------------------------------------------------------
config_path = sys.argv[1]
config = yaml.load(open(config_path), Loader=yaml.FullLoader)
run_name = sys.argv[2]

epochs = config["epochs"]
sample_rate = config["sample_rate"]
pool_factor = config["pool_factor"]
mert_batch_size = config.get("mert_batch_size")
backbone = config.get("backbone", "pretrained")

# Shared with infer.SlidingKeyExtractor.
geom = sliding_geometry(config)
crop_samples = geom.crop_samples
context_samples = geom.context_samples
hop_samples = geom.hop_samples
n_frames_per_hop = geom.n_frames_per_hop
pad_left = geom.pad_left
n_hops = geom.n_hops
print(geom.describe())


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------

def run_mert_windows(
    mert: nn.Module,
    processor: Wav2Vec2FeatureExtractor,
    windows_flat: torch.Tensor,   # [N, 1, context_samples], already on device
) -> torch.Tensor:
    """extract_MERT_features over a flat window batch, in sub-batches of mert_batch_size."""
    if mert_batch_size is None or windows_flat.shape[0] <= mert_batch_size:
        return extract_MERT_features(mert, processor, windows_flat, pool_factor=pool_factor)
    parts = [
        extract_MERT_features(mert, processor, windows_flat[i : i + mert_batch_size], pool_factor=pool_factor)
        for i in range(0, windows_flat.shape[0], mert_batch_size)
    ]
    return torch.cat(parts, dim=0)


def run_epoch(
    loader: DataLoader,
    mert: nn.Module,
    processor: Wav2Vec2FeatureExtractor,
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
        audio = audio.to(device)
        labels = labels.to(device)
        B = audio.shape[0]

        # ---- 0. pitch-shift augmentation (audio and label together) --------
        if train and pitch_shifts:
            audio, labels = apply_pitch_shift_batch(audio, labels, n_steps)

        # ---- 1-2. left-pad the buffer, unfold into per-hop context windows --
        audio_trim = audio[:, :, :crop_samples]
        audio_padded = F.pad(audio_trim, (pad_left, 0))
        windows = audio_padded.unfold(2, context_samples, hop_samples)   # [B, 1, n_hops, context_samples]
        windows_flat = windows.reshape(B * n_hops, 1, context_samples)

        # ---- 3. MERT (frozen) on every window ------------------------------
        emb_flat = run_mert_windows(mert, processor, windows_flat)       # [B*n_hops, L, T_ctx, D]
        n_layers, T_ctx, D = emb_flat.shape[1:]

        # ---- 4. keep this hop's own frames, concatenate across hops --------
        emb_hop = emb_flat[:, :, T_ctx - n_frames_per_hop :, :]
        emb = emb_hop.reshape(B, n_hops, n_layers, n_frames_per_hop, D)
        emb = emb.permute(0, 2, 1, 3, 4).reshape(B, n_layers, n_hops * n_frames_per_hop, D)

        # ---- 5. pool, classify, loss ---------------------------------------
        with torch.set_grad_enabled(train):
            loss, logits = readout_loss(classifier, emb, labels, criterion, train_readout)
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
            config={**config, "n_hops": n_hops, "hop_samples": hop_samples,
                    "context_samples": context_samples},
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

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

    print(f"Loading {backbone} backbone (MERT_model={config['MERT_model']})...")
    mert, processor = load_backbone_and_processor(config, device)

    model_type = config.get("model_type", "LSTM")
    train_readout = resolve_train_readout(config)
    classifier = build_classifier(model_type, N_KEY_CLASSES, config).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=float(config["lr"]))
    criterion = nn.CrossEntropyLoss()

    pitch_shifts = resolve_pitch_shifts(config)
    print(
        f"\nHead ({model_type}) params : {sum(p.numel() for p in classifier.parameters()):,}\n"
        f"Readout                  : {train_readout}\n"
        f"{describe_augment(pitch_shifts)}\n"
        f"Device                   : {device}\n"
        f"Train / val crops        : {len(train_dataset)} / {len(val_dataset)}\n"
    )

    best_val_weighted = -1.0
    epoch_bar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        tr_loss, tr_acc, tr_w = run_epoch(
            train_loader, mert, processor, classifier, optimizer, criterion, device,
            train=True, train_readout=train_readout, desc=f"Epoch {epoch}/{epochs} [train]",
            pitch_shifts=pitch_shifts)
        val_loss, val_acc, val_w = run_epoch(
            val_loader, mert, processor, classifier, optimizer, criterion, device,
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
