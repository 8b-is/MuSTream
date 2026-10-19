"""
Train a beat/downbeat probe on a frozen MERT run over the whole crop (offline, bidirectional).

Usage (from beat_tracking/):
  python train_offline.py config/MERT_offline.yaml <run_name>
"""

import os
import shutil
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from beat_dataloader import BeatDataset, mert_input_for_output, mert_output_len, kernels_strides_for_backbone
from models import (
    BEST_CHECKPOINT_METRIC, build_probe, compute_loss, extract_mert_features, frame_stats,
    load_backbone_and_processor, precision_recall_f1, val_selection_score,
)
from data_utils import set_seed

config_path = sys.argv[1]
run_name = sys.argv[2]
config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

raw_fps = config["sample_rate"] / config["hop_size"]
fps = raw_fps / config["pool_factor"]
backbone = config.get("backbone", "pretrained")  # 'pretrained' (HF AutoModel) or 'small_bidirectional'


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------

def run_epoch(loader, mert, processor, probe, optimizer, device, train: bool, desc: str = ""):
    probe.train(train)
    total_loss = 0.0
    n_batches = 0
    beat_correct = beat_valid = 0.0
    beat_tp = beat_fp = beat_fn = 0.0
    downbeat_correct = downbeat_valid = 0.0
    downbeat_tp = downbeat_fp = downbeat_fn = 0.0

    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for audio, beat_seq, downbeat_seq, mask, downbeat_available in bar:
        audio = audio.to(device)
        beat_seq = beat_seq.to(device)
        downbeat_seq = downbeat_seq.to(device)
        mask = mask.to(device)
        downbeat_available = downbeat_available.to(device)

        embeddings = extract_mert_features(
            mert, processor, audio,
            sample_rate=config["sample_rate"],
            pool_factor=config["pool_factor"],
            device=device,
        )
        logits = probe(embeddings)  # [B, T, 2]

        loss, beat_loss, downbeat_loss = compute_loss(
            logits, beat_seq, downbeat_seq, mask, downbeat_available, config
        )

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        (b_correct, b_valid, (b_tp, b_fp, b_fn)), (d_correct, d_valid, (d_tp, d_fp, d_fn)) = frame_stats(
            logits, beat_seq, downbeat_seq, mask, downbeat_available
        )
        beat_correct += b_correct; beat_valid += b_valid
        beat_tp += b_tp; beat_fp += b_fp; beat_fn += b_fn
        downbeat_correct += d_correct; downbeat_valid += d_valid
        downbeat_tp += d_tp; downbeat_fp += d_fp; downbeat_fn += d_fn

        total_loss += loss.item()
        n_batches += 1
        bar.set_postfix(
            loss=f"{loss.item():.4f}",
            beat_acc=f"{b_correct / max(b_valid, 1):.3f}",
            db_acc=f"{d_correct / max(d_valid, 1):.3f}",
        )

    beat_acc = beat_correct / max(beat_valid, 1)
    downbeat_acc = downbeat_correct / max(downbeat_valid, 1)
    beat_precision, beat_recall, beat_f1 = precision_recall_f1(beat_tp, beat_fp, beat_fn)
    downbeat_precision, downbeat_recall, downbeat_f1 = precision_recall_f1(downbeat_tp, downbeat_fp, downbeat_fn)

    return {
        "loss": total_loss / n_batches,
        "beat_acc": beat_acc,
        "downbeat_acc": downbeat_acc,
        "beat_precision": beat_precision,
        "beat_recall": beat_recall,
        "beat_f1": beat_f1,
        "downbeat_precision": downbeat_precision,
        "downbeat_recall": downbeat_recall,
        "downbeat_f1": downbeat_f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(config["seed"])
    device = torch.device(config["device"])
    save_dir = os.path.join("logs", run_name)
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(save_dir, "config.yaml"))

    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            entity=config.get("wandb_entity"),
            project=config.get("wandb_project", "beat-tracking"),
            name=run_name,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    # Crop length in samples that yields exactly crop_duration * fps frames.
    kernels, strides = kernels_strides_for_backbone(backbone)
    n_frames = round(config["crop_duration"] * fps)
    raw_frames = n_frames * config["pool_factor"]
    crop_samples = mert_input_for_output(raw_frames, kernels=kernels, strides=strides)
    assert mert_output_len(crop_samples, kernels=kernels, strides=strides) == raw_frames

    dataset_kwargs = dict(
        audio_dir=config["audio_dir"],
        annotation_dir=config["annotation_dir"],
        split_dir=config["split_dir"],
        sr=config["sample_rate"],
        fps=fps,
        pool_factor=config["pool_factor"],
        crop_samples=crop_samples,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        split_seed=config["split_seed"],
        target_widen=config["target_widen"],
        audio_ext=config.get("audio_ext", "wav"),
        kernels=kernels,
        strides=strides,
    )
    train_dataset = BeatDataset(split="train", mode="train", **dataset_kwargs)
    val_dataset = BeatDataset(split="val", mode="train", **dataset_kwargs)
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
    probe = build_probe(model_type, config).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=float(config["lr"]))

    n_params = sum(p.numel() for p in probe.parameters())
    print(f"Probe params: {n_params:,}  |  train={len(train_dataset)}  val={len(val_dataset)}  |  device={device}")

    # best.pt = highest val score (models.val_selection_score).
    best_val_score = -1.0
    epoch_bar = tqdm(range(1, config["epochs"] + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        tr_stats = run_epoch(
            train_loader, mert, processor, probe, optimizer, device, train=True,
            desc=f"Epoch {epoch}/{config['epochs']} [train]",
        )
        val_stats = run_epoch(
            val_loader, mert, processor, probe, optimizer, device, train=False,
            desc=f"Epoch {epoch}/{config['epochs']} [val]",
        )
        val_loss = val_stats["loss"]
        val_score = val_selection_score(val_stats)

        tqdm.write(
            f"Epoch {epoch:>3}/{config['epochs']}  "
            f"tr_loss={tr_stats['loss']:.4f} tr_beat_acc={tr_stats['beat_acc']:.3f} tr_db_acc={tr_stats['downbeat_acc']:.3f} "
            f"tr_beat_f1={tr_stats['beat_f1']:.3f} tr_db_f1={tr_stats['downbeat_f1']:.3f}  "
            f"val_loss={val_loss:.4f} val_beat_acc={val_stats['beat_acc']:.3f} val_db_acc={val_stats['downbeat_acc']:.3f} "
            f"val_beat_f1={val_stats['beat_f1']:.3f} val_db_f1={val_stats['downbeat_f1']:.3f} "
            f"val_f1={val_score:.4f}"
        )
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": tr_stats["loss"],
                "train/beat_acc": tr_stats["beat_acc"], 
                "train/downbeat_acc": tr_stats["downbeat_acc"],
                "train/beat_precision": tr_stats["beat_precision"], 
                "train/beat_recall": tr_stats["beat_recall"],
                "train/beat_f1": tr_stats["beat_f1"],
                "train/downbeat_precision": tr_stats["downbeat_precision"],
                "train/downbeat_recall": tr_stats["downbeat_recall"], 
                "train/downbeat_f1": tr_stats["downbeat_f1"],
                "val/loss": val_loss,
                "val/f1": val_score,
                "val/beat_acc": val_stats["beat_acc"], 
                "val/downbeat_acc": val_stats["downbeat_acc"],
                "val/beat_precision": val_stats["beat_precision"], 
                "val/beat_recall": val_stats["beat_recall"],
                "val/beat_f1": val_stats["beat_f1"],
                "val/downbeat_precision": val_stats["downbeat_precision"],
                "val/downbeat_recall": val_stats["downbeat_recall"], 
                "val/downbeat_f1": val_stats["downbeat_f1"],
            })

        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(
                {"epoch": epoch, "model_state": probe.state_dict(), "config": config},
                os.path.join(save_dir, "best.pt"),
            )
        if epoch % config["ckpt_interval"] == 0:
            torch.save(
                {"epoch": epoch, "model_state": probe.state_dict(), "config": config},
                os.path.join(save_dir, f"epoch_{epoch:03d}_{val_score:.4f}.pt"),
            )

    print(f"\nBest {BEST_CHECKPOINT_METRIC}: {best_val_score:.4f}  ->  {save_dir}/best.pt")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
