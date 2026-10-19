"""
Train a beat/downbeat probe on a frozen causal streaming MERT
(modeling/streaming/streaming_mert.py, KV-cache). The backbone is causal by
construction, so one forward pass over the whole crop equals streaming inference.

config['backbone']:
  'streaming'       -- published MERT-330M/95M in the streaming architecture;
                       mert_align_checkpoint (causal distillation) is optional.
  'small_streaming' -- the 26M causal student; mert_align_checkpoint is required
                       and emb_dim must equal its hidden size.

Usage (from beat_tracking/):
  python train_streaming.py config/MERT_streaming.yaml <run_name>
"""

import os
import shutil
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from beat_dataloader import BeatDataset, mert_input_for_output, mert_output_len
from models import (
    BEST_CHECKPOINT_METRIC, build_probe, compute_loss,
    extract_streaming_features, frame_stats,
    load_streaming_backbone, precision_recall_f1, val_selection_score,
)
from data_utils import set_seed

config_path = sys.argv[1]
run_name = sys.argv[2]
config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

sample_rate = config["sample_rate"]
pool_factor = config["pool_factor"]
backbone_name = config.get("backbone", "streaming")
# 'streaming' (published MERT) or 'small_streaming' (26M causal student).
STREAMING_BACKBONES = ("streaming", "small_streaming")
assert backbone_name in STREAMING_BACKBONES, (
    f"invalid backbone {backbone_name!r}: only {STREAMING_BACKBONES} "
    "(modeling/streaming/streaming_mert.py) are supported here"
)


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------

def run_epoch(
    loader, model, probe, optimizer, device, target_raw_frames, train: bool, desc: str = "",
):
    probe.train(train)
    total_loss = 0.0
    n_batches = 0
    beat_correct = beat_valid = 0.0
    beat_tp = beat_fp = beat_fn = 0.0
    downbeat_correct = downbeat_valid = 0.0
    downbeat_tp = downbeat_fp = downbeat_fn = 0.0

    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for audio, beat_seq, downbeat_seq, mask, downbeat_available in bar:
        beat_seq = beat_seq.to(device)
        downbeat_seq = downbeat_seq.to(device)
        mask = mask.to(device)
        downbeat_available = downbeat_available.to(device)

        embeddings = extract_streaming_features(
            model, audio,
            pool_factor=pool_factor,
            target_raw_frames=target_raw_frames, device=device,
        )  # [B, L, T_r, D]
        logits = probe(embeddings)  # [B, T_r, 2]

        T = min(logits.shape[1], beat_seq.shape[1])
        logits = logits[:, :T]
        beat_seq_t = beat_seq[:, :T]
        downbeat_seq_t = downbeat_seq[:, :T]
        mask_t = mask[:, :T]

        loss, beat_loss, downbeat_loss = compute_loss(
            logits, beat_seq_t, downbeat_seq_t, mask_t, downbeat_available, config
        )

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        (b_correct, b_valid, (b_tp, b_fp, b_fn)), (d_correct, d_valid, (d_tp, d_fp, d_fn)) = frame_stats(
            logits, beat_seq_t, downbeat_seq_t, mask_t, downbeat_available
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

    raw_fps = sample_rate / config["hop_size"]
    fps = raw_fps / pool_factor
    model_type = config.get("model_type", "LSTM")
    n_frames = round(config["crop_duration"] * fps)
    target_raw_frames = n_frames * pool_factor
    mert_crop_samples = target_raw_frames * config["hop_size"]

    print(f"Backbone: {backbone_name}")
    dataset_kwargs = dict(
        audio_dir=config["audio_dir"],
        annotation_dir=config["annotation_dir"],
        split_dir=config["split_dir"],
        sr=sample_rate,
        fps=fps,
        pool_factor=pool_factor,
        crop_samples=mert_crop_samples,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        split_seed=config["split_seed"],
        target_widen=config["target_widen"],
        audio_ext=config.get("audio_ext", "wav"),
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

    probe = build_probe(model_type, config).to(device)

    lr = float(config["lr"])
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    n_params = sum(p.numel() for p in probe.parameters())
    print(f"Probe params: {n_params:,} (lr={lr:g})  |  train={len(train_dataset)}  val={len(val_dataset)}  |  device={device}")
    print("Backbone: frozen (probe-only training)")

    # best.pt = highest val score (models.val_selection_score).
    best_val_score = -1.0
    epoch_bar = tqdm(range(1, config["epochs"] + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        tr_stats = run_epoch(
            train_loader, model, probe, optimizer, device, target_raw_frames, train=True,
            desc=f"Epoch {epoch}/{config['epochs']} [train]",
        )
        val_stats = run_epoch(
            val_loader, model, probe, optimizer, device, target_raw_frames, train=False,
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
                "train/beat_acc": tr_stats["beat_acc"], "train/downbeat_acc": tr_stats["downbeat_acc"],
                "train/beat_precision": tr_stats["beat_precision"], "train/beat_recall": tr_stats["beat_recall"],
                "train/beat_f1": tr_stats["beat_f1"],
                "train/downbeat_precision": tr_stats["downbeat_precision"],
                "train/downbeat_recall": tr_stats["downbeat_recall"], "train/downbeat_f1": tr_stats["downbeat_f1"],
                "val/loss": val_loss,
                "val/f1": val_score,
                "val/beat_acc": val_stats["beat_acc"], "val/downbeat_acc": val_stats["downbeat_acc"],
                "val/beat_precision": val_stats["beat_precision"], "val/beat_recall": val_stats["beat_recall"],
                "val/beat_f1": val_stats["beat_f1"],
                "val/downbeat_precision": val_stats["downbeat_precision"],
                "val/downbeat_recall": val_stats["downbeat_recall"], "val/downbeat_f1": val_stats["downbeat_f1"],
            })

        # best.pt plus one checkpoint every ckpt_interval epochs.
        is_best = val_score > best_val_score
        is_periodic = epoch % config["ckpt_interval"] == 0
        if is_best or is_periodic:
            ckpt = {"epoch": epoch, "model_state": probe.state_dict(), "config": config}
            if is_best:
                best_val_score = val_score
                torch.save(ckpt, os.path.join(save_dir, "best.pt"))
            if is_periodic:
                torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch:03d}_{val_score:.4f}.pt"))

    print(f"\nBest {BEST_CHECKPOINT_METRIC}: {best_val_score:.4f}  ->  {save_dir}/best.pt")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
