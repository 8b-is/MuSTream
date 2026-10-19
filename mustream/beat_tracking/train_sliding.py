"""
Train a beat/downbeat probe on a frozen bidirectional MERT run in sliding windows:
every hop_duration seconds, MERT encodes the last context_duration seconds and only
the newest hop's features are kept (causal per hop). Optional `lookahead` (seconds)
slides each window that far into the future (<= context_duration - hop_duration).

Usage (from beat_tracking/):
  python train_sliding.py config/MERT_sliding.yaml <run_name>
"""

import os
import shutil
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from beat_dataloader import BeatDataset, mert_input_for_output, mert_output_len, kernels_strides_for_backbone
from models import (
    BEST_CHECKPOINT_METRIC, build_probe, compute_loss, extract_mert_features, frame_stats,
    load_backbone_and_processor, lookahead_frames, precision_recall_f1,
    val_selection_score,
)
from data_utils import set_seed

config_path = sys.argv[1]
run_name = sys.argv[2]
config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

raw_fps = config["sample_rate"] / config["hop_size"]
fps = raw_fps / config["pool_factor"]
pool_factor = config["pool_factor"]
crop_duration = config["crop_duration"]
hop_duration = config["hop_duration"]         # streaming hop, e.g. 0.2 s
context_duration = config["context_duration"]  # encoder lookback, e.g. 4.0 s
mert_batch_size = config.get("mert_batch_size")  # optional: cap flat-batch size fed to MERT at once
backbone = config.get("backbone", "pretrained")  # 'pretrained' (HF AutoModel) or 'small_bidirectional'

if context_duration < hop_duration:
    raise ValueError(
        f"context_duration={context_duration}s must be >= hop_duration={hop_duration}s"
    )

n_frames = round(crop_duration * fps)   # total label windows per crop

# Labels per hop / per context window (rounded; warns if not integral).
n_labels_per_hop = round(hop_duration * fps)
n_labels_per_context = round(context_duration * fps)
if n_labels_per_hop < 1:
    raise ValueError(
        f"hop_duration={hop_duration}s is too short: "
        f"need at least {1/fps:.3f}s for one label window."
    )
for name, duration in (("hop_duration", hop_duration), ("context_duration", context_duration)):
    if abs(duration * fps - round(duration * fps)) > 0.01:
        print(
            f"Warning: {name}={duration}s x fps={fps} Hz = "
            f"{duration*fps:.2f} (not an integer); rounding to "
            f"{round(duration * fps)} labels."
        )

# Raw-sample sizes aligned to label frames.
kernels, strides = kernels_strides_for_backbone(backbone)
hop_samples = n_labels_per_hop * pool_factor * config["hop_size"]
context_samples = mert_input_for_output(n_labels_per_context * pool_factor, kernels=kernels, strides=strides)
assert mert_output_len(context_samples, kernels=kernels, strides=strides) == n_labels_per_context * pool_factor

# Lookahead in label frames: each window slides this far ahead of its hop.
lookahead = lookahead_frames(config)
lookahead_samples = lookahead * pool_factor * config["hop_size"]
if lookahead > n_labels_per_context - n_labels_per_hop:
    raise ValueError(
        f"lookahead={config.get('lookahead')}s ({lookahead} label frames) leaves no room for the "
        f"hop itself in a {context_duration}s window ({n_labels_per_context} frames, "
        f"{n_labels_per_hop} of them the hop): need lookahead <= "
        f"{(n_labels_per_context - n_labels_per_hop) / fps:.4g}s, or a longer context_duration"
    )
pad_left = context_samples - hop_samples - lookahead_samples

# Hops per crop and the audio they need (plus the last window's lookahead tail).
n_hops = n_frames // n_labels_per_hop
crop_samples = n_hops * hop_samples + lookahead_samples   # exact; slightly > crop_duration due to CNN overhead

print(
    f"Hop            : {hop_duration}s  ({hop_samples} samples, {n_labels_per_hop} labels/hop)\n"
    f"Context        : {context_duration}s  ({context_samples} samples, {n_labels_per_context} labels)\n"
    f"Lookahead      : {lookahead / fps:.4g}s  ({lookahead} labels) -- window covers "
    f"{(n_labels_per_context - lookahead) / fps:.4g}s past + {lookahead / fps:.4g}s future\n"
    f"Hops/crop      : {n_hops}  (covering {n_hops * n_labels_per_hop}/{n_frames} label windows)"
)


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------

def run_mert_windows(mert, processor, windows_flat, device):
    """Run MERT on context windows in sub-batches of mert_batch_size; return each window's hop frames [N, L, n_labels_per_hop, D]."""
    batch = mert_batch_size or windows_flat.shape[0]
    hop_embeddings = []
    for start in range(0, windows_flat.shape[0], batch):
        emb_sub = extract_mert_features(
            mert, processor, windows_flat[start : start + batch],
            sample_rate=config["sample_rate"], pool_factor=pool_factor, device=device,
        )
        hop_end = emb_sub.shape[2] - lookahead
        # .contiguous() so the full-window output can be freed.
        hop_embeddings.append(emb_sub[:, :, hop_end - n_labels_per_hop : hop_end, :].contiguous())
    return torch.cat(hop_embeddings, dim=0)


def run_epoch(loader, mert, processor, probe, optimizer, device, train: bool, desc: str = ""):
    """One train/val epoch: left-pad, unfold into per-hop windows, run MERT, keep hop frames, probe, loss."""
    probe.train(train)
    total_loss = 0.0
    n_batches = 0
    beat_correct = beat_valid = 0.0
    beat_tp = beat_fp = beat_fn = 0.0
    downbeat_correct = downbeat_valid = 0.0
    downbeat_tp = downbeat_fp = downbeat_fn = 0.0

    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for audio, beat_seq, downbeat_seq, mask, downbeat_available in bar:
        # audio : [B, 1, T_samples]   beat_seq/downbeat_seq/mask : [B, n_frames]
        audio = audio.to(device)
        beat_seq = beat_seq.to(device)
        downbeat_seq = downbeat_seq.to(device)
        mask = mask.to(device)
        downbeat_available = downbeat_available.to(device)
        B = audio.shape[0]

        # ---- 1. trim + left-pad the streaming buffer ---------------------------
        audio_trim = audio[:, :, :crop_samples]                   # [B, 1, n_hops*hop_samples]
        audio_padded = F.pad(audio_trim, (pad_left, 0))            # [B, 1, context_samples+(n_hops-1)*hop_samples]

        # ---- 2. unfold into overlapping per-hop context windows ---------------
        windows = audio_padded.unfold(2, context_samples, hop_samples)  # [B, 1, n_hops, context_samples]
        windows_flat = windows.reshape(B * n_hops, 1, context_samples)

        # ---- 3./4. MERT (frozen) on all windows, keeping each hop's frames ----
        emb_hop = run_mert_windows(mert, processor, windows_flat, device)
        # [B*n_hops, n_layers, n_labels_per_hop, D]

        n_layers, _T_hop, D = emb_hop.shape[1:]

        # ---- concatenate the hops back into one per-crop sequence -------------
        emb = emb_hop.reshape(B, n_hops, n_layers, n_labels_per_hop, D)
        emb = emb.permute(0, 2, 1, 3, 4).reshape(B, n_layers, n_hops * n_labels_per_hop, D)

        # ---- 5. probe the concatenated embedding sequence in one pass -------
        logits = probe(emb)  # [B, N*T_c, 2]

        # ---- 6. align with labels --------------------------------------------
        T_r = min(logits.shape[1], beat_seq.shape[1])
        logits_use = logits[:, :T_r]
        beat_seq_use = beat_seq[:, :T_r]
        downbeat_seq_use = downbeat_seq[:, :T_r]
        mask_use = mask[:, :T_r]

        loss, beat_loss, downbeat_loss = compute_loss(
            logits_use, beat_seq_use, downbeat_seq_use, mask_use, downbeat_available, config
        )

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        (b_correct, b_valid, (b_tp, b_fp, b_fn)), (d_correct, d_valid, (d_tp, d_fp, d_fn)) = frame_stats(
            logits_use, beat_seq_use, downbeat_seq_use, mask_use, downbeat_available
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
            config={**config, "n_hops": n_hops, "hop_samples": hop_samples, "context_samples": context_samples},
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    dataset_kwargs = dict(
        audio_dir=config["audio_dir"],
        annotation_dir=config["annotation_dir"],
        split_dir=config["split_dir"],
        sr=config["sample_rate"],
        fps=fps,
        pool_factor=pool_factor,
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
