"""Train a chord probe on frozen bidirectional MERT with sliding-window inference.

Every hop_duration seconds, MERT re-encodes the last context_duration seconds
(zero-padded at the clip start) and only the newest hop's embeddings are kept,
so features see no audio past the current hop. config['lookahead'] (seconds)
slides each window that far into the future; it must be
<= context_duration - hop_duration. Set mert_batch_size to bound memory.

Usage (from chord_recognition/):
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

# Let sibling-module imports resolve when called by path.
sys.path.insert(0, os.path.dirname(__file__))
from models import (
    build_classifier, extract_MERT_features, load_backbone_and_processor, lookahead_frames,
)
from chord_augment import (
    apply_pitch_shift_batch, describe as describe_augment, resolve_pitch_shifts,
)
from chord_dataloader import (
    MultiDatasetChordDataset, mert_input_for_output, mert_output_len, kernels_strides_for_backbone,
)
from chord_vocab import build_transpose_table, describe as describe_vocab
import pickle


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
config_path = sys.argv[1]
config      = yaml.load(open(config_path), Loader=yaml.FullLoader)
run_name    = sys.argv[2]

epochs        = config["epochs"]
crop_duration = config["crop_duration"]
sample_rate   = config["sample_rate"]
hop_size      = config["hop_size"]
pool_factor   = config["pool_factor"]
hop_duration     = config["hop_duration"]       # streaming hop, e.g. 0.2 s
context_duration = config["context_duration"]   # encoder lookback, e.g. 4.0 s
mert_batch_size  = config.get("mert_batch_size")  # optional: cap flat-batch size fed to MERT at once
backbone         = config.get("backbone", "pretrained")  # 'pretrained' (HF AutoModel) or 'small_bidirectional'

label_rate  = sample_rate / hop_size / pool_factor   # e.g. 15 Hz
window_size = 1.0 / label_rate                       # seconds per label window
n_windows   = round(crop_duration * label_rate)      # total label windows per crop

if context_duration < hop_duration:
    raise ValueError(
        f"context_duration={context_duration}s must be >= hop_duration={hop_duration}s"
    )

# Labels per hop / per context window (rounded; warns if not a whole number).
n_labels_per_hop = round(hop_duration * label_rate)
n_labels_per_context = round(context_duration * label_rate)
if n_labels_per_hop < 1:
    raise ValueError(
        f"hop_duration={hop_duration}s is too short: "
        f"need at least {1/label_rate:.3f}s for one label window."
    )
for name, duration in (("hop_duration", hop_duration), ("context_duration", context_duration)):
    if abs(duration * label_rate - round(duration * label_rate)) > 0.01:
        print(
            f"Warning: {name}={duration}s × label_rate={label_rate} Hz = "
            f"{duration*label_rate:.2f} (not an integer); rounding to "
            f"{round(duration * label_rate)} labels."
        )

# Raw-sample sizes, aligned to label-window boundaries.
_kernels, _strides = kernels_strides_for_backbone(backbone)
context_samples = mert_input_for_output(n_labels_per_context * pool_factor, kernels=_kernels, strides=_strides)
assert mert_output_len(context_samples, kernels=_kernels, strides=_strides) == n_labels_per_context * pool_factor
hop_samples     = n_labels_per_hop * pool_factor * hop_size

# Lookahead in label frames: each window slides that far ahead.
lookahead = lookahead_frames(config)
lookahead_samples = lookahead * pool_factor * hop_size
if lookahead > n_labels_per_context - n_labels_per_hop:
    raise ValueError(
        f"lookahead={config.get('lookahead')}s ({lookahead} label frames) leaves no room for the "
        f"hop itself in a {context_duration}s window ({n_labels_per_context} frames, "
        f"{n_labels_per_hop} of them the hop): need lookahead <= "
        f"{(n_labels_per_context - n_labels_per_hop) / label_rate:.4g}s, or a longer context_duration"
    )
pad_left = context_samples - hop_samples - lookahead_samples

# Hops per crop and the audio they need (plus the lookahead tail).
n_hops = n_windows // n_labels_per_hop
mert_crop_samples = n_hops * hop_samples + lookahead_samples

print(
    f"Hop            : {hop_duration}s  ({hop_samples} samples, {n_labels_per_hop} labels/hop)\n"
    f"Context        : {context_duration}s  ({context_samples} samples, {n_labels_per_context} labels)\n"
    f"Lookahead      : {lookahead / label_rate:.4g}s  ({lookahead} labels) -- window covers "
    f"{(n_labels_per_context - lookahead) / label_rate:.4g}s past + {lookahead / label_rate:.4g}s future\n"
    f"Hops/crop      : {n_hops}  (covering {n_hops * n_labels_per_hop}/{n_windows} label windows)"
)


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------

def run_mert_windows(
    mert: nn.Module,
    processor: Wav2Vec2FeatureExtractor,
    windows_flat: torch.Tensor,   # [N, 1, context_samples]
    device: torch.device,
) -> torch.Tensor:
    """Run MERT on a flat batch of windows, in sub-batches of mert_batch_size if set."""
    if mert_batch_size is None or windows_flat.shape[0] <= mert_batch_size:
        return extract_MERT_features(
            mert, processor, windows_flat, pool_factor=pool_factor
        )
    chunks = []
    for start in range(0, windows_flat.shape[0], mert_batch_size):
        sub = windows_flat[start : start + mert_batch_size]
        chunks.append(extract_MERT_features(
            mert, processor, sub, pool_factor=pool_factor
        ))
    return torch.cat(chunks, dim=0)


def run_epoch(
    loader: DataLoader,
    mert: nn.Module,
    processor: Wav2Vec2FeatureExtractor,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    desc: str = "",
    pitch_shifts=None,        # training only: allowed semitone shifts
    transpose_table=None,     # [V, 12] chord_vocab table; required with pitch_shifts
) -> Tuple[float, float]:
    """Run one epoch: encode one window per hop, keep each hop's frames, classify."""
    classifier.train(train)
    total_loss    = 0.0
    total_correct = 0
    total_frames  = 0

    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for audio, labels, _mask, n_steps in bar:
        # audio : [B, 1, T_samples]   labels : [B, N_windows]
        audio  = audio.to(device)
        labels = labels.to(device)
        if train and pitch_shifts:
            # Batched GPU pitch shift; transposes audio and labels together.
            audio, labels = apply_pitch_shift_batch(
                audio, labels, n_steps, transpose_table)
        B      = audio.shape[0]

        # ---- 1. trim + left-pad the streaming buffer ---------------------------
        audio_trim = audio[:, :, :mert_crop_samples]              # [B, 1, n_hops*hop_samples]
        audio_padded = F.pad(audio_trim, (pad_left, 0))            # [B, 1, context_samples+(n_hops-1)*hop_samples]

        # ---- 2. unfold into overlapping per-hop context windows ---------------
        windows = audio_padded.unfold(2, context_samples, hop_samples)  # [B, 1, n_hops, context_samples]
        windows_flat = windows.reshape(B * n_hops, 1, context_samples)

        # ---- 3. MERT (frozen) on all windows -----------------------------------
        emb_flat = run_mert_windows(mert, processor, windows_flat, device)
        # [B*n_hops, n_layers, n_labels_per_context, D]

        n_layers, T_ctx, D = emb_flat.shape[1:]

        # ---- 4. keep this hop's embeddings, concatenate across hops -----------
        # The hop's frames end `lookahead` frames before the window's right edge.
        hop_end = T_ctx - lookahead
        emb_hop = emb_flat[:, :, hop_end - n_labels_per_hop : hop_end, :]  # [B*n_hops, n_layers, n_labels_per_hop, D]
        emb = emb_hop.reshape(B, n_hops, n_layers, n_labels_per_hop, D)
        emb = emb.permute(0, 2, 1, 3, 4).reshape(B, n_layers, n_hops * n_labels_per_hop, D)

        # ---- 5. classify the concatenated embedding sequence in one pass ------
        with torch.set_grad_enabled(train):
            logits = classifier(emb)                            # [B, N*T_c, V]
            V = logits.shape[-1]

            # ---- 6. align with labels ------------------------------------------
            assert logits.shape[1] == labels.shape[1]
            T_r = min(logits.shape[1], labels.shape[1])
            logits_use = logits[:, :T_r]                        # [B, T_r, V]
            labels_use = labels[:, :T_r]                        # [B, T_r]

            loss = criterion(
                logits_use.reshape(-1, V),
                labels_use.reshape(-1),
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds         = logits_use.argmax(dim=-1)               # [B, T_r]
        batch_correct = (preds == labels_use).sum().item()
        batch_frames  = B * T_r

        total_loss    += loss.item() * B
        total_correct += batch_correct
        total_frames  += batch_frames

        bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{batch_correct / batch_frames:.4f}",
        )

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_frames
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device  = torch.device(config["device"])
    save_dir = f"logs/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(save_dir, "config.yaml"))
    torch.manual_seed(config.get("seed", 42))

    # --- wandb ---------------------------------------------------------------
    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            entity=config.get("wandb_entity"),
            project=config.get("wandb_project", "chord-recognition"),
            name=run_name,
            config={**config, "n_hops": n_hops, "hop_samples": hop_samples, "context_samples": context_samples},
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    # --- data ----------------------------------------------------------------
    chord2idx, idx2chord = pickle.load(open(config["vocab_path"], 'rb'))
    vocab = chord2idx
    # Built up front so a vocabulary not closed under transposition fails early.
    pitch_shifts = resolve_pitch_shifts(config)
    transpose_table = build_transpose_table(idx2chord) if pitch_shifts else None
    print(describe_vocab(idx2chord))
    print("Loading datasets …")

    dataset_kwargs = dict(
        split_dir=config["split_dir"],
        vocab_file=config["vocab_path"],
        sr=sample_rate,
        window=window_size,
        crop_samples=mert_crop_samples,
        crop_duration=crop_duration,
    )
    train_dataset = MultiDatasetChordDataset(
        split="train", pitch_shifts=pitch_shifts, **dataset_kwargs)
    val_dataset   = MultiDatasetChordDataset(split="val",   **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True,
    )

    # --- MERT (frozen) -------------------------------------------------------
    print(f"Loading {backbone} backbone (MERT_model={config['MERT_model']}) …")
    mert, processor = load_backbone_and_processor(config, device)

    # --- classifier ----------------------------------------------------------
    model_type = config.get("model_type", "LSTM")
    classifier = build_classifier(model_type, len(vocab), config).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in classifier.parameters())
    print(
        f"\nClassifier params : {n_params:,}\n"
        f"{describe_augment(pitch_shifts)}\n"
        f"Seed              : {config.get('seed', 42)}\n"
        f"Device            : {device}\n"
        f"Train / val clips : {len(train_dataset)} / {len(val_dataset)}\n"
    )

    # --- training loop -------------------------------------------------------
    best_val_acc = -1.0
    epoch_bar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        tr_loss, tr_acc = run_epoch(
            train_loader, mert, processor, classifier,
            optimizer, criterion, device, train=True,
            desc=f"Epoch {epoch}/{epochs} [train]",
            pitch_shifts=pitch_shifts, transpose_table=transpose_table,
        )
        val_loss, val_acc = run_epoch(
            val_loader, mert, processor, classifier,
            optimizer, criterion, device, train=False,
            desc=f"Epoch {epoch}/{epochs} [val]  ",
        )

        epoch_bar.set_postfix(
            tr_loss=f"{tr_loss:.4f}", tr_acc=f"{tr_acc:.4f}",
            val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}",
        )
        tqdm.write(
            f"Epoch {epoch:>3}/{epochs}  "
            f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": tr_loss, "train/acc": tr_acc,
                "val/loss":   val_loss, "val/acc":   val_acc,
            })

        # config is saved so evaluation can check training settings (e.g. lookahead).
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
