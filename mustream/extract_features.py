"""
Extract MuSTream features from audio with a released (or trained) MuSTream backbone.

The audio is fed chunk by chunk through one streaming session (KV cache), exactly as
in real-time use; the result does not depend on the chunk size. Each file is saved as
<out-dir>/<name>.npz with `features` (float32), `frame_rate` (Hz) and `layer`.

Usage (from anywhere):
  python mustream/extract_features.py song.wav --model 95M --variant r15
  python mustream/extract_features.py a.wav b.mp3 --model 26M --layer average --pool 5
  python mustream/extract_features.py song.wav --checkpoint path/to/your_run/step=100000.ckpt

Python:
  from extract_features import load_backbone, load_audio, extract
  model = load_backbone("ckpt/pytorch/backbone/streaming_95M_r15.ckpt")
  feats = extract(model, load_audio("song.wav"))     # [L, T, D] at 75 fps
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from modeling.load_checkpoint import load_streaming_mert  # noqa: E402

SAMPLE_RATE = 24000
HOP = 320                                   # samples per frame -> 75 fps
FRAME_RATE = SAMPLE_RATE / HOP
DEFAULT_CKPT_DIR = os.path.join(HERE, "ckpt", "pytorch", "backbone")


def load_backbone(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """A MuSTream backbone checkpoint (released or trained) -> frozen model on `device`."""
    return load_streaming_mert(checkpoint_path).to(device)


def load_audio(path: str) -> torch.Tensor:
    """Audio file -> mono float32 waveform at 24 kHz, [T_samples]."""
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform


@torch.no_grad()
def extract(model: torch.nn.Module, waveform: torch.Tensor, chunk_seconds: float = 0.2) -> torch.Tensor:
    """MuSTream features of a [T_samples] waveform -> [L, T_frames, D] at 75 fps (on the CPU).

    L = number of hidden states (13 for 26M/95M, 25 for 330M; index 0 is the transformer
    input). chunk_seconds must be a multiple of the backbone's block (0.2 s).
    """
    device = next(model.parameters()).device
    chunk_frames = round(chunk_seconds * FRAME_RATE)
    block = getattr(model, "lookahead_block", 1)
    if chunk_frames < 1 or chunk_frames % block:
        raise ValueError(f"chunk_seconds={chunk_seconds} is not a multiple of the backbone's "
                         f"{block}-frame block ({block / FRAME_RATE:.2f} s)")
    n_frames = waveform.shape[-1] // HOP
    if n_frames == 0:
        raise ValueError("audio is shorter than one frame")
    chunk_samples = chunk_frames * HOP
    n_chunks = -(-n_frames // chunk_frames)
    audio = torch.nn.functional.pad(waveform[: n_frames * HOP], (0, n_chunks * chunk_samples - n_frames * HOP))
    audio = audio[None].to(device)                               # [1, T]

    outputs = []
    with model.feature_extractor.streaming(1), model.transformer.streaming(1):
        for c in range(n_chunks):
            _, hidden_states = model(audio[:, c * chunk_samples:(c + 1) * chunk_samples],
                                     output_hidden_states=True)
            outputs.append(torch.stack(hidden_states, dim=1)[0].cpu())   # [L, chunk_frames, D]
    return torch.cat(outputs, dim=1)[:, :n_frames]               # drop the padded tail


def select_and_pool(features: torch.Tensor, layer: str = "all", pool: int = 1) -> torch.Tensor:
    """[L, T, D] -> layer selection ('all' | 'average' | an index) and mean-pooling over `pool` frames."""
    if layer == "average":
        features = features.mean(dim=0)
    elif layer != "all":
        features = features[int(layer)]
    if pool > 1:
        T = features.shape[-2] // pool * pool
        features = features[..., :T, :]
        features = features.reshape(*features.shape[:-2], T // pool, pool, features.shape[-1]).mean(-2)
    return features


def main():
    parser = argparse.ArgumentParser(description="Extract MuSTream features from audio files.")
    parser.add_argument("audio", nargs="+", help="audio file(s); any format torchaudio reads")
    parser.add_argument("--model", choices=["26M", "95M", "330M"], default="95M",
                        help="released MuSTream size (default: %(default)s)")
    parser.add_argument("--variant", choices=["r0", "r15"], default="r15",
                        help="released MuSTream variant (default: %(default)s)")
    parser.add_argument("--checkpoint", default=None,
                        help="a backbone .ckpt to use instead of --model/--variant (e.g. your own run)")
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR,
                        help="folder of the released MuSTream backbones (default: mustream/ckpt/pytorch/backbone)")
    parser.add_argument("--layer", default="all",
                        help="'all' -> [L, T, D], 'average' -> [T, D] mean over layers, or a layer index -> [T, D]")
    parser.add_argument("--pool", type=int, default=1,
                        help="mean-pool this many frames (1 = 75 fps, 5 = 15 fps)")
    parser.add_argument("--chunk-seconds", type=float, default=0.2,
                        help="audio per streaming step; a multiple of 0.2 s (speed only, same result)")
    parser.add_argument("--out-dir", default="features", help="where to write <name>.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    # Full fp32 on GPU, so the features do not depend on the chunk size.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    checkpoint = args.checkpoint or os.path.join(args.ckpt_dir, f"streaming_{args.model}_{args.variant}.ckpt")
    if not os.path.isfile(checkpoint):
        sys.exit(f"backbone checkpoint not found: {checkpoint} (see mustream/README.md to download it)")
    print(f"Loading {checkpoint} on {args.device} ...")
    model = load_backbone(checkpoint, args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    frame_rate = FRAME_RATE / args.pool
    for path in args.audio:
        feats = select_and_pool(extract(model, load_audio(path), args.chunk_seconds), args.layer, args.pool)
        out = Path(args.out_dir) / (Path(path).stem + ".npz")
        np.savez(out, features=feats.numpy().astype(np.float32), frame_rate=frame_rate, layer=args.layer)
        print(f"{path}: features {tuple(feats.shape)} at {frame_rate:g} fps -> {out}")


if __name__ == "__main__":
    main()
