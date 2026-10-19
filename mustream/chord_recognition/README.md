# Chord Recognition

Frame-level chord recognition with probes on frozen MERT backbones. The backbone
produces per-frame embeddings (pooled to 15 fps); a small trainable head (`MLP`,
`LSTM` or `Transformer`) predicts one of 121 chord classes (12 roots x 10
qualities + `N`) per frame. Only the head is trained.

## Quick Start

Download the released checkpoints to `mustream/ckpt/` (see `mustream/README.md`), then run a
probe on your own audio; here the 95M backbone, `r15`:

```bash
cd chord_recognition
python infer_streaming.py ../ckpt/pytorch/chord/streaming_95M_r15/config.yaml \
                          ../ckpt/pytorch/chord/streaming_95M_r15/best.pt --audio song.wav
```

This writes `predictions/song.lab` (`start end chord` per line; `--out-dir` changes the
folder), decoded with the config's `decoder:` block (`blocksync`).

Any of `streaming_{26M,95M,330M}_{r0,r15}` works the same way, and `--audio` takes
several files (any format torchaudio reads; audio is resampled to 24 kHz mono).

Without `--audio`, the script evaluates on your own labelled split instead: set
`split_dir` (and `eval_split`) in that `config.yaml` first.

## Backbones

| Backbone | `backbone` | `MERT_model` | `emb_dim` |
|---|---|---|---|
| MuSTream-330M (causal streaming) | `streaming` | `m-a-p/MERT-v1-330M` | 1024 |
| MuSTream-95M (causal streaming) | `streaming` | `m-a-p/MERT-v1-95M` | 768 |
| MuSTream-26M (causal streaming) | `small_streaming` | `m-a-p/MERT-v1-95M` (base family) | 432 |
| MERT-330M (published, bidirectional) | `pretrained` | `m-a-p/MERT-v1-330M` | 1024 |
| MERT-95M (published, bidirectional) | `pretrained` | `m-a-p/MERT-v1-95M` | 768 |
| 26M bidirectional student (size distillation) | `small_bidirectional` | `m-a-p/MERT-v1-95M` (base family) | 432 |

MuSTream backbones are used by the causal streaming setup; the bidirectional ones by
the offline, chunked and sliding-window setups.

Each config has the alternatives as commented-out blocks (A / B / C); swap the
whole block. For MuSTream, `mert_align_checkpoint` is the backbone checkpoint: a
released one in `ckpt/pytorch/backbone/`, or a `train_mert_causal_distill.py` run. It is
required for MuSTream-26M, and for `small_bidirectional` (a `train_mert_size_distill.py`
checkpoint). For the 26M backbones, `MERT_model` must be the MERT family the student
was distilled from, and `student_*` must match the distillation run.

## Data

The datasets and splits used in our experiments are not distributed.

> **Note:** put your own split files in `chord_recognition/data_splits/splits/`
> (or point `split_dir` in the configs elsewhere). The streaming configs already point `mert_align_checkpoint` at the released
> backbones in `ckpt/pytorch/backbone/`.

The code expects, relative to `chord_recognition/`:

- `data_splits/splits/{train,val,test}.pkl`: each a pickled
  list of dicts with `audio_path` (any torchaudio-readable file; 24 kHz mono wav
  is fastest) and `chord_label_path` (a `.lab` file of `start end chord` lines,
  Harte chord syntax).
- `data_splits/dictionary_121.pkl`: the vocabulary, created with
  `python chord_vocab.py --write data_splits/dictionary_121.pkl`.

Point `split_dir` / `vocab_path` in the config elsewhere if needed.

## Inference modes

- **Offline** (`train_offline.py`, `config/MERT_offline.yaml`): bidirectional MERT
  sees whole `crop_duration`-second windows (10 s). Not causal; the reference.
- **Chunked** (`train_chunked.py`, `config/MERT_chunked.yaml`): bidirectional MERT
  sees independent, non-overlapping chunks of `chunk_duration` seconds. Causal at
  chunk granularity; latency is one chunk, but each chunk has no past context.
- **Sliding-window** (`train_sliding.py`, `config/MERT_sliding.yaml`): every
  `hop_duration` seconds, bidirectional MERT re-encodes the last `context_duration`
  seconds and keeps only the newest hop. Causal with one-hop latency and a bounded
  past, at the cost of re-encoding each sample `context_duration / hop_duration` times.
- **Causal streaming** (`train_streaming.py`, `config/MERT_streaming.yaml`):
  MuSTream, the causal KV-cache streaming backbone (`modeling/streaming/streaming_mert.py`), so
  frame *t* only depends on audio up to *t*. Inference feeds the song chunk by
  chunk with bounded memory (`streaming_context` frames of history); latency is
  the backbone's own frame/block latency. `mert_align_checkpoint` (optional for
  330M/95M) loads the causally distilled weights; `backbone_positional_embedding`
  must match that checkpoint (use `rope`).

All probes are strictly causal over frames; `model_type` selects
`MLP` (per-frame), `LSTM` (unidirectional) or `Transformer` (causal attention).

## Train and evaluate

Run everything from `chord_recognition/` (configs and imports use relative paths).
Checkpoints go to `logs/<run_name>/` (`best.pt` = best val frame accuracy, plus
`epoch_XXX_<acc>.pt` every `ckpt_interval` epochs, and a copy of the config).
Weights & Biases logging is off by default; set `use_wandb: true` (plus `wandb_entity` / `wandb_project`) to enable it.

```bash
cd chord_recognition

# Offline
python train_offline.py   config/MERT_offline.yaml   <run_name>
python infer.py           logs/<run_name>/config.yaml logs/<run_name>/best.pt

# Chunked
python train_chunked.py   config/MERT_chunked.yaml   <run_name>
python infer.py           logs/<run_name>/config.yaml logs/<run_name>/best.pt

# Sliding-window
python train_sliding.py   config/MERT_sliding.yaml   <run_name>
python infer.py           logs/<run_name>/config.yaml logs/<run_name>/best.pt

# Causal streaming
python train_streaming.py config/MERT_streaming.yaml <run_name>
python infer_streaming.py logs/<run_name>/config.yaml logs/<run_name>/best.pt
```

`infer.py` picks the matching extractor from the config (sliding if it has
`hop_duration`/`context_duration`, otherwise offline/chunked). Both evaluators
score the split named by `eval_split` (`val` by default; set `test` for final
numbers) with `mir_eval`, averaging per-song weighted accuracy over songs, and
print `majmin`, `sevenths`, `tetrads`, `mirex` (plus `root`, `triads`).
`infer_streaming.py` options in the config: `stream_chunk_duration` (speed only),
`single_pass: true` (whole song in one call).

Post-processing is set by the config's `decoder:` block (evaluation only; no
retraining needed to change it):

- `none`: per-frame argmax.
- `blocksync`: self-bias CRF decoded once per block of frames the backbone emits
  (`block: auto`). Streamable, no added latency. Default for chunked, sliding-window
  and streaming.
- `offline`: self-bias CRF over the whole track. Not causal. Default for offline.

Optional fields: `self_bias` (CRF self-transition bonus in nats, default 1.0;
higher = fewer chord changes) and `emission_mode` (`logprobs` or `logits`).

Key config fields: `backbone`, `MERT_model`, `emb_dim`, `mert_align_checkpoint`
(distilled backbones), `model_type`, `mert_layer` (`average` or a layer index),
`chunk_duration` / `hop_duration` + `context_duration` (mode-specific),
`augment_pitch` + `pitch_shift_range` (training-only pitch-shift augmentation).

## Files

| File | Purpose |
|---|---|
| `train_{offline,chunked,sliding,streaming}.py` | training scripts |
| `infer.py`, `infer_streaming.py` | evaluation / `.lab` prediction |
| `models.py` | probe heads, backbone loaders, CRF decoders |
| `chord_dataloader.py`, `chord_vocab.py`, `chord_augment.py` | data, vocabulary, pitch-shift augmentation |
| `sweep_crf.py`, `sweep_self_bias.py` | CRF post-processing sweeps |
