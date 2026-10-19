# Key Detection

Global key detection (24 classes: 12 pitch classes x major/minor) with probes on
frozen MERT backbones. The backbone produces frame embeddings and a small
trainable head pools them into one key prediction. Every head is prefix-causal,
so besides the whole-track prediction it also gives a *running* prediction after
each amount of audio. That lets you measure how fast the key is found and how
stable it stays.

## Quick Start

Download the released checkpoints to `mustream/ckpt/` (see `mustream/README.md`), then run a
probe on your own audio; here the 95M backbone, `r15`:

```bash
cd key_detection
python infer_streaming.py ../ckpt/pytorch/key/streaming_95M_r15/config.yaml \
                          ../ckpt/pytorch/key/streaming_95M_r15/best.pt --audio song.wav
```

This prints the final key and how the running estimate changed over time, and writes
`predictions/song.key` (the final key) and `predictions/song.key_running.tsv` (`time key`
after each block of audio); `--out-dir` changes the folder.

Any of `streaming_{26M,95M,330M}_{r0,r15}` works the same way, and `--audio` takes
several files (any format torchaudio reads; audio is resampled to 24 kHz mono).

Without `--audio`, the script evaluates on your own labelled split instead (`--split`):
set `audio_dir`, `annotation_dir` and `split_dir` in that `config.yaml` first.

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
the offline, chunked and sliding-window setups. MuSTream-26M and the 26M bidirectional student have no published
weights, so they always need `mert_align_checkpoint`.

## Probe heads

`model_type`: `MLP` (cumulative-mean pooling followed by an MLP),
`LSTM`, or `Transformer` (causal). `train_readout: final` (default) trains on the
segment prediction only; `running` trains on every prefix prediction.

## Data

The datasets and splits used in our experiments are not distributed.

> **Note:** create `key_detection/data_splits/splits/` with your own split
> files (or point `split_dir` elsewhere), and set `audio_dir` and `annotation_dir`
> in `config/*.yaml`. The streaming configs already point `mert_align_checkpoint` at the released
> backbones in `ckpt/pytorch/backbone/`.

The code expects pre-built splits in `split_dir` (default `data_splits/splits/`):

- `{train,val,test}.txt`: one track id per line (ids may contain `/`).
- `audio_dir/<id>.<audio_ext>`: the audio (any sample rate; it is resampled to 24 kHz).
- `annotation_dir/<id>.key`: one key per file, e.g. `A# minor` or `Bb minor`.
  The first tab-separated field is used.

Set `audio_dir`, `annotation_dir` and `split_dir` in `config/*.yaml`.

## Experiments

Run every command from `key_detection/`. Training writes `logs/<run_name>/`,
which holds a copy of the config, `best.pt` (best val MIREX weighted score) and
periodic snapshots. To evaluate, pass the run's own `logs/<run_name>/config.yaml`.

### Inference options

- **Offline (bidirectional)**: MERT sees the whole crop, or at evaluation the
  whole track, in one pass. Not causal; it gives one prediction at the end of the
  track and is the upper-bound reference.
- **Chunked**: MERT encodes independent, non-overlapping `chunk_duration`
  chunks. It is bidirectional inside a chunk, so it gives one prediction per
  chunk with up to `chunk_duration` of latency.
- **Sliding-window**: every `hop_duration`, a bidirectional MERT re-encodes the
  last `context_duration` seconds and keeps only the newest hop. This makes it
  causal at hop granularity (latency is one hop), but each sample is encoded
  `context_duration / hop_duration` times, so it is expensive.
- **Causal streaming**: a causal MERT with a rolling KV cache
  (`streaming_context` frames) that is fed chunk by chunk in one session. Memory
  is O(context), and you get a prediction for every frame (or every 15-frame
  block for the block-causal MuSTream backbones). Chunk size only affects speed.

### Train

```bash
python train_offline.py   config/MERT_offline.yaml   <run_name>
python train_chunked.py   config/MERT_chunked.yaml   <run_name>
python train_sliding.py   config/MERT_sliding.yaml   <run_name>
python train_streaming.py config/MERT_streaming.yaml <run_name>
```

### Evaluate

```bash
# offline / chunked / sliding-window (chosen automatically from the config)
python infer.py           logs/<run_name>/config.yaml logs/<run_name>/best.pt
# causal streaming
python infer_streaming.py logs/<run_name>/config.yaml logs/<run_name>/best.pt
```

Optional flags (both scripts):

- `--split {train,val,test}` (default `test`)
- `--settle-tol F` (default 0, strict; `docs/key_detection.md` uses 0.05)
- `--max-duration SEC`
- `--out results.json` (summary plus per-track metrics)
- `--curves curves.npz` (raw running curves)

The output reports:

- **Global metrics**: the MIREX weighted score and accuracy of the whole-track
  prediction.
- **Convergence profile**: TAWS (time-averaged weighted score), `t_first_correct`,
  `t_settle` / `p_settle`, `flip_rate`, and `W(t)` / `Acc(t)` at fixed prefix
  lengths. Time is measured as the seconds of audio each prediction has consumed.

### Key config fields

- **Backbone block**: each config has commented A/B/C blocks (330M / 95M /
  26M). Switch `backbone`, `MERT_model`, `emb_dim` and `mert_align_checkpoint`
  together. `emb_dim` must equal the backbone's hidden size.
- **`mert_align_checkpoint`**: the distilled weights. For `streaming` /
  `small_streaming` it is a `train_mert_causal_distill.py` checkpoint (the released
  MuSTream backbones are in `ckpt/pytorch/backbone/`); for the 26M
  `small_bidirectional` student it is a `train_mert_size_distill.py` checkpoint,
  which is not released. It is required for the 26M backbones. For MuSTream-330M/95M
  it is optional; `null` uses the published MERT weights remapped into the streaming
  architecture.
- **Streaming only**:
  - `streaming_context` (KV window in raw 75 fps frames) must match the
    checkpoint's distillation run (750 for the released backbones). A mismatch
    prints a warning if the run's `args.yml` is next to the checkpoint.
  - `backbone_positional_embedding` must match the checkpoint. Use `rope`.
  - `stream_chunk_duration` must be a whole number of 15-frame (0.2 s) blocks
    for the block-causal MuSTream backbones. `single_pass: true` runs the whole track in one forward pass
    instead.
- **Setup-specific settings**: `chunk_duration` (chunked); `context_duration` and
  `hop_duration` (sliding-window); `mert_batch_size` limits how many chunks or
  windows go to MERT per call.
- **Augmentation**: `augment_pitch` / `pitch_shift_range` add pitch-shift
  augmentation to the training split only, transposing the key label to match.

## Files

```
key_utils.py        key vocabulary, MIREX weighted score, convergence metrics
key_augment.py      pitch-shift augmentation (audio + label)
key_dataloader.py   KeyDataset and the crop/window geometry for each setup
models.py           heads (KeyMLP/KeyLSTM/KeyTransformer), backbone loading, feature extraction
train_{offline,chunked,sliding,streaming}.py   training scripts
infer.py            offline / chunked / sliding-window evaluation
infer_streaming.py  causal streaming evaluation
config/             one config per setup
```

`python key_utils.py`, `python key_augment.py` and `python models.py` run self-tests.
