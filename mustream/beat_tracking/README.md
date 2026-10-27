# Beat Tracking

Frame-level beat and downbeat tracking with a frozen MERT backbone. The backbone produces per-frame embeddings (75 fps). A small trainable probe head (`MLP`, `LSTM` or `Transformer`) maps them to two logits per frame (beat, downbeat). A post-processor then turns these into beat and downbeat times, which are scored with `mir_eval`.

## Quick Start

Download the released checkpoints to `mustream/ckpt/` (see `mustream/README.md`), then run a
probe on your own audio; here the 95M backbone, `r15`:

```bash
cd beat_tracking
python infer_streaming.py ../ckpt/pytorch/beat/streaming_95M_r15/config.yaml \
                          ../ckpt/pytorch/beat/streaming_95M_r15/best.pt --audio song.wav
```

This writes `predictions/song.beats` (`time is_downbeat` per beat, with `is_downbeat` 1 or 0;
`--out-dir` changes the folder), post-processed with the config's `postproc_type`
(`dbn_blocksync`).

Any of `streaming_{26M,95M,330M}_{r0,r15}` works the same way, and `--audio` takes
several files (any format torchaudio reads; audio is resampled to 24 kHz mono).

Without `--audio`, the script evaluates on your own labelled split instead: set
`audio_dir`, `annotation_dir` and `split_dir` in that `config.yaml` first.

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

Each config has a backbone section with three blocks (A: 330M, B: 95M, C: 26M). Uncomment exactly one block and switch all of its keys together. MuSTream-26M and the 26M bidirectional student have no published weights, so `mert_align_checkpoint` must point to their checkpoint: a released MuSTream backbone (`ckpt/pytorch/backbone/`) or a `train_mert_causal_distill.py` run for `small_streaming`, a `train_mert_size_distill.py` run for `small_bidirectional`.

## Data

The datasets and splits used in our experiments are not distributed.

> **Note:** create `beat_tracking/data_splits/splits/` with your own split files (or point `split_dir` elsewhere), and set `audio_dir` and `annotation_dir` in `config/*.yaml`. The streaming config already points `mert_align_checkpoint` at the released backbones in `ckpt/pytorch/backbone/`.

The code expects pre-built splits under `data_splits/<name>/`, which `split_dir` points to (the default is `data_splits/splits`). Each split has `train.txt`, `val.txt` and `test.txt`, with one track id per line. If they are missing and `annotation_dir` is a single flat corpus, the dataloader writes them itself, splitting every annotated track by `train_ratio` / `val_ratio` with seed `split_seed`. For each id the code reads:

- `audio_dir/<id>.<audio_ext>`: the audio, resampled on load to 24 kHz mono.
- `annotation_dir/<id>/beat.lab`: two columns, `time  bar_position`. A `bar_position` of 1 marks a downbeat.

Set `audio_dir` and `annotation_dir` in the config to your own paths.

## Inference modes

| Mode | Script | Backbone behaviour | Latency / causality |
|---|---|---|---|
| Offline | `train_offline.py` | Bidirectional MERT runs over the whole crop or track in one pass. | Not causal. This is the reference. |
| Chunked | `train_chunked.py` | Bidirectional MERT runs on independent chunks of `chunk_duration` s. | Each frame sees only its own chunk (latency ≈ one chunk). |
| Sliding-window | `train_sliding.py` | Bidirectional MERT re-encodes the last `context_duration` s every `hop_duration` s, and only the newest hop is kept. | Causal per hop (latency ≈ one hop, plus optional `lookahead`). This mode is expensive: each crop is encoded about `context/hop` times. |
| Causal streaming | `train_streaming.py` | MuSTream, the causal streaming backbone (KV cache, `streaming_context`-frame window) built by causal distillation. | Causal by construction. One forward pass over a crop equals true streaming. |

All probe heads are causal: `LSTM` is unidirectional and `Transformer` uses causal attention. For `transformer_positional_embedding` and `backbone_positional_embedding`, use `rope`.

### Post-processing (`postproc_type`, applied at evaluation only)

- `minimal`: peak-picking. Streamable with 3 frames of lookahead.
- `dbn`: madmom's DBN, decoded offline over the whole track. Gives the best accuracy but cannot stream. Requires `madmom`.
- `dbn_blocksync`: the same DBN, decoded once per block of frames that the backbone emits per forward call (`dbn_block: auto`). Streamable with no added latency. Works with chunked, sliding-window and streaming backbones, but not offline.

Optional settings for `dbn_blocksync`: `dbn_beats_per_bar` (3 or 4) pins the meter, and `dbn_transition_lambda` sets the tempo-change penalty. Changing the post-processor does not require retraining.

## Train and evaluate

Run every command from `beat_tracking/`. The scripts import local modules and use relative `data_splits/` and `logs/` paths. Each run writes `logs/<run_name>/config.yaml`, `best.pt` (the epoch with the best mean val frame F1) and `epoch_<n>_<score>.pt` every `ckpt_interval` epochs.

```bash
cd beat_tracking

# Offline
python train_offline.py config/MERT_offline.yaml <run_name>
python infer.py logs/<run_name>/config.yaml logs/<run_name>/best.pt

# Chunked
python train_chunked.py config/MERT_chunked.yaml <run_name>
python infer.py logs/<run_name>/config.yaml logs/<run_name>/best.pt

# Sliding-window
python train_sliding.py config/MERT_sliding.yaml <run_name>
python infer.py logs/<run_name>/config.yaml logs/<run_name>/best.pt

# Causal streaming
python train_streaming.py config/MERT_streaming.yaml <run_name>
python infer_streaming.py logs/<run_name>/config.yaml logs/<run_name>/best.pt
```

`infer.py` chooses the offline/chunked or sliding-window extractor from the config, based on whether it contains `chunk_duration` or `hop_duration`/`context_duration`. Always evaluate with the run's own `logs/<run_name>/config.yaml`, because the window geometry must match training. To try a different post-processor, edit `postproc_type` (and `dbn_*`) in that file. Both evaluation scripts print mir_eval F-measure, precision, recall, CMLc/CMLt and AMLc/AMLt for beats and downbeats on the test split. `infer_streaming.py` also prints per-dataset results.

Key config fields:

- `backbone`, `MERT_model`, `emb_dim`, and `mert_align_checkpoint` (plus `student_*` for MuSTream-26M): see the backbone table above.
- `mert_layer`: `average`, or an int layer index.
- `model_type`: `MLP`, `LSTM` or `Transformer`.
- `pos_weight_beat` and `pos_weight_downbeat`: positive-class weights for the BCE loss.
- Mode-specific settings:
  - Chunked: `chunk_duration`.
  - Sliding-window: `hop_duration`, `context_duration`, `lookahead`, and `mert_batch_size` (lower it on OOM).
  - Streaming: `streaming_context`.
- `infer_streaming.py` only:
  - `stream_chunk_duration` or `stream_chunk_frames` (per-call work; results don't depend on valid chunk size).
  - `single_pass: true` runs each track as one forward call.

Chunk-by-chunk extraction bounds the backbone KV cache, but evaluation retains
the whole waveform and full-track features. Total memory therefore grows with
track length even when `streaming_context` and chunk size stay fixed.

### Comparing post-processors on one checkpoint

```bash
python sweep_postproc.py logs/<run_name>/config.yaml logs/<run_name>/best.pt \
    --blocks 1,auto --cache logits_<run_name>.npz
```

This prints the accuracy and measured per-beat emission latency for `minimal`, offline `dbn`, and `dbn_blocksync` at each block. The model forward pass is cached, so later runs skip it. `python postprocessor.py` runs the decoder self-test, which requires `madmom`.

## Files

- `train_{offline,chunked,sliding,streaming}.py`: training, one script per inference mode.
- `infer.py` and `infer_streaming.py`: evaluation.
- `sweep_postproc.py`: post-processor comparison.
- `models.py`: backbone loading, feature extraction, probe heads, loss and metrics.
- `postprocessor.py`: `minimal`, DBN and block-synchronous DBN decoders.
- `beat_dataloader.py` and `data_utils.py`: the dataset.
- `config/`: one config per mode.
