# MuSTream

MuSTream is a causal, streaming music representation model: a bidirectional MERT
distilled into a causal model with a KV cache, which processes audio chunk by chunk
in real time. This folder contains the MuSTream backbones and music-understanding
probes on top of them (chord recognition, key detection, beat tracking).

## Installation

The environment is managed with [uv](https://docs.astral.sh/uv/) (Python 3.10). From
the repository root:

```bash
uv sync                       # creates .venv from pyproject.toml / uv.lock
source .venv/bin/activate
cd mustream
```

All commands below are run from this folder (`mustream/`) with the environment
activated. PyTorch is installed with CUDA 11.8 by default; for another CUDA version
or CPU, change the `pytorch-cu118` index in `../pyproject.toml` and run `uv lock`.
The browser demo (`../streaming-demo`) needs Node.js instead; see its README.

## Pretrained checkpoints

The released checkpoints are on Hugging Face: <https://huggingface.co/yamaha-research/MuSTream>.
Download them into `mustream/ckpt/`:

```bash
hf download yamaha-research/MuSTream --local-dir ckpt
```

```
ckpt/
├── pytorch/
│   ├── backbone/                   # MuSTream backbones
│   │   └── streaming_{26M,95M,330M}_{r0,r15}.ckpt
│   └── {chord,key,beat}/           # LSTM probes, one folder per backbone
│       └── streaming_{26M,95M,330M}_{r0,r15}/{best.pt, config.yaml}
└── onnx/                           # the same models in ONNX (../streaming-demo, onnx_export/)
    ├── backbone/
    │   └── streaming_{26M,95M,330M}_{r0,r15}.streaming.{onnx,json}
    └── {chord,key,beat}/
        └── streaming_{26M,95M,330M}_{r0,r15}.streaming.{onnx,json}
```

- **Backbones.** MuSTream at three sizes: MuSTream-26M (a distilled student) and
  MuSTream-95M / MuSTream-330M (distilled from MERT-95M / MERT-330M). Each frame attends to the past
  plus the rest of its 15-frame (0.2 s) block, over a 750-frame (10 s) KV cache.
- **`r0` / `r15`.** The local-interval teacher used in distillation sees 0 or 15
  frames past each supervised block (`teacher_predict_frames`).
- **Probes.** Each `config.yaml` already points `mert_align_checkpoint` at its
  backbone (`../ckpt/pytorch/backbone/...`, relative to the task folder), so they run as
  downloaded. Each task README starts with a Quick Start on your own audio file:

  ```bash
  cd chord_recognition
  python infer_streaming.py ../ckpt/pytorch/chord/streaming_95M_r15/config.yaml \
                            ../ckpt/pytorch/chord/streaming_95M_r15/best.pt --audio song.wav
  ```

## 1. Distillation data (MTG-Jamendo)

The backbones are distilled on [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset)
audio only (no labels). Download the full-length audio following the instructions in
that repository, e.g. into `path/to/MTG-Jamendo`, then prepare it in three steps
(each script takes `--root` / `--out` to override its default paths):

```bash
# 1. length and sample rate of every readable file -> data_splits/mtg_jamendo/audio_info.pkl
python data_processing/collect_audio_info.py --root path/to/MTG-Jamendo --workers 16

# 2. train / val / test split by song (90 / 5 / 5) -> data_splits/mtg_jamendo/{train,val,test}.pkl
python data_processing/create_splits.py --root path/to/MTG-Jamendo

# 3. resample to 24 kHz mono and write 256 HDF5 shards (resumable)
python data_processing/collect_audio_wav.py --root path/to/MTG-Jamendo \
    --out path/to/MTG-Jamendo-h5/MTG-Jamendo --sr 24000 --shards 256
```

The splits used in our experiments are included in `data_splits/mtg_jamendo/`, so you
can skip steps 1 and 2: by default they overwrite those files (pass `--out` to write
elsewhere). Step 2 reproduces them when the same files are readable (fixed seed 42). In both distillation configs, set `h5_root` to
the `--out` prefix of step 3 (`train_split` / `valid_split` already point at
`data_splits/mtg_jamendo/`).

## 2. Training the backbones from scratch

Training logs to [Weights & Biases](https://wandb.ai): run `wandb login` first, or set
`WANDB_MODE=offline` (or `disabled`). Each run keeps one checkpoint, the one with the
best validation loss, as `<save_dir>/step=<N>.ckpt`.


### 2a. MuSTream-95M / MuSTream-330M (causal distillation)

The published MERT is the frozen teacher; the student is MuSTream, the same MERT in
the causal streaming architecture, initialized from the teacher's weights.

```bash
python train_mert_causal_distill.py --args.load=config/mert_causal_distill.yml \
    --mert_model m-a-p/MERT-v1-95M --batch_size 128 --teacher_predict_frames 15 \
    --save_dir ckpt/runs/streaming_95M_r15

python train_mert_causal_distill.py --args.load=config/mert_causal_distill.yml \
    --mert_model m-a-p/MERT-v1-330M --batch_size 64 --teacher_predict_frames 15 \
    --save_dir ckpt/runs/streaming_330M_r15
```

Use `--teacher_predict_frames 0` for the `r0` variants. Other key settings (defaults
in the config): `student_lookahead_block: 15`, `streaming_context: 750`,
`teacher_window_frames: 375`, `teacher_supervise_frames: 15`,
`student_positional_embedding: rope`, 100k steps.

### 2b. MuSTream-26M (two stages)

1. **Size distillation** -- MERT-95M -> a 26M bidirectional student
   (`modeling/mert/small_mert.py`, randomly initialized):

   ```bash
   python train_mert_size_distill.py --args.load=config/mert_size_distill.yml \
       --save_dir ckpt/runs/bidirectional_26M
   ```

2. **Causal distillation** -- the 26M bidirectional student (frozen) -> MuSTream-26M,
   warm-started from its weights:

   ```bash
   python train_mert_causal_distill.py --args.load=config/mert_causal_distill.yml \
       --mert_model m-a-p/MERT-v1-95M --student_model small_streaming --batch_size 128 \
       --teacher_checkpoint ckpt/runs/bidirectional_26M/step=<N>.ckpt \
       --teacher_predict_frames 15 --save_dir ckpt/runs/streaming_26M_r15
   ```

`mert_model` stays `m-a-p/MERT-v1-95M` here: it supplies the configuration the 26M
student inherits.

Optionally, both stages can probe the student on chord / beat data every
`eval_interval` steps (`MERTChordEvaluator.*`, `MERTBeatEvaluator.*`, see
`evaluate_mert_distill.py`). This is off by default (`enable: false`); it needs your
own labelled data (`split_dir`, `audio_dir`, `annotation_dir`).

A trained run can be used directly as a probe backbone: set `mert_align_checkpoint`
in a task config to its checkpoint, e.g.
`../ckpt/runs/streaming_95M_r15/step=<N>.ckpt` (relative to the task folder).

## 3. Training the understanding probes

Each task has its own folder and README with the data format, the inference modes
(offline, chunked, sliding-window, causal streaming), and the training and evaluation
commands. The datasets used in our experiments are not distributed; the READMEs
describe the expected split layout.

| Task | Folder | Streaming training |
|---|---|---|
| Chord recognition (121 classes, 15 fps) | [`chord_recognition/`](chord_recognition/README.md) | `python train_streaming.py config/MERT_streaming.yaml <run_name>` |
| Key detection (24 classes, running estimate) | [`key_detection/`](key_detection/README.md) | `python train_streaming.py config/MERT_streaming.yaml <run_name>` |
| Beat / downbeat tracking (75 fps) | [`beat_tracking/`](beat_tracking/README.md) | `python train_streaming.py config/MERT_streaming.yaml <run_name>` |

Probes train on a frozen backbone (`backbone`, `MERT_model`, `emb_dim`,
`mert_align_checkpoint` in each config) and write `logs/<run_name>/{best.pt,config.yaml}`
inside the task folder.

## 4. ONNX export

`onnx_export/` exports a streaming backbone and its probes to ONNX and runs them without
PyTorch; the exported graphs drive the browser demo in `../streaming-demo`. See
[`onnx_export/README.md`](onnx_export/README.md).

## Files

| Path | Description |
|---|---|
| `extract_features.py` | extract MuSTream features from audio with a released (or trained) backbone |
| `train_mert_causal_distill.py` | entry point: causal distillation (bidirectional teacher -> MuSTream) |
| `train_mert_size_distill.py` | entry point: size distillation (MERT-95M -> 26M bidirectional student) |
| `config/` | distillation configs (`mert_causal_distill.yml`, `mert_size_distill.yml`) |
| `lit_module/` | Lightning modules for the two distillations; `distill_targets.py` builds the local-interval teacher targets |
| `base_trainer.py` | Lightning trainer wrapper: checkpointing, resuming, W&B logging |
| `lr_scheduler.py` | linear warm-up + cosine decay schedule |
| `dataloader.py` | MTG-Jamendo audio crops from the HDF5 shards (or raw files) |
| `audio_augment.py` | batched pitch-shift augmentation (chord and key probes) |
| `evaluate_mert_distill.py` | optional chord / beat probe evaluation during distillation |
| `modeling/` | model code: `mert/` (MERT and the small student), `streaming/` (the MuSTream backbone), `moshi/` (streaming modules), loaders (`load_mert.py`, `init_streaming.py`, `load_checkpoint.py`) |
| `data_processing/` | MTG-Jamendo preparation scripts (section 1) |
| `data_splits/mtg_jamendo/` | MTG-Jamendo train / val / test splits |
| `chord_recognition/`, `key_detection/`, `beat_tracking/` | the understanding tasks: probes, data loaders, training and inference scripts, configs |
| `onnx_export/` | ONNX export and ONNX runtime runners |
| `ckpt/` | released checkpoints (downloaded) and your own runs |
