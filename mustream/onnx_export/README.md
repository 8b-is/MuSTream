# ONNX export

Export the streaming backbones and LSTM probes to ONNX, and run them without
PyTorch. The released models (`mustream/ckpt/onnx/`, see `mustream/README.md`) were
produced with these scripts and are what `streaming-demo/` runs.

Each graph is a `.onnx` file plus a `.json` spec (inputs, outputs, state tensors,
labels); the same spec is embedded in the ONNX metadata. Streaming graphs take one
200 ms chunk (15 frames at 75 fps) plus the carried state, and return that chunk's
output plus the updated state.

## Export

Run from `mustream/`. Each export checks the ONNX graph against PyTorch
(`--no-check` skips it).

```bash
# backbone: a streaming backbone checkpoint
python onnx_export/export_mert_onnx.py ckpt/pytorch/backbone/streaming_95M_r15.ckpt \
    --out ckpt/onnx/backbone/streaming_95M_r15.streaming.onnx

# probes: a probe checkpoint (its config is read from the checkpoint)
python onnx_export/export_chord_probe_onnx.py ckpt/pytorch/chord/streaming_95M_r15/best.pt \
    --decoder blocksync --out ckpt/onnx/chord/streaming_95M_r15.streaming.onnx
python onnx_export/export_beat_probe_onnx.py ckpt/pytorch/beat/streaming_95M_r15/best.pt \
    --out ckpt/onnx/beat/streaming_95M_r15.streaming.onnx
python onnx_export/export_key_probe_onnx.py ckpt/pytorch/key/streaming_95M_r15/best.pt \
    --out ckpt/onnx/key/streaming_95M_r15.streaming.onnx
```

- `--decoder blocksync` (chord only) puts the self-bias CRF in the graph, which then
  also outputs `tags`; `--self-bias` sets its strength (default 1.0). `none` outputs
  logits only.
- `--chunk-frames` changes the chunk size (raw 75 fps frames; a multiple of the
  backbone's 15-frame lookahead block).
- `--context` (backbone) changes the KV-cache window; the default is the trained
  `streaming_context`.

## Run

Pair each probe with the backbone it was trained with (same `streaming_<size>_<r>`).

```bash
O=ckpt/onnx
python onnx_export/run_chord_onnx.py song.wav --backbone $O/backbone/streaming_95M_r15.streaming.onnx \
    --probe $O/chord/streaming_95M_r15.streaming.onnx --lab out/
python onnx_export/run_beat_onnx.py song.wav --backbone $O/backbone/streaming_95M_r15.streaming.onnx \
    --probe $O/beat/streaming_95M_r15.streaming.onnx --postproc dbn_blocksync --beats out/
python onnx_export/run_key_onnx.py song.wav --backbone $O/backbone/streaming_95M_r15.streaming.onnx \
    --probe $O/key/streaming_95M_r15.streaming.onnx --curve out/
```

- Chord writes `<name>.lab`; beat writes `<name>.beats`; key prints the final key and
  (`--curve`) writes the running estimate.
- Beat `--postproc`: `minimal` (peak picking, no extra dependencies), `dbn_blocksync`
  (streaming DBN, needs madmom) or `dbn` (offline DBN).
- `--live` prints results chunk by chunk; `--threads` / `--providers` configure
  onnxruntime.

## Files

| File | Purpose |
|---|---|
| `export_mert_onnx.py` | streaming backbone export |
| `export_{chord,beat,key}_probe_onnx.py` | probe export |
| `run_{chord,beat,key}_onnx.py` | run a backbone + probe pair on audio files |
| `onnx_stream.py` | shared streaming driver for the runners |
