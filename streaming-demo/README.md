# Real-time Music Understanding Demo

Chord recognition, beat tracking and key detection running for real in the
browser.

## Running the demo

There are two ways to run it. Either way the models run on your own device.

### Option 1: open the hosted page (nothing to install)

**https://yamaha-research-mustream.static.hf.space**

Choose a task, upload an audio file and press **Start streaming**, or press
**Try a sample** to stream the bundled `docs/test.wav` in one click. The first visit
downloads the 26M graphs: about 107 MB for one task, or 218 MB for "All tasks".
They are cached in the browser after that. Add `?model=95M` or `?model=330M` to
the URL for a larger student, which is a much bigger download (about 0.7 GB and
2.5 GB for all tasks).

### Option 2: run it on your own machine

This needs [Node.js](https://nodejs.org) 18 or newer and the Hugging Face CLI
(`pip install -U huggingface_hub`). The page and the graphs are then served from
your own machine, so the demo still works offline once installed. 

```bash
git clone https://github.com/yamaha-research/MuSTream.git && cd MuSTream

# The 26M ONNX graphs only (~235 MB); --include "onnx/*" gets every model (~3.5 GB).
hf download yamaha-research/MuSTream --include "onnx/*/streaming_26M_*" --local-dir mustream/ckpt

cd streaming-demo
npm install
npm run dev            # links the graphs into public/models, then serves
```

Open the address `npm run dev` prints (http://localhost:5173 by default). To use
a larger student, download its graphs (`--include "onnx/*/streaming_95M_*"`) and
switch with `npm run link-models 95M`; see [Choosing a model](#choosing-a-model).

## What runs

| Task | Head | Backbone | Emits | Status |
| --- | --- | --- | --- | --- |
| Chord recognition | `chord_probe` | `r0` | 3 labels per chunk, 15 fps | real inference |
| Beat tracking | `beat_probe` | `r15` | 15 (beat, downbeat) logits per chunk, 75 fps | real inference, bar-pointer DBN |
| Key detection | `key_probe` | `r0` | one running estimate per chunk | real inference |

Every page works the same way: upload audio, press **Start streaming**, and
watch the model's output arrive as the track plays. "All tasks" runs the three
heads over one audio stream.

**Two backbones, not one.** The demo runs the chord and key probes on the `r0`
streaming backbone and the beat probe on `r15` for best results. 

## Choosing a model

```bash
npm run models                     # list the models, marking the linked one
npm run link-models 330M           # switch; reload the page, no restart needed
```

`npm run dev` keeps whatever is linked, so a choice survives a restart. All
three model sizes have all three heads exported.

| Model | Both backbones | Carried state (one task / all) |
| --- | --- | --- |
| 26M | 203 MB | 31 MB / 62 MB |
| 95M | 718 MB | 55 MB / 111 MB |
| 330M | 2457 MB | 147 MB / 295 MB |
