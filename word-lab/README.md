# Browser model comparison

Public URL: https://nipunbatra.github.io/attention/word-lab/

This is a static GitHub Pages application. ONNX Runtime Web 1.30.0 executes
the actual trained FP32 models locally. The automatic mode attempts WebGPU,
then falls back to single-threaded WASM. There is no inference server, prompt
upload, ChatGPT Site dependency, or playback of saved text. The pinned runtime
is downloaded from jsDelivr; this is not an offline application.

## Run locally

From the repository root:

```sh
python3 -m http.server 8778 --bind 127.0.0.1
```

Open http://127.0.0.1:8778/word-lab/ in a current browser. WebGPU requires a
supported browser/device and a secure context (HTTPS or localhost). WASM is
the compatibility option. Each generation recomputes a 64-token window; there
is no KV cache. Long prompts are cropped and short ones are left-padded.

## Models and evidence

- All models use the same word/punctuation tokenizer, 4,000-token vocabulary,
  6,000-story subset, document splits and supervised targets.
- MLP: 64 ordered token embeddings, then a hidden MLP and vocabulary head.
- Attention: one block with either one 64-coordinate head or four 16-coordinate
  heads. Learned absolute position rows are added to token embeddings. The
  position is the slot in the current window, not an unbounded document index.
- Both attention variants retain total width 64 and have the same parameter
  count. They are not complete Transformer stacks.
- `models/comparison.json` records nine fresh training runs, validation-selected
  steps, configuration, source hashes, environment and test results. The earlier
  two-model benchmark is preserved separately in the notebook directory.
- `models/metadata.json` identifies the seed-11 checkpoints actually exported,
  their test scores, file sizes, SHA-256 hashes and CPU parity errors.
- `models/*-parity.json` contains full reference logits for three inputs.
  `check_browser.mjs` compares every browser output against these references.
- `browser-timing.json` records three warm WASM measurements, their raw results,
  browser/device and prompt. These are local measurements, not a promise about
  another computer. The UI measures each new generation on the current device.
- `saved-examples.json` is only for lecture figures and regression tests.
  `app.mjs` never fetches it. Presets were selected without comparing outputs.

Training time includes the training loop and validation checks. It excludes
data preparation and final test scoring. Browser generation time excludes model
downloads/warm-up but includes token selection and UI updates. Per-call time
measures `session.run` only. EOS costs a call without adding a text token.

## Reproduce

From `notebooks/wordlm`, using an environment with `requirements.txt` installed:

```sh
python prepare_data.py --profile dgx --data-dir work/data
python run_head_comparison.py --data-dir work/data
python -m pip install -r requirements-export.txt
python export_browser_lab.py --data-dir work/data
python build_head_lesson.py
```

Execute notebook 6 before exporting its HTML and the download ZIP. The
`export_bundle` function in `build_slow_lesson.py` performs that export without
rebuilding the earlier source notebooks. After measuring browser timings,
`python build_head_lesson.py --slides-only` refreshes the ending of Part II.

From the repository root, with a local server running:

```sh
node word-lab/check_core.mjs
PLAYWRIGHT_PATH=/path/to/playwright node word-lab/check_browser.mjs
PLAYWRIGHT_PATH=/path/to/playwright node word-lab/check_webgpu.mjs
```

Set `LAB_URL=https://nipunbatra.github.io/attention/word-lab/` to test the public
deployment without overwriting the recorded local timings. Tests create screenshots
in a temporary directory and print its path.
The GPU check records adapter availability, full-logit parity and automatic-runtime
generation in `webgpu-check.json`. GPU timing there is a functional test, not a
multi-run speed benchmark.

## Text attribution

The two training stories and one test story in `examples.json` are unchanged
TinyStories documents by Ronen Eldan and Yuanzhi Li, from the pinned source
revision in `examples.json`, under [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/).
Prompts are tokenized prefixes, and display labels are ours. The three authored
prompts are marked separately. The models can produce repetitions, unsupported
claims and poor continuations; this is a small classroom experiment.
