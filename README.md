# From characters to Transformers

Three interactive, offline-capable lessons for a deep-learning course. [Open the series](https://nipunbatra.github.io/attention/).

- [Part 1](https://nipunbatra.github.io/attention/part1.html): characters, embeddings, an MLP, next-token probabilities, training, and generation.
- [Part 2](https://nipunbatra.github.io/attention/attention.html): query, key, and value; causal self-attention; a contextual update; next-token prediction.
- [Part 3](https://nipunbatra.github.io/attention/part3.html): loss and learning, multiple heads, Transformer blocks, and autoregressive generation.

Build sources, plans and the handover guide are in `src/` (start with `src/HANDOVER.md`).

## Read, present, or inspect the diagram

Open any part for the article. Press **P** or choose **Present** for the classroom view; use **←/→** to move through frames and interactive steps, **O** for the overview, **S** for notes, and **Esc** to return to reading. Presenter view opens a separate notes window.

The sources are slide-first: one bounded teaching idea per 16:9 frame, with large classroom type and no internal scrollbars. Reading mode unfolds those same frames and their companion explanations into a responsive article. There is one source, one set of widgets, and one numerical model per part, not a second deck to keep synchronized.

Part 1 uses a trained small name model. Part 2 uses hand-designed weights so every step can be inspected; Part 3 trains that toy before introducing the larger Transformer architecture. The toy's position coordinate is added at the same width, but its initial projections ignore it: the example demonstrates content routing, not sensitivity to word order. The text marks this boundary explicitly.

The standalone SVG preview is at `figures/attention-diagram-preview/index.html`. Its twelve stages build one causal attention head, then the output projection, residual addition, and the final-token vocabulary prediction. Section16 embeds the same diagram source and reads the article's live numerical model.

```sh
python3 src/assemble.py --part 1 --out part1.html
python3 src/assemble.py --part 2 --out attention.html
python3 src/assemble.py --part 3 --out part3.html
python3 -m http.server 8776 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8776/attention.html?present#s16/1/0` for the staged diagram in classroom mode. Reading-mode anchors such as `#s16` do not force presentation mode.

## Export the slide views to PDF

```sh
# One fully revealed 16:9 page per frame
node src/export_slides.mjs attention.html output/pdf/attention-part2-slides.pdf

# Every build and managed interactive step becomes a page
node src/export_slides.mjs attention.html output/pdf/attention-part2-builds.pdf --builds all
```

The exporter checks every frame for clipping before writing the PDF. It captures the exact classroom stage with navigation removed, at 2× resolution by default (`--scale 1|2|3`). These appearance-faithful PDFs use images, so their text is not selectable. Browser Print remains available for a reading-oriented handout.

Exported PDFs in `output/pdf/` are local build artifacts, not committed files. Managed builds are advanced automatically; free-form quiz, slider, and disclosure state stays at its default.

## Checks

```sh
node src/check_part1.mjs
node src/toy_ref.mjs src/toy.json --compare src/py_check.json
node src/check-live-model.mjs attention.html
python3 src/check_training.py
node src/check_position_capacity.mjs
node src/pres_test.mjs
node src/frame_audit.mjs part1.html
node src/frame_audit.mjs attention.html
node src/frame_audit.mjs part3.html
node src/check-routing-scaling.mjs
node src/check-diagram.mjs attention.html
node src/qa.mjs attention.html --width 1280 --height 720
node src/qa.mjs attention.html --width 390 --height 844
```

Browser checks reuse an installed Playwright runtime; they do not add a production dependency. See `src/PRESENT.md` for authoring, frame-fit validation, presenter controls, and PDF options. `src/CLASSROOM_QA.md` records the release checks.
