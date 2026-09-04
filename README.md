# Attention, vision and language

Two connected, interactive, offline-capable series for a deep-learning course. [Open the lessons](https://nipunbatra.github.io/attention/).

- [Part 1](https://nipunbatra.github.io/attention/part1.html): characters, embeddings, an MLP, next-token probabilities, training, and generation.
- [Part 2](https://nipunbatra.github.io/attention/attention.html): query, key, and value; causal self-attention; a contextual update; next-token prediction.
- [Part 3](https://nipunbatra.github.io/attention/part3.html): loss and learning, multiple heads, Transformer blocks, and autoregressive generation.
- [Part 4](https://nipunbatra.github.io/attention/part4.html): cross-attention through English-to-French translation, from source encoding to prediction, training, and generation.

The four-part **Vision to language** sequence continues with:

- [Vision I](https://nipunbatra.github.io/attention/vision1.html): one image followed through patch projection, CLS attention, a class prediction, loss, and actual learning on two images. Includes a pixel-by-pixel projection and a value-only experiment.
- [Vision II](https://nipunbatra.github.io/attention/vision2.html): visual pretraining through MAE, DINO, and I-JEPA; exact reconstruction and teacher-distribution worksheets.
- [Vision III](https://nipunbatra.github.io/attention/vision3.html): CLIP-style image–text matching, symmetric contrastive learning, candidate classification, and retrieval.
- [Vision IV](https://nipunbatra.github.io/attention/vision4.html): a visual connector, an image-conditioned prefix decoder, actual answer generation, training, and grounding checks.

Build sources, plans and the handover guide are in `src/` (start with `src/HANDOVER.md`).

## Read, present, or inspect the diagram

Open any part for the article. Press **P** or choose **Present** for the classroom view; use **←/→** to move through frames and interactive steps, **O** for the overview, and **S** for notes. There is no permanent slide header or footer. Click **Controls** or press **C** for navigation and Presenter view. **Esc** closes an open panel, then returns to reading.

A focused slider keeps its native arrow and Page Up/Down controls. Press **N** to advance the presentation without changing the slider, or **Esc** to return focus to the slide before using arrows.

The sources are slide-first: one bounded teaching idea per 16:9 frame, with large classroom type and no internal scrollbars. Reading mode unfolds those same frames and their companion explanations into a responsive article. There is one source, one set of widgets, and one numerical model per part, not a second deck to keep synchronized.

Part 1 uses a trained small name model. Part 2 uses hand-designed weights so every step can be inspected; Part 3 trains that toy before introducing the larger Transformer architecture. The toy's position coordinate is added at the same width, but its initial projections ignore it: the example demonstrates content routing, not sensitivity to word order. The text marks this boundary explicitly.

The standalone SVG preview is at `figures/attention-diagram-preview/index.html`. Its twelve stages build one causal attention head, then the output projection, residual addition, and the final-token vocabulary prediction. Section16 embeds the same diagram source and reads the article's live numerical model.

Part 1 also has four model-backed diagrams: the trained embedding space, lookup-to-concatenation wiring, the forward/backward learning graph, and the shared model inside training versus generation. They build incrementally in class and stay interactive in the article. Every displayed coordinate, probability, and sampled character comes from the same bundled name model.

Part 4 uses a separate three-coordinate model fitted to two phrase pairs: “the river bank” → “la rive”, and “the financial bank” → “la banque”. It includes an encoder, masked target self-attention, cross-attention, residual additions, and a vocabulary head. The numerical toy omits FFNs and LayerNorm; it is not evidence of general translation ability. Source and target positions are learned, added vectors. The diagrams and calculations read the same model. An independent NumPy reference reproduces training and checks every scalar gradient.

The vision sequence adapts the earlier `vision-transformer`, `vision-ssl`, `clip-zero-shot`, and `vlm` articles into this shared article/classroom system. Vision I keeps one block-counting question through the forward and learning passes. Its initial 4×4-image encoder is also the frozen snapshot used in Vision IV; Vision I's later training experiment does not replace that snapshot. Vision II's calculators illustrate objectives, not pretrained-model outputs. Vision I and IV fit only two training images, and Vision III fits three image–caption pairs. These exercises demonstrate the computations, not general counting or zero-shot transfer. See `src/VISION_SOURCE_AUDIT.md` for provenance and changes.

```sh
python3 src/assemble.py --part 1 --out part1.html
python3 src/assemble.py --part 2 --out attention.html
python3 src/assemble.py --part 3 --out part3.html
python3 src/assemble.py --part 4 --out part4.html
# Internal source IDs 5–8 display as Vision Parts I–IV.
python3 src/assemble.py --part 5 --out vision1.html
python3 src/assemble.py --part 6 --out vision2.html
python3 src/assemble.py --part 7 --out vision3.html
python3 src/assemble.py --part 8 --out vision4.html
python3 -m http.server 8776 --bind 127.0.0.1
```

One build per part is enough, even in a clean output directory. Navigation availability comes from complete lesson sources and configs, not from which HTML file happens to exist first. Build and distribute all eight pages together for working offline series links. A planned config can set `"published": false`; a specific navigation entry can set `"available": false`. Undeclared targets stay unavailable even if an old placeholder HTML exists.

Then open `http://127.0.0.1:8776/attention.html?present#s16/1/0` for the staged diagram in classroom mode. Reading-mode anchors such as `#s16` do not force presentation mode.

## Export the slide views to PDF

```sh
# One fully revealed 16:9 page per frame
node src/export_slides.mjs attention.html output/pdf/attention-part2-slides.pdf

# Every build and managed interactive step becomes a page
node src/export_slides.mjs attention.html output/pdf/attention-part2-builds.pdf --builds all

# The same exporter works for the vision series.
node src/export_slides.mjs vision1.html output/pdf/vision-part1-slides.pdf
```

The exporter checks every frame for clipping before writing the PDF. Reveal/quiz answers are shown on completed frames by default, since PDF readers cannot click them; use `--answers authored` for a question handout. It captures the exact classroom stage with navigation removed, at 2× resolution by default (`--scale 1|2|3`). These appearance-faithful PDFs use images, so their text is not selectable. Browser Print also opens answer panels for a reading-oriented handout.

Exported PDFs in `output/pdf/` are local build artifacts, not committed files. Managed builds are advanced automatically and answer reveals are opened; sliders and other manual controls stay at their defaults.

## Checks

```sh
python3 src/check_metadata.py
node src/check_part1.mjs
node src/check_part1_diagrams.mjs
python3 src/train_part4.py --check
node src/check_part4.mjs
node src/check_vision1.mjs
python3 src/verify_vision1_learning.py
node src/check_vision2.mjs
python3 src/train_vision3.py --check
node src/check_vision3.mjs
python3 src/train_vision4.py --check
node src/check_vision4.mjs
node src/check_vision_pixels.mjs vision1.html vision2.html vision3.html vision4.html
node src/toy_ref.mjs src/toy.json --compare src/py_check.json
node src/check-live-model.mjs attention.html
python3 src/check_training.py
node src/check_position_capacity.mjs
node src/pres_test.mjs
node src/interaction_test.mjs
node src/export_test.mjs
node src/frame_audit.mjs part1.html
node src/frame_audit.mjs attention.html
node src/frame_audit.mjs part3.html
node src/frame_audit.mjs part4.html
node src/frame_audit.mjs vision1.html
node src/frame_audit.mjs vision2.html
node src/frame_audit.mjs vision3.html
node src/frame_audit.mjs vision4.html
node src/check_tables.mjs part1.html attention.html part3.html part4.html
node src/check_tables.mjs vision1.html vision2.html vision3.html vision4.html
node src/check-routing-scaling.mjs
node src/check-diagram.mjs attention.html
node src/qa.mjs attention.html --width 1280 --height 720
node src/qa.mjs attention.html --width 390 --height 844
```

Browser checks reuse an installed Playwright runtime; they do not add a production dependency. See `src/PRESENT.md` for authoring, frame-fit validation, presenter controls, and PDF options. `src/CLASSROOM_QA.md` records the release checks.

`check_metadata.py` checks all eight configs against section IDs, headings, roadmap order, landing-card titles, and navigation labels. It also builds each lesson in an isolated temporary directory and tests unavailable planned links, without changing the published HTML. `interaction_test.mjs` exercises open arithmetic dialogs, changing masks, focused sliders, and presenter notes; a default frame walk alone does not cover those states.
