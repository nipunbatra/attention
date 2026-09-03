# Self-attention, from first principles

Interactive single-file lessons for a deep-learning course. Live: https://nipunbatra.github.io/attention/

- `attention.html` Part 2: self-attention for next-token prediction, with reading and classroom modes
- `part1.html`, `part3.html`: planned (see `src/GUIDE1.md`, `src/GUIDE3.md`)

Build sources, plans and the handover guide are in `src/` (start with `src/HANDOVER.md`).

## Read, present, or inspect the diagram

Open `attention.html` for the article. Press **P** or choose **Present** for the classroom view; use **←/→** to move through frames and interactive steps, **O** for the overview, **S** for notes, and **Esc** to return to reading. Presenter view opens a separate notes window. The same section sources power both modes; no duplicate deck is maintained.

The standalone SVG preview is at `figures/attention-diagram-preview/index.html`. Its twelve stages build one causal attention head, then the output projection, residual addition, and the final-token vocabulary prediction. Section16 embeds the same diagram source and reads the article's live numerical model.

```sh
python3 src/assemble.py --part 2 --out attention.html
python3 -m http.server 8776 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8776/attention.html?present#s16/1/0` for the staged diagram in classroom mode. Reading-mode anchors such as `#s16` do not force presentation mode.

## Checks

```sh
node src/toy_ref.mjs src/toy.json --compare src/py_check.json
node src/check-live-model.mjs attention.html
node src/pres_test.mjs
node src/check-routing-scaling.mjs
node src/check-diagram.mjs attention.html
node src/qa.mjs attention.html --width 1280 --height 720
node src/qa.mjs attention.html --width 390 --height 844
```

Browser checks reuse an installed Playwright runtime; they do not add a production dependency. See `src/PRESENT.md` for frame authoring and presenter controls. Parts1 and3 remain planned; their links are marked as unavailable rather than pointing to missing pages.
