# Incremental attention diagram preview

Open `index.html` to step through the diagram. It is standalone and works offline; no build or third-party dependencies are required.

- 12 cumulative reveals; boxes never move between stages.
- Stages 1–11 follow receiver 7 (`bank`) in the ten-token river sentence.
- Stage 12 recomputes the same path for receiver 10 (the final `the`) and adds the vocabulary predictor outside the attention core.
- All projections in a layer use the same original input snapshot. The update of `bank` is not fed into the final `the` within the same layer.
- Numerical data is frozen from `src/toy.json`, not trained. The current model uses five representation coordinates, three query/key coordinates, and two value coordinates. Shapes and labels are read from the model.
- Uses row vectors; `E` is the matrix of current input representations, not the shared vocabulary embedding table.
- This is one attention head with an output projection and residual addition, not a full Transformer block. FFN, normalization, training and multiple heads are omitted.

## Deliverables

- `index.html`: interactive preview (arrow keys, autoplay, all-stages overview, current-SVG download).
- `stages/*.svg`: standalone editable vector stages.
- `all-stages.svg`: contact sheet of all reveals.
- `diagram.js`: SVG authoring and stage descriptions.
- `toy-data.js`: frozen numerical input generated from the existing toy.

To refresh the numerical snapshot and SVG exports:

```sh
node figures/attention-diagram-preview/sync-data.mjs
node figures/attention-diagram-preview/check-data.mjs
node figures/attention-diagram-preview/export.mjs
node figures/attention-diagram-preview/check-preview.mjs
```

Section 16 of the article uses the same SVG authoring source. Its data adapter reads the article's live `AT.forward` result instead of the frozen preview snapshot. Reading mode shows the arithmetic panel; classroom mode gives the fixed diagram more space and uses larger explanatory text.

Diagram structure adapted from Figure 2 of Vaswani et al., *Attention Is All You Need* (https://arxiv.org/abs/1706.03762); the projections, residual path, token labels and numerical walkthrough are contextual additions.
