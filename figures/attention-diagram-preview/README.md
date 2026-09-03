# Incremental attention diagram preview

Open `index.html` to step through the diagram. It is standalone and works offline; no build or third-party dependencies are required.

- 12 cumulative reveals; boxes never move between stages.
- Stages 1–11 follow receiver 7 (`bank`) in the ten-token river sentence.
- Stage 12 recomputes the same path for receiver 10 (the final `the`) and adds the vocabulary predictor outside the attention core.
- All projections in a layer use the same original input snapshot. The update of `bank` is not fed into the final `the` within the same layer.
- Numerical data is frozen from `src/toy.json`, not trained. This snapshot has equal `W_K` and `W_V`, which is disclosed on the Values stage.
- Uses row vectors; `E` is the matrix of current input representations, not the shared vocabulary embedding table.
- This is one attention head with an output projection and residual addition, not a full Transformer block. FFN, normalization, training and multiple heads are omitted.

## Deliverables

- `index.html`: interactive preview (arrow keys, autoplay, all-stages overview, current-SVG download).
- `stages/*.svg`: standalone editable vector stages.
- `all-stages.svg`: contact sheet of all reveals.
- `diagram.js`: SVG authoring and stage descriptions.
- `toy-data.js`: frozen numerical input generated from the existing toy.

To regenerate SVG exports after a preview edit:

```sh
node figures/attention-diagram-preview/export.mjs
```

The original `attention.html` and its source sections are not modified by this preview.

Diagram structure adapted from Figure 2 of Vaswani et al., *Attention Is All You Need* (https://arxiv.org/abs/1706.03762); the projections, residual path, token labels and numerical walkthrough are contextual additions.
