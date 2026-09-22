# Part III: multi-head attention

Requested 2026-09-22: make the third lesson a direct continuation of Parts I–II,
not a survey of optimization, normalization and the complete Transformer.

The old Part III stays reproducible as `part2b.html` (optional reference), using
`part2b.json`, the unchanged `sections3/`, `part3.js` and `toy3.json`.
The new Part III uses `sections3-heads/`, `part3-heads.js`, and the Part II toy.

Teaching sequence:

1. The same name prediction and river-bank sentence; what could two heads read?
2. Keep the full input row; learn separate Q/K/V projections.
3. Work both score rows, softmaxes and value messages numerically.
4. Concatenate messages, apply W_O, add the residual and predict one next token.
5. Keep the complete map in view while adding shapes and short executable code.
6. Train and generate with the same loss/loop as before.
7. Replace only the attention calculation with `nn.MultiheadAttention`, and
   check numerical equivalence after copying the weights.
8. Inspect the genuine three-model experiment and browser demo.

Two separate sources of numbers: an explicitly hand-chosen two-head worksheet
using Part II's exact embeddings/positions, and trained four-head TinyStories
results. Neither implies that heads are assigned semantic roles in training.
Do not promise that every extra head or every continuation improves.

Keep Part IV about cross-attention. LayerNorm, full block FFNs, depth, KV-cache
cost and long complexity derivations are optional Part 2B material, not required
steps in the new Part III walkthrough.

## Build and check

Use the notebook environment (PyTorch, nbformat, nbconvert) for the generators:

```sh
python src/build_multihead_lesson.py
python notebooks/wordlm/build_head_walkthrough.py
python src/assemble.py --part 3 --out part3.html
python src/assemble.py --part 3 --config src/part2b.json --out part2b.html
```

Execute Notebook 7 from `notebooks/wordlm/`, then call
`build_slow_lesson.export_bundle(ROOT)` there to rebuild its HTML and the ZIP.
Do not rerun the training benchmark to regenerate teaching figures.
The canonical scratch model is `examples/multihead_from_scratch.py`; the notebook
generator copies it into the downloadable directory. Edit the canonical file.

`check_multihead_lesson.mjs` checks independent numbers, live controls, SVG label
bounds, notebook execution, identical figures, local links and mobile fit.
`tests/test_scratch_multihead.py` checks PyTorch parity, causality, padding,
gradients, parameter counts and every worksheet output.
The older `check_part3_continuity.mjs` now targets Part 2B, as do the cost and
position-capacity checks. The original numerical model and sections are intact.
