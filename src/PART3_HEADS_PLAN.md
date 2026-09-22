# Part III: multi-head attention

Requested 2026-09-22: make the third lesson a direct continuation of Parts I–II,
not a survey of optimization, normalization and the complete Transformer.

The old Part III stays reproducible as `part2b.html` (optional reference), using
`part2b.json`, the unchanged `sections3/`, `part3.js` and `toy3.json`.
The new Part III uses `sections3-heads/`, `part3-heads.js`, and the Part II toy.

## What the first rewrite got wrong

The initial 58-frame lecture introduced tensor bookkeeping before students could
see why a second head helps. Generic box chains and tables replaced the held
SVG drawings used in Part II. It also reused M for messages even though Part II
already reserved M for the mask, and used H as the head count instead of the
message matrix. Fit tests did not catch those teaching failures.

## Revised visual sequence (45 frames)

1. Start with the river-bank sentence. A separate two-source example computes
   one shared value mixture and two independent mixtures before the full
   ten-token setting/person patterns. One head is not limited to one word.
2. Recall Part II’s Maya query/key/value roles before the matrices. Translate
   those roles to the river prefix. Compute each head’s query on its own slide.
   Work through Head 1 completely: numerical Q/K matrices, the final query
   times Kᵀ, all ten scaled scores and softmax weights, then every weighted
   value contribution. Repeat that same four-frame layout for Head 2 before
   the combined calculation. Keep receiver 10 fixed throughout.
   Repack those exact coordinates into one wide head. Compare the single
   softmax with the two independent rows, without claiming universal superiority.
3. Join the messages, multiply by W_O and add the update to the original e.
   The recurring diagram shows concatenation and projection separately, even
   though joined width and embedding width both equal four in this worksheet.
   Return to the familiar next-token MLP and try the other bank context.
4. Stack the rows: show actual 10×4, 4×2, 10×2 and 10×10 matrix silhouettes.
   Trace the same receiving row. Keep Q/K, the mask, A, V and H identifiable.
5. Write one short head function, call it twice, then show the PyTorch equivalent.
   Explain a bias with a two-number offset and locate it in all four projections.
   Match `bias=False` to the worksheet and distinguish the separate MLP biases.
   Keep batching, packed projections and the full training loop in the notebook.
6. Compare measured trained results and open the real browser demo.

The original detailed steps remain in `figures/multihead/lab-manifest.json` and
Notebook 7, after the new visual story. `multihead_story.py` authors the lecture;
`build_multihead_lesson.py` builds both resources from `head_worksheet.py`.
Notebook 7 pairs each of the eight head-arithmetic figures with code that
reproduces its matrices, dot products, normalization or value contributions.

Notation follows Part II: row vectors; e, Δe, e′; A for attention weights;
α_ij for one entry; M for the mask; H=AV for messages. Parenthesized superscripts
label heads. The number of heads is `n_heads`, not H. The main visual story uses
one sequence before the optional lab introduces the batch axis.

Visual teaching references are credited in the article and notebook:
3Blue1Brown’s attention lesson and Jay Alammar’s Illustrated Transformer.
The river-bank diagrams and numerical example are our own. We deliberately
retain Part II’s row-vector convention rather than import 3Blue1Brown’s columns.

The introductory two-source calculation is labelled as a separate illustration.
The main lesson has two sources of numbers: a hand-chosen two-head worksheet
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
