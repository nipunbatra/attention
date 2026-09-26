# Part III: return from the arithmetic to the trained models

2026-09-26

The four added frames retain the existing Part II/III diagram language and
leave the worked example, training artifacts and browser inference unchanged.

## Teaching changes

- Return to the full two-head worksheet path after the calculations and code:
  head messages, concatenation, output projection, residual, prediction MLP.
  Reading-mode links revisit each head's calculations and the output projection.
- Trace the actual trained MLP: 64 token rows of width 64, concatenated to
  4,096 coordinates, then 256 hidden units and 4,000 vocabulary logits.
  Concatenation distinguishes slots; this baseline has no position table.
- Trace the actual single-head model: token plus learned position rows, Q/K/V,
  masked score softmax, AV, output projection, residual and final-row readout.
- Keep those landmarks for the four-head model. Each head receives the same E,
  produces 64×16 messages, and joins to 64×64 before the 64×64 output projection.
- Distinguish the full-sequence diagram from the browser's equivalent
  final-query-only inference optimization. The existing measured losses,
  perplexities, parameter counts, training times and live-generation link follow.
  There is no new training or position-free ablation, and no new score claim.

## Verification

- Checked the paths against FixedWindowMLP/CausalAttentionLM in `wordlm.py`,
  MultiHeadAttentionLM in `multihead.py`, and the saved comparison protocol.
- 63 content frames / 64 presentation states pass at 1280×720 and 760×1041.
  All 37 unique formulas parse; no runtime errors or stage overflows.
- Inspected all four new diagrams on desktop and the four-head diagram in
  portrait. All head inputs/outputs visibly branch and join; labels fit boxes.
- Notebook 7 reruns all 44 code cells; the same SVGs are embedded before the
  existing 58-step detailed lab. Regenerated the HTML notebook and ZIP bundle.
- All eight scratch/PyTorch tests pass. The browser regression checks 1,840
  numerical values, model-frame order/shapes, head count, arithmetic links,
  SVG label bounds, live controls, notebook parity and mobile reading fit.

The existing Vision I is being revised in a separate task. This change only
retains the two-frame transition to that image-classification lecture.
