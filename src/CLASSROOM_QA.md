# Part 2: classroom layer and diagram checks

Checked locally on 2026-09-03. This pass improves the existing Part2 article; Parts1 and3 are still planned. The model remains a hand-designed teaching example, not a trained language model.

## What changed

- The article and classroom mode use the same section sources and live numerical model. Long derivations remain available in reading mode; classroom frames emphasize an active question, operation, and result.
- Question-led frames, presenter notes, keyboard/visible navigation, accessible reveal state, overview, fullscreen fallback, and short PyTorch snippets.
- Consistent row-vector equations and clearer separation of token representation, query/key matching coordinates, value payload, message, output projection, and residual update.
- A controlled value-only intervention: muting the finance payload leaves Q, K, scores and attention weights unchanged, but changes the message and update.
- Scaling separates two ideas: a fixed-score divisor demonstration and a seeded variance-growth simulation with stated assumptions.
- Bypassing attention is labelled as no contextual update, not fabricated self-only attention.
- One twelve-stage SVG source powers the standalone preview and section16. It follows bank at position7, then switches to the last token at position10 for next-token prediction. All same-layer projections read the same input snapshot.

## Verified

| Check | Result |
| --- | --- |
| Assembled article versus independent reference | 8 cases; 8,220 finite values and180 masked infinities; max difference1.34×10⁻¹⁵ |
| Reference versus saved Python output | 6,615 values; max difference3.55×10⁻¹⁵; all hard model targets pass |
| Frozen SVG evidence | 307 numerical comparisons; shapes, masks, normalization, causal prefix equivalence, provenance and deep freeze |
| Diagram classroom layout | All12 stages fit1280×720; model-derived dimensions and receiver switch verified |
| Article interaction sweep | 19 sections,128 controls, no errors or invalid displayed values |
| Reading layout | No horizontal document overflow at1280px or390px; no JavaScript/KaTeX errors |
| Presentation runtime suite | Forward/back, nested/manual steppers, inert pending controls, overview focus, notes/presenter, print restore, exit/reload and mobile toolbar tests pass |
| Full assembled classroom walk | 64 frames;195 forward states and194 matching reverse transitions; zero violations |
| Dense-frame visual checks | Major matrix/lifetime/walkthrough frames reviewed at1280×720; optional worksheets and some dense interactive views retain internal scrolling |

The current toy has model/query-key/value widths5/3/2. Its dedicated position coordinate is carried in the representation but ignored by the chosen projections and prediction head; the causal mask still restricts available sources. This limitation is disclosed, not presented as a property of trained Transformers.

## Reproduce

Run from the repository root:

```sh
python3 src/assemble.py --part 2 --out attention.html
node src/check-live-model.mjs attention.html
node src/toy_ref.mjs src/toy.json --compare src/py_check.json
node src/check-routing-scaling.mjs
node src/check-diagram.mjs attention.html
node src/pres_test.mjs
node src/qa.mjs attention.html --width 390 --height 844
node src/sweep.mjs attention.html
node figures/attention-diagram-preview/check-data.mjs
node figures/attention-diagram-preview/check-preview.mjs
```

The preview's `sync-data.mjs` refreshes its frozen evidence after a model change, and `export.mjs` regenerates the SVG stages. The article instead reads `AT.forward` directly through `attention-flow-data.js`.
