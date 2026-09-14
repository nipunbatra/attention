# Part 3: learning and the Transformer

Students start with the exact attention model and 4→8→20 ReLU prediction MLP from Part 2. First train it; then introduce a larger Transformer architecture. Keep computed toy results separate from illustrative block examples.

## Numerical contract

- train_part3.py reads toy.json unchanged and writes toy3.json with offline SGD results. Initial token/position tables, attention projections, both predictor matrices, and both biases must match Part 2 exactly.
- The prediction MLP computes h = ReLU(e′ W_hidden + b_hidden), then logits = h W_vocab + b_vocab. The matrices are 4×8 and 8×20. Classroom notation calls them W₁ and W₂.
- The river prefix ends with “the” at position 10. The observed target is “water” at position 11. Single-target steps use learning rates 0.05, 0.1, and 0.3. A separate mean-loss step averages all ten shifted targets at learning rate 0.1.
- Saved results and parameters keep full precision; only displayed text is rounded. A displayed probability of 1.0000 does not mean certainty.
- part3.js supplies the saved training results, a four-coordinate LayerNorm calculator, a separate illustrative 4→8→4 FFN, and hand-chosen head patterns. These do not run a trained full Transformer.
- The full-model schematic uses pre-norm causal blocks, final LayerNorm, and a linear vocabulary readout. Its nonlinear FFNs sit inside blocks. Do not substitute its linear readout into the numerical toy.

## Classroom route

50 authored frames plus the generated title replace the previous 70-frame sequence. Detailed arithmetic, additional checks, and repeated comparisons remain available in reading mode.

| Sections | Frames | Teaching purpose |
|---|---:|---|
| 01 | 1 | One recap: attention → residual → the same prediction MLP. |
| 02 | 4 | Learning break; observed target, probability, and cross-entropy. |
| 03–04 | 8 | Parameters versus activations; shapes; complete forward and backward graphs, including the MLP. |
| 05–06 | 6 | A numerical SGD step before the general equation; step sizes, shifted targets, causal mask, and mean loss. |
| 07 | 6 | Block break; concrete needs of “she”, full-input heads, masks, dimensions, and combination. |
| 08–09 | 5 | Separate 4→8→4 block FFN; actual residual bypasses; four-coordinate LayerNorm. |
| 10–11 | 6 | Branches, stacking and multi-hop example; full decoder flowchart, final normalization, and linear readout. |
| 12 | 1 | The full model uses the same mean next-token objective. |
| 13 | 4 | Generation break; choose/append/repeat, KV cache, and costs with units. |
| 14–15 | 4 | Block arrangements, cross-attention, and context routes. Reused toy numbers are labelled. |
| 16–19 | 5 | Limits, fixed-window comparison, two checkpoints, and one closing summary linking to Part 4. |

## Verification

Run python3 src/train_part3.py, python3 src/check_training.py, then python3 src/assemble.py --part 3 --out part3.html. Browser checks are node src/check_part3_continuity.mjs, node src/check_position_capacity.mjs, node src/frame_audit.mjs part3.html, node src/check_tables.mjs part3.html, and node src/qa.mjs part3.html --width 390 --height 844.

The gradient check reports exact-zero ReLU kinks separately. It checks all 332 gradients on a smooth copy for both single-target and mean-loss objectives, and checks the zero-derivative convention at inactive units. It never changes the canonical model. The continuity regression independently reproduces the SGD steps and predictions, then checks diagrams, colours, controls, and phone layout.
