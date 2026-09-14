# Toy model v3 — four coordinates and additive position vectors

The source of truth is `make_toy2.py`, which generates `toy.json`. Parts II and III share this model. All values and feature names are hand-chosen for teaching, not measurements from trained embeddings. A coordinate is one numerical feature, not a token ID or a probability. Real learned features can be distributed across many coordinates.

## Coordinate spaces

| Object | Width | Illustrative labels |
|---|---:|---|
| e, Δe, e′ | 4 | water, finance, person, glue |
| q, k | 3 | setting: water?, setting: finance?, who? |
| v, m | 2 | says: water scene, says: finance scene |

The word-feature labels mean water-related scenes, money-related scenes, people, and function words respectively. Examples of lookup rows: river `[3,0,0,0]`, bank `[.7,.7,0,.7]`, fisherman `[2,0,2.2,0]`, the `[0,0,0,2.4]`.

## Position is not an extra coordinate

`pos_emb` has 20 rows of four small, hand-chosen offsets. These are illustrative table entries, not sinusoidal values or learned measurements. Form `e_i = tok_emb[token_i] + pos_emb[i]` before the projections. The same four coordinates now mix word and position information; the labels do not imply a clean separation after addition.

For the repeated word “the”, positions 1, 5 and 10 give starting rows `[.1,0,.1,2.3]`, `[.1,.1,0,2.3]` and `[0,0,0,2.3]`. River at position 6 starts at `[3.1,-.1,0,.1]`. All sentence matrices in the ten-token example are 10×4.

This replaces the historical five-coordinate model. Do not restore a `pos` axis or zero out position in the projections. T9 verifies that swapping earlier words while keeping the final token fixed can change the prediction. Zeroing the position table makes that same permutation invariant for this one-layer final-position calculation.

## Projection shapes and roles

- W_Q and W_K: 4×3. Water/finance coordinates match scene clues; person contributes to who? Glue contributes to the query, not the key.
- W_V: 4×2. The water and finance coordinates supply value content.
- W_O: 2×4. Map the message back onto water and finance coordinates; the residual row retains all four coordinates.
- Part II predictor: W_hidden (W1) is 4×8, b_hidden (b1) has 8 entries; h = ReLU(e′ W1 + b1). W_vocab (W2) is 8×20 and b_vocab (b2) has 20 entries. Both layers use the same fixed parameters for baseline, pooling and attention examples. Hidden units are numbered, not additional token positions. Parameters are hand-designed, not trained.
- Part III retains its separately saved direct 4×20 linear vocabulary head for the attention-gradient worksheet; it does not use Part II's MLP parameters. Its full-Transformer architecture sketch also has the usual linear LM head.
- Part III's illustrative FFN: 4→8→4, so residual addition preserves width.

Queries and keys share a width because they are compared. Values have their own width because they are mixed and sent, then mapped back by W_O.

## Numerical contract

Keep the river-context bank query focused first on river (≥.40), then fisherman. The cheque-context query focuses first on cheque, then deposited. Final predictions rank water > boats > fish > ducks in context A and teller > clerk > queue > money in context B. Other tokens each stay ≤.04. The last-token-only baseline remains identical in both contexts because it receives the same token at the same position.

Run the Python generator, the independent JavaScript reference comparison and the browser regressions whenever these numbers change. Regenerate toy_report.md, preview evidence, SVG exports and the Part II HTML. The live flowchart reads that same model at runtime. Rebuild Part III and its training checks when its own numerical model changes; its direct linear readout is intentionally distinct. Historical reports are not the current numerical contract.
