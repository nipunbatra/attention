# Vision I: image classification teaching route

The lecture explains one image classifier. Keep the same photograph as the anchor, use the small worksheet for arithmetic, then return to the real architecture. All new backward-pass numbers are chosen-parameter calculations, not a fitted model or benchmark.

## Main route

1. Sections 1–3: pet dataset, text parallels, image shapes, patches and shared projection. Section 2 ends with 196 content rows. Section 3 first locates P63 on the unchanged photograph and adds position. The purple CLS detour explains why the classifier needs one image summary, how attention fills that row, why the shared starting CLS produces image-dependent summaries, and how mean pooling can replace it. Resume the forward pass with all 197 rows present, then make Q/K/V and complete the measured classifier. Name LayerNorm here; save its details for section 7.
2. Sections 4–5: calculate one query, its scores, softmax and value message. Work the second head, concatenate and project. Return to possible visual head roles; explain that these are hypotheses rather than assigned jobs.
3. Section 6: readout, class probabilities, loss and learning. The new reverse sequence follows the exact earlier worksheet through the class head, residual, both heads, values, attention softmax, Q/K and patch projection. Show the single query-weight update as an arithmetic example. The two softmaxes have different axes and purposes.
4. Section 7: restore LayerNorm and the MLP. Follow both residual gradient paths, then return to the whole forward/loss/backward diagram. Compare information flow and readout with a conventional CNN.
5. Section 9: explain the proposed dog/cat adaptation. A Linear(192,2) head has 386 parameters. Distinguish frozen-encoder head training from fine-tuning. Show one batch and the train/validation/test procedure. No new training has been run.
6. Sections 10–11: use the saved real-photo predictions and measured attention/occlusion examples. Keep ImageNet outputs separate from the proposed two-class model. A confidence on one photo is not test accuracy.
7. Section 13: ask students to narrate the shapes and reverse path. The optional patch-rearrangement check belongs here: explicitly call it a thought experiment about location, not a preprocessing step. Its coarse 4×4 grid illustrates the idea; the model uses 14×14 patches. The animal label need not change. Section 14 closes classification and previews CLIP.

## Choose the depth for the audience

In the CLS detour, keep the dog photograph visible. `real-cls-purpose` shows two distinct origins: pixels pass through the patch layer, while CLS is a separate trainable parameter with no pixels. `cls-parameter-origin` explains initialization and why the width is 192. `cls-parameter-learning` illustrates the image-label gradient that trains it; it does not claim this photograph was a checkpoint training example. `cls-stored-start` shows actual saved parameter and position values. `cls-collect` follows those input activations to this dog's measured final summary. Distinguish a parameter changed by training from an activation changed during a forward pass.

Close that loop with `cls-shared-start` and `cls-two-image-readout`: the dog and cat have identical 192-coordinate CLS inputs, different measured outputs after the first attention residual, and different final CLS summaries and image labels. Use the same coordinates in both rows. The trace in `cls-two-image-trace.json` stores all 192 coordinates and verifies that the starting parameter stays fixed. `trace_cls_comparison.py` reproduces these two inference passes with the cached checkpoint; it performs no training.

Then follow the same dog through the two readout choices. `cls-without` now traces the known CLS route with real patch thumbnails, 197 input rows and the measured Newfoundland prediction. `cls-pool-dog` mirrors that diagram with 196 rows and mean pooling; this is an alternative design to train, with no claimed prediction. `cls-pool-arithmetic` uses a separate miniature dog example with four large patches and two chosen final features per row: [2,0], [4,2], [2,4], [0,2] average to [2,2]. These are illustrative values, not checkpoint outputs. Return to 196 × 192 → 1 × 192, then use `cls-readout-return` to resume the saved CLS model. Emphasize that pooling averages contextual features after the blocks, not raw pixels; both designs require training for their readout.

At `model-journey-checkpoint`, introduce just one Transformer block: 197 × 192 input rows → attention → MLP → 197 × 192 updated rows. Open those operations on the following slides. Only after the MLP, use `real-block-handoff` to show that block 1’s output is block 2’s input; its attention computes new Q/K/V with its own weights. `real-cls-depth` then shows the 12 distinct blocks in sequence and the single prediction at the end. Twelve is the model depth, not repeated image preprocessing, weight sharing across blocks, or twelve predictions. The overview and route ribbons label attention plus MLP as the inside of one block.

Inside that first block, `real-cls-attention` draws one normalized matrix X feeding three separate learned projections. Follow the unchanged row identities into Q, K and V; values change. `real-attention-product` shows Q × Kᵀ / 8, with CLS first and P1 through P196 on both score axes. Track the highlighted CLS/P63 entry: receiving query versus source key. `real-attention-cls-zoom` then outlines the CLS row in the full matrix and enlarges it beside the grid. Reveal its scores first, then softmax and the matching weight row: 196 patch scores plus one self-score give 197 weights summing to one. Nothing is causally masked because the full image is available. `real-attention-weights` returns to the full matrices and applies that same softmax to every query row, preserving shape while turning scores into source weights. `real-attention-mask` compares causal text access with the fully connected image matrix: even the first CLS row can read the last patch. `real-cls-values-origin` reconnects V to the same dog and the shared 192→64 value projection. `real-cls-value-scaling` selects the saved CLS weight for P63 and multiplies it across the patch’s 64 actual value features. `real-cls-value-sum` zooms out to (1×197) × (197×64) → (1×64), highlighting one V column and its output coordinate. Sum over sources, keep the feature width. `real-attention-values` then restores every query to form H=AV with 197 message rows. Matrix grids abbreviate entries with symbols and ellipses. The P63 values, its CLS weight, and the CLS message preview come from the saved dog traces; products are computed at full stored precision before rounding. The later small worksheet supplies a complete hand calculation.

For a first pass through backward propagation, use `backward-route`, `backward-scores`, `backward-cls`, `backward-heads`, `backward-one-weight` and `backward-patches`. The detailed softmax/QK derivatives can be a calculation workshop after the main mechanism is understood. Sections 8 (code), 12 (cost) and the existing notebooks are optional extensions; the lecture does not require a live notebook.

## Keep the examples distinct

- Pet photographs motivate the real task.
- The exact checkpoint trace is a pretrained 1,000-class ImageNet model: 224×224 RGB, 16×16 patches, 196 patch rows, D=192, 3 heads, 12 blocks.
- The four-patch grayscale worksheet has engineered 4-wide embeddings, two heads and two arrangement labels. Its simplified block omits LayerNorm and the MLP.
- The new pet adaptation is a procedure students could run, with no claimed measured accuracy.
- Earlier noisy-stripe training results remain in the optional worked lab and saved `training.json`; they are not pet-classification evidence.

## Useful questions

- Is CLS an image patch, a label, an activation, or a learned starting parameter?
- Why does it become image-dependent? How does class loss train it?
- Without CLS, how could we obtain one image vector? Why is deleting it from a pretrained CLS model a different experiment?
- Why are three heads compatible with two classes? What exactly differs between heads?
- Can CNNs see the whole image? How do their neighborhoods grow?
- What receives a gradient if only CLS is read? Why do source patch values still receive gradients?
- Which operation computes gradients and which changes parameters?
- How do we check generalization without selecting on the test set?

## Rebuild without executing notebooks

```sh
python3 src/build_vision1_lesson.py --slides-only
PYTHONPATH=src python3 src/vision1_gradients.py
```

The slide-only build reuses saved experiment artifacts. The second command checks analytic gradients against central differences for all 116 worksheet parameter coordinates; it runs no model-training loop.
