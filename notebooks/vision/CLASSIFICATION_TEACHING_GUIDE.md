# Vision I: image classification teaching route

The lecture follows one 224×224 RGB photograph through the 192-wide, three-head, 12-block ImageNet classifier. A second end-to-end four-patch walkthrough is kept in the optional notebook, outside the main lecture. No new training is run for this revision.

## Main route

1. **Sections 1–3 — one forward pass.** Motivate image classification, form patches and embeddings, add position and CLS, then follow attention, heads, residuals, MLPs, depth and the final class head. Four useful points from the former worksheet now appear where they first matter: weighted filters versus averaging, controlled key/value changes, possible head roles, and the two softmax axes.
2. **Section 4 — loss and learning.** Attach a hypothetical Newfoundland label to the same saved prediction, then follow the class-head gradient, both residual branches, attention weights and values, and the shared input parameters. One bias update illustrates SGD. Parameters, forward activations, gradients and optimizer updates remain distinct.
3. **Section 5 — CNNs and inductive bias.** Compare local receptive fields with global patch attention; illustrate shared detectors and translation equivariance; compare both image-to-representation-to-classifier routes. Discuss locality, data and pretraining without claiming either architecture always wins.
4. **Section 6 — the same model in code.** Keep B×3×224×224 throughout. A full slide opens Conv2d(3,192,16,16), then show token preparation, explicit three-head attention, two residual branches, 12 distinct blocks, CLS readout and a training step. SVG snippets come directly from `vit_image_classifier.py`. Random initialization does not reproduce the saved pretrained prediction.
5. **Section 7 — dog/cat adaptation.** Replace the 1,000-class head with Linear(192,2). Compare frozen-encoder training and fine-tuning, then discuss batches and train/validation/test splits. This remains a proposed procedure.
6. **Sections 8–9 — measured predictions and inspection.** Use saved dog/cat outputs, attention and occlusion. Separate an image confidence from test accuracy and attention weights from causal explanations.
7. **Sections 10–12 — cost, transfer questions and next tasks.** Count tokens and score entries, ask students to reason about new shapes and patch arrangements, then close the classification story. The two-source arithmetic and 128×128 shape questions are short transfer exercises, not a repeated forward pass.

## Choose the depth for the audience

In the CLS detour, keep the dog photograph visible. `real-cls-purpose` shows two distinct origins: pixels pass through the patch layer, while CLS is a separate trainable parameter with no pixels. `cls-parameter-origin` explains initialization and why the width is 192. `cls-parameter-learning` illustrates the image-label gradient that trains it; it does not claim this photograph was a checkpoint training example. `cls-stored-start` shows actual saved parameter and position values. `cls-collect` follows those input activations to this dog's measured final summary. Distinguish a parameter changed by training from an activation changed during a forward pass.

Close that loop with `cls-shared-start` and `cls-two-image-readout`: the dog and cat have identical 192-coordinate CLS inputs, different measured outputs after the first attention residual, and different final CLS summaries and image labels. Use the same coordinates in both rows. The trace in `cls-two-image-trace.json` stores all 192 coordinates and verifies that the starting parameter stays fixed. `trace_cls_comparison.py` reproduces these two inference passes with the cached checkpoint; it performs no training.

Then follow the same dog through the two readout choices. `cls-without` now traces the known CLS route with real patch thumbnails, 197 input rows and the measured Newfoundland prediction. `cls-pool-dog` mirrors that diagram with 196 rows and mean pooling; this is an alternative design to train, with no claimed prediction. `cls-pool-arithmetic` uses a separate miniature dog example with four large patches and two chosen final features per row: [2,0], [4,2], [2,4], [0,2] average to [2,2]. These are illustrative values, not checkpoint outputs. Return to 196 × 192 → 1 × 192, then use `cls-readout-return` to resume the saved CLS model. Emphasize that pooling averages contextual features after the blocks, not raw pixels; both designs require training for their readout.

At `model-journey-checkpoint`, introduce just one Transformer block: 197 × 192 input rows → attention → MLP → 197 × 192 updated rows. Open those operations on the following slides. Only after the MLP, use `real-block-handoff` to show that block 1’s output is block 2’s input; its attention computes new Q/K/V with its own weights. `real-cls-depth` then shows the 12 distinct blocks in sequence and the single prediction at the end. Twelve is the model depth, not repeated image preprocessing, weight sharing across blocks, or twelve predictions. The overview and route ribbons label attention plus MLP as the inside of one block.

Inside that first block, `real-cls-attention` draws one normalized matrix X feeding three separate learned projections. Follow the unchanged row identities into Q, K and V; values change. `real-attention-product` shows Q × Kᵀ / 8, with CLS first and P1 through P196 on both score axes. Track the highlighted CLS/P63 entry: receiving query versus source key. `real-attention-cls-zoom` then outlines the CLS row in the full matrix and enlarges it beside the grid. Reveal its scores first, then softmax and the matching weight row: 196 patch scores plus one self-score give 197 weights summing to one. Nothing is causally masked because the full image is available. `real-attention-weights` returns to the full matrices and applies that same softmax to every query row, preserving shape while turning scores into source weights. `real-attention-mask` compares causal text access with the fully connected image matrix: even the first CLS row can read the last patch. `real-cls-values-origin` reconnects V to the same dog and the shared 192→64 value projection. `real-cls-value-scaling` selects the saved CLS weight for P63 and multiplies it across the patch’s 64 actual value features. `real-cls-value-sum` zooms out to (1×197) × (197×64) → (1×64), highlighting one V column and its output coordinate. Sum over sources, keep the feature width. `real-attention-values` then restores every query to form H=AV with 197 message rows. Matrix grids abbreviate entries with symbols and ellipses. The P63 values, its CLS weight, and the CLS message preview come from the saved dog traces; products are computed at full stored precision before rounding. The later small worksheet supplies a complete hand calculation.

For a shorter class, use the six-slide photo learning section at a conceptual level. The companion text includes the attention derivatives. Code (section 6), measured inspection, cost and the optional notebook can be assigned for independent study; no live notebook is required.

## Keep the examples distinct

- Pet photographs motivate the real task.
- The exact checkpoint trace is a pretrained 1,000-class ImageNet model: 224×224 RGB, 16×16 patches, 196 patch rows, D=192, 3 heads, 12 blocks.
- The optional four-patch grayscale worksheet has engineered 4-wide embeddings, two heads and two arrangement labels. Its simplified block omits LayerNorm and the MLP.
- The pet adaptation is a procedure students could run, with no claimed measured accuracy.
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
python3 src/check_vision1_closure.py
python3 src/check_vision1_photo_code.py
```

The slide-only build reuses saved experiment artifacts. The closure check validates the saved arithmetic and slide artifacts. The code check uses synthetic inputs to verify Conv2d against unfold-plus-linear, attention against PyTorch SDPA, batch independence, tensor shapes and gradient reach; it performs no optimizer step.

### Completing one head before introducing three

Finish the CLS weighted sum and the full H matrix first. `real-heads-intro` marks a visible topic break: head 1 has completed its attention calculation. `real-heads-qkv` fans the same X (196 patches plus one CLS) into nine learned projections, three per head. `real-heads-messages` draws each complete Q/K → scores → softmax → weighted V path. `real-heads-cls` compares measured 64-feature messages while keeping the normalized CLS input fixed. There is one CLS token, shared by all heads; messages are computed activations. `real-heads-concat` appends those messages in feature order, with no learned parameters. `real-cls-message` connects the entire attention branch to an explicit residual skip path, matching the text-attention diagram. Follow E through LayerNorm, three heads, concatenation and the learned output projection to ΔE; carry the original E around that branch to a visible addition node. The contextualized representation is U=E+ΔE. `real-cls-residual` then zooms into the same dog's CLS row, with measured vectors and the first coordinate addition: −0.704+1.167≈0.463. All 192 coordinates are added this way. Continue to the MLP, later blocks and final classifier; this intermediate representation is not yet a prediction.

The numeric examples come from `figures/vision1/multihead-cls-trace.json`, generated by the cached-model, inference-only script `notebooks/vision/trace_multihead_messages.py`. The trace checks separate head projections against the fused checkpoint layer, every concatenated coordinate, the output projection, and the earlier dog previews. No training is needed.

### Open the MLP, then close the block

`real-cls-mlp` returns to the whole block before opening its second branch. First show E → block 1 → E¹, then reveal attention plus its residual, the intermediate U, and the MLP plus its residual. All three matrices are 197 × 192. `real-mlp-network` draws representative neurons and dense connections for LayerNorm(U) → Linear(192,768) → GELU → Linear(768,192). Its first two displayed activations come from the saved dog trace. The hidden width of 768 is a feature count, unrelated to the patch pixel count. Both linear layers have biases, and GELU is the intervening nonlinearity. `real-mlp-residual` adds the MLP result to U, showing 0.463 − 0.100 ≈ 0.363 for the first CLS coordinate. The full sequence receives this same row-wise operation with shared MLP parameters.

`real-block-handoff` draws all rows leaving block 1 and entering a separate block 2. Keep CLS and every patch row; nothing is patchified again. `real-block-changes` distinguishes computed activations from stored parameters: Q/K/V, softmax weights, messages, MLP activations and representations are recomputed in each block. Each block has its own LayerNorm, attention and MLP parameters, which stay fixed during this forward pass. Training changes them later. `real-cls-depth` closes with the 12-block stack and one final classification. This is a sequence of distinct blocks, with the same architecture and tensor shapes, rather than feedback through one shared block.

### Open the final classification layer

`real-cls-readout` draws the full 197 × 192 output matrix, final normalization and extraction of the highlighted CLS row. That 1 × 192 vector contains image features, not class probabilities. `real-classifier-network` opens `nn.Linear(192,1000)` as a dense neuron diagram: every class reads all 192 features with its own weights and bias. Its 1,000 outputs are logits. There is no hidden layer or GELU in this head. `real-classifier-score` follows the first two CLS feature × weight products for Newfoundland, the sum of the other 190 products, and the bias into the measured score 15.4763. Rounded diagram inputs are previews; the displayed product totals were computed before rounding.

`real-classifier-softmax` normalizes over all 1,000 labels, including the 997 omitted from the drawing. It shows their combined probability mass and explains that class softmax ranges over labels while attention softmax ranges over source rows. `real-cls-prediction` closes the same dog forward pass with argmax → Newfoundland at 95.73%. This is a probability on one image, not accuracy. Numerical evidence is in `figures/vision1/classifier-readout-trace.json`; `notebooks/vision/trace_classifier_readout.py` uses the cached head and saved final CLS without another image forward pass or training. The checks independently reproduce all displayed class dot products and softmax using all 1,000 logits.
