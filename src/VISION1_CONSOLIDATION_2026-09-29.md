# Vision I consolidation audit

Date: 2026-09-29

## Finding and action

The dog walkthrough in sections I–III already completes patch projection,
position/CLS preparation, Q/K/V, attention, multiple heads, both residual branches,
the MLP, the 12-block stack and classification. Former sections IV–VI repeated
these operations with a separate four-patch classifier. Former section VII
reopened block operations, and VIII switched shapes for the implementation.

The main lecture now has **164 teaching frames plus the cover**, down from
**256 teaching frames plus the cover**: 92 fewer frames (35.9%).

| Former material | Decision | Current location |
| --- | --- | --- |
| IV–VI: another complete four-patch forward pass | Remove from the main slide sequence; keep the optional lab and calculation sources | Optional notebook |
| Projection can retain spatial patterns that an average loses | One two-filter diagnostic beside the first patch projection | II, `patch-filter-patterns` |
| Distinguish changing a key from changing a value | One controlled-intervention check after the first full message matrix | III, `photo-key-value-check` |
| What different heads might mean | Move the existing photo-based hypotheses beside the head introduction | III, `heads-visual-roles` |
| Attention softmax versus class softmax | Compare the real model's 197 sources and 1,000 labels | III, `photo-two-softmaxes` |
| Loss and backward pass | Six slides following the same saved dog output and the same model dimensions | IV |
| VII: LayerNorm arithmetic, MLP, residual and depth restart | Remove repeated forward explanations; retain full-block gradient paths in the new backward pass | III and IV |
| VII: CNN versus ViT | Three slides: receptive fields, shared local detectors/equivariance, architectural assumptions and image readout | V |
| VIII: implementation | Seven SVG code-and-shape slides using B×3×224×224 throughout | VI |
| VIII: Conv2d patch embedding | Dedicated full slide: 192 filters, each 3×16×16; stride 16; B×192×14×14 output | VI, `code-photo-conv` |

Later adaptation, measured inspection, cost and transfer questions remain, with
consecutive section numbering VII–XII. The two-source arithmetic and 128×128
shape questions are short transfer exercises with their inputs supplied; they
do not depend on the removed worksheet walkthrough.

## Content and implementation checks

- The same image remains the anchor: 224×224 RGB, 196 patches, width 192,
  three heads of width 64, 12 distinct blocks, and 1,000 ImageNet outputs.
- Loss examples use the saved Newfoundland probability as a hypothetical
  labelled example. They do not claim the photo was in the checkpoint's
  training set. The one SGD bias update is illustrative arithmetic.
- Backward diagrams distinguish parameter updates from gradients, show both
  residual paths, and explain why keys/values carry gradients into patch rows.
- CNN comparison distinguishes translation equivariance from invariance,
  states the stride/boundary assumptions, and identifies ViT's remaining
  patch/position biases. Possible head roles remain hypotheses.
- SVG code snippets are extracted from the accompanying teaching model,
  `notebooks/vision/vit_image_classifier.py`. It is randomly initialized;
  it does not silently load or claim the pretrained photo result.
- Conv2d was checked against `unfold` plus `linear`, including the same
  channel/pixel order, weights and biases.
- Explicit attention was checked against PyTorch's scaled dot-product attention.
- A two-image batch matches independent image forwards; all model parameters
  receive finite, nonzero gradients in the synthetic check. No optimizer step
  or checkpoint download was performed.
- The complete manifest, SVG XML, local links, captions, teacher notes,
  numerical traces and optional worksheet parity checks pass.
- All 23 revised/relocated early and middle slides were checked in the browser
  for stage overflow and SVG text clipping. Representative gradient, CNN,
  code and section-break layouts were also inspected visually.

## Reproduce

```sh
python3 src/build_vision1_lesson.py --slides-only
python3 src/check_vision1_closure.py
uv run --offline --with torch --with numpy python src/check_vision1_lesson.py
uv run --offline --with torch --with numpy python src/check_vision1_photo_code.py
```

The consolidation is applied in `src/vision1_lecture_focus.py`. Original
worksheet builders and unused generated illustrations remain available for
the optional lab; only the main lecture sequence is shortened. The earlier
untracked flow audit and prototype work are preserved.
