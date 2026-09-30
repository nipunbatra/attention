# Trained ViT patch explorer

Open [the lecture, section IX](https://nipunbatra.github.io/attention/vision1.html?present#s09/2/0).

## Start with the nine guided examples

The explorer opens in **Guided examples**. Locate the purple reference, read the gold map, then discuss the takeaway. The solid gold box marks the source whose measured value is shown beside the explanation.

- **Next example** sets the query, block, head and view for you. **Previous** revisits the last example.
- The **Guided example** menu jumps directly to any of the nine presets.
- After the last example, **Explore freely** exposes the original controls. You can also enter **Free exploration** at any time.
- **Guided examples** returns to the exact preset you left; custom settings do not overwrite it.

| Example | Preset | What to notice | Takeaway |
|---|---|---|---|
| 1. Background finds background | Block 12, P182, feature similarity | Purple marks the lower-right background patch. Look for gold around the dog, with less gold on its body. | Patch features can separate regions that look like foreground and background. This is a similarity pattern, not a predicted segmentation mask. |
| 2. Check a true corner | Block 12, P1, feature similarity | Now the reference is the top-left corner. Its strongest matches are the other three corners, with cosine similarity almost 1. | A strong feature match need not identify an object part. Inspect where the matches occur before assigning them a meaning. |
| 3. One ear finds the other side | Block 12, P74, feature similarity | Purple marks the left side of the dog’s head. P82, on the opposite side, is the strongest other match. | Distant patches can have similar learned features. This correspondence appears in this trained model and photograph; it is not guaranteed for every image. |
| 4. Move the reference to the nose | Block 12, P63, feature similarity | The query moves to P63 near the nose. The brightest matches move toward nearby face and muzzle patches. | The map answers a question about the selected patch. Changing the reference changes which features are being compared. |
| 5. Follow the body’s dark fur | Block 12, P147, feature similarity | The reference is now on the chest. Notice the group of similar patches lower on the dog, rather than around its nose. | Different regions of one object can have different features. Similarity need not highlight the entire dog uniformly. |
| 6. Rewind the ear example to block 1 | Block 1, P74, feature similarity | Keep the same P74-to-P82 ear pair as example 3. Its cosine similarity is now 0.322; after block 12 it was 0.902. | The pixels stay fixed while the representation changes through the blocks. The late-block match was not already this strong at the start. |
| 7. Ask what the ear reads | Block 4, P74, attention head 1 | Switch to attention: block 4, head 1, query P74. P60’s value row is multiplied by 9.09% in this head’s message. | An attention weight scales a source’s value in the query’s message. Feature similarity compares patch representations. |
| 8. Change only the attention head | Block 4, P74, attention head 2 | The image, query and block stay fixed. Head 2 puts its largest patch weight on P38, at about 2.29%. | Heads learn different ways to gather information. Gold is rescaled within each attention map, so compare percentages, not brightness, across heads. |
| 9. Let CLS gather an image summary | Block 12, CLS, attention head 1 | The query is now CLS, not an image patch. In this head, P64 near the face receives about 24.86% of the weight. | CLS can gather patch information for classification. This is one head in one block, not a complete explanation of the final label. |

## Explore freely afterward

1. Click a patch on the left. Purple marks the query. Each selection covers **16×16 pixels**.
2. Click a source on the right, a ranked thumbnail, or the CLS weight to inspect the number.
3. Change the head or block. **Play blocks** advances once through blocks 1–12 with the same query and head. It stops on a manual change, leaving the slide, or hiding the tab.
4. Compare **Attention to keys** with **Patch feature similarity**. The view label and color scale state which measurement is shown.

The nine examples are steps within one interactive frame, so they do not repeat the model walkthrough or add nine slides.

## What the colors mean

**Attention:** `softmax(q_i Kᵀ / sqrt(64))` for one head. The softmax includes all 197 sources: CLS plus 196 image patches. The image displays the 196 patch weights without renormalizing them; the CLS weight is shown separately. Gold opacity uses a scale from zero to the largest visible patch weight **in this map**, and that maximum is labelled. Compare the numeric weights when changing heads or blocks. Each weight multiplies its source's 64-dimensional value row.

**Feature similarity:** cosine between the selected patch's 192-dimensional representation and every other patch representation **after the selected block's attention, MLP, and both residual additions**, before the next/final LayerNorm. Blue is negative, clear is zero, gold is positive, on a fixed −1…1 scale. The selected patch has a self-match of one and is omitted from the top-three ranking. The head selector is disabled: this representation combines the heads and the MLP.

This is the same supervised classification ViT as the rest of the lecture. It adopts the click-to-inspect interaction associated with [DINO](https://github.com/facebookresearch/dino); it does not present DINO features. Attention is directed query–key matching, whereas feature cosine similarity is symmetric. Neither is a causal explanation of the prediction. Some trained attention heads favor background corners.

## Provenance

- Model: [`timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k), pretrained on ImageNet-21k and fine-tuned on ImageNet-1k.
- Photograph: the lecture's Oxford-IIIT Pet image `figures/vision1/newfoundland_31.jpg`.
- Exact input view: `figures/vision1/model-input.png`, 224×224 after the checkpoint's evaluation resize/center crop. The model also applies its configured RGB normalization.
- Saved prediction: Newfoundland, approximately 95.73%.
- No training is run. One evaluation forward pass captures all 12 blocks. Hooks independently reconstruct attention and verify it against the model's attention-module outputs and the previously saved lecture maps.
- `manifest.json` records source/display hashes, preprocessing, package versions, shapes, file hashes and verification flags. `reference.json` contains independent PyTorch rows for browser-math checks.

Each block file contains little-endian float32 Q (3×197×64), K (3×197×64), and post-block features (197×192), in that order. The browser lazily loads one 453,888-byte block at a time, caches it, and calculates only the selected row. Total activation data is 5.45 MB. These are activations for one public photograph, not model weights. A failed load offers Retry and the saved SVG example. The SVG also provides the no-JavaScript/print view.

## Reproduce and verify

From the repository root, with the existing cached checkpoint and environment:

```sh
HF_HUB_OFFLINE=1 uv run --offline --with timm --with pillow python notebooks/vision/export_attention_explorer.py
python3 src/build_vision1_lesson.py --slides-only
python3 src/check_vision1_explorer.py
python3 src/check_vision1_closure.py
python3 -m http.server 8791
```

Open `http://127.0.0.1:8791/vision1.html?present#s09/2/0`.
The exporter needs the pretrained checkpoint cached locally; it deliberately does not train or download another model in this workflow. The checks compare 240 JavaScript rows with independent PyTorch outputs across all blocks, all heads, patch queries, and CLS. Keyboard users can tab into either image, move with arrow keys, and select with Enter/Space.

Source files: `src/vision1_attention_explorer.py` (slide/print SVG), `src/vision1-inspector.js` (math and interaction), `src/vision1-inspector.css` (layout), `src/vision1_explorer_tour.py` (guided presets), and `notebooks/vision/export_attention_explorer.py` (measurements).
