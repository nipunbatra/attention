# Trained ViT patch explorer

Open [the lecture, section IX](https://nipunbatra.github.io/attention/vision1.html?present#s09/2/0).

1. Click a patch on the left. Purple marks the query. Each selection covers **16×16 pixels**.
2. The right image shows its attention over source patches. Click any source, a top-ranked thumbnail, or the CLS weight to see the score and its softmax weight.
3. Change the head or block. **Play blocks** advances once through blocks 1–12 with the same selected query and head. It stops on a manual change, leaving the slide, or hiding the tab.
4. Switch to **Patch feature similarity**, choose **Ear**, and select **Block 12**. The nearest other patches are P82 and P81, on the opposite side of this dog's head. This is an observation for this image/model, not a guarantee that corresponding parts always match.

The interactive frame replaces three static head/depth/query comparison frames. The lecture now has 161 teaching frames plus the cover.

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

Source files: `src/vision1_attention_explorer.py` (slide/print SVG), `src/vision1-inspector.js` (math and interaction), `src/vision1-inspector.css` (layout), and `notebooks/vision/export_attention_explorer.py` (measurements).
