# One scene across the vision lessons

These are AI-generated photographic illustrations, created with OpenAI's built-in image generation tool on 2026-09-04. They are teaching inputs, not photographs from a dataset and not predictions from ViT, MAE, DINO, I-JEPA, CLIP or a VLM. Provenance is recorded here and in the article's scope notes; classroom figures omit the repeated production caption.

- `two-mugs.jpg`: terracotta mug, teal mug, blue book, plant.
- `one-mug.jpg`: edited version with the teal mug removed.

Both are 1536 × 1024. JPEG conversion uses quality 88. The second image is an illustrative intervention, not a pixel-identical controlled photograph: image generation can change details outside the requested edit. Human-readable counts are expected answers, not measured model outputs.

The figure component is `src/vision-scene.js`. `src/assemble.py` embeds both assets once in each standalone vision HTML file. SVG windows display crops from those same embedded pixels. The enlarged boundary-crossing patch is `(x=512, y=512, width=256, height=256)`, using top-left image coordinates. It contains a mug edge, tabletop and part of the book. No reconstruction is fabricated for a masked patch.

The realistic scene illustrates the task. The 4 × 4 images and their exact numerical models remain a separate, explicitly labelled worked example.

## Generation prompts

### Base image

Use case: photorealistic-natural. Asset type: recurring photographic illustration for four university computer-vision lessons. Generate one landscape 3:2 image, natural camera photograph, with exactly two ceramic mugs on a pale oak tabletop: a terracotta mug on the left, a teal mug on the right. Both mugs are fully visible, separated, handles obvious. A closed plain dark-blue book rests horizontally in front of them; a small green leafy plant in a simple off-white pot sits behind them. Slightly elevated three-quarter camera viewpoint, eye-level enough to see mug bodies, soft daylight from left, realistic ceramic texture and shadows, neutral pale wall, uncluttered everyday scene. Objects occupy the middle of frame with generous but not excessive tabletop around them. Use no text, no lettering or logos, no diagram annotations, no extra cups, no plates, no people. Sharp enough for patch crops. This is a clearly labelled illustrative asset in the teaching material, not a model result.

### One-mug edit

Use case: precise-object-edit. Input image is the edit target, a photorealistic classroom illustration of two mugs, a book and a plant. Remove ONLY the teal mug on the right including its handle and its own shadow, filling that area naturally with the same oak tabletop. Keep the terracotta mug on the left, the blue book, plant, lighting, viewpoint, exact framing and image dimensions unchanged. Do not add or move any other object. Exactly one mug must remain. No text or diagram annotations.
