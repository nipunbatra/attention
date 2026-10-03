# Generated bird intervention — prompt and provenance record

Built-in OpenAI image tool, 29 September 2026. Newly generated teaching illustrations; no creator photographs were reused. Original and both edits are retained. The experiment plan was written before inference; the four class descriptions and softmax scale were fixed in advance.

## Original — `figures/bird-toucan-original.png`

Use case: scientific-educational. Asset type: realistic synthetic photograph for an undergraduate machine-learning lecture on CLIP visual-feature interventions. Generate ONE square image, no collage, no text or labels. A toco toucan perched on a simple horizontal brown branch, clear side profile facing right. Its entire oversized orange-yellow beak with black tip is clearly visible, along with its black body, white throat, blue eye ring, feet and tail. The bird fills most of the square but no part is cropped. Soft green leafy bokeh background, daylight, sharp natural feather texture, plausible proportions, uncluttered wildlife-photo composition. This image will later be edited to shorten only the beak, so keep the silhouette clean, the beak unobstructed and leave empty background around it. No other animals, no watermarks. Realistic photograph style; this is a generated teaching illustration, not a sourced wildlife photograph.

## Beak edit — `figures/bird-toucan-small-beak.png`

Reference: the original generated image, not the background control.

Use case: precise-object-edit. Edit target: this generated toucan teaching photograph. Make ONE localized visual intervention: replace the entire oversized orange-yellow toucan beak with a very short, small, dark charcoal bird beak, approximately one quarter of the original beak length, attached plausibly at the same mouth position and pointing right. Fill the vacated beak area with the matching green leafy background. Preserve the rest of the image as closely as possible: exact square dimensions and crop, camera, bird silhouette outside the beak, black crown, blue eye ring, eye position, white throat, black body and wings, red under-tail patch, feet, tail, branch, green bokeh and lighting. Do not shrink or change the head, do not recolor any feathers, do not alter the eye, do not reposition the bird. No text, labels, arrows or borders. This is a deliberately artificial counterfactual bird illustration, not a biological species claim. Return the edited full photograph only.

## Background control — `figures/bird-toucan-background.png`

Reference: the original generated image, not the beak edit.

Use case: precise-object-edit. Edit target: this generated toucan teaching photograph. This is a background-only control for a machine-learning class. Change ONLY the soft green leafy bokeh background to soft blue-grey bokeh with similar brightness. Preserve the ENTIRE bird, its oversized orange-yellow beak and black tip, blue eye ring, white throat, black wings and body, red under-tail patch, exact head shape, feet, tail, pose, crop and horizontal brown branch as closely as possible. Keep the bird and branch in exactly the same locations and at the same scale; do not change any part of the bird or its beak. Preserve square dimensions, photographic texture and daylight illumination on the bird. No text, labels, borders or arrows. Return the full edited photograph only.

## Interpretation and sources

The intended edits are approximate: the image generator may also alter incidental pixels. This single three-image illustration is not a benchmark, a change of real species, or proof that a particular CLIP coordinate represents beak size.

- CLIP encoding and similarity: [Radford et al., 2021](https://proceedings.mlr.press/v139/radford21a.html), [official implementation](https://github.com/openai/CLIP/blob/main/clip/model.py).
- The distinction between image editing and a future intervention on a predicted concept comes from [Koh et al., Concept Bottleneck Models, 2020](https://proceedings.mlr.press/v119/koh20a.html).
- This toucan experiment, candidate menu, control, diagrams and classroom exercises are newly authored.
