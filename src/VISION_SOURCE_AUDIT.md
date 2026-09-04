# Vision to language: adaptation notes

Reviewed 2026-09-04. The new four-part series lives beside the attention lessons and uses their existing article/presentation/PDF runtime.

## Sources retained

The original material is in `/Users/nipun/git/interactive/src/articles/` under `vision-transformer`, `vision-ssl`, `clip-zero-shot`, and `vlm`. Each has `index.html`, `main.js`, `styles.css`, and metadata. The source files are preserved. The earlier lecture worksheet is `/Users/nipun/git/dl-teaching/lecture17/L17-vit-multimodal.typ`.

The adaptation retains patch extraction, visual pretraining objectives, image–text matching, connectors, generation, and the thermal-calibration example. It changes their presentation to bounded frames with article companions, progressive SVGs, small PyTorch snippets, and explicit numerical provenance. The recurring 4×4 grayscale grid connects the parts.

## Correctness changes

| Original risk | Adaptation |
|---|---|
| A pooled MobileNet vector treated as a 7×7 spatial feature map | Exact displayed patch projection and image attention; no disguised CNN feature-map claim |
| Patch slider changed only a picture, not network tokenization | Patch extraction is identified as a separate exercise; live pixel edits recompute the fixed worksheet |
| “Position embedding” toggle actually added a distance penalty to attention logits | Same-width position vectors added before projection; no extra positional dimension required |
| Synthetic attention rollout presented as unsynthesized model evidence | All hand-chosen parameters and illustrative figures are labelled; arithmetic is recomputed |
| Position-free self-attention called invariant | Row outputs are permutation-equivariant; the unchanged CLS readout is invariant to patch-only permutation |
| Early ViT attention described as necessarily local; full block shown with the wrong norm order | All image positions are available; the full original ViT architecture is pre-norm, distinct from the simplified worksheet |
| Interpolation shown as MAE behavior | Exact reconstruction loss for a user-selected guess, explicitly not a trained MAE |
| I-JEPA target described as coming from the same encoder or a pre-cropped target | Separate EMA target encoder processes the full image; target block representations are selected afterward |
| Unsupported DINO low-label performance claims | Remove benchmark claims; explain objective and evaluation protocol using the primary paper |
| Temperature odds example overstated or miscomputed | Use exact normalized embeddings, scaled logits, denominators, and candidate probabilities |
| Contrastive trainer omitted normalization gradients, used inconsistent averaging and in-place mixed updates | Independent NumPy reference and finite differences; all parameter updates use one gradient snapshot |
| One architecture or staged training recipe presented as universal across VLMs | Distinguish projected prefix, separate cross-attention, and learned query/resampling bridges with named sources |
| A lower loss implied general improvement | VLM shows both training pairs: a one-example update improves one response but breaks the other |
| Thermal palette alone implied a recoverable absolute temperature | Show identical normalized colours under different calibration ranges; require a visible scale or sensor data |

The earlier interactive toys are not evidence of trained-model behavior. New fitted toys likewise demonstrate computation, not general translation, image recognition, zero-shot transfer, or counting ability. The VLM's two-image conditioning test reuses its training images.

## Vision I teaching rebuild

The first adaptation passed arithmetic and fit checks but skipped too many reasons and intermediate steps. Its replacement keeps the same image in view and asks a concrete question—one occupied block or two—from the opening frame. It works one patch-projection column at a time, changes a pixel to expose what the map does, motivates a global collection row before introducing CLS, and keeps source crops beside keys and values.

A value-only intervention demonstrates unchanged weights with changed information. Scores, exponentials, normalization, value contributions, residual addition, classifier columns, and loss are separate teaching steps. The initial model gets the one-block image wrong. The same model then trains on both images with real full-batch SGD; students see the graph, autograd call, an actual parameter update, and recomputed predictions. Manual derivative worksheets stay out of the lesson.

All learned numbers in this two-image experiment are updated, but this is not evidence of a general block counter. The first numerical encoder snapshot remains unchanged for the later VLM example. The prose uses direct teacher questions; formal permutation arguments and optional shape/cost detail stay in the article companions.

## Primary references

- ViT: https://arxiv.org/abs/2010.11929
- MAE: https://arxiv.org/abs/2111.06377
- DINO: https://arxiv.org/abs/2104.14294
- I-JEPA: https://arxiv.org/abs/2301.08243
- CLIP: https://arxiv.org/abs/2103.00020
- LLaVA: https://arxiv.org/abs/2304.08485
- Flamingo: https://arxiv.org/abs/2204.14198
- BLIP-2: https://arxiv.org/abs/2301.12597

The learner-facing articles link the relevant papers near architectural claims. Release test results belong in `CLASSROOM_QA.md`, not in this provenance record.
