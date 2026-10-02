# Vision I — teaching guide

## Scope and prerequisite

**146 teaching frames plus the cover, in 10 sections.** This lecture follows **Transformers beyond next-token prediction** and leads into CLIP. The opening connects to the text encoder immediately: **same encoder idea, different tokens**.

The first October 2026 revision was selective: 26 repeated introductions or recaps moved to expandable reading notes, seven bridge/summary frames were added, and the net reduction is 19 frames (11.2%). The hand calculations, source-to-receiver attention walkthrough, multihead diagrams, Linear/Conv2d equivalence, transfer diagrams and nine guided explorer examples remain. The visual pass and removal of the grayscale filter detour reduce the main route to 146 frames from 150. Detailed patch arithmetic and attention calculations stay in the presentation. Two CLS parameter digressions, the historical paper figure, the dense whole-model map, the nine-state interactive widget and three repeated quadrant slides remain available in reading notes. Three clean interpretation figures replace the widget in the teaching/PDF route.

Presentation: Right/Left advances reveals, **S** opens presenter notes, **O** opens the overview, and **C** shows classroom controls. Reading mode includes longer explanations, sources, numerical tables and optional recaps. The [slide map](VISION1_SLIDE_MAP.md) lists current routes. Old `#section/frame/build` links can change when a section is reordered.

## Route through the lecture

| Section | Main question | Teaching emphasis |
|---|---|---|
| 1 | What changes from text to vision? | Recall encoder/decoder routes; identify ViT as an encoder; compare embedding lookup with pixel projection; establish the image-label task. |
| 2 | How do pixels become rows? | Calculate the small RGB projection, reuse its weights, then follow the actual 224×224 dog input into 196 rows of width 192. |
| 3 | How do rows become a prediction? | Add position and CLS; follow a query through scores, weights, value contributions and its own residual update; join heads, apply the MLP, repeat blocks and classify. |
| 4 | Can we recount the complete model? | One enlarged shape trace, a clean pre-LN block, then the backward path. The complete dense map is an optional reference. |
| 5 | How do CNNs and ViTs differ? | Compare local filters, global mixing, receptive fields and inductive bias with diagrams and tables. |
| 6 · optional lab | How does the code implement that model? | Preserve the same 224×224 input and annotate tensor shapes; prove Linear and Conv2d use identical pixel–weight products. |
| 7 | How do we adapt the pretrained model? | Original labels → new dog/cat task or changed domain; replacement head, frozen encoder, fine-tuning, held-out evaluation and inference. |
| 8 | What did this model actually predict? | Saved dog/cat outputs and the limits of two examples. |
| 9 | What can we measure inside it? | Three measured hero examples, a link to all nine guided examples, then controlled occlusion experiments. |
| 10 | What costs more, and what comes next? | Token count and quadratic score count; architecture and takeaways; fixed class vectors → language-derived candidates for CLIP. |

Use multiple meetings rather than treating 146 frames as a one-class target. A practical split is sections 1–3 for representation and computation, sections 4–7 for architecture/code/adaptation, and sections 8–10 for interpretation, cost and the CLIP handoff. Short frames allow students to predict the next step before revealing it.

## One model, clearly named exceptions

The main measured example uses `vit_tiny_patch16_224.augreg_in21k_ft_in1k`: 224×224 RGB input, 16×16 patches, 196 patch rows plus CLS, D=192, 3 heads of width 64, MLP hidden width 768, 12 blocks, and 1,000 ImageNet class outputs. These are this checkpoint's choices, not universal ViT dimensions.

The 2×2 RGB example uses chosen teaching weights for `Linear(12,2)` so both outputs can be computed by hand. It is not a second trained classifier. The original four-patch worksheet and separate synthetic training experiment remain optional lab material. Their numbers must not be described as the dog checkpoint's activations or performance.

The dog/cat task motivates adaptation; the saved ImageNet model predicts fine-grained labels such as Newfoundland and Persian cat. The 95.73% dog-image probability is one model output, not test accuracy. No new training run was performed for this revision.

## Consistent flattening and Linear/Conv2d equivalence

All RGB examples now use **channel-major order**: all R values, then G, then B. Within each channel, scan left to right and top to bottom.

- Small patch: `R_A, R_B, R_C, R_D, G_A, …, B_D`. Its two outputs remain `[2.5, −0.5]` and `[0.5, −2.5]` for the two illustrated patches.
- Real patch: 256 red values, 256 green values, 256 blue values. The first pixel's RGB entries occupy positions 1, 257 and 513 (one-based).
- `F.unfold` returns `(B,768,196)`. Transpose to `(B,196,768)` and apply `Linear(768,192)`.
- `Conv2d(3,192,16,stride=16)` stores `(192,3,16,16)` weights. `conv.weight.flatten(1)` is the equivalent Linear weight `(192,768)`; bias is identical.
- Both use 147,456 weights + 192 biases = **147,648 parameters**. Conv2d expresses the shared patch operation directly and uses optimized kernels; speed depends on the backend and workload.

`notebooks/vision/trace_real_patch.py` verifies all 196 flattened patch projections against the checkpoint, plus position addition and the first head's Q/K/V projections. The saved JSON records the flattening convention. Negative numbers come from the checkpoint's `(RGB/255−0.5)/0.5` normalization.

## Keep attention's receiver visible

The detailed sequence in section 3 is retained intentionally. Use the same receiver/source analogy from text:

1. Fix the receiver (initially CLS).
2. Compare its query with every source key. Score-matrix rows are queries; columns are keys.
3. Softmax across the chosen row gives its source weights.
4. Select a weight and the corresponding **source value row**. Repeat for several sources.
5. Add all weighted value rows to make the message **for that receiver**.
6. Join that receiver's head messages, project to D features, and add the update to **that receiver's incoming embedding**.
7. Other receivers have their own query rows and their own messages. All rows update in parallel from the current block input.

Q, K and V come from the same normalized current image sequence. In translation cross-attention, queries and keys/values instead come from different streams. Separate Q/K projections make the score matrix generally asymmetric. Whole-input access does not mean uniform weights. Attention weights and class probabilities normalize over different axes with different meanings.

CLS is initialized once before training, learned with the model, and reused at inference. Its activation becomes image dependent through repeated attention and MLP updates. Later blocks read updated CLS and updated patch representations. Both directions of interaction are possible. Mean pooling is a valid alternative trained readout; CLS is not inherently guaranteed to outperform it. Keep the checkpoint's trained readout for its saved measurements.

## Interpretation and evidence

The main route presents three measured examples: ear-side feature similarity, the same query in two attention heads, and CLS attention. [The standalone interactive lab](../vision1-explorer.html) retains all nine guided presets and free exploration. It needs HTTP serving so saved model arrays can load; use the same local server as the lecture. Feature cosine similarity, attention weights and occlusion are three different measurements:

- Similarity compares contextual feature vectors; visually coherent regions do not establish segmentation accuracy.
- Attention shows one head's source weights for one query in one block; it does not completely explain a label.
- Occlusion changes input pixels, reruns the same fixed model, and compares the same class probability. Quadrants give a coarse intervention; the 196 one-patch trials give finer spatial sensitivity. Results depend on replacement values and region size.

Possible head roles are illustrative hypotheses; measured examples are labeled separately. Patches and CLS retain 192 coordinates throughout the blocks even though their numbers change.

## Ending and the next lecture

Keep the architecture and four takeaways after the patch-cost control. Then reinterpret the supervised head as `score_k = hᵀw_k + b_k`: one learned weight vector per fixed label. Ask **what if a candidate class vector could come from language?**

The final diagram motivates CLIP's image/text encoders and shared space. Arbitrary text embeddings cannot replace today's ImageNet class weights. Alignment training, suitable projections and normalization are needed. The final slide is a bridge to the next lecture, not an invented zero-shot result from this checkpoint.

Sources: [ViT](https://arxiv.org/abs/2010.11929), [CLIP](https://arxiv.org/abs/2103.00020), [PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html), [PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).

## Rebuild and verification

```sh
uv run --offline --with timm --with pillow python src/build_vision1_lesson.py --slides-only
uv run --offline --with timm --with pillow python src/check_vision1_lesson.py
uv run --offline --with timm --with pillow python src/check_vision1_closure.py
uv run --offline --with timm --with pillow python src/check_vision1_photo_code.py
```

The audit PDF includes 151 pages: cover + 146 teaching frames, followed by four labelled optional reference pages (complete model map, original paper diagram, CLS initialization and CLS parameter learning). Its three interpretation figures replace screenshots of nine UI states; all nine remain in the HTML lab. It captures final reveals; live controls and intermediate animations remain in HTML. The searchable audit transcript supplies diagram labels, code and speaker notes alongside the rendered PDF.

## Visual semantics across the lecture series

| Meaning | Color |
|---|---|
| Image / patch / contextual vision states | Blue `#3478E5` |
| Language states | Purple `#8B5BB5` |
| Attention, information mixing, comparison | Teal `#178F82` |
| Learned CLS, position and projection parameters | Amber `#B98224` |
| Loss, gradients and trainable updates | Coral `#D45555` |
| Neutral architecture | Charcoal `#30343B` |
| CNN comparison column | Muted green-gray `#6E817B` |

Detailed attention calculations explicitly label a local Q/receiver, K/source, V/message palette. It is a local role convention, separate from modality colors. RGB-channel teaching diagrams retain their literal channel colors. Route bars use neutral boxes and blue for the current stage. Feature similarity uses a fixed −1 to 1 scale; the two-head static attention comparison uses one common scale. Interactive attention maps state their per-map rescaling.

`vision1_palette.py` defines the shared colors. `vision1_visual_refinement.py` adds the presentation figures and preserves original reference material. `figures/vision1/vit-canonical-block.svg` is the reusable pre-LayerNorm block; `vision-language-handoff.svg` uses the CLIP lecture's two branches, learned projections, normalized u/v and dot-product comparison.
