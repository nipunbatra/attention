# Vision I — teaching guide

## Main lecture and optional references

The main lecture contains **54 content frames plus the cover: 55 slides**. It follows **Transformers beyond next-token prediction** and ends with the question leading to CLIP. The revised route derives the architecture through five questions while retaining visual Q/K/V intuition, CNN comparison and measured image interventions.

The separate reference deck has **150 content frames plus cover: 151 pages**. It preserves all 144 frames from the previous detailed sequence, two optional dividers and four reference figures. This includes neuron arithmetic, complete attention calculations, forward/backward diagrams, Linear/Conv2d equivalence, transfer learning, evaluation and interpretation. Nothing is silently discarded from that detailed sequence.

- Main: `vision1.html` and `pdf/vision1.pdf`
- Optional details: `vision1-reference.html` and `pdf/vision1-reference.pdf`
- Nine guided examples and free exploration: `vision1-explorer.html`
- Searchable audit companions: `pdf/vision1-transcript.md` and `pdf/vision1-reference-transcript.md`
- [Main slide map](VISION1_SLIDE_MAP.md) and [reference map](VISION1_REFERENCE_MAP.md)

Presentation controls: Right/Left advances builds, **S** opens notes, **O** opens the overview and **C** shows controls. Hash routes change when frames are reordered; stable frame IDs remain in the maps.

## Route through the main lecture

| Section | Frames | Teaching purpose |
|---|---:|---|
| 1. The task | 4 | Recall the prior lecture’s architecture summary; establish the actual checkpoint, image and five questions. |
| 2. Image to tokens | 9 | Motivate patching, count 196 patches, flatten RGB consistently, reuse one learned projection and inspect a measured output. |
| 3. Position | 3 | Separate content from location; add one learned positional vector to each content row. |
| 4. Readout | 6 | Reuse the known CLS idea; distinguish the shared starting parameter from the image-dependent state; prepare all 197 rows. |
| 5. Information exchange | 14 | Explain query/key/value roles and hypothetical examples, match keys, mix values progressively, contrast causal/full attention, inspect a trained map, and build the multihead pre-LN block. |
| 6. Prediction and intervention | 7 | Read final CLS, compute ImageNet class probabilities, cover a quadrant, rerun the model, compare four measured results, then inspect smaller covers. |
| 7. Comparison and synthesis | 11 | Compare CNNs and ViTs with three visual frames; explain patch cost; recap architecture/shapes/code; link optional material; end on language-derived class vectors. |

The main projected captions stay under 25 words. Speaker notes retain technical qualifications. Separate frames give the instructor room to explain each operation without packing the diagram with prose.

## One measured model throughout

The checkpoint is `vit_tiny_patch16_224.augreg_in21k_ft_in1k`: 224×224 RGB input, 16×16 patches, 196 patch rows plus CLS, D=192, 3 heads of width 64, MLP hidden width 768, 12 blocks and 1,000 ImageNet class outputs.

Oxford-IIIT Pet supplies `newfoundland_31`; it is not the classifier’s output vocabulary. The saved probability is **95.726752% Newfoundland**, followed by Tibetan mastiff and briard. This is one prediction, not test accuracy. Dog/cat adaptation is explicitly an optional new task. No new training was performed for the refactor.

The main two-feature, three-source value-mixture example is labelled **toy**. Its chosen weights `[0.6, 0.3, 0.1]` mix values `[2,0]`, `[0,1]`, `[1,1]` into `[1.3,0.4]`. The hypothetical verbal queries are intuition, not decoded meanings of trained vectors. The measured map uses block 4, head 1, query P74 from the saved model arrays.

## Input preparation before attention

All RGB examples use channel-major order: all R values, then G, then B; spatial order is row-major inside each channel. Pixel-major order can also work if the weight columns are permuted consistently. The code and figures use one convention throughout.

For each real patch, `x_i` has 768 values and `c_i = x_i W_E + b_E` has 192 features. The same projection is reused at every location. The main notation uses row vectors and `W_E: 768×192`; PyTorch stores the equivalent Linear weight as `192×768`.

Optional implementation lab:

- `F.unfold`: `(B,768,196)` → transpose → `(B,196,768)` → `Linear(768,192)`.
- `Conv2d(3,192,16,stride=16)`: `(B,192,14,14)` → flatten/transpose → `(B,196,192)`.
- `conv.weight.flatten(1)` is the equivalent Linear weight; biases are identical.
- Both have **147,456 weights + 192 biases = 147,648 parameters**.
- Conv2d expresses the shared operation directly and uses optimized kernels; speed depends on the backend and workload.

Content is blue; learned projection, position and starting CLS parameters are amber. The final computed CLS state is blue and input dependent. CLS and patch rows may exchange information in both directions. A block computes updates from its incoming states; the next block reads the updated states.

## Keep the query’s receiver visible

1. Fix the receiving token and its query.
2. Compare with each source key. Matrix rows are queries; columns are keys.
3. Softmax across the receiving row produces source weights.
4. Multiply each source’s value vector by its scalar weight.
5. Sum these contributions into the message for that receiver.
6. Join the receiver’s three head messages, project to 192 features and add to its incoming row.
7. Other receivers get their own messages in parallel.

ViT self-attention gets Q/K/V from the same normalized image sequence. Translation cross-attention uses target states for Q and source-encoder states for K/V. The mask comparison shows **allowed pairs**, not attention magnitudes: the text decoder masks future sources; ViT can read the complete image. Full access does not imply equal weights, and the attention matrix is generally asymmetric.

The MLP transforms features within each row; attention mixes information between rows. Both branches have pre-LayerNorm and residual additions. Blocks preserve `(197,192)` while changing the represented information. Parameters stay fixed at inference; activations and attention weights change with the image.

## Attention and occlusion answer different questions

The trained attention map shows source weights for one query, head and block. P60 receives about 9.09% in block 4, head 1, query P74. The displayed patch weights exclude CLS visually but are not renormalized. The scale is printed. This is neither segmentation nor a complete explanation of the final prediction.

Occlusion changes pixels before a fresh forward pass. For this checkpoint, normalized zero corresponds to gray RGB `(0.5,0.5,0.5)`. A 112×112 cover replaces 49 patch inputs; no tokens are deleted. The model weights, input size and target class stay fixed. Each intervention starts from a fresh original.

The four quadrant probabilities are 83.00%, 79.62%, 84.54% and 82.16%, against 95.73% originally. Top-right covering gives the largest tested drop, **16.11 percentage points**; all four top labels remain Newfoundland. The result is sensitivity to these gray replacements on this image.

The finer experiment covers one 16×16 patch in each of 196 separate runs. P78 gives the largest drop, **3.84 points**, to 91.89%. All top labels remain Newfoundland. A small drop does not establish that a region is useless, and drops across interventions must not be added. The main map uses a fixed −4 to +4 point scale. Full data and protocol remain in the reference deck and notebooks.

Feature similarity is a third measurement: it compares contextual vectors. All nine guided similarity/attention examples remain in the interactive lab.

## CNN comparison and ending

Three main slides compare context gathering, architecture assumptions and a concrete local-filter reuse example. The connection diagrams are schematic. CNN locality and sharing supply useful priors; global ViT attention computes content-dependent mixing. The comparison is qualified by training data, pretraining and compute rather than declaring a universal winner.

The two summary figures show the canonical seven-stage pipeline and its shape trace. The six-line pseudocode follows exactly those operations. Optional links precede the final two slides: the fixed classifier stores one learned vector per ImageNet class; what if that vector could come from language? End on the question. The present checkpoint does not already support arbitrary text class vectors.

## Build and audit

Run from the repository root:

```sh
uv run --offline --with timm --with pillow python src/build_vision1_lesson.py --slides-only
uv run --offline --with timm --with pillow python src/check_vision1_story.py
uv run --offline --with timm --with pillow python src/check_vision1_lesson.py
uv run --offline --with timm --with pillow python src/check_vision1_closure.py
uv run --offline --with timm --with pillow python src/check_vision1_photo_code.py
```

Render both HTML decks with the existing `src/export_slides.mjs`, using `--frames` to retain page PNGs. The exporter checks presentation overflow and browser errors, then captures final reveals at 2× resolution. The output prefix expected by the packager is:

```sh
node src/export_slides.mjs vision1.html tmp/pdfs/vision1-restored-main.pdf --frames tmp/pdfs/vision1-restored-main-frames
node src/export_slides.mjs vision1-reference.html tmp/pdfs/vision1-restored-reference.pdf --frames tmp/pdfs/vision1-restored-reference-frames
python src/package_vision1_audit.py --render-prefix tmp/pdfs/vision1-restored
```

Use an installed Chromium via `PLAYWRIGHT_CHROMIUM_EXECUTABLE` if needed. Packaging uses the existing Pillow/pypdf runtime; it adds bookmarks, compresses screenshots, checks every resulting page image against its rendered PNG and creates searchable transcripts including speaker notes. The PDFs themselves are raster snapshots; HTML retains builds and interactions.

The main authoring source is `src/vision1_question_story.py`. It derives the short route only after preserving the detailed library generated by the existing modules. `frame-manifest.json`, `reference-frame-manifest.json` and `detailed-frame-manifest.json` keep those routes distinct for validation.

## Visual semantics

| Meaning | Color |
|---|---|
| Image / patch / contextual vision states | Blue `#3478E5` |
| Language states | Purple `#8B5BB5` |
| Attention and information mixing | Teal `#178F82` |
| Learned CLS, positions and projection parameters | Amber `#B98224` |
| Loss, gradients and probability drops | Coral `#D45555` |
| Neutral architecture | Charcoal `#30343B` |
| CNN | Muted green-gray `#6E817B` |

Q/K/V diagrams use a labelled local role palette (purple query, amber key, teal value). `vision1_palette.py` defines shared colors. `vit-canonical-pipeline.svg` and its seven stage variants use consistent geometry. RGB channel teaching diagrams retain literal channel colors.
