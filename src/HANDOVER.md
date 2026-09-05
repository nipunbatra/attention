# Handover: interactive teaching series

Updated 2026-09-04. Owner: Nipun Batra.

- Repository: https://github.com/nipunbatra/attention
- Published series: https://nipunbatra.github.io/attention/
- Canonical local checkout on this Mac: `/Users/nipun/git/attention`.

All four attention parts and the four-part Vision to language extension are implemented. They share a slide-first reading/presentation system, not separate article and slide sources. The first slide-first checkpoint was `a49f811`; `a1d609c` completed Part 3 and the numerical-correctness pass. `CLASSROOM_QA.md` records local verification. Check the checkout's Git log and the GitHub Pages workflow for the current published commit. A temporary checkout is not evidence of what is live.

## Start here

The shared-layout pass removes permanent presentation header/footer bars. A small **Controls** button (or **C**) opens
navigation without changing the stage scale. Slides show one title without repeated section labels. The article's notation
strip is no longer sticky, and the end note is collapsed under “About this page”.

Use direct teacher explanations and a short presenter cue followed by supporting guidance. Preserve the computations,
notation, and technical caveats. PDFs open `details.reveal` answers by default
(`--answers show`); `--answers authored` preserves the closed questions. `export_test.mjs` checks actual answer pixels and
PDF page counts, including dynamically created reveals. Browser Print opens answers and restores them afterwards.

Read this file, then `PRESENT.md` for the current layout/runtime contract and `CLASSROOM_QA.md` for recorded verification. Inspect the actual section and shared runtime before editing.

`SPEC.md`, `BRIEF.md`, `GUIDE.md`, `GUIDE1.md`, `GUIDE3.md`, `AXES.md`, `PARTS.md`, the older feedback files, and `REV2_TASK.md` preserve useful design history. Some contain superseded notation, planned work that is now done, or old layout assumptions. They are historical guidance, not instructions to undo the current implementation. `CONTRACT.md` documents components; confirm its examples against `shared.js` when they differ.

## Source and outputs

| Part | Editable sections | Data/runtime | Assembled output |
|---|---|---|---|
| 1: characters to prediction | `sections1/secNN.html` | `toy1.json`, `part1.js`, `part1-diagrams.js`, `part1.json` | `../part1.html` |
| 2: self-attention | `sections/secNN.html` | `toy.json`, `part2.json` | `../attention.html` |
| 3: learning and Transformer blocks | `sections3/secNN.html` | `toy3.json`, `part3.js`, `part3.json` | `../part3.html` |
| 4: cross-attention and translation | `sections4/secNN.html` | `toy4.json`, `part4.js`, `part4.json` | `../part4.html` |
| Vision I: ViT (source ID 5) | `sections5/secNN.html` | `toy5.json`, `part5.js`, `part5-diagrams.js`, `part5-learning.js`, `part5.json` | `../vision1.html` |
| Vision II: visual pretraining (source ID 6) | `sections6/secNN.html` | `toy6.json`, `part6.js`, `part6.json` | `../vision2.html` |
| Vision III: CLIP (source ID 7) | `sections7/secNN.html` | `toy7.json`, `part7.js`, `part7.json` | `../vision3.html` |
| Vision IV: VLM (source ID 8) | `sections8/secNN.html` | `toy8.json`, `part8.js`, `part8.json` | `../vision4.html` |

Paths in this table are relative to `src/`. Shared files are `shell.html` (layout/CSS), `shared.js` (math, widgets, notation, presentation runtime), and `assemble.py`. Do not edit the generated root HTML or the inlined `katex-bundle.html` by hand. `index.html` is the series landing page; `part2.html` redirects to `attention.html`.

The internal source IDs 5–8 are not displayed part numbers. Their configs set `series: "Vision to language"`, `part: 1..4`, `partLabel: "Vision I".."Vision IV"`, and separate `vision1..4` notation filters. `assemble.py --part 5 --out vision1.html` selects source 5 and displays Vision I. The text configs use `series: "Attention and language"` and display Part 1–4.

Build each part once in any order. `assemble.py` derives available lesson targets from complete section/config/data sources, not existing output files. The conventional outputs are the filenames in the table above; a future source config can declare a different `output`. Set `published: false` for an unpublished draft, or `available: false` on an individual navigation entry to leave it disabled. Unknown destinations stay unavailable even if a stale HTML placeholder exists. Distribute the complete set of outputs for offline series navigation. `check_metadata.py` tests clean-directory builds and those planned-link cases.

The standalone staged diagram lives in `figures/attention-diagram-preview/`. Part 2 embeds its same `diagram.js` source through `src/attention-flow-data.js`; keep the preview and article synchronized by changing that shared source.

Part 1's four diagrams live in `part1-diagrams.js` and are inserted by `assemble.py`. They adapt the original handwritten
notes' visual sequence while reading current `AT.mlp` numbers: actual embedding geometry, repeated lookup and ordered
concatenation, forward/backward learning, and training versus generation. Human position/slot labels are 1-based.
The generation diagram uses a reproducible temperature-1 sample, not a claim that the observed target is the argmax.
Boundary `-` stops generation. The observed target enters loss on a separate branch and must never be drawn as an MLP input.

## Notation and conceptual agreements

- Use row vectors. `E_tok` is the learned vocabulary lookup table, shape `|Vocab| × d_model`. `E` is the current sequence stack, shape `T × d_model`. They are not interchangeable.
- Before attention, `e_i` denotes an embedding/current representation. Introduce position addition explicitly: `e_i^(0) = E_tok[token_i] + P[i]`. Position is added as a same-width vector, not appended as a compulsory new dimension in real models.
- `q_i = e_i W_Q` asks what to retrieve; `k_j = e_j W_K` supplies matching features; `v_j = e_j W_V` supplies information to send. A value is a learned projection, not a renamed embedding. Its width may differ from the embedding width.
- Raw scores are `r_ij = q_i · k_j`; scaled scores are `s_ij = r_ij / sqrt(d_k)`. Apply the causal mask before row-wise softmax. `alpha_ij` weights input positions, not vocabulary outcomes.
- `m_i = sum_j alpha_ij v_j` is one retrieved message. The matrix of messages is `H = A V` (`Mmsg` in runtime results). Reserve `M` for the causal mask. `Delta e_i = m_i W_O`, then `e_i' = e_i + Delta e_i`. Delta denotes an activation update, not an optimizer step.
- The vocabulary head maps the final known token's contextual row to next-token logits. During generation, append the chosen token and recompute for the new last position. The unknown next token does not provide a query.
- Keep the bank/river/finance examples and work arithmetic progressively. Intuitive English questions and named axes explain a contrived model; real projections learn vectors, not literal questions or guaranteed semantic axes.
- Preserve object colours: embedding blue, query purple, key amber, value teal, attention weight rose, update green, updated representation blue/green. Do not use these colours for unrelated decorations.
- Part 2 teaches one attention head plus output projection and residual. Part 3 adds heads, FFN, normalization, and blocks. Label simplified numerical worksheets as such; they do not calculate the full pre-norm stack. That stack includes final LayerNorm before the vocabulary head.
- Show the forward pass and generation before learning. For backpropagation, show the true branching graph and short autograd code, not hand-derived Jacobians. Parameters are updated; intermediate Q/K/V, weights, and messages are recomputed.

## The numerical model and its limits

Part 1 uses the trained name-model data in `toy1.json`. Parts 2 and 3 share the hand-designed single-head attention model: `d_model=5`, `d_k=3`, `d_v=2`, and a 20-token vocabulary. The worked river/cheque prefixes contain ten tokens. Twenty same-width position vectors support generation beyond those prefixes; inputs beyond the supported capacity must fail clearly, not silently receive zero positions.

The five illustrative representation axes are water, finance, person, glue, and position. **The hand-designed attention projections and vocabulary head ignore the position axis.** Position remains in the residual row, but this toy does not demonstrate learned positional use or general order sensitivity. The causal mask still restricts each row to its prefix. Do not turn this limitation into a claim that positions are unnecessary or that real embeddings reserve one coordinate for position.

Read numbers from `AT.model`, `AT.forward`, `AT.mlp`, or the stored `AT.train` results. Distinguish unrounded computation from displayed precision. `make_toy2.py` regenerates the base model; `train_part3.py` derives `toy3.json`; `gen_report.py` generates the numerical report. These are writing generators, not read-only tests. If parameters change, regenerate dependent data and assembled parts, then rerun numerical checks. Do not casually retrain `toy1.json` during a layout edit.

`toy3.json` retains the established four-decimal training arrays, but its two finite-difference spot checks retain full precision (ε = 1e−4). The separate `check_training.py` checks all 260 used scalar parameters at ε = 1e−5 and verifies that unused position rows do not affect this prefix or change in its SGD update. Do not round numerical-error diagnostics to the worksheet display precision.

## Slide-first, article-unfolded

Part IV follows `the river bank <eos>` to `la rive <eos>`. It has separate learned source/target tables and full-width
added positions, bidirectional encoder attention, causal decoder self-attention, cross-attention, output projections,
residuals, and a vocabulary head. Its three-dimensional one-head toy omits FFN, LayerNorm, and dropout. Cross queries
come from decoder rows after self-attention; cross keys/values come from encoded source rows. The source length is four
and the teacher-forced target length is three, giving a 3 by 4 cross-attention matrix.

`train_part4.py --check` reproduces the saved two-pair fitted checkpoint and one further river-only SGD step. `check_part4.mjs`
independently checks JS tensors, gradients, causal prefixes, source sensitivity, and actual greedy generation. The one step
reduces the river example's mean loss, not every position's loss or the financial example's loss. Keep that caveat.

Every section uses explicit `.frame` wrappers with a `data-title`, optional `data-build` reveals, and `text/x-notes` presenter cues. Presentation uses a fixed logical **1280 × 720** stage, uniformly scaled to the available screen. Classroom typography is larger than article typography; the current tokens are 28px body, 42px heading, 22px captions, and 32px math.

Frames must not contain nested scrollbars or shrink their own text to fit. Split an overfull idea into a continuation frame. The live warning and `AT.present.preflight()` identify overflow, including open reveals and managed stepper states. Do not suppress those warnings or hide overflowing content.

Reading mode unfolds the same frames and widgets into responsive longform. `.companion` holds article-only explanation and detailed worksheets; it is hidden in presentation and lean reading. Preserve widget IDs/listeners when moving content. Test both modes and intermediate states, not only the fully revealed default slide.

Table cells have semantic `auto`, `number`, `text`, and `code` kinds. Do not put prose back into right-aligned numeric
styling. Use the central table API and `check_tables.mjs` for regressions; worksheet headers are intentionally larger in
presentation. Managed stepper toolbars are hidden in presentation because the global toolbar already advances them;
manual widgets retain their local controls.

Use **P** to present, arrows to advance, **Esc** to return to the same article section. Bare `#sNN` links are reading anchors; `?present#sNN/frame/build` is a classroom deep link. See `PRESENT.md` for controls, state restoration, and presenter view.

Focused sliders keep native arrow, Page Up/Down, and Home/End behaviour. **N** advances without changing a slider; **Esc** first returns focus to the slide. Test these interaction states as well as passive frame fit.

## Build, verify, and export

### Vision models and provenance

`VISION_SOURCE_AUDIT.md` records the original four articles and the important corrections. Keep their original files untouched; the adapted release lives in this repository. All core worksheets work without network/model downloads.

`vision-scene.js` carries one generated tabletop scene through all four vision lessons. `assemble.py` embeds its two
JPEG assets from `figures/vision-scene/`; no runtime fetch or image server is needed. See that directory's README for
the prompts, edit limitations and exact crop coordinates. The scene motivates a task and returns after its worked
example. The scene never borrows scores, attention maps or predictions from the separate 4×4 toy.

The photo-style figures use one shared type/layout component. Narrow reading views stack the same crops and captions;
the classroom view keeps the comparison on one 16:9 stage. Scope SVG sizing rules to direct children so they do not
override nested crop viewports. `check_vision_scene.mjs` verifies image decoding, crop transforms, label bounds, offline
loading and mobile containment at every scene frame. It also records representative classroom and phone screenshots.

- `AT.vision`: exact four-patch + CLS worksheet; full ViT pre-norm architecture is clearly separate from the no-LN/no-FFN numerical model. Position addition is same-width. Token permutation without positions is equivariant; the CLS readout is invariant to patch-only permutation.
- `AT.vision.story`: image crops stay attached to representations through 21 SVG stages. Important operations use separate authored frames so their workings survive ordinary PDF export. Projection columns, score/exponential/normalization stages, and value-only interventions are shown explicitly.
- `AT.vision.learning`: separate 44-parameter two-image classifier. Initial numbers match `AT.vision.forward()`; full-batch SGD uses learning rate 0.05 for 600 updates. The first step reduces mean loss but worsens the two-block example. `verify_vision1_learning.py` independently reproduces its tensors and checks every parameter gradient with NumPy and central differences. **Do not copy the fitted encoder over the initial frozen snapshot used by Vision IV.**
- `AT.visionSSL`: exact masked-pixel losses and illustrative teacher logits. These are not trained MAE, DINO, or I-JEPA outputs. EMA and stop-gradient roles must remain explicit.
- `AT.mae`: a separate, genuinely trained 72-parameter MAE-style model in `part6-learning.js`. Eight repeated-tile images, four single-patch masks each, 800 full-batch SGD updates at 0.05. `mae6.json` stores checkpoints; `part6-learning-view.js` draws actual predictions. Six held-out brightnesses/image combinations test interpolation only. The visible-tile average baseline is exact on this deliberately repetitive dataset. Frozen-encoder probes are already 100% accurate before reconstruction training, so do not claim classification improves. `train_vision2.mjs --check` reproduces 881 saved numbers; `check_vision2_learning.mjs` checks 288 finite differences and hidden-pixel leakage. The training implementation uses explicit JavaScript gradients; the classroom snippets show the equivalent PyTorch operations.
- `AT.clip`: three trained image–caption pairs, 16×3 pixel map, bag-of-words text map, unit normalization, symmetric loss, and a learned logit scale. `train_vision3.py --check` reproduces training; `check_vision3.mjs` checks the normalization derivative and simultaneous updates. Candidate softmax is not calibrated truth confidence. `checkpoint(0|1|20|60)` reproduces the plotted directions; `projectDirection` is a fixed orthographic camera, never refitted to a checkpoint. Initial normalization uses an exact zero-third-coordinate slice. Probability bars and the learned-directions/initial-temperature comparison isolate scale from geometry. The classroom has 35 frames; static exports retain initial, one-step, and 60-step vector views. Do not infer true 3D angles from their foreshortened screen view.
- `AT.vlm`: reuses Vision I's fixed encoder, discards CLS only after image attention, then uses a 2×3 bridge and a width-three decoder. Image rows read only image rows; text reads every image plus current/earlier text. No future-answer route is allowed. The prompt's final `?` row predicts the first answer token; subsequent generated tokens provide the new last query. `train_vision4.py --check` reproduces two-pair fitting and a further SGD step. That one-example step **harms the other image's answer**; keep the visible before/after regression rather than claiming universal improvement.
- `AT.vlm.contributions`: exact per-source weighted values, output-projected terms, and decomposition of `logit(two) - logit(one)` for the single-block linear-head toy. This is forward-pass bookkeeping, not a raw-pixel causal attribution. The visual-slot diagram preserves the full attention denominator, including text. The signed image terms favor `one` for both fitted examples; do not replace them with a generic object-localization story. Generation figures retain the same image while updating the last known token/query.

These tiny fitted models do not establish transfer to new images or text. Full-model claims are scoped to cited papers. Values remain attention projections; CLIP's global embeddings use `g_img/g_txt`, and VLM visual memory uses `G`.

The vision checkers have optional `--browser` modes; inspect each header for arguments. Use the shared `frame_audit.mjs`, `check_tables.mjs`, and `export_slides.mjs` on all four `visionN.html` outputs. Save PDFs as `output/pdf/vision-partN-slides.pdf` (ignored by Git).

### Commands

Run these from the repository root. Reuse existing Python/NumPy and Playwright environments; browser tools should not install dependencies. In a restricted agent sandbox, Chromium's macOS Mach-port error requires an approved browser test run outside that sandbox, not a claimed pass.

```sh
python3 src/assemble.py --part 1 --out part1.html
python3 src/assemble.py --part 2 --out attention.html
python3 src/assemble.py --part 3 --out part3.html
python3 src/assemble.py --part 4 --out part4.html
python3 src/assemble.py --part 5 --out vision1.html
python3 src/assemble.py --part 6 --out vision2.html
python3 src/assemble.py --part 7 --out vision3.html
python3 src/assemble.py --part 8 --out vision4.html

python3 src/check_metadata.py
node src/check_part1.mjs
node src/check_part1_diagrams.mjs
python3 src/train_part4.py --check
node src/check_part4.mjs
node src/check_vision1.mjs
python3 src/verify_vision1_learning.py
node src/check_vision2.mjs
node src/check_vision2_learning.mjs
node src/train_vision2.mjs --check
python3 src/train_vision3.py --check
node src/check_vision3.mjs
python3 src/train_vision4.py --check
node src/check_vision4.mjs
node src/toy_ref.mjs src/toy.json --compare src/py_check.json
python3 src/check_training.py
node src/check-live-model.mjs attention.html
node src/check_position_capacity.mjs attention.html part3.html
node src/check-routing-scaling.mjs
node src/check-diagram.mjs attention.html

node src/pres_test.mjs
node src/interaction_test.mjs
node src/export_test.mjs
node src/frame_audit.mjs part1.html
node src/frame_audit.mjs attention.html
node src/frame_audit.mjs part3.html
node src/frame_audit.mjs part4.html
node src/frame_audit.mjs vision1.html
node src/frame_audit.mjs vision2.html
node src/frame_audit.mjs vision3.html
node src/frame_audit.mjs vision4.html
node src/check_tables.mjs part1.html attention.html part3.html part4.html
node src/check_tables.mjs vision1.html vision2.html vision3.html vision4.html
node src/qa.mjs attention.html --width 1280 --height 720
node src/qa.mjs attention.html --width 390 --height 844
node src/sweep.mjs attention.html
git diff --check
```

Repeat article QA and interaction sweeps for each changed part. `check_training.py` independently checks saved training results and all used parameter gradients without writing data. `check_position_capacity.mjs` checks the exactly tied baseline candidates, 10/11/20-token evaluation, invalid-input rejection, and generation with the new last-position query. `frame_audit.mjs` walks presentation states and can save failure screenshots with `--shots /tmp/attention-overflow`. `interaction_test.mjs` opens arithmetic dialogs, changes masks, exercises focused controls, and displays presenter notes. Inspect representative screenshots and PDF pages as well as test output. Record actual commands/results in `CLASSROOM_QA.md`; this handover does not certify an unrun release.

```sh
# Exact slide views: one fully revealed page per authored frame
node src/export_slides.mjs attention.html output/pdf/attention-part2-slides.pdf

# Every build and managed stepper state as a separate page
node src/export_slides.mjs attention.html output/pdf/attention-part2-builds.pdf --builds all
```

The exporter captures the actual 16:9 stage without navigation, preflights fit, and refuses overfull frames. Default scale is 2×; `--scale 1|2|3` changes raster resolution and `--frames DIR` retains PNGs. PDFs preserve browser/SVG appearance as images, so text is not selectable and widgets are no longer interactive. Final export advances authored builds and managed steppers and opens reveal answers. Other manual widgets keep their current states. Browser Print is a separate reading-oriented handout, not this exact slide export.

## Publishing and parallel work

The user requested regular GitHub checkpoints. Verify the checkout, branch, remote, existing edits, and available GitHub authentication first. Preserve unrelated work. Stage only reviewed task files with explicit paths, inspect the staged diff, then commit and push after the relevant checks pass. Never use broad `git add -A` as a handover shortcut. Verify the pushed commit and Pages status before calling a checkpoint published.

Give each parallel agent an explicit file boundary and a bounded task. Shared runtime changes affect both series and require all-part fit checks. Ask for changed files, tests actually run, screenshots inspected, and remaining limitations. Keep temporary previews outside the source tree and do not overwrite another agent's changes.

## 7. Vision rebuild (2026-09-05)
The four vision parts are being rebuilt to the standard of Part 2. Design: `VISION_FEEDBACK.md` (diagnosis and per-part plans) and `VISION_AXES.md`
(one 8×8 scene with named regions, a fixed patch encoder with named axes brightness / contrast / row / col, keys "bright region? / on the right?",
values "sends: brightness / sends: contrast", the shared figures). Foundation files: `vision-shared.js` (scene, encoder, grid, thumb, overlay, scatter,
circle, triptych, curve; injected before partN.js for parts 5 to 8), `make_vision_toy.py` → `toy5.json` (Vision I asks "is there a mug on the right half?";
scene C is the generalisation probe), `part5.js` (Vision I runtime), `sections5/sec00_vision_demo.html` (component gallery, not shipped), and the
"Vision components" section of `CONTRACT.md`. Per-part briefs: `TASK_VISION_B.md` (Vision I), `_C` (II), `_D` (III), `_E` (IV); parts 6 to 8 copy
`toy5.json.trained` into their toy as `encoder` and define `V.encode` in their runtime. Rebuild acceptance: named axes everywhere, thumbnails as row
labels, one figure per section that is not a table, an end-to-end run with a loss curve, one caveat box per part, clean qa/sweep/mathdiff/walk.
