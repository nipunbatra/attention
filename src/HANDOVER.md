# Handover: three-part interactive teaching series

Updated 2026-09-03. Owner: Nipun Batra.

- Repository: https://github.com/nipunbatra/attention
- Published series: https://nipunbatra.github.io/attention/
- Canonical local checkout on this Mac: `/Users/nipun/git/attention`.

All three parts are implemented. They share a slide-first reading/presentation system, not separate article and slide sources. The first slide-first checkpoint was `a49f811`. This handover accompanies the follow-up Part 3 and correctness checkpoint; `CLASSROOM_QA.md` records its completed local verification. Check the checkout's Git log and the GitHub Pages workflow for the current published commit. A temporary checkout is not evidence of what is live.

## Start here

Read this file, then `PRESENT.md` for the current layout/runtime contract and `CLASSROOM_QA.md` for recorded verification. Inspect the actual section and shared runtime before editing.

`SPEC.md`, `BRIEF.md`, `GUIDE.md`, `GUIDE1.md`, `GUIDE3.md`, `AXES.md`, `PARTS.md`, the older feedback files, and `REV2_TASK.md` preserve useful design history. Some contain superseded notation, planned work that is now done, or old layout assumptions. They are historical guidance, not instructions to undo the current implementation. `CONTRACT.md` documents components; confirm its examples against `shared.js` when they differ.

## Source and outputs

| Part | Editable sections | Data/runtime | Assembled output |
|---|---|---|---|
| 1: characters to prediction | `sections1/secNN.html` | `toy1.json`, `part1.js`, `part1.json` | `../part1.html` |
| 2: self-attention | `sections/secNN.html` | `toy.json`, `part2.json` | `../attention.html` |
| 3: learning and Transformer blocks | `sections3/secNN.html` | `toy3.json`, `part3.js`, `part3.json` | `../part3.html` |

Paths in this table are relative to `src/`. Shared files are `shell.html` (layout/CSS), `shared.js` (math, widgets, notation, presentation runtime), and `assemble.py`. Do not edit the generated root HTML or the inlined `katex-bundle.html` by hand. `index.html` is the series landing page; `part2.html` redirects to `attention.html`.

The standalone staged diagram lives in `figures/attention-diagram-preview/`. Part 2 embeds its same `diagram.js` source through `src/attention-flow-data.js`; keep the preview and article synchronized by changing that shared source.

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

## Slide-first, article-unfolded

Every section uses explicit `.frame` wrappers with a `data-title`, optional `data-build` reveals, and `text/x-notes` presenter cues. Presentation uses a fixed logical **1280 × 720** stage, uniformly scaled to the available screen. Classroom typography is larger than article typography; the current tokens are 28px body, 42px heading, 22px captions, and 32px math.

Frames must not contain nested scrollbars or shrink their own text to fit. Split an overfull idea into a continuation frame. The live warning and `AT.present.preflight()` identify overflow, including open reveals and managed stepper states. Do not suppress those warnings or hide overflowing content.

Reading mode unfolds the same frames and widgets into responsive longform. `.companion` holds article-only explanation and detailed worksheets; it is hidden in presentation and lean reading. Preserve widget IDs/listeners when moving content. Test both modes and intermediate states, not only the fully revealed default slide.

Use **P** to present, arrows to advance, **Esc** to return to the same article section. Bare `#sNN` links are reading anchors; `?present#sNN/frame/build` is a classroom deep link. See `PRESENT.md` for controls, state restoration, and presenter view.

## Build, verify, and export

Run these from the repository root. Reuse existing Python/NumPy and Playwright environments; browser tools should not install dependencies. In a restricted agent sandbox, Chromium's macOS Mach-port error requires an approved browser test run outside that sandbox, not a claimed pass.

```sh
python3 src/assemble.py --part 1 --out part1.html
python3 src/assemble.py --part 2 --out attention.html
python3 src/assemble.py --part 3 --out part3.html

node src/check_part1.mjs
node src/toy_ref.mjs src/toy.json --compare src/py_check.json
python3 src/check_training.py
node src/check-live-model.mjs attention.html
node src/check_position_capacity.mjs attention.html part3.html
node src/check-routing-scaling.mjs
node src/check-diagram.mjs attention.html

node src/pres_test.mjs
node src/frame_audit.mjs part1.html
node src/frame_audit.mjs attention.html
node src/frame_audit.mjs part3.html
node src/qa.mjs attention.html --width 1280 --height 720
node src/qa.mjs attention.html --width 390 --height 844
node src/sweep.mjs attention.html
git diff --check
```

Repeat article QA and interaction sweeps for each changed part. `check_training.py` independently checks saved training results and all used parameter gradients without writing data. `check_position_capacity.mjs` checks 10/11/20-token evaluation, invalid-input rejection, and generation with the new last-position query. `frame_audit.mjs` walks presentation states and can save failure screenshots with `--shots /tmp/attention-overflow`. Inspect representative screenshots and PDF pages as well as test output. Record actual commands/results in `CLASSROOM_QA.md`; this handover does not certify an unrun release.

```sh
# Exact slide views: one fully revealed page per authored frame
node src/export_slides.mjs attention.html output/pdf/attention-part2-slides.pdf

# Every build and managed stepper state as a separate page
node src/export_slides.mjs attention.html output/pdf/attention-part2-builds.pdf --builds all
```

The exporter captures the actual 16:9 stage without navigation, preflights fit, and refuses overfull frames. Default scale is 2×; `--scale 1|2|3` changes raster resolution and `--frames DIR` retains PNGs. PDFs preserve browser/SVG appearance as images, so text is not selectable and widgets are no longer interactive. Final export advances authored builds and managed steppers; sliders, quizzes, and manual disclosures retain their authored defaults. Browser Print is a separate reading-oriented handout, not this exact slide export.

## Publishing and parallel work

The user requested regular GitHub checkpoints. Verify the checkout, branch, remote, existing edits, and available GitHub authentication first. Preserve unrelated work. Stage only reviewed task files with explicit paths, inspect the staged diff, then commit and push after the relevant checks pass. Never use broad `git add -A` as a handover shortcut. Verify the pushed commit and Pages status before calling a checkpoint published.

Give each parallel agent an explicit file boundary and a bounded task. Shared runtime changes affect all three parts and require all-part fit checks. Ask for changed files, tests actually run, screenshots inspected, and remaining limitations. Keep temporary previews outside the source tree and do not overwrite another agent's changes.
