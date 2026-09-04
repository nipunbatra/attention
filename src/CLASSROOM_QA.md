# Three-part classroom release checks

## 2026-09-04: tone, whitespace, and PDF answers

- Humanizer pass: all 54 section files in Parts I–III; computation and notation retained and embedded scripts parsed.
- No persistent presentation bars. One heading per slide, quiet margins, and on-demand keyboard-accessible controls.
- Presentation regression suite passes: hidden-control focus, scale invariance on opening controls, navigation, overview,
  presenter window, print reveal/restore, mobile stage, live fit diagnostics, and preflight restoration.
- All 563 existing live states pass strict maths and fit checks: Part I 132, Part II 297, Part III 134.
- Table flow: 507 desktop/mobile/presentation table views checked. No errors or alignment/overflow failures; two existing
  narrow-prose-column advisories remain in Part II's comparison tables.
- `export_test.mjs`: final/all-build × shown/authored answer modes, 12 PDF pages, actual PNG answer pixels checked.
  Default exports show quiz answers on completed frames; earlier all-build pages retain the question.
- Rebuilt PDFs: Part I 61 pages (6 newly opened answers), Part II 133 (11), Part III 68 (13).
- Part I arithmetic and sampler match reference values. Part II routing/scaling checks pass, including value-only changes,
  independent score normalization, mobile width, ≤4-line code snippets, and no JS/KaTeX errors.

## Previous release record

Verified locally on 2026-09-03. All three parts use the same slide-first runtime. Each frame fits a logical 1280 × 720 stage; the article unfolds the same content and its companion explanations. Exact-slide PDF exports contain no browser header or footer.

## Classroom and reading checks

| Part | Authored frames / PDF pages | Presentation states walked | Unique rendered formulas reparsed |
|---|---:|---:|---:|
| 1: characters to prediction | 61 | 132 | 79 |
| 2: self-attention | 133 | 297 | 487 |
| 3: learning and Transformer blocks | 68 | 134 | 140 |

- All 563 states passed: no stage/frame overflow, nested scrolling, JavaScript errors, or invalid rendered formulas. Preflight also checks open reveals and intermediate managed steps. The walkthrough explicitly starts at frame 1, independent of the article's current scroll position.
- Strict math validation reparses each unique rendered formula with KaTeX `throwOnError: true`. A regression fixture confirmed that unknown commands, malformed math, and fallback text fail the audit, including commands introduced only in a later step.
- Reading layouts fit 390px phone width without horizontal document overflow. Interaction sweeps exercised 32 controls in Part 1, 128 in Part 2, and 34 in Part 3 without errors.
- Runtime regression checks passed for forward/back navigation, multiple managed steppers, manual-widget state/focus/selection, pending-control accessibility, overview focus, presenter view, fullscreen fallback, print restoration, deep links, and exit/reload. Managed steppers use the global presentation controls; manual widgets retain their local toolbar.
- Title-only overflow, native disclosure changes, and intermediate-step disclosure overflow are detected. Preflight preserves edited manual widgets rather than rebuilding their DOM.
- Representative diagram, dense-table, intro, and continuation frames were inspected visually. The browser header and footer are compact; body/title/caption/math sizes are 28/42/22/32 logical pixels.

## Table and Part 1 diagram refinement

The dedicated table audit sampled 507 tables across desktop reading, 390px phone reading, and presentation. It checks cell-text overlap, table overlap, semantic numeric/text/code styling, header sizes, contained mobile scrolling, and runtime errors. Classification, update, footer, and colour-tint regression fixtures pass for all three assembled parts. Numeric precision was preserved; prose no longer inherits right-aligned number styling. Dense token headers and projection columns were visually checked, and 42 targeted screenshots were saved during review. Intentional row-label wrapping remains readable.

Four Part 1 diagrams adapt the original handwritten notes' visual sequence: the learned embedding scatter, repeated lookup to ordered concatenation, the prediction/learning graph, and the two training/generation loops. The regression creates 30 SVG instances and checks all 662 labels against their viewBoxes, all 184 arrow-marker references against their owning SVG, unique IDs, and accessible titles/descriptions. It verifies all 27 plotted embedding coordinates, six concatenated values, model shapes, target probability/loss, and a reproducible sample of `i` followed by the context `a b i`. A sampled boundary stops generation. Drawing diagrams does not mutate model parameters; the observed target has its own loss branch and is not an input.

Part 1 one-hot worksheets use the neutral label `products`, not query/key notation. Diagram stages use the full classroom width. Wide diagrams and tables may pan within a bounded container in phone reading mode; presentation frames never scroll.

## Numerical and conceptual checks

| Check | Result |
|---|---|
| Part 1 forward pass | All six saved example rows agree with the JS model; maximum error 1.67e−16 |
| Part 1 generation | Displayed probabilities and sampler agree at four temperatures |
| Part 2 reference versus Python | 6,615 values; maximum error 3.55e−15; all hard targets pass |
| Live article versus reference | Eight original cases; 8,220 finite values and 180 masked infinities; maximum error 1.34e−15 |
| Position capacity and generation | 48 cases at lengths 10, 11, and 20 across both attention pages; 84,968 finite values and 2,680 masked infinities; maximum error 1.34e−15 |
| Invalid inputs | Twelve checks reject unsupported length, unknown tokens, and missing, wrong-width, or non-finite position vectors |
| Part 3 generation UI | Appends `water`, evaluates position 11, and displays the correct distribution for position 12 |
| Stored training results | All 1,678 saved numeric entries match recomputation at stored precision |
| Backward implementation | All 260 used parameter gradients checked by central differences; maximum absolute error 4.51e−11 |
| Diagram evidence | 307 comparisons; shapes, causal zeros, normalization, prefix equivalence, provenance, and deep freeze pass |
| Staged diagram | All twelve stages fit; receiver changes from bank at position 7 to the final token at position 10 |
| Routing/scaling | Value-only intervention leaves weights unchanged; divisor/softmax calculations and variance simulation pass |

The review also corrected the incomplete vocabulary-softmax worksheet, vocabulary bias shape, residual-gradient graph, LayerNorm epsilon, final pre-norm LayerNorm, and a future-leaking multi-layer example. The three-part notation separates the learned vocabulary table `E_tok` from the sequence stack `E`, values from embeddings, message matrix `H` from causal mask `M`, and attention weights from vocabulary probabilities. Head-message equations are introduced before concatenation.

The learner-facing backpropagation explanation uses the branched computation graph and autograd, not a manual Jacobian derivation. Numerical gradients remain a behind-the-scenes regression test.

## PDF verification

The exact-slide exporter produced 61-, 133-, and 68-page PDFs at 2× resolution. Every page has a 16:9 media box and an embedded slide image. First, middle, and last pages of each exported PDF were rendered and visually inspected, along with the new Part 1 diagrams and representative changed tables.

The exporter also passed progressive-build and multiple-stepper fixtures, and rejected an intentionally overfull frame instead of writing a clipped PDF. `--builds all` records each managed step; default export records the final state of each authored frame. Sliders, quizzes, and manual disclosures remain at their authored defaults.

These PDFs are appearance-faithful raster snapshots: text is not selectable, and controls are no longer interactive. Local PDF outputs are ignored by Git and can be regenerated from the committed sources.

## Deliberate teaching boundaries

Part 1 uses a trained small name model. Part 2 is a hand-designed, single-head model with widths 5/3/2 and a 20-token vocabulary, not a trained language model. Part 3 demonstrates actual updates to that toy before explaining a larger Transformer architecture.

The toy has twenty same-width position vectors. Its initial projection/head matrices ignore the dedicated position coordinate. Therefore it demonstrates content routing, not word-order sensitivity; this is now explained explicitly in the introduction and summary. Position is added, not appended as an obligatory extra dimension. The causal mask still restricts available sources.

Full Transformer diagrams are architectural explanations. The original one-head numerical worksheet does not silently become a multi-head, normalized, stacked model. This release is a checked teaching implementation, not a claim of exhaustive browser/accessibility coverage or an absence of every possible pedagogical improvement.

## Reproduce from the repository root

```sh
python3 src/assemble.py --part 1 --out part1.html
python3 src/assemble.py --part 2 --out attention.html
python3 src/assemble.py --part 3 --out part3.html
node src/check_part1.mjs
node src/check_part1_diagrams.mjs part1.html
node src/toy_ref.mjs src/toy.json --compare src/py_check.json
node src/check-live-model.mjs attention.html
python3 src/check_training.py
node src/check_position_capacity.mjs
node src/check-routing-scaling.mjs
node src/check-diagram.mjs attention.html
node src/pres_test.mjs
node src/frame_audit.mjs part1.html
node src/frame_audit.mjs attention.html
node src/frame_audit.mjs part3.html
node src/check_tables.mjs part1.html attention.html part3.html
node src/qa.mjs attention.html --width 390 --height 844
node src/sweep.mjs attention.html
node figures/attention-diagram-preview/check-data.mjs
node src/export_slides.mjs attention.html output/pdf/attention-part2-slides.pdf
git diff --check
```

Repeat article QA, interaction sweeps, and PDF export for each changed part. Check the GitHub Pages workflow before assuming the latest pushed source is live.
