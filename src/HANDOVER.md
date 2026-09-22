# Handover: interactive teaching series

Updated 2026-09-22. Owner: Nipun Batra.

- Repository: https://github.com/nipunbatra/attention
- Published series: https://nipunbatra.github.io/attention/
- Canonical local checkout on this Mac: `/Users/nipun/git/attention`.

All four attention parts and the four-part Vision to language extension are implemented. They share a slide-first reading/presentation system, not separate article and slide sources. The first slide-first checkpoint was `a49f811`; `a1d609c` completed Part 3 and the numerical-correctness pass. `CLASSROOM_QA.md` records local verification. Check the checkout's Git log and the GitHub Pages workflow for the current published commit. A temporary checkout is not evidence of what is live.

## Start here

2026-09-22 document-boundary clarification:

- BOS and EOS wrap each complete story/document once, not each sentence or
  training window. The Lily sentence is a complete toy document. The boundary
  slide states this rule, and Notebook 5 runs a two-sentence story with a single
  BOS/EOS pair. Keep `boundaries=True` explicit and the original eight toy IDs
  and seven targets unchanged. Other datasets may use different conventions.

2026-09-22 unknown-token clarification:

- Keep the special-token slide's explicit result: blue is absent from the toy
  vocabulary, so encode_tokens returns [3], the UNK ID. boundaries=False omits
  BOS/EOS. Notebook 5 also runs boundaries=True to show [1,3,2] and distinguishes
  token-to-ID lookup from embedding lookup. The 88-step order is unchanged.
- After build_slow_lesson.py, execute Notebook 5 with nbconvert in an isolated
  JUPYTER_CONFIG_DIR, then call export_bundle(ROOT) to refresh the ZIP with the
  fully executed notebook. The builder alone leaves the setup cell unexecuted.

2026-09-22 position-section pacing and limitations (current structure):

- The position lesson has 42 presentation frames instead of 49. Its five
  topic pauses introduce the motivation, adding position to a word, Method 1
  (learned positions), Method 2 (sinusoidal positions), and relative positions.
  Keep the numerical examples within their topic blocks.
- Three frames now motivate the relative-position divider before it appears:
  a four-row learned table has no rows 4 and 5, the same today/Ravi pair shifts
  five slots when a prefix is added, and the held-fixed additive score changes
  from 2.366 to 0.500 despite the gap staying one. The relative-bias table reuses
  exactly the same receiver/source indices (3,2) and (8,7), followed by ALiBi
  and RoPE. Preserve the distinction between table coverage and same-gap
  matching; relative methods do not guarantee arbitrary-length predictions.
  The explanatory cross-term algebra remains in reading notes.
- Nine recap/reference frames are now reading companions with headings:
  the second addition example, concat-plus-projection, Q/K/V routing, static
  clock comparison, sinusoid row table, rotary score table, mean pooling,
  length extension and the methods summary. Their calculations and caveats
  remain available in reading mode; do not restore them as duplicate slides.
- ALiBi immediately follows the general distance-bias formula, before RoPE.
  The closing attention map appears once. Its accessible selector highlights
  input, matching/mixing or update/prediction without moving any nodes.
  Keep the map on learned additive positions to match the notebook.

2026-09-22 appended-position example:

- The former abstract `s17-position-append` frame is now a five-slide Maya/Ravi
  calculation: appended rows, word/slot score contributions, full softmax,
  an interactive scale comparison, and design tradeoffs. The new helper
  `AT.positionLesson.appendedExperiment` uses the existing two-coordinate word
  rows plus `c*i`, with identity Q/K projections and matching width three.
  It deliberately does not add the earlier two-dimensional position offsets.
- At c=1, today's weights are 92.3% and 89.7%; at c=0.1, they are 28.9% and
  28.8%. These are untrained scale diagnostics, not model-quality evidence.
  Preserve the visible caveat that appropriate scaling and learned projections
  can make concatenation work. Addition is a convenient fixed-width choice,
  not a theorem that concatenation is inferior.
- Section 17 consistently uses e for the representation entering attention,
  E for its stacked rows, and delta e for the contextual update. The word lookup
  is E_tok[t_i], and the position offset remains p_i. Keep this distinction in
  the numeric table, residual diagram and full position-aware attention map.

2026-09-21 slower lab and position revision (supersedes the entries below):

- Section 19 has 88 walkthrough frames plus its opening topic break. After the
  data introduction, two slides show three real TinyStories documents: one
  complete 52-word story, then excerpts from 107- and 248-word stories. Full-story
  token counts are computed with the notebook tokenizer; the shared source and
  downloadable bundle include three attributed texts in `story_examples.json`.
  One authored six-token sentence yields seven targets; a B=2, w=4, C=10 batch is
  traced through the MLP, attention, loss, an SGD update and generation. Random
  toy arithmetic is explicitly separate from the saved TinyStories benchmark.
- Twenty visible map checkpoints precede the relevant calculations, including
  splitting, tokenization, boundaries, window/batch preparation, MLP layers,
  attention operations, learning and generation. `ROUTE_CHECKPOINTS` in
  `slow_walkthrough.py` controls their order. They execute no model code.
  Keep node locations fixed across highlight states and regenerate all slide links.
  The new `lookup-flow` figure traces X[1] through selected rows of T [C,d]
  into E[1], then labels the full batch E [B,w,d]. Every value is computed.
  MLP logits now feed the loss vertically; observed y follows an outer lane.
  A footer explains reusing updated parameters instead of a crossing return arrow.
  Each SVG uses its own colored arrow markers so hidden frames cannot hide them.
- Before `tokenize(sentence)`, three short tokenization frames explain the unit
  of prediction, compare word/character/illustrative subword splits of `redder!`,
  and demonstrate the actual notebook tokenizer. The first is a topic break.
  Preserve the distinction between tokens, IDs and embeddings, the context-token
  count, training-only vocabulary fitting and fixed rules at generation time.
  Subword splits are illustrative, not measured BPE output. The English teaching
  tokenizer intentionally loses case and spacing and is not multilingual.
- Window preparation explicitly traces `context_ids` and `target_id` into the
  two growing Python lists before conversion to `all_X` and `all_y`. The first
  two appends are separate frames, then the loop continues at position 3.
  Preserve both visible append calls and the position/index/ID distinction.
  Eight additional frames keep the seven toy pairs and all model arithmetic
  unchanged. The batch figure maps dataset rows 2 and 3 to batch rows 0 and 1.
  It also decodes both inputs and targets beside their IDs. The following
  `batch-ids` frame distinguishes the integer tensors from their shapes and
  makes clear that tokenization is complete but embedding lookup comes later.
  Only X goes through input embedding lookup; y stays as target IDs for the loss.
- Slide code now has an explicit excerpt (or no code) for every walkthrough
  stage. Never restore first-N-line slicing. Counting and generation have six
  further frames, each showing a complete operation. `code_display.py` uses
  Python's tokenizer for offline syntax colors and rejects incomplete snippets.
  The Part II assembler also highlights the earlier PyTorch code blocks;
  `python-code.js` preserves the same colors in the two live steppers. Both
  work offline. Keep full notebook code/figures synchronized and check rendered
  snippet text, including after changing interactive steps.
- Use short, descriptive walkthrough titles. The humanizer pass replaces long
  contrasts such as the former examples-versus-updates heading with
  "Seven examples, four batches". Keep the conditional one-update-per-batch
  explanation in the body, along with the separate benchmark sampling rule.
- The `shapes` frame uses two tables: data dimensions (B, w, C) and model
  dimensions (d, h, dₖ, dᵥ). Keep one symbol, value and definition per row;
  slash-separated pairs look like division. Values come from the executed
  notebook namespace, and the shared figure keeps the guide and slide aligned.
- `notebooks/wordlm/05_training_and_inference_maps.html` is the public illustrated
  companion. Each step links to its exact slide and shares its numeric SVG with
  the executable notebook. `slow_walkthrough.py` and `build_slow_lesson.py` are
  the shared authoring source. All five notebooks and their support files are
  downloadable; the full raw corpus and the owner-private Site are excluded.
- All 28 detailed cost frames moved to `sections3/sec16_cost.html` and are
  included in Part III Section 16, after training and cached generation. Part II
  Section 16 retains the matrix calculation, with a reading link to Part III.
- Section 17 now has 42 position frames. Its opening computes both sentence
  orders explicitly: query, scores, exponentials, normalized weights, weighted
  values and the identical message. The independent toy's `today` row is
  `[0.8,0.2]`, so Maya and Ravi have unequal weights. It then adds positions and
  computes why Maya's score changes with its slot. The topic break says
  "Positional encoding" and visibly credits both videos. Before the addition
  table, equal-axis SVG plots show all four original word dots. Four reversible
  builds move each sentence's dots by the exact slot offsets used in the table.
  The positioned rows, messages and residual updates are calculated from the
  same data. Position offsets and contextual updates are explicitly separate.
- Adjustable clocks expose collisions at indices 4 and 12. Fast/slow wave plots
  and a full slow cycle precede the sinusoidal formula and numeric construction.
  Integer-period toy clocks are distinct from the standard radian frequencies.
  An additive common-shift counterexample motivates the rotation and relative
  dot-product demonstration. The section closes with one full position-aware
  attention map with selectable focus states, before the TinyStories walkthrough.
  `position-visuals.js` and `position-journey.js` supply the original SVGs.
  Serrano and Huang are credited; mathematical checks use the papers. New
  builds are authored in the HTML before presenter boot, then populated by JS.
- Preserve the caveats: appending position features is valid, addition is not
  uniquely invertible, the one-layer final-row swap example is scoped, RoPE
  rotates Q/K rather than V, and a formula accepting longer indices does not
  establish reliable longer-context behavior.

Use `check_wordlm_pipeline.mjs`, `check_wordlm_companion.mjs`,
`check_cost_position.mjs attention.html part3.html`, and
`check_cost_networks.mjs part3.html` for these changes. The model, benchmark
parameters and saved measurements were not changed.

2026-09-21 diagram-first notebook walkthrough: Section 19 now has 16 additional
`pipeline-lesson` frames after the training/generation introduction. The source
is `sections/sec19_pipeline.html`, included at `<!--WORDLM_PIPELINE-->`.
Four master SVGs in `figures/wordlm-pipeline/` cover MLP/attention ×
training/inference; the editable Python generator is beside them. Focus views
keep the same node positions, and “Show full map” replaces the explanation
temporarily without overflowing the slide. Preserve the one-target-per-window
contract, final-query attention optimization, PAD-key mask, hidden prediction
layer, and the uncached generation loop. This is a separate trained TinyStories
companion, not a change to the hand-chosen `toy.json` model. The bounded three-seed
comparison and its provenance are in `benchmark-summary.json`.

The companion notebook work lives in the existing Word-level prediction task's
worktree, under `lecture11/notebooks/word-level-next-token`. Notebooks 1 and 3
now use these maps; notebook 5 is the complete sentence-to-loss and
checkpoint-to-generation trace. `pipeline_maps.py` is identical in both projects.
All five notebooks have fresh executed outputs. The owner-private notebook Site
was not modified or republished. Run `check_wordlm_pipeline.mjs` in addition to
the general frame audit; the core-tail regression excludes the explicit
`pipeline-lesson` extension, just as it excludes the cost/position extension.

2026-09-17 Part II cost and position extensions: 45 classroom frames follow the matrix-form calculation in Section 16 and the alternatives comparison in Section 17. The sources are `sections/sec16_cost.html` (28 frames) and `sections/sec17_positions.html` (17 frames), included by `assemble.py` at explicit markers in their owning sections. Existing section IDs and core frames are unchanged. The two new topic breaks are `s16-cost-break` and `s17-position-break`; the opening reading guide identifies these as a later teaching sitting.

The cost section now introduces each method visually before its operation table. Three comparable neural-network diagrams use the same ten-token prefix: concatenation selects the last three four-number rows (12 inputs), averaging makes four mean inputs, and attention projects a two-number message into a four-number update and adds the final input row. All retain the eight-unit ReLU hidden layer, both weight matrices/biases, twenty vocabulary logits and softmax. Two further editable SVGs expand attention into literal Q/K/V matrix shapes and the score/mix/output products. These are architecture sketches, not new fitted models or invented activations. `check_cost_networks.mjs` checks context selection, neuron/connection counts, matrix cells, residual routing, forward/reverse SVG reveals and label bounds. Keep the diagrams before the corresponding cost tables; retain the all-row versus final-query-only distinction. Phone reading scrolls the diagrams inside their own containers rather than shrinking their labels.

The cost lesson derives dense MACs for each product, compares fixed-window concatenation, cumulative averaging and attention on equal workloads, and separates all-position training/scoring, prompt prefill and cached generation. Preserve the important one-layer exception: Part II's final-only prediction can compute all K/V and just its final query, without forming every attention row. Stacked-model prefill is a different workload. The calculator counts operations, not runtime or measured speedups. Pair grids use the same cell scale so doubled side lengths show quadrupled area.

The position lesson includes an independent two-coordinate order-swap experiment, addition versus appended features, learned/sinusoidal absolute positions, relative score bias, RoPE and ALiBi. Appending features is valid; addition is not universally optimal. A plain mean of token-plus-position rows remains invariant to reassigning the same tokens to the same slots. Do not generalize the one-layer final-row invariance example to all stacked causal networks. Both extensions leave `toy.json`, shared model computations and Part III unchanged. `check_cost_position.mjs` checks independent arithmetic, every new build, reverse math reveals, controls, model immutability and projector/phone layouts. See the latest entry in `CLASSROOM_QA.md` for the verification record.

2026-09-14 Part III continuity revision: training now starts from the exact Part II 4→8→20 ReLU prediction MLP, including both matrices and biases. This supersedes the temporary separate linear-readout experiment. Saved SGD parameters and outputs keep full precision. The classroom path has 51 frames including the title, with three section breaks and repeated material retained as reading companions. The block FFN remains a separate illustrative 4→8→4 network; the full Transformer schematic has FFNs inside blocks and a linear readout after final LayerNorm. See GUIDE3.md, AXES.md, check_training.py, and check_part3_continuity.mjs.

2026-09-12 position revision: Parts II and III now share a four-coordinate toy, with no dedicated `pos` axis. The hand-chosen position table adds small offsets across the same word-feature coordinates. The opening defines the illustrative features before introducing position; every worked example, reference result, training trace and exported diagram must use this model. This supersedes historical five-coordinate/ignored-position guidance below and in REV2_TASK.md. See AXES.md and check_position_intro.mjs; T9 in both numerical references tests position-sensitive predictions.

The section 4 lookup primer now distinguishes the placeholder c (dark grey, pSymbol) from the literal "a"
(orange, pToken). Quoted monospaced literals carry through id("a") and the concrete embedding subscript. Generic
e_c = E_tok[id(c)] keeps both c occurrences grey. Matching prose explains the quotes and the character/value
distinction. The notation regression checks the quotes, distinct colours, and adjacent equations in article/print.

Section 14 now explores both window w and embedding width d. The coordinate-grid diagram concatenates w rows of d
numbers into one wd-wide activation, with explicit W1, hidden, W2, logits, and bias labels. Both controls keep their
state across navigation. The following worksheet recalculates all five parameter groups and their total. w affects
W1; d affects E_tok and W1. Hidden/vocabulary widths stay 32/27. These are architecture choices, never changes to the
saved trained model. A five-line PyTorch example uses w=5,d=4 and counts 1,671 parameters. The larger-width worked
example retains (100*256)*1024 = 26,214,400 W1 weights. Tests cover all 25 control pairs, shapes, counts, bounded labels,
keyboard controls, retained state, and trained-model immutability. There are still 33 snippets and 230 progressive states.

The abstract section 11 "Many windows or one growing output" table is replaced by concrete training and generation
frames. Training shows all six contexts/targets from aabid, six independent score rows, and one shared MLP batch.
The batch code now follows that picture and names both axes of X/logits plus the six target IDs in y. The generation
frame uses the actual seed-1 run from aab: draw h, shift to abh, draw stop, keep aabh. It explicitly distinguishes
observed i in the next training window from sampled h in generation, and notes that independent names can be batched.
Browser tests verify source pairs, dependency transitions, probabilities, boundary handling, and frame order. Torch
independently reproduces both new displayed probabilities. There are 230 progressive states and still 33 snippets.

Section 11's three-stage training/generation comparison now marks all five parameter groups explicitly. The training
MLP has an update symbol and an orange gradient/optimizer return arrow. The generation MLP has a lock plus a snowflake
beside E_tok, W1, b1, W2, and b2. A separate blue loop changes the input, never the parameters. Each stage explains its
meaning in plain text, including the distinction between frozen parameters and recomputed activations/probabilities.
The article clarifies that eval() alone does not freeze weights. The saved-model arithmetic and three-stage structure
remain unchanged. SVG tests require five correctly attached snowflakes and exclude them from activation boxes.

Section 10's live generator now accepts a sampling seed and starting text (up to 16 letters, case-insensitive).
Blank text uses `---`; shorter fragments are left-padded, longer ones keep only their last three letters in the
model input. The full starting fragment stays in the output. Generate uses the entered seed exactly, without the
old hidden increment. New seed selects another uint32 seed and clears the run; Start over replays the same settings.
Invalid settings block generation instead of falling back to boundary tokens or silently wrapping the seed.
The controls preserve their state across frame navigation, and Next stops at END or the 18-call safety cap.
The chart now shows four individual tokens plus the remaining mass; a sampled tail token stays visible.
`node src/check_part1_generator.mjs part1.html` checks replay, prefix handling, actual model probabilities, validation,
terminal states, keyboard controls, navigation, and slide/phone/print behaviour. Source and shared model stay separate;
no retraining, shared CSS changes, new snippets, extra frames, or PDF regeneration were needed.

Section 10 introduces temperature before the sampling function and live generator. Three frames compare the same
`- s a` prediction at 0.5/1.0/1.5, explain coloured `p = softmax(z/T)` symbols, and work through the actual divided
logits and n probability. All 27 logits enter each softmax. The model stays fixed, T is a positive user setting,
and greedy preserves its choice. The existing five-line sampling function follows the explanation unchanged.
A mode-aware note beside the generator slider explains that temperature changes bars but not greedy choices.
There are still 33 executable snippets. Browser and independent Torch regressions check the three distributions.

Section 10 now animates a complete saved-model generation run from `- - -`: seed 3, temperature 1 samples s, a, m,
then the boundary. Its 12 managed steps separate prediction, selection, and append/shift (or stop). Next/Back gives
manual pacing, Play replays the run, leaving the frame pauses playback, and reduced-motion/print disables movement.
The following frame records all four calls so final-state PDF export retains the worked trace. A greedy-versus-sampling
comparison at `- s a` shows n (0.161) versus this draw of m (0.064), just before the unchanged live generator.
The literal PyTorch comparison and synchronized download bring the total to 33 snippets. Browser and Torch RNGs differ.
New checks cover all stages, chosen tokens outside the top five, the boundary stop, controls, and independent Torch probabilities.

Section 9 now follows the learning graph with a full `E_tok` view: all 27 rows and both actual stored coordinates,
split into three blocks with no omissions. Only a and b are outlined. The text explains that two a occurrences send
gradients to the same row, where autograd adds them, while b contributes once. The other 25 rows, including target i,
have zero embedding gradient from this single example's cross-entropy. Extra regularization is explicitly excluded.
This is a separate frame so final-state slide exports retain both the computation graph and the expanded table.
The new autograd regression verifies selected rows, repeated-row accumulation, and a plain SGD step touching only a/b.

The classroom stretch from the worked hidden unit through final probabilities is now **four frames instead of nine**.
One frame combines a hidden activation and the i logit, followed by softmax with its two-line code, the numerical table,
and the probability bars. Detailed product worksheets and the stability derivation/code remain in full article mode.
The classroom table uses exp(z) directly to match its preceding formula; PyTorch still evaluates softmax stably. All
classroom numbers in this stretch use three decimals. Model parameters, probability values, and all 32 snippets are unchanged.

Part 1's shape explanation now separates activations from parameters. For a0/a1/z, rows count examples and columns
count input features, hidden activations, or vocabulary scores. The entire a a b window is one example. A separate
parameter table explains W axes, shared bias rows, and PyTorch bias broadcasting. A five-line, four-example snippet
shows [4, 6] → [4, 32] → [4, 27] with unchanged parameters. `check_part1_torch.py` checks its shapes and each batched
row against a separate forward pass. The lesson and synchronized download now contain 32 executable snippets.

Part 1 now uses **ReLU**, following Nipun's September 11 annotation. `train_names.py`, its backward pass, `part1.js`,
all classroom equations, and the 32 literal PyTorch snippets agree. The model was retrained with the same names, split,
seed, dimensions, sign penalty, and 6,000 steps. `toy1.json` explicitly records `activation: "relu"`. All worked values,
embedding plots, losses, and generation therefore use new numbers. Historical tanh numbers must not be copied back.
The seed-1 generation illustration now samples **h**, giving `a b h`; the observed training target is still **i**.
`check_part1_torch.py` checks every NumPy parameter gradient against autograd for both fresh and trained parameters.

The full-network diagram labels W1/W2 on the connection sets and b1/b2 below their receiving columns. Purple parameter
labels match the equation explanations, blue marks the six inputs, and teal marks hidden activations. The Part 1-only
`annotateMLP` helper extends the shared sketch without changing other parts. A three-number ReLU example precedes the
hidden-layer PyTorch frame. Weight matrices and biases have separate explanations and shapes.

Part 1's two MLP sketches now pass the real hidden width (32) to `netSketch`. They show eight numbered hidden units,
1–4 and 29–32, with an explicit gap. All six scalar inputs remain visible on the full-network frame. Output gaps appear
both before and after the highlighted i. The shared helper abbreviates hidden widths above eight and connects only real
displayed nodes. Keep the true counts in captions and accessibility labels. `check_part1_diagrams.mjs` covers both sketches.

The September 11 pass keeps Claude's verified September 5 content on `main`; the partial whiteboard work remains on
`wip/whiteboard-pass`. Presentation navigation is now one compact row, hidden after three idle seconds. **C**, lower-right
pointer movement, and touch restore access; hover and keyboard focus prevent a timeout. `controls_test.mjs` checks these
behaviours. All cover slides contain only the series label, title, and subtitle, without opening formulas or duration text.
The historical opener convention in section 8 is superseded by this paragraph.

Part 1 now opens with a concrete example per language task: spam, sentiment, entities, translation, summarization, and
question answering. A comparison precedes next-word prediction and the `aabid` character example. The generated-name
teaser and “every number by hand” claim were removed. Three-character probability notation is introduced with explicit
positions before the general expression. All trained data and arithmetic remain unchanged.

The subsequent notation pass adds same-colour, plain-language keys beside Part 1's probability, embedding lookup,
concatenation, MLP, softmax, and loss equations. It explains the conditioning bar and index meanings explicitly.
The MLP layers, stable-softmax calculation, loss comparison, and mathematical recap have room for those explanations
on separate frames. Part 1's role macros are scoped to its runtime; other parts retain their existing Q/K/V palette.
Use the equation-key convention in `PRESENT.md` for future revisions and `notation_test.mjs` to check this implementation.

Part 1 also pairs each computational stage with a 2–5-line PyTorch snippet, a plain-language explanation, and tensor
shapes. There are 32 executable snippets, including `nn.Embedding`, `nn.Linear`, ReLU, the storage transpose, stable softmax,
cross-entropy, autograd/SGD, and generation. Run `check_part1_torch.py` with PyTorch installed; it checks the literal HTML
code and the matching download `../examples/part1_pytorch.py`, then reproduces the saved model's six probability rows.
The classroom code starts fresh and illustrates one update; it does not claim to reproduce training from a single batch.
The current source data and JS computations use the retrained ReLU model described above. `notation_test.mjs` locates its frames by title, not fragile indices.

The embedding introduction now motivates learned rows before the table: one saved two-coordinate row becomes a point,
then three points become table rows. The existing vowel-sign constraint is explicitly disclosed, not presented as a
discovered semantic axis. A 22-frame addition in section 4 develops representations for words, documents, images, and
time series, then returns to the original `a a b` character context. Word2vec is a six-frame, king/queen/man/woman detour:
learning from surrounding words, a simple 2D analogy, and four lines of PyTorch. Its parallelogram uses invented numbers,
not pretrained vectors or a claimed exact rule. Detailed skip-gram gradients and CBOW arithmetic were intentionally omitted
at Nipun's request. Preserve this introductory pacing.

The lookup equation is now introduced through six small frames: character `c`, mapping `id(c)`, full table `E_tok`,
square-bracket row selection, retrieved vector `e_a`, then the complete equation. The two blue coordinates alone form
the embedding; character and numeric ID are row labels. New Part 1 macros `pToken` and `pIndex` colour characters rust
and lookup IDs muted violet, with matching prose. The established embedding/table blue is unchanged. These local lookup
roles do not redefine Part 2's Q/K/V palette. A separate three-line PyTorch example retrieves just a and distinguishes
its `[1, 2]` output shape from coordinate values. `notation_test.mjs` covers all seven frames, label colours, IDs, and rows.

`embedding-primer.js` supplies 11 editable SVG instances. Character coordinates come from `toy1.json`; analogy coordinates
are hand-chosen; the domain encoders are schematic. The existing two-mug scene is embedded by `assemble.py` for offline use.
`check_embedding_primer.mjs` checks coordinate consistency, accessibility, label bounds, full-size classroom diagrams,
frame fit, and phone layout. The document/CNN/signal snippets create separate fresh models and do not load pretrained
encoders or replace the character model. See `CLASSROOM_QA.md` for this checkpoint's verification.

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
| 1: characters to prediction | `sections1/secNN.html` | `toy1.json`, `part1.js`, `part1-diagrams.js`, `embedding-primer.js`, `part1.json` | `../part1.html` |
| 2: self-attention | `sections/secNN.html` | `toy.json`, `part2.json` | `../attention.html` |
| 3: multi-head attention | `multihead_story.py` + `build_multihead_lesson.py` → `sections3-heads/secNN.html` | `head_worksheet.py` → `toy3-heads.json`, `part3-heads.js`, `part3.json` | `../part3.html` |
| Optional 2B: training, blocks and cost | `sections3/secNN.html` (original Part III) | `toy3.json`, `part3.js`, `part2b.json` | `../part2b.html` |
| 4: cross-attention and translation | `sections4/secNN.html` | `toy4.json`, `part4.js`, `part4.json` | `../part4.html` |
| Vision II: visual pretraining (source ID 6) | `sections6/secNN.html` | `toy6.json`, `part6.js`, `part6.json` | `../vision2.html` |
| Vision III: CLIP (source ID 7) | `sections7/secNN.html` | `toy7.json`, `part7.js`, `part7.json` | `../vision3.html` |
| Vision IV: VLM (source ID 8) | `sections8/secNN.html` | `toy8.json`, `part8.js`, `part8.json` | `../vision4.html` |

Paths in this table are relative to `src/`. Shared files are `shell.html` (layout/CSS), `shared.js` (math, widgets, notation, presentation runtime), and `assemble.py`. Do not edit the generated root HTML or the inlined `katex-bundle.html` by hand. `index.html` is the series landing page; `part2.html` redirects to `attention.html`.

The internal source IDs 5–8 are not displayed part numbers. Their configs set `series: "Vision to language"`, `part: 1..4`, `partLabel: "Vision I".."Vision IV"`, and separate `vision1..4` notation filters. `assemble.py --part 5 --out vision1.html` selects source 5 and displays Vision I. The text configs use `series: "Attention and language"` and display Part 1–4.

Build each part once in any order. `assemble.py` derives available lesson targets from complete section/config/data sources, not existing output files. The conventional outputs are the filenames in the table above; a future source config can declare a different `output`. Set `published: false` for an unpublished draft, or `available: false` on an individual navigation entry to leave it disabled. Unknown destinations stay unavailable even if a stale HTML placeholder exists. Distribute the complete set of outputs for offline series navigation. `check_metadata.py` tests clean-directory builds and those planned-link cases.

The standalone staged diagram lives in `figures/attention-diagram-preview/`. Part 2 embeds its same `diagram.js` source through `src/attention-flow-data.js`; keep the preview and article synchronized by changing that shared source.

Part III now has a 45-frame visual story in `multihead_story.py`. Its generator
also retains the 58-step detailed tensor lab for Notebook 7 only. The two manifests
are `figures/multihead/manifest.json` and `lab-manifest.json`; do not put the
full lab back into the lecture. The notebook embeds the visual story first,
then the executable lab. See `PART3_HEADS_PLAN.md` for the critique and sequence.
Keep M for the mask and H=AV for message rows, as in Part II. Use `n_heads` for
the head count and parenthesized superscripts for individual heads.
The four width/bias clarification frames keep the same worksheet projections:
one wide head normalizes once, while two heads normalize separately. Optional
projection biases are distinct from positions, masks and prediction-MLP biases.
The opening two-source mixture illustration is separate from the ten-token
worksheet. A Maya Q/K/V recap bridges Part II to the two query calculations.
All three section dividers use topic labels, not ambiguous numeric transitions.
The repeated architecture map separates concatenation from W_O and labels
their shapes; W_O mixes into embedding coordinates even when width is unchanged.
The arithmetic passage finishes Head 1 before starting Head 2. For each head,
four figures show all numerical Q/K rows, the final query times Kᵀ, all ten
softmax terms and all ten weighted value rows. `head_walkthrough` shares their
layout. The notebook executes each phase directly after its matching figure.

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
- Part 2 teaches one attention head plus output projection and residual. Part 3 adds multiple heads. FFNs, normalization and full blocks remain in optional Part 2B. Label simplified numerical worksheets as such; they do not calculate the full pre-norm stack.
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
override nested crop viewports. (retired) `check_vision_scene.mjs` verifies image decoding, crop transforms, label bounds, offline
loading and mobile containment at every scene frame. It also records representative classroom and phone screenshots.

- `AT.vision`: exact four-patch + CLS worksheet; full ViT pre-norm architecture is clearly separate from the no-LN/no-FFN numerical model. Position addition is same-width. Token permutation without positions is equivariant; the CLS readout is invariant to patch-only permutation.
- `AT.vision.story`: image crops stay attached to representations through 21 SVG stages. Important operations use separate authored frames so their workings survive ordinary PDF export. Projection columns, score/exponential/normalization stages, and value-only interventions are shown explicitly.
- `AT.vision.learning`: separate 44-parameter two-image classifier. Initial numbers match `AT.vision.forward()`; full-batch SGD uses learning rate 0.05 for 600 updates. The first step reduces mean loss but worsens the two-block example. `verify_vision1_learning.py` independently reproduces its tensors and checks every parameter gradient with NumPy and central differences. **Do not copy the fitted encoder over the initial frozen snapshot used by Vision IV.**
- `AT.visionSSL`: exact masked-pixel losses and illustrative teacher logits. These are not trained MAE, DINO, or I-JEPA outputs. EMA and stop-gradient roles must remain explicit.
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
node src/train_vision2.py --check
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

## 7. Vision rebuild (2026-09-05, complete: all four vision parts rebuilt on the shared scene and pushed)
The four vision parts are being rebuilt to the standard of Part 2. Design: `VISION_FEEDBACK.md` (diagnosis and per-part plans) and `VISION_AXES.md`
(one 8×8 scene with named regions, a fixed patch encoder with named axes brightness / contrast / row / col, keys "bright region? / on the right?",
values "sends: brightness / sends: contrast", the shared figures). Foundation files: `vision-shared.js` (scene, encoder, grid, thumb, overlay, scatter,
circle, triptych, curve; injected before partN.js for parts 5 to 8), `make_vision_toy.py` → `toy5.json` (Vision I asks "is there a mug on the right half?";
scene C is the generalisation probe), `part5.js` (Vision I runtime), `sections5/sec00_vision_demo.html` (component gallery, not shipped), and the
"Vision components" section of `CONTRACT.md`. Per-part briefs: `TASK_VISION_B.md` (Vision I), `_C` (II), `_D` (III), `_E` (IV); parts 6 to 8 copy
`toy5.json.trained` into their toy as `encoder` and define `V.encode` in their runtime. Rebuild acceptance: named axes everywhere, thumbnails as row
labels, one figure per section that is not a table, an end-to-end run with a loss curve, one caveat box per part, clean qa/sweep/mathdiff/walk.

## 8. Presentation openers (2026-09-05)
Present mode now begins on a cover frame (`#s00/1/0`) that `shared.js` builds from the part config: `series · partLabel`, `title`, `subtitle`, the
`hook` line (one question per part, in `partN.json`), the `central` chain, and `audience · minutes`. It exists only in present mode. Every part's first
section frame is a hook, not machinery: Part 1 shows eight invented names, Part 2 the sentence with a blank, Part 3 one guess and one truth with its
loss, Part 4 banque or rive, Vision I the photo and the question, Vision II the 142 million unlabelled images, Vision III the fixed menu, Vision IV the
four-word question. Rule for new parts: cover, then a hook frame, then machinery; the fit check (`#at-fit-warning`) must stay empty on every frame.

## 9. Whiteboard pass (2026-09-05)
`STYLE_WHITEBOARD.md` distils how the instructor teaches (his handwritten decks in ~/git/ml-teaching and Part 2 here): example before machinery, one
held drawing that gains one mark per build, exact small numbers on the drawing, colour = identity, at most 40 words of prose on a frame, notes that
ask a question. `FRAME_AUDIT.md` lists the frames of every part that break those rules (measured from the built pages). Parts 1, 3, 4 and Vision I
to IV are being reworked frame by frame against that list; Part 2 is the reference and is left alone. Re-run the audit after any change: build the
part, then count per frame words, builds, svg, tables (the script is small; see git history of FRAME_AUDIT.md for the measurement).

## 10. STATE AT HANDOVER (2026-09-05 evening) — read this first
Live site: https://nipunbatra.github.io/attention/ serves main. Everything on main is VERIFIED (zero errors at 1280 and 390, sweeps clean, present-mode
fit checks clean) as of the commit that adds this section. What is done:
- Parts 1, 2, 3, 4 (language) and Vision I to IV rebuilt on one design system; vision parts rebuilt on the shared 8×8 scene with named axes (sections 7, 8).
- Every part opens in presentation mode with a cover frame and a hook frame (section 8).
- STYLE_WHITEBOARD.md and FRAME_AUDIT.md define the next improvement: the whiteboard pass (section 9). It has NOT been done. Seven agents were started in
  parallel and all died on a session limit before writing frames. Their few partial edits (part4.js, part6.js, sections6/sec01-02) are preserved on the
  branch `wip/whiteboard-pass` and REVERTED on main, so main is clean. TASK_WHITEBOARD.md holds the exact prompt to rerun the pass, one part at a time.
How to resume (any agent, Codex, or a person): read HANDOVER.md sections 1 to 4, STYLE_WHITEBOARD.md, FRAME_AUDIT.md, TASK_WHITEBOARD.md; do one part;
verify with section 3's commands plus the fit-warning walk; push the built page together with its sources. Never push a built page that was not verified.
Known small issues: vision pages are about 2 MB each (photo hook, KaTeX, toys); Vision III's image encoder picks its axes by a small search because Vision I's
encoder maps the three CLIP scenes onto one line (the page says so); the hero roadmap of Part 2 lists sections 1 to 14 only.
