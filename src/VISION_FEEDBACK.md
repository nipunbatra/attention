# Vision parts: what is missing, and what to build

Written 2026-09-05 after reading vision1 to vision4 end to end (text and every section screenshot) and comparing them with Part 2.
Audience for this file: whoever improves the vision lessons next (Codex, Gemini, or a person). Everything below is specific enough to act on.

Sizes today, for reference:

| page | sections | words | height |
|---|---:|---:|---:|
| attention.html (Part 2) | 19 | 7,391 | 74,677 px |
| part3.html | 19 | 3,301 | 39,279 px |
| vision1.html | 8 | 2,807 | 31,175 px |
| vision2.html | 7 | 2,226 | 16,351 px |
| vision3.html | 7 | 1,889 | 18,474 px |
| vision4.html | 7 | 1,890 | 17,172 px |

The vision parts are a third of Part 2 by words and a quarter by height. Length is a symptom, not the disease. The disease is in section 1 below.

## 1. Why they read as poor: eight causes

**1. They start at the mechanism, not at a question.** Part 2 opens with a sentence, a blank, and probability bars: the reader wants to know the answer before any machinery appears. Vision II opens with the definition of self-supervised learning, Vision III with the definition of CLIP, Vision IV with a definition of a VLM. Vision I has the right instinct (a photo and a question) but abandons it in the same paragraph: "we will switch to a separate 4×4 worksheet". No part earns its machinery with a concrete problem that the previous tools cannot solve.

**2. Two worlds that never touch.** Every vision page has a beautiful tabletop photo (mugs, book, plant) that is never computed on, and a 4×4 grid of 0/1/2 values that is computed on but says nothing. The photo carries the caption "AI-generated teaching illustration. No model output is shown" fifteen times across the four pages, which tells the student, fifteen times, that the interesting picture is inert. In Part 2 the intuition example and the worked example are the same sentence ("The fisherman sat beside the river bank"); that is why Part 2 feels whole. The vision parts need one scene that is both drawable and computable.

**3. Anonymous coordinates.** The single biggest win in Part 2 was naming the axes (water, finance, person, glue, position) so a student can read a row. The vision toys went back to "coordinate 1" and "coordinate 2" (vision1, vision4), "matching 1" (vision2), and empty axis lists (vision3). The page even says "coordinate 1 adds the left column of the patch" and still labels it "coordinate 1". CLIP's initial W_txt rows are one-hot on squares, stripes, dot, so its three axes already have names and the page does not use them.

**4. Caveats crowd out intuition.** Count the sentences that begin with "not", "it does not", "this is not evidence". Vision I s01 alone: "It is a separate numerical example, not a downsample of the scene and not evidence that a model trained on two block images can count mugs." Every section of every vision part ends with two or three such paragraphs. The honesty is right (and rare) but its placement kills momentum. Part 2 solved this with one provenance line in the hero and one "what this toy can and cannot show" box; the vision parts should do the same and let the sections breathe.

**5. The signature figures of the field are absent.** A student who has seen any ViT talk expects: the attention heatmap painted over the image (which patches does CLS read?); MAE's triptych of masked, reconstructed, original; CLIP's image and caption points pulling together in embedding space; a loss curve next to a before and after prediction. None of these exist. The pages have flow-box schematics (input row → encoder → output row) and tables of numbers instead. Flow boxes explain plumbing; heatmaps and scatter plots explain meaning.

**6. End-to-end working is uneven.** Vision I trains its toy and shows a failure case (good). Vision III trains for 60 steps and does zero-shot (good). Vision II trains nothing: MAE is a loss calculator with a scalar guess the reader types in, DINO uses invented teacher logits (2, 1, 0) unconnected to any image, and I-JEPA is drawn symbolically. Vision IV runs generation, but its numbers are opaque (keys like −8.652, values like 20.571) and the only interaction is a two-grid switch.

**7. The language parts' idioms are not reused.** Part 2's mock search with per-card key and value strips, the records view (each token as a card that opens to show k_j and v_j), the routing tree with a live dot-product worksheet, the mixTable with a weighted sum footer, the eighteen-step walkthrough, and the recurring motif e → Q,K,V → attention → Δe → e+Δe. The vision parts use the same runtime but almost none of those forms, so the series stops feeling like one author at vision1.

**8. Loose ends.** Vision IV s07 refers to "the earlier article's thermal example" and "the earlier VLM article", which do not exist in this series. Vision I s03 introduces p_j for positions while Part 2 shows a position column; the notation card does not list r_j, G, B, g_img, g_txt, W_bridge, E_tok. Section titles are fine.

## 2. What to keep

Do not lose these while rebuilding: the value-only intervention in Vision I s05 (weights unchanged, message changes); the CLS-versus-stored-parameter table; the position ON/OFF permutation check; the symmetric cross-entropy worked in both directions in Vision III; the "removing the correct candidate does not create none-of-the-above" point; the temperature ratio example (e^5 ≈ 148, not 5×); the failed one-example update in Vision IV s06; the response-only loss table; the PyTorch two-liners beside each calculation; and the scoping sentences, once they are gathered into one box per part.

## 3. Cross-cutting fixes (do these before touching individual sections)

**A. One scene, drawable and computable.** Replace the 4×4 block grid with an 8×8 grayscale "tabletop" designed by hand: a bright 3×3 square (the mug), a 2×5 stripe (the book), a small dot (the plant), on a flat background. With 2×2 patches that is 16 patch rows, small enough to compute in the page (d_model = 2 or 3), large enough that an attention overlay means something. Give the regions names (mug, book, plant, table) and use them in every row label, so tables read "patch 6 · mug edge" instead of "patch 6". Keep the photo as the opening hook of Vision I only, then never show a picture the model cannot see. If the 4×4 must stay for hand arithmetic, use it for one worked frame and switch to the 8×8 for everything trained.

**B. Named axes in every vision toy**, designed the way AXES.md did it for language: pick the names first, then choose weights so the names are true. Suggested names:
- patch embedding (Vision I): "brightness", "edge" (left column minus right column) or, for the current weights, "left-column brightness" and "right-column brightness", which is literally what the page says they compute;
- CLS query and patch keys: "bright region?", "edge here?";
- values: "sends: brightness", "sends: edge";
- CLIP (Vision III): "squares", "stripes", "dot" (the initial text rows are exactly these);
- VLM connector output (Vision IV): whatever the connector carries after fitting; if nothing readable survives, say so in one line and show the attention overlay instead of the value table.
Add the "coordinate → meaning" table to each part's notation card, and drop "coordinate 1" everywhere.

**C. Attention painted on the image.** Add one shared component `AT.vision.overlay(grid, alpha, {receiver, hover})` that tints each patch with the rose heat colour in proportion to α, with the receiver outlined, and use it in Vision I (what CLS reads, before and after training), Vision II (what the visible patches read in MAE), and Vision IV (what the answer token "two" reads: the grounding figure that the whole series builds towards). Hovering a patch shows its key and value rows. This is the most valuable single addition in this file.

**D. Embedding-space pictures.** The toys are 2-D or 3-D already. Draw them: a scatter of patch embeddings (Vision I s02) with the four or sixteen points labelled by region; the CLS point with its Δe arrow after attention (as Part 2 s10 does for bank); for CLIP, the three image points and three caption points on the unit circle (2-D) or a projected sphere (3-D), animated over the 60 training steps so pairs visibly pull together and competitors push apart; for the two-view section of Vision II, the two views' encoder outputs as two points that training moves together.

**E. An end-to-end run in every part**, with a loss curve and a before/after that a student can see: Vision I has it (add the overlay before/after); Vision II must gain one (see below); Vision III has it (add the embedding animation); Vision IV should show the overlay for both grids and for both answers.

**F. Gather the caveats.** One "What this toy can and cannot show" box per part, placed after the hero, written once. In sections keep at most one short scoping sentence where a student might over-read a result (for example after the failure case in Vision I s07). Everything else moves to the box. Target: no section ends with two consecutive caveat paragraphs.

**G. A motivation opener for each part, with one number a student will remember** (the way Part 1 s14 uses 26 million weights):
- Vision I: "224×224 pixels as tokens is 50,176 rows and 2.5 billion attention scores per head. Sixteen-pixel patches give 196 rows." Build it as the first frame with a patch-size slider that shows rows and scores, reusing Part 1's window-growth pattern.
- Vision II: the cost of labels versus the supply of images. Give a sourced number (ImageNet's labelling effort, or the count of unlabelled web images used by a named model) and then the one-line idea: "the image is its own teacher: hide part of it and ask for it back."
- Vision III: a classifier can only say the names on its fixed menu; a caption can say anything. Open with the crane example (bird or machine) that is currently buried in s06, and with a class list that lacks the true answer.
- Vision IV: show CLIP failing a question. It can rank "two mugs" against "one mug" but cannot answer "what is on the book?". That failure is the reason for a decoder, and the whole part should be the answer to it.

**H. Reuse Part 2's forms.** Records view for patches (a card per patch with its thumbnail, e_j, k_j, v_j); the routing tree with thumbnails at the leaves and the click-to-open dot-product worksheet; the mixTable with thumbnails as row labels and the Σαv footer; a stepper walkthrough for the full CLS update; the motif at the top of Vision I and IV with the stage names unchanged. The student should recognise every form from Part 2 and see only the data change.

**I. Fix the loose ends.** Remove or rebuild the thermal example in Vision IV s07 (as it stands it refers to a page that is not in the series); extend the notation card; standardise the position symbol with Part 2.

## 4. Per-part plans

### Vision I: An image becomes a sequence (8 → 11 sections, target 4,000 words)
1. new, "Why patches?": the tokens-per-image slider and the 2.5-billion number; one sentence on why not one row per pixel and why not one row per image.
2. s01 The scene: keep the photo hook, then the 8×8 named scene; the question "how many mugs?" stays the question for the whole part.
3. s02 Patch embeddings: name the two axes; add the 2-D scatter of patch points labelled by region; keep the worksheet for one patch.
4. s03 Position and CLS: keep; make the "swap two patches" check visual on the grid.
5. s04 Routing: the records view and routing tree with thumbnails; the attention overlay on the image for CLS, untrained.
6. s05 Values and the update: mixTable with thumbnails; the CLS point and its Δe arrow on the scatter.
7. s06 Prediction: keep.
8. s07 Learning: keep the failure case; add the loss curve and the overlay before versus after training ("what CLS reads now").
9. new, "What a trained model attends to": the overlay on the 8×8 scene for a few patches as receivers, and one honest paragraph about real ViT attention maps with a link, no fabricated maps on the photo.
10. s08 Scale-up: keep; connect back to the slider from section 1.
11. Summaries and pause-and-think, as in Part 2.

### Vision II: Learn visual representations without class labels (7 → 10 sections, target 3,500 words)
1. new motivation: labels are expensive, images are free; "the image is its own teacher".
2. MAE, the idea: the triptych (masked, prediction, original) on the 8×8 scene.
3. MAE, trained: fit a tiny decoder (a few parameters) on the scene's patches in the page; loss curve; the triptych at steps 0, 10, 100; then the ambiguous-patch case (two valid completions) as the motivation for feature targets.
4. Which pixels enter the loss: keep the calculator, now fed by the trained decoder's guesses instead of a typed scalar.
5. Two views: the flip and brightness views drawn correctly (one shared grayscale ramp; the darker view must look darker); the two encoder outputs as points that agreement pulls together.
6. DINO: teacher logits computed from the toy encoder on view B, not typed in; the centering toggle: switch it off and watch every output slot converge and the loss sit at log N (this is the collapse intuition, and it is currently a sentence).
7. I-JEPA: the predicted feature and the target feature as two points; one honest line that the numbers come from the toy encoder, not the paper's.
8. What did it learn: a linear probe on the toy's frozen features, drawn as a decision line in the 2-D feature plane separating "two objects" from "one object" scenes.
9. Comparison table (keep) plus the caveat box.
10. Summaries.

### Vision III: CLIP (7 → 9 sections, target 3,500 words)
1. new motivation: the fixed menu versus the open caption; crane first.
2. Pairs: the three scenes and captions; name the three axes.
3. Encode and normalise: keep; add the unit-circle picture with six points.
4. The similarity matrix: keep the heatmap; put thumbnails on the rows and caption text on the columns (the page has labels; make them pictures).
5. Both directions: keep.
6. Training animated: 0 → 60 steps on the circle; pairs pull together, competitors push apart; the loss curve beside it.
7. Batch size: a slider from 3 to 8 pairs showing how more negatives sharpen the softmax; the collapse case (all points equal, loss = log N) drawn, not only stated.
8. Zero-shot with prompts: keep; make the prompt template demo show a real effect on the toy.
9. Scope box, summaries.

### Vision IV: An image becomes context for an answer (7 → 10 sections, target 3,500 words)
1. new motivation: CLIP fails a question; we need to write an answer.
2. The pipeline end to end, drawn once with the scene: image patches as thumbnails in the token row, text tokens as chips, the decoder, the answer. This is the signature VLM figure and it should be the frame students photograph.
3. Visual rows enter the decoder: keep; name what the connector carries if it can be named.
4. Two routes: keep the mask picture; shorten the prose.
5. One prediction: replace the value table with the overlay of what "?" reads; keep the worksheet for q·k behind a click.
6. Generation stepper: keep; add the overlay for each generated token ("two" reads the mug patches).
7. The grounding test: change the image, keep the question; both overlays side by side; this is the payoff of the series.
8. Training on the answer: keep the failed update, it is a good lesson.
9. What the pixels cannot tell you: rebuild the thermal idea with the scene ("how heavy is the mug?") or drop it.
10. Scope box, summaries, and a closing frame that returns to Part 1's `a a b → i`.

## 5. Order of work and effort

1. Cross-cutting A and B (scene and named axes): one day; everything else depends on them.
2. C and D (overlay and scatter components): half a day, shared by all four parts.
3. Vision II real training and the DINO collapse toggle: one day.
4. Vision IV pipeline figure and grounding overlays: half a day.
5. Motivation openers, caveat consolidation, notation card, loose ends: half a day.
6. Vision III animation and batch slider: half a day.

Acceptance for each part: every displayed number comes from the toy; every table has named columns; at least one figure per section that is not a table or a flow box; one end-to-end run with a loss curve; one caveat box; qa.mjs, sweep.mjs, mathdiff.mjs clean at 1280 and 390; present mode walks without overflow.
