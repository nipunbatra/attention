# Vision series: an independent teaching review

Reviewed 5 September 2026, against repository commit `4c638e8`.

## My recommendation

The vision series is still weaker than the text series. The recent changes made the diagrams more legible and the calculations easier to inspect, but they did less to explain why a learner should want each calculation. Your discomfort is justified.

I would keep the four-part structure and most of the verified numerical machinery. I would substantially rebuild the teaching sequence in **Vision II**, then **Vision IV**. Vision III needs a better account of the geometry and purpose of image-text matching. Vision I already has a fairly complete lesson; it needs more visual interpretation and a few better experiments, not a wholesale restart.

The aim should be that a student can follow an image from pixels to a prediction, explain what each intermediate represents, and see what changes after learning. Adding more slides will help only where it gives students time to make those connections.

This is a review and implementation brief. No article sources have been changed for this audit. Proposed experiments and figures below have not been built or measured unless explicitly described as existing results.

## What I inspected

- Claude's [VISION_FEEDBACK.md](VISION_FEEDBACK.md), including its proposed rebuild.
- All 29 vision section fragments: `sections5` through `sections8`, the part configurations, and the relevant numerical/diagram code and toy data.
- Representative text lessons, especially the search example in [Text II §5](sections/sec05.html), the return to the bank example, the loss lesson in [Text I §7](sections1/sec07.html), and the continuation into training in [Text III §1](sections3/sec01.html).
- Selected live article and presentation views, including the MAE loss calculator, DINO distributions, CLIP training, and VLM value retrieval. This was a visual spot-check, not a new screenshot audit of every frame or PDF page.
- All four non-browser numerical regression scripts. They pass. Vision III checks 900 reference scalars and 246 finite-difference gradients; Vision IV checks 4,779 reference values and 1,096 gradients. These checks support the implemented arithmetic, not every explanatory claim or every proposed improvement.
- An additional calculation using the current VLM to examine which query predicts the first answer and how image and text values affect its logits.

The latest commit adds Claude's feedback file; it does not contain a newer article rewrite that this review overlooks.

## 1. Where the gap actually is

| Part | Current state | Main teaching gap | Revision priority |
| --- | --- | --- | --- |
| Vision I: image tokens and classification | 60 frames; complete forward pass, actual training, loss curve, and a revealing failure | The image becomes numbers, then mostly disappears while those numbers are processed | Targeted improvements |
| Vision II: visual self-supervision | 26 frames covering MAE, DINO, I-JEPA, and evaluation | Several target/loss explanations, but no complete image-to-prediction-to-update experiment | Highest |
| Vision III: image-text matching | 28 frames; specified encoders, symmetric loss, actual updates, candidate controls | Matching geometry is mostly a table; the reason to replace a fixed classifier is underdeveloped | Substantial additions |
| Vision IV: image-conditioned generation | 36 frames; real fitted decoder, generation, response loss, and an update | Too much happens inside supplied fitted matrices; the source-to-value-to-answer connection is hard to see | Second |

Vision I is not particularly short. Vision II is short *for the number of ideas it attempts*. Frame counts are useful evidence of pacing, but I would not use equal word counts or equal numbers of slides as a target.

### What the text lessons do better

The search lesson first gives the learner a request and a collection of recognizable items. It changes the request while keeping the collection fixed. Then it separates the key used for matching from the value returned. Students have a reason to inspect each table: they want to explain a change they have just seen.

The bank example also has a persistent receiver. We know whose representation is being updated, which words can help it, and what prediction the update will support. The training continuation retains that example.

In vision, the calculations are often individually sound, but the learner changes situations too frequently: a mug photograph, a four-patch grid, three caption patterns, then a fitted block-counting decoder. The transitions need to explain what carries over. Otherwise the series feels like several small demonstrations placed next to one another.

**A good revision should repeatedly answer:** What are we trying to predict? What can the model see? Where does this number come from? What does the result let us do next?

## 2. Which of Claude's recommendations I would use

Claude identifies several important weaknesses. I would not implement the document literally.

| Recommendation or claim | My assessment |
| --- | --- |
| Vision II needs a real learning example | Strongly agree. Moving a supplied pixel guess is not training an encoder or decoder. |
| Real scenes and computed grids feel disconnected | Agree. Keep the distinction honest, but teach the transition instead of repeatedly apologizing for it. |
| Add spatial attention views, reconstructions, and embedding geometry | Agree, provided the figures use actual model outputs and explain what they measure. |
| Reuse the text series' visual components | Agree selectively. Reuse row-by-column multiplication, source-linked value contributions, and progressive disclosure. Do not turn every vision operation into another attention diagram. |
| Vision I needs a training loss curve | Already present in `sections5/sec07.html`, at `s07-training-curve`, together with training predictions and a moved-block failure. Improve its interpretation instead of duplicating it. |
| The parts begin with definitions rather than questions | Too broad. Vision II already opens with a masked scene; Vision III with captions; Vision IV with an image-dependent answer. Their openings need stronger follow-through, not simply a question pasted above them. |
| The notation card is missing almost all new symbols | Overstated. Cards already include raw patch rows, encoded visual rows, image/text vectors, the connector, and the token lookup table. There are specific omissions and collisions, listed below. |
| Give every learned coordinate a semantic name | Reject as a blanket rule. Name coordinates when the disclosed construction makes the name true. Do not call a fitted decoder coordinate “mug count” because that would make the story convenient. |
| Replace the small grid with an 8×8 scene immediately | Not a prerequisite. Sixteen patches plus CLS produce a 17×17 attention matrix. That can improve spatial experiments while making first-pass arithmetic harder. Introduce it only where the extra image structure earns its cost. |
| Turn centering off in DINO and show a guaranteed collapse at loss log N | Incorrect as a promised outcome. Collapse has different forms, and its loss depends on the output distributions. See the correction below. |
| More CLIP negatives sharpen the softmax | Incorrect for fixed logits. Adding competitors increases the denominator and lowers an existing candidate's probability. Training with more negatives is a separate experiment. |
| Show the “two” token reading the mug patches to explain its prediction | Wrong receiving position for the current model. The `?` query predicts `two`; the `two` query predicts `blocks`. Also, this numerical model sees block grids, not mug pixels. |

### Two technical corrections worth preserving in the rebuild brief

**DINO collapse.** If both branches return the same uniform distribution over K slots, their cross-entropy is log K. If both always return the same sharply concentrated slot, the loss can instead be near zero. Neither outcome proves useful representations. An ablation should show measured distributions across several different images, not a pre-scripted “collapse” animation. Centering, sharpening, and the momentum teacher interact; removing one component does not establish a universal numerical outcome. The [DINO paper's collapse discussion](https://arxiv.org/html/2104.14294v2) is the right reference.

**CLIP negatives.** With fixed positive logit a and other logits b_j,

`p(positive) = exp(a) / [exp(a) + sum_j exp(b_j)]`.

Adding another finite competitor lowers that probability. A larger training batch may supply more useful comparisons, but it may also introduce false negatives. Show those as separate questions. The existing distinction between row and column losses is worth retaining; it follows the paired objective in the [CLIP paper](https://proceedings.mlr.press/v139/radford21a.html).

## 3. Establish continuity before adding content

### Say what is reused, not just “as before”

There is an important distinction between reusing an architecture and reusing its learned weights.

| Transition | What the current code actually does | What students should be told |
| --- | --- | --- |
| Vision I to II | Changes from a fully specified classifier to separate self-supervised objective illustrations/calculators | We are changing how an encoder could be trained. We have not continued the classifier's training run. |
| Vision II to III | Introduces a new linear image encoder and bag-of-words text encoder | The loss is new, and these small encoders are supplied for arithmetic. They are not the pretrained MAE/DINO/I-JEPA models from the previous discussion. |
| Vision I to IV | Reuses the initial five-row vision calculation, retaining four updated patch rows | This is the disclosed initial encoder, not the 600-step classifier checkpoint. |
| Vision III to IV | Does not reuse the trained CLIP image encoder or alignment space | We are changing from candidate matching to image-conditioned generation, not attaching a decoder to that exact fitted CLIP checkpoint. |

Put a small “what we keep / what changes” diagram at each transition. In the longer term, an optional experiment could genuinely transfer a saved encoder between parts. If we do that, it must use the checkpoint in the code and report the resulting behavior. Similar-looking diagrams are not evidence of transfer.

### Keep one family of images, with two distinct purposes

Keep the realistic mug scene for questions people can answer immediately: what is in the image, which crop changes the count, what a patch boundary cuts through, and which answer needs visual evidence.

Keep the current 4×4 block image for the first exact calculation. Its virtue is that every pixel, matrix product, and output fits on a page. Do not relabel a bright square as a detected mug.

Then add a small, disclosed family of larger synthetic images for learning and diagnostic tests: translated blocks, different counts, changed brightnesses, and distractors. The training and test split should vary nuisance properties deliberately. An 8×8 grid is a reasonable candidate here, but the dataset and task matter more than that resolution.

Use the same synthetic family across the new MAE experiment, a matching experiment, and an optional VLM test where practical. Keep a tiny worked subset for hand calculation. This provides continuity without forcing a 17×17 matrix onto every slide.

If we want numerical claims about realistic images, run an actual image model and disclose its input, checkpoint, processing, and output. Generated photographs and hand-drawn overlays can illustrate a question; they cannot stand in for an inference result.

## 4. Vision I: make the numbers visibly belong to the image

Relevant files: [patch embeddings](sections5/sec02.html), [positions and CLS](sections5/sec03.html), [attention](sections5/sec04.html), [values and update](sections5/sec05.html), [training](sections5/sec07.html), and [full ViT](sections5/sec08.html).

### Keep

The pixel perturbation, explicit patch projection, same-width position addition, value-only intervention, distinction between the learned CLS parameter and its per-image output, complete training run, and moved-block failure are all useful. They are stronger than Claude's summary suggests.

### Change the teaching sequence at these points

1. **Motivate patches before processing them.** Show a familiar image with a coarse grid and a finer grid. Let students see the trade-off between local detail and number of tokens. Give the existing 224×224, patch-side-16 example: 196 patch rows and 197 rows with CLS. Put the larger matrix-cost comparison after students understand what those rows represent.
2. **Keep the selected patch beside every calculation.** In the flattening and projection sequence, the same crop should remain visible while its four entries become a row. Highlight the left pixel column for the first output and the right column for the second. The current chosen weights genuinely support those labels.
3. **Explain what the projection discards.** Compare patches `[[2,0],[0,2]]` and `[[0,2],[2,0]]`. The initial W_patch maps both to `(2,2)`: their left/right sums agree, although their diagonal patterns differ. Ask what this particular map can no longer distinguish. This gives students a reason to learn the projection and to use more features.
4. **Make spatial mixing visible.** Put the selected receiver next to the original four-patch grid. Show the actual weights on the four source patches, with CLS as a separate fifth source. Then select a different receiver and recompute. Students should see that an attention row belongs to a receiving token.
5. **Carry each value back to its source.** Add a small patch thumbnail to the existing value/mixing rows. Reveal one `alpha_j v_j` contribution, then the sum, then the residual addition. Keep the value-only intervention: it answers why matching weights alone do not determine the update.
6. **Spend less time repeating familiar softmax arithmetic.** Retain one complete score-to-weight calculation. Use the recovered space for the spatial view, the projection collision, and the question of whether the model really learned to count.
7. **Interpret the training curve through predictions.** Alongside the existing curve, show the two training images and the moved-block image at selected saved checkpoints. Do not imply that every example improves on every step; the current counterexample is educational.
8. **Test a shortcut explicitly.** Add a proposed matched-total-brightness diagnostic: one 2×2 block with pixel intensity 2 versus two separated 2×2 blocks with intensity 1. Both have total intensity 8 but different object counts. Decide the layouts before evaluating. If the model fails, use that result to discuss what the training set did not require it to learn.

### Figure brief: one patch sends information to CLS

Build it in five stages: image and receiver; source patch outlines and actual weights; one selected source's value vector; weighted contributions; updated CLS row and class logits. Use the same source colors and patch IDs throughout. Include CLS's own contribution separately rather than hiding it in an image heatmap.

**Completion question:** Can a student point to a pixel, trace how it changes a patch embedding, and explain how it can affect the final class prediction?

## 5. Vision II: teach one complete self-supervised experiment

Relevant files: [MAE architecture](sections6/sec02.html), [pixel loss](sections6/sec03.html), [views](sections6/sec04.html), [DINO](sections6/sec05.html), [I-JEPA](sections6/sec06.html), and [evaluation](sections6/sec07.html).

This part needs the largest revision. The MAE calculator currently scores a supplied guess. DINO starts from supplied logits. I-JEPA shows a symbolic target. These are valid demonstrations of their stated operations, but together they do not show how an image teaches an encoder.

### Build MAE as the complete example

Use a slow sequence with a visible image at every step:

1. **Remove the class label.** Revisit the classifier's image and hide its label. Ask what information remains available for training.
2. **Hide a region, keep its target.** Show the original image to the class, then show the model's restricted input. Put the withheld pixels in a separate target panel. Distinguish what the learner sees from what the loss can use.
3. **First work one missing patch.** State that this first example hides one of four patches for arithmetic. Later increase to three of four to illustrate 75% masking. The current transition between those settings is easy to miss.
4. **Encode the visible patches.** Show their IDs, position vectors, input rows, and the resulting encoded rows. The masked pixels must not enter this branch.
5. **Assemble the decoder input.** Restore the slot order, insert learned mask tokens, and add decoder position vectors. A mask token should not look like a true zero-valued image patch.
6. **Calculate one prediction.** Use a specified small decoder and actually produce the four predicted pixel values. Then expose one output coordinate's multiplication. A slider may explore the result afterward, but it must not substitute for this forward pass.
7. **Show the error as an image.** Place masked input, predicted completion, and original target side by side. Add a fourth, optional error view. Keep the color scale fixed across all panels and steps.
8. **Compute masked-pixel loss.** Score only the hidden entries. Carry the values from the figure into the existing squared-error table.
9. **Follow the graph backward.** Show gradients reaching the decoder and visible-patch encoder, then use `loss.backward()` and `optimizer.step()`. No hand-derived Jacobians are needed.
10. **Run the same input again.** Update the reconstruction, loss, and one tracked parameter. Then show several saved training checkpoints.
11. **Try an image or mask not used to fit this example.** Include a simple mean-pixel or position-only baseline. One memorized image is not evidence that the encoder uses context.
12. **Ask whether its features help classification.** Freeze the encoder, fit a small probe, and evaluate on a held-out set. Report a failure honestly if this tiny pretraining task does not help. A low reconstruction loss alone cannot answer the question.

This is a small MAE-style teaching model, not a reproduction of a full research training recipe. State the important departures once near the model definition, with details in the companion material. The visible-only encoder and masked-pixel target should remain consistent with the [MAE paper](https://arxiv.org/abs/2111.06377).

### Give DINO a concrete connection to images

Keep the useful crop comparison, including the warning that a crop can change a counting label. Then use two actual views of the same synthetic image, followed by a view of a different image.

Show the path from each view through a specified encoder and output head before introducing its distribution. Use aligned bars for the teacher target and student output. Explain that these slots are learned outputs, not supplied object classes. One actual student update and one explicit EMA parameter update are enough for the first pass.

For collapse, put three different source images above their output distributions. First show identical uniform outputs, then identical concentrated outputs, then image-dependent outputs. Students need to understand what information was lost, not merely recognize the word “collapse.” If these are illustrative states rather than training outcomes, label them that way.

A full, stable DINO training reproduction is optional. Do not promise it just to satisfy a rule that every method needs a large training widget. The minimum is that the displayed distributions have a disclosed source and the student/teacher updates have distinct roles.

### Make I-JEPA's changed target tangible

Reuse the same hidden region from MAE. Compute a target row with a specified target encoder: full image first, select the target position afterward. Beside it, show the context branch's prediction for that row. Give their dimensions and one feature-error calculation.

The learner should be able to say, “We are predicting this encoder's vector for the region; we are no longer being asked to reconstruct its four pixel intensities.” Do not promise that an arbitrary toy feature is semantic or texture-invariant. Preserve the distinction between gradient-updated context parameters and the EMA target copy. This follows the target construction described in [I-JEPA](https://arxiv.org/abs/2301.08243).

### End with a measured result

The current title “What did the encoder learn?” should lead to a small observed test, not only a definition of linear probing. If that experiment is deferred, change the title to “How would we test the representation?” until the result exists.

**Completion question:** Can a student explain where a target came from without a manual class label, identify which parameters changed, and distinguish fitting a pretext task from learning transferable features?

## 6. Vision III: show what image-text matching buys us

Relevant files: [opening](sections7/sec01.html), [encoders](sections7/sec02.html), [matching](sections7/sec03.html), [loss](sections7/sec04.html), [training](sections7/sec05.html), and [candidate prompts](sections7/sec06.html).

### Start with the limitation of the previous classifier

The classifier had fixed output labels. What if we want to compare the image with a new description? Show the old output menu next to three supplied captions. Make the distinction visible before naming CLIP.

Keep the mug-caption opening, but return to it after the numerical example. Otherwise it functions as a cover illustration while the actual lesson happens elsewhere.

### Preserve the exact three-pair calculation

The current small encoders, unit normalization, symmetric cross-entropy, updates, and candidate experiments are a useful base. Prompt-template changes already have numerical consequences; do not rebuild a feature that is present.

Improve these points:

- Keep image thumbnails and full caption text attached to every row/column through the pair matrix, both losses, and training. Avoid making students remember whether I2 or T3 was the striped image.
- Explain the two retrieval directions with an actual question in each direction. One image chooses among captions; one caption chooses among images. Then average the two losses.
- Show normalization geometrically before the cosine matrix. The current representation is three-dimensional. Use a genuine 3D view or a clearly labeled fixed projection. A unit-circle illustration would require a different two-dimensional example, not silently dropping the third coordinate.
- During training, show vector movement alongside the matrix and loss. Keep the display projection fixed across checkpoints. Otherwise apparent movement may be a plotting artifact.
- Track learned temperature separately. The loss can improve through the scale as well as changing directions. Include a control or comparison that holds the vectors fixed while changing temperature.
- Demonstrate a wrong but plausible candidate and a missing correct candidate. The latter already exists and is worth retaining: the softmax must distribute its mass among the supplied options.
- If claiming transfer to a new description or image, reserve that case from training. Choosing among the three training captions is not evidence of zero-shot generalization. The bag-of-words encoder also cannot establish understanding of word order.

### Figure brief: the pair matrix becomes a retrieval result

Start with three image-caption pairs. Separate the two columns while preserving pair colors. Reveal the six normalized vectors, then one cosine comparison, then the full matrix. Highlight a single image row to retrieve a caption; highlight a single caption column to retrieve an image. After an update, move the actual vectors and refresh the same matrix.

Use the unchanged image beside the changing candidate menu in the next section. Students should see why the output can change even though the image did not.

**Completion question:** Can a student explain why a new candidate list changes probabilities, why that is not vocabulary generation, and what evidence would count as transfer beyond the training pairs?

## 7. Vision IV: make the image's effect on the answer traceable

Relevant files: [visual rows and connector](sections8/sec02.html), [architecture and mask](sections8/sec03.html), [first prediction](sections8/sec04.html), [generation](sections8/sec05.html), [learning](sections8/sec06.html), and [limits](sections8/sec07.html). Numerical source: [part8.js](part8.js) and [toy8.json](toy8.json).

### Complete one route before comparing architectures

Use the existing visual-prefix model as the main route. Move the alternative cross-attention/resampler discussion until after students have seen a complete prediction and generation cycle. At that point the architectural comparison answers a question they can understand: where else could the language model read the image?

The first pass should proceed in this order:

1. Show the image, the fixed prompt `<bos> count ?`, and the unknown answer.
2. Retain the four contextual patch outputs G. Explain why the classifier's single CLS row is not the only possible visual output.
3. Apply the connector to one row, then all four rows, producing B. Show `4×2 → 4×3`; explain the width change before filling in numbers.
4. Stack those rows with the three known text rows and add same-width positions. Display all seven row labels.
5. Show the mask for these seven rows. Introduce the longer nine-row training view later, when answer tokens and shifted targets are present.
6. Select the `?` row as the receiver. Calculate one coordinate of its query, then show its complete query and the seven keys.
7. Calculate one score, then the full score row and softmax weights. Keep the image and text source labels visible.
8. Show each source value and its weighted contribution. The current main frame jumps directly to the total message, output projection, and residual result; the missing source terms belong in the main teaching sequence.
9. Project the message, add the residual, calculate the answer logits, and interpret the output.
10. Change only the image. Compare the same prompt and first query, the changed visual rows, and the new probabilities.
11. Generate the remaining answer, keeping the image strip visible throughout.
12. Supply the observed response, show answer-only loss and the computational graph, then one optimizer step and both images' new predictions.

### An important finding from the actual model

For the current `before` checkpoint, the first query is approximately `(0.628, 0.401, 0.969)` for both images. This is expected in this one-layer decoder: the incoming `?` row and its position are the same. The visual sources differ.

| First answer prediction, same prompt | Two-block image | One-block image |
| --- | ---: | ---: |
| Total attention weight on the four image rows | 5.04% | 16.04% |
| Probability of `two` | 0.782 | 0.159 |
| Probability of `one` | 0.216 | 0.794 |
| `logit(two) - logit(one)` | 1.286 | -1.610 |

These are recalculated outputs, not proposed illustrative values. They show why a generic “the model looks at the two objects” explanation would be misleading. The two-block answer does not coincide with high total visual attention. Value magnitude, output projection, residual input, and the competition with text sources all matter.

For an optional explanatory figure, this one-head, linear-head toy permits an exact breakdown of the logit difference:

`z_two - z_one = residual-and-bias contribution + sum_j alpha_ij [v_j W_O (u_two - u_one)]`.

Here `u_two` and `u_one` are columns of the vocabulary matrix. Show a simple signed contribution chart rather than asking students to derive this equation. I checked that the per-source terms sum to the implemented contrast for both images. It describes this forward computation; it is not a general causal attribution method for arbitrary VLMs.

If these fitted numbers still obstruct the first explanation, add a small hand-chosen introductory calculation with interpretable projections, clearly separated from the fitted checkpoint. Recompute its complete forward pass. Do not change labels on learned coordinates or draw an attention map that the fitted model did not produce.

### Keep the receiver and the generated token distinct

The generation strip should say:

| Last known token supplying the query | Next predicted token |
| --- | --- |
| `?` | `two` |
| `two` | `blocks` |
| `blocks` | `<eos>` |

A map explaining the prediction of `two` must use the `?` row. Recomputing with the `two` query explains the following token instead. Keep the original image above this strip so the sequence does not appear to become text-only after the first step.

### Be precise about spatial overlays

G already contains contextual visual rows. Each row has mixed information from the image before the language decoder reads it. A decoder attention overlay therefore describes weights on visual token slots, not a direct measurement of which raw pixels caused the answer.

Also show attention assigned to text. If we renormalize only the visual subset to make a heatmap visible, label that conditional normalization and display its original total mass. A bright-looking map must not conceal that only 5% of the total weight went to the image rows.

### Keep the failed update

The current update improves the two-block response loss from about 0.085 to 0.008, while worsening the one-block response loss from about 0.079 to 0.637 and changing its generated answer incorrectly. This is worth teaching. It shows why checking one improved example is not enough.

Keep response-only loss, the frozen-versus-trainable graph, and the explanation that a frozen language model can still transmit gradients to its input connector. Show the origin of the fitted checkpoint more clearly. The present `before` label means before this extra update, not before all training.

Move the thermal calibration material into a clearly marked optional application or provide its context locally. It currently asks the learner to recover too much from an earlier external article. The unanswerable mug-temperature question is sufficient for the main lesson's limit test.

**Completion question:** Can a student identify the query that predicts each answer token, trace an image value into that prediction, and explain why an update that helps one example can hurt another?

## 8. Fix notation where it actually conflicts

1. **P is overloaded.** Vision I's notation card uses P for patch side length, while its training graph uses “CLS, P” for learned positional inputs. Vision IV also uses P as the position matrix. Keep P for the position matrix across the series and use a distinct patch-side symbol such as `s_patch`. Update the pixel-row shape accordingly.
2. **Introduce B at the connector.** Vision IV defines `B = G W_bridge + b`, but B is missing from its notation card. Add it with `N × d_model` and the concrete `4×3` shape.
3. **Keep the representation chain beside the relevant operation.** For vision: raw pixels r_j, patch embedding, position-added e_j, q/k/v projections, message m_i, update Δe_i, updated e'_i. For the VLM connector: G, then B, then combined E. Do not require a trip to the notation card to understand a newly introduced letter.
4. **Name only demonstrably interpretable coordinates.** The initial patch map computes left/right pixel sums. After adding positions, applying contextual mixing, or learning arbitrary weights, those entries no longer have that unchanged literal meaning. Say what transformed, then use neutral coordinate labels where necessary.
5. **Keep three different normalizations separate.** Attention softmax is over source positions. CLIP normalization creates unit vectors, followed by a softmax over supplied candidates. Vocabulary softmax is over next-token choices. Label the denominator's set at each transition.
6. **Do not make position an extra “location feature” column.** The current full-width addition is the convention to retain. When coordinates have hand-chosen meanings, distinguish the content contribution from its positional offset.
7. **Use local notation introductions even when the card is complete.** The current cards are more complete than Claude reports. A complete glossary does not fix a symbol that appears too early in a slide.

## 9. Make the prose sound like teaching

I used the humanizer skill for this review. The article edits should preserve their care with evidence while reducing repetitive warnings and abstract labels. The human quality comes from explaining a particular difficulty and helping the learner through it, not from adding chatty fillers.

Here are concrete rewrites to guide the later edit. They are proposed replacements, not changes already made.

### MAE: lead with the operation

Current: “This is an exact loss calculation, not a trained image reconstructor.”

Proposed, until the trained example exists: “For now, we supplied the four guesses ourselves. This table scores them. We still need an encoder and decoder to produce the guesses.”

This preserves the boundary and names the missing step. Once a trained example is added, replace the sentence with a reference to the actual decoder output.

### Values: explain the missing information

Current: “A scalar amount alone cannot update a two-number row.”

Proposed: “The attention weight tells CLS how much to read from this patch. The value vector tells it which numbers to add.”

Follow immediately with the selected patch, its value, and the multiplication. The sentence should point to something visible.

### VLM inputs: replace a general claim with the concrete rows

Current: “The connector supplies vectors, but an attention mask controls how they are used.”

Proposed: “Put the four image rows first, followed by `<bos> count ?`. The question-mark row can read all seven known rows. Later, when we include answer tokens for training, it must not read those future answers.”

### Evaluation: ask a question the example can answer

Current title: “What did the encoder learn?”

Proposed before adding a real probe: “Would these features help classify a new image?”

Proposed after adding a real probe: use a result-specific title, such as “The probe succeeds on these layouts but fails on this change,” only if that is what the experiment shows.

### Learning: interpret one visible change

Current: “Pretraining produces encoder parameters, not a guarantee of task performance.”

Proposed: “The reconstruction loss has fallen. Now move the block. Does the encoder still give the classifier useful information?”

This replacement belongs only after the new reconstruction experiment has actually produced the stated result.

### Editing rules for the whole series

- Name the image, patch, receiving token, or prediction under discussion. Replace an abstract “representation” with its role when that role is known.
- Keep a limitation next to the claim it qualifies. Put implementation details and repeated qualifications in one companion note rather than repeating them on successive slides.
- Preserve genuine questions whose answers are shown next. Remove rhetorical questions that lead only to another definition.
- Let the longform article explain the transitions between frames. It should not read as a sequence of captions pasted together.
- Put essential meaning in the article and on the teaching frame. Presenter notes may help the instructor, but they should not be the only place explaining where a query or target came from.
- Use short PyTorch snippets after their operation is understood. State tensor shapes and whether code is executable or schematic. Avoid a two-line snippet that hides five unexplained helper functions.

## 10. Figure and presentation standards for this revision

More pictures will help if they reveal part of the computation. Repeating the same photograph above unrelated equations will not solve the problem.

| Figure | Source of its content | Incremental stages | What the learner should infer |
| --- | --- | --- | --- |
| Patch projection | Actual crop and W_patch | Crop, flatten, first column, second column, perturbation | A patch embedding is computed from pixels, and a projection can lose information |
| Spatial attention and values | Actual A, V, and source IDs | Receiver, source weights, one value contribution, sum, residual | Matching and sending information are different jobs |
| MAE reconstruction | Actual model predictions | Restricted input, prediction, target, error, updated prediction | The target is available to the loss while hidden from the predictor |
| DINO views and distributions | Specified views, encoder/head outputs | Two views, distributions, loss, student update, EMA | Agreement is learned across views; identical outputs for all images would lose information |
| CLIP geometry | Actual normalized vectors | Paired items, vector positions, comparisons, update | The pair objective changes a shared matching space |
| VLM answer evidence | Actual prefix-decoder computation | Image rows, text receiver, contributions, logits, changed image | Visual inputs alter a particular next-token distribution |
| Generation | Actual decoder trace | Known prefix, current receiver, predicted token, append, repeat | The query source changes as the answer grows |

For presentation mode, split a long explanation at a meaningful intermediate result. Do not shrink the type or introduce an internal scrollbar to preserve a one-frame layout. Keep a small persistent source image or token strip when continuation frames need it.

For article mode, unfold the same stages with connective prose. Readers should be able to understand a final state without having watched an animation. Every interaction needs a useful initial state, labeled controls, and an explanation of what changed.

For PDF, export a declared state or selected before/after states. Include reveal answers, source labels, legends, and model provenance. A hover-only patch value or an animation-only reconstruction is missing content in the PDF.

## 11. A practical revision order

### First: close the largest explanatory holes

1. Fix the P collision, introduce B locally, and add the architecture-versus-checkpoint transition notes.
2. Build Vision II's complete MAE-style forward, update, reconstruction, and held-out test.
3. Fill Vision IV's missing per-source value contributions and same-prompt/different-image comparison. Keep the image visible through generation.

### Next: give the calculations a visual interpretation

4. Add Vision III's actual embedding geometry, separate temperature effects, and connect candidate matching to the old classifier's limitation.
5. Add Vision I's source-linked spatial readout and projection-information-loss example. Improve the interpretation of its existing training and failure figures.
6. Ground DINO distributions and I-JEPA targets in specified image computations. Add the evaluation result or narrow the claim until it exists.

### Then: read and present it as a lesson

7. Humanize each section after the calculations and figures are settled. Check whether every paragraph answers the learner's next likely question.
8. Recheck article, presentation, and PDF modes, including reveal answers and continuation frames.
9. Keep detailed matrices available for inspection, but remove duplicated arithmetic from the main route when it no longer teaches a new idea.

Do not begin with a new CSS redesign, a universal image-generation pass, or a fixed word-count target. Those would consume effort before addressing the main gaps.

## 12. Acceptance checks before calling it finished

### Teaching

- Each part opens with a task, returns to it after the worked example, and shows a result that answers it.
- New symbols appear with a source, a role, and a shape before they are used in a calculation.
- Every worked example carries one identifiable image through its forward pass.
- The student can distinguish a supplied number, a computed number, an updated parameter, and an illustrative sketch.
- Vision II includes actual learning from images, not only manual adjustment of output guesses.
- Vision IV identifies the last known token for every prediction and shows the contribution path from image rows to logits.
- Every claimed capability is supported by a suitable test. A two-example fit is not described as general object counting or broad visual understanding.

### Numerical and experimental

- Derive figures, tables, and saved states from the same calculation; do not maintain separate hard-coded “nice” diagrams.
- Preserve the current regression checks and add tests for new reconstructions, checkpoint states, masks, and contribution sums.
- Use full precision internally, a consistent displayed precision, and an explicit rounding note where displayed terms do not sum exactly.
- Keep training and held-out examples separate; include simple baselines for new self-supervised claims.
- Do not promise monotonic improvement for each example, a particular collapse outcome, or semantically named learned axes without evidence.

### Classroom and reading modes

- No frame needs an internal scrollbar at the supported presentation sizes.
- A student at the back can read the source labels and the one calculation being discussed.
- Long tables become coherent continuation frames, not tiny text.
- Figures still explain something when printed; reveal answers appear in the PDF.
- Longform readers get the reasoning between stages, not just the revealed slide fragments.

## Bottom line

The series has enough working numerical material to support a much better lesson. I would spend the next pass on the learner's path through that material: make one self-supervised model visibly learn, make the image's effect on a generated answer traceable, and keep sources attached to their vectors throughout. That will bring the vision parts closer to the text series more effectively than making every part longer by the same amount.
