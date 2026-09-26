# Vision I: research and redesign plan

Status: complete local Vision I lecture built, 26 September 2026. The initial 19-frame prototype has been expanded into **140 teaching frames plus cover**, integrated into `../vision1.html`. See [the teaching guide](VISION1_TEACHING_GUIDE.md) and [the executed full lab](../notebooks/vision/03_vision_transformer_lab.ipynb).

The completed lecture includes real-photo motivation; the entire two-head calculation; full-block code; actual training on 512 independently generated images with separate validation and test splits; and measured real-photo predictions, attention maps and occlusion. The original proposed Pets species fine-tuning benchmark was replaced in the teaching sequence by a controlled synthetic training experiment and an explicitly separate pretrained ImageNet demonstration. No Pets benchmark result is claimed. The research and design notes below retain the original proposals for provenance.

## 1. What needs to improve

The current lesson covers the operations, but the experience of discovering them is weaker than Parts I–III. The redesign should make students want the next operation, let them predict its effect, and then give them numbers that settle the question.

Concrete findings from the existing material:

| Current choice | Consequence | Proposed change |
|---|---|---|
| The photograph illustrates a scene that the numerical model never receives (`sections5/sec02.html`). | The appealing image and the calculation lead separate lives. | A real photograph remains the anchor for the architecture and the final real inference. A small arithmetic exercise gets an explicit transition and an honest geometric task. |
| Values 0–3 effectively identify table, plant, book, and mug in the synthetic scene. | Much of the difficult visual recognition problem has already been supplied by construction. | Use visibly simple geometry for the worksheet; reserve object recognition claims for measured inference on real images. |
| Section 3 introduces named axes, a projection table, two full sets of content rows, and a scatter plot. | Students do bookkeeping before seeing what a patch representation enables. | Keep the source crop on screen; flatten it; compute one output coordinate; then reuse the same projection on a second crop. |
| Section 10 introduces a separate hand-chosen two-head classifier after the trained one-head model. Its output projection is the identity and its class head reads only coordinate four. | Multihead attention becomes an appendix, and the reason for combining head outputs is hard to see. | Carry one two-head worksheet through routing, values, a nontrivial output projection, residual, and prediction. Both head messages should influence the displayed result. |
| Two training scenes, one held-out probe, and a pretrained model with an unrelated ImageNet answer form the empirical ending. | The lesson lacks a satisfying answer to its opening question. | Use a checkpoint with an explicitly matching label space and report a real held-out evaluation, including errors. |
| Presentation frames often contain cards, full tables, or several sentences. | The look and pacing diverge from the existing whiteboard rules. | Follow `STYLE_WHITEBOARD.md`: a held drawing, one new mark per reveal, big type, numbers where they are used, and at most 40 words of prose. |

Part II's “Which earlier tokens should bank read?” is the local model: a particular receiver has a particular uncertainty; the drawing stays stable as scores become weights and weights mix values. Part III's coat example motivates independent readings before introducing head dimensions. Those teaching moves should carry into vision.

## 2. What the supplied references contribute

Review method: article and notebook text/code, PDF text plus selected rendered pages, and YouTube transcripts with selected silent still-frame inspection. This was not uninterrupted viewing of every video. Video timestamps below are approximate navigation points from captions. Playback was muted; the temporary video tab was closed after review.

| Reference | Most useful contribution | How to use it here |
|---|---|---|
| [Encord: Vision Transformers](https://encord.com/blog/vision-transformers/) | Accessible architecture overview and visual application context. | Use as an orientation reference; derive equations and technical claims from primary sources. |
| [Dive into Deep Learning: ViT](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html) | Patch embedding, encoder block, and classifier connect directly to executable code. Exercises include pooling and patch-size choices. | The pattern for a short diagram-to-code bridge and useful student experiments. |
| [Paperspace: Vision Transformers Explained](https://blog.paperspace.com/vision-transformers/) | Grayscale patch diagrams make flattening and projection concrete. | Borrow the visual decomposition. Check dimensions independently: the article's multihead concatenation description is inconsistent. |
| [Gertjan Burghouts: Transformers for Vision](https://gertjanburghouts.github.io/pictures/transformers4vision.pdf) | Moves from ViT to detection, objects, interactions, and video. Page 25 juxtaposes attention distance and receptive fields. | A reference for the final vision roadmap. Its historical data-scaling conclusions need their original experimental context. |
| [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Repeated zooming from a whole computation into one token, then back out; consistent visual identity through Q/K/V, weighted values, heads, and output projection. | The principal external reference for explanatory pacing. Create original drawings using this lesson's notation. |
| [CodeEmporium: Vision Transformers — Explained!](https://www.youtube.com/watch?v=Hnsrm1ezI3g) | Approximately 10:00–15:00 follows image and token dimensions; 15:00–18:30 separates fine-tuning from pretraining. | Keep training, adapting a classifier, and inference visibly distinct. Put resolution interpolation in optional material. |
| [UvA Tutorial 15: Vision Transformers](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html) | Actual images become patch strips; the implementation leads to a saved-model experiment and comparison with CNNs. | Best supplied practical-lab model. Treat its CIFAR-10 outcome as a particular experiment, not a universal verdict on ViTs. |
| [UvA deep learning course](https://uvadlc.github.io/) | A route to the course's tutorial materials and archives. | Recommend the public notebooks directly. The current homepage says lecture documents and recordings are not publicly provided. |
| [AGI Lambda: Vision Transformer](https://www.youtube.com/watch?v=vJF3TBI8esQ) | The short sequence from image/RGB values to patches, projected vectors, positions, and classification. | Adapt the visual choreography at a slower teaching pace. Keep illustrative attention maps explicitly distinguished from measurements. |
| [DeepFindr: Vision Transformer Quick Guide](https://www.youtube.com/watch?v=j3VNqtJUoz0) | Approximately 2:00–8:00 connects patchification, tensor shapes, CLS, and positions to implementation; later sections assemble the model. | Pair each small piece of code with the drawing it implements. Avoid treating attention weights as guaranteed explanations. |
| [Engineering Visualized: What Is a Vision Transformer?](https://www.youtube.com/watch?v=vuqc2aD1dCg) | A recognizable dog stays central across MLP, CNN, and ViT motivation; around 3:00, separated visible parts motivate context. | Strong opening-story reference. Explain local-to-global CNN receptive fields accurately. |
| [Yannic Kilcher: An Image Is Worth 16×16 Words](https://www.youtube.com/watch?v=TrdevFK_am4&t=640s) | The requested 10:40 region and following minutes explain local patches, global interaction, positions, the shared projection, and CLS with annotations on the architecture. | Good lecturer preparation for “why this design?”; approximately 10:40–16:30 is the targeted passage. |

## 3. Additional international teaching shortlist

These are selected for a specific teaching strength, rather than institutional name alone. Some are lecture slides or notes, not public video recordings.

| Resource | Why it earns a place | Best use |
|---|---|---|
| **Stanford, USA — Justin Johnson, CS231n, Lecture 8 (2025)**: [slides](https://cs231n.stanford.edu/slides/2025/lecture_8.pdf), [video](https://www.youtube.com/watch?v=RQowiOF_FvQ), [course schedule](https://cs231n.stanford.edu/2025/schedule.html) | Slides 100–109 preserve one cat-image diagram while adding patches, projection, positions, transformer, and output. Slide 104 asks students to recognize patch projection as a strided convolution. | Strongest additional reference for the main classroom sequence. Its pooling example is a useful contrast after our CLS explanation. |
| **University of Amsterdam, Netherlands — Phillip Lippe, Tutorial 15**: [notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html) | Complete implementation, data splits, saved checkpoint, training curves, and a concrete CNN comparison. | Strongest practical companion; already in Nipun's list and worth keeping central. |
| **Carnegie Mellon, USA — 10-423/623/723, Lecture 5 (2025)**: [slides](https://www.cs.cmu.edu/~mgormley/courses/10423-f25/slides/lecture5-vision.pdf), [course](https://www.cs.cmu.edu/~mgormley/courses/10423-f25/) | The ViT section includes a pointed question about recovering 2D position from 1D positional embeddings. See PDF pages 51–53, with printed slide numbers 57–59. | Build prediction pauses around positions and data requirements. The ViT segment is short; it does not supply a full numerical walkthrough. |
| **Australian National University, Australia — Stephen Gould, Deep Learning for Computer Vision**: [lecture notes](https://users.cecs.anu.edu.au/~sgould/papers/comp8536_lecture_notes.pdf) | Pages 95–98 connect attention mathematics, sequence classification, and image tokenization; page 98 keeps actual patch crops beside the sequence. | A rigorous reading companion and source for the image-to-sequence bridge. |
| **Deep Learning IndabaX Tanzania — Anurag Arnab (2021)**: [lecture slides](https://www.robots.ox.ac.uk/~aarnab/talks/transformers_talk.pdf), [event attribution](https://www.robots.ox.ac.uk/~aarnab/) | Begins with contextual ambiguity, includes a permutation exercise, then extends image tokens to video tubelets. | Conceptual questions and a later video extension. This is an invited lecture in Tanzania, hosted on the author's Oxford site. |
| **IIT Hyderabad / NPTEL, India — Vineeth N Balasubramanian**: [Self-Attention and Transformers lecture page](https://dl4cv-nptel.github.io/DL4CVBK/notebooks/Week_9/Week_9_Lecture_5.html) | An accessible course path through attention in vision, with linked lecture video and slides. | Supplementary background for students needing the attention prerequisite; not the central ViT worked example. The public lesson page was verified; the full linked recording was not reviewed. |

Preparation priority: Stanford's ViT builds → the supplied Engineering Visualized/AGI Lambda visual sequences → Jay's numerical pacing → D2L's implementation → UvA's experiment. Arnab, ANU, and CMU deepen specific questions.

For technical grounding, use the [original ViT paper, Section 3 and Appendix A](https://arxiv.org/html/2010.11929v2). The small worksheet below is our own teaching construction.

## 4. Proposed example design

### Real image: a question that returns at the end

Use a clearly recognizable dog photograph, with a second cat image for comparison, from the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). Its species annotations support the proposed cat/dog task. Record the sample IDs, attribution, crop, resize, and normalization; the dataset page provides its license.

The first photograph must be an actual input to the final model. Preserve its identity through the grid, enlarged crop, token thumbnail, selected attention row, and final prediction. A patch can cross an object boundary; it is not an automatically discovered object.

For the real endpoint, adapt a pretrained ViT to a two-class species head using the official training partition and a separate validation split, then evaluate on the held-out test partition. Reserve the opening images from fitting. Record model identifier, pretrained data provenance, classifier mapping, and checkpoint. Do not promise a particular accuracy or invent the final answer before running it. Student-facing inference loads a saved checkpoint; long training is an optional lab.

### Arithmetic image: an honest visual task

Use a **4×4 grayscale image, divided into four 2×2 patches**, showing two occupied quadrants. Ask whether they form a horizontal or vertical pair. For example:

```text
Horizontal                  Vertical
1 1 | 1 1                   1 1 | 0 0
1 1 | 1 1                   1 1 | 0 0
----+----                   ----+----
0 0 | 0 0                   1 1 | 0 0
0 0 | 0 0                   1 1 | 0 0
```

The two images contain the same multiset of patches: two occupied and two empty. Their arrangement supplies the label. Students can understand the task without interpreting tiny blocks as mugs or plants.

Use this one worksheet throughout the arithmetic, with `d_model = 4`, two heads, `d_head = 2`, and four patch tokens plus CLS. Hand-chosen parameters are declared once. The worksheet illustrates a computation; success on these arrangements is not evidence of natural-image recognition or the necessity of multiple heads.

The transition is explicit: “We will shrink the image and model so every number is visible, then return to the photograph.” Keep notation, stage order, and visual grammar unchanged. A small scope annotation identifies the active example. No later switch to an unrelated one-head classifier.

## 5. Proposed lecture sequence: approximately 80 minutes

| Time | Student question | Drawing or worked result |
|---|---|---|
| 0–6 min | What is in this image? What can one crop tell us? | Dog photo → ambiguous crop → crop restored to context. Relate it briefly to “bank” from Part II. |
| 6–14 min | How can an image become the rows our transformer expects? | Fixed grid → actual crop strip → pixels → shared projection. First a picture, then dimensions. |
| 14–23 min | If the patches stay the same, can the answer change? | Introduce the four-patch worksheet. Rearrange occupied patches; predict what a model without positions can distinguish; add location vectors. Reserve a learned CLS row to hold the eventual image summary. |
| 23–34 min | How does one patch read other patches? | Keep one receiver fixed. Show Q/K comparison, every source score, normalization, and arrows returning to the source thumbnails. Recall Part II instead of re-deriving all attention theory. |
| 34–46 min | What information comes back? What changes with two heads? | Values beside their own weights; products; two messages; concatenation; nontrivial output projection; residual. Reuse Part III's distinction between independent mixtures. |
| 46–54 min | How does the summary row become an image answer? | Return to the CLS row introduced alongside the patches; work its readout and two class scores. Distinguish attention softmax over sources from class softmax over labels. |
| 54–62 min | What turns this attention worksheet into a full ViT? | Restore pre-LayerNorm, tokenwise MLP, residual paths, and depth. Expand one block in the existing drawing. Show matching short code. |
| 62–70 min | Does it answer the original photograph's question? | Load the species checkpoint; run the opening image; show actual probabilities, a held-out score, and representative errors. Identify what was pretrained and what was fitted here. |
| 70–77 min | What changes if we move, hide, or resize patches? | Position ablation; measured occlusion experiment; patch-size/token-count control. Ask for a prediction before revealing each outcome. |
| 77–80 min | What carries forward to the rest of vision? | One completed image-to-label drawing; brief bridge to label-free features, dense outputs, and vision-language models. |

Target roughly 35–45 main presentation frames, with short builds where needed. This is a pacing estimate, not a quality target. Full tables and longer derivations belong in reading mode and the notebook.

### The numerical contract

The complete computation should exist before drawing its frames:

1. Flatten one actual 2×2 worksheet patch into four scalars. Expand a projection coordinate as all four products and their sum. Reuse the same projection for every patch.
2. Add content and position vectors visibly. Show all five initial rows once, with thumbnail labels.
3. For one receiving patch and one head, calculate its two-coordinate query and all five keys; include the receiver itself and CLS. Expand at least one dot product, divide by `sqrt(2)`, and show all five scores and the full softmax denominator.
4. Keep the five source labels fixed while displaying their two-coordinate values, weighted contributions, and sum. The attention weight multiplies both coordinates of a value.
5. Repeat the routing/mixing for the second head. Its numbers must make the different mixture visible. Avoid assigning permanent semantic jobs such as “the shape head.”
6. Concatenate two two-coordinate messages into a four-coordinate row. Use a non-identity `4×4` output projection with a visible contribution from each head, then add the residual. Work the relevant CLS calculation with the same parameters for the final readout.
7. Multiply the final CLS row by a `4×2` class matrix; show both logits, class probabilities, and the loss for the known label. Return to the horizontal/vertical picture beside the answer.
8. Re-run the changed arrangement and compare the result. Show a single parameter update as a learning illustration, then explain that real learning repeats this over many examples. No benchmark claims from the worksheet.

For clarity, the first arithmetic pass isolates attention, its residual, and the class readout. Label the omitted LayerNorm/MLP once. The subsequent full-block code and diagram must include those operations; their outputs must be computed separately rather than presented as identical to the simplified pass.

## 6. Twelve-frame prototype for the next stage

After agreement on this plan, make a small representative prototype before rebuilding the full lesson. Include the difficult numerical frames as well as the opening, so the review can judge the whole teaching style.

| Frame | Question before the reveal | Held drawing and next mark |
|---|---|---|
| 1 | “What animal is this?” | The real photo and an unfilled answer slot. |
| 2 | “Would this crop be enough?” | Same photo location; enlarge one outlined crop, then restore its surroundings. |
| 3 | “What should count as a token?” | Keep the photo; add a regular patch grid. |
| 4 | “Where did this row come from?” | Pull actual crops into a strip; retain source IDs and image miniatures. |
| 5 | “How do pixels become an embedding?” | Introduce the small worksheet; open one patch into four numbers and one projection sum. |
| 6 | “Same patches—same answer?” | Horizontal and vertical arrangements; slide the same patch pieces between fixed locations. |
| 7 | “Which source gets the larger weight?” | One fixed receiver, five sources, two dot products revealed first, then the complete score row. |
| 8 | “What does that weight multiply?” | Preserve the same source order; reveal values, products, and the resulting message. |
| 9 | “What does the other head contribute?” | Hold the first message; add the second head's independent weights and message. |
| 10 | “How do both messages change this row?” | Concatenation → nontrivial projection → residual; reveal one coordinate's arithmetic at a time. |
| 11 | “How does one image get one prediction?” | CLS readout, two logits, and two class probabilities next to the worksheet image. |
| 12 | “Can the real model answer our opening question?” | Return to the exact opening photograph; show the full model path and measured output when the checkpoint is ready. |

Frames 7–11 are selected middle-of-lesson samples, not a claim that the full explanation fits into five slides. No invented model result should appear in frame 12 during a visual-only prototype.

## 7. Visual and technical acceptance criteria

**Teaching and appearance**

- Begin each section with an example or question. At each transition, a student should be able to state why the next operation is needed.
- Keep patch identity traceable from image to row to attention edge. Use thumbnails and IDs; preserve the existing seven-object colors for mathematical roles.
- One held drawing per frame, one new mark per build, at most six builds before splitting, and at most 40 words of prose. Put a question and a pointing instruction in presenter notes.
- Keep numbers beside their source patches, arrows, or vector coordinates. Move complete matrix dumps and explanatory paragraphs to read mode.
- Verify the representative frames at 1280×720 and the reading layout on a phone. Core explanations must not depend on horizontally panning an oversized diagram.
- Check comprehension with prediction pauses: patch count; output width; permutation outcome; the sum of weights; the object multiplied by a weight; and the difference between the two softmax operations.

**Correctness**

- Keep row-vector notation and existing symbols from Parts II–III. Avoid reusing `H` for head count when it already denotes message matrices.
- In a deterministic transformer with shared token operations and no position signal, patch outputs are permutation equivariant; the fixed CLS readout is invariant to patch permutations. Moving patch contents while location embeddings stay fixed is a different experiment from merely reordering token-position pairs.
- Initial CLS is a learned vector shared across inputs. Its first-layer query does not already encode the current image. Later layers can read with an image-dependent summary.
- Head maps are learned mixing weights, not guaranteed object masks or causal explanations. Label layer, head, receiver, and source axes in every measured visualization.
- CNNs can develop global receptive fields. Flattening retains values and their indexed order; the architectural treatment of spatial structure is the issue.
- Patch-size controls show exact token counts and attention-score counts. They must not label the quadratic score-matrix ratio as the runtime ratio of the entire model.
- One executable source generates worksheet numbers, diagram annotations, and notebook checks. Verify the complete trace against an independent NumPy/PyTorch computation and check the changed-arrangement case.
- Keep real-image preprocessing, labels, checkpoint provenance, and evaluation split inspectable. Display measured failures alongside successes.

## 8. Work sequence after plan review

1. **Freeze the story and examples.** Select the attributed real images, construct the four-patch worksheet, and calculate parameters that yield a readable trace. Confirm the classifier uses both head messages. Record a manifest of every input, parameter, and output.
2. **Build the twelve representative frames.** Review continuity, pacing, numerical legibility, and resemblance to Parts I–III before expanding the deck.
3. **Rebuild Vision I and its companion lab.** Integrate the worksheet into the main sequence; create matching code; run and save the real-image classifier experiment. Keep lengthy training outside the live-lecture path.
4. **Audit the complete lesson.** Check numerical consistency, frame layouts, reading mode, links, and the student questions above. Preserve downstream use of the existing vision toy, especially Vision IV, through a compatibility layer or a separate new worksheet module.
5. **Review, then publish.** Publication follows the agreed implementation workflow. This research pass does not alter the live lesson.

Keep the existing notation, navigation, read/present modes, and useful code infrastructure. Rework the example and sequence. Move the current two-image fitting experiment and bulky tables to optional historical/lab material if they remain useful. Detection, segmentation, and video stay a short roadmap here; their detailed lessons get their own subsequent plans.

The central standard: a student should be able to explain what information each step adds, trace a particular patch through the computation, and connect the resulting class score back to the image.
