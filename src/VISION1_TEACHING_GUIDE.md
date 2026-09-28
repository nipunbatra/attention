# Vision I — complete teaching guide

**Deck:** [vision1.html](../vision1.html) · **Present:** open the deck and press **P** · **Lab:** [03_vision_transformer_lab.ipynb](../notebooks/vision/03_vision_transformer_lab.ipynb)

211 teaching frames plus cover; 14 sections. Silent, self-contained HTML slides with image assets and math embedded. Reading mode includes the longer explanations, source links, and numerical tables. Arrow keys advance one reveal; **S** opens presenter notes; **O** opens the overview; **C** shows classroom controls. Every frame has a question to ask and a note about what to point at.

## The teaching thread

Start with six labeled Oxford-IIIT Pet examples. Establish the dataset size (7,349), species task (2 classes), alternative breed task (37 classes), and variable original dimensions versus 224×224 RGB model inputs. Then choose one photograph and ask students to name the clues. Isolate one genuine crop and restore its context. Recall **aabid** from Part I, **river/bank** from Part II, and **red/wool/coat** from Part III. Then introduce patches, a tiny exact worksheet, the complete block, executable code, learning, and the original photograph again.

Introduce the opening task examples explicitly: classification returns an image label, detection returns labels and boxes, captioning generates a sentence, and image–text search ranks photos for supplied words. Use the same dog throughout. These are illustrative desired outputs. After the task tour, “Our task today” selects image classification; the text comparison and dark crop then motivate attention. The captioning and retrieval examples also preview the later vision-to-language lessons.

Use the three opening context slides as one held example. Identify the dark receiver, reveal the face source, and trace the arrow into the receiving numerical row. Add the branch source and compare illustrative contributions; thickness is not measured attention. Then show the weighted message added to the current patch row. Keep the unchanged pixels visible. The target remains one image label; students are not training a fur-versus-background classifier for each patch. The text recap and question-led parallels come next, before the diagram that introduces the patch-row construction.

Before “How can we give this photograph to attention?”, use the seven-slide text-to-image bridge. The recap redraws the Part II fisherman/river-bank computation with tokens, initial embeddings plus position, attention and updated rows. Trace bank at position 7, then distinguish the final the at position 10 used for next-token prediction. Ask students for an image token, representation, query, key, value and target before revealing each counterpart. The bank embedding is read directly from the original text toy. Query/key/value diagrams share a layout so students can reuse the roles. Only then return to constructing rows from actual pixels.

The two-crop Q/K/V warm-up and the pooling example use their own clearly labeled, hand-chosen numbers. Then keep the three larger settings explicit:

1. **Hand worksheet:** a 4×4 binary image, four 2×2 patches, D=4, two heads of width 2, five rows including CLS. Chosen weights; no LayerNorm or block MLP. Labels name two specific arrangements. Students can calculate every number.
2. **Trained small ViT:** 8×8 noisy grayscale images, sixteen 2×2 patches, D=16, two complete pre-LayerNorm blocks, two heads per block and MLP width 32. All trainable components learn. Data splits are independent random draws, with opposite-label pairs sharing exactly the same patch multiset.
3. **Pretrained real ViT:** `vit_tiny_patch16_224.augreg_in21k_ft_in1k`; 224×224 RGB, 196 patches plus CLS, D=192, 12 blocks, three heads per block, 1,000 ImageNet outputs. Exact photos, preprocessing, probabilities, attention arrays and interventions are saved.

## One model map, then a deliberate worksheet

Each section opens on a tinted panel with an explicit **SECTION** label and large section number. These chapter breaks use their own CSS. Worked-image steps instead use the eyebrow **Photo walkthrough · Step n of 12** above the recurring model diagram. The main slide title names the operation without a competing number.

Section 2 starts with the full architecture SVG: image → patches → projection → CLS + position → attention → MLP → final CLS → class scores → softmax. A compact version uses exactly the same box order on detailed slides and highlights the active operation. The full drawing shows both LayerNorm operations and residual paths, as well as the 12-block repetition. This is the actual pretrained ViT-Tiny: D=192, three 64-feature heads, and MLP hidden width 768.

After step 10 has stacked the patch content rows, use the three photograph-based position slides. Move the same face patches between image slots; their content stays fixed but their locations change. Then perform the measured position addition in step 11. This puts the motivation before the calculation. Keep the distinction between moving content among fixed slots and reordering whole content-plus-position pairs.

Step 12 follows one patch into Q/K/V. The next full-map checkpoint explicitly says that this is not yet an image prediction. Section 3 completes the **same photograph and same checkpoint**: introduce CLS, assemble 197 positioned rows, compare queries with keys, mix values, join heads, add the attention residual, run the MLP and its residual, repeat all 12 blocks, normalize and read CLS, and compute 1,000 ImageNet class scores and probabilities. CLS is present before attention in the actual computation; it is not appended after the blocks.

The opening cat/dog question motivates image classification. The pretrained example predicts 1,000 ImageNet labels; the slides name that change explicitly. Its measured top label is Newfoundland (95.73%), not a claimed test accuracy. `notebooks/vision/trace_real_classifier.py` verifies the explicit first-block attention, both residuals, MLP, complete 12-block path and final logits against the checkpoint. `figures/vision1/real-classifier-path.json` stores the shapes, selected vectors, source weights and top-three outputs.

Only after the real prediction does Section 4 introduce the smaller worksheet: classify the arrangement of four grayscale patches using chosen weights. The worksheet preserves hand calculations from the earlier lesson. It omits LayerNorm and the block MLP to keep the arithmetic manageable; Section 7 restores those operations. Do not present its numbers as a second pass through the dog model. The shared route diagram locates each calculation within the architecture.

## Suggested pacing

Use three meetings, or teach sections 1–7 first and assign the implementation as a lab. The 211 frames are short steps; the total is not a target for one class. Pause for predictions and hand calculations.

| Meeting | Sections | Student activity |
|---|---|---|
| A | 1–4 | Compare tasks, turn pixels into rows, reason about position, explain Q/K/V, calculate a first attention message |
| B | 5–7 | Work the second head, combine messages, predict a class, compare CLS with pooling, restore the full block |
| C / lab | 8–14 | Run the code, interpret the training control and real-image measurements, solve transfer exercises |

For a short conceptual introduction, use the task comparison, the two-crop Q/K/V example, CLS and pooling, the whole-block drawing and the real-photo predictions. Keep the full four-patch calculation for a session with time to work alongside the class.

## Dataset introduction

Three slides precede the single-photo question: `dataset-gallery`, `dataset-counts`, and `dataset-dimensions`. Read the species and breed under each example, add 3,680 and 3,669 to get 7,349, and compute 224×224×3=150,528 pixel values. The counts describe the full labeled dataset; the gallery contains six selected examples. The original photo files vary in size. Dimensions on the slide use height × width × channels. Our real checkpoint receives the displayed square crop, followed by channel normalization.

The opening uses the cat/dog task to establish input and target. The later synthetic training experiment and 1,000-class pretrained inference keep their existing scopes; these slides do not introduce a Pets training result. Sources, original dimensions and file hashes are recorded in `figures/vision1/dataset-intro.json`. Re-fetch the gallery with `python notebooks/vision/fetch_dataset_examples.py` (Pillow required).

## Recall both text prediction examples

The opening task recap keeps the name example from Part I, then adds the river-bank sentence from Part II before the text/image comparison. In the name model, the fixed character window goes through embedding lookup, concatenation and an MLP. In the attention model, the updated final “the” row predicts the word after the whole prefix. “Water” is a plausible continuation, not a new measured output. Distinguish bank’s contextual row from the final row used for this next-word prediction. Both examples choose a token, append it, and predict again; their token units and architectures differ.

## Give the visual queries a concrete purpose

After the text/image query diagram, show three receiver examples: a dark coat patch, a partial face and a branch. Locate each receiver in the whole photograph before revealing a possible question and two actual source crops. All use the same query projection within a head/layer; the receiver row changes. Keep the task fixed: one image label. These are possible learned behaviours, with no patch-level labels or claimed measured head meanings.

For keys and values, reuse the same eye/muzzle, coat and branch crops. A key supplies matching features; its relevance depends on the receiver's query. A value supplies visual information to mix, and the source's value is shared across receivers even though their weights can differ. Explain a₁₀,₇ as P10 reading P7, then reveal the symbolic sum. The attention part of Section 4 reconnects the roles to the fully numerical face/branches example, then works out the four-patch attention matrix.

## Draw the patch layer as a 12-to-2 network

The weight introduction uses twelve nodes containing the actual RGB values, connected to two output nodes. Keep this drawing for both calculations. Reveal the four nonzero incoming weights for the selected output; its other eight weights are zero. Read the source values along those edges, add their contributions, then add the bias. The two results remain 2.5 and −0.5. The outputs are two embedding features of one patch, not two class scores. Mobile reading mode keeps a taller version of the network diagram.

## State where the activation appears

The text/image embedding comparison now says that the patch output has no activation. The next slide follows the patch's single affine map xW+b and separately shows the later block MLP: Linear(D,H) → GELU → Linear(H,D). Keep D (embedding width) and H (hidden width) explicit. The later worked patch output retains its negative coordinate. The absence of a patch activation does not make the full Transformer linear; attention softmax and LayerNorm also appear in the complete model.

## Introduce terms before using them

The opening task comparison uses “one image summary.” The full-model map first names CLS as the classification token, and Section 3 explains its learned starting row before assembling the full sequence. Section 4 then chooses a small CLS vector for the worksheet. Section 6 names mean pooling alongside the coordinate-by-coordinate average. The previous text lessons supply the familiar operations; new vision terms are defined where their role becomes visible.

## Opening transition and later comparison

After the ambiguous crop and its face clues, `image-to-rows` recalls the input attention needs: rows of numbers. It previews photograph → patches → one row per patch → attention. The next section constructs those inputs step by step. The CNN comparison now follows the worked Conv2d patch projection in section 8, where both local filters and attention have been explained. Its two-layer dependency diagram distinguishes growing local context from a direct global-attention path; the next slide implements the attention calculation.

## Build the patch embedding before naming its output

Section 2 keeps the RGB entries grouped by pixel A, B, C and D. Count 2×2×3=12 entries, then compare a text embedding lookup with a computed patch embedding. Use `nn.Linear(12, 2)` for a hand calculation: show all 24 weights and two biases, calculate each coordinate, and collect `[2.5, −0.5]`. There is no activation after this patch layer; the separate block MLP later uses two linear layers with GELU between them.

Reuse the exact same layer on a second patch to get `[0.5, −2.5]`. A single call maps `X` of shape `(2,12)` to `C` of shape `(2,2)`. These two output features are not two class scores. `cᵢ` names the content row of patch i; `eᵢ=cᵢ+pᵢ` adds position before attention. Subscripts identify patches, while D gives the embedding width. The real-photo trace returns to the same dog with the checkpoint’s actual patch grid, then defines each row along its full computational path.

Then scale to the saved real model: 16×16×3=768 input values, `nn.Linear(768,192)`, and 196 rows for a 224×224 input. The row-vector equation uses W shaped `(768,192)`; PyTorch stores the transposed weight `(192,768)`. This is equivalent to the checkpoint's Conv2d patch embedding with the corresponding input order. `figures/vision1/patch-embedding-example.json` contains the independently executed warm-up. The lab contains the same calculation as an editable, executed code cell.

## One continuous photograph-to-embeddings walkthrough

The real-image run in section 2 is a twelve-step sequence. Each operation consumes the preceding slide’s output; the same prepared dog image and patch identities stay visible throughout. The earlier 12-input, 2-output network remains the hand-calculation warm-up.

1. **Start with the image:** show the actual 224×224×3 model input and identify each axis.
2. **Cut into patches:** draw 16×16 boundaries on the photograph, with x pointing right and y pointing down. Mark the 0-to-224 pixel extent on both axes. Save the count for the next slide.
3. **Number the pieces:** show actual crops in a truncated row-major grid: P1, P2, …, P14; P15, P16, …, P28; …; P183, P184, …, P196. Reveal 224÷16=14 on each axis, then 14×14=196 and the array shape 196×16×16×3. Pause before each answer.
4. **Read one patch’s RGB:** keep P63 visible, outline its first two pixels, and read [16,17,12] and [41,42,37]. Count 256 pixels × 3 channels = 768 values.
5. **Normalize the same values:** explain (value/255−0.5)/0.5 before displaying negative inputs. The checkpoint normally normalizes before patch extraction; this independent channel operation gives the identical result on the extracted crop.
6. **Flatten:** carry those normalized triples into x₆₃, shape 1×768. Follow the scan arrow and keep RGB together for each pixel.
7. **Apply the shared projection:** nn.Linear(768,192), W shaped 768×192 in row notation and 192 bias entries. No activation follows. The same 147,648 parameters serve every patch.
8. **Inspect its output:** c₆₃ contains 192 measured features. Show the first three and the last coordinate with their indices.
9. **Repeat for P64:** retain P63’s path while revealing the neighboring crop, its different input values and its different output. Both paths cross the same layer.
10. **Stack all output rows:** X (196×768) becomes C (196×192). Match representative image crops to their actual output vectors. Reading mode includes a collapsible table of all 196 rows’ first three coordinates; the complete vectors are in the saved JSON.
11. **Add position:** illustrate c₆₃+p₆₃=e₆₃, then show C+P=E for all patch rows, each matrix 196×192. The preceding three-slide photograph interlude motivates this position information before the addition.
12. **Enter the first block:** LayerNorm preserves the 192-feature width. Each Q/K/V projection for one of three heads has 192×64 weights and 64 biases, producing a 1×64 row for this patch.

P63 is row 5, column 7; P64 is row 5, column 8. The subscript is a patch identity, while 768 and 192 are feature counts. All shapes omit the batch axis because the walkthrough follows one image. This grid differs explicitly from the earlier coarse 4×4 illustration.

Reproduce the trace with `uv run --with timm --with pillow python notebooks/vision/trace_real_patch.py`. `figures/vision1/real-patch-path.json` stores raw RGB values and normalized rows for selected patches, all 196 full output embeddings, position addition, first-head Q/K/V vectors, model identity and preprocessing. The script checks the displayed input, linear/Conv2d equivalence for every patch, position addition and Q/K/V slices against the pretrained checkpoint. Printed decimals are rounded.

## Places to stop and ask

Use the [complete slide map](VISION1_SLIDE_MAP.md) for current frame numbers and direct presentation links.

- **`patch-context`:** What is the whole-image task? Could the isolated dark crop be fur, shadow or background? Reveal the eye and face clues, then explain how richer local information could support the image label. The arrows illustrate possible context, not measured attention.
- **`task-side-by-side`:** What is the input, target and readout in each task? Why does the loss still look familiar?
- **`task-mask-reason`:** Which target would a future training token reveal? Why is a later raster patch already available?
- **`cls-start`, `cls-two-images`:** How can a shared initial vector lead to different image summaries?
- **`qkv-match-numbers`:** Exponentials 3 and 1 give which two shares?
- **`qkv-read-numbers`:** What does each source send after weighting? Is the result a patch index or a vector?
- **`qkv-change-key`, `qkv-change-value`:** Which intervention changes the heatmap? Which changes the message?
- **`weight-denominator`:** Why does CLS belong in the denominator too?
- **`one-value-product`:** Can an empty patch still send position information?
- **`s04-join`:** Which operation joins messages, and which actually mixes their coordinates?
- **`two-softmaxes`:** Are the alternatives source rows or class labels?
- **`pooling-example`:** Calculate the mean. Does image classification require CLS?
- **`conv-one-patch`, `conv-trainable`:** Calculate one filter output, then count its learned weights. Why does Conv2d match our linear patch projection?
- **`batch-axis`, `batch-check`:** Average three rows in A, then average the first rows across A and B. Which result should be unchanged when another image enters the batch?
- **`trained-position-control`:** Why can the chosen architecture not separate opposite-label pairs without positions?
- **`cover-1` through `cover-4`:** Predict the change before revealing each measured probability.

## Main calculation answers

For “Across the top”, E rows are CLS `[0,0,0,1]`, P1 `[1,0,0,1]`, P2 `[1,0,1,1]`, P3 `[0,1,0,1]`, P4 `[0,1,1,1]`.

- Head 1 CLS message: `[0.457888, 0.457888]`.
- Head 2 CLS message: `[0.681748, 0.681748]`.
- After concatenate and W_O: `ΔCLS = [-0.223860, 0.457888, 1.139636, 0.681748]`.
- After residual: `[-0.223860, 0.457888, 1.139636, 1.681748]`.
- Class logits: `[0.895440, -0.895440]`; probabilities `[0.857035, 0.142965]`.
- The new class-bias gradient is `[-0.142965, +0.142965]`. One SGD step at 0.5 improves the correct-class probability to about 0.874.
- In this deliberately simplified worksheet all queries are `[1,1]`, hence attention rows within a head are equal. The residual rows still differ. Real-model patch-query maps show the more general case.

## What the measured experiments establish

The small ViT is trained on **512 images**, selected on **128 validation images**, then evaluated on **256 test images**. With positions it gets **256/256**; without positions it gets **128/256**. One deterministic seed, 80 epochs, AdamW, batch 64, learning rate .003, weight decay .01. This is in-distribution synthetic generalization; do not present it as a natural-image benchmark.

For the exact Newfoundland photograph, the pretrained model gives **95.73%** Newfoundland. The Persian photograph gives **96.71%** Persian cat. These are two inference examples, not dataset accuracy or proof of pretraining holdout.

Attention maps specify block, head, receiver and source grid. CLS self-weight is reported separately. Compare heads using a shared linear colour scale. Values, W_O, residuals and later blocks also affect the final prediction. Occlusion fills predetermined quadrants with model-mean RGB; it probes one intervention, not causal importance. All four occlusions keep Newfoundland as the top label.

## Exercises and answers

1. Scaled scores `[ln 2,0]`, values `[2,0]`, `[0,3]`: weights `[2/3,1/3]`; message `[4/3,1]`.
2. 128×128 RGB, P=16, D=64, four heads: 64 patches; N=65 with CLS; W_patch 768×64; Q per head 65×16; A per head 65×65; W_O 64×64.
3. Permuting complete positioned rows preserves content–position associations and CLS output under standard shared self-attention. Moving contents while retaining location vectors changes those associations.
4. At 224 pixels, P=32/16/8 gives N=50/197/785 and N²=2,500/38,809/616,225 coefficients per head. These are coefficient counts, not complete FLOPs or measured memory allocation.

## Reproduce and rebuild

```sh
# NumPy + PyTorch required; no network needed for the saved experiments.
python notebooks/vision/train_small_vit.py
python src/build_vision1_lesson.py
python src/check_vision1_lesson.py

# Optional: rerun the real checkpoint (timm + Pillow; network if not cached).
uv run --with timm --with pillow python notebooks/vision/run_real_images.py
uv run --with timm --with pillow python notebooks/vision/inspect_real_vit.py
python src/build_vision1_lesson.py
```

The notebook executes all 19 code cells, including independent PyTorch attention parity, the learning step, Conv2d/Linear equivalence, every full-model gradient, an explicit wrong-batch-axis negative control, and re-evaluation of saved checkpoints. The checked numerical values are also used by the slides and browser controls.

## References that shaped the lecture

- [UCSD CSE252D, Manmohan Chandraker (2024)](https://cseweb.ucsd.edu/~mkchandraker/classes/CSE252D/Spring2024/Lectures/lec02_visiontransformers.pdf): a visual question motivates matching, then each image token supplies its own query.
- [MIT VisionBook, Chapter 26](https://visionbook.mit.edu/transformers.html): separate mixing rows from modifying a row; make the output depend on the task.

- [Stanford CS231n, Lecture 8 (2025)](https://cs231n.stanford.edu/slides/2025/lecture_8.pdf): hold an actual image while exposing the patch-to-token path.
- [UvA Vision Transformer tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html): complete implementation, training and image-patch interpretation.
- [D2L Vision Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html): patch projection, shapes and encoder structure.
- [CMU 10-423 Lecture 5 (2025)](https://www.cs.cmu.edu/~mgormley/courses/10423-f25/slides/lecture5-vision.pdf): motivate position through lost layout.
- [ANU COMP8536 notes](https://users.cecs.anu.edu.au/~sgould/papers/comp8536_lecture_notes.pdf): connect visual learning and transformer representations.
- [Arnab’s Transformers for Vision lecture](https://www.robots.ox.ac.uk/~aarnab/talks/transformers_talk.pdf): context and permutation structure.
- [Jay Alammar](https://jalammar.github.io/illustrated-transformer/): persistent objects and a visible calculation path.
- [Original ViT paper](https://arxiv.org/abs/2010.11929): architecture and claims about scale and training.

The [latest reference review](VISION1_REFERENCE_REVIEW.md) records the new sources and the successful review of both Vizuara recordings. The earlier [research plan](VISION1_REDESIGN_PLAN.md) records all the supplied articles and videos, including their review status. Videos were consulted silently through available text/transcripts. This is an original teaching sequence; it does not reproduce those lectures' slides.

Photographs: Oxford-IIIT Pet dataset, Parkhi, Vedaldi, Zisserman and Jawahar, via the timm Hugging Face mirror. Image filenames, revision, checksums and attribution are in `figures/vision1/images.json`. Original image ownership and CC BY-SA 4.0 attribution are retained. `model-input.png` shows the checkpoint's exact evaluation crop.
