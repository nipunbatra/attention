# Vision I — complete teaching guide

**Deck:** [vision1.html](../vision1.html) · **Present:** open the deck and press **P** · **Lab:** [03_vision_transformer_lab.ipynb](../notebooks/vision/03_vision_transformer_lab.ipynb)

146 teaching frames plus cover; 14 sections. Silent, self-contained HTML slides with image assets and math embedded. Reading mode includes the longer explanations, source links, and numerical tables. Arrow keys advance one reveal; **S** opens presenter notes; **O** opens the overview; **C** shows classroom controls. Every frame has a question to ask and a note about what to point at.

## The teaching thread

Start with a photograph and ask students to name the clues. Isolate one genuine crop and restore its context. Recall **aabid** from Part I, **river/bank** from Part II, and **red/wool/coat** from Part III. Then introduce patches, a tiny exact worksheet, the complete block, executable code, learning, and the original photograph again.

The two-crop Q/K/V warm-up and the pooling example use their own clearly labeled, hand-chosen numbers. Then keep the three larger settings explicit:

1. **Hand worksheet:** a 4×4 binary image, four 2×2 patches, D=4, two heads of width 2, five rows including CLS. Chosen weights; no LayerNorm or block MLP. Labels name two specific arrangements. Students can calculate every number.
2. **Trained small ViT:** 8×8 noisy grayscale images, sixteen 2×2 patches, D=16, two complete pre-LayerNorm blocks, two heads per block and MLP width 32. All trainable components learn. Data splits are independent random draws, with opposite-label pairs sharing exactly the same patch multiset.
3. **Pretrained real ViT:** `vit_tiny_patch16_224.augreg_in21k_ft_in1k`; 224×224 RGB, 196 patches plus CLS, D=192, 12 blocks, three heads per block, 1,000 ImageNet outputs. Exact photos, preprocessing, probabilities, attention arrays and interventions are saved.

## Suggested pacing

Use three meetings, or teach sections 1–7 first and assign the implementation as a lab. The 146 frames are short steps; the total is not a target for one class. Pause for predictions and hand calculations.

| Meeting | Sections | Student activity |
|---|---|---|
| A | 1–4 | Compare tasks, turn pixels into rows, reason about position, explain Q/K/V, calculate a first attention message |
| B | 5–7 | Work the second head, combine messages, predict a class, compare CLS with pooling, restore the full block |
| C / lab | 8–14 | Run the code, interpret the training control and real-image measurements, solve transfer exercises |

For a short conceptual introduction, use the task comparison, the two-crop Q/K/V example, CLS and pooling, the whole-block drawing and the real-photo predictions. Keep the full four-patch calculation for a session with time to work alongside the class.

## Introduce terms before using them

The opening task comparison uses “one image summary.” Section 3 introduces the classification token (CLS) with the crop-to-summary diagram, then its shared learned starting vector, before CLS appears in the position-vector table. Section 6 names mean pooling alongside the coordinate-by-coordinate average. The previous text lessons supply the familiar operations; new vision terms are defined where their role becomes visible.

## Places to stop and ask

Use the [complete slide map](VISION1_SLIDE_MAP.md) for current frame numbers and direct presentation links.

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
