# Vision I — complete teaching guide

**Deck:** [vision1.html](../vision1.html) · **Present:** open the deck and press **P** · **Lab:** [03_vision_transformer_lab.ipynb](../notebooks/vision/03_vision_transformer_lab.ipynb)

58 teaching frames plus cover; 14 sections. Silent, self-contained HTML slides with image assets and math embedded. Reading mode includes the longer explanations, source links, and numerical tables. Arrow keys advance one reveal; **S** opens presenter notes; **O** opens the overview; **C** shows classroom controls. Every frame has a question to ask and a note about what to point at.

## The teaching thread

Start with a photograph and ask students to name the clues. Isolate one genuine crop and restore its context. Recall **aabid** from Part I, **river/bank** from Part II, and **red/wool/coat** from Part III. Then introduce patches, a tiny exact worksheet, the complete block, executable code, learning, and the original photograph again.

Keep three settings explicit:

1. **Hand worksheet:** a 4×4 binary image, four 2×2 patches, D=4, two heads of width 2, five rows including CLS. Chosen weights; no LayerNorm or block MLP. Labels name two specific arrangements. Students can calculate every number.
2. **Trained small ViT:** 8×8 noisy grayscale images, sixteen 2×2 patches, D=16, two complete pre-LayerNorm blocks, two heads per block and MLP width 32. All trainable components learn. Data splits are independent random draws, with opposite-label pairs sharing exactly the same patch multiset.
3. **Pretrained real ViT:** `vit_tiny_patch16_224.augreg_in21k_ft_in1k`; 224×224 RGB, 196 patches plus CLS, D=192, 12 blocks, three heads per block, 1,000 ImageNet outputs. Exact photos, preprocessing, probabilities, attention arrays and interventions are saved.

## Suggested pacing

Use two sessions so the calculations have room. The estimates include short student responses; adjust to the class.

| Session | Sections | What students should do | Suggested time |
|---|---|---|---:|
| A | 1–2 | Identify useful context; count RGB values and projection parameters | 15 min |
| A | 3 | Reconstruct layouts from the same patch bag; compute the projection and positions | 12 min |
| A | 4–5 | Calculate both source mixtures, then concatenate and apply W_O | 25 min |
| A | 6 | Compute logits, probabilities, loss and one gradient step; run the position control | 15 min |
| B | 7–8 | Explain both residual paths; trace the complete executable model | 20 min |
| B | 9 | Predict the training control; interpret the validation and test evidence | 12 min |
| B | 10–11 | Return to real photos; interpret attention maps and occlusion carefully | 18 min |
| B | 12–14 | Count cost, solve the three exercises, connect to later vision topics | 15 min |

For an 80-minute introduction, retain sections 1–6, the full-block and real-scale frames in section 7, the trained position control in section 9, the opening-photo prediction in section 10, and the final shape exercise. Assign code, interpretation and cost as the follow-up lab. The complete deck remains available; no arithmetic is hidden from the student notes.

## Places to stop and ask

- **s01 / frame 2:** Could the isolated dark crop be fur, shadow, or another object? Restore the same photograph.
- **s03 / frame 2:** How can two filled and two empty patches reconstruct more than one image?
- **s04 / frame 2:** What information would leak if a patch reads a later raster position? All input patches are already observed for classification.
- **s04 / frame 6:** All five weights must sum to one, including CLS. Scores `[0,1,1,1,1]` give normalizer `1+4e`.
- **s04 / frame 7:** A scalar weight multiplies every coordinate of its own source value. Sum the products, not the keys.
- **s05 / frame 1:** Why is P2 now the highest-weight source? Head 2 compares ink and column.
- **s05 / frame 4:** Which two message coordinates are subtracted by W_O? Concatenation itself does not subtract them.
- **s06 / frame 2:** Which alternatives does this softmax normalize: source rows or class labels?
- **s06 / frame 3:** For the correct class, why is `p-y` negative? Subtracting the gradient increases its logit.
- **s06 / frame 4:** Move contents first, then disable positions. Expect 85.7/14.3 to reverse with the arrangement, and 50/50 for both without positions.
- **s09 / frame 3:** Why can training never separate the paired layouts without position information in this architecture?
- **s11 / frame 4:** Predict the effect of each quadrant removal before revealing the measured target-class probabilities.
- **s12 / frame 2:** Change 224→384 with P=16: 576 patches, 577 tokens, 332,929 scores per head.

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

The notebook executes all 14 code cells, including independent PyTorch attention parity, the learning step, Conv2d/Linear equivalence, every full-model gradient, and re-evaluation of saved checkpoints. The checked numerical values are also used by the slides and browser controls.

## References that shaped the lecture

- [Stanford CS231n, Lecture 8 (2025)](https://cs231n.stanford.edu/slides/2025/lecture_8.pdf): hold an actual image while exposing the patch-to-token path.
- [UvA Vision Transformer tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html): complete implementation, training and image-patch interpretation.
- [D2L Vision Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html): patch projection, shapes and encoder structure.
- [CMU 10-423 Lecture 5 (2025)](https://www.cs.cmu.edu/~mgormley/courses/10423-f25/slides/lecture5-vision.pdf): motivate position through lost layout.
- [ANU COMP8536 notes](https://users.cecs.anu.edu.au/~sgould/papers/comp8536_lecture_notes.pdf): connect visual learning and transformer representations.
- [Arnab’s Transformers for Vision lecture](https://www.robots.ox.ac.uk/~aarnab/talks/transformers_talk.pdf): context and permutation structure.
- [Jay Alammar](https://jalammar.github.io/illustrated-transformer/): persistent objects and a visible calculation path.
- [Original ViT paper](https://arxiv.org/abs/2010.11929): architecture and claims about scale and training.

The earlier [research plan](VISION1_REDESIGN_PLAN.md) records all the supplied articles and videos, including their review status. Videos were consulted silently through available text/transcripts. This is an original teaching sequence; it does not reproduce those lectures' slides.

Photographs: Oxford-IIIT Pet dataset, Parkhi, Vedaldi, Zisserman and Jawahar, via the timm Hugging Face mirror. Image filenames, revision, checksums and attribution are in `figures/vision1/images.json`. Original image ownership and CC BY-SA 4.0 attribution are retained. `model-input.png` shows the checkpoint's exact evaluation crop.
