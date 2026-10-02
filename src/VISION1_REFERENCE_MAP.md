# Vision I — optional reference map

150 content frames plus the cover: 151 reference pages. All 144 frames from the previous detailed teaching sequence are preserved.

Right/Left advances reveals; S opens notes. PDF captures the final reveal of each frame.

| PDF page | Route | Stable ID | Title |
|---:|---|---|---|
| 2 | [s01/1/0](../vision1-reference.html?present#s01/1/0) | `optional-library-start` | Optional labs, worked examples and reference diagrams |
| 3 | [s01/2/0](../vision1-reference.html?present#s01/2/0) | `prior-encoder-recap` | We already know how an encoder builds context |
| 4 | [s01/3/0](../vision1-reference.html?present#s01/3/0) | `vit-same-encoder` | A ViT is an encoder over image patches |
| 5 | [s01/4/0](../vision1-reference.html?present#s01/4/0) | `vit-token-inputs` | Text looks up a row; an image patch computes one |
| 6 | [s01/5/0](../vision1-reference.html?present#s01/5/0) | `vision-topic-01` | Section 1 · The image classification task |
| 7 | [s01/6/0](../vision1-reference.html?present#s01/6/0) | `dataset-gallery` | What does our animal dataset look like? |
| 8 | [s01/7/0](../vision1-reference.html?present#s01/7/0) | `s01-photo` | What animal do you see? |
| 9 | [s01/8/0](../vision1-reference.html?present#s01/8/0) | `task-image-label` | Our task today: classify the whole image |
| 10 | [s01/9/0](../vision1-reference.html?present#s01/9/0) | `patch-context` | Attention: what can the face tell this patch? |
| 11 | [s01/10/0](../vision1-reference.html?present#s01/10/0) | `patch-context-weights` | Should every source contribute equally? |
| 12 | [s01/11/0](../vision1-reference.html?present#s01/11/0) | `patch-context-update` | What changes when the patch gets context? |
| 13 | [s01/12/0](../vision1-reference.html?present#s01/12/0) | `bridge-image-query` | What could a query be in the image? |
| 14 | [s01/13/0](../vision1-reference.html?present#s01/13/0) | `bridge-image-key` | What could a key be in the image? |
| 15 | [s01/14/0](../vision1-reference.html?present#s01/14/0) | `bridge-image-value` | What information would a value send? |
| 16 | [s01/15/0](../vision1-reference.html?present#s01/15/0) | `image-to-rows` | How can we give this photograph to attention? |
| 17 | [s01/16/0](../vision1-reference.html?present#s01/16/0) | `vit-house-architecture` | The image classifier, drawn as one encoder pipeline |
| 18 | [s02/1/0](../vision1-reference.html?present#s02/1/0) | `vision-topic-02` | Section 2 · From pixels to patch embeddings |
| 19 | [s02/2/0](../vision1-reference.html?present#s02/2/0) | `model-journey-overview` | The whole route: photograph to prediction |
| 20 | [s02/3/0](../vision1-reference.html?present#s02/3/0) | `s01-patches` | Where do the patch boundaries go? |
| 21 | [s02/4/0](../vision1-reference.html?present#s02/4/0) | `rgb-flatten-step-1` | Read the RGB values of each pixel |
| 22 | [s02/5/0](../vision1-reference.html?present#s02/5/0) | `rgb-flatten` | Flatten one channel at a time: R, then G, then B |
| 23 | [s02/6/0](../vision1-reference.html?present#s02/6/0) | `patch-activation-location` | Do we apply an activation after the patch layer? |
| 24 | [s02/7/0](../vision1-reference.html?present#s02/7/0) | `patch-linear-shapes` | What does “projection” mean here? |
| 25 | [s02/8/0](../vision1-reference.html?present#s02/8/0) | `patch-linear-weights` | 12 input numbers, 2 output numbers |
| 26 | [s02/9/0](../vision1-reference.html?present#s02/9/0) | `patch-linear-first` | Follow the connections into output 1 |
| 27 | [s02/10/0](../vision1-reference.html?present#s02/10/0) | `patch-linear-second` | Now follow the connections into output 2 |
| 28 | [s02/11/0](../vision1-reference.html?present#s02/11/0) | `patch-linear-result` | These two numbers are the patch embedding |
| 29 | [s02/12/0](../vision1-reference.html?present#s02/12/0) | `patch-shared-code` | Apply the very same layer to another patch |
| 30 | [s02/13/0](../vision1-reference.html?present#s02/13/0) | `s01-rows` | Start with the same dog photograph |
| 31 | [s02/14/0](../vision1-reference.html?present#s02/14/0) | `projection-size` | Split the image into 16 × 16 patches |
| 32 | [s02/15/0](../vision1-reference.html?present#s02/15/0) | `real-patch-crops` | Number the patches row by row |
| 33 | [s02/16/0](../vision1-reference.html?present#s02/16/0) | `patch-real-dimensions` | Read the RGB values inside patch 63 |
| 34 | [s02/17/0](../vision1-reference.html?present#s02/17/0) | `real-patch-normalize` | Normalize those same RGB values |
| 35 | [s02/18/0](../vision1-reference.html?present#s02/18/0) | `patch-one-row-shape` | Flatten patch 63 in the same order as the code |
| 36 | [s02/19/0](../vision1-reference.html?present#s02/19/0) | `patch-one-row-projection` | Pass that row through the shared linear layer |
| 37 | [s02/20/0](../vision1-reference.html?present#s02/20/0) | `real-patch-projection` | Read the 192 output features for patch 63 |
| 38 | [s02/21/0](../vision1-reference.html?present#s02/21/0) | `real-patch-shared` | Pass patch 64 through the very same layer |
| 39 | [s02/22/0](../vision1-reference.html?present#s02/22/0) | `patch-projection-parameters` | Stack the 196 output rows into C |
| 40 | [s03/1/0](../vision1-reference.html?present#s03/1/0) | `vision-topic-03` | Section 3 · Prepare the rows, then classify the image |
| 41 | [s03/2/0](../vision1-reference.html?present#s03/2/0) | `position-where` | Give each patch its location in the photograph |
| 42 | [s03/3/0](../vision1-reference.html?present#s03/3/0) | `position-table` | Position is a learned lookup table |
| 43 | [s03/4/0](../vision1-reference.html?present#s03/4/0) | `real-patch-position` | Add position to these content rows |
| 44 | [s03/5/0](../vision1-reference.html?present#s03/5/0) | `position-learning` | The image loss teaches the position table |
| 45 | [s03/6/0](../vision1-reference.html?present#s03/6/0) | `cls-detour` | Why add CLS? Give the classifier one image summary |
| 46 | [s03/7/0](../vision1-reference.html?present#s03/7/0) | `real-cls-purpose` | Add a summary row beside the dog’s patch rows |
| 47 | [s03/8/0](../vision1-reference.html?present#s03/8/0) | `cls-summary-refinement` | What makes CLS an image summary? |
| 48 | [s03/9/0](../vision1-reference.html?present#s03/9/0) | `cls-shared-start` | The same CLS start reads two different photographs |
| 49 | [s03/10/0](../vision1-reference.html?present#s03/10/0) | `real-cls-sequence` | Add the summary row: 196 + 1 = 197 |
| 50 | [s03/11/0](../vision1-reference.html?present#s03/11/0) | `model-journey-checkpoint` | Start with one Transformer block |
| 51 | [s03/12/0](../vision1-reference.html?present#s03/12/0) | `vit-self-vs-cross` | Self-attention: Q, K and V share the same input |
| 52 | [s03/13/0](../vision1-reference.html?present#s03/13/0) | `real-patch-qkv` | Make queries, keys and values from these rows |
| 53 | [s03/14/0](../vision1-reference.html?present#s03/14/0) | `real-cls-attention` | The same input matrix feeds three learned projections |
| 54 | [s03/15/0](../vision1-reference.html?present#s03/15/0) | `real-attention-product` | One query–key comparison fills one matrix cell |
| 55 | [s03/16/0](../vision1-reference.html?present#s03/16/0) | `real-attention-cls-zoom` | Follow the CLS row from scores to weights |
| 56 | [s03/17/0](../vision1-reference.html?present#s03/17/0) | `real-attention-weights` | Turn each query’s 197 scores into 197 source weights |
| 57 | [s03/18/0](../vision1-reference.html?present#s03/18/0) | `real-attention-mask` | Every image row can read every image row |
| 58 | [s03/19/0](../vision1-reference.html?present#s03/19/0) | `real-message-text-analogy` | A message for CLS works like a message for bank |
| 59 | [s03/20/0](../vision1-reference.html?present#s03/20/0) | `real-cls-values-origin` | The dog’s feature rows become value rows |
| 60 | [s03/21/0](../vision1-reference.html?present#s03/21/0) | `real-cls-value-scaling` | One weight scales all 64 features in its value row |
| 61 | [s03/22/0](../vision1-reference.html?present#s03/22/0) | `real-cls-value-contributions` | Each source contributes a weighted value row |
| 62 | [s03/23/0](../vision1-reference.html?present#s03/23/0) | `real-cls-value-sum` | Add the contributions to make one CLS message |
| 63 | [s03/24/0](../vision1-reference.html?present#s03/24/0) | `real-cls-message-destination` | Where does the CLS message go? |
| 64 | [s03/25/0](../vision1-reference.html?present#s03/25/0) | `real-attention-values` | Each query gets its own message |
| 65 | [s03/26/0](../vision1-reference.html?present#s03/26/0) | `real-heads-intro` | From one completed head to three parallel heads |
| 66 | [s03/27/0](../vision1-reference.html?present#s03/27/0) | `heads-visual-roles` | What might different heads look for in this photograph? |
| 67 | [s03/28/0](../vision1-reference.html?present#s03/28/0) | `real-heads-qkv` | The same rows feed three sets of Q, K and V |
| 68 | [s03/29/0](../vision1-reference.html?present#s03/29/0) | `real-heads-messages` | Each head repeats the complete attention calculation |
| 69 | [s03/30/0](../vision1-reference.html?present#s03/30/0) | `real-heads-cls` | One CLS input produces three different messages |
| 70 | [s03/31/0](../vision1-reference.html?present#s03/31/0) | `real-heads-concat` | Concatenate the three CLS messages |
| 71 | [s03/32/0](../vision1-reference.html?present#s03/32/0) | `real-cls-message` | Keep the embedding; add the context from attention |
| 72 | [s03/33/0](../vision1-reference.html?present#s03/33/0) | `real-cls-residual` | The dog’s CLS keeps its input and gains context |
| 73 | [s03/34/0](../vision1-reference.html?present#s03/34/0) | `real-cls-mlp` | Open block 1: attention, then the MLP |
| 74 | [s03/35/0](../vision1-reference.html?present#s03/35/0) | `real-mlp-network` | Open the MLP: 192 inputs, 768 hidden units, 192 outputs |
| 75 | [s03/36/0](../vision1-reference.html?present#s03/36/0) | `real-mlp-residual` | Add the MLP update to finish block 1 |
| 76 | [s03/37/0](../vision1-reference.html?present#s03/37/0) | `real-block-handoff` | Pass the complete output of block 1 into block 2 |
| 77 | [s03/38/0](../vision1-reference.html?present#s03/38/0) | `real-block-changes` | What changes as the rows move through the blocks? |
| 78 | [s03/39/0](../vision1-reference.html?present#s03/39/0) | `real-cls-depth` | Continue through the stack, then classify the image |
| 79 | [s03/40/0](../vision1-reference.html?present#s03/40/0) | `real-cls-readout` | Select CLS from the final feature matrix |
| 80 | [s03/41/0](../vision1-reference.html?present#s03/41/0) | `real-classifier-network` | Open the classifier: 192 features become 1,000 scores |
| 81 | [s03/42/0](../vision1-reference.html?present#s03/42/0) | `real-classifier-score` | One class score is a weighted sum plus a bias |
| 82 | [s03/43/0](../vision1-reference.html?present#s03/43/0) | `real-classifier-softmax` | Turn all 1,000 scores into class probabilities |
| 83 | [s03/44/0](../vision1-reference.html?present#s03/44/0) | `photo-two-softmaxes` | Two softmaxes, two different questions |
| 84 | [s03/45/0](../vision1-reference.html?present#s03/45/0) | `real-cls-prediction` | The same dog now has its final prediction |
| 85 | [s04/1/0](../vision1-reference.html?present#s04/1/0) | `vision-topic-04` | Section 4 · Whole model walkthrough |
| 86 | [s04/2/0](../vision1-reference.html?present#s04/2/0) | `vit-shape-trace` | One shape trace from pixels to class scores |
| 87 | [s04/3/0](../vision1-reference.html?present#s04/3/0) | `vit-canonical-block` | Inside each block: mix, transform, keep the residual |
| 88 | [s04/4/0](../vision1-reference.html?present#s04/4/0) | `photo-optimizer-step` | Backward: compute gradients, then update the model |
| 89 | [s05/1/0](../vision1-reference.html?present#s05/1/0) | `vision-topic-05` | Section 5 · CNNs, ViTs and inductive bias |
| 90 | [s05/2/0](../vision1-reference.html?present#s05/2/0) | `cnn-receptive-field` | Two ways to build an image representation |
| 91 | [s05/3/0](../vision1-reference.html?present#s05/3/0) | `cnn-context-readout` | A wider view, then one image label |
| 92 | [s05/4/0](../vision1-reference.html?present#s05/4/0) | `cnn-inductive-bias` | Inductive bias: a useful starting assumption |
| 93 | [s05/5/0](../vision1-reference.html?present#s05/5/0) | `cnn-vit-design` | Which is a sensible starting point? |
| 94 | [s06/1/0](../vision1-reference.html?present#s06/1/0) | `vision-topic-06` | Implementation lab · optional |
| 95 | [s06/2/0](../vision1-reference.html?present#s06/2/0) | `code-photo-input` | Keep the same photograph and add the batch axis |
| 96 | [s06/3/0](../vision1-reference.html?present#s06/3/0) | `code-patch-goal` | Our target: turn every patch into 192 features |
| 97 | [s06/4/0](../vision1-reference.html?present#s06/4/0) | `code-patch-dense` | See the patch projection as a layer of neurons |
| 98 | [s06/5/0](../vision1-reference.html?present#s06/5/0) | `code-patch-linear` | Implementation 1: extract patches, then use Linear |
| 99 | [s06/6/0](../vision1-reference.html?present#s06/6/0) | `code-photo-conv` | Implementation 2: the same projection with Conv2d |
| 100 | [s06/7/0](../vision1-reference.html?present#s06/7/0) | `code-patch-parameters` | Reshape the weights; keep the same parameter count |
| 101 | [s06/8/0](../vision1-reference.html?present#s06/8/0) | `code-patch-same-products` | Same pixel × same weight, in both implementations |
| 102 | [s06/9/0](../vision1-reference.html?present#s06/9/0) | `code-patch-equivalence` | Verify it on both photographs: the features match |
| 103 | [s06/10/0](../vision1-reference.html?present#s06/10/0) | `code-patch-why-conv` | Why package patch projection as Conv2d? |
| 104 | [s06/11/0](../vision1-reference.html?present#s06/11/0) | `code-photo-tokens` | First turn the feature grid into patch rows |
| 105 | [s06/12/0](../vision1-reference.html?present#s06/12/0) | `code-photo-tokens-cls` | Then prepend CLS and add position |
| 106 | [s06/13/0](../vision1-reference.html?present#s06/13/0) | `code-photo-attention` | Make queries, keys and values for three heads |
| 107 | [s06/14/0](../vision1-reference.html?present#s06/14/0) | `code-photo-attention-messages` | Compute one message for every query in every head |
| 108 | [s06/15/0](../vision1-reference.html?present#s06/15/0) | `code-photo-attention-join` | Join the head messages and project back to 192 |
| 109 | [s06/16/0](../vision1-reference.html?present#s06/16/0) | `code-photo-block-layers` | Build the layers inside one Transformer block |
| 110 | [s06/17/0](../vision1-reference.html?present#s06/17/0) | `code-photo-block` | Use the two residual paths in order |
| 111 | [s06/18/0](../vision1-reference.html?present#s06/18/0) | `code-photo-stack` | Create twelve blocks with separate learned parameters |
| 112 | [s06/19/0](../vision1-reference.html?present#s06/19/0) | `code-photo-readout` | Run the stack, then read the final CLS |
| 113 | [s06/20/0](../vision1-reference.html?present#s06/20/0) | `code-photo-training` | Connect the loss diagram to one training step |
| 114 | [s07/1/0](../vision1-reference.html?present#s07/1/0) | `optional-extensions-start` | Optional extensions |
| 115 | [s07/2/0](../vision1-reference.html?present#s07/2/0) | `vision-topic-07` | Section 7 · Adapt and evaluate the classifier |
| 116 | [s07/3/0](../vision1-reference.html?present#s07/3/0) | `pets-original-task` | What was this model trained to predict? |
| 117 | [s07/4/0](../vision1-reference.html?present#s07/4/0) | `pets-new-task` | Same photographs, a different label vocabulary |
| 118 | [s07/5/0](../vision1-reference.html?present#s07/5/0) | `pets-new-domain` | What if our users supply sketches? |
| 119 | [s07/6/0](../vision1-reference.html?present#s07/6/0) | `pets-head` | Replace the ImageNet head with our two-class head |
| 120 | [s07/7/0](../vision1-reference.html?present#s07/7/0) | `pets-frozen` | Freeze the encoder; train the new head |
| 121 | [s07/8/0](../vision1-reference.html?present#s07/8/0) | `pets-fine-tune` | Next option: fine-tune the last block as well |
| 122 | [s07/9/0](../vision1-reference.html?present#s07/9/0) | `pets-training-step` | One batch follows the same forward and backward paths |
| 123 | [s07/10/0](../vision1-reference.html?present#s07/10/0) | `pets-learning-stages` | Three stages: learn, adapt, then predict |
| 124 | [s07/11/0](../vision1-reference.html?present#s07/11/0) | `pets-inference` | What happens when we classify a new photograph? |
| 125 | [s07/12/0](../vision1-reference.html?present#s07/12/0) | `pets-evaluation` | How would we check whether the classifier learned? |
| 126 | [s08/1/0](../vision1-reference.html?present#s08/1/0) | `vision-topic-08` | Section 8 · Return to the real photographs |
| 127 | [s08/2/0](../vision1-reference.html?present#s08/2/0) | `real-input` | Which pixels are we giving the real model? |
| 128 | [s08/3/0](../vision1-reference.html?present#s08/3/0) | `s06-answer` | What did the model call our dog? |
| 129 | [s08/4/0](../vision1-reference.html?present#s08/4/0) | `one-photo-limit` | Does one correct photograph tell us the accuracy? |
| 130 | [s08/5/0](../vision1-reference.html?present#s08/5/0) | `real-cat` | What happens when we give it the cat? |
| 131 | [s09/1/0](../vision1-reference.html?present#s09/1/0) | `vision-topic-09` | Section 9 · Look inside the trained model |
| 132 | [s09/2/0](../vision1-reference.html?present#s09/2/0) | `interpret-similarity` | Similar patch features can connect distant image regions |
| 133 | [s09/3/0](../vision1-reference.html?present#s09/3/0) | `interpret-heads` | Keep the query fixed; change only the attention head |
| 134 | [s09/4/0](../vision1-reference.html?present#s09/4/0) | `interpret-cls` | CLS gathers a message for the image summary |
| 135 | [s09/5/0](../vision1-reference.html?present#s09/5/0) | `cover-pixels` | “Cover” means replace these pixels with gray |
| 136 | [s09/6/0](../vision1-reference.html?present#s09/6/0) | `cover-1` | Run the covered image through the same trained model |
| 137 | [s09/7/0](../vision1-reference.html?present#s09/7/0) | `occlusion` | Four covers, four new forward passes |
| 138 | [s09/8/0](../vision1-reference.html?present#s09/8/0) | `occlusion-small-setup` | Use smaller covers to ask a more local question |
| 139 | [s09/9/0](../vision1-reference.html?present#s09/9/0) | `occlusion-small-result` | Smaller covers reveal local sensitivity |
| 140 | [s10/1/0](../vision1-reference.html?present#s10/1/0) | `vision-topic-10` | Section 10 · The cost of smaller patches |
| 141 | [s10/2/0](../vision1-reference.html?present#s10/2/0) | `patch-cost` | What changes when the patch size is halved? |
| 142 | [s10/3/0](../vision1-reference.html?present#s10/3/0) | `real-work-count` | How much matching happens inside the tiny real model? |
| 143 | [s10/4/0](../vision1-reference.html?present#s10/4/0) | `cost-control` | What happens if we use a larger image? |
| 144 | [s10/5/0](../vision1-reference.html?present#s10/5/0) | `vision-summary-architecture` | The whole ViT: pixels → context → one label |
| 145 | [s10/6/0](../vision1-reference.html?present#s10/6/0) | `vision-summary-takeaways` | Four ideas to carry forward |
| 146 | [s10/7/0](../vision1-reference.html?present#s10/7/0) | `vision-fixed-class-vectors` | Our classifier stores a vector for each known label |
| 147 | [s10/8/0](../vision1-reference.html?present#s10/8/0) | `vision-language-handoff` | What if a class vector could come from language? |
| 148 | [s11/1/0](../vision1-reference.html?present#s11/1/0) | `reference-photo-label-loss` | The whole Vision Transformer in one figure |
| 149 | [s11/2/0](../vision1-reference.html?present#s11/2/0) | `reference-paper-vision-transformer` | To images: An Image is Worth 16 × 16 Words |
| 150 | [s11/3/0](../vision1-reference.html?present#s11/3/0) | `reference-cls-parameter-origin` | Create 192 trainable numbers for CLS |
| 151 | [s11/4/0](../vision1-reference.html?present#s11/4/0) | `reference-cls-parameter-learning` | The image label teaches the starting CLS numbers |
