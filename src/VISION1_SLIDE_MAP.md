# Vision I — current slide map

144 teaching frames plus the cover. Right/Left advances reveals; S opens notes. Detailed calculations are preserved. Section 6 is an optional implementation lab; historical and dense reference figures remain in reading notes.

| Section | Frame | ID | Title |
|---|---:|---|---|
| s01 | 1 | `prior-encoder-recap` | We already know how an encoder builds context |
| s01 | 2 | `vit-same-encoder` | A ViT is an encoder over image patches |
| s01 | 3 | `vit-token-inputs` | Text looks up a row; an image patch computes one |
| s01 | 4 | `vision-topic-01` | Section 1 · The image classification task |
| s01 | 5 | `dataset-gallery` | What does our animal dataset look like? |
| s01 | 6 | `s01-photo` | What animal do you see? |
| s01 | 7 | `task-image-label` | Our task today: classify the whole image |
| s01 | 8 | `patch-context` | Attention: what can the face tell this patch? |
| s01 | 9 | `patch-context-weights` | Should every source contribute equally? |
| s01 | 10 | `patch-context-update` | What changes when the patch gets context? |
| s01 | 11 | `bridge-image-query` | What could a query be in the image? |
| s01 | 12 | `bridge-image-key` | What could a key be in the image? |
| s01 | 13 | `bridge-image-value` | What information would a value send? |
| s01 | 14 | `image-to-rows` | How can we give this photograph to attention? |
| s01 | 15 | `vit-house-architecture` | The image classifier, drawn as one encoder pipeline |
| s02 | 1 | `vision-topic-02` | Section 2 · From pixels to patch embeddings |
| s02 | 2 | `model-journey-overview` | The whole route: photograph to prediction |
| s02 | 3 | `s01-patches` | Where do the patch boundaries go? |
| s02 | 4 | `rgb-flatten-step-1` | Read the RGB values of each pixel |
| s02 | 5 | `rgb-flatten` | Flatten one channel at a time: R, then G, then B |
| s02 | 6 | `patch-activation-location` | Do we apply an activation after the patch layer? |
| s02 | 7 | `patch-linear-shapes` | What does “projection” mean here? |
| s02 | 8 | `patch-linear-weights` | 12 input numbers, 2 output numbers |
| s02 | 9 | `patch-linear-first` | Follow the connections into output 1 |
| s02 | 10 | `patch-linear-second` | Now follow the connections into output 2 |
| s02 | 11 | `patch-linear-result` | These two numbers are the patch embedding |
| s02 | 12 | `patch-shared-code` | Apply the very same layer to another patch |
| s02 | 13 | `s01-rows` | Start with the same dog photograph |
| s02 | 14 | `projection-size` | Split the image into 16 × 16 patches |
| s02 | 15 | `real-patch-crops` | Number the patches row by row |
| s02 | 16 | `patch-real-dimensions` | Read the RGB values inside patch 63 |
| s02 | 17 | `real-patch-normalize` | Normalize those same RGB values |
| s02 | 18 | `patch-one-row-shape` | Flatten patch 63 in the same order as the code |
| s02 | 19 | `patch-one-row-projection` | Pass that row through the shared linear layer |
| s02 | 20 | `real-patch-projection` | Read the 192 output features for patch 63 |
| s02 | 21 | `real-patch-shared` | Pass patch 64 through the very same layer |
| s02 | 22 | `patch-projection-parameters` | Stack the 196 output rows into C |
| s03 | 1 | `vision-topic-03` | Section 3 · Prepare the rows, then classify the image |
| s03 | 2 | `position-where` | Give each patch its location in the photograph |
| s03 | 3 | `position-table` | Position is a learned lookup table |
| s03 | 4 | `real-patch-position` | Add position to these content rows |
| s03 | 5 | `position-learning` | The image loss teaches the position table |
| s03 | 6 | `cls-detour` | Why add CLS? Give the classifier one image summary |
| s03 | 7 | `real-cls-purpose` | Add a summary row beside the dog’s patch rows |
| s03 | 8 | `cls-summary-refinement` | What makes CLS an image summary? |
| s03 | 9 | `cls-shared-start` | The same CLS start reads two different photographs |
| s03 | 10 | `real-cls-sequence` | Add the summary row: 196 + 1 = 197 |
| s03 | 11 | `model-journey-checkpoint` | Start with one Transformer block |
| s03 | 12 | `vit-self-vs-cross` | Self-attention: Q, K and V share the same input |
| s03 | 13 | `real-patch-qkv` | Make queries, keys and values from these rows |
| s03 | 14 | `real-cls-attention` | The same input matrix feeds three learned projections |
| s03 | 15 | `real-attention-product` | One query–key comparison fills one matrix cell |
| s03 | 16 | `real-attention-cls-zoom` | Follow the CLS row from scores to weights |
| s03 | 17 | `real-attention-weights` | Turn each query’s 197 scores into 197 source weights |
| s03 | 18 | `real-attention-mask` | Every image row can read every image row |
| s03 | 19 | `real-message-text-analogy` | A message for CLS works like a message for bank |
| s03 | 20 | `real-cls-values-origin` | The dog’s feature rows become value rows |
| s03 | 21 | `real-cls-value-scaling` | One weight scales all 64 features in its value row |
| s03 | 22 | `real-cls-value-contributions` | Each source contributes a weighted value row |
| s03 | 23 | `real-cls-value-sum` | Add the contributions to make one CLS message |
| s03 | 24 | `real-cls-message-destination` | Where does the CLS message go? |
| s03 | 25 | `real-attention-values` | Each query gets its own message |
| s03 | 26 | `real-heads-intro` | From one completed head to three parallel heads |
| s03 | 27 | `heads-visual-roles` | What might different heads look for in this photograph? |
| s03 | 28 | `real-heads-qkv` | The same rows feed three sets of Q, K and V |
| s03 | 29 | `real-heads-messages` | Each head repeats the complete attention calculation |
| s03 | 30 | `real-heads-cls` | One CLS input produces three different messages |
| s03 | 31 | `real-heads-concat` | Concatenate the three CLS messages |
| s03 | 32 | `real-cls-message` | Keep the embedding; add the context from attention |
| s03 | 33 | `real-cls-residual` | The dog’s CLS keeps its input and gains context |
| s03 | 34 | `real-cls-mlp` | Open block 1: attention, then the MLP |
| s03 | 35 | `real-mlp-network` | Open the MLP: 192 inputs, 768 hidden units, 192 outputs |
| s03 | 36 | `real-mlp-residual` | Add the MLP update to finish block 1 |
| s03 | 37 | `real-block-handoff` | Pass the complete output of block 1 into block 2 |
| s03 | 38 | `real-block-changes` | What changes as the rows move through the blocks? |
| s03 | 39 | `real-cls-depth` | Continue through the stack, then classify the image |
| s03 | 40 | `real-cls-readout` | Select CLS from the final feature matrix |
| s03 | 41 | `real-classifier-network` | Open the classifier: 192 features become 1,000 scores |
| s03 | 42 | `real-classifier-score` | One class score is a weighted sum plus a bias |
| s03 | 43 | `real-classifier-softmax` | Turn all 1,000 scores into class probabilities |
| s03 | 44 | `photo-two-softmaxes` | Two softmaxes, two different questions |
| s03 | 45 | `real-cls-prediction` | The same dog now has its final prediction |
| s04 | 1 | `vision-topic-04` | Section 4 · Whole model walkthrough |
| s04 | 2 | `vit-shape-trace` | One shape trace from pixels to class scores |
| s04 | 3 | `vit-canonical-block` | Inside each block: mix, transform, keep the residual |
| s04 | 4 | `photo-optimizer-step` | Backward: compute gradients, then update the model |
| s05 | 1 | `vision-topic-05` | Section 5 · CNNs, ViTs and inductive bias |
| s05 | 2 | `cnn-receptive-field` | Two ways to build an image representation |
| s05 | 3 | `cnn-context-readout` | A wider view, then one image label |
| s05 | 4 | `cnn-inductive-bias` | Inductive bias: a useful starting assumption |
| s05 | 5 | `cnn-vit-design` | Which is a sensible starting point? |
| s06 | 1 | `vision-topic-06` | Implementation lab · optional |
| s06 | 2 | `code-photo-input` | Keep the same photograph and add the batch axis |
| s06 | 3 | `code-patch-goal` | Our target: turn every patch into 192 features |
| s06 | 4 | `code-patch-dense` | See the patch projection as a layer of neurons |
| s06 | 5 | `code-patch-linear` | Implementation 1: extract patches, then use Linear |
| s06 | 6 | `code-photo-conv` | Implementation 2: the same projection with Conv2d |
| s06 | 7 | `code-patch-parameters` | Reshape the weights; keep the same parameter count |
| s06 | 8 | `code-patch-same-products` | Same pixel × same weight, in both implementations |
| s06 | 9 | `code-patch-equivalence` | Verify it on both photographs: the features match |
| s06 | 10 | `code-patch-why-conv` | Why package patch projection as Conv2d? |
| s06 | 11 | `code-photo-tokens` | First turn the feature grid into patch rows |
| s06 | 12 | `code-photo-tokens-cls` | Then prepend CLS and add position |
| s06 | 13 | `code-photo-attention` | Make queries, keys and values for three heads |
| s06 | 14 | `code-photo-attention-messages` | Compute one message for every query in every head |
| s06 | 15 | `code-photo-attention-join` | Join the head messages and project back to 192 |
| s06 | 16 | `code-photo-block-layers` | Build the layers inside one Transformer block |
| s06 | 17 | `code-photo-block` | Use the two residual paths in order |
| s06 | 18 | `code-photo-stack` | Create twelve blocks with separate learned parameters |
| s06 | 19 | `code-photo-readout` | Run the stack, then read the final CLS |
| s06 | 20 | `code-photo-training` | Connect the loss diagram to one training step |
| s07 | 1 | `vision-topic-07` | Section 7 · Adapt and evaluate the classifier |
| s07 | 2 | `pets-original-task` | What was this model trained to predict? |
| s07 | 3 | `pets-new-task` | Same photographs, a different label vocabulary |
| s07 | 4 | `pets-new-domain` | What if our users supply sketches? |
| s07 | 5 | `pets-head` | Replace the ImageNet head with our two-class head |
| s07 | 6 | `pets-frozen` | Freeze the encoder; train the new head |
| s07 | 7 | `pets-fine-tune` | Next option: fine-tune the last block as well |
| s07 | 8 | `pets-training-step` | One batch follows the same forward and backward paths |
| s07 | 9 | `pets-learning-stages` | Three stages: learn, adapt, then predict |
| s07 | 10 | `pets-inference` | What happens when we classify a new photograph? |
| s07 | 11 | `pets-evaluation` | How would we check whether the classifier learned? |
| s08 | 1 | `vision-topic-08` | Section 8 · Return to the real photographs |
| s08 | 2 | `real-input` | Which pixels are we giving the real model? |
| s08 | 3 | `s06-answer` | What did the model call our dog? |
| s08 | 4 | `one-photo-limit` | Does one correct photograph tell us the accuracy? |
| s08 | 5 | `real-cat` | What happens when we give it the cat? |
| s09 | 1 | `vision-topic-09` | Section 9 · Look inside the trained model |
| s09 | 2 | `interpret-similarity` | Similar patch features can connect distant image regions |
| s09 | 3 | `interpret-heads` | Keep the query fixed; change only the attention head |
| s09 | 4 | `interpret-cls` | CLS gathers a message for the image summary |
| s09 | 5 | `cover-pixels` | “Cover” means replace these pixels with gray |
| s09 | 6 | `cover-1` | Run the covered image through the same trained model |
| s09 | 7 | `occlusion` | Four covers, four new forward passes |
| s09 | 8 | `occlusion-small-setup` | Use smaller covers to ask a more local question |
| s09 | 9 | `occlusion-small-result` | Smaller covers reveal local sensitivity |
| s10 | 1 | `vision-topic-10` | Section 10 · The cost of smaller patches |
| s10 | 2 | `patch-cost` | What changes when the patch size is halved? |
| s10 | 3 | `real-work-count` | How much matching happens inside the tiny real model? |
| s10 | 4 | `cost-control` | What happens if we use a larger image? |
| s10 | 5 | `vision-summary-architecture` | The whole ViT: pixels → context → one label |
| s10 | 6 | `vision-summary-takeaways` | Four ideas to carry forward |
| s10 | 7 | `vision-fixed-class-vectors` | Our classifier stores a vector for each known label |
| s10 | 8 | `vision-language-handoff` | What if a class vector could come from language? |
