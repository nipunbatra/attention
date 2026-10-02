# Vision I — main slide map

54 content frames plus the cover: 55 main slides. Q/K/V intuition, full attention, CNN comparison and measured occlusion are part of the main route.

Right/Left advances reveals; S opens notes. PDF captures the final reveal of each frame.

| PDF page | Route | Stable ID | Title |
|---:|---|---|---|
| 2 | [s01/1/0](../vision1.html?present#s01/1/0) | `story-recall` | We already know how an encoder builds context |
| 3 | [s01/2/0](../vision1.html?present#s01/2/0) | `story-task` | What should this photograph produce? |
| 4 | [s01/3/0](../vision1.html?present#s01/3/0) | `story-contract` | An encoder expects vectors. How can an image supply them? |
| 5 | [s01/4/0](../vision1.html?present#s01/4/0) | `story-questions` | Five questions build the architecture |
| 6 | [s02/1/0](../vision1.html?present#s02/1/0) | `story-pixel-budget` | Why not make every pixel a token? |
| 7 | [s02/2/0](../vision1.html?present#s02/2/0) | `story-patch-grid` | Split the same photograph into 196 patches |
| 8 | [s02/3/0](../vision1.html?present#s02/3/0) | `story-patch-768` | What is inside one patch? |
| 9 | [s02/4/0](../vision1.html?present#s02/4/0) | `story-flatten` | Flatten one channel at a time: R, then G, then B |
| 10 | [s02/5/0](../vision1.html?present#s02/5/0) | `story-projection` | How do 768 pixel values become 192 features? |
| 11 | [s02/6/0](../vision1.html?present#s02/6/0) | `story-shared-projection` | Should every patch get a different projection? |
| 12 | [s02/7/0](../vision1.html?present#s02/7/0) | `story-stack` | Stack the patches into one feature matrix |
| 13 | [s02/8/0](../vision1.html?present#s02/8/0) | `story-measured-patch` | What does the trained projection actually return? |
| 14 | [s02/9/0](../vision1.html?present#s02/9/0) | `story-pipeline-patches` | We have solved the image-to-token problem |
| 15 | [s03/1/0](../vision1.html?present#s03/1/0) | `story-where` | The projection is shared. Where does location enter? |
| 16 | [s03/2/0](../vision1.html?present#s03/2/0) | `story-content-position` | Add WHAT and WHERE |
| 17 | [s03/3/0](../vision1.html?present#s03/3/0) | `story-position-table` | Which positional vector goes with each row? |
| 18 | [s04/1/0](../vision1.html?present#s04/1/0) | `story-readout-question` | Many patch representations, one image label |
| 19 | [s04/2/0](../vision1.html?present#s04/2/0) | `story-cls-analogy` | Same CLS idea, different input tokens |
| 20 | [s04/3/0](../vision1.html?present#s04/3/0) | `story-cls-start` | Every image starts with the same learned CLS vector |
| 21 | [s04/4/0](../vision1.html?present#s04/4/0) | `story-cls-dependent` | After reading the image, CLS becomes image-dependent |
| 22 | [s04/5/0](../vision1.html?present#s04/5/0) | `story-cls-reads` | CLS reads the current patch states at every block |
| 23 | [s04/6/0](../vision1.html?present#s04/6/0) | `story-prepared` | We now have the token sequence the encoder needs |
| 24 | [s05/1/0](../vision1.html?present#s05/1/0) | `story-reuse-encoder` | From here, reuse the encoder we already know |
| 25 | [s05/2/0](../vision1.html?present#s05/2/0) | `story-self-cross` | Where do Q, K and V come from? |
| 26 | [s05/3/0](../vision1.html?present#s05/3/0) | `story-qkv-roles` | One patch representation has three jobs |
| 27 | [s05/4/0](../vision1.html?present#s05/4/0) | `story-query-examples` | What might different queries try to gather? |
| 28 | [s05/5/0](../vision1.html?present#s05/5/0) | `story-key-value-pair` | A source supplies both a key and a value |
| 29 | [s05/6/0](../vision1.html?present#s05/6/0) | `story-value-mixture` | Several source values form one receiver’s message |
| 30 | [s05/7/0](../vision1.html?present#s05/7/0) | `story-full-attention` | ViT uses full attention, not causal attention |
| 31 | [s05/8/0](../vision1.html?present#s05/8/0) | `story-cls-message` | How does one CLS query collect one message? |
| 32 | [s05/9/0](../vision1.html?present#s05/9/0) | `story-multihead` | Three heads form three views of the same sequence |
| 33 | [s05/10/0](../vision1.html?present#s05/10/0) | `story-measured-attention` | Where does the trained query look? |
| 34 | [s05/11/0](../vision1.html?present#s05/11/0) | `story-block` | What exactly is inside one pre-LN encoder block? |
| 35 | [s05/12/0](../vision1.html?present#s05/12/0) | `story-mlp` | What job remains for the MLP? |
| 36 | [s05/13/0](../vision1.html?present#s05/13/0) | `story-depth` | Repeat the block twelve times |
| 37 | [s05/14/0](../vision1.html?present#s05/14/0) | `story-stored-computed` | What is stored, and what changes with the image? |
| 38 | [s06/1/0](../vision1.html?present#s06/1/0) | `story-final-cls` | Which representation enters the classifier? |
| 39 | [s06/2/0](../vision1.html?present#s06/2/0) | `story-head` | How do 192 features score 1,000 classes? |
| 40 | [s06/3/0](../vision1.html?present#s06/3/0) | `story-prediction` | What does this checkpoint predict for our photograph? |
| 41 | [s06/4/0](../vision1.html?present#s06/4/0) | `story-cover-question` | What happens if we hide one quarter of the image? |
| 42 | [s06/5/0](../vision1.html?present#s06/5/0) | `story-cover-code` | Replace pixels, then recompute the whole forward pass |
| 43 | [s06/6/0](../vision1.html?present#s06/6/0) | `story-cover-results` | Four independent covers, four measured predictions |
| 44 | [s06/7/0](../vision1.html?present#s06/7/0) | `story-cover-small` | Would smaller covers tell us more? |
| 45 | [s07/1/0](../vision1.html?present#s07/1/0) | `story-cnn-context` | Two ways to gather image context |
| 46 | [s07/2/0](../vision1.html?present#s07/2/0) | `story-cnn-bias` | Which assumptions are built into the architecture? |
| 47 | [s07/3/0](../vision1.html?present#s07/3/0) | `story-cnn-example` | What does an inductive bias buy us? |
| 48 | [s07/4/0](../vision1.html?present#s07/4/0) | `story-patch-sizes` | How much detail should one token cover? |
| 49 | [s07/5/0](../vision1.html?present#s07/5/0) | `story-patch-cost` | What does a finer grid cost? |
| 50 | [s07/6/0](../vision1.html?present#s07/6/0) | `story-summary-pipeline` | Once an image becomes tokens, the encoder is familiar |
| 51 | [s07/7/0](../vision1.html?present#s07/7/0) | `story-summary-shapes` | Follow the whole model through its shapes |
| 52 | [s07/8/0](../vision1.html?present#s07/8/0) | `story-six-lines` | The whole ViT in six lines |
| 53 | [s07/9/0](../vision1.html?present#s07/9/0) | `story-optional-routes` | Choose a deeper dive when you need it |
| 54 | [s07/10/0](../vision1.html?present#s07/10/0) | `story-fixed-vocabulary` | The classifier stores a learned vector per known class |
| 55 | [s07/11/0](../vision1.html?present#s07/11/0) | `story-clip-question` | What if our class vocabulary could come from words? |
