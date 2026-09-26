# Vision I: slide map

146 teaching frames plus the cover. Each link opens the first reveal of that slide. Use Right/Left for reveals and S for presenter notes.

## Review these additions first

- [What were we asking the text model to predict?](https://nipunbatra.github.io/attention/vision1.html?present#s01/3/0)
- [What are we asking the image model to predict?](https://nipunbatra.github.io/attention/vision1.html?present#s01/4/0)
- [What changed, and what stayed the same?](https://nipunbatra.github.io/attention/vision1.html?present#s01/5/0)
- [Where does CLS come from?](https://nipunbatra.github.io/attention/vision1.html?present#s04/2/0)
- [How can the same starting CLS describe different pictures?](https://nipunbatra.github.io/attention/vision1.html?present#s04/3/0)
- [Why did our text predictor hide later tokens?](https://nipunbatra.github.io/attention/vision1.html?present#s04/5/0)
- [Which part of the picture could help this dark crop?](https://nipunbatra.github.io/attention/vision1.html?present#s04/7/0)
- [How could the face crop get more weight?](https://nipunbatra.github.io/attention/vision1.html?present#s04/9/0)
- [What do we receive after choosing those weights?](https://nipunbatra.github.io/attention/vision1.html?present#s04/10/0)
- [Change only the keys. What happens?](https://nipunbatra.github.io/attention/vision1.html?present#s04/11/0)
- [Change only a value. What happens?](https://nipunbatra.github.io/attention/vision1.html?present#s04/12/0)
- [Do we type a question into this classifier?](https://nipunbatra.github.io/attention/vision1.html?present#s04/13/0)
- [Who teaches CLS what information to collect?](https://nipunbatra.github.io/attention/vision1.html?present#s06/11/0)
- [Could we classify the image without CLS?](https://nipunbatra.github.io/attention/vision1.html?present#s06/12/0)
- [So why use CLS in our ViT?](https://nipunbatra.github.io/attention/vision1.html?present#s06/13/0)

## Additions from the video review

- [Why does our ViT code use Conv2d?](https://nipunbatra.github.io/attention/vision1.html?present#s08/3/0)
- [How far should the filter move?](https://nipunbatra.github.io/attention/vision1.html?present#s08/4/0)
- [Does this layer merely cut up the image?](https://nipunbatra.github.io/attention/vision1.html?present#s08/5/0)
- [Can a patch in image A read image B?](https://nipunbatra.github.io/attention/vision1.html?present#s08/7/0)
- [A tensor can have the right shape and the wrong meaning](https://nipunbatra.github.io/attention/vision1.html?present#s08/8/0)
- [A quick check before training](https://nipunbatra.github.io/attention/vision1.html?present#s08/9/0)

## Complete sequence

| Section / frame | Question or teaching step | Stable source ID |
|---|---|---|
| s01 / 1 | [What animal do you see?](https://nipunbatra.github.io/attention/vision1.html?present#s01/1/0) | `s01-photo` |
| s01 / 2 | [Find the dog photos in a folder](https://nipunbatra.github.io/attention/vision1.html?present#s01/2/0) | `photo-folder` |
| s01 / 3 | [What were we asking the text model to predict?](https://nipunbatra.github.io/attention/vision1.html?present#s01/3/0) | `task-next-token` |
| s01 / 4 | [What are we asking the image model to predict?](https://nipunbatra.github.io/attention/vision1.html?present#s01/4/0) | `task-image-label` |
| s01 / 5 | [What changed, and what stayed the same?](https://nipunbatra.github.io/attention/vision1.html?present#s01/5/0) | `task-side-by-side` |
| s01 / 6 | [Suppose we also want to crop out the animal](https://nipunbatra.github.io/attention/vision1.html?present#s01/6/0) | `find-animal` |
| s01 / 7 | [Find the photo that matches this description](https://nipunbatra.github.io/attention/vision1.html?present#s01/7/0) | `photo-search` |
| s01 / 8 | [Would you recognize this crop on its own?](https://nipunbatra.github.io/attention/vision1.html?present#s01/8/0) | `s01-context` |
| s01 / 9 | [What can we carry over from our text models?](https://nipunbatra.github.io/attention/vision1.html?present#s01/9/0) | `bridge-text` |
| s01 / 10 | [Which other patches could help here?](https://nipunbatra.github.io/attention/vision1.html?present#s01/10/0) | `patch-context` |
| s01 / 11 | [Can a CNN use the rest of the image too?](https://nipunbatra.github.io/attention/vision1.html?present#s01/11/0) | `cnn-context` |
| s02 / 1 | [Where do the patch boundaries go?](https://nipunbatra.github.io/attention/vision1.html?present#s02/1/0) | `s01-patches` |
| s02 / 2 | [How can a red pixel be three numbers?](https://nipunbatra.github.io/attention/vision1.html?present#s02/2/0) | `one-rgb` |
| s02 / 3 | [How many numbers are in four RGB pixels?](https://nipunbatra.github.io/attention/vision1.html?present#s02/3/0) | `rgb-flatten-step-1` |
| s02 / 4 | [Write those twelve numbers in one row](https://nipunbatra.github.io/attention/vision1.html?present#s02/4/0) | `rgb-flatten` |
| s02 / 5 | [Which pixel goes first in the row?](https://nipunbatra.github.io/attention/vision1.html?present#s02/5/0) | `flatten-order` |
| s02 / 6 | [Give each patch its own row of pixels](https://nipunbatra.github.io/attention/vision1.html?present#s02/6/0) | `s01-rows-step-1` |
| s02 / 7 | [Use the same projection on every patch](https://nipunbatra.github.io/attention/vision1.html?present#s02/7/0) | `s01-rows` |
| s02 / 8 | [How many weights turn pixels into a patch row?](https://nipunbatra.github.io/attention/vision1.html?present#s02/8/0) | `projection-size` |
| s03 / 1 | [Same pieces, different picture?](https://nipunbatra.github.io/attention/vision1.html?present#s03/1/0) | `s02-small` |
| s03 / 2 | [Could you put the picture back together?](https://nipunbatra.github.io/attention/vision1.html?present#s03/2/0) | `position-question` |
| s03 / 3 | [Flatten P1 so we can multiply it](https://nipunbatra.github.io/attention/vision1.html?present#s03/3/0) | `s02-projection-step-1` |
| s03 / 4 | [Compute the first coordinate of P1](https://nipunbatra.github.io/attention/vision1.html?present#s03/4/0) | `s02-projection-step-2` |
| s03 / 5 | [Write the whole content row](https://nipunbatra.github.io/attention/vision1.html?present#s03/5/0) | `s02-projection` |
| s03 / 6 | [Where does each number in the patch row come from?](https://nipunbatra.github.io/attention/vision1.html?present#s03/6/0) | `patch-matrix` |
| s03 / 7 | [What row does an empty patch get?](https://nipunbatra.github.io/attention/vision1.html?present#s03/7/0) | `empty-patch` |
| s03 / 8 | [Would the mean pixel value tell these patches apart?](https://nipunbatra.github.io/attention/vision1.html?present#s03/8/0) | `mean-loses-edge` |
| s03 / 9 | [Could two projection columns keep that difference?](https://nipunbatra.github.io/attention/vision1.html?present#s03/9/0) | `edge-filters` |
| s03 / 10 | [These patches look the same. How do we tell them apart?](https://nipunbatra.github.io/attention/vision1.html?present#s03/10/0) | `two-identical-patches` |
| s03 / 11 | [Give each patch a location vector](https://nipunbatra.github.io/attention/vision1.html?present#s03/11/0) | `s02-positions-step-1` |
| s03 / 12 | [Add content and location, coordinate by coordinate](https://nipunbatra.github.io/attention/vision1.html?present#s03/12/0) | `s02-positions` |
| s04 / 1 | [We have several patch rows. Where does the answer go?](https://nipunbatra.github.io/attention/vision1.html?present#s04/1/0) | `why-cls` |
| s04 / 2 | [Where does CLS come from?](https://nipunbatra.github.io/attention/vision1.html?present#s04/2/0) | `cls-start` |
| s04 / 3 | [How can the same starting CLS describe different pictures?](https://nipunbatra.github.io/attention/vision1.html?present#s04/3/0) | `cls-two-images` |
| s04 / 4 | [Can the top-left patch read the bottom-right?](https://nipunbatra.github.io/attention/vision1.html?present#s04/4/0) | `image-mask` |
| s04 / 5 | [Why did our text predictor hide later tokens?](https://nipunbatra.github.io/attention/vision1.html?present#s04/5/0) | `task-mask-reason` |
| s04 / 6 | [Why do we make three versions of each row?](https://nipunbatra.github.io/attention/vision1.html?present#s04/6/0) | `qkv-roles` |
| s04 / 7 | [Which part of the picture could help this dark crop?](https://nipunbatra.github.io/attention/vision1.html?present#s04/7/0) | `qkv-photo-question` |
| s04 / 8 | [Each row makes a query, a key and a value](https://nipunbatra.github.io/attention/vision1.html?present#s04/8/0) | `qkv-three-roles` |
| s04 / 9 | [How could the face crop get more weight?](https://nipunbatra.github.io/attention/vision1.html?present#s04/9/0) | `qkv-match-numbers` |
| s04 / 10 | [What do we receive after choosing those weights?](https://nipunbatra.github.io/attention/vision1.html?present#s04/10/0) | `qkv-read-numbers` |
| s04 / 11 | [Change only the keys. What happens?](https://nipunbatra.github.io/attention/vision1.html?present#s04/11/0) | `qkv-change-key` |
| s04 / 12 | [Change only a value. What happens?](https://nipunbatra.github.io/attention/vision1.html?present#s04/12/0) | `qkv-change-value` |
| s04 / 13 | [Do we type a question into this classifier?](https://nipunbatra.github.io/attention/vision1.html?present#s04/13/0) | `qkv-no-prompt` |
| s04 / 14 | [Which weights create the query, key and value?](https://nipunbatra.github.io/attention/vision1.html?present#s04/14/0) | `all-qkv` |
| s04 / 15 | [Where does the CLS query [1,1] come from?](https://nipunbatra.github.io/attention/vision1.html?present#s04/15/0) | `q-dot` |
| s04 / 16 | [How does P1 get the key [√2,0]?](https://nipunbatra.github.io/attention/vision1.html?present#s04/16/0) | `one-key-dot` |
| s04 / 17 | [Which features does this head compare?](https://nipunbatra.github.io/attention/vision1.html?present#s04/17/0) | `s03-query` |
| s04 / 18 | [Can we work out one score before filling the table?](https://nipunbatra.github.io/attention/vision1.html?present#s04/18/0) | `one-score` |
| s04 / 19 | [What does softmax do when the scores tie?](https://nipunbatra.github.io/attention/vision1.html?present#s04/19/0) | `softmax-relative` |
| s04 / 20 | [Repeat the dot product for every source](https://nipunbatra.github.io/attention/vision1.html?present#s04/20/0) | `s03-weights-step-1` |
| s04 / 21 | [Exponentiate the five scores](https://nipunbatra.github.io/attention/vision1.html?present#s04/21/0) | `s03-weights-step-2` |
| s04 / 22 | [Divide by one shared sum](https://nipunbatra.github.io/attention/vision1.html?present#s04/22/0) | `s03-weights` |
| s04 / 23 | [Where does the 0.229 beside P1 come from?](https://nipunbatra.github.io/attention/vision1.html?present#s04/23/0) | `weight-denominator` |
| s04 / 24 | [Put each value beside its weight](https://nipunbatra.github.io/attention/vision1.html?present#s04/24/0) | `s03-values-step-1` |
| s04 / 25 | [Multiply the value by its weight](https://nipunbatra.github.io/attention/vision1.html?present#s04/25/0) | `s03-values-step-2` |
| s04 / 26 | [Add the contributions to get one message](https://nipunbatra.github.io/attention/vision1.html?present#s04/26/0) | `s03-values` |
| s04 / 27 | [Can an empty patch still send something?](https://nipunbatra.github.io/attention/vision1.html?present#s04/27/0) | `one-value-product` |
| s04 / 28 | [Do equal weights send equal information?](https://nipunbatra.github.io/attention/vision1.html?present#s04/28/0) | `weight-message` |
| s04 / 29 | [What if the query cared only about ink?](https://nipunbatra.github.io/attention/vision1.html?present#s04/29/0) | `change-query` |
| s05 / 1 | [Would a second way of reading the image help?](https://nipunbatra.github.io/attention/vision1.html?present#s05/1/0) | `heads-question` |
| s05 / 2 | [Head 2 compares ink and column](https://nipunbatra.github.io/attention/vision1.html?present#s05/2/0) | `s03-second-step-1` |
| s05 / 3 | [Give Head 2 its own softmax](https://nipunbatra.github.io/attention/vision1.html?present#s05/3/0) | `s03-second` |
| s05 / 4 | [What information does Head 2 send?](https://nipunbatra.github.io/attention/vision1.html?present#s05/4/0) | `s03-second-values-step-1` |
| s05 / 5 | [Calculate each Head 2 contribution](https://nipunbatra.github.io/attention/vision1.html?present#s05/5/0) | `s03-second-values-step-2` |
| s05 / 6 | [Add Head 2’s contributions](https://nipunbatra.github.io/attention/vision1.html?present#s05/6/0) | `s03-second-values` |
| s05 / 7 | [Do the patch rows get updated too?](https://nipunbatra.github.io/attention/vision1.html?present#s05/7/0) | `all-receivers` |
| s05 / 8 | [Keep both messages by putting them side by side](https://nipunbatra.github.io/attention/vision1.html?present#s05/8/0) | `s04-join-step-1` |
| s05 / 9 | [Use W_O to make the first update coordinate](https://nipunbatra.github.io/attention/vision1.html?present#s05/9/0) | `s04-join-step-2` |
| s05 / 10 | [Collect all four output coordinates](https://nipunbatra.github.io/attention/vision1.html?present#s05/10/0) | `s04-join` |
| s05 / 11 | [What do the other columns of W_O produce?](https://nipunbatra.github.io/attention/vision1.html?present#s05/11/0) | `other-output-coordinates` |
| s06 / 1 | [What if attention sent a zero message?](https://nipunbatra.github.io/attention/vision1.html?present#s06/1/0) | `residual-zero` |
| s06 / 2 | [Add the message to the starting CLS row](https://nipunbatra.github.io/attention/vision1.html?present#s06/2/0) | `s04-residual-step-1` |
| s06 / 3 | [Use the updated CLS row to score the two labels](https://nipunbatra.github.io/attention/vision1.html?present#s06/3/0) | `s04-residual` |
| s06 / 4 | [We used softmax twice. What changed?](https://nipunbatra.github.io/attention/vision1.html?present#s06/4/0) | `two-softmaxes` |
| s06 / 5 | [Turn the class scores into probabilities](https://nipunbatra.github.io/attention/vision1.html?present#s06/5/0) | `s04-probability-step-1` |
| s06 / 6 | [Use the known label to calculate the loss](https://nipunbatra.github.io/attention/vision1.html?present#s06/6/0) | `s04-probability` |
| s06 / 7 | [How much does a confident wrong answer cost?](https://nipunbatra.github.io/attention/vision1.html?present#s06/7/0) | `loss-comparison` |
| s06 / 8 | [Use the gradient to change the class bias](https://nipunbatra.github.io/attention/vision1.html?present#s06/8/0) | `one-update-step-1` |
| s06 / 9 | [Run the prediction again after that update](https://nipunbatra.github.io/attention/vision1.html?present#s06/9/0) | `one-update` |
| s06 / 10 | [Move the patches. Does the answer change?](https://nipunbatra.github.io/attention/vision1.html?present#s06/10/0) | `s04-experiment` |
| s06 / 11 | [Who teaches CLS what information to collect?](https://nipunbatra.github.io/attention/vision1.html?present#s06/11/0) | `cls-learns` |
| s06 / 12 | [Could we classify the image without CLS?](https://nipunbatra.github.io/attention/vision1.html?present#s06/12/0) | `pooling-example` |
| s06 / 13 | [So why use CLS in our ViT?](https://nipunbatra.github.io/attention/vision1.html?present#s06/13/0) | `readout-choice` |
| s07 / 1 | [Put the familiar attention inside a full block](https://nipunbatra.github.io/attention/vision1.html?present#s07/1/0) | `s05-block` |
| s07 / 2 | [Which operation lets one patch borrow from another?](https://nipunbatra.github.io/attention/vision1.html?present#s07/2/0) | `mix-across-rows` |
| s07 / 3 | [What does the MLP change?](https://nipunbatra.github.io/attention/vision1.html?present#s07/3/0) | `mix-within-row` |
| s07 / 4 | [What average are we subtracting?](https://nipunbatra.github.io/attention/vision1.html?present#s07/4/0) | `ln-mean` |
| s07 / 5 | [How spread out is the centered row?](https://nipunbatra.github.io/attention/vision1.html?present#s07/5/0) | `ln-variance` |
| s07 / 6 | [What does LayerNorm do to one row?](https://nipunbatra.github.io/attention/vision1.html?present#s07/6/0) | `layernorm` |
| s07 / 7 | [How does the MLP make a wider row?](https://nipunbatra.github.io/attention/vision1.html?present#s07/7/0) | `mlp-first-linear` |
| s07 / 8 | [Apply GELU to each hidden coordinate](https://nipunbatra.github.io/attention/vision1.html?present#s07/8/0) | `mlp-row-step-1` |
| s07 / 9 | [How does the MLP return to the original width?](https://nipunbatra.github.io/attention/vision1.html?present#s07/9/0) | `mlp-second-linear` |
| s07 / 10 | [Add the MLP message to the starting row](https://nipunbatra.github.io/attention/vision1.html?present#s07/10/0) | `mlp-row` |
| s07 / 11 | [What does the next block get to see?](https://nipunbatra.github.io/attention/vision1.html?present#s07/11/0) | `depth` |
| s07 / 12 | [How many rows does the real photograph produce?](https://nipunbatra.github.io/attention/vision1.html?present#s07/12/0) | `s05-scale-step-1` |
| s07 / 13 | [How many blocks process those rows?](https://nipunbatra.github.io/attention/vision1.html?present#s07/13/0) | `s05-scale-step-2` |
| s07 / 14 | [What changed when we made the model larger?](https://nipunbatra.github.io/attention/vision1.html?present#s07/14/0) | `s05-scale` |
| s08 / 1 | [These two 16s mean different things](https://nipunbatra.github.io/attention/vision1.html?present#s08/1/0) | `two-sixteens` |
| s08 / 2 | [How do image pixels become rows in code?](https://nipunbatra.github.io/attention/vision1.html?present#s08/2/0) | `code-patch` |
| s08 / 3 | [Why does our ViT code use Conv2d?](https://nipunbatra.github.io/attention/vision1.html?present#s08/3/0) | `conv-one-patch` |
| s08 / 4 | [How far should the filter move?](https://nipunbatra.github.io/attention/vision1.html?present#s08/4/0) | `conv-stride` |
| s08 / 5 | [Does this layer merely cut up the image?](https://nipunbatra.github.io/attention/vision1.html?present#s08/5/0) | `conv-trainable` |
| s08 / 6 | [Can you match each line to our calculation?](https://nipunbatra.github.io/attention/vision1.html?present#s08/6/0) | `code-attention` |
| s08 / 7 | [Can a patch in image A read image B?](https://nipunbatra.github.io/attention/vision1.html?present#s08/7/0) | `batch-boundary` |
| s08 / 8 | [A tensor can have the right shape and the wrong meaning](https://nipunbatra.github.io/attention/vision1.html?present#s08/8/0) | `batch-axis` |
| s08 / 9 | [A quick check before training](https://nipunbatra.github.io/attention/vision1.html?present#s08/9/0) | `batch-check` |
| s08 / 10 | [What is the complete pre-LayerNorm block?](https://nipunbatra.github.io/attention/vision1.html?present#s08/10/0) | `code-block` |
| s08 / 11 | [How do we add one CLS row per image?](https://nipunbatra.github.io/attention/vision1.html?present#s08/11/0) | `code-add-cls` |
| s08 / 12 | [Where does location enter the code?](https://nipunbatra.github.io/attention/vision1.html?present#s08/12/0) | `code-add-pos` |
| s08 / 13 | [Which row reaches the classifier?](https://nipunbatra.github.io/attention/vision1.html?present#s08/13/0) | `code-cls-readout` |
| s08 / 14 | [How does the full model produce image logits?](https://nipunbatra.github.io/attention/vision1.html?present#s08/14/0) | `code-model` |
| s08 / 15 | [Which call makes this model learn?](https://nipunbatra.github.io/attention/vision1.html?present#s08/15/0) | `code-train` |
| s08 / 16 | [When are gradients computed?](https://nipunbatra.github.io/attention/vision1.html?present#s08/16/0) | `code-backward` |
| s08 / 17 | [Which line changes the weights?](https://nipunbatra.github.io/attention/vision1.html?present#s08/17/0) | `code-step` |
| s09 / 1 | [Will the model recognize a new noisy stripe?](https://nipunbatra.github.io/attention/vision1.html?present#s09/1/0) | `training-data` |
| s09 / 2 | [Which images are allowed to influence the weights?](https://nipunbatra.github.io/attention/vision1.html?present#s09/2/0) | `three-splits` |
| s09 / 3 | [Suppose the model gets this training image wrong](https://nipunbatra.github.io/attention/vision1.html?present#s09/3/0) | `training-one-image` |
| s09 / 4 | [What does one epoch mean in this experiment?](https://nipunbatra.github.io/attention/vision1.html?present#s09/4/0) | `one-epoch` |
| s09 / 5 | [Is the model improving on its training images?](https://nipunbatra.github.io/attention/vision1.html?present#s09/5/0) | `learning-curves-step-1` |
| s09 / 6 | [Does the improvement carry over to validation?](https://nipunbatra.github.io/attention/vision1.html?present#s09/6/0) | `learning-curves` |
| s09 / 7 | [Can training make up for missing positions?](https://nipunbatra.github.io/attention/vision1.html?present#s09/7/0) | `trained-position-control` |
| s10 / 1 | [Which pixels are we giving the real model?](https://nipunbatra.github.io/attention/vision1.html?present#s10/1/0) | `real-input` |
| s10 / 2 | [What did the model call our dog?](https://nipunbatra.github.io/attention/vision1.html?present#s10/2/0) | `s06-answer` |
| s10 / 3 | [Does one correct photograph tell us the accuracy?](https://nipunbatra.github.io/attention/vision1.html?present#s10/3/0) | `one-photo-limit` |
| s10 / 4 | [What happens when we give it the cat?](https://nipunbatra.github.io/attention/vision1.html?present#s10/4/0) | `real-cat` |
| s10 / 5 | [Where did the checkpoint learn its visual features?](https://nipunbatra.github.io/attention/vision1.html?present#s10/5/0) | `three-phases-step-1` |
| s10 / 6 | [How would we adapt it to our own labels?](https://nipunbatra.github.io/attention/vision1.html?present#s10/6/0) | `three-phases-step-2` |
| s10 / 7 | [What happens when we classify a new photograph?](https://nipunbatra.github.io/attention/vision1.html?present#s10/7/0) | `three-phases` |
| s11 / 1 | [What is one coloured square actually showing?](https://nipunbatra.github.io/attention/vision1.html?present#s11/1/0) | `read-attention-map` |
| s11 / 2 | [Do the heads read the same places?](https://nipunbatra.github.io/attention/vision1.html?present#s11/2/0) | `real-heads` |
| s11 / 3 | [Does CLS read differently in a later block?](https://nipunbatra.github.io/attention/vision1.html?present#s11/3/0) | `real-depth` |
| s11 / 4 | [What does this particular patch read?](https://nipunbatra.github.io/attention/vision1.html?present#s11/4/0) | `real-patch-query` |
| s11 / 5 | [What happens if we cover the top left?](https://nipunbatra.github.io/attention/vision1.html?present#s11/5/0) | `cover-1` |
| s11 / 6 | [What happens if we cover the top right?](https://nipunbatra.github.io/attention/vision1.html?present#s11/6/0) | `cover-2` |
| s11 / 7 | [What happens if we cover the bottom left?](https://nipunbatra.github.io/attention/vision1.html?present#s11/7/0) | `cover-3` |
| s11 / 8 | [What happens if we cover the bottom right?](https://nipunbatra.github.io/attention/vision1.html?present#s11/8/0) | `cover-4` |
| s11 / 9 | [Which covered region changed the answer most?](https://nipunbatra.github.io/attention/vision1.html?present#s11/9/0) | `occlusion` |
| s12 / 1 | [What changes when the patch size is halved?](https://nipunbatra.github.io/attention/vision1.html?present#s12/1/0) | `patch-cost` |
| s12 / 2 | [How much matching happens inside the tiny real model?](https://nipunbatra.github.io/attention/vision1.html?present#s12/2/0) | `real-work-count` |
| s12 / 3 | [What happens if we use a larger image?](https://nipunbatra.github.io/attention/vision1.html?present#s12/3/0) | `cost-control` |
| s13 / 1 | [Your turn: work out the message](https://nipunbatra.github.io/attention/vision1.html?present#s13/1/0) | `exercise-message` |
| s13 / 2 | [Your turn: trace every important shape](https://nipunbatra.github.io/attention/vision1.html?present#s13/2/0) | `exercise-shapes` |
| s13 / 3 | [Did we move the image, or just reorder its rows?](https://nipunbatra.github.io/attention/vision1.html?present#s13/3/0) | `exercise-position` |
| s14 / 1 | [What else could we ask the image model to do?](https://nipunbatra.github.io/attention/vision1.html?present#s14/1/0) | `next-vision` |
| s14 / 2 | [Can you talk us through the whole model?](https://nipunbatra.github.io/attention/vision1.html?present#s14/2/0) | `closing` |
