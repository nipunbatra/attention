# Vision I: slide map

198 teaching frames plus the cover. Each link opens the first reveal of that slide. Use Right/Left for reveals and S for presenter notes.

## Review these additions first

- [What were we asking the text model to predict?](https://nipunbatra.github.io/attention/vision1.html?present#s01/12/0)
- [Our task today: classify the whole image](https://nipunbatra.github.io/attention/vision1.html?present#s01/11/0)
- [What changed, and what stayed the same?](https://nipunbatra.github.io/attention/vision1.html?present#s01/14/0)
- [Where does CLS come from?](https://nipunbatra.github.io/attention/vision1.html?present#s03/15/0)
- [How can the same starting CLS describe different pictures?](https://nipunbatra.github.io/attention/vision1.html?present#s03/16/0)
- [Why did our text predictor hide later tokens?](https://nipunbatra.github.io/attention/vision1.html?present#s04/3/0)
- [Which part of the picture could help this dark crop?](https://nipunbatra.github.io/attention/vision1.html?present#s04/5/0)
- [How could the face crop get more weight?](https://nipunbatra.github.io/attention/vision1.html?present#s04/7/0)
- [What do we receive after choosing those weights?](https://nipunbatra.github.io/attention/vision1.html?present#s04/8/0)
- [Change only the keys. What happens?](https://nipunbatra.github.io/attention/vision1.html?present#s04/9/0)
- [Change only a value. What happens?](https://nipunbatra.github.io/attention/vision1.html?present#s04/10/0)
- [Do we type a question into this classifier?](https://nipunbatra.github.io/attention/vision1.html?present#s04/11/0)
- [Who teaches CLS what information to collect?](https://nipunbatra.github.io/attention/vision1.html?present#s06/12/0)
- [Could we classify the image without CLS?](https://nipunbatra.github.io/attention/vision1.html?present#s06/13/0)
- [So why use CLS in our ViT?](https://nipunbatra.github.io/attention/vision1.html?present#s06/14/0)

## Additions from the video review

- [Why does our ViT code use Conv2d?](https://nipunbatra.github.io/attention/vision1.html?present#s08/4/0)
- [How far should the filter move?](https://nipunbatra.github.io/attention/vision1.html?present#s08/5/0)
- [Does this layer merely cut up the image?](https://nipunbatra.github.io/attention/vision1.html?present#s08/6/0)
- [Can a patch in image A read image B?](https://nipunbatra.github.io/attention/vision1.html?present#s08/9/0)
- [A tensor can have the right shape and the wrong meaning](https://nipunbatra.github.io/attention/vision1.html?present#s08/10/0)
- [A quick check before training](https://nipunbatra.github.io/attention/vision1.html?present#s08/11/0)

## Start with the dataset

- [What does our animal dataset look like?](https://nipunbatra.github.io/attention/vision1.html?present#s01/2/0)
- [How many images and classes are there?](https://nipunbatra.github.io/attention/vision1.html?present#s01/3/0)
- [What shape is one image?](https://nipunbatra.github.io/attention/vision1.html?present#s01/4/0)

## Opening task overview

- [One photograph, several possible tasks](https://nipunbatra.github.io/attention/vision1.html?present#s01/6/0)
- [Classification: name the animal](https://nipunbatra.github.io/attention/vision1.html?present#s01/7/0)
- [Detection: name and locate each object](https://nipunbatra.github.io/attention/vision1.html?present#s01/8/0)
- [Captioning: describe the image in words](https://nipunbatra.github.io/attention/vision1.html?present#s01/9/0)
- [Image–text search: find a matching photo](https://nipunbatra.github.io/attention/vision1.html?present#s01/10/0)
- [Our task today: classify the whole image](https://nipunbatra.github.io/attention/vision1.html?present#s01/11/0)

## Follow information into one patch

- [What can the face tell this dark patch?](https://nipunbatra.github.io/attention/vision1.html?present#s01/17/0)
- [Should every source contribute equally?](https://nipunbatra.github.io/attention/vision1.html?present#s01/18/0)
- [What changes when the patch gets context?](https://nipunbatra.github.io/attention/vision1.html?present#s01/19/0)

## Recall text, then ask the image parallels

- [Back to text: what did attention update?](https://nipunbatra.github.io/attention/vision1.html?present#s01/20/0)
- [What could be the image equivalent of a token?](https://nipunbatra.github.io/attention/vision1.html?present#s01/21/0)
- [What could be the image equivalent of an embedding?](https://nipunbatra.github.io/attention/vision1.html?present#s01/22/0)
- [What could a query be in the image?](https://nipunbatra.github.io/attention/vision1.html?present#s01/23/0)
- [What could a key be in the image?](https://nipunbatra.github.io/attention/vision1.html?present#s01/27/0)
- [What information would a value send?](https://nipunbatra.github.io/attention/vision1.html?present#s01/29/0)
- [What is the “next token” for this image?](https://nipunbatra.github.io/attention/vision1.html?present#s01/31/0)

## RGB pixels to patch embeddings

- [Read the RGB values of each pixel](https://nipunbatra.github.io/attention/vision1.html?present#s02/4/0)
- [Put the four RGB triples in one row](https://nipunbatra.github.io/attention/vision1.html?present#s02/5/0)
- [What does “projection” mean here?](https://nipunbatra.github.io/attention/vision1.html?present#s02/9/0)
- [Follow the connections into output 1](https://nipunbatra.github.io/attention/vision1.html?present#s02/11/0)
- [Now follow the connections into output 2](https://nipunbatra.github.io/attention/vision1.html?present#s02/12/0)
- [Apply the very same layer to another patch](https://nipunbatra.github.io/attention/vision1.html?present#s02/14/0)
- [Back to the dog: follow one real patch](https://nipunbatra.github.io/attention/vision1.html?present#s02/15/0)
- [How many rows come from the whole image?](https://nipunbatra.github.io/attention/vision1.html?present#s02/20/0)

## Read the dimensions

- [Where do the 768 input values come from?](https://nipunbatra.github.io/attention/vision1.html?present#s02/16/0)
- [What does the 1 in 1 × 768 count?](https://nipunbatra.github.io/attention/vision1.html?present#s02/17/0)
- [Does the embedding need 768 coordinates too?](https://nipunbatra.github.io/attention/vision1.html?present#s02/18/0)
- [How many rows come from the whole image?](https://nipunbatra.github.io/attention/vision1.html?present#s02/20/0)
- [How many parameters does this one layer learn?](https://nipunbatra.github.io/attention/vision1.html?present#s02/21/0)

## Photograph first, then position arithmetic

- [03 · Remembering where patches belong](https://nipunbatra.github.io/attention/vision1.html?present#s03/1/0)
- [Move the face patches. What changes?](https://nipunbatra.github.io/attention/vision1.html?present#s03/2/0)
- [Does the patch layer notice the move?](https://nipunbatra.github.io/attention/vision1.html?present#s03/3/0)
- [Give the row its location as well as its content](https://nipunbatra.github.io/attention/vision1.html?present#s03/4/0)
- [A smaller task: classify the arrangement](https://nipunbatra.github.io/attention/vision1.html?present#s03/5/0)
- [Would just counting the patches solve it?](https://nipunbatra.github.io/attention/vision1.html?present#s03/6/0)

## Both familiar text prediction examples

- [What were we asking the text model to predict?](https://nipunbatra.github.io/attention/vision1.html?present#s01/12/0)
- [And what were we predicting in the bank example?](https://nipunbatra.github.io/attention/vision1.html?present#s01/13/0)
- [What changed, and what stayed the same?](https://nipunbatra.github.io/attention/vision1.html?present#s01/14/0)

## Concrete visual queries, keys and values

- [Could this dark texture belong to the animal?](https://nipunbatra.github.io/attention/vision1.html?present#s01/24/0)
- [Where is the rest of this face?](https://nipunbatra.github.io/attention/vision1.html?present#s01/25/0)
- [Where does this branch continue?](https://nipunbatra.github.io/attention/vision1.html?present#s01/26/0)
- [What could each source offer for matching?](https://nipunbatra.github.io/attention/vision1.html?present#s01/28/0)
- [What information could these values carry?](https://nipunbatra.github.io/attention/vision1.html?present#s01/30/0)

## Patch projection and activation

- [How does this connect to text embeddings?](https://nipunbatra.github.io/attention/vision1.html?present#s02/7/0)
- [Do we apply an activation after the patch layer?](https://nipunbatra.github.io/attention/vision1.html?present#s02/8/0)
- [These two numbers are the patch embedding](https://nipunbatra.github.io/attention/vision1.html?present#s02/13/0)

## Follow one real patch through every step

- [Back to the dog: follow one real patch](https://nipunbatra.github.io/attention/vision1.html?present#s02/15/0)
- [The patch projection produces c₆₃](https://nipunbatra.github.io/attention/vision1.html?present#s02/19/0)
- [Content + position gives the block’s input row](https://nipunbatra.github.io/attention/vision1.html?present#s02/22/0)
- [Then the block makes queries, keys and values](https://nipunbatra.github.io/attention/vision1.html?present#s02/23/0)

## Complete sequence

| Slide | Question | Frame ID |
|---|---|---|
| s01 / 1 | [01 · The image classification task](https://nipunbatra.github.io/attention/vision1.html?present#s01/1/0) | `vision-topic-01` |
| s01 / 2 | [What does our animal dataset look like?](https://nipunbatra.github.io/attention/vision1.html?present#s01/2/0) | `dataset-gallery` |
| s01 / 3 | [How many images and classes are there?](https://nipunbatra.github.io/attention/vision1.html?present#s01/3/0) | `dataset-counts` |
| s01 / 4 | [What shape is one image?](https://nipunbatra.github.io/attention/vision1.html?present#s01/4/0) | `dataset-dimensions` |
| s01 / 5 | [What animal do you see?](https://nipunbatra.github.io/attention/vision1.html?present#s01/5/0) | `s01-photo` |
| s01 / 6 | [One photograph, several possible tasks](https://nipunbatra.github.io/attention/vision1.html?present#s01/6/0) | `vision-tasks` |
| s01 / 7 | [Classification: name the animal](https://nipunbatra.github.io/attention/vision1.html?present#s01/7/0) | `photo-folder` |
| s01 / 8 | [Detection: name and locate each object](https://nipunbatra.github.io/attention/vision1.html?present#s01/8/0) | `find-animal` |
| s01 / 9 | [Captioning: describe the image in words](https://nipunbatra.github.io/attention/vision1.html?present#s01/9/0) | `image-caption` |
| s01 / 10 | [Image–text search: find a matching photo](https://nipunbatra.github.io/attention/vision1.html?present#s01/10/0) | `photo-search` |
| s01 / 11 | [Our task today: classify the whole image](https://nipunbatra.github.io/attention/vision1.html?present#s01/11/0) | `task-image-label` |
| s01 / 12 | [What were we asking the text model to predict?](https://nipunbatra.github.io/attention/vision1.html?present#s01/12/0) | `task-next-token` |
| s01 / 13 | [And what were we predicting in the bank example?](https://nipunbatra.github.io/attention/vision1.html?present#s01/13/0) | `task-bank-next-token` |
| s01 / 14 | [What changed, and what stayed the same?](https://nipunbatra.github.io/attention/vision1.html?present#s01/14/0) | `task-side-by-side` |
| s01 / 15 | [Would you recognize this crop on its own?](https://nipunbatra.github.io/attention/vision1.html?present#s01/15/0) | `s01-context` |
| s01 / 16 | [What can we carry over from our text models?](https://nipunbatra.github.io/attention/vision1.html?present#s01/16/0) | `bridge-text` |
| s01 / 17 | [What can the face tell this dark patch?](https://nipunbatra.github.io/attention/vision1.html?present#s01/17/0) | `patch-context` |
| s01 / 18 | [Should every source contribute equally?](https://nipunbatra.github.io/attention/vision1.html?present#s01/18/0) | `patch-context-weights` |
| s01 / 19 | [What changes when the patch gets context?](https://nipunbatra.github.io/attention/vision1.html?present#s01/19/0) | `patch-context-update` |
| s01 / 20 | [Back to text: what did attention update?](https://nipunbatra.github.io/attention/vision1.html?present#s01/20/0) | `text-context-recap` |
| s01 / 21 | [What could be the image equivalent of a token?](https://nipunbatra.github.io/attention/vision1.html?present#s01/21/0) | `bridge-image-token` |
| s01 / 22 | [What could be the image equivalent of an embedding?](https://nipunbatra.github.io/attention/vision1.html?present#s01/22/0) | `bridge-image-embedding` |
| s01 / 23 | [What could a query be in the image?](https://nipunbatra.github.io/attention/vision1.html?present#s01/23/0) | `bridge-image-query` |
| s01 / 24 | [Could this dark texture belong to the animal?](https://nipunbatra.github.io/attention/vision1.html?present#s01/24/0) | `query-example-dark` |
| s01 / 25 | [Where is the rest of this face?](https://nipunbatra.github.io/attention/vision1.html?present#s01/25/0) | `query-example-face` |
| s01 / 26 | [Where does this branch continue?](https://nipunbatra.github.io/attention/vision1.html?present#s01/26/0) | `query-example-branch` |
| s01 / 27 | [What could a key be in the image?](https://nipunbatra.github.io/attention/vision1.html?present#s01/27/0) | `bridge-image-key` |
| s01 / 28 | [What could each source offer for matching?](https://nipunbatra.github.io/attention/vision1.html?present#s01/28/0) | `key-example-sources` |
| s01 / 29 | [What information would a value send?](https://nipunbatra.github.io/attention/vision1.html?present#s01/29/0) | `bridge-image-value` |
| s01 / 30 | [What information could these values carry?](https://nipunbatra.github.io/attention/vision1.html?present#s01/30/0) | `value-example-messages` |
| s01 / 31 | [What is the “next token” for this image?](https://nipunbatra.github.io/attention/vision1.html?present#s01/31/0) | `bridge-image-target` |
| s01 / 32 | [How can we give this photograph to attention?](https://nipunbatra.github.io/attention/vision1.html?present#s01/32/0) | `image-to-rows` |
| s02 / 1 | [02 · From pixels to patch embeddings](https://nipunbatra.github.io/attention/vision1.html?present#s02/1/0) | `vision-topic-02` |
| s02 / 2 | [Where do the patch boundaries go?](https://nipunbatra.github.io/attention/vision1.html?present#s02/2/0) | `s01-patches` |
| s02 / 3 | [How can a red pixel be three numbers?](https://nipunbatra.github.io/attention/vision1.html?present#s02/3/0) | `one-rgb` |
| s02 / 4 | [Read the RGB values of each pixel](https://nipunbatra.github.io/attention/vision1.html?present#s02/4/0) | `rgb-flatten-step-1` |
| s02 / 5 | [Put the four RGB triples in one row](https://nipunbatra.github.io/attention/vision1.html?present#s02/5/0) | `rgb-flatten` |
| s02 / 6 | [Which pixel goes first in the row?](https://nipunbatra.github.io/attention/vision1.html?present#s02/6/0) | `flatten-order` |
| s02 / 7 | [How does this connect to text embeddings?](https://nipunbatra.github.io/attention/vision1.html?present#s02/7/0) | `s01-rows-step-1` |
| s02 / 8 | [Do we apply an activation after the patch layer?](https://nipunbatra.github.io/attention/vision1.html?present#s02/8/0) | `patch-activation-location` |
| s02 / 9 | [What does “projection” mean here?](https://nipunbatra.github.io/attention/vision1.html?present#s02/9/0) | `patch-linear-shapes` |
| s02 / 10 | [12 input numbers, 2 output numbers](https://nipunbatra.github.io/attention/vision1.html?present#s02/10/0) | `patch-linear-weights` |
| s02 / 11 | [Follow the connections into output 1](https://nipunbatra.github.io/attention/vision1.html?present#s02/11/0) | `patch-linear-first` |
| s02 / 12 | [Now follow the connections into output 2](https://nipunbatra.github.io/attention/vision1.html?present#s02/12/0) | `patch-linear-second` |
| s02 / 13 | [These two numbers are the patch embedding](https://nipunbatra.github.io/attention/vision1.html?present#s02/13/0) | `patch-linear-result` |
| s02 / 14 | [Apply the very same layer to another patch](https://nipunbatra.github.io/attention/vision1.html?present#s02/14/0) | `patch-shared-code` |
| s02 / 15 | [Back to the dog: follow one real patch](https://nipunbatra.github.io/attention/vision1.html?present#s02/15/0) | `s01-rows` |
| s02 / 16 | [Where do the 768 input values come from?](https://nipunbatra.github.io/attention/vision1.html?present#s02/16/0) | `patch-real-dimensions` |
| s02 / 17 | [What does the 1 in 1 × 768 count?](https://nipunbatra.github.io/attention/vision1.html?present#s02/17/0) | `patch-one-row-shape` |
| s02 / 18 | [Does the embedding need 768 coordinates too?](https://nipunbatra.github.io/attention/vision1.html?present#s02/18/0) | `patch-one-row-projection` |
| s02 / 19 | [The patch projection produces c₆₃](https://nipunbatra.github.io/attention/vision1.html?present#s02/19/0) | `real-patch-projection` |
| s02 / 20 | [How many rows come from the whole image?](https://nipunbatra.github.io/attention/vision1.html?present#s02/20/0) | `projection-size` |
| s02 / 21 | [How many parameters does this one layer learn?](https://nipunbatra.github.io/attention/vision1.html?present#s02/21/0) | `patch-projection-parameters` |
| s02 / 22 | [Content + position gives the block’s input row](https://nipunbatra.github.io/attention/vision1.html?present#s02/22/0) | `real-patch-position` |
| s02 / 23 | [Then the block makes queries, keys and values](https://nipunbatra.github.io/attention/vision1.html?present#s02/23/0) | `real-patch-qkv` |
| s03 / 1 | [03 · Remembering where patches belong](https://nipunbatra.github.io/attention/vision1.html?present#s03/1/0) | `vision-topic-03` |
| s03 / 2 | [Move the face patches. What changes?](https://nipunbatra.github.io/attention/vision1.html?present#s03/2/0) | `position-photo-layout` |
| s03 / 3 | [Does the patch layer notice the move?](https://nipunbatra.github.io/attention/vision1.html?present#s03/3/0) | `position-photo-content` |
| s03 / 4 | [Give the row its location as well as its content](https://nipunbatra.github.io/attention/vision1.html?present#s03/4/0) | `position-photo-add` |
| s03 / 5 | [A smaller task: classify the arrangement](https://nipunbatra.github.io/attention/vision1.html?present#s03/5/0) | `s02-small` |
| s03 / 6 | [Would just counting the patches solve it?](https://nipunbatra.github.io/attention/vision1.html?present#s03/6/0) | `position-question` |
| s03 / 7 | [Flatten P1 so we can multiply it](https://nipunbatra.github.io/attention/vision1.html?present#s03/7/0) | `s02-projection-step-1` |
| s03 / 8 | [Compute the first coordinate of P1](https://nipunbatra.github.io/attention/vision1.html?present#s03/8/0) | `s02-projection-step-2` |
| s03 / 9 | [Write the whole content row](https://nipunbatra.github.io/attention/vision1.html?present#s03/9/0) | `s02-projection` |
| s03 / 10 | [Where does each number in the patch row come from?](https://nipunbatra.github.io/attention/vision1.html?present#s03/10/0) | `patch-matrix` |
| s03 / 11 | [What row does an empty patch get?](https://nipunbatra.github.io/attention/vision1.html?present#s03/11/0) | `empty-patch` |
| s03 / 12 | [Would the mean pixel value tell these patches apart?](https://nipunbatra.github.io/attention/vision1.html?present#s03/12/0) | `mean-loses-edge` |
| s03 / 13 | [Could two projection columns keep that difference?](https://nipunbatra.github.io/attention/vision1.html?present#s03/13/0) | `edge-filters` |
| s03 / 14 | [We have several patch rows. Where does the answer go?](https://nipunbatra.github.io/attention/vision1.html?present#s03/14/0) | `why-cls` |
| s03 / 15 | [Where does CLS come from?](https://nipunbatra.github.io/attention/vision1.html?present#s03/15/0) | `cls-start` |
| s03 / 16 | [How can the same starting CLS describe different pictures?](https://nipunbatra.github.io/attention/vision1.html?present#s03/16/0) | `cls-two-images` |
| s03 / 17 | [These patches look the same. How do we tell them apart?](https://nipunbatra.github.io/attention/vision1.html?present#s03/17/0) | `two-identical-patches` |
| s03 / 18 | [Give each patch a location vector](https://nipunbatra.github.io/attention/vision1.html?present#s03/18/0) | `s02-positions-step-1` |
| s03 / 19 | [Add content and location, coordinate by coordinate](https://nipunbatra.github.io/attention/vision1.html?present#s03/19/0) | `s02-positions` |
| s04 / 1 | [04 · Queries, keys and values in vision](https://nipunbatra.github.io/attention/vision1.html?present#s04/1/0) | `vision-topic-04` |
| s04 / 2 | [Can the top-left patch read the bottom-right?](https://nipunbatra.github.io/attention/vision1.html?present#s04/2/0) | `image-mask` |
| s04 / 3 | [Why did our text predictor hide later tokens?](https://nipunbatra.github.io/attention/vision1.html?present#s04/3/0) | `task-mask-reason` |
| s04 / 4 | [Why do we make three versions of each row?](https://nipunbatra.github.io/attention/vision1.html?present#s04/4/0) | `qkv-roles` |
| s04 / 5 | [Which part of the picture could help this dark crop?](https://nipunbatra.github.io/attention/vision1.html?present#s04/5/0) | `qkv-photo-question` |
| s04 / 6 | [Each row makes a query, a key and a value](https://nipunbatra.github.io/attention/vision1.html?present#s04/6/0) | `qkv-three-roles` |
| s04 / 7 | [How could the face crop get more weight?](https://nipunbatra.github.io/attention/vision1.html?present#s04/7/0) | `qkv-match-numbers` |
| s04 / 8 | [What do we receive after choosing those weights?](https://nipunbatra.github.io/attention/vision1.html?present#s04/8/0) | `qkv-read-numbers` |
| s04 / 9 | [Change only the keys. What happens?](https://nipunbatra.github.io/attention/vision1.html?present#s04/9/0) | `qkv-change-key` |
| s04 / 10 | [Change only a value. What happens?](https://nipunbatra.github.io/attention/vision1.html?present#s04/10/0) | `qkv-change-value` |
| s04 / 11 | [Do we type a question into this classifier?](https://nipunbatra.github.io/attention/vision1.html?present#s04/11/0) | `qkv-no-prompt` |
| s04 / 12 | [Which weights create the query, key and value?](https://nipunbatra.github.io/attention/vision1.html?present#s04/12/0) | `all-qkv` |
| s04 / 13 | [Where does the CLS query [1,1] come from?](https://nipunbatra.github.io/attention/vision1.html?present#s04/13/0) | `q-dot` |
| s04 / 14 | [How does P1 get the key [√2,0]?](https://nipunbatra.github.io/attention/vision1.html?present#s04/14/0) | `one-key-dot` |
| s04 / 15 | [Which features does this head compare?](https://nipunbatra.github.io/attention/vision1.html?present#s04/15/0) | `s03-query` |
| s04 / 16 | [Can we work out one score before filling the table?](https://nipunbatra.github.io/attention/vision1.html?present#s04/16/0) | `one-score` |
| s04 / 17 | [What does softmax do when the scores tie?](https://nipunbatra.github.io/attention/vision1.html?present#s04/17/0) | `softmax-relative` |
| s04 / 18 | [Repeat the dot product for every source](https://nipunbatra.github.io/attention/vision1.html?present#s04/18/0) | `s03-weights-step-1` |
| s04 / 19 | [Exponentiate the five scores](https://nipunbatra.github.io/attention/vision1.html?present#s04/19/0) | `s03-weights-step-2` |
| s04 / 20 | [Divide by one shared sum](https://nipunbatra.github.io/attention/vision1.html?present#s04/20/0) | `s03-weights` |
| s04 / 21 | [Where does the 0.229 beside P1 come from?](https://nipunbatra.github.io/attention/vision1.html?present#s04/21/0) | `weight-denominator` |
| s04 / 22 | [Put each value beside its weight](https://nipunbatra.github.io/attention/vision1.html?present#s04/22/0) | `s03-values-step-1` |
| s04 / 23 | [Multiply the value by its weight](https://nipunbatra.github.io/attention/vision1.html?present#s04/23/0) | `s03-values-step-2` |
| s04 / 24 | [Add the contributions to get one message](https://nipunbatra.github.io/attention/vision1.html?present#s04/24/0) | `s03-values` |
| s04 / 25 | [Can an empty patch still send something?](https://nipunbatra.github.io/attention/vision1.html?present#s04/25/0) | `one-value-product` |
| s04 / 26 | [Do equal weights send equal information?](https://nipunbatra.github.io/attention/vision1.html?present#s04/26/0) | `weight-message` |
| s04 / 27 | [What if the query cared only about ink?](https://nipunbatra.github.io/attention/vision1.html?present#s04/27/0) | `change-query` |
| s05 / 1 | [05 · More than one attention head](https://nipunbatra.github.io/attention/vision1.html?present#s05/1/0) | `vision-topic-05` |
| s05 / 2 | [Would a second way of reading the image help?](https://nipunbatra.github.io/attention/vision1.html?present#s05/2/0) | `heads-question` |
| s05 / 3 | [Head 2 compares ink and column](https://nipunbatra.github.io/attention/vision1.html?present#s05/3/0) | `s03-second-step-1` |
| s05 / 4 | [Give Head 2 its own softmax](https://nipunbatra.github.io/attention/vision1.html?present#s05/4/0) | `s03-second` |
| s05 / 5 | [What information does Head 2 send?](https://nipunbatra.github.io/attention/vision1.html?present#s05/5/0) | `s03-second-values-step-1` |
| s05 / 6 | [Calculate each Head 2 contribution](https://nipunbatra.github.io/attention/vision1.html?present#s05/6/0) | `s03-second-values-step-2` |
| s05 / 7 | [Add Head 2’s contributions](https://nipunbatra.github.io/attention/vision1.html?present#s05/7/0) | `s03-second-values` |
| s05 / 8 | [Do the patch rows get updated too?](https://nipunbatra.github.io/attention/vision1.html?present#s05/8/0) | `all-receivers` |
| s05 / 9 | [Keep both messages by putting them side by side](https://nipunbatra.github.io/attention/vision1.html?present#s05/9/0) | `s04-join-step-1` |
| s05 / 10 | [Use W_O to make the first update coordinate](https://nipunbatra.github.io/attention/vision1.html?present#s05/10/0) | `s04-join-step-2` |
| s05 / 11 | [Collect all four output coordinates](https://nipunbatra.github.io/attention/vision1.html?present#s05/11/0) | `s04-join` |
| s05 / 12 | [What do the other columns of W_O produce?](https://nipunbatra.github.io/attention/vision1.html?present#s05/12/0) | `other-output-coordinates` |
| s06 / 1 | [06 · From messages to an image label](https://nipunbatra.github.io/attention/vision1.html?present#s06/1/0) | `vision-topic-06` |
| s06 / 2 | [What if attention sent a zero message?](https://nipunbatra.github.io/attention/vision1.html?present#s06/2/0) | `residual-zero` |
| s06 / 3 | [Add the message to the starting CLS row](https://nipunbatra.github.io/attention/vision1.html?present#s06/3/0) | `s04-residual-step-1` |
| s06 / 4 | [Use the updated CLS row to score the two labels](https://nipunbatra.github.io/attention/vision1.html?present#s06/4/0) | `s04-residual` |
| s06 / 5 | [We used softmax twice. What changed?](https://nipunbatra.github.io/attention/vision1.html?present#s06/5/0) | `two-softmaxes` |
| s06 / 6 | [Turn the class scores into probabilities](https://nipunbatra.github.io/attention/vision1.html?present#s06/6/0) | `s04-probability-step-1` |
| s06 / 7 | [Use the known label to calculate the loss](https://nipunbatra.github.io/attention/vision1.html?present#s06/7/0) | `s04-probability` |
| s06 / 8 | [How much does a confident wrong answer cost?](https://nipunbatra.github.io/attention/vision1.html?present#s06/8/0) | `loss-comparison` |
| s06 / 9 | [Use the gradient to change the class bias](https://nipunbatra.github.io/attention/vision1.html?present#s06/9/0) | `one-update-step-1` |
| s06 / 10 | [Run the prediction again after that update](https://nipunbatra.github.io/attention/vision1.html?present#s06/10/0) | `one-update` |
| s06 / 11 | [Move the patches. Does the answer change?](https://nipunbatra.github.io/attention/vision1.html?present#s06/11/0) | `s04-experiment` |
| s06 / 12 | [Who teaches CLS what information to collect?](https://nipunbatra.github.io/attention/vision1.html?present#s06/12/0) | `cls-learns` |
| s06 / 13 | [Could we classify the image without CLS?](https://nipunbatra.github.io/attention/vision1.html?present#s06/13/0) | `pooling-example` |
| s06 / 14 | [So why use CLS in our ViT?](https://nipunbatra.github.io/attention/vision1.html?present#s06/14/0) | `readout-choice` |
| s07 / 1 | [07 · The complete Transformer block](https://nipunbatra.github.io/attention/vision1.html?present#s07/1/0) | `vision-topic-07` |
| s07 / 2 | [Put the familiar attention inside a full block](https://nipunbatra.github.io/attention/vision1.html?present#s07/2/0) | `s05-block` |
| s07 / 3 | [Which operation lets one patch borrow from another?](https://nipunbatra.github.io/attention/vision1.html?present#s07/3/0) | `mix-across-rows` |
| s07 / 4 | [What does the MLP change?](https://nipunbatra.github.io/attention/vision1.html?present#s07/4/0) | `mix-within-row` |
| s07 / 5 | [What average are we subtracting?](https://nipunbatra.github.io/attention/vision1.html?present#s07/5/0) | `ln-mean` |
| s07 / 6 | [How spread out is the centered row?](https://nipunbatra.github.io/attention/vision1.html?present#s07/6/0) | `ln-variance` |
| s07 / 7 | [What does LayerNorm do to one row?](https://nipunbatra.github.io/attention/vision1.html?present#s07/7/0) | `layernorm` |
| s07 / 8 | [How does the MLP make a wider row?](https://nipunbatra.github.io/attention/vision1.html?present#s07/8/0) | `mlp-first-linear` |
| s07 / 9 | [Apply GELU to each hidden coordinate](https://nipunbatra.github.io/attention/vision1.html?present#s07/9/0) | `mlp-row-step-1` |
| s07 / 10 | [How does the MLP return to the original width?](https://nipunbatra.github.io/attention/vision1.html?present#s07/10/0) | `mlp-second-linear` |
| s07 / 11 | [Add the MLP message to the starting row](https://nipunbatra.github.io/attention/vision1.html?present#s07/11/0) | `mlp-row` |
| s07 / 12 | [What does the next block get to see?](https://nipunbatra.github.io/attention/vision1.html?present#s07/12/0) | `depth` |
| s07 / 13 | [How many rows does the real photograph produce?](https://nipunbatra.github.io/attention/vision1.html?present#s07/13/0) | `s05-scale-step-1` |
| s07 / 14 | [How many blocks process those rows?](https://nipunbatra.github.io/attention/vision1.html?present#s07/14/0) | `s05-scale-step-2` |
| s07 / 15 | [What changed when we made the model larger?](https://nipunbatra.github.io/attention/vision1.html?present#s07/15/0) | `s05-scale` |
| s08 / 1 | [08 · Build the model in PyTorch](https://nipunbatra.github.io/attention/vision1.html?present#s08/1/0) | `vision-topic-08` |
| s08 / 2 | [These two 16s mean different things](https://nipunbatra.github.io/attention/vision1.html?present#s08/2/0) | `two-sixteens` |
| s08 / 3 | [How do image pixels become rows in code?](https://nipunbatra.github.io/attention/vision1.html?present#s08/3/0) | `code-patch` |
| s08 / 4 | [Why does our ViT code use Conv2d?](https://nipunbatra.github.io/attention/vision1.html?present#s08/4/0) | `conv-one-patch` |
| s08 / 5 | [How far should the filter move?](https://nipunbatra.github.io/attention/vision1.html?present#s08/5/0) | `conv-stride` |
| s08 / 6 | [Does this layer merely cut up the image?](https://nipunbatra.github.io/attention/vision1.html?present#s08/6/0) | `conv-trainable` |
| s08 / 7 | [Can local filters gather distant clues too?](https://nipunbatra.github.io/attention/vision1.html?present#s08/7/0) | `cnn-context` |
| s08 / 8 | [Can you match each line to our calculation?](https://nipunbatra.github.io/attention/vision1.html?present#s08/8/0) | `code-attention` |
| s08 / 9 | [Can a patch in image A read image B?](https://nipunbatra.github.io/attention/vision1.html?present#s08/9/0) | `batch-boundary` |
| s08 / 10 | [A tensor can have the right shape and the wrong meaning](https://nipunbatra.github.io/attention/vision1.html?present#s08/10/0) | `batch-axis` |
| s08 / 11 | [A quick check before training](https://nipunbatra.github.io/attention/vision1.html?present#s08/11/0) | `batch-check` |
| s08 / 12 | [What is the complete pre-LayerNorm block?](https://nipunbatra.github.io/attention/vision1.html?present#s08/12/0) | `code-block` |
| s08 / 13 | [How do we add one CLS row per image?](https://nipunbatra.github.io/attention/vision1.html?present#s08/13/0) | `code-add-cls` |
| s08 / 14 | [Where does location enter the code?](https://nipunbatra.github.io/attention/vision1.html?present#s08/14/0) | `code-add-pos` |
| s08 / 15 | [Which row reaches the classifier?](https://nipunbatra.github.io/attention/vision1.html?present#s08/15/0) | `code-cls-readout` |
| s08 / 16 | [How does the full model produce image logits?](https://nipunbatra.github.io/attention/vision1.html?present#s08/16/0) | `code-model` |
| s08 / 17 | [Which call makes this model learn?](https://nipunbatra.github.io/attention/vision1.html?present#s08/17/0) | `code-train` |
| s08 / 18 | [When are gradients computed?](https://nipunbatra.github.io/attention/vision1.html?present#s08/18/0) | `code-backward` |
| s08 / 19 | [Which line changes the weights?](https://nipunbatra.github.io/attention/vision1.html?present#s08/19/0) | `code-step` |
| s09 / 1 | [09 · Check what the model learned](https://nipunbatra.github.io/attention/vision1.html?present#s09/1/0) | `vision-topic-09` |
| s09 / 2 | [Will the model recognize a new noisy stripe?](https://nipunbatra.github.io/attention/vision1.html?present#s09/2/0) | `training-data` |
| s09 / 3 | [Which images are allowed to influence the weights?](https://nipunbatra.github.io/attention/vision1.html?present#s09/3/0) | `three-splits` |
| s09 / 4 | [Suppose the model gets this training image wrong](https://nipunbatra.github.io/attention/vision1.html?present#s09/4/0) | `training-one-image` |
| s09 / 5 | [What does one epoch mean in this experiment?](https://nipunbatra.github.io/attention/vision1.html?present#s09/5/0) | `one-epoch` |
| s09 / 6 | [Is the model improving on its training images?](https://nipunbatra.github.io/attention/vision1.html?present#s09/6/0) | `learning-curves-step-1` |
| s09 / 7 | [Does the improvement carry over to validation?](https://nipunbatra.github.io/attention/vision1.html?present#s09/7/0) | `learning-curves` |
| s09 / 8 | [Can training make up for missing positions?](https://nipunbatra.github.io/attention/vision1.html?present#s09/8/0) | `trained-position-control` |
| s10 / 1 | [10 · Return to the real photographs](https://nipunbatra.github.io/attention/vision1.html?present#s10/1/0) | `vision-topic-10` |
| s10 / 2 | [Which pixels are we giving the real model?](https://nipunbatra.github.io/attention/vision1.html?present#s10/2/0) | `real-input` |
| s10 / 3 | [What did the model call our dog?](https://nipunbatra.github.io/attention/vision1.html?present#s10/3/0) | `s06-answer` |
| s10 / 4 | [Does one correct photograph tell us the accuracy?](https://nipunbatra.github.io/attention/vision1.html?present#s10/4/0) | `one-photo-limit` |
| s10 / 5 | [What happens when we give it the cat?](https://nipunbatra.github.io/attention/vision1.html?present#s10/5/0) | `real-cat` |
| s10 / 6 | [Where did the checkpoint learn its visual features?](https://nipunbatra.github.io/attention/vision1.html?present#s10/6/0) | `three-phases-step-1` |
| s10 / 7 | [How would we adapt it to our own labels?](https://nipunbatra.github.io/attention/vision1.html?present#s10/7/0) | `three-phases-step-2` |
| s10 / 8 | [What happens when we classify a new photograph?](https://nipunbatra.github.io/attention/vision1.html?present#s10/8/0) | `three-phases` |
| s11 / 1 | [11 · Look inside the trained model](https://nipunbatra.github.io/attention/vision1.html?present#s11/1/0) | `vision-topic-11` |
| s11 / 2 | [What is one coloured square actually showing?](https://nipunbatra.github.io/attention/vision1.html?present#s11/2/0) | `read-attention-map` |
| s11 / 3 | [Do the heads read the same places?](https://nipunbatra.github.io/attention/vision1.html?present#s11/3/0) | `real-heads` |
| s11 / 4 | [Does CLS read differently in a later block?](https://nipunbatra.github.io/attention/vision1.html?present#s11/4/0) | `real-depth` |
| s11 / 5 | [What does this particular patch read?](https://nipunbatra.github.io/attention/vision1.html?present#s11/5/0) | `real-patch-query` |
| s11 / 6 | [What happens if we cover the top left?](https://nipunbatra.github.io/attention/vision1.html?present#s11/6/0) | `cover-1` |
| s11 / 7 | [What happens if we cover the top right?](https://nipunbatra.github.io/attention/vision1.html?present#s11/7/0) | `cover-2` |
| s11 / 8 | [What happens if we cover the bottom left?](https://nipunbatra.github.io/attention/vision1.html?present#s11/8/0) | `cover-3` |
| s11 / 9 | [What happens if we cover the bottom right?](https://nipunbatra.github.io/attention/vision1.html?present#s11/9/0) | `cover-4` |
| s11 / 10 | [Which covered region changed the answer most?](https://nipunbatra.github.io/attention/vision1.html?present#s11/10/0) | `occlusion` |
| s12 / 1 | [12 · The cost of smaller patches](https://nipunbatra.github.io/attention/vision1.html?present#s12/1/0) | `vision-topic-12` |
| s12 / 2 | [What changes when the patch size is halved?](https://nipunbatra.github.io/attention/vision1.html?present#s12/2/0) | `patch-cost` |
| s12 / 3 | [How much matching happens inside the tiny real model?](https://nipunbatra.github.io/attention/vision1.html?present#s12/3/0) | `real-work-count` |
| s12 / 4 | [What happens if we use a larger image?](https://nipunbatra.github.io/attention/vision1.html?present#s12/4/0) | `cost-control` |
| s13 / 1 | [13 · Your turn to work it out](https://nipunbatra.github.io/attention/vision1.html?present#s13/1/0) | `vision-topic-13` |
| s13 / 2 | [Your turn: work out the message](https://nipunbatra.github.io/attention/vision1.html?present#s13/2/0) | `exercise-message` |
| s13 / 3 | [Your turn: trace every important shape](https://nipunbatra.github.io/attention/vision1.html?present#s13/3/0) | `exercise-shapes` |
| s13 / 4 | [Did we move the image, or just reorder its rows?](https://nipunbatra.github.io/attention/vision1.html?present#s13/4/0) | `exercise-position` |
| s14 / 1 | [14 · What can we build next?](https://nipunbatra.github.io/attention/vision1.html?present#s14/1/0) | `vision-topic-14` |
| s14 / 2 | [What else could we ask the image model to do?](https://nipunbatra.github.io/attention/vision1.html?present#s14/2/0) | `next-vision` |
| s14 / 3 | [Can you talk us through the whole model?](https://nipunbatra.github.io/attention/vision1.html?present#s14/3/0) | `closing` |
