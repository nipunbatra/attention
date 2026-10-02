# Vision Transformer · optional labs and references — audit transcript

Companion PDF: [vision1-reference.pdf](vision1-reference.pdf) · **151 pages**.

Main lecture: 54 conceptual slides plus cover. The optional reference deck preserves the detailed material separately. Final reveals are captured in the PDF; progressive builds and interactions remain in HTML.

Diagram labels are listed in source order; use the PDF to judge spatial layout. Speaker notes contain the detail omitted from the projected slide.

## Page index

| PDF page | Route | Title |
|---:|---|---|
| 1 | Cover | Vision Transformer · optional labs and references |
| 2 | #s01/1 | Optional labs, worked examples and reference diagrams |
| 3 | #s01/2 | We already know how an encoder builds context |
| 4 | #s01/3 | A ViT is an encoder over image patches |
| 5 | #s01/4 | Text looks up a row; an image patch computes one |
| 6 | #s01/5 | Section 1 · The image classification task |
| 7 | #s01/6 | What does our animal dataset look like? |
| 8 | #s01/7 | What animal do you see? |
| 9 | #s01/8 | Our task today: classify the whole image |
| 10 | #s01/9 | Attention: what can the face tell this patch? |
| 11 | #s01/10 | Should every source contribute equally? |
| 12 | #s01/11 | What changes when the patch gets context? |
| 13 | #s01/12 | What could a query be in the image? |
| 14 | #s01/13 | What could a key be in the image? |
| 15 | #s01/14 | What information would a value send? |
| 16 | #s01/15 | How can we give this photograph to attention? |
| 17 | #s01/16 | The image classifier, drawn as one encoder pipeline |
| 18 | #s02/1 | Section 2 · From pixels to patch embeddings |
| 19 | #s02/2 | The whole route: photograph to prediction |
| 20 | #s02/3 | Where do the patch boundaries go? |
| 21 | #s02/4 | Read the RGB values of each pixel |
| 22 | #s02/5 | Flatten one channel at a time: R, then G, then B |
| 23 | #s02/6 | Do we apply an activation after the patch layer? |
| 24 | #s02/7 | What does “projection” mean here? |
| 25 | #s02/8 | 12 input numbers, 2 output numbers |
| 26 | #s02/9 | Follow the connections into output 1 |
| 27 | #s02/10 | Now follow the connections into output 2 |
| 28 | #s02/11 | These two numbers are the patch embedding |
| 29 | #s02/12 | Apply the very same layer to another patch |
| 30 | #s02/13 | Start with the same dog photograph |
| 31 | #s02/14 | Split the image into 16 × 16 patches |
| 32 | #s02/15 | Number the patches row by row |
| 33 | #s02/16 | Read the RGB values inside patch 63 |
| 34 | #s02/17 | Normalize those same RGB values |
| 35 | #s02/18 | Flatten patch 63 in the same order as the code |
| 36 | #s02/19 | Pass that row through the shared linear layer |
| 37 | #s02/20 | Read the 192 output features for patch 63 |
| 38 | #s02/21 | Pass patch 64 through the very same layer |
| 39 | #s02/22 | Stack the 196 output rows into C |
| 40 | #s03/1 | Section 3 · Prepare the rows, then classify the image |
| 41 | #s03/2 | Give each patch its location in the photograph |
| 42 | #s03/3 | Position is a learned lookup table |
| 43 | #s03/4 | Add position to these content rows |
| 44 | #s03/5 | The image loss teaches the position table |
| 45 | #s03/6 | Why add CLS? Give the classifier one image summary |
| 46 | #s03/7 | Add a summary row beside the dog’s patch rows |
| 47 | #s03/8 | What makes CLS an image summary? |
| 48 | #s03/9 | The same CLS start reads two different photographs |
| 49 | #s03/10 | Add the summary row: 196 + 1 = 197 |
| 50 | #s03/11 | Start with one Transformer block |
| 51 | #s03/12 | Self-attention: Q, K and V share the same input |
| 52 | #s03/13 | Make queries, keys and values from these rows |
| 53 | #s03/14 | The same input matrix feeds three learned projections |
| 54 | #s03/15 | One query–key comparison fills one matrix cell |
| 55 | #s03/16 | Follow the CLS row from scores to weights |
| 56 | #s03/17 | Turn each query’s 197 scores into 197 source weights |
| 57 | #s03/18 | Every image row can read every image row |
| 58 | #s03/19 | A message for CLS works like a message for bank |
| 59 | #s03/20 | The dog’s feature rows become value rows |
| 60 | #s03/21 | One weight scales all 64 features in its value row |
| 61 | #s03/22 | Each source contributes a weighted value row |
| 62 | #s03/23 | Add the contributions to make one CLS message |
| 63 | #s03/24 | Where does the CLS message go? |
| 64 | #s03/25 | Each query gets its own message |
| 65 | #s03/26 | From one completed head to three parallel heads |
| 66 | #s03/27 | What might different heads look for in this photograph? |
| 67 | #s03/28 | The same rows feed three sets of Q, K and V |
| 68 | #s03/29 | Each head repeats the complete attention calculation |
| 69 | #s03/30 | One CLS input produces three different messages |
| 70 | #s03/31 | Concatenate the three CLS messages |
| 71 | #s03/32 | Keep the embedding; add the context from attention |
| 72 | #s03/33 | The dog’s CLS keeps its input and gains context |
| 73 | #s03/34 | Open block 1: attention, then the MLP |
| 74 | #s03/35 | Open the MLP: 192 inputs, 768 hidden units, 192 outputs |
| 75 | #s03/36 | Add the MLP update to finish block 1 |
| 76 | #s03/37 | Pass the complete output of block 1 into block 2 |
| 77 | #s03/38 | What changes as the rows move through the blocks? |
| 78 | #s03/39 | Continue through the stack, then classify the image |
| 79 | #s03/40 | Select CLS from the final feature matrix |
| 80 | #s03/41 | Open the classifier: 192 features become 1,000 scores |
| 81 | #s03/42 | One class score is a weighted sum plus a bias |
| 82 | #s03/43 | Turn all 1,000 scores into class probabilities |
| 83 | #s03/44 | Two softmaxes, two different questions |
| 84 | #s03/45 | The same dog now has its final prediction |
| 85 | #s04/1 | Section 4 · Whole model walkthrough |
| 86 | #s04/2 | One shape trace from pixels to class scores |
| 87 | #s04/3 | Inside each block: mix, transform, keep the residual |
| 88 | #s04/4 | Backward: compute gradients, then update the model |
| 89 | #s05/1 | Section 5 · CNNs, ViTs and inductive bias |
| 90 | #s05/2 | Two ways to build an image representation |
| 91 | #s05/3 | A wider view, then one image label |
| 92 | #s05/4 | Inductive bias: a useful starting assumption |
| 93 | #s05/5 | Which is a sensible starting point? |
| 94 | #s06/1 | Implementation lab · optional |
| 95 | #s06/2 | Keep the same photograph and add the batch axis |
| 96 | #s06/3 | Our target: turn every patch into 192 features |
| 97 | #s06/4 | See the patch projection as a layer of neurons |
| 98 | #s06/5 | Implementation 1: extract patches, then use Linear |
| 99 | #s06/6 | Implementation 2: the same projection with Conv2d |
| 100 | #s06/7 | Reshape the weights; keep the same parameter count |
| 101 | #s06/8 | Same pixel × same weight, in both implementations |
| 102 | #s06/9 | Verify it on both photographs: the features match |
| 103 | #s06/10 | Why package patch projection as Conv2d? |
| 104 | #s06/11 | First turn the feature grid into patch rows |
| 105 | #s06/12 | Then prepend CLS and add position |
| 106 | #s06/13 | Make queries, keys and values for three heads |
| 107 | #s06/14 | Compute one message for every query in every head |
| 108 | #s06/15 | Join the head messages and project back to 192 |
| 109 | #s06/16 | Build the layers inside one Transformer block |
| 110 | #s06/17 | Use the two residual paths in order |
| 111 | #s06/18 | Create twelve blocks with separate learned parameters |
| 112 | #s06/19 | Run the stack, then read the final CLS |
| 113 | #s06/20 | Connect the loss diagram to one training step |
| 114 | #s07/1 | Optional extensions |
| 115 | #s07/2 | Section 7 · Adapt and evaluate the classifier |
| 116 | #s07/3 | What was this model trained to predict? |
| 117 | #s07/4 | Same photographs, a different label vocabulary |
| 118 | #s07/5 | What if our users supply sketches? |
| 119 | #s07/6 | Replace the ImageNet head with our two-class head |
| 120 | #s07/7 | Freeze the encoder; train the new head |
| 121 | #s07/8 | Next option: fine-tune the last block as well |
| 122 | #s07/9 | One batch follows the same forward and backward paths |
| 123 | #s07/10 | Three stages: learn, adapt, then predict |
| 124 | #s07/11 | What happens when we classify a new photograph? |
| 125 | #s07/12 | How would we check whether the classifier learned? |
| 126 | #s08/1 | Section 8 · Return to the real photographs |
| 127 | #s08/2 | Which pixels are we giving the real model? |
| 128 | #s08/3 | What did the model call our dog? |
| 129 | #s08/4 | Does one correct photograph tell us the accuracy? |
| 130 | #s08/5 | What happens when we give it the cat? |
| 131 | #s09/1 | Section 9 · Look inside the trained model |
| 132 | #s09/2 | Similar patch features can connect distant image regions |
| 133 | #s09/3 | Keep the query fixed; change only the attention head |
| 134 | #s09/4 | CLS gathers a message for the image summary |
| 135 | #s09/5 | “Cover” means replace these pixels with gray |
| 136 | #s09/6 | Run the covered image through the same trained model |
| 137 | #s09/7 | Four covers, four new forward passes |
| 138 | #s09/8 | Use smaller covers to ask a more local question |
| 139 | #s09/9 | Smaller covers reveal local sensitivity |
| 140 | #s10/1 | Section 10 · The cost of smaller patches |
| 141 | #s10/2 | What changes when the patch size is halved? |
| 142 | #s10/3 | How much matching happens inside the tiny real model? |
| 143 | #s10/4 | What happens if we use a larger image? |
| 144 | #s10/5 | The whole ViT: pixels → context → one label |
| 145 | #s10/6 | Four ideas to carry forward |
| 146 | #s10/7 | Our classifier stores a vector for each known label |
| 147 | #s10/8 | What if a class vector could come from language? |
| 148 | #s11/1 | The whole Vision Transformer in one figure |
| 149 | #s11/2 | To images: An Image is Worth 16 × 16 Words |
| 150 | #s11/3 | Create 192 trainable numbers for CLS |
| 151 | #s11/4 | The image label teaches the starting CLS numbers |

## Transcript

### Page 1 — Vision Transformer · optional labs and references

Detailed arithmetic, PyTorch implementation, transfer learning and interpretation.

### Page 2 — Optional labs, worked examples and reference diagrams

```text
OPTIONAL REFERENCE DECK
Open the boxes when you need the details.
The main 54-slide lecture is complete without this material.
```

**Caption:** The main 54-slide lecture is complete without this material.

**Speaker notes**

Optional labs, worked examples and reference diagrams
The main 54-slide lecture is complete without this material.

### Page 3 — We already know how an encoder builds context

```text
ENCODER
Encode the input
DECODER
Predict the next token
ENCODER-DECODER
Generate from a source
Encoder
one vector per token
labels · retrieval · readouts
Decoder
next
append → repeat
chat · code · continuation
Encoder
target prefix
source K,V
Decoder
next
source + target prefix
translation · summarization
Today we reuse the encoder on the left: image patches become tokens, and the final image representation predicts a label.
```

**Caption:** Today we reuse the encoder on the left: image patches become tokens, and the final image representation predicts a label.

**Speaker notes**

Which architecture reads a complete supplied input and returns contextual representations?
The encoder, shown on the left. The full attention square lets every token read every token. The decoder uses causal attention and repeats next-token prediction. The encoder–decoder adds a source pathway into the target decoder through cross-attention. For ViT, replace text tokens with image patches, then use a classification head.

### Page 4 — A ViT is an encoder over image patches

```text
TEXT
CLS
Raghav
goes
to
school
Encoder blocks
whole-input attention
Read final CLS
→ classifier
IMAGE
CLS
P1
P2
…
P196
Encoder blocks
whole-input attention
Read final CLS
→ classifier
Same encoder idea. Different tokens.
Blue: vision states     Purple: language states     Amber: learned CLS / position
Attention builds context in either sequence. The input representation and the task head determine how we use the encoder.
```

**Caption:** Attention builds context in either sequence. The input representation and the task head determine how we use the encoder.

**Speaker notes**

What changes when we replace the text sequence with patch tokens?
The input embedding and task head change. The encoder machinery remains: attention, MLPs, residuals and normalization. Both pictured classifiers read final CLS.

### Page 5 — Text looks up a row; an image patch computes one

```text
TEXT
IMAGE
“bank” → token ID
Embedding table
look up a learned row
token content row
D features
RGB patch: 16 × 16 × 3
flatten → 768 values
Shared Linear(768, D)
compute a learned projection
patch content row
D features
Add learned position → send the rows into the encoder
Both routes produce a row of D features. Text learns a vocabulary table; vision learns a shared pixel projection. Our image checkpoint chooses D = 192.
```

**Caption:** Both routes produce a row of D features. Text learns a vocabulary table; vision learns a shared pixel projection. Our image checkpoint chooses D = 192.

**Speaker notes**

Does an image patch need a vocabulary ID?
No. Its pixel values go through a learned affine layer. Once position is added, the encoder works with feature rows in either case.

### Page 6 — Section 1 · The image classification task

```text
SECTION
01
The image classification task
A text encoder built context from supplied tokens.
What should a model predict from a photograph?
Labeled photos
The image task
Useful clues
Start with the dataset, choose an image label, then ask which parts of the photograph help us decide.
```

**Caption:** Start with the dataset, choose an image label, then ask which parts of the photograph help us decide.

**Speaker notes**

What should a model predict from a photograph?
Pause at the section question. Connect the previous result to the three steps, then advance to the concrete example.

### Page 7 — What does our animal dataset look like?

```text
dog
Newfoundland
dog
Pug
dog
Great Pyrenees
cat
Persian
cat
Sphynx
cat
Birman
Oxford-IIIT Pet: six examples. Each photo has a species label and a breed label.
```

**Caption:** Oxford-IIIT Pet: six examples. Each photo has a species label and a breed label.

**Speaker notes**

What changes across photos that share the same dog or cat label?
Point to pose, coat and background. Read the species label first, then the breed beneath it.

### Page 8 — What animal do you see?

```text
One photograph.
One image label.
What clues did you use?
Which parts of the photograph helped you decide?
```

**Caption:** Which parts of the photograph helped you decide?

**Speaker notes**

What animal is this, and which parts made you decide?
Point to the face, fur, and silhouette before mentioning an architecture.

### Page 9 — Our task today: classify the whole image

```text
image summary
dog
cat
class scores
For the rest of this lecture, we observe the whole photograph and predict one class label.
```

**Caption:** For the rest of this lecture, we observe the whole photograph and predict one class label.

**Speaker notes**

Are we trying to guess a missing patch in this task?
Point to the full observed image and the two possible class labels.

### Page 10 — Attention: what can the face tell this patch?

```text
receiver: dark patch
fur, shadow, background?
patch numbers
source: face
face information
send clues
Recall Q, K, V from text
Q (receiver): what to look for
K (source): what can match
V (source): information to send
Next: the image versions, then a numerical example.
LOCAL ROLES
Q / receiver
K / source
V / message
Queries and keys determine weights; values supply the information to combine. The dark patch is our receiver, and the face patch is one possible source.
```

**Caption:** Queries and keys determine weights; values supply the information to combine. The dark patch is our receiver, and the face patch is one possible source.

**Speaker notes**

Could nearby face information help us interpret the dark texture as animal fur?
Name attention and recall Q, K, V from text. Start at the dark receiver, which makes a query. Reveal the face source, which supplies a key and a value. Trace the information arrow into the receiver’s numbers. Preview the coming image-vector diagrams and numerical exercise.

### Page 11 — Should every source contribute equally?

```text
receiver: dark patch
fur, shadow, background?
patch numbers
source: face
face information
source: branches
branch information
larger share
smaller share
Illustrative shares
LOCAL ROLES
Q / receiver
K / source
V / message
Here, face clues could help more than branches. Attention weights control how much each source contributes to the message.
```

**Caption:** Here, face clues could help more than branches. Attention weights control how much each source contributes to the message.

**Speaker notes**

For this dark receiver, which source would you expect to be more useful: the face or the branches?
Keep the receiver fixed. Reveal the second source, then compare the thick and thin arrows ending on the same receiver row.

### Page 12 — What changes when the patch gets context?

```text
receiver: dark patch
same pixels
current patch row
face patch
face value
branches
branch value
× its weight
× its weight
weighted context
sum values + project
+
updated patch row
Two sources shown; all image rows can contribute.
LOCAL ROLES
Q / receiver
K / source
V / message
The face and branches contribute value vectors. Attention weights mix those vectors, and an output projection forms the message added to the dark patch’s row. The pixels stay fixed.
```

**Caption:** The face and branches contribute value vectors. Attention weights mix those vectors, and an output projection forms the message added to the dark patch’s row. The pixels stay fixed.

**Speaker notes**

Are we changing the photo, assigning a fur label to this crop, or updating its numerical representation?
Start at the dark patch and its current row. Reveal the face and branches from the preceding slide, then follow their value vectors through weighting, summation and projection. Finally follow both the context message and original row into the plus sign.

### Page 13 — What could a query be in the image?

```text
Text · receiver
bank
Image · receiver
e_bank
W_Q
q_bank
e_dark
W_Q
q_dark
Which context could help bank?
Could this dark texture belong to the animal?
LOCAL ROLES
Q / receiver
K / source
V / message
The row being updated makes a query. Its learned projection determines what kinds of source information it can match.
```

**Caption:** The row being updated makes a query. Its learned projection determines what kinds of source information it can match.

**Speaker notes**

Which patch should make the query if we want to update the dark patch?
Follow the receiver row through W_Q in each case. Read the questions as intuition for a learned vector, not words typed by a user.

### Page 14 — What could a key be in the image?

```text
Text · source
river
Image · source
e_river
W_K
k_river
e_face
W_K
k_face
q_bank · k_river → score
q_dark · k_face → score
LOCAL ROLES
Q / receiver
K / source
V / message
Each source row makes a key. Comparing the receiver’s query with source keys gives the scores used to choose attention weights.
```

**Caption:** Each source row makes a key. Comparing the receiver’s query with source keys gives the scores used to choose attention weights.

**Speaker notes**

How could the dark patch compare a face patch with a branches patch?
Keep the query on the receiver. Point to river and the face as sources, each with its own key.

### Page 15 — What information would a value send?

```text
Text · source
river
Image · source
e_river
W_V
v_river
e_face
W_V
v_face
weight × v_river
weight × v_face
LOCAL ROLES
Q / receiver
K / source
V / message
Q and K choose the weights. V supplies the information to combine using those weights; the combined message updates the receiving representation.
```

**Caption:** Q and K choose the weights. V supplies the information to combine using those weights; the combined message updates the receiving representation.

**Speaker notes**

Once a source receives a weight, which vector actually contributes to the message?
Reuse the same source crops from the key slide. Replace W_K with W_V, then multiply each value by its weight.

### Page 16 — How can we give this photograph to attention?

```text
whole photograph
numbers for a patch
numbers for a patch
numbers for a patch
⋮
one row per patch
attention
share information
Attention works on rows of numbers. Next, we choose image patches and turn their pixels into those rows.
```

**Caption:** Attention works on rows of numbers. Next, we choose image patches and turn their pixels into those rows.

**Speaker notes**

In the text lessons, what did attention receive as its input?
Keep the photograph visible. Reveal its pieces, one row per piece, then the familiar attention operation.

### Page 17 — The image classifier, drawn as one encoder pipeline

```text
224 × 224 RGB
Shared patch projection
196 rows · 192 features
+ learned CLS
+ learned position
Encoder
× 12
Final LayerNorm
read CLS: 192 features
Linear class head
1,000 class scores
ViT: Dosovitskiy et al., 2020 · dimensions shown for our ViT-Tiny checkpoint
Patches become feature rows. Learned CLS and position prepare the input. Encoder blocks build context; the final CLS feeds the classifier.
```

**Caption:** Patches become feature rows. Learned CLS and position prepare the input. Encoder blocks build context; the final CLS feeds the classifier.

**Speaker notes**

Where does the image become a sequence?
The shared patch projection creates 196 rows. CLS adds row 197; position adds location information without changing the shape.

### Page 18 — Section 2 · From pixels to patch embeddings

```text
SECTION
02
From pixels to patch embeddings
Attention updates a row of numbers for each token.
How do we make those rows from an image?
RGB pixels
Shared linear layer
Patch embeddings
Read one small patch, calculate its embedding, then scale the same operation to a real photograph.
```

**Caption:** Read one small patch, calculate its embedding, then scale the same operation to a real photograph.

**Speaker notes**

How do we make those rows from an image?
Pause at the section question. Connect the previous result to the three steps, then advance to the concrete example.

### Page 19 — The whole route: photograph to prediction

```text
Image
one
photograph
HERE
Patches
small
image crops
HERE
Projection
pixels into
features
HERE
Prepare rows
location +
summary
Attention
share
information
MLP
transform
features
Read summary
one image
vector
Class scores
score each
image class
Inside one block
softmax → label
First: make the input rows
pixels → patch features
Then: build context
patches share information
Finally: make a prediction
image summary → label
The photograph stays fixed. The rows of features change as we follow the arrows.
First turn pixels into patch features. Prepare those rows, let them exchange information, then read one image summary to score the classes. We will open each box as we reach it.
```

**Caption:** First turn pixels into patch features. Prepare those rows, let them exchange information, then read one image summary to score the classes. We will open each box as we reach it.

**Speaker notes**

Where do pixels become features, and where do features become class scores?
Trace left to right. For now, call the extra row an image summary. Explain its name and how it works in the dedicated detour before attention.

### Page 20 — Where do the patch boundaries go?

```text
P6
P7
P10
P11
The grid cuts through the photograph before the model knows where the dog is.
```

**Caption:** The grid cuts through the photograph before the model knows where the dog is.

**Speaker notes**

Does each patch contain one whole object?
Reveal the grid, then match each enlarged crop back to its location.

### Page 21 — Read the RGB values of each pixel

```text
A
B
C
D
2 × 2 × 3
pixel
R
G
B
A
1
0
0
B
0
1
0
C
0
0
1
D
1
1
1
4 pixels × 3 channels = 12 values
For this calculation, divide RGB values by 255. Read A, B, C, D: top-left, top-right, bottom-left, bottom-right.
```

**Caption:** For this calculation, divide RGB values by 255. Read A, B, C, D: top-left, top-right, bottom-left, bottom-right.

**Speaker notes**

How many numbers does pixel A contribute?
Point to the red pixel, read its RGB triple, then reveal one pixel at a time.

### Page 22 — Flatten one channel at a time: R, then G, then B

```text
4 pixels × 3 channels = 12 values
A
B
C
D
one RGB patch
R values
A.R
1
1
B.R
0
2
C.R
0
3
D.R
1
4
G values
A.G
0
5
B.G
1
6
C.G
0
7
D.G
1
8
B values
A.B
0
9
B.B
0
10
C.B
1
11
D.B
1
12
A → B → C → D inside each channel
One row x₁: inputs and weights use this same order.
We use all R values, then G, then B: 12 values for this patch. Pixel-by-pixel RGB also works if the weight columns follow that order. Our PyTorch code uses the channel-first order.
```

**Caption:** We use all R values, then G, then B: 12 values for this patch. Pixel-by-pixel RGB also works if the weight columns follow that order. Our PyTorch code uses the channel-first order.

**Speaker notes**

Where is the green value of pixel B in this row?
It is entry 6 (one-based): four red entries, then green A and green B. Within each channel, read A, B, C, D in image row order.

### Page 23 — Do we apply an activation after the patch layer?

```text
Patch embedding · at the image input
12 pixel values
nn.Linear(12, 2)
2 coordinates
c₁ = x₁W + b. Bias makes this affine; no ReLU or GELU follows.
Later, after attention · the block MLP
one row
Linear(D, H)
GELU
Linear(H, D)
D features
H features
H features
D features
Our patch embedding uses one affine layer. The later block MLP puts GELU between two linear layers. D is the embedding width; H is the MLP’s hidden width.
```

**Caption:** Our patch embedding uses one affine layer. The later block MLP puts GELU between two linear layers. D is the embedding width; H is the MLP’s hidden width.

**Speaker notes**

Which part of these two paths applies an activation function?
Follow the direct pixel-to-embedding path first. Then reveal the separate block MLP and point to GELU between its two linear layers.

### Page 24 — What does “projection” mean here?

```text
Input: x₁
Output: c₁
12 values
nn.Linear(12, 2)
2 coordinates
(1, 12)
(1, 2)
c₁ = x₁ W + b
(1, 2) = (1, 12) × (12, 2) + (1, 2)
A projection forms weighted sums of the pixel values and adds a bias. This one linear layer turns 12 inputs into a 2-coordinate patch embedding.
```

**Caption:** A projection forms weighted sums of the pixel values and adds a bias. This one linear layer turns 12 inputs into a 2-coordinate patch embedding.

**Speaker notes**

How many weighted sums do we need to produce two output coordinates?
Match 12 to in_features and 2 to out_features, then follow the matrix dimensions.

### Page 25 — 12 input numbers, 2 output numbers

```text
12 inputs
2 outputs
nn.Linear(12, 2)
A.R
1
B.R
0
C.R
0
D.R
1
A.G
0
B.G
1
C.G
0
D.G
1
A.B
0
B.B
0
C.B
1
D.B
1
y₁
y₂
One patch in.
Two features out.
Each line has a weight.
Each output adds a bias.
c₁ = [y₁, y₂]
24 weights + 2 biases
Each input node holds one RGB value. Both outputs read all 12 inputs. Together, y₁ and y₂ form the embedding for this one patch.
```

**Caption:** Each input node holds one RGB value. Both outputs read all 12 inputs. Together, y₁ and y₂ form the embedding for this one patch.

**Speaker notes**

How many connections enter each output node?
Count the twelve actual pixel values. Follow their connections into each of the two outputs, then reveal the weights and biases.

### Page 26 — Follow the connections into output 1

```text
12 inputs
2 outputs
nn.Linear(12, 2)
+1
+1
+1
+1
A.R
1
B.R
0
C.R
0
D.R
1
A.G
0
B.G
1
C.G
0
D.G
1
A.B
0
B.B
0
C.B
1
D.B
1
y₁
y₂
Output 1
A.R + B.R
+ C.R + D.R
Other incoming weights: 0
1 + 0 + 0 + 1
+ 0.5 (bias)
= 2.5
Multiply each input by its connection weight, add the contributions, then add the bias. The highlighted connections show the nonzero weights for this output.
```

**Caption:** Multiply each input by its connection weight, add the contributions, then add the bias. The highlighted connections show the nonzero weights for this output.

**Speaker notes**

What does this output receive before we add its bias?
Reveal the highlighted edges and their weights. Read the connected input values, compute the sum, then add the bias and reveal the answer.

### Page 27 — Now follow the connections into output 2

```text
12 inputs
2 outputs
nn.Linear(12, 2)
+1
+1
−1
−1
A.R
1
B.R
0
C.R
0
D.R
1
A.G
0
B.G
1
C.G
0
D.G
1
A.B
0
B.B
0
C.B
1
D.B
1
y₁
y₂
Output 2
A.G + B.G
− C.G − D.G
Other incoming weights: 0
0 + 1 − 0 − 1
− 0.5 (bias)
= −0.5
Multiply each input by its connection weight, add the contributions, then add the bias. The highlighted connections show the nonzero weights for this output.
```

**Caption:** Multiply each input by its connection weight, add the contributions, then add the bias. The highlighted connections show the nonzero weights for this output.

**Speaker notes**

What does this output receive before we add its bias?
Reveal the highlighted edges and their weights. Read the connected input values, compute the sum, then add the bias and reveal the answer.

### Page 28 — These two numbers are the patch embedding

```text
A
B
C
D
nn.Linear(12, 2)
[2.5, −0.5]
x₁: (1, 12)
c₁: (1, 2)
No activation: −0.5 stays −0.5.
Later block MLP: Linear → GELU → Linear
c₁ represents the content of patch 1. Its two coordinates are features for the model; the image classifier comes later.
```

**Caption:** c₁ represents the content of patch 1. Its two coordinates are features for the model; the image classifier comes later.

**Speaker notes**

Should the negative output become zero?
Collect the two calculated coordinates and point to the absence of an activation after the layer.

### Page 29 — Apply the very same layer to another patch

```text
Create once: proj = nn.Linear(12, 2)
A
B
C
D
P1
x1 · (1, 12)
[2.5, -0.5]
A
B
C
D
P2
x2 · (1, 12)
[0.5, -2.5]
same W
same b
C = proj(X):   (2, 12) → (2, 2)
The pixel rows differ. Both use the same 26 parameters, so we compute both embeddings with one call: C = proj(X).
```

**Caption:** The pixel rows differ. Both use the same 26 parameters, so we compute both embeddings with one call: C = proj(X).

**Speaker notes**

Do we create a new nn.Linear layer when we move to patch 2?
Reveal the second patch. Follow both arrows through the single weight-and-bias box, then read the stacked input and output shapes.

### Page 30 — Start with the same dog photograph

```text
The same dog photograph
Input image
224 × 224 × 3
224 rows of pixels
224 columns of pixels
3 values per pixel: R, G, B
Next: cut this image into equal-sized patches.
We now carry this one image through the real model’s input pipeline. It has already been resized and cropped to 224 × 224 RGB pixels.
```

**Caption:** We now carry this one image through the real model’s input pipeline. It has already been resized and cropped to 224 × 224 RGB pixels.

**Speaker notes**

What do the three image dimensions count?
Keep the photograph fixed. Identify rows, columns and RGB channels before introducing any patch or embedding dimension.

### Page 31 — Split the image into 16 × 16 patches

```text
0
0
224
224
x (pixels)
y (pixels)
P63 · enlarged
One patch
16 pixels wide
16 pixels high
3 RGB channels
Next: number the pieces, one image row at a time.
The x-axis runs right; the y-axis runs down. Both span 224 pixels. Cut every 16 pixels along each axis. Each piece keeps its RGB values; we will count the pieces next.
```

**Caption:** The x-axis runs right; the y-axis runs down. Both span 224 pixels. Cut every 16 pixels along each axis. Each piece keeps its RGB values; we will count the pieces next.

**Speaker notes**

How many 16-pixel-wide pieces fit along each image axis?
Point to the x-axis arrow, then the y-axis arrow and their endpoints. Follow the highlighted image region into the enlarged 16×16 patch. Leave the count for the next slide.

### Page 32 — Number the patches row by row

```text
column 1
column 2
last column
row 1
…
P1
P2
P14
row 2
…
P15
P16
P28
last row
…
P183
P184
P196
⋮
⋮
⋮
Use the image axes:
columns: 224 ÷ 16
rows: 224 ÷ 16
= 14
= 14
total = columns × rows
14 × 14 = 196 patches
shape: 196 × 16 × 16 × 3
Number left to right, then continue on the next row. Use the image and patch sizes to calculate the row length and total before revealing the answers.
```

**Caption:** Number left to right, then continue on the next row. Use the image and patch sizes to calculate the row length and total before revealing the answers.

**Speaker notes**

What is the last patch number in the first row? What starts the second row, and what is the final patch number?
Start with P1 and P2. Ask students to divide each 224-pixel extent by 16, then reveal P14 and the second row. Multiply the two counts before revealing P196 and the full array shape.

### Page 33 — Read the RGB values inside patch 63

```text
P63 · enlarged
16 × 16 = 256 pixels
Pixel in P63
R
G
B
first
16
17
12
second
41
42
37
256 pixels × 3 RGB values
= 768 numbers in this patch
P63 contains 256 pixels. Its first pixel is RGB [16, 17, 12]; its next pixel is [41, 42, 37]. Three values per pixel give 768 numbers in this same patch.
```

**Caption:** P63 contains 256 pixels. Its first pixel is RGB [16, 17, 12]; its next pixel is [41, 42, 37]. Three values per pixel give 768 numbers in this same patch.

**Speaker notes**

How many input numbers does each pixel contribute?
Use the two outlined pixels in the real crop. Read each RGB triple, then count 256 triples rather than introducing 768 without its source.

### Page 34 — Normalize those same RGB values

```text
Same P63 pixels; apply the checkpoint’s normalization.
Pixel
8-bit RGB
Normalized RGB
1
[16, 17, 12]
[−0.875, −0.867, −0.906]
2
[41, 42, 37]
[−0.678, −0.671, −0.710]
Each channel: (value / 255 − 0.5) / 0.5
The checkpoint rescales every channel using the same formula. For the first red value, (16 / 255 − 0.5) / 0.5 ≈ −0.875. P63 still contains 768 values.
```

**Caption:** The checkpoint rescales every channel using the same formula. For the first red value, (16 / 255 − 0.5) / 0.5 ≈ −0.875. P63 still contains 768 values.

**Speaker notes**

Where does the first negative number come from?
Carry the first RGB triple from the previous slide across the arrow. Work out the red channel before revealing the second pixel.

### Page 35 — Flatten patch 63 in the same order as the code

```text
P63
All R values → all G values → all B values
R (256)
[−0.875, −0.678, −0.561, …]
G (256)
[−0.867, −0.671, −0.553, …]
B (256)
[−0.906, −0.710, −0.592, …]
x₆₃: 1 × 768
Within each channel: left to right, top to bottom.
Concatenate 256 red values, 256 green values and 256 blue values. This gives one 768-number patch row. F.unfold and the reshaped Conv2d weights use this exact ordering.
```

**Caption:** Concatenate 256 red values, 256 green values and 256 blue values. This gives one 768-number patch row. F.unfold and the reshaped Conv2d weights use this exact ordering.

**Speaker notes**

Where does the first green value appear?
At entry 257 (one-based), after all 256 red values. The first pixel’s RGB values occupy entries 1, 257 and 513.

### Page 36 — Pass that row through the shared linear layer

```text
P63 is now x₆₃: a row of 768 normalized pixel values.
x₆₃ · 1 × 768
nn.Linear(768, 192)
c₆₃ · 1 × 192
768 inputs
W_patch: 768 × 192
bias: 192 values
192 outputs
Each output = a weighted sum of the 768 inputs + its bias. No activation.
This layer reads 768 values and computes 192 output features. Its learned weights and biases are shared by every patch. One patch row enters; one embedding row leaves.
```

**Caption:** This layer reads 768 values and computes 192 output features. Its learned weights and biases are shared by every patch. One patch row enters; one embedding row leaves.

**Speaker notes**

How many weighted sums does this layer compute for P63?
Carry x63 from the previous slide into the layer. Follow its row count and feature width separately, then identify the weight and bias shapes.

### Page 37 — Read the 192 output features for patch 63

```text
nn.Linear(768, 192)
c₆₃ · 192 output features
feature 1
−0.852
feature 2
1.339
feature 3
0.504
feature 192
−1.199
…
c₆₃ = [−0.852, 1.339, 0.504, …, −1.199]      shape: 1 × 192
The layer produces c₆₃, the content embedding for P63. These are actual outputs from the pretrained model, rounded here. The first feature is −0.852 and the last is −1.199.
```

**Caption:** The layer produces c₆₃, the content embedding for P63. These are actual outputs from the pretrained model, rounded here. The first feature is −0.852 and the last is −1.199.

**Speaker notes**

Which patch do all 192 numbers describe?
Trace the same crop through the same layer, then read individual output coordinates before collecting them into c63.

### Page 38 — Pass patch 64 through the very same layer

```text
patch
input row · 1 × 768
output row · 1 × 192
P63
[−0.875, −0.678, …]
[−0.852, 1.339, …]
P64
[−0.286, −0.271, …]
[0.161, 0.988, …]
same layer
768 → 192
same W and b
P63 and P64 produce different features using the same learned parameters.
P64 is the patch immediately to the right of P63. Its different pixels give a different embedding. Both patches use the same weights and biases, and both produce 192 output features.
```

**Caption:** P64 is the patch immediately to the right of P63. Its different pixels give a different embedding. Both patches use the same weights and biases, and both produce 192 output features.

**Speaker notes**

Do we create new projection weights for P64?
Keep P63’s path visible while revealing P64 underneath it. Point to the one shared layer and compare their measured input and output values.

### Page 39 — Stack the 196 output rows into C

```text
One content embedding for every image patch
P1
[−0.205, 0.082, −0.205, …]
P63
[−0.852, 1.339, 0.504, …]
P64
[0.161, 0.988, −1.148, …]
P196
[−1.309, 1.327, −2.234, …]
⋮
⋮
X: 196 × 768
same Linear(768, 192)
C: 196 × 192
196 patch rows; 192 features each
Repeat the same projection for all 196 patches and keep their order. The input matrix X is 196 × 768. The output matrix C is 196 × 192: one content embedding per image patch.
```

**Caption:** Repeat the same projection for all 196 patches and keep their order. The input matrix X is 196 × 768. The output matrix C is 196 × 192: one content embedding per image patch.

**Speaker notes**

Which axis changes when we apply the shared projection to all patches?
Match the illustrated crops to their measured output rows. Then carry the unchanged row count 196 from X through the layer to C.

### Page 40 — Section 3 · Prepare the rows, then classify the image

```text
SECTION
03
Prepare the rows, then classify
the image
Our photograph has become 196 rows of patch features.
Where is each patch? How do we get one image
summary?
Add location
Introduce CLS
Resume the forward pass
Keep the photograph fixed. Give each patch its location, introduce the summary row, then follow the complete input through the classifier.
```

**Caption:** Keep the photograph fixed. Give each patch its location, introduce the summary row, then follow the complete input through the classifier.

**Speaker notes**

Where is each patch? How do we get one image summary?
Pause at the section question. Connect the previous result to the three steps, then advance to the concrete example.

### Page 41 — Give each patch its location in the photograph

```text
Keep the photograph exactly as it is.
column 7
row 5
P63 · one of 196 patches
Content c₆₃: features from this crop
Already computed by Linear(768, 192)
Position p₆₃: a vector for this grid slot
Row 5, column 7 → patch number 63
The shared pixel projection was never given
the row or column. Add that information next.
Recognizing a face depends on how its parts are arranged. c₆₃ describes this crop; p₆₃ tells attention where it belongs. Add them so the model can use both appearance and layout.
```

**Caption:** Recognizing a face depends on how its parts are arranged. c₆₃ describes this crop; p₆₃ tells attention where it belongs. Add them so the model can use both appearance and layout.

**Speaker notes**

Which part of the shared patch projection received row 5, column 7?
None: the projection received only pixel values. Point to the location on the unchanged photograph, then introduce its position vector.

### Page 42 — Position is a learned lookup table

```text
P63: row 5, column 7
P: 196 grid slots × 192 coordinates
P1
[ … 192 coordinates … ]
P2
[ … 192 coordinates … ]
⋮
⋮
P63
[−0.815, −0.083, …]
⋮
⋮
P196
[ … 192 coordinates … ]
Slot 63 selects p₆₃ for every image.
196 × 192 = 37,632 learned numbers
Compare: patch projection has 147,648 parameters.
Each grid slot has a trainable 192-number row, shared across images. Slot 63 always selects p₆₃; its pixel-derived content changes from photo to photo. The patch-position table has 37,632 parameters.
```

**Caption:** Each grid slot has a trainable 192-number row, shared across images. Slot 63 always selects p₆₃; its pixel-derived content changes from photo to photo. The patch-position table has 37,632 parameters.

**Speaker notes**

Would a second photograph require a second table of position vectors?
No: point from row 5, column 7 into the same stored row. Read the parameter count and compare it with the patch projection already calculated. The displayed p63 entries come from the saved checkpoint.

### Page 43 — Add position to these content rows

```text
P63
We already computed c₆₃ using the patch projection.
Add its position row; repeat for every image patch.
content from pixels
c₆₃ · 1 × 192
[−0.852, 1.339, …]
row 63 of the learned table
p₆₃ · 1 × 192
[−0.815, −0.083, …]
+
input to the block
e₆₃ · 1 × 192
[−1.666, 1.256, …]
=
All patch rows:  C (196 × 192) + P (196 × 192) = E (196 × 192)
c₆₃ comes from the patch pixels. p₆₃ is learned for its grid location. Add matching coordinates to get e₆₃. All three rows have 192 coordinates; addition does not double the width.
```

**Caption:** c₆₃ comes from the patch pixels. p₆₃ is learned for its grid location. Add matching coordinates to get e₆₃. All three rows have 192 coordinates; addition does not double the width.

**Speaker notes**

After adding position, does this row have 192 coordinates or 384?
Keep the crop visible. Recall the projection that made c63, then add its learned position row coordinate by coordinate to make e63.

### Page 44 — The image loss teaches the position table

```text
Start with small random values; train with the class loss.
p₆₃: 192 entries
+
c₆₃ from pixels
e₆₃ = c₆₃ + p₆₃
Transformer blocks
+ classifier
class loss
L
Backpropagation: gradients reach every trainable position row.
Illustrative SGD update of one entry
p₆₃,₁: 0.020 − 0.1 × 0.30 = −0.010
old value − learning rate × gradient
Alternative: fixed sinusoidal
sin/cos(position, coordinate)
No learned table entries.
The class loss supplies gradients for the position table and other model parameters. Training updates the table; inference reuses it unchanged. Fixed sinusoidal encodings instead compute position vectors from a formula.
```

**Caption:** The class loss supplies gradients for the position table and other model parameters. Training updates the table; inference reuses it unchanged. Fixed sinusoidal encodings instead compute position vectors from a formula.

**Speaker notes**

What tells the optimizer how to change a position coordinate: a position label, or the image classification loss?
Trace p63 through addition into the classifier, then follow the backward arrow. The loss updates the table jointly with other weights. Use the marked illustrative SGD step to explain one coordinate, then distinguish fixed sinusoidal vectors.

### Page 45 — Why add CLS? Give the classifier one image summary

```text
A SHORT DETOUR · WHY ADD CLS?
One image label from 196 patch rows
Which representation should the classifier read?
The whole dog photo
196 patch rows
P1 features
P2 features
P196 features
⋮
Classifier
One image label
CLS · learned start
+
Transformer
blocks
all 197 rows
CLS reads patch features
Final CLS
image summary
Averaging final patch rows is another readout option.
Patch embeddings give us many rows, while the target is one label for the photograph. We need a way to combine patch information into the representation that the classifier will read.
```

**Caption:** Patch embeddings give us many rows, while the target is one label for the photograph. We need a way to combine patch information into the representation that the classifier will read.

**Speaker notes**

We have 196 patch rows and want one label for the whole photograph. Which representation should the classifier read?
Start with the dog and its patch rows. Reveal one extra learned CLS row before the blocks. All rows enter together, and attention updates CLS using patch information. Then reveal the readout from final CLS into the classifier. CLS is one design choice; averaging final patch rows also works with a model trained for that readout.

### Page 46 — Add a summary row beside the dog’s patch rows

```text
The photograph still has exactly 196 patches.
The same dog photo
P63
768 pixel values
Linear(768, 192)
Patch content c₆₃
192 features
Create a parameter
192 trainable numbers
CLS row
192 features
196 image rows + 1 summary row = 197 rows
A patch row comes from pixels. CLS comes from a separate set of model parameters. No extra crop is cut from the photograph. Its 192 coordinates match the width of every patch row.
```

**Caption:** A patch row comes from pixels. CLS comes from a separate set of model parameters. No extra crop is cut from the photograph. Its 192 coordinates match the width of every patch row.

**Speaker notes**

Where are the pixels for CLS?
There are none. Trace the two different origins: a crop through the patch layer, and a separately stored parameter row.

### Page 47 — What makes CLS an image summary?

```text
CLS and patch rows can read one another’s current features.
Same photo
Input
Shared CLS start
196 patch rows
After block 1
Updated CLS
Updated patch rows
Block 1
After block 2
Updated CLS
Updated patch rows
Block 2
197 × 192 at each stage. All updates use that block’s input rows.
Continue through block 12. The classifier then reads final CLS.
The class loss trains this row to carry useful image information.
CLS can read all patch rows; each patch can read CLS and every patch. All updates use the rows entering that block. Block 2 reads the updated CLS and patches produced by block 1.
```

**Caption:** CLS can read all patch rows; each patch can read CLS and every patch. All updates use the rows entering that block. Block 2 reads the updated CLS and patches produced by block 1.

**Speaker notes**

Can patch rows read CLS too, and when do they see its updated version?
Yes. Follow both diagonals: follow the upward arrows from patch rows to CLS and the downward arrows from CLS to patch updates. Each attention operation computes all its queries, keys and values from the same incoming rows. It does not first update CLS and then let patches read that new CLS in the same attention operation. Block 2 receives both updated CLS and updated patch rows from block 1. Continue through block 12: the classifier reads final CLS, so the class loss trains that row to be useful.

### Page 48 — The same CLS start reads two different photographs

```text
Shared learned CLS s = [−0.356, −0.038, −0.072, …]
Add the same position vector before either image enters attention.
Starting CLS input e₀
After attention + residual
Dog
[−0.704, −0.067, −0.146, …]
Attention
same parameters
[0.463, −0.118, 0.216, …]
196 dog patch rows
Cat
[−0.704, −0.067, −0.146, …]
Attention
same parameters
[0.441, −0.147, 0.254, …]
196 cat patch rows
First 3 of 192 coordinates shown. Both CLS rows remain 192 numbers wide.
Both images use the same starting CLS and the same model parameters. Their patch rows differ, so attention produces different updates. These are measured CLS values after the first attention update and residual addition.
```

**Caption:** Both images use the same starting CLS and the same model parameters. Their patch rows differ, so attention produces different updates. These are measured CLS values after the first attention update and residual addition.

**Speaker notes**

The starting numbers match exactly. Why do the two outputs differ?
Point to the different photos. Their patches supply different keys and values. The starting CLS query is the same; the resulting attention message can differ.

### Page 49 — Add the summary row: 196 + 1 = 197

```text
The full sequence entering block 1
CLS
learned token
+
p₀
=
e₀
1 × 192
P1
c₁
+
p₁
=
e₁
1 × 192
P63
c₆₃
+
p₆₃
=
e₆₃
1 × 192
P196
c₁₉₆
+
p₁₉₆
=
e₁₉₆
1 × 192
Stack: [e₀; e₁; …; e₁₉₆] → E has shape 197 × 192
Each patch keeps its content-plus-position row. CLS gets its own learned position too. Stack the summary and patch rows into E: 197 rows, each with 192 features. Batch size one is omitted here.
```

**Caption:** Each patch keeps its content-plus-position row. CLS gets its own learned position too. Stack the summary and patch rows into E: 197 rows, each with 192 features. Batch size one is omitted here.

**Speaker notes**

Why do we now have 197 rows, while the width is still 192?
Identify the extra row at the top. Position is added coordinate by coordinate. The subscript is the row’s identity, not its width.

### Page 50 — Start with one Transformer block

```text
Our dog photograph
Prepared rows
197 × 192
Transformer block 1
Attention
share context
MLP
refine features
Updated rows
197 × 192
Each row keeps 192 features. Its numbers are updated.
First, let’s follow this one block.
Our 196 patch rows and one CLS row enter block 1. Attention shares information across rows; the MLP transforms each row. The output still has 197 rows and 192 features per row.
```

**Caption:** Our 196 patch rows and one CLS row enter block 1. Attention shares information across rows; the MLP transforms each row. The output still has 197 rows and 192 features per row.

**Speaker notes**

What goes into one block, and what comes out?
Keep the scope to block 1. Follow prepared rows through attention and then the MLP. Same matrix shape, updated feature values. Explain the stack only after these operations.

### Page 51 — Self-attention: Q, K and V share the same input

```text
TRANSLATION · cross-attention
Target-prefix states
Q
Encoded source states
K, V
Two language streams meet at attention.
ViT · self-attention
Current CLS + patch states
one image sequence
Project the same rows into Q, K and V
then mix their information
ViT has one stream. Translation had two streams.
Self-attention uses one input sequence. Cross-attention takes queries from one stream and keys and values from another.
```

**Caption:** Self-attention uses one input sequence. Cross-attention takes queries from one stream and keys and values from another.

**Speaker notes**

Does CLS query a separate image encoder through cross-attention?
No. CLS is one row inside the image sequence. In this pre-LN model, all three projections read the same normalized current rows.

### Page 52 — Make queries, keys and values from these rows

```text
P63: pixels → patch projection → c₆₃ → add p₆₃ → e₆₃
All 197 rows are ready. Follow patch 63 into the first block.
e₆₃
1 × 192
LayerNorm
still 1 × 192
× W_Q + b_Q
q₆₃
1 × 64
× W_K + b_K
k₆₃
1 × 64
× W_V + b_V
v₆₃
1 × 64
Each W: 192 × 64; each bias: 64
LOCAL ROLES
Q / receiver
K / source
V / message
LayerNorm rescales a row’s features and keeps its width. We name it here and study its details later. Three learned maps then make Q, K and V; the diagram shows one of three heads.
```

**Caption:** LayerNorm rescales a row’s features and keeps its width. We name it here and study its details later. Three learned maps then make Q, K and V; the diagram shows one of three heads.

**Speaker notes**

Which projection reads pixels, and which projections read the prepared embedding row?
Trace the small recap first. Enter the block with e63, normalize it, then split into the three separate learned projections for head one.

### Page 53 — The same input matrix feeds three learned projections

```text
Follow one head in block 1 · 192 features / 3 heads = 64 per head
X = LayerNorm(E)
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P196
1
2
…
192
197 × 192
·
·
·
·
·
·
·
·
·
× W_Q (192 × 64)
+ b_Q
64 values
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P196
Q
197 × 64
·
·
·
·
·
·
·
·
·
× W_K (192 × 64)
+ b_K
64 values
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P196
K
197 × 64
·
·
·
·
·
·
·
·
·
× W_V (192 × 64)
+ b_V
64 values
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P196
V
197 × 64
LOCAL ROLES
Q / receiver
K / source
V / message
One normalized matrix X feeds three different learned projections. Q, K and V each contain 197 rows of 64 features, in the same token order. This is one head; dots stand for omitted values.
```

**Caption:** One normalized matrix X feeds three different learned projections. Q, K and V each contain 197 rows of 64 features, in the same token order. This is one head; dots stand for omitted values.

**Speaker notes**

Do Q, K and V come from three different images?
Trace all three arrows back to the same X. The learned weights differ. Each projection processes every row using the same weights within that projection.

### Page 54 — One query–key comparison fills one matrix cell

```text
Rows: receiving queries. Columns: source keys. CLS is first on both axes.
Q · 197 × 64
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P63
…
P196
1
2
…
64
×
Kᵀ · 64 × 197
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
1
2
…
64
CLS
P1
…
P63
…
P196
Turn K’s rows into columns
÷ 8
S · 197 × 197
Columns = keys →
Rows = queries
·
·
·
s
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P63
…
P196
CLS
P1
…
P63
…
P196
Highlighted cell: s(CLS, P63) = q(CLS) · k(P63) / 8
CLS row: what CLS reads. CLS column: how each query scores CLS.
LOCAL ROLES
Q / receiver
K / source
V / message
Fix the receiving query to CLS and move across the columns. These scores determine CLS’s source weights after softmax. Moving down the CLS column changes the query while keeping the CLS key fixed.
```

**Caption:** Fix the receiving query to CLS and move across the columns. These scores determine CLS’s source weights after softmax. Moving down the CLS column changes the query while keeping the CLS key fixed.

**Speaker notes**

Which query and which key produced the highlighted cell?
Follow the highlighted row of Q and column of Kᵀ to row CLS, column P63 of S. The dot product sums 64 coordinate products. The CLS row fixes q_CLS and compares every key; the CLS column fixes k_CLS and compares every query. S can contain negative scores and need not be symmetric.

### Page 55 — Follow the CLS row from scores to weights

```text
S · 197 × 197 scores
Columns = source keys →
Rows = queries
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P63
…
P196
CLS
P1
…
P63
…
P196
CLS row, enlarged
CLS query stays fixed; compare every source key.
zoom
s₀
CLS
s₁
P1
s₂
P2
…
…
s₆₃
P63
…
…
s₁₉₆
P196
softmax
across all 197 scores
a₀
a₁
a₂
…
a₆₃
…
a₁₉₆
a₀ + a₁ + … + a₁₉₆ = 1
1 CLS score + 196 patch scores
Patch queries update patches → the next block’s CLS reads those updated patches.
LOCAL ROLES
Q / receiver
K / source
V / message
Earlier blocks update patches so later CLS queries can read their new features. In the final block, a CLS-only readout could compute just the CLS output, using all 197 incoming keys and values.
```

**Caption:** Earlier blocks update patches so later CLS queries can read their new features. In the final block, a CLS-only readout could compute just the CLS output, using all 197 incoming keys and values.

**Speaker notes**

If the classifier reads CLS, why compute the other score rows?
We follow one row to explain the calculation. Each patch query needs its own scores and weights to update that patch. The next block computes its keys and values from these updated patches, which CLS can then read. For this one CLS message alone, only its 197 scores and all 197 value rows are needed. In the final block, if only CLS is used, patch outputs can be skipped in a specialized implementation. All incoming keys and values are still needed. The standard implementation shown computes every row. There are 196 patch scores plus the CLS self-score. Softmax normalizes all 197; there is no causal mask.

### Page 56 — Turn each query’s 197 scores into 197 source weights

```text
S: matching scores
A: attention weights
·
·
·
s
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P63
…
P196
CLS
P1
…
P63
…
P196
softmax
across each row
·
·
·
a
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P63
…
P196
CLS
P1
…
P63
…
P196
Highlighted CLS row: sum = 1
a(CLS, P63): how much weight CLS gives to P63’s value row.
LOCAL ROLES
Q / receiver
K / source
V / message
Apply softmax across each score row. The CLS row becomes weights over CLS and all 196 patches, summing to one. Each other query gets its own weight row. These weights tell us how to mix the value rows.
```

**Caption:** Apply softmax across each score row. The CLS row becomes weights over CLS and all 196 patches, summing to one. Each other query gets its own weight row. These weights tell us how to mix the value rows.

**Speaker notes**

What are the alternatives in this softmax: animal classes or source rows?
They are sources, including CLS itself. a(CLS,P63) is normalized against all 197 scores in the CLS row. Every row sums to one; columns need not.

### Page 57 — Every image row can read every image row

```text
Text: predict the next token
ViT: classify the whole image
•
×
×
×
×
×
•
•
×
×
×
×
•
•
•
×
×
×
•
•
•
•
×
×
•
•
•
•
•
×
•
•
•
•
•
•
t₁
t₂
t₃
t₄
t₅
t₆
t₁
t₂
t₃
t₄
t₅
t₆
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
CLS
P1
…
P63
…
P196
CLS
P1
…
P63
…
P196
Read the current and earlier positions
Read every row, including itself
Filled cell = permitted connection. × = blocked. This shows access, not weight size.
No causal mask here: CLS can read P196, and P196 can read CLS.
LOCAL ROLES
Q / receiver
K / source
V / message
Next-token prediction hides later text positions. Here the complete photograph is available before classification. Every query can use every key and value, including itself. The ViT attention matrix has no causal triangle removed.
```

**Caption:** Next-token prediction hides later text positions. Here the complete photograph is available before classification. Every query can use every key and value, including itself. The ViT attention matrix has no causal triangle removed.

**Speaker notes**

If CLS is the first row, what would a causal mask let it read?
Only itself. That would prevent it from gathering patch information. This classifier uses full attention because all image patches are available for the image-label prediction.

### Page 58 — A message for CLS works like a message for bank

```text
Choose a receiver. Mix source values to make a message for that receiver.
Text · receiver: bank
river
other allowed
tokens
Mix their values
using bank’s weights
Message for bank
Image · receiver: CLS
P1
P63
P196
…
CLS
Mix their values
using CLS’s weights
Message for CLS
Same weighted-sum operation. Different receiver and sources.
LOCAL ROLES
Q / receiver
K / source
V / message
In text, bank receives a weighted mixture of allowed token values. Here, CLS receives a weighted mixture of image-token values, including its own. The query chooses the receiver; the values supply the message.
```

**Caption:** In text, bank receives a weighted mixture of allowed token values. Here, CLS receives a weighted mixture of image-token values, including its own. The query chooses the receiver; the values supply the message.

**Speaker notes**

What corresponds to bank in the calculation we are following?
CLS is our selected receiver. River was one text source; P63 is one image source. Both calculations sum weighted value vectors. CLS also has its own value row.

### Page 59 — The dog’s feature rows become value rows

```text
Same dog · block 1 · head 1
196 patches + CLS
X = LayerNorm(E)
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P63
…
P196
1
2
…
192
197 × 192
Linear(192,64)
× W_V + b_V
V: one value row per source
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P63
…
P196
1
2
…
64
197 × 64
P63 → v₆₃ = [−0.174, −1.201, …, −1.308]
64 learned features · rounded values from this dog
LOCAL ROLES
Q / receiver
K / source
V / message
Use the same normalized input X that produced Q and K. The value projection makes 64 features for each of its 197 rows. P63’s crop identifies one source; its value row contains learned features.
```

**Caption:** Use the same normalized input X that produced Q and K. The value projection makes 64 features for each of its 197 rows. P63’s crop identifies one source; its value row contains learned features.

**Speaker notes**

Where does row P63 of V come from?
Follow P63’s normalized 192-feature row through the shared value projection. It produces 64 features. CLS has a value row too, although it has no image crop.

### Page 60 — One weight scales all 64 features in its value row

```text
A · 197 × 197 weights
Columns = source keys →
Rows = queries
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P63
…
P196
CLS
P1
…
P63
…
P196
×
V · 197 × 64 values
Columns = value features →
Rows = sources
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P63
…
P196
1
2
…
64
1 · Pick receiver CLS
2 · Pick source P63
a₆₃
A[CLS, P63] · one weight
0.000504
3 · Pick P63’s value row
V[P63, :] · all 64 features
×
−0.174
−1.201
…
−1.308
v₆₃,₁
4 · Scale each feature
Contribution to CLS · 1 × 64
=
−0.000088
v₆₃,₂
−0.000606
…
−0.000659
LOCAL ROLES
Q / receiver
K / source
V / message
Use Next to select CLS’s row in A, P63’s column, P63’s row in V, then its feature coordinates. One scalar scales all 64 values. This contribution is addressed to CLS.
```

**Caption:** Use Next to select CLS’s row in A, P63’s column, P63’s row in V, then its feature coordinates. One scalar scales all 64 values. This contribution is addressed to CLS.

**Speaker notes**

Which row, column and value feature are we selecting?
Next 1: choose the CLS query row in A. Next 2: choose its P63 source column and extract A[CLS,P63]. Next 3: choose row P63 of V. Next 4: choose value feature 1 and multiply. Next 5: repeat for feature 2 and the remaining features. Every product is a contribution to CLS; P63 is the source.

### Page 61 — Each source contributes a weighted value row

```text
Keep the receiver fixed: every weight below comes from the CLS row.
Source
CLS weight
Value row · 64 features
Contribution to CLS
CLS
CLS
0.860556
×
0.418
0.071
…
0.359857
0.060762
…
P1
0.002298
×
−0.211
−0.027
…
−0.000485
−0.000063
…
P63
0.000504
×
−0.174
−1.201
…
−0.000088
−0.000606
…
P196
0.000233
×
−1.607
1.181
…
−0.000374
0.000275
…
Four source examples. Every contribution still has 64 features.
LOCAL ROLES
Q / receiver
K / source
V / message
Repeat the P63 calculation for other sources. Each uses its own CLS weight and value row. These are four measured examples from the same head; every contribution is a 64-feature vector addressed to CLS.
```

**Caption:** Repeat the P63 calculation for other sources. Each uses its own CLS weight and value row. These are four measured examples from the same head; every contribution is a 64-feature vector addressed to CLS.

**Speaker notes**

Are these messages for four different receivers?
No. All four weights come from the CLS attention row. These are four source contributions to one receiver. Each source supplies 64 features, scaled by its own weight.

### Page 62 — Add the contributions to make one CLS message

```text
Add the same feature across sources. Keep all 64 features.
Weighted contributions for CLS · each 1 × 64
CLS
0.359857
0.060762
…
P1
+
−0.000485
−0.000063
…
P63
+
−0.000088
−0.000606
…
P196
+
−0.000374
0.000275
…
Other 193
+
−0.005982
−0.003703
…
Message for CLS
0.353
0.057
…
1 × 64
197 source contributions → one message for the receiving CLS query.
LOCAL ROLES
Q / receiver
K / source
V / message
Add corresponding coordinates from all 197 weighted value rows, including CLS itself. The result has 64 features: one message for CLS in this head. The other 193 sources are grouped to keep the addition readable.
```

**Caption:** Add corresponding coordinates from all 197 weighted value rows, including CLS itself. The result has 64 features: one message for CLS in this head. The other 193 sources are grouped to keep the addition readable.

**Speaker notes**

Why is the result one row with 64 features?
For feature 1, sum 197 weighted feature-1 values. Repeat for features 2 through 64. We sum over sources, keeping the feature dimension. Every term belongs to the fixed CLS query.

### Page 63 — Where does the CLS message go?

```text
Same pattern as text: current token row + context update → updated token row.
Head 1 message
for CLS · 1 × 64
Combine 3 head messages
then project to 192 features
We open this box next.
CLS context update
1 × 192
Current CLS row
1 × 192
+
Updated CLS row
1 × 192
The CLS representation changes. It is still the same receiving token.
LOCAL ROLES
Q / receiver
K / source
V / message
The CLS messages become a 192-feature context update after combining heads and projecting. Add this update to the CLS row entering the block. The result is a new representation of the same CLS token, as in text.
```

**Caption:** The CLS messages become a 192-feature context update after combining heads and projecting. Add this update to the CLS row entering the block. The result is a new representation of the same CLS token, as in text.

**Speaker notes**

Can we add the 64-feature message directly to the 192-feature CLS row?
No. First combine all three heads and project. Then add the 192-feature context update to the original 192-feature CLS row. We will open the middle box in the multi-head slides.

### Page 64 — Each query gets its own message

```text
Keep the source values. Change the receiving query and its weights.
Receiver
Its own 197 weights
Same V
Its own message · 1 × 64
CLS query
0.8606
0.0023
…
×
197 × 64
0.353
0.057
…
P63 query
0.2644
0.0001
…
×
197 × 64
0.050
−0.358
…
CLS’s messages update CLS. P63’s messages update P63. Every row has its own.
LOCAL ROLES
Q / receiver
K / source
V / message
P63’s query mixes the same source values using P63’s weights, producing P63’s own message. CLS and every other patch do the same. Each receiver’s messages ultimately update its own input row through the residual path.
```

**Caption:** P63’s query mixes the same source values using P63’s weights, producing P63’s own message. CLS and every other patch do the same. Each receiver’s messages ultimately update its own input row through the residual path.

**Speaker notes**

Does the CLS message also update P63?
No. P63 uses its own query, its own attention weights and its own message. Both receivers read the same V in this head. All messages are computed from the rows entering this block.

### Page 65 — From one completed head to three parallel heads

```text
Head 1 is complete.
Q and K → source weights → mix V → a message for every row.
197 input rows
197 messages, each 64 features
Now use three heads in parallel
Each reads the same input rows
First combine their messages. Then continue to the MLP.
We have finished one head’s attention calculation. This model has three heads. Each starts from the same input rows and computes its own messages. Next, follow those parallel paths and combine their outputs before the MLP.
```

**Caption:** We have finished one head’s attention calculation. This model has three heads. Each starts from the same input rows and computes its own messages. Next, follow those parallel paths and combine their outputs before the MLP.

**Speaker notes**

Have we finished head 1, or already updated the input E?
We have its message matrix H¹. The output projection and residual addition still follow. Introduce the other heads only now; they run alongside head 1 and do not consume its output.

### Page 66 — What might different heads look for in this photograph?

```text
Illustrative possibilities · learned heads are not assigned these jobs
Nearby texture
a different weighted
message
Does this fur continue nearby?
Animal parts
a different weighted
message
Which face clues belong together?
Wider context
a different weighted
message
How does this region fit the scene?
Different heads can learn different matching patterns and different messages. Local texture, relationships between parts and wider context are possible intuitions. These are hypotheses, not measured head labels.
```

**Caption:** Different heads can learn different matching patterns and different messages. Local texture, relationships between parts and wider context are possible intuitions. These are hypotheses, not measured head labels.

**Speaker notes**

Could one mixture preserve all the different clues that matter for this image?
Connect to coat reading colour and material in Part III. Each visual role is a possible interpretation, not a hand-written rule or guaranteed specialization.

### Page 67 — The same rows feed three sets of Q, K and V

```text
196 patch rows + one CLS row = 197 rows
Each Q/K/V: 197 × 64
X = LayerNorm(E)
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P196
197 × 192
Head 1
own Q/K/V projections
Q¹
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P196
K¹
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
V¹
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
Head 2
own Q/K/V projections
Q²
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P196
K²
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
V²
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
Head 3
own Q/K/V projections
Q³
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
CLS
P1
…
P196
K³
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
V³
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
One shared CLS input; each head projects it differently.
All three heads read the same normalized matrix X. Each has its own Q, K and V projection weights and biases. Every output matrix has 197 rows and 64 features. The highlighted first row is always CLS.
```

**Caption:** All three heads read the same normalized matrix X. Each has its own Q, K and V projection weights and biases. Every output matrix has 197 rows and 64 features. The highlighted first row is always CLS.

**Speaker notes**

Do we create three CLS tokens or divide the patches between the heads?
Neither. Every head reads all 197 rows, including the same 192-feature CLS input. Each applies a different learned projection. Superscripts 1, 2 and 3 name heads.

### Page 68 — Each head repeats the complete attention calculation

```text
Head 1
Q¹
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 64
(K¹)ᵀ
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
64 × 197
S¹
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 197
A¹
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 197
V¹
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 64
H¹
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 64
×
÷ 8
softmax
×
mix values
Head 2
Q²
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 64
(K²)ᵀ
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
64 × 197
S²
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 197
A²
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 197
V²
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 64
H²
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 64
×
÷ 8
softmax
×
mix values
Head 3
Q³
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 64
(K³)ᵀ
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
64 × 197
S³
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 197
A³
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 197
V³
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 64
H³
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
·
197 × 64
×
÷ 8
softmax
×
mix values
Each head has its own matching scores, attention weights and messages.
Within each head, compare Q with K, scale the scores, apply softmax across each row, then multiply by that head’s V. Each path returns 197 messages of 64 features. The heads operate in parallel.
```

**Caption:** Within each head, compare Q with K, scale the scores, apply softmax across each row, then multiply by that head’s V. Each path returns 197 messages of 64 features. The heads operate in parallel.

**Speaker notes**

Does head 2 use head 1’s attention weights or its value matrix?
No. Keep each colored path intact: Qʰ and Kʰ produce Aʰ, which mixes Vʰ to make Hʰ. Every head has its own 197×197 weight matrix.

### Page 69 — One CLS input produces three different messages

```text
Follow the same dog and the same receiving CLS row.
CLS row in X · 1 × 192
[0.044, −0.011, 0.154, …]
same input to every head
Head 1 attention
[0.353, 0.057, 0.024, …]
CLS message · 1 × 64
Head 2 attention
[−0.091, −0.112, −0.086, …]
CLS message · 1 × 64
Head 3 attention
[−0.002, −0.566, 0.020, …]
CLS message · 1 × 64
One CLS input row.
Three sets of learned projections.
CLS is the same input row for all heads. Their learned projections produce different queries, keys and values, so their attention computations produce different messages. These measured messages each contain 64 features. There is still one CLS token.
```

**Caption:** CLS is the same input row for all heads. Their learned projections produce different queries, keys and values, so their attention computations produce different messages. These measured messages each contain 64 features. There is still one CLS token.

**Speaker notes**

What is shared, and what differs, across these three CLS paths?
The normalized CLS input is shared. The projection parameters, projected queries/keys/values, source weights and resulting messages differ. Each message also depends on the patch values and keys from this photograph.

### Page 70 — Concatenate the three CLS messages

```text
Head 1 CLS message
[0.353, 0.057, …]
1 × 64
Head 2 CLS message
[−0.091, −0.112, …]
1 × 64
Head 3 CLS message
[−0.002, −0.566, …]
1 × 64
[0.353, 0.057, …]
features 1–64
[−0.091, −0.112, …]
features 65–128
[−0.002, −0.566, …]
features 129–192
One joined row · 1 × 192     (64 + 64 + 64)
Keep all three messages by placing their coordinates side by side. The joined CLS row has 192 features: 64 from head 1, then 64 from head 2, then 64 from head 3. Concatenation has no learned parameters.
```

**Caption:** Keep all three messages by placing their coordinates side by side. The joined CLS row has 192 features: 64 from head 1, then 64 from head 2, then 64 from head 3. Concatenation has no learned parameters.

**Speaker notes**

Does concatenation add corresponding numbers from different heads?
No. Follow each vector into its own feature range. The coordinates retain their values and order. Joining happens across the feature dimension, keeping one receiver row.

### Page 71 — Keep the embedding; add the context from attention

```text
Original
E
197 × 192
LayerNorm
X
Head 1
H¹ · 197 × 64
Head 2
H² · 197 × 64
Head 3
H³ · 197 × 64
Concatenate
H¹
H²
H³
Joined messages
197 × 192
Linear
(192,192)
× W_O + b_O
ΔE
Update
Skip path: carry the original E unchanged
+
U
Updated
U = E + ΔE
both 197 × 192
Each H contains one message per row: CLS, P1, …, P196.
The same residual pattern as text: original embedding + context update.
Follow two paths from E. The heads produce messages; concatenate and project them to make ΔE. The skip path carries E directly to addition. U = E + ΔE is the contextualized representation passed to the MLP.
```

**Caption:** Follow two paths from E. The heads produce messages; concatenate and project them to make ΔE. The skip path carries E directly to addition. U = E + ΔE is the contextualized representation passed to the MLP.

**Speaker notes**

Which operation learns to combine the head features, and which row gets the residual update?
Linear(192,192) learns the combination. Add the result to E, before the attention branch’s LayerNorm. The MLP will receive this updated row through its own LayerNorm.

### Page 72 — The dog’s CLS keeps its input and gains context

```text
Same dog · block 1 · zoom in on the CLS row
Every vector below contains 192 numbers.
Original CLS embedding
[−0.704, −0.067, …]
e₀ · 1 × 192
After concatenation + projection
[1.167, −0.051, …]
Δe₀ · attention update
skip path
+
Contextualized CLS
[0.463, −0.118, …]
u₀ · 1 × 192
First coordinate: −0.704 + 1.167 ≈ 0.463
Next: MLP → later blocks → final CLS → class scores.
LOCAL ROLES
Q / receiver
K / source
V / message
Add the projected attention update coordinate by coordinate to the original CLS embedding. This produces a contextualized CLS row of the same width. It continues through the MLP and later blocks before the classifier reads the final summary.
```

**Caption:** Add the projected attention update coordinate by coordinate to the original CLS embedding. This produces a contextualized CLS row of the same width. It continues through the MLP and later blocks before the classifier reads the final summary.

**Speaker notes**

Is the attention update already the contextualized embedding, or do we still add something?
The update is Δe₀. The contextualized row is u₀=e₀+Δe₀. Add the original block input, before LayerNorm. The 192 output features are a representation; class scores are computed later.

### Page 73 — Open block 1: attention, then the MLP

```text
Input E
197 × 192
Transformer
block 1
Output E¹
197 × 192
INSIDE BLOCK 1
E
Normalize → attention → + E
share context across rows
U
Normalize → MLP → add U
transform each row’s features
E¹
OPEN THIS PART NEXT
All three matrices: 197 × 192. One CLS row: 1 × 192.
One block makes two updates. Attention lets rows exchange information; the MLP transforms each resulting row. Each branch adds its update to its input. The block receives and returns 197 rows, each 192 features wide.
```

**Caption:** One block makes two updates. Attention lets rows exchange information; the MLP transforms each resulting row. Each branch adds its update to its input. The block receives and returns 197 rows, each 192 features wide.

**Speaker notes**

Have we finished the entire block after the attention residual?
We have reached U, the midpoint. Open the MLP branch next. E¹ names the output of block 1; it is not a power. Both branches update CLS and every patch row.

### Page 74 — Open the MLP: 192 inputs, 768 hidden units, 192 outputs

```text
Follow the dog’s CLS row after LayerNorm.
Normalized row
1 × 192
n₁
n₂
n₁₉₂
⋮
Hidden pre-activation
1 × 768
−0.091
0.248
z₇₆₈
⋮
After GELU
1 × 768
−0.042
0.148
a₇₆₈
⋮
MLP update
1 × 192
−0.100
−0.041
m₁₉₂
⋮
Linear(192,768)
GELU
Linear(768,192)
192 → 768 → 768 → 192 features · biases in both linear layers
Each linear layer connects every input feature to every output unit. GELU transforms each hidden activation. Only a few neurons are drawn; dots omit the rest. The displayed numbers are measured values for this dog’s CLS row.
```

**Caption:** Each linear layer connects every input feature to every output unit. GELU transforms each hidden activation. Only a few neurons are drawn; dots omit the rest. The displayed numbers are measured values for this dog’s CLS row.

**Speaker notes**

Does 768 mean patches, pixels, or hidden features here?
It is the number of hidden features for one row. GELU is the nonlinearity between the two learned linear layers. The MLP returns a 192-feature update; the residual addition follows.

### Page 75 — Add the MLP update to finish block 1

```text
Same dog · finish the second residual addition
The MLP produces an update for the current CLS row.
After attention: u₀
[0.463, −0.118, …]
1 × 192
[−0.100, −0.041, …]
MLP update · 1 × 192
keep u₀
+
Block 1 output: e₀¹
[0.363, −0.159, …]
1 × 192
First coordinate: 0.463 − 0.100 ≈ 0.363
Do this for CLS and every patch: E¹ = U + MLP(LayerNorm(U)).
Keep the row produced by attention and add the MLP’s update. The result is this row’s output from block 1. Apply the same operation to all 197 rows. Their feature values change; their identities and width stay the same.
```

**Caption:** Keep the row produced by attention and add the MLP’s update. The result is this row’s output from block 1. Apply the same operation to all 197 rows. Their feature values change; their identities and width stay the same.

**Speaker notes**

Do we add the MLP update to E or to U?
Add it to U, the result of the attention residual. There are two successive residual additions in the block. Its final output includes both updates.

### Page 76 — Pass the complete output of block 1 into block 2

```text
Same dog · carry every updated row forward
Input E
CLS
· · ·
P1
· · ·
…
· · ·
P196
· · ·
197 × 192
E¹
CLS
· · ·
P1
· · ·
…
· · ·
P196
· · ·
197 × 192
E²
CLS
· · ·
P1
· · ·
…
· · ·
P196
· · ·
197 × 192
Block 1
Attention + add
MLP + add
own learned weights
Block 2
Attention + add
MLP + add
own learned weights
Block 1’s output is exactly block 2’s input.
All 197 rows continue. Both blocks return 192 features per row.
The entire output matrix continues to the next block: CLS and all 196 patch rows. Block 2 has its own attention and MLP weights. It applies the same sequence of operations to the feature values produced by block 1.
```

**Caption:** The entire output matrix continues to the next block: CLS and all 196 patch rows. Block 2 has its own attention and MLP weights. It applies the same sequence of operations to the feature values produced by block 1.

**Speaker notes**

Which rows do we keep, and which block processes them next?
Keep every row. Follow the straight arrow into a distinct block 2. There is no return to block 1, no repeated patchification, and no new CLS token.

### Page 77 — What changes as the rows move through the blocks?

```text
Feature values flow forward; each block uses its own parameter set.
E
input features
Block 1
attention + MLP
E¹
new features
Block 2
attention + MLP
Block 1 weights
Block 2 weights
Recomputed in each block:
Q, K, V → attention weights → messages → MLP activations → updated rows
Learned weights stay fixed in this forward pass. Training updates them later.
Each block recomputes queries, keys, values, attention weights and MLP activations from its input. Its learned parameters stay fixed during this forward pass. The next block uses a different parameter set. Matrix dimensions and row identities stay the same.
```

**Caption:** Each block recomputes queries, keys, values, attention weights and MLP activations from its input. Its learned parameters stay fixed during this forward pass. The next block uses a different parameter set. Matrix dimensions and row identities stay the same.

**Speaker notes**

Are we changing the weights or the feature values when we run this dog forward?
The feature values change. The blocks already have their own stored weights, which stay fixed during inference. Training later computes gradients and the optimizer updates those stored parameters.

### Page 78 — Continue through the stack, then classify the image

```text
One forward pass through 12 distinct Transformer blocks
E
197 × 192
Block 1
attention + MLP
own weights
Block 2
attention + MLP
own weights
Block 12
attention + MLP
own weights
…
E¹
E²
E¹¹
Every arrow carries all 197 × 192 feature values.
After block 12 → final normalization → read CLS → class scores
One image. Twelve blocks. One prediction at the end.
Pass the updated features through blocks 1 to 12 in order. The architecture repeats, with separate learned parameters in each block. After block 12 and final normalization, read CLS and calculate one set of class scores for this image.
```

**Caption:** Pass the updated features through blocks 1 to 12 in order. The architecture repeats, with separate learned parameters in each block. After block 12 and final normalization, read CLS and calculate one set of class scores for this image.

**Speaker notes**

What does “12 blocks” repeat, and when do we make a prediction?
The attention-plus-MLP architecture repeats. A different block processes each successive representation. All rows continue through the stack; one classifier reads the final CLS.

### Page 79 — Select CLS from the final feature matrix

```text
The same dog has now passed through all 12 blocks.
After block 12
CLS
⋯
⋯
⋯
⋯
P1
⋯
⋯
⋯
⋯
…
⋯
⋯
⋯
⋯
P196
⋯
⋯
⋯
⋯
197 × 192
Final
LayerNorm
Normalized rows
CLS
⋯
⋯
⋯
⋯
P1
⋯
⋯
⋯
⋯
…
⋯
⋯
⋯
⋯
P196
⋯
⋯
⋯
⋯
197 × 192
Select CLS
[−0.144, −0.340, …]
1 × 192
One image summary
192 features
Read the highlighted row; the patch rows have already supplied context.
After block 12, all 197 rows are still present. Final LayerNorm keeps the same shape. Select the highlighted CLS row: one image summary with 192 features. Its values now reflect information gathered from this dog’s patches.
```

**Caption:** After block 12, all 197 rows are still present. Final LayerNorm keeps the same shape. Select the highlighted CLS row: one image summary with 192 features. Its values now reflect information gathered from this dog’s patches.

**Speaker notes**

Which part of the 197 × 192 matrix enters the classifier?
Extract the entire first row, all 192 features. This CLS is the result after the blocks and final normalization, not the initial learned CLS parameter. We do not average the rows in this model.

### Page 80 — Open the classifier: 192 features become 1,000 scores

```text
This checkpoint predicts 1,000 ImageNet classes.
CLS features · 1 × 192
Class scores · 1 × 1,000
−0.144
h₁
−0.340
h₂
6.738
h₁₉₂
⋮
15.476
Newfoundland
11.401
Tibetan mastiff
10.521
briard
⋮
997 more class scores
nn.Linear(192,1000)
192 weights + 1 bias
for each class
One affine layer produces scores. Softmax follows.
Every class has one output neuron connected to all 192 CLS features. Each neuron uses its own learned weights and bias to produce a score. Only three of the 1,000 class neurons are drawn. Their scores are measured.
```

**Caption:** Every class has one output neuron connected to all 192 CLS features. Each neuron uses its own learned weights and bias to produce a score. Only three of the 1,000 class neurons are drawn. Their scores are measured.

**Speaker notes**

Does one feature correspond to one class, or does every class read the whole summary?
Each class reads all 192 features with its own weighted sum. This class head is a single affine layer. It has no hidden layer or GELU; softmax converts its scores to probabilities next.

### Page 81 — One class score is a weighted sum plus a bias

```text
Zoom into one output neuron: Newfoundland
−0.144
h₁
−0.340
h₂
× 0.00682
× 0.01947
Other 190 weighted features
sum = 15.4822
Bias: 0.0017
Σ
15.4763
one class score
−0.0010 −0.0066 + 15.4822 + 0.0017 ≈ 15.4763
Multiply each CLS feature by this class’s learned weight. Sum all 192 products and add its bias. For this dog, the Newfoundland neuron produces 15.4763. Every other class performs the same calculation with its own weights and bias.
```

**Caption:** Multiply each CLS feature by this class’s learned weight. Sum all 192 products and add its bias. For this dog, the Newfoundland neuron produces 15.4763. Every other class performs the same calculation with its own weights and bias.

**Speaker notes**

What makes the Newfoundland score different from the Tibetan mastiff score?
They read the same image summary but use different learned weights and biases. The 190 omitted products still contribute. Keep the score separate from its later probability.

### Page 82 — Turn all 1,000 scores into class probabilities

```text
The alternatives are class labels: normalize over all 1,000.
Class scores (logits)
Class probabilities
Newfoundland
15.476
95.73%
Tibetan mastiff
11.401
1.63%
briard
10.521
0.67%
997 more classes
⋮
1.97% across 997 others
softmax
all 1,000 scores
p(Newfoundland) = exp(15.476) / sum of exp(all 1,000 scores)
≈ 0.9573     ·     All 1,000 probabilities sum to 1.
Exponentiate each score and divide by the sum over all 1,000 classes. Newfoundland receives 95.73% probability. The displayed classes are only three alternatives; the remaining 997 also enter the denominator. Choose the class with the largest probability.
```

**Caption:** Exponentiate each score and divide by the sum over all 1,000 classes. Newfoundland receives 95.73% probability. The displayed classes are only three alternatives; the remaining 997 also enter the denominator. Choose the class with the largest probability.

**Speaker notes**

Would softmax over only these three displayed scores give the same answer?
No. All 1,000 scores contribute to the denominator. The remaining probability mass is shown explicitly. Attention softmax normalized over source rows; this softmax normalizes over image classes.

### Page 83 — Two softmaxes, two different questions

```text
The same normalization, applied to two different questions
ATTENTION · one query
Which source rows help?
197 scores → softmax over sources → 197 weights
CLS + P1 + P2 + … + P196
CLASSIFIER · one image
Which image label fits?
1,000 logits → softmax over labels → probabilities
Newfoundland, Persian cat, …
Attention normalizes across 197 source rows for each query and head. The classifier normalizes across 1,000 labels for each image. Both sum to one along their chosen axis; only the second distribution predicts the image label.
```

**Caption:** Attention normalizes across 197 source rows for each query and head. The classifier normalizes across 1,000 labels for each image. Both sum to one along their chosen axis; only the second distribution predicts the image label.

**Speaker notes**

Does a large attention weight mean a high probability of the dog class?
No. It means that one source contributes strongly to that query’s value mixture. The class head later scores labels using the final CLS features.

### Page 84 — The same dog now has its final prediction

```text
The same photograph
Final CLS
192 image features
Linear + softmax
1,000 probabilities
Choose the largest probability
Newfoundland · 95.73%
Pixels → patches → contextual features → one image prediction.
The classifier reads the final CLS, scores 1,000 labels, and selects Newfoundland as the most probable class. Its probability for this photograph is 95.73%. We have completed one forward pass from pixels to an image prediction.
```

**Caption:** The classifier reads the final CLS, scores 1,000 labels, and selects Newfoundland as the most probable class. Its probability for this photograph is 95.73%. We have completed one forward pass from pixels to an image prediction.

**Speaker notes**

Did each patch predict its own label, or did the model make one image prediction?
The final CLS summarizes the contextual features for this one image. One class head reads that summary. The 95.73% is a model probability for this photograph, not an accuracy measurement.

### Page 85 — Section 4 · Whole model walkthrough

```text
SECTION
04
Whole model walkthrough
We have opened each part of the classifier.
How do forward and backward fit together?
One example: follow the full model to a label loss, then reverse the path to compute gradients and update the parameters.
```

**Caption:** One example: follow the full model to a label loss, then reverse the path to compute gradients and update the parameters.

**Speaker notes**

How do forward and backward fit together?
Connect the previous result to this new question. Keep the same image classifier as the reference.

### Page 86 — One shape trace from pixels to class scores

```text
ONE IMAGE · batch axis omitted
Pixels
3 × 224 × 224
Patch projection
196 × 192
Add CLS + position
197 × 192
12 encoder blocks
197 × 192
Final LN → read CLS
192
Class head
1,000 logits
Blocks preserve the shape. They change what each row represents.
```

**Caption:** Blocks preserve the shape. They change what each row represents.

**Speaker notes**

Which step changes the number of rows, and which step changes the final output width?
Prepending CLS changes 196 rows to 197. The head maps the selected 192-feature row to 1,000 logits. Expand the next diagram to see inside every block.

### Page 87 — Inside each block: mix, transform, keep the residual

```text
ONE PRE-LAYERNORM BLOCK · input and output: 197 × 192
E
LayerNorm
Self-attention
3 heads · 64 features each
+
U
U
LayerNorm
MLP
192 → 768 → 192 · GELU
+
Carry the input along the skip path
E′
Attention mixes rows. The MLP transforms each row. Both add a residual update.
Repeat this structure 12 times, with different learned weights in each block. All 197 rows are updated in parallel.
```

**Caption:** Repeat this structure 12 times, with different learned weights in each block. All 197 rows are updated in parallel.

**Speaker notes**

Which operation exchanges information between rows?
Self-attention mixes value rows. The MLP is applied independently to each row. LayerNorm precedes each branch; residual additions preserve the current representation.

### Page 88 — Backward: compute gradients, then update the model

```text
Same model, reverse direction: loss → parameter gradients
Image x
3 × 224 × 224
Input embedding
197 × 192
12 blocks
197 × 192
Read CLS + head
192 → 1,000
Label loss
L = 0.0437
Fixed data
Gradients reach the head, all blocks, and the learned input parameters.
Compute gradients
θ.grad = ∂L/∂θ
Optimizer step
θ ← θ − η ∇θ L
Next forward pass
use updated parameters
backward() computes gradients; step() changes the parameters.
Reverse the forward dependencies to compute gradients for the trainable parameters. The optimizer uses these gradients to update them. The next forward pass uses the updated model. The image and its label remain fixed.
```

**Caption:** Reverse the forward dependencies to compute gradients for the trainable parameters. The optimizer uses these gradients to update them. The next forward pass uses the updated model. The image and its label remain fixed.

**Speaker notes**

Does backward itself change the weights?
No. Backward computes gradients through the same model. The optimizer step applies the update. The learned input parameters include the patch projection, positions and starting CLS; the photograph and label are fixed data.

### Page 89 — Section 5 · CNNs, ViTs and inductive bias

```text
SECTION
05
CNNs, ViTs and inductive bias
Attention lets patches exchange information.
How do they differ, and which should we try?
Compare context, mixing, wider views and readouts, then connect those choices to inductive bias and practical data and compute constraints.
```

**Caption:** Compare context, mixing, wider views and readouts, then connect those choices to inductive bias and practical data and compute constraints.

**Speaker notes**

How do they differ, and which should we try?
Connect the previous result to this new question. Keep the same image classifier as the reference.

### Page 90 — Two ways to build an image representation

```text
1 · Gather context: which inputs can interact in one layer?
CNN · local filters
ViT · global attention
One local neighbourhood
feeds one output location.
Start with nearby clues.
One query can use
every source patch.
Near and far are available.
2 · Mix information: what determines each contribution?
Learned filter
Reuse the same weights
at every image location.
Query + keys
Compute weights
for this query + image.
Both learn parameters. ViT recomputes attention weights from the current features.
First compare the available connections. Then press Next to compare the mixing weights. CNN filters stay fixed during a forward pass. ViT projection parameters also stay fixed, while attention weights are computed from the current query and source features.
```

**Caption:** First compare the available connections. Then press Next to compare the mixing weights. CNN filters stay fixed during a forward pass. ViT projection parameters also stay fixed, while attention weights are computed from the current query and source features.

**Speaker notes**

What is shared, and what changes when we show a new image?
Pause on point 1: a conventional convolution reads local neighbours; global attention can connect distant patches immediately. Reveal point 2: both models keep their learned parameters fixed during inference. A CNN reuses its learned kernel across locations; ViT computes attention coefficients from query–key matches. These coefficients depend on the input and receiver. They are not the stored projection weights.

### Page 91 — A wider view, then one image label

```text
3 · Build a wider view: how do distant clues meet?
CNN · local filters
ViT · global attention
Local features
Combine them
Wider context
Successive layers connect larger regions.
Near + far
patches
One global
attention layer
Later blocks refine those relationships.
4 · Read out a label: turn many locations into one vector.
Spatial average
one feature vector
Class head
image scores
CLS
P1
…
P196
Read final CLS
one feature vector
Class head
image scores
Both produce an image-level summary. Pooling is also a valid ViT readout.
Trace context through the layers, then press Next to inspect the readout. Both models can use the whole image. A classifier needs one image-level vector: CNNs often use spatial pooling; this ViT reads final CLS.
```

**Caption:** Trace context through the layers, then press Next to inspect the readout. Both models can use the whole image. A classifier needs one image-level vector: CNNs often use spatial pooling; this ViT reads final CLS.

**Speaker notes**

Are distant clues unavailable to a CNN, and is CLS required for every ViT?
Pause on point 3: repeated local layers broaden a conventional CNN’s receptive field; a global attention layer allows direct distant interactions. Later ViT blocks still refine features. Reveal point 4: average spatial features or read a trained CLS representation, then apply the class head. Both routes produce one vector. Return to the earlier CLS-versus-pooling discussion without re-deriving it.

### Page 92 — Inductive bias: a useful starting assumption

```text
Inductive bias = assumptions built into how the model learns.
Example: move
the same stripe.
new location
Same local
pattern
What is built in?
CNN
ViT · global attention
Spatial structure
Local filters, reused
across the image
Patches + position;
flexible global mixing
The shifted stripe
The same filter can
detect it elsewhere
Position can change
the attention pattern
Learned from data
Which local filters help
Which patch relations help
A CNN builds in local reuse: one learned detector can look for a pattern at many locations. ViT also has structure—patches, shared projections and position signals—but gives attention more freedom to learn spatial relationships.
```

**Caption:** A CNN builds in local reuse: one learned detector can look for a pattern at many locations. ViT also has structure—patches, shared projections and position signals—but gives attention more freedom to learn spatial relationships.

**Speaker notes**

Does inductive bias mean the edge detector’s weights were chosen by hand?
No. The architecture chooses local connections and parameter sharing; training learns the filter values. Both architectures still learn from data. The translated stripe is an illustration, not a model prediction.

### Page 93 — Which is a sensible starting point?

```text
Practical starting points · validate on your task and hardware.
Your situation
CNN
ViT · global attention
Few labels;
train from scratch
Useful baseline:
built-in local reuse
Pay close attention
to the training recipe
Few labels;
pretrained weights
Fine-tune a suitable
pretrained CNN
Fine-tune a suitable
pretrained ViT
W
Phone or tight
latency budget
Try a compact CNN;
measure on the device
Try an efficient ViT;
measure on the device
Distant clues
across the image
Use enough depth
for broad context
Global attention links
distant patches directly
Compare validation accuracy, speed and memory under the same budget.
Treat these as starting points, not a ranking. Pretraining can matter more than the architecture name on a small dataset. Compare suitable models using the same validation set and measure speed and memory on the target hardware.
```

**Caption:** Treat these as starting points, not a ranking. Pretraining can matter more than the architecture name on a small dataset. Compare suitable models using the same validation set and measure speed and memory on the target hardware.

**Speaker notes**

We have few labelled pet photos and access to pretrained weights. Which column should we choose?
Both are plausible: adapt suitable checkpoints and validate. Few task labels do not rule out a pretrained ViT. For device use, compare measured latency and memory rather than assuming every CNN is faster.

### Page 94 — Implementation lab · optional

```text
IMPLEMENTATION LAB · OPTIONAL
Build the same ViT in PyTorch
One image
B × 3 × 224 × 224
Code + tensor shapes
Linear and Conv2d
The same prediction
B × 1,000
Use this lab in class, or work through it after the conceptual lecture.
All implementation details are retained. Every operation is paired with shapes, diagrams or a numerical equivalence check.
```

**Caption:** All implementation details are retained. Every operation is paired with shapes, diagrams or a numerical equivalence check.

**Speaker notes**

What should both patch implementations compute?
Exactly the same affine map with the same weights and biases, up to floating-point roundoff.

### Page 95 — Keep the same photograph and add the batch axis

```text
# rgb: (224, 224, 3), RGB per pixel
x = rgb.permute(2, 0, 1)  # 3, 224, 224
x = x.unsqueeze(0)       # 1, 3, 224, 224
# Patch order: all R, then G, then B
Height × width × RGB
PyTorch: B × C × H × W
B × 3 × 224 × 224
The prepared photograph is 224×224 with three RGB channels. PyTorch puts channels before height and width, then adds a batch axis. B counts images; it does not count patches.
```

**Caption:** The prepared photograph is 224×224 with three RGB channels. PyTorch puts channels before height and width, then adds a batch axis. B counts images; it does not count patches.

**Speaker notes**

Which axis says how many photographs are in the batch?
The leading axis B. One photograph uses B=1; a batch of photographs uses the same model parameters for each.

### Page 96 — Our target: turn every patch into 192 features

```text
Implement the same patch projection we used for the dog.
224 × 224 × 3
196 patches
One RGB patch
3 × 16 × 16
= 768 numbers
Shared projection
768 → 192
Patch row
192 features
Apply the same learned weights and biases at every location.
One image → 196 rows, each with 192 features.
Two implementations: flatten + Linear     or     Conv2d
Each RGB patch supplies 768 numbers. One shared projection turns it into 192 features. Reusing that projection across the 14×14 grid produces 196 rows. We will implement this exact operation in two ways.
```

**Caption:** Each RGB patch supplies 768 numbers. One shared projection turns it into 192 features. Reusing that projection across the 14×14 grid produces 196 rows. We will implement this exact operation in two ways.

**Speaker notes**

Do we learn a separate projection for each of the 196 patches?
No. Each patch supplies different inputs to the same learned projection. The same parameters also serve the cat and every other image.

### Page 97 — See the patch projection as a layer of neurons

```text
One affine layer: nn.Linear(768, 192, bias=True)
768 inputs
192 outputs
w₁,₁, w₁,₂, …, w₁,₇₆₈
x₁
x₂
x₇₆₈
c₁
c₂
c₁₉₂
⋮
⋮
+ b₁
Open just the first output neuron
c₁ = w₁,₁x₁ + w₁,₂x₂ + …
+ w₁,₇₆₈x₇₆₈ + b₁
Every output has its own
768 weights + 1 bias.
192 × 768 weights + 192 biases = 147,648 parameters
Patch projection: no hidden layer or activation. Later MLP: Linear → GELU → Linear.
All 768 inputs connect to every output neuron. Each neuron computes a weighted sum and adds its own bias. The patch projection is one dense layer; the later Transformer MLP has two layers and a nonlinear activation.
```

**Caption:** All 768 inputs connect to every output neuron. Each neuron computes a weighted sum and adds its own bias. The patch projection is one dense layer; the later Transformer MLP has two layers and a nonlinear activation.

**Speaker notes**

Where does the 192 in the bias count come from?
There is one bias per output feature, not one per input pixel or patch. Every output has 768 weights, so there are 147,456 weights and 192 biases in total.

### Page 98 — Implementation 1: extract patches, then use Linear

```text
linear = nn.Linear(768, 192, bias=True)
def project_with_linear(images, linear):
    patches = F.unfold(images, kernel_size=16, stride=16)
    # B × 768 × 196; each patch: all R, then G, then B
    # Within each channel: left to right, top to bottom
    patches = patches.transpose(1, 2)
    # B × 196 × 768: one flattened patch per row
    return linear(patches)  # B × 196 × 192
images · B × 3 × 224 × 224
unfold: flattened columns
B × 768 × 196
transpose: patch rows
B × 196 × 768
Linear: project each row
B × 196 × 192
One patch row: 256 R values | 256 G values | 256 B values.
Unfold extracts each non-overlapping patch as a column. Transpose makes patches into rows. Linear changes only the last axis, from 768 inputs to 192 features, reusing its weights for every patch and image.
```

**Caption:** Unfold extracts each non-overlapping patch as a column. Transpose makes patches into rows. Linear changes only the last axis, from 768 inputs to 192 features, reusing its weights for every patch and image.

**Speaker notes**

Does applying Linear to 196 rows create 196 copies of its weights?
No. A Linear module broadcasts the same affine operation across the leading axes. Only activations grow with B and the number of patches.

### Page 99 — Implementation 2: the same projection with Conv2d

```text
conv = nn.Conv2d(3, 192, kernel_size=16, stride=16,
                 padding=0, bias=True)
14 × 14 locations
One filter = one output neuron
+ bⱼ
3 × 16 × 16 weights
all 768 pixels contribute
192 filters → 192 features
at each patch location
Stride 16: move by one patch.
Output: B × 192 × 14 × 14
Same filters at every location. Kernel size = stride = 16; patches do not overlap.
Reshape each neuron’s 768 weights into a 3×16×16 filter. It reads the same pixels and adds the same bias. Conv2d applies all 192 filters at every patch location, giving a 192-channel feature grid.
```

**Caption:** Reshape each neuron’s 768 weights into a 3×16×16 filter. It reads the same pixels and adds the same bias. Conv2d applies all 192 filters at every patch location, giving a 192-channel feature grid.

**Speaker notes**

What does one output channel represent?
One of the 192 learned patch features, computed at all 14×14 locations using one shared filter and bias. The filter spans all three RGB channels.

### Page 100 — Reshape the weights; keep the same parameter count

```text
Identical numbers, stored with different shapes
Parameter
Linear
Conv2d
Weight shape
192 × 768
192 × 3 × 16 × 16
Weights
147,456
147,456
Biases
192
192
Total
147,648
147,648
Match the input order: each filter flattens R, then G, then B.
with torch.no_grad():
    linear.weight.copy_(conv.weight.flatten(1))
    linear.bias.copy_(conv.bias)
Both implementations have 147,456 weights and 192 biases. Copying the exact weights and biases makes their computations equivalent. More patches or images create more output values, while the learned parameter count stays 147,648.
```

**Caption:** Both implementations have 147,456 weights and 192 biases. Copying the exact weights and biases makes their computations equivalent. More patches or images create more output values, while the learned parameter count stays 147,648.

**Speaker notes**

Would two independently initialized layers produce the same features?
No. Equal parameter counts are not enough. They must use the same weights, biases and pixel order. Flatten each Conv2d filter to obtain the corresponding row of Linear.weight.

### Page 101 — Same pixel × same weight, in both implementations

```text
Dog patch P63 · first output feature · indices below start at 0
Flatten order: R₀ … R₂₅₅  |  G₀ … G₂₅₅  |  B₀ … B₂₅₅
Green pixel at (row 2, column 3) → flat index 256 + 2×16 + 3 = 291
Linear: one neuron
same 768 pixels
R₀
R₁
…
G₃₅
…
B₂₅₅
W[0,291] selected
w₀
w₁
…
w₂₉₁
…
w₇₆₇
×
×
…
×
…
×
Conv2d: one filter, flattened
same 768 pixels
R₀
R₁
…
G₃₅
…
B₂₅₅
W[0,1,2,3] selected
w₀
w₁
…
w₂₉₁
…
w₇₆₇
×
×
…
×
…
×
Selected product in both:  −0.623529 × 0.019536 = −0.012181
Add all 768 products + the same bias (−0.414880) → −0.851541
Both paths pair the same 768 pixels with the same 768 weights, sum their products, and add the same bias.
```

**Caption:** Both paths pair the same 768 pixels with the same 768 weights, sum their products, and add the same bias.

**Speaker notes**

Which Conv2d weight corresponds to Linear.weight[0,291]?
conv.weight[0,1,2,3]: output 0, green channel, patch row 2, column 3. R, G and B each contain 256 pixels in row-major order.

### Page 102 — Verify it on both photographs: the features match

```text
Same pretrained weights · same prepared dog and cat images
P63 · first 3 of 192 features
Linear
Conv2d
Dog
[−0.852, 1.339, 0.504, …]
[−0.852, 1.339, 0.504, …]
768 → 192
Cat
[−0.189, 0.104, −0.208, …]
[−0.189, 0.104, −0.208, …]
768 → 192
Checked all 2 × 196 × 192 = 75,264 features.
Largest float32 difference: 5.2e-06. Same result within rounding.
Both produce B × 196 × 192 rows for the same classifier.
Using identical pretrained weights, both paths produce the same dog and cat features within floating-point rounding. Conv2d packages patch extraction and projection into one image operation. Flatten and transpose its grid before adding CLS and positions.
```

**Caption:** Using identical pretrained weights, both paths produce the same dog and cat features within floating-point rounding. Conv2d packages patch extraction and projection into one image operation. Flatten and transpose its grid before adding CLS and positions.

**Speaker notes**

Why use Conv2d if Linear can compute the same thing?
Conv2d accepts the image grid directly and uses optimized convolution implementations. It avoids explicitly constructing a patch matrix in our code. This is a choice of implementation, not a change in the learned model.

### Page 103 — Why package patch projection as Conv2d?

```text
One image operation applies all 192 filters at all 196 locations.
Linear route
Image grid
B × 3 × 224 × 224
Explicit patch rows
B × 196 × 768
Shared Linear
B × 196 × 192
Conv2d route
Image grid
B × 3 × 224 × 224
Conv2d
B × 192 × 14 × 14
Flatten + transpose
B × 196 × 192
Less reshaping code; optimized convolution backends. Same learned model.
Conv2d accepts the image layout directly and handles the repeated patch computations. It avoids an explicit unfolded patch tensor in our code and uses optimized convolution backends. Actual speed and memory depend on the hardware.
```

**Caption:** Conv2d accepts the image layout directly and handles the repeated patch computations. It avoids an explicit unfolded patch tensor in our code and uses optimized convolution backends. Actual speed and memory depend on the hardware.

**Speaker notes**

Does Conv2d learn fewer parameters or a different function?
No. The same 147,648 parameters compute the same outputs. The advantage is a convenient image operation and backend support; benchmark actual speed and memory.

### Page 104 — First turn the feature grid into patch rows

```text
# x: (B, 3, 224, 224)
B = x.shape[0]                          # integer: number of images
grid = self.patch(x)                    # (B, 192, 14, 14)
rows = grid.flatten(2)                  # (B, 192, 196)
rows = rows.transpose(1, 2)             # (B, 196, 192)
Feature grid
B × 192 × 14 × 14
Flatten spatial axes
B × 192 × 196
Features last
B × 196 × 192
Flatten combines the 14×14 spatial grid into 196 locations. Transpose puts the 192 features last. Each row now describes one patch.
```

**Caption:** Flatten combines the 14×14 spatial grid into 196 locations. Transpose puts the 192 features last. Each row now describes one patch.

**Speaker notes**

Did flatten mix images or merge their features?
No. flatten(2) starts at axis 2. The batch and 192 feature channels remain separate.

### Page 105 — Then prepend CLS and add position

```text
# Continue with rows: (B, 196, 192)
cls = self.cls.expand(B, -1, -1)        # (B, 1, 192)
rows = torch.cat([cls, rows], dim=1)    # (B, 197, 192)
return rows + self.pos                  # (B, 197, 192)
# self.cls: (1, 1, 192)   self.pos: (1, 197, 192)
Patch rows
B × 196 × 192
Prepend one CLS
B × 197 × 192
Add position
B × 197 × 192
Expand reuses the learned CLS start across the batch. Concatenation adds one row. Position addition changes the numbers while keeping 197 rows and 192 features.
```

**Caption:** Expand reuses the learned CLS start across the batch. Concatenation adds one row. Position addition changes the numbers while keeping 197 rows and 192 features.

**Speaker notes**

Which operation changes the number of rows?
torch.cat adds CLS along dim=1. Adding positions leaves the shape unchanged.

### Page 106 — Make queries, keys and values for three heads

```text
B, N, D = x.shape                         # B, 197, 192
qkv = self.qkv(x)                        # B, 197, 576
qkv = qkv.reshape(B, N, 3, 3, 64)        # B, N, QKV, heads, features
q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
# q, k, v: each (B, 3, 197, 64)
Input rows
B × 197 × 192
QKV projection
B × 197 × 576
Split Q, K, V
each B × 3 × 197 × 64
LOCAL ROLES
Q / receiver
K / source
V / message
Project each row once to produce all queries, keys and values. Split the 576 outputs into three roles, each with three heads of width 64.
```

**Caption:** Project each row once to produce all queries, keys and values. Split the 576 outputs into three roles, each with three heads of width 64.

**Speaker notes**

What do the two size-3 axes mean?
One selects Q, K or V. The other selects the attention head. B always remains a separate image axis.

### Page 107 — Compute one message for every query in every head

```text
scores = (q @ k.transpose(-2, -1)) / 8  # B, 3, 197, 197
weights = scores.softmax(dim=-1)       # B, 3, 197, 197
messages = weights @ v                # B, 3, 197, 64
Query–key scores
197 × 197
Softmax over sources
197 × 197
Weighted value sums
197 × 64
LOCAL ROLES
Q / receiver
K / source
V / message
For each image and head, every query scores all 197 sources. Softmax normalizes each query row. Multiplying by V combines the sources into one 64-feature message per query.
```

**Caption:** For each image and head, every query scores all 197 sources. Softmax normalizes each query row. Multiplying by V combines the sources into one 64-feature message per query.

**Speaker notes**

Along which axis do the weights sum to one?
The last axis: source keys. Every receiving query gets its own distribution.

### Page 108 — Join the head messages and project back to 192

```text
joined = messages.transpose(1, 2)     # B, 197, 3, 64
joined = joined.reshape(B, N, D)      # B, 197, 192
return self.proj(joined)             # B, 197, 192
Three messages per row
3 × 64
Concatenate features
192
Output projection
192
LOCAL ROLES
Q / receiver
K / source
V / message
Move the head axis beside its features, then join 3×64 into 192. A learned output projection mixes these features before the residual addition.
```

**Caption:** Move the head axis beside its features, then join 3×64 into 192. A learned output projection mixes these features before the residual addition.

**Speaker notes**

Do we concatenate different images or different query rows?
Neither. Each image and query keeps its own three head messages. Only head features are joined.

### Page 109 — Build the layers inside one Transformer block

```text
self.norm1 = nn.LayerNorm(192, eps=1e-6)
self.attn = Attention()
self.norm2 = nn.LayerNorm(192, eps=1e-6)
self.mlp = nn.Sequential(
    nn.Linear(192, 768), nn.GELU(),
    nn.Linear(768, 192))
Attention
mix information across rows
MLP
192 → 768 → 192
Create two normalization layers, one attention module and one MLP. The MLP expands and then restores the feature width for each row.
```

**Caption:** Create two normalization layers, one attention module and one MLP. The MLP expands and then restores the feature width for each row.

**Speaker notes**

Does the 768-wide hidden layer change the number of patches?
No. Only the feature axis expands. B and N stay unchanged.

### Page 110 — Use the two residual paths in order

```text
# x: (B, 197, 192)
x = x + self.attn(self.norm1(x))
x = x + self.mlp(self.norm2(x))
return x
Input x
B × 197 × 192
Attention + input
B × 197 × 192
MLP + input
B × 197 × 192
Normalize, compute the attention update, and add the input. Then normalize the updated rows, compute the MLP update, and add them again.
```

**Caption:** Normalize, compute the attention update, and add the input. Then normalize the updated rows, compute the MLP update, and add them again.

**Speaker notes**

Which x enters the second line?
The x already updated by attention. Each branch returns the same shape as the input it is added to.

### Page 111 — Create twelve blocks with separate learned parameters

```text
self.blocks = nn.ModuleList([Block() for _ in range(12)])
self.norm = nn.LayerNorm(192, eps=1e-6)
self.head = nn.Linear(192, num_classes)
Block 1
B × 197 × 192
Blocks 2–11
B × 197 × 192
Block 12
B × 197 × 192
ModuleList creates twelve distinct blocks. The final normalization and class head read the result of the entire stack.
```

**Caption:** ModuleList creates twelve distinct blocks. The final normalization and class head read the result of the entire stack.

**Speaker notes**

Are these twelve calls to one shared set of weights?
No. The list comprehension creates twelve separate Block objects.

### Page 112 — Run the stack, then read the final CLS

```text
rows = self.embed(x)                 # B, 197, 192
for block in self.blocks:
    rows = block(rows)               # B, 197, 192
summary = self.norm(rows)[:, 0]      # B, 192: final CLS
return self.head(summary)           # B, 1000 logits
All rows through 12 blocks
B × 197 × 192
Normalize; select CLS
B × 192
Class head
B × 1000
Feed each block’s output into the next. Normalize the final rows and select CLS for each image. The class head returns 1,000 scores.
```

**Caption:** Feed each block’s output into the next. Normalize the final rows and select CLS for each image. The class head returns 1,000 scores.

**Speaker notes**

Why is there no softmax inside this forward method?
Cross-entropy accepts logits directly. During inference, logits.softmax(-1) converts scores into class probabilities.

### Page 113 — Connect the loss diagram to one training step

```text
model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(images)                   # B, 1000
loss = F.cross_entropy(logits, labels)    # labels: B
loss.backward()
optimizer.step()
Images B × 3 × 224 × 224
labels B
Forward → cross-entropy
one scalar loss
backward: gradients
step: parameters change
Cross-entropy consumes logits and class-index labels. Backward computes gradients; the optimizer updates the parameters it owns. Clear previous gradients before the next batch. This code shows the procedure without claiming a trained result.
```

**Caption:** Cross-entropy consumes logits and class-index labels. Backward computes gradients; the optimizer updates the parameters it owns. Clear previous gradients before the next batch. This code shows the procedure without claiming a trained result.

**Speaker notes**

Should we apply softmax before cross_entropy?
No. It already includes log-softmax. labels is a length-B integer tensor; the model returns B×1000 logits for this example.

### Page 114 — Optional extensions

```text
OPTIONAL EXTENSIONS
Transfer · evaluation · interpretation
These experiments extend the core ImageNet classification story.
```

**Caption:** These experiments extend the core ImageNet classification story.

**Speaker notes**

Optional extensions
These experiments extend the core ImageNet classification story.

### Page 115 — Section 7 · Adapt and evaluate the classifier

```text
SECTION
07
Adapt and evaluate the
classifier
Our checkpoint predicts 1,000 ImageNet labels.
What if our labels or image domain change?
Choose trainable parameters, follow one batch, then evaluate on held-out images. We will describe the procedure without running a new experiment.
```

**Caption:** Choose trainable parameters, follow one batch, then evaluate on held-out images. We will describe the procedure without running a new experiment.

**Speaker notes**

What if our labels or image domain change?
Connect the previous result to this new question. Keep the same image classifier as the reference.

### Page 116 — What was this model trained to predict?

```text
The checkpoint already learned from labelled photographs.
Pretrain on ImageNet-21k
then fine-tune on ImageNet-1k
Current task: choose among 1,000 labels
animal breeds, objects and other categories
ViT encoder
192 final CLS features
Trained 1,000-class head
Top label: Newfoundland
Our saved dog prediction used this existing task and head.
The checkpoint was pretrained on ImageNet-21k and fine-tuned on ImageNet-1k. Its current head predicts 1,000 categories. Newfoundland is one of those labels.
```

**Caption:** The checkpoint was pretrained on ImageNet-21k and fine-tuned on ImageNet-1k. Its current head predicts 1,000 categories. Newfoundland is one of those labels.

**Speaker notes**

Did our earlier dog example train a pet classifier?
No. It ran inference with the existing ImageNet checkpoint. Its learned encoder and 1,000-class head were already available.

### Page 117 — Same photographs, a different label vocabulary

```text
Now our application asks a simpler question: dog or cat?
Old label vocabulary
Newfoundland
Our new target
dog
Old label vocabulary
Persian cat
Our new target
cat
Our pet application groups breeds into two categories. We want two scores for every image, trained and evaluated for this dog/cat task. Start by reusing the encoder’s visual features.
```

**Caption:** Our pet application groups breeds into two categories. We want two scores for every image, trained and evaluated for this dog/cat task. Start by reusing the encoder’s visual features.

**Speaker notes**

Could we reuse the visual features even though the output labels changed?
Yes. The encoder already represents visual patterns. A new head can learn how those features separate the two target classes.

### Page 118 — What if our users supply sketches?

```text
A second kind of change: same labels, different-looking inputs.
Pet photographs
New domain
Sketches · schematic examples
dog
cat
Fur texture and colour disappear; outlines become more important.
Keep two outputs. Test on sketches; use labelled sketches to adapt if needed.
The labels can stay dog and cat while the image domain changes. A photo classifier may struggle with sketches. Evaluate on the target domain, then compare head training with encoder fine-tuning.
```

**Caption:** The labels can stay dog and cat while the image domain changes. A photo classifier may struggle with sketches. Evaluate on the target domain, then compare head training with encoder fine-tuning.

**Speaker notes**

Would a different-looking image automatically require more output neurons?
No. The two labels are unchanged. The features may need adaptation because sketches remove colour and texture cues.

### Page 119 — Replace the ImageNet head with our two-class head

```text
pretrained ViT encoder
final CLS: 192 features
new nn.Linear(192, 2)
scores: [dog, cat]
384 weights + 2 biases = 386 parameters
Keep the learned visual features to begin with.
For our dog/cat task, replace the ImageNet head with a new two-output linear layer. The encoder supplies a 192-number image representation. The new 386 head parameters must learn from labelled pet images.
```

**Caption:** For our dog/cat task, replace the ImageNet head with a new two-output linear layer. The encoder supplies a 192-number image representation. The new 386 head parameters must learn from labelled pet images.

**Speaker notes**

Why can we not simply rename two of the old 1,000 outputs?
We have changed the label vocabulary. The new head learns a new mapping; the existing pretrained probabilities do not measure this proposed pet classifier.

### Page 120 — Freeze the encoder; train the new head

```text
1 · Train only the new class head
trainable
Pet photo
ViT encoder
All encoder weights fixed
patches → blocks → CLS
192 features
Linear(192, 2)
⋮
dog
cat
192×2 weights + 2 biases
loss
update head
Frozen encoder: every image still gets its own 192 features.
Only the 386 head parameters receive optimizer updates.
Snowflake: weights stay fixed. Flame: weights can learn. Train the two-class head on labelled pet images. The frozen encoder still computes different features for different images.
```

**Caption:** Snowflake: weights stay fixed. Flame: weights can learn. Train the two-class head on labelled pet images. The frozen encoder still computes different features for different images.

**Speaker notes**

When only the head is trained, do the Q/K/V weights change?
No. Trace the photograph through the blue vision encoder into 192 CLS features. Only the orange head learns. Reveal the loss update.

### Page 121 — Next option: fine-tune the last block as well

```text
If head-only training is insufficient, adapt some features too.
2 · Optional:
fine-tune the
last block too
Earlier encoder
weights stay fixed
Last block
unfreeze weights
Head
dog
cat
The loss can now change the last block and the class head.
Compare on validation data before choosing how much to unfreeze.
Unfreeze the last block and train it together with the head. Earlier encoder weights stay fixed. This lets some visual features adapt to the target data; validation determines whether it helps.
```

**Caption:** Unfreeze the last block and train it together with the head. Earlier encoder weights stay fixed. This lets some visual features adapt to the target data; validation determines whether it helps.

**Speaker notes**

When might a head alone be insufficient?
If the current features do not separate the new classes well, or the images look different, changing some encoder features may help. Evaluate rather than assume.

### Page 122 — One batch follows the same forward and backward paths

```text
batch of images
B × 3 × 224 × 224
ViT + new head
B × 2 scores
known labels
B targets
mean cross-entropy
one loss
zero gradients → forward → loss → backward → optimizer step
Repeat on training batches. Evaluate separately with weights fixed.
Pair every image with its dog/cat label. Compute the average loss for the batch, backpropagate once, and update the trainable parameters. Repeat this procedure over training batches.
```

**Caption:** Pair every image with its dog/cat label. Compute the average loss for the batch, backpropagate once, and update the trainable parameters. Repeat this procedure over training batches.

**Speaker notes**

What is shared across the images in a batch?
The model parameters are shared. Each image has its own activations and attention matrix; attention does not mix different images in the batch.

### Page 123 — Three stages: learn, adapt, then predict

```text
1 · Pretrain
Many labelled images
Learn visual features
All weights learn
2 · Adapt
Our images + labels
Learn the new task
Chosen weights learn
3 · Infer
One new image
Predict its label
Weights stay fixed
Orange connections and flames mark learning. Blue connections and snowflakes mark fixed parameters. During inference, a new image changes the features while the stored weights stay fixed.
```

**Caption:** Orange connections and flames mark learning. Blue connections and snowflakes mark fixed parameters. During inference, a new image changes the features while the stored weights stay fixed.

**Speaker notes**

During which stages can the optimizer change weights?
Pretraining and adaptation use learning signals. Inference only computes outputs using the learned parameters. Reveal each stage and follow the image-to-network-to-output arrows.

### Page 124 — What happens when we classify a new photograph?

```text
One forward pass through the trained model
Input photograph
Trained ViT
192 features
Final CLS
depends on this image
Trained class head
dog score
cat score
192 inputs → 2 scores
Softmax → two probabilities → choose the larger one.
The image produces its own CLS features and class scores. Both snowflakes mark fixed weights. Inference computes a prediction without a label, loss or optimizer update.
```

**Caption:** The image produces its own CLS features and class scores. Both snowflakes mark fixed weights. Inference computes a prediction without a label, loss or optimizer update.

**Speaker notes**

Do we need to know the new image’s label to run inference?
No. First reveal the image-dependent CLS features, then the trained head and its two scores. Labels are needed later to evaluate the prediction.

### Page 125 — How would we check whether the classifier learned?

```text
Before fitting: separate training, validation and test images
TRAIN
update weights
VALIDATION
choose settings
TEST
final held-out check
Inspect: dog → dog    dog → cat    cat → dog    cat → cat
Accuracy = correct predictions / number of test images
Keep mistakes beside correct examples. A confident score can still be wrong.
Use held-out images to measure accuracy and the two kinds of confusion. Inspect successes and mistakes, including confident errors. A probability on one photograph is not the classifier’s test accuracy.
```

**Caption:** Use held-out images to measure accuracy and the two kinds of confusion. Inspect successes and mistakes, including confident errors. A probability on one photograph is not the classifier’s test accuracy.

**Speaker notes**

Could we choose the best epoch by repeatedly looking at test accuracy?
Use validation for selection. Reserve the test set for the chosen procedure; related or near-duplicate images must not leak across splits.

### Page 126 — Section 8 · Return to the real photographs

```text
SECTION
08
Return to the real photographs
Training and inference have different jobs.
What do the saved photo predictions establish?
Return to the saved dog and cat predictions. These outputs come from the ImageNet checkpoint, not from a newly trained pet classifier.
```

**Caption:** Return to the saved dog and cat predictions. These outputs come from the ImageNet checkpoint, not from a newly trained pet classifier.

**Speaker notes**

What do the saved photo predictions establish?
Connect the previous result to this new question. Keep the same image classifier as the reference.

### Page 127 — Which pixels are we giving the real model?

```text
original photograph → supplied resize / crop → normalize RGB
The model receives the square crop on the right, after RGB normalization.
```

**Caption:** The model receives the square crop on the right, after RGB normalization.

**Speaker notes**

Are we feeding the original rectangular photograph directly to the network?
Match the original to the evaluation crop, then mention normalization.

### Page 128 — What did the model call our dog?

```text
Top 3 of 1,000 labels
Newfoundland
95.73%
Tibetan mastiff
1.63%
briard
0.67%
These probabilities come from running this photograph through the pretrained model.
```

**Caption:** These probabilities come from running this photograph through the pretrained model.

**Speaker notes**

What answer should the model assign to our opening image?
Reveal the saved top-three predictions. Compare the model’s label with the students’ original answer.

### Page 129 — Does one correct photograph tell us the accuracy?

```text
probability of the known breed: 0.957
one successful prediction
How often does that happen on new photos?
To estimate accuracy, count correct predictions over an appropriate test set.
```

**Caption:** To estimate accuracy, count correct predictions over an appropriate test set.

**Speaker notes**

Can we report 95.7% accuracy from a 95.7% probability on one dog?
Separate a model’s probability on one image from a fraction correct over many images.

### Page 130 — What happens when we give it the cat?

```text
Same checkpoint. Another photograph.
Persian cat
96.71%
Angora
1.26%
plastic bag
0.43%
Keep the same model and preprocessing. Change only the photograph.
```

**Caption:** Keep the same model and preprocessing. Change only the photograph.

**Speaker notes**

Do we change model weights when a different test image arrives?
Reveal the top three ImageNet labels and probabilities.

### Page 131 — Section 9 · Look inside the trained model

```text
SECTION
09
Look inside the trained model
A prediction tells us the model’s answer.
What can we measure inside its attention blocks?
Read attention maps, then cover parts of the photograph and measure how the prediction changes.
```

**Caption:** Read attention maps, then cover parts of the photograph and measure how the prediction changes.

**Speaker notes**

What can we measure inside its attention blocks?
Connect the previous result to this new question. Keep the same image classifier as the reference.

### Page 132 — Similar patch features can connect distant image regions

```text
Query · P74
Feature similarity · block 12
Similar features
across the dog
P74 ↔ P82: 0.902
Compare the final
192-number patch vectors.
Violet outline: selected patch
Gold: positive cosine · fixed −1 to 1
Feature similarity ≠ attention weight
Explore all nine examples ↗
The selected ear-side patch matches patches on the other side of the dog. This measures representation similarity, rather than which values attention mixes.
```

**Caption:** The selected ear-side patch matches patches on the other side of the dog. This measures representation similarity, rather than which values attention mixes.

**Speaker notes**

Does a similar representation imply a large attention weight?
No. This compares contextual patch features after block 12 using cosine similarity. P82 is a measured high-similarity patch; a correspondence is not a guaranteed segmentation.

### Page 133 — Keep the query fixed; change only the attention head

```text
Query · P74
Same query · block 4
Head 1
P60: 9.09%
Head 2
P38: 2.29%
Different heads gather different mixtures.
Explore all nine examples ↗
Teal shows source weights. Both maps use the same 0–10.35% scale. Within each head, all 197 source weights—including CLS—sum to 100%.
```

**Caption:** Teal shows source weights. Both maps use the same 0–10.35% scale. Within each head, all 197 source weights—including CLS—sum to 100%.

**Speaker notes**

Why can the maps differ although the image and query location stay fixed?
Each head has its own learned query and key projections. These lead to different weights on the source value rows. Head 2’s weaker peak is shown on the same color scale.

### Page 134 — CLS gathers a message for the image summary

```text
Query = CLS
current image state
Attention · block 12 · head 1
CLS reads
patch information
P64: 24.86%
of this head’s weight
Teal: 0 → 24.86%
One attention head is one part of the classifier.
Explore all nine examples ↗
The query is CLS. Its weights select a mixture of all source value rows. Later operations and the classifier turn the final CLS features into class scores.
```

**Caption:** The query is CLS. Its weights select a mixture of all source value rows. Later operations and the classifier turn the final CLS features into class scores.

**Speaker notes**

Is this a complete explanation of the Newfoundland prediction?
No. It is the last block’s first attention head. P64 receives 24.86% of this query’s source weight; CLS itself receives 0.08%. Other heads, residuals, MLPs and the class head also contribute.

### Page 135 — “Cover” means replace these pixels with gray

```text
Original input x
Copy with gray pixels
112 × 112 pixels replaced
RGB fill = (0.5, 0.5, 0.5)
After normalization:
(0.5 − 0.5) / 0.5 = 0
Same 224 × 224 image size.
All 196 patches remain.
x = transform(photo).unsqueeze(0)  # (1, 3, 224, 224)
covered = x.clone()                # a separate copy
covered[:, :, :112, :112] = 0       # all RGB channels, top left
Copy the normalized input, then overwrite one quadrant. The rest of the image stays unchanged; no patch rows are deleted.
```

**Caption:** Copy the normalized input, then overwrite one quadrant. The rest of the image stays unchanged; no patch rows are deleted.

**Speaker notes**

Does zero here mean black pixels, or gray pixels?
x is already resized, center-cropped and normalized by this checkpoint’s evaluation transform. Its mean and standard deviation are both 0.5 per channel. Zero in x therefore means RGB 0.5 before normalization.

### Page 136 — Run the covered image through the same trained model

```text
Two independent forward passes
P(Newfoundland)
Original x
Same trained ViT
patches → blocks → CLS → head
95.73%
Covered copy
Same trained ViT
patches → blocks → CLS → head
83.00%
Same weights; recompute all activations
model.eval()
with torch.inference_mode():
    p_before = model(x).softmax(-1)[0]        # (1000,)
    p_after = model(covered).softmax(-1)[0]   # (1000,)
before, after = p_before[256], p_after[256]   # Newfoundland
Recompute patch features, attention, CLS and class scores. Softmax gives 1,000 probabilities; compare the Newfoundland entry in both runs. The weights stay fixed.
```

**Caption:** Recompute patch features, attention, CLS and class scores. Softmax gives 1,000 probabilities; compare the Newfoundland entry in both runs. The weights stay fixed.

**Speaker notes**

What must be recomputed after the pixels change?
The entire forward pass runs again. Reveal the measured change from 95.73% to 83.00%: 12.73 percentage points lower. The top label remains Newfoundland. No backward pass or optimizer step occurs.

### Page 137 — Four covers, four new forward passes

```text
Original P(Newfoundland): 95.73%
Top left
83.00%
−12.73 points
Top right
79.62%
−16.11 points
Bottom left
84.54%
−11.19 points
Bottom right
82.16%
−13.57 points
Top-right covering causes the largest drop: 16.11 percentage points. All four images still predict Newfoundland. Attention shows internal mixing; occlusion tests how a changed input changes the prediction.
```

**Caption:** Top-right covering causes the largest drop: 16.11 percentage points. All four images still predict Newfoundland. Attention shows internal mixing; occlusion tests how a changed input changes the prediction.

**Speaker notes**

Which intervention changes this probability most?
The top-right cover gives the largest drop among these four tests. Keep the target class, fill value and model fixed. This does not isolate a semantic object part; the next slide uses smaller covers.

### Page 138 — Use smaller covers to ask a more local question

```text
Large cover: 112 × 112
Small cover: 16 × 16
49 patches covered · 4 tests
1 patch covered · 196 tests
Fresh original → cover one region → rerun → measure the drop
Only the cover size changes. The trained model still uses 16×16 patches and a 224×224 input. Every test starts from the original image.
```

**Caption:** Only the cover size changes. The trained model still uses 16×16 patches and a 224×224 input. Every test starts from the original image.

**Speaker notes**

Are we changing the model’s patch size, or the region we cover?
Keep the model fixed. Cover one of its 196 patch locations at a time, producing 196 independent images. A smaller cover probes a smaller region, but may have a smaller effect because other useful information remains.

### Page 139 — Smaller covers reveal local sensitivity

```text
One test: cover P78
All 196 test results
Each square = one separate test
Largest drop: P78
95.73% → 91.89%
Down 3.84 percentage points
Red: probability falls
Blue: probability rises
Drop scale: −4 to +4 points
All 196 tests still predict Newfoundland.
Smaller covers localize sensitivity. P78 causes the largest drop here, but no single-patch cover changes the top label. This map measures probability changes, not attention weights.
```

**Caption:** Smaller covers localize sensitivity. P78 causes the largest drop here, but no single-patch cover changes the top label. This map measures probability changes, not attention weights.

**Speaker notes**

Does a small probability drop mean a patch contains no useful information?
P78 causes the largest measured drop: 3.84 percentage points. Read each square as a different forward pass with one gray patch. Other regions can retain useful clues; these drops are not additive.

### Page 140 — Section 10 · The cost of smaller patches

```text
SECTION
10
The cost of smaller patches
Every query compares all source rows.
What happens when we make the patches smaller?
Halve the patch width and height. Predict the change in attention work before calculating it.
```

**Caption:** Halve the patch width and height. Predict the change in attention work before calculating it.

**Speaker notes**

What happens when we make the patches smaller?
Connect the previous result to this new question. Keep the same image classifier as the reference.

### Page 141 — What changes when the patch size is halved?

```text
32 × 32 patches
49 patches + CLS
50² = 2,500 scores / head
16 × 16 patches
196 patches + CLS
197² = 38,809 scores / head
8 × 8 patches
784 patches + CLS
785² = 616,225 scores / head
At fixed image size, half the patch width gives four times as many patch rows.
```

**Caption:** At fixed image size, half the patch width gives four times as many patch rows.

**Speaker notes**

Will attention-score count grow by four or roughly sixteen?
Count rows first, then square the total including CLS.

### Page 142 — How much matching happens inside the tiny real model?

```text
196 patch rows + CLS
197 × 197 = 38,809 scores per head
3 heads × 12 blocks
1,397,124 scores for one image
Even this small ViT compares many pairs of rows.
```

**Caption:** Even this small ViT compares many pairs of rows.

**Speaker notes**

How can a 14-by-14 patch grid produce more than a million scores?
Count pairs, then heads, then blocks. Keep token count separate from pixel count.

### Page 143 — What happens if we use a larger image?

```text
image width / height
patch width / height
Predict the token count first.
Image size
224
384Patch size
32
16
8
Work out the token count before you change the controls.
```

**Caption:** Work out the token count before you change the controls.

**Speaker notes**

What happens to the 197-token sequence at resolution 384?
Ask for a count, change the control, and compare with the prediction.

### Page 144 — The whole ViT: pixels → context → one label

```text
224 × 224 RGB
Shared patch layer
+ CLS + position
196 patches + 1 CLS
12 Transformer blocks
197 × 192
All rows gain context
Final LN
read CLS
192 features
Linear head
1,000 scores
softmax → label
INSIDE EACH BLOCK
Each block has its own learned weights
E
LN
Head 1
Head 2
Head 3
Join + project
192 features
+
LN
MLP
192→768→192
+
Keep the row + add an update
E′
Attention: mix across rows
One shared patch layer builds the rows. Twelve blocks refine every row. The classifier reads final CLS to score image labels.
```

**Caption:** One shared patch layer builds the rows. Twelve blocks refine every row. The classifier reads final CLS to score image labels.

**Speaker notes**

Can you trace the image path, then name the two updates inside a block?
Trace projection, CLS and position, twelve blocks, final normalization, CLS and class head. Open one block: normalize, three attention heads, concatenate and project, residual; normalize, MLP, residual.

### Page 145 — Four ideas to carry forward

```text
1 · Content + location
patch features
+
position
Shared projection; each slot has a position.
2 · Every row gains context
CLS
CLS′
P1
P1′
P2
P2′
Attention
+ MLP
Attention mixes; the MLP transforms.
3 · The task teaches the summary
Final CLS
Class head
Label loss
Train with loss; infer with fixed weights.
4 · More tokens cost more
197²
785²
≈16×
224 → 448
patch = 16
Twice the side length: ≈16× scores.
Pixels supply content. Position supplies location. Attention builds context. Training makes the final representation useful for the task.
```

**Caption:** Pixels supply content. Position supplies location. Attention builds context. Training makes the final representation useful for the task.

**Speaker notes**

Which operation supplies content, location, context and the learning signal?
Patch projection supplies content; learned positions identify slots; attention mixes source values for every query; label loss trains the whole classifier. More tokens increase the number of query–key comparisons quadratically.

### Page 146 — Our classifier stores a vector for each known label

```text
Final image CLS
h · 192 features
Newfoundland
learned class vector wₖ
hᵀwₖ + bₖ
Persian cat
learned class vector wₖ
hᵀwₖ + bₖ
… 998 other labels
learned class vector wₖ
hᵀwₖ + bₖ
The head stores one learned vector and one bias per label.
A class score is a dot product with its learned weight vector, plus a bias. The current head has 1,000 fixed label slots. New tasks can train a replacement head.
```

**Caption:** A class score is a dot product with its learned weight vector, plus a bias. The current head has 1,000 fixed label slots. New tasks can train a replacement head.

**Speaker notes**

Where does the vector for the label Newfoundland come from?
It is a learned row of the classifier weight matrix. The class name itself is not read by this classifier.

### Page 147 — What if a class vector could come from language?

```text
Image encoder
→ final readout hᵢ
hᵢ × Wᵢ
learned projection
u
unit length
“a photo
of a dog”
Text encoder
→ final readout hₜ
hₜ × Wₜ
learned projection
v
unit length
u · v
cosine
Separate encoders. Aligned vectors. Compare in one shared space.
CLIP trains image and text representations to match. Learned projections and unit normalization make their vectors comparable; prompts can then describe candidate classes.
```

**Caption:** CLIP trains image and text representations to match. Learned projections and unit normalization make their vectors comparable; prompts can then describe candidate classes.

**Speaker notes**

Could we replace today’s class weights with arbitrary text embeddings?
Not directly. Image and text encoders need compatible projections and joint alignment training. CLIP compares normalized image and text vectors, with a learned score scale.

### Page 148 — The whole Vision Transformer in one figure

```text
1 · RGB image
3 × 224 × 224
16 × 16 patches
flatten → 196 × 768
Shared projection
768 → 192 + bias
Prepend learned CLS
196 + 1 = 197 rows
+ learned positions
E⁰: 197 × 192
2 · Blocks 1–12 · all 197 rows, each 192 features wide
1
2
3
4
5
6
7
8
9
10
11
12
E¹²
INSIDE BLOCK 1
Blocks 2–12 repeat these operations with their own weights.
skip: E
E
197 × 192
LN 1
197 × 192
Head 1: QKV
each 197 × 64
QKᵀ / √64
197 × 197
A = softmax
over source keys
AV
197 × 64
V
Head 2: QKV
each 197 × 64
QKᵀ / √64
197 × 197
A = softmax
over source keys
AV
197 × 64
V
Head 3: QKV
each 197 × 64
QKᵀ / √64
197 × 197
A = softmax
over source keys
AV
197 × 64
V
Concat
3 × 64 = 192
W_O + b
192 → 192
+
U
3 parallel heads
No causal mask
U
197 × 192
LN 2
197 × 192
MLP: Linear + bias
192 → 768
GELU
197 × 768
Linear + bias
768 → 192
+
skip: U
E¹
197 × 192
3 · After block 12: normalize → CLS → class scores → prediction / loss
Final LN
197 × 192
Select CLS
1 × 192
Linear head + bias
192 → 1,000 logits
Softmax
1,000 probs
Newfoundland
top label · 95.73%
Label loss
L = 0.0437
Cross-entropy(logits, y) · y = Newfoundland
Follow the numbered path. The expanded block shows three parallel heads, both residual additions, and the MLP. All 12 blocks keep every row. Final CLS feeds the classifier; the known label enters only at the loss.
```

**Caption:** Follow the numbered path. The expanded block shows three parallel heads, both residual additions, and the MLP. All 12 blocks keep every row. Final CLS feeds the classifier; the known label enters only at the loss.

**Speaker notes**

Can you trace one image through every operation without skipping a box?
Start at pixels, then patch projection, CLS and positions. Trace all 12 blocks. Open block 1: normalize, three parallel Q/K/V → scores → row softmax → AV lanes, concatenate, output projection, add the original E. Normalize U, apply Linear → GELU → Linear, and add U. Continue through the remaining distinct blocks, final normalization, final CLS and the class head. Softmax gives the prediction; logits and the known label give cross-entropy.

### Page 149 — To images: An Image is Worth 16 × 16 Words

```text
Dosovitskiy et al. (2020; ICLR 2021), Figure 1 · arXiv:2010.11929
Image patches become tokens. The encoder builds a summary for classification.
```

**Caption:** Image patches become tokens. The encoder builds a summary for classification.

**Speaker notes**

What changed between the two paper figures?
Read this figure from the image upward. Reveal the patch projection, the encoder, and the classification readout in that order. Point to the expanded encoder on the right. Then begin our dog-photo example.

### Page 150 — Create 192 trainable numbers for CLS

```text
The model uses 192 features per row.
Give the extra row the same width.
s₁
1
s₂
2
s₃
3
…
…
s₁₉₂
192
Initialize once with small random values; mark them trainable.
cls = nn.Parameter(torch.randn(1, 1, 192) * 0.02)
Before training, initialize one 192-number vector. It is stored in the model, like a learned token embedding in text. The numbers do not come from this photograph. Training will adjust them.
```

**Caption:** Before training, initialize one 192-number vector. It is stored in the model, like a learned token embedding in text. The numbers do not come from this photograph. Training will adjust them.

**Speaker notes**

Why 192, and who supplies the initial numbers?
The chosen embedding width is 192. The initialization routine supplies the numbers once, before training; no image pixels are needed.

### Page 151 — The image label teaches the starting CLS numbers

```text
Training idea · imagine this is a labelled training example
Dog photograph
Patch rows + CLS
Transformer blocks
Class scores
prediction
Compare with label
loss
Known label
Newfoundland
192 starting CLS parameters
adjusted by the optimizer
backward through classifier + blocks
The loss trains CLS along with the other trainable model parameters.
The label supplies a loss on the prediction. Backpropagation reaches the starting CLS vector through the classifier and blocks. The optimizer adjusts its 192 parameters, along with other trainable weights, across many labelled images.
```

**Caption:** The label supplies a loss on the prediction. Backpropagation reaches the starting CLS vector through the classifier and blocks. The optimizer adjusts its 192 parameters, along with other trainable weights, across many labelled images.

**Speaker notes**

Do we need 192 target numbers for CLS?
No. The image’s class label provides the loss; the chain rule supplies gradients for the starting vector.

## Interactive lab — all nine examples

### Example 1: Background finds background
similarity · block 12 · head 1 · query 182 · source 70

The background query finds gold around the dog.

Learned features can separate foreground and background.

Purple marks the lower-right background patch. Gold appears around the dog, with less on its body. This is a similarity pattern, not a predicted segmentation mask.

### Example 2: Check a true corner
similarity · block 12 · head 1 · query 1 · source 183

The top-left corner matches the other three corners.

A strong match need not be an object part.

The strongest matches have cosine similarity almost 1. Inspect where the matches occur before assigning them a meaning.

### Example 3: One ear finds the other side
similarity · block 12 · head 1 · query 74 · source 82

The ear query matches the other side of the head.

Similar features can connect distant patches.

Purple marks P74 on the left side of the dog’s head. P82 on the opposite side is the strongest other match. This correspondence is specific to this trained model and photograph; it is not guaranteed for every image.

### Example 4: Move the reference to the nose
similarity · block 12 · head 1 · query 63 · source 77

The nose query finds nearby face and muzzle patches.

Change the query; change the matches.

The query moves to P63 near the nose. The map compares every patch representation with this selected reference.

### Example 5: Follow the body’s dark fur
similarity · block 12 · head 1 · query 147 · source 162

The chest query finds similar fur lower on the dog.

One object can contain several kinds of features.

The reference is now on the chest. Similarity need not highlight the entire dog uniformly: the lower body and nose have different features.

### Example 6: Rewind to block 1
similarity · block 1 · head 1 · query 74 · source 82

Same ear pair: 0.322 here → 0.902 at block 12.

Same pixels. Different features after each block.

Keep the P74-to-P82 ear pair from example 3. Its cosine similarity is 0.322 after block 1 and 0.902 after block 12. The representation changes while the pixels stay fixed; not every pair’s similarity must increase.

### Example 7: Ask what the ear reads
attention · block 4 · head 1 · query 74 · source 60

The ear query gives P60’s value 9.09% weight.

Attention mixes values into the query’s message.

This is block 4, head 1, query P74. An attention weight scales a source’s value; feature similarity instead compares the patch representations. These are different measurements.

### Example 8: Change only the attention head
attention · block 4 · head 2 · query 74 · source 38

Same query and block. Head 2 favours P38.

Different heads gather different information.

Head 2 puts its largest patch weight on P38, at about 2.29%. Teal is rescaled within each attention map: compare percentages, not brightness, across heads.

### Example 9: Let CLS gather an image summary
attention · block 12 · head 1 · query 0 · source 64

CLS gives the face patch P64 about 25% weight.

CLS gathers image information for classification.

CLS is the query: an extra token, not an image patch. P64 receives about 24.86% of the weight in block 12, head 1. This is one head in one block, not a complete explanation of the final label.
