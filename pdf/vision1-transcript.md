# Vision Transformer — audit transcript

Companion PDF: [vision1.pdf](vision1.pdf) · **55 pages**.

Main lecture: 54 conceptual slides plus cover. The optional reference deck preserves the detailed material separately. Final reveals are captured in the PDF; progressive builds and interactions remain in HTML.

Diagram labels are listed in source order; use the PDF to judge spatial layout. Speaker notes contain the detail omitted from the projected slide.

## Page index

| PDF page | Route | Title |
|---:|---|---|
| 1 | Cover | Vision Transformer |
| 2 | #s01/1 | We already know how an encoder builds context |
| 3 | #s01/2 | What should the model predict for this photograph? |
| 4 | #s01/3 | An encoder expects vectors. How can an image supply them? |
| 5 | #s01/4 | Five questions build the architecture |
| 6 | #s02/1 | Why not make every pixel a token? |
| 7 | #s02/2 | Split the same photograph into 196 patches |
| 8 | #s02/3 | What is inside one patch? |
| 9 | #s02/4 | Flatten one channel at a time: R, then G, then B |
| 10 | #s02/5 | How do 768 pixel values become 192 features? |
| 11 | #s02/6 | Should every patch get a different projection? |
| 12 | #s02/7 | Stack the patches into one feature matrix |
| 13 | #s02/8 | What does the trained projection return? |
| 14 | #s02/9 | We now have a token for each patch |
| 15 | #s03/1 | The projection is shared. Where does location enter? |
| 16 | #s03/2 | Add WHAT and WHERE |
| 17 | #s03/3 | Which positional vector goes with each row? |
| 18 | #s04/1 | Many patch representations, one image label |
| 19 | #s04/2 | Same CLS idea, different input tokens |
| 20 | #s04/3 | Every image starts with the same learned CLS vector |
| 21 | #s04/4 | After the encoder, CLS depends on the image |
| 22 | #s04/5 | CLS reads the current patch states at every block |
| 23 | #s04/6 | We now have the token sequence the encoder needs |
| 24 | #s05/1 | From here, reuse the encoder we already know |
| 25 | #s05/2 | Where do Q, K and V come from? |
| 26 | #s05/3 | One patch representation has three jobs |
| 27 | #s05/4 | What might different queries try to gather? |
| 28 | #s05/5 | A source supplies both a key and a value |
| 29 | #s05/6 | Several source values form one receiver’s message |
| 30 | #s05/7 | ViT uses full attention, not causal attention |
| 31 | #s05/8 | How does one CLS query collect one message? |
| 32 | #s05/9 | Three heads gather three messages for each token |
| 33 | #s05/10 | Where does the trained query look? |
| 34 | #s05/11 | Add the attention update to the original embedding |
| 35 | #s05/12 | The MLP adds one more update to each embedding |
| 36 | #s05/13 | Every block updates the patches and CLS again |
| 37 | #s05/14 | What is stored, and what changes with the image? |
| 38 | #s06/1 | After the blocks, read the updated CLS embedding |
| 39 | #s06/2 | How do 192 features score 1,000 classes? |
| 40 | #s06/3 | What does the model predict for our photograph? |
| 41 | #s06/4 | What happens if we hide one quarter of the image? |
| 42 | #s06/5 | Change the pixels, then run the model again |
| 43 | #s06/6 | Cover each quarter, then compare predictions |
| 44 | #s06/7 | Would smaller covers tell us more? |
| 45 | #s07/1 | Two ways to gather image context |
| 46 | #s07/2 | Which assumptions are built into the architecture? |
| 47 | #s07/3 | How can these built-in assumptions help? |
| 48 | #s07/4 | How much detail should one token cover? |
| 49 | #s07/5 | What does a finer grid cost? |
| 50 | #s07/6 | Once an image becomes tokens, the encoder is familiar |
| 51 | #s07/7 | Follow the whole model through its shapes |
| 52 | #s07/8 | The whole ViT in six lines |
| 53 | #s07/9 | More detail, when you want it |
| 54 | #s07/10 | The classifier stores a learned vector per known class |
| 55 | #s07/11 | What if our class vocabulary could come from words? |

## Transcript

### Page 1 — Vision Transformer

Five questions turn one photograph into tokens, context and an ImageNet prediction.

### Page 2 — We already know how an encoder builds context

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
We can reuse the encoder once we turn the image into tokens.
```

**Caption:** We can reuse the encoder once we turn the image into tokens.

**Speaker notes**

We already know how an encoder builds context
The previous lecture, Transformers beyond next-token prediction, compared these three model families. The encoder on the left uses full attention to read the complete input. Here we work out how to give it an image.

### Page 3 — What should the model predict for this photograph?

```text
Pretrained ViT-Tiny
ImageNet-1k
1,000 possible classes
Newfoundland: 95.73%
Our input photograph
One measured forward pass
This photograph comes from Oxford-IIIT Pet. The model chooses from 1,000 ImageNet classes.
```

**Caption:** This photograph comes from Oxford-IIIT Pet. The model chooses from 1,000 ImageNet classes.

**Speaker notes**

What should the model predict for this photograph?
Our photograph is newfoundland_31 from Oxford-IIIT Pet. We use vit_tiny_patch16_224.augreg_in21k_ft_in1k throughout; it assigns Newfoundland a probability of 0.9572675228. Oxford-IIIT Pet provides the image, while ImageNet defines the classifier’s output labels. The optional extension shows how to adapt the model to dog/cat classification.

### Page 4 — An encoder expects vectors. How can an image supply them?

```text
?
vector 1
vector 2
…
vector N
The encoder takes a sequence of vectors. We need to turn the pixels into that sequence.
```

**Caption:** The encoder takes a sequence of vectors. We need to turn the pixels into that sequence.

**Speaker notes**

An encoder expects vectors. How can an image supply them?
The image is an RGB array arranged in a spatial grid. The encoder takes rows of features. Our first job is to turn that grid into feature rows; Q, K and V come later.

### Page 5 — Five questions build the architecture

```text
1
Make tokens
Pixels → vectors
2
Keep location
Which patch goes where?
3
Choose a readout
Many rows → one label
4
Share information
Use the known encoder
5
Predict a class
Summary → class scores
Each question gives us a reason to add the next part of the model.
```

**Caption:** Each question gives us a reason to add the next part of the model.

**Speaker notes**

Five questions build the architecture
We will answer each question before introducing the component that solves it. By the end, we will have followed one photograph all the way to a class prediction and seen what a fixed set of output classes leaves out.

### Page 6 — Why not make every pixel a token?

```text
Pixel tokens
224 × 224
50,176 tokens
16 × 16 patches
14 × 14
196 tokens
Attention compares token pairs: its score matrix grows as N².
Using patches also keeps the number of attention comparisons manageable.
```

**Caption:** Using patches also keeps the number of attention comparisons manageable.

**Speaker notes**

Why not make every pixel a token?
Pixel tokens would give 50,176 spatial positions, each initially holding RGB. The fine grid is schematic. Patch tokens give 196 positions before CLS. Attention score counts scale quadratically with sequence length. This comparison ignores CLS to emphasize the scale, not memory implementation details.

### Page 7 — Split the same photograph into 196 patches

```text
224 ÷ 16 = 14 patches per side
14 × 14 = 196 patches
Each non-overlapping crop is 16 × 16 pixels, with all three colour channels.
```

**Caption:** Each non-overlapping crop is 16 × 16 pixels, with all three colour channels.

**Speaker notes**

Split the same photograph into 196 patches
The photograph has already been resized and centre-cropped using the checkpoint transform. We now work with its 224 by 224 RGB input. Patches are ordered row by row.

### Page 8 — What is inside one patch?

```text
P63
16 × 16 × 3
768 input numbers
A patch has 768 pixel values. We will add its location separately.
```

**Caption:** A patch has 768 pixel values. We will add its location separately.

**Speaker notes**

What is inside one patch?
Patch P63 lies at zero-based grid row 4, column 6. The crop illustration is magnified. Before projection, use the checkpoint normalization: RGB to [0,1], subtract 0.5 and divide by 0.5 per channel.

### Page 9 — Flatten one channel at a time: R, then G, then B

```text
R channel
16 × 16 values
256 numbers
G channel
16 × 16 values
256 numbers
B channel
16 × 16 values
256 numbers
xᵢ = [ all R pixels | all G pixels | all B pixels ] ∈ ℝ⁷⁶⁸
Flattening only rearranges the values; it does not learn features.
```

**Caption:** Flattening only rearranges the values; it does not learn features.

**Speaker notes**

Flatten one channel at a time: R, then G, then B
Read pixels row by row within each channel, then concatenate R, G and B. This is channel-major ordering, matching PyTorch Unfold and flattened Conv2d kernels. Pixel-major ordering is also valid if the weight columns are permuted to match.

### Page 10 — How do 768 pixel values become 192 features?

```text
xᵢ
768 values
Shared learned projection
W_E: 768 × 192
b_E: 192
cᵢ
192 features
cᵢ = xᵢ W_E + b_E
The learned projection turns those 768 pixel values into 192 features.
```

**Caption:** The learned projection turns those 768 pixel values into 192 features.

**Speaker notes**

How do 768 pixel values become 192 features?
Use row-vector notation: x_i is 1 by 768, W_E is 768 by 192, and b_E has 192 entries. This affine map has no following activation in the patch layer. PyTorch stores the Linear weight transposed, as 192 by 768.

### Page 11 — Should every patch get a different projection?

```text
x1: 768 values
same W_E, b_E
c1: 192
x63: 768 values
same W_E, b_E
c63: 192
x196: 768 values
same W_E, b_E
c196: 192
Every patch uses the same W_E and b_E.
```

**Caption:** Every patch uses the same W_E and b_E.

**Speaker notes**

Should every patch get a different projection?
Every patch passes through the same projection with its own pixel values. We learn one set of weights and biases, shared across all 196 patches. We still need a way to tell the model where each patch came from.

### Page 12 — Stack the patches into one feature matrix

```text
x₁: 768
x₂: 768
…
x₁₉₆: 768
W_E, b_E
shared
c₁: 192
c₂: 192
…
c₁₉₆: 192
X: 196 × 768
C: 196 × 192
The projection changes 768 features to 192. We still have 196 patch rows.
```

**Caption:** The projection changes 768 features to 192. We still have 196 patch rows.

**Speaker notes**

Stack the patches into one feature matrix
X contains all flattened patches. C = X W_E + b_E broadcasts the same bias across every row. The batch axis is omitted in the main lecture.

### Page 13 — What does the trained projection return?

```text
Measured P63
768 inputs
trained projection
192 output features
c₆₃
[−0.852, 1.339, 0.504, …]
These are the first three of P63’s 192 features.
```

**Caption:** These are the first three of P63’s 192 features.

**Speaker notes**

What does the trained projection return?
The first three values come from real-patch-path.json for our saved checkpoint. The dots stand for the other 189 coordinates. These features describe the patch before we add position. We have not assigned a meaning to each individual coordinate.

### Page 14 — We now have a token for each patch

```text
IMAGE
3 × 224 × 224
PATCHES
196 crops
PATCH
PROJECTION
196 × 192
+ POSITION
+ CLS
197 × 192
ENCODER
× 12
197 × 192
FINAL
CLS
192
CLASS
HEAD
1,000 logits
HERE
The image is now 196 rows, each with 192 learned features.
```

**Caption:** The image is now 196 rows, each with 192 learned features.

**Speaker notes**

We now have a token for each patch
This diagram tracks our progress through the model. Completed stages fade, and the current stage is larger. We have patch features now; next we need to add where each patch came from.

### Page 15 — The projection is shared. Where does location enter?

```text
content c₆₃
Where?
row 5, column 7
The content vector describes the crop. A position vector tells the model where it came from.
```

**Caption:** The content vector describes the crop. A position vector tells the model where it came from.

**Speaker notes**

The projection is shared. Where does location enter?
P63 is in row 5, column 7 when we count from one. The shared projection has no explicit position index. The picture itself may offer clues about location, but the learned position embedding gives the model that information explicitly.

### Page 16 — Add WHAT and WHERE

```text
content cᵢ
WHAT
+
position pᵢ
WHERE
eᵢ
eᵢ = cᵢ + pᵢ     ·     192 features
Adding position preserves the 196 × 192 shape.
```

**Caption:** Adding position preserves the 196 × 192 shape.

**Speaker notes**

Add WHAT and WHERE
Content vectors are blue and learned positional vectors amber. We add corresponding coordinates, not concatenate them. The vectors remain 192 features wide. The CLS position is added when CLS is included.

### Page 17 — Which positional vector goes with each row?

```text
P1 content
P63 content
P196 content
+
position 1
position 63
position 196
P1 + location
P63 + location
P196 + location
Learned table: one 192-number row per token position
Every image with this grid uses the same learned position table.
```

**Caption:** Every image with this grid uses the same learned position table.

**Speaker notes**

Which positional vector goes with each row?
Training updates the position table along with the rest of the model. At inference, the table stays fixed. Our checkpoint has 197 position vectors, including one for the CLS slot.

### Page 18 — Many patch representations, one image label

```text
patch 1
patch 63
…
patch 196
Which readout?
one image label
Mean pooling is another valid readout; this checkpoint was trained with CLS.
Reuse the CLS readout we already know from text.
```

**Caption:** Reuse the CLS readout we already know from text.

**Speaker notes**

Many patch representations, one image label
Averaging final patch features is a valid alternative when the model is trained for that readout. We keep the actual checkpoint’s CLS design, rather than substituting a different architecture.

### Page 19 — Same CLS idea, different input tokens

```text
TEXT
CLS
sentence tokens
encoder
final CLS
class
IMAGE
CLS
patch tokens
encoder
final CLS
class
We use patch tokens where the text model used word tokens.
```

**Caption:** We use patch tokens where the text model used word tokens.

**Speaker notes**

Same CLS idea, different input tokens
As in the text model, the starting CLS vector is a learned parameter. It is shown in amber. After the encoder, CLS depends on the input; its colour matches the text or image branch.

### Page 20 — Every image starts with the same learned CLS vector

```text
One learned CLS start
192 stored numbers
same CLS start
same CLS start
Both photographs start with exactly the same CLS vector.
```

**Caption:** Both photographs start with exactly the same CLS vector.

**Speaker notes**

Every image starts with the same learned CLS vector
Both photographs use the same trained model, starting CLS vector and CLS position vector. Only their patch inputs differ. The cat helps us see how the input changes the final representation; we are still using the original classifier.

### Page 21 — After the encoder, CLS depends on the image

```text
same CLS start
same encoder
+ these patch rows
dog image summary
same CLS start
same encoder
+ these patch rows
cat image summary
The same model produces a different CLS representation for each photograph.
```

**Caption:** The same model produces a different CLS representation for each photograph.

**Speaker notes**

After the encoder, CLS depends on the image
The arrows show the computation. The dog and cat produce different final CLS states even though they use the same model. The reference deck includes measured traces for both images; this diagram does not display numerical predictions.

### Page 22 — CLS reads the current patch states at every block

```text
CLS
state 0
196 patch rows
state 0
CLS
state 1
196 patch rows
state 1
CLS
state 2
196 patch rows
state 2
block
next block
CLS and patch rows can exchange information in both directions.
```

**Caption:** CLS and patch rows can exchange information in both directions.

**Speaker notes**

CLS reads the current patch states at every block
Each block computes all updates from its incoming states, in parallel. The next block reads the updated CLS and patch states. Arrows show allowed dependencies, including patch-to-CLS and CLS-to-patch, not guaranteed large weights.

### Page 23 — We now have the token sequence the encoder needs

```text
IMAGE
3 × 224 × 224
PATCHES
196 crops
PATCH
PROJECTION
196 × 192
+ POSITION
+ CLS
197 × 192
ENCODER
× 12
197 × 192
FINAL
CLS
192
CLASS
HEAD
1,000 logits
HERE
[ CLS ; P1 ; P2 ; … ; P196 ]  +  position  →  197 × 192
196 projected patches plus one CLS row, each with 192 features.
```

**Caption:** 196 projected patches plus one CLS row, each with 192 features.

**Speaker notes**

We now have the token sequence the encoder needs
Prepend CLS, then add a position vector to each of the 197 rows. This gives us the complete image sequence. We can now use that sequence to compute Q, K and V.

### Page 24 — From here, reuse the encoder we already know

```text
CLS
P1
…
P196
The encoder
we already know
CLS updated
P1 updated
…
P196 updated
We can now apply the same encoder operations we used for text.
```

**Caption:** We can now apply the same encoder operations we used for text.

**Speaker notes**

From here, reuse the encoder we already know
Normalization, self-attention, residual additions and the MLP all work on feature rows. The small pictures in our diagrams identify patches; the encoder itself receives their vectors.

### Page 25 — Where do Q, K and V come from?

```text
TRANSLATION · cross-attention
target-prefix states → Q
source-encoder states → K, V
ViT · self-attention
CLS + patch states
one image sequence
Q, K and V
ViT gets Q, K and V from one image sequence. Translation cross-attention gets them from two sequences.
```

**Caption:** ViT gets Q, K and V from one image sequence. Translation cross-attention gets them from two sequences.

**Speaker notes**

Where do Q, K and V come from?
In ViT, three learned projections read the same normalized image sequence. In translation cross-attention, queries come from target decoder states and keys/values from source encoder states. Translation also contains self-attention; the comparison concerns its cross-attention operation.

### Page 26 — One patch representation has three jobs

```text
Receiver: P74
current patch state
192 features
QUERY · Q
Context I seek
KEY · K
What can match
VALUE · V
Information I send
Every token makes a query, a key and a value through three learned projections.
```

**Caption:** Every token makes a query, a key and a value through three learned projections.

**Speaker notes**

One patch representation has three jobs
Recall the same three roles from text. In this head q_i = LN(e_i) W_Q, k_i = LN(e_i) W_K and v_i = LN(e_i) W_V (with the checkpoint biases). Each is 64 features wide. These are learned numeric vectors; the questions are an analogy for their roles. P74 can receive information using its query and send information using its key and value.

### Page 27 — What might different queries try to gather?

```text
Illustrative questions: the model uses vectors, not sentences
A patch on the dog
Which other regions help interpret this texture?
Other fur / face regions
A background patch
Which other regions provide scene context?
Branches / bright background
CLS
CLS
Which image features help classify the whole image?
Useful parts across the photograph
Each query can gather different information from the same image.
```

**Caption:** Each query can gather different information from the same image.

**Speaker notes**

What might different queries try to gather?
These questions illustrate what a query might do; they are not translations of the trained vectors. A key has features that can match a query. The value carries information from the same source through a separate projection. An ear need not attend to another ear, and a head has no fixed semantic role. We will follow the calculation and then inspect a measured attention map.

### Page 28 — A source supplies both a key and a value

```text
P74 query q₇₄
64 features
Source P60
key k₆₀
64 features
value v₆₀
64 features
q₇₄ · k₆₀ / √64
one score → row softmax
a₇₄,₆₀ × v₆₀
Q and K set a scalar weight. V supplies the vector being weighted.
The weight for P60 scales all 64 coordinates of P60’s value vector.
```

**Caption:** The weight for P60 scales all 64 coordinates of P60’s value vector.

**Speaker notes**

A source supplies both a key and a value
The pictured source provides both k_60 and v_60, using different projections of its current normalized state. q_74 dot k_60 divided by 8 is one score. Its attention weight comes from softmax against every source score in row P74, including CLS. The weight is a scalar; the value and weighted contribution are 64-feature vectors. This diagram illustrates the roles of Q, K and V; it does not show measured attention weights.

### Page 29 — Several source values form one receiver’s message

```text
Toy example · one receiver · three sources · two value features
source A
0.6
×
[2, 0]
[1.2, 0]
source B
0.3
×
[0, 1]
[0, 0.3]
source C
0.1
×
[1, 1]
[0.1, 0.1]
Add the contributions: m = [1.3, 0.4]
This message goes to the receiver whose query chose the weights.
Multiply each value vector by its weight, then add the results to get one message.
```

**Caption:** Multiply each value vector by its weight, then add the results to get one message.

**Speaker notes**

Several source values form one receiver’s message
These chosen toy weights sum to one; they are not from the dog checkpoint. Reveal the second source, third source, then their sum: 0.6[2,0] + 0.3[0,1] + 0.1[1,1] = [1.3,0.4]. The real head sums 197 source contributions with 64 features each. It produces one message per receiving query. The multihead output projection converts the joined message into the update added to that receiver’s original row.

### Page 30 — ViT uses full attention, not causal attention

```text
TEXT DECODER · causal
source keys →
queries
↓
Earlier + current tokens
ViT ENCODER · full
source keys →
queries
↓
CLS + every patch token
Filled = allowed. ViT sees the complete image: no causal mask.
Each row is a query; each column is a key. Every pair is allowed, but the weights can differ.
```

**Caption:** Each row is a query; each column is a key. Every pair is allowed, but the weights can differ.

**Speaker notes**

ViT uses full attention, not causal attention
The small matrices show permitted query-key pairs, not learned weights. In a next-token decoder, future source positions are masked; a ViT encoder receives the complete image, so all 197 by 197 pairs are allowed, including CLS and self-pairs. Actual scores and row-softmax weights are generally asymmetric because Q and K differ. Patch order in the sequence does not impose a temporal or reading direction.

### Page 31 — How does one CLS query collect one message?

```text
CLS query
one receiver
compare with 197 keys
197 scores
row softmax
197 weights
weight the 197 values
sum their contributions
one CLS message
64 features in this head
Other query rows receive their own messages in parallel.
We used the CLS query, so this message updates CLS.
```

**Caption:** We used the CLS query, so this message updates CLS.

**Speaker notes**

How does one CLS query collect one message?
In one head, all 197 weighted value rows contribute to a 64-feature message. We join the heads, project the result to 192 features, and add it to the incoming CLS embedding. The optional attention appendix works through each source’s contribution.

### Page 32 — Three heads gather three messages for each token

```text
same sequence
197 × 192
Head 1
197 × 64
Head 2
197 × 64
Head 3
197 × 64
concatenate messages
197 × 192
then output projection
Join each row’s three 64-feature messages, then project them back to 192 features.
```

**Caption:** Join each row’s three 64-feature messages, then project them back to 192 features.

**Speaker notes**

Three heads gather three messages for each token
The heads run in parallel on the same normalized input, each with its own learned projections. They can gather different messages without having fixed roles such as an "ear head". Joining them increases the feature width while keeping the same token rows.

### Page 33 — Where does the trained query look?

```text
Query: P74
Measured source weights
P60
Weight: 9.09%
P60’s value contributes
to P74’s message.
Trained ViT · block 4 · head 1
Teal: 0 → 10.35%
Explore the nine examples ↗
Choose one query and look at its source weights. Another head or block may gather a different mixture.
```

**Caption:** Choose one query and look at its source weights. Another head or block may gather a different mixture.

**Speaker notes**

Where does the trained query look?
The maps use saved Q/K from trained ViT-Tiny block 4, head 1, query P74. P60 receives 9.088265% and CLS receives 2.089876%. All 197 weights sum to one; the displayed 196 patch weights are not renormalized. Teal intensity is scaled to this map’s maximum. The violet outline marks the query and white marks source P60. An attention map visualizes mixing, not a segmentation or a complete explanation of the class prediction.

### Page 34 — Add the attention update to the original embedding

```text
P74 input
192 features
Attention
gather messages
Project message
192 features
+
Updated P74
192 features
Keep the original embedding
original embedding + projected message = updated embedding
Every patch gets its own update. So does CLS.
Gather a message, project it to 192 features, then add it to the receiving token’s original embedding.
```

**Caption:** Gather a message, project it to 192 features, then add it to the receiving token’s original embedding.

**Speaker notes**

Add the attention update to the original embedding
Follow P74 as one receiver. Its attention heads gather weighted values from the current image sequence. Concatenate the three 64-feature messages and apply the learned output projection to obtain a 192-feature update. Add that update to P74’s incoming embedding; the message alone is not the new embedding. All other patch rows and CLS receive their own updates in parallel. Normalization is omitted from this conceptual diagram; the actual checkpoint uses U = E + MSA(LN(E)). The complete pre-LN block remains in the optional reference deck.

### Page 35 — The MLP adds one more update to each embedding

```text
After attention
192 features
MLP
192-feature update
+
New embedding
192 features
Keep the embedding after attention
The MLP processes each token separately.
This new embedding is ready for the next block.
Attention gathers context from other rows; the MLP then processes each row’s features.
```

**Caption:** Attention gathers context from other rows; the MLP then processes each row’s features.

**Speaker notes**

The MLP adds one more update to each embedding
The MLP receives the attention-updated representation and computes another 192-feature update. Add it to that same attention-updated representation, not to the original input from before attention. The same MLP parameters are used independently for every patch and CLS. This completes one block. Normalization and the internal 192 → 768 → 192 layers with GELU are omitted from this conceptual figure; the exact formula is E_next = U + MLP(LN(U)). Implementation details remain in the reference deck.

### Page 36 — Every block updates the patches and CLS again

```text
Each block: attention update, then MLP update.
Block 1
Block 2
Block 3
Block 4
Block 5
Block 6
Block 7
Block 8
Block 9
Block 10
Block 11
Block 12
197 token embeddings · still 192 features each
The next block starts from these new embeddings. After block 12, we read CLS.
```

**Caption:** The next block starts from these new embeddings. After block 12, we read CLS.

**Speaker notes**

Every block updates the patches and CLS again
All 197 rows continue through the stack: 196 patch embeddings and one CLS embedding. Each block reads the states produced by the preceding block and performs attention and MLP updates. Blocks have their own learned weights. The feature width remains 192; what each row represents changes. After the last block, the checkpoint’s final normalization and CLS readout give one image representation for classification.

### Page 37 — What is stored, and what changes with the image?

```text
LEARNED PARAMETERS
COMPUTED FOR EACH IMAGE
Patch projection W, b
Patch embeddings
Position table + starting CLS
Q, K, V and attention weights
Transformer weights, including W_Q/K/V
Contextual patch states + final CLS
Classifier weights + biases
Logits + probabilities
Training learns W_Q, W_K and W_V. Each new image gets its own attention weights.
```

**Caption:** Training learns W_Q, W_K and W_V. Each new image gets its own attention weights.

**Speaker notes**

What is stored, and what changes with the image?
At inference, learned parameters stay fixed and the model recomputes its activations for each image. Training uses gradients to update the parameters. The query and key projection weights are learned parameters; the attention matrix is computed from the current input.

### Page 38 — After the blocks, read the updated CLS embedding

```text
After all 12 blocks
Updated CLS
192 features
Updated patch embeddings
196 rows still exist
Read final CLS
one image embedding
192 features
Next: the class head
The classifier uses the image summary carried by CLS.
The classifier reads the final CLS embedding as its image summary.
```

**Caption:** The classifier reads the final CLS embedding as its image summary.

**Speaker notes**

After the blocks, read the updated CLS embedding
After the final LayerNorm, we select the CLS row to get a 192-feature image embedding. Selecting the row does not update it. This diagram leaves out normalization so we can focus on the readout. All 196 final patch embeddings still exist, but the trained ImageNet head uses CLS. Next, the head turns those features into 1,000 class scores.

### Page 39 — How do 192 features score 1,000 classes?

```text
h_CLS
192 features
Linear(192, 1000)
1,000 logits
softmax across classes
1,000 probabilities
One logit and one probability for every ImageNet class
The class head learns how each image feature contributes to each class score.
```

**Caption:** The class head learns how each image feature contributes to each class score.

**Speaker notes**

How do 192 features score 1,000 classes?
The head applies a learned affine map. Softmax turns its class scores into probabilities across the output classes. Attention softmax instead normalizes weights across source tokens. We predict the class with the highest score.

### Page 40 — What does the model predict for our photograph?

```text
Newfoundland
95.73%
Tibetan mastiff
1.63%
briard
0.67%
The model assigns Newfoundland a probability of 95.73% among its 1,000 classes.
```

**Caption:** The model assigns Newfoundland a probability of 95.73% among its 1,000 classes.

**Speaker notes**

What does the model predict for our photograph?
These are saved measured predictions from real-inference.json: Newfoundland 95.726752%, Tibetan mastiff 1.625724%, briard 0.674269%. They are not performance or accuracy estimates. The remaining probability belongs to the other ImageNet classes.

### Page 41 — What happens if we hide one quarter of the image?

```text
Original: 95.73%
Cover one quarter
Same model.
Changed pixels.
Same label?
Same probability?
What do you expect to change when we pass the covered image through the same model?
```

**Caption:** What do you expect to change when we pass the covered image through the same model?

**Speaker notes**

What happens if we hide one quarter of the image?
Keep Newfoundland as the target class throughout this experiment. Compare the original photograph with a fresh copy whose top-left 112 by 112 pixels are replaced with mid-gray. We change the pixels, keeping all tokens and the attention mask unchanged. Here we measure the final prediction; the earlier attention map showed weights inside the model.

### Page 42 — Change the pixels, then run the model again

```text
covered = x.clone()              # x: (1, 3, 224, 224), normalized
covered[:, :, :112, :112] = 0     # gray RGB; 112 × 112 pixels
model.eval()
with torch.inference_mode():
    before = model(x).softmax(-1)[0, 256]
    after = model(covered).softmax(-1)[0, 256]
same ViT
95.73%
ViT
83.00%
The model keeps its learned weights and recomputes patch features, attention weights and class probabilities.
```

**Caption:** The model keeps its learned weights and recomputes patch features, attention weights and class probabilities.

**Speaker notes**

Change the pixels, then run the model again
The checkpoint transform normalizes each channel as (RGB − 0.5) / 0.5, so normalized zero means gray RGB 0.5, not black. Index 256 is Newfoundland. Both calls use the same frozen model in evaluation mode. Image shape, all 196 patch tokens and CLS remain; 49 patch inputs now contain gray pixels. Each test uses a fresh clone, so covers never accumulate.

### Page 43 — Cover each quarter, then compare predictions

```text
Original P(Newfoundland) = 95.73%
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
Covering the top right gives the biggest drop: 16.11 percentage points. All four covered images still predict Newfoundland.
```

**Caption:** Covering the top right gives the biggest drop: 16.11 percentage points. All four covered images still predict Newfoundland.

**Speaker notes**

Cover each quarter, then compare predictions
Choose the four quadrants before viewing results. Run each covered copy separately and measure the same target-class probability. Top-right drops most among these four tests. This shows sensitivity to a specified pixel replacement on this photograph, not unique importance of an ear, face or other semantic part. The covers add gray edges and change the input distribution. Probability drops are not additive.

### Page 44 — Would smaller covers tell us more?

```text
Cover one patch: P78
196 separate tests
Largest local drop
95.73% → 91.89%
P78: −3.84 points
Red: probability falls
Blue: probability rises
Scale: −4 to +4 points
All 196 tests keep the same label. Smaller covers probe more local sensitivity.
Each coloured square records a separate forward pass with one 16 × 16 patch covered.
```

**Caption:** Each coloured square records a separate forward pass with one 16 × 16 patch covered.

**Speaker notes**

Would smaller covers tell us more?
The model and its patch size are unchanged; only the covered area shrinks. A fresh input is used for each of 196 tests. The largest drop is P78, 3.836018 points. This is a probability-change map, not attention. Small drops can occur because other regions preserve related clues; they do not prove a patch is useless. Detailed experimental protocol and all measurements remain in the reference deck.

### Page 45 — Two ways to gather image context

```text
CNN
ViT
Nearby first → wider context
Direct access to distant patches
CNNs gather wider context through local layers. Global ViT attention can connect distant patches in a single block.
```

**Caption:** CNNs gather wider context through local layers. Global ViT attention can connect distant patches in a single block.

**Speaker notes**

Two ways to gather image context
These drawings illustrate receptive fields and attention connections; they are not measured maps. Both families can use information from the whole image. We are comparing ordinary local CNNs with this ViT’s global attention. The diagrams do not tell us which model will perform better on a particular task.

### Page 46 — Which assumptions are built into the architecture?

```text
CNN
ViT
Basic units
Local feature maps
Patch-token rows
Mixing rule
Shared local convolution
Content-dependent attention
Locality
Built into each filter
Less spatial structure built in
Position
Grid structure
Explicit position embeddings
An inductive bias is an assumption built into the model before it learns from data.
```

**Caption:** An inductive bias is an assumption built into the model before it learns from data.

**Speaker notes**

Which assumptions are built into the architecture?
Convolution builds in local neighborhoods and shared filters. ViT also has built-in assumptions: it uses patches and shared projections. Its global attention weights depend on the content, and it receives position information explicitly. The data, training setup and pretrained weights all affect performance; the optional material discusses model choice.

### Page 47 — How can these built-in assumptions help?

```text
CNN: reuse a local detector
ViT: choose context from content
Same filter, different locations
Different weights for each query
A built-in image prior can help with limited data; pretraining changes the comparison.
CNN filters assume that nearby pixels matter and that the same pattern can occur in different places.
```

**Caption:** CNN filters assume that nearby pixels matter and that the same pattern can occur in different places.

**Speaker notes**

How can these built-in assumptions help?
The boxes and arrows are schematic. A learned local CNN filter is reused across locations; standard convolution mixes a fixed neighborhood with the same kernel weights, although its activations depend on the image. ViT query-key matching computes different source weights for each receiver and image, using shared projection parameters. With limited task data, useful priors or pretrained representations can help; there is no universal winner. Compare models under the actual data, accuracy and compute constraints.

### Page 48 — How much detail should one token cover?

```text
32 × 32 patches
49 patch tokens
16 × 16 patches
196 patch tokens
8 × 8 patches
784 patch tokens
Smaller patches give us a finer grid, with more tokens to process.
```

**Caption:** Smaller patches give us a finer grid, with more tokens to process.

**Speaker notes**

How much detail should one token cover?
For a 224 by 224 image: 32-pixel patches give 49 patch tokens, 16 gives 196, and 8 gives 784. Add one CLS token for this architecture. These are architectural comparisons, not a claim that the saved checkpoint accepts all three patch sizes without adaptation.

### Page 49 — What does a finer grid cost?

```text
16 × 16 patches
196 patch tokens
197 tokens with CLS
8 × 8 patches
784 patch tokens
785 tokens with CLS
Patch tokens × 4  →  attention scores ≈ × 16
38,809 scores  →  616,225 scores per head
Each token compares with every token, so doubling the token count gives four times as many scores.
```

**Caption:** Each token compares with every token, so doubling the token count gives four times as many scores.

**Speaker notes**

What does a finer grid cost?
Including CLS, 197 squared is 38,809 and 785 squared is 616,225, about 15.88 times larger. The approximately 16-fold statement is exact for the patch-only counts. This concerns attention scores, not a 16-fold claim about total runtime or all model operations.

### Page 50 — Once an image becomes tokens, the encoder is familiar

```text
1 · Make tokens
2 · Build context
3 · Read CLS
Shared
projection
224 × 224 RGB
+ CLS + position
CLS
P1
P2
…
197 rows × 192 features
12 encoder blocks
Full attention
gather messages + add
MLP
update each row + add
197 updated rows × 192
Final CLS · 192
Updated patch rows
Linear class head
1,000 class scores
Newfoundland
Turn patches into tokens, update them with the encoder, then use final CLS to predict the class.
```

**Caption:** Turn patches into tokens, update them with the encoder, then use final CLS to predict the class.

**Speaker notes**

Once an image becomes tokens, the encoder is familiar
Read the three panels from left to right, as in the opening model-family recap. Split the image into 196 RGB patches and apply one shared projection; prepend CLS and add position embeddings. All 197 rows pass through 12 encoder blocks. Within each block, full attention gathers messages, joins and projects the head outputs, and adds an update to each incoming row; the MLP then adds its own update to that attention-updated row. Each block has its own parameters. The shape remains 197 by 192. After final normalization, select only the CLS row for the trained 1,000-class head. The patch rows still exist. LayerNorm and the output projection are omitted from this overview; the nearby code and the detailed block slides make them explicit. The final label is the measured prediction for our photograph.

### Page 51 — Follow the whole model through its shapes

```text
Pixels
3 × 224 × 224
Patch projection
196 × 192
+ CLS + position
197 × 192
12 encoder blocks
197 × 192
Final LN → read CLS
192
Class head
1,000 logits
The blocks keep the same shape.
Each row now includes context.
These shapes describe one photograph. The batch dimension is left out.
```

**Caption:** These shapes describe one photograph. The batch dimension is left out.

**Speaker notes**

Follow the whole model through its shapes
The patch projection sets the feature width. Adding CLS takes us from 196 rows to 197, and all twelve encoder blocks keep the 197 by 192 shape. Reading CLS selects one row. The classifier then maps its 192 features to 1,000 logits.

### Page 52 — The whole ViT in six lines

```text
# Project each image patch to 192 features.
x = patch_embed(image)        # (B, 196, 192)
# Add CLS at index 0 and position to every token.
x = prepend_cls(x) + pos      # (B, 197, 192)
# Update all patch embeddings and CLS in each block.
for block in blocks:
    x = block(x)              # (B, 197, 192)
# Normalize, then extract token 0: final CLS.
h = norm(x)[:, 0]             # (B, 192)
# Turn the image summary into 1,000 class scores.
logits = head(h)              # (B, 1000)
B is the batch size. Token 0 is CLS; the head returns scores for the 1,000 ImageNet classes.
```

**Caption:** B is the batch size. Token 0 is CLS; the head returns scores for the 1,000 ImageNet classes.

**Speaker notes**

The whole ViT in six lines
This is readable pseudocode for the same pre-LN encoder classifier. The input image tensor has shape B by 3 by 224 by 224. patch_embed includes patch extraction, shared projection and conversion to patch rows. pos has shape 1 by 197 by 192 and broadcasts across the batch. Starting CLS is shared across the batch and prepended at token index zero. Each block updates every patch row and CLS through attention and MLP residual branches. norm(x) normalizes all final rows; [:, 0] selects CLS for every image in the batch, yielding B by 192. The linear head maps that image summary to 1,000 unnormalized class scores. No softmax is performed in these six executable lines.

### Page 53 — More detail, when you want it

```text
OPTIONAL LABS & EXTENSIONS
Implementation lab
Linear ↔ Conv2d
PyTorch shapes + code
Extensions
Transfer + evaluation
Similarity · attention · occlusion
Open the boxes
Patch arithmetic
Real attention numbers
Use these labs to work through the details.
The reference deck has the full calculations, code, diagrams and exercises.
```

**Caption:** The reference deck has the full calculations, code, diagrams and exercises.

**Speaker notes**

More detail, when you want it
The reference deck at vision1-reference.html has the full detail. Section 6 covers implementation; sections 7 to 9 contain the optional extensions. You can also find initialization, optimization and CLS-versus-pooling comparisons there. Feature similarity, attention weights and the effect of covering pixels measure different things.

### Page 54 — The classifier stores a learned vector per known class

```text
h_CLS
192 features
w_Newfoundland
class score
w_Persian cat
class score
w_sports car
class score
…
class score
s_c = h_CLSᵀ w_c + b_c
Its 1,000 output classes form a fixed vocabulary.
```

**Caption:** Its 1,000 output classes form a fixed vocabulary.

**Speaker notes**

The classifier stores a learned vector per known class
The 192-feature CLS is compared with a learned weight vector for every ImageNet class, with one bias per class. Persian cat and sports car name actual ImageNet classes. The equation uses column-vector notation for h. The classifier cannot accept an arbitrary new class description without changing its readout or model.

### Page 55 — What if our class vocabulary could come from words?

```text
“a photo of a dog”
? ? ?
v_dog
What if a class vector came from language?
NEXT: CLIP
We will pick up this question in the CLIP lecture.
```

**Caption:** We will pick up this question in the CLIP lecture.

**Speaker notes**

What if our class vocabulary could come from words?
The missing piece is a way to turn a class description into a vector we can compare with the image. Our checkpoint cannot do that for arbitrary text. The next lecture explains how CLIP learns image and text representations that can be compared.
