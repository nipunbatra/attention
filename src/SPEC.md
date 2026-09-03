# SPEC — Interactive self-attention teaching page (verbatim brief from the instructor)

Create a **single, self-contained interactive HTML file** that teaches **self-attention from first principles in the context of autoregressive next-token prediction**.
The target audience is students in a Deep Learning course who understand:
* token embeddings,
* neural networks,
* dot products,
* softmax,
* next-token prediction,
but have **not yet learned attention or Transformers**.
The main pedagogical goal is not to make students memorize
$$\operatorname{softmax}(QK^\top/\sqrt d)V.$$
Instead, by the end they should understand:
1. **Why attention is needed.**
2. **What problem \(Q,K,V\) solve.**
3. **What information actually flows between tokens.**
4. **How attention produces a contextual update to a token representation.**
5. **How that updated representation helps next-token prediction.**
6. **Why the residual addition exists.**
7. **Which pieces are standard Transformer machinery versus pedagogical notation.**

---
# Central notation: keep it extremely consistent
Notation confusion is a major concern.
Throughout the article use $e_i$ to mean: **the current representation of token \(i\)**.
At the very beginning, $e_i^{(0)} = \text{token embedding}_i+\text{positional information}_i.$
After attention, it becomes a contextual representation.
Do **not** continually switch between \(x,h,z,e\). Use \(e\) consistently.
For teaching, introduce $\Delta e_i$ to mean: the context-dependent update produced by the attention sublayer.
Then show:
$$\boxed{e_i^{\text{new}} = e_i^{\text{old}} + \Delta e_i}$$
Very explicitly say:
> The notation \(\Delta e\) is a pedagogical convention used in this article. It is not standard Transformer notation. The **addition itself is standard**: it is the residual connection around the attention sublayer.
This distinction should appear prominently.
The central visual idea of the whole article should be:
$$\boxed{e \rightarrow Q,K,V \rightarrow \text{attention} \rightarrow \Delta e \rightarrow e+\Delta e}$$

---
# 1. Begin with next-token prediction
Start with a very concrete language-modeling task. For example:
> The fisherman sat near the river bank and watched the ___
We want $p(x_{t+1}\mid x_1,\ldots,x_t)$.
Show a few possible next tokens with probabilities.
Explain:
> To predict the next token well, the model needs a useful representation of everything relevant in the preceding context.
Do not introduce attention yet.

---
# 2. Simplest baseline: only the current/last token
Suppose prediction uses only $e_t$. Then $\ell = W_{\text{out}}e_t+b$ and $p(x_{t+1}\mid x_{\le t}) = \operatorname{softmax}(\ell)$.
Visually demonstrate why this throws away most of the context.

---
# 3. Baseline: fixed previous-K-token context
Now propose the seemingly obvious solution:
> Why not simply concatenate the previous \(K\) token representations?
For example: $c_t = [e_{t-K+1};e_{t-K+2};\ldots;e_t]$. Then predict: $p(x_{t+1}\mid x_{\le t}) = \operatorname{softmax}(Wc_t+b)$.
Use an interactive slider for \(K\).
Show visually:
* what tokens are included,
* what tokens fall outside the context,
* how dimensionality increases with \(K\).
Discuss:
### Hard context boundary
Anything before the previous \(K\) tokens is unavailable.
### Everything is included equally structurally
The model receives all \(K\) representations, regardless of relevance.
### Relevant context changes dynamically
For different predictions, different previous tokens matter.
### Increasing K does not fundamentally solve the problem
Larger \(K\): increases input dimensionality, increases parameters, includes more irrelevant information, still has a fixed cutoff.
Lead students toward the question:
> Could we dynamically retrieve only the information that is useful right now?

---
# 4. Before attention: introduce weighted pooling
First show the simplest aggregation: $c_i = \frac{1}{i}\sum_{j\le i}e_j.$
Ask: > Why not simply average all previous token representations?
Explain that this treats every token equally.
Then introduce: $c_i = \sum_{j\le i}\alpha_{ij}e_j,$ where $\sum_j\alpha_{ij}=1.$
This is weighted pooling.
Now ask the crucial question:
> Where should the weights \(\alpha_{ij}\) come from?
They should depend on **what token \(i\) currently needs** and **what information token \(j\) contains**.
This is the bridge to attention.

---
# 5. Take a deliberate information-retrieval detour
Before mentioning \(Q,K,V\) in language models, introduce them through a familiar **search/retrieval analogy**.
Imagine a website containing educational videos:
* Backpropagation
* Gradient Descent
* CNNs
* Transformers
* Batch Normalization
* Regularization
Suppose the user searches:
> "How does a neural network send gradient information backwards?"
Build a simple visual search engine.
Each item should conceptually have two representations:
$\boxed{\text{Key}}$ used for matching, and $\boxed{\text{Value}}$ representing the information returned when that item is retrieved.
The user's search becomes: $\boxed{\text{Query}}$.
Use: $q=\text{query representation}$, $k_j=\text{key of item }j$, $v_j=\text{value of item }j$.
Similarity: $s_j=q^\top k_j$.
Show a table such as:
| Video            | Similarity |
| ---------------- | ---------: |
| Gradient Descent |        1.4 |
| Backpropagation  |        4.8 |
| CNN              |        0.6 |
| Transformers     |        0.2 |
Explain very explicitly:
> The **key is used to decide whether something matches**.
> The **value is the information we actually retrieve**.
This distinction is essential.
Use a memorable formulation:
$\boxed{Q:\text{ What am I looking for?}}$
$\boxed{K:\text{ When should you retrieve me?}}$
$\boxed{V:\text{ What information do I send if retrieved?}}$

---
# 6. Hard retrieval → soft retrieval
A normal search engine might retrieve only: $j^* = \arg\max_j q^\top k_j$. Then return $v_{j^*}$.
Attention instead performs **soft retrieval**.
Compute: $\alpha_j = \operatorname{softmax}_j(q^\top k_j)$. Then return: $\boxed{\sum_j \alpha_jv_j}$
Explain:
> Attention does not necessarily retrieve one thing. It retrieves a weighted mixture of information.
Use an interactive numerical example where changing similarity scores changes:
* softmax weights,
* retrieved mixture.
This information-retrieval section should make \(Q,K,V\) feel natural before returning to language.

---
# 7. Return to language: every token becomes a tiny searchable record
Now say:
> What if every token in the sequence behaved like one item in this searchable memory?
Token \(j\), with current representation \(e_j\), generates: $k_j=W_Ke_j$ and $v_j=W_Ve_j$.
Token \(i\), which is trying to gather useful context, generates: $q_i=W_Qe_i$.
Thus: $\boxed{q_i=W_Qe_i}$ $\boxed{k_j=W_Ke_j}$ $\boxed{v_j=W_Ve_j}$
Do not yet show matrix notation. Teach one query token at a time.

---
# 8. Critical conceptual distinction: Q/K versus V
Make this one of the strongest visual sections.
Separate attention into two phases.
## Phase A: READ ROUTING
Compare: $q_i^\top k_j$. This answers: > Where should token \(i\) retrieve information from?
No representation has yet been updated.
Visually:
```text
                           q_bank
                              |
             --------------------------------
             |               |              |
        k_fisherman       k_river         k_the
             |               |              |
          score           score          score
```
Then softmax generates $\alpha_{ij}$.
Explain: $\boxed{Q,K \text{ determine routing / relevance}}$
## Phase B: MESSAGE PASSING
The actual information sent from token \(j\) is: $v_j$.
So $\alpha_{ij}v_j$ is the message from \(j\) to \(i\).
Aggregate: $m_i = \sum_j\alpha_{ij}v_j.$
Explain: $\boxed{V \text{ carries the information}}$
This section should make clear:
> Queries and keys determine **who communicates with whom**.
> Values determine **what gets communicated**.

---
# 9. Introduce the contextual update Δe_i
Now connect the retrieved message back to the token representation.
For a simple single-head pedagogical version, show: $m_i = \sum_j\alpha_{ij}v_j.$
Then possibly an output projection: $\boxed{\Delta e_i=W_Om_i}$ and finally: $\boxed{e_i^{\text{new}} = e_i+\Delta e_i.}$
This should be the centerpiece of the article.
Animate this sequence:
```text
current representation
        e_i
         |
         +------------------------+
         |                        |
         |                       Q_i
         |                        |
         |               compare against K_j
         |                        |
         |                 attention weights
         |                        |
         |                weighted V_j values
         |                        |
         |                    Δe_i
         |                        |
         +----------- + ----------+
                     |
                     v
                  e_i'
```
Use a prominent caption:
> **Attention does not replace the existing representation. It computes context-dependent information that can be added to it.**
Then immediately add the technical clarification:
> Calling this update \(\Delta e_i\) is pedagogical notation. Standard implementations usually call this the attention output and add it through a residual connection.

---
# 10. Use the "bank" ambiguity example carefully
Use examples compatible with **causal attention**.
Prefer: > The fisherman sat beside the river bank
versus: > She deposited the cheque at the bank
Do **not** demonstrate the representation of "bank" depending on future words.
For "The fisherman sat beside the river bank" show the `bank` token querying earlier tokens.
Potential attention: river → high, fisherman → moderate, beside → moderate, the → low.
Show: $\Delta e_{\text{bank}} = \alpha_1v_1+\cdots+\alpha_tv_t.$ Then: $e_{\text{bank}}' = e_{\text{bank}} + \Delta e_{\text{bank}}.$
Conceptually visualize:
$\text{generic BANK representation} + \text{river-context update} \rightarrow \text{RIVER-BANK contextual representation}.$
For "She deposited the cheque at the bank" show a different attention pattern: cheque → high, deposited → high.
Thus $e_{\text{bank}}'$ becomes a different contextual representation.
Make it visually clear that $e_{\text{bank}}^{\text{initial}}$ may begin similarly in both sentences, while $e_{\text{bank}}^{\text{contextual}}$ becomes different after attention.

---
# 11. Explicitly answer: are Q/K/V transformed embeddings?
Have a dedicated section titled: ## "What exactly are Q, K, and V?"
State: They are temporary linear projections of the **current token representation**:
$e_i \rightarrow \begin{cases} q_i=W_Qe_i\\ k_i=W_Ke_i\\ v_i=W_Ve_i \end{cases}$
They are **not three new contextual embeddings**. They play temporary functional roles during attention.
Use: $\boxed{e \rightarrow Q,K,V \rightarrow \Delta e \rightarrow e+\Delta e}$ and keep returning to this diagram.

---
# 12. Explain scaling
Then introduce: $s_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}}.$
Explain that if components are roughly zero-mean unit-variance, then $\operatorname{Var}(q^\top k) \propto d_k.$
Therefore dot products become larger as dimension grows. Large logits make softmax extremely sharp.
Use an interactive comparison:
### Without scaling
$[2,5,9,1]$
### With scaling
$\frac{[2,5,9,1]}{\sqrt{d_k}}$
and display resulting softmax probabilities.

---
# 13. Explain causal masking
Now emphasize that this is **next-token prediction**.
Token \(i\) may only retrieve from $j\le i$. Never $j>i$.
Build an attention matrix: $S=QK^\top$.
Show the causal mask visually:
$\begin{bmatrix} \checkmark & \times & \times & \times\\ \checkmark & \checkmark & \times & \times\\ \checkmark & \checkmark & \checkmark & \times\\ \checkmark & \checkmark & \checkmark & \checkmark \end{bmatrix}.$
Add a toggle: **Causal Mask ON/OFF**
When OFF, visually demonstrate information leakage.
Explain why training would become invalid if a token could inspect future tokens while learning next-token prediction.

---
# 14. Connect attention directly to next-token prediction
Do not stop at contextual embeddings.
Show $e_t^{\text{contextual}}$ feeding into the language-model output head: $\ell = W_{\text{vocab}}e_t+b.$ Then: $p(x_{t+1}\mid x_{\leq t}) = \operatorname{softmax}(\ell).$
For "The fisherman sat beside the river bank and watched the ___" show candidate probabilities.
Then modify earlier context and show probabilities change.
Make the causal chain obvious:
$\boxed{\text{previous tokens} \rightarrow \text{attention} \rightarrow \Delta e_t \rightarrow e_t' \rightarrow \text{next-token logits} \rightarrow \text{next-token probabilities}}$

---
# 15. Build one complete interactive walkthrough
Have a large **Next Step** button.
Use a short sequence and walk through:
1. Tokens
2. Initial representations \(e_i\)
3. Choose query token
4. Compute \(q_i\)
5. Compute each \(k_j\)
6. Compare \(q_i^\top k_j\)
7. Scale
8. Mask
9. Softmax
10. Obtain \(\alpha_{ij}\)
11. Produce values \(v_j\)
12. Weight each value
13. Sum values
14. Obtain message \(m_i\)
15. Output projection \(W_Om_i\)
16. Call this pedagogically \(\Delta e_i\)
17. Residual addition $e_i'=e_i+\Delta e_i$
18. Use contextual representation for next-token prediction.
Only one stage should be highlighted at once.
This should be the major teaching visualization.

---
# 16. Only after intuition: introduce matrix notation
Once one-token attention is completely clear, show:
$E= \begin{bmatrix} e_1^\top\\ \vdots\\ e_T^\top \end{bmatrix}.$
Then: $Q=EW_Q, \qquad K=EW_K, \qquad V=EW_V.$
Dimensions should always be visible:
$E\in\mathbb{R}^{T\times d_{\text{model}}}$, $Q,K\in\mathbb{R}^{T\times d_k}$, $V\in\mathbb{R}^{T\times d_v}.$
Then: $S = \frac{QK^\top}{\sqrt{d_k}}.$ $A = \operatorname{softmax}(S+M).$ $H=AV.$ $\Delta E=HW_O.$
Finally: $\boxed{E'=E+\Delta E.}$
Again state: > \(\Delta E\) is our pedagogical notation for the attention sublayer's residual update.
Animate the matrices one operation at a time.

---
# 17. Explain multi-head attention only after single-head attention
Motivate: > There may be multiple useful notions of relevance.
For the bank example: one head may retrieve semantic context, another nearby syntax, another long-range information.
For each head \(h\): $Q^{(h)}=EW_Q^{(h)}$, $K^{(h)}=EW_K^{(h)}$, $V^{(h)}=EW_V^{(h)}.$
Then: $H^{(h)} = \operatorname{Attention}(Q^{(h)},K^{(h)},V^{(h)}).$
Concatenate: $H= [H^{(1)};\ldots;H^{(H)}].$
Then: $\boxed{\Delta E=HW_O.}$ Finally: $E'=E+\Delta E.$
Use \(W_O\) to reinforce:
> Each head retrieves information in its own smaller space. \(W_O\) combines those retrieved messages into an update in the model's representation space.

---
# 18. Explain repeated contextualization across layers
Clarify that \(e_i\) is only the raw token embedding at layer 0.
Initially: $e_i^{(0)} = \text{token embedding}+\text{position}.$
After layer 1: $e_i^{(1)} = e_i^{(0)} + \Delta e_i^{(0)}.$
Then the next attention layer creates new $Q^{(1)},K^{(1)},V^{(1)}$ from these already-contextualized representations.
Then: $e_i^{(2)} = e_i^{(1)} + \Delta e_i^{(1)}.$
Show: $e^{(0)} \rightarrow e^{(1)} \rightarrow e^{(2)} \rightarrow \cdots$ with progressively richer context.
Explain: > We are repeatedly updating token representations, not repeatedly returning to the original token embedding.

---
# 19. Contrast against alternatives
Include compact visual comparisons.
## Fixed window: $[e_{t-K+1};\ldots;e_t]$
## Mean pooling: $\frac1t\sum_j e_j$
## Fixed positional weights: $\sum_j w_je_j$
## Attention: $\sum_j \underbrace{\alpha_{ij}}_{\text{depends on current content}} v_j.$
Emphasize: $\boxed{\text{Attention = content-dependent soft retrieval}}$

---
# 20. Include "Pause and Think" questions
Examples:
### Why do we need separate keys and values?
Reveal: > Matching information and returned information serve different functions.
### If every attention score were identical, what would attention resemble?
Approximately uniform/mean pooling.
### Is \(q_i\) the new representation of token \(i\)?
No.
### Are \(Q,K,V\) stored permanently as token representations?
No. They are projections used by the attention computation.
### What is \(\Delta e_i\)?
The context-dependent update generated by the attention sublayer.
### Is "delta embedding" standard terminology?
No. It is a pedagogical interpretation of the standard residual update.
### What operation is standard?
$e_i^{\text{new}} = e_i^{\text{old}} + \text{AttentionOutput}_i.$
### Why can't `bank` attend to a later `river` in a causal language model?
Because that would expose future information.

---
# 21. End with three levels of summary
## Intuitive
> Each token searches previous tokens for useful information, retrieves a weighted mixture of what they have to say, and uses that information to update its own representation.
## Operational
$\boxed{\text{Query} \rightarrow \text{match Keys} \rightarrow \text{softmax} \rightarrow \text{retrieve Values} \rightarrow \Delta e \rightarrow e+\Delta e}$
## Mathematical
$Q=EW_Q, \qquad K=EW_K, \qquad V=EW_V$
$A = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)$
$\Delta E=AVW_O$
$\boxed{E'=E+\Delta E.}$
Then finally connect: $E' \rightarrow \text{vocabulary logits} \rightarrow \text{softmax} \rightarrow p(\text{next token}).$

---
# Design requirements
Make this look like a polished interactive educational artifact rather than an article containing large amounts of prose.
Use: token chips, arrows, animated message passing, SVG diagrams, small numerical vectors, sliders, hover effects, attention heatmaps, matrices, expandable explanations, step-by-step buttons, prediction probability bars.
Keep text concise. Prefer: visual → intuition → equation rather than long paragraphs.
Use visually distinct styles for:
* \(e\): current token representation,
* \(Q\): query,
* \(K\): key,
* \(V\): value,
* \(\alpha\): attention weights,
* \(\Delta e\): contextual update,
* \(e+\Delta e\): updated representation.
The distinction between these objects must remain visually consistent throughout the page.

---
# Technical constraints
Produce exactly one `attention.html` file. It must contain all HTML, CSS, JavaScript, SVG inline.
No backend. No React. No npm. No build step. Prefer no external CDN dependencies.
It should work by double-clicking the HTML file. Make it responsive and projector-friendly.

---
# Pedagogical style
The page should feel like a combination of: 3Blue1Brown's geometric intuition, Andrej Karpathy's concrete language-model reasoning, Andrew Ng's gradual conceptual buildup. But do not copy their wording or visuals.
The sequence should feel inevitable:
Next-token prediction ↓ Fixed context is unsatisfactory ↓ We need dynamic retrieval ↓ Information retrieval: Query / Key / Value ↓ Soft retrieval ↓ Tokens retrieve information from tokens ↓ Q,K decide where to read ↓ V determines what gets transmitted ↓ Δe = retrieved contextual update ↓ e' = e + Δe ↓ better contextual representation ↓ next-token prediction
Do not introduce full Transformer blocks, LayerNorm, MLPs, or architectural details until the attention mechanism itself is completely clear.
The artifact should support approximately **60 minutes of teaching**.
Optimize relentlessly for eliminating conceptual confusion.
