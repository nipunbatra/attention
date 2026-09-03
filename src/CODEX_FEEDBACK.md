# Reviewer feedback (from Codex, forwarded by the instructor) — every item must be addressed

NOTE (already done by the integrator, keep it): the whole page now uses the ROW-VECTOR convention consistently:
q_i = e_i W_Q, k_j = e_j W_K, v_j = e_j W_V, Δe_i = m_i W_O, ℓ = e_t W_vocab + b, matching the displayed shapes (e_i is 1×4, W_Q is 4×3)
and the matrix form Q = E W_Q. Do not reintroduce W_Q e_i. Item A0 below adds the one-line explanation.

## Important fixes
A0. (s07) Add one short sentence where q_i = e_i W_Q first appears: vectors are written as rows here, so a 4-number row times a 4×3 matrix gives 3 numbers; many texts write the column form W_Q e_i; the numbers are the same.
A2. (s16) The numerical matrix equation is labelled QK^T = S but the numbers shown for S are already divided by √3 (bank→river raw product 3.0098, displayed 1.7377). Make the numerical equation show the division explicitly: Q K^T / √d_k = S (or label the product step as the raw product and add the scaling step).
A3. (s15) The walkthrough follows bank (position 7) through step 17, then step 18 predicts from the(10) while the status still names bank as the query. Make the switch explicit: at step 18 say that the same computation was run for position 10, show q_10's row (attention weights of the(10)), and label the prediction as coming from e'_10, not from bank's update. The status/label must change to the(10) at that step.
A4. (s11) Do not say q/k/v are "discarded" or "temporary" in a way that suggests they cannot be kept (generation caches keys and values). Better wording: "These are computed working vectors, not separately learned embedding tables, and not the updated token representation that is passed onward." Adjust the lifetime strip tags accordingly (e.g. "computed → used in attention" instead of "computed → used → discarded").
A5. (s15) The highlighted row is labelled "largest message" but it is the largest attention weight; message magnitude also depends on v_j (with query "and", the highlighted contribution has norm ≈0.222 while another has ≈0.445). Rename the highlight "largest attention weight" (and the caption/notes that say largest message/contribution).

## Explanations to tighten
B1. (s03, s04) Fixed-window models can learn relevance through W; concatenation does not force equal influence. Reword the "everything is included equally" card: the valid criticism is the fixed boundary and the absence of content-dependent pooling (the weights that combine the K blocks are fixed by position, not chosen by reading the content), not an inability to distinguish useful words. Also drop "fundamentally".
B2. (s12) √d_k scaling does not guarantee diffuse attention; it controls dimension-driven score variance (under simplifying independence assumptions). Attention can still be sharply concentrated; score differences, not a common offset, determine sharpness. Reword the callout and eyebrow.
B3. (s21) Prediction is not generation. Add a short closing note: the head gives probabilities for one step; generation picks a token, appends it, forms the new last-position query, and repeats.

## Additional corrections
C1. (s05, s06) The selected query silently changes between sections: choosing "How do I stop overfitting?" in s05 makes Regularization win, but s06 says "same query" while using the gradient question's scores. Either carry the selection forward or make the reset explicit ("We go back to the gradient question").
C2. (s13) After inspecting a future-token cell and then toggling the mask off, the explanation still reports "0.000 — masked" while the matrix shows 0.17. Re-run the cell explanation (or clear it) whenever the mask toggles.
C3. (s06) Rounded arithmetic shown as exact: six equal weights display as 0.17 and the equation joins them with "=" although they sum to 1.02. Use ≈ (\approx) in the displayed equation with rounded numbers, or show 1/6, and say the computation uses unrounded weights.
C4. (s06) "as τ → 0 soft retrieval becomes hard retrieval" needs "when there is a unique maximum"; with tied maxima softmax keeps sharing weight among the tied items.
C5. (s10) At 50% on the "fraction of Δe added" slider the numbers are e + 0.5Δe but the label still says e' = e + Δe. Label intermediate states e(t) = e + tΔe and reserve e' for t = 1.

## Teaching improvements
D1. (s14, and reinforce in s18) Make the layer boundary explicit: after contextualising bank, the page predicts after the final "the"; students may infer that "the" reads the newly updated bank vector immediately. Within one layer it does not: every position reads values computed from the same layer-input representations; the updated bank becomes available to the next layer. This is also a concrete reason to stack layers. Add a short note callout in s14 and one sentence in s18.
D2. (s20) The equal-scores answer must say: equal scores give exactly uniform weights over the allowed positions; the message is the mean of the VALUE vectors v_j, not in general the mean of the original representations e_j; the output projection and residual addition still follow. (Protect the e vs v distinction.)
D3. (s17) Clarify what heads divide: every head reads the full input representation through its own projections; it does not receive a disjoint slice of the embedding; the smaller projected outputs are concatenated. Do not use H for both the head count and the matrix: write the head count as n_h (H = [H^{(1)}; …; H^{(n_h)}]) and fix the dims text (d_k = d_model / n_h).
D4. (shell hero "How to read this page", and s01 or s07 briefly) Label the toy's provenance visibly: all numbers come from a tiny demonstration model (4-number representations, 20-word vocabulary) whose weights were tuned by hand to make the patterns visible; the arithmetic is real; it is not a trained general-purpose language model.
D5. (s19) "blind to content" → "content-independent weights" (a mean still changes when its input words change). (s18) moving farther from the initial embedding shows change, not necessarily better understanding; reword "richer, more contextual" claims to "changed by context" unless supported.
