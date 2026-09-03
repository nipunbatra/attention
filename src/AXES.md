# AXES.md — interpretable toy model v2 (named coordinates everywhere)

Instructor request: students should be able to read what each column of e, q, k, v means. Illustrative names only,
but the numbers must AGREE with the names. So the toy is rebuilt by hand with sparse, readable matrices.

## Axes (add to toy.json as "axes")
- e (d_model = 4): ["water", "finance", "person", "glue"]. Meaning: how much the token is about a water/nature scene;
  about finance; about a person or agent; how much it is a grammatical glue word. Position adds a small offset (see below).
- q and k (d_k = 3): ["setting: water?", "setting: finance?", "who?"]. For a QUERY row read "what I am asking for";
  for a KEY row read "what I offer".
- v (d_v = 3): ["says: water scene", "says: finance scene", "says: a person is here"]. What the token sends if retrieved.
- Δe uses the e axes (W_O maps value axes back onto them: water→water, finance→finance, person→person, nothing→glue).
- Short forms for narrow tables: e: water / finance / person / glue; q,k: water? / finance? / who?; v: →water / →finance / →person.

## Embeddings (1-decimal, non-negative, sparse; entries up to 3.0 so softmax is decisive)
river [3.0,0,0,0]  water [3.0,0,0,0]  boats [2.4,0,0.4,0]  fish [2.6,0,0.6,0]  ducks [2.2,0,0.8,0]
cheque [0,3.0,0,0]  money [0,3.0,0,0]  teller [0,2.6,1.8,0]  clerk [0,2.4,1.8,0]  queue [0,2.0,0.6,0]
deposited [0,2.2,0.6,0.4]  fisherman [1.4,0,2.4,0]  she [0,0,2.6,0.4]  bank [1.5,1.5,0,0.4]
the [0,0,0,2.4]  and [0,0,0,2.2]  at [0,0,0,2.0]  beside [0.6,0,0,2.0]  sat [0.4,0,0.6,1.8]  watched [0.6,0,0.8,1.6]
(These are a starting point; adjust minimally to hit the targets. Keep the story: bank is half water, half finance.)
Position: pos_emb rows are small (|x| ≤ 0.3), one decimal, mostly on the person/glue axes, distinct per position, so the three
"the" tokens differ but their meaning does not. State on the page: "plus a small position offset".

## Projections (sparse, 1-decimal; students should be able to read each row as "axis → asks/offers/says")
W_Q (e axes → q axes):  water → water? 1.0;  finance → finance? 1.0;  person → who? 0.4;  glue → water? 0.7 and finance? 0.7
   (glue words ask "what setting am I in?", which is why the final "the" reads river or cheque).
W_K (e → k): water → water? 1.0; finance → finance? 1.0; person → who? 1.0; glue → nothing (glue offers no evidence).
W_V (e → v): water → says water 1.0; finance → says finance 1.0; person → says person 1.0; glue → nothing.
W_O (v → e): says water → water 1.0; says finance → finance 1.0; says person → person 0.5; nothing → glue.
W_vocab (e → logits): water axis → water 1.2, boats 0.9, fish 0.8, ducks 0.5; finance axis → teller 1.2, clerk 0.9, queue 0.6, money 0.5;
   person axis → small boosts (teller 0.2, clerk 0.2, fisherman 0.2); glue → 0. b_vocab: candidates 0, everything else −1.5 (so the
   baseline over "the" alone is diffuse across the eight candidates). Tune the few free numbers so the targets below hold.

## Targets (verify on the final rounded numbers with toy_ref.mjs; loosened from the first toy because interpretability comes first)
T1 S_A (The fisherman sat beside the river bank), query bank(7): river clearly highest (≥ 0.40), fisherman second, bank(self) and the glue words low.
T2 S_B (She deposited the cheque at the bank), query bank(7): cheque highest, deposited second, others low.
T3 S_A, query the(10): probabilities water > boats > fish > ducks, all other tokens ≤ 0.04 each; attention mostly on river, bank, fisherman.
T4 S_B, query the(10): teller > clerk > queue > money, others ≤ 0.04; attention mostly on cheque, bank, deposited.
T5 baseline (e_the^(0) through the head, no attention): eight candidates each between 0.06 and 0.18, others ≤ 0.04; identical for both sentences.
T6 mask off, S_A: some early token puts ≥ 0.25 on future river/bank (report; soft).
Keep schema of BRIEF section 4 exactly, plus: "axes": {"e": [...], "qk": [...], "v": [...], "short": {"e": [...], "qk": [...], "v": [...]}}
and "notes" explaining that coordinates were named and the matrices designed by hand so the names are true.

## Revision 2 (instructor decisions after seeing the page)
1. Position gets its own coordinate. d_model = 5 with e axes ["water", "finance", "person", "glue", "position"] (short: water / finance / person / glue / pos).
   Token embeddings have 0 in the position column; pos_emb[i] is 0 on the four meaning axes and 0.1*(i+1) on the position axis (i zero-based),
   so e^{(0)} = token + position changes exactly one column. The position rows of W_Q, W_K, W_V and W_vocab are all zero (the toy carries position but
   does not use it; the page says so). Old rule "position offsets on person/glue" is withdrawn.
2. Values are deliberately narrower than queries and keys: d_k = 3 (q and k axes unchanged: setting: water? / setting: finance? / who?) and
   d_v = 2 with v axes ["says: water scene", "says: finance scene"] (short: →water / →finance). W_V is 5×2 (water→says water 1.0, finance→says finance 1.0,
   everything else 0); W_O is 2×5 (says water→water 1.0, says finance→finance 1.0). The message m_i and Δe_i follow (m_i has 2 columns on the v axes,
   Δe_i has 5 columns on the e axes with only water and finance non-zero). Teaching point to state where q, k, v first sit together (s07, s11):
   q and k share columns because they are compared; v has its own columns and its own width because it is only mixed and sent.
3. Targets T1 to T6 unchanged (attention depends only on q·k; predictions use water/finance). Re-verify on the rounded numbers.
4. Sweep every shape and count on the page: 1×4 → 1×5, 4×3 → 5×3 for W_Q/W_K, W_V 5×2, W_O 2×5, W_vocab 5×20, "four numbers" → "five numbers",
   d_model = 5, d_v = 2; the notation card and the hero provenance line; the shared runtime picks axis names by width (5 → e, 3 → qk, 2 → v).
