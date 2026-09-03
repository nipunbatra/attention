# toy_report.md - numbers produced by the hand-designed toy.json (v2, named axes)

All values below are computed from the ONE-DECIMAL parameters in `toy.json` (268 numbers, max |x| = 3.0) by `make_toy2.py`; `node toy_ref.mjs --compare py_check.json` reproduces every intermediate to < 1e-12. Attention weights are shown to 2 decimals, probabilities to 3. The previous, optimised toy is kept as `toy_v1.json` (its report: `toy_report_v1.md`).

## How the numbers were obtained

Nothing was optimised. AXES.md names every coordinate, and the embeddings and projection matrices were written by hand so that the names are true: each row of a projection reads as 'input axis -> asks / offers / says'. The forward pass (the same arithmetic as `toy_ref.mjs`) was then evaluated and a few magnitudes were adjusted from the AXES.md starting point until the targets held on the rounded numbers: `bank` went from [1.5, 1.5, 0, 0.4] to [0.7, 0.7, 0, 0.6] and `fisherman` from [1.4, 0, 2.4, 0] to [2.0, 0, 2.2, 0], because the self score q_bank . k_bank grows with the square of bank's water/finance entries and at 1.5 bank attended to itself as much as to river; the W_vocab weights were raised to 1.5/1.2/1.0/0.7 (water words) and 1.5/1.2/0.9/0.7 (finance words) so that every non-candidate word stays at or below 0.04. Position offsets are a circle of radius 0.3 in the (person, glue) plane, one decimal, nothing on the water/finance axes.

## Axes (from `toy.json` -> `axes`)

| object | coordinate 1 | coordinate 2 | coordinate 3 | coordinate 4 |
|---|---|---|---|---|
| e, Delta e, e' (d_model = 4) | water | finance | person | glue |
| q, k (d_k = 3) | setting: water? | setting: finance? | who? | |
| v, m (d_v = 3) | says: water scene | says: finance scene | says: a person is here | |
| short forms | e: water, finance, person, glue | q,k: water?, finance?, who? | v: →water, →finance, →person | |

Reading rule: a query row is 'what I ask for', a key row is 'what I offer', a value row is 'what I send if retrieved'. W_O maps the v axes back onto the e axes (water -> water, finance -> finance, person -> person, nothing -> glue).

## Token embeddings (e axes)

| token | water | finance | person | glue | reading |
|---|---:|---:|---:|---:|---|
| the | 0 | 0 | 0 | 2.4 | glue 2.4 |
| fisherman | 2.0 | 0 | 2.2 | 0 | water 2.0, person 2.2 |
| sat | 0.4 | 0 | 0.6 | 1.8 | water 0.4, person 0.6, glue 1.8 |
| beside | 0.6 | 0 | 0 | 2.0 | water 0.6, glue 2.0 |
| river | 3.0 | 0 | 0 | 0 | water 3.0 |
| bank | 0.7 | 0.7 | 0 | 0.6 | water 0.7, finance 0.7, glue 0.6 |
| and | 0 | 0 | 0 | 2.2 | glue 2.2 |
| watched | 0.6 | 0 | 0.8 | 1.6 | water 0.6, person 0.8, glue 1.6 |
| she | 0 | 0 | 2.6 | 0.4 | person 2.6, glue 0.4 |
| deposited | 0 | 2.2 | 0.6 | 0.4 | finance 2.2, person 0.6, glue 0.4 |
| cheque | 0 | 3.0 | 0 | 0 | finance 3.0 |
| at | 0 | 0 | 0 | 2.0 | glue 2.0 |
| water | 3.0 | 0 | 0 | 0 | water 3.0 |
| boats | 2.4 | 0 | 0.4 | 0 | water 2.4, person 0.4 |
| fish | 2.6 | 0 | 0.6 | 0 | water 2.6, person 0.6 |
| ducks | 2.2 | 0 | 0.8 | 0 | water 2.2, person 0.8 |
| teller | 0 | 2.6 | 1.8 | 0 | finance 2.6, person 1.8 |
| clerk | 0 | 2.4 | 1.8 | 0 | finance 2.4, person 1.8 |
| queue | 0 | 2.0 | 0.6 | 0 | finance 2.0, person 0.6 |
| money | 0 | 3.0 | 0 | 0 | finance 3.0 |

## Position offsets (added to the token row; e^(0)_i = tok_emb[token_i] + pos_emb[i])

| position | water | finance | person | glue |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0.2 | -0.2 |
| 2 | 0 | 0 | 0.1 | -0.3 |
| 3 | 0 | 0 | -0.1 | -0.3 |
| 4 | 0 | 0 | -0.2 | -0.2 |
| 5 | 0 | 0 | -0.3 | 0 |
| 6 | 0 | 0 | -0.2 | 0.2 |
| 7 | 0 | 0 | -0.1 | 0.3 |
| 8 | 0 | 0 | 0.1 | 0.3 |
| 9 | 0 | 0 | 0.2 | 0.2 |
| 10 | 0 | 0 | 0.3 | 0 |

Radius 0.3 in the (person, glue) plane, so no position looks like a setting; all ten rows differ.

## Projection matrices (rows = input axis, columns = output axis; zeros left as 0)

### W_Q (e -> q): 'axis -> asks'

| e axis \ q axis | water? | finance? | who? |
|---|---:|---:|---:|
| water | 1.0 | 0 | 0 |
| finance | 0 | 1.0 | 0 |
| person | 0 | 0 | 0.4 |
| glue | 0.7 | 0.7 | 0 |

Reading: water asks water?, finance asks finance?, person asks who? (0.4), glue asks both settings (0.7, 0.7). Glue words ask 'what setting am I in?', which is why the final 'the' reads river or cheque.

### W_K (e -> k): 'axis -> offers'

| e axis \ k axis | water? | finance? | who? |
|---|---:|---:|---:|
| water | 1.0 | 0 | 0 |
| finance | 0 | 1.0 | 0 |
| person | 0 | 0 | 1.0 |
| glue | 0 | 0 | 0 |

Reading: water offers water, finance offers finance, person offers a person, glue offers nothing.

### W_V (e -> v): 'axis -> says'

| e axis \ v axis | →water | →finance | →person |
|---|---:|---:|---:|
| water | 1.0 | 0 | 0 |
| finance | 0 | 1.0 | 0 |
| person | 0 | 0 | 1.0 |
| glue | 0 | 0 | 0 |

Reading: water says water scene, finance says finance scene, person says a person is here, glue says nothing.

### W_O (v -> e): back onto the e axes

| v axis \ e axis | water | finance | person | glue |
|---|---:|---:|---:|---:|
| →water | 1.0 | 0 | 0 | 0 |
| →finance | 0 | 1.0 | 0 | 0 |
| →person | 0 | 0 | 0.5 | 0 |

Reading: says water -> water 1.0, says finance -> finance 1.0, says person -> person 0.5, nothing lands on glue.

### W_vocab (e -> logits): 'axis -> votes for these words' (only the non-zero columns; every other entry is 0)

| e axis \ word | fisherman | water | boats | fish | ducks | teller | clerk | queue | money |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| water | 0 | 1.5 | 1.2 | 1.0 | 0.7 | 0 | 0 | 0 | 0 |
| finance | 0 | 0 | 0 | 0 | 0 | 1.5 | 1.2 | 0.9 | 0.7 |
| person | 0.2 | 0 | 0 | 0 | 0 | 0.2 | 0.2 | 0 | 0 |
| glue | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

b_vocab: -1.5 for the, fisherman, sat, beside, river, bank, and, watched, she, deposited, cheque, at; 0 for water, boats, fish, ducks, teller, clerk, queue, money.

## T1 - S_A `The fisherman sat beside the river bank`, query bank(7), causal

### attention row of bank(7)

| position | token | weight |
|---:|---|---:|
| 1 | The | 0.04 |
| 2 | fisherman | 0.20 |
| 3 | sat | 0.06 |
| 4 | beside | 0.07 |
| 5 | the | 0.05 |
| 6 | river | 0.45 |
| 7 | bank | 0.13 |

Target: river >= .40, fisherman second, bank(self) and the glue words below fisherman: **PASS**

## T2 - S_B `She deposited the cheque at the bank`, query bank(7), causal

### attention row of bank(7)

| position | token | weight |
|---:|---|---:|
| 1 | She | 0.04 |
| 2 | deposited | 0.24 |
| 3 | the | 0.04 |
| 4 | cheque | 0.45 |
| 5 | at | 0.05 |
| 6 | the | 0.05 |
| 7 | bank | 0.13 |

Target: cheque highest, deposited second, others low: **PASS**

## T3 - S_A, query the(10): attention and next-token probabilities

### attention row of the(10)

| position | token | weight |
|---:|---|---:|
| 1 | The | 0.03 |
| 2 | fisherman | 0.21 |
| 3 | sat | 0.04 |
| 4 | beside | 0.04 |
| 5 | the | 0.02 |
| 6 | river | 0.46 |
| 7 | bank | 0.10 |
| 8 | and | 0.03 |
| 9 | watched | 0.05 |
| 10 | the | 0.03 |

### p(next token | S_A) from e'_the(10)

| token | probability |
|---|---:|
| water | 0.392 |
| boats | 0.219 |
| fish | 0.149 |
| ducks | 0.083 |
| every other token (max: teller) | 0.026 |
| (sum of the other 16) | 0.156 |

Target: water > boats > fish > ducks, every other token <= .04; attention mostly on river, bank, fisherman: **PASS**

## T4 - S_B, query the(10): attention and next-token probabilities

### attention row of the(10)

| position | token | weight |
|---:|---|---:|
| 1 | She | 0.03 |
| 2 | deposited | 0.23 |
| 3 | the | 0.03 |
| 4 | cheque | 0.46 |
| 5 | at | 0.03 |
| 6 | the | 0.03 |
| 7 | bank | 0.10 |
| 8 | and | 0.03 |
| 9 | watched | 0.05 |
| 10 | the | 0.03 |

### p(next token | S_B) from e'_the(10)

| token | probability |
|---|---:|
| teller | 0.420 |
| clerk | 0.233 |
| queue | 0.120 |
| money | 0.081 |
| every other token (max: water) | 0.024 |
| (sum of the other 16) | 0.146 |

Target: teller > clerk > queue > money, every other token <= .04; attention mostly on cheque, bank, deposited: **PASS**

## T5 - baseline (no attention): softmax(e^(0)_the(10) W_vocab + b)

Identical for both sentences (same token, same position 10). e^(0)_the(10) has nothing on the water or finance axes, so the only differences come from the position offset on the person axis (which votes a little for teller and clerk).

| token | probability |
|---|---:|
| water | 0.092 |
| boats | 0.092 |
| fish | 0.092 |
| ducks | 0.092 |
| teller | 0.098 |
| clerk | 0.098 |
| queue | 0.092 |
| money | 0.092 |
| every other token (max) | 0.022 |

Target: each candidate in [.06, .18], every other token <= .04: **PASS**

## T6 - causal mask OFF on S_A (leakage onto future positions)

| query | The(1) | fisherman(2) | sat(3) | beside(4) | the(5) | river(6) | bank(7) | and(8) | watched(9) | the(10) | river+bank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| The(1) | 0.03 | 0.20 | 0.04 | 0.05 | 0.03 | 0.43 | 0.10 | 0.03 | 0.05 | 0.03 | **0.53** |
| fisherman(2) | 0.02 | 0.45 | 0.03 | 0.03 | 0.01 | 0.34 | 0.03 | 0.02 | 0.05 | 0.02 | **0.37** |
| sat(3) | 0.03 | 0.23 | 0.05 | 0.05 | 0.03 | 0.39 | 0.09 | 0.03 | 0.06 | 0.03 | **0.48** |
| beside(4) | 0.02 | 0.17 | 0.03 | 0.04 | 0.02 | 0.55 | 0.08 | 0.02 | 0.04 | 0.02 | **0.63** |
| the(5) | 0.03 | 0.16 | 0.04 | 0.05 | 0.03 | 0.50 | 0.10 | 0.03 | 0.04 | 0.03 | **0.60** |

Soft target (>= .25 on river+bank from some early token): **PASS**, strongest leak from beside(4) (0.63). Every glue word asks for its setting, so with the mask off each of them reads river before river has been written; the page can highlight any of The(1), beside(4) or the(5).

## The 'bank' story for section 10 (2 decimals, e axes: water, finance, person, glue)

| quantity | S_A (river) | S_B (cheque) |
|---|---|---|
| e_bank^(0) (identical) | [0.70, 0.70, -0.10, 0.90] | [0.70, 0.70, -0.10, 0.90] |
| q_bank (q axes) | [1.33, 1.33, -0.04] | [1.33, 1.33, -0.04] |
| m_bank (v axes) | [1.90, 0.09, 0.36] | [0.09, 1.97, 0.15] |
| Delta e_bank | [1.90, 0.09, 0.18, 0.00] | [0.09, 1.97, 0.08, 0.00] |
| e'_bank = e + Delta e | [2.60, 0.79, 0.08, 0.90] | [0.79, 2.67, -0.02, 0.90] |
| e_the(10)^(0) (identical) | [0.00, 0.00, 0.30, 2.40] | [0.00, 0.00, 0.30, 2.40] |
| q_the(10) (q axes) | [1.68, 1.68, 0.12] | [1.68, 1.68, 0.12] |
| m_the(10) (v axes) | [1.93, 0.07, 0.44] | [0.10, 1.96, 0.19] |
| Delta e_the(10) | [1.93, 0.07, 0.22, 0.00] | [0.10, 1.96, 0.09, 0.00] |
| e'_the(10) | [1.93, 0.07, 0.52, 2.40] | [0.10, 1.96, 0.39, 2.40] |

Reading: in S_A the update to bank lands on the water axis, in S_B on the finance axis; the same e_bank^(0) ends up as two different e'_bank. The final 'the' receives the same kind of update, and the output head turns it into water words or finance words.

## Keys and values of every token in both sentences (what each record offers and says)

### S_A

| pos | token | e (water, finance, person, glue) | q (water?, finance?, who?) | k (same axes) | v (→water, →finance, →person) |
|---:|---|---|---|---|---|
| 1 | The | [0.00, 0.00, 0.20, 2.20] | [1.54, 1.54, 0.08] | [0.00, 0.00, 0.20] | [0.00, 0.00, 0.20] |
| 2 | fisherman | [2.00, 0.00, 2.30, -0.30] | [1.79, -0.21, 0.92] | [2.00, 0.00, 2.30] | [2.00, 0.00, 2.30] |
| 3 | sat | [0.40, 0.00, 0.50, 1.50] | [1.45, 1.05, 0.20] | [0.40, 0.00, 0.50] | [0.40, 0.00, 0.50] |
| 4 | beside | [0.60, 0.00, -0.20, 1.80] | [1.86, 1.26, -0.08] | [0.60, 0.00, -0.20] | [0.60, 0.00, -0.20] |
| 5 | the | [0.00, 0.00, -0.30, 2.40] | [1.68, 1.68, -0.12] | [0.00, 0.00, -0.30] | [0.00, 0.00, -0.30] |
| 6 | river | [3.00, 0.00, -0.20, 0.20] | [3.14, 0.14, -0.08] | [3.00, 0.00, -0.20] | [3.00, 0.00, -0.20] |
| 7 | bank | [0.70, 0.70, -0.10, 0.90] | [1.33, 1.33, -0.04] | [0.70, 0.70, -0.10] | [0.70, 0.70, -0.10] |
| 8 | and | [0.00, 0.00, 0.10, 2.50] | [1.75, 1.75, 0.04] | [0.00, 0.00, 0.10] | [0.00, 0.00, 0.10] |
| 9 | watched | [0.60, 0.00, 1.00, 1.80] | [1.86, 1.26, 0.40] | [0.60, 0.00, 1.00] | [0.60, 0.00, 1.00] |
| 10 | the | [0.00, 0.00, 0.30, 2.40] | [1.68, 1.68, 0.12] | [0.00, 0.00, 0.30] | [0.00, 0.00, 0.30] |

### S_B

| pos | token | e (water, finance, person, glue) | q (water?, finance?, who?) | k (same axes) | v (→water, →finance, →person) |
|---:|---|---|---|---|---|
| 1 | She | [0.00, 0.00, 2.80, 0.20] | [0.14, 0.14, 1.12] | [0.00, 0.00, 2.80] | [0.00, 0.00, 2.80] |
| 2 | deposited | [0.00, 2.20, 0.70, 0.10] | [0.07, 2.27, 0.28] | [0.00, 2.20, 0.70] | [0.00, 2.20, 0.70] |
| 3 | the | [0.00, 0.00, -0.10, 2.10] | [1.47, 1.47, -0.04] | [0.00, 0.00, -0.10] | [0.00, 0.00, -0.10] |
| 4 | cheque | [0.00, 3.00, -0.20, -0.20] | [-0.14, 2.86, -0.08] | [0.00, 3.00, -0.20] | [0.00, 3.00, -0.20] |
| 5 | at | [0.00, 0.00, -0.30, 2.00] | [1.40, 1.40, -0.12] | [0.00, 0.00, -0.30] | [0.00, 0.00, -0.30] |
| 6 | the | [0.00, 0.00, -0.20, 2.60] | [1.82, 1.82, -0.08] | [0.00, 0.00, -0.20] | [0.00, 0.00, -0.20] |
| 7 | bank | [0.70, 0.70, -0.10, 0.90] | [1.33, 1.33, -0.04] | [0.70, 0.70, -0.10] | [0.70, 0.70, -0.10] |
| 8 | and | [0.00, 0.00, 0.10, 2.50] | [1.75, 1.75, 0.04] | [0.00, 0.00, 0.10] | [0.00, 0.00, 0.10] |
| 9 | watched | [0.60, 0.00, 1.00, 1.80] | [1.86, 1.26, 0.40] | [0.60, 0.00, 1.00] | [0.60, 0.00, 1.00] |
| 10 | the | [0.00, 0.00, 0.30, 2.40] | [1.68, 1.68, 0.12] | [0.00, 0.00, 0.30] | [0.00, 0.00, 0.30] |

## T7 / T9 - magnitudes

- 268 parameters, all one-decimal, max |x| = 3.0 (limit 3.0).
- token-embedding norms: min 1.16, mean 2.46, max 3.16; positional norms: min 0.28, max 0.32 (13% of the mean token norm).
- e^(0) of the three 'the' in S_A: pos 1 [0.00, 0.00, 0.20, 2.20], pos 5 [0.00, 0.00, -0.30, 2.40], pos 10 [0.00, 0.00, 0.30, 2.40] (all different).

## Score ranges (for heat-map colour scales)

- S_A: scaled causal scores s_ij in [-0.07, 5.45]; raw q.k in [-0.12, 9.44]; |Delta e| rows up to 2.81; logits of the(10) in [-1.50, 2.90].
- S_B: scaled causal scores s_ij in [-0.19, 4.96]; raw q.k in [-0.34, 8.60]; |Delta e| rows up to 2.81; logits of the(10) in [-1.50, 3.03].

## Check summary (make_toy2.py --check-only)

- [PASS] T1 hard: bank(7) in S_A: river >= .40, fisherman second, bank(self) and glue words below fisherman: bank(7) row S_A: The=0.04 fisherman=0.20 sat=0.06 beside=0.07 the=0.05 river=0.45 bank=0.13
- [PASS] T2 hard: bank(7) in S_B: cheque highest, deposited second, others low: bank(7) row S_B: She=0.04 deposited=0.24 the=0.04 cheque=0.45 at=0.05 the=0.05 bank=0.13
- [PASS] T3 hard: the(10) in S_A: water > boats > fish > ducks, every other token <= .04: water=0.392 boats=0.219 fish=0.149 ducks=0.083 | max other=0.026
- [PASS] T3 hard: attention of the(10) mostly on river, bank, fisherman (sum >= .60, each of them top-3): the(10) row S_A: The=0.03 fisherman=0.21 sat=0.04 beside=0.04 the=0.02 river=0.46 bank=0.10 and=0.03 watched=0.05 the=0.03
- [PASS] T4 hard: the(10) in S_B: teller > clerk > queue > money, every other token <= .04: teller=0.420 clerk=0.233 queue=0.120 money=0.081 | max other=0.024
- [PASS] T4 hard: attention of the(10) mostly on cheque, bank, deposited (sum >= .60, each of them top-3): the(10) row S_B: She=0.03 deposited=0.23 the=0.03 cheque=0.46 at=0.03 the=0.03 bank=0.10 and=0.03 watched=0.05 the=0.03
- [PASS] T5 hard: baseline (no attention) candidates each in [.06,.18], others <= .04, identical for S_A/S_B: water=0.092 boats=0.092 fish=0.092 ducks=0.092 teller=0.098 clerk=0.098 queue=0.092 money=0.092 | max other=0.022
- [PASS] (soft) T6 soft: mask off, some early token of S_A puts >= .25 on future river/bank: The(1):0.53 fisherman(2):0.37 sat(3):0.48 beside(4):0.63 the(5):0.60
- [PASS] T7 hard: all |x| <= 3.0 and one decimal: max|x|=3.0
- [PASS] T8 hard: vocabulary is the 20 tokens: 20 tokens
- [PASS] T9 hard: pos norms <= 40% of mean token norm; the three 'the' differ: max pos norm=0.32, mean tok norm=2.46 (13%)
- [PASS] T10 hard: d_model=4, d_k=d_v=3: ok

