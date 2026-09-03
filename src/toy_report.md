# toy_report.md - numbers produced by the hand-designed toy.json (v2, named axes)

All values below are computed from the ONE-DECIMAL parameters in `toy.json` (370 numbers, max |x| = 3.0) by `make_toy2.py`; `node toy_ref.mjs --compare py_check.json` reproduces every intermediate to < 1e-12. Attention weights are shown to 2 decimals, probabilities to 3. The previous, optimised toy is kept as `toy_v1.json` (its report: `toy_report_v1.md`).

## How the numbers were obtained

Nothing was optimised. AXES.md names every coordinate, and the embeddings and projection matrices were written by hand so that the names are true: each row of a projection reads as 'input axis -> asks / offers / says'. The forward pass (the same arithmetic as `toy_ref.mjs`) was then evaluated and a few magnitudes were adjusted from the AXES.md starting point until the targets held on the rounded numbers: `bank` became [0.7, 0.7, 0, 0.7, 0] and `fisherman` became [2.0, 0, 2.2, 0, 0], because the self score q_bank . k_bank grows with the square of bank's water/finance entries and at 1.5 bank attended to itself as much as to river; the W_vocab weights were raised to 1.5/1.2/1.0/0.7 (water words) and 1.5/1.2/0.9/0.7 (finance words) so that every non-candidate word stays at or below 0.04. Position has a dedicated coordinate only as a toy simplification. It rises from 0.1 to 2.0 across 20 supported positions, while its rows in W_Q, W_K, W_V and W_vocab are zero. The toy demonstrates content routing, not sensitivity to word order.

## Axes (from `toy.json` -> `axes`)

| object | coordinate 1 | coordinate 2 | coordinate 3 | coordinate 4 | coordinate 5 |
|---|---|---|---|---|---|
| e, Delta e, e' (d_model = 5) | water | finance | person | glue | position |
| q, k (d_k = 3) | setting: water? | setting: finance? | who? |  |  |
| v, m (d_v = 2) | says: water scene | says: finance scene |  |  |  |

Reading rule: a query row is 'what I ask for', a key row is 'what I offer', a value row is 'what I send if retrieved'. W_O maps the two v axes back onto the water and finance axes of e.

## Token embeddings (e axes)

| token | water | finance | person | glue | pos | reading |
|---|---:|---:|---:|---:|---:|---|
| the | 0 | 0 | 0 | 2.4 | 0 | glue 2.4 |
| fisherman | 2.0 | 0 | 2.2 | 0 | 0 | water 2.0, person 2.2 |
| sat | 0.4 | 0 | 0.6 | 1.8 | 0 | water 0.4, person 0.6, glue 1.8 |
| beside | 0.6 | 0 | 0 | 2.0 | 0 | water 0.6, glue 2.0 |
| river | 3.0 | 0 | 0 | 0 | 0 | water 3.0 |
| bank | 0.7 | 0.7 | 0 | 0.7 | 0 | water 0.7, finance 0.7, glue 0.7 |
| and | 0 | 0 | 0 | 2.2 | 0 | glue 2.2 |
| watched | 0.6 | 0 | 0.8 | 1.6 | 0 | water 0.6, person 0.8, glue 1.6 |
| she | 0 | 0 | 2.6 | 0.4 | 0 | person 2.6, glue 0.4 |
| deposited | 0 | 2.2 | 0.6 | 0.4 | 0 | finance 2.2, person 0.6, glue 0.4 |
| cheque | 0 | 3.0 | 0 | 0 | 0 | finance 3.0 |
| at | 0 | 0 | 0 | 2.0 | 0 | glue 2.0 |
| water | 3.0 | 0 | 0 | 0 | 0 | water 3.0 |
| boats | 2.4 | 0 | 0.4 | 0 | 0 | water 2.4, person 0.4 |
| fish | 2.6 | 0 | 0.6 | 0 | 0 | water 2.6, person 0.6 |
| ducks | 2.2 | 0 | 0.8 | 0 | 0 | water 2.2, person 0.8 |
| teller | 0 | 2.6 | 1.8 | 0 | 0 | finance 2.6, person 1.8 |
| clerk | 0 | 2.4 | 1.8 | 0 | 0 | finance 2.4, person 1.8 |
| queue | 0 | 2.0 | 0.6 | 0 | 0 | finance 2.0, person 0.6 |
| money | 0 | 3.0 | 0 | 0 | 0 | finance 3.0 |

## Position offsets (added to the token row; e^(0)_i = tok_emb[token_i] + pos_emb[i])

| position | water | finance | person | glue | pos |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0.1 |
| 2 | 0 | 0 | 0 | 0 | 0.2 |
| 3 | 0 | 0 | 0 | 0 | 0.3 |
| 4 | 0 | 0 | 0 | 0 | 0.4 |
| 5 | 0 | 0 | 0 | 0 | 0.5 |
| 6 | 0 | 0 | 0 | 0 | 0.6 |
| 7 | 0 | 0 | 0 | 0 | 0.7 |
| 8 | 0 | 0 | 0 | 0 | 0.8 |
| 9 | 0 | 0 | 0 | 0 | 0.9 |
| 10 | 0 | 0 | 0 | 0 | 1.0 |
| 11 | 0 | 0 | 0 | 0 | 1.1 |
| 12 | 0 | 0 | 0 | 0 | 1.2 |
| 13 | 0 | 0 | 0 | 0 | 1.3 |
| 14 | 0 | 0 | 0 | 0 | 1.4 |
| 15 | 0 | 0 | 0 | 0 | 1.5 |
| 16 | 0 | 0 | 0 | 0 | 1.6 |
| 17 | 0 | 0 | 0 | 0 | 1.7 |
| 18 | 0 | 0 | 0 | 0 | 1.8 |
| 19 | 0 | 0 | 0 | 0 | 1.9 |
| 20 | 0 | 0 | 0 | 0 | 2.0 |

Only the position coordinate changes. The attention projections and output head ignore that coordinate in this toy.

## Projection matrices (rows = input axis, columns = output axis; zeros left as 0)

### W_Q (e -> q): 'axis -> asks'

| e axis \ q axis | water? | finance? | who? |
|---|---:|---:|---:|
| water | 1.0 | 0 | 0 |
| finance | 0 | 1.0 | 0 |
| person | 0 | 0 | 0.4 |
| glue | 0.7 | 0.7 | 0 |
| pos | 0 | 0 | 0 |

Reading: water asks water?, finance asks finance?, person asks who? (0.4), glue asks both settings (0.7, 0.7). Glue words ask 'what setting am I in?', which is why the final 'the' reads river or cheque.

### W_K (e -> k): 'axis -> offers'

| e axis \ k axis | water? | finance? | who? |
|---|---:|---:|---:|
| water | 1.0 | 0 | 0 |
| finance | 0 | 1.0 | 0 |
| person | 0 | 0 | 1.0 |
| glue | 0 | 0 | 0 |
| pos | 0 | 0 | 0 |

Reading: water offers water, finance offers finance, person offers a person, glue offers nothing.

### W_V (e -> v): 'axis -> says'

| e axis \ v axis | →water | →finance |
|---|---:|---:|
| water | 1.0 | 0 |
| finance | 0 | 1.0 |
| person | 0 | 0 |
| glue | 0 | 0 |
| pos | 0 | 0 |

Reading: water says water scene and finance says finance scene. The other input axes send zeros in this toy.

### W_O (v -> e): back onto the e axes

| v axis \ e axis | water | finance | person | glue | pos |
|---|---:|---:|---:|---:|---:|
| →water | 1.0 | 0 | 0 | 0 | 0 |
| →finance | 0 | 1.0 | 0 | 0 | 0 |

Reading: says water -> water 1.0 and says finance -> finance 1.0. The other e axes receive zero.

### W_vocab (e -> logits): 'axis -> votes for these words' (only the non-zero columns; every other entry is 0)

| e axis \ word | fisherman | water | boats | fish | ducks | teller | clerk | queue | money |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| water | 0 | 1.5 | 1.2 | 1.0 | 0.7 | 0 | 0 | 0 | 0 |
| finance | 0 | 0 | 0 | 0 | 0 | 1.5 | 1.2 | 0.9 | 0.7 |
| person | 0.2 | 0 | 0 | 0 | 0 | 0.2 | 0.2 | 0 | 0 |
| glue | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pos | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

b_vocab: -1.5 for the, fisherman, sat, beside, river, bank, and, watched, she, deposited, cheque, at; 0 for water, boats, fish, ducks, teller, clerk, queue, money.

## T1 - S_A `The fisherman sat beside the river bank`, query bank(7), causal

### attention row of bank(7)

| position | token | weight |
|---:|---|---:|
| 1 | The | 0.05 |
| 2 | fisherman | 0.21 |
| 3 | sat | 0.07 |
| 4 | beside | 0.08 |
| 5 | the | 0.05 |
| 6 | river | 0.41 |
| 7 | bank | 0.14 |

Target: river >= .40, fisherman second, bank(self) and the glue words below fisherman: **PASS**

## T2 - S_B `She deposited the cheque at the bank`, query bank(7), causal

### attention row of bank(7)

| position | token | weight |
|---:|---|---:|
| 1 | She | 0.05 |
| 2 | deposited | 0.24 |
| 3 | the | 0.05 |
| 4 | cheque | 0.41 |
| 5 | at | 0.05 |
| 6 | the | 0.05 |
| 7 | bank | 0.14 |

Target: cheque highest, deposited second, others low: **PASS**

## T3 - S_A, query the(10): attention and next-token probabilities

### attention row of the(10)

| position | token | weight |
|---:|---|---:|
| 1 | The | 0.03 |
| 2 | fisherman | 0.18 |
| 3 | sat | 0.04 |
| 4 | beside | 0.05 |
| 5 | the | 0.03 |
| 6 | river | 0.48 |
| 7 | bank | 0.10 |
| 8 | and | 0.03 |
| 9 | watched | 0.05 |
| 10 | the | 0.03 |

### p(next token | S_A) from e'_the(10)

| token | probability |
|---|---:|
| water | 0.396 |
| boats | 0.221 |
| fish | 0.150 |
| ducks | 0.084 |
| every other token (max: teller) | 0.024 |
| (sum of the other 16) | 0.150 |

Target: water > boats > fish > ducks, every other token <= .04; attention mostly on river, bank, fisherman: **PASS**

## T4 - S_B, query the(10): attention and next-token probabilities

### attention row of the(10)

| position | token | weight |
|---:|---|---:|
| 1 | She | 0.03 |
| 2 | deposited | 0.22 |
| 3 | the | 0.03 |
| 4 | cheque | 0.48 |
| 5 | at | 0.03 |
| 6 | the | 0.03 |
| 7 | bank | 0.10 |
| 8 | and | 0.03 |
| 9 | watched | 0.05 |
| 10 | the | 0.03 |

### p(next token | S_B) from e'_the(10)

| token | probability |
|---|---:|
| teller | 0.413 |
| clerk | 0.228 |
| queue | 0.125 |
| money | 0.084 |
| every other token (max: water) | 0.024 |
| (sum of the other 16) | 0.150 |

Target: teller > clerk > queue > money, every other token <= .04; attention mostly on cheque, bank, deposited: **PASS**

## T5 - baseline (no attention): softmax(e^(0)_the(10) W_vocab + b)

Identical for both sentences (same token, same position 10). e^(0)_the(10) has nothing on the water or finance axes, and the output head ignores its position coordinate.

| token | probability |
|---|---:|
| water | 0.094 |
| boats | 0.094 |
| fish | 0.094 |
| ducks | 0.094 |
| teller | 0.094 |
| clerk | 0.094 |
| queue | 0.094 |
| money | 0.094 |
| every other token (max) | 0.021 |

Target: each candidate in [.06, .18], every other token <= .04: **PASS**

## T6 - causal mask OFF on S_A (leakage onto future positions)

| query | The(1) | fisherman(2) | sat(3) | beside(4) | the(5) | river(6) | bank(7) | and(8) | watched(9) | the(10) | river+bank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| The(1) | 0.03 | 0.18 | 0.04 | 0.05 | 0.03 | 0.48 | 0.10 | 0.03 | 0.05 | 0.03 | **0.58** |
| fisherman(2) | 0.01 | 0.40 | 0.03 | 0.03 | 0.01 | 0.42 | 0.03 | 0.01 | 0.04 | 0.01 | **0.45** |
| sat(3) | 0.03 | 0.23 | 0.04 | 0.04 | 0.03 | 0.45 | 0.08 | 0.03 | 0.05 | 0.03 | **0.53** |
| beside(4) | 0.02 | 0.18 | 0.03 | 0.04 | 0.02 | 0.58 | 0.07 | 0.02 | 0.04 | 0.02 | **0.65** |
| the(5) | 0.03 | 0.18 | 0.04 | 0.05 | 0.03 | 0.48 | 0.10 | 0.03 | 0.05 | 0.03 | **0.58** |

Soft target (>= .25 on river+bank from some early token): **PASS**, strongest leak from beside(4) (0.65). Every glue word asks for its setting, so with the mask off each of them reads river before river has been written; the page can highlight any of The(1), beside(4) or the(5).

## The 'bank' story for section 10 (2 decimals, e axes: water, finance, person, glue, pos)

| quantity | S_A (river) | S_B (cheque) |
|---|---|---|
| e_bank^(0) (identical) | [0.70, 0.70, 0.00, 0.70, 0.70] | [0.70, 0.70, 0.00, 0.70, 0.70] |
| q_bank (q axes) | [1.19, 1.19, 0.00] | [1.19, 1.19, 0.00] |
| m_bank (v axes) | [1.80, 0.10] | [0.10, 1.86] |
| Delta e_bank | [1.80, 0.10, 0.00, 0.00, 0.00] | [0.10, 1.86, 0.00, 0.00, 0.00] |
| e'_bank = e + Delta e | [2.50, 0.80, 0.00, 0.70, 0.70] | [0.80, 2.56, 0.00, 0.70, 0.70] |
| e_the(10)^(0) (identical) | [0.00, 0.00, 0.00, 2.40, 1.00] | [0.00, 0.00, 0.00, 2.40, 1.00] |
| q_the(10) (q axes) | [1.68, 1.68, 0.00] | [1.68, 1.68, 0.00] |
| m_the(10) (v axes) | [1.95, 0.07] | [0.10, 1.98] |
| Delta e_the(10) | [1.95, 0.07, 0.00, 0.00, 0.00] | [0.10, 1.98, 0.00, 0.00, 0.00] |
| e'_the(10) | [1.95, 0.07, 0.00, 2.40, 1.00] | [0.10, 1.98, 0.00, 2.40, 1.00] |

Reading: in S_A the update to bank lands on the water axis, in S_B on the finance axis; the same e_bank^(0) ends up as two different e'_bank. The final 'the' receives the same kind of update, and the output head turns it into water words or finance words.

## Keys and values of every token in both sentences (what each record offers and says)

### S_A

| pos | token | e (water, finance, person, glue, pos) | q (water?, finance?, who?) | k (same axes) | v (→water, →finance) |
|---:|---|---|---|---|---|
| 1 | The | [0.00, 0.00, 0.00, 2.40, 0.10] | [1.68, 1.68, 0.00] | [0.00, 0.00, 0.00] | [0.00, 0.00] |
| 2 | fisherman | [2.00, 0.00, 2.20, 0.00, 0.20] | [2.00, 0.00, 0.88] | [2.00, 0.00, 2.20] | [2.00, 0.00] |
| 3 | sat | [0.40, 0.00, 0.60, 1.80, 0.30] | [1.66, 1.26, 0.24] | [0.40, 0.00, 0.60] | [0.40, 0.00] |
| 4 | beside | [0.60, 0.00, 0.00, 2.00, 0.40] | [2.00, 1.40, 0.00] | [0.60, 0.00, 0.00] | [0.60, 0.00] |
| 5 | the | [0.00, 0.00, 0.00, 2.40, 0.50] | [1.68, 1.68, 0.00] | [0.00, 0.00, 0.00] | [0.00, 0.00] |
| 6 | river | [3.00, 0.00, 0.00, 0.00, 0.60] | [3.00, 0.00, 0.00] | [3.00, 0.00, 0.00] | [3.00, 0.00] |
| 7 | bank | [0.70, 0.70, 0.00, 0.70, 0.70] | [1.19, 1.19, 0.00] | [0.70, 0.70, 0.00] | [0.70, 0.70] |
| 8 | and | [0.00, 0.00, 0.00, 2.20, 0.80] | [1.54, 1.54, 0.00] | [0.00, 0.00, 0.00] | [0.00, 0.00] |
| 9 | watched | [0.60, 0.00, 0.80, 1.60, 0.90] | [1.72, 1.12, 0.32] | [0.60, 0.00, 0.80] | [0.60, 0.00] |
| 10 | the | [0.00, 0.00, 0.00, 2.40, 1.00] | [1.68, 1.68, 0.00] | [0.00, 0.00, 0.00] | [0.00, 0.00] |

### S_B

| pos | token | e (water, finance, person, glue, pos) | q (water?, finance?, who?) | k (same axes) | v (→water, →finance) |
|---:|---|---|---|---|---|
| 1 | She | [0.00, 0.00, 2.60, 0.40, 0.10] | [0.28, 0.28, 1.04] | [0.00, 0.00, 2.60] | [0.00, 0.00] |
| 2 | deposited | [0.00, 2.20, 0.60, 0.40, 0.20] | [0.28, 2.48, 0.24] | [0.00, 2.20, 0.60] | [0.00, 2.20] |
| 3 | the | [0.00, 0.00, 0.00, 2.40, 0.30] | [1.68, 1.68, 0.00] | [0.00, 0.00, 0.00] | [0.00, 0.00] |
| 4 | cheque | [0.00, 3.00, 0.00, 0.00, 0.40] | [0.00, 3.00, 0.00] | [0.00, 3.00, 0.00] | [0.00, 3.00] |
| 5 | at | [0.00, 0.00, 0.00, 2.00, 0.50] | [1.40, 1.40, 0.00] | [0.00, 0.00, 0.00] | [0.00, 0.00] |
| 6 | the | [0.00, 0.00, 0.00, 2.40, 0.60] | [1.68, 1.68, 0.00] | [0.00, 0.00, 0.00] | [0.00, 0.00] |
| 7 | bank | [0.70, 0.70, 0.00, 0.70, 0.70] | [1.19, 1.19, 0.00] | [0.70, 0.70, 0.00] | [0.70, 0.70] |
| 8 | and | [0.00, 0.00, 0.00, 2.20, 0.80] | [1.54, 1.54, 0.00] | [0.00, 0.00, 0.00] | [0.00, 0.00] |
| 9 | watched | [0.60, 0.00, 0.80, 1.60, 0.90] | [1.72, 1.12, 0.32] | [0.60, 0.00, 0.80] | [0.60, 0.00] |
| 10 | the | [0.00, 0.00, 0.00, 2.40, 1.00] | [1.68, 1.68, 0.00] | [0.00, 0.00, 0.00] | [0.00, 0.00] |

## T7 / T9 - magnitudes

- 370 parameters, all one-decimal, max |x| = 3.0 (limit 3.0).
- token-embedding norms: min 1.21, mean 2.47, max 3.16; positional norms: min 0.10, max 2.00 (81% of the mean token norm).
- e^(0) of the three 'the' in S_A: pos 1 [0.00, 0.00, 0.00, 2.40, 0.10], pos 5 [0.00, 0.00, 0.00, 2.40, 0.50], pos 10 [0.00, 0.00, 0.00, 2.40, 1.00] (all different).

## Score ranges (for heat-map colour scales)

- S_A: scaled causal scores s_ij in [0.00, 5.20]; raw q.k in [0.00, 9.00]; |Delta e| rows up to 2.77; logits of the(10) in [-1.50, 2.92].
- S_B: scaled causal scores s_ij in [0.00, 5.20]; raw q.k in [0.00, 9.00]; |Delta e| rows up to 2.81; logits of the(10) in [-1.50, 2.98].

## Check summary (make_toy2.py --check-only)

- [PASS] T1 hard: bank(7) in S_A: river >= .40, fisherman second, bank(self) and glue words below fisherman: bank(7) row S_A: The=0.05 fisherman=0.21 sat=0.07 beside=0.08 the=0.05 river=0.41 bank=0.14
- [PASS] T2 hard: bank(7) in S_B: cheque highest, deposited second, others low: bank(7) row S_B: She=0.05 deposited=0.24 the=0.05 cheque=0.41 at=0.05 the=0.05 bank=0.14
- [PASS] T3 hard: the(10) in S_A: water > boats > fish > ducks, every other token <= .04: water=0.396 boats=0.221 fish=0.150 ducks=0.084 | max other=0.024
- [PASS] T3 hard: attention of the(10) mostly on river, bank, fisherman (sum >= .60, each of them top-3): the(10) row S_A: The=0.03 fisherman=0.18 sat=0.04 beside=0.05 the=0.03 river=0.48 bank=0.10 and=0.03 watched=0.05 the=0.03
- [PASS] T4 hard: the(10) in S_B: teller > clerk > queue > money, every other token <= .04: teller=0.413 clerk=0.228 queue=0.125 money=0.084 | max other=0.024
- [PASS] T4 hard: attention of the(10) mostly on cheque, bank, deposited (sum >= .60, each of them top-3): the(10) row S_B: She=0.03 deposited=0.22 the=0.03 cheque=0.48 at=0.03 the=0.03 bank=0.10 and=0.03 watched=0.05 the=0.03
- [PASS] T5 hard: baseline (no attention) candidates each in [.06,.18], others <= .04, identical for S_A/S_B: water=0.094 boats=0.094 fish=0.094 ducks=0.094 teller=0.094 clerk=0.094 queue=0.094 money=0.094 | max other=0.021
- [PASS] (soft) T6 soft: mask off, some early token of S_A puts >= .25 on future river/bank: The(1):0.58 fisherman(2):0.45 sat(3):0.53 beside(4):0.65 the(5):0.58
- [PASS] T7 hard: all |x| <= 3.0 and one decimal: max|x|=3.0
- [PASS] T8 hard: vocabulary is the 20 tokens: 20 tokens
- [PASS] T9 hard: position has its own coordinate and the three 'the' rows differ: position rows 0.1 to 2.0; token and projection position rows are zero
- [PASS] T10 hard: parameter shapes match d_model, d_k and d_v: ok

