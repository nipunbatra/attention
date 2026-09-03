# toy_report.md - numbers produced by the rounded toy.json

All values below are computed from the ONE-DECIMAL parameters in `toy.json` (268 numbers, max |x| = 1.9) by `make_toy.py --check-only`; `node toy_ref.mjs --compare py_check.json` reproduces every intermediate to < 1e-14. Attention weights are shown to 2 decimals, probabilities to 3.

## How the numbers were obtained

Only numpy was available, so `make_toy.py` contains a ~60-line reverse-mode autodiff (checked against finite differences, relative error 2e-6). The loss is cross-entropy to the soft target rows/distributions of T1-T5, plus hinge penalties with safety margins (0.02 on the T1/T2 orderings, 0.015 on the T3/T4 thresholds, 0.01 on the '<= .03 / <= .05' caps) for every HARD target, a small L2 term, a penalty on |x| > 2.5, a weak term encouraging the unmasked leakage of T6, and a term keeping token norms above 1.4 so the fixed positional table (a scaled-down sinusoid, amplitude 0.4, already 1-decimal, not trained) stays at about a third of a token norm (T9). Adam ran 2000 steps on the continuous parameters (lr 0.03), then 2000 steps with uniform +-0.05 parameter noise (lr 0.01) so the optimum is flat with respect to rounding. Every entry was then rounded to one decimal; the rounded model already satisfied all hard targets, and 12 sweeps of greedy +-0.1 coordinate descent on the grid (accepting any move that lowers the same loss) polished the soft targets. Seed 0; the run is deterministic and takes about 10 s. One visible artefact of the tuning: the 12 non-candidate columns of W_vocab and b_vocab are identical, because nothing in the targets distinguishes those tokens (they only ever need to be improbable).

## T1 - S_A `The fisherman sat beside the river bank`, query bank(7), causal

### attention row of bank(7)

| position | token | weight | target |
|---:|---|---:|---:|
| 1 | The | 0.05 | 0.05 |
| 2 | fisherman | 0.18 | 0.18 |
| 3 | sat | 0.06 | 0.06 |
| 4 | beside | 0.14 | 0.14 |
| 5 | the | 0.05 | 0.05 |
| 6 | river | 0.42 | 0.42 |
| 7 | bank | 0.11 | 0.10 |

Hard ordering river > fisherman > beside > max(sat, the, The, bank): **PASS**

## T2 - S_B `She deposited the cheque at the bank`, query bank(7), causal

### attention row of bank(7)

| position | token | weight | target |
|---:|---|---:|---:|
| 1 | She | 0.07 | 0.07 |
| 2 | deposited | 0.32 | 0.32 |
| 3 | the | 0.04 | 0.04 |
| 4 | cheque | 0.38 | 0.38 |
| 5 | at | 0.05 | 0.05 |
| 6 | the | 0.05 | 0.04 |
| 7 | bank | 0.09 | 0.10 |

Hard: min(cheque, deposited) > every other weight and cheque >= deposited: **PASS**

## T3 - S_A, query the(10): attention and next-token probabilities

### attention row of the(10)

| position | token | weight |
|---:|---|---:|
| 1 | The | 0.02 |
| 2 | fisherman | 0.16 |
| 3 | sat | 0.04 |
| 4 | beside | 0.06 |
| 5 | the | 0.01 |
| 6 | river | 0.43 |
| 7 | bank | 0.24 |
| 8 | and | 0.01 |
| 9 | watched | 0.01 |
| 10 | the | 0.02 |

Hard: river >= .25, bank >= .20, fisherman >= .10, each function word <= .08: **PASS**

### p(next token | S_A) from e'_the(10)

| token | probability | target |
|---|---:|---:|
| water | 0.365 | 0.36 |
| boats | 0.217 | 0.22 |
| fish | 0.183 | 0.18 |
| ducks | 0.097 | 0.10 |
| every other token (max: queue) | 0.010 | <= .03 |
| (sum of the other 16) | 0.138 | |

## T4 - S_B, query the(10): attention and next-token probabilities

### attention row of the(10)

| position | token | weight |
|---:|---|---:|
| 1 | She | 0.05 |
| 2 | deposited | 0.30 |
| 3 | the | 0.01 |
| 4 | cheque | 0.34 |
| 5 | at | 0.03 |
| 6 | the | 0.02 |
| 7 | bank | 0.22 |
| 8 | and | 0.01 |
| 9 | watched | 0.01 |
| 10 | the | 0.02 |

Hard: cheque >= .25, bank >= .20, deposited >= .10, each function word <= .08: **PASS**

### p(next token | S_B) from e'_the(10)

| token | probability | target |
|---|---:|---:|
| teller | 0.337 | 0.34 |
| clerk | 0.240 | 0.24 |
| queue | 0.163 | 0.16 |
| money | 0.119 | 0.12 |
| every other token (max: water) | 0.010 | <= .03 |
| (sum of the other 16) | 0.141 | |

## T5 - baseline (no attention): softmax(e^(0)_the(10) W_vocab + b)

Identical for both sentences (same token, same position 10).

| token | probability |
|---|---:|
| water | 0.105 |
| boats | 0.116 |
| fish | 0.110 |
| ducks | 0.107 |
| teller | 0.106 |
| clerk | 0.111 |
| queue | 0.112 |
| money | 0.108 |
| every other token (max) | 0.011 |

Hard: each candidate in [.06, .16], every other token <= .05: **PASS**

## T6 - causal mask OFF on S_A (leakage onto future positions)

| query | The(1) | fisherman(2) | sat(3) | beside(4) | the(5) | river(6) | bank(7) | and(8) | watched(9) | the(10) | river+bank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sat(3) | 0.11 | 0.03 | 0.08 | 0.05 | 0.12 | 0.02 | 0.04 | 0.21 | 0.21 | 0.13 | **0.06** |
| beside(4) | 0.10 | 0.02 | 0.06 | 0.04 | 0.13 | 0.01 | 0.01 | 0.27 | 0.25 | 0.12 | **0.02** |
| the(5) | 0.06 | 0.15 | 0.08 | 0.09 | 0.04 | 0.23 | 0.22 | 0.03 | 0.04 | 0.06 | **0.46** |

Soft target (>= .25 on {river, bank} from at least one of sat/beside/the(5)): **met by the(5)** (0.46). sat(3) and beside(4) instead leak onto `and`/`watched` (about 0.2 each), which is still future information - a usable teaching point: with the mask off, every early token reads from words it should not be able to see. bank(7) itself, with the mask off, is unaffected in what matters (river still dominates); the page should highlight the(5).

## The 'bank' story for section 10 (2 decimals)

| quantity | S_A (river) | S_B (cheque) |
|---|---|---|
| e_bank^(0) (identical) | [-0.70, 1.30, -0.10, 0.10] | [-0.70, 1.30, -0.10, 0.10] |
| Delta e_bank | [-0.70, -1.52, -1.04, -0.61] | [-0.98, 0.90, 0.41, 1.28] |
| e'_bank = e + Delta e | [-1.40, -0.22, -1.14, -0.51] | [-1.68, 2.20, 0.31, 1.38] |
| e_the(10)^(0) (identical) | [1.30, 0.90, 0.10, -0.80] | [1.30, 0.90, 0.10, -0.80] |
| Delta e_the(10) | [-0.89, -1.48, -1.04, -0.45] | [-1.07, 0.61, 0.22, 1.14] |
| e'_the(10) | [0.41, -0.58, -0.94, -1.25] | [0.23, 1.51, 0.32, 0.34] |

## T7 / T9 - magnitudes

- 268 parameters, all one-decimal, max |x| = 1.9 (limit 3.0).
- token-embedding norms: min 1.42, mean 1.70, max 2.26; positional norms: min 0.55, max 0.57 (33% of the mean token norm).
- e^(0) of the three 'the' in S_A: pos 1 [1.50, 0.80, 0.20, 0.00], pos 5 [1.30, 0.10, 0.50, -0.40], pos 10 [1.30, 0.90, 0.10, -0.80] (all different).
- Positional table: pos_emb[p-1] = 0.4 * [sin(2 pi p/10), cos(2 pi p/10), sin(2 pi p/20), cos(2 pi p/20)] rounded to one decimal, p = 1..10.

## Score ranges (for heat-map colour scales)

- S_A: scaled causal scores s_ij in [-1.86, 2.31]; raw q.k in [-3.23, 4.00]; |Delta e| rows up to 3.15; logits of the(10) in [-0.92, 2.89].
- S_B: scaled causal scores s_ij in [-1.86, 2.17]; raw q.k in [-3.23, 3.76]; |Delta e| rows up to 3.56; logits of the(10) in [-0.86, 2.82].
