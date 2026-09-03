#!/usr/bin/env python3
"""gen_report.py - writes toy_report.md from toy.json using make_toy2.py's numpy forward pass (toy v2, named axes)."""
import json, os
import numpy as np
from make_toy2 import load_json_params, run, check, VOCAB, VI, SA, SB, CAND_A, CAND_B, CANDS, HERE, OUT_JSON

toy = json.load(open(OUT_JSON))
AX = toy["axes"]
P = load_json_params()
fa, fb, fu = run(P, SA), run(P, SB), run(P, SA, mask=False)
res, _ = check(P, verbose=False)
base = fa["base_probs"][9]
nparams = sum(int(np.size(P[k])) for k in P)
maxabs = max(float(np.abs(P[k]).max()) for k in P)
L = []
w = L.append
fmt = lambda v: "[" + ", ".join(f"{x:.2f}" for x in v) + "]"
num = lambda x: ("0" if abs(x) < 1e-12 else f"{x:.1f}")

w("# toy_report.md - numbers produced by the hand-designed toy.json (v2, named axes)\n")
w(f"All values below are computed from the ONE-DECIMAL parameters in `toy.json` ({nparams} numbers, max |x| = {maxabs:.1f}) by "
  "`make_toy2.py`; `node toy_ref.mjs --compare py_check.json` reproduces every intermediate to < 1e-12. "
  "Attention weights are shown to 2 decimals, probabilities to 3. The previous, optimised toy is kept as `toy_v1.json` "
  "(its report: `toy_report_v1.md`).\n")
w("## How the numbers were obtained\n")
w("Nothing was optimised. AXES.md names every coordinate, and the embeddings and projection matrices were written by hand "
  "so that the names are true: each row of a projection reads as 'input axis -> asks / offers / says'. The forward pass "
  "(the same arithmetic as `toy_ref.mjs`) was then evaluated and a few magnitudes were adjusted from the AXES.md starting "
  "point until the targets held on the rounded numbers: `bank` went from [1.5, 1.5, 0, 0.4] to [0.7, 0.7, 0, 0.6] and "
  "`fisherman` from [1.4, 0, 2.4, 0] to [2.0, 0, 2.2, 0], because the self score q_bank . k_bank grows with the square of "
  "bank's water/finance entries and at 1.5 bank attended to itself as much as to river; the W_vocab weights were raised to "
  "1.5/1.2/1.0/0.7 (water words) and 1.5/1.2/0.9/0.7 (finance words) so that every non-candidate word stays at or below 0.04. "
  "Position offsets are a circle of radius 0.3 in the (person, glue) plane, one decimal, nothing on the water/finance axes.\n")

w("## Axes (from `toy.json` -> `axes`)\n")
w("| object | coordinate 1 | coordinate 2 | coordinate 3 | coordinate 4 |")
w("|---|---|---|---|---|")
w("| e, Delta e, e' (d_model = 4) | " + " | ".join(AX["e"]) + " |")
w("| q, k (d_k = 3) | " + " | ".join(AX["qk"]) + " | |")
w("| v, m (d_v = 3) | " + " | ".join(AX["v"]) + " | |")
w("| short forms | e: " + ", ".join(AX["short"]["e"]) + " | q,k: " + ", ".join(AX["short"]["qk"]) + " | v: " + ", ".join(AX["short"]["v"]) + " | |")
w("")
w("Reading rule: a query row is 'what I ask for', a key row is 'what I offer', a value row is 'what I send if retrieved'. "
  "W_O maps the v axes back onto the e axes (water -> water, finance -> finance, person -> person, nothing -> glue).\n")

w("## Token embeddings (e axes)\n")
w("| token | " + " | ".join(AX["short"]["e"]) + " | reading |")
w("|---|" + "---:|" * 4 + "---|")
def reading(v):
    names = AX["short"]["e"]
    parts = [f"{n} {v[i]:.1f}" for i, n in enumerate(names) if abs(v[i]) > 1e-9]
    return ", ".join(parts) if parts else "all zero"
for word in VOCAB:
    v = toy["tok_emb"][word]
    w(f"| {word} | " + " | ".join(num(x) for x in v) + f" | {reading(v)} |")
w("")
w("## Position offsets (added to the token row; e^(0)_i = tok_emb[token_i] + pos_emb[i])\n")
w("| position | " + " | ".join(AX["short"]["e"]) + " |")
w("|---:|" + "---:|" * 4)
for i, row in enumerate(toy["pos_emb"]):
    w(f"| {i+1} | " + " | ".join(num(x) for x in row) + " |")
w("")
w("Radius 0.3 in the (person, glue) plane, so no position looks like a setting; all ten rows differ.\n")

def wtable(title, W, rows, cols, rowname, colname):
    w(f"### {title}\n")
    w(f"| {rowname} \\ {colname} | " + " | ".join(cols) + " |")
    w("|---|" + "---:|" * len(cols))
    for r, name in zip(W, rows):
        w(f"| {name} | " + " | ".join(num(x) for x in r) + " |")
    w("")

w("## Projection matrices (rows = input axis, columns = output axis; zeros left as 0)\n")
wtable("W_Q (e -> q): 'axis -> asks'", toy["W_Q"], AX["short"]["e"], AX["short"]["qk"], "e axis", "q axis")
w("Reading: water asks water?, finance asks finance?, person asks who? (0.4), glue asks both settings (0.7, 0.7). "
  "Glue words ask 'what setting am I in?', which is why the final 'the' reads river or cheque.\n")
wtable("W_K (e -> k): 'axis -> offers'", toy["W_K"], AX["short"]["e"], AX["short"]["qk"], "e axis", "k axis")
w("Reading: water offers water, finance offers finance, person offers a person, glue offers nothing.\n")
wtable("W_V (e -> v): 'axis -> says'", toy["W_V"], AX["short"]["e"], AX["short"]["v"], "e axis", "v axis")
w("Reading: water says water scene, finance says finance scene, person says a person is here, glue says nothing.\n")
wtable("W_O (v -> e): back onto the e axes", toy["W_O"], AX["short"]["v"], AX["short"]["e"], "v axis", "e axis")
w("Reading: says water -> water 1.0, says finance -> finance 1.0, says person -> person 0.5, nothing lands on glue.\n")
w("### W_vocab (e -> logits): 'axis -> votes for these words' (only the non-zero columns; every other entry is 0)\n")
Wv = toy["W_vocab"]
nz_cols = [j for j in range(20) if any(abs(Wv[r][j]) > 1e-9 for r in range(4))]
w("| e axis \\ word | " + " | ".join(VOCAB[j] for j in nz_cols) + " |")
w("|---|" + "---:|" * len(nz_cols))
for r in range(4):
    w(f"| {AX['short']['e'][r]} | " + " | ".join(num(Wv[r][j]) for j in nz_cols) + " |")
w("")
bset = sorted(set(toy["b_vocab"]))
w("b_vocab: " + "; ".join(f"{num(b)} for " + ", ".join(VOCAB[j] for j in range(20) if toy['b_vocab'][j] == b) for b in bset) + ".\n")

def att_table(title, toks, row, n):
    w(f"### {title}\n")
    w("| position | token | weight |")
    w("|---:|---|---:|")
    for j in range(n):
        w(f"| {j+1} | {toks[j]} | {row[j]:.2f} |")
    w("")

def prob_table(title, p, cands):
    w(f"### {title}\n")
    w("| token | probability |"); w("|---|---:|")
    for c in cands: w(f"| {c} | {p[VI[c]]:.3f} |")
    oth = [(VOCAB[i], p[i]) for i in range(20) if VOCAB[i] not in cands]
    w(f"| every other token (max: {max(oth, key=lambda x: x[1])[0]}) | {max(o[1] for o in oth):.3f} |")
    w(f"| (sum of the other 16) | {sum(o[1] for o in oth):.3f} |")
    w("")

def status(prefix):
    hits = [ok for name, ok, _, _ in res if name.startswith(prefix)]
    return "**PASS**" if all(hits) else "**FAIL**"

w("## T1 - S_A `The fisherman sat beside the river bank`, query bank(7), causal\n")
att_table("attention row of bank(7)", SA, fa["A"][6], 7)
w(f"Target: river >= .40, fisherman second, bank(self) and the glue words below fisherman: {status('T1')}\n")
w("## T2 - S_B `She deposited the cheque at the bank`, query bank(7), causal\n")
att_table("attention row of bank(7)", SB, fb["A"][6], 7)
w(f"Target: cheque highest, deposited second, others low: {status('T2')}\n")
w("## T3 - S_A, query the(10): attention and next-token probabilities\n")
att_table("attention row of the(10)", SA, fa["A"][9], 10)
prob_table("p(next token | S_A) from e'_the(10)", fa["probs"][9], CAND_A)
w(f"Target: water > boats > fish > ducks, every other token <= .04; attention mostly on river, bank, fisherman: {status('T3')}\n")
w("## T4 - S_B, query the(10): attention and next-token probabilities\n")
att_table("attention row of the(10)", SB, fb["A"][9], 10)
prob_table("p(next token | S_B) from e'_the(10)", fb["probs"][9], CAND_B)
w(f"Target: teller > clerk > queue > money, every other token <= .04; attention mostly on cheque, bank, deposited: {status('T4')}\n")
w("## T5 - baseline (no attention): softmax(e^(0)_the(10) W_vocab + b)\n")
w("Identical for both sentences (same token, same position 10). e^(0)_the(10) has nothing on the water or finance axes, so the "
  "only differences come from the position offset on the person axis (which votes a little for teller and clerk).\n")
w("| token | probability |"); w("|---|---:|")
for c in CANDS: w(f"| {c} | {base[VI[c]]:.3f} |")
oth = [base[i] for i in range(20) if VOCAB[i] not in CANDS]
w(f"| every other token (max) | {max(oth):.3f} |"); w("")
w(f"Target: each candidate in [.06, .18], every other token <= .04: {status('T5')}\n")
w("## T6 - causal mask OFF on S_A (leakage onto future positions)\n")
w("| query | " + " | ".join(f"{t}({j+1})" for j, t in enumerate(SA)) + " | river+bank |")
w("|---|" + "---:|" * 10 + "---:|")
for i in range(5):
    w(f"| {SA[i]}({i+1}) | " + " | ".join(f"{fu['A'][i][j]:.2f}" for j in range(10)) + f" | **{fu['A'][i][5] + fu['A'][i][6]:.2f}** |")
w("")
best = max(range(5), key=lambda i: fu['A'][i][5] + fu['A'][i][6])
w(f"Soft target (>= .25 on river+bank from some early token): {status('T6')}, strongest leak from {SA[best]}({best+1}) "
  f"({fu['A'][best][5] + fu['A'][best][6]:.2f}). Every glue word asks for its setting, so with the mask off each of them reads "
  "river before river has been written; the page can highlight any of The(1), beside(4) or the(5).\n")
w("## The 'bank' story for section 10 (2 decimals, e axes: " + ", ".join(AX["short"]["e"]) + ")\n")
w("| quantity | S_A (river) | S_B (cheque) |"); w("|---|---|---|")
w(f"| e_bank^(0) (identical) | {fmt(fa['E'][6])} | {fmt(fb['E'][6])} |")
w(f"| q_bank (q axes) | {fmt(fa['Q'][6])} | {fmt(fb['Q'][6])} |")
w(f"| m_bank (v axes) | {fmt(fa['Mmsg'][6])} | {fmt(fb['Mmsg'][6])} |")
w(f"| Delta e_bank | {fmt(fa['Delta'][6])} | {fmt(fb['Delta'][6])} |")
w(f"| e'_bank = e + Delta e | {fmt(fa['Enew'][6])} | {fmt(fb['Enew'][6])} |")
w(f"| e_the(10)^(0) (identical) | {fmt(fa['E'][9])} | {fmt(fb['E'][9])} |")
w(f"| q_the(10) (q axes) | {fmt(fa['Q'][9])} | {fmt(fb['Q'][9])} |")
w(f"| m_the(10) (v axes) | {fmt(fa['Mmsg'][9])} | {fmt(fb['Mmsg'][9])} |")
w(f"| Delta e_the(10) | {fmt(fa['Delta'][9])} | {fmt(fb['Delta'][9])} |")
w(f"| e'_the(10) | {fmt(fa['Enew'][9])} | {fmt(fb['Enew'][9])} |")
w("")
w("Reading: in S_A the update to bank lands on the water axis, in S_B on the finance axis; the same e_bank^(0) ends up as two "
  "different e'_bank. The final 'the' receives the same kind of update, and the output head turns it into water words or finance words.\n")
w("## Keys and values of every token in both sentences (what each record offers and says)\n")
for name, toks, f in (("S_A", SA, fa), ("S_B", SB, fb)):
    w(f"### {name}\n")
    w("| pos | token | e (" + ", ".join(AX["short"]["e"]) + ") | q (" + ", ".join(AX["short"]["qk"]) + ") | k (same axes) | v (" + ", ".join(AX["short"]["v"]) + ") |")
    w("|---:|---|---|---|---|---|")
    for i, t in enumerate(toks):
        w(f"| {i+1} | {t} | {fmt(f['E'][i])} | {fmt(f['Q'][i])} | {fmt(f['K'][i])} | {fmt(f['V'][i])} |")
    w("")
tn = np.linalg.norm(P["tok"], axis=1); pn = np.linalg.norm(P["pos"], axis=1)
w("## T7 / T9 - magnitudes\n")
w(f"- {nparams} parameters, all one-decimal, max |x| = {maxabs:.1f} (limit 3.0).")
w(f"- token-embedding norms: min {tn.min():.2f}, mean {tn.mean():.2f}, max {tn.max():.2f}; positional norms: min {pn.min():.2f}, max {pn.max():.2f} "
  f"({pn.max()/tn.mean()*100:.0f}% of the mean token norm).")
w(f"- e^(0) of the three 'the' in S_A: pos 1 {fmt(fa['E'][0])}, pos 5 {fmt(fa['E'][4])}, pos 10 {fmt(fa['E'][9])} (all different).")
w("")
w("## Score ranges (for heat-map colour scales)\n")
for name, f in (("S_A", fa), ("S_B", fb)):
    S = f["Sraw"] / np.sqrt(3); tri = S[np.tril_indices(10)]
    w(f"- {name}: scaled causal scores s_ij in [{tri.min():.2f}, {tri.max():.2f}]; raw q.k in [{(f['Sraw'][np.tril_indices(10)]).min():.2f}, {(f['Sraw'][np.tril_indices(10)]).max():.2f}]; "
      f"|Delta e| rows up to {np.linalg.norm(f['Delta'], axis=1).max():.2f}; logits of the(10) in [{f['logits'][9].min():.2f}, {f['logits'][9].max():.2f}].")
w("")
w("## Check summary (make_toy2.py --check-only)\n")
for name, ok, detail, hard in res:
    w(f"- [{'PASS' if ok else 'FAIL'}]{'' if hard else ' (soft)'} {name}: {detail}")
w("")
open(os.path.join(HERE, "toy_report.md"), "w").write("\n".join(L) + "\n")
print("wrote toy_report.md")
