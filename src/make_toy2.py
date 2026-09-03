#!/usr/bin/env python3
"""
make_toy2.py - toy model v2 for attention.html, designed BY HAND from AXES.md.

Every coordinate has a name and every matrix entry is a readable, one-decimal number chosen so that
the names are true (see AXES.md). Nothing is optimised: the tables below ARE the design; this script
only evaluates the forward pass (the same arithmetic as toy_ref.mjs), checks the AXES.md targets on
the rounded numbers, and writes toy.json + py_check.json.

Usage:  python3 make_toy2.py              # build from the tables, check, write toy.json + py_check.json
        python3 make_toy2.py --check-only # re-check the toy.json on disk (no write except py_check.json)
        python3 make_toy2.py --dry        # build and check, write nothing
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "toy.json")
OUT_CHECK = os.path.join(HERE, "py_check.json")

VOCAB = ("the fisherman sat beside river bank and watched she deposited cheque at "
         "water boats fish ducks teller clerk queue money").split()
VI = {w: i for i, w in enumerate(VOCAB)}
SA = ["The", "fisherman", "sat", "beside", "the", "river", "bank", "and", "watched", "the"]
SB = ["She", "deposited", "the", "cheque", "at", "the", "bank", "and", "watched", "the"]
CAND_A = ["water", "boats", "fish", "ducks"]
CAND_B = ["teller", "clerk", "queue", "money"]
CANDS = CAND_A + CAND_B
D_MODEL, D_K, D_V, T = 5, 3, 2, 10
MAX_CONTEXT = 20  # ten-token examples plus room for the generation walkthrough
BANK, LAST = 6, 9  # 0-indexed positions of bank(7) and the(10)

# ----------------------------------------------------------------------------------------------
# 0. Named axes (AXES.md)
# ----------------------------------------------------------------------------------------------
AXES = {
    "e": ["water", "finance", "person", "glue", "position"],
    "qk": ["setting: water?", "setting: finance?", "who?"],
    "v": ["says: water scene", "says: finance scene"],
    "short": {"e": ["water", "finance", "person", "glue", "pos"],
              "qk": ["water?", "finance?", "who?"],
              "v": ["→water", "→finance"]},
}

# ----------------------------------------------------------------------------------------------
# 1. Token embeddings on the e axes  [water, finance, person, glue, position]
# ----------------------------------------------------------------------------------------------
TOK = {
    # water scene
    "river":     [3.0, 0.0, 0.0, 0.0, 0.0],
    "water":     [3.0, 0.0, 0.0, 0.0, 0.0],
    "boats":     [2.4, 0.0, 0.4, 0.0, 0.0],
    "fish":      [2.6, 0.0, 0.6, 0.0, 0.0],
    "ducks":     [2.2, 0.0, 0.8, 0.0, 0.0],
    # finance scene
    "cheque":    [0.0, 3.0, 0.0, 0.0, 0.0],
    "money":     [0.0, 3.0, 0.0, 0.0, 0.0],
    "teller":    [0.0, 2.6, 1.8, 0.0, 0.0],
    "clerk":     [0.0, 2.4, 1.8, 0.0, 0.0],
    "queue":     [0.0, 2.0, 0.6, 0.0, 0.0],
    "deposited": [0.0, 2.2, 0.6, 0.4, 0.0],
    # people
    "fisherman": [2.0, 0.0, 2.2, 0.0, 0.0],   # AXES.md starts at water 1.4; raised so bank(7) ranks him second (see notes)
    "she":       [0.0, 0.0, 2.6, 0.4, 0.0],
    # the ambiguous word: equal parts water and finance, plus a little glue (it has to ask for its setting)
    "bank":      [0.7, 0.7, 0.0, 0.7, 0.0],   # the glue entry keeps river above the T1 threshold after position moves to its own unused axis
    # glue words
    "the":       [0.0, 0.0, 0.0, 2.4, 0.0],
    "and":       [0.0, 0.0, 0.0, 2.2, 0.0],
    "at":        [0.0, 0.0, 0.0, 2.0, 0.0],
    "beside":    [0.6, 0.0, 0.0, 2.0, 0.0],
    "sat":       [0.4, 0.0, 0.6, 1.8, 0.0],
    "watched":   [0.6, 0.0, 0.8, 1.6, 0.0],
}

# Position has one dedicated coordinate. The projections below carry a zero row for it, so this toy
# preserves position in e^(0) without using it for attention or the vocabulary head.
POS = [[0.0, 0.0, 0.0, 0.0, 0.1 * (i + 1)] for i in range(MAX_CONTEXT)]

# ----------------------------------------------------------------------------------------------
# 2. Projections (rows = input axis, columns = output axis)
# ----------------------------------------------------------------------------------------------
#            q axes:  water?  finance?  who?
W_Q = [
    [1.0, 0.0, 0.0],   # water   -> asks water?
    [0.0, 1.0, 0.0],   # finance -> asks finance?
    [0.0, 0.0, 0.4],   # person  -> asks who? (weakly)
    [0.7, 0.7, 0.0],   # glue    -> asks "what setting am I in?" (both)
    [0.0, 0.0, 0.0],   # position -> unused by this toy
]
#            k axes:  water?  finance?  who?
W_K = [
    [1.0, 0.0, 0.0],   # water   -> offers water
    [0.0, 1.0, 0.0],   # finance -> offers finance
    [0.0, 0.0, 1.0],   # person  -> offers a person
    [0.0, 0.0, 0.0],   # glue    -> offers nothing
    [0.0, 0.0, 0.0],   # position -> unused by this toy
]
#            v axes:  ->water  ->finance
W_V = [
    [1.0, 0.0],   # water    -> says water scene
    [0.0, 1.0],   # finance  -> says finance scene
    [0.0, 0.0],   # person   -> says nothing in this toy
    [0.0, 0.0],   # glue     -> says nothing
    [0.0, 0.0],   # position -> unused by this toy
]
#            e axes:  water  finance  person  glue  position
W_O = [
    [1.0, 0.0, 0.0, 0.0, 0.0],   # says water   -> water
    [0.0, 1.0, 0.0, 0.0, 0.0],   # says finance -> finance
]
# Output head: which e axis votes for which word (columns = vocabulary). Everything not listed is 0.
W_VOCAB_ROWS = {
    "water":   {"water": 1.5, "boats": 1.2, "fish": 1.0, "ducks": 0.7},     # AXES.md starts at 1.2/0.9/0.8/0.5
    "finance": {"teller": 1.5, "clerk": 1.2, "queue": 0.9, "money": 0.7},   # AXES.md starts at 1.2/0.9/0.6/0.5
    "person":  {"teller": 0.2, "clerk": 0.2, "fisherman": 0.2},
    "glue":    {},
    "position": {},
}
B_OTHER = -1.5   # bias for every non-candidate word; candidates get 0


def build_params():
    W_vocab = np.zeros((D_MODEL, len(VOCAB)))
    for r, ax in enumerate(AXES["e"]):
        for w, x in W_VOCAB_ROWS[ax].items():
            W_vocab[r, VI[w]] = x
    b = np.full(len(VOCAB), B_OTHER)
    for c in CANDS:
        b[VI[c]] = 0.0
    return dict(tok=np.array([TOK[w] for w in VOCAB], float), pos=np.array(POS, float),
                W_Q=np.array(W_Q, float), W_K=np.array(W_K, float), W_V=np.array(W_V, float),
                W_O=np.array(W_O, float), W_vocab=W_vocab, b_vocab=b)


# ----------------------------------------------------------------------------------------------
# 3. Forward pass (mirrors toy_ref.mjs: row vectors, q = e W_Q, mask j > i, scale 1/sqrt(d_k))
# ----------------------------------------------------------------------------------------------
def softmax_rows(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def run(P, tokens, mask=True, scale=True):
    n = len(tokens)
    if n > len(P["pos"]):
        raise ValueError(f"Context length {n} exceeds positional capacity {len(P['pos'])}.")
    idx = [VI[t.lower()] for t in tokens]
    E = P["tok"][idx] + P["pos"][:n]
    Q, K, V = E @ P["W_Q"], E @ P["W_K"], E @ P["W_V"]
    Sraw = Q @ K.T
    S = Sraw / np.sqrt(D_K) if scale else Sraw.copy()
    if mask:
        S = S + np.triu(np.full((n, n), -1e9), k=1)
    A = softmax_rows(S)
    M = A @ V
    Delta = M @ P["W_O"]
    Enew = E + Delta
    logits = Enew @ P["W_vocab"] + P["b_vocab"]
    base_logits = E @ P["W_vocab"] + P["b_vocab"]
    return dict(E=E, Q=Q, K=K, V=V, Sraw=Sraw, S=S, A=A, Mmsg=M, Delta=Delta, Enew=Enew,
                logits=logits, probs=softmax_rows(logits),
                base_logits=base_logits, base_probs=softmax_rows(base_logits))


# ----------------------------------------------------------------------------------------------
# 4. Targets (AXES.md T1-T6, plus the structural checks T7-T10 of the first toy)
# ----------------------------------------------------------------------------------------------
def check(P, verbose=True):
    fa, fb, fu = run(P, SA), run(P, SB), run(P, SA, mask=False)
    res = []

    def rec(name, ok, detail, hard=True):
        res.append((name, bool(ok), detail, hard))
    rowstr = lambda toks, row, n: " ".join(f"{toks[j]}={row[j]:.2f}" for j in range(n))

    rA = fa["A"][BANK, :7]
    rec("T1 hard: bank(7) in S_A: river >= .40, fisherman second, bank(self) and glue words below fisherman",
        rA[5] >= .40 and rA[1] > max(rA[0], rA[2], rA[3], rA[4], rA[6]),
        "bank(7) row S_A: " + rowstr(SA, rA, 7))
    rB = fb["A"][BANK, :7]
    rec("T2 hard: bank(7) in S_B: cheque highest, deposited second, others low",
        rB[3] > rB[1] > max(rB[0], rB[2], rB[4], rB[5], rB[6]),
        "bank(7) row S_B: " + rowstr(SB, rB, 7))

    pA, aA = fa["probs"][LAST], fa["A"][LAST]
    othA = [pA[i] for i in range(20) if VOCAB[i] not in CAND_A]
    cA = [pA[VI[c]] for c in CAND_A]
    rec("T3 hard: the(10) in S_A: water > boats > fish > ducks, every other token <= .04",
        cA[0] > cA[1] > cA[2] > cA[3] and max(othA) <= .04,
        " ".join(f"{c}={pA[VI[c]]:.3f}" for c in CAND_A) + f" | max other={max(othA):.3f}")
    rec("T3 hard: attention of the(10) mostly on river, bank, fisherman (sum >= .60, each of them top-3)",
        aA[5] + aA[6] + aA[1] >= .60 and set(np.argsort(-aA)[:3]) == {5, 6, 1},
        "the(10) row S_A: " + rowstr(SA, aA, 10))
    pB, aB = fb["probs"][LAST], fb["A"][LAST]
    othB = [pB[i] for i in range(20) if VOCAB[i] not in CAND_B]
    cB = [pB[VI[c]] for c in CAND_B]
    rec("T4 hard: the(10) in S_B: teller > clerk > queue > money, every other token <= .04",
        cB[0] > cB[1] > cB[2] > cB[3] and max(othB) <= .04,
        " ".join(f"{c}={pB[VI[c]]:.3f}" for c in CAND_B) + f" | max other={max(othB):.3f}")
    rec("T4 hard: attention of the(10) mostly on cheque, bank, deposited (sum >= .60, each of them top-3)",
        aB[3] + aB[6] + aB[1] >= .60 and set(np.argsort(-aB)[:3]) == {3, 6, 1},
        "the(10) row S_B: " + rowstr(SB, aB, 10))

    pb, pb2 = fa["base_probs"][LAST], fb["base_probs"][LAST]
    cp = [pb[VI[c]] for c in CANDS]
    oth = [pb[i] for i in range(20) if VOCAB[i] not in CANDS]
    rec("T5 hard: baseline (no attention) candidates each in [.06,.18], others <= .04, identical for S_A/S_B",
        min(cp) >= .06 and max(cp) <= .18 and max(oth) <= .04 and np.allclose(pb, pb2, atol=1e-12),
        " ".join(f"{c}={pb[VI[c]]:.3f}" for c in CANDS) + f" | max other={max(oth):.3f}")
    leaks = {f"{SA[i]}({i+1})": fu["A"][i, 5] + fu["A"][i, 6] for i in range(5)}
    rec("T6 soft: mask off, some early token of S_A puts >= .25 on future river/bank",
        max(leaks.values()) >= .25, " ".join(f"{k}:{v:.2f}" for k, v in leaks.items()), hard=False)

    allv = np.concatenate([P[k].ravel() for k in P])
    rec("T7 hard: all |x| <= 3.0 and one decimal",
        np.all(np.abs(allv) <= 3.0) and np.allclose(allv, np.round(allv, 1), atol=1e-12), f"max|x|={np.abs(allv).max():.1f}")
    rec("T8 hard: vocabulary is the 20 tokens", P["tok"].shape == (len(VOCAB), D_MODEL), "20 tokens")
    pos_axis = AXES["e"].index("position")
    e_the = [fa["E"][i] for i in (0, 4, 9)]
    want_pos = np.array([[0.1 * (i + 1) if d == pos_axis else 0.0 for d in range(D_MODEL)] for i in range(len(P["pos"]))])
    rec("T9 hard: position has its own coordinate and the three 'the' rows differ",
        np.allclose(P["pos"], want_pos, atol=1e-12) and np.allclose(P["tok"][:, pos_axis], 0) and
        np.allclose(P["W_Q"][pos_axis], 0) and np.allclose(P["W_K"][pos_axis], 0) and
        np.allclose(P["W_V"][pos_axis], 0) and np.allclose(P["W_vocab"][pos_axis], 0) and
        len({tuple(np.round(e, 6)) for e in e_the}) == 3,
        f"position rows 0.1 to {0.1 * len(P['pos']):.1f}; token and projection position rows are zero")
    rec("T10 hard: parameter shapes match d_model, d_k and d_v",
        P["pos"].shape == (MAX_CONTEXT, D_MODEL) and
        P["W_Q"].shape == (D_MODEL, D_K) and P["W_K"].shape == (D_MODEL, D_K) and
        P["W_V"].shape == (D_MODEL, D_V) and P["W_O"].shape == (D_V, D_MODEL) and
        P["W_vocab"].shape == (D_MODEL, len(VOCAB)), "ok")
    if verbose:
        print("=" * 100)
        for name, ok, detail, hard in res:
            print(f"[{'PASS' if ok else 'FAIL'}]{'' if hard else ' (soft)'} {name}\n        {detail}")
        print("=" * 100)
    return res, dict(fa=fa, fb=fb, fu=fu)


def hard_ok(res):
    return all(ok for _, ok, _, hard in res if hard)


# ----------------------------------------------------------------------------------------------
# 5. Files
# ----------------------------------------------------------------------------------------------
def fl(a):
    return [fl(x) for x in a] if np.ndim(a) > 0 else float(round(float(a), 1)) + 0.0


NOTES = (
    "Toy v2: single-head causal self-attention (d_model=5, d_k=3, d_v=2, vocab=20, ten-token examples, max_context=20), designed BY HAND so that every "
    "coordinate has a name that the numbers agree with (see 'axes'). e axes: water, finance, person, glue. q and k axes: "
    "setting: water?, setting: finance?, who? (a query row reads 'what I ask for', a key row 'what I offer'). The fifth e "
    "axis carries position, but its zero rows in W_Q, W_K, W_V and W_vocab mean this toy does not use it. v has its own "
    "narrower axes: says: water scene and says: finance scene; W_O maps them back onto the e axes. Names are "
    "illustrative; the matrices are sparse and one-decimal so a student can read each row of W_Q as 'axis -> asks', "
    "W_K as 'axis -> offers', W_V as 'axis -> says', W_vocab as 'axis -> votes for these words'. Values are narrower on "
    "purpose because they are mixed and sent rather than compared. Glue words have zero key and value rows but ask for "
    "both settings, which is why the final 'the' reads river or cheque. bank is equal parts water "
    "and finance (0.7, 0.7) plus a little glue (0.7); its entries are kept below 1.0 so that bank does not mostly attend to itself "
    "(the self score q_bank.k_bank grows with the square of those entries). Position rises from 0.1 to 2.0 in its own "
    "coordinate, with twenty same-width position vectors available for generation. The first ten positions are used in the worked examples. "
    "This toy's output ignores position; it illustrates content routing, not word-order sensitivity. Patterns produced (all "
    "checked on these rounded numbers by make_toy2.py and toy_ref.mjs): (1) in 'The fisherman sat beside the river bank', "
    "bank(7) attends river first, fisherman second, itself and the glue words little; (2) in 'She deposited the cheque at "
    "the bank', bank(7) attends cheque first, deposited second; (3) the final 'the'(10) reads mostly river/bank/fisherman "
    "or cheque/bank/deposited and the output head predicts water > boats > fish > ducks or teller > clerk > queue > money "
    "with every other word at most 0.04; (4) the baseline from e_the(10) alone is spread over the eight candidates and is "
    "identical for both sentences; (5) with the causal mask off, the early glue words leak attention onto river and bank. "
    "Nothing was optimised; a few magnitudes (bank, fisherman, the W_vocab weights) were adjusted by hand from AXES.md."
)


def write_json(P):
    toy = {
        "d_model": D_MODEL, "d_k": D_K, "d_v": D_V, "max_context": MAX_CONTEXT,
        "vocab": VOCAB,
        "tok_emb": {w: fl(P["tok"][i]) for i, w in enumerate(VOCAB)},
        "pos_emb": fl(P["pos"]),
        "W_Q": fl(P["W_Q"]), "W_K": fl(P["W_K"]), "W_V": fl(P["W_V"]),
        "W_O": fl(P["W_O"]),
        "W_vocab": fl(P["W_vocab"]), "b_vocab": fl(P["b_vocab"]),
        "sentences": {"river": SA, "cheque": SB},
        "candidates": {"river": CAND_A, "cheque": CAND_B},
        "axes": AXES,
        "notes": NOTES,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(toy, f, indent=1)
    print(f"wrote {OUT_JSON}")


def write_check(P):
    out = {}
    for name, toks, mask, scale in (("river_masked", SA, True, True), ("cheque_masked", SB, True, True),
                                    ("river_unmasked", SA, False, True), ("cheque_unmasked", SB, False, True),
                                    ("river_noscale", SA, True, False)):
        f = run(P, toks, mask, scale)
        out[name] = {k: f[k].tolist() for k in ("E", "Q", "K", "V", "Sraw", "S", "A", "Mmsg", "Delta", "Enew", "logits", "probs")}
        if mask:  # JS uses -Infinity for masked scores, python -1e9: compare only the visible entries
            out[name]["S"] = [[(None if j > i else f["S"][i, j]) for j in range(T)] for i in range(T)]
        out[name]["base_logits"] = f["base_logits"].tolist()
        out[name]["base_probs"] = f["base_probs"].tolist()
    with open(OUT_CHECK, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {OUT_CHECK}")


def load_json_params(path=OUT_JSON):
    with open(path) as f:
        toy = json.load(f)
    return dict(tok=np.array([toy["tok_emb"][w] for w in VOCAB]), pos=np.array(toy["pos_emb"]),
                W_Q=np.array(toy["W_Q"]), W_K=np.array(toy["W_K"]), W_V=np.array(toy["W_V"]), W_O=np.array(toy["W_O"]),
                W_vocab=np.array(toy["W_vocab"]), b_vocab=np.array(toy["b_vocab"]))


def main():
    args = sys.argv[1:]
    if "--check-only" in args:
        P = load_json_params()
        res, _ = check(P)
        write_check(P)
        sys.exit(0 if hard_ok(res) else 1)
    P = build_params()
    res, _ = check(P)
    ok = hard_ok(res)
    print(f"HARD TARGETS: {'ALL PASS' if ok else 'FAILURES REMAIN'}")
    if ok and "--dry" not in args:
        write_json(P)
        write_check(P)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
