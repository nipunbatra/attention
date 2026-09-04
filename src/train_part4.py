#!/usr/bin/env python3
"""Build the small encoder-decoder translation example used in Part IV.

The numerical model has one head and three coordinates throughout. It omits
FFNs and LayerNorm deliberately. NumPy implements both the forward pass and
its analytic reverse pass; central differences independently check every
scalar gradient. No deep-learning package is needed.

Two pairs are fitted so the French noun must depend on the English source:
  the river bank <eos>      -> <bos> la rive <eos>
  the financial bank <eos>  -> <bos> la banque <eos>
This tiny overfit model is not a general translator. The visible learning step
then uses only the river pair. Every saved number is computed, not painted.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
D = 3
SRC_VOCAB = ["the", "river", "bank", "financial", "<eos>"]
TGT_VOCAB = ["<bos>", "la", "rive", "banque", "<eos>"]
SOURCE = ["the", "river", "bank", "<eos>"]
CONTRAST = ["the", "financial", "bank", "<eos>"]
INPUT = ["<bos>", "la", "rive"]
TARGETS = ["la", "rive", "<eos>"]
PAIRS = [
    (SOURCE, INPUT, TARGETS),
    (CONTRAST, ["<bos>", "la", "banque"], ["la", "banque", "<eos>"]),
]
SEED = 21
WARMUP_RATE = 0.06
UPDATE_RATE = 0.05
MAX_SOURCE = 4
MAX_TARGET = 6


def softmax(rows):
    rows = np.asarray(rows, dtype=np.float64)
    exp = np.exp(rows - rows.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def initialize():
    rng = np.random.default_rng(SEED)
    p = {
        "E_src": rng.normal(0, 0.45, (len(SRC_VOCAB), D)),
        "P_src": np.array([[0.10, -0.05, 0.02], [0.02, 0.10, -0.05],
                           [-0.05, 0.02, 0.10], [0.06, -0.04, 0.08]]),
        "E_tgt": rng.normal(0, 0.45, (len(TGT_VOCAB), D)),
        "P_tgt": np.array([[0.10, 0.02, -0.04], [-0.04, 0.10, 0.02],
                           [0.02, -0.04, 0.10], [0.06, 0.03, -0.02],
                           [-0.02, 0.06, 0.03], [0.03, -0.02, 0.06]]),
    }
    for block in ("enc", "dec", "cross"):
        for role in ("Q", "K", "V", "O"):
            p[f"{block}_W_{role}"] = rng.normal(0, 0.4, (D, D))
    p["W_vocab"] = rng.normal(0, 0.4, (D, len(TGT_VOCAB)))
    p["b_vocab"] = np.zeros(len(TGT_VOCAB))
    return p


def attention(p, query_input, memory, name, causal=False):
    q = query_input @ p[name + "_W_Q"]
    k = memory @ p[name + "_W_K"]
    v = memory @ p[name + "_W_V"]
    raw = q @ k.T
    scaled = raw / math.sqrt(D)
    blocked = np.triu(np.ones(scaled.shape, dtype=bool), 1) if causal else np.zeros(scaled.shape, dtype=bool)
    scores = np.where(blocked, -np.inf, scaled)
    a = softmax(scores)
    message = a @ v
    delta = message @ p[name + "_W_O"]
    return {
        "input": query_input, "memory": memory,
        "Q": q, "K": k, "V": v,
        "Sraw": raw, "Sfull": scaled, "S": scores, "A": a,
        "Mmsg": message, "Delta": delta, "Enew": query_input + delta,
        "causal": causal,
    }


def forward(p, prefix=INPUT, source=SOURCE):
    if not source or len(source) > MAX_SOURCE:
        raise ValueError("Source length is outside the available position table.")
    if not prefix or len(prefix) > MAX_TARGET:
        raise ValueError("Target prefix length is outside the available position table.")
    si = [SRC_VOCAB.index(t) for t in source]
    ti = [TGT_VOCAB.index(t) for t in prefix]
    source_lookup = p["E_src"][si]
    target_lookup = p["E_tgt"][ti]
    source_positions = p["P_src"][:len(source)]
    target_positions = p["P_tgt"][:len(prefix)]
    source_rows = source_lookup + source_positions
    target_rows = target_lookup + target_positions
    encoder = attention(p, source_rows, source_rows, "enc")
    decoder_self = attention(p, target_rows, target_rows, "dec", causal=True)
    cross = attention(p, decoder_self["Enew"], encoder["Enew"], "cross")
    logits = cross["Enew"] @ p["W_vocab"] + p["b_vocab"]
    probs = softmax(logits)
    return {
        "source": source, "prefix": prefix,
        "sourceIds": si, "targetIds": ti,
        "sourceLookup": source_lookup, "sourcePositions": source_positions,
        "sourceRows": source_rows, "targetLookup": target_lookup,
        "targetPositions": target_positions, "targetRows": target_rows,
        "encoder": encoder, "decoderSelf": decoder_self, "cross": cross,
        "logits": logits, "probs": probs,
    }


def attention_backward(p, cache, gradient, name):
    # The residual branch contributes gradient directly to the query input.
    dmessage = gradient @ p[name + "_W_O"].T
    da = dmessage @ cache["V"].T
    dv = cache["A"].T @ dmessage
    ds = cache["A"] * (da - (da * cache["A"]).sum(axis=1, keepdims=True))
    ds /= math.sqrt(D)
    dq = ds @ cache["K"]
    dk = ds.T @ cache["Q"]
    grads = {
        name + "_W_Q": cache["input"].T @ dq,
        name + "_W_K": cache["memory"].T @ dk,
        name + "_W_V": cache["memory"].T @ dv,
        name + "_W_O": cache["Mmsg"].T @ gradient,
    }
    dquery = gradient + dq @ p[name + "_W_Q"].T
    dmemory = dk @ p[name + "_W_K"].T + dv @ p[name + "_W_V"].T
    return dquery, dmemory, grads


def loss_and_grads(p, prefix=INPUT, source=SOURCE, targets=TARGETS):
    f = forward(p, prefix, source)
    ids = [TGT_VOCAB.index(t) for t in targets]
    if len(ids) != len(prefix):
        raise ValueError("Each decoder input needs one next-token target.")
    # Stable log-sum-exp cross-entropy, one term per decoder row.
    z = f["logits"]
    peak = z.max(axis=1)
    losses = peak + np.log(np.exp(z - peak[:, None]).sum(axis=1)) - z[np.arange(len(ids)), ids]
    gz = f["probs"].copy()
    gz[np.arange(len(ids)), ids] -= 1
    gz /= len(ids)
    grads = {"W_vocab": f["cross"]["Enew"].T @ gz, "b_vocab": gz.sum(axis=0)}
    gcross = gz @ p["W_vocab"].T
    gdec, genc, g = attention_backward(p, f["cross"], gcross, "cross")
    grads.update(g)
    gq, gm, g = attention_backward(p, f["decoderSelf"], gdec, "dec")
    grads.update(g)
    gtgt = gq + gm
    gq, gm, g = attention_backward(p, f["encoder"], genc, "enc")
    grads.update(g)
    gsrc = gq + gm
    for name in ("E_src", "P_src", "E_tgt", "P_tgt"):
        grads[name] = np.zeros_like(p[name])
    np.add.at(grads["E_src"], f["sourceIds"], gsrc)
    grads["P_src"][:len(source)] = gsrc
    np.add.at(grads["E_tgt"], f["targetIds"], gtgt)
    grads["P_tgt"][:len(prefix)] = gtgt
    return float(losses.mean()), losses, grads, f


def step(p, grads, rate):
    return {name: values - rate * grads[name] for name, values in p.items()}


def generate(p, source=SOURCE, max_tokens=6):
    prefix = ["<bos>"]
    trace = []
    reason = "limit"
    for index in range(min(max_tokens, MAX_TARGET)):
        f = forward(p, prefix, source)
        probabilities = f["probs"][-1]
        chosen_id = int(np.argmax(probabilities))
        chosen = TGT_VOCAB[chosen_id]
        trace.append({
            "step": index + 1, "prefix": prefix[:],
            "query": f["cross"]["Q"][-1], "weights": f["cross"]["A"][-1],
            "logits": f["logits"][-1], "probabilities": probabilities,
            "chosen": chosen, "chosenId": chosen_id,
        })
        prefix.append(chosen)
        if chosen == "<eos>":
            reason = "eos"
            break
    return {"tokens": prefix[1:], "prefix": prefix, "trace": trace, "stoppedBy": reason}


def finite_differences(p, grads, epsilon=1e-5):
    checks = []
    for name in p:
        for index in np.ndindex(p[name].shape):
            original = p[name][index]
            try:
                p[name][index] = original + epsilon
                plus = loss_and_grads(p)[0]
                p[name][index] = original - epsilon
                minus = loss_and_grads(p)[0]
            finally:
                p[name][index] = original
            numeric = (plus - minus) / (2 * epsilon)
            analytic = float(grads[name][index])
            checks.append({"parameter": name, "index": list(index),
                           "analytic": analytic, "numeric": numeric,
                           "absError": abs(numeric - analytic)})
    return {"epsilon": epsilon, "count": len(checks),
            "maxAbsError": max(c["absError"] for c in checks), "checks": checks}


def native(value):
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        # Only a forbidden causal score may become null in the saved data.
        if value == -math.inf:
            return None
        raise ValueError("A nonfinite model value cannot be saved as a masked score.")
    return value


def build():
    p = initialize()
    history = []
    initial = []
    for source, prefix, targets in PAIRS:
        initial.append(loss_and_grads(p, prefix, source, targets)[0])
    for iteration in range(4000):
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        pair_losses = []
        for source, prefix, targets in PAIRS:
            loss, _, g, _ = loss_and_grads(p, prefix, source, targets)
            pair_losses.append(loss)
            for name in p:
                grads[name] += g[name] / len(PAIRS)
        if iteration % 25 == 0:
            history.append({"step": iteration, "loss": float(np.mean(pair_losses))})
        if iteration >= 80:
            target_probs = []
            for source, prefix, targets in PAIRS:
                f = forward(p, prefix, source)
                target_probs.extend(f["probs"][i, TGT_VOCAB.index(t)] for i, t in enumerate(targets))
            if min(target_probs) >= 0.88:
                break
        p = step(p, grads, WARMUP_RATE)
    else:
        raise RuntimeError("The deterministic two-pair training did not reach its target.")
    before = copy.deepcopy(p)
    before_loss, before_losses, grads, before_f = loss_and_grads(before)
    after = step(before, grads, UPDATE_RATE)
    after_loss, after_losses, _, after_f = loss_and_grads(after)
    checks = finite_differences(copy.deepcopy(before), grads)
    if checks["maxAbsError"] > 1e-7:
        raise RuntimeError("Part IV gradient check failed.")
    if after_loss >= before_loss:
        raise RuntimeError("The visible SGD step did not lower the worked-example loss.")
    generated = {name: {"river": generate(params), "financial": generate(params, CONTRAST)}
                 for name, params in (("before", before), ("after", after))}
    for name in generated:
        if generated[name]["river"]["tokens"] != TARGETS:
            raise RuntimeError("River generation failed at snapshot " + name)
        if generated[name]["financial"]["tokens"] != ["la", "banque", "<eos>"]:
            raise RuntimeError("Financial generation failed at snapshot " + name)
    history.append({"step": iteration, "loss": float(np.mean([
        loss_and_grads(before, prefix, source, targets)[0] for source, prefix, targets in PAIRS]))})
    translation = {
        "sourceVocab": SRC_VOCAB, "targetVocab": TGT_VOCAB,
        "source": SOURCE, "sourceContrast": CONTRAST,
        "targetInput": INPUT, "targets": TARGETS,
        "contrastTargets": ["la", "banque", "<eos>"],
        "maxSource": MAX_SOURCE, "maxTarget": MAX_TARGET,
        "defaultSnapshot": "before", "snapshots": {"before": before, "after": after},
        "training": {"seed": SEED, "warmupSteps": iteration, "warmupLearningRate": WARMUP_RATE,
                     "pairs": [{"source": s, "input": i, "targets": t} for s, i, t in PAIRS],
                     "initialPairLosses": initial, "history": history,
                     "overfit": True,
                     "note": "Fitted only to two bank phrases. This checks source dependence, not translation quality or generalisation."},
        "update": {"learningRate": UPDATE_RATE, "source": SOURCE, "input": INPUT, "targets": TARGETS,
                   "lossBefore": before_loss, "lossAfter": after_loss,
                   "perPositionBefore": before_losses, "perPositionAfter": after_losses,
                   "gradients": grads, "parameterCount": sum(v.size for v in before.values()),
                   "gradientCheck": checks},
        "reference": {"before": before_f, "after": after_f, "generation": generated},
        "notes": [
            "One bidirectional encoder self-attention block, one causal decoder self-attention block, then cross-attention.",
            "Every block includes its output projection and residual addition. FFNs, LayerNorm and dropout are omitted from the numerical toy.",
            "Source and target embeddings are separate learned tables. Positions are added as three-coordinate vectors, not appended as another coordinate.",
            "Coordinates are generic learned axes. Attention weights are computed from the model and are not asserted to be word alignments.",
            "The visible SGD update uses only the river translation after the two-pair warm-up.",
            "Full precision is used in every calculation. Tables may round for display. A null saved score denotes a forbidden causal position."
        ],
    }
    return native({
        "d_model": D, "d_k": D, "d_v": D, "max_context": MAX_TARGET,
        "vocab": TGT_VOCAB,
        "sentences": {"river": SOURCE, "translation": INPUT},
        "axes": {"e": ["representation coordinate 1", "representation coordinate 2", "representation coordinate 3"],
                 "qk": ["matching coordinate 1", "matching coordinate 2", "matching coordinate 3"],
                 "v": ["message coordinate 1", "message coordinate 2", "message coordinate 3"],
                 "short": {"e": ["1", "2", "3"], "qk": ["1", "2", "3"], "v": ["1", "2", "3"]}},
        "translation": translation,
        "notes": "A three-coordinate, one-head encoder-decoder toy fitted to two translations. No FFN or LayerNorm. Use AT.translation, not the Part II AT.forward helper."
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="reproduce and compare the saved JSON without writing")
    args = parser.parse_args()
    result = build()
    destination = HERE / "toy4.json"
    if args.check:
        saved = json.loads(destination.read_text(encoding="utf-8"))
        if saved != result:
            raise RuntimeError("Saved Part IV toy does not reproduce exactly.")
        print("PASS: Part IV training, every saved tensor, and gradient checks reproduce exactly.")
    else:
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("wrote", destination)
    t = result["translation"]
    print("warm-up steps:", t["training"]["warmupSteps"])
    print("worked-example loss:", t["update"]["lossBefore"], "->", t["update"]["lossAfter"])
    print("gradient scalars:", t["update"]["gradientCheck"]["count"], "max error:", t["update"]["gradientCheck"]["maxAbsError"])
    for name, generation in t["reference"]["generation"].items():
        print(name, {k: v["tokens"] for k, v in generation.items()})


if __name__ == "__main__":
    main()
