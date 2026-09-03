#!/usr/bin/env python3
"""Build toy3.json by adding exact, offline training calculations to toy.json.

The forward and backward passes use row vectors, matching shared.js.  The code
keeps the attention graph explicit so each gradient can be checked or taught.
Only the values written to JSON are rounded; every update uses full precision.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "toy.json"
OUTPUT = HERE / "toy3.json"
ETAS = (0.05, 0.1, 0.3)
FD_EPS = 1e-4


def array(model, name):
    return np.asarray(model[name], dtype=np.float64)


def softmax(rows):
    rows = np.asarray(rows, dtype=np.float64)
    shifted = rows - np.max(rows, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def parameter_arrays(model, tokens):
    """Return the differentiable arrays used by this sentence."""
    used = list(dict.fromkeys(t.lower() for t in tokens))
    return {
        "tok_emb_used": {t: np.asarray(model["tok_emb"][t], dtype=np.float64) for t in used},
        "pos_emb_used": array(model, "pos_emb")[: len(tokens)].copy(),
        "W_Q": array(model, "W_Q").copy(),
        "W_K": array(model, "W_K").copy(),
        "W_V": array(model, "W_V").copy(),
        "W_O": array(model, "W_O").copy(),
        "W_vocab": array(model, "W_vocab").copy(),
        "b_vocab": array(model, "b_vocab").copy(),
    }


def forward(model, tokens):
    vocab = model["vocab"]
    token_rows = np.stack([model["tok_emb"][t.lower()] for t in tokens]).astype(np.float64)
    positions = array(model, "pos_emb")[: len(tokens)]
    E = token_rows + positions
    W_Q, W_K, W_V = array(model, "W_Q"), array(model, "W_K"), array(model, "W_V")
    W_O, W_vocab, b_vocab = array(model, "W_O"), array(model, "W_vocab"), array(model, "b_vocab")
    Q, K, V = E @ W_Q, E @ W_K, E @ W_V
    scale = math.sqrt(model["d_k"])
    scores = Q @ K.T / scale
    causal = np.triu(np.ones_like(scores, dtype=bool), k=1)
    masked_scores = np.where(causal, -1e30, scores)
    A = softmax(masked_scores)
    messages = A @ V
    delta = messages @ W_O
    Enew = E + delta
    logits = Enew @ W_vocab + b_vocab
    probs = softmax(logits)
    return {
        "tokens": tokens,
        "vocab": vocab,
        "E": E,
        "Q": Q,
        "K": K,
        "V": V,
        "scores": masked_scores,
        "A": A,
        "messages": messages,
        "delta": delta,
        "Enew": Enew,
        "logits": logits,
        "probs": probs,
    }


def loss_and_grads(model, tokens, targets, output_positions):
    """Cross-entropy and a reverse pass through causal self-attention.

    ``targets[k]`` is predicted by ``output_positions[k]``.  A single final
    position and all positions therefore share the same implementation.
    """
    cache = forward(model, tokens)
    vocab_index = {t: i for i, t in enumerate(model["vocab"])}
    rows = np.asarray(output_positions, dtype=int)
    target_ids = np.asarray([vocab_index[t.lower()] for t in targets], dtype=int)
    chosen = cache["probs"][rows, target_ids]
    losses = -np.log(chosen)
    loss = float(np.mean(losses))

    dlogits = np.zeros_like(cache["logits"])
    for row, target_id in zip(rows, target_ids):
        dlogits[row] += cache["probs"][row]
        dlogits[row, target_id] -= 1.0
    dlogits /= len(rows)

    W_Q, W_K, W_V = array(model, "W_Q"), array(model, "W_K"), array(model, "W_V")
    W_O, W_vocab = array(model, "W_O"), array(model, "W_vocab")
    dW_vocab = cache["Enew"].T @ dlogits
    db_vocab = np.sum(dlogits, axis=0)
    dEnew = dlogits @ W_vocab.T

    ddelta = dEnew
    dmessages = ddelta @ W_O.T
    dW_O = cache["messages"].T @ ddelta
    dA = dmessages @ cache["V"].T
    dV = cache["A"].T @ dmessages
    dscore = cache["A"] * (dA - np.sum(dA * cache["A"], axis=1, keepdims=True))
    scale = math.sqrt(model["d_k"])
    dQ = dscore @ cache["K"] / scale
    dK = dscore.T @ cache["Q"] / scale

    dW_Q = cache["E"].T @ dQ
    dW_K = cache["E"].T @ dK
    dW_V = cache["E"].T @ dV
    dE = dEnew + dQ @ W_Q.T + dK @ W_K.T + dV @ W_V.T

    tok_grads = {}
    for i, token in enumerate(tokens):
        key = token.lower()
        tok_grads.setdefault(key, np.zeros(model["d_model"], dtype=np.float64))
        tok_grads[key] += dE[i]

    grads = {
        "tok_emb_used": tok_grads,
        "pos_emb_used": dE,
        "W_Q": dW_Q,
        "W_K": dW_K,
        "W_V": dW_V,
        "W_O": dW_O,
        "W_vocab": dW_vocab,
        "b_vocab": db_vocab,
    }
    return loss, losses, grads, cache


def apply_step(model, tokens, grads, eta):
    updated = copy.deepcopy(model)
    for token, grad in grads["tok_emb_used"].items():
        updated["tok_emb"][token] = (np.asarray(updated["tok_emb"][token]) - eta * grad).tolist()
    pos = array(updated, "pos_emb")
    pos[: len(tokens)] -= eta * grads["pos_emb_used"]
    updated["pos_emb"] = pos.tolist()
    for name in ("W_Q", "W_K", "W_V", "W_O", "W_vocab", "b_vocab"):
        updated[name] = (array(updated, name) - eta * grads[name]).tolist()
    return updated


def top_probabilities(cache, row, k=5):
    order = np.argsort(-cache["probs"][row])[:k]
    return [{"token": cache["vocab"][i], "p": cache["probs"][row, i]} for i in order]


def stored_parameters(model, tokens):
    p = parameter_arrays(model, tokens)
    return p


def finite_difference(model, tokens, target, name, index, analytic):
    def evaluate(m):
        return loss_and_grads(m, tokens, [target], [len(tokens) - 1])[0]

    plus, minus = copy.deepcopy(model), copy.deepcopy(model)
    plus[name][index[0]][index[1]] += FD_EPS
    minus[name][index[0]][index[1]] -= FD_EPS
    numeric = (evaluate(plus) - evaluate(minus)) / (2 * FD_EPS)
    return {
        "parameter": name,
        "index": list(index),
        "epsilon": FD_EPS,
        "analytic": analytic,
        "numeric": numeric,
        "abs_error": abs(analytic - numeric),
    }


def rounded(value):
    if isinstance(value, np.ndarray):
        return rounded(value.tolist())
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    if isinstance(value, float):
        result = round(value, 4)
        return 0.0 if result == 0 else result
    return value


def build_training(model):
    tokens = model["sentences"]["river"]
    target = "water"
    final = len(tokens) - 1
    before_loss, _, single_grads, before = loss_and_grads(model, tokens, [target], [final])
    target_id = model["vocab"].index(target)

    steps = {}
    for eta in ETAS:
        updated = apply_step(model, tokens, single_grads, eta)
        after_loss, _, _, after = loss_and_grads(updated, tokens, [target], [final])
        steps[str(eta)] = {
            "eta": eta,
            "p_target": after["probs"][final, target_id],
            "loss": after_loss,
            "e_prime_10": after["Enew"][final],
            "top_probabilities": top_probabilities(after, final),
            "parameters": stored_parameters(updated, tokens),
        }

    shifted_targets = [t.lower() for t in tokens[1:]] + [target]
    all_positions = list(range(len(tokens)))
    par_loss, par_losses, par_grads, par_before = loss_and_grads(
        model, tokens, shifted_targets, all_positions
    )
    par_updated = apply_step(model, tokens, par_grads, 0.1)
    par_after_loss, par_after_losses, _, par_after = loss_and_grads(
        par_updated, tokens, shifted_targets, all_positions
    )
    per_position = []
    for i, tok in enumerate(shifted_targets):
        tid = model["vocab"].index(tok)
        per_position.append(
            {
                "position": i + 1,
                "input": tokens[i],
                "target": tok,
                "loss_before": par_losses[i],
                "loss_after": par_after_losses[i],
                "p_target_before": par_before["probs"][i, tid],
                "p_target_after": par_after["probs"][i, tid],
            }
        )

    checks = [
        finite_difference(model, tokens, target, "W_Q", (3, 0), single_grads["W_Q"][3, 0]),
        finite_difference(
            model,
            tokens,
            target,
            "W_vocab",
            (0, target_id),
            single_grads["W_vocab"][0, target_id],
        ),
    ]
    max_error = max(c["abs_error"] for c in checks)
    if max_error > 1e-7:
        raise RuntimeError(f"finite-difference check failed: max error {max_error}")

    return {
        "sentence": tokens,
        "target": target,
        "target_position": len(tokens) + 1,
        "single": {
            "loss_before": before_loss,
            "p_target_before": before["probs"][final, target_id],
            "e_prime_10_before": before["Enew"][final],
            "top_probabilities_before": top_probabilities(before, final),
            "gradients": single_grads,
            "steps": steps,
        },
        "parallel": {
            "eta": 0.1,
            "targets": shifted_targets,
            "mean_loss_before": par_loss,
            "mean_loss_after": par_after_loss,
            "per_position": per_position,
            "gradients": par_grads,
            "parameters_after": stored_parameters(par_updated, tokens),
        },
        "finite_difference": {
            "checks": checks,
            "max_abs_error": max_error,
            "passed": True,
        },
        "note": "Stored parameters include every embedding row and position row used by the sentence, plus every projection and output-head parameter.",
    }


def main():
    model = json.loads(SOURCE.read_text(encoding="utf-8"))
    output = copy.deepcopy(model)
    output["training"] = rounded(build_training(model))
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    training = output["training"]
    print(f"wrote {OUTPUT.name}")
    print(f"single loss before: {training['single']['loss_before']:.4f}")
    for eta, step in training["single"]["steps"].items():
        print(f"eta {eta}: p(target) {step['p_target']:.4f}, loss {step['loss']:.4f}")
    print(
        "parallel mean loss: "
        f"{training['parallel']['mean_loss_before']:.4f} -> "
        f"{training['parallel']['mean_loss_after']:.4f}"
    )
    print(f"finite differences max abs error: {training['finite_difference']['max_abs_error']:.4g}")


if __name__ == "__main__":
    main()
