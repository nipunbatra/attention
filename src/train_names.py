#!/usr/bin/env python3
"""Train the small Part 1 character model and write toy1.json.

The implementation uses NumPy for all model arithmetic.  Names are split
before windows are made, so the reported held-out loss is name-level held out.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np


SEED = 1729
SAMPLE_SEED = 20260903
WINDOW = 3
EMBED_DIM = 2
HIDDEN_DIM = 32
VOCAB = ["-"] + list("abcdefghijklmnopqrstuvwxyz")
STOI = {token: i for i, token in enumerate(VOCAB)}
VOWELS = set("aeiou")
STEPS = 6000
BATCH_SIZE = 512
LEARNING_RATE = 0.012
VOWEL_MARGIN = 0.55
VOWEL_PENALTY = 0.16

ROOT = Path(__file__).resolve().parent


def load_names(path: Path) -> list[str]:
    """Lowercase, keep a-z only, and deduplicate without reordering."""
    names: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = re.sub(r"[^a-z]", "", (row.get("Name") or "").lower())
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    if not names:
        raise ValueError(f"no names found in {path}")
    return names


def make_windows(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    contexts: list[list[int]] = []
    targets: list[int] = []
    for name in names:
        context = [0] * WINDOW
        for char in name + "-":
            target = STOI[char]
            contexts.append(context.copy())
            targets.append(target)
            context = context[1:] + [target]
    return np.asarray(contexts, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def initialise(rng: np.random.Generator) -> dict[str, np.ndarray]:
    scale_1 = np.sqrt(1.0 / (WINDOW * EMBED_DIM))
    scale_2 = np.sqrt(1.0 / HIDDEN_DIM)
    return {
        "E": rng.normal(0.0, 0.22, (len(VOCAB), EMBED_DIM)),
        "W1": rng.normal(0.0, scale_1, (WINDOW * EMBED_DIM, HIDDEN_DIM)),
        "b1": np.zeros(HIDDEN_DIM),
        "W2": rng.normal(0.0, scale_2, (HIDDEN_DIM, len(VOCAB))),
        "b2": np.zeros(len(VOCAB)),
    }


def forward(params: dict[str, np.ndarray], ids: np.ndarray) -> dict[str, np.ndarray]:
    rows = params["E"][ids]
    a0 = rows.reshape(ids.shape[0], WINDOW * EMBED_DIM)
    a1 = np.tanh(a0 @ params["W1"] + params["b1"])
    z = a1 @ params["W2"] + params["b2"]
    shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    p = exp / exp.sum(axis=1, keepdims=True)
    return {"rows": rows, "a0": a0, "a1": a1, "z": z, "p": p}


def cross_entropy(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray, chunk: int = 8192) -> float:
    total = 0.0
    for start in range(0, len(X), chunk):
        xb, yb = X[start : start + chunk], y[start : start + chunk]
        p = forward(params, xb)["p"]
        total += float(-np.log(np.clip(p[np.arange(len(yb)), yb], 1e-12, 1.0)).sum())
    return total / len(X)


def loss_and_grads(
    params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray
) -> tuple[float, float, dict[str, np.ndarray]]:
    result = forward(params, X)
    p, a0, a1 = result["p"], result["a0"], result["a1"]
    n = len(X)
    ce = float(-np.log(np.clip(p[np.arange(n), y], 1e-12, 1.0)).mean())

    dz = p.copy()
    dz[np.arange(n), y] -= 1.0
    dz /= n
    grads = {
        "W2": a1.T @ dz,
        "b2": dz.sum(axis=0),
    }
    da1 = dz @ params["W2"].T
    du = da1 * (1.0 - a1 * a1)
    grads["W1"] = a0.T @ du
    grads["b1"] = du.sum(axis=0)
    da0 = (du @ params["W1"].T).reshape(n, WINDOW, EMBED_DIM)
    grads["E"] = np.zeros_like(params["E"])
    for pos in range(WINDOW):
        np.add.at(grads["E"], X[:, pos], da0[:, pos, :])

    # The first embedding axis is trained to have a readable sign.  Vowels
    # should lie above +margin and consonants below -margin.  This is a soft
    # penalty, so the language-model loss still decides the exact locations.
    signs = np.asarray([1.0 if token in VOWELS else -1.0 for token in VOCAB[1:]])
    signed = signs * params["E"][1:, 0]
    violation = np.maximum(0.0, VOWEL_MARGIN - signed)
    penalty = VOWEL_PENALTY * float(np.mean(violation * violation))
    grads["E"][1:, 0] += VOWEL_PENALTY * (-2.0 / len(signs)) * signs * violation
    boundary_penalty = 0.02 * float(params["E"][0, 0] ** 2)
    grads["E"][0, 0] += 0.04 * params["E"][0, 0]
    return ce, penalty + boundary_penalty, grads


def adam_train(
    params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> list[dict[str, float | int]]:
    first = {name: np.zeros_like(value) for name, value in params.items()}
    second = {name: np.zeros_like(value) for name, value in params.items()}
    curve: list[dict[str, float | int]] = []
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for step in range(1, STEPS + 1):
        idx = rng.integers(0, len(X), size=min(BATCH_SIZE, len(X)))
        ce, penalty, grads = loss_and_grads(params, X[idx], y[idx])
        for name in params:
            first[name] = beta1 * first[name] + (1.0 - beta1) * grads[name]
            second[name] = beta2 * second[name] + (1.0 - beta2) * (grads[name] * grads[name])
            m_hat = first[name] / (1.0 - beta1**step)
            v_hat = second[name] / (1.0 - beta2**step)
            params[name] -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + eps)
        if step % 50 == 0:
            curve.append({"step": step, "loss": ce, "objective": ce + penalty})
    return curve


def one_forward(params: dict[str, np.ndarray], ids: list[int]) -> dict[str, list[float]]:
    result = forward(params, np.asarray([ids], dtype=np.int64))
    return {key: result[key][0].tolist() for key in ("a0", "a1", "z", "p")}


def aabid_rows(params: dict[str, np.ndarray]) -> list[dict[str, object]]:
    name = "aabid"
    context = [0] * WINDOW
    rows: list[dict[str, object]] = []
    for target in name + "-":
        target_id = STOI[target]
        result = one_forward(params, context)
        rows.append(
            {
                "context": [VOCAB[i] for i in context],
                "ids": context.copy(),
                "target": target,
                "target_id": target_id,
                "probability": result["p"][target_id],
                "loss": -float(np.log(max(result["p"][target_id], 1e-12))),
                "probabilities": result["p"],
            }
        )
        context = context[1:] + [target_id]
    return rows


def sample_name(
    params: dict[str, np.ndarray], seed: int, temperature: float = 0.85, max_length: int = 18
) -> str:
    rng = np.random.default_rng(seed)
    context = [0] * WINDOW
    chars: list[str] = []
    for _ in range(max_length):
        z = np.asarray(one_forward(params, context)["z"], dtype=np.float64) / temperature
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        nxt = int(rng.choice(len(VOCAB), p=p))
        if nxt == 0:
            break
        chars.append(VOCAB[nxt])
        context = context[1:] + [nxt]
    return "".join(chars)


def jsonable(params: dict[str, np.ndarray]) -> dict[str, object]:
    return {name: value.tolist() for name, value in params.items()}


def main() -> None:
    names = load_names(ROOT / "names.csv")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(names))
    split = int(0.9 * len(names))
    train_names = [names[i] for i in order[:split]]
    held_names = [names[i] for i in order[split:]]
    if "aabid" in held_names:
        held_index = held_names.index("aabid")
        train_names[0], held_names[held_index] = held_names[held_index], train_names[0]
    X_train, y_train = make_windows(train_names)
    X_held, y_held = make_windows(held_names)

    params = initialise(rng)
    before = {name: value.copy() for name, value in params.items()}
    initial_train_loss = cross_entropy(params, X_train, y_train)
    curve = [{"step": 0, "loss": initial_train_loss, "objective": initial_train_loss}]
    curve.extend(adam_train(params, X_train, y_train, rng))
    train_loss = cross_entropy(params, X_train, y_train)
    held_loss = cross_entropy(params, X_held, y_held)

    samples: list[dict[str, object]] = []
    used: set[str] = set()
    candidate_seed = SAMPLE_SEED
    while len(samples) < 20:
        name = sample_name(params, candidate_seed, temperature=0.75)
        if len(name) >= 3 and name not in used:
            samples.append({"name": name, "seed": candidate_seed})
            used.add(name)
        candidate_seed += 1
        if candidate_seed - SAMPLE_SEED > 2000:
            raise RuntimeError("could not draw 20 distinct sample names")

    rows = aabid_rows(params)
    export: dict[str, object] = {
        "vocab": VOCAB,
        "E": params["E"].tolist(),
        "W1": params["W1"].tolist(),
        "b1": params["b1"].tolist(),
        "W2": params["W2"].tolist(),
        "b2": params["b2"].tolist(),
        "w": WINDOW,
        "d_model": EMBED_DIM,
        "d_h": HIDDEN_DIM,
        "axes": {"e": ["vowel-ness", "learned axis 2"]},
        "aabid_rows": rows,
        "training_curve": curve,
        "before": jsonable(before),
        "sample_seed": SAMPLE_SEED,
        "sample_temperature": 0.75,
        "sampled_names": samples,
        "metrics": {
            "train_loss": train_loss,
            "held_out_loss": held_loss,
            "initial_train_loss": initial_train_loss,
            "train_names": len(train_names),
            "held_out_names": len(held_names),
            "train_examples": len(X_train),
            "held_out_examples": len(X_held),
            "total_examples": len(X_train) + len(X_held),
            "steps": STEPS,
            "seed": SEED,
        },
        "notes": {
            "data": "Names were lowercased, stripped to a-z, deduplicated, and split by name before windows were made.",
            "boundary": "The token '-' has id 0 and serves as start padding and the end target.",
            "axis": "Embedding axis 1 was constrained during training with a margin penalty: vowels positive, consonants negative.",
            "display": "The page displays two decimal places; all forward passes use these full-precision values.",
            "sampling": "The exported names use seeded sampling at temperature 0.75.",
        },
    }
    (ROOT / "toy1.json").write_text(json.dumps(export, separators=(",", ":")), encoding="utf-8")

    signs_ok = all(params["E"][STOI[c], 0] > 0 for c in VOWELS) and all(
        params["E"][STOI[c], 0] < 0 for c in "abcdefghijklmnopqrstuvwxyz" if c not in VOWELS
    )
    if not signs_ok:
        raise RuntimeError("vowel-ness signs did not satisfy the export constraint")
    print(f"wrote toy1.json with {len(names)} unique names and {len(X_train) + len(X_held)} examples")
    print(f"train loss: {train_loss:.6f}")
    print(f"held-out loss: {held_loss:.6f}")
    print("samples: " + ", ".join(item["name"] for item in samples))
    print("aabid target probabilities: " + ", ".join(f"{row['probability']:.6f}" for row in rows))


if __name__ == "__main__":
    main()
