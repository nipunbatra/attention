#!/usr/bin/env python3
"""Independent NumPy check of the current Vision I fitted toy.

The browser uses vision-shared.js and part5.js. This script recomputes the
forward pass from toy5.json without importing either runtime. It checks the
stored A/B fit, the held-out C prediction, every recorded CLS attention row,
and the training-loss summary. It does not claim general image recognition.

Run: python3 src/verify_vision1_learning.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TOY = json.loads((Path(__file__).parent / "toy5.json").read_text())


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(shifted)
    return ex / ex.sum(axis=-1, keepdims=True)


def forward(key: str, which: str) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(TOY["scenes"][key]["pixels"], dtype=float)
    assert image.shape == (8, 8) and np.isin(image, (0, 1, 2, 3)).all()
    # Sixteen 2x2 patches, row-major. Four pixels in TL, TR, BL, BR order.
    patches = image.reshape(4, 2, 4, 2).transpose(0, 2, 1, 3).reshape(16, 4)
    assert patches.shape == (16, 4)

    # Use exact thirds. The position table in JSON is rounded for display;
    # the teaching runtime also computes these fractions exactly.
    positions = np.asarray([[0, 0, -1, -1]] +
                           [[0, 0, j // 4 / 3, j % 4 / 3] for j in range(16)])
    E = np.vstack([TOY["cls"], patches @ np.asarray(TOY["W_patch"])]) + positions
    assert E.shape == (17, 4)

    p = TOY[which]
    Q, K, V = (E @ np.asarray(p["W_" + role]) for role in "QKV")
    scores = Q @ K.T / np.sqrt(2)
    A = softmax(scores)
    H = A @ V
    updated = E + H @ np.asarray(p["W_O"])
    logits = updated[0] @ np.asarray(p["W_cls"]) + np.asarray(p["b_cls"])
    probs = softmax(logits)
    assert Q.shape == K.shape == V.shape == H.shape == (17, 2)
    assert scores.shape == A.shape == (17, 17)
    np.testing.assert_allclose(A.sum(axis=1), 1, atol=1e-14)
    return probs, A[0]


def main() -> None:
    assert TOY["classes"] == ["mug on the right", "no mug on the right"]
    losses = {}
    for which in ("initial", "trained"):
        losses[which] = []
        for key in "ABC":
            probs, cls_attention = forward(key, which)
            stored = TOY["scenes"][key]
            np.testing.assert_allclose(probs, stored["probs_" + which], atol=0.00051)
            np.testing.assert_allclose(cls_attention, stored["cls_attention_" + which], atol=0.00051)
            label = TOY["classes"].index(stored["label"])
            if key in "AB":
                losses[which].append(-np.log(probs[label]))
            if which == "trained":
                prediction = TOY["classes"][int(np.argmax(probs))]
                assert prediction == stored["label"]
                print(f"scene {key} ({'held out' if key == 'C' else 'fitted on'}): "
                      f"{prediction}, p(correct)={probs[label]:.3f}")

    initial = float(np.mean(losses["initial"]))
    trained = float(np.mean(losses["trained"]))
    curve = np.asarray(TOY["curve"], dtype=float)
    assert curve.ndim == 2 and curve.shape[1] == 2
    assert curve[0, 0] == 1 and curve[-1, 0] == 1500
    assert curve[-1, 1] < curve[0, 1] < initial
    assert trained < 0.01 and trained < initial
    print(f"mean A/B loss: {initial:.6f} → {trained:.6f}; "
          f"{len(curve)} recorded training points; NumPy parity PASS")


if __name__ == "__main__":
    main()
