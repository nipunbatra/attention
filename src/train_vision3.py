#!/usr/bin/env python3
"""Reproduce the exact offline CLIP-style toy. No pretrained model is used.

python3 src/train_vision3.py         # write toy7.json
python3 src/train_vision3.py --check # compare without writing
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VOCAB = ["two", "bright", "squares", "horizontal", "stripes", "one", "dot", "a", "grid", "of", "with"]
CAPTIONS = ["two bright squares", "horizontal stripes", "one bright dot"]
IMAGES = [
    [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]],
    [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
    [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
]
LR = 0.1
STEPS = 60


def initial():
    wi = np.zeros((16, 3), dtype=np.float64)
    wi[0], wi[2], wi[6] = [2, 1, 0], [-2, 1, 1], [1, 0, 2]
    wt = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0],
                   [0, 0, 0], [0, 1, 2], [0, 0, 0], [.1, .2, .1], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
    return {"W_img": wi, "W_txt": wt, "log_scale": np.array(math.log(2.0))}


def features(captions):
    result = np.zeros((len(captions), len(VOCAB)))
    for i, caption in enumerate(captions):
        for word in caption.lower().split():
            result[i, VOCAB.index(word)] += 1
    return result


def row_softmax(logits):
    ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)


def forward(params):
    pixels = np.array(IMAGES, dtype=np.float64).reshape(3, 16)
    words = features(CAPTIONS)
    image_raw, text_raw = pixels @ params["W_img"], words @ params["W_txt"]
    image_norm, text_norm = np.linalg.norm(image_raw, axis=1, keepdims=True), np.linalg.norm(text_raw, axis=1, keepdims=True)
    image_unit, text_unit = image_raw / image_norm, text_raw / text_norm
    cosine = image_unit @ text_unit.T
    scale = np.exp(params["log_scale"])
    logits = scale * cosine
    row_prob = row_softmax(logits)
    column_prob = row_softmax(logits.T).T
    row_losses = np.logaddexp.reduce(logits, axis=1) - logits.diagonal()
    column_losses = np.logaddexp.reduce(logits, axis=0) - logits.diagonal()
    return {"pixels": pixels, "wordCounts": words, "imageRaw": image_raw, "textRaw": text_raw,
            "imageNorm": image_norm[:, 0], "textNorm": text_norm[:, 0],
            "imageUnit": image_unit, "textUnit": text_unit, "cosine": cosine, "scale": scale,
            "tau": 1 / scale, "logits": logits, "rowProb": row_prob, "columnProb": column_prob,
            "rowLosses": row_losses, "columnLosses": column_losses,
            "rowLoss": row_losses.mean(), "columnLoss": column_losses.mean(),
            "loss": (row_losses.mean() + column_losses.mean()) / 2}


def backward(params, f):
    n = len(IMAGES)
    dl = (f["rowProb"] + f["columnProb"] - 2 * np.eye(n)) / (2 * n)
    dc = dl * f["scale"]
    du_image = dc @ f["textUnit"]
    du_text = dc.T @ f["imageUnit"]
    # Exact derivative of g / ||g||: project away the radial component.
    dg_image = (du_image - f["imageUnit"] * np.sum(du_image * f["imageUnit"], axis=1, keepdims=True)) / f["imageNorm"][:, None]
    dg_text = (du_text - f["textUnit"] * np.sum(du_text * f["textUnit"], axis=1, keepdims=True)) / f["textNorm"][:, None]
    return {"W_img": f["pixels"].T @ dg_image, "W_txt": f["wordCounts"].T @ dg_text,
            "log_scale": np.array(np.sum(dl * f["logits"]))}


def step(params):
    grads = backward(params, forward(params))
    # Every gradient uses the same pre-update parameters.
    return {key: value - LR * grads[key] for key, value in params.items()}


def serial(value):
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("All saved CLIP toy values must be finite.")
    return value


def build():
    p = initial()
    gradients = backward(p, forward(p))
    checks = []
    for key, values in p.items():
        for index in np.ndindex(values.shape):
            original = values[index].copy()
            eps = 1e-5
            values[index] = original + eps
            plus = forward(p)["loss"]
            values[index] = original - eps
            minus = forward(p)["loss"]
            values[index] = original
            numeric = (plus - minus) / (2 * eps)
            checks.append({"parameter": key, "index": list(index), "analytic": gradients[key][index],
                           "numeric": numeric, "error": abs(numeric - gradients[key][index])})
    p_after = step(p)
    history = [forward(p)["loss"]]
    learned = {key: value.copy() for key, value in p.items()}
    for _ in range(STEPS):
        learned = step(learned)
        history.append(forward(learned)["loss"])
    clip = {"dimension": 3, "vocab": VOCAB, "captions": CAPTIONS, "imageNames": ["two-square grid", "striped grid", "dot grid"],
            "images": IMAGES, "learningRate": LR, "trainedSteps": STEPS,
            "snapshots": {"initial": p, "afterOne": p_after, "trained": learned},
            "reference": {"initial": forward(p), "afterOne": forward(p_after), "trained": forward(learned)},
            "initialGradients": gradients, "history": history,
            "gradientCheck": {"epsilon": 1e-5, "count": len(checks), "maxError": max(c["error"] for c in checks), "checks": checks},
            "notes": ["Hand-chosen starting matrices give readable three-coordinate outputs.",
                      "The image encoder is a trainable linear map of 16 pixels; the text encoder sums trainable word rows.",
                      "This is not a pretrained CLIP checkpoint, ViT, or language Transformer. The 60-step snapshot is fitted only to the three displayed pairs.",
                      "Both encoders and log_scale learn. Gradients include exact unit-normalization derivatives and average both directions over the batch.",
                      "Prompt controls use this finite toy vocabulary. They illustrate the calculation, not natural-language generalization."]}
    return serial({"d_model": 3, "d_k": 3, "d_v": 3, "vocab": VOCAB, "sentences": {}, "axes": {"e": [], "qk": [], "v": []}, "clip": clip})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build()
    path = HERE / "toy7.json"
    if args.check:
        if json.loads(path.read_text()) != result:
            raise AssertionError("Saved toy7.json does not reproduce exactly.")
        print("PASS: exact CLIP toy reproduction.")
    else:
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, ref in result["clip"]["reference"].items():
        print(name, "row", ref["rowLoss"], "column", ref["columnLoss"], "mean", ref["loss"], "tau", ref["tau"])
    print("gradient check:", result["clip"]["gradientCheck"]["count"], result["clip"]["gradientCheck"]["maxError"])
