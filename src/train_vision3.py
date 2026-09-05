#!/usr/bin/env python3
"""Vision III (CLIP) toy, version 2: the frozen Vision I encoder, a learned 4 x 3 image map, a bag-of-words text map.

python3 src/train_vision3.py          # write src/toy7.json
python3 src/train_vision3.py --check  # recompute and compare with the saved file, writing nothing

The image encoder is the trained Vision I encoder from toy5.json ("trained": W_Q, W_K, W_V, W_O, W_cls, b_cls) run over
the shared 8x8 scene exactly as vision-shared.js runs it (fixed W_patch, unrounded positions, CLS start row). Its updated
CLS row (brightness, contrast, row, col) goes through a learned 4 x 3 map onto the joint axes mug / book / plant. The text
encoder adds one learned row per word. Both outputs are unit-normalised, compared by cosine, scaled by a learned logit
scale, and trained with the symmetric contrastive loss for 60 SGD steps on three pairs. Every parameter gradient is checked
by central differences. NumPy only; no pretrained model.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AXES = ["mug", "book", "plant"]
VOCAB = ["a", "two", "one", "mug", "mugs", "book", "plant", "small", "on", "table", "and", "photo", "of", "the"]
# The first three pairs are the training batch. The rest extend the batch-size demonstration only.
PAIRS = [
    {"scene": "A", "caption": "two mugs on a table"},
    {"scene": "D", "caption": "a book on a table"},
    {"scene": "E", "caption": "a small plant"},
    {"scene": "B", "caption": "one mug on a table"},
    {"scene": "F", "caption": "two mugs"},
    {"scene": "G", "caption": "a book and a plant"},
    {"scene": "H", "caption": "a mug and a plant"},
    {"scene": "I", "caption": "a mug and a book"},
]
TRAIN = 3
LR = 0.05
STEPS = 60
SEED = 0


# ---------- the scene and the frozen encoder (mirrors vision-shared.js) ----------
def blank():
    return np.zeros((8, 8))


def mug(g, c0):
    g[1:4, c0:c0 + 3] = 3


def book(g):
    g[5:7, 1:6] = 2


def plant(g):
    g[4][7] = 1


def scenes():
    s = {}
    s["A"] = blank(); mug(s["A"], 1); mug(s["A"], 5); book(s["A"]); plant(s["A"])
    s["B"] = blank(); mug(s["B"], 1); book(s["B"]); plant(s["B"])
    s["C"] = blank(); mug(s["C"], 4); book(s["C"]); plant(s["C"])
    s["D"] = blank(); book(s["D"])
    s["E"] = blank(); plant(s["E"])
    s["F"] = blank(); mug(s["F"], 1); mug(s["F"], 5)
    s["G"] = blank(); book(s["G"]); plant(s["G"])
    s["H"] = blank(); mug(s["H"], 1); plant(s["H"])
    s["I"] = blank(); mug(s["I"], 5); book(s["I"])
    return s


SCENE_NAMES = {"A": "two mugs", "B": "one mug", "C": "one mug, moved right", "D": "book only", "E": "plant only",
               "F": "two mugs only", "G": "book and plant", "H": "one mug and plant", "I": "right mug and book"}
W_PATCH = np.array([[.25, .5, 0, 0], [.25, -.5, 0, 0], [.25, .5, 0, 0], [.25, -.5, 0, 0]])
CLS_START = np.array([1.0, 0, 0, 0])
POS = np.array([[0, 0, -1, -1]] + [[0, 0, (j // 4) / 3, (j % 4) / 3] for j in range(16)], dtype=float)


def patchify(g):
    return np.array([[g[2 * pr][2 * pc], g[2 * pr][2 * pc + 1], g[2 * pr + 1][2 * pc], g[2 * pr + 1][2 * pc + 1]]
                     for pr in range(4) for pc in range(4)], dtype=float)


def softmax_rows(x):
    ex = np.exp(x - x.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


def attend(g, enc):
    """One attention layer over CLS + 16 patches, exactly as V.attend in vision-shared.js."""
    E = np.vstack([CLS_START, patchify(g) @ W_PATCH]) + POS
    Q, K, Vv = E @ np.array(enc["W_Q"]), E @ np.array(enc["W_K"]), E @ np.array(enc["W_V"])
    S = (Q @ K.T) / math.sqrt(2)
    A = softmax_rows(S)
    H = A @ Vv
    D = H @ np.array(enc["W_O"])
    Enew = E + D
    logits = np.array(enc["b_cls"]) + Enew[0] @ np.array(enc["W_cls"])
    return {"E": E, "A": A, "H": H, "Delta": D, "Enew": Enew, "logits": logits, "probs": softmax_rows(logits)}


# ---------- the CLIP toy ----------
def words(captions):
    m = np.zeros((len(captions), len(VOCAB)))
    for i, caption in enumerate(captions):
        for w in caption.lower().split():
            m[i, VOCAB.index(w)] += 1
    return m


def initial():
    # Names first. Image map rows are the CLS axes (brightness, contrast, row, col); columns are mug, book, plant.
    # Bright material speaks for every object, most for a mug and least for a plant; the update that Vision I's
    # encoder writes onto the row and col coordinates when it finds a mug on the right adds to mug and subtracts from plant.
    w_img = np.array([[1.0, 1.0, 0.5], [0.0, 0.0, 0.0], [1.0, 0.0, -0.5], [1.0, 0.0, -0.5]])
    w_txt = np.zeros((len(VOCAB), 3))
    for word, axis in (("mug", 0), ("mugs", 0), ("book", 1), ("plant", 2)):
        w_txt[VOCAB.index(word), axis] = 1.0
    return {"W_img": w_img, "W_txt": w_txt, "log_scale": np.array(math.log(2.0))}


def forward(params, cls_rows, word_rows):
    img_raw, txt_raw = cls_rows @ params["W_img"], word_rows @ params["W_txt"]
    img_norm = np.linalg.norm(img_raw, axis=1, keepdims=True)
    txt_norm = np.linalg.norm(txt_raw, axis=1, keepdims=True)
    img_unit, txt_unit = img_raw / img_norm, txt_raw / txt_norm
    cosine = img_unit @ txt_unit.T
    scale = float(np.exp(params["log_scale"]))
    logits = scale * cosine
    n = min(logits.shape)
    row_prob = softmax_rows(logits)
    col_prob = softmax_rows(logits.T).T
    diag = np.array([logits[i, i] for i in range(n)])
    row_losses = np.logaddexp.reduce(logits, axis=1)[:n] - diag
    col_losses = np.logaddexp.reduce(logits, axis=0)[:n] - diag
    return {"clsRows": cls_rows, "wordRows": word_rows, "imgRaw": img_raw, "txtRaw": txt_raw,
            "imgNorm": img_norm[:, 0], "txtNorm": txt_norm[:, 0], "imgUnit": img_unit, "txtUnit": txt_unit,
            "cosine": cosine, "scale": scale, "tau": 1 / scale, "logits": logits, "rowProb": row_prob, "colProb": col_prob,
            "rowLosses": row_losses, "colLosses": col_losses, "rowLoss": row_losses.mean(), "colLoss": col_losses.mean(),
            "loss": (row_losses.mean() + col_losses.mean()) / 2}


def backward(params, f):
    n = f["logits"].shape[0]
    dlogits = (f["rowProb"] + f["colProb"] - 2 * np.eye(n)) / (2 * n)
    dcos = dlogits * f["scale"]
    du_img, du_txt = dcos @ f["txtUnit"], dcos.T @ f["imgUnit"]
    # exact derivative of g / ||g||: remove the radial component, then divide by the length
    dg_img = (du_img - f["imgUnit"] * np.sum(du_img * f["imgUnit"], axis=1, keepdims=True)) / f["imgNorm"][:, None]
    dg_txt = (du_txt - f["txtUnit"] * np.sum(du_txt * f["txtUnit"], axis=1, keepdims=True)) / f["txtNorm"][:, None]
    return {"W_img": f["clsRows"].T @ dg_img, "W_txt": f["wordRows"].T @ dg_txt,
            "log_scale": np.array(np.sum(dlogits * f["logits"]))}


def step(params, cls_rows, word_rows, lr=LR):
    grads = backward(params, forward(params, cls_rows, word_rows))
    return {key: value - lr * grads[key] for key, value in params.items()}   # all gradients from the same old parameters


def copy(params):
    return {key: np.array(value, dtype=float).copy() for key, value in params.items()}


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
        raise ValueError("every saved number must be finite")
    return value


def build():
    toy5 = json.loads((HERE / "toy5.json").read_text())
    enc = toy5["trained"]
    S = scenes()
    frozen = {key: attend(S[key], enc) for key in S}
    cls_all = {key: frozen[key]["Enew"][0] for key in S}
    train_scenes = [p["scene"] for p in PAIRS[:TRAIN]]
    train_caps = [p["caption"] for p in PAIRS[:TRAIN]]
    cls_rows = np.array([cls_all[k] for k in train_scenes])
    word_rows = words(train_caps)

    p = initial()
    f0 = forward(p, cls_rows, word_rows)
    grads = backward(p, f0)

    # central-difference check of every parameter, plus one random direction (seeded)
    rng = np.random.default_rng(SEED)
    eps = 1e-5
    checks = []
    for key, values in p.items():
        for index in np.ndindex(values.shape):
            original = values[index].copy()
            values[index] = original + eps
            plus = forward(p, cls_rows, word_rows)["loss"]
            values[index] = original - eps
            minus = forward(p, cls_rows, word_rows)["loss"]
            values[index] = original
            numeric = (plus - minus) / (2 * eps)
            checks.append({"parameter": key, "index": list(index), "analytic": grads[key][index], "numeric": numeric,
                           "error": abs(numeric - grads[key][index])})
    direction = {key: rng.standard_normal(value.shape) for key, value in p.items()}
    dir_plus = forward({k: p[k] + eps * direction[k] for k in p}, cls_rows, word_rows)["loss"]
    dir_minus = forward({k: p[k] - eps * direction[k] for k in p}, cls_rows, word_rows)["loss"]
    dir_numeric = (dir_plus - dir_minus) / (2 * eps)
    dir_analytic = sum(float(np.sum(grads[k] * direction[k])) for k in p)

    # 60 steps; keep every checkpoint and the six unit vectors at every step for the animation
    checkpoints, history, trajectory = [], [], []
    q = copy(p)
    for t in range(STEPS + 1):
        f = forward(q, cls_rows, word_rows)
        checkpoints.append(copy(q))
        history.append(f["loss"])
        trajectory.append({"img": f["imgUnit"], "txt": f["txtUnit"], "loss": f["loss"], "tau": f["tau"],
                           "cosine": f["cosine"], "diag": [f["rowProb"][i][i] for i in range(TRAIN)]})
        if t < STEPS:
            q = step(q, cls_rows, word_rows)
    after_one, trained = checkpoints[1], checkpoints[STEPS]
    f1, ft = forward(after_one, cls_rows, word_rows), forward(trained, cls_rows, word_rows)

    # zero-shot references with the frozen trained encoders: a few candidate lists, several scenes
    def classify(scene, captions, params=trained, tau=None):
        f = forward(params, np.array([cls_all[scene]]), words(captions))
        cosine = f["cosine"][0]
        t = f["tau"] if tau is None else tau
        probs = softmax_rows(np.array([cosine / t]))[0]
        return {"scene": scene, "captions": list(captions), "tau": t, "cosine": cosine, "probs": probs, "best": int(np.argmax(cosine))}

    zero_shot = [
        classify("A", ["two mugs", "a book", "a plant"]),
        classify("D", ["two mugs", "a book", "a plant"]),
        classify("E", ["two mugs", "a book", "a plant"]),
        classify("B", ["one mug", "two mugs", "a book", "a plant"]),
        classify("A", ["a book", "a plant"]),
        classify("A", ["a photo of two mugs", "a photo of a book", "a photo of a plant"]),
        classify("B", ["a photo of one mug", "a photo of two mugs", "a photo of a book", "a photo of a plant"]),
        classify("H", ["one mug", "two mugs", "a book", "a plant"]),
    ]

    # batch-size references: the first N pairs with the trained parameters, and the collapsed loss log N
    batches = []
    for n in range(TRAIN, len(PAIRS) + 1):
        rows = np.array([cls_all[pp["scene"]] for pp in PAIRS[:n]])
        f = forward(trained, rows, words([pp["caption"] for pp in PAIRS[:n]]))
        batches.append({"n": n, "loss": f["loss"], "rowLoss": f["rowLoss"], "colLoss": f["colLoss"],
                        "diag": [f["rowProb"][i][i] for i in range(n)], "meanDiag": float(np.mean([f["rowProb"][i][i] for i in range(n)])),
                        "collapse": math.log(n), "cosine": f["cosine"]})

    clip = {
        "axes": AXES, "vocab": VOCAB, "pairs": PAIRS, "trainPairs": TRAIN, "learningRate": LR, "steps": STEPS, "seed": SEED,
        "clsAxes": toy5["axes"]["e"], "sceneNames": SCENE_NAMES,
        "clsRows": {key: cls_all[key] for key in S},
        "frozen": {key: {"attention": frozen[key]["A"][0], "probs": frozen[key]["probs"], "logits": frozen[key]["logits"]} for key in S},
        "snapshots": {"initial": p, "afterOne": after_one, "trained": trained},
        "checkpoints": checkpoints,
        "reference": {"initial": f0, "afterOne": f1, "trained": ft},
        "initialGradients": grads,
        "history": history, "trajectory": trajectory,
        "gradientCheck": {"epsilon": eps, "count": len(checks), "maxError": max(c["error"] for c in checks), "checks": checks,
                          "direction": {"analytic": dir_analytic, "numeric": dir_numeric, "error": abs(dir_analytic - dir_numeric)}},
        "zeroShot": zero_shot, "batches": batches,
        "notes": [
            "Image encoder: the frozen trained Vision I encoder (toy5.json 'trained') run on the shared 8x8 scene; its updated CLS row (brightness, contrast, row, col) times a learned 4x3 map onto the joint axes mug, book, plant.",
            "Text encoder: bag of words; one learned 3-number row per word, named by the same axes; mug, mugs, book and plant start one-hot, every other word starts at zero.",
            "The three CLS rows lie on one line (the frozen encoder was trained for one question), so the initial image map was chosen by a small grid search over half-integer entries to make each scene's largest coordinate its own axis.",
            "Training: plain SGD, learning rate 0.05, 60 steps, symmetric cross-entropy over the three observed pairs; log_scale is learned; exact unit-normalisation gradients; every gradient checked by central differences.",
            "Pairs 4 to 8 are never trained. They only extend the batch-size demonstration and the zero-shot probes.",
        ],
    }
    return serial({"d_model": 3, "d_k": 3, "d_v": 3, "vocab": VOCAB, "sentences": {},
                   "axes": {"e": AXES, "qk": AXES, "v": AXES, "short": {"e": AXES, "qk": AXES, "v": AXES}},
                   "encoder": enc, "encoderClasses": toy5["classes"], "clip": clip})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build()
    path = HERE / "toy7.json"
    if args.check:
        if json.loads(path.read_text()) != result:
            raise AssertionError("saved toy7.json does not reproduce exactly")
        print("PASS: toy7.json reproduces exactly.")
    else:
        path.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
        print("wrote", path)
    c = result["clip"]
    for name, ref in c["reference"].items():
        print(f"{name:9s} loss {ref['loss']:.4f} (rows {ref['rowLoss']:.4f}, columns {ref['colLoss']:.4f}) tau {ref['tau']:.3f} diag p {[round(ref['rowProb'][i][i], 3) for i in range(3)]}")
    print("cosine before", np.round(np.array(c["reference"]["initial"]["cosine"]), 3).tolist())
    print("cosine after ", np.round(np.array(c["reference"]["trained"]["cosine"]), 3).tolist())
    print("gradient check:", c["gradientCheck"]["count"], "parameters, max error", c["gradientCheck"]["maxError"],
          "; random direction error", c["gradientCheck"]["direction"]["error"])
    for z in c["zeroShot"]:
        print("zero-shot", z["scene"], {cap: round(cos, 3) for cap, cos in zip(z["captions"], z["cosine"])}, "->", z["captions"][z["best"]])
    for b in c["batches"]:
        print(f"batch N={b['n']}: loss {b['loss']:.3f}, mean p(correct) {b['meanDiag']:.3f}, collapse log N = {b['collapse']:.3f}")
