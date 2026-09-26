#!/usr/bin/env python3
"""Build and execute the small Vision I notebook from readable code cells.

Run from the repository root with Python, NumPy, Pillow, PyTorch and timm.
The two-head worksheet is deliberately hand chosen. The trained one-head
parameters are read from toy5.json; this script never changes them.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks/vision/01_classification_from_patches.ipynb"
TOY = json.loads((ROOT / "src/toy5.json").read_text())
RAMP = np.array([58, 110, 163, 216], dtype=np.uint8)

for key in "ABC":
    image = np.asarray(TOY["scenes"][key]["pixels"], dtype=np.uint8)
    name = {"A": "scene-a", "B": "scene-b", "C": "scene-c-heldout"}[key]
    Image.fromarray(RAMP[image], "L").save(ROOT / f"figures/vision-scene/{name}.png")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


cells = [
    md("""# Vision I lab: one image, two heads, one class

This lab follows the same 8×8 tabletop image as [Vision I](../../vision1.html). A and B are the two images used to fit the course toy; C is held out. The PNG files below encode exactly the 0–3 grayscale pixels shown in the lecture, so the classification runs on image files, not labels or region names. These are synthetic teaching images, not evidence of natural-image recognition.

The first two-head numbers are **hand chosen** to show the arithmetic. The later one-head parameters are the already fitted course toy. A final, [separately pretrained ViT](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k) runs on the two natural-looking JPEG illustrations as a reality check; its ImageNet head does not have a “mug on the right” class. The JPGs were never used by the course toy.

References: [original ViT paper](https://arxiv.org/abs/2010.11929), [Hugging Face image classification guide](https://huggingface.co/docs/transformers/main/tasks/image_classification), [Jay Alammar's visual explanation of attention](https://jalammar.github.io/illustrated-transformer/). All diagrams and numbers here are original to this course.
"""),
]

code_sources = [
"""from pathlib import Path
import json
import numpy as np
from PIL import Image

repo = Path.cwd().resolve()
if not (repo / "src/toy5.json").exists():
    repo = repo.parents[1]  # when launched in notebooks/vision
toy = json.loads((repo / "src/toy5.json").read_text())
names = {"A": "scene-a", "B": "scene-b", "C": "scene-c-heldout"}
ramp = np.array([58, 110, 163, 216])

def read_image(key):
    path = repo / "figures/vision-scene" / (names[key] + ".png")
    grey = np.asarray(Image.open(path).convert("L"))
    # The exact four grey levels carry toy pixel values 0, 1, 2, 3.
    assert grey.shape == (8, 8) and np.isin(grey, ramp).all()
    image = np.searchsorted(ramp, grey).astype(float)
    np.testing.assert_array_equal(image, toy["scenes"][key]["pixels"])
    return image

def patchify(image):
    # Row-major 2x2 patches, with pixels ordered TL, TR, BL, BR.
    return image.reshape(4, 2, 4, 2).transpose(0, 2, 1, 3).reshape(16, 4)

def embed(image):
    R = patchify(image)                     # 16 x 4 raw pixel rows
    content = R @ np.array(toy["W_patch"]) # 16 x 4, shared projection
    # The JSON display rows round thirds; the browser computes exact thirds.
    position = np.array([[0,0,-1,-1]] +
                        [[0,0,j//4/3,j%4/3] for j in range(16)])
    E = np.vstack([toy["cls"], content]) + position
    return R, E                             # E: 17 x 4, CLS first

def softmax(rows):
    e = np.exp(rows - np.max(rows, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

A = read_image("A")
R, E = embed(A)
print("image", A.shape, "patches", R.shape, "E", E.shape)
print("patch 8 pixels", R[7], "embedding", np.round(E[8], 3))
print("CLS", E[0])
""",
"""# The first head is the toy's initial one-head attention.
# The second head is an independent hand-chosen worksheet head.
first = toy["initial"]
second = {
    "W_Q": [[0, 1], [0, 0], [0, 0], [0, 0]],
    "W_K": [[0, 0], [1, 0], [0, 0], [0, 2]],
    "W_V": [[0, 1], [1, 0], [0, 0], [0, 0]],
}

def attention_head(E, weights):
    Q = E @ np.array(weights["W_Q"])
    K = E @ np.array(weights["W_K"])
    V = E @ np.array(weights["W_V"])
    scores = Q @ K.T / np.sqrt(Q.shape[-1])  # 17 x 17, no mask
    A = softmax(scores)                       # each row sums to one
    H = A @ V                                 # 17 x 2 value messages
    return Q, K, V, scores, A, H

q1, k1, v1, s1, a1, h1 = attention_head(E, first)
q2, k2, v2, s2, a2, h2 = attention_head(E, second)
for h, q, k, v, s, a, m in [(1,q1,k1,v1,s1,a1,h1),(2,q2,k2,v2,s2,a2,h2)]:
    print(f"head {h}: q_cls={q[0]}, k_patch8={k[8]}, score={s[0,8]:.3f}, "
          f"weight={a[0,8]:.3f}, v_patch8={v[8]}, message={np.round(m[0],3)}")
    assert q.shape == k.shape == v.shape == m.shape == (17, 2)
    assert s.shape == a.shape == (17, 17)
    np.testing.assert_allclose(a.sum(axis=1), 1)

joined = np.concatenate([h1, h2], axis=1)  # 17 x 4
W_O = np.eye(4)                              # hand chosen; easy to inspect
E_new = E + joined @ W_O                     # residual update: 17 x 4
W_class = np.array([[0,0],[0,0],[0,0],[2,-2]], dtype=float)
logits = E_new[0] @ W_class                   # one CLS row -> two class logits
probs = softmax(logits)
print("concat CLS", np.round(joined[0],3), "updated CLS", np.round(E_new[0],3))
print("class probabilities [right mug, no right mug]", np.round(probs,3))
""",
"""# The fitted course toy has one head. A and B were used for fitting.
# Read actual PNG pixels again for the held-out scene C.
def fitted_forward(image):
    _, rows = embed(image)
    p = toy["trained"]
    q, k, v, scores, attn, msg = attention_head(rows, p)
    updated = rows + msg @ np.array(p["W_O"])
    logits = updated[0] @ np.array(p["W_cls"]) + np.array(p["b_cls"])
    return softmax(logits), attn[0]

for key in "ABC":
    image = read_image(key)
    p, alpha = fitted_forward(image)
    prediction = toy["classes"][int(np.argmax(p))]
    print(key, "held out" if key == "C" else "fitted on", "label:",
          toy["scenes"][key]["label"], "prediction:", prediction,
          "p(right mug)=", round(float(p[0]),3))
    np.testing.assert_allclose(p, toy["scenes"][key]["probs_trained"], atol=0.0006)
print("C's eight image rows:\\n", read_image("C").astype(int))
""",
"""# PyTorch parity for the fitted image classifier; no new training run.
import torch
im = torch.tensor(read_image("C"), dtype=torch.float64)
patches = im.reshape(4,2,4,2).permute(0,2,1,3).reshape(16,4)
P = {k: torch.tensor(v, dtype=torch.float64) for k,v in toy["trained"].items()}
rows = torch.cat([torch.tensor(toy["cls"], dtype=torch.float64)[None],
                  patches @ torch.tensor(toy["W_patch"], dtype=torch.float64)])
position = [[0,0,-1,-1]] + [[0,0,j//4/3,j%4/3] for j in range(16)]
rows = rows + torch.tensor(position, dtype=torch.float64)
Q, K, V = rows @ P["W_Q"], rows @ P["W_K"], rows @ P["W_V"]
weights = torch.softmax(Q @ K.T / 2**0.5, dim=-1)
updated = rows + (weights @ V) @ P["W_O"]
torch_logits = updated[0] @ P["W_cls"] + P["b_cls"]
target = torch.tensor([0])  # scene C's ground-truth label: mug on the right
loss = torch.nn.functional.cross_entropy(torch_logits[None], target)
np.testing.assert_allclose(torch_logits.softmax(-1).numpy(), fitted_forward(read_image("C"))[0], atol=1e-12)
print("PyTorch C probabilities", torch_logits.softmax(-1).tolist(), "loss", round(loss.item(),3))
""",
"""# A separately pretrained ViT on actual JPEG illustrations.
# ImageNet has object categories; it has no left/right-mug class.
import timm
from timm.data import resolve_model_data_config, create_transform, ImageNetInfo
model_name = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
model = timm.create_model(model_name, pretrained=True).eval()
transform = create_transform(**resolve_model_data_config(model), is_training=False)
info = ImageNetInfo()
with torch.inference_mode():
    for filename in ("one-mug.jpg", "two-mugs.jpg"):
        jpg = Image.open(repo / "figures/vision-scene" / filename).convert("RGB")
        p = model(transform(jpg)[None]).softmax(-1)[0]
        i = int(p.argmax())
        print(filename, "top ImageNet class:", info.label_descriptions()[i],
              "p=", round(float(p[i]),3), "class index", i)
""",
]

cells += [md("""## From pixels to two independent messages

The second head asks about horizontal position because of its chosen matrices; that is a **possible interpretation of this worksheet**, not a role assigned to trained heads. The full 17×17 softmax includes CLS as a source. The browser lesson draws the same values in editable SVG.
""")]

namespace: dict = {}
for index, source in enumerate(code_sources):
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(source, f"notebook-cell-{index+1}", "exec"), namespace)
    stdout = capture.getvalue()
    cells.append({
        "cell_type": "code", "execution_count": index + 1, "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [{"output_type": "stream", "name": "stdout", "text": stdout.splitlines(keepends=True)}],
    })
    if index == 1:
        cells.append(md("""## A held-out image file

Scene C is never used by the fitted one-head model. It moves one mug into the right half. The code reads its PNG pixels, runs the *fitted* matrices, and reports the prediction whether it succeeds or fails. Two training images and one probe do not establish general image recognition.
"""))
    if index == 3:
        cells.append(md("""## From the worksheet to a pretrained image classifier

The next cell runs an ImageNet-pretrained ViT on the existing JPEG illustrations. It performs a real forward pass on photo-sized images. Its answer vocabulary describes objects; the two JPEGs both contain a plant and at least one mug, so do not treat its top class as an answer to the right-half question.
"""))

cells.append(md("""## Hugging Face usage on a new image

For a pretrained ImageNet classifier, the [official ViT API](https://huggingface.co/docs/transformers/main/model_doc/vit) keeps preprocessing with the checkpoint. This optional example downloads its checkpoint on first use. Its labels are ImageNet objects, not this lesson's right-half classes; adapting that task needs a new two-class head and training images.

```python
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
import torch

name = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(name)
model = ViTForImageClassification.from_pretrained(name).eval()
image = Image.open("figures/vision-scene/two-mugs.jpg").convert("RGB")
batch = processor(image, return_tensors="pt")
with torch.inference_mode():
    logits = model(**batch).logits
print(model.config.id2label[int(logits.argmax(-1))])
```
"""))

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4, "nbformat_minor": 4,
}
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
print("wrote", OUT.relative_to(ROOT), "with", len(cells), "cells")
