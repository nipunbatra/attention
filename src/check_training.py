#!/usr/bin/env python3
"""Read-only regression checks for the Part 3 training example.

Run ``python3 src/check_training.py`` from any directory. This imports the
training functions without running their JSON-writing main(), reproduces every
saved training result, and checks every scalar parameter used by the final-token
loss with central finite differences. No model or other file is written.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

# Importing the implementation must not create __pycache__ in a read-only check.
sys.dont_write_bytecode = True

import numpy as np


HERE = Path(__file__).resolve().parent
EPSILON = 1e-5
GRADIENT_TOLERANCE = 1e-8


def load_training_module():
    spec = importlib.util.spec_from_file_location("attention_train_part3", HERE / "train_part3.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load train_part3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_saved(actual, expected, path="training"):
    """Compare the rounded JSON tree exactly, including keys and array lengths."""
    numbers = 0
    max_error = 0.0
    mismatches = []

    def visit(left, right, location):
        nonlocal numbers, max_error
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                missing = sorted(left.keys() - right.keys())
                extra = sorted(right.keys() - left.keys())
                mismatches.append(f"{location}: missing saved keys {missing}, extra saved keys {extra}")
            for key in left:
                if key in right:
                    visit(left[key], right[key], f"{location}.{key}")
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                mismatches.append(f"{location}: computed length {len(left)}, saved length {len(right)}")
            for i, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f"{location}[{i}]")
        elif isinstance(left, bool) or isinstance(right, bool):
            if type(left) is not type(right) or left != right:
                mismatches.append(f"{location}: computed {left!r}, saved {right!r}")
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            numbers += 1
            if not math.isfinite(left) or not math.isfinite(right):
                mismatches.append(f"{location}: non-finite value")
                max_error = math.inf
            else:
                error = abs(left - right)
                max_error = max(max_error, error)
                if left != right:
                    mismatches.append(f"{location}: computed {left!r}, saved {right!r}")
        elif type(left) is not type(right) or left != right:
            mismatches.append(f"{location}: computed {left!r}, saved {right!r}")

    visit(actual, expected, path)
    return numbers, max_error, mismatches


def parameters(grads):
    """Yield model paths and analytic gradients, including shared token rows."""
    for token, row in grads["tok_emb_used"].items():
        for i, value in enumerate(row):
            yield ("tok_emb", token, i), float(value)
    for index in np.ndindex(grads["pos_emb_used"].shape):
        yield ("pos_emb", *index), float(grads["pos_emb_used"][index])
    for name in ("W_Q", "W_K", "W_V", "W_O", "W_vocab", "b_vocab"):
        for index in np.ndindex(grads[name].shape):
            yield (name, *index), float(grads[name][index])


def final_token_loss(module, model, tokens, target):
    """Evaluate only the forward graph, independently of the reverse pass."""
    logits = module.forward(model, tokens)["logits"][-1]
    target_id = model["vocab"].index(target)
    # Stable log-sum-exp cross-entropy, without calling loss_and_grads.
    peak = float(np.max(logits))
    return peak + math.log(float(np.exp(logits - peak).sum())) - float(logits[target_id])


def check_gradients(module, model, tokens, target):
    loss, _, grads, _ = module.loss_and_grads(model, tokens, [target], [len(tokens) - 1])
    forward_loss = final_token_loss(module, model, tokens, target)
    if not math.isfinite(loss) or abs(loss - forward_loss) > 1e-12:
        raise RuntimeError(f"Forward loss mismatch: reverse-pass loss {loss}, independent loss {forward_loss}")

    count = 0
    max_error = 0.0
    worst = None
    failures = []
    work = copy.deepcopy(model)
    for path, analytic in parameters(grads):
        parent = work
        for part in path[:-1]:
            parent = parent[part]
        index = path[-1]
        original = parent[index]
        try:
            parent[index] = original + EPSILON
            plus = final_token_loss(module, work, tokens, target)
            parent[index] = original - EPSILON
            minus = final_token_loss(module, work, tokens, target)
        finally:
            parent[index] = original
        numeric = (plus - minus) / (2 * EPSILON)
        error = abs(numeric - analytic)
        label = str(path[0]) + "".join(f"[{part}]" for part in path[1:])
        count += 1
        if not math.isfinite(error):
            failures.append(f"{label}: non-finite analytic/numeric gradient")
            max_error, worst = math.inf, label
        elif error > max_error:
            max_error, worst = error, label
        if math.isfinite(error) and error > GRADIENT_TOLERANCE:
            failures.append(f"{label}: analytic {analytic:.12g}, numeric {numeric:.12g}, error {error:.3g}")

    # Count from actual model shapes, not the gradient arrays, to catch omissions.
    used_tokens = list(dict.fromkeys(token.lower() for token in tokens))
    expected = sum(len(model["tok_emb"][token]) for token in used_tokens)
    expected += np.asarray(model["pos_emb"][: len(tokens)]).size
    expected += sum(np.asarray(model[name]).size for name in ("W_Q", "W_K", "W_V", "W_O", "W_vocab", "b_vocab"))
    if count != expected:
        failures.append(f"Checked {count} scalar parameters, expected {expected}")
    return count, len(used_tokens), max_error, worst, failures


def main():
    module = load_training_module()
    model = json.loads((HERE / "toy.json").read_text(encoding="utf-8"))
    saved = json.loads((HERE / "toy3.json").read_text(encoding="utf-8"))
    computed = module.rounded(module.build_training(model))
    numbers, saved_error, failures = compare_saved(computed, saved.get("training"))
    print(f"Saved training: {numbers} numeric entries; max rounded error {saved_error:.3g}")

    tokens, target = computed["sentence"], computed["target"]
    count, token_rows, gradient_error, worst, gradient_failures = check_gradients(module, model, tokens, target)
    failures.extend(gradient_failures)
    print(f"Gradients: {count} scalars ({token_rows} token rows, {len(tokens)} position rows, all projections/head)")
    print(f"Central differences: epsilon {EPSILON:g}; max absolute error {gradient_error:.3g} at {worst}")
    if failures:
        for failure in failures[:12]:
            print(f"FAIL: {failure}", file=sys.stderr)
        if len(failures) > 12:
            print(f"... {len(failures) - 12} additional mismatches", file=sys.stderr)
        return 1
    print("PASS: saved training and all used parameter gradients agree; no files written")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
