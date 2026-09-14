#!/usr/bin/env python3
"""Read-only regression checks for the Part 3 training example.

Run ``python3 src/check_training.py`` from any directory. This imports the
training functions without running their JSON-writing main(), reproduces every
saved training result, and checks every scalar parameter used by the final-token
loss with central finite differences wherever ReLU is differentiable. A smooth
probe checks every parameter; exact-zero ReLU units use derivative zero.
No model or other file is written.
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
    """Compare lesson arrays and full-precision diagnostics, including shapes."""
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
                # Keep full-precision results, allowing only floating-point
                # summation differences across BLAS implementations.
                tolerance = 2e-11 if ".finite_difference." in location else 1e-12
                if error > tolerance:
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
    for name in ("W_Q", "W_K", "W_V", "W_O", "W_hidden", "b_hidden", "W_vocab", "b_vocab"):
        for index in np.ndindex(grads[name].shape):
            yield (name, *index), float(grads[name][index])


def final_token_loss(module, model, tokens, target, parallel=False):
    """Evaluate only the forward graph, independently of the reverse pass."""
    logits = module.forward(model, tokens)["logits"]
    targets = [token.lower() for token in tokens[1:]] + [target] if parallel else [target]
    positions = list(range(len(tokens))) if parallel else [len(tokens)-1]
    # Stable log-sum-exp cross-entropy, without calling loss_and_grads.
    losses = []
    for position, token in zip(positions, targets):
        row = logits[position]
        peak = float(np.max(row))
        losses.append(peak + math.log(float(np.exp(row - peak).sum())) - float(row[model["vocab"].index(token)]))
    return sum(losses) / len(losses)


def check_gradients(module, model, tokens, target, parallel=False):
    targets = [token.lower() for token in tokens[1:]] + [target] if parallel else [target]
    positions = list(range(len(tokens))) if parallel else [len(tokens)-1]
    loss, _, grads, _ = module.loss_and_grads(model, tokens, targets, positions)
    forward_loss = final_token_loss(module, model, tokens, target, parallel)
    if not math.isfinite(loss) or abs(loss - forward_loss) > 1e-12:
        raise RuntimeError(f"Forward loss mismatch: reverse-pass loss {loss}, independent loss {forward_loss}")

    count = 0
    max_error = 0.0
    worst = None
    failures = []
    kinks = []
    work = copy.deepcopy(model)
    for path, analytic in parameters(grads):
        parent = work
        for part in path[:-1]:
            parent = parent[part]
        index = path[-1]
        original = parent[index]
        try:
            parent[index] = original + EPSILON
            plus = final_token_loss(module, work, tokens, target, parallel)
            plus_active = module.forward(work, tokens)["head_pre"][positions] > 0
            parent[index] = original - EPSILON
            minus = final_token_loss(module, work, tokens, target, parallel)
            minus_active = module.forward(work, tokens)["head_pre"][positions] > 0
        finally:
            parent[index] = original
        numeric = (plus - minus) / (2 * EPSILON)
        error = abs(numeric - analytic)
        label = str(path[0]) + "".join(f"[{part}]" for part in path[1:])
        count += 1
        if np.any(plus_active != minus_active):
            kinks.append(label)
            continue  # A central slope across a ReLU kink is not its derivative.
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
    expected += sum(np.asarray(model[name]).size for name in module.PARAMETER_NAMES)
    if count != expected:
        failures.append(f"Checked {count} scalar parameters, expected {expected}")
    inactive = np.all(module.forward(model, tokens)["head_pre"][positions] <= 0, axis=0)
    if np.any(grads["W_hidden"][:, inactive] != 0) or np.any(grads["b_hidden"][inactive] != 0):
        failures.append("Inactive/exact-zero ReLU units must use derivative zero")
    return count, len(used_tokens), max_error, worst, failures, kinks


def main():
    module = load_training_module()
    model = module.training_model(json.loads((HERE / "toy.json").read_text(encoding="utf-8")))
    saved = json.loads((HERE / "toy3.json").read_text(encoding="utf-8"))
    for key, value in model.items():
        if saved.get(key) != value:
            raise RuntimeError(f"Part III must start from the exact Part II model: {key} differs")
    raw = module.build_training(model)
    computed = module.serialize_training(raw)
    numbers, saved_error, failures = compare_saved(computed, saved.get("training"))
    print(f"Saved training: {numbers} numeric entries; max absolute error {saved_error:.3g}")
    diagnostic = saved["training"]["finite_difference"]
    if len(diagnostic["checks"]) != 3 or not 0 < diagnostic["max_abs_error"] < 1e-7:
        failures.append("Saved finite-difference diagnostics were rounded away or have the wrong check count")
    for check in diagnostic["checks"]:
        if check["epsilon"] != module.FD_EPS or check["analytic"] == check["numeric"]:
            failures.append("Saved finite-difference estimates must retain full precision and epsilon")
        if abs(abs(check["analytic"] - check["numeric"]) - check["abs_error"]) > 1e-15:
            failures.append("Saved absolute error does not match the two saved estimates")

    # Only the prefix's first T position rows participate. The full stored
    # table is larger, so all unused rows have zero gradient on this example.
    unused_checks = 0
    work = copy.deepcopy(model)
    for row in range(len(raw["sentence"]), len(model["pos_emb"])):
        for col, original in enumerate(model["pos_emb"][row]):
            work["pos_emb"][row][col] = original + EPSILON
            plus = final_token_loss(module, work, raw["sentence"], raw["target"])
            work["pos_emb"][row][col] = original - EPSILON
            minus = final_token_loss(module, work, raw["sentence"], raw["target"])
            work["pos_emb"][row][col] = original
            unused_checks += 1
            if plus != minus:
                failures.append(f"Unused position [{row},{col}] affects this prefix's loss")
    for step in raw["single"]["steps"].values():
        updated = module.apply_step(model, raw["sentence"], raw["single"]["gradients"], step["eta"])
        if updated["pos_emb"][len(raw["sentence"]):] != model["pos_emb"][len(raw["sentence"]):]:
            failures.append("An optimizer step changed an unused position row")

    tokens, target = computed["sentence"], computed["target"]
    count, token_rows, gradient_error, worst, gradient_failures, kinks = check_gradients(module, model, tokens, target)
    failures.extend(gradient_failures)
    print(f"Gradients: {count} scalars ({token_rows} token rows, {len(tokens)} position rows, all projections/head)")
    print(f"Central differences: {count - len(kinks)} smooth coordinates; {len(kinks)} ReLU-kink directions excluded; max error {gradient_error:.3g} at {worst}")
    # The hand-designed Part II model intentionally has exact zeros. Keep that
    # model unchanged and check all reverse-pass paths on a separate smooth copy.
    probe = copy.deepcopy(model)
    probe["b_hidden"] = [x + .013 for x in probe["b_hidden"]]
    probe_count, _, probe_error, probe_worst, probe_failures, probe_kinks = check_gradients(module, probe, tokens, target)
    failures.extend(probe_failures)
    if probe_kinks:
        failures.append(f"Smooth probe still crosses ReLU kinks: {probe_kinks}")
    print(f"Smooth probe: all {probe_count} scalar gradients; max error {probe_error:.3g} at {probe_worst}")
    par_count, _, par_error, par_worst, par_failures, par_kinks = check_gradients(module, probe, tokens, target, parallel=True)
    failures.extend(par_failures)
    if par_kinks:
        failures.append(f"Parallel smooth probe crosses ReLU kinks: {par_kinks}")
    print(f"Parallel mean-loss smooth probe: all {par_count} gradients; max error {par_error:.3g} at {par_worst}")
    print(f"Unused positions: {unused_checks} scalar perturbations leave loss unchanged; optimizer leaves those rows unchanged")
    if failures:
        for failure in failures[:12]:
            print(f"FAIL: {failure}", file=sys.stderr)
        if len(failures) > 12:
            print(f"... {len(failures) - 12} additional mismatches", file=sys.stderr)
        return 1
    print("PASS: exact Part II initialization, saved training, smooth gradients, and zero-ReLU convention; no files written")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
