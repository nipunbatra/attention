"""Proposed eight-word teaching reader, not the speech app's unpublished reader.

Consumes cached real 768D embeddings; it does not extract audio features.
Run --check for a shape/gradient test, not a model-quality experiment.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

WORDS = ["yes", "no", "up", "down", "left", "right", "stop", "go"]


def make_reader():
    return nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 768))


def features(array, name):
    a = np.asarray(array, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 768 or not len(a) or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected nonempty finite (N, 768) array")
    if (np.linalg.norm(a, axis=1) < 1e-8).any():
        raise ValueError(f"{name}: zero feature vector")
    return F.normalize(torch.from_numpy(a), dim=-1)


def read_cache(path):
    z = np.load(path, allow_pickle=False)
    if z["words"].tolist() != WORDS:
        raise ValueError(f"Use this exact label order: {WORDS}")
    e = features(z["E"], "E")
    if e.shape != (8, 768):
        raise ValueError("Expected eight candidate text vectors")
    splits, speakers = {}, {}
    for split in ("train", "val", "test"):
        x = features(z[f"x_{split}"], f"x_{split}")
        a = z[f"y_{split}"]
        ids = z[f"speakers_{split}"]
        if a.shape != (len(x),) or not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"{split}: one integer label required per feature row")
        if set(a.tolist()) != set(range(8)):
            raise ValueError(f"{split}: labels must cover all eight classes")
        if ids.shape != (len(x),) or ids.dtype.kind not in "US" or any(not s for s in ids.tolist()):
            raise ValueError(f"{split}: one nonempty string speaker ID required per row")
        speakers[split] = set(ids.tolist())
        splits[split] = (x, torch.from_numpy(a.astype(np.int64)))
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if speakers[a] & speakers[b]:
            raise ValueError(f"Speaker leakage between {a} and {b}")
    metadata = json.loads(str(z["metadata_json"].item()))
    for key in ("model", "revision", "audio_preprocessing", "text_prompt", "dataset"):
        if not metadata.get(key):
            raise ValueError(f"Missing cache provenance: {key}")
    return splits, e, metadata


@torch.no_grad()
def evaluate(reader, x, y, e, temperature):
    total_loss, correct = 0.0, 0
    if reader is not None:
        reader.eval()
    for xb, yb in DataLoader(TensorDataset(x, y), batch_size=256):
        r = xb if reader is None else F.normalize(reader(xb), dim=-1)
        scores = r @ e.T / temperature
        total_loss += F.cross_entropy(scores, yb, reduction="sum").item()
        correct += (scores.argmax(-1) == yb).sum().item()
    return {"loss": total_loss / len(y), "accuracy": correct / len(y), "examples": len(y)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path, nargs="?")
    p.add_argument("--output", type=Path, default=Path("reader-result"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    torch.manual_seed(7)
    reader = make_reader()
    if args.check:
        x, y = torch.randn(4, 768), torch.tensor([0, 1, 2, 3])
        e = F.normalize(torch.randn(8, 768), dim=-1)
        r = F.normalize(reader(x), dim=-1)
        logits = r @ e.T / 0.1
        loss = F.cross_entropy(logits, y)
        loss.backward()
        assert r.shape == (4, 768) and logits.shape == (4, 8)
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in reader.parameters())
        assert e.grad is None and sum(t.numel() for t in reader.parameters()) == 197504
        print("PASS: shapes, 197504 reader parameters, finite gradients; random tensors only")
        return
    if args.cache is None or args.epochs < 1:
        p.error("Provide a real feature cache and a positive epoch count, or use --check")
    if args.output.exists():
        p.error("Choose a new output directory to preserve previous results")
    splits, e, metadata = read_cache(args.cache)
    loader = DataLoader(TensorDataset(*splits["train"]), batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=1e-3)
    best_loss, best, history = float("inf"), None, []
    for epoch in range(args.epochs):
        reader.train()
        for x, y in loader:
            r = F.normalize(reader(x), dim=-1)
            loss = F.cross_entropy(r @ e.T / 0.1, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val = evaluate(reader, *splits["val"], e, 0.1)
        history.append({"epoch": epoch + 1, "validation": val})
        print(json.dumps(history[-1]))
        if val["loss"] < best_loss:
            best_loss, best = val["loss"], copy.deepcopy(reader.state_dict())
    reader.load_state_dict(best)
    report = {"cache": metadata, "history": history,
              "test": evaluate(reader, *splits["test"], e, 0.1),
              "zero_shot_test": evaluate(None, *splits["test"], e, 0.1)}
    args.output.mkdir(parents=True)
    torch.save({"reader": best, "word_vectors": e, "words": WORDS,
                "temperature": 0.1, "cache_provenance": metadata}, args.output / "reader.pt")
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"test": report["test"], "zero_shot_test": report["zero_shot_test"]}))


if __name__ == "__main__":
    main()
