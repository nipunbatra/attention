"""Small text-only retrieval baseline. See ../PRACTICAL.md for the input format."""
import argparse
import json
from pathlib import Path
import numpy as np

MODEL = "google/embeddinggemma-2"


def unit_rows(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite two-dimensional feature matrix")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("Cannot normalize a zero vector")
    return x / norms


def load_model(revision):
    import torch
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        MODEL, revision=revision,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    ix = sub.add_parser("index")
    ix.add_argument("pieces", type=Path, help="JSONL: one text piece and source location per line")
    ix.add_argument("directory", type=Path, help="A new output directory")
    ix.add_argument("--revision", default="main", help="Resolved to a commit and saved")
    query = sub.add_parser("query")
    query.add_argument("directory", type=Path)
    query.add_argument("question")
    query.add_argument("--top-k", type=int, default=5)
    sub.add_parser("check", help="Check array arithmetic only; no model download")
    args = p.parse_args()
    if args.command == "check":
        x = unit_rows([[0.8, 0.6, 0], [0, 0.6, 0.8], [0, 0, 1]])
        q = unit_rows([[0.6, 0.8, 0]])[0]
        assert np.allclose(x @ q, [0.96, 0.48, 0])
        print("PASS: normalization, scores and ordering; illustrative arrays only")
        return
    if args.command == "index":
        rows = [json.loads(line) for line in args.pieces.read_text().splitlines() if line.strip()]
        if not rows or any(not isinstance(r.get("text"), str) or not r["text"].strip()
                           or not r.get("file") or not r.get("piece_id") for r in rows):
            raise ValueError("Each piece needs nonempty text, file and piece_id")
        if len({r["piece_id"] for r in rows}) != len(rows):
            raise ValueError("piece_id must be unique")
        if args.directory.exists():
            raise ValueError("Use a new directory so the previous index remains available")
        from huggingface_hub import HfApi
        revision = HfApi().model_info(MODEL, revision=args.revision).sha
        model = load_model(revision)
        x = unit_rows(model.encode([r["text"] for r in rows], prompt_name="Document",
                                  normalize_embeddings=True, show_progress_bar=True))
        if x.shape != (len(rows), 768):
            raise ValueError(f"Expected {(len(rows), 768)}, got {x.shape}")
        args.directory.mkdir(parents=True)
        np.save(args.directory / "vectors.npy", x)
        (args.directory / "index.json").write_text(json.dumps({
            "model": MODEL, "revision": revision, "dimensions": 768,
            "document_prompt": "Document", "query_prompt": "SearchQuery",
            "normalized": True, "pieces": rows,
        }, indent=2))
        print(f"Saved {x.shape} vectors and {len(rows)} source records to {args.directory}")
    else:
        data = json.loads((args.directory / "index.json").read_text())
        x = np.load(args.directory / "vectors.npy", allow_pickle=False)
        if data["model"] != MODEL or x.shape != (len(data["pieces"]), 768):
            raise ValueError("Incompatible index or metadata")
        x = unit_rows(x)
        if args.top_k < 1:
            raise ValueError("top-k must be positive")
        q = unit_rows(load_model(data["revision"]).encode(
            [args.question], prompt_name=data["query_prompt"], normalize_embeddings=True))[0]
        scores = x @ q
        for rank, row in enumerate(np.argsort(-scores)[:args.top_k], 1):
            item = data["pieces"][int(row)]
            print(json.dumps({"rank": rank, "cosine": round(float(scores[row]), 4),
                              "source": item}, ensure_ascii=False))


if __name__ == "__main__":
    main()
