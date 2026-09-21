#!/usr/bin/env python3
"""Download, checksum, split, tokenize, and audit a bounded TinyStories sample."""

from __future__ import annotations

import argparse
from pathlib import Path

from wordlm import PROFILE_SPECS, build_corpus, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILE_SPECS), default="smoke")
    parser.add_argument("--data-dir", type=Path, default=Path("work/data"))
    args = parser.parse_args()

    corpus = build_corpus(args.data_dir, args.profile)
    audit_path = args.data_dir / args.profile / "corpus_audit.json"
    write_json(
        audit_path,
        {
            "source": corpus.manifest,
            "audit": corpus.audit,
            "special_tokens": list(corpus.vocab.itos[:4]),
            "most_common_train_tokens": [
                {"token": token, "count": corpus.vocab.counts[token]}
                for token in corpus.vocab.itos[4:24]
            ],
        },
    )
    print(f"Prepared {args.profile!r} profile in {args.data_dir / args.profile}")
    print(f"Documents: {corpus.audit['documents']}")
    print(f"Vocabulary: {len(corpus.vocab.itos):,} tokens")
    print(
        "Held-out OOV: "
        f"validation={corpus.audit['oov']['validation']['rate']:.2%}, "
        f"test={corpus.audit['oov']['test']['rate']:.2%}"
    )
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
