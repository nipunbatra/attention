#!/usr/bin/env python3
"""Run controlled MLP sweeps and a fair MLP-vs-attention benchmark."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from wordlm import (
    CausalAttentionLM,
    FixedWindowMLP,
    PROFILE_SPECS,
    TrainConfig,
    attention_invariants,
    build_corpus,
    count_parameters,
    environment_summary,
    evaluate_loss,
    make_all_windows,
    model_state_to_npz,
    read_json,
    resolve_device,
    seed_everything,
    train_model,
    write_json,
)


BASELINE: dict[str, Any] = {
    "context_len": 16,
    "d_embed": 64,
    "hidden": 256,
    "learning_rate": 3e-3,
    "weight_decay": 1e-4,
}

# The architecture comparison deliberately uses a longer shared input.  This
# is where fixed concatenation becomes costly and a content-weighted summary is
# meant to help; both models still receive the exact same 64 tokens and targets.
COMPARISON_CONTEXT_LEN = 64


def sweep_definitions() -> list[dict[str, Any]]:
    choices = {
        "context_len": [8, 16, 32, 64],
        "d_embed": [32, 64, 96],
        "hidden": [128, 256, 384],
        "learning_rate": [1e-3, 3e-3, 6e-3],
        "weight_decay": [0.0, 1e-4, 1e-3],
    }
    definitions: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for axis, values in choices.items():
        for value in values:
            config = copy.deepcopy(BASELINE)
            config[axis] = value
            key = tuple(sorted(config.items()))
            if key in seen:
                continue
            seen.add(key)
            config["sweep_axis"] = axis if value != BASELINE[axis] else "baseline"
            config["sweep_value"] = value
            definitions.append(config)
    return definitions


def build_mlp(vocab_size: int, pad_id: int, config: dict[str, Any], seed: int) -> FixedWindowMLP:
    seed_everything(seed)
    return FixedWindowMLP(
        vocab_size,
        int(config["context_len"]),
        int(config["d_embed"]),
        int(config["hidden"]),
        pad_id,
    )


def build_attention(
    vocab_size: int, pad_id: int, config: dict[str, Any], seed: int
) -> CausalAttentionLM:
    seed_everything(seed)
    d_model = int(config["d_embed"])
    return CausalAttentionLM(
        vocab_size,
        int(config["context_len"]),
        d_model=d_model,
        d_k=d_model,
        d_v=d_model,
        hidden=int(config["hidden"]),
        pad_id=pad_id,
    )


def train_one(
    kind: str,
    corpus,
    windows,
    config: dict[str, Any],
    train_config: TrainConfig,
    device: torch.device,
    capture_embeddings: bool = False,
):
    model = (
        build_mlp(len(corpus.vocab.itos), corpus.vocab.pad_id, config, train_config.seed)
        if kind == "mlp"
        else build_attention(len(corpus.vocab.itos), corpus.vocab.pad_id, config, train_config.seed)
    )
    result, embedding_trace = train_model(
        model,
        windows["train"][0],
        windows["train"][1],
        windows["validation"][0],
        windows["validation"][1],
        train_config,
        device=device,
        capture_embeddings=capture_embeddings,
    )
    result.update({"kind": kind, "model_config": copy.deepcopy(config)})
    return model, result, embedding_trace


def run_sweep(args, corpus, device: torch.device) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    windows_cache: dict[int, Any] = {}
    for number, definition in enumerate(sweep_definitions(), start=1):
        context_len = int(definition["context_len"])
        windows = windows_cache.setdefault(context_len, make_all_windows(corpus, context_len))
        train_config = TrainConfig(
            steps=args.sweep_steps,
            batch_size=args.batch_size,
            learning_rate=float(definition["learning_rate"]),
            weight_decay=float(definition["weight_decay"]),
            seed=args.selection_seed,
            eval_every=max(1, args.sweep_steps // 6),
        )
        print(
            f"[sweep {number:02d}/{len(sweep_definitions()):02d}] "
            f"{definition['sweep_axis']}={definition['sweep_value']}"
        )
        _, result, _ = train_one("mlp", corpus, windows, definition, train_config, device)
        records.append(result)

    primary = [record for record in records if record["model_config"]["context_len"] == 16]
    selected = min(primary, key=lambda record: record["validation_loss"])
    payload = {
        "protocol": {
            "selection_metric": "token-weighted validation cross-entropy",
            "test_set_used": False,
            "one_variable_at_a_time": True,
            "baseline": BASELINE,
            "selection_seed": args.selection_seed,
            "steps_per_run": args.sweep_steps,
            "batch_size": args.batch_size,
        },
        "selected_primary_mlp_config": selected["model_config"],
        "records": records,
    }
    write_json(args.output_dir / "mlp_sweep.json", payload)

    flat_rows = []
    for record in records:
        row = {
            **record["model_config"],
            "validation_loss": record["validation_loss"],
            "validation_perplexity": record["validation_perplexity"],
            "parameter_count": record["parameter_count"],
            "tokens_seen": record["tokens_seen"],
            "runtime_seconds": record["runtime_seconds"],
        }
        flat_rows.append(row)
    with (args.output_dir / "mlp_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    return payload


def tune_comparison_models(args, corpus, device: torch.device) -> dict[str, Any]:
    """Tune both models fairly for the shared long-context comparison."""
    windows = make_all_windows(corpus, COMPARISON_CONTEXT_LEN)
    settings = {
        "mlp": [
            (1e-3, 1e-4),
            (2e-3, 1e-4),
            (3e-3, 1e-4),
            (3e-3, 1e-3),
            (3e-3, 1e-2),
        ],
        "attention": [
            (1.5e-3, 1e-4),
            (2e-3, 1e-4),
            (3e-3, 1e-4),
            (4e-3, 1e-4),
            (3e-3, 1e-3),
        ],
    }
    records: dict[str, list[dict[str, Any]]] = {"mlp": [], "attention": []}
    for kind, choices in settings.items():
        for number, (learning_rate, weight_decay) in enumerate(choices, start=1):
            candidate = copy.deepcopy(BASELINE)
            candidate.update(
                {
                    "context_len": COMPARISON_CONTEXT_LEN,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "sweep_axis": "comparison_tune",
                    "sweep_value": f"lr={learning_rate},wd={weight_decay}",
                }
            )
            print(
                f"[{kind} comparison tune {number:02d}/{len(choices):02d}] "
                f"lr={learning_rate}, wd={weight_decay}"
            )
            train_config = TrainConfig(
                steps=args.tune_steps,
                batch_size=args.batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                seed=args.selection_seed,
                eval_every=max(1, args.tune_steps // 4),
            )
            _, result, _ = train_one(kind, corpus, windows, candidate, train_config, device)
            records[kind].append(result)
    selected = {
        kind: min(kind_records, key=lambda record: record["validation_loss"])["model_config"]
        for kind, kind_records in records.items()
    }
    payload = {
        "protocol": {
            "selection_metric": "token-weighted validation cross-entropy",
            "test_set_used": False,
            "same_context_length": COMPARISON_CONTEXT_LEN,
            "same_embedding_and_hidden_widths": True,
            "same_steps_batch_size_and_seed": True,
            "selection_seed": args.selection_seed,
            "steps_per_run": args.tune_steps,
            "batch_size": args.batch_size,
        },
        "selected_configs": selected,
        "records": records,
    }
    write_json(args.output_dir / "comparison_tuning.json", payload)
    return payload


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {}
    for name in [
        "validation_loss",
        "validation_perplexity",
        "test_loss",
        "test_perplexity",
        "runtime_seconds",
    ]:
        values = [float(record[name]) for record in records]
        metrics[name] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return metrics


def run_benchmark(
    args,
    corpus,
    device: torch.device,
    selected_mlp: dict[str, Any],
    selected_attention: dict[str, Any],
) -> dict[str, Any]:
    if int(selected_mlp["context_len"]) != int(selected_attention["context_len"]):
        raise RuntimeError("Primary comparison must use the same context length")
    context_len = int(selected_mlp["context_len"])
    windows = make_all_windows(corpus, context_len)
    expected_test_targets = sum(len(story.tokens) + 1 for story in corpus.stories["test"])
    if len(windows["test"][1]) != expected_test_targets:
        raise AssertionError("Test windows do not contain exactly one target per story token plus EOS")

    seeds = [11, 29, 47][: args.seeds]
    resume_settings = {
        "profile": args.profile,
        "context_len": context_len,
        "steps": args.benchmark_steps,
        "batch_size": args.batch_size,
        "seeds": seeds,
        "mlp_config": selected_mlp,
        "attention_config": selected_attention,
        "raw_sha256": corpus.manifest["raw_sha256"],
        "checkpoint_selection": "lowest validation loss on the fixed selection slice",
    }
    partial_path = args.output_dir / "benchmark_partial.json"
    by_kind: dict[str, list[dict[str, Any]]] = {"mlp": [], "attention": []}
    if partial_path.exists():
        partial = read_json(partial_path)
        if partial.get("settings") == resume_settings:
            by_kind = partial["runs"]
            print(
                "Resuming completed runs:",
                {kind: [run["seed"] for run in runs] for kind, runs in by_kind.items()},
            )
    for kind, config in [("mlp", selected_mlp), ("attention", selected_attention)]:
        for seed in seeds:
            if any(int(run["seed"]) == seed for run in by_kind[kind]):
                print(f"[benchmark] skip completed kind={kind} seed={seed}")
                continue
            print(f"[benchmark] kind={kind} seed={seed}")
            train_config = TrainConfig(
                steps=args.benchmark_steps,
                batch_size=args.batch_size,
                learning_rate=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
                seed=seed,
                eval_every=max(1, args.benchmark_steps // 8),
            )
            model, result, embedding_trace = train_one(
                kind,
                corpus,
                windows,
                config,
                train_config,
                device,
                capture_embeddings=(seed == seeds[0]),
            )
            # The test split is touched only here, after both configurations are frozen.
            result["test_loss"] = evaluate_loss(
                model,
                windows["test"][0],
                windows["test"][1],
                device=device,
            )
            result["test_perplexity"] = math.exp(result["test_loss"])
            result["seed"] = seed
            if kind == "attention":
                result["attention_invariants"] = attention_invariants(
                    model, windows["validation"][0][:2].to(device)
                )
            by_kind[kind].append(result)
            if seed == seeds[0]:
                model_state_to_npz(
                    args.output_dir / f"{kind}_seed{seed}.npz",
                    model,
                    {
                        "kind": kind,
                        "seed": seed,
                        "model_config": config,
                        "train_config": asdict(train_config),
                        "vocab_size": len(corpus.vocab.itos),
                        "dataset_revision": corpus.manifest["dataset_revision"],
                        "raw_sha256": corpus.manifest["raw_sha256"],
                        "selected_step": result["selected_step"],
                        "checkpoint_selection": result["checkpoint_selection"],
                        "validation_loss": result["validation_loss"],
                        "test_loss": result["test_loss"],
                    },
                    embedding_trace,
                )
            write_json(
                partial_path,
                {"settings": resume_settings, "runs": by_kind},
            )

    payload = {
        "protocol": {
            "profile": args.profile,
            "same_documents_tokenizer_vocabulary_targets": True,
            "same_context_length": context_len,
            "same_hidden_width": selected_mlp["hidden"] == selected_attention["hidden"],
            "same_embedding_width": selected_mlp["d_embed"] == selected_attention["d_embed"],
            "same_steps_batch_size_and_seeds": True,
            "one_target_per_window_for_both": True,
            "test_used_after_selection_only": True,
            "seeds": seeds,
            "steps": args.benchmark_steps,
            "batch_size": args.batch_size,
            "tokens_seen_per_run": args.benchmark_steps * args.batch_size,
            "checkpoint_selection": "lowest validation loss on the fixed selection slice",
        },
        "source": corpus.manifest,
        "corpus_audit": corpus.audit,
        "environment": environment_summary(),
        "configs": {"mlp": selected_mlp, "attention": selected_attention},
        "runs": by_kind,
        "aggregate": {kind: _aggregate(records) for kind, records in by_kind.items()},
        "test_windows": len(windows["test"][1]),
    }
    mlp_mean = payload["aggregate"]["mlp"]["test_loss"]["mean"]
    attention_mean = payload["aggregate"]["attention"]["test_loss"]["mean"]
    payload["comparison"] = {
        "attention_minus_mlp_test_loss": attention_mean - mlp_mean,
        "relative_perplexity_change": (
            payload["aggregate"]["attention"]["test_perplexity"]["mean"]
            / payload["aggregate"]["mlp"]["test_perplexity"]["mean"]
            - 1
        ),
        "winner": "attention" if attention_mean < mlp_mean else "mlp",
        "claim_scope": "bounded TinyStories subset and the documented training budget only",
    }
    write_json(args.output_dir / "benchmark.json", payload)
    write_json(
        args.output_dir / "vocab.json",
        {
            "itos": corpus.vocab.itos,
            "counts": corpus.vocab.counts,
            "max_vocab": corpus.vocab.max_vocab,
            "min_freq": corpus.vocab.min_freq,
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILE_SPECS), default="smoke")
    parser.add_argument("--mode", choices=["pilot", "sweep", "benchmark", "all"], default="pilot")
    parser.add_argument("--data-dir", type=Path, default=Path("work/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/results"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--selection-seed", type=int, default=7)
    parser.add_argument("--sweep-steps", type=int, default=240)
    parser.add_argument("--tune-steps", type=int, default=3000)
    parser.add_argument("--benchmark-steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Environment: {environment_summary()}")
    corpus = build_corpus(args.data_dir, args.profile)
    write_json(args.output_dir / "corpus.json", {"source": corpus.manifest, "audit": corpus.audit})
    print(f"Corpus documents: {corpus.audit['documents']}; vocabulary={len(corpus.vocab.itos)}")

    if args.mode == "pilot":
        windows = make_all_windows(corpus, int(BASELINE["context_len"]))
        results = []
        for kind in ["mlp", "attention"]:
            config = copy.deepcopy(BASELINE)
            train_config = TrainConfig(
                steps=args.sweep_steps,
                batch_size=args.batch_size,
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                seed=args.selection_seed,
                eval_every=max(1, args.sweep_steps // 4),
            )
            model, result, _ = train_one(kind, corpus, windows, config, train_config, device)
            result["test_set_used"] = False
            if kind == "attention":
                result["attention_invariants"] = attention_invariants(
                    model, windows["validation"][0][:2].to(device)
                )
            results.append(result)
        write_json(args.output_dir / "pilot.json", {"environment": environment_summary(), "runs": results})
        for result in results:
            print(
                f"{result['kind']}: val_loss={result['validation_loss']:.4f}, "
                f"params={result['parameter_count']:,}, runtime={result['runtime_seconds']:.1f}s"
            )
        return

    if args.mode in {"sweep", "all"}:
        sweep = run_sweep(args, corpus, device)
        comparison_tuning = tune_comparison_models(args, corpus, device)
    else:
        sweep = read_json(args.output_dir / "mlp_sweep.json")
        comparison_tuning = read_json(args.output_dir / "comparison_tuning.json")

    if args.mode in {"benchmark", "all"}:
        payload = run_benchmark(
            args,
            corpus,
            device,
            comparison_tuning["selected_configs"]["mlp"],
            comparison_tuning["selected_configs"]["attention"],
        )
        print(f"Comparison: {payload['comparison']}")


if __name__ == "__main__":
    main()
