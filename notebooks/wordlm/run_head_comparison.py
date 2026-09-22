"""Repeat the fixed protocol for MLP, one head and four heads on one device."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import math
import platform
import statistics
import torch
from multihead import MultiHeadAttentionLM
from wordlm import (build_corpus, make_all_windows, seed_everything, FixedWindowMLP,
                    CausalAttentionLM, TrainConfig, train_model, evaluate_loss,
                    model_state_to_npz, read_json, write_json, sha256_file,
                    resolve_device, environment_summary)

ROOT = Path(__file__).resolve().parent


def build_model(kind, vocab_size=4000, seed=11):
    seed_everything(seed)
    if kind == 'mlp':
        return FixedWindowMLP(vocab_size, 64, 64, 256)
    if kind == 'attention':
        return CausalAttentionLM(vocab_size, 64, 64, 64, 64, 256)
    return MultiHeadAttentionLM(vocab_size, 64, 64, 256, heads=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts' / 'heads')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--steps', type=int, default=6000)
    args = parser.parse_args()
    device = resolve_device(args.device)
    old = read_json(ROOT / 'artifacts' / 'benchmark.json')
    corpus = build_corpus(args.data_dir, 'dgx')
    assert corpus.manifest['raw_sha256'] == old['source']['raw_sha256']
    assert corpus.vocab.itos == read_json(ROOT / 'artifacts' / 'vocab.json')['itos']
    windows = make_all_windows(corpus, 64)
    protocol = dict(context_length=64, embedding_width=64, hidden_width=256,
                    steps=args.steps, batch_size=512, seeds=[11, 29, 47],
                    heads={'mlp': 0, 'attention': 1, 'multihead': 4},
                    positional_encoding='learned absolute position rows, added to token rows',
                    hyperparameters='Frozen baseline settings; four heads reuses one-head settings. No test tuning.',
                    raw_sha256=corpus.manifest['raw_sha256'],
                    train_config_mlp=old['configs']['mlp'],
                    train_config_attention=old['configs']['attention'],
                    source_hashes={p: sha256_file(ROOT / p) for p in ['wordlm.py', 'multihead.py']})
    result_path = args.output_dir / 'comparison.json'
    payload = dict(protocol=protocol, environment=environment_summary(),
                   machine=platform.processor(), created_utc=datetime.now(timezone.utc).isoformat(),
                   source=corpus.manifest, audit=corpus.audit, runs={k: [] for k in protocol['heads']})
    if result_path.exists():
        payload = read_json(result_path)
        assert payload['protocol'] == protocol, 'Resume settings differ'
    # All choices are frozen before evaluating any test target.
    for kind in ['mlp', 'attention', 'multihead']:
        cfg = old['configs']['mlp' if kind == 'mlp' else 'attention']
        for seed in protocol['seeds']:
            if any(r['seed'] == seed for r in payload['runs'][kind]):
                continue
            print(f'Training {kind}, seed {seed}, device {device}', flush=True)
            model = build_model(kind, len(corpus.vocab.itos), seed)
            config = TrainConfig(steps=args.steps, batch_size=512, seed=seed,
                                 learning_rate=cfg['learning_rate'], weight_decay=cfg['weight_decay'],
                                 eval_every=max(1, args.steps // 8))
            result, _ = train_model(model, *windows['train'][:2], *windows['validation'][:2], config, device=device)
            result.pop('approximate_forward_multiplies', None)  # separate cost lesson covers this
            result['test_loss'] = evaluate_loss(model, *windows['test'][:2], device=device)
            result['test_perplexity'] = math.exp(result['test_loss'])
            result['seed'] = seed
            payload['runs'][kind].append(result)
            if seed == 11:
                model_state_to_npz(args.output_dir / f'{kind}_seed11.npz', model,
                                   dict(kind=kind, seed=seed, selected_step=result['selected_step'],
                                        protocol=protocol, test_loss=result['test_loss']))
            write_json(result_path, payload)
            print(f"Done: test CE {result['test_loss']:.4f}, {result['runtime_seconds']:.1f}s", flush=True)
    payload['aggregate'] = {}
    for kind, runs in payload['runs'].items():
        payload['aggregate'][kind] = {metric: dict(mean=statistics.mean(r[metric] for r in runs),
                                                  sample_std=statistics.stdev(r[metric] for r in runs))
                                      for metric in ['test_loss', 'test_perplexity', 'runtime_seconds']}
    write_json(result_path, payload)
    print(payload['aggregate'], flush=True)


if __name__ == '__main__':
    main()
