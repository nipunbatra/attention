"""Check the saved experiment's provenance, arithmetic and browser payloads."""
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parents[1] / 'word-lab'


def read(path):
    return json.loads(path.read_text())


def test_run_protocol_and_aggregates():
    result = read(ROOT / 'artifacts/heads/comparison.json')
    for name, digest in result['protocol']['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    for kind, runs in result['runs'].items():
        assert [r['seed'] for r in runs] == [11, 29, 47]
        for run in runs:
            assert run['tokens_seen'] == 6000 * 512
            assert 1 <= run['selected_step'] <= 6000
            assert math.isclose(math.exp(run['test_loss']), run['test_perplexity'])
        for metric, summary in result['aggregate'][kind].items():
            values = [r[metric] for r in runs]
            assert math.isclose(summary['mean'], statistics.mean(values))
            assert math.isclose(summary['sample_std'], statistics.stdev(values))


def test_export_hashes_and_prompt_ids():
    # Browser artifacts are only present in the full lecture repository, not the ZIP.
    if not LAB.exists():
        return
    metadata = read(LAB / 'models/metadata.json')
    assert metadata['models']['attention']['parameters'] == metadata['models']['multihead']['parameters']
    for kind, model in metadata['models'].items():
        path = LAB / model['file']
        assert path.stat().st_size == model['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == model['sha256']
        checkpoint = ROOT / f'artifacts/heads/{kind}_seed11.npz'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == model['checkpoint_sha256']
        assert model['onnx_cpu_max_absolute_error'] < 2e-4
    itos = read(LAB / 'models/vocab.json')['itos']
    stoi = {word: i for i, word in enumerate(itos)}
    prompts = read(LAB / 'examples.json')['prompts']
    assert len([p for p in prompts if p.get('split') == 'train']) == 2
    assert len([p for p in prompts if p.get('split') == 'test']) == 1
    for prompt in prompts:
        assert prompt['token_ids'] == [stoi.get(t, 3) for t in prompt['tokens']]
    saved = read(LAB / 'saved-examples.json')['examples']
    assert len(saved) == 3 * len(prompts)
    for sample in saved:
        assert len(sample['generated_ids']) <= 24
        assert all(4 <= t < 4000 for t in sample['generated_ids'])
