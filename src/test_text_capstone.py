"""Verify the restored TinyStories slides against data and the actual model."""
import json
import math
import statistics
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'notebooks/wordlm'))
from multihead import MultiHeadAttentionLM
sys.path.insert(0, str(REPO/'src'))
from tinystories_setup import SETUP_STEPS
from tinystories_map import SETUP_CODE, SETUP_FOCUS, MODEL_FOCUS
from slow_walkthrough import STAGES, initial_namespace, render_figure


def test_capstone_data_and_forward_shapes():
    evidence = json.loads((REPO/'figures/multihead/capstone-evidence.json').read_text())
    vocab = json.loads((REPO/'word-lab/models/vocab.json').read_text())['itos']
    X = torch.tensor([w['ids'] for w in evidence['windows']])
    y = torch.tensor([w['target_id'] for w in evidence['windows']])
    assert X.shape == (2,64) and y.shape == (2,)
    for row, example in zip(X, evidence['windows']):
        assert [vocab[i] for i in row.tolist() if i] == example['tokens']
        assert vocab[example['target_id']] == example['target']
    assert [vocab[i] for i in y.tolist()] == ['a','red']
    torch.manual_seed(11)
    model = MultiHeadAttentionLM(len(vocab),64,64,256,4)
    detail = model.forward_details(X)
    for key, shape in {'E':(2,64,64),'Q':(2,4,64,16),'K':(2,4,64,16),
                       'V':(2,4,64,16),'weights':(2,4,64,64),
                       'messages':(2,64,64),'contextual':(2,64,64),
                       'logits':(2,4000)}.items():
        assert detail[key].shape == shape
    torch.testing.assert_close(detail['logits'], model(X), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(detail['weights'].sum(-1),torch.ones(2,4,64))
    for b in range(2):
        assert (detail['weights'][b,:,-1,X[b]==0] == 0).all()
    # Execute the short snippets actually printed in class, using the same model.
    steps = {s['key']:s for s in json.loads((REPO/'figures/multihead/manifest.json').read_text())}
    scope = dict(B=2, T=64, X=X, y=y, model=model, F=F, torch=torch, math=math,
                 token_embedding=model.token_embedding, position_embedding=model.position_embedding,
                 W_Q=model.W_Q, W_K=model.W_K, W_V=model.W_V, W_O=model.W_O,
                 split_heads=model.split_heads, vocab_head=model.vocab_head,
                 hidden_layer=model.hidden_layer, relu=F.relu, cross_entropy=F.cross_entropy,
                 A=detail['weights'], optimizer=torch.optim.AdamW(model.parameters(),lr=.001))
    for key in ['s04-cap-lookup','s04-cap-position','s04-cap-embed','s04-cap-scores','s04-cap-weights','s04-cap-mix','s04-cap-join','s04-cap-predict']:
        exec(steps[key]['code'],scope)
    torch.testing.assert_close(scope['logits'],detail['logits'])
    before = model.W_Q.weight.detach().clone()
    exec(steps['s04-cap-train']['code'],scope)
    assert not torch.equal(before,model.W_Q.weight)
    assert torch.isfinite(scope['loss'])
    # Generation excerpts use the same saved tokenizer/IDs and a frozen model.
    from wordlm import tokenize
    scope.update(model=model, real_stoi={t:i for i,t in enumerate(vocab)}, tokenize=tokenize)
    before = {k:v.detach().clone() for k,v in model.state_dict().items()}
    for key in ['s04-cap-prompt', 's04-cap-decode', 's04-cap-generate']:
        exec(steps[key]['code'], scope)
    assert scope['X'][0, -3:].tolist() == [1,23,110]
    assert scope['next_id'] not in {0,1,3}
    for k,v in model.state_dict().items():
        assert torch.equal(v,before[k])


def test_restored_setup_reuses_the_original_computed_figures():
    steps = {s['key']:s for s in json.loads((REPO/'figures/multihead/manifest.json').read_text())}
    ns = initial_namespace()
    for source in STAGES:
        exec(source['code'], ns)
        if source['id'] in SETUP_STEPS:
            key = source['id']
            original = (REPO/'figures/wordlm-pipeline/steps'/f'{key}.svg').read_text()
            assert render_figure(source, ns) == original
            assert (REPO/'figures/multihead'/f's03-data-{key}.svg').read_text() == original
            assert f's03-data-{key}' in steps
        if source['id'] == 'batches':
            break
    assert ns['ids'] == [1,8,7,5,9,6,4,2]
    assert ns['all_X'].shape == (7,4) and ns['all_y'].tolist() == [8,7,5,9,6,4,2]
    assert ns['X'].tolist() == [[0,1,8,7],[1,8,7,5]]
    assert ns['y'].tolist() == [5,9]
    results = json.loads((REPO/'word-lab/models/comparison.json').read_text())
    assert ns['audit']['documents'] == results['audit']['documents']
    assert ns['audit']['oov'] == results['audit']['oov']
    assert ns['window_counts']['train'] == 969160
    vocab = json.loads((REPO/'word-lab/models/vocab.json').read_text())['itos']
    assert [vocab.index(t) for t in ['<BOS>','lily','found']] == [1,23,110]
    assert len(SETUP_STEPS) == 25


def test_every_setup_checkpoint_has_executable_matching_code():
    steps = {s['key']:s for s in json.loads((REPO/'figures/multihead/manifest.json').read_text())}
    ns = initial_namespace()
    for key in SETUP_STEPS:
        checkpoint = steps['s03-map-' + key]
        assert checkpoint['code'] == SETUP_CODE[key]
        exec(checkpoint['code'], ns)
        figure = (REPO/'figures/multihead'/f's03-map-{key}.svg').read_text()
        assert f'data-focus="{" ".join(SETUP_FOCUS[key])}"' in figure
    assert ns['all_X'].tolist() == [[0,0,0,1],[0,0,1,8],[0,1,8,7],[1,8,7,5],[8,7,5,9],[7,5,9,6],[5,9,6,4]]
    assert ns['all_y'].tolist() == [8,7,5,9,6,4,2]
    assert ns['X'].tolist() == [[0,1,8,7],[1,8,7,5]]
    assert ns['window_counts']['train'] == 969160
    assert ns['batch_sizes'] == [2,2,2,1]
    vocab = json.loads((REPO/'word-lab/models/vocab.json').read_text())['itos']
    ns['real_stoi'] = {t:i for i,t in enumerate(vocab)}
    for key in ['scale', 'real-windows']:
        exec(steps['s03-map-' + key]['code'], ns)
    evidence = json.loads((REPO/'figures/multihead/capstone-evidence.json').read_text())
    assert ns['X'].tolist() == [w['ids'] for w in evidence['windows']]
    assert ns['y'].tolist() == [w['target_id'] for w in evidence['windows']]
    exec(steps['s04-cap-model']['code'], ns)
    assert ns['model'].heads == 4 and ns['model'].head_width == 16
    assert isinstance(ns['model'].position_embedding, torch.nn.Embedding)
    assert ns['model'].position_embedding.weight.shape == (64, 64)
    assert ns['model'].position_embedding.weight.requires_grad
    position_step = steps['s04-cap-position']
    assert 'learned position' in position_step['title']
    assert 'not sinusoidal' in position_step['code']
    assert 'Trained together with the token embeddings' in position_step['code']
    assert ns['model'](ns['X']).shape == (2,4000)
    for key, focus in MODEL_FOCUS.items():
        figure = (REPO/'figures/multihead'/f's04-cap-{key}.svg').read_text()
        assert f'data-focus="{" ".join(focus)}"' in figure
        assert steps['s04-cap-'+key]['code'].strip()


def test_results_and_samples_are_saved_evidence():
    evidence = json.loads((REPO/'figures/multihead/capstone-evidence.json').read_text())
    results = json.loads((REPO/'word-lab/models/comparison.json').read_text())
    source = json.loads((REPO/'notebooks/wordlm/artifacts/heads/comparison.json').read_text())
    assert source == results
    assert evidence['protocol'] == results['protocol']
    for i, kind in enumerate(['mlp','attention','multihead']):
        run = results['runs'][kind][0]
        avg = results['aggregate'][kind]
        assert str(run['parameter_count']) == evidence['costs'][i][1].replace(',','')
        assert evidence['scores'][i][1].startswith(f'{avg["test_loss"]["mean"]:.3f}')
        assert evidence['scores'][i][2].startswith(f'{avg["test_perplexity"]["mean"]:.2f}')
        for metric in ['test_loss', 'test_perplexity']:
            values = [r[metric] for r in results['runs'][kind]]
            assert math.isclose(avg[metric]['mean'], statistics.mean(values))
            assert math.isclose(avg[metric]['sample_std'], statistics.stdev(values))
        for r in results['runs'][kind]:
            assert math.isclose(math.exp(r['test_loss']),r['test_perplexity'])
    steps = {s['key']:s for s in json.loads((REPO/'figures/multihead/manifest.json').read_text())}
    result_step = steps['s05-cap-scores']
    for phrase in ['Perplexity = exp(test loss)', 'probability ¼', 'perplexity is 4',
                   '11, 29, 47', 'sample standard deviation', '28.78 ± 0.61']:
        assert phrase in result_step['body']
    for run in results['runs']['multihead']:
        assert f'{run["test_perplexity"]:.2f}' in result_step['body']
    assert 'not a confidence interval' in result_step['companion']
    assert math.isclose(math.exp(-math.log(.25)), 4)
    saved = json.loads((REPO/'word-lab/saved-examples.json').read_text())
    assert evidence['samples'] == [s for s in saved['examples'] if s['prompt_id']=='heldout']
    assert results['aggregate']['multihead']['test_loss']['mean'] < results['aggregate']['attention']['test_loss']['mean']
    assert results['aggregate']['multihead']['runtime_seconds']['mean'] > results['aggregate']['attention']['runtime_seconds']['mean']
