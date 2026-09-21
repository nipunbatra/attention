"""Keep the teaching diagrams aligned with the executed window-model contract."""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_maps import graph, pipeline_svg


@pytest.mark.parametrize('kind', ['mlp', 'attention'])
@pytest.mark.parametrize('mode', ['training', 'inference'])
def test_graph_is_complete_and_svg_is_valid(kind, mode):
    g = graph(kind, mode)
    keys = {n['key'] for n in g['nodes']}
    assert len(keys) == len(g['nodes'])
    assert all(e['start'] in keys and e['end'] in keys for e in g['edges'])
    ET.fromstring(pipeline_svg(kind, mode))
    for key in keys:
        ET.fromstring(pipeline_svg(kind, mode, key))
    with pytest.raises(ValueError):
        pipeline_svg(kind, mode, 'nonexistent-stage')


@pytest.mark.parametrize('kind', ['mlp', 'attention'])
def test_targets_only_enter_training_loss_and_eos_precedes_append(kind):
    training = graph(kind, 'training')
    inference = graph(kind, 'inference')
    train_edges = {(e['start'], e['end']) for e in training['edges']}
    infer_edges = {(e['start'], e['end']) for e in inference['edges']}
    assert ('windows', 'loss') in train_edges
    assert ('logits', 'loss') in train_edges
    assert ('backward', 'optimizer') in train_edges
    assert not {'loss', 'optimizer', 'backward'} & {n['key'] for n in inference['nodes']}
    assert {('choose', 'stop'), ('stop', 'append'), ('append', 'windows')} <= infer_edges
    assert ('checkpoint', 'embedding') in infer_edges


def test_attention_shapes_and_bypasses():
    g = graph('attention', 'training')
    n = {x['key']: x for x in g['nodes']}
    assert '[B,1,dₖ]' in n['qkv']['detail']
    assert '[B,1,w]' in n['weights']['detail']
    assert '[B,1,dᵥ]' in n['message']['detail']
    edges = {(e['start'], e['end']) for e in g['edges']}
    assert {('qkv', 'message'), ('embedding', 'residual'), ('message', 'projection')} <= edges


def test_model_nodes_do_not_move_between_training_and_inference():
    for kind in ['mlp', 'attention']:
        train = {n['key']: n for n in graph(kind, 'training')['nodes']}
        infer = {n['key']: n for n in graph(kind, 'inference')['nodes']}
        for key in (train.keys() & infer.keys()) - {'windows', 'tokenize'}:
            assert (train[key]['x'], train[key]['y']) == (infer[key]['x'], infer[key]['y'])
