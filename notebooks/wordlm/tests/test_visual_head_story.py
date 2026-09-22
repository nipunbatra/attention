"""Execute the short, unbatched code printed in Part III itself."""
import json
import math
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def test_intro_mixtures_and_projected_source_rows():
    values = torch.tensor([[10., 1.], [2., 8.]])
    a_setting = torch.tensor([.8, .2])
    a_person = torch.tensor([.2, .8])
    torch.testing.assert_close(a_setting @ values, torch.tensor([8.4, 2.4]))
    separate = torch.stack([a_setting @ values[:, 0], a_person @ values[:, 1]])
    torch.testing.assert_close(separate, torch.tensor([8.4, 6.6]))
    data = json.loads((ROOT / 'multihead-worksheet.json').read_text())['headsLesson']
    case = data['cases']['river']
    E = torch.tensor(case['E'])
    for h, projection in enumerate(data['projections']):
        for kind in ['Q', 'K', 'V']:
            actual = E @ torch.tensor(projection[kind], dtype=torch.float32)
            torch.testing.assert_close(actual, torch.tensor(case['heads'][h][kind]))
    assert case['heads'][0]['Q'][-1] == [2.3, 2.3]
    assert case['heads'][1]['Q'][-1] == [2.3, 0.0]


def test_printed_head_code_and_pytorch():
    steps = {s['key']: s for s in json.loads((ROOT / 'multihead-story.json').read_text())}
    data = json.loads((ROOT / 'multihead-worksheet.json').read_text())['headsLesson']
    case = data['cases']['river']
    scope = dict(torch=torch, math=math, nn=nn, E=torch.tensor(case['E']))
    scope['W_O'] = torch.tensor(data['W_O'])
    for h, projection in enumerate(data['projections'], 1):
        for kind in ['Q', 'K', 'V']:
            scope[f'W_{kind}{h}'] = torch.tensor(projection[kind], dtype=torch.float32)
    exec(steps['s05-v-scratch']['code'], scope)
    exec(steps['s05-v-combine']['code'], scope)
    expected = torch.tensor(case['updated'])
    torch.testing.assert_close(scope['E_prime'], expected)
    # The API cell creates random parameters; copy the *same* toy parameters
    # before judging parity, rather than comparing unrelated random layers.
    exec(steps['s05-v-pytorch']['code'], scope)
    mha = scope['mha']
    with torch.no_grad():
        packed = [torch.cat([scope[f'W_{k}1'], scope[f'W_{k}2']], dim=1).T for k in ['Q', 'K', 'V']]
        mha.in_proj_weight.copy_(torch.cat(packed, dim=0))
        mha.out_proj.weight.copy_(scope['W_O'].T)
    delta, weights = mha(scope['E'], scope['E'], scope['E'], attn_mask=scope['future'], average_attn_weights=False)
    torch.testing.assert_close(scope['E'] + delta, expected)
    torch.testing.assert_close(weights, torch.tensor([h['A'] for h in case['heads']]))
    assert weights.shape == (2, 10, 10)


def test_each_head_dot_softmax_and_value_walkthrough():
    data = json.loads((ROOT / 'multihead-worksheet.json').read_text())['headsLesson']
    for case in data['cases'].values():
        for head in case['heads']:
            Q, K, V = [torch.tensor(head[kind], dtype=torch.float64) for kind in ['Q', 'K', 'V']]
            q = Q[-1:]
            raw = q @ K.T
            assert raw.shape == (1, 10)
            for j in range(10):
                torch.testing.assert_close(raw[0, j], (q[0] * K[j]).sum())
            scores = raw / math.sqrt(2)
            torch.testing.assert_close(scores[0], torch.tensor(head['scores'][-1], dtype=torch.float64))
            exponentials = scores.exp()
            A_row = exponentials / exponentials.sum(-1, keepdim=True)
            torch.testing.assert_close(A_row, scores.softmax(-1))
            torch.testing.assert_close(A_row[0], torch.tensor(head['A'][-1], dtype=torch.float64))
            contributions = A_row.T * V
            torch.testing.assert_close(contributions.sum(0, keepdim=True), A_row @ V)
            torch.testing.assert_close((A_row @ V)[0], torch.tensor(head['messages'][-1], dtype=torch.float64))


def test_wide_head_uses_one_softmax_for_the_same_coordinates():
    data = json.loads((ROOT / 'multihead-worksheet.json').read_text())['headsLesson']
    for case in data['cases'].values():
        E = torch.tensor(case['E'])
        layer = nn.MultiheadAttention(4, 1, bias=False)
        with torch.no_grad():
            packed = [torch.cat([torch.tensor(p[k], dtype=torch.float32)
                                 for p in data['projections']], dim=1).T
                      for k in ['Q', 'K', 'V']]
            layer.in_proj_weight.copy_(torch.cat(packed))
            layer.out_proj.weight.copy_(torch.tensor(data['W_O']).T)
        future = torch.ones(10, 10, dtype=torch.bool).triu(1)
        delta, A = layer(E, E, E, attn_mask=future, average_attn_weights=False)
        torch.testing.assert_close(A[0], torch.tensor(case['wide']['A']))
        torch.testing.assert_close(delta, torch.tensor(case['wide']['messages']) @ torch.tensor(data['W_O']))
        assert not torch.allclose(torch.tensor(case['wide']['messages']), torch.tensor(case['joined']))
        assert sum(p.numel() for p in layer.parameters()) == 64


def test_projection_bias_scope_and_broadcasting():
    from multihead_from_scratch import TinyMultiHeadLM
    plain = nn.MultiheadAttention(4, 2, bias=False)
    biased = nn.MultiheadAttention(4, 2, bias=True)
    assert plain.in_proj_bias is None and plain.out_proj.bias is None
    assert biased.in_proj_bias.shape == (12,) and biased.out_proj.bias.shape == (4,)
    count = lambda layer: sum(p.numel() for p in layer.parameters())
    assert count(biased) - count(plain) == 16
    E = torch.tensor([[0., 0., 0., 2.3], [0., 0., 0., 1.]])
    W_Q = torch.tensor([[0., 0.], [0., 0.], [0., 0.], [1., 1.]])
    b_Q = torch.tensor([.2, -.1])
    torch.testing.assert_close(E @ W_Q + b_Q, torch.tensor([[2.5, 2.2], [1.2, .9]]))
    model = TinyMultiHeadLM(20)
    assert model.hidden.bias is not None and model.readout.bias is not None
    for name in ['W_Q', 'W_K', 'W_V', 'W_O']:
        assert getattr(model.attention, name).bias is None
