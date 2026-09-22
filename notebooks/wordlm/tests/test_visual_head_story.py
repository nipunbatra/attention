"""Execute the short, unbatched code printed in Part III itself."""
import json
import math
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


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
