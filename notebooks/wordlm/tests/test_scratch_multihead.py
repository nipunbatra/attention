"""The Part III worksheet and explicit implementation agree with PyTorch."""
import json
from pathlib import Path
import pytest
import torch
from multihead_from_scratch import ScratchMultiHead, TinyMultiHeadLM, copy_to_pytorch, load_worksheet_weights

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('heads', [1, 2, 4])
def test_pytorch_parity_and_causality(heads):
    torch.manual_seed(83)
    scratch = ScratchMultiHead(8, heads).double()
    E = torch.randn(2, 6, 8, dtype=torch.float64, requires_grad=True)
    delta, weights = scratch(E)
    api = copy_to_pytorch(scratch)
    mask = torch.ones(6, 6, dtype=torch.bool).triu(1)
    expected, A = api(E, E, E, attn_mask=mask, average_attn_weights=False)
    torch.testing.assert_close(delta, expected)
    torch.testing.assert_close(weights, A)
    assert weights.shape == (2, heads, 6, 6)
    assert not weights.triu(1).any()
    changed = E.detach().clone()
    changed[:, 4:] += 100
    torch.testing.assert_close(scratch(changed)[0][:, :4], delta[:, :4])
    delta.square().sum().backward()
    assert E.grad.isfinite().all()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in scratch.parameters())


def test_fixed_width_parameter_count():
    assert [sum(p.numel() for p in ScratchMultiHead(64, h).parameters()) for h in [1, 2, 4]] == [4 * 64 * 64] * 3


def test_padding_is_ignored_for_real_queries():
    torch.manual_seed(4)
    model = ScratchMultiHead(4, 2)
    E = torch.randn(2, 5, 4)
    pad = torch.tensor([[True, True, False, False, False], [True, False, False, False, False]])
    delta, weights = model(E, pad)
    assert delta.isfinite().all() and weights.isfinite().all()
    for b in range(2):
        assert not weights[b, :, ~pad[b]][..., pad[b]].any()
    changed = E.clone()
    changed[pad] += 90
    torch.testing.assert_close(model(changed, pad)[0][~pad], delta[~pad])


@pytest.mark.parametrize('name', ['river', 'cheque'])
def test_every_worksheet_output(name):
    w = json.loads((ROOT / 'multihead-worksheet.json').read_text())
    model = load_worksheet_weights(TinyMultiHeadLM(len(w['vocab'])), w)
    ids = torch.tensor([[w['vocab'].index(t.lower()) for t in w['sentences'][name]]])
    E = model.token_embedding(ids) + model.position_embedding(torch.arange(10))
    delta, weights = model.attention(E)
    target = w['headsLesson']['cases'][name]
    torch.testing.assert_close(delta[0], torch.tensor(target['delta']))
    torch.testing.assert_close(weights[0], torch.tensor([h['A'] for h in target['heads']]))
    torch.testing.assert_close(model(ids)[0], torch.tensor(target['logits'][-1]))


def test_context_and_head_validation():
    with pytest.raises(ValueError):
        ScratchMultiHead(4, 3)
    with pytest.raises(ValueError):
        ScratchMultiHead(4, 0)
    model = TinyMultiHeadLM(20)
    with pytest.raises(ValueError, match='Crop'):
        model(torch.zeros(1, 11, dtype=torch.long))
    assert model(torch.zeros(1, 3, dtype=torch.long)).shape == (1, 20)
