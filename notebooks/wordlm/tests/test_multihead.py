import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multihead import MultiHeadAttentionLM
from wordlm import CausalAttentionLM, count_parameters


def test_one_head_matches_original_and_four_heads_keep_parameter_count():
    torch.manual_seed(1)
    old = CausalAttentionLM(20, 4, 8, 8, 8, 16)
    one = MultiHeadAttentionLM(20, 4, 8, 16, heads=1)
    one.load_state_dict(old.state_dict())
    x = torch.tensor([[0, 1, 4, 5], [1, 6, 7, 8]])
    torch.testing.assert_close(one(x), old(x))
    four = MultiHeadAttentionLM(20, 4, 8, 16, heads=4)
    assert count_parameters(old) == count_parameters(four)


def test_multihead_mask_shapes_causality_gradients_and_final_query():
    model = MultiHeadAttentionLM(20, 4, 8, 16, heads=4)
    x = torch.tensor([[0, 1, 4, 5], [1, 6, 7, 8]])
    details = model.forward_details(x)
    assert details['Q'].shape == (2, 4, 4, 2)
    A = details['weights']
    assert A.shape == (2, 4, 4, 4)
    assert not A.triu(1).any()
    assert not A[0, :, 1:, 0].any()
    torch.testing.assert_close(A.sum(-1), torch.ones(2, 4, 4))
    torch.testing.assert_close(model(x), details['logits'])
    changed = x.clone()
    changed[:, -1] = 9
    torch.testing.assert_close(model.forward_details(changed)['logits_all'][:, :-1], details['logits_all'][:, :-1])
    model(x).sum().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    with pytest.raises(ValueError):
        MultiHeadAttentionLM(20, 4, 7, 16, heads=4)
