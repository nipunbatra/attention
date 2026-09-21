from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wordlm import (  # noqa: E402
    BOS,
    EOS,
    PAD,
    UNK,
    CausalAttentionLM,
    FixedWindowMLP,
    Vocabulary,
    attention_invariants,
    detokenize,
    make_windows,
    Story,
    TrainConfig,
    stable_story_split,
    tiny_batch_overfit,
    train_model,
    tokenize,
)


def tiny_vocab() -> Vocabulary:
    itos = [PAD, BOS, EOS, UNK, "the", "cat", "sat", "."]
    return Vocabulary(itos, {token: i for i, token in enumerate(itos)}, {}, len(itos), 1)


def test_tokenizer_round_trip_is_token_stable() -> None:
    text = "Lily's cat, sleepy?!"
    tokens = tokenize(text)
    assert tokenize(detokenize(tokens)) == tokens


def test_story_split_is_deterministic() -> None:
    text = "Once upon a time, a tiny test was written."
    assert stable_story_split(text, 17) == stable_story_split(text, 17)


def test_windows_never_cross_documents() -> None:
    vocab = tiny_vocab()
    stories = [
        Story(1, "doc-a", "the cat.", ["the", "cat", "."], "train"),
        Story(2, "doc-b", "the cat sat.", ["the", "cat", "sat", "."], "train"),
    ]
    X, y, doc_ids = make_windows(stories, vocab, context_len=3)
    assert len(y) == (3 + 1) + (4 + 1)
    boundary = 4
    assert doc_ids[:boundary] == ["doc-a"] * boundary
    assert doc_ids[boundary:] == ["doc-b"] * 5
    assert X[boundary].tolist() == [vocab.pad_id, vocab.pad_id, vocab.bos_id]
    assert y[boundary].item() == vocab.stoi["the"]


def test_fixed_window_shapes_and_finite_gradients() -> None:
    model = FixedWindowMLP(vocab_size=8, context_len=4, d_embed=6, hidden=10)
    X = torch.tensor([[0, 1, 4, 5], [1, 4, 5, 6]])
    y = torch.tensor([6, 7])
    logits = model(X)
    assert logits.shape == (2, 8)
    F.cross_entropy(logits, y).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_attention_shapes_mask_rows_and_future_invariance() -> None:
    model = CausalAttentionLM(
        vocab_size=8,
        context_len=4,
        d_model=6,
        d_k=4,
        d_v=5,
        hidden=10,
    )
    X = torch.tensor([[1, 4, 5, 6], [1, 5, 4, 7]])
    details = model.forward_details(X)
    assert details["Q"].shape == (2, 4, 4)
    assert details["K"].shape == (2, 4, 4)
    assert details["V"].shape == (2, 4, 5)
    assert details["scores"].shape == (2, 4, 4)
    assert details["messages"].shape == (2, 4, 5)
    assert details["updates"].shape == (2, 4, 6)
    assert details["contextual"].shape == (2, 4, 6)
    assert details["logits"].shape == (2, 8)
    assert torch.allclose(model(X), details["logits"], atol=1e-6)
    checks = attention_invariants(model, X)
    assert checks["max_future_weight"] == 0.0
    assert checks["max_row_sum_error"] < 1e-6
    assert checks["max_earlier_representation_change_after_future_edit"] < 1e-6
    assert checks["all_finite"]


def test_attention_has_finite_gradients() -> None:
    model = CausalAttentionLM(8, 4, 6, 4, 5, 10)
    X = torch.tensor([[1, 4, 5, 6], [1, 5, 4, 7]])
    y = torch.tensor([7, 6])
    F.cross_entropy(model(X), y).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_real_queries_do_not_attend_to_left_padding() -> None:
    model = CausalAttentionLM(8, 4, 6, 4, 5, 10, pad_id=0)
    X = torch.tensor([[0, 0, 1, 4]])
    details = model.forward_details(X)
    assert torch.isfinite(details["weights"]).all()
    assert details["weights"][0, -1, :2].abs().max().item() == 0.0
    assert abs(details["weights"][0, -1].sum().item() - 1.0) < 1e-6


def test_tiny_batch_can_be_overfit() -> None:
    torch.manual_seed(2)
    model = FixedWindowMLP(vocab_size=8, context_len=3, d_embed=8, hidden=24)
    X = torch.tensor([[1, 4, 5], [4, 5, 6], [5, 6, 4], [6, 4, 5]])
    y = torch.tensor([6, 4, 5, 7])
    before, after = tiny_batch_overfit(model, X, y, steps=180, learning_rate=3e-2)
    assert after < min(0.02, before / 20)


def test_training_restores_lowest_validation_checkpoint() -> None:
    torch.manual_seed(4)
    model = FixedWindowMLP(vocab_size=8, context_len=3, d_embed=5, hidden=9)
    X = torch.randint(0, 8, (24, 3))
    y = torch.randint(0, 8, (24,))
    result, _ = train_model(
        model,
        X,
        y,
        X,
        y,
        TrainConfig(steps=8, batch_size=8, learning_rate=1e-2, seed=5, eval_every=1),
        device="cpu",
    )
    best_trace = min(result["trace"], key=lambda point: point["validation_loss"])
    assert result["selected_step"] == best_trace["step"]
    assert abs(result["validation_loss"] - best_trace["validation_loss"]) < 1e-7
