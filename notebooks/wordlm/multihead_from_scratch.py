"""Part III: explicit multi-head self-attention and a small next-token model."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ScratchMultiHead(nn.Module):
    def __init__(self, width=4, heads=2):
        super().__init__()
        if heads < 1 or width % heads:
            raise ValueError('width must be divisible by a positive head count')
        self.width, self.heads = width, heads
        self.head_width = width // heads
        self.W_Q = nn.Linear(width, width, bias=False)
        self.W_K = nn.Linear(width, width, bias=False)
        self.W_V = nn.Linear(width, width, bias=False)
        self.W_O = nn.Linear(width, width, bias=False)

    def split_heads(self, rows):
        B, T, _ = rows.shape
        return rows.reshape(B, T, self.heads, self.head_width).transpose(1, 2)

    def forward(self, E, padding_mask=None):
        B, T, _ = E.shape
        Q = self.split_heads(self.W_Q(E))
        K = self.split_heads(self.W_K(E))
        V = self.split_heads(self.W_V(E))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_width)
        blocked = torch.ones(T, T, dtype=torch.bool, device=E.device).triu(1)
        if padding_mask is not None:
            # Ignore PAD sources for real queries. PAD-query outputs are unused;
            # leave their causal prefix available to avoid an all-masked softmax.
            blocked = blocked | (padding_mask[:, None, :] & ~padding_mask[:, :, None])
            blocked = blocked[:, None]
        weights = scores.masked_fill(blocked, float('-inf')).softmax(dim=-1)
        messages = weights @ V
        joined = messages.transpose(1, 2).contiguous().reshape(B, T, self.width)
        delta = self.W_O(joined)
        return delta, weights


class TinyMultiHeadLM(nn.Module):
    def __init__(self, vocab_size, context=10, width=4, heads=2, hidden=8):
        super().__init__()
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context, width)
        self.attention = ScratchMultiHead(width, heads)
        self.hidden = nn.Linear(width, hidden)
        self.readout = nn.Linear(hidden, vocab_size)

    def forward(self, ids):
        T = ids.shape[1]
        if T > self.context:
            raise ValueError('Crop the prompt to the configured context window')
        E = self.token_embedding(ids) + self.position_embedding(torch.arange(T, device=ids.device))
        delta, _ = self.attention(E)
        updated = E + delta
        return self.readout(F.relu(self.hidden(updated[:, -1])))


def copy_to_pytorch(scratch):
    """Use exactly the same parameters, not a newly randomized comparison."""
    layer = nn.MultiheadAttention(scratch.width, scratch.heads, bias=False,
                                  dropout=0.0, batch_first=True)
    layer = layer.to(device=scratch.W_Q.weight.device, dtype=scratch.W_Q.weight.dtype)
    with torch.no_grad():
        layer.in_proj_weight.copy_(torch.cat([scratch.W_Q.weight,
                                             scratch.W_K.weight,
                                             scratch.W_V.weight], dim=0))
        layer.out_proj.weight.copy_(scratch.W_O.weight)
    return layer


def load_worksheet_weights(model, worksheet):
    """Load the printed, hand-chosen example; this is not a training algorithm."""
    heads = worksheet['headsLesson']['projections']
    with torch.no_grad():
        model.token_embedding.weight.copy_(torch.tensor([worksheet['tok_emb'][w] for w in worksheet['vocab']]))
        model.position_embedding.weight.copy_(torch.tensor(worksheet['pos_emb'][:model.context]))
        for letter in ['Q', 'K', 'V']:
            packed = [a + b for a, b in zip(heads[0][letter], heads[1][letter])]
            getattr(model.attention, 'W_' + letter).weight.copy_(torch.tensor(packed).T)
        model.attention.W_O.weight.copy_(torch.tensor(worksheet['headsLesson']['W_O']).T)
        model.hidden.weight.copy_(torch.tensor(worksheet['W_hidden']).T)
        model.hidden.bias.copy_(torch.tensor(worksheet['b_hidden']))
        model.readout.weight.copy_(torch.tensor(worksheet['W_vocab']).T)
        model.readout.bias.copy_(torch.tensor(worksheet['b_vocab']))
    return model
