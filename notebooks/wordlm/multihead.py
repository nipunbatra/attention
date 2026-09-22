"""One attention block with several heads; the Part II prediction head is unchanged."""
import math
import torch
from torch.nn import functional as F
from wordlm import CausalAttentionLM


class MultiHeadAttentionLM(CausalAttentionLM):
    def __init__(self, vocab_size, context_len, d_model=64, hidden=256, heads=4, pad_id=0):
        if heads < 1 or d_model % heads:
            raise ValueError('d_model must be divisible by the positive head count')
        super().__init__(vocab_size, context_len, d_model, d_model, d_model, hidden, pad_id)
        self.heads = heads
        self.head_width = d_model // heads

    def split_heads(self, rows):
        B, T, _ = rows.shape
        return rows.reshape(B, T, self.heads, self.head_width).transpose(1, 2)

    def forward_details(self, context_ids):
        B, T = context_ids.shape
        positions = torch.arange(T, device=context_ids.device)
        E = self.token_embedding(context_ids) + self.position_embedding(positions)[None]
        Q, K, V = (self.split_heads(layer(E)) for layer in (self.W_Q, self.W_K, self.W_V))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_width)
        future = torch.ones(T, T, dtype=torch.bool, device=E.device).triu(1)
        real = context_ids.ne(self.pad_id)
        mask = future[None] | ((~real[:, None, :]) & real[:, :, None])
        A = F.softmax(scores.masked_fill(mask[:, None], float('-inf')), dim=-1)
        messages = (A @ V).transpose(1, 2).reshape(B, T, self.d_model)
        contextual = E + self.W_O(messages)
        logits = self.vocab_head(F.relu(self.hidden_layer(contextual)))
        return dict(E=E, Q=Q, K=K, V=V, weights=A, messages=messages,
                    contextual=contextual, logits_all=logits, logits=logits[:, -1])

    def forward(self, context_ids):
        B, T = context_ids.shape
        positions = torch.arange(T, device=context_ids.device)
        E = self.token_embedding(context_ids) + self.position_embedding(positions)[None]
        Q = self.split_heads(self.W_Q(E[:, -1:]))
        K, V = self.split_heads(self.W_K(E)), self.split_heads(self.W_V(E))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_width)
        mask = context_ids.eq(self.pad_id)[:, None, None, :]
        A = F.softmax(scores.masked_fill(mask, float('-inf')), dim=-1)
        message = (A @ V).transpose(1, 2).reshape(B, self.d_model)
        updated = E[:, -1] + self.W_O(message)
        return self.vocab_head(F.relu(self.hidden_layer(updated)))
