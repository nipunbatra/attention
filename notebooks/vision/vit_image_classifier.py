"""Teaching ViT: the lecture's 224 RGB / 16 patch / 192 width architecture.

Randomly initialized; this does not load the pretrained dog checkpoint.
Dropout and stochastic depth are omitted to expose the core computation.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(192, 3 * 192)
        self.proj = nn.Linear(192, 192)

    def forward(self, x):
        # slide:attention
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, 3, 64)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / 8
        weights = scores.softmax(dim=-1)
        messages = weights @ v
        joined = messages.transpose(1, 2).reshape(B, N, D)
        return self.proj(joined)
        # endslide


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        # slide:block-init
        self.norm1 = nn.LayerNorm(192, eps=1e-6)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(192, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(192, 768), nn.GELU(),
            nn.Linear(768, 192))
        # endslide

    def forward(self, x):
        # slide:block-forward
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
        # endslide


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # slide:conv
        self.patch = nn.Conv2d(
            3, 192, kernel_size=16, stride=16,
            padding=0, bias=True)
        # endslide
        # slide:parameters
        self.cls = nn.Parameter(torch.randn(1, 1, 192) * .02)
        self.pos = nn.Parameter(torch.randn(1, 197, 192) * .02)
        # endslide
        # slide:stack
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.norm = nn.LayerNorm(192, eps=1e-6)
        self.head = nn.Linear(192, num_classes)
        # endslide

    def embed(self, x):
        # slide:embed
        # x: (B, 3, 224, 224)
        B = x.shape[0]                          # integer: number of images
        grid = self.patch(x)                    # (B, 192, 14, 14)
        rows = grid.flatten(2)                  # (B, 192, 196)
        rows = rows.transpose(1, 2)             # (B, 196, 192)
        cls = self.cls.expand(B, -1, -1)        # (B, 1, 192)
        rows = torch.cat([cls, rows], dim=1)    # (B, 197, 192)
        return rows + self.pos                  # (B, 197, 192)
        # endslide

    def forward(self, x):
        assert x.ndim == 4 and x.shape[1:] == (3, 224, 224)
        # slide:readout
        rows = self.embed(x)
        for block in self.blocks:
            rows = block(rows)
        summary = self.norm(rows)[:, 0]
        return self.head(summary)
        # endslide


def training_step(model, images, labels, optimizer):
    # slide:training
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)                   # B, 1000
    loss = F.cross_entropy(logits, labels)    # labels: B
    loss.backward()
    optimizer.step()
    # endslide
    return loss.detach()
