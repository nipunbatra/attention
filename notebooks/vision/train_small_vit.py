"""A complete small ViT, trained on new noisy images, with a position control.

Run: python notebooks/vision/train_small_vit.py (requires torch and numpy).
No downloads. Splits use independent random seeds; each horizontal example
has a vertical partner with exactly the same multiset of 2x2 patches.
"""
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]

class Block(nn.Module):
    def __init__(self, d=16, heads=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))

    def forward(self, x):
        z = self.norm1(x)
        x = x + self.attn(z, z, z, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))

class SmallViT(nn.Module):
    def __init__(self, positions=True, d=16):
        super().__init__()
        self.patch = nn.Conv2d(1, d, kernel_size=2, stride=2)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, 17, d), requires_grad=positions)
        self.use_positions = positions
        self.blocks = nn.Sequential(Block(d), Block(d))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)
        nn.init.normal_(self.cls, std=.02)
        if positions: nn.init.normal_(self.pos, std=.02)

    def forward(self, images):
        x = self.patch(images).flatten(2).transpose(1, 2)  # B,16,16
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        if self.use_positions: x = x + self.pos
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])  # raw class logits


def dataset(pairs, seed):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for _ in range(pairs):
        row = int(rng.integers(0, 4))
        patches = rng.normal(.10, .07, (4, 4, 2, 2))
        patches[row] += rng.uniform(.60, .85)
        patches = np.clip(patches, 0, 1)
        # Transpose patch LOCATIONS, keeping pixels inside each patch unchanged.
        for grid, label in [(patches, 0), (patches.transpose(1, 0, 2, 3), 1)]:
            images.append(grid.transpose(0, 2, 1, 3).reshape(8, 8))
            labels.append(label)
    return torch.tensor(np.array(images)[:, None], dtype=torch.float32), torch.tensor(labels)


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data[0])
        return {'loss': float(F.cross_entropy(logits, data[1])),
                'correct': int((logits.argmax(-1) == data[1]).sum()),
                'n': len(data[1]), 'accuracy': float((logits.argmax(-1) == data[1]).float().mean())}


def run():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    train, validation, test = [dataset(n, s) for n, s in [(256, 11), (64, 22), (128, 33)]]
    results = []
    for positions in [True, False]:
        torch.manual_seed(7)
        model = SmallViT(positions)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
        generator = torch.Generator().manual_seed(44)
        history = [{'epoch': 0, 'train': evaluate(model, train), 'validation': evaluate(model, validation)}]
        best_loss, best_state, best_epoch = math.inf, None, None
        for epoch in range(1, 81):
            model.train()
            for indices in torch.randperm(len(train[1]), generator=generator).split(64):
                loss = F.cross_entropy(model(train[0][indices]), train[1][indices])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            metrics = evaluate(model, validation)
            if metrics['loss'] < best_loss:
                best_loss, best_epoch = metrics['loss'], epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch in [1, 5, 10, 20, 40, 60, 80]:
                history.append({'epoch': epoch, 'train': evaluate(model, train), 'validation': metrics})
        model.load_state_dict(best_state)
        test_metrics = evaluate(model, test)
        with torch.no_grad():
            probabilities = model(test[0]).softmax(-1)
            # Without position, paired images must have equal CLS predictions.
            pair_gap = float((probabilities[0::2] - probabilities[1::2]).abs().max())
        if not positions: assert pair_gap < 1e-5
        results.append({'positions': positions, 'parameters': sum(p.numel() for p in model.parameters()),
                        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                        'best_validation_epoch': best_epoch, 'test': test_metrics,
                        'max_paired_probability_difference': pair_gap, 'history': history,
                        'example_probabilities': probabilities[:4].tolist()})
        print('positions:', positions, 'test:', test_metrics, 'selected epoch:', best_epoch, flush=True)
        torch.save(best_state, ROOT/'notebooks/vision'/('small-vit-positions.pt' if positions else 'small-vit-no-positions.pt'))
    result = {'torch': torch.__version__, 'numpy': np.__version__, 'seed_model': 7,
              'seed_order': 44, 'split_seeds': [11, 22, 33], 'split_sizes': [512,128,256],
              'epochs': 80, 'batch_size':64, 'learning_rate': .003, 'weight_decay': .01,
              'checkpoint_selection': 'minimum validation loss; test evaluated only after selection',
              'scope': 'One seed; in-distribution synthetic generalization. Not a natural-image benchmark.',
              'examples': test[0][:4, 0].tolist(), 'results': results}
    (ROOT/'figures/vision1/training.json').write_text(json.dumps(result, indent=2)+'\n')

if __name__ == '__main__': run()
