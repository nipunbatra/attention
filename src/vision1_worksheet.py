"""Four visible patches, two attention heads, and one reproducible worksheet.

Parameters are chosen for arithmetic, not fitted. The two named arrangements
are the exercise; this is not a general horizontal/vertical recognizer.
"""
from pathlib import Path
import json
import numpy as np


def parameters():
    root2 = float(np.sqrt(2))
    return {
        'd_model': 4, 'd_k': 2, 'd_v': 2,
        'W_patch': [[.25, 0, 0, 0]] * 4,
        'b_patch': [0, 0, 0, 1],
        'cls': [0, 0, 0, 1],
        'positions': [[0, 0, 0, 0], [0, 0, 0, 0],
                      [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]],
        'heads': [
            {'W_Q': [[0, 0], [0, 0], [0, 0], [1, 1]],
             'W_K': [[root2, 0], [0, root2], [0, 0], [0, 0]],
             'W_V': [[1, 0], [0, 1], [0, 0], [0, 0]]},
            {'W_Q': [[0, 0], [0, 0], [0, 0], [1, 1]],
             'W_K': [[root2, 0], [0, 0], [0, root2], [0, 0]],
             'W_V': [[1, 0], [0, 0], [0, 1], [0, 0]]},
        ],
        'W_O': [[1, 0, 1, 0], [0, 1, 0, 0],
                [-1, 0, 1, 0], [0, 0, 0, 1]],
        'W_class': [[-4, 4], [0, 0], [0, 0], [0, 0]],
        'classes': ['Across the top', 'Down the left'],
        'images': {
            'horizontal': [[1, 1, 1, 1], [1, 1, 1, 1],
                           [0, 0, 0, 0], [0, 0, 0, 0]],
            'vertical': [[1, 1, 0, 0], [1, 1, 0, 0],
                         [1, 1, 0, 0], [1, 1, 0, 0]],
        },
    }


def softmax(x):
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


def forward(image, use_positions=True):
    p = parameters()
    pixels = np.asarray(image, dtype=float)
    # Four row-major patches; each patch's pixels are TL, TR, BL, BR.
    patches = pixels.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    content = patches @ np.asarray(p['W_patch']) + p['b_patch']
    E = np.vstack([p['cls'], content])
    if use_positions:
        E = E + p['positions']
    heads = []
    for w in p['heads']:
        Q, K, V = [E @ np.asarray(w[key]) for key in ['W_Q', 'W_K', 'W_V']]
        scores = Q @ K.T / np.sqrt(p['d_k'])
        A = softmax(scores)
        H = A @ V
        heads.append(dict(Q=Q, K=K, V=V, scores=scores, A=A, H=H,
                          contributions=A[0, :, None] * V))
    joined = np.concatenate([h['H'] for h in heads], axis=-1)
    delta = joined @ np.asarray(p['W_O'])
    updated = E + delta
    logits = updated[0] @ np.asarray(p['W_class'])
    probability = softmax(logits)
    return dict(pixels=pixels, patches=patches, content=content, E=E,
                heads=heads, joined=joined, delta=delta, updated=updated,
                logits=logits, probability=probability)


def serializable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: serializable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [serializable(v) for v in x]
    return x


def export():
    p = parameters()
    p['cases'] = {
        name + ('_positions' if positions else '_no_positions'):
        serializable(forward(image, positions))
        for name, image in p['images'].items() for positions in [True, False]
    }
    path = Path(__file__).with_name('vision1-worksheet.json')
    path.write_text(json.dumps(p, indent=2) + '\n')
    return p


if __name__ == '__main__':
    data = export()
    for name, case in data['cases'].items():
        print(name, 'class probabilities:', np.round(case['probability'], 6))
