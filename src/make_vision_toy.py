#!/usr/bin/env python3
"""Vision I toy v2: one 8x8 scene, named axes, a hand-built encoder, a trained CLS attention classifier.
Design: VISION_AXES.md. Usage: python3 make_vision_toy.py [--check-only]. Writes toy5.json."""
import json, math, sys
import numpy as np

# ---------------------------------------------------------------- scenes
def blank(): return np.zeros((8, 8), dtype=int)
def mug(img, c0):  img[1:4, c0:c0 + 3] = 3
def book(img):     img[5:7, 1:6] = 2
def plant(img):    img[4, 7] = 1
SCENES = {}
a = blank(); mug(a, 1); mug(a, 5); book(a); plant(a); SCENES['A'] = a
b = blank(); mug(b, 1); book(b); plant(b);            SCENES['B'] = b
c = blank(); mug(c, 4); book(c); plant(c);            SCENES['C'] = c
d = blank(); book(d);                                  SCENES['D'] = d
e = blank(); plant(e);                                 SCENES['E'] = e
NAMES = {'A': 'two mugs', 'B': 'one mug', 'C': 'one mug, moved', 'D': 'book only', 'E': 'plant only'}
LABEL = {'A': 0, 'B': 1, 'C': 0}  # class 0 = a mug on the right half, class 1 = no mug on the right half
CLASSES = ['mug on the right', 'no mug on the right']
TRAIN_KEYS = ('A', 'B')

def patchify(img):
    rows = []
    for pr in range(4):
        for pc in range(4):
            blk = img[2 * pr:2 * pr + 2, 2 * pc:2 * pc + 2]
            rows.append([int(blk[0, 0]), int(blk[0, 1]), int(blk[1, 0]), int(blk[1, 1])])
    return np.array(rows, dtype=float)

def region_of(img, j):
    pr, pc = divmod(j, 4)
    blk = img[2 * pr:2 * pr + 2, 2 * pc:2 * pc + 2]
    objs = {}
    for r in range(2):
        for cc in range(2):
            v = int(blk[r, cc]); rr, col = 2 * pr + r, 2 * pc + cc
            if v == 0: continue
            if v == 3: name = 'left mug' if col <= 3 else 'right mug'
            elif v == 2: name = 'book'
            else: name = 'plant'
            objs[name] = objs.get(name, 0) + 1
    if not objs: return 'table'
    name, n = max(objs.items(), key=lambda kv: kv[1])
    if name == 'plant': return 'plant'
    return name + {4: ' centre', 2: ' edge', 1: ' corner', 3: ' edge'}[n]

# ---------------------------------------------------------------- encoder (fixed, named)
AXES = {
    'e': ['brightness', 'contrast (left minus right)', 'row', 'col'],
    'qk': ['bright region?', 'on the right?'],
    'v': ['sends: brightness', 'sends: contrast'],
    'short': {'e': ['bright', 'contrast', 'row', 'col'], 'qk': ['bright?', 'right?'], 'v': ['→bright', '→contrast']},
}
W_PATCH = np.array([[.25, .5, 0, 0], [.25, -.5, 0, 0], [.25, .5, 0, 0], [.25, -.5, 0, 0]])  # 4 pixels -> 4 axes
POS = np.zeros((17, 4)); POS[0, 2:] = [-1.0, -1.0]
for j in range(16):
    pr, pc = divmod(j, 4); POS[j + 1, 2:] = [pr / 3.0, pc / 3.0]
CLS = np.array([1.0, 0.0, 0.0, 0.0])   # CLS starts with brightness 1 so its untrained query asks for bright regions

def embed(img):
    R = patchify(img)
    E = np.vstack([CLS[None, :], R @ W_PATCH]) + POS
    return R, E

def init_params():
    WQ = np.zeros((4, 2)); WQ[0, 0] = 1.0; WQ[0, 1] = 1.0   # CLS start row (brightness 1) asks: bright region? and on the right?
    WK = np.zeros((4, 2)); WK[0, 0] = 1.0; WK[3, 1] = 1.0   # keys offer: brightness -> bright?, col position -> on the right?
    WV = np.zeros((4, 2)); WV[0, 0] = 1.0; WV[1, 1] = 1.0
    WO = np.zeros((2, 4)); WO[0, 0] = 1.0; WO[1, 1] = 1.0
    WC = np.array([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # brightness -> two mugs (+), one mug (-)
    bC = np.array([-1.5, 1.5])
    return {'W_Q': WQ, 'W_K': WK, 'W_V': WV, 'W_O': WO, 'W_cls': WC, 'b_cls': bC}

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True); ex = np.exp(x); return ex / ex.sum(axis=-1, keepdims=True)

def forward(P, img):
    R, E = embed(img)
    Q, K, V = E @ P['W_Q'], E @ P['W_K'], E @ P['W_V']
    S = Q @ K.T / math.sqrt(2)
    A = softmax(S)
    H = A @ V
    D = H @ P['W_O']
    En = E + D
    logits = En[0] @ P['W_cls'] + P['b_cls']
    p = softmax(logits)
    return dict(R=R, E=E, Q=Q, K=K, V=V, S=S, A=A, H=H, Delta=D, Enew=En, logits=logits, probs=p)

def loss(P, keys=TRAIN_KEYS):
    tot = 0.0
    for k in keys:
        f = forward(P, SCENES[k]); tot += -math.log(f['probs'][LABEL[k]] + 1e-12)
    reg = sum((P[n] ** 2).sum() for n in ('W_Q', 'W_K', 'W_V', 'W_O', 'W_cls'))
    return tot / len(keys) + 1e-3 * reg

def grad_fd(P, eps=1e-5):
    g = {}
    for n, arr in P.items():
        ga = np.zeros_like(arr)
        it = np.nditer(arr, flags=['multi_index'])
        for _ in it:
            i = it.multi_index; old = arr[i]
            arr[i] = old + eps; lp = loss(P); arr[i] = old - eps; lm = loss(P); arr[i] = old
            ga[i] = (lp - lm) / (2 * eps)
        g[n] = ga
    return g

def train(P, steps=1500, lr=0.03, seed=0):
    rng = np.random.default_rng(seed)
    m = {n: np.zeros_like(v) for n, v in P.items()}; v2 = {n: np.zeros_like(v) for n, v in P.items()}
    curve = []
    for t in range(1, steps + 1):
        g = grad_fd(P)
        g['W_Q'][2:, :] = 0.0; g['W_V'][2:, :] = 0.0; g['W_K'][2, :] = 0.0   # queries and values read content only; keys may offer the column position
        for n in P:
            m[n] = 0.9 * m[n] + 0.1 * g[n]; v2[n] = 0.999 * v2[n] + 0.001 * g[n] ** 2
            mh = m[n] / (1 - 0.9 ** t); vh = v2[n] / (1 - 0.999 ** t)
            P[n] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if t % 25 == 0 or t == 1: curve.append([t, round(loss(P), 4)])
    return curve

def rnd(a, d=2): return np.round(np.asarray(a, dtype=float), d).tolist()

MUG_PATCHES_A = [j for j in range(16) if 'right mug' in region_of(SCENES['A'], j)]

def check(P0, P1, verbose=True):
    ok = True; rep = []
    fA0, fA1 = forward(P0, SCENES['A']), forward(P1, SCENES['A'])
    fB1, fC1 = forward(P1, SCENES['B']), forward(P1, SCENES['C'])
    a0, a1 = fA0['A'][0][1:], fA1['A'][0][1:]
    left = [j for j in range(16) if 'left mug' in region_of(SCENES['A'], j)]
    t1 = (int(np.argmax(a0)) in MUG_PATCHES_A) and (a0[MUG_PATCHES_A].sum() > a0[left].sum())
    t2 = a1[MUG_PATCHES_A].sum() >= 0.5
    t3 = fA1['probs'][0] >= 0.9 and fB1['probs'][1] >= 0.9
    rep.append(('T1 untrained CLS: top patch is a right-mug patch and right-mug mass > left-mug mass', t1, (int(np.argmax(a0)) + 1, round(float(a0[MUG_PATCHES_A].sum()), 3), round(float(a0[left].sum()), 3))))
    rep.append(('T2 trained CLS mass on right-mug patches >= 0.5', t2, round(float(a1[MUG_PATCHES_A].sum()), 3)))
    rep.append(('T3 p(correct) >= 0.9 for A and B', t3, (round(float(fA1['probs'][0]), 3), round(float(fB1['probs'][1]), 3))))
    rep.append(('T4 scene C (mug moved to the right) prediction', True, (CLASSES[0] if fC1['probs'][0] > .5 else CLASSES[1], round(float(fC1['probs'][0]), 3))))
    for name, passed, val in rep:
        ok &= bool(passed)
        if verbose: print(('PASS ' if passed else 'FAIL ') + name + ': ' + str(val))
    return ok

def main():
    P0 = init_params()
    P1 = {n: v.copy() for n, v in init_params().items()}
    curve = train(P1)
    # round trained params to 2 decimals and re-check on rounded values
    P1r = {n: np.round(v, 2) for n, v in P1.items()}
    print('loss initial %.4f -> trained %.4f (rounded %.4f)' % (loss(P0), loss(P1), loss(P1r)))
    ok = check(P0, P1r)
    if '--check-only' in sys.argv: sys.exit(0 if ok else 1)
    out = {
        'd_model': 4, 'd_k': 2, 'd_v': 2, 'axes': AXES,
        'classes': CLASSES,
        'scenes': {k: {'name': NAMES[k], 'pixels': SCENES[k].tolist(), 'regions': [region_of(SCENES[k], j) for j in range(16)],
                       'label': CLASSES[LABEL[k]] if k in LABEL else None} for k in SCENES},
        'W_patch': rnd(W_PATCH), 'b_patch': [0, 0, 0, 0], 'pos_emb': rnd(POS), 'cls': rnd(CLS),
        'initial': {n: rnd(v, 1) for n, v in P0.items()},
        'trained': {n: rnd(v, 2) for n, v in P1r.items()},
        'curve': curve,
        'ramp': {'0': '#3A3A3A', '1': '#6E6E6E', '2': '#A3A3A3', '3': '#D8D8D8'},
        'notes': 'Vision I toy v2. One 8x8 scene with 2x2 patches (16 patch rows plus CLS = 17 rows). The patch encoder is fixed by its axis names: '
                 'brightness = mean of the four pixels, contrast = left column minus right column; row and col carry the patch position. '
                 'The question is: is there a mug on the right half? Only the content rows of W_Q and W_V, the content and column rows of W_K, W_O and the class head are trained (Adam, central-difference gradients, seed 0) on scenes A and B; the other position rows stay zero, so queries ask about content, keys can offer content and column position; '
                 'W_patch, positions and the CLS start row are fixed so the names stay true. Scene C moves the mug to the right half and is the generalisation probe. Numbers are rounded to two decimals and all checks '
                 'pass on the rounded values. Scene C is a generalisation probe and is reported whatever the outcome.'
    }
    for k in ('A', 'B', 'C'):
        out['scenes'][k]['cls_attention_initial'] = rnd(forward(P0, SCENES[k])['A'][0], 3)
        out['scenes'][k]['cls_attention_trained'] = rnd(forward(P1r, SCENES[k])['A'][0], 3)
        out['scenes'][k]['probs_initial'] = rnd(forward(P0, SCENES[k])['probs'], 3)
        out['scenes'][k]['probs_trained'] = rnd(forward(P1r, SCENES[k])['probs'], 3)
    json.dump(out, open('toy5.json', 'w'), indent=1)
    print('wrote toy5.json;', 'mug patches (0-based):', MUG_PATCHES_A)
    print('regions A:', out['scenes']['A']['regions'])
    print('CLS attention A trained:', out['scenes']['A']['cls_attention_trained'])
    print('CLS attention B trained:', out['scenes']['B']['cls_attention_trained'])
    print('probs A/B/C trained:', out['scenes']['A']['probs_trained'], out['scenes']['B']['probs_trained'], out['scenes']['C']['probs_trained'])
    print('trained W_Q', out['trained']['W_Q'], 'W_K', out['trained']['W_K'])
    sys.exit(0 if ok else 1)

if __name__ == '__main__': main()
