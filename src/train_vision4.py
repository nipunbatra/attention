#!/usr/bin/env python3
"""Vision IV toy (toy8.json v2): the frozen Vision I encoder's sixteen patch rows enter a
width-three prefix decoder through a fitted connector and answer "how many mugs?".

  python3 src/train_vision4.py            # fit, check gradients, write src/toy8.json
  python3 src/train_vision4.py --check    # recompute and compare with the saved file, write nothing
  python3 src/train_vision4.py --explore  # print the fit and the answer token's attention for several seeds

The encoder is exactly the one Vision I shows: vision-shared.js's fixed patch map, CLS row and
positions, with toy5.json's "trained" attention parameters, applied once to the 8x8 scene. Its
sixteen updated patch rows (CLS dropped) are the frozen visual rows. Everything after them,
connector W_bridge/b_bridge, the text table E_tok, the text positions P, W_Q/W_K/W_V/W_O and the
vocabulary head, is fitted on scenes A (two mugs) and B (one mug) only. Scene C (one mug, moved
right) is a probe; the page reports whatever it gives. The fitted parameters are rounded to two
decimals before anything is exported, so every displayed number can be recomputed by hand.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VOCAB = ['<bos>', 'how', 'many', 'mugs', '?', 'one', 'two', '<eos>', 'book', 'plant']
PROMPT = ['<bos>', 'how', 'many', 'mugs', '?']
ANSWERS = {'A': ['two', '<eos>'], 'B': ['one', '<eos>']}
SCENE_NAMES = {'A': 'two mugs', 'B': 'one mug', 'C': 'one mug, moved right'}
D = 3            # decoder width (d_model = d_k = d_v)
NP = 16          # image rows in front of the text
TP = 8           # text positions that own a learned position vector (5 prompt + up to 3 answer)
SEED = 9
STEP_RATE = 0.1  # the one further SGD step on scene A alone
E_AXES = ['brightness', 'contrast (left minus right)', 'row', 'col']
E_SHORT = ['bright', 'contrast', 'row', 'col']


# ---------- the scene and the frozen encoder (must agree with vision-shared.js) ----------
def scenes():
    def blank():
        return [[0] * 8 for _ in range(8)]

    def mug(g, c0):
        for r in range(1, 4):
            for c in range(c0, c0 + 3):
                g[r][c] = 3

    def book(g):
        for r in range(5, 7):
            for c in range(1, 6):
                g[r][c] = 2

    def plant(g):
        g[4][7] = 1
    S = {}
    S['A'] = blank(); mug(S['A'], 1); mug(S['A'], 5); book(S['A']); plant(S['A'])
    S['B'] = blank(); mug(S['B'], 1); book(S['B']); plant(S['B'])
    S['C'] = blank(); mug(S['C'], 4); book(S['C']); plant(S['C'])
    return S


SCENES = scenes()
W_PATCH = np.array([[.25, .5, 0, 0], [.25, -.5, 0, 0], [.25, .5, 0, 0], [.25, -.5, 0, 0]])
CLS = np.array([1., 0, 0, 0])
POS = np.array([[0, 0, -1, -1]] + [[0, 0, (j // 4) / 3, (j % 4) / 3] for j in range(16)], dtype=float)


def sm(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def patchify(g):
    g = np.array(g, dtype=float)
    return np.array([[g[2 * pr][2 * pc], g[2 * pr][2 * pc + 1], g[2 * pr + 1][2 * pc], g[2 * pr + 1][2 * pc + 1]]
                     for pr in range(4) for pc in range(4)])


def region_of(g, j):
    pr, pc = j // 4, j % 4
    objs = {}
    for r in range(2):
        for c in range(2):
            v = g[2 * pr + r][2 * pc + c]
            col = 2 * pc + c
            if not v:
                continue
            name = ('left mug' if col <= 3 else 'right mug') if v == 3 else ('book' if v == 2 else 'plant')
            objs[name] = objs.get(name, 0) + 1
    if not objs:
        return 'table'
    best = max(objs, key=lambda k: objs[k])
    n = objs[best]
    if best == 'plant':
        return 'plant'
    return best + (' centre' if n == 4 else ' corner' if n == 1 else ' edge')


def mug_patches(scene):
    return [j for j in range(16) if 'mug' in region_of(SCENES[scene], j)]


def load_encoder():
    return json.loads((HERE / 'toy5.json').read_text())['trained']


def encode(scene, enc):
    R = patchify(SCENES[scene])
    E = np.vstack([CLS, R @ W_PATCH]) + POS
    Q, K, V = E @ np.array(enc['W_Q']), E @ np.array(enc['W_K']), E @ np.array(enc['W_V'])
    S = Q @ K.T / math.sqrt(2)
    A = sm(S)
    Enew = E + (A @ V) @ np.array(enc['W_O'])
    return dict(R=R, E=E, Q=Q, K=K, V=V, S=S, A=A, Enew=Enew, G=Enew[1:])


# ---------- the decoder ----------
def initial(seed):
    rng = np.random.default_rng(seed)
    p = {'W_bridge': rng.normal(0, .3, (4, D)), 'b_bridge': np.zeros(D),
         'E_tok': rng.normal(0, .4, (len(VOCAB), D)), 'P': rng.normal(0, .1, (TP, D))}
    for key in ('W_Q', 'W_K', 'W_V', 'W_O'):
        p[key] = rng.normal(0, .3, (D, D))
    p['W_vocab'] = rng.normal(0, .35, (D, len(VOCAB)))
    p['b_vocab'] = np.zeros(len(VOCAB))
    return p


def forward(p, scene, prefix, enc):
    if scene not in SCENES or not 1 <= len(prefix) <= TP or any(t not in VOCAB for t in prefix):
        raise ValueError('Unknown scene or unsupported prefix.')
    G = encode(scene, enc)['G']
    ids = [VOCAB.index(t) for t in prefix]
    T = len(ids)
    B = G @ p['W_bridge'] + p['b_bridge']
    E = np.vstack([B, p['E_tok'][ids] + p['P'][:T]])
    n = NP + T
    Q, K, V = (E @ p['W_' + r] for r in 'QKV')
    raw = Q @ K.T
    allowed = np.zeros((n, n), dtype=bool)
    allowed[:NP, :NP] = True
    for i in range(NP, n):
        allowed[i, :i + 1] = True
    scores = np.where(allowed, raw / math.sqrt(D), -np.inf)
    A = sm(scores)
    msg = A @ V
    delta = msg @ p['W_O']
    out = E + delta
    z = out @ p['W_vocab'] + p['b_vocab']
    return dict(scene=scene, prefix=list(prefix), ids=ids, G=G, B=B, E=E, Q=Q, K=K, V=V, raw=raw, scores=scores,
                allowed=allowed, A=A, message=msg, delta=delta, out=out, logits=z, probs=sm(z))


def loss_grad(p, scene, enc):
    ans = ANSWERS[scene]
    f = forward(p, scene, PROMPT + ans[:-1], enc)
    rows = np.arange(NP + len(PROMPT) - 1, NP + len(PROMPT) - 1 + len(ans))
    targets = np.array([VOCAB.index(t) for t in ans])
    z = f['logits'][rows]
    peak = z.max(axis=1)
    losses = peak + np.log(np.exp(z - peak[:, None]).sum(axis=1)) - z[np.arange(len(ans)), targets]
    dz = np.zeros_like(f['logits'])
    dz[rows] = f['probs'][rows]
    dz[rows, targets] -= 1
    dz /= len(ans)
    g = {'W_vocab': f['out'].T @ dz, 'b_vocab': dz.sum(axis=0)}
    dout = dz @ p['W_vocab'].T
    g['W_O'] = f['message'].T @ dout
    dm = dout @ p['W_O'].T
    da = dm @ f['V'].T
    dv = f['A'].T @ dm
    ds = f['A'] * (da - (da * f['A']).sum(axis=1, keepdims=True)) / math.sqrt(D)
    dq = ds @ f['K']
    dk = ds.T @ f['Q']
    g['W_Q'] = f['E'].T @ dq
    g['W_K'] = f['E'].T @ dk
    g['W_V'] = f['E'].T @ dv
    de = dout + dq @ p['W_Q'].T + dk @ p['W_K'].T + dv @ p['W_V'].T
    g['W_bridge'] = f['G'].T @ de[:NP]
    g['b_bridge'] = de[:NP].sum(axis=0)
    g['E_tok'] = np.zeros_like(p['E_tok'])
    np.add.at(g['E_tok'], f['ids'], de[NP:])
    g['P'] = np.zeros_like(p['P'])
    g['P'][:len(f['ids'])] = de[NP:]
    return float(losses.mean()), losses, g, f


def generate(p, scene, enc, limit=3):
    prefix = PROMPT[:]
    trace = []
    reason = 'limit'
    for _ in range(limit):
        f = forward(p, scene, prefix, enc)
        row = len(f['E']) - 1
        chosen = VOCAB[int(f['probs'][row].argmax())]
        trace.append(dict(prefix=prefix[:], row=row, query=f['Q'][row], weights=f['A'][row], logits=f['logits'][row],
                          probs=f['probs'][row], chosen=chosen))
        prefix.append(chosen)
        if chosen == '<eos>':
            reason = 'eos'
            break
    return dict(tokens=prefix[len(PROMPT):], trace=trace, stoppedBy=reason)


def target_probs(p, scene, enc):
    """p(target) for each teacher-forced answer token."""
    ans = ANSWERS[scene]
    f = forward(p, scene, PROMPT + ans[:-1], enc)
    return [float(f['probs'][NP + len(PROMPT) - 1 + t][VOCAB.index(tok)]) for t, tok in enumerate(ans)]


def fitted(p, enc, margin):
    return all(generate(p, s, enc)['tokens'] == ANSWERS[s] and min(target_probs(p, s, enc)) >= margin for s in ANSWERS)


def rounded(p):
    return {k: np.round(v, 2) for k, v in p.items()}


def fit(seed, enc, margin=0.94, rate=0.01, limit=20000):
    p = initial(seed)
    hist = []
    first = {k: np.zeros_like(v) for k, v in p.items()}
    second = {k: np.zeros_like(v) for k, v in p.items()}
    for it in range(limit):
        g = {k: np.zeros_like(v) for k, v in p.items()}
        losses = []
        for name in ANSWERS:
            l, _, gi, _ = loss_grad(p, name, enc)
            losses.append(l)
            for k in p:
                g[k] += gi[k] / len(ANSWERS)
        hist.append([it, float(np.mean(losses))])
        if it > 50 and fitted(p, enc, margin) and fitted(rounded(p), enc, 0.9):
            break
        for k in p:
            first[k] = .9 * first[k] + .1 * g[k]
            second[k] = .999 * second[k] + .001 * g[k] ** 2
            p[k] -= rate * (first[k] / (1 - .9 ** (it + 1))) / (np.sqrt(second[k] / (1 - .999 ** (it + 1))) + 1e-8)
    else:
        raise RuntimeError('Toy did not fit for seed %d.' % seed)
    return p, hist, it


def attention_report(p, scene, enc):
    g = generate(p, scene, enc)
    w = g['trace'][0]['weights']
    mugs = mug_patches(scene)
    image = float(w[:NP].sum())
    mug = float(sum(w[j] for j in mugs))
    return dict(tokens=g['tokens'], image=image, mug=mug, mug_share=mug / image if image else 0.0,
                top=[int(j) for j in np.argsort(-w[:NP])[:6]])


# ---------- names for the learned decoder axes: what each one tracks across the sixteen patches ----------
PIXEL_QUANTITIES = ['bright', 'contrast', 'row', 'col']


def pixel_quantities(scene='A'):
    """Four named quantities per patch, from its pixels and its place: mean brightness, contrast (left minus right),
    patch row and patch column."""
    R = patchify(SCENES[scene])
    return np.array([R.mean(axis=1), (R[:, 0] + R[:, 2] - R[:, 1] - R[:, 3]) / 2,
                     [j // 4 for j in range(16)], [j % 4 for j in range(16)]], dtype=float)


def corr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 1e-12 else 0.0


def axis_names(M, scene='A'):
    """M: the sixteen image rows of one decoder-side matrix (B, K or V). Each learned axis is named by the pixel
    quantity it tracks most strongly across the patches (sign included); a second quantity is added when it is
    nearly as strong or when the name would repeat. Returns (short names, full descriptions with r)."""
    Q = pixel_quantities(scene)
    names, full, taken = [], [], set()
    for c in range(M.shape[1]):
        r = [corr(M[:, c], Q[i]) for i in range(4)]
        order = list(np.argsort(-np.abs(r)))
        def join(idx):
            s = ''
            for n, i in enumerate(idx):
                s += ('\u2212' if r[i] < 0 else '') + PIXEL_QUANTITIES[i] if n == 0 else (' \u2212 ' if r[i] < 0 else ' + ') + PIXEL_QUANTITIES[i]
            return s
        k = 2 if abs(r[order[1]]) >= .7 * abs(r[order[0]]) else 1
        name = join(order[:k])
        while name in taken and k < 4:
            k += 1
            name = join(order[:k])
        taken.add(name)
        names.append(name)
        full.append('tracks ' + ', '.join('%s (r = %+.2f)' % (PIXEL_QUANTITIES[i], r[i]) for i in order) + ' across the 16 patches')
    return names, full


def native(x):
    if isinstance(x, np.ndarray):
        return native(x.tolist())
    if isinstance(x, np.generic):
        return native(x.item())
    if isinstance(x, dict):
        return {k: native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [native(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        if x == -math.inf:
            return None
        raise ValueError('Nonfinite model value.')
    return x


def build(seed=SEED):
    enc = load_encoder()
    init = initial(seed)
    p, hist, steps = fit(seed, enc)
    trained = rounded(p)
    assert fitted(trained, enc, 0.9)
    # finite-difference check of every trainable scalar on both scenes at the exported checkpoint
    checks = []
    for scene in ANSWERS:
        _, _, g, _ = loss_grad(trained, scene, enc)
        for k in trained:
            for ix in np.ndindex(trained[k].shape):
                old = trained[k][ix]
                trained[k][ix] = old + 1e-5
                hi = loss_grad(trained, scene, enc)[0]
                trained[k][ix] = old - 1e-5
                lo = loss_grad(trained, scene, enc)[0]
                trained[k][ix] = old
                numeric = (hi - lo) / 2e-5
                checks.append(dict(scene=scene, parameter=k, index=list(ix), analytic=g[k][ix], numeric=numeric,
                                   error=abs(numeric - g[k][ix])))
    err = max(c['error'] for c in checks)
    assert err < 1e-7, err
    # one further SGD step on scene A alone
    lA, lsA, gA, fA = loss_grad(trained, 'A', enc)
    lB, lsB, _, fB = loss_grad(trained, 'B', enc)
    step = {k: v - STEP_RATE * gA[k] for k, v in trained.items()}
    lA2, lsA2, _, fA2 = loss_grad(step, 'A', enc)
    lB2, lsB2, _, fB2 = loss_grad(step, 'B', enc)
    assert lA2 < lA
    snaps = {'init': rounded(init), 'trained': trained, 'step': step}
    fnames = forward(trained, 'A', PROMPT, enc)
    names_e, full_e = axis_names(fnames['B'])
    names_qk, full_qk = axis_names(fnames['K'][:NP])
    names_v, full_v = axis_names(fnames['V'][:NP])
    reports = {s: {sc: attention_report(snaps[s], sc, enc) for sc in SCENES} for s in snaps}
    result = dict(
        d_model=D, d_k=D, d_v=D, vocab=VOCAB,
        axes=dict(e=names_e, qk=names_qk, v=names_v, short=dict(e=names_e, qk=names_qk, v=names_v),
                  full=dict(e=full_e, qk=full_qk, v=full_v),
                  note='Learned decoder axes, named by the pixel quantity each one tracks across the sixteen image rows of scene A (connector output for e, keys for q and k, values for v); r is the correlation.'),
        encoder=enc,
        encoderAxes=dict(e=E_AXES, short=E_SHORT),
        vlm=dict(
            vocab=VOCAB, prompt=PROMPT, answers=ANSWERS, sceneNames=SCENE_NAMES, scenes=SCENES,
            imageRows=NP, textPositions=TP, snapshots=snaps, defaultSnapshot='trained',
            mugPatches={s: mug_patches(s) for s in SCENES},
            visualRows={s: encode(s, enc)['G'] for s in SCENES},
            reference={'trained': {'A': fA, 'B': fB}, 'step': {'A': fA2, 'B': fB2}},
            generation={s: {sc: generate(snaps[s], sc, enc) for sc in SCENES} for s in snaps},
            attention=reports,
            training=dict(seed=seed, steps=steps, optimizer='Adam', rate=0.01, margin=0.94, rounding=2, curve=hist),
            update=dict(rate=STEP_RATE, scene='A', lossBefore={'A': lA, 'B': lB}, lossAfter={'A': lA2, 'B': lB2},
                        perTokenBefore={'A': lsA, 'B': lsB}, perTokenAfter={'A': lsA2, 'B': lsB2},
                        gradients=gA, gradientChecks=checks, maxGradientError=err),
            notes=['Scenes A and B only; scene C is a probe, not a training example.',
                   'The encoder is Vision I\'s exact frozen worksheet; CLS is dropped after its attention.',
                   'Positions: image rows carry row and col from the encoder; text rows add a learned P.',
                   'One head, width three; no FFN, LayerNorm or dropout. Loss on answer tokens only.']))
    return native(result)


def explore():
    enc = load_encoder()
    for seed in range(12):
        try:
            p, hist, steps = fit(seed, enc)
        except RuntimeError as e:
            print(seed, e)
            continue
        t = rounded(p)
        rep = {s: attention_report(t, s, enc) for s in SCENES}
        pa, pb = target_probs(t, 'A', enc), target_probs(t, 'B', enc)
        fn = forward(t, 'A', PROMPT, enc)
        names = axis_names(fn['B'])[0] + axis_names(fn['K'][:NP])[0] + axis_names(fn['V'][:NP])[0]
        print('seed %d: %d steps; p(two|A)=%.3f p(one|B)=%.3f; C -> %s; mug share of image mass A %.2f B %.2f C %.2f; image mass A %.2f; axes %s'
              % (seed, steps, pa[0], pb[0], rep['C']['tokens'], rep['A']['mug_share'], rep['B']['mug_share'], rep['C']['mug_share'], rep['A']['image'], names))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--explore', action='store_true')
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()
    if args.explore:
        explore()
        raise SystemExit
    result = build(args.seed)
    dest = HERE / 'toy8.json'
    if args.check:
        assert result == json.loads(dest.read_text()), 'Saved toy differs from the deterministic computation.'
        print('toy8.json matches the deterministic computation.')
    else:
        dest.write_text(json.dumps(result, indent=1) + '\n')
    m = result['vlm']
    print('Vision IV: seed %d, %d Adam steps; one SGD step on A at %.2f: loss A %.4f -> %.4f, loss B %.4f -> %.4f'
          % (m['training']['seed'], m['training']['steps'], m['update']['rate'], m['update']['lossBefore']['A'],
             m['update']['lossAfter']['A'], m['update']['lossBefore']['B'], m['update']['lossAfter']['B']))
    print('max gradient error', m['update']['maxGradientError'])
    for s in ('trained', 'step'):
        print(s, {sc: (m['generation'][s][sc]['tokens'], round(m['attention'][s][sc]['mug_share'], 2)) for sc in m['generation'][s]})
    print('axes', result['axes'])
