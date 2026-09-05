#!/usr/bin/env python3
"""Build toy6.json for Vision II (learn visual representations without class labels).

Everything numerical on the Vision II page comes from this file:
  * the frozen Vision I encoder (toy5.json "trained") on the shared 8x8 scene,
  * a tiny MAE decoder trained to predict the hidden right-mug pixels of scenes A, B, D,
  * an I-JEPA predictor trained to predict the hidden patches' (brightness, contrast) features,
  * a DINO student/teacher head on two views, four runs (centring and sharpening on/off),
  * a linear probe on pooled frozen features (two mugs versus one mug).

    python3 train_vision2.py            # writes toy6.json
    python3 train_vision2.py --check    # recomputes everything and compares with toy6.json

Seeded numpy only; central-difference gradients (the models have 10 to 32 parameters).
The scene, patch order, W_patch, positions and the attention formula copy vision-shared.js exactly.
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'toy6.json')
SQ2 = math.sqrt(2.0)

# ---------------------------------------------------------------- the scene (vision-shared.js)
def blank(): return np.zeros((8, 8))
def mug(g, c0): g[1:4, c0:c0 + 3] = 3
def book(g): g[5:7, 1:6] = 2
def plant(g): g[4, 7] = 1
SCENES = {}
g = blank(); mug(g, 1); mug(g, 5); book(g); plant(g); SCENES['A'] = g
g = blank(); mug(g, 1); book(g); plant(g); SCENES['B'] = g
g = blank(); mug(g, 4); book(g); plant(g); SCENES['C'] = g
g = blank(); book(g); SCENES['D'] = g
g = blank(); plant(g); SCENES['E'] = g
SCENE_NAMES = {'A': 'two mugs', 'B': 'one mug', 'C': 'one mug, moved right', 'D': 'book only', 'E': 'plant only'}

W_PATCH = np.array([[.25, .5, 0, 0], [.25, -.5, 0, 0], [.25, .5, 0, 0], [.25, -.5, 0, 0]])
CLS = np.array([1., 0., 0., 0.])
POS = np.array([[0, 0, -1, -1]] + [[0, 0, (j // 4) / 3, (j % 4) / 3] for j in range(16)], float)
HIDDEN = [2, 3, 6, 7]                      # the right-mug patches (0-based, row-major)
MAE_SCENES = ['A', 'B', 'D']

toy5 = json.load(open(os.path.join(HERE, 'toy5.json'), encoding='utf-8'))
ENC = {k: np.array(v, float) for k, v in toy5['trained'].items()}

def patchify(g):
    rows = []
    for pr in range(4):
        for pc in range(4):
            rows.append([g[2 * pr][2 * pc], g[2 * pr][2 * pc + 1], g[2 * pr + 1][2 * pc], g[2 * pr + 1][2 * pc + 1]])
    return np.array(rows, float)

def softmax(x):
    x = np.asarray(x, float); m = x.max(axis=-1, keepdims=True); e = np.exp(x - m); return e / e.sum(axis=-1, keepdims=True)

def embed(g):
    return np.vstack([CLS, patchify(g) @ W_PATCH]) + POS

def attend(E, P):
    Q, K, Vv = E @ P['W_Q'], E @ P['W_K'], E @ P['W_V']
    S = (Q @ K.T) / SQ2
    A = softmax(S)
    H = A @ Vv
    D = H @ P['W_O']
    return {'E': E, 'Q': Q, 'K': K, 'V': Vv, 'S': S, 'A': A, 'H': H, 'D': D, 'Enew': E + D}

def encode(g):
    """The frozen Vision I encoder on the full image: 17 updated rows, CLS first."""
    return attend(embed(g), ENC)['Enew']

def encode_visible(g, hidden):
    """MAE: only CLS and the visible patch rows enter the encoder."""
    idx = [0] + [j + 1 for j in range(16) if j not in hidden]
    out = attend(embed(g)[idx], ENC)
    return idx, out

# ---------------------------------------------------------------- views
def flip(g): return np.array(g)[:, ::-1].copy()
def dim(g): return 0.75 * np.array(g)
def crop(g):
    idx = [1 + (r * 7) // 8 for r in range(8)]
    return np.array(g)[np.ix_(idx, idx)].copy()
VIEWS = {'identity': lambda g: np.array(g, float), 'flip': flip, 'dim': dim, 'crop': crop, 'crop+dim': lambda g: dim(crop(g))}

# ---------------------------------------------------------------- parameter vectors
class Params:
    def __init__(self, shapes):
        self.shapes = shapes
    def unflat(self, v):
        out, k = {}, 0
        for name, shape in self.shapes:
            n = int(np.prod(shape)); out[name] = np.array(v[k:k + n]).reshape(shape); k += n
        return out
    def flat(self, d):
        return np.concatenate([np.asarray(d[name], float).ravel() for name, _ in self.shapes])

def fd_grad(f, v, eps=1e-6):
    g = np.zeros_like(v)
    for i in range(len(v)):
        vp = v.copy(); vp[i] += eps; vm = v.copy(); vm[i] -= eps
        g[i] = (f(vp) - f(vm)) / (2 * eps)
    return g

class Adam:
    def __init__(self, n, lr, b1=.9, b2=.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps; self.m = np.zeros(n); self.v = np.zeros(n); self.t = 0
    def step(self, v, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g; self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t); vh = self.v / (1 - self.b2 ** self.t)
        return v - self.lr * mh / (np.sqrt(vh) + self.eps)

# ---------------------------------------------------------------- the placeholder read (shared by MAE and I-JEPA)
def placeholder_read(theta, g, hidden):
    """For every hidden patch j: u_j = m + p_j asks q_j = u_j W_Qd; the visible encoded patch rows offer the frozen
    keys and values; the message g_j and the slot position form the 4-number decoder row z_j."""
    idx, out = encode_visible(g, hidden)
    vis = [i - 1 for i in idx if i > 0]              # visible patch indices (0-based)
    Ev = out['Enew'][1:]
    K, Vv = Ev @ ENC['W_K'], Ev @ ENC['W_V']
    res = {}
    for j in hidden:
        u = theta['m'] + POS[j + 1]
        q = u @ theta['W_Qd']
        s = (K @ q) / SQ2
        a = softmax(s)
        msg = a @ Vv
        z = np.array([msg[0], msg[1], POS[j + 1][2], POS[j + 1][3]])
        res[j] = {'u': u, 'q': q, 'scores': s, 'alpha': a, 'msg': msg, 'z': z}
    return vis, res

MAE = Params([('m', (4,)), ('W_Qd', (4, 2)), ('W_dec', (4, 4)), ('b_dec', (4,))])
def mae_forward(theta, g, hidden=HIDDEN):
    vis, res = placeholder_read(theta, g, hidden)
    R = patchify(g)
    preds, se = {}, []
    for j in hidden:
        r = res[j]['z'] @ theta['W_dec'] + theta['b_dec']
        preds[j] = r; se.append((r - R[j]) ** 2)
    return {'vis': vis, 'read': res, 'pred': preds, 'loss': float(np.mean(se))}

def mae_loss(v, scenes=MAE_SCENES):
    th = MAE.unflat(v)
    return float(np.mean([mae_forward(th, SCENES[s])['loss'] for s in scenes]))

JEPA = Params([('m', (4,)), ('W_Qd', (4, 2)), ('W_pred', (4, 2)), ('b_pred', (2,))])
def jepa_target(g, j):
    return encode(g)[j + 1][:2]
def jepa_forward(theta, g, hidden=HIDDEN):
    vis, res = placeholder_read(theta, g, hidden)
    preds, targets, se = {}, {}, []
    for j in hidden:
        y = res[j]['z'] @ theta['W_pred'] + theta['b_pred']
        t = jepa_target(g, j)
        preds[j] = y; targets[j] = t; se.append((y - t) ** 2)
    return {'vis': vis, 'read': res, 'pred': preds, 'target': targets, 'loss': float(np.mean(se))}
def jepa_loss(v, scenes=MAE_SCENES):
    th = JEPA.unflat(v)
    return float(np.mean([jepa_forward(th, SCENES[s])['loss'] for s in scenes]))

def train(params, loss_fn, v0, steps, lr, keep):
    opt = Adam(len(v0), lr); v = v0.copy(); curve = []; snaps = {}
    for t in range(steps + 1):
        L = loss_fn(v); curve.append([t, L])
        if t in keep: snaps[t] = params.unflat(v)
        if t == steps: break
        v = opt.step(v, fd_grad(loss_fn, v))
    return curve, snaps

# ---------------------------------------------------------------- DINO
def cls_feature(g): return encode(g)[0]
DINO_SCENES = ['A', 'B', 'D', 'E']
DINO_VIEWS = [('flip', flip), ('crop+dim', lambda g: dim(crop(g)))]
TAU_S, TAU_T_SHARP, MOMENTUM, CENTER_MOMENTUM = 1.0, 0.5, 0.9, 0.9
HEAD = Params([('W', (4, 3)), ('b', (3,))])

def head_logits(f, W, b): return f @ W + b

def dino_loss_batch(student_v, teacher_v, center, tau_t, feats, use_center):
    S, T = HEAD.unflat(student_v), HEAD.unflat(teacher_v)
    total = 0.0
    for f1, f2 in feats:
        zt1, zt2 = head_logits(f1, T['W'], T['b']), head_logits(f2, T['W'], T['b'])
        zs1, zs2 = head_logits(f1, S['W'], S['b']), head_logits(f2, S['W'], S['b'])
        c = center if use_center else np.zeros(3)
        pt1, pt2 = softmax((zt1 - c) / tau_t), softmax((zt2 - c) / tau_t)
        ls1, ls2 = np.log(softmax(zs1 / TAU_S)), np.log(softmax(zs2 / TAU_S))
        total += 0.5 * (-(pt2 * ls1).sum() - (pt1 * ls2).sum())
    return total / len(feats)

def dino_run(use_center, sharpen, steps=200, lr=0.5, keep=(0, 20, 200), seed=1):
    rng = np.random.default_rng(seed)
    W0 = rng.normal(0, .5, (4, 3)); b0 = np.zeros(3)
    student = HEAD.flat({'W': W0, 'b': b0}); teacher = student.copy(); center = np.zeros(3)
    tau_t = TAU_T_SHARP if sharpen else TAU_S
    feats = [(cls_feature(DINO_VIEWS[0][1](SCENES[s])), cls_feature(DINO_VIEWS[1][1](SCENES[s]))) for s in DINO_SCENES]
    curve, entropy, snaps = [], [], {}
    def outputs(sv, tv, c):
        S, T = HEAD.unflat(sv), HEAD.unflat(tv)
        rows = []
        for (f1, f2), s in zip(feats, DINO_SCENES):
            cc = c if use_center else np.zeros(3)
            rows.append({'scene': s,
                         'student_view1': softmax(head_logits(f1, S['W'], S['b']) / TAU_S).tolist(),
                         'teacher_view2': softmax((head_logits(f2, T['W'], T['b']) - cc) / tau_t).tolist(),
                         'teacher_logits_view2': head_logits(f2, T['W'], T['b']).tolist()})
        return rows
    for t in range(steps + 1):
        L = dino_loss_batch(student, teacher, center, tau_t, feats, use_center)
        curve.append([t, float(L)])
        T = HEAD.unflat(teacher)
        ent = 0.0
        for f1, f2 in feats:
            cc = center if use_center else np.zeros(3)
            for f in (f1, f2):
                p = softmax((head_logits(f, T['W'], T['b']) - cc) / tau_t); ent += float(-(p * np.log(p)).sum())
        entropy.append([t, ent / (2 * len(feats))])
        if t in keep:
            snaps[t] = {'student': HEAD.unflat(student), 'teacher': HEAD.unflat(teacher), 'center': center.copy(), 'outputs': outputs(student, teacher, center)}
        if t == steps: break
        grad = fd_grad(lambda v: dino_loss_batch(v, teacher, center, tau_t, feats, use_center), student)
        student = student - lr * grad
        teacher = MOMENTUM * teacher + (1 - MOMENTUM) * student
        Tn = HEAD.unflat(teacher)
        batch_mean = np.mean([head_logits(f, Tn['W'], Tn['b']) for f1, f2 in feats for f in (f1, f2)], axis=0)
        center = CENTER_MOMENTUM * center + (1 - CENTER_MOMENTUM) * batch_mean
    return {'center': use_center, 'sharpen': sharpen, 'tau_t': tau_t, 'curve': curve, 'entropy': entropy, 'snapshots': snaps}

# ---------------------------------------------------------------- linear probe
PROBE_SCENES = [('A', 1), ('B', 0), ('C', 0)]
PROBE_TRAIN_VIEWS = ['identity', 'flip', 'dim']
PROBE_TEST_VIEWS = ['crop']
def pooled(g): return encode(g)[1:].mean(axis=0)[:2]
def probe_points():
    pts = []
    for s, y in PROBE_SCENES:
        for name in PROBE_TRAIN_VIEWS + PROBE_TEST_VIEWS:
            f = pooled(VIEWS[name](SCENES[s]))
            pts.append({'scene': s, 'view': name, 'label': y, 'feature': f.tolist(), 'split': 'train' if name in PROBE_TRAIN_VIEWS else 'test'})
    return pts
def probe_train(pts, steps=400, lr=1.0):
    X = np.array([p['feature'] for p in pts if p['split'] == 'train']); y = np.array([p['label'] for p in pts if p['split'] == 'train'], float)
    w = np.zeros(2); b = 0.0; curve = []
    for t in range(steps + 1):
        z = X @ w + b; p = 1 / (1 + np.exp(-z))
        L = float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))); curve.append([t, L])
        if t == steps: break
        gw = ((p - y)[:, None] * X).mean(axis=0); gb = float((p - y).mean())
        w = w - lr * gw; b = b - lr * gb
    return w, b, curve

# ---------------------------------------------------------------- assemble
def r(x, d=6):
    if isinstance(x, np.ndarray): x = x.tolist()
    if isinstance(x, (list, tuple)): return [r(v, d) for v in x]
    if isinstance(x, dict): return {k: r(v, d) for k, v in x.items()}
    if isinstance(x, (float, np.floating)): return round(float(x), d)
    if isinstance(x, (np.integer,)): return int(x)
    return x

def build():
    rng = np.random.default_rng(0)
    mae0 = {'m': rng.normal(0, .1, 4), 'W_Qd': rng.normal(0, .1, (4, 2)), 'W_dec': rng.normal(0, .1, (4, 4)), 'b_dec': np.zeros(4)}
    mae_curve, mae_snaps = train(MAE, mae_loss, MAE.flat(mae0), 100, 0.05, {0, 10, 100})
    rng = np.random.default_rng(2)
    jepa0 = {'m': rng.normal(0, .1, 4), 'W_Qd': rng.normal(0, .1, (4, 2)), 'W_pred': rng.normal(0, .1, (4, 2)), 'b_pred': np.zeros(2)}
    jepa_curve, jepa_snaps = train(JEPA, jepa_loss, JEPA.flat(jepa0), 100, 0.05, {0, 10, 100})

    def mae_block(step):
        th = mae_snaps[step]; per = {}
        for s in MAE_SCENES:
            f = mae_forward(th, SCENES[s])
            per[s] = {'loss': f['loss'], 'pred': {str(j): f['pred'][j].tolist() for j in HIDDEN},
                      'alpha': {str(j): f['read'][j]['alpha'].tolist() for j in HIDDEN},
                      'z': {str(j): f['read'][j]['z'].tolist() for j in HIDDEN}}
        return {'params': {k: v.tolist() for k, v in th.items()}, 'scenes': per, 'mean_loss': float(np.mean([per[s]['loss'] for s in MAE_SCENES]))}
    def jepa_block(step):
        th = jepa_snaps[step]; per = {}
        for s in MAE_SCENES:
            f = jepa_forward(th, SCENES[s])
            per[s] = {'loss': f['loss'], 'pred': {str(j): f['pred'][j].tolist() for j in HIDDEN}, 'target': {str(j): f['target'][j].tolist() for j in HIDDEN}}
        return {'params': {k: v.tolist() for k, v in th.items()}, 'scenes': per, 'mean_loss': float(np.mean([per[s]['loss'] for s in MAE_SCENES]))}

    runs = {}
    for name, (c, s) in {'center_sharpen': (True, True), 'nocenter_sharpen': (False, True), 'center_nosharpen': (True, False), 'nocenter_nosharpen': (False, False)}.items():
        run = dino_run(c, s)
        runs[name] = {'center': c, 'sharpen': s, 'tau_t': run['tau_t'], 'curve': run['curve'], 'entropy': run['entropy'],
                      'checkpoints': {str(t): {'student': {k: v.tolist() for k, v in snap['student'].items()}, 'teacher': {k: v.tolist() for k, v in snap['teacher'].items()},
                                               'center': snap['center'].tolist(), 'outputs': snap['outputs']} for t, snap in run['snapshots'].items()}}
    pts = probe_points(); w, b, pcurve = probe_train(pts)
    for p in pts:
        z = float(np.dot(w, p['feature']) + b); p['p_two'] = 1 / (1 + math.exp(-z)); p['correct'] = int((p['p_two'] >= .5) == (p['label'] == 1))
    # a second fit that also sees the crop views (the pause-and-think question: would a line separate all twelve?)
    all_pts = [dict(q, split='train') for q in pts]; w2, b2, curve2 = probe_train(all_pts)
    for p in all_pts:
        z = float(np.dot(w2, p['feature']) + b2); p['p_two'] = 1 / (1 + math.exp(-z)); p['correct'] = int((p['p_two'] >= .5) == (p['label'] == 1))
    probe = {'w': w.tolist(), 'b': float(b), 'curve': pcurve, 'points': pts, 'axes': ['brightness', 'contrast (left minus right)'],
             'train_accuracy': float(np.mean([p['correct'] for p in pts if p['split'] == 'train'])), 'test_accuracy': float(np.mean([p['correct'] for p in pts if p['split'] == 'test'])),
             'all': {'w': w2.tolist(), 'b': float(b2), 'final_loss': curve2[-1][1], 'accuracy': float(np.mean([p['correct'] for p in all_pts])), 'p_two': [p['p_two'] for p in all_pts]}}

    toy = {
        'd_model': 4, 'd_k': 2, 'd_v': 2,
        'axes': toy5['axes'],
        'question': 'Vision I trained this encoder to answer: is there a mug on the right half? Vision II keeps it frozen.',
        'encoder': toy5['trained'],
        'hidden': HIDDEN,
        'mae': {'scenes': MAE_SCENES, 'hidden': HIDDEN, 'decoder_axes': ['read: brightness', 'read: contrast', 'row', 'col'], 'query_axes': toy5['axes']['qk'],
                'pixel_names': ['top left', 'top right', 'bottom left', 'bottom right'], 'optimizer': 'Adam', 'lr': 0.05, 'steps': 100, 'seed': 0,
                'curve': mae_curve, 'checkpoints': {str(t): mae_block(t) for t in (0, 10, 100)}},
        'jepa': {'scenes': MAE_SCENES, 'hidden': HIDDEN, 'target_axes': toy5['axes']['e'][:2], 'optimizer': 'Adam', 'lr': 0.05, 'steps': 100, 'seed': 2,
                 'curve': jepa_curve, 'checkpoints': {str(t): jepa_block(t) for t in (0, 10, 100)}},
        'dino': {'scenes': DINO_SCENES, 'views': [v[0] for v in DINO_VIEWS], 'slots': 3, 'tau_s': TAU_S, 'tau_t_sharpen': TAU_T_SHARP, 'momentum': MOMENTUM,
                 'center_momentum': CENTER_MOMENTUM, 'lr': 0.5, 'steps': 200, 'seed': 1, 'log3': math.log(3), 'runs': runs},
        'probe': probe,
        'sources': {'mae': 'https://arxiv.org/abs/2111.06377', 'dino': 'https://arxiv.org/abs/2104.14294', 'ijepa': 'https://arxiv.org/abs/2301.08243', 'dinov2': 'https://arxiv.org/abs/2304.07193'},
        'notes': 'Vision II toy. The encoder is the frozen Vision I encoder (toy5.json trained). MAE: the visible rows (CLS + 12 patches) go through the encoder; for each hidden patch a placeholder u_j = m + p_j asks a learned question q_j = u_j W_Qd, reads the visible patches through the frozen W_K, W_V, and the 4-number row z_j = (message, row, col) predicts 4 pixels through W_dec, b_dec; masked-pixel MSE on scenes A, B, D with the right-mug patches hidden; Adam 0.05, 100 steps, seed 0, central-difference gradients. I-JEPA: the same read, a 4-to-2 head, target = the full-image encoder row of the hidden patch on (brightness, contrast). DINO: frozen CLS features of two views (flip; crop then x0.75), a 4x3 head plus bias, student SGD 0.5, teacher EMA 0.9, centre EMA 0.9, tau_s 1, tau_t 0.5 when sharpening, symmetrised cross-entropy over scenes A, B, D, E; four runs switch centring and sharpening. Probe: logistic regression on the mean patch row (brightness, contrast) of the frozen encoder; train on identity/flip/dim views of A (two mugs) versus B, C (one mug), test on the crop views.'
    }
    return r(toy)

def compare(a, b, path='', tol=1e-6, stats=None):
    stats = stats if stats is not None else {'n': 0, 'max': 0.0}
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a: compare(a[k], b[k], path + '/' + k, tol, stats)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): compare(x, y, path + '[%d]' % i, tol, stats)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        d = abs(float(a) - float(b)); stats['n'] += 1; stats['max'] = max(stats['max'], d)
        assert d <= tol, '%s: %r != %r' % (path, a, b)
    else:
        assert a == b, path
    return stats

if __name__ == '__main__':
    toy = build()
    if '--check' in sys.argv:
        saved = json.load(open(OUT, encoding='utf-8'))
        st = compare(toy, saved)
        print(json.dumps({'reproduced': st['n'], 'maxError': st['max'], 'mae': [toy['mae']['checkpoints'][k]['mean_loss'] for k in ('0', '10', '100')],
                          'jepa': [toy['jepa']['checkpoints'][k]['mean_loss'] for k in ('0', '10', '100')],
                          'dino_final': {k: v['curve'][-1][1] for k, v in toy['dino']['runs'].items()}, 'probe': [toy['probe']['train_accuracy'], toy['probe']['test_accuracy']]}))
    else:
        json.dump(toy, open(OUT, 'w', encoding='utf-8'), indent=1)
        print(json.dumps({'wrote': OUT, 'mae': [toy['mae']['checkpoints'][k]['mean_loss'] for k in ('0', '10', '100')],
                          'jepa': [toy['jepa']['checkpoints'][k]['mean_loss'] for k in ('0', '10', '100')],
                          'dino_final': {k: v['curve'][-1][1] for k, v in toy['dino']['runs'].items()}, 'log3': math.log(3),
                          'probe': [toy['probe']['train_accuracy'], toy['probe']['test_accuracy']]}))
