#!/usr/bin/env python3
"""Read-only independent checks for Vision I's complete training computation.

Run ``python3 src/verify_vision1_learning.py``. Requires NumPy and Node, but no
training framework. Reproduces the browser's 600 full-batch SGD updates with
independent NumPy operations; compares forward tensors, every reverse-graph
tensor, all parameter gradients, and the unchanged original encoder. Central
finite differences test every one of the 44 trainable scalar parameters for
each image and the mean loss, at three different parameter snapshots.

Nothing is installed or written. This checks fitting the two training images;
it provides no evidence of performance on unseen images.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ('W_patch', 'b_patch', 'cls', 'positions', 'W_Q', 'W_K', 'W_V', 'W_O', 'W_class', 'b_class')
EPSILON = 1e-5
GRADIENT_TOLERANCE = 2e-8
PARITY_TOLERANCE = 2e-10


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def forward(p, image, target):
    image = np.asarray(image, dtype=float)
    patches = np.stack([image[y:y+2, x:x+2].reshape(-1) for y in (0, 2) for x in (0, 2)])
    embeddings = patches @ p['W_patch'] + p['b_patch']
    content = np.vstack([p['cls'], embeddings])
    positions = p['positions'].copy()
    E = content + positions
    Q, K, V = (E @ p['W_' + role] for role in ('Q', 'K', 'V'))
    raw = Q @ K.T
    S = raw / math.sqrt(2)
    A = softmax(S)
    message = A @ V
    delta = message @ p['W_O']
    updated = E + delta
    logits = updated[0] @ p['W_class'] + p['b_class']
    probs = softmax(logits)
    peak = logits.max()
    loss = float(peak + np.log(np.exp(logits - peak).sum()) - logits[target])
    return dict(image=image, patches=patches, embeddings=embeddings, content=content,
                positions=positions, E=E, Q=Q, K=K, V=V, raw=raw, S=S, A=A,
                message=message, delta=delta, updated=updated, logits=logits,
                probs=probs, loss=loss, target=target)


def backward(p, image, target):
    f = forward(p, image, target)
    dz = f['probs'].copy()
    dz[target] -= 1
    g = {'W_class': np.outer(f['updated'][0], dz), 'b_class': dz.copy()}
    du = np.zeros_like(f['updated'])
    du[0] = dz @ p['W_class'].T
    g['W_O'] = f['message'].T @ du
    dm = du @ p['W_O'].T
    da = dm @ f['V'].T
    dv = f['A'].T @ dm
    ds = f['A'] * (da - (f['A'] * da).sum(axis=1, keepdims=True))
    draw = ds / math.sqrt(2)
    dq, dk = draw @ f['K'], draw.T @ f['Q']
    g['W_Q'], g['W_K'], g['W_V'] = f['E'].T @ dq, f['E'].T @ dk, f['E'].T @ dv
    deq, dek, dev = dq @ p['W_Q'].T, dk @ p['W_K'].T, dv @ p['W_V'].T
    de = du + deq + dek + dev
    dembed = de[1:]
    g['positions'], g['cls'] = de.copy(), de[0].copy()
    g['W_patch'], g['b_patch'] = f['patches'].T @ dembed, dembed.sum(axis=0)
    dp = dembed @ p['W_patch'].T
    dimage = np.zeros((4, 4))
    for j, (y, x) in enumerate(( (0, 0), (0, 2), (2, 0), (2, 2) )):
        dimage[y:y+2, x:x+2] = dp[j].reshape(2, 2)
    graph = dict(logits=dz, updated=du, delta=du, message=dm, A=da, V=dv,
                 S=ds, raw=draw, Q=dq, K=dk, E=de, EResidual=du, EQuery=deq,
                 EKey=dek, EValue=dev, content=de, embeddings=dembed, patches=dp, image=dimage)
    return dict(loss=f['loss'], grads=g, forward=f, graph=graph)


def batch(p, images, targets):
    examples = [backward(p, im, target) for im, target in zip(images, targets)]
    grads = {k: sum(ex['grads'][k] for ex in examples) / len(examples) for k in NAMES}
    losses = [ex['loss'] for ex in examples]
    return dict(loss=float(np.mean(losses)), losses=losses, grads=grads, examples=examples)


def step(p, grads, lr):
    return {name: p[name] - lr * grads[name] for name in NAMES}


def stage(p, images, targets):
    fs = [forward(p, im, target) for im, target in zip(images, targets)]
    losses = [f['loss'] for f in fs]
    return dict(params=p, forwards=fs, losses=losses, meanLoss=float(np.mean(losses)),
                predictions=[int(np.argmax(f['probs'])) for f in fs])


def compare(actual, expected, label):
    """Shape/key-aware parity checking without rounding away small errors."""
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f'{label}: key mismatch {set(actual) ^ set(expected)}')
        return max((compare(actual[k], expected[k], f'{label}.{k}') for k in expected), default=0.)
    if isinstance(expected, (list, tuple)) and expected and isinstance(expected[0], dict):
        assert len(actual) == len(expected), f'{label}: list length mismatch'
        return max(compare(a, b, f'{label}[{i}]') for i, (a, b) in enumerate(zip(actual, expected)))
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, f'{label}: shape mismatch {a.shape} vs {b.shape}'
    assert np.isfinite(a).all() and np.isfinite(b).all(), f'{label}: nonfinite value'
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    assert error < PARITY_TOLERANCE, f'{label}: max error {error:.3g}'
    return error


def browser_results():
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Node is required to check the browser implementation.')
    script = r'''
const fs = require('fs'), vm = require('vm'), path = require('path');
const src = process.argv[1];
global.window = {AT:{axes:{},objects:[],notation:[]},__TOY__:JSON.parse(fs.readFileSync(path.join(src,'toy5.json'),'utf8'))};
vm.runInThisContext(fs.readFileSync(path.join(src,'part5.js'),'utf8'));
const T=window.AT.vision, untouched=JSON.stringify(T.data), oldForward=T.forward();
const start=performance.now();
vm.runInThisContext(fs.readFileSync(path.join(src,'part5-learning.js'),'utf8'));
const runtimeMs=performance.now()-start, L=T.learning;
if (JSON.stringify(T.data)!==untouched) throw new Error('The original model was mutated.');
const examples={};
for (const name of ['before','afterSingle','afterTraining']) {
  examples[name]=L.images.map((im,i)=>L.backward(L.experiment[name].params,im,L.targets[i]));
}
const p0=JSON.stringify(L.initial), grads0=JSON.stringify(L.experiment.firstBatch.grads);
L.step(L.initial,L.experiment.firstBatch.grads,L.experiment.lr);
if (JSON.stringify(L.initial)!==p0 || JSON.stringify(L.experiment.firstBatch.grads)!==grads0) throw new Error('SGD mutated its inputs.');
console.log(JSON.stringify({initial:L.initial,images:L.images,targets:L.targets,classes:L.classes,
  experiment:L.experiment,examples,oldForward,runtimeMs}));
'''
    result = subprocess.run([node, '-e', script, str(HERE)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def finite_differences(p, images, targets):
    """Compare analytic gradients to independently perturbed forward losses."""
    checked, max_error, worst = 0, 0., ''
    b = batch(p, images, targets)
    parameter_count = sum(p[name].size for name in NAMES)
    assert parameter_count == 44, f'Expected 44 trainable scalar parameters, got {parameter_count}'
    for which in (0, 1, 'mean'):
        grads = b['grads'] if which == 'mean' else b['examples'][which]['grads']

        def loss():
            if which == 'mean':
                return np.mean([forward(p, im, t)['loss'] for im, t in zip(images, targets)])
            return forward(p, images[which], targets[which])['loss']

        for name in NAMES:
            for index in np.ndindex(p[name].shape):
                original = p[name][index]
                try:
                    p[name][index] = original + EPSILON
                    plus = loss()
                    p[name][index] = original - EPSILON
                    minus = loss()
                finally:
                    p[name][index] = original
                numeric = (plus-minus) / (2*EPSILON)
                error = abs(float(numeric-grads[name][index]))
                checked += 1
                label = f'{which}:{name}{index}'
                assert math.isfinite(error) and error < GRADIENT_TOLERANCE, (
                    f'{label}: analytic {grads[name][index]:.12g}, finite difference {numeric:.12g}, error {error:.3g}')
                if error > max_error:
                    max_error, worst = error, label
    assert checked == 3*parameter_count
    return checked, max_error, worst


def main():
    js = browser_results()
    data = json.loads((HERE / 'toy5.json').read_text())['vision']
    initial = {name: np.array(data[name], dtype=float) for name in NAMES}
    images = [data['image'], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]]]
    targets = [0, 1]
    assert js['classes'] == ['two blocks', 'one block'] and js['targets'] == targets
    compare(js['images'], images, 'training images')
    compare(js['initial'], initial, 'initial parameters')
    initial_f = forward(initial, images[0], 0)
    # Compare the old worksheet too: trainable snapshots must not change it.
    compare({k: js['oldForward'][k] for k in initial_f if k != 'target'},
            {k: v for k, v in initial_f.items() if k != 'target'}, 'unchanged default forward')

    exp = js['experiment']
    assert exp['lr'] == .05 and exp['nsteps'] == 600 and exp['batchSize'] == 2 and exp['parameterCount'] == 44
    assert exp['objective'] == 'mean cross-entropy on these two training images'
    p = {name: x.copy() for name, x in initial.items()}
    first = batch(p, images, targets)
    worst_parity = compare(exp['firstBatch'], first, 'firstBatch')
    stages = {'before': stage(p, images, targets)}
    history = [dict(step=0, loss=first['loss'], losses=first['losses'])]
    for it in range(1, exp['nsteps']+1):
        p = step(p, first['grads'] if it == 1 else batch(p, images, targets)['grads'], exp['lr'])
        if it == 1:
            stages['afterSingle'] = stage(p, images, targets)
        if it == 1 or it % 10 == 0 or it == exp['nsteps']:
            fs = stage(p, images, targets)
            history.append(dict(step=it, loss=fs['meanLoss'], losses=fs['losses']))
    stages['afterTraining'] = stage(p, images, targets)
    worst_parity = max(worst_parity, compare(exp['history'], history, 'history'))

    checked, gradient_error, worst_gradient = 0, 0., ''
    for name, snapshot in stages.items():
        worst_parity = max(worst_parity, compare(exp[name], snapshot, name))
        independent = [backward(snapshot['params'], im, t) for im, t in zip(images, targets)]
        worst_parity = max(worst_parity, compare(js['examples'][name], independent, f'{name}.reverseGraph'))
        count, error, path = finite_differences(snapshot['params'], images, targets)
        checked += count
        if error > gradient_error:
            gradient_error, worst_gradient = error, f'{name}:{path}'

    for cell in exp['selectedCells']:
        def read(params):
            return params[cell['name']][tuple(cell['index'])]
        assert cell['label'] == cell['name'] + ''.join(f'[{i}]' for i in cell['index'])
        for key, source in [('before', initial), ('gradient', first['grads']),
                            ('afterSingle', stages['afterSingle']['params']), ('afterTraining', p)]:
            worst_parity = max(worst_parity, compare(cell[key], read(source), f'{cell["label"]}.{key}'))
        assert abs(cell['afterSingle'] - (cell['before']-exp['lr']*cell['gradient'])) < 1e-12

    assert stages['afterSingle']['meanLoss'] < stages['before']['meanLoss']
    assert stages['afterSingle']['losses'][0] > stages['before']['losses'][0], 'The lesson must disclose the first-image tradeoff.'
    assert stages['afterTraining']['predictions'] == targets
    assert all(after < before for after, before in zip(stages['afterTraining']['losses'], stages['before']['losses']))
    assert all(f['probs'][target] > .99 for f, target in zip(stages['afterTraining']['forwards'], targets))
    print(f'Browser experiment: {exp["nsteps"]} full-batch SGD updates, lr={exp["lr"]}, 44 parameters; {js["runtimeMs"]:.1f} ms')
    print(f'NumPy/browser parity: every forward/reverse tensor and training checkpoint; max error {worst_parity:.3g}')
    print(f'Central differences: {checked} checks (44 parameters × 3 objectives × 3 snapshots), epsilon={EPSILON:g}')
    print(f'Max gradient error: {gradient_error:.3g} at {worst_gradient}')
    for name, s in stages.items():
        print(f'{name}: losses {s["losses"][0]:.9f}, {s["losses"][1]:.9f}; mean {s["meanLoss"]:.9f}; predictions {s["predictions"]}')
    print('PASS: original encoder unchanged; both training images fitted; no held-out/generalization claim; no files written')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (AssertionError, KeyError, TypeError, ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        sys.exit(1)
