"""Check the shipped browser math against independent PyTorch measurements."""
from pathlib import Path
import hashlib
import json
import subprocess
import re
import numpy as np
from vision1_explorer_tour import EXAMPLES

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'figures/vision1/attention-explorer'
meta = json.loads((folder/'manifest.json').read_text())
refs = json.loads((folder/'reference.json').read_text())
assert meta['verification']['training_run'] is False
assert meta['verification']['explicit_attention_matches_model'] is True
for name, field in [('newfoundland_31.jpg','source_sha256'),('model-input.png','display_sha256')]:
    assert hashlib.sha256((folder.parent/name).read_bytes()).hexdigest() == meta[field]
for entry in meta['files']:
    raw = (folder/entry['file']).read_bytes()
    assert len(raw) == entry['bytes'] == 453888
    assert hashlib.sha256(raw).hexdigest() == entry['sha256']

js = r'''
const fs=require('fs'),path=require('path');
const {decode,attention,similarity}=require(process.argv[1]);
const base=process.argv[2],refs=JSON.parse(fs.readFileSync(path.join(base,'reference.json')));
const blocks=new Map();
for(let b=1;b<=12;b++) {
 const bytes=fs.readFileSync(path.join(base,`block-${String(b).padStart(2,'0')}.f32`));
 blocks.set(b,decode(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)));
}
let rejected=false;try{decode(new ArrayBuffer(4))}catch{rejected=true}if(!rejected)throw Error('Bad data accepted');
console.log(JSON.stringify(refs.map(r => r.weights ? attention(blocks.get(r.block),r.head-1,r.query).values : similarity(blocks.get(r.block),r.query).values)));
'''
actual = json.loads(subprocess.check_output(['node','-e',js,str(ROOT/'src/vision1-inspector.js'),str(folder)],text=True))
max_error = 0
for ref, values in zip(refs, actual):
    expected = ref.get('weights', ref.get('similarities'))
    max_error = max(max_error, float(np.max(np.abs(np.array(values)-expected))))
    np.testing.assert_allclose(values,expected,atol=2e-6,rtol=2e-5)
    if 'weights' in ref:
        assert min(values) >= 0
        np.testing.assert_allclose(sum(values),1,atol=1e-12)
    else:
        np.testing.assert_allclose(values[ref['query']],1,atol=1e-12)
        assert min(values)>=-1 and max(values)<=1
assert len(actual) == 240
# A measured correspondence used by the teaching prompt, not a fabricated heatmap.
ear = next(values for ref,values in zip(refs,actual) if ref['block']==12 and ref['query']==74 and 'similarities' in ref)
top = sorted((j for j in range(1,197) if j!=74),key=lambda j:-ear[j])[:2]
assert top == [82,81]
manifest = json.loads((folder.parent/'frame-manifest.json').read_text())
ids = [frame['id'] for frame in manifest]
assert ids.count('read-attention-map')==1
assert not {'real-heads','real-depth','real-patch-query'} & set(ids)
page = (ROOT/'vision1.html').read_text()
assert 'id="vit-explorer"' in page and 'function attention(data, head, query)' in page
assert 'additionalRuntimeFiles' in page
assert len(EXAMPLES) == 9
assert json.loads(re.search(r'<script type="application/json" data-examples>(.*?)</script>', page, re.S).group(1)) == EXAMPLES
# The tour makes specific claims about corners, ear correspondence and weights.
# Check those against the saved measurements independently of the UI settings.
tour_values = []
for e in EXAMPLES:
    d = np.fromfile(folder/f'block-{e["block"]:02}.f32', dtype='<f4').astype(np.float64)
    if e['mode'] == 'similarity':
        f = d[75648:].reshape(197, 192)
        v = f @ f[e['query']] / (np.linalg.norm(f, axis=1)*np.linalg.norm(f[e['query']]))
    else:
        q, k = d[:37824].reshape(3,197,64), d[37824:75648].reshape(3,197,64)
        scores = q[e['head']-1,e['query']] @ k[e['head']-1].T / 8
        v = np.exp(scores-scores.max()); v /= v.sum()
    tour_values.append(v[e['source']])
    if e['id'] == 'corners':
        assert sorted(sorted((j for j in range(1,197) if j != 1), key=lambda j:-v[j])[:3]) == [14,183,196]
np.testing.assert_allclose(tour_values, [.7834,.9999,.9017,.9110,.9427,.3221,.0909,.0229,.2486], atol=5e-5, rtol=0)
print(f'PASS: 240 browser rows match PyTorch across all 12 blocks and 3 heads (max error {max_error:.2g}).')
print('PASS: data hashes, normalized weights, cosine self-matches, real ear correspondence, and consolidated slide route.')
print('PASS: all nine guided presets and their numerical teaching claims match the saved model data.')
