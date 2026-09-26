"""Numerical and artifact checks for the new worksheet, independent of the UI."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import subprocess
import sys
import numpy as np
import torch
from vision1_worksheet import parameters, forward

ROOT=Path(__file__).resolve().parents[1]
p=parameters()
layer=torch.nn.MultiheadAttention(4,2,bias=False,batch_first=True,dtype=torch.float64)
with torch.no_grad():
    layer.in_proj_weight.copy_(torch.tensor(np.concatenate([
        np.concatenate([h[key] for h in p['heads']],axis=1).T
        for key in ['W_Q','W_K','W_V']],axis=0)))
    layer.out_proj.weight.copy_(torch.tensor(np.array(p['W_O']).T))
layer.eval()
cases=[]
for name,image in p['images'].items():
    for positions in [True,False]:
        r=forward(image,positions)
        E=torch.tensor(r['E']).unsqueeze(0)
        with torch.no_grad():
            delta,weights=layer(E,E,E,need_weights=True,average_attn_weights=False)
            updated=E+delta
            logits=updated[0,0]@torch.tensor(p['W_class'],dtype=torch.float64)
            probability=logits.softmax(-1)
        np.testing.assert_allclose(delta[0],r['delta'],atol=1e-12)
        np.testing.assert_allclose(probability,r['probability'],atol=1e-12)
        for h in range(2):
            np.testing.assert_allclose(weights[0,h],r['heads'][h]['A'],atol=1e-12)
            np.testing.assert_allclose(r['heads'][h]['A'].sum(-1),1,atol=1e-12)
        if not positions:np.testing.assert_allclose(probability,[.5,.5],atol=1e-12)
        else:assert probability.argmax().item()==(['horizontal','vertical'].index(name))
        cases.append({'image':name,'positions':positions,'probability':probability.tolist()})

# The browser runtime uses a separate JavaScript implementation.
js="""const p=JSON.parse(require('fs').readFileSync(process.argv[1],'utf8'));
const {forward}=require(process.argv[2]);
console.log(JSON.stringify(Object.keys(p.images).flatMap(name=>[true,false].map(pos=>forward(p,name,pos)))));"""
actual=json.loads(subprocess.check_output(['node','-e',js,str(ROOT/'src/vision1-worksheet.json'),str(ROOT/'src/vision1-lesson.js')],text=True))
for i,(name,image) in enumerate(p['images'].items()):
    for j,positions in enumerate([True,False]):
        expected=forward(image,positions)
        for key in ['patches','content','E','joined','delta','updated','logits','probability']:
            np.testing.assert_allclose(actual[2*i+j][key],expected[key],atol=1e-12)
        for h in range(2):
            for key in ['Q','K','V','scores','A','H']:
                np.testing.assert_allclose(actual[2*i+j]['heads'][h][key],expected['heads'][h][key],atol=1e-12)

manifest=json.loads((ROOT/'figures/vision1/frame-manifest.json').read_text())
assert len(manifest)==140
required={'task-side-by-side','task-mask-reason','qkv-match-numbers','qkv-read-numbers','qkv-change-key','qkv-change-value','qkv-no-prompt','cls-start','cls-two-images','cls-learns','pooling-example','readout-choice'}
assert required <= {x['id'] for x in manifest}
assert all(len(x['caption'].split())<=40 and '\n' in x['notes'] for x in manifest)
report=json.loads((ROOT/'figures/vision1/real-inference.json').read_text())
for item in report['results']:
    path=ROOT/'figures/vision1'/f'{item["image_id"]}.jpg'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']

class Links(HTMLParser):
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='a' and 'href' in a:
            link=a['href'].split('#')[0].split('?')[0]
            if link and not link.startswith(('http:','https:','mailto:','#')):
                assert (ROOT/link).exists(),link
Links().feed((ROOT/'vision1.html').read_text())
notebook=json.loads((ROOT/'notebooks/vision/03_vision_transformer_lab.ipynb').read_text())
assert all(c.get('execution_count') for c in notebook['cells'] if c['cell_type']=='code')
assert len({x['id'] for x in manifest})==len(manifest)
assert all(x['section'] and x['frame'] for x in manifest)
inspection=json.loads((ROOT/'figures/vision1/inspection.json').read_text())
for record in inspection['attention']:
    np.testing.assert_allclose(np.sum(record['patch_weights'])+record['cls_weight'],1.,atol=2e-6)
assert len(inspection['occlusion'])==4
assert inspection['source_sha256']==report['results'][0]['sha256']
training=json.loads((ROOT/'figures/vision1/training.json').read_text())
assert training['split_sizes']==[512,128,256]
assert [r['test']['correct'] for r in training['results']]==[256,128]
assert training['results'][1]['max_paired_probability_difference']<1e-5
assert all(r['best_validation_epoch']<=training['epochs'] for r in training['results'])
page=(ROOT/'vision1.html').read_text()
assert '__VISION_SCENES__' not in page
assert '<audio' not in page and '<video' not in page
assert 'torch.nn' in str(notebook)
out={'attention_rows_normalized':True,'independent_split_sizes':training['split_sizes'],
     'position_control_correct':[r['test']['correct'] for r in training['results']],
     'silent':True,'numerical_parity':'NumPy, torch.nn.MultiheadAttention, and JavaScript agree within 1e-12',
     'cases':cases,'teaching_frames':len(manifest),'captions_at_most_40_words':True,
     'real_image_checksums_match':True,'local_links_exist':True,'notebook_executed':True}
(ROOT/'figures/vision1/checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
