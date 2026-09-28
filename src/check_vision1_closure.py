"""Check the new teaching calculations and generated lecture without PyTorch."""
from pathlib import Path
from html.parser import HTMLParser
import json
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from vision1_gradients import checked_trace
from vision1_worksheet import forward,parameters

ROOT=Path(__file__).resolve().parents[1]
trace=checked_trace()
base=forward(parameters()['images']['horizontal'])
for actual,old in [('U','updated'),('J','joined'),('z','logits'),('p','probability')]:
    np.testing.assert_allclose(trace['forward'][actual],base[old],atol=1e-12)
for h in trace['gradients']['heads']:
    np.testing.assert_allclose(np.array(h['dS']).sum(axis=1),0,atol=1e-12)
assert trace['single_update']['loss_after']<trace['single_update']['loss_before']
manifest=json.loads((ROOT/'figures/vision1/frame-manifest.json').read_text())
assert len(manifest)==220 and len({x['id'] for x in manifest})==220
assert all(len(x['caption'].split())<=40 and '\n' in x['notes'] for x in manifest)
required={'cls-shared-start','cls-without','heads-visual-roles','heads-independent','backward-route','backward-qk','backward-patches','backward-full-block','classification-training-map','cnn-receptive-field','cnn-classifier-parallel','pets-head','pets-frozen','pets-training-step','pets-evaluation','classification-exit'}
assert required<={x['id'] for x in manifest}
assert not {'find-animal','image-caption','photo-search','learning-curves','training-data'}&{x['id'] for x in manifest}
for x in manifest:
    ET.parse(ROOT/'figures/vision1'/(x['id']+'.svg'))
page=(ROOT/'vision1.html').read_text()
class Links(HTMLParser):
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='a' and 'href' in a:
            p=a['href'].split('#')[0].split('?')[0]
            if p and not p.startswith(('https:','http:','mailto:')):assert (ROOT/p).exists(),p
Links().feed(page)
assert '<audio' not in page and '<video' not in page
js="""const p=JSON.parse(require('fs').readFileSync(process.argv[1],'utf8'));const {forward}=require(process.argv[2]);console.log(JSON.stringify(Object.keys(p.images).flatMap(name=>[true,false].map(pos=>forward(p,name,pos)))));"""
actual=json.loads(subprocess.check_output(['node','-e',js,str(ROOT/'src/vision1-worksheet.json'),str(ROOT/'src/vision1-lesson.js')],text=True))
for i,(name,image) in enumerate(parameters()['images'].items()):
    for j,pos in enumerate([True,False]):
        expected=forward(image,pos)
        for k in ['E','joined','delta','updated','logits','probability']:np.testing.assert_allclose(actual[2*i+j][k],expected[k],atol=1e-12)
report={'teaching_frames':len(manifest),'new_gradient_coordinates_checked':trace['verification']['finite_difference_coordinates'],
        'gradient_max_absolute_error':trace['verification']['max_absolute_error'],'original_forward_and_js_match':True,
        'valid_svg_and_local_links':True,'caption_limit_and_notes':True,'silent':True,'training_jobs_run':False,
        'single_query_weight_update':trace['single_update']}
(ROOT/'figures/vision1/classification-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
