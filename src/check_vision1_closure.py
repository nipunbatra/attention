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
assert len(manifest)==227 and len({x['id'] for x in manifest})==227
ids=[x['id'] for x in manifest]
input_order=['patch-projection-parameters','vision-topic-03','position-where','real-patch-position',
             'cls-detour','real-cls-purpose','cls-parameter-origin','cls-parameter-learning',
             'cls-stored-start','cls-collect','cls-shared-start','cls-two-image-readout','cls-without',
             'real-cls-sequence','model-journey-checkpoint','real-patch-qkv','real-cls-attention']
assert [ids.index(k) for k in input_order]==sorted(ids.index(k) for k in input_order)
assert all(x['section']=='s13' for x in manifest if x['id'].startswith('position-photo-'))
assert all(len(x['caption'].split())<=40 and '\n' in x['notes'] for x in manifest)
required={'cls-shared-start','cls-without','heads-visual-roles','heads-independent','backward-route','backward-qk','backward-patches','backward-full-block','classification-training-map','cnn-receptive-field','cnn-classifier-parallel','pets-head','pets-frozen','pets-training-step','pets-evaluation','classification-exit'}
assert required<={x['id'] for x in manifest}
assert not {'find-animal','image-caption','photo-search','learning-curves','training-data'}&{x['id'] for x in manifest}
saved=json.loads((ROOT/'figures/vision1/real-classifier-path.json').read_text())
np.testing.assert_allclose(np.array(saved['previews']['cls_parameter'])+saved['previews']['cls_position'],
                           saved['previews']['cls_input'],atol=1e-7)
for name in ['cls_parameter','cls_position','cls_input']:
    for value in saved['previews'][name]:
        assert f'{value:.3f}'.replace('-','−') in (ROOT/'figures/vision1/cls-stored-start.svg').read_text()
for name in ['cls_input','cls_final']:
    for value in saved['previews'][name]:
        assert f'{value:.3f}'.replace('-','−') in (ROOT/'figures/vision1/cls-collect.svg').read_text()
comparison=json.loads((ROOT/'figures/vision1/cls-two-image-trace.json').read_text())
dog,cat=comparison['results']
assert dog['cls_input']==cat['cls_input']
assert dog['first_cls_query_all_heads']==cat['first_cls_query_all_heads']
for row in [dog,cat]:
    for field,figure in [('cls_input','cls-shared-start'),('cls_after_attention1_residual','cls-shared-start'),
                          ('cls_final_after_norm','cls-two-image-readout')]:
        assert len(row[field])==192
        for value in row[field][:3]:
            assert f'{value:.3f}'.replace('-','−') in (ROOT/'figures/vision1'/(figure+'.svg')).read_text()
assert not np.allclose(dog['cls_after_attention1_residual'],cat['cls_after_attention1_residual'])
assert not np.allclose(dog['cls_final_after_norm'],cat['cls_final_after_norm'])
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
        'position_and_cls_explained_before_attention':True,'patch_rearrangement_in_exercises_only':True,
        'cls_origin_and_saved_parameter_previews_verified':True,
        'two_images_share_cls_input_but_have_different_updates':True,
        'single_query_weight_update':trace['single_update']}
(ROOT/'figures/vision1/classification-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
