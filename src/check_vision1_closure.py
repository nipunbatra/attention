"""Check the new teaching calculations and generated lecture without PyTorch."""
from pathlib import Path
from html.parser import HTMLParser
import json
import math
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
assert len(manifest)==161 and len({x['id'] for x in manifest})==161
ids=[x['id'] for x in manifest]
input_order=['patch-projection-parameters','vision-topic-03','position-where','position-table','real-patch-position','position-learning',
             'cls-detour','real-cls-purpose','cls-parameter-origin','cls-parameter-learning',
             'cls-stored-start','cls-summary-refinement','cls-collect','cls-shared-start','cls-two-image-readout','cls-without',
             'cls-pool-dog','cls-pool-arithmetic','cls-readout-return',
             'real-cls-sequence','model-journey-checkpoint','real-patch-qkv','real-cls-attention',
             'real-attention-product','real-attention-cls-zoom','real-attention-weights','real-attention-mask',
             'real-cls-values-origin','real-cls-value-scaling','real-cls-value-sum','real-attention-values',
             'real-heads-intro','real-heads-qkv','real-heads-messages','real-heads-cls','real-heads-concat',
             'real-cls-message','real-cls-residual','real-cls-mlp','real-mlp-network','real-mlp-residual',
             'real-block-handoff','real-block-changes','real-cls-depth','real-cls-readout',
             'real-classifier-network','real-classifier-score','real-classifier-softmax','real-cls-prediction']
assert [ids.index(k) for k in input_order]==sorted(ids.index(k) for k in input_order)
assert all(x['section']=='s11' for x in manifest if x['id'].startswith('position-photo-'))
assert all(len(x['caption'].split())<=40 and '\n' in x['notes'] for x in manifest)
required={'task-side-by-side','cls-shared-start','cls-without','heads-visual-roles',
          'patch-filter-patterns','photo-key-value-check','photo-two-softmaxes',
          'photo-label-loss','photo-backward-head','photo-backward-block',
          'photo-backward-attention','photo-backward-inputs','photo-optimizer-step',
          'cnn-receptive-field','cnn-inductive-bias','cnn-vit-design',
          'code-photo-input','code-photo-conv','code-photo-tokens','code-photo-attention',
          'code-photo-block','code-photo-readout','code-photo-training',
          'pets-evaluation','classification-exit'}
assert required<={x['id'] for x in manifest}
assert not {'s02-small','s03-query','s04-probability','heads-independent','code-patch','code-model'} & {x['id'] for x in manifest}
assert sorted({x['section'] for x in manifest}) == [f's{i:02}' for i in range(1,13)]
assert not {'find-animal','image-caption','photo-search','learning-curves','training-data'}&{x['id'] for x in manifest}
saved=json.loads((ROOT/'figures/vision1/real-classifier-path.json').read_text())
assert saved['with_cls']==[197,192] and saved['mlp_hidden']==[197,768] and saved['block_count']==12
hidden=saved['previews']['cls_mlp_hidden']
np.testing.assert_allclose([.5*x*(1+math.erf(x/math.sqrt(2))) for x in hidden],
                           saved['previews']['cls_mlp_activated'],atol=2e-7)
np.testing.assert_allclose(np.array(saved['previews']['cls_after_attention'])+saved['previews']['cls_mlp_update'],
                           saved['previews']['cls_after_block1'],atol=1e-7)
for name in ['cls_mlp_hidden','cls_mlp_activated','cls_mlp_update']:
    for value in saved['previews'][name][:2]:
        assert f'{value:.3f}'.replace('-','−') in (ROOT/'figures/vision1/real-mlp-network.svg').read_text()
heads=json.loads((ROOT/'figures/vision1/multihead-cls-trace.json').read_text())
assert heads['input_shape']==[197,192] and len(heads['heads'])==3
for h in heads['heads']:
    assert all(len(h[k])==64 for k in ['cls_query','cls_key','cls_value','cls_message'])
    assert len(h['cls_weights'])==197
    np.testing.assert_allclose(sum(h['cls_weights']),1,atol=1e-6)
np.testing.assert_array_equal(np.concatenate([h['cls_message'] for h in heads['heads']]),heads['cls_joined'])
np.testing.assert_allclose(np.array(heads['cls_input'])+heads['cls_projected_update'],heads['cls_after_residual'],atol=1e-6)
np.testing.assert_allclose(heads['heads'][0]['cls_message'][:3],saved['previews']['cls_head1_message'],atol=2e-5)
assert all(not np.allclose(heads['heads'][i]['cls_message'],heads['heads'][j]['cls_message'])
           for i in range(3) for j in range(i+1,3))
assert heads['verification']['training_run'] is False
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
readout=json.loads((ROOT/'figures/vision1/classifier-readout-trace.json').read_text())
np.testing.assert_array_equal(readout['cls_final'],dog['cls_final_after_norm'])
assert readout['input_shape']==[1,192] and readout['output_shape']==[1,1000]
assert readout['weight_math_shape']==[192,1000] and readout['weight_pytorch_shape']==[1000,192]
logits=np.array(readout['logits'])
assert logits.shape==(1000,) and readout['bias_shape']==[1000]
shifted=np.exp(logits-logits.max())
probability=shifted/shifted.sum()
np.testing.assert_allclose(probability.sum(),1,atol=1e-12)
np.testing.assert_allclose(shifted.sum(),readout['softmax']['shifted_denominator'],atol=2e-7)
for item, ref in zip(readout['selected_classes'],saved['top3']):
    assert len(item['weights'])==192 and item['index']==ref['index']
    products=np.array(readout['cls_final'])*item['weights']
    np.testing.assert_allclose(products.sum()+item['bias'],item['logit'],atol=4e-6)
    np.testing.assert_allclose(products[:2],item['first_two_products'],atol=1e-7)
    np.testing.assert_allclose(products[2:].sum(),item['remaining_190_products_sum'],atol=4e-6)
    np.testing.assert_allclose(probability[item['index']],item['probability'],atol=1e-7)
    np.testing.assert_allclose(item['probability'],ref['probability'],atol=2e-6)
assert readout['verification']['training_run'] is False
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
        'multihead_messages_concatenation_and_residual_verified':True,
        'mlp_gelu_and_second_residual_previews_verified':True,
        'classifier_weighted_sums_and_all_class_softmax_verified':True,
        'single_query_weight_update':trace['single_update']}
(ROOT/'figures/vision1/classification-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
