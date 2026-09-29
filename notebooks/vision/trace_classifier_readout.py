"""Open the cached classifier using the lecture's already saved final CLS.

HF_HUB_OFFLINE=1 uv run --offline --with timm --with pillow python notebooks/vision/trace_classifier_readout.py
No image forward pass, training, or downloads. Saves the final linear/softmax calculation.
"""
from pathlib import Path
import json
import torch
import timm

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT/'figures/vision1'
reference = json.loads((ASSETS/'real-classifier-path.json').read_text())
comparison = json.loads((ASSETS/'cls-two-image-trace.json').read_text())
dog = next(r for r in comparison['results'] if r['image_id']=='newfoundland_31')
assert dog['source_sha256']==reference['source_sha256']
assert comparison['model']==reference['model']
torch.set_num_threads(4)
model = timm.create_model(reference['model'], pretrained=True).eval()
with torch.inference_mode():
    cls = torch.tensor(dog['cls_final_after_norm']).reshape(1,192)
    weight, bias = model.head.weight, model.head.bias
    assert tuple(weight.shape)==(1000,192) and tuple(bias.shape)==(1000,)
    logits = model.head(cls)
    torch.testing.assert_close(logits, cls @ weight.T + bias, atol=2e-6, rtol=1e-5)
    # Use double precision for a classroom scalar calculation that can be
    # reproduced from the stored logits independently of float32 kernels.
    probability = logits.double().softmax(-1)
    indices = probability[0].topk(3).indices.tolist()
    assert indices==[r['index'] for r in reference['top3']]
    selected = []
    for item in reference['top3']:
        i = item['index']
        products = cls[0]*weight[i]
        torch.testing.assert_close(products.sum()+bias[i], logits[0,i], atol=2e-6, rtol=1e-5)
        assert abs(float(logits[0,i])-item['logit'])<2e-5
        assert abs(float(probability[0,i])-item['probability'])<2e-6
        selected.append({'index':i, 'label':item['label'].split(',')[0],
                         'weights':weight[i].tolist(), 'bias':float(bias[i]),
                         'first_two_products':products[:2].tolist(),
                         'remaining_190_products_sum':float(products[2:].sum()),
                         'logit':float(logits[0,i]), 'probability':float(probability[0,i])})
    shifted = torch.exp(logits[0].double()-logits.max())
    other = torch.ones(1000,dtype=torch.bool)
    other[indices] = False

report = {'model':reference['model'], 'source_sha256':reference['source_sha256'],
          'cls_final':cls[0].tolist(), 'input_shape':[1,192],
          'weight_math_shape':[192,1000], 'weight_pytorch_shape':[1000,192],
          'bias_shape':[1000], 'output_shape':[1,1000],
          'logits':logits[0].tolist(), 'selected_classes':selected,
          'softmax':{'dtype':'float64', 'max_logit':float(logits.max()),
                     'selected_shifted_exponentials':shifted[indices].tolist(),
                     'other_997_shifted_sum':float(shifted[other].sum()),
                     'shifted_denominator':float(shifted.sum())},
          'verification':{'all_logits_equal_affine_head':True,
                          'selected_weighted_sums_equal_logits':True,
                          'predictions_match_saved_dog_result':True,
                          'training_run':False, 'image_forward_run':False}}
(ASSETS/'classifier-readout-trace.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'selected_classes':[{k:v for k,v in x.items() if k!='weights'} for x in selected],
                  'softmax':report['softmax'], 'verification':report['verification']},indent=2))
