"""Measured single-head attention and mean-colour occlusion on the opening image.
Run: uv run --with timm --with pillow python notebooks/vision/inspect_real_vit.py
The saved plots are measurements, not causal explanations or accuracy estimates.
"""
from pathlib import Path
import hashlib
import json
import math
import torch
import timm
from PIL import Image
from timm.data import ImageNetInfo, create_transform, resolve_model_data_config
ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT/'figures/vision1'
MODEL = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
torch.set_num_threads(4)
model = timm.create_model(MODEL, pretrained=True).eval()
config = resolve_model_data_config(model)
transform = create_transform(**config, is_training=False)
x = transform(Image.open(ASSETS/'newfoundland_31.jpg').convert('RGB')).unsqueeze(0)
# Show the exact resize/centre-crop supplied to the model, before normalization.
mean = torch.tensor(config['mean']).view(1,3,1,1)
std = torch.tensor(config['std']).view(1,3,1,1)
pixels = ((x*std+mean)[0].permute(1,2,0).clamp(0,1)*255).round().byte().numpy()
Image.fromarray(pixels).save(ASSETS/'model-input.png')
maps=[]
handles=[]
def hook(layer):
    def capture(module, inputs):
        z=inputs[0]; B,N,C=z.shape
        q,k,v=module.qkv(z).reshape(B,N,3,module.num_heads,-1).permute(2,0,3,1,4).unbind(0)
        q,k=module.q_norm(q),module.k_norm(k)
        a=((q*module.scale)@k.transpose(-2,-1)).softmax(-1)[0]
        assert torch.allclose(a.sum(-1),torch.ones_like(a.sum(-1)),atol=1e-6)
        for h in range(module.num_heads):
            for query,name in [(0,'CLS'),(104,'patch row 7, column 5 (zero-based)')]:
                row=a[h,query].detach().cpu()
                maps.append({'block':layer+1,'head':h+1,'query':name,'query_index':query,
                    'cls_weight':float(row[0]),'patch_weights':row[1:].reshape(14,14).tolist(),
                    'patch_sum':float(row[1:].sum()),'maximum_patch_weight':float(row[1:].max())})
    return capture
for i in [0,11]: handles.append(model.blocks[i].attn.register_forward_pre_hook(hook(i)))
with torch.inference_mode(): baseline=model(x).softmax(-1)[0]
for handle in handles:handle.remove()
labels=ImageNetInfo(); target=int(baseline.argmax())
regions=[('Top left',0,0),('Top right',0,112),('Bottom left',112,0),('Bottom right',112,112)]
results=[]
with torch.inference_mode():
    for name,r,c in regions:
        masked=x.clone(); masked[:,:,r:r+112,c:c+112]=0
        probs=model(masked).softmax(-1)[0]; top=int(probs.argmax())
        results.append({'region':name,'row':r,'column':c,'size':112,
          'target_probability':float(probs[target]),'change_percentage_points':100*float(probs[target]-baseline[target]),
          'top_label':labels.index_to_description(top),'top_probability':float(probs[top])})
result={'model':MODEL,'timm':timm.__version__,'torch':torch.__version__,
        'source_sha256':hashlib.sha256((ASSETS/'newfoundland_31.jpg').read_bytes()).hexdigest(),
        'preprocessing':config,'target_index':target,'target_label':labels.index_to_description(target),
        'baseline_probability':float(baseline[target]),'attention':maps,'occlusion':results,
        'occlusion_fill':'normalized RGB zero = model normalization mean RGB',
        'scope':'One image; fixed quadrants chosen before results. Occlusion changes the input distribution; no causal importance claim.'}
(ASSETS/'inspection.json').write_text(json.dumps(result,indent=2)+'\n')
print('Baseline',result['target_label'],result['baseline_probability'])
for r in results: print(r['region'],r['target_probability'],r['top_label'])
