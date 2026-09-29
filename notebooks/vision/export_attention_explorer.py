"""Export one pretrained forward pass for the lecture's interactive patch explorer.

HF_HUB_OFFLINE=1 uv run --offline --with timm --with pillow python notebooks/vision/export_attention_explorer.py
No training, gradients, or model weights are shipped to the browser.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import timm
from PIL import Image
from timm.data import ImageNetInfo, create_transform, resolve_model_data_config

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'figures/vision1'
OUT = ASSETS / 'attention-explorer'
MODEL = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
torch.set_num_threads(4)
model = timm.create_model(MODEL, pretrained=True).eval()
config = resolve_model_data_config(model)
x = create_transform(**config, is_training=False)(
    Image.open(ASSETS / 'newfoundland_31.jpg').convert('RGB')).unsqueeze(0)
saved = {}
references = []
handles = []

def capture_attention(index):
    def capture(module, inputs):
        z = inputs[0]
        B, N, C = z.shape
        q, k, v = module.qkv(z).reshape(B, N, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)
        a = ((q * module.scale) @ k.transpose(-2, -1)).softmax(-1)[0]
        assert q.shape == (1, 3, 197, 64)
        torch.testing.assert_close(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-6, rtol=0)
        # Verify the explicit QK/softmax calculation against timm's actual attention output.
        h = (a.unsqueeze(0) @ v).transpose(1, 2).reshape(B, N, C)
        expected = module.proj_drop(module.proj(h))
        saved[index] = {'q': q[0].detach().cpu().numpy(), 'k': k[0].detach().cpu().numpy(),
                        'expected': expected, 'a': a.detach().cpu().numpy()}
        for head in range(3):
            for query in [0, 63, 74, 104, 196]:
                references.append({'block': index+1, 'head': head+1, 'query': query,
                                   'weights': a[head, query].tolist()})
    return capture

def check_attention(index):
    def capture(module, inputs, output):
        torch.testing.assert_close(output, saved[index].pop('expected'), atol=2e-5, rtol=2e-5)
    return capture

def capture_features(index):
    def capture(module, inputs, output):
        features = output[0].detach().cpu()
        saved[index]['features'] = features.numpy()
        unit = torch.nn.functional.normalize(features, dim=-1)
        for query in [1, 63, 74, 104, 196]:
            references.append({'block': index+1, 'query': query, 'similarities': (unit[query] @ unit.T).tolist()})
    return capture

for i, block in enumerate(model.blocks):
    handles.extend([block.attn.register_forward_pre_hook(capture_attention(i)),
                    block.attn.register_forward_hook(check_attention(i)),
                    block.register_forward_hook(capture_features(i))])
with torch.inference_mode():
    probability = model(x).softmax(-1)[0]
for handle in handles:
    handle.remove()

# Independently saved maps from the existing lecture must still agree.
old = json.loads((ASSETS / 'inspection.json').read_text())
for item in old['attention']:
    row = saved[item['block']-1]['a'][item['head']-1, item['query_index']]
    np.testing.assert_allclose(row, [item['cls_weight'], *np.array(item['patch_weights']).flatten()], atol=2e-7)

OUT.mkdir(exist_ok=True)
files = []
for i, data in saved.items():
    values = np.concatenate([data['q'].flatten(), data['k'].flatten(), data['features'].flatten()]).astype('<f4')
    raw = values.tobytes()
    path = OUT / f'block-{i+1:02}.f32'
    path.write_bytes(raw)
    files.append({'block': i+1, 'file': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
target = int(probability.argmax())
parameter_hash = hashlib.sha256()
for name, tensor in sorted(model.state_dict().items()):
    parameter_hash.update(name.encode())
    parameter_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
meta = {
    'version': 1, 'parameter_sha256': parameter_hash.hexdigest(), 'model': MODEL, 'timm': timm.__version__, 'torch': torch.__version__,
    'model_card': 'https://huggingface.co/timm/' + MODEL,
    'source_image': 'newfoundland_31.jpg',
    'source_sha256': hashlib.sha256((ASSETS / 'newfoundland_31.jpg').read_bytes()).hexdigest(),
    'display_image': 'model-input.png',
    'display_sha256': hashlib.sha256((ASSETS / 'model-input.png').read_bytes()).hexdigest(),
    'preprocessing': config, 'input_shape': [1, 3, 224, 224], 'grid': 14, 'patch_size': 16,
    'rows': 197, 'heads': 3, 'head_width': 64, 'feature_width': 192, 'scale': float(model.blocks[0].attn.scale),
    'layout': {'dtype': 'little-endian float32', 'q': {'offset': 0, 'shape': [3, 197, 64]},
               'k': {'offset': 37824, 'shape': [3, 197, 64]},
               'features': {'offset': 75648, 'shape': [197, 192]}, 'offset_unit': 'float32 elements'},
    'feature_stage': 'Output of the selected block, after both residual additions; before any following LayerNorm.',
    'token_order': 'CLS at 0, then P1..P196 in row-major order across the 14 by 14 patch grid.',
    'prediction': {'label': ImageNetInfo().index_to_description(target), 'probability': float(probability[target])},
    'default': {'block': 4, 'head': 1, 'query': 74}, 'files': files,
    'verification': {'training_run': False, 'explicit_attention_matches_model': True,
                     'matches_existing_lecture_maps': True, 'reference_file': 'reference.json'},
}
(OUT / 'manifest.json').write_text(json.dumps(meta, separators=(',', ':')) + '\n')
(OUT / 'reference.json').write_text(json.dumps(references, separators=(',', ':')) + '\n')
print(f'Exported {len(files)} blocks; {sum(f["bytes"] for f in files):,} bytes. {meta["prediction"]}')
for i in [0, 5, 11]:
    f = saved[i]['features']
    f = f / np.linalg.norm(f, axis=-1, keepdims=True)
    for query in [63, 74]:
        similarities = f @ f[query]
        order = [int(j) for j in np.argsort(-similarities) if j not in [0, query]][:5]
        print('Block', i+1, 'query', query, 'similar patches', [(j, round(float(similarities[j]), 3)) for j in order])
