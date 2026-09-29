"""Save the three CLS messages from block 1 of the lecture's cached dog model.

HF_HUB_OFFLINE=1 uv run --offline --with timm --with pillow python notebooks/vision/trace_multihead_messages.py
Inference only. No training, downloads, or changes to the earlier traces.
"""
from pathlib import Path
import hashlib
import json
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT/'figures/vision1'
MODEL = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
torch.set_num_threads(4)
reference = json.loads((ASSETS/'real-classifier-path.json').read_text())
photo = ASSETS/'newfoundland_31.jpg'
assert hashlib.sha256(photo.read_bytes()).hexdigest() == reference['source_sha256']
model = timm.create_model(MODEL, pretrained=True).eval()
transform = create_transform(**resolve_model_data_config(model), is_training=False)
parameter_before = model.cls_token.detach().clone()

with torch.inference_mode():
    image = transform(Image.open(photo).convert('RGB')).unsqueeze(0)
    E = model._pos_embed(model.patch_embed(image))
    block = model.blocks[0]
    X = block.norm1(E)
    q, k, v = block.attn.qkv(X).reshape(1,197,3,3,64).permute(2,0,3,1,4).unbind(0)
    A = ((q @ k.transpose(-2,-1))*block.attn.scale).softmax(-1)
    H = A @ v
    joined = H.transpose(1,2).reshape(1,197,192)
    torch.testing.assert_close(joined[:,0], torch.cat([H[:,h,0] for h in range(3)], dim=-1))
    update = block.attn.proj(joined)
    torch.testing.assert_close(update, block.attn(X), atol=2e-5, rtol=1e-5)
    U = E + update
    for head in range(3):
        for part, expected in enumerate([q,k,v]):
            start = part*192 + head*64
            actual = F.linear(X[:,0], block.attn.qkv.weight[start:start+64],
                              block.attn.qkv.bias[start:start+64])
            torch.testing.assert_close(actual, expected[:,head,0], atol=2e-5, rtol=1e-5)
    for value, key in [(E[:,0], 'cls_input'), (H[:,0,0], 'cls_head1_message'),
                       (update[:,0], 'cls_attention_update'), (U[:,0], 'cls_after_attention')]:
        torch.testing.assert_close(value.reshape(-1)[:3], torch.tensor(reference['previews'][key]),
                                   atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(model.cls_token, parameter_before, atol=0, rtol=0)

def row(tensor):
    return tensor.reshape(-1).tolist()

report = {
    'model': MODEL, 'torch': torch.__version__, 'timm': timm.__version__,
    'source_sha256': reference['source_sha256'], 'block': 1,
    'input_shape': [197,192], 'qkv_shape_per_head': [197,64],
    'attention_shape_per_head': [197,197], 'message_shape_per_head': [197,64],
    'output_projection_math_shape': [192,192],
    'cls_input': row(E[:,0]), 'cls_normalized': row(X[:,0]),
    'heads': [{'head': h+1, 'cls_query': row(q[:,h,0]), 'cls_key': row(k[:,h,0]),
               'cls_value': row(v[:,h,0]), 'cls_weights': row(A[:,h,0]),
               'cls_message': row(H[:,h,0])} for h in range(3)],
    'cls_joined': row(joined[:,0]), 'cls_projected_update': row(update[:,0]),
    'cls_after_residual': row(U[:,0]),
    'verification': {'separate_projections_read_same_normalized_cls': True,
                     'concatenation_preserves_each_64_coordinate_segment': True,
                     'output_projection_matches_checkpoint_attention': True,
                     'previews_match_previous_dog_trace': True,
                     'cls_parameter_unchanged': True, 'training_run': False},
}
(ASSETS/'multihead-cls-trace.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'verification': report['verification'], 'head_message_previews':
                 [h['cls_message'][:3] for h in report['heads']]}, indent=2))
