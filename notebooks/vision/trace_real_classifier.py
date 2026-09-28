"""Trace the lecture's saved dog photo through the full pretrained classifier.

uv run --with timm --with pillow python notebooks/vision/trace_real_classifier.py
Only writes numerical evidence. No training, playback or publication.
"""
from pathlib import Path
import hashlib
import json
import torch
import timm
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'figures/vision1'
MODEL = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
torch.set_num_threads(4)
model = timm.create_model(MODEL, pretrained=True).eval()
photo = ASSETS/'newfoundland_31.jpg'
transform = create_transform(**resolve_model_data_config(model), is_training=False)
x = transform(Image.open(photo).convert('RGB')).unsqueeze(0)

with torch.inference_mode():
    C = model.patch_embed(x)
    E = model._pos_embed(C)
    torch.testing.assert_close(E[:, 0], model.cls_token[:, 0] + model.pos_embed[:, 0])
    block = model.blocks[0]
    z = block.norm1(E)
    q, k, v = block.attn.qkv(z).reshape(1,197,3,3,64).permute(2,0,3,1,4).unbind(0)
    A = ((q @ k.transpose(-2,-1)) * block.attn.scale).softmax(-1)
    H = A @ v
    joined = H.transpose(1,2).reshape(1,197,192)
    update = block.attn.proj(joined)
    torch.testing.assert_close(update, block.attn(z), atol=2e-5, rtol=1e-5)
    after_attention = E + update
    normalized = block.norm2(after_attention)
    hidden = block.mlp.fc1(normalized)
    activated = block.mlp.act(hidden)
    mlp_update = block.mlp.fc2(activated)
    after_block = after_attention + mlp_update
    torch.testing.assert_close(after_block, block(E), atol=3e-5, rtol=1e-5)
    current = after_block
    for other in model.blocks[1:]:
        current = other(current)
    final = model.norm(current)
    logits = model.head(final[:,0])
    torch.testing.assert_close(logits, model(x), atol=5e-5, rtol=1e-5)
    probs = logits.softmax(-1)

reference = json.loads((ASSETS/'real-inference.json').read_text())['results'][0]['top3']
top_values, top_indices = probs[0].topk(3)
top3 = [{'index':int(i), 'label':ref['label'], 'logit':float(logits[0,i]), 'probability':float(p)}
        for i,p,ref in zip(top_indices,top_values,reference)]
for actual, ref in zip(top3, reference):
    assert abs(actual['probability']-ref['probability']) < 1e-5

def preview(t):
    return t.reshape(-1)[:3].tolist()

report = {
    'model':MODEL, 'timm':timm.__version__, 'source_sha256':hashlib.sha256(photo.read_bytes()).hexdigest(),
    'patch_rows':[196,192], 'with_cls':[197,192], 'head_count':3, 'block_count':12,
    'per_head_qkv':[197,64], 'per_head_attention':[197,197], 'mlp_hidden':[197,768],
    'classification_head':[192,1000],
    'previews':{name:preview(t) for name,t in [
        ('cls_parameter',model.cls_token), ('cls_position',model.pos_embed[:,0]),
        ('cls_input',E[:,0]), ('cls_head1_message',H[:,0,0]), ('cls_attention_update',update[:,0]),
        ('cls_after_attention',after_attention[:,0]), ('cls_mlp_hidden',hidden[:,0]),
        ('cls_mlp_activated',activated[:,0]), ('cls_mlp_update',mlp_update[:,0]),
        ('cls_after_block1',after_block[:,0]), ('cls_final',final[:,0])]},
    'cls_head1_weights':A[0,0,0].tolist(), 'top3':top3,
    'verified':['CLS plus its position matches the checkpoint input',
                'Explicit softmax(QKᵀ/√64)V and output projection match attention',
                'Both residual additions and Linear-GELU-Linear match block 1',
                'All 12 blocks, final normalization and CLS head match model(image)',
                'Top-three probabilities match the previously published dog result'],
}
(ASSETS/'real-classifier-path.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['with_cls','mlp_hidden','top3','verified']},indent=2))
