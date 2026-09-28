"""Trace one actual dog-image patch through the saved lecture checkpoint.

Run: uv run --with timm --with pillow python notebooks/vision/trace_real_patch.py
Writes numerical evidence only; the slide builder renders the existing image.
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
config = resolve_model_data_config(model)
transform = create_transform(**config, is_training=False)
photo = ASSETS / 'newfoundland_31.jpg'
inputs = transform(Image.open(photo).convert('RGB')).unsqueeze(0)
mean = torch.tensor(config['mean']).view(1, 3, 1, 1)
std = torch.tensor(config['std']).view(1, 3, 1, 1)
display = ((inputs*std+mean)[0].permute(1, 2, 0).clamp(0, 1)*255).round().byte()
assert bytes(display.numpy().tobytes()) == Image.open(ASSETS/'model-input.png').convert('RGB').tobytes()
row, col, size = 4, 6, 16
index = row * 14 + col

with torch.inference_mode():
    patch = inputs[0, :, row*size:(row+1)*size, col*size:(col+1)*size]
    x = patch.permute(1, 2, 0).reshape(1, -1)
    projection = model.patch_embed.proj
    # Match the slides' pixel-major RGB row to the Conv2d parameter ordering.
    W = projection.weight.permute(2, 3, 1, 0).reshape(768, 192)
    content = x @ W + projection.bias
    all_content = model.patch_embed(inputs)
    torch.testing.assert_close(content, all_content[:, index], atol=2e-5, rtol=1e-5)
    # Extract every patch in the same raster order, retaining RGB per pixel.
    X = inputs[0].permute(1, 2, 0).reshape(14, 16, 14, 16, 3).permute(0, 2, 1, 3, 4).reshape(196, 768)
    C = X @ W + projection.bias
    torch.testing.assert_close(C, all_content[0], atol=2e-5, rtol=1e-5)
    position = model.pos_embed[:, index+1]
    embedded = content + position
    all_embedded = model._pos_embed(all_content)
    torch.testing.assert_close(embedded, all_embedded[:, index+1], atol=2e-5, rtol=1e-5)
    normalized = model.blocks[0].norm1(embedded)
    attn = model.blocks[0].attn
    packed = attn.qkv(normalized).reshape(1, 3, 3, 64)
    q, k, v = (packed[:, j, 0] for j in range(3))
    # First head uses one 64-row slice of each Q/K/V parameter group.
    for j, expected in enumerate([q, k, v]):
        start = j*192
        actual = normalized @ attn.qkv.weight[start:start+64].T + attn.qkv.bias[start:start+64]
        torch.testing.assert_close(actual, expected)

def values(tensor):
    return tensor.detach().cpu().reshape(-1).tolist()

selected = {}
for number in [1, 63, 64, 196]:
    r, c = divmod(number-1, 14)
    selected[str(number)] = {
        'row_col_one_based': [r+1, c+1],
        'rgb_pixels': display[r*16:(r+1)*16, c*16:(c+1)*16].reshape(256, 3).tolist(),
        'normalized_rgb_row': values(X[number-1]),
        'content': values(C[number-1]),
    }

report = {
    'model': MODEL, 'timm': timm.__version__, 'torch': torch.__version__,
    'source_sha256': hashlib.sha256(photo.read_bytes()).hexdigest(),
    'display_image_sha256': hashlib.sha256((ASSETS/'model-input.png').read_bytes()).hexdigest(),
    'preprocessing': config,
    'patch_number_one_based': index+1,
    'patch_grid_row_col_zero_based': [row, col], 'patch_size': [size, size, 3],
    'input_shape': list(x.shape), 'projection_weight_shape': list(W.shape),
    'bias_shape': list(projection.bias.shape), 'content_shape': list(content.shape),
    'position_shape': list(position.shape), 'embedding_shape': list(embedded.shape),
    'head_count': 3, 'per_head_qkv_shape': [1, 64],
    'normalized_rgb_row': values(x), 'content': values(content),
    'position': values(position), 'embedding': values(embedded),
    'normalized_embedding': values(normalized),
    'head1_q': values(q), 'head1_k': values(k), 'head1_v': values(v),
    'selected_patches': selected,
    'all_content_rows': C.tolist(),
    'all_input_shape': list(X.shape), 'all_content_shape': list(C.shape),
    'verified': ['displayed image equals checkpoint input before normalization',
                 'pixel-major linear equals checkpoint patch embedding',
                 'all 196 separately flattened patches match the checkpoint output rows',
                 'content plus position equals checkpoint input row',
                 'head-one projections equal checkpoint QKV slices'],
}
(ASSETS/'real-patch-path.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['patch_number_one_based','input_shape','content_shape','position_shape','per_head_qkv_shape','verified']},indent=2))
print('First three coordinates:', {k:report[k][:3] for k in ['content','position','embedding']})
