"""Run an existing ImageNet ViT on the exact motivation photographs.

This is inference on two examples, not a Pets benchmark or species fine-tuning.
Run: uv run --with timm --with pillow python notebooks/vision/run_real_images.py
"""
from pathlib import Path
import hashlib
import json
import PIL
from PIL import Image
import torch
import timm
from timm.data import ImageNetInfo, create_transform, resolve_model_data_config

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'figures/vision1'
MODEL = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
torch.set_num_threads(4)
model = timm.create_model(MODEL, pretrained=True).eval()
config = resolve_model_data_config(model)
transform = create_transform(**config, is_training=False)
labels = ImageNetInfo()
results = []
for item in json.loads((ASSETS / 'images.json').read_text()):
    path = ASSETS / item['file']
    x = transform(Image.open(path).convert('RGB')).unsqueeze(0)
    with torch.inference_mode():
        probs = model(x).softmax(dim=-1)[0]
    values, indices = probs.topk(3)
    results.append({
        'image_id': item['image_id'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'top3': [{'index': int(i), 'label': labels.index_to_description(int(i)),
                  'probability': float(v)} for v, i in zip(values, indices)],
    })
report = {
    'model': MODEL, 'model_url': 'https://huggingface.co/timm/' + MODEL,
    'timm': timm.__version__, 'torch': torch.__version__, 'pillow': PIL.__version__,
    'preprocessing': config, 'classes': 1000, 'patches': 196, 'tokens': 197,
    'd_model': 192, 'n_heads': 3, 'd_head': 64, 'blocks': 12,
    'scope': 'Two-image ImageNet inference; no training or Pets benchmark.',
    'results': results,
}
(ASSETS / 'real-inference.json').write_text(json.dumps(report, indent=2) + '\n')
for r in results:
    print(r['image_id'], r['top3'])
