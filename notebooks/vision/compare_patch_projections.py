"""Compare two implementations of the same trained patch projection.

Run from the repository root (cached checkpoint, inference only):
HF_HUB_OFFLINE=1 uv run --offline --with timm --with pillow python notebooks/vision/compare_patch_projections.py

The two modules are alternatives. A deployed model needs only one of them.
Slide snippets are extracted from this executable example.
"""
from pathlib import Path
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F
import timm
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'figures/vision1'
MODEL = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'


def project_with_linear(images, linear):
    # slide:linear-forward
    patches = F.unfold(images, kernel_size=16, stride=16)
    # B × 768 × 196: one flattened patch per column
    patches = patches.transpose(1, 2)
    # B × 196 × 768: one flattened patch per row
    return linear(patches)  # B × 196 × 192
    # endslide


def project_with_conv(images, conv):
    # slide:conv-forward
    grid = conv(images)  # B × 192 × 14 × 14
    return grid.flatten(2).transpose(1, 2)  # B × 196 × 192
    # endslide


def main():
    torch.set_num_threads(4)
    model = timm.create_model(MODEL, pretrained=True).eval()
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)
    references = json.loads((ASSETS / 'real-inference.json').read_text())['results']
    paths = [ASSETS / (ref['image_id'] + '.jpg') for ref in references]
    for path, ref in zip(paths, references):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256']
    images = torch.stack([transform(Image.open(path).convert('RGB')) for path in paths])
    assert list(images.shape) == [2, 3, 224, 224]

    # slide:linear-init
    linear = nn.Linear(768, 192, bias=True)
    # endslide
    # slide:conv-init
    conv = nn.Conv2d(3, 192, kernel_size=16, stride=16,
                     padding=0, bias=True)
    # endslide
    conv.load_state_dict(model.patch_embed.proj.state_dict())
    # Same trained numbers; reshape each filter into one neuron's weights.
    # slide:copy-weights
    with torch.no_grad():
        linear.weight.copy_(conv.weight.flatten(1))
        linear.bias.copy_(conv.bias)
    # endslide
    linear.eval()
    conv.eval()

    with torch.inference_mode():
        # slide:compare
        dense_rows = project_with_linear(images, linear)
        conv_rows = project_with_conv(images, conv)
        torch.testing.assert_close(
            dense_rows, conv_rows, rtol=1e-5, atol=2e-5)
        # endslide
        assert list(dense_rows.shape) == [2, 196, 192]
        torch.testing.assert_close(conv_rows, model.patch_embed(images), rtol=1e-5, atol=2e-5)
        # P63 (one-based) is row 4, column 6 on the 14 × 14 grid.
        crops = images[:, :, 64:80, 96:112].flatten(1)
        patches = F.unfold(images, 16, stride=16).transpose(1, 2)
        torch.testing.assert_close(crops, patches[:, 62], rtol=0, atol=0)
        torch.testing.assert_close(linear(crops), conv_rows[:, 62], rtol=1e-5, atol=2e-5)
        torch.testing.assert_close(conv_rows, torch.cat([
            project_with_conv(image.unsqueeze(0), conv) for image in images
        ]), rtol=1e-5, atol=2e-5)
        torch.testing.assert_close(linear.weight, conv.weight.flatten(1), rtol=0, atol=0)
        torch.testing.assert_close(linear.bias, conv.bias, rtol=0, atol=0)

    counts = {}
    for name, layer in [('linear', linear), ('conv2d', conv)]:
        counts[name] = {'weight_shape': list(layer.weight.shape),
                       'bias_shape': list(layer.bias.shape),
                       'weights': layer.weight.numel(), 'biases': layer.bias.numel(),
                       'total': sum(p.numel() for p in layer.parameters())}
        assert counts[name]['total'] == 147648
    report = {
        'model': MODEL, 'torch': torch.__version__, 'timm': timm.__version__,
        'preprocessing': config, 'input_shape': list(images.shape),
        'output_shape': list(conv_rows.shape), 'counts': counts,
        'flatten_order': 'channel, patch row, patch column (all R, then G, then B)',
        'max_absolute_error': (dense_rows-conv_rows).abs().max().item(),
        'comparison_tolerance': {'rtol': 1e-5, 'atol': 2e-5},
        'results': [{
            'image_id': ref['image_id'], 'source_sha256': ref['sha256'],
            'patch_index_one_based': 63, 'patch_grid_zero_based': [4, 6],
            'linear_first_features': dense_rows[i, 62, :3].tolist(),
            'conv_first_features': conv_rows[i, 62, :3].tolist(),
            'max_absolute_error': (dense_rows[i]-conv_rows[i]).abs().max().item(),
        } for i, ref in enumerate(references)],
        'verification': {
            'all_75264_output_features_match_within_tolerance': True,
            'matches_pretrained_patch_embed': True, 'matching_pixel_order': True,
            'same_weights_and_biases': True, 'batch_matches_individual_images': True,
            'training_run': False,
        },
    }
    (ASSETS / 'patch-projection-equivalence.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
