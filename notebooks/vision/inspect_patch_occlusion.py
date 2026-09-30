"""Replace one 16×16 patch at a time and measure a fixed class probability.

Run with the existing cached environment and checkpoint:
HF_HUB_OFFLINE=1 uv run --offline --with timm --with pillow python notebooks/vision/inspect_patch_occlusion.py

This is inference only. Each trial starts from the original normalized image.
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


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def measure(model, x, regions, target, baseline, batch_size=16):
    results = []
    with torch.inference_mode():
        for start in range(0, len(regions), batch_size):
            batch_regions = regions[start:start+batch_size]
            covered = x.expand(len(batch_regions), -1, -1, -1).clone()
            for i, (row, column, size) in enumerate(batch_regions):
                covered[i, :, row:row+size, column:column+size] = 0
            probabilities = model(covered).softmax(-1)
            for (row, column, size), p in zip(batch_regions, probabilities):
                results.append(dict(row=row, column=column, size=size,
                    target_probability=float(p[target]), top_index=int(p.argmax()),
                    drop_percentage_points=100*(baseline-float(p[target]))))
    return results


def main():
    torch.set_num_threads(4)
    model = timm.create_model(MODEL, pretrained=True).eval()
    before_hash = parameter_hash(model)
    config = resolve_model_data_config(model)
    x = create_transform(**config, is_training=False)(
        Image.open(ASSETS/'newfoundland_31.jpg').convert('RGB')).unsqueeze(0)
    original = x.clone()
    saved = json.loads((ASSETS/'inspection.json').read_text())
    target = saved['target_index']
    with torch.inference_mode():
        baseline = float(model(x).softmax(-1)[0, target])
    assert abs(baseline-saved['baseline_probability']) < 2e-6
    coarse = measure(model, x, [(r['row'], r['column'], r['size']) for r in saved['occlusion']], target, baseline)
    assert all(abs(a['target_probability']-b['target_probability']) < 2e-6
               for a, b in zip(coarse, saved['occlusion']))
    fine = measure(model, x, [(r, c, 16) for r in range(0,224,16) for c in range(0,224,16)], target, baseline)
    for i, row in enumerate(fine, 1):
        row['patch_index'] = i
    largest = max(fine, key=lambda row: row['drop_percentage_points'])
    # Confirm the displayed extreme with its own independent forward pass.
    check = measure(model, x, [(largest['row'], largest['column'], 16)], target, baseline)[0]
    assert abs(check['target_probability']-largest['target_probability']) < 2e-6
    assert torch.equal(x, original) and before_hash == parameter_hash(model)
    report = dict(model=MODEL, parameter_sha256=before_hash,
        torch=torch.__version__, timm=timm.__version__,
        source_sha256=hashlib.sha256((ASSETS/'newfoundland_31.jpg').read_bytes()).hexdigest(),
        preprocessing=config, input_shape=list(x.shape), target_index=target,
        target_label=saved['target_label'], baseline_probability=baseline,
        cover_size=16, cover_stride=16, grid=[14,14], fill_normalized_rgb=[0,0,0],
        fill_raw_rgb=config['mean'], definition='drop = 100 * (original probability - covered probability)',
        trials=fine, largest_drop=largest,
        summary=dict(trials=len(fine), all_top_labels_unchanged=all(r['top_index']==target for r in fine),
            minimum_probability=min(r['target_probability'] for r in fine),
            maximum_probability=max(r['target_probability'] for r in fine),
            lower_probability_count=sum(r['drop_percentage_points']>0 for r in fine),
            higher_probability_count=sum(r['drop_percentage_points']<0 for r in fine)),
        verification=dict(matches_saved_quadrants=True, largest_drop_checked_separately=True,
            original_input_preserved=True, parameters_unchanged=True, training_run=False),
        scope='One image and checkpoint, gray replacement, fixed non-overlapping locations. '
              'This is sensitivity to this intervention, not attention, segmentation, or a unique semantic attribution.')
    (ASSETS/'patch-occlusion.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(baseline=baseline, largest_drop=largest, summary=report['summary'],
                         verification=report['verification']), indent=2))


if __name__ == '__main__':
    main()
