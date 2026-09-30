"""Show the exact pixel intervention before comparing two model predictions."""
from html import escape
import json
import re

from vision1_focus_common import Figures


PIXEL_CODE = '''x = transform(photo).unsqueeze(0)  # (1, 3, 224, 224)
covered = x.clone()                # a separate copy
covered[:, :, :112, :112] = 0       # all RGB channels, top left'''

FORWARD_CODE = '''model.eval()
with torch.inference_mode():
    p_before = model(x).softmax(-1)[0]        # (1000,)
    p_after = model(covered).softmax(-1)[0]   # (1000,)
before, after = p_before[256], p_after[256]   # Newfoundland'''


def explain(b, sections):
    f = Figures(b)
    t, g, arrow = f.t, f.g, f.arrow
    saved = json.loads((b['ASSETS'] / 'inspection.json').read_text())
    record = saved['occlusion'][0]
    assert record['region'] == 'Top left' and record['size'] == 112
    assert saved['preprocessing']['mean'] == saved['preprocessing']['std'] == [.5, .5, .5]
    assert saved['target_index'] == 256

    def photo(x, y, size, covered=False, region=None):
        out = f.image(x, y, size, size, f.photo)
        if covered:
            row, column, width = region or (0, 0, 112)
            out += (f'<rect x="{x+column/224*size}" y="{y+row/224*size}" '
                    f'width="{width/224*size}" height="{width/224*size}" fill="rgb(128,128,128)"/>')
        return out

    body = photo(35, 36, 235) + t(152, 306, 'Original input x', 27, 'c-e', 'middle')
    body += arrow(295, 150, 355, 150, 'c-q')
    body += photo(380, 36, 235, True) + t(497, 306, 'Copy with gray pixels', 27, 'c-q', 'middle')
    body += t(680, 66, '112 × 112 pixels replaced', 30, 'c-q')
    body += t(680, 118, 'RGB fill = (0.5, 0.5, 0.5)', 27)
    body += t(680, 158, 'After normalization:', 25, 'ink-2')
    body += t(680, 194, '(0.5 − 0.5) / 0.5 = 0', 29, 'c-e')
    body += t(680, 251, 'Same 224 × 224 image size.', 26)
    body += t(680, 290, 'All 196 patches remain.', 26)
    body += f.code(PIXEL_CODE, x=35, y=358, width=1120, size=25, spacing=34)
    f.add('cover-pixels', '“Cover” means replace these pixels with gray', body,
          'Copy the normalized input, then overwrite one quadrant. The rest of the image stays unchanged; no patch rows are deleted.',
          'Does zero here mean black pixels, or gray pixels?',
          'x is already resized, center-cropped and normalized by this checkpoint’s evaluation transform. '
          'Its mean and standard deviation are both 0.5 per channel. Zero in x therefore means RGB 0.5 before normalization.',
          '<p>The original RGB photo is loaded with Pillow. The checkpoint’s evaluation transform resizes, '
          'center-crops to 224×224 and normalizes each channel as (pixel − 0.5) / 0.5. '
          'unsqueeze(0) adds the batch axis. clone() makes a separate tensor so the original is preserved.</p>'
          '<p>The four indices are batch, channel, row, column. Both colons select the whole batch and all RGB channels; '
          ':112 selects indices 0 through 111 in each spatial axis. This changes 112×112×3 input values. '
          'The 14×14 patch grid remains: 49 of its patches now contain constant gray pixels. '
          'Their projected features still include the learned projection bias and position information.</p>'
          '<p>This is an input-pixel intervention. It does not remove tokens or apply a mask to the attention matrix. '
          'The diagram draws the gray cover on the exact model crop; the experiment changes the tensor before running the model.</p>'
          '<pre class="language-python"><code>'+escape(PIXEL_CODE)+'</code></pre>',
          '<p>Original normalized input: (1, 3, 224, 224). Copy it and set the first 112 rows and columns, '
          'in all three channels, to zero. This is mid-gray RGB; image size and patch count stay the same.</p>'
          '<pre class="language-python"><code>'+escape(PIXEL_CODE)+'</code></pre>')

    body = t(35, 24, 'Two independent forward passes', 28)
    body += t(1030, 24, 'P(Newfoundland)', 26, 'c-e', 'middle')
    for i, label in enumerate(['Original x', 'Covered copy']):
        y = 46 + i*127
        body += photo(35, y, 100, bool(i)) + t(155, y+58, label, 25, 'c-q' if i else 'c-e')
        body += arrow(335, y+50, 385, y+50)
        body += f.box(400, y+7, 420, ['Same trained ViT', 'patches → blocks → CLS → head'], h=86, size=23)
        body += arrow(835, y+50, 897, y+50)
        result = t(1030, y+66, f'{100*(record["target_probability"] if i else saved["baseline_probability"]):.2f}%',
                   39, 'c-e', 'middle')
        body += g(result, 1) if i else result
    body += t(610, 163, 'Same weights; recompute all activations', 22, 'ink-2', 'middle')
    body += f.code(FORWARD_CODE, x=35, y=325, width=1120, size=24, spacing=29)
    f.add('cover-1', 'Run the covered image through the same trained model', body,
          'Recompute patch features, attention, CLS and class scores. Softmax gives 1,000 probabilities; compare the Newfoundland entry in both runs. The weights stay fixed.',
          'What must be recomputed after the pixels change?',
          f'The entire forward pass runs again. Reveal the measured change from {saved["baseline_probability"]:.2%} '
          f'to {record["target_probability"]:.2%}: {abs(record["change_percentage_points"]):.2f} percentage points lower. '
          'The top label remains Newfoundland. No backward pass or optimizer step occurs.',
          '<p>Both inputs use the same pretrained checkpoint in evaluation mode. model.eval() selects evaluation '
          'behaviour; torch.inference_mode() avoids recording gradients. Neither call trains the model. '
          'Each model call recomputes patch embeddings, all 12 blocks, final CLS and the 1,000 class scores. '
          'The CLS start vector, position embeddings and every learned weight remain fixed between runs.</p>'
          '<p>softmax(-1) converts the 1×1,000 logits to class probabilities; [0] selects the only image in the batch. '
          'Index 256 is Newfoundland in this checkpoint’s ImageNet label order. Keep that index fixed '
          'when comparing probabilities, even if an intervention changes the top prediction.</p>'
          '<pre class="language-python"><code>'+escape(FORWARD_CODE)+'</code></pre>'
          '<p>The results are saved measurements, not a live browser inference. '
          '<a href="notebooks/vision/inspect_real_vit.py">Reproduce the preprocessing and both model calls</a> · '
          '<a href="figures/vision1/inspection.json">Saved probabilities and metadata</a>. '
          'This measures sensitivity to this particular gray replacement; it does not assign a unique importance '
          'to the removed visual content.</p>',
          f'<p>Original → same trained ViT → P(Newfoundland) = {saved["baseline_probability"]:.2%}. '
          f'Covered copy → same trained ViT → {record["target_probability"]:.2%}. '
          'All activations are recomputed; weights remain fixed.</p>'
          '<pre class="language-python"><code>'+escape(FORWARD_CODE)+'</code></pre>', height=465)

    # State the hypothesis and use probability drops, not four similar bars.
    baseline = saved['baseline_probability']
    quadrants = saved['occlusion']
    worst = min(quadrants, key=lambda item: item['target_probability'])
    assert all(item['top_label'] == saved['target_label'] for item in quadrants)
    body = t(35, 33, 'Question: how much does one gray cover lower P(Newfoundland)?', 28)
    body += photo(35, 85, 235)
    body += t(152, 360, f'Original: {baseline:.2%}', 27, 'c-e', 'middle')
    body += t(340, 88, 'Covered region', 24, 'ink-2')
    body += t(825, 88, 'Probability', 24, 'ink-2', 'end')
    body += t(1115, 88, 'Drop (points)', 24, 'ink-2', 'end')
    for i, item in enumerate(quadrants):
        y = 125 + i*62
        strongest = item == worst
        if strongest:
            body += f.rect(324, y-28, 812, 59, 'c-a', 'card')
        body += photo(340, y-22, 48, True, (item['row'], item['column'], item['size']))
        color = 'c-a' if strongest else 'ink'
        body += t(410, y+10, item['region'], 27, color)
        body += t(825, y+10, f'{item["target_probability"]:.2%}', 29, color, 'end')
        body += t(1115, y+10, f'{100*(baseline-item["target_probability"]):.2f}', 29, color, 'end')
    body += g(t(35, 430, 'Same top label in all four tests: Newfoundland.', 33, 'c-e', weight=650), 1)
    f.add('occlusion', 'The label stays; its probability falls', body,
          'Top-right covering causes the largest drop: 16.11 percentage points. This photo still receives the same label after every quadrant cover.',
          'Did the class label change, or only its probability?',
          'Compare each probability with the same 95.73% baseline. The largest tested drop is 16.11 points, '
          'for the top-right cover. Reveal that all four top labels remain Newfoundland. This is a coarse sensitivity experiment.',
          '<p><strong>Experiment:</strong> choose four quadrants before looking at results. For each trial, '
          'start from the original input, replace only that quadrant with gray, and run the same frozen classifier. '
          'Record the probability of the fixed Newfoundland class, then subtract it from the original probability.</p>'
          '<p><strong>Result:</strong> all four probabilities are lower, between 79.62% and 84.54%; '
          'all four top labels remain Newfoundland. The top-right cover causes the largest drop among these four replacements. '
          'The label survives these particular changes to this particular photograph. '
          'This does not establish robustness to other images, covers or perturbations.</p>'
          '<p>Each cover hides 49 model patches at once, spanning several visual features and introducing a gray boundary. '
          'The ranking cannot isolate a small feature or assign a unique importance to a semantic object part. '
          'The next slides reduce the cover size while keeping the model unchanged.</p>',
          b['mobile_rows'](['Gray cover', 'P(Newfoundland)', 'Drop (percentage points)'],
              [[r['region'], f'{r["target_probability"]:.2%}', f'{100*(baseline-r["target_probability"]):.2f}'] for r in quadrants])+
          '<p>All four top labels remain Newfoundland. Top-right covering causes the largest probability drop.</p>')

    body = t(290, 30, 'Large cover: 112 × 112', 31, 'c-q', 'middle')
    body += t(870, 30, 'Small cover: 16 × 16', 31, 'c-q', 'middle')
    for x, width in [(175, 112), (755, 16)]:
        body += photo(x, 60, 230, True, (0, 0, width))
        for j in range(1, 14):
            body += f.line(x+j*230/14, 60, x+j*230/14, 290, 'card', .8)
            body += f.line(x, 60+j*230/14, x+230, 60+j*230/14, 'card', .8)
        body += f.rect(x, 60, width/224*230, width/224*230, 'c-q', 'transparent', 0)
    body += t(290, 336, '49 patches covered · 4 tests', 28, 'ink', 'middle')
    body += t(870, 336, '1 patch covered · 196 tests', 28, 'ink', 'middle')
    body += t(580, 403, 'Fresh original → cover one region → rerun → measure the drop', 29, 'c-e', 'middle')
    f.add('occlusion-small-setup', 'Use smaller covers to ask a more local question', body,
          'Only the cover size changes. The trained model still uses 16×16 patches and a 224×224 input. Every test starts from the original image.',
          'Are we changing the model’s patch size, or the region we cover?',
          'Keep the model fixed. Cover one of its 196 patch locations at a time, producing 196 independent images. '
          'A smaller cover probes a smaller region, but may have a smaller effect because other useful information remains.',
          '<p>The original experiment used four non-overlapping 112×112 covers. '
          'The finer experiment uses 196 non-overlapping 16×16 covers aligned with the existing 14×14 patch grid. '
          'The model’s patch projection, weights, input resolution and preprocessing stay fixed.</p>'
          '<p>Each trial starts with a fresh copy. Replace one patch with normalized zero, rerun the whole model '
          'and compute 100 × (original probability − covered probability). The covers are never accumulated. '
          'A small change does not show that a region is useless: other regions can carry related information, '
          'and features can interact. Drops from different trials should not be added together.</p>'
          '<p><a href="notebooks/vision/inspect_patch_occlusion.py">Reproduce the 196 trained-model tests</a>.</p>',
          '<p>Large cover: 112×112 pixels, 49 model patches, four tests. '
          'Small cover: 16×16 pixels, one model patch, 196 tests. '
          'Each test starts with the original image and uses the same model.</p>')

    fine = json.loads((b['ASSETS']/'patch-occlusion.json').read_text())
    assert abs(fine['baseline_probability']-baseline) < 2e-6 and fine['summary']['all_top_labels_unchanged']
    largest = fine['largest_drop']
    region = (largest['row'], largest['column'], largest['size'])
    body = t(175, 30, f'One test: cover P{largest["patch_index"]}', 27, 'c-q', 'middle')
    body += t(530, 30, 'All 196 test results', 27, 'c-e', 'middle')
    body += photo(35, 65, 280, True, region)
    body += f.rect(35+region[1]/224*280, 65+region[0]/224*280, 20, 20, 'c-q', 'transparent', 0)
    body += photo(390, 65, 280)
    # Fixed diverging scale: positive drop lowers P; negative drop raises it.
    # Every colored cell is the result of a different covered-image run.
    for item in fine['trials']:
        drop = item['drop_percentage_points']
        color = '#be123c' if drop >= 0 else '#245edb'
        alpha = min(abs(drop)/4, 1)*.9
        body += (f'<rect x="{390+item["column"]/16*20}" y="{65+item["row"]/16*20}" '
                 f'width="20" height="20" fill="{color}" opacity="{alpha:.4f}"/>')
    body += f.rect(390+region[1]/16*20, 65+region[0]/16*20, 20, 20, 'c-q', 'transparent', 0)
    body += t(530, 374, 'Each square = one separate test', 22, 'ink-2', 'middle')
    body += t(725, 83, f'Largest drop: P{largest["patch_index"]}', 29, 'c-a', weight=650)
    body += t(725, 137, f'{baseline:.2%} → {largest["target_probability"]:.2%}', 37, 'c-e')
    body += t(725, 186, f'Down {largest["drop_percentage_points"]:.2f} percentage points', 26, 'c-a')
    body += t(725, 250, 'Red: probability falls', 25, 'c-a')
    body += t(725, 287, 'Blue: probability rises', 25, 'c-e')
    body += t(725, 326, 'Drop scale: −4 to +4 points', 23, 'ink-2')
    body += g(t(35, 431, 'All 196 tests still predict Newfoundland.', 34, 'c-e', weight=650), 1)
    f.add('occlusion-small-result', 'Smaller covers reveal local sensitivity', body,
          'Smaller covers localize sensitivity. P78 causes the largest drop here, but no single-patch cover changes the top label. This map measures probability changes, not attention weights.',
          'Does a small probability drop mean a patch contains no useful information?',
          f'P{largest["patch_index"]} causes the largest measured drop: {largest["drop_percentage_points"]:.2f} percentage points. '
          'Read each square as a different forward pass with one gray patch. Other regions can retain useful clues; these drops are not additive.',
          '<p>This is a new saved experiment on the same trained checkpoint and photograph. '
          f'The largest drop is at P{largest["patch_index"]}, grid row {region[0]//16+1}, column {region[1]//16+1} '
          '(counting from one). All 196 interventions retain Newfoundland as the top label. '
          f'The target probabilities range from {fine["summary"]["minimum_probability"]:.2%} '
          f'to {fine["summary"]["maximum_probability"]:.2%}. '
          'Three replacements slightly increase the target probability, so a cover need not always reduce it.</p>'
          '<p>Red means a positive drop, blue a negative drop, with a fixed symmetric −4 to +4 percentage-point color scale. '
          'The outlined patch is the same location in the one-test image and in the result map. '
          'This is neither a similarity map nor an attention map: it summarizes changes in the final class probability.</p>'
          '<p>The result depends on this image, target, model, cover size and gray fill. '
          'It locates sensitivity to these replacements, not an object segmentation or a complete explanation of recognition. '
          '<a href="figures/vision1/patch-occlusion.json">All measurements and verification</a> · '
          '<a href="notebooks/vision/inspect_patch_occlusion.py">Reproduce the experiment</a>.</p>',
          f'<p>Among 196 separate tests, covering P{largest["patch_index"]} gives the largest drop: '
          f'{baseline:.2%} → {largest["target_probability"]:.2%}, or {largest["drop_percentage_points"]:.2f} percentage points. '
          'All top labels remain Newfoundland. Red marks a probability decrease; blue marks an increase.</p>')

    result = []
    for title, frames in sections:
        out = []
        for markup in frames:
            key = re.search(r'class="frame[^\"]*" id="([^\"]+)"', markup).group(1)
            if key == 'cover-1':
                out.extend([f.frames['cover-pixels'], f.frames['cover-1']])
            elif key == 'occlusion':
                out.extend(f.frames[k] for k in ['occlusion', 'occlusion-small-setup', 'occlusion-small-result'])
            else:
                out.append(markup)
        result.append((title, out))
    return result
