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

    def photo(x, y, size, covered=False):
        out = f.image(x, y, size, size, f.photo)
        if covered:
            out += f'<rect x="{x}" y="{y}" width="{size/2}" height="{size/2}" fill="rgb(128,128,128)"/>'
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

    result = []
    for title, frames in sections:
        out = []
        for markup in frames:
            key = re.search(r'class="frame[^\"]*" id="([^\"]+)"', markup).group(1)
            if key == 'cover-1':
                out.extend([f.frames['cover-pixels'], f.frames['cover-1']])
            else:
                out.append(markup)
        result.append((title, out))
    return result
