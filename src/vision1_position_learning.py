"""Explain the stored position table and how image-label loss trains it."""
import base64
import json


def build_position_learning(b):
    t, g, rect, arrow, line, image, frame, mobile_rows = (b[k] for k in
        ['t', 'g', 'rect', 'arrow', 'line', 'image', 'frame', 'mobile_rows'])
    data = json.loads((b['ASSETS'] / 'real-patch-path.json').read_text())
    width = len(data['position'])
    patches = data['all_content_shape'][0]
    count = patches * width
    projection_count = data['projection_weight_shape'][0] * width + width
    preview = '[' + ', '.join(f'{v:.3f}'.replace('-', '−') for v in data['position'][:2]) + ', …]'
    photo = 'data:image/png;base64,' + base64.b64encode((b['ASSETS'] / 'model-input.png').read_bytes()).decode()
    sources = (' <a href="https://arxiv.org/html/2010.11929v2#S3.SS1">ViT paper, §3.1</a> · '
               '<a href="https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py">timm implementation</a> · '
               '<a href="https://arxiv.org/html/1706.03762v7#S3.SS5">Original Transformer position encodings</a>.')
    frames = {}

    def add(key, title, body, caption, question, explanation, prose, mobile):
        frames[key] = frame(key, title, body, caption, question+'\n'+explanation, prose+sources, mobile)

    body = image(35, 80, 252, 252, photo)
    for k in range(1, 14):
        body += line(35+18*k, 80, 35+18*k, 332, 'card', .7)
        body += line(35, 80+18*k, 287, 80+18*k, 'card', .7)
    body += rect(143, 152, 18, 18, 'c-q', 'transparent', 0)
    body += t(161, 373, 'P63: row 5, column 7', 25, 'c-q', 'middle')
    body += t(390, 35, f'P: {patches} grid slots × {width} coordinates', 30, 'c-q')
    body += rect(390, 62, 710, 240, 'line', 'transparent')
    body += g(rect(392, 181, 706, 43, 'c-q', 't-q'), 1)
    for y, label, values in [(94, 'P1', '[ … 192 coordinates … ]'),
                             (134, 'P2', '[ … 192 coordinates … ]'),
                             (167, '⋮', '⋮'), (210, 'P63', preview),
                             (247, '⋮', '⋮'), (286, 'P196', '[ … 192 coordinates … ]')]:
        body += t(415, y, label, 25, 'c-q' if label=='P63' else 'ink-2')
        body += t(560, y, values, 27, 'c-q' if label=='P63' else 'ink-2')
    lookup = line(161, 161, 320, 161, 'c-q') + line(320, 161, 320, 202, 'c-q')
    lookup += arrow(320, 202, 390, 202, 'c-q')
    body += g(lookup+t(390, 342, 'Slot 63 selects p₆₃ for every image.', 28, 'c-q'), 1)
    body += g(t(390, 386, f'{patches} × {width} = {count:,} learned numbers', 30, 'c-q')
              +t(390, 429, f'Compare: patch projection has {projection_count:,} parameters.', 24, 'ink-2'), 2)
    add('position-table', 'Position is a learned lookup table', body,
        'Each grid slot has a trainable 192-number row, shared across images. Slot 63 always selects p₆₃; its pixel-derived content changes from photo to photo. The patch-position table has 37,632 parameters.',
        'Would a second photograph require a second table of position vectors?',
        'No: point from row 5, column 7 into the same stored row. Read the parameter count and compare it with the patch projection already calculated. The displayed p63 entries come from the saved checkpoint.',
        'This checkpoint uses learned absolute position embeddings, as in the original ViT. '
        'Patch positions are numbered in raster order. The index selects a parameter row, much like an embedding lookup for a text token ID. '
        'A slot has the same position vector for every image at this input resolution; the vector is not computed from that slot’s pixels. '
        'The table begins with small random values when training from scratch. The values shown for p63 are already trained values from the lecture checkpoint. '
        f'The patch slots require {patches}×{width}={count:,} parameters, compared with {projection_count:,} in the patch projection. '
        'This table does not grow when more training images are added. It is shared across the dataset. '
        f'The full checkpoint also reserves a position row for the CLS summary introduced next: {patches+1}×{width}={count+width:,} parameters. '
        'That CLS position row and the learned CLS token are separate parameters. '
        '<a href="figures/vision1/real-patch-path.json">Saved patch and position vectors</a>.',
        mobile_rows(['Item', 'Meaning'], [
            ['P', '196 patch slots × 192 coordinates'], ['P63', 'Row 5, column 7 selects p₆₃'],
            ['Saved p₆₃ preview', preview], ['Across images', 'Reuse the same table'],
            ['Patch-position parameters', f'{count:,}'], ['Patch-projection parameters', f'{projection_count:,}'],
            ['Including the later CLS position', f'197 × 192 = {count+width:,}']]))

    def box(x, y, w, labels, color='c-e'):
        result = rect(x, y, w, 66, color, 'transparent')
        for j, label in enumerate(labels):
            result += t(x+w/2, y+(41 if len(labels)==1 else 27+j*27), label, 25, color, 'middle')
        return result

    body = t(35, 32, 'Start with small random values; train with the class loss.', 28)
    body += box(35, 112, 190, ['p₆₃: 192 entries'], 'c-q')
    body += arrow(225, 145, 308, 145, 'c-q')
    body += '<circle cx="330" cy="145" r="22" fill="none" stroke="var(--ink)" stroke-width="2"/>'
    body += t(330, 155, '+', 35, 'ink', 'middle')
    body += t(330, 78, 'c₆₃ from pixels', 25, 'c-e', 'middle')+arrow(330, 89, 330, 123, 'c-e')
    body += arrow(352, 145, 415, 145)+box(415, 112, 190, ['e₆₃ = c₆₃ + p₆₃'])
    body += arrow(605, 145, 670, 145)+box(670, 112, 255, ['Transformer blocks', '+ classifier'])
    body += arrow(925, 145, 1000, 145)+box(1000, 112, 130, ['class loss', 'L'], 'c-a')
    backward = line(1065, 178, 1065, 228, 'c-a')+line(1065, 228, 130, 228, 'c-a')
    backward += arrow(130, 228, 130, 178, 'c-a')
    backward += t(580, 267, 'Backpropagation: gradients reach every trainable position row.', 26, 'c-a', 'middle')
    body += g(backward, 1)
    body += g(t(35, 319, 'Illustrative SGD update of one entry', 25, 'ink-2')
              +t(35, 366, 'p₆₃,₁: 0.020 − 0.1 × 0.30 = −0.010', 29, 'c-q')
              +t(35, 409, 'old value − learning rate × gradient', 25, 'ink-2'), 2)
    alternative = rect(670, 294, 460, 133, 'c-v', 'transparent')
    alternative += t(900, 327, 'Alternative: fixed sinusoidal', 26, 'c-v', 'middle')
    alternative += t(900, 367, 'sin/cos(position, coordinate)', 25, 'c-v', 'middle')
    alternative += t(900, 405, 'No learned table entries.', 24, 'ink-2', 'middle')
    body += g(alternative, 3)
    add('position-learning', 'The image loss teaches the position table', body,
        'The class loss supplies gradients for the position table and other model parameters. Training updates the table; inference reuses it unchanged. Fixed sinusoidal encodings instead compute position vectors from a formula.',
        'What tells the optimizer how to change a position coordinate: a position label, or the image classification loss?',
        'Trace p63 through addition into the classifier, then follow the backward arrow. The loss updates the table jointly with other weights. Use the marked illustrative SGD step to explain one coordinate, then distinguish fixed sinusoidal vectors.',
        'Every image uses the same trainable table. During training, automatic differentiation follows the classification loss back through '
        'the Transformer and the addition eᵢ=cᵢ+pᵢ. For one image, the direct addition gives ∂L/∂pᵢ=∂L/∂eᵢ. '
        'When a table is broadcast across a minibatch, its gradient combines contributions from those images according to the loss reduction. '
        'The optimizer updates its coordinates together with the patch projection, attention and classifier parameters. '
        'Grid slot identity is already known from patch extraction; no extra position-label prediction loss is needed. '
        'All coordinates are updated by tensor operations, rather than 196 separately trained models. '
        'The SGD numbers are a hand-chosen arithmetic example, not a measured update to the pretrained checkpoint; practical training may use AdamW. '
        'Inference performs the additions with the trained table but makes no optimizer updates. '
        'Fixed sinusoidal encodings are another design: a deterministic sin/cos function of position and coordinate supplies the vector, '
        'so there are no trainable entries for that encoding. The original Transformer studied both choices; the original ViT and this checkpoint use a learned table. '
        'The choice is part of the architecture, not something implied by the word position.',
        mobile_rows(['Step', 'What happens'], [
            ['Initialize', 'Small random values, once when training from scratch'],
            ['Forward', 'c₆₃ + p₆₃ → Transformer → class scores → loss'],
            ['Backward', 'The loss supplies gradients for the shared position table'],
            ['Illustrative SGD', '0.020 − 0.1 × 0.30 = −0.010'],
            ['Inference', 'Reuse the trained table without updates'],
            ['Fixed sinusoidal alternative', 'Compute sin/cos values; no trainable encoding entries']]))
    return frames
