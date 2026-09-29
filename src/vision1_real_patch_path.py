"""Follow one measured photo patch from pixels to the first attention head."""
import base64
import json


def add_real_path(b, add):
    t, g, rect, arrow, line, image, mobile_rows = (b[k] for k in
        ['t', 'g', 'rect', 'arrow', 'line', 'image', 'mobile_rows'])
    data = json.loads((b['ASSETS'] / 'real-patch-path.json').read_text())
    uri = 'data:image/png;base64,' + base64.b64encode((b['ASSETS']/'model-input.png').read_bytes()).decode()
    evidence = (' <a href="figures/vision1/real-patch-path.json">Measured vectors and shapes</a> · '
                '<a href="notebooks/vision/trace_real_patch.py">Reproduce this trace</a>.')

    def num(value):
        return f'{value:.3f}'.replace('-', '−')

    def vec(values, n=3):
        return '[' + ', '.join(num(v) for v in values[:n]) + ', …]'

    def box(x, y, w, label, color='c-e', size=28):
        return rect(x, y, w, 64, color, 'transparent') + t(x+w/2, y+42, label, size, color, 'middle')

    def patch(x, y, side):
        return (f'<svg x="{x}" y="{y}" width="{side}" height="{side}" '
                'viewBox="96 64 16 16" overflow="hidden">' + image(0, 0, 224, 224, uri) + '</svg>'
                + rect(x, y, side, side, 'c-q', 'transparent', 0))

    def photo_grid(x, y, side):
        out = image(x, y, side, side, uri)
        cell = side/14
        for k in range(1, 14):
            out += line(x+k*cell, y, x+k*cell, y+side, 'card', .65)
            out += line(x, y+k*cell, x+side, y+k*cell, 'card', .65)
        out += rect(x+6*cell, y+4*cell, cell, cell, 'c-q', 'transparent', 0)
        return out

    def mobile_patch():
        return ('<svg viewBox="0 0 360 160" role="img" aria-label="Patch 63, a 16 by 16 crop from the dog photo">'
                + patch(15, 12, 125) + t(165, 60, 'P63', 26, 'c-q')
                + t(165, 100, '16 × 16 × 3', 23, 'c-e') + '</svg>')

    from vision1_photo_walkthrough import add_photo_walkthrough
    add_photo_walkthrough(b, add)

    # One coordinate-wise addition; do not hide the measured patch projection.
    body = patch(35, 45, 100) + t(85, 180, 'P63', 26, 'c-q', 'middle')
    body += t(175, 85, 'We already computed c₆₃ using the patch projection.', 28)
    body += t(175, 142, 'Add its position row; repeat for every image patch.', 30, 'c-e')
    headings = [(35, 'content from pixels', 'c₆₃', 'content', 'c-e'),
                (455, 'row 63 of the learned table', 'p₆₃', 'position', 'c-q'),
                (865, 'input to the block', 'e₆₃', 'embedding', 'c-v')]
    for i, (x, label, symbol, key, color) in enumerate(headings):
        marks = t(x, 240, label, 25, 'ink-2') + t(x, 287, symbol+' · 1 × 192', 31, color)
        marks += t(x, 340, vec(data[key], 2), 26, color)
        if i: marks += t(x-57, 306, '+' if i==1 else '=', 38)
        body += marks if i==0 else g(marks, i)
    body += g(t(35, 416, 'All patch rows:  C (196 × 192) + P (196 × 192) = E (196 × 192)', 28, 'c-v'), 2)
    add('real-patch-position', '11 · Add position to these content rows', body,
        'c₆₃ comes from the patch pixels. p₆₃ is learned for its grid location. Add matching coordinates to get e₆₃. All three rows have 192 coordinates; addition does not double the width.',
        'After adding position, does this row have 192 coordinates or 384?',
        'Keep the crop visible. Recall the projection that made c63, then add its learned position row coordinate by coordinate to make e63.',
        'The pretrained checkpoint learns a 192-entry position vector for each input slot. P63 uses the position row for grid row 5, column 7. '
        'That learned row is shared across images at this input resolution; the content row changes with the pixels in the slot. '
        'The first two coordinates shown are '+vec(data['content'], 2)+' + '+vec(data['position'], 2)+' = '+vec(data['embedding'], 2)+'. '
        'For the first coordinate, −0.851541 − 0.814817 ≈ −1.666358. The full precision calculation gives e₆₃[0]=−1.6663575. The printed rounded inputs introduce rounding error, so the displayed arithmetic uses ≈. '
        'cᵢ, pᵢ and eᵢ all have shape (1,192). We add them, rather than concatenate them. '
        'This is the same content-plus-position idea used for text tokens. e63 is the input row to the first Transformer block. '
        'The preceding diagram showed how this grid slot selects row 63 of the shared position table. Next we trace how the image loss trains that table, then introduce the summary row before entering the first block. '
        'The trace checks this sum against the checkpoint’s own position-addition operation; its position table also contains the classification-token slot introduced later.' + evidence,
        mobile_patch()+mobile_rows(['Row · shape 1 × 192', 'First two coordinates'],[
            ['c₆₃: projected content', vec(data['content'], 2)],
            ['+ p₆₃: position', vec(data['position'], 2)],
            ['= e₆₃: block input', vec(data['embedding'], 2)]])
        +'<p>All coordinates are added in matching positions. The width remains 192. Values are rounded.</p>')

    body = patch(35, 25, 80) + t(140, 52, 'P63: pixels → patch projection → c₆₃ → add p₆₃ → e₆₃', 27)
    body += t(140, 88, 'All 197 rows are ready. Follow patch 63 into the first block.', 25, 'ink-2')
    body += box(35, 215, 190, 'e₆₃', size=31) + t(130, 321, '1 × 192', 25, 'ink-2', 'middle') + arrow(225, 247, 282, 247)
    body += g(box(300, 215, 235, 'LayerNorm', size=29) + t(417, 321, 'still 1 × 192', 25, 'ink-2', 'middle'), 1)
    body += g(line(535, 247, 585, 247) + line(585, 171, 585, 377), 2)
    for y, name, color in [(139, 'Q', 'c-q'), (242, 'K', 'c-k'), (345, 'V', 'c-v')]:
        body += g(arrow(585, y+32, 632, y+32, color) + box(650, y, 280, '× W_'+name+' + b_'+name, color, 28)
                  + arrow(930, y+32, 980, y+32, color) + t(1002, y+29, name.lower()+'₆₃', 31, color)
                  + t(1002, y+62, '1 × 64', 23, color), 2)
    body += t(650, 126, 'Each W: 192 × 64; each bias: 64', 24, 'ink-2')
    add('real-patch-qkv', '12 · Make queries, keys and values from these rows', body,
        'LayerNorm rescales a row’s features and keeps its width. We name it here and study its details later. Three learned maps then make Q, K and V; the diagram shows one of three heads.',
        'Which projection reads pixels, and which projections read the prepared embedding row?',
        'Trace the small recap first. Enter the block with e63, normalize it, then split into the three separate learned projections for head one.',
        'In this pretrained pre-LN Transformer, z₆₃=LayerNorm(e₆₃). For head one, q₆₃=z₆₃W_Q+b_Q, k₆₃=z₆₃W_K+b_K and v₆₃=z₆₃W_V+b_V. '
        'Each W has row-vector shape (192,64), each bias has 64 entries, and each output is a (1,64) row. '
        'The branch boxes show multiplication by the appropriate weight matrix and addition of the corresponding bias. '
        'The model’s embedding width is 192 and it uses three heads, so each head has 64 features. Other model choices may use different widths. '
        'The checkpoint packs Q/K/V parameters into one layer; our script verifies the head-one slices against the packed result. '
        'Every patch follows the same path. The query for this patch is compared with source keys; attention weights then mix source values to update this patch’s representation. '
        'That exchange is worked out later in the lecture. Neither patch embeddings nor Q/K/V coordinates are class probabilities.' + evidence,
        mobile_patch()+'<p><strong>Pixels → patch projection → c₆₃ → add p₆₃ → e₆₃</strong></p>'
        +'<p>e₆₃ (1 × 192) → LayerNorm → z₆₃ (1 × 192).</p>'
        +mobile_rows(['Head one · separate learned maps', 'Output shape'],[
            ['q₆₃ = z₆₃W_Q + b_Q', '1 × 64'], ['k₆₃ = z₆₃W_K + b_K', '1 × 64'], ['v₆₃ = z₆₃W_V + b_V', '1 × 64']])
        +'<p>Each W: 192 × 64. Each bias: 64 entries. This model uses three heads.</p>')
