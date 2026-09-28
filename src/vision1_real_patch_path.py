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

    # Return to the actual photograph before using realistic feature dimensions.
    body = t(35, 33, 'Same photograph; now the model’s actual patch grid.', 28)
    body += photo_grid(35, 72, 280) + t(175, 393, '224 × 224 × 3', 28, 'c-e', 'middle')
    body += g(line(175, 152, 342, 152, 'c-q') + arrow(342, 152, 398, 205, 'c-q')
              + patch(405, 132, 160) + t(485, 331, 'P63 · enlarged', 25, 'c-q', 'middle'), 1)
    body += g(t(660, 115, 'One patch: 16 × 16 pixels', 30, 'c-e')
              + t(660, 173, '3 RGB values at each pixel', 28)
              + t(660, 256, '14 × 14 = 196 patches', 30)
              + t(660, 320, 'P63: row 5, column 7', 27, 'c-q')
              + t(660, 394, 'We will follow this patch throughout.', 25, 'ink-2'), 2)
    mobile = ('<svg viewBox="0 0 360 420" role="img" aria-label="Dog image with a 14 by 14 grid and patch 63 highlighted">'
              + photo_grid(40, 8, 280) + arrow(180, 100, 280, 335, 'c-q')
              + patch(270, 335, 65) + t(25, 337, 'P63 · row 5, column 7', 20, 'c-q')
              + t(25, 386, '224 × 224 RGB image', 23, 'c-e') + '</svg>')
    add('s01-rows', 'Back to the dog: follow one real patch', body,
        'The earlier 4 × 4 grid was an illustration. This model uses a 14 × 14 grid. We now follow P63, a 16 × 16 RGB patch from the same photograph.',
        'Which part of this photograph will supply our next row of numbers?',
        'Find the purple square in the real input grid, follow the line to its enlarged crop, and keep P63 as the example for the next steps.',
        'The saved ViT-Tiny checkpoint receives this exact 224×224 resize and center crop of our Newfoundland photograph. '
        'It divides the image into 196 non-overlapping 16×16 patches. Numbering patches from 1 in row-major order, row 5 and column 7 identify P63. '
        'The earlier P6 and P7 labels belonged to a coarse 4×4 illustration; changing the grid changes those indices. '
        'We use the model’s actual pixel values and learned parameters for the following trace. Values printed on slides are rounded; full precision is saved with the model name and preprocessing.' + evidence,
        mobile + '<p>14 × 14 = 196 patches. Each patch contains 16 × 16 pixels, each with three RGB channel values.</p>')

    # Show the shared projection explicitly, with a measured input and output.
    body = patch(35, 78, 135) + t(102.5, 253, 'P63', 29, 'c-q', 'middle')
    body += arrow(178, 143, 258, 143) + t(280, 70, 'x₆₃ · normalized RGB values', 28, 'c-e')
    body += t(280, 126, vec(data['normalized_rgb_row'], 3), 32, 'c-e')
    body += t(280, 170, 'first pixel: R, G, B', 23, 'ink-2') + t(280, 229, '1 × 768 · one patch row', 27)
    body += g(arrow(632, 143, 718, 143) + box(740, 105, 365, 'nn.Linear(768, 192)', size=29)
              + t(922, 219, 'W_patch: 768 × 192', 25, 'c-e', 'middle')
              + t(922, 260, 'b_patch: 192 values', 25, 'c-e', 'middle'), 1)
    body += g(arrow(922, 274, 922, 325, 'c-e') + box(690, 335, 420, 'c₆₃ · 1 × 192')
              + arrow(680, 367, 570, 367, 'c-e')
              + t(35, 337, 'c₆₃ = x₆₃W_patch + b_patch', 28, 'c-e')
              + t(35, 393, vec(data['content']), 32, 'c-e'), 2)
    add('real-patch-projection', 'The patch projection produces c₆₃', body,
        'The shared patch layer turns P63’s 768 pixel values into c₆₃, a row of 192 learned features. These are measured coordinates from the dog image, rounded here. No activation follows this projection.',
        'Which operation changes 768 input values into 192 content features?',
        'Follow the actual crop to x63, through the named patch layer, and down to c63. Separate the patch index 63 from the feature width 192.',
        'The first three entries shown are the normalized R, G and B of P63’s top-left pixel. '
        'Continue with RGB of the next pixel, move left to right across the patch, then start the next row. '
        'This checkpoint uses (RGB/255−0.5)/0.5 for channel normalization. All 768 entries contribute to the 192 learned weighted sums. '
        'The first three content coordinates are '+vec(data['content'])+'. The subscript 63 identifies the patch; it does not count coordinates. '
        'The patch layer is mathematically nn.Linear(768,192), with no following activation. In this checkpoint it is implemented as a stride-16 Conv2d. '
        'Our verification script rearranges its kernel weights to match pixel-major RGB order and checks the linear result against the actual patch embedding output. '
        'The same weights and biases are used for every patch.' + evidence,
        mobile_patch() + '<p><strong>x₆₃ · 1 × 768</strong><br>First RGB triple: '+vec(data['normalized_rgb_row'])+'</p>'
        + '<p>↓ <strong>Shared nn.Linear(768, 192)</strong><br>W_patch: 768 × 192; bias: 192 values.</p>'
        + '<p>↓ <strong>c₆₃ · 1 × 192</strong><br>'+vec(data['content'])+'</p><p>No activation after this layer.</p>')

    # One coordinate-wise addition; do not hide the measured patch projection.
    body = patch(35, 45, 100) + t(85, 180, 'P63', 26, 'c-q', 'middle')
    body += t(175, 85, 'We already computed c₆₃ using the patch projection.', 28)
    body += t(175, 142, 'Now give this row its location in the image.', 30, 'c-e')
    headings = [(35, 'content from pixels', 'c₆₃', 'content', 'c-e'),
                (455, 'learned position row', 'p₆₃', 'position', 'c-q'),
                (865, 'input to the block', 'e₆₃', 'embedding', 'c-v')]
    for i, (x, label, symbol, key, color) in enumerate(headings):
        marks = t(x, 240, label, 25, 'ink-2') + t(x, 287, symbol+' · 1 × 192', 31, color)
        marks += t(x, 340, vec(data[key], 2), 26, color)
        if i: marks += t(x-57, 306, '+' if i==1 else '=', 38)
        body += marks if i==0 else g(marks, i)
    body += g(t(35, 416, 'First coordinate:  −0.851541 − 0.814817 ≈ −1.666358', 30, 'c-v'), 2)
    add('real-patch-position', 'Content + position gives the block’s input row', body,
        'c₆₃ comes from the patch pixels. p₆₃ is learned for its grid location. Add matching coordinates to get e₆₃. All three rows have 192 coordinates; addition does not double the width.',
        'After adding position, does this row have 192 coordinates or 384?',
        'Keep the crop visible. Recall the projection that made c63, then add its learned position row coordinate by coordinate to make e63.',
        'The pretrained checkpoint learns a 192-entry position vector for each input slot. P63 uses the position row for grid row 5, column 7. '
        'That learned row is shared across images at this input resolution; the content row changes with the pixels in the slot. '
        'The first two coordinates shown are '+vec(data['content'], 2)+' + '+vec(data['position'], 2)+' = '+vec(data['embedding'], 2)+'. '
        'The full precision calculation gives e₆₃[0]=−1.6663575. The printed rounded inputs introduce rounding error, so the displayed arithmetic uses ≈. '
        'cᵢ, pᵢ and eᵢ all have shape (1,192). We add them, rather than concatenate them. '
        'This is the same content-plus-position idea used for text tokens. e63 is the input row to the first Transformer block. '
        'The next slide shows its LayerNorm and Q/K/V projections; the next section motivates why the model needs position information. '
        'The trace checks this sum against the checkpoint’s own position-addition operation; its position table also contains the classification-token slot introduced later.' + evidence,
        mobile_patch()+mobile_rows(['Row · shape 1 × 192', 'First two coordinates'],[
            ['c₆₃: projected content', vec(data['content'], 2)],
            ['+ p₆₃: position', vec(data['position'], 2)],
            ['= e₆₃: block input', vec(data['embedding'], 2)]])
        +'<p>All coordinates are added in matching positions. The width remains 192. Values are rounded.</p>')

    body = patch(35, 25, 80) + t(140, 52, 'P63: pixels → patch projection → c₆₃ → add p₆₃ → e₆₃', 27)
    body += t(140, 88, 'That completes the input row. Now enter the first Transformer block.', 25, 'ink-2')
    body += box(35, 215, 190, 'e₆₃', size=31) + t(130, 321, '1 × 192', 25, 'ink-2', 'middle') + arrow(225, 247, 282, 247)
    body += g(box(300, 215, 235, 'LayerNorm', size=29) + t(417, 321, 'still 1 × 192', 25, 'ink-2', 'middle'), 1)
    body += g(line(535, 247, 585, 247) + line(585, 171, 585, 377), 2)
    for y, name, color in [(139, 'Q', 'c-q'), (242, 'K', 'c-k'), (345, 'V', 'c-v')]:
        body += g(arrow(585, y+32, 632, y+32, color) + box(650, y, 280, '× W_'+name+' + b_'+name, color, 28)
                  + arrow(930, y+32, 980, y+32, color) + t(1002, y+29, name.lower()+'₆₃', 31, color)
                  + t(1002, y+62, '1 × 64', 23, color), 2)
    body += t(650, 126, 'Each W: 192 × 64; each bias: 64', 24, 'ink-2')
    add('real-patch-qkv', 'Then the block makes queries, keys and values', body,
        'The patch projection made content features from pixels. Inside this block, LayerNorm comes first; three more learned projections produce Q, K and V. This model has 3 heads; we show one head’s shapes.',
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
