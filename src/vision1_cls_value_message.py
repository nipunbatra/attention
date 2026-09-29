"""Follow one CLS weight row through the value matrix to its 64-feature message."""
import base64
import json


def build_cls_value_message(b, matrix):
    t, g, rect, arrow, line, image, frame, mobile_rows = (b[k] for k in
        ['t', 'g', 'rect', 'arrow', 'line', 'image', 'frame', 'mobile_rows'])
    result = {}
    photo = 'data:image/png;base64,' + base64.b64encode(
        (b['ASSETS']/'model-input.png').read_bytes()).decode()
    patch = json.loads((b['ASSETS']/'real-patch-path.json').read_text())
    trace = json.loads((b['ASSETS']/'real-classifier-path.json').read_text())
    a63 = trace['cls_head1_weights'][63]
    v63 = patch['head1_v']
    message = trace['previews']['cls_head1_message']
    tokens = ['CLS', 'P1', '…', 'P63', '…', 'P196']
    features = ['1', '2', '…', '64']
    evidence = (' <a href="figures/vision1/real-patch-path.json">Saved P63 value row</a> · '
                '<a href="figures/vision1/real-classifier-path.json">Saved CLS weights and message</a> · '
                '<a href="notebooks/vision/trace_real_classifier.py">Verified forward computation</a>.')

    def number(value, places=3):
        return f'{value:.{places}f}'.replace('-', '−')

    def crop(x, y, side):
        return (f'<svg x="{x}" y="{y}" width="{side}" height="{side}" '
                'viewBox="96 64 16 16" overflow="hidden">'
                + image(0, 0, 224, 224, photo) + '</svg>')

    def strip(x, y, w, entries, color='c-v', height=56, size=27, selected=None):
        out = ''
        for j, entry in enumerate(entries):
            cell = w/len(entries)
            out += rect(x+j*cell, y, cell, height, color, 'transparent', 0)
            if j == selected:
                out += (f'<rect x="{x+j*cell}" y="{y}" width="{cell}" height="{height}" '
                        f'fill="var(--{color})" fill-opacity=".14"/>')
            out += t(x+(j+.5)*cell, y+height/2+9, entry, size, color, 'middle')
        return out

    def add(key, title, body, caption, notes, prose, mobile):
        result[key] = frame(key, title, body, caption, notes, prose+evidence, mobile)

    body = t(35, 34, 'Same dog · block 1 · head 1', 29)
    body += image(35, 106, 200, 200, photo)
    for j in range(1, 14):
        body += line(35+j*200/14, 106, 35+j*200/14, 306, 'card', .6)
        body += line(35, 106+j*200/14, 235, 106+j*200/14, 'card', .6)
    body += rect(35+6*200/14, 106+4*200/14, 200/14, 200/14, 'c-q', 'transparent', 0)
    body += t(135, 335, '196 patches + CLS', 22, 'ink-2', 'middle')
    body += arrow(245, 210, 300, 210)
    body += t(433, 82, 'X = LayerNorm(E)', 23, 'c-e', 'middle')
    body += matrix(368, 121, 130, 184, tokens, ['1', '2', '…', '192'], 'c-e', row=3, size=17)
    body += t(433, 335, '197 × 192', 25, 'c-e', 'middle')
    body += g(arrow(519, 210, 547, 210, 'c-v')+rect(558, 170, 174, 80, 'c-v', 'transparent')
              +t(645, 203, 'Linear(192,64)', 22, 'c-v', 'middle')
              +t(645, 234, '× W_V + b_V', 21, 'c-v', 'middle')
              +arrow(738, 210, 768, 210, 'c-v'), 1)
    body += g(t(947, 82, 'V: one value row per source', 23, 'c-v', 'middle')
              +matrix(825, 121, 244, 184, tokens, features, 'c-v', row=3, size=17)
              +t(947, 335, '197 × 64', 25, 'c-v', 'middle'), 1)
    body += g(crop(35, 360, 60)+t(124, 389, 'P63 → v₆₃ = ['
              +number(v63[0])+', '+number(v63[1])+', …, '+number(v63[-1])+']', 28, 'c-v')
              +t(124, 425, '64 learned features · rounded values from this dog', 23, 'ink-2'), 2)
    add('real-cls-values-origin', 'The dog’s feature rows become value rows', body,
        'Use the same normalized input X that produced Q and K. The value projection makes 64 features for each of its 197 rows. P63’s crop identifies one source; its value row contains learned features.',
        'Where does row P63 of V come from?\nFollow P63’s normalized 192-feature row through the shared value projection. It produces 64 features. CLS has a value row too, although it has no image crop.',
        'This is head 1 of block 1 in the saved dog forward pass. The image’s patch projection and positions '
        'already produced E; we do not flatten or project pixels again here. X=LayerNorm(E), then '
        'V=XW_V+b_V with W_V shaped 192×64 and a 64-coordinate bias, shared across source rows. '
        'There is no additional activation after this value projection. The diagram abbreviates matrix entries; '
        'the P63 preview is read from the saved head1_v array, whose length is 64. These coordinates are learned '
        'features rather than RGB channels. The displayed dog grid connects source identity to a crop; CLS '
        'comes from the extra learned row explained earlier.',
        '<img src="figures/vision1/model-input.png" alt="The same dog photograph" width="160">'
        '<p>X = LayerNorm(E): <strong>197 × 192</strong>. Apply the same Linear(192,64) to every row.</p>'
        '<p>V = X W_V + b_V: <strong>197 × 64</strong>, ordered CLS, P1, …, P196.</p>'
        '<p>P63’s value row begins ['+number(v63[0])+', '+number(v63[1])+', …] and has 64 learned features.</p>')

    sources = ['CLS', 'P1', 'P2', '…', 'P63', '…', 'P196']
    body = t(35, 34, 'Select P63’s weight from the CLS row', 28, 'c-q')
    for j, label in enumerate(sources):
        body += t(270+(j+.5)*850/7, 74, label, 22, 'ink-2', 'middle')
    body += strip(270, 88, 850, ['a₀', 'a₁', 'a₂', '…', 'a₆₃', '…', 'a₁₉₆'], 'c-q', selected=4)
    body += t(150, 125, '1 × 197', 27, 'c-q', 'middle')
    body += crop(35, 203, 78)+t(74, 316, 'P63', 24, 'c-v', 'middle')
    body += t(194, 231, 'a₆₃', 31, 'c-q', 'middle')+t(194, 274, number(a63, 6), 25, 'c-q', 'middle')
    body += t(290, 242, '×', 36)
    body += t(685, 190, 'Value row v₆₃ · 1 × 64', 26, 'c-v', 'middle')
    body += strip(330, 210, 710, [number(v63[0]), number(v63[1]), '…', number(v63[-1])])
    scaled = arrow(685, 278, 685, 333, 'c-v')+t(711, 313, 'scale every feature', 24, 'c-v')
    scaled += t(194, 387, 'a₆₃ v₆₃', 30, 'c-q', 'middle')
    scaled += strip(330, 345, 710, [number(a63*v63[0], 6), number(a63*v63[1], 6), '…', number(a63*v63[-1], 6)], size=25)
    body += g(scaled, 1)
    body += t(330, 436, 'One weighted value row · still 1 × 64', 26, 'c-v')
    add('real-cls-value-scaling', 'One weight scales all 64 features in its value row', body,
        'P63’s CLS weight multiplies every coordinate of P63’s value row. The result still has 64 features. Repeat for CLS itself and every patch. The numbers shown are rounded from the saved dog computation.',
        'Does a₆₃ multiply one feature, or all 64 features?\nIt scales the entire P63 value row. This is one source’s contribution to the CLS message, not the completed message.',
        'For this fixed CLS query, a_j means A[CLS,j]. Source 0 is CLS itself. '
        'The saved weight a₆₃ is '+str(a63)+'. Its product with the first value feature is '
        +str(a63*v63[0])+', and the second product is '+str(a63*v63[1])+'. '
        'The displayed products use the full stored precision before rounding. The same scalar multiplies '
        'all 64 coordinates; no feature dimension is removed. Values may be negative, so weighted '
        'features can be negative even though the softmax weights are nonnegative. This crop does not '
        'supply a probability for a class; it supplies a learned value vector. Repeat this multiplication '
        'for all 197 sources, including CLS. The next slide adds their contributions.',
        '<p>From the CLS weight row (1 × 197), select a₆₃ = '+number(a63, 6)+'.</p>'
        +mobile_rows(['Feature of P63', 'Value', 'Weight × value'],
                     [[str(j+1), number(v63[j]), number(a63*v63[j], 6)] for j in [0, 1, 63]])
        +'<p><strong>a₆₃ × v₆₃ → one weighted row of shape 1 × 64.</strong> '
        'Every feature gets the same scalar weight. Repeat for all 197 sources.</p>')

    body = t(35, 34, 'Same image and head · keep the CLS query fixed', 28)
    body += t(170, 134, 'CLS weights · 1 × 197', 25, 'c-q', 'middle')
    for j, label in enumerate(['CLS', 'P1', '…', 'P196']):
        body += t(35+(j+.5)*270/4, 177, label, 19, 'ink-2', 'middle')
    body += strip(35, 193, 270, ['a₀', 'a₁', '…', 'a₁₉₆'], 'c-q', size=28)
    body += t(348, 232, '×', 37)
    body += t(585, 82, 'V · 197 × 64', 28, 'c-v', 'middle')
    subscripts = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
    entries = {(r, c): ('v'+str(source).translate(subscripts)+','+str(feature).translate(subscripts)
                       if source is not None and feature is not None else '…')
               for r, source in enumerate([0, 1, None, 63, None, 196])
               for c, feature in enumerate([1, 2, None, 64])}
    body += matrix(465, 126, 240, 216, tokens, features, 'c-v', col=0, size=19, entries=entries)
    # An actual column sum is exposed next to the result, rather than hiding AV in a box.
    output = arrow(740, 221, 821, 221, 'c-v')
    output += t(985, 134, 'CLS message · 1 × 64', 25, 'c-v', 'middle')
    for j, label in enumerate(features):
        output += t(850+(j+.5)*270/4, 177, label, 19, 'ink-2', 'middle')
    output += strip(850, 193, 270, ['h₁', 'h₂', '…', 'h₆₄'], selected=0)
    output += t(985, 286, '64 feature coordinates', 24, 'c-v', 'middle')
    body += g(output, 1)
    body += g(t(35, 384, 'h₁ = a₀v₀,₁ + a₁v₁,₁ + … + a₁₉₆v₁₉₆,₁', 30, 'c-v')
              +t(35, 431, 'Add down the 197 sources. Keep all 64 feature columns.', 28), 1)
    body += g(t(985, 329, '['+number(message[0])+', '+number(message[1])+', …]', 26, 'c-v', 'middle'), 2)
    add('real-cls-value-sum', 'Add the weighted rows to make one CLS message', body,
        'Multiply the CLS weight row by V: (1 × 197) × (197 × 64) gives 1 × 64. Each output feature adds 197 weighted contributions. This is the message for one query in one head.',
        'Why does the result have 64 coordinates, rather than 197?\nWe sum over sources. Each column of V contributes one feature to the output row. All 64 feature columns remain; each contains a sum over 197 weighted sources.',
        'Here h₁ denotes feature 1 of the CLS message, not head 1; the entire diagram follows head 1. '
        'For source j and feature d, v_{j,d} is entry V[j,d]. Thus h_d = Σ_{j=0}^{196} a_j v_{j,d}. '
        'The first column highlight shows precisely the 197 values used to compute h₁. '
        'The row count one refers to the receiving CLS query, not to the batch dimension. '
        'Other patch queries have their own weight rows and receive their own 64-feature messages; '
        'the next slide restores them to make H with shape 197×64. The saved CLS message begins '
        +', '.join(number(v) for v in message)+'. The matrix dots omit entries; the numerical preview '
        'is measured from the same dog, block and head. This message precedes head concatenation, '
        'output projection and the residual addition; it is not the final 192-feature image representation.',
        '<p><strong>CLS weights (1 × 197) × V (197 × 64) → CLS message (1 × 64).</strong></p>'
        '<p>For output feature d: h_d = a₀v₀,d + a₁v₁,d + … + a₁₉₆v₁₉₆,d.</p>'
        '<p>Sum over the 197 sources. Retain all 64 feature columns.</p>'
        '<p>Measured message begins ['+', '.join(number(v) for v in message)+', …]. '
        'This is one CLS query’s message in one head, before the heads are joined.</p>')
    return result
