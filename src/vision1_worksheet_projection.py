"""Draw the worksheet's patch projection as input and output neurons."""


def build_worksheet_projection(b):
    t, g, rect, arrow, line, pixels, frame = (b[k] for k in
        ['t', 'g', 'rect', 'arrow', 'line', 'pixels', 'frame'])
    result = []
    prose = ('This worksheet chooses the parameters for easy arithmetic. The row-vector '
             'weight matrix has four identical rows [¼,0,0,0], and the bias is [0,0,0,1]. '
             'Thus c=xW+b. In PyTorch this is nn.Linear(4,4), with weight shape (4,4) '
             'and bias shape (4,). The first output measures the fraction of filled pixels. '
             'These meanings are chosen for this exercise; real ViT feature meanings are learned. '
             'There is no hidden layer or activation in this patch projection. The Transformer '
             'block MLP, introduced earlier, has two linear layers with GELU between them.')

    def node(x, y, label, color='c-e', radius=29):
        return (f'<circle cx="{x}" cy="{y}" r="{radius}" fill="var(--card)" '
                f'stroke="var(--{color})" stroke-width="2.5"/>'
                +t(x, y+9, label, 28, color, 'middle'))

    def add(key, title, body, caption, question, point, mobile):
        # Replace the earlier unsplit drawing's metadata as well as its figure.
        b['FRAMES'][:] = [m for m in b['FRAMES'] if m['id'] != key]
        result.append(frame(key, title, body, caption, question+'\n'+point, prose, mobile))

    body = pixels(55, 110, [[1, 1], [1, 1]], 90, True)
    body += t(145, 335, 'P1 · 2 × 2 pixels', 27, 'ink', 'middle')
    body += arrow(280, 200, 405, 200)
    row = t(725, 92, 'Keep the order: top row, then bottom row', 26, 'ink-2', 'middle')
    for j, label in enumerate(['top-left', 'top-right', 'bottom-left', 'bottom-right']):
        x = 475+175*j
        row += rect(x-65, 145, 130, 100, 'c-e', 't-e')+t(x, 207, '1', 38, 'c-e', 'middle')
        row += t(x, 285, label, 24, 'ink-2', 'middle')
    row += t(737, 359, 'x = [1, 1, 1, 1]     shape: 1 × 4', 32, 'c-e', 'middle')
    body += g(row, 1)
    add('s02-projection-step-1', 'Flatten P1 into four inputs', body,
        'The same four grayscale pixels become four input numbers. Flattening rearranges them; it has no weights or biases.',
        'Which input comes from the bottom-left pixel?',
        'Trace top-left, top-right, bottom-left, bottom-right. Each pixel supplies one grayscale value in this worksheet.',
        '<p>P1: a 2 × 2 patch, all four pixels equal to 1.</p>'
        '<p>Read top-left, top-right, bottom-left, bottom-right: x = [1,1,1,1], shape 1 × 4.</p>')

    body = t(270, 38, 'Four pixel inputs', 27, 'c-e', 'middle')
    body += pixels(30, 150, [[1, 1], [1, 1]], 48, True)+t(78, 280, 'P1', 26, 'ink', 'middle')
    body += arrow(146, 200, 204, 200)
    for j, y in enumerate([100, 180, 260, 340]):
        body += line(299, y, 620, 220, 'c-e', 2.5)
        label_y = y+(220-y)*.30
        body += rect(374, label_y-19, 50, 35, 'transparent', 'card')
        body += t(399, label_y+8, '¼', 28, 'c-e', 'middle')
        body += node(270, y, '1')+t(224, y+8, ['x₁', 'x₂', 'x₃', 'x₄'][j], 24, 'ink-2', 'end')
    body += t(413, 53, 'weights', 24, 'c-e', 'middle')
    body += rect(550, 70, 140, 48, 'c-v', 'transparent')+t(620, 102, 'bias = 0', 27, 'c-v', 'middle')
    body += arrow(620, 122, 620, 178, 'c-v')+node(620, 220, 'Σ', 'c-e', 38)
    body += t(620, 284, 'weighted sum + bias', 25, 'ink-2', 'middle')
    output = arrow(674, 220, 823, 220)
    output += rect(840, 170, 250, 100, 'c-e', 't-e')+t(965, 207, 'first coordinate', 24, 'c-e', 'middle')
    output += t(965, 248, 'ink = 1', 35, 'c-e', 'middle')
    output += t(740, 356, '¼·1 + ¼·1 + ¼·1 + ¼·1 + 0 = 1', 30, 'c-e', 'middle')
    body += g(output, 1)
    body += t(580, 423, 'One output neuron: multiply each input, add the products, then add its bias.', 26, 'ink', 'middle')
    add('s02-projection-step-2', 'Compute the first coordinate with one neuron', body,
        'Four inputs, four weights of ¼, and one bias of 0 produce ink = 1. This projection applies no activation after the weighted sum.',
        'What does this neuron output when all four pixels are 1?',
        'Follow each weighted edge into the sum, then add the separate bias. Reveal the output. This is one of the layer’s four output neurons.',
        '<p>Four input neurons: x₁=x₂=x₃=x₄=1. Each connects to the first output with weight ¼.</p>'
        '<p>Bias = 0. Output = ¼·1 + ¼·1 + ¼·1 + ¼·1 + 0 = 1. No activation follows.</p>')

    body = t(580, 30, 'nn.Linear(4, 4) · four inputs → four outputs', 31, 'c-e', 'middle')
    body += t(230, 78, 'Pixel values', 25, 'ink-2', 'middle')+t(660, 78, 'Content coordinates', 25, 'c-e', 'middle')
    body += t(947, 78, 'One bias per output', 25, 'c-v', 'middle')
    ys = [126, 196, 266, 336]
    for y in ys:
        for out_y in ys[1:]:
            body += line(259, y, 631, out_y, 'line', 1.7, '4 5')
    for y in ys:
        body += line(259, y, 631, ys[0], 'c-e', 2.5)
    body += rect(324, 102, 220, 36, 'transparent', 'card')+t(434, 128, 'blue weights = ¼', 24, 'c-e', 'middle')
    body += rect(320, 264, 235, 36, 'transparent', 'card')+t(437, 290, 'dashed weights = 0', 24, 'ink-2', 'middle')
    for j, y in enumerate(ys):
        body += node(230, y, '1')+t(180, y+8, ['x₁', 'x₂', 'x₃', 'x₄'][j], 24, 'ink-2', 'end')
        body += node(660, y, ['1', '0', '0', '1'][j])
        body += t(755, y-12, ['ink', 'output 2', 'output 3', 'output 4'][j], 22, 'ink-2', 'middle')
        body += arrow(850, y, 699, y, 'c-v')
        body += t(887, y+8, ['+ 0', '+ 0', '+ 0', '+ 1'][j], 28, 'c-v')
    body += g(t(580, 410, 'c = [1, 0, 0, 1]     shape: 1 × 4     ·     no activation', 31, 'c-e', 'middle'), 1)
    add('s02-projection', 'Open the whole patch projection layer', body,
        'Each output has four weights and one bias: 16 weights + 4 biases. Zero weights produce zero sums for outputs 2–4; the fourth bias adds 1. Use this same layer on every patch.',
        'Why is the fourth output 1 even though all its weights are zero?',
        'Read the bias beside each output. The fourth coordinate is 0+1. The middle coordinates stay zero until position is added on a later slide.',
        '<p>One nn.Linear(4,4) layer, with 16 weights and 4 biases, is shared by all patches.</p>'
        '<table class="vp-table"><thead><tr><th>Output</th><th>Four input weights</th><th>Bias</th><th>Result</th></tr></thead><tbody>'
        '<tr><td>1 (ink)</td><td>[¼,¼,¼,¼]</td><td>0</td><td>1</td></tr>'
        '<tr><td>2</td><td>[0,0,0,0]</td><td>0</td><td>0</td></tr>'
        '<tr><td>3</td><td>[0,0,0,0]</td><td>0</td><td>0</td></tr>'
        '<tr><td>4</td><td>[0,0,0,0]</td><td>1</td><td>1</td></tr></tbody></table>'
        '<p>Content row c = [1,0,0,1], shape 1 × 4. No activation follows the linear layer.</p>')
    return result


def build_worksheet_images(b):
    """Replace two redundant single-patch slides with both complete images."""
    t, g, rect, arrow, line, pixels, frame, mobile_rows = (b[k] for k in
        ['t', 'g', 'rect', 'arrow', 'line', 'pixels', 'frame', 'mobile_rows'])
    data = b['DATA']
    colors = ['c-e', 'c-q', 'c-v', 'c-k']
    result = {}

    def row(values):
        return '['+', '.join(str(int(v)) for v in values)+']'

    for key, name, title in [
        ('patch-matrix', 'horizontal', 'Project all four patches of the first image'),
        ('empty-patch', 'vertical', 'Project the second image with the same layer'),
    ]:
        case = data['cases'][name+'_positions']
        body = t(118, 38, 'All 16 pixels', 28, 'ink', 'middle')
        body += t(418, 38, 'Four pixel rows', 28, 'c-e', 'middle')
        body += t(710, 38, 'Shared Linear(4, 4)', 27, 'c-e', 'middle')
        body += t(1012, 38, 'Four content rows', 27, 'c-e', 'middle')
        body += t(418, 76, 'X: 4 patches × 4 values', 23, 'ink-2', 'middle')
        body += t(1012, 76, 'C: 4 patches × 4 features', 22, 'ink-2', 'middle')
        body += pixels(26, 120, data['images'][name], 46, True)
        for j, color in enumerate(colors):
            x, y = 26+(j % 2)*92, 120+(j//2)*92
            body += rect(x, y, 92, 92, color, 'transparent', 0)
            body += t(x+46, 110 if j<2 else 335, f'P{j+1}', 24, color, 'middle')
        body += arrow(227, 212, 273, 212)

        # All 16 pixel values remain visible, with one row for each 2x2 patch.
        output = ''
        for j, color in enumerate(colors):
            y = 133+59*j
            body += t(308, y+9, f'P{j+1}', 23, color, 'end')
            body += rect(321, y-24, 192, 48, color, 'transparent')
            body += t(417, y+9, row(case['patches'][j]), 27, color, 'middle')
            output += rect(916, y-24, 192, 48, color, 'transparent')
            output += t(1012, y+9, row(case['content'][j]), 27, color, 'middle')
            output += t(900, y+9, f'P{j+1}', 23, color, 'end')
        body += arrow(528, 212, 575, 212)

        # One weight matrix and one bias, reused for every patch in both images.
        body += rect(590, 93, 239, 257, 'c-e', 't-e')
        body += t(609, 121, 'W', 23, 'c-e')
        for r, weights in enumerate(data['W_patch']):
            for c, value in enumerate(weights):
                body += t(644+c*49, 145+r*39, '¼' if value==.25 else str(int(value)), 28, 'c-e', 'middle')
        body += line(611, 281, 808, 281, 'c-e')
        body += t(710, 316, 'b = [0, 0, 0, 1]', 25, 'c-v', 'middle')
        output += arrow(839, 212, 878, 212)
        output += t(580, 399, 'C = XW + b     ·     (4 × 4)(4 × 4) + bias → 4 × 4', 29, 'c-e', 'middle')
        body += g(output, 1)
        if name == 'horizontal':
            body += t(580, 439, 'One 2 × 2 patch → four inputs → four output features. Repeat for P1–P4.', 25, 'ink', 'middle')
            caption = ('The 16 grayscale pixels form four patches. Each patch supplies four inputs '
                       'to the same layer and gets four output features. Even an empty patch '
                       'produces [0, 0, 0, 1] because of the bias.')
            question = 'Are four output features describing one patch or the whole image?'
            point = ('One patch. Match each coloured 2×2 region to its input row and output row. '
                     'There are four output rows. Add the same bias to each row; no activation follows.')
        else:
            body += t(580, 439, 'P2 becomes empty; P3 becomes filled. The same W and b give the new rows.', 25, 'ink', 'middle')
            caption = ('Change the image from “Across the top” to “Down the left”. P2 and P3 '
                       'exchange their pixel rows and content rows. The layer uses exactly '
                       'the same weights and biases for both images.')
            question = 'Which output rows change when the filled patches move to the left?'
            point = ('P2 changes to [0,0,0,1]; P3 changes to [1,0,0,1]. P1 and P4 stay the same. '
                     'Position has not been added yet. Keep row order: top-left, top-right, bottom-left, bottom-right.')
        prose = ('This diagram shows every input pixel and every patch content feature. '
                 'The input X has shape 4×4: four patches, with four grayscale values per patch. '
                 'Each patch uses the same W and b shown in the center. C=XW+b has shape 4×4: '
                 'four patch embeddings, each four features wide. Bias is broadcast to every row. '
                 'This is nn.Linear(4,4) without an activation. The 16 weights and four biases '
                 'are hand-chosen for this worksheet, independently of the 16 input pixel values. '
                 'No positions, CLS or attention have entered this calculation. The two images '
                 'have the same set of content rows in a different order. Their locations are added later.')
        mobile = ('<p>All 16 pixels, arranged in four image rows: '
                  +'; '.join(row(r) for r in data['images'][name])+'.</p>'
                  '<p>Four 2×2 patches. Shared nn.Linear(4,4): each row of W is [¼,0,0,0]; '
                  'bias = [0,0,0,1]. No activation.</p>'
                  +mobile_rows(['Patch', 'Four input pixels', 'Four output features'],
                               [[f'P{j+1}', row(case['patches'][j]), row(case['content'][j])] for j in range(4)])
                  +'<p>X: 4×4 → C=XW+b: 4×4. One output row per patch.</p>')
        notes = question+'\n'+point
        for meta in b['FRAMES']:
            if meta['id']==key:
                meta.update(title=title, caption=caption, notes=notes)
        result[key] = frame(key, title, body, caption, notes, prose, mobile)
    return result
