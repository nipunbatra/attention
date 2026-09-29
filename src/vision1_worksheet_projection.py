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

    # The matrix and network encode the same connections. Separate xW from
    # bias addition so a zero-weight output does not appear to invent a 1.
    weights, biases = b['DATA']['W_patch'], b['DATA']['b_patch']
    inputs = b['R']['patches'][0]
    sums = [sum(inputs[i]*weights[i][j] for i in range(4)) for j in range(4)]
    outputs = [sums[j]+biases[j] for j in range(4)]
    ys = [126, 192, 258, 324]
    body = t(176, 28, 'Weight matrix W', 28, 'c-e', 'middle')
    body += t(591, 28, '1 · Multiply and sum', 28, 'c-e', 'middle')
    body += t(948, 28, '2 · Add one bias', 28, 'c-v', 'middle')
    body += t(178, 67, 'one column per output', 23, 'ink-2', 'middle')
    body += line(338, 54, 338, 356)
    body += t(436, 73, 'P1 inputs', 24, 'ink-2', 'middle')
    body += t(728, 73, 'sum z', 24, 'c-e', 'middle')
    body += t(865, 73, 'bias b', 24, 'c-v', 'middle')
    body += t(1060, 73, 'output c', 24, 'c-e', 'middle')
    body += rect(71, 93, 47, 258, 'transparent', 't-e')
    body += rect(236, 93, 47, 258, 'transparent', 't-q')
    for j in range(4):
        body += t(95+55*j, 94, str(j+1), 19, 'ink-2', 'middle')
    for i, y in enumerate(ys):
        body += t(50, y+9, ['x₁', 'x₂', 'x₃', 'x₄'][i], 22, 'ink-2', 'end')
        for j in range(4):
            color = 'c-e' if j==0 else 'c-q' if j==3 else 'ink-2'
            value = '¼' if weights[i][j]==.25 else str(int(weights[i][j]))
            body += t(95+55*j, y+9, value, 29, color, 'middle')
        body += node(436, y, str(int(inputs[i])))
        body += t(392, y+8, ['x₁', 'x₂', 'x₃', 'x₄'][i], 22, 'ink-2', 'end')
    wires = ''
    for j, out_y in enumerate(ys):
        for i, in_y in enumerate(ys):
            if weights[i][j]==0:
                color = 'c-q' if j==3 else 'ink-3'
                wires += '<g opacity=".25">'+line(465, in_y, 699, out_y, color, 1.5, '4 5')+'</g>'
    for in_y in ys:
        wires += line(465, in_y, 699, ys[0], 'c-e', 2.5)
    wires += rect(488, 99, 163, 34, 'transparent', 'card')+t(570, 124, 'solid: weight ¼', 21, 'c-e', 'middle')
    wires += rect(485, 293, 172, 34, 'transparent', 'card')+t(571, 318, 'dashed: weight 0', 21, 'ink-2', 'middle')
    additions, final = '', ''
    for j, y in enumerate(ys):
        color = 'c-q' if j==3 else 'c-e'
        wires += node(728, y, str(int(sums[j])), color)
        additions += t(799, y+10, '+', 31, 'ink-2', 'middle')
        additions += rect(837, y-26, 56, 52, 'c-v', 'transparent')
        additions += t(865, y+10, str(int(biases[j])), 30, 'c-v', 'middle')
        final += t(957, y+10, '=', 31, 'ink-2', 'middle')
        final += node(1060, y, str(int(outputs[j])), color)
    body += g(wires, 1)+g(additions, 2)+g(final, 3)
    body += t(580, 387, 'Column j of W gives the four connection weights for output j.', 27, 'ink', 'middle')
    body += g(t(580, 431, 'xW = [1, 0, 0, 0]   +   b = [0, 0, 0, 1]   →   c = [1, 0, 0, 1]', 27, 'c-e', 'middle'), 3)
    add('s02-projection', 'The same projection: matrix, neurons, then biases', body,
        'First multiply inputs by connection weights and sum at each neuron: [1, 0, 0, 0]. '
        'Then add one bias per output: [0, 0, 0, 1]. The fourth output is 0 + 1 = 1.',
        'Why does output 4 become 1 when its four connection weights are zero?',
        'Match each W column to one neuron’s incoming weights. Reveal the weighted sums, '
        'then the four biases, then the outputs. Output 4 adds its bias of 1 to a zero sum. '
        'This is one nn.Linear(4,4) layer with no activation. Reuse it for every patch.',
        '<p>P1 inputs: x = [1,1,1,1]. The four columns of W correspond to the four output neurons. '
        'Each column contains that neuron’s four incoming connection weights.</p>'
        '<table class="vp-table"><thead><tr><th>Output</th><th>Connection weights</th>'
        '<th>Multiply and sum</th><th>Add bias</th><th>Result</th></tr></thead><tbody>'
        '<tr><td>1 (ink)</td><td>[¼,¼,¼,¼]</td><td>¼·1 + ¼·1 + ¼·1 + ¼·1 = 1</td><td>+ 0</td><td>1</td></tr>'
        '<tr><td>2</td><td>[0,0,0,0]</td><td>0·1 + 0·1 + 0·1 + 0·1 = 0</td><td>+ 0</td><td>0</td></tr>'
        '<tr><td>3</td><td>[0,0,0,0]</td><td>0·1 + 0·1 + 0·1 + 0·1 = 0</td><td>+ 0</td><td>0</td></tr>'
        '<tr><td>4</td><td>[0,0,0,0]</td><td>0·1 + 0·1 + 0·1 + 0·1 = 0</td><td>+ 1</td><td>1</td></tr>'
        '</tbody></table><p>xW = [1,0,0,0]; xW+b = [1,0,0,1]. All rows have shape 1×4. '
        'These are the linear layer’s biases. No activation follows this layer.</p>')
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
