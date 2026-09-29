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
