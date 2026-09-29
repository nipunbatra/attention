"""Finish the dog forward pass by opening CLS selection, the head and softmax."""
import json


def build_classifier_journey(b):
    t, g, rect, arrow, line, image, frame, mobile_rows = (b[k] for k in
        ['t', 'g', 'rect', 'arrow', 'line', 'image', 'frame', 'mobile_rows'])
    data = json.loads((b['ASSETS']/'classifier-readout-trace.json').read_text())
    cls, classes = data['cls_final'], data['selected_classes']
    result = {}
    evidence = (' <a href="figures/vision1/classifier-readout-trace.json">Saved classifier calculation</a> · '
                '<a href="notebooks/vision/trace_classifier_readout.py">Reproduce the readout from saved CLS</a>.')

    def num(v, places=3):
        return f'{v:.{places}f}'.replace('-', '−')

    def box(x, y, w, labels, color='c-e', h=80, size=25):
        out = rect(x, y, w, h, color, 'transparent', 6)
        for j, label in enumerate(labels):
            out += t(x+w/2, y+h/2+8+(j-(len(labels)-1)/2)*31, label, size, color, 'middle')
        return out

    def neuron(x, y, label, color='c-e', radius=35):
        return (f'<circle cx="{x}" cy="{y}" r="{radius}" fill="var(--card)" '
                f'stroke="var(--{color})" stroke-width="2"/>'+t(x, y+7, label, 21, color, 'middle'))

    def matrix(x, y, label, color='c-e'):
        out = t(x+99, y-25, label, 24, color, 'middle')
        for row, name in enumerate(['CLS', 'P1', '…', 'P196']):
            tint = 'c-q' if row==0 else color
            out += t(x-12, y+row*45+29, name, 20, tint, 'end')
            for col in range(4):
                out += rect(x+col*49.5, y+row*45, 49.5, 45, tint,
                            't-q' if row==0 else 'transparent', 0)
                out += t(x+(col+.5)*49.5, y+row*45+28, '⋯', 22, tint, 'middle')
        out += t(x+99, y+217, '197 × 192', 26, color, 'middle')
        return out

    def add(key, title, body, caption, question, point, prose, mobile):
        notes = question+'\n'+point
        for meta in b['FRAMES']:
            if meta['id']==key:
                meta.update(title=title, caption=caption, notes=notes)
        result[key] = frame(key, title, body, caption, notes, prose+evidence, mobile)

    body = image(35, 5, 125, 83)+t(187, 47, 'The same dog has now passed through all 12 blocks.', 29)
    body += matrix(93, 152, 'After block 12')
    body += arrow(304, 241, 337, 241)
    body += box(351, 203, 149, ['Final', 'LayerNorm'], 'c-e', 76, 25)
    body += arrow(509, 241, 552, 241)
    body += matrix(607, 152, 'Normalized rows')
    selected = arrow(817, 175, 866, 175, 'c-q')
    selected += t(1002, 126, 'Select CLS', 29, 'c-q', 'middle')
    selected += box(879, 140, 249, ['['+num(cls[0])+', '+num(cls[1])+', …]', '1 × 192'], 'c-q', 78, 24)
    selected += t(1002, 281, 'One image summary', 24, 'c-q', 'middle')
    selected += t(1002, 321, '192 features', 27, 'c-q', 'middle')
    body += g(selected, 1)
    body += t(35, 430, 'Read the highlighted row; the patch rows have already supplied context.', 28)
    add('real-cls-readout', 'Select CLS from the final feature matrix', body,
        'After block 12, all 197 rows are still present. Final LayerNorm keeps the same shape. Select the highlighted CLS row: one image summary with 192 features. Its values now reflect information gathered from this dog’s patches.',
        'Which part of the 197 × 192 matrix enters the classifier?',
        'Extract the entire first row, all 192 features. This CLS is the result after the blocks and final normalization, not the initial learned CLS parameter. We do not average the rows in this model.',
        'The matrix still contains CLS and P1 through P196. Final LayerNorm operates across the '
        '192 features within each row. The classifier takes h=final[:,0,:], a 1×192 vector for '
        'this one image. The matrix drawings abbreviate feature entries, while the selected vector '
        'shows the measured first two coordinates. The patch rows have contributed through attention; '
        'this model’s final head reads only CLS. Batch size one is omitted from the matrix diagrams.',
        '<img src="figures/vision1/model-input.png" alt="The same dog after the full forward pass" width="150">'
        '<p>Block 12 output (197 × 192) → final LayerNorm (197 × 192) → select CLS (1 × 192).</p>'
        '<p>Final CLS begins ['+num(cls[0])+', '+num(cls[1])+', …]. These are image-summary features.</p>')

    body = t(580, 34, 'This checkpoint predicts 1,000 ImageNet classes.', 29, 'ink', 'middle')
    body += t(210, 83, 'CLS features · 1 × 192', 26, 'c-q', 'middle')
    body += t(870, 83, 'Class scores · 1 × 1,000', 26, 'c-e', 'middle')
    input_ys, output_ys = [144, 234, 344], [144, 234, 324]
    body += '<g opacity=".3">'
    for y1 in input_ys:
        for y2 in output_ys:
            body += line(249, y1, 681, y2, 'c-e', 1.6)
    body += '</g>'
    for y, value, label in zip(input_ys, [cls[0], cls[1], cls[-1]], ['h₁', 'h₂', 'h₁₉₂']):
        body += neuron(210, y, num(value), 'c-q', 37)+t(141, y+8, label, 24, 'c-q', 'end')
    body += t(210, 300, '⋮', 28, 'c-q', 'middle')
    for y, item in zip(output_ys, classes):
        body += neuron(720, y, num(item['logit']), 'c-e', 38)
        body += t(788, y+9, item['label'], 28)
    body += t(720, 392, '⋮', 28, 'c-e', 'middle')+t(788, 393, '997 more class scores', 24, 'ink-2')
    body += t(453, 131, 'nn.Linear(192,1000)', 26, 'c-e', 'middle')
    body += t(453, 374, '192 weights + 1 bias', 24, 'c-e', 'middle')
    body += t(453, 408, 'for each class', 24, 'c-e', 'middle')
    body += t(580, 439, 'One affine layer produces scores. Softmax follows.', 27, 'ink', 'middle')
    add('real-classifier-network', 'Open the classifier: 192 features become 1,000 scores', body,
        'Every class has one output neuron connected to all 192 CLS features. Each neuron uses its own learned weights and bias to produce a score. Only three of the 1,000 class neurons are drawn. Their scores are measured.',
        'Does one feature correspond to one class, or does every class read the whole summary?',
        'Each class reads all 192 features with its own weighted sum. This class head is a single affine layer. It has no hidden layer or GELU; softmax converts its scores to probabilities next.',
        'The operation is z=hW_class+b_class: (1×192)(192×1000)+(1000) gives 1×1000. '
        'PyTorch nn.Linear(192,1000) stores its weight tensor transposed, as 1000×192. '
        'The head has 192,000 weights plus 1,000 biases. The drawn connections are representative; '
        'the omitted 189 inputs and 997 outputs are also fully connected. Input values and class '
        'scores come from the same saved dog representation. The output width is the label count '
        'used when training this checkpoint. A cat-versus-dog classifier could use Linear(192,2), '
        'with weights trained for those two labels. These scores are logits, not probabilities.',
        '<p>Final CLS: 1 × 192 → nn.Linear(192,1000) → class scores: 1 × 1000.</p>'
        '<p>Weight matrix: 192 × 1000 in row-vector math; 1000 × 192 in PyTorch. Bias: 1000.</p>'
        +mobile_rows(['Class', 'Measured score'], [[r['label'], num(r['logit'])] for r in classes])
        +'<p>Every output reads all 192 features. No hidden layer or GELU in this head.</p>')

    first = classes[0]
    body = t(35, 35, 'Zoom into one output neuron: Newfoundland', 31)
    body += neuron(165, 134, num(cls[0]), 'c-q', 38)+t(165, 191, 'h₁', 25, 'c-q', 'middle')
    body += neuron(165, 234, num(cls[1]), 'c-q', 38)+t(165, 291, 'h₂', 25, 'c-q', 'middle')
    body += arrow(211, 134, 682, 219, 'c-e')+t(418, 132, '× '+num(first['weights'][0],5), 27, 'c-e', 'middle')
    body += arrow(211, 234, 682, 234, 'c-e')+t(418, 218, '× '+num(first['weights'][1],5), 27, 'c-e', 'middle')
    body += box(35, 318, 343, ['Other 190 weighted features', 'sum = '+num(first['remaining_190_products_sum'],4)], 'c-e', 66, 23)
    body += arrow(393, 351, 686, 258, 'c-e')
    body += box(627, 67, 214, ['Bias: '+num(first['bias'],4)], 'c-e', 59, 25)
    body += arrow(734, 134, 734, 184, 'c-e')+neuron(734, 234, 'Σ', 'c-e', 43)
    body += arrow(787, 234, 893, 234, 'c-e')
    body += box(908, 191, 217, [num(first['logit'],4), 'one class score'], 'c-e', 85, 26)
    equation = (num(first['first_two_products'][0],4)+' '+num(first['first_two_products'][1],4)
                +' + '+num(first['remaining_190_products_sum'],4)+' + '+num(first['bias'],4)
                +' ≈ '+num(first['logit'],4))
    body += t(580, 437, equation, 29, 'c-e', 'middle')
    add('real-classifier-score', 'One class score is a weighted sum plus a bias', body,
        'Multiply each CLS feature by this class’s learned weight. Sum all 192 products and add its bias. For this dog, the Newfoundland neuron produces 15.4763. Every other class performs the same calculation with its own weights and bias.',
        'What makes the Newfoundland score different from the Tibetan mastiff score?',
        'They read the same image summary but use different learned weights and biases. The 190 omitted products still contribute. Keep the score separate from its later probability.',
        'For class c, z_c=Σᵢ hᵢWᵢc+b_c, summing all 192 coordinates. This zoom displays '
        'the first two feature values and corresponding learned weights, groups the remaining 190 '
        'products, and includes the bias. The bottom arithmetic uses products computed before '
        'rounding; the displayed inputs and weights are rounded separately. All 192 stored weights '
        'for each displayed class are linked in the trace. The readout check reproduces each dot '
        'product and matches the dog’s earlier full-model logits. The classifier weights stay '
        'fixed during inference; different photographs supply different CLS feature values.',
        '<p>Newfoundland: sum all 192 feature × weight products, then add its bias.</p>'
        '<p>'+equation+'</p><p>The same CLS features feed every class, with separate learned weights and biases.</p>')

    body = t(580, 34, 'The alternatives are class labels: normalize over all 1,000.', 29, 'ink', 'middle')
    body += t(206, 88, 'Class scores (logits)', 27, 'c-e', 'middle')
    body += t(968, 88, 'Class probabilities', 27, 'c-v', 'middle')
    body += (line(309, 109, 301, 109, 'c-e')+line(301, 109, 301, 366, 'c-e')
             +line(301, 366, 309, 366, 'c-e')+line(409, 109, 417, 109, 'c-e')
             +line(417, 109, 417, 366, 'c-e')+line(417, 366, 409, 366, 'c-e'))
    for j, item in enumerate(classes):
        y = 127+j*65
        body += t(35, y+25, item['label'], 25)+t(403, y+25, num(item['logit']), 26, 'c-e', 'end')
        body += t(1090, y+25, f'{100*item["probability"]:.2f}%', 27, 'c-v', 'end')
        body += rect(822, y+5, max(2,170*item['probability']), 22, 'c-v', 'transparent', 0)
    other_probability = 1-sum(item['probability'] for item in classes)
    body += t(35, 352, '997 more classes', 24, 'ink-2')+t(355, 352, '⋮', 27, 'c-e', 'middle')
    body += t(968, 352, f'{100*other_probability:.2f}% across 997 others', 23, 'ink-2', 'middle')
    body += arrow(424, 220, 460, 220)+box(474, 178, 277, ['softmax', 'all 1,000 scores'], 'c-v', 84, 27)
    body += arrow(763, 220, 802, 220, 'c-v')
    body += t(580, 405, 'p(Newfoundland) = exp(15.476) / sum of exp(all 1,000 scores)', 26, 'c-v', 'middle')
    body += t(580, 439, '≈ 0.9573     ·     All 1,000 probabilities sum to 1.', 27, 'c-v', 'middle')
    add('real-classifier-softmax', 'Turn all 1,000 scores into class probabilities', body,
        'Exponentiate each score and divide by the sum over all 1,000 classes. Newfoundland receives 95.73% probability. The displayed classes are only three alternatives; the remaining 997 also enter the denominator. Choose the class with the largest probability.',
        'Would softmax over only these three displayed scores give the same answer?',
        'No. All 1,000 scores contribute to the denominator. The remaining probability mass is shown explicitly. Attention softmax normalized over source rows; this softmax normalizes over image classes.',
        'For every class c, p_c=exp(z_c)/Σⱼexp(z_j). The actual computation subtracts '
        'the largest logit before exponentiation for numerical stability; this leaves the '
        'probabilities unchanged. For the winning class the shifted numerator is 1 and the '
        'denominator is '+num(data['softmax']['shifted_denominator'],6)+', giving '
        +num(first['probability'],6)+'. The trace includes all 1,000 logits, so the complete '
        'normalization can be checked. The displayed 997-class probability is an aggregate of '
        'the omitted alternatives, not another class. Softmax has no learned parameters. '
        'The prediction uses argmax; the highest score and highest probability identify the same class.',
        mobile_rows(['Class', 'Score', 'Probability'],
                    [[r['label'], num(r['logit']), f'{100*r["probability"]:.2f}%'] for r in classes])
        +f'<p>Other 997 classes: {100*other_probability:.2f}% combined. Softmax uses all 1,000 scores.</p>'
        '<p>p(c) = exp(z_c) / sum_j exp(z_j). Select the largest probability.</p>')

    body = image(35, 58, 280, 280)+t(175, 376, 'The same photograph', 25, 'ink-2', 'middle')
    body += box(425, 80, 299, ['Final CLS', '192 image features'], 'c-q', 92, 28)
    body += arrow(333, 218, 398, 126, 'c-q')
    body += arrow(736, 126, 795, 126)+box(810, 80, 316, ['Linear + softmax', '1,000 probabilities'], 'c-e', 92, 27)
    body += arrow(968, 184, 968, 241, 'c-e')
    body += t(430, 273, 'Choose the largest probability', 28)
    body += box(425, 292, 701, [first['label']+' · '+f'{100*first["probability"]:.2f}%'], 'c-v', 86, 38)
    body += t(580, 435, 'Pixels → patches → contextual features → one image prediction.', 29, 'ink', 'middle')
    add('real-cls-prediction', 'The same dog now has its final prediction', body,
        'The classifier reads the final CLS, scores 1,000 labels, and selects Newfoundland as the most probable class. Its probability for this photograph is 95.73%. We have completed one forward pass from pixels to an image prediction.',
        'Did each patch predict its own label, or did the model make one image prediction?',
        'The final CLS summarizes the contextual features for this one image. One class head reads that summary. The 95.73% is a model probability for this photograph, not an accuracy measurement.',
        'The photograph is the original Newfoundland example used throughout the lecture. '
        'Its patches supplied features that were updated through the 12-block stack alongside '
        'CLS. The final normalized CLS feeds the learned classifier, and softmax plus argmax '
        'gives the image label shown here. The saved classifier-only calculation matches the '
        'earlier full forward pass; no training was performed for this illustration. The next '
        'section uses smaller numbers so students can calculate the attention mechanism themselves.',
        '<img src="figures/vision1/model-input.png" alt="The dog classified as Newfoundland" width="200">'
        '<p>Final CLS → Linear(192,1000) → softmax → select the largest probability.</p>'
        '<p><strong>Newfoundland: 95.73%</strong> for this photograph.</p>')
    return result
