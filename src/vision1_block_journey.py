"""Open one Transformer block, trace its MLP, and hand its rows to block 2."""
import json


def build_block_journey(b):
    t, g, rect, arrow, line, image, frame, mobile_rows = (b[k] for k in
        ['t', 'g', 'rect', 'arrow', 'line', 'image', 'frame', 'mobile_rows'])
    data = json.loads((b['ASSETS']/'real-classifier-path.json').read_text())
    result = {}
    evidence = (' <a href="figures/vision1/real-classifier-path.json">Saved dog forward pass</a> · '
                '<a href="notebooks/vision/trace_real_classifier.py">Check the block computation</a>.')

    def number(v):
        return f'{v:.3f}'.replace('-', '−')

    def vec(name):
        return '['+', '.join(number(v) for v in data['previews'][name][:2])+', …]'

    def box(x, y, w, labels, color='c-e', h=80, size=25):
        out = rect(x, y, w, h, color, 'transparent', 6)
        for j, label in enumerate(labels):
            out += t(x+w/2, y+h/2+8+(j-(len(labels)-1)/2)*31, label, size, color, 'middle')
        return out

    def plus(x, y, color='c-e'):
        return (f'<circle cx="{x}" cy="{y}" r="23" fill="var(--card)" '
                f'stroke="var(--{color})" stroke-width="2.5"/>'+t(x, y+10, '+', 33, color, 'middle'))

    def matrix(x, y, w, label, color='c-e'):
        out = t(x+w/2, y-22, label, 27, color, 'middle')
        for r, name in enumerate(['CLS', 'P1', '…', 'P196']):
            out += rect(x, y+r*35, w, 35, color, 't-e' if r==0 else 'transparent', 0)
            out += t(x+8, y+r*35+24, name, 18, color)
            out += t(x+w-10, y+r*35+24, '· · ·', 20, color, 'end')
        out += t(x+w/2, y+176, '197 × 192', 22, color, 'middle')
        return out

    def add(key, title, body, caption, question, point, prose, mobile):
        notes = question+'\n'+point
        for meta in b['FRAMES']:
            if meta['id'] == key:
                meta.update(title=title, caption=caption, notes=notes)
        result[key] = frame(key, title, body, caption, notes, prose+evidence, mobile)

    body = box(35, 22, 230, ['Input E', '197 × 192'])+arrow(278, 62, 448, 62)
    body += box(465, 22, 230, ['Transformer', 'block 1'], 'c-q')+arrow(708, 62, 878, 62)
    body += box(895, 22, 230, ['Output E¹', '197 × 192'])
    opened = line(475, 107, 83, 175, 'c-q', 2, '6 5')+line(685, 107, 1077, 175, 'c-q', 2, '6 5')
    opened += rect(35, 175, 1090, 177, 'c-q', 'transparent', 8)
    opened += t(65, 209, 'INSIDE BLOCK 1', 22, 'c-q', weight=600)
    opened += t(75, 286, 'E', 32, 'c-e')+arrow(108, 276, 146, 276)
    opened += box(158, 233, 316, ['Normalize → attention → + E', 'share context across rows'], 'ink-2', 85, 22)
    opened += arrow(485, 276, 523, 276)+t(546, 286, 'U', 32, 'c-e', 'middle')
    opened += arrow(566, 276, 606, 276)
    opened += box(620, 233, 322, ['Normalize → MLP → add U', 'transform each row’s features'], 'c-v', 85, 23)
    opened += arrow(956, 276, 1002, 276)+t(1058, 286, 'E¹', 32, 'c-e', 'middle')
    opened += t(781, 385, 'OPEN THIS PART NEXT', 23, 'c-v', 'middle', 600)
    body += g(opened, 1)
    body += t(35, 431, 'All three matrices: 197 × 192. One CLS row: 1 × 192.', 30)
    add('real-cls-mlp', 'Open block 1: attention, then the MLP', body,
        'One block makes two updates. Attention lets rows exchange information; the MLP transforms each resulting row. Each branch adds its update to its input. The block receives and returns 197 rows, each 192 features wide.',
        'Have we finished the entire block after the attention residual?',
        'We have reached U, the midpoint. Open the MLP branch next. E¹ names the output of block 1; it is not a power. Both branches update CLS and every patch row.',
        'For this pre-LayerNorm model, U=E+Attention(LayerNorm(E)), then '
        'E¹=U+MLP(LayerNorm(U)). Attention here includes all heads, concatenation and the output '
        'projection. Both additions are coordinate by coordinate. E, U and E¹ all have shape 197×192, '
        'while a single selected row has shape 1×192. This is still the same dog forward pass. '
        'The next drawings open the MLP and then close the entire block before introducing depth.',
        '<p>Input E (197 × 192) → <strong>block 1</strong> → output E¹ (197 × 192).</p>'
        '<ol><li>Normalize → attention → add E: obtain U.</li>'
        '<li>Normalize U → MLP → add U: obtain E¹.</li></ol>'
        '<p>Both branches update all rows. Each CLS or patch row stays 192 features wide.</p>')

    # A few drawn neurons represent the complete dense layers. Numbers in the
    # first two output nodes are saved activations, not invented weights.
    body = t(35, 31, 'Follow the dog’s CLS row after LayerNorm.', 28)
    cols = [(130, 'Normalized row', '1 × 192', 'c-e'),
            (430, 'Hidden pre-activation', '1 × 768', 'c-q'),
            (700, 'After GELU', '1 × 768', 'c-v'),
            (1000, 'MLP update', '1 × 192', 'c-d')]
    ys = [156, 236, 346]
    for left, right, color in [(130, 430, 'c-q'), (700, 1000, 'c-d')]:
        wires = '<g opacity=".25">'
        for a in ys:
            for c in ys:
                wires += line(left+37, a, right-37, c, color, 1.4)
        body += wires+'</g>'
    for y in ys:
        body += arrow(470, y, 660, y, 'c-v')
    labels = [['n₁', 'n₂', 'n₁₉₂'],
              [number(v) for v in data['previews']['cls_mlp_hidden'][:2]]+['z₇₆₈'],
              [number(v) for v in data['previews']['cls_mlp_activated'][:2]]+['a₇₆₈'],
              [number(v) for v in data['previews']['cls_mlp_update'][:2]]+['m₁₉₂']]
    for j, (x, name, shape, color) in enumerate(cols):
        body += t(x, 76, name, 22, color, 'middle')+t(x, 106, shape, 25, color, 'middle')
        for y, label in zip(ys, labels[j]):
            body += f'<circle cx="{x}" cy="{y}" r="35" fill="var(--card)" stroke="var(--{color})" stroke-width="2"/>'
            body += t(x, y+7, label, 19, color, 'middle')
        body += t(x, 303, '⋮', 26, color, 'middle')
    body += t(281, 393, 'Linear(192,768)', 25, 'c-q', 'middle')
    body += t(565, 393, 'GELU', 25, 'c-v', 'middle')
    body += t(850, 393, 'Linear(768,192)', 25, 'c-d', 'middle')
    body += t(580, 438, '192 → 768 → 768 → 192 features · biases in both linear layers', 27, 'ink', 'middle')
    add('real-mlp-network', 'Open the MLP: 192 inputs, 768 hidden units, 192 outputs', body,
        'Each linear layer connects every input feature to every output unit. GELU transforms each hidden activation. Only a few neurons are drawn; dots omit the rest. The displayed numbers are measured values for this dog’s CLS row.',
        'Does 768 mean patches, pixels, or hidden features here?',
        'It is the number of hidden features for one row. GELU is the nonlinearity between the two learned linear layers. The MLP returns a 192-feature update; the residual addition follows.',
        'Let n=LayerNorm(u₀). The first affine layer computes z=nW₁+b₁, with '
        'W₁ shaped 192×768 and b₁ of width 768. Apply GELU to every coordinate: a=GELU(z), '
        'still 1×768. The second layer computes m=aW₂+b₂, with W₂ shaped 768×192 and b₂ '
        'of width 192. Its output m has shape 1×192, with no activation after this final linear layer. '
        'The first two measured values are drawn in the corresponding nodes; the final symbolic node '
        'and ellipsis stand for the remaining units. Lines show learned connections, not their weight '
        'values. All 197 rows use this same MLP independently. Hidden width 768 is a model choice; '
        'it is unrelated to the 768 RGB values in a 16×16 patch.',
        mobile_rows(['Operation', 'Shape for CLS', 'First values'],
                    [['LayerNorm(u₀)', '1 × 192', 'n₁, n₂, …'],
                     ['Linear(192,768)', '1 × 768', vec('cls_mlp_hidden')],
                     ['GELU', '1 × 768', vec('cls_mlp_activated')],
                     ['Linear(768,192)', '1 × 192', vec('cls_mlp_update')]])
        +'<p>Both linear layers include a learned bias. The same parameters process each row.</p>')

    body = image(35, 14, 130, 87)+t(188, 49, 'Same dog · finish the second residual addition', 29)
    body += t(188, 93, 'The MLP produces an update for the current CLS row.', 27, 'ink-2')
    body += t(192, 187, 'After attention: u₀', 26, 'c-e', 'middle')
    body += box(35, 213, 313, [vec('cls_after_attention'), '1 × 192'], 'c-e', 87, 27)
    body += box(454, 135, 330, [vec('cls_mlp_update'), 'MLP update · 1 × 192'], 'c-d', 77, 25)
    body += arrow(619, 219, 619, 240, 'c-d')
    addition = arrow(361, 264, 586, 264, 'c-e')+t(471, 298, 'keep u₀', 24, 'c-e', 'middle')
    addition += plus(619, 264)+arrow(649, 264, 821, 264, 'c-e')
    addition += t(984, 187, 'Block 1 output: e₀¹', 26, 'c-e', 'middle')
    addition += box(836, 213, 290, [vec('cls_after_block1'), '1 × 192'], 'c-e', 87, 26)
    addition += t(580, 367, 'First coordinate: 0.463 − 0.100 ≈ 0.363', 31, 'c-e', 'middle')
    body += g(addition, 1)
    body += t(35, 432, 'Do this for CLS and every patch: E¹ = U + MLP(LayerNorm(U)).', 28)
    add('real-mlp-residual', 'Add the MLP update to finish block 1', body,
        'Keep the row produced by attention and add the MLP’s update. The result is this row’s output from block 1. Apply the same operation to all 197 rows. Their feature values change; their identities and width stay the same.',
        'Do we add the MLP update to E or to U?',
        'Add it to U, the result of the attention residual. There are two successive residual additions in the block. Its final output includes both updates.',
        'For the dog’s first CLS coordinate, the stored values verify '
        '0.462772608−0.099954687≈0.362817913. For all rows, E¹=U+MLP(LayerNorm(U)), '
        'with U and E¹ shaped 197×192. Each row uses the same MLP weights but receives a different '
        'update because its input features differ. The MLP does not mix rows: attention has already '
        'supplied cross-patch context. The next block receives the complete E¹ matrix, including '
        'all 196 patch rows and CLS.',
        '<p>CLS after attention: '+vec('cls_after_attention')+'.</p>'
        '<p>Add the MLP update: '+vec('cls_mlp_update')+'.</p>'
        '<p><strong>Block 1 CLS output:</strong> '+vec('cls_after_block1')+' (1 × 192).</p>'
        '<p>For all rows: E¹ = U + MLP(LayerNorm(U)), with shape 197 × 192.</p>')

    body = image(35, 6, 100, 67)+t(166, 47, 'Same dog · carry every updated row forward', 29)
    body += matrix(35, 157, 122, 'Input E')
    body += matrix(490, 157, 140, 'E¹')
    body += matrix(988, 157, 137, 'E²')
    for x, label, color in [(210, 'Block 1', 'c-q'), (709, 'Block 2', 'c-v')]:
        body += rect(x, 97, 217, 247, color, 'transparent', 8)
        body += t(x+108, 128, label, 28, color, 'middle')
        body += box(x+13, 146, 191, ['Attention + add'], color, 61, 23)
        body += arrow(x+108, 214, x+108, 237, color)
        body += box(x+13, 245, 191, ['MLP + add'], color, 61, 23)
        body += t(x+108, 330, 'own learned weights', 19, color, 'middle')
    body += arrow(164, 228, 197, 228)+arrow(439, 228, 477, 228)
    body += arrow(641, 228, 696, 228)+arrow(938, 228, 975, 228)
    body += t(580, 390, 'Block 1’s output is exactly block 2’s input.', 31, 'c-e', 'middle')
    body += t(580, 436, 'All 197 rows continue. Both blocks return 192 features per row.', 28, 'ink', 'middle')
    add('real-block-handoff', 'Pass the complete output of block 1 into block 2', body,
        'The entire output matrix continues to the next block: CLS and all 196 patch rows. Block 2 has its own attention and MLP weights. It applies the same sequence of operations to the feature values produced by block 1.',
        'Which rows do we keep, and which block processes them next?',
        'Keep every row. Follow the straight arrow into a distinct block 2. There is no return to block 1, no repeated patchification, and no new CLS token.',
        'E¹ is both block 1’s output and block 2’s input. This model has separate block instances, '
        'each with its own LayerNorm parameters, Q/K/V projections, attention output projection and '
        'MLP layers. The small attention and MLP boxes each include their normalization and residual '
        'addition. The displayed matrices abbreviate their coordinates; the highlighted first row '
        'tracks CLS. Patch identities stay in the same row order as their representations acquire context.',
        '<p>E (CLS, P1, …, P196) → block 1 → E¹ → block 2 → E².</p>'
        '<p>Every matrix is 197 × 192. Each block includes attention + residual, then MLP + residual.</p>'
        '<p>Block 2 has its own learned parameters and receives all the updated rows.</p>')

    body = t(35, 38, 'Feature values flow forward; each block uses its own parameter set.', 29)
    body += box(35, 177, 188, ['E', 'input features'], 'c-e', 86, 25)
    body += box(287, 163, 239, ['Block 1', 'attention + MLP'], 'c-q', 114, 27)
    body += box(590, 177, 188, ['E¹', 'new features'], 'c-e', 86, 25)
    body += box(842, 163, 239, ['Block 2', 'attention + MLP'], 'c-v', 114, 27)
    body += arrow(234, 220, 273, 220)+arrow(538, 220, 576, 220)+arrow(790, 220, 829, 220)
    body += arrow(1091, 220, 1134, 220)
    body += box(287, 73, 239, ['Block 1 weights'], 'c-q', 48, 24)+arrow(406, 125, 406, 153, 'c-q')
    body += box(842, 73, 239, ['Block 2 weights'], 'c-v', 48, 24)+arrow(961, 125, 961, 153, 'c-v')
    body += t(35, 334, 'Recomputed in each block:', 26, 'c-e', weight=600)
    body += t(35, 374, 'Q, K, V → attention weights → messages → MLP activations → updated rows', 25)
    body += t(35, 434, 'Learned weights stay fixed in this forward pass. Training updates them later.', 27)
    add('real-block-changes', 'What changes as the rows move through the blocks?', body,
        'Each block recomputes queries, keys, values, attention weights and MLP activations from its input. Its learned parameters stay fixed during this forward pass. The next block uses a different parameter set. Matrix dimensions and row identities stay the same.',
        'Are we changing the weights or the feature values when we run this dog forward?',
        'The feature values change. The blocks already have their own stored weights, which stay fixed during inference. Training later computes gradients and the optimizer updates those stored parameters.',
        'The two boxes denote complete parameter sets, including each block’s two LayerNorms, Q/K/V '
        'projections, attention output projection and two MLP layers with biases. They are different '
        'stored sets, not the same weights being repeatedly overwritten. In a forward pass, the input '
        'matrix changes between blocks; consequently the computed attention patterns and hidden '
        'activations can change too. Within a block, its MLP and projections share parameters across '
        'all token rows. Across depth, this ViT uses separate parameters for each block.',
        mobile_rows(['Quantity', 'What happens'],
                    [['Features, Q/K/V, weights from softmax, MLP activations', 'Recomputed from each block’s input'],
                     ['Learned parameters', 'Separate set per block; fixed during the forward pass'],
                     ['Rows and output width', 'CLS + 196 patches; 192 features per row']])
        +'<p>Training changes the learned parameters through backpropagation and an optimizer step.</p>')

    body = t(35, 40, 'One forward pass through 12 distinct Transformer blocks', 32)
    body += box(35, 167, 150, ['E', '197 × 192'], 'c-e', 83, 24)
    for x, name in [(245, 'Block 1'), (505, 'Block 2'), (935, 'Block 12')]:
        body += box(x, 135, 189, [name, 'attention + MLP', 'own weights'], 'c-q', 145, 23)
    body += arrow(196, 208, 233, 208)+arrow(446, 208, 493, 208)
    body += arrow(707, 208, 758, 208)+t(813, 219, '…', 40, 'ink-2', 'middle')+arrow(866, 208, 922, 208)
    body += t(469, 119, 'E¹', 25, 'c-e', 'middle')+t(731, 119, 'E²', 25, 'c-e', 'middle')
    body += t(901, 119, 'E¹¹', 25, 'c-e', 'middle')
    body += t(580, 331, 'Every arrow carries all 197 × 192 feature values.', 30, 'c-e', 'middle')
    body += g(t(580, 381, 'After block 12 → final normalization → read CLS → class scores', 28, 'ink', 'middle')
              +t(580, 433, 'One image. Twelve blocks. One prediction at the end.', 30, 'c-q', 'middle'), 1)
    add('real-cls-depth', 'Continue through the stack, then classify the image', body,
        'Pass the updated features through blocks 1 to 12 in order. The architecture repeats, with separate learned parameters in each block. After block 12 and final normalization, read CLS and calculate one set of class scores for this image.',
        'What does “12 blocks” repeat, and when do we make a prediction?',
        'The attention-plus-MLP architecture repeats. A different block processes each successive representation. All rows continue through the stack; one classifier reads the final CLS.',
        'Each block preserves shape 197×192 and has its own parameter set. E¹, E² and E¹¹ '
        'label outputs of blocks 1, 2 and 11; the superscript is a block index, not an exponent. '
        'The photograph is patchified and projected once before this stack. The trained initial CLS '
        'and position embeddings are added once. Later blocks receive the preceding block’s contextual '
        'representations. Twelve is this checkpoint’s chosen depth. The final LayerNorm and class '
        'head come after block 12; the next slide opens that readout.',
        '<p>E → block 1 → E¹ → block 2 → E² → … → block 12.</p>'
        '<p>All 12 blocks have their own weights. Every block receives and returns 197 × 192 features.</p>'
        '<p>After block 12: final normalization → select CLS → class scores → one prediction.</p>')
    return result
