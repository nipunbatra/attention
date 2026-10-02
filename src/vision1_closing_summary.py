"""Two visual takeaways end the lecture; retain practice in the reading view."""
import re
from html import escape

from vision1_focus_common import Figures


def conclude(b, sections):
    f = Figures(b)
    t, line, arrow, rect = f.t, f.line, f.arrow, f.rect

    def key(markup):
        return re.search(r'class="frame[^\"]*" id="([^\"]+)"', markup).group(1)

    def plus(x, y):
        return (f'<circle cx="{x}" cy="{y}" r="17" fill="var(--card)" '
                'stroke="var(--c-e)" stroke-width="2"/>'
                + t(x, y+10, '+', 32, 'c-e', 'middle'))

    def route(points, color='c-e', dashed=False):
        return ''.join(line(*a, *c, color, 2, '5 5' if dashed else '')
                       for a, c in zip(points[:-2], points[1:-1])) + arrow(*points[-2], *points[-1], color)

    # The photo-to-label route stays large; one block is opened below it.
    body = f.image(25, 20, 128, 128, f.photo)
    for i in range(1, 14):
        body += line(25+i*128/14, 20, 25+i*128/14, 148, 'card', .5)
        body += line(25, 20+i*128/14, 153, 20+i*128/14, 'card', .5)
    body += t(89, 181, '224 × 224 RGB', 23, 'ink-2', 'middle')
    body += arrow(164, 83, 199, 83)
    body += f.box(210, 37, 240, ['Shared patch layer', '+ CLS + position'], h=92, size=26)
    body += t(330, 172, '196 patches + 1 CLS', 25, 'c-e', 'middle')
    body += arrow(460, 83, 494, 83)
    body += f.box(505, 37, 245, ['12 Transformer blocks', '197 × 192'], h=92, size=24)
    body += t(627, 172, 'All rows gain context', 25, 'c-e', 'middle')
    body += arrow(760, 83, 795, 83)
    body += f.box(805, 37, 130, ['Final LN', 'read CLS'], 'vision', h=92, size=25)
    body += t(870, 172, '192 features', 23, 'vision', 'middle')
    body += arrow(945, 83, 970, 83)
    body += f.box(980, 37, 155, ['Linear head', '1,000 scores'], h=92, size=24)
    body += arrow(1057, 139, 1057, 178, 'c-v')
    body += t(1057, 202, 'softmax → label', 23, 'vision', 'middle')

    body += line(516, 189, 35, 234, 'c-e', 1.5, '5 5')
    body += line(740, 189, 1125, 234, 'c-e', 1.5, '5 5')
    body += rect(25, 235, 1110, 245, 'line', 'transparent', 6)
    body += t(46, 270, 'INSIDE EACH BLOCK', 23, 'c-e', weight=700)
    body += t(1108, 270, 'Each block has its own learned weights', 23, 'ink-2', 'end')

    y = 356
    body += t(55, y+9, 'E', 31, 'c-e', 'middle')
    body += arrow(74, y, 104, y)
    body += f.box(113, y-30, 70, 'LN', h=60, size=25)
    body += arrow(193, y, 218, y)
    # Three visibly parallel head messages enter the common output projection.
    for i in range(3):
        hy = 304+i*42
        body += f.box(228, hy, 140, f'Head {i+1}', 'mixing', h=34, size=23)
        body += route([(210, y), (210, hy+17), (223, hy+17)], 'mixing')
        body += route([(373, hy+17), (393, hy+17), (393, y), (412, y)], 'mixing')
    body += f.box(422, y-37, 166, ['Join + project', '192 features'], 'c-v', h=74, size=23)
    body += arrow(598, y, 622, y, 'c-v') + plus(642, y)
    body += route([(55, 335), (55, 289), (642, 289), (642, y-20)])
    body += arrow(663, y, 699, y)
    body += f.box(709, y-30, 70, 'LN', h=60, size=25)
    body += arrow(789, y, 814, y)
    body += f.box(824, y-37, 160, ['MLP', '192→768→192'], 'neutral', h=74, size=21)
    body += arrow(994, y, 1012, y, 'c-v') + plus(1032, y)
    body += route([(679, y), (679, 437), (1032, 437), (1032, y+20)])
    body += t(854, 464, 'Keep the row + add an update', 24, 'c-e', 'middle')
    body += arrow(1053, y, 1084, y) + t(1101, y+9, 'E′', 31, 'c-e', 'middle')
    body += t(355, 464, 'Attention: mix across rows', 25, 'mixing', 'middle')
    f.add('vision-summary-architecture', 'The whole ViT: pixels → context → one label', body,
          'One shared patch layer builds the rows. Twelve blocks refine every row. The classifier reads final CLS to score image labels.',
          'Can you trace the image path, then name the two updates inside a block?',
          'Trace projection, CLS and position, twelve blocks, final normalization, CLS and class head. '
          'Open one block: normalize, three attention heads, concatenate and project, residual; normalize, MLP, residual.',
          '<p>This is the same trained tiny ViT throughout the lecture: 16×16 RGB patches, 196 patch rows, '
          '192 features, three 64-wide attention heads per block, and twelve blocks. LN means LayerNorm. '
          'The shared patch projection maps 768 input values to 192 features; Conv2d with kernel and stride 16 '
          'implements the same affine map as a shared Linear layer on flattened patches.</p>'
          '<p>Within a head, softmax(QKᵀ/√64)V gives one message per query. Concatenation and the output '
          'projection make a 192-wide update for every row. Each plus sign adds that update to the incoming '
          'row. The MLP transforms each row separately. Final LayerNorm precedes selection of CLS.</p>'
          '<p>The original detailed reference figure, including Q/K/V, shapes, class probabilities and the label '
          'loss, remains in <a href="figures/vision1/photo-label-loss.svg">the optional whole-model reference</a>. '
          'The top path is the full model; the lower panel is a zoom of one block, not an additional block.</p>',
          '<p>224×224 RGB → shared 768-to-192 patch projection → 196 patch rows → add CLS and positions '
          '→ 197×192 → twelve blocks → final LayerNorm → CLS (192) → Linear head → 1,000 class scores.</p>'
          '<p>Each block: LayerNorm → three attention heads → join and project → residual addition; '
          'LayerNorm → MLP → residual addition. All rows update.</p>', height=500)

    # Four small drawings anchor four short ideas. Details live in the notes.
    body = line(580, 10, 580, 472) + line(25, 246, 1135, 246)
    body += t(35, 42, '1 · Content + location', 32, 'ink', weight=650)
    body += ('<svg x="40" y="80" width="90" height="90" viewBox="96 64 16 16" overflow="hidden">'
             + f.image(0, 0, 224, 224, f.photo) + '</svg>')
    body += arrow(142, 125, 176, 125)
    body += f.box(187, 97, 160, 'patch features', h=57, size=24)
    body += t(373, 134, '+', 35, 'special', 'middle')
    body += f.box(401, 97, 140, 'position', 'special', h=57, size=25)
    body += t(35, 217, 'Shared projection; each slot has a position.', 25, 'ink-2')

    body += t(610, 42, '2 · Every row gains context', 32, 'ink', weight=650)
    for i, name in enumerate(['CLS', 'P1', 'P2']):
        y = 76+i*40
        color = 'vision'
        body += f.box(615, y, 86, name, color, h=30, size=22)
        body += f.box(1005, y, 110, name+'′', color, h=30, size=22)
        body += arrow(708, y+15, 761, 127, color)
        body += arrow(938, 127, 995, y+15, color)
    body += f.box(773, 93, 153, ['Attention', '+ MLP'], 'c-v', h=68, size=25)
    body += t(610, 217, 'Attention mixes; the MLP transforms.', 25, 'ink-2')

    body += t(35, 286, '3 · The task teaches the summary', 31, 'ink', weight=650)
    body += f.box(35, 323, 119, 'Final CLS', 'c-q', h=57, size=24)
    body += arrow(164, 351, 201, 351)
    body += f.box(212, 323, 124, 'Class head', h=57, size=24)
    body += arrow(346, 351, 383, 351)
    body += f.box(394, 323, 147, 'Label loss', 'c-a', h=57, size=25)
    body += route([(467, 388), (467, 414), (94, 414), (94, 388)], 'c-a')
    body += t(35, 462, 'Train with loss; infer with fixed weights.', 25, 'ink-2')

    body += t(610, 286, '4 · More tokens cost more', 32, 'ink', weight=650)
    for x, size, count in [(625, 52, 197), (866, 104, 785)]:
        body += rect(x, 318, size, size, 'mixing', 'card', 0)
        for i in range(1, 4):
            body += line(x+i*size/4, 318, x+i*size/4, 318+size, 'mixing', .8)
            body += line(x, 318+i*size/4, x+size, 318+i*size/4, 'mixing', .8)
        body += t(x+size/2, 311, f'{count}²', 23, 'mixing', 'middle')
    body += arrow(707, 365, 827, 365, 'mixing')
    body += t(768, 349, '≈16×', 30, 'mixing', 'middle', 650)
    body += t(991, 346, '224 → 448', 23, 'ink-2')
    body += t(991, 380, 'patch = 16', 23, 'ink-2')
    body += t(610, 462, 'Twice the side length: ≈16× scores.', 25, 'ink-2')

    # Optional materials remain readable without becoming additional slides.
    practice = '<h3>Optional practice and next steps</h3><p>The main lecture ends above. '
    practice += 'Use these questions to check your understanding in the reading view.</p>'
    metadata = {item['id']: item for item in b['FRAMES']}
    for _, frames in sections[10:]:
        for markup in frames:
            k = key(markup)
            if k.startswith('vision-topic-') or k == 'closing':
                continue
            markup = re.sub(r'class="frame[^\"]*"', 'class="vp-optional-practice"', markup, count=1)
            markup = re.sub(r' data-build="\d+"', '', markup)
            practice += '<h3>'+escape(metadata[k]['title'])+'</h3>'+markup
    f.add('vision-summary-takeaways', 'Four ideas to carry forward', body,
          'Pixels supply content. Position supplies location. Attention builds context. Training makes the final representation useful for the task.',
          'Which operation supplies content, location, context and the learning signal?',
          'Patch projection supplies content; learned positions identify slots; attention mixes source values for every query; '
          'label loss trains the whole classifier. More tokens increase the number of query–key comparisons quadratically.',
          '<p>The token diagram is schematic: the real model updates CLS and all 196 patches. '
          'The loss diagram summarizes backpropagation through the head and all preceding operations. '
          'CLS is a learned readout choice; mean pooling is another valid choice when the model is trained for it.</p>'
          '<p>At fixed 16×16 patch size, 224×224 gives 197 tokens with CLS and 448×448 gives 785. '
          'The score counts per head and block are 38,809 and 616,225, a factor of 15.88. '
          'The matrix icons are schematic. This counts dense attention scores, not total model computation or '
          'the memory allocated by an optimized attention kernel. Adapting this checkpoint to a larger grid '
          'also requires adapting its positional embeddings.</p>'+practice,
          '<ol><li>Content + location: shared patch projection plus learned position.</li>'
          '<li>Every token gains context through attention and is transformed by the MLP.</li>'
          '<li>Image-label loss trains useful representations; inference keeps weights fixed.</li>'
          '<li>At patch size 16, increasing the image from 224 to 448 gives 197 to 785 tokens: '
          'about sixteen times as many attention scores.</li></ol>', height=490)

    assert len(sections) == 12 and key(sections[9][1][-1]) == 'cost-control'
    result = list(sections[:10])
    title, frames = result[-1]
    result[-1] = (title, frames + [f.frames['vision-summary-architecture'], f.frames['vision-summary-takeaways']])
    return result
