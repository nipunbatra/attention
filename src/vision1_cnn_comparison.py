"""Three table-led, visual comparisons of conventional CNNs and a plain ViT."""
from vision1_focus_common import Figures


def build(b):
    f = Figures(b)
    t, box, line, arrow = f.t, f.box, f.line, f.arrow
    mobile_rows = b['mobile_rows']
    frames = []
    vit_source = '<a href="https://arxiv.org/html/2010.11929v2#S3.SS1">ViT: architecture and spatial assumptions</a>'
    deit_source = '<a href="https://proceedings.mlr.press/v139/touvron21a.html">DeiT: the training recipe matters</a>'
    convnext_source = '<a href="https://arxiv.org/abs/2201.03545">ConvNeXt: modern CNNs remain competitive</a>'

    def cell_text(x, y, lines, h, color='ink', size=25):
        lines = [lines] if isinstance(lines, str) else lines
        return ''.join(t(x, y+h/2+8+(j-(len(lines)-1)/2)*29, s, size, color)
                       for j, s in enumerate(lines))

    def header(y, first='Compare', widths=(265, 410, 415)):
        out = ''
        x = 35
        for w, label, color in zip(widths, [first, 'CNN', 'ViT · global attention'], ['ink-2', 'c-v', 'c-q']):
            out += f.rect(x, y, w, 44, 'line', 'card', 0)
            out += cell_text(x+18, y, label, 44, color, 27)
            x += w
        return out

    def row(y, values, h=54, widths=(265, 410, 415), label_inset=18):
        out = ''
        x = 35
        for j, (w, value) in enumerate(zip(widths, values)):
            out += f.rect(x, y, w, h, 'line', 'transparent', 0)
            out += cell_text(x+(label_inset if j == 0 else 18), y, value, h,
                             ['ink-2', 'c-v', 'c-q'][j], 24)
            x += w
        return out

    def add(key, title, body, caption, question, point, prose, headers, rows):
        flat_rows = [[(' '.join(v) if isinstance(v, list) else v) for v in r] for r in rows]
        f.add(key, title, body, caption, question, point, prose,
              mobile_rows(headers, flat_rows)+'<p>'+caption+'</p>')
        frames.append(f.frames[key])

    # The photographs carry the comparison; the short rows name what students see.
    body = t(35, 29, 'Same photograph. Same goal: one image label.', 29)
    body += header(51)
    for x, w in [(35, 265), (300, 410), (710, 415)]:
        body += f.rect(x, 95, w, 157, 'line', 'transparent', 0)
    body += cell_text(53, 95, ['How context', 'is gathered'], 157, 'ink-2', 26)
    body += f.image(322, 103, 140, 140, f.photo)
    for size in [26, 58, 94]:
        body += f.rect(392-size/2, 168-size/2, size, size, 'c-v', 'transparent', 0)
    body += cell_text(484, 95, ['Nearby first', '↓', 'Wider context'], 157, 'c-v', 25)
    body += f.image(731, 103, 140, 140, f.photo)
    for j in range(1, 14):
        body += line(731+j*10, 103, 731+j*10, 243, 'card', .5)
        body += line(731, 103+j*10, 871, 103+j*10, 'card', .5)
    body += f.rect(791, 143, 10, 10, 'c-q', 'transparent', 0)
    for x, y in [(746, 118), (856, 128), (746, 228), (856, 228)]:
        body += arrow(x, y, 796, 148, 'c-q')
    body += cell_text(892, 95, ['Connect near', 'and distant', 'patches'], 157, 'c-q', 25)
    rows = [
        ['Mix information', 'Shared filters on neighbours', 'Weights depend on the image'],
        ['Use a wider view', 'Combine regions through layers', 'Mix distant patches in one block'],
        ['Read out a label', 'Often pool → class head', 'CLS (here) → class head'],
    ]
    for j, values in enumerate(rows):
        body += row(252+j*49, values, 49)
    body += t(35, 434, 'Both can learn local detail and whole-image context.', 29)
    add('cnn-receptive-field', 'Two ways to build an image representation', body,
        'A CNN builds a wider view through layers of local filters. Global ViT attention lets distant patches interact within a block. Both can use the whole image and produce class scores. The connections shown are schematic.',
        'Can both models use clues from distant parts of the dog?',
        'Yes. Follow the growing CNN windows and the ViT arrows. The difference is the route information takes, not whether a model can ever see the whole image.',
        'This compares a conventional CNN with the plain global-attention ViT in this lecture. '
        'The CNN windows illustrate growing context, not exact pixel-sized kernels. The ViT arrows show '
        'permitted connections, not measured attention strengths. CNN readouts often pool spatial features; '
        'this ViT reads CLS. Other readouts and hybrid architectures exist. '+vit_source+'.',
        ['Question', 'CNN', 'ViT'], rows)

    # One shared example motivates the term; the table supplies the comparison.
    body = t(35, 29, 'Inductive bias = assumptions built into how the model learns.', 29)
    body += cell_text(35, 58, ['Example: move', 'the same stripe.'], 110, 'ink-2', 26)
    for x, shift in [(327, 0), (749, 3)]:
        for r in range(5):
            for c in range(7):
                active = c == 1+shift and r in [1, 2, 3]
                body += f.rect(x+c*20, 59+r*20, 20, 20, 'line', 'ink' if active else 'card', 0)
        body += f.rect(x+shift*20, 79, 60, 60, 'c-v', 'transparent', 0)
    body += arrow(496, 107, 720, 107, 'c-e')
    body += t(608, 80, 'new location', 23, 'c-e', 'middle')
    body += cell_text(926, 58, ['Same local', 'pattern'], 110, 'c-v', 25)
    body += header(184, 'What is built in?')
    rows = [
        ['Spatial structure', ['Local filters, reused', 'across the image'], ['Patches + position;', 'flexible global mixing']],
        ['The shifted stripe', ['The same filter can', 'detect it elsewhere'], ['Position can change', 'the attention pattern']],
        ['Learned from data', 'Which local filters help', 'Which patch relations help'],
    ]
    for j, values in enumerate(rows):
        body += row(228+j*64, values, 64)
    add('cnn-inductive-bias', 'Inductive bias: a useful starting assumption', body,
        'A CNN builds in local reuse: one learned detector can look for a pattern at many locations. ViT also has structure—patches, shared projections and position signals—but gives attention more freedom to learn spatial relationships.',
        'Does inductive bias mean the edge detector’s weights were chosen by hand?',
        'No. The architecture chooses local connections and parameter sharing; training learns the filter values. Both architectures still learn from data. The translated stripe is an illustration, not a model prediction.',
        'The highlighted neighbourhood is identical before and after the shift, so a shared local '
        'filter can reuse its response. Exact convolutional shift equivariance depends on stride '
        'and boundaries; it does not guarantee an unchanged final class prediction. ViT shares '
        'projection parameters across rows too. Its position signals can affect matching after a shift. '
        'The table describes architectural preferences, not hard limits on what either model can learn. '
        +vit_source+'.',
        ['Comparison', 'CNN', 'ViT'], rows)

    # Small pictograms distinguish the four practical situations without more theory.
    def scenario_icon(kind, x, y):
        out = ''
        if kind == 'data':
            for dx, dy in [(0, 0), (9, 9), (18, 18)]:
                out += f.rect(x+dx, y+dy, 25, 25, 'c-e', 'card', 2)
                out += line(x+dx+5, y+dy+18, x+dx+12, y+dy+10, 'c-e')
                out += line(x+dx+12, y+dy+10, x+dx+21, y+dy+18, 'c-e')
        elif kind == 'pretrained':
            out += f.rect(x+1, y, 41, 25, 'c-e', 'card', 3)
            out += t(x+21, y+19, 'W', 20, 'c-e', 'middle')
            out += arrow(x+21, y+29, x+21, y+47, 'c-e')
        elif kind == 'phone':
            out += f.rect(x+8, y-2, 30, 47, 'c-e', 'transparent', 4)
            out += line(x+13, y+5, x+32, y+5, 'c-e')+line(x+18, y+39, x+27, y+39, 'c-e')
        else:
            out += f.rect(x-2, y+3, 48, 34, 'line', 'transparent', 0)
            out += f.rect(x+2, y+7, 9, 9, 'c-e', 't-e', 0)
            out += f.rect(x+33, y+24, 9, 9, 'c-q', 't-q', 0)
            out += line(x+12, y+16, x+32, y+24, 'c-q')
        return out

    widths = (340, 375, 375)
    body = t(35, 29, 'Practical starting points · validate on your task and hardware.', 28)
    body += header(51, 'Your situation', widths)
    rows = [
        [['Few labels;', 'train from scratch'], ['Useful baseline:', 'built-in local reuse'], ['Pay close attention', 'to the training recipe']],
        [['Few labels;', 'pretrained weights'], ['Fine-tune a suitable', 'pretrained CNN'], ['Fine-tune a suitable', 'pretrained ViT']],
        [['Phone or tight', 'latency budget'], ['Try a compact CNN;', 'measure on the device'], ['Try an efficient ViT;', 'measure on the device']],
        [['Distant clues', 'across the image'], ['Use enough depth', 'for broad context'], ['Global attention links', 'distant patches directly']],
    ]
    for j, (kind, values) in enumerate(zip(['data', 'pretrained', 'phone', 'context'], rows)):
        y = 95+j*74
        body += row(y, values, 74, widths, label_inset=83)
        body += scenario_icon(kind, 52, y+16)
    body += t(35, 435, 'Compare validation accuracy, speed and memory under the same budget.', 27)
    add('cnn-vit-design', 'Which is a sensible starting point?', body,
        'Treat these as starting points, not a ranking. Pretraining can matter more than the architecture name on a small dataset. Compare suitable models using the same validation set and measure speed and memory on the target hardware.',
        'We have few labelled pet photos and access to pretrained weights. Which column should we choose?',
        'Both are plausible: adapt suitable checkpoints and validate. Few task labels do not rule out a pretrained ViT. For device use, compare measured latency and memory rather than assuming every CNN is faster.',
        'These are practical recommendations inferred from architectural differences and published studies, '
        'not results from a new benchmark. The original ViT benefited from large-scale pretraining. '
        'DeiT subsequently demonstrated competitive transformers trained on ImageNet without external data, '
        'showing why the training recipe matters. Modern CNNs such as ConvNeXt also remain competitive. '
        'There is no universal image-count threshold or architecture winner. Model size, resolution, '
        'checkpoint suitability, augmentation and implementation affect the comparison. '
        +vit_source+' · '+deit_source+' · '+convnext_source+'.',
        ['Situation', 'CNN starting point', 'ViT starting point'], rows)
    return frames
