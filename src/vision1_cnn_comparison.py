"""Four visual comparisons of conventional CNNs and a plain ViT."""
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

    # Two teaching beats per frame give each comparison point its own pause.
    # The drawings show permitted routes, not attention measured from this photo.
    def column_titles(y):
        return (t(35, y, 'CNN · local filters', 29, 'c-v', weight=600)
                +t(620, y, 'ViT · global attention', 29, 'c-q', weight=600))

    body = t(35, 29, '1 · Gather context: which inputs can interact in one layer?', 29)
    body += column_titles(77)
    body += f.image(35, 96, 170, 170, f.photo)
    body += f.rect(97, 151, 51, 51, 'c-v', 'transparent', 0)
    body += f.rect(114, 168, 17, 17, 'c-v', 'transparent', 0)
    body += arrow(155, 177, 231, 177, 'c-v')
    body += t(248, 137, 'One local neighbourhood', 26, 'c-v')
    body += t(248, 174, 'feeds one output location.', 26, 'c-v')
    body += t(248, 227, 'Start with nearby clues.', 25, 'ink-2')
    body += f.image(620, 96, 170, 170, f.photo)
    for j in range(1,14):
        body += line(620+j*170/14,96,620+j*170/14,266,'card',.5)
        body += line(620,96+j*170/14,790,96+j*170/14,'card',.5)
    body += f.rect(692,144,13,13,'c-q','transparent',0)
    for x,y in [(635,110),(777,116),(631,252),(777,250)]:
        body += arrow(x,y,699,151,'c-q')
    body += t(816, 137, 'One query can use', 26, 'c-q')
    body += t(816, 174, 'every source patch.', 26, 'c-q')
    body += t(816, 227, 'Near and far are available.', 25, 'ink-2')
    mixing = line(35,286,1125,286,'line')
    mixing += t(35,323,'2 · Mix information: what determines each contribution?',29)
    mixing += box(35,348,182,['Learned filter'],'c-v',h=50,size=24)
    mixing += arrow(226,373,258,373,'c-v')
    mixing += t(274,365,'Reuse the same weights',25,'c-v')
    mixing += t(274,400,'at every image location.',25,'c-v')
    mixing += box(620,348,174,['Query + keys'],'c-q',h=50,size=24)
    mixing += arrow(803,373,835,373,'c-q')
    mixing += t(851,365,'Compute weights',25,'c-q')
    mixing += t(851,400,'for this query + image.',25,'c-q')
    mixing += t(35,438,'Both learn parameters. ViT recomputes attention weights from the current features.',25,'ink-2')
    body += f.g(mixing,1)
    rows = [
        ['1. Gather context', 'One output uses a local neighbourhood.', 'One query can use near and distant patches.'],
        ['2. Mix information', 'The learned filter is reused at every location.', 'Query–key matches determine the source weights for this input.'],
    ]
    add('cnn-receptive-field', 'Two ways to build an image representation', body,
        'First compare the available connections. Then press Next to compare the mixing weights. CNN filters stay fixed during a forward pass. ViT projection parameters also stay fixed, while attention weights are computed from the current query and source features.',
        'What is shared, and what changes when we show a new image?',
        'Pause on point 1: a conventional convolution reads local neighbours; global attention can connect distant patches immediately. Reveal point 2: both models keep their learned parameters fixed during inference. A CNN reuses its learned kernel across locations; ViT computes attention coefficients from query–key matches. These coefficients depend on the input and receiver. They are not the stored projection weights.',
        'This compares a conventional CNN with the plain global-attention ViT in this lecture. '
        'The neighbourhood and patch arrows are schematic, not measured responses. A convolution computes '
        'weighted sums using a learned kernel reused across spatial positions. Its activations still change '
        'with the image. In ViT, learned Q/K/V projections stay fixed during inference, but the resulting '
        'query/key vectors and attention coefficients are input-dependent. Value vectors supply the '
        'features that the coefficients mix. This distinction concerns spatial mixing; both systems '
        'also contain nonlinear feature transformations. The next slide follows the wider context and readout. '
        +vit_source+'.',
        ['Point', 'CNN', 'ViT'], rows)

    body = t(35,29,'3 · Build a wider view: how do distant clues meet?',29)
    body += column_titles(77)
    for x,w,labels in [(35,150,['Local features']), (220,150,['Combine them']), (405,150,['Wider context'])]:
        body += box(x,112,w,labels,'c-v',h=72,size=22)
    body += arrow(190,148,213,148,'c-v')+arrow(377,148,398,148,'c-v')
    body += t(35,221,'Successive layers connect larger regions.',25,'c-v')
    body += box(620,112,185,['Near + far','patches'],'c-q',h=72,size=23)
    body += arrow(812,148,850,148,'c-q')
    body += box(859,112,266,['One global','attention layer'],'c-q',h=72,size=23)
    body += t(620,221,'Later blocks refine those relationships.',25,'c-q')
    readout = line(35,251,1125,251,'line')
    readout += t(35,291,'4 · Read out a label: turn many locations into one vector.',29)
    for r in range(3):
        for c in range(4):
            readout += f.rect(35+c*17,325+r*17,17,17,'c-v','transparent',0)
    readout += arrow(113,351,145,351,'c-v')
    readout += box(155,318,176,['Spatial average','one feature vector'],'c-v',h=67,size=21)
    readout += arrow(341,351,373,351,'c-v')
    readout += box(383,318,172,['Class head','image scores'],'c-v',h=67,size=22)
    for r,label in enumerate(['CLS','P1','…','P196']):
        readout += f.rect(620,314+r*20,63,20,'c-q','t-q' if r==0 else 'transparent',0)
        readout += t(651,329+r*20,label,15,'c-q','middle')
    readout += arrow(691,324,727,324,'c-q')
    readout += box(739,302,188,['Read final CLS','one feature vector'],'c-q',h=67,size=21)
    readout += arrow(935,336,967,336,'c-q')
    readout += box(977,302,148,['Class head','image scores'],'c-q',h=67,size=22)
    readout += t(35,436,'Both produce an image-level summary. Pooling is also a valid ViT readout.',27,'ink-2')
    body += f.g(readout,1)
    rows = [
        ['3. Build a wider view', 'Successive local layers combine larger regions.', 'A global attention layer connects distant patches; later blocks refine the features.'],
        ['4. Read out one label', 'Often average the final spatial features, then apply a class head.', 'This model reads final CLS, then applies a class head. Pooling is another option.'],
    ]
    add('cnn-context-readout','A wider view, then one image label',body,
        'Trace context through the layers, then press Next to inspect the readout. Both models can use the whole image. A classifier needs one image-level vector: CNNs often use spatial pooling; this ViT reads final CLS.',
        'Are distant clues unavailable to a CNN, and is CLS required for every ViT?',
        'Pause on point 3: repeated local layers broaden a conventional CNN’s receptive field; a global attention layer allows direct distant interactions. Later ViT blocks still refine features. Reveal point 4: average spatial features or read a trained CLS representation, then apply the class head. Both routes produce one vector. Return to the earlier CLS-versus-pooling discussion without re-deriving it.',
        'The CNN chain is schematic: depth, kernel size, stride and dilation determine its receptive field. '
        'Global attention permits a direct dependency between distant patches within a layer, without '
        'guaranteeing that a trained head assigns every distant patch a large weight. Both models still '
        'need learned features and useful training. A common CNN readout averages each channel across '
        'spatial locations; it preserves the feature/channel axis for the classifier. This ViT selects '
        'the final normalized CLS row. Its other rows have helped build that summary through attention. '
        'Mean pooling is another valid ViT design when the model is trained for that readout. '+vit_source+'.',
        ['Point', 'CNN', 'ViT'], rows)

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
