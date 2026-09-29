"""Explain location and CLS before resuming the real-image forward pass."""
import base64
import re


def clarify_inputs(b, sections):
    t, g, rect, arrow, line, image, frame = (b[k] for k in
        ['t', 'g', 'rect', 'arrow', 'line', 'image', 'frame'])
    photo = 'data:image/png;base64,' + base64.b64encode(
        (b['ASSETS'] / 'model-input.png').read_bytes()).decode()

    def key(markup):
        return re.search(r'class="frame[^\"]*" id="([^\"]+)"', markup).group(1)

    def without_route(markup):
        markup = re.sub(r'<div class="vp-pathbar">.*?</svg></div>', '', markup, flags=re.S)
        return markup.replace(' vp-with-route', '')

    def box(x, y, w, labels, color='c-e', h=86):
        out = rect(x, y, w, h, color, 't-q' if color == 'c-q' else 't-e', 6)
        for j, label in enumerate(labels):
            out += t(x+w/2, y+35+j*32, label, 25, color, 'middle')
        return out

    body = t(70, 37, 'Keep the photograph exactly as it is.', 29)
    body += image(100, 87, 280, 280, photo)
    for k in range(1, 14):
        body += line(100+20*k, 87, 100+20*k, 367, 'card', .7)
        body += line(100, 87+20*k, 380, 87+20*k, 'card', .7)
    body += rect(220, 167, 20, 20, 'c-q', 'transparent', 0)
    body += t(230, 68, 'column 7', 22, 'c-q', 'middle')
    body += arrow(230, 78, 230, 163, 'c-q')
    body += t(15, 184, 'row 5', 22, 'c-q') + arrow(78, 177, 215, 177, 'c-q')
    body += t(240, 410, 'P63 · one of 196 patches', 26, 'c-q', 'middle')
    body += box(480, 85, 630, ['Content c₆₃: features from this crop', 'Already computed by Linear(768, 192)'])
    body += g(box(480, 214, 630, ['Position p₆₃: a vector for this grid slot', 'Row 5, column 7 → patch number 63'], 'c-q'), 1)
    body += g(t(480, 362, 'The shared pixel projection was never given', 27)
              + t(480, 402, 'the row or column. Add that information next.', 27), 2)
    location = frame('position-where', 'Give each patch its location in the photograph', body,
        'Recognizing a face depends on how its parts are arranged. c₆₃ describes this crop; p₆₃ tells attention where it belongs. Add them so the model can use both appearance and layout.',
        'Which part of the shared patch projection received row 5, column 7?\n'
        'None: the projection received only pixel values. Point to the location on the unchanged photograph, then introduce its position vector.',
        'Number patches from left to right, then top to bottom. P63 is in row 5, column 7 because 4×14+7=63. '
        'The model learns a 192-coordinate vector for each slot; it does not literally append the integers 5 and 7. '
        'The content vector can describe visual features, but its shared projection has no explicit input specifying the crop’s grid location. '
        'Position supplies this information before attention. We keep the image and the patch order unchanged throughout the forward pass.',
        '<p>The photograph is unchanged. P63 is in <strong>row 5, column 7</strong> of the 14 × 14 grid: 4 × 14 + 7 = 63.</p>'
        '<p><strong>c₆₃:</strong> 192 content features computed from its pixels.</p>'
        '<p><strong>p₆₃:</strong> 192 learned coordinates for this grid slot.</p>'
        '<p>Recognizing a face depends on how its parts are arranged. The shared pixel projection has not received the row or column. Add position so attention can use appearance and layout.</p>')

    body = t(55, 45, 'A SHORT DETOUR · WHY ADD CLS?', 23, 'c-q', weight=600)
    body += t(55, 100, 'One image label from 196 patch rows', 44, weight=700)
    body += t(55, 143, 'Which representation should the classifier read?', 29, 'ink-2')
    body += image(55,208,158,158,photo)
    for k in range(1,14):
        offset=158*k/14
        body += line(55+offset,208,55+offset,366,'card',.65)
        body += line(55,208+offset,213,208+offset,'card',.65)
    body += t(134,397,'The whole dog photo',22,'ink-2','middle')
    body += arrow(219,287,266,287,'c-e')
    body += t(385,232,'196 patch rows',21,'c-e','middle')
    for y,label in [(240,'P1 features'),(283,'P2 features'),(350,'P196 features')]:
        body += rect(285,y,200,34,'c-e','t-e',3)+t(385,y+25,label,23,'c-e','middle')
    body += t(385,338,'⋮',26,'c-e','middle')
    body += rect(885,353,220,45,'c-e','t-e',5)+t(995,383,'Classifier',25,'c-e','middle')
    body += arrow(995,399,995,411,'c-e')+t(995,434,'One image label',25,'c-e','middle')

    # Reveal the design only after students see the many-rows / one-label problem.
    collect = rect(285,170,200,38,'c-q','t-q',4)+t(385,197,'CLS · learned start',22,'c-q','middle')
    collect += t(265,198,'+',29,'c-q','end')
    collect += line(491,189,532,189,'c-q')+line(532,189,532,367,'c-q')
    for y in [257,300,367]:
        collect += line(491,y,532,y,'c-e')
    collect += arrow(532,280,568,280,'c-q')
    collect += rect(585,217,210,125,'c-q','transparent',6)
    collect += t(690,251,'Transformer',25,'c-q','middle')+t(690,284,'blocks',25,'c-q','middle')
    collect += t(690,322,'all 197 rows',23,'ink-2','middle')
    collect += t(690,375,'CLS reads patch features',22,'c-q','middle')
    body += g(collect,1)

    readout = arrow(802,280,868,280,'c-q')
    readout += box(885,226,220,['Final CLS','image summary'],'c-q',h=82)
    readout += arrow(995,314,995,343,'c-q')
    body += g(readout,2)
    body += g(t(55,437,'Averaging final patch rows is another readout option.',24,'ink-2'),2)
    detour = frame('cls-detour', 'Why add CLS? Give the classifier one image summary', body,
        'Patch embeddings give us many rows, while the target is one label for the photograph. We need a way to combine patch information into the representation that the classifier will read.',
        'We have 196 patch rows and want one label for the whole photograph. Which representation should the classifier read?\n'
        'Start with the dog and its patch rows. Reveal one extra learned CLS row before the blocks. All rows enter together, and attention updates CLS using patch information. Then reveal the readout from final CLS into the classifier. CLS is one design choice; averaging final patch rows also works with a model trained for that readout.',
        'The task gives a label for the whole image, while patch embedding has supplied 196 separate feature rows. '
        'We need a rule for turning those rows into an image-level prediction. This checkpoint chooses a designated summary row called CLS, '
        'short for classification token. Add it before the Transformer blocks so that it can participate in self-attention alongside the patch rows. '
        'Its initial learned vector is shared across images; it does not yet contain information about this particular dog. '
        'Through attention it receives weighted mixtures of source value vectors, and the blocks transform its representation. '
        'The final CLS therefore depends on this image. After the blocks and final normalization, the classifier reads that row to produce class scores. '
        'The class-label training loss teaches the model which information makes this readout useful. '
        'All patch rows are updated too; this overview follows only CLS at the output. There is no extra image crop, supplied answer label, or separately supervised patch label. '
        'CLS is a learned readout mechanism, not a mathematical requirement for image classification. A model can instead average its final patch rows '
        'and train a classifier on that vector; the later comparison explains this alternative with the same dog. '
        'Position and CLS have separate jobs: position supplies location, while CLS is the row selected for the image-level readout. '
        'The next slide compares the origins of patch rows and CLS; we then explain the stored parameters and how training changes them. '
        '<a href="https://arxiv.org/html/2010.11929v2#S3.SS1">Original ViT, §3.1</a>.',
        '<img src="figures/vision1/model-input.png" alt="The same dog photograph whose 196 patches must support one image prediction">'
        '<p><strong>Problem:</strong> we have 196 patch feature rows, but want one label for the whole photo. Which representation should the classifier read?</p>'
        '<ol><li><strong>Add CLS:</strong> reserve an extra learned row before the Transformer blocks.</li>'
        '<li><strong>Gather image information:</strong> all 197 rows pass through the blocks. Attention lets CLS read patch features; the patch rows are updated too.</li>'
        '<li><strong>Read final CLS:</strong> its updated, image-dependent features go to the classifier, which scores image labels.</li></ol>'
        '<p>CLS starts as shared model parameters. The next slides show where those numbers come from.</p>'
        '<p>CLS is one design choice. Averaging the final patch rows is another readout option.</p>')
    detour = detour.replace('class="frame vp-frame', 'class="frame vp-frame vp-topic-break vp-cls-detour', 1)

    from vision1_cls_story import build_cls_story
    cls_story = build_cls_story(b)
    from vision1_cls_readouts import build_cls_readouts
    cls_readouts = build_cls_readouts(b)
    from vision1_position_learning import build_position_learning
    position_story = build_position_learning(b)

    all_frames = {key(m): m for _, frames in sections for m in frames}
    experiments = ['position-photo-layout', 'position-photo-content', 'position-photo-add']
    transfer = experiments + ['real-patch-position', 'real-patch-qkv', 'model-journey-checkpoint']
    sections[1] = (sections[1][0], [m for m in sections[1][1] if key(m) not in transfer])
    # Two prerequisites are explained separately before any Q/K/V computation.
    continuation = [m for m in sections[2][1] if key(m) in [
        'real-cls-attention', 'real-attention-product', 'real-attention-cls-zoom', 'real-attention-weights',
        'real-attention-mask', 'real-cls-values-origin', 'real-cls-value-scaling', 'real-cls-value-sum',
        'real-attention-values', 'real-heads-intro', 'real-heads-qkv', 'real-heads-messages',
        'real-heads-cls', 'real-heads-concat', 'real-cls-message', 'real-cls-residual', 'real-cls-mlp',
        'real-mlp-network', 'real-mlp-residual', 'real-block-handoff', 'real-block-changes', 'real-cls-depth',
        'real-cls-readout', 'real-classifier-network', 'real-classifier-score',
        'real-classifier-softmax', 'real-cls-prediction']]
    third = [all_frames['vision-topic-03'], location, position_story['position-table'],
             all_frames['real-patch-position'], position_story['position-learning'], detour,
             cls_story['real-cls-purpose'], cls_story['cls-parameter-origin'],
             cls_story['cls-parameter-learning'], cls_story['cls-stored-start'], cls_story['cls-collect'],
             cls_story['cls-shared-start'], cls_story['cls-two-image-readout'],
             *cls_readouts.values(),
             all_frames['real-cls-sequence'], all_frames['model-journey-checkpoint'],
             all_frames['real-patch-qkv']] + continuation
    sections[2] = ('Prepare the rows, then classify the image', third)
    exercises = []
    for m in sections[12][1]:
        if key(m) == 'exercise-position':
            exercises.extend(without_route(all_frames[k]) for k in experiments)
        exercises.append(m)
    sections[12] = (sections[12][0], exercises)
    return sections
