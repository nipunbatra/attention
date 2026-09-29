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

    body = t(55, 56, 'A SHORT DETOUR · THE IMAGE SUMMARY', 23, 'c-q', weight=600)
    body += t(55, 145, '196 patch descriptions.', 47, weight=700)
    body += t(55, 207, 'One image label.', 47, weight=700)
    body += box(55, 284, 280, ['Patch features', '196 rows'])
    body += arrow(350, 326, 421, 326, 'c-q')
    body += box(437, 284, 280, ['One summary', 'CLS'], 'c-q')
    body += arrow(733, 326, 804, 326, 'c-q')
    body += box(820, 284, 280, ['Classifier', 'One label'])
    detour = frame('cls-detour', 'CLS: one summary for the whole image', body,
        'Before we enter attention, decide where the image summary will live. This model reserves one extra row called CLS, short for classification token. Then we return to the same forward pass.',
        'We have 196 descriptions. Which representation should the classifier read?\n'
        'Pause the computation. Explain the need for one image summary before introducing an extra row of numbers.',
        'This is a conceptual detour. CLS is inserted into the sequence before the Transformer blocks, '
        'and its final updated representation is read after the last block. Position and CLS have separate jobs: '
        'position provides location; CLS provides one learned place to collect features for the image-level prediction.',
        '<p><strong>A short detour: 196 patch descriptions → one image summary → one label.</strong></p>'
        '<p>CLS means classification token. This model reserves one extra row for the summary.</p>'
        '<p>Position describes where a patch belongs. CLS provides the image summary. Then we return to the forward pass.</p>')
    detour = detour.replace('class="frame vp-frame', 'class="frame vp-frame vp-topic-break vp-cls-detour', 1)

    from vision1_cls_story import build_cls_story
    cls_story = build_cls_story(b)

    all_frames = {key(m): m for _, frames in sections for m in frames}
    experiments = ['position-photo-layout', 'position-photo-content', 'position-photo-add']
    transfer = experiments + ['real-patch-position', 'real-patch-qkv', 'model-journey-checkpoint']
    sections[1] = (sections[1][0], [m for m in sections[1][1] if key(m) not in transfer])
    # Two prerequisites are explained separately before any Q/K/V computation.
    continuation = [m for m in sections[2][1] if key(m) in [
        'real-cls-attention', 'real-cls-message', 'real-cls-mlp', 'real-cls-depth',
        'real-cls-readout', 'real-cls-prediction']]
    third = [all_frames['vision-topic-03'], location, all_frames['real-patch-position'], detour,
             cls_story['real-cls-purpose'], cls_story['cls-parameter-origin'],
             cls_story['cls-parameter-learning'], cls_story['cls-stored-start'], cls_story['cls-collect'],
             without_route(all_frames['cls-shared-start']), without_route(all_frames['cls-without']),
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
