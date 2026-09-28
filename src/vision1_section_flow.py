"""Visible topic openings and a photograph-first explanation of patch position."""
import re
from html import escape
from textwrap import wrap


# Each opener connects something students have seen to a concrete next question.
TOPICS = [
    ('The image classification task',
     'In text, a prefix led to a prediction.',
     'What should a model predict from a photograph?',
     ('Labeled photos', 'The image task', 'Useful clues'),
     'Start with the dataset, choose an image label, then ask which parts of the photograph help us decide.'),
    ('From pixels to patch embeddings',
     'Attention updates a row of numbers for each token.',
     'How do we make those rows from an image?',
     ('RGB pixels', 'Shared linear layer', 'Patch embeddings'),
     'Read one small patch, calculate its embedding, then scale the same operation to a real photograph.'),
    ('From patch rows to an image prediction',
     'We followed one real patch into the first attention head.',
     'How does the whole photograph get a label?',
     ('Add a summary row', 'Attention and MLP', 'Predict the label'),
     'Continue with the same photograph and checkpoint. Follow all 197 rows through the blocks, then read the image summary.'),
    ('Calculate attention with four patches',
     'We have seen the complete photograph-to-prediction path.',
     'Can we calculate an attention message ourselves?',
     ('A small image', 'Five input rows', 'One message'),
     'Switch explicitly to a four-patch worksheet with chosen weights. Work out the operations that the real model performed at a larger scale.'),
    ('More than one attention head',
     'One head has produced a weighted message.',
     'What could a second head tell us?',
     ('Another projection', 'Another message', 'Join the messages'),
     'Keep the same input rows. Change the head’s learned projections and follow the second calculation.'),
    ('From messages to an image label',
     'Each head has sent its message.',
     'How do those messages become a prediction?',
     ('Update the rows', 'Score the classes', 'Learn from the label'),
     'Follow the summary row through the classifier and loss. Then compare it with averaging the patch rows.'),
    ('The complete Transformer block',
     'Our worksheet made attention small enough to calculate.',
     'What else belongs in the model we train?',
     ('Normalize', 'Attention + MLP', 'Repeat the block'),
     'Put attention back inside the full block and follow what each operation changes.'),
    ('Build the model in PyTorch',
     'We can now trace the full block on paper.',
     'Can our code follow the same computation?',
     ('Patch layer', 'Attention block', 'Training step'),
     'Match code to the drawings. Track the batch, patch and feature dimensions at every step.'),
    ('Check what the model learned',
     'The training code can change the model’s parameters.',
     'Does it work on images it did not train on?',
     ('New noisy images', 'Learning curves', 'Remove positions'),
     'Use separate training, validation and test images. Compare models with and without position information.'),
    ('Return to the real photographs',
     'The small experiment let us inspect learning.',
     'What does a pretrained ViT predict here?',
     ('Prepare the photo', 'Run the model', 'Read class scores'),
     'Return to the dog and cat with a pretrained image classifier. Read actual outputs from the saved experiment.'),
    ('Look inside the trained model',
     'We have seen the model’s image predictions.',
     'Which image regions does it use?',
     ('Inspect heads', 'Compare layers', 'Cover a region'),
     'Read attention maps, then cover parts of the photograph and measure how the prediction changes.'),
    ('The cost of smaller patches',
     'Attention compares every row with every allowed source.',
     'What happens when we use a finer patch grid?',
     ('Count patches', 'Count comparisons', 'Compare the cost'),
     'Halve the patch width and height. Predict the change in attention work before calculating it.'),
    ('Your turn to work it out',
     'We have followed the computation from pixels to a label.',
     'Can you carry it through a new example?',
     ('One message', 'Tensor shapes', 'Position reasoning'),
     'Pause at each question. Write an answer before revealing the calculation or explanation.'),
    ('What can we build next?',
     'A visual row can now carry information from its image.',
     'How could these rows help with other tasks?',
     ('Visual features', 'Image and text', 'New outputs'),
     'Connect this classifier to the next vision lectures, then explain the complete photograph-to-answer path.'),
]


def introduce(b, sections):
    t, g, rect, arrow, line, image, pixels, frame = (b[k] for k in
        ['t', 'g', 'rect', 'arrow', 'line', 'image', 'pixels', 'frame'])
    old_meta = {f['id']: f for f in b['FRAMES']}
    new = {}

    def add(key, title, body, caption, question, point, prose='', mobile=''):
        notes = question + '\n' + point
        new[key] = frame(key, title, body, caption, notes, prose, mobile)
        if key in old_meta:
            old_meta[key].update(title=title, caption=caption, notes=notes)
        return new[key]

    def photo_def(uid):
        # One embedded image per SVG; cropped <use> elements reuse its pixels.
        return '<defs><g id="'+uid+'">'+image(0, 0, 500, 334)+'</g></defs>'

    def piece(uid, index, x, y, width, height):
        col, row = index % 4, index // 4
        return (f'<svg x="{x}" y="{y}" width="{width}" height="{height}" '
                f'viewBox="{col*125} {row*83.5} 125 83.5" overflow="hidden">'
                f'<use href="#{uid}"/></svg>')

    def symbol(x, y, letter, subscript, size=35, color='c-e'):
        return (f'<text x="{x}" y="{y}" font-size="{size}" fill="var(--{color})" '
                f'text-anchor="middle" font-weight="500">{escape(letter)}'
                f'<tspan baseline-shift="sub" font-size="{size*.65}">{escape(subscript)}</tspan></text>')

    def mosaic(uid, x, y, width, order):
        out = ''
        w, h = width / 4, width * 334 / 500 / 4
        for slot, source in enumerate(order):
            px, py = x + (slot % 4) * w, y + (slot // 4) * h
            out += piece(uid, source, px, py, w, h)
        for i in range(1, 4):
            out += line(x+i*w, y, x+i*w, y+4*h, 'card', 1.5)
            out += line(x, y+i*h, x+4*w, y+i*h, 'card', 1.5)
        return out

    original = list(range(16))
    moved = original.copy()
    for a, c in [(5, 13), (6, 14)]:
        moved[a], moved[c] = moved[c], moved[a]
    assert sorted(moved) == original

    def photo_pair(uid, x1=35, x2=675, y=80, width=450):
        w, h = width/4, width*334/500/4
        body = photo_def(uid)
        body += mosaic(uid, x1, y, width, original)
        body += mosaic(uid, x2, y, width, moved)
        body += rect(x1+w, y+h, 2*w, h, 'c-q', 'transparent', 0)
        body += rect(x2+w, y+3*h, 2*w, h, 'c-q', 'transparent', 0)
        return body

    body = photo_pair('position-layout-photo')
    body += t(260, 44, 'Original photograph', 29, 'ink', 'middle')
    body += t(900, 44, 'Same patches, moved', 29, 'ink', 'middle')
    # Connect the actual outlined regions, with a white underlay over the photo.
    route = 'M372.5 192.725 H580 V343.025 H787.5'
    connector = f'<path d="{route}" fill="none" stroke="white" stroke-width="7"/>'
    connector += f'<path d="{route}" fill="none" stroke="var(--c-q)" stroke-width="3"/>'
    connector += arrow(754, 343.025, 787.5, 343.025, 'c-q')
    body += g(connector, 1)
    body += t(260, 424, 'Face patches in row 2', 26, 'c-q', 'middle')
    body += t(900, 424, 'Face patches in row 4', 26, 'c-q', 'middle')
    mobile = '<div class="vp-position-mobile">'
    for uid, order, label, row in [('position-original-mobile', original, 'Original: face in row 2', 1),
                                  ('position-moved-mobile', moved, 'Rearranged: face in row 4', 3)]:
        mobile += '<p>'+label+'</p><svg viewBox="0 0 500 334" role="img" aria-label="'+label+'">'
        mobile += photo_def(uid)+mosaic(uid, 0, 0, 500, order)+rect(125, row*83.5, 250, 83.5, 'c-q', 'transparent', 0)+'</svg>'
    mobile += '</div>'
    add('position-photo-layout', 'Move the face patches. What changes?', body,
        'We kept every patch and every pixel inside it. The face is now lower in the picture. Which information describes this change?',
        'Are these the same pieces? Are they arranged the same way?',
        'Compare both photos before revealing the arrow. Trace the two outlined face patches from row 2 to row 4; the displaced patches move back into row 2.',
        'This is a rearrangement of the same 16 non-overlapping crops from the opening dog photograph. We exchange two face patches with two patches in the bottom row. '
        'Nothing rotates and no pixel within a patch changes. The task in this thought experiment is to describe the changed layout. '
        'We do not claim that the animal label must change or show a model prediction on the rearranged photograph. The grid is enlarged for teaching; it is not the real model’s 14×14 patch grid.', mobile)

    uid = 'position-content-photo'
    body = photo_def(uid)
    for i, (label, slot) in enumerate([('Before the move', 'row 2, column 2'), ('After the move', 'row 4, column 2')]):
        y = 40+i*212
        body += t(40, y, label+' · '+slot, 27, 'ink')
        body += piece(uid, 5, 40, y+23, 180, 120.24)
        body += g(arrow(235, y+83, 405, y+83)+rect(425, y+47, 290, 73, 'c-e', 'transparent')
                  +t(570, y+93, 'same W and b', 28, 'c-e', 'middle'), 1)
        body += g(arrow(730, y+83, 900, y+83)+symbol(1000, y+93, 'c', 'face'), 2)
    body += g(t(1000, 43, 'content vector', 25, 'c-e', 'middle'), 2)
    add('position-photo-content', 'Does the patch layer notice the move?', body,
        'The same pixels pass through the same linear layer, so they produce the same content embedding. The layer has not been given the patch’s location.',
        'What changed in the inputs to the shared linear layer?',
        'Point to the identical crops, then the identical parameters. Read c_face as the content embedding of this one face crop.',
        'c_face is a name for the vector produced by this crop, not a scalar, a class score or a patch ID. '
        'Moving the intact crop leaves its flattened pixel vector unchanged. The shared affine map therefore returns the same vector. '
        'Visual content may suggest a typical location, but this patch layer receives no explicit index identifying its current slot. '
        'The numerical patch-layer calculation in the previous section explains exactly how this vector is computed.',
        '<p>The identical face crop moves from row 2, column 2 to row 4, column 2.</p>'
        +'<p>Same pixels → same W and b → same content vector <strong>c_face</strong>.</p>'
        +'<p>Its new location has not entered that calculation.</p>')

    uid = 'position-add-photo'
    body = photo_def(uid)+piece(uid, 5, 35, 117, 180, 120.24)
    body += t(125, 288, 'same crop', 27, 'ink', 'middle')
    body += t(355, 56, 'CONTENT', 23, 'c-e', 'middle')
    body += t(680, 56, 'POSITION', 23, 'c-q', 'middle')
    body += t(1015, 56, 'INPUT ROW', 23, 'ink', 'middle')
    for i, (pos, label) in enumerate([('2,2', 'row 2, column 2'), ('4,2', 'row 4, column 2')]):
        y = 135+i*145
        body += symbol(355, y, 'c', 'face', 36)
        body += g(t(515, y, '+', 35)+symbol(680, y, 'p', pos, 39, 'c-q')
                  +t(680, y+46, label, 23, 'c-q', 'middle'), 1)
        body += g(arrow(800, y-10, 900, y-10)+symbol(1015, y, 'e', 'before' if i==0 else 'after'), 2)
    body += g(t(335, 403, 'Content and position are both D-dimensional vectors.', 27, 'ink-2'), 2)
    add('position-photo-add', 'Give the row its location as well as its content', body,
        'Each grid slot has a position vector. Add it to the content embedding before attention, just as we added position to the text embeddings.',
        'Which part stays the same when the face crop moves? Which part should change?',
        'Keep pointing to c_face. Then change the position vector from the upper slot to the lower one; reveal the resulting input rows.',
        'Here p₂,₂ means the position vector for row 2, column 2, not a vector with just two coordinates. '
        'The row and column labels describe the diagram. The original ViT can store one learned D-dimensional vector per flattened grid slot. '
        'Content and position have the same width so they can be added coordinate by coordinate. '
        'This supplies location to attention; it does not promise correct classification. '
        'Moving image content between fixed position slots changes the content–position pairings. Reordering whole rows after addition keeps these pairings intact and is a different operation. '
        'See section 26.11 in <a href="https://visionbook.mit.edu/transformers.html">MIT’s Transformer chapter</a>.',
        '<p>Before: <strong>c_face + p₂,₂ = e_before</strong>.</p>'
        +'<p>After: <strong>c_face + p₄,₂ = e_after</strong>.</p>'
        +'<p>The same content is paired with its current location. Each vector has D coordinates; p₂,₂ names a slot, not two feature values.</p>')

    body = t(580, 40, '4 × 4 pixels · four 2 × 2 patches · two classes', 29, 'ink-2', 'middle')
    for x, name in [(100, 'horizontal'), (740, 'vertical')]:
        body += pixels(x, 92, b['DATA']['images'][name], 57, False, True)
    body += g(t(214, 378, '“Across the top”', 30, 'c-e', 'middle')
              +t(854, 378, '“Down the left”', 30, 'c-e', 'middle'), 1)
    worksheet_mobile = '<div class="vp-position-mobile">'
    for name, label in [('horizontal', 'Across the top'), ('vertical', 'Down the left')]:
        worksheet_mobile += '<p>'+label+'</p><svg viewBox="0 0 290 290" role="img" aria-label="'+label+'">'
        worksheet_mobile += pixels(15, 15, b['DATA']['images'][name], 65, False, True)+'</svg>'
    worksheet_mobile += '</div>'
    add('s02-small', 'A smaller task: classify the arrangement', body,
        'Now switch to a tiny grayscale image so we can calculate every step. Its label describes the arrangement of the filled patches.',
        'What should the labels be for these two arrangements?',
        'State the new worksheet task before revealing either label. Count four patches in each image: two filled and two empty.',
        'The photograph motivated why location matters. We now change to a deliberately small classification problem: distinguish filled patches across the top from filled patches down the left. '
        'This is a new two-class worksheet, not a lower-resolution dog classifier. Both images contain the same two filled and two empty patches. '
        'Pixels are 1 for filled and 0 for empty. We choose small parameters so students can compute the patch rows, attention messages and class scores. '
        'The worksheet uses attention plus a residual; normalization and the block MLP enter in the later full-model section.',
        worksheet_mobile+'<p><strong>Worksheet task:</strong> classify a 4×4 grayscale image as “across the top” or “down the left”.</p>'
        +'<p>There are four 2×2 patches. Each image contains two filled and two empty patches; their locations distinguish the labels.</p>')

    body = t(580, 40, 'Both images contain exactly these pieces', 30, 'ink', 'middle')
    for i, val in enumerate([1, 1, 0, 0]):
        x = 275+i*165
        body += pixels(x, 90, [[val, val], [val, val]], 48)
    body += g(arrow(580, 206, 580, 251)+t(580, 311, 'Two filled patches + two empty patches', 33, 'c-e', 'middle'), 1)
    body += g(t(580, 403, 'That count fits both labels. We need the arrangement.', 31, 'c-q', 'middle'), 2)
    add('position-question', 'Would just counting the patches solve it?', body,
        'The pieces tell us what is present. Their locations tell us how those pieces are arranged. Our worksheet label depends on that arrangement.',
        'Which label can you choose if I only tell you “two filled, two empty”?',
        'Count the pieces, then refer back to the two layouts. The same count is compatible with both answers.',
        'More precisely, shared patch operations and unrestricted self-attention without any positional signal are permutation-equivariant: permuting patch rows permutes their output rows. '
        'An image readout using a fixed CLS row or a mean of the patch rows is invariant to this patch permutation. '
        'Such a classifier cannot distinguish our two layouts solely from their identical set of content vectors. This argument assumes no other source of location information. '
        'We will return to this claim in the measured training control. '
        'See section 26.8 in <a href="https://visionbook.mit.edu/transformers.html">MIT’s Transformer chapter</a>.',
        '<p>Both layouts: filled, filled, empty, empty.</p><p>Counting the patch types cannot choose between the two labels. We need their arrangement.</p>')

    result = []
    for i, (_, frames) in enumerate(sections, 1):
        topic, previous, question, steps, caption = TOPICS[i-1]
        title = f'Section {i} · {topic}'
        body = t(48, 58, 'SECTION', 23, 'c-e', weight=600)
        body += t(48, 177, f'{i:02}', 110, 'c-e', weight=600)
        body += line(220, 35, 220, 405, 'c-e', 2)
        for j, phrase in enumerate(wrap(topic, 34)):
            body += t(264, 78+j*62, phrase, 46, 'ink', weight=700)
        for j, phrase in enumerate(wrap(previous, 61)):
            body += t(264, 217+j*34, phrase, 26, 'ink-2')
        for j, phrase in enumerate(wrap(question, 51)):
            body += t(264, 309+j*36, phrase, 30, 'ink', weight=600)
        for j, label in enumerate(steps):
            x = 264+j*290
            body += t(x, 414, label, 22, 'c-e')
            if j < 2:
                body += arrow(x+245, 406, x+278, 406, 'c-e')
        opener = add(f'vision-topic-{i:02}', title, body, caption, question,
                     'Pause at the section question. Connect the previous result to the three steps, then advance to the concrete example.',
                     '', '<p class="vp-section-label">Section '+str(i)+'</p><h3>'+escape(topic)+'</h3><p>'+escape(previous)+'</p><p><strong>'+escape(question)+'</strong></p>'
                     +'<p>'+' → '.join(escape(s) for s in steps)+'</p>')
        opener = opener.replace('class="frame vp-frame"', 'class="frame vp-frame vp-topic-break"', 1)
        ordered = [opener]
        if i == 3:
            ordered += [new[key] for key in ['position-photo-layout', 'position-photo-content', 'position-photo-add']]
        for markup in frames:
            key = re.search(r'class="frame[^"]*" id="([^"]+)"', markup).group(1)
            ordered.append(new.get(key, markup))
        result.append((topic, ordered))
    return result
