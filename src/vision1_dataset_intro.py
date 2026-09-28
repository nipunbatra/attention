"""Introduce the actual photo dataset before inspecting a single image."""
import base64
import hashlib
import json
from html import escape


def introduce(b, sections):
    frame, t, g, image, arrow, line = (b[k] for k in ['frame', 't', 'g', 'image', 'arrow', 'line'])
    assets = b['ASSETS']
    data = json.loads((assets / 'dataset-intro.json').read_text())
    assert data['images'] == sum(data['splits'].values()) == 7349
    assert len(data['breed_labels']) == 37 and len(data['species_labels']) == 2
    assert data['model_input_numbers'] == 224 * 224 * 3
    photos = {}
    for item in data['examples']:
        blob = (assets / item['file']).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == item['sha256']
        photos[item['image_id']] = 'data:image/jpeg;base64,' + base64.b64encode(blob).decode()
    source = ('<a href="https://www.robots.ox.ac.uk/~vgg/data/pets/">Oxford-IIIT Pet dataset</a>, '
              'Parkhi, Vedaldi, Zisserman and Jawahar (2012). '
              '<a href="' + data['metadata_source'] + '">Pinned mirror metadata</a>. '
              'Photos: CC BY-SA 4.0; original image copyrights retained. '
              '<a href="figures/vision1/dataset-intro.json">Example labels, dimensions and provenance</a>.')
    intro = []
    body = ''
    gallery_mobile = '<div class="vp-pet-gallery">'
    for i, item in enumerate(data['examples']):
        x = 45 + (i % 3) * 385
        # Keep each photo and its two labels together, then leave a distinct
        # gutter before the next row. The taller canvas preserves photo size.
        y = 8 + (i // 3) * 282
        body += image(x, y, 285, 132, photos[item['image_id']])
        body += t(x + 142.5, y + 178, item['species'], 28, 'c-e', 'middle')
        body += t(x + 142.5, y + 226, item['breed'], 22, 'ink-2', 'middle')
        gallery_mobile += ('<figure><img src="'
                           + photos[item['image_id']] + '" alt="' + escape(item['breed']) + '">'
                           '<figcaption><strong>' + item['species'] + '</strong>'
                           '<span>' + escape(item['breed']) + '</span></figcaption></figure>')
    gallery_mobile += '</div>'
    gallery = frame(
        'dataset-gallery', 'What does our animal dataset look like?', body,
        'Oxford-IIIT Pet: six examples. Each photo has a species label and a breed label.',
        'What changes across photos that share the same dog or cat label?\nPoint to pose, coat and background. Read the species label first, then the breed beneath it.',
        'A dataset pairs each image with its target labels. Here the first line under each photo is its species; the second is its breed. '
        'These are six selected examples from the test split, chosen to show variety. We use the photos to motivate classification. '
        'Later, our training experiment uses synthetic stripe images, and our pretrained demonstration predicts ImageNet categories. '
        'The lecture does not report training or accuracy on the full Pets dataset. ' + source,
        gallery_mobile)
    gallery = gallery.replace('viewBox="0 0 1160 440"', 'viewBox="0 0 1160 540"', 1)
    gallery_asset = assets / 'dataset-gallery.svg'
    gallery_asset.write_text(gallery_asset.read_text().replace('viewBox="0 0 1160 440"', 'viewBox="0 0 1160 540"', 1))
    intro.append(gallery)

    body = t(35,90,f"{data['images']:,}",62,'c-e') + t(35,145,'photographs',29,'ink-2')
    body += g(line(35,185,610,185,'line') + t(35,270,'3,680 train / validation',32,'c-e')
              + t(35,350,'3,669 test',32,'c-e'),1)
    body += g(t(730,90,'2 species',45,'c-e') + t(730,145,'cat or dog',30,'c-e'),2)
    body += g(t(730,275,'37 breeds',45,'c-k') + t(730,330,'Persian, Pug, …',28,'c-k'),3)
    intro.append(frame(
        'dataset-counts', 'How many images and classes are there?', body,
        'Our opening question uses 2 classes: cat or dog. Predicting the breed would use 37 classes.',
        'Could the same photograph have a two-class target in one task and a 37-class target in another?\nPoint back to the two labels under each photo. Add the two split sizes to recover the total.',
        'The labeled dataset has 7,349 images. The standard training/validation pool contains 3,680 images; the test split contains 3,669. '
        'The timm mirror calls the first pool train. Its label_cat_dog field has two species classes, while label has 37 breed classes. '
        'We begin with the species question to establish what the input and target mean. The size of the output layer follows the chosen target vocabulary. ' + source,
        b['mobile_rows'](['Dataset quantity','Count'],[
            ['Images','7,349'],['Train / validation','3,680'],['Test','3,669'],
            ['Species classes','2: cat and dog'],['Breed classes','37']])))

    dog = next(x for x in data['examples'] if x['image_id'] == 'newfoundland_31')
    processed = 'data:image/png;base64,' + base64.b64encode((assets / 'model-input.png').read_bytes()).decode()
    body = t(35,40,'Dimensions: height × width × channels',28,'ink-2')
    body += image(35,85,390,261,photos[dog['image_id']])
    body += t(230,404,f"{dog['height']} × {dog['width']} × {dog['channels']}",31,'c-e','middle')
    body += g(arrow(460,225,695,225,'c-e') + t(578,140,'resize +',28,'ink-2','middle')
              + t(578,180,'center-crop',28,'ink-2','middle'),1)
    body += g(image(745,85,275,275,processed) + t(882,404,'224 × 224 × 3',32,'c-e','middle'),2)
    intro.append(frame(
        'dataset-dimensions', 'What shape is one image?', body,
        'Original sizes vary. Our demo uses 224 × 224 pixels, each with red, green and blue values: 150,528 numbers.',
        'What does each of the three dimensions count?\nTrace the original photograph to the exact model crop. Multiply height by width by the three RGB channels.',
        'This Newfoundland file is 334 pixels high and 500 wide. The Persian example is 500 high and 375 wide. '
        'Both are RGB, so each pixel has three channel values. The diagram uses height × width × channels; '
        'PyTorch stores one preprocessed image as [3,224,224], with the same number of entries. '
        'For the supplied pretrained checkpoint, preprocessing preserves aspect ratio during resize, takes a 224×224 center crop, '
        'and then normalizes the channels. The right-hand image is the saved model-input crop, displayed before normalization. '
        '224×224×3=150,528 scalar inputs. These are pixels, not patch embeddings. ' + source,
        '<p>Height × width × channels</p><img src="' + photos[dog['image_id']]
        + '" alt="Original Newfoundland photograph"><p>Original: 334 × 500 × 3</p>'
        '<p>Resize and center-crop ↓</p><img src="' + processed
        + '" alt="The exact square model-input crop"><p>Model input: 224 × 224 × 3 = 150,528 numbers</p>'))
    revised = list(sections)
    title, frames = revised[0]
    revised[0] = (title, intro + frames)
    return revised
