"""Fetch the six labeled Pets examples used to introduce the lecture dataset.

Run with Python + Pillow. Reuses the two original lecture photos. The public
viewer must still describe the pinned dataset revision; no credentials needed.
"""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import io
import json
import re
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'figures/vision1'
REVISION = '089695c834a7deb60505b7cc506672db1c31a6aa'
REPO = 'timm/oxford-iiit-pet'
ROWS_URL = ('https://datasets-server.huggingface.co/rows?'
            'dataset=timm%2Foxford-iiit-pet&config=default&split=test&offset=0&length=30')
CARD_URL = f'https://huggingface.co/datasets/{REPO}/raw/{REVISION}/README.md'
SELECTED = [
    ('newfoundland_31', 'Newfoundland', 'dog'),
    ('pug_57', 'Pug', 'dog'),
    ('great_pyrenees_22', 'Great Pyrenees', 'dog'),
    ('Persian_98', 'Persian', 'cat'),
    ('Sphynx_59', 'Sphynx', 'cat'),
    ('Birman_62', 'Birman', 'cat'),
]


def fetch(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def main():
    repo = json.loads(fetch(f'https://huggingface.co/api/datasets/{REPO}'))
    assert repo['sha'] == REVISION, 'Check the viewer against the pinned revision before updating.'
    card = fetch(CARD_URL).decode()
    splits = dict((name, int(count)) for name, count in re.findall(
        r'- name: (train|test)\s+num_bytes: [^\n]+\s+num_examples: (\d+)', card))
    assert splits == {'train': 3680, 'test': 3669}
    viewer = json.loads(fetch(ROWS_URL))
    labels = {f['name']: f['type']['names'] for f in viewer['features']
              if f['type'].get('_type') == 'ClassLabel'}
    assert len(labels['label']) == 37 and labels['label_cat_dog'] == ['cat', 'dog']
    assert viewer['num_rows_total'] == splits['test']
    rows = {r['row']['image_id']: r for r in viewer['rows']}
    (ASSETS / 'dataset').mkdir(exist_ok=True)
    examples = []
    for image_id, breed, species in SELECTED:
        record = rows[image_id]
        row = record['row']
        assert labels['label'][row['label']].replace('_', ' ').lower() == breed.lower()
        assert labels['label_cat_dog'][row['label_cat_dog']] == species
        existing = ASSETS / f'{image_id}.jpg'
        path = existing if existing.exists() else ASSETS / 'dataset' / f'{image_id}.jpg'
        blob = path.read_bytes() if path.exists() else fetch(row['image']['src'])
        im = Image.open(io.BytesIO(blob))
        assert im.size == (row['image']['width'], row['image']['height'])
        assert im.mode == 'RGB'
        path.write_bytes(blob)
        examples.append({
            'file': str(path.relative_to(ASSETS)), 'image_id': image_id,
            'breed': breed, 'species': species, 'breed_id': row['label'],
            'species_id': row['label_cat_dog'], 'split': 'test',
            'row': record['row_idx'], 'height': im.height, 'width': im.width,
            'channels': 3, 'sha256': hashlib.sha256(blob).hexdigest(),
        })
    report = {
        'dataset': REPO, 'revision': REVISION,
        'source': f'https://huggingface.co/datasets/{REPO}',
        'original_dataset': 'https://www.robots.ox.ac.uk/~vgg/data/pets/',
        'metadata_source': CARD_URL, 'viewer_rows_source': ROWS_URL,
        'images': sum(splits.values()), 'splits': splits,
        'species_labels': labels['label_cat_dog'], 'breed_labels': labels['label'],
        'original_sizes': 'Variable; example dimensions verified from image bytes.',
        'model_input_hwc': [224, 224, 3],
        'model_input_numbers': 224 * 224 * 3,
        'scope': 'Six dataset examples for motivation. No Pets training or benchmark result is claimed.',
        'license': 'CC BY-SA 4.0; original image copyrights retained.',
        'attribution': 'Oxford-IIIT Pet: Parkhi, Vedaldi, Zisserman and Jawahar (2012), via timm.',
        'examples': examples,
    }
    (ASSETS / 'dataset-intro.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Saved {len(examples)} labeled examples; {report['images']} dataset images, 37 breeds, 2 species.")


if __name__ == '__main__':
    main()
