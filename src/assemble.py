#!/usr/bin/env python3
"""Assemble one part of the series from shell.html + katex-bundle.html + shared.js + the part's sections + toy + config.
Usage:
  python3 assemble.py --part 2 --out attention.html            # Part 2: sections/, toy.json, part2.json
  python3 assemble.py --part 1 --out part1.html                # Part 1: sections1/, toy1.json, part1.json, runtime + diagrams (if present)
  python3 assemble.py --part 3 --out part3.html                # Part 3: sections3/, toy3.json, part3.json, runtime part3.js (if present)
  python3 assemble.py --only sections/sec07.html --out t.html  # any part: test one or more fragments (pass --part so the right toy/config is used)
Replaces <!--KATEX-->, <!--SHARED-->, <!--SECTIONS--> in shell.html and the <title>. Injects window.__TOY__ and window.__PART__
before shared.js, then the optional part runtime (partN.js) after it."""
import argparse, base64, glob, os, sys, json, re
from urllib.parse import urlsplit


def source_outputs(source_dir):
    """Find complete lesson targets from source, independently of build order.

    Source IDs 1–4 are the language sequence (2 uses attention.html); IDs 5–8
    are vision1.html–vision4.html. A future config can declare another output
    filename. ``published: false`` keeps a planned lesson unavailable even if
    its draft source is present.
    """
    outputs = set()
    for config_path in glob.glob(os.path.join(source_dir, 'part[0-9]*.json')):
        match = re.fullmatch(r'part(\d+)\.json', os.path.basename(config_path))
        if not match:
            continue
        number = int(match.group(1))
        with open(config_path, encoding='utf-8') as file:
            config = json.load(file)
        if config.get('published') is False:
            continue
        suffix = '' if number == 2 else str(number)
        sections = config.get('sections', [])
        section_dir = os.path.join(source_dir, 'sections' + suffix)
        complete = bool(sections) and os.path.isfile(os.path.join(source_dir, 'toy' + suffix + '.json'))
        complete = complete and all(
            re.fullmatch(r's\d\d', section.get('id', ''))
            and os.path.isfile(os.path.join(section_dir, 'sec' + section['id'][1:] + '.html'))
            for section in sections
        )
        if complete:
            default = 'attention.html' if number == 2 else (
                'vision%d.html' % (number - 4) if 5 <= number <= 8 else 'part%d.html' % number
            )
            outputs.add(config.get('output', default))
    if os.path.isfile(os.path.join(source_dir, '..', 'index.html')):
        outputs.add('index.html')
    return outputs


here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--part', type=int, default=2, help='part number (default 2)')
ap.add_argument('--only', nargs='*', default=None, help='section files to include (default: all of the part)')
ap.add_argument('--shell', default=os.path.join(here, 'shell.html'))
ap.add_argument('--shared', default=os.path.join(here, 'shared.js'))
ap.add_argument('--toy', default=None, help='toy JSON (default: toy.json for part 2, toyN.json otherwise)')
ap.add_argument('--config', default=None, help='part config JSON (default partN.json)')
ap.add_argument('--runtime', default=None, help='part runtime JS (default partN.js when it exists; none for part 2)')
ap.add_argument('--sections', default=None, help='sections directory (default sections/ for part 2, sectionsN/ otherwise)')
a = ap.parse_args()
N = a.part
suffix = '' if N == 2 else str(N)
toy_path = a.toy or os.path.join(here, 'toy%s.json' % suffix)
cfg_path = a.config or os.path.join(here, 'part%d.json' % N)
rt_path = a.runtime or os.path.join(here, 'part%d.js' % N)
sec_dir = a.sections or os.path.join(here, 'sections%s' % suffix)

shell = open(a.shell, encoding='utf-8').read()
for ph in ('<!--KATEX-->', '<!--SHARED-->', '<!--SECTIONS-->'):
    if ph not in shell:
        sys.exit('shell.html is missing placeholder ' + ph)
katex = open(os.path.join(here, 'katex-bundle.html'), encoding='utf-8').read()
if not os.path.exists(toy_path):
    sys.exit('toy file not found: ' + toy_path)
toy = json.load(open(toy_path, encoding='utf-8'))
if os.path.exists(cfg_path):
    part = json.load(open(cfg_path, encoding='utf-8'))
else:
    print('warning: %s not found, using a minimal part config' % os.path.basename(cfg_path))
    part = {'part': N, 'title': 'Part %d' % N, 'subtitle': '', 'chain': [], 'sections': [], 'objects': ['e', 'q', 'k', 'v', 'a', 'd', 'ep'], 'prev': None, 'next': None, 'notation': 'part%d' % N}
part.setdefault('part', N)
# The complete series is built in one pass into any directory. A sibling's
# absence from that output directory does not make a published lesson planned.
available_outputs = source_outputs(here)
for direction in ('prev', 'next', 'index'):
    link = part.get(direction)
    if link and link.get('href') and not re.match(r'^(?:https?:)?//', link['href']):
        target = os.path.normpath(urlsplit(link['href']).path)
        link['available'] = link.get('available') is not False and target in available_outputs
shared = open(a.shared, encoding='utf-8').read()
runtime = ''
if os.path.exists(rt_path):
    runtime = '<script>\n' + open(rt_path, encoding='utf-8').read() + '\n</script>\n'
def js(obj):
    return json.dumps(obj, separators=(',', ':')).replace('</', '<\\/')
vision_shared = ''
if 5 <= N <= 8:
    # The vision parts share one scene, one fixed patch encoder and one set of figures; it must load before the part runtime.
    with open(os.path.join(here, 'vision-shared.js'), encoding='utf-8') as module:
        vision_shared = '<script>\n' + module.read() + '\n</script>\n'
shared_block = ('<script>\nwindow.__TOY__ = ' + js(toy) + ';\nwindow.__PART__ = ' + js(part) + ';\n</script>\n'
                '<script>\n' + shared + '\n</script>\n' + vision_shared + runtime)
if 5 <= N <= 8:
    # Embed the recurring scene once per standalone lesson. Every SVG crop uses
    # this same data URI, so file://, presentation and PDF need no image server.
    scene_dir = os.path.join(here, '..', 'figures', 'vision-scene')
    scene_assets = {}
    for variant, filename in (('two', 'two-mugs.jpg'), ('one', 'one-mug.jpg')):
        with open(os.path.join(scene_dir, filename), 'rb') as asset:
            scene_assets[variant] = 'data:image/jpeg;base64,' + base64.b64encode(asset.read()).decode('ascii')
    shared_block += '<script>\nwindow.__VISION_SCENES__ = ' + js(scene_assets) + ';\n</script>\n'
    with open(os.path.join(here, 'vision-scene.js'), encoding='utf-8') as module:
        shared_block += '<script>\n' + module.read() + '\n</script>\n'
if N == 1:
    # Part I's diagrams read the same trained toy exposed by part1.js.
    diagrams = os.path.join(here, 'part1-diagrams.js')
    if os.path.isfile(diagrams):
        shared_block += '<script>\n' + open(diagrams, encoding='utf-8').read() + '\n</script>\n'
if N == 2:
    # One SVG source powers both the standalone preview and the article stepper.
    diagram = os.path.join(here, '..', 'figures', 'attention-diagram-preview', 'diagram.js')
    data_adapter = os.path.join(here, 'attention-flow-data.js')
    if os.path.isfile(diagram) and os.path.isfile(data_adapter):
        shared_block += ('<script>\n' + open(data_adapter, encoding='utf-8').read() + '\n</script>\n'
                         '<script>\n' + open(diagram, encoding='utf-8').read() + '\n</script>\n')
if a.only:
    files = a.only
else:
    files = sorted(glob.glob(os.path.join(sec_dir, 'sec[0-9][0-9].html')))
    if not files:
        sys.exit('no section files in ' + sec_dir)
parts = []
ids = []
for f in files:
    txt = open(f, encoding='utf-8').read()
    m = re.search(r'<section\s+id="(s\d+)"', txt)
    if m:
        ids.append(m.group(1))
    parts.append('<!-- ===== ' + os.path.basename(f) + ' ===== -->\n' + txt.strip() + '\n')
if not a.only:
    listed = [s.get('id') for s in part.get('sections', [])]
    missing = [i for i in listed if i not in ids]
    extra = [i for i in ids if i not in listed]
    if missing or extra:
        print('warning: part config sections differ from the files (missing on disk: %s; not in config: %s)' % (missing, extra))
out = shell.replace('<!--KATEX-->', katex).replace('<!--SHARED-->', shared_block).replace('<!--SECTIONS-->', '\n'.join(parts))
title = part.get('title') or ('Part %d' % N)
out = re.sub(r'<title>.*?</title>', '<title>' + title.replace('&', '&amp;').replace('<', '&lt;') + '</title>', out, count=1, flags=re.S)
open(a.out, 'w', encoding='utf-8').write(out)
print('wrote %s: %.0f KB, part %d, %d section file(s): %s%s' % (a.out, len(out) / 1024, N, len(files), [os.path.basename(f) for f in files], ' + ' + os.path.basename(rt_path) if runtime else ''))
