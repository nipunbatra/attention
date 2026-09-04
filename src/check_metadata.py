#!/usr/bin/env python3
"""Check eight lesson configs, navigation names, and clean-directory builds.

Run ``python3 src/check_metadata.py``. Builds go into a temporary directory,
never the repository's published HTML. Uses only the Python standard library.
"""
from __future__ import annotations

import copy
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUTS = ['part1.html', 'attention.html', 'part3.html', 'part4.html',
           'vision1.html', 'vision2.html', 'vision3.html', 'vision4.html']
ERRORS = []


def check(condition, message):
    if not condition:
        ERRORS.append(message)


def text(markup):
    plain = html.unescape(re.sub(r'<[^>]+>', '', markup))
    # The one mathematical heading uses KaTeX in its h2 and plain text in the
    # navigation label. Compare the same expression, not the rendering syntax.
    plain = re.sub(r'\$\\sqrt\{([^{}]+)\}\$', r'√\1', plain)
    return ' '.join(plain.split())


class SectionParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sections = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'section' and 'sec' in attrs.get('class', '').split():
            self.sections.append(attrs)


def config_in(page):
    match = re.search(r'window\.__PART__ = (.+?);\n', page)
    if not match:
        raise ValueError('Assembled page is missing its part config')
    return json.loads(match[1])


def build(source, number, output, config=None):
    command = [sys.executable, str(source / 'assemble.py'), '--part', str(number), '--out', str(output)]
    if config:
        command.extend(['--config', str(config)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output.read_text(encoding='utf-8')


def main():
    configs = [json.loads((HERE / f'part{n}.json').read_text(encoding='utf-8')) for n in range(1, 9)]
    destinations = dict(zip(OUTPUTS, configs))
    index = (ROOT / 'index.html').read_text(encoding='utf-8')
    cards = dict((href, text(title)) for href, title in re.findall(
        r'<h2><a href="([^"]+)">(.*?)</a></h2>', index, flags=re.S))
    check(set(cards) == set(OUTPUTS), 'Landing-page cards must cover exactly the eight lessons')
    section_count = 0
    for number, (output, config) in enumerate(zip(OUTPUTS, configs), 1):
        name = f'part{number}.json'
        prefix = 'Part ' + str(number) if number <= 4 else 'Vision ' + ['I', 'II', 'III', 'IV'][number - 5]
        check(config.get('partLabel', 'Part ' + str(config.get('part'))) == prefix, f'{name}: displayed part label')
        check(config.get('series') == ('Attention and language' if number <= 4 else 'Vision to language'), f'{name}: series name')
        check(cards.get(output) == config['title'], f'{name}: landing-card title does not match destination h1')
        directory = HERE / ('sections' + ('' if number == 2 else str(number)))
        source_sections = []
        for path in sorted(directory.glob('sec[0-9][0-9].html')):
            markup = path.read_text(encoding='utf-8')
            parser = SectionParser()
            parser.feed(markup)
            check(len(parser.sections) == 1, f'{path.name}: expected one lesson section')
            if not parser.sections:
                continue
            section = parser.sections[0]
            source_sections.append(section)
            heading = re.search(r'<h2[^>]*>(.*?)</h2>', markup, flags=re.S)
            check(heading and text(heading[1]) == section.get('data-title'), f'{name}/{section.get("id")}: header differs from data-title')
        ids = [section.get('id') for section in source_sections]
        listed = config.get('sections', [])
        listed_ids = [section.get('id') for section in listed]
        check(len(set(ids)) == len(ids), f'{name}: duplicate source section IDs')
        check(listed_ids == ids, f'{name}: section list must match source IDs and order')
        for declared, source in zip(listed, source_sections):
            check(declared.get('title') == source.get('data-title'), f'{name}/{source.get("id")}: config section title differs from source')
        chain_ids = [item.get('section') for item in config.get('chain', [])]
        check(chain_ids == listed_ids, f'{name}: roadmap must visit each section once, in order')
        check(all(item.get('label', '').strip() for item in config.get('chain', [])), f'{name}: empty roadmap label')
        section_count += len(ids)
        check(config.get('index') == {'label': 'Series home', 'href': 'index.html'}, f'{name}: series-home link')
        for direction, expected in [('prev', number - 2), ('next', number)]:
            link = config.get(direction)
            if not 0 <= expected < len(OUTPUTS):
                check(link is None, f'{name}: unexpected {direction} link at sequence boundary')
                continue
            target = OUTPUTS[expected]
            check(bool(link) and urlsplit(link.get('href', '')).path == target, f'{name}: wrong {direction} destination')
            target_config = destinations[target]
            target_prefix = target_config.get('partLabel', 'Part ' + str(target_config['part']))
            check(bool(link) and link.get('label') == target_prefix + ': ' + target_config['title'], f'{name}: {direction} label differs from destination title')

    with tempfile.TemporaryDirectory(prefix='attention-metadata-') as temporary:
        work = Path(temporary)
        for number, output in enumerate(OUTPUTS, 1):
            directory = work / str(number)
            directory.mkdir()
            page = build(HERE, number, directory / output)
            part = config_in(page)
            check(set(p.name for p in directory.iterdir()) == {output}, 'Clean build unexpectedly needs or writes sibling pages')
            check('<title>' + html.escape(configs[number - 1]['title'], quote=False) + '</title>' in page,
                  f'{output}: document title differs from config')
            for direction in ('prev', 'next', 'index'):
                if part.get(direction):
                    check(part[direction].get('available') is True, f'{output}: {direction} unavailable in clean directory')

        # A stale output must not promote a future lesson to published status.
        planned = copy.deepcopy(configs[0])
        planned['next'] = {'label': 'Planned lesson', 'href': 'future.html'}
        planned_file = work / 'planned.json'
        planned_file.write_text(json.dumps(planned), encoding='utf-8')
        (work / 'future.html').write_text('old placeholder', encoding='utf-8')
        check(config_in(build(HERE, 1, work / 'planned.html', planned_file))['next']['available'] is False,
              'A stale file made an undeclared planned destination available')
        planned['next'] = dict(configs[0]['next'], available=False)
        planned_file.write_text(json.dumps(planned), encoding='utf-8')
        check(config_in(build(HERE, 1, work / 'disabled.html', planned_file))['next']['available'] is False,
              'An explicitly unavailable link became available')

        # Complete draft source can explicitly remain unpublished.
        fixture = work / 'draft-source'
        fixture.mkdir()
        for filename in ['assemble.py', 'shell.html', 'katex-bundle.html', 'shared.js',
                         'part1.json', 'part2.json', 'toy1.json', 'toy.json']:
            shutil.copy2(HERE / filename, fixture / filename)
        for dirname in ['sections1', 'sections']:
            shutil.copytree(HERE / dirname, fixture / dirname)
        draft = dict(configs[1], published=False)
        (fixture / 'part2.json').write_text(json.dumps(draft), encoding='utf-8')
        check(config_in(build(fixture, 1, work / 'draft.html'))['next']['available'] is False,
              'A complete but unpublished draft became available')

    if ERRORS:
        for message in ERRORS:
            print('FAIL: ' + message, file=sys.stderr)
        return 1
    print(f'PASS: eight configs, {section_count} sections, roadmap IDs, card/header/navigation names, '
          'eight isolated builds, and three planned-link cases. No published HTML changed.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        sys.exit(1)
