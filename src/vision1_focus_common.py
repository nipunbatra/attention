"""Drawing helpers for the compact continuation of the photo walkthrough."""
import base64
import builtins
import io
import keyword
import tokenize
from html import escape
from textwrap import dedent


class Figures:
    def __init__(self, b):
        self.b = b
        for key in ['t', 'g', 'rect', 'line', 'arrow', 'image', 'pixels']:
            setattr(self, key, b[key])
        self.photo = 'data:image/png;base64,' + base64.b64encode(
            (b['ASSETS'] / 'model-input.png').read_bytes()).decode()
        self.frames = {}

    def box(self, x, y, w, labels, color='c-e', h=76, size=24):
        labels = [labels] if isinstance(labels, str) else labels
        out = self.rect(x, y, w, h, color, {'c-q': 't-q', 'special': 't-special', 'vision': 't-e', 'language': 't-language', 'mixing': 't-mixing'}.get(color, 'card'), 5)
        for i, label in enumerate(labels):
            out += self.t(x+w/2, y+h/2+8+(i-(len(labels)-1)/2)*30,
                          label, size, color, 'middle')
        return out

    def code(self, source, x=35, y=55, width=690, size=22, spacing=34):
        source = dedent(source).strip()
        lines = source.splitlines()
        # SVG code needs its own static spans: HTML <pre> highlighting does not
        # reach it. Token positions preserve indentation and exported figures.
        colors = {'keyword': '#7135a5', 'string': '#14613d', 'number': '#9a410c',
                  'call': '#174f9e', 'comment': '#526174', 'operator': '#374151'}
        runs = [[] for _ in lines]
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        for i, token in enumerate(tokens):
            kind = {tokenize.STRING: 'string', tokenize.NUMBER: 'number',
                    tokenize.COMMENT: 'comment', tokenize.OP: 'operator'}.get(token.type)
            if token.type == tokenize.NAME:
                if keyword.iskeyword(token.string):
                    kind = 'keyword'
                elif token.string in vars(builtins) or (i+1 < len(tokens) and tokens[i+1].string == '('):
                    kind = 'call'
            if kind:
                for row in range(token.start[0], token.end[0]+1):
                    start = token.start[1] if row == token.start[0] else 0
                    end = token.end[1] if row == token.end[0] else len(lines[row-1])
                    runs[row-1].append((start, end, kind))
        out = self.rect(x-15, y-32, width, len(lines)*spacing+22, 'line', 'card', 5)
        for i, line in enumerate(lines):
            cursor, spans = 0, ''
            for start, end, kind in runs[i]:
                spans += escape(line[cursor:start])
                spans += f'<tspan fill="{colors[kind]}" class="py-{kind}">{escape(line[start:end])}</tspan>'
                cursor = end
            spans += escape(line[cursor:])
            out += (f'<text x="{x}" y="{y+i*spacing}" font-size="{size}" fill="#17202a" '
                    'xml:space="preserve" font-family="SFMono-Regular,Consolas,monospace">'+spans+'</text>')
        return out

    def add(self, key, title, body, caption, question, point, prose='', mobile='', *, height=440):
        notes = question + '\n' + point
        for item in self.b['FRAMES']:
            if item['id'] == key:
                item.update(title=title, caption=caption, notes=notes)
        if not mobile:
            mobile = '<p>'+escape(caption)+'</p><p>'+escape(point)+'</p>'
        markup = self.b['frame'](key, title, body, caption, notes, prose, mobile, height=height)
        self.frames[key] = markup
        return markup
