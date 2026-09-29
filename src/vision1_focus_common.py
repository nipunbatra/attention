"""Drawing helpers for the compact continuation of the photo walkthrough."""
import base64
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
        out = self.rect(x, y, w, h, color, 't-q' if color == 'c-q' else 'card', 5)
        for i, label in enumerate(labels):
            out += self.t(x+w/2, y+h/2+8+(i-(len(labels)-1)/2)*30,
                          label, size, color, 'middle')
        return out

    def code(self, source, x=35, y=55, width=690, size=22, spacing=34):
        lines = dedent(source).strip().splitlines()
        out = self.rect(x-15, y-32, width, len(lines)*spacing+22, 'line', 'card', 5)
        for i, line in enumerate(lines):
            out += self.t(x, y+i*spacing, line, size, 'ink').replace(
                '<text ', '<text xml:space="preserve" font-family="SFMono-Regular,Consolas,monospace" ', 1)
        return out

    def add(self, key, title, body, caption, question, point, prose='', mobile=''):
        notes = question + '\n' + point
        for item in self.b['FRAMES']:
            if item['id'] == key:
                item.update(title=title, caption=caption, notes=notes)
        if not mobile:
            mobile = '<p>'+escape(caption)+'</p><p>'+escape(point)+'</p>'
        markup = self.b['frame'](key, title, body, caption, notes, prose, mobile)
        self.frames[key] = markup
        return markup
