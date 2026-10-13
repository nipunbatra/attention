"""Opt-in, teacher-controlled builds; HTML reading and print show the full figure."""
import json
import re
from html import escape


def attrs(html, attributes):
    return re.sub(r'(<[A-Za-z][\w:-]*)', lambda m: m[1]+' '+attributes, html, count=1)


def step(html, number, until=None):
    return attrs(html, f'data-reveal-step="{number}"'+(f' data-reveal-until="{until}"' if until is not None else ''))


def sequence(html, titles):
    return attrs(html, 'data-reveal-titles="'+escape(json.dumps(titles), quote=True)+'"')


def group(html, number):
    return f'<g data-reveal-step="{number}">{html}</g>'
