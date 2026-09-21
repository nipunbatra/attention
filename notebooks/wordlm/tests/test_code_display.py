"""Visible code must remain complete, small, escaped and copyable."""
import ast
from html.parser import HTMLParser
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_slow_lesson import SLIDE_CODE, slide_code
from code_display import highlight_python, validate_python
from slow_walkthrough import STAGES, initial_namespace, render_figure


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def test_every_slide_makes_an_explicit_complete_code_choice():
    assert set(SLIDE_CODE) == {stage['id'] for stage in STAGES}
    for stage in STAGES:
        source = slide_code(stage)
        if source is not None:
            compile(source, stage['id'], 'exec')
            parser = TextOnly()
            parser.feed(highlight_python(source))
            assert ''.join(parser.parts) == source
            assert len(source.splitlines()) <= (6 if stage['id']=='pairs-loop' else 4)
    assert SLIDE_CODE['counts'] is None
    assert SLIDE_CODE['counts-train'].splitlines() == [
        'train_tokens = 964_338', 'train_stories = 4_822',
        'train_examples = train_tokens + train_stories']
    assert SLIDE_CODE['tokenize'] == 'pieces = tokenize(sentence)'
    assert SLIDE_CODE['mix'] == 'message = A @ V'


@pytest.mark.parametrize('source', [
    'counts = {\n    name: 7 for name in names',
    'for token in tokens:', 'break', 'return x', 'x = ...',
])
def test_incomplete_or_placeholder_code_fails_before_export(source):
    with pytest.raises((SyntaxError, ValueError)):
        validate_python(source)


def test_highlighter_preserves_whitespace_and_escapes_markup():
    source = 'name = "<script>&x</script>"\n# keep < and >\nif name:\n    print(1.5, name)\n'
    html = highlight_python(source)
    assert '<script>' not in html
    assert 'py-string' in html and 'py-keyword' in html and 'py-number' in html
    assert 'py-comment' in html and 'py-call' in html
    parser = TextOnly()
    parser.feed(html)
    assert ''.join(parser.parts) == source


def test_new_count_and_generation_steps_match_their_evidence():
    ns = initial_namespace()
    for stage in STAGES:
        exec(stage['code'], ns)
        validate_python(stage['code'], stage['id'])
        if stage['id']=='counts-story':
            assert ns['examples_in_story']==ns['N']==7
        elif stage['id']=='counts-train':
            assert ns['train_examples']==969160
        elif stage['id']=='prompt-window':
            assert ns['inference_X'].tolist()==[[0,1,8,7]]
        elif stage['id']=='generation-probabilities':
            assert ns['next_p'][ns['blocked_ids']].eq(0).all()
        elif stage['id']=='generation-append':
            before=ns['history_before_choice']
            expected=before if ns['chosen']==ns['vocab'].eos_id else before+[ns['chosen']]
            assert ns['history']==expected
        elif stage['id']=='append':
            assert ns['generation_trace'][0][0]==[0,1,8,7]
            assert all(ns['torch'].equal(p, ns['frozen'][n]) for n,p in ns['mlp'].named_parameters())
