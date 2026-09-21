"""Offline Python highlighting that preserves the exact, copyable source text."""
import ast
import builtins
from html import escape
import io
import keyword
import tokenize


PYTHON_CSS = '''
code.python-code{color:#17202a}
code.python-code .py-keyword{color:#7135a5;font-weight:600}
code.python-code .py-string{color:#14613d}
code.python-code .py-number{color:#9a410c}
code.python-code .py-call{color:#174f9e}
code.python-code .py-comment{color:#526174;font-style:italic}
code.python-code .py-operator{color:#374151}
'''


def validate_python(source, filename='<slide>'):
    """Fail the build on truncated code, bare break/return, or ellipsis stand-ins."""
    tree = ast.parse(source, filename=filename)
    compile(tree, filename, 'exec')
    if any(isinstance(node, ast.Constant) and node.value is Ellipsis
           for node in ast.walk(tree)):
        raise ValueError(f'{filename}: replace omitted code with a complete small operation')


def highlight_python(source):
    """Tokenize rather than regex-replace, so comments/strings cannot inject HTML."""
    validate_python(source)
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    def offset(position):
        row, column = position
        return offsets[min(row - 1, len(offsets) - 1)] + column

    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    fragments, cursor = [], 0
    for i, token in enumerate(tokens):
        if token.type in {tokenize.ENDMARKER, tokenize.ENCODING, tokenize.INDENT,
                          tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL}:
            continue
        start, end = offset(token.start), offset(token.end)
        kind = {tokenize.STRING: 'string', tokenize.NUMBER: 'number',
                tokenize.COMMENT: 'comment', tokenize.OP: 'operator'}.get(token.type)
        if token.type == tokenize.NAME:
            if keyword.iskeyword(token.string):
                kind = 'keyword'
            elif (token.string in vars(builtins)
                  or (i + 1 < len(tokens) and tokens[i + 1].string == '(')):
                kind = 'call'
        fragments.append(escape(source[cursor:start]))
        text = escape(source[start:end])
        fragments.append(f'<span class="py-{kind}">{text}</span>' if kind else text)
        cursor = end
    fragments.append(escape(source[cursor:]))
    return '<code class="language-python python-code">' + ''.join(fragments) + '</code>'
