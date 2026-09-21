/* Small offline lexer for the short Python snippets produced by live steppers.
   Static slides use Python's tokenize module at build time. Both share CSS.
   All source is inserted as text, never interpreted as markup. */
(function () {
  const keywords = new Set('and as assert async await break class continue def del elif else except False finally for from global if import in is lambda None nonlocal not or pass raise return True try while with yield'.split(' '));
  const builtins = new Set('bool dict enumerate float int len list max min print range set str sum tuple zip'.split(' '));
  AT.pythonCode = function (source) {
    const code = document.createElement('code');
    code.className = 'language-python python-code';
    // Strings/comments precede names and numbers so their contents stay intact.
    const tokens = /#[^\n]*|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|\b\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?\b|\b[A-Za-z_]\w*\b|[+*/%@=<>!&|^~:.,()[\]{}-]+/g;
    let cursor = 0;
    for (const match of source.matchAll(tokens)) {
      const text = match[0], start = match.index;
      code.append(document.createTextNode(source.slice(cursor, start)));
      const kind = text.startsWith('#') ? 'comment'
        : /^["']/.test(text) ? 'string'
        : /^\d/.test(text) ? 'number'
        : keywords.has(text) ? 'keyword'
        : builtins.has(text) || (/^[A-Za-z_]/.test(text) && /^\s*\(/.test(source.slice(start + text.length))) ? 'call'
        : /^[^A-Za-z_]/.test(text) ? 'operator' : null;
      const node = document.createElement('span');
      if (kind) node.className = 'py-' + kind;
      node.textContent = text;
      code.append(node);
      cursor = start + text.length;
    }
    code.append(document.createTextNode(source.slice(cursor)));
    return code;
  };
})();
