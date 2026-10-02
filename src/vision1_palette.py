"""Lecture-series colors. Q/K/V use a labelled, local role palette."""
PALETTE = {
    'ink': '#30343B', 'ink-2': '#59616D', 'ink-3': '#6B7280',
    'line': '#D9DDE5', 'card': '#FFFFFF', 'transparent': 'transparent',
    'vision': '#3478E5', 'language': '#8B5BB5', 'mixing': '#178F82',
    'special': '#B98224', 'loss': '#D45555', 'neutral': '#30343B',
    'cnn': '#6E817B', 't-special': '#FBF1DB', 't-language': '#F1EAF7',
    't-mixing': '#E6F3EF', 't-e': '#EAF1FC', 't-q': '#F1EAF7',
    'c-e': '#3478E5', 'c-q': '#8B5BB5', 'c-k': '#B98224',
    'c-v': '#178F82', 'c-a': '#D45555', 'c-d': '#6E817B',
}
SVG_COLORS = ''.join(f'--{name}:{value};' for name, value in PALETTE.items())
CSS = ':root{' + SVG_COLORS + '--bg:#FAFAF8;--paper:#FAFAF8;}\n'
