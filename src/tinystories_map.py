"""A held, editable end-to-end map for every TinyStories teaching step.

Node positions never depend on the highlighted operation. The four head lanes
are parallel, and the target and generation paths stay separate.
"""
from html import escape
from multihead_story import COLORS


def pipeline_svg(focus, uid, *, toy=False):
    active = set(focus)
    T, C = ('4', '10') if toy else ('64', '4,000')
    nodes = []

    def node(key, x, y, w, title, detail='', role='e', h=52):
        nodes.append((key, x, y, w, h, title, detail, role))

    for i, (key, title, detail) in enumerate([
        ('stories', 'Stories', 'whole documents'), ('split', 'Split', 'train / val / test'),
        ('tokens', 'Tokenize', 'words, punctuation'), ('vocab', 'Train vocab', f'C = {C} IDs'),
        ('boundaries', 'BOS / EOS', 'one pair per story'), ('windows', 'Windows', f'last {T} IDs + y'),
        ('batch', 'Batch X', f'[B, {T}]'),
    ]):
        node(key, 14+i*164, 17, 146, title, detail)
    node('lookup', 14, 157, 169, 'Token lookup', f'[B, {T}, 64]')
    node('position', 14, 251, 169, '+ position', f'{T} rows, width 64', 'd')
    for h in range(4):
        y = 151 + 43*h
        node(f'qkv{h}', 331, y, 141, 'Q, K, V', '', 'q', 32)
        node(f'scores{h}', 490, y, 119, 'QKᵀ / √16', '', 'k', 32)
        node(f'weights{h}', 629, y, 112, 'Mask + A', '', 'a', 32)
        node(f'mix{h}', 761, y, 92, 'AV', '', 'v', 32)
    node('join', 876, 211, 109, 'Join', '4 × 16 = 64', 'v', 56)
    node('project', 1009, 157, 137, 'W_O', '[B, T, 64]', 'd')
    node('residual', 1009, 278, 137, '+ original E', '[B, T, 64]', 'd')
    node('predict', 894, 350, 252, 'Final row + classifier', '[B, 64] → [B, C]', 'e')
    node('loss', 625, 350, 233, 'Loss against y', 'cross-entropy', 'a')
    node('train', 346, 350, 241, 'Backward + update', 'repeat with next batch', 'q')
    node('decode', 625, 413, 233, 'Choose next token', 'greedy or sample', 'a')
    node('append', 346, 413, 241, 'Append / stop at EOS', 'parameters stay fixed', 'd')
    keys = {n[0] for n in nodes}
    if active - keys:
        raise ValueError(f'Unknown pipeline focus: {active - keys}')
    prefix = 'ts-' + uid
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 475" role="img" aria-label="Complete TinyStories pipeline; highlighted: {escape(", ".join(focus))}" data-pipeline="tinystories" data-focus="{escape(" ".join(focus))}" style="font-family:Avenir Next,Segoe UI,Arial,sans-serif">',
           '<title>Complete TinyStories pipeline with four parallel attention heads</title>',
           f'<defs><marker id="{prefix}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#6b7280"/></marker></defs>']

    def label(x, y, text, size=20, anchor='start', color='#50586a', weight=500):
        return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{escape(text)}</text>'

    def edge(a, b, d, dashed=False):
        out.append(f'<path data-edge="{a}:{b}" d="{d}" fill="none" stroke="#6b7280" stroke-width="1.8"'+(' stroke-dasharray="6 5"' if dashed else '')+f' marker-end="url(#{prefix}-arrow)"/>')

    for i in range(6):
        edge(nodes[i][0], nodes[i+1][0], f'M{160+i*164} 43H{178+i*164}')
    # Batch X enters lookup. Observed y takes its own route to the loss.
    edge('batch', 'lookup', 'M1071 69V89H98V157')
    out.append(label(220, 83, 'X: input IDs', 18))
    edge('windows', 'loss', 'M907 69V78H1159V331H741V350', True)
    out.append(label(835, 347, 'observed y', 18, 'middle'))
    edge('lookup', 'E', 'M183 183H220V223')
    edge('position', 'E', 'M183 277H220V249')
    out.append('<circle cx="220" cy="236" r="13" fill="white" stroke="#50586a"/>')
    out.append(label(220, 243, '+', 23, 'middle'))
    out.append(label(248, 224, 'E', 22, color=COLORS['e'], weight=650))
    edge('E', 'qkv', 'M233 236H308')
    out.append('<path d="M308 167V296" fill="none" stroke="#6b7280" stroke-width="1.8"/>')
    out.append(label(578, 119, 'Four parallel heads. Each learns its own Q, K and V.', 22, 'middle'))
    for x, shape in [(401, f'[B, {T}, 16]'), (549, f'[B, {T}, {T}]'),
                     (685, f'[B, {T}, {T}]'), (807, f'[B, {T}, 16]')]:
        out.append(label(x, 143, shape, 18, 'middle'))
    for h in range(4):
        y = 167 + h*43
        out.append(label(292, y+6, str(h+1), 21, 'middle'))
        edge('E', f'qkv{h}', f'M308 {y}H331')
        edge(f'qkv{h}', f'scores{h}', f'M472 {y}H490')
        edge(f'scores{h}', f'weights{h}', f'M609 {y}H629')
        edge(f'weights{h}', f'mix{h}', f'M741 {y}H761')
        # The value route is distinct from the score/softmax route.
        edge(f'qkv{h}', f'mix{h}', f'M451 {y+16}V{y+21}H807V{y+16}', True)
        edge(f'mix{h}', 'join', f'M853 {y}H866V239H876')
    edge('join', 'project', 'M985 239H997V183H1009')
    edge('project', 'residual', 'M1077 209V278')
    edge('E', 'residual', 'M220 223V101H1153V304H1146', True)
    out.append(label(972, 119, 'keep E', 18, 'middle'))
    edge('residual', 'predict', 'M1077 330V350')
    edge('predict', 'loss', 'M894 376H858')
    edge('loss', 'train', 'M625 376H587')
    out.append(label(301, 382, 'Training', 21, 'end'))
    edge('predict', 'decode', 'M1020 402V439H858')
    edge('decode', 'append', 'M625 439H587')
    out.append(label(301, 427, 'Generation', 21, 'end'))
    out.append(label(897, 464, 'same trained model', 18))
    edge('append', 'windows', 'M346 439H7V6H907V17', True)
    # Dim only the nodes, retaining readable labels and a fixed complete graph.
    for key, x, y, w, h, title, detail, role in nodes:
        on = key in active
        color = COLORS[role] if on else '#77808f'
        out.append(f'<g data-stage="{key}" data-active="{str(on).lower()}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{COLORS[role]+"17" if on else "#f7f8fa"}" stroke="{color}" stroke-width="{3.5 if on else 1}"/>')
        out.append(label(x+w/2, y+21, title, 21 if w>120 else 19, 'middle', COLORS[role] if on else '#50586a', 700 if on else 500))
        if detail:
            out.append(label(x+w/2, y+h-8, detail, 16, 'middle', '#3d4656' if on else '#657080'))
        out.append('</g>')
    out.append('</svg>')
    return ''.join(out)


HEADS = lambda name: tuple(f'{name}{h}' for h in range(4))
SETUP_FOCUS = {
    **{k: ('stories',) for k in ('data', 'story-complete', 'story-excerpts', 'sentence')},
    'split': ('split',),
    **{k: ('tokens',) for k in ('tokenization-intro', 'tokenization-choices', 'tokenization-rules', 'tokenize')},
    'vocabulary': ('vocab',), 'special': ('vocab', 'boundaries'), 'boundaries': ('boundaries',),
    **{k: ('windows',) for k in ('story-indices', 'one-pair', 'pair-first', 'pair-second', 'windows-first', 'windows-last', 'counts-story', 'context')},
    'counts': ('split', 'windows'), 'pairs-tensors': ('windows', 'batch'),
    **{k: ('batch',) for k in ('batch', 'batch-ids', 'batches')},
    'scale': ('vocab', 'windows', 'batch'), 'real-windows': ('windows', 'batch'),
}
MODEL_FOCUS = {
    'model': ('lookup', 'position', *HEADS('qkv'), 'project', 'predict'),
    'lookup': ('lookup',), 'position': ('position',), 'embed': HEADS('qkv'),
    'scores': HEADS('scores'), 'weights': HEADS('weights'), 'mix': HEADS('mix'),
    'join': ('join', 'project', 'residual'), 'predict': ('predict', 'loss'),
    'train': ('train',), 'prompt': ('tokens', 'vocab', 'windows', 'batch'),
    'decode': ('predict', 'decode'), 'generate': ('append', 'windows'),
}

# Small executable excerpts, followed by the original data/example reveal.
SETUP_CODE = {
    'data': "sample = evidence['sample']\nprint(sample['text_excerpt'])",
    'story-complete': "short_story = story_examples['examples'][0]\nprint(short_story['text'])",
    'story-excerpts': "long_stories = story_examples['examples'][1:]\nprint([s['token_count'] for s in long_stories])",
    'split': "audit = evidence['audit']\nprint(audit['documents'])  # whole-story splits",
    'sentence': "sentence = 'Lily found a red ball.'",
    'tokenization-intro': "tokenization_text = 'redder!'\nprint(tokenize(tokenization_text))",
    'tokenization-choices': "word_tokens = tokenize('redder!')\ncharacter_tokens = list('redder!')\nillustrative_subwords = ['red', 'der', '!']  # hand-chosen",
    'tokenization-rules': "print(tokenize(\"Lily can't find 12 balls!\"))",
    'tokenize': 'pieces = tokenize(sentence)',
    'vocabulary': "words = list(SPECIAL_TOKENS) + sorted(set(pieces))\nvocab = Vocabulary(words, {t:i for i,t in enumerate(words)}, {}, 10, 1)",
    'special': "print(vocab.encode_tokens(['blue'], boundaries=False))  # UNK: [3]\nprint(vocab.encode_tokens(['blue'], boundaries=True))   # [1, 3, 2]",
    'boundaries': 'ids = vocab.encode_tokens(pieces, boundaries=True)',
    'story-indices': 'w = 4\ntarget_position = 4\nprint(list(enumerate(ids)))',
    'one-pair': 'context_ids = ids[target_position-w:target_position]\ntarget_id = ids[target_position]',
    'pair-first': 't = 1\nvisible = ids[max(0, t-w):t]\ncontext_ids, target_id = [0] * (w-len(visible)) + visible, ids[t]',
    'pair-second': 't = 2\nvisible = ids[max(0, t-w):t]\ncontext_ids, target_id = [0] * (w-len(visible)) + visible, ids[t]',
    'windows-first': 'contexts, targets = [], []\nfor t in range(1, len(ids)):\n    visible = ids[max(0, t-w):t]\n    contexts.append([0] * (w-len(visible)) + visible); targets.append(ids[t])',
    'windows-last': "print(contexts[4:], targets[4:])  # older tokens fall outside the window",
    'pairs-tensors': 'all_X = torch.tensor(contexts, dtype=torch.long)\nall_y = torch.tensor(targets, dtype=torch.long)',
    'counts-story': 'examples_in_story = len(pieces) + 1  # one extra target: EOS',
    'counts': "window_counts = {s: audit['oov'][s]['tokens'] + n\n                 for s, n in audit['documents'].items()}",
    'context': 'history = ids[:6]\ncontexts_by_width = {w: history[-w:] for w in [2, 4, 6]}',
    'batch': 'selected = torch.tensor([2, 3])\nX, y = all_X[selected], all_y[selected]',
    'batch-ids': 'print(X.shape, X.dtype)  # two rows of four IDs\nprint(y.shape, y.dtype)  # two observed next-token IDs',
    'batches': 'B, N = 2, len(all_y)\nbatch_sizes = [len(all_y[start:start+B]) for start in range(0, N, B)]',
    'scale': 'C, T, D, heads = 4000, 64, 64, 4\ntraining_batch_size = 512',
    'real-windows': "prefixes = [['<BOS>', 'lily', 'found'], ['<BOS>', 'lily', 'found', 'a']]\nX = torch.tensor([[0]*(T-len(p)) + [real_stoi[t] for t in p] for p in prefixes])\ny = torch.tensor([real_stoi['a'], real_stoi['red']])",
}


def setup_checkpoint(stage, key, title):
    uid = 's03-map-' + key
    toy = key not in {'scale', 'real-windows'}
    return stage(uid, title, pipeline_svg(SETUP_FOCUS[key], uid, toy=toy),
                 code=SETUP_CODE[key],
                 notes='The layout stays fixed. Trace only the outlined operation, then reveal the original worked example. T is the input-window width; C is vocabulary size. The model below is the four-head architecture we will use after the setup.',
                 companion='<p>The coloured, thick outline marks the current operation. The complete map remains visible. The top row prepares data. Four parallel head lanes form the model. The bottom branches separate learning from generation. Dashed V routes bypass score calculation. The dashed E route carries the residual.</p><p>The small data example uses C=10 and T=4. We switch to C=4,000 and T=64 before running the model. The diagram previews its width-64, four-head architecture. The original data example follows on the next frame.</p>')
