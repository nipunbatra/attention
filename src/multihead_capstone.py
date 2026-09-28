"""Compact classroom route: head intuition, then the real TinyStories experiment.

The longer arithmetic story remains in part3-worked.html and Notebook 7.
All reported results and continuations below come from saved experiment files.
"""
from html import escape
import json
from pathlib import Path
import textwrap

from multihead_story import svg, t, rect, arrow, path, full_map, COLORS

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'mlp': 'Fixed-window MLP', 'attention': 'One-head attention',
          'multihead': 'Four-head attention'}


def read(name):
    return json.loads((ROOT / name).read_text())


def rows(headers, data, widths=None):
    widths = widths or [1120 / len(headers)] * len(headers)
    body = ''
    for i, row in enumerate([headers] + data):
        x, y = 20, 40 + i * 66
        for j, (value, width) in enumerate(zip(row, widths)):
            body += f'<g data-cell-left="{x}" data-cell-right="{x+width}">'
            body += t(x+10, y, value, 24 if i == 0 else 26,
                      'muted' if i == 0 else 'ink', weight=650 if i == 0 else 500) + '</g>'
            x += width
        body += path(f'M20 {y+17} H1140', 'line', 1)
    return svg(body, 80 + len(data)*66, ' · '.join(headers))


def flow(items, footer=''):
    """One left-to-right operation sequence; one label and shape per node."""
    width = (1120 - 38*(len(items)-1))/len(items)
    body = ''
    for i, (name, shape, color) in enumerate(items):
        x = 20+i*(width+38)
        body += '<g>' + rect(x, 40, width, 95, color)
        body += t(x+width/2, 77, name, 26, color, 'middle', 650)
        body += t(x+width/2, 113, shape, 24, 'muted', 'middle') + '</g>'
        if i < len(items)-1:
            body += arrow(x+width, 87, x+width+36, 87)
    if footer:
        body += t(580, 200, footer, 25, 'ink', 'middle')
    return svg(body, 230 if footer else 165, ' → '.join(i[0] for i in items))


def windows(vocab):
    body = t(20, 40, 'Teaching sentence: “Lily found a red ball.”', 30)
    examples = [(['<BOS>', 'lily', 'found'], 'a'),
                (['<BOS>', 'lily', 'found', 'a'], 'red')]
    evidence = []
    for n, (words, target) in enumerate(examples):
        y = 97 + n*115
        ids = [vocab.index(w) for w in words]
        body += t(20, y, f'Input {n+1}:', 24, 'muted')
        body += t(180, y, f'{64-len(words)} PADs  ·  '+ '  '.join(words), 27, 'e')
        body += t(180, y+37, 'IDs:  '+f'{64-len(words)} zeros  ·  '+str(ids), 23, 'muted')
        body += arrow(805, y-9, 861, y-9)
        body += t(885, y, target, 30, 'd', weight=650)
        body += t(885, y+37, f'target ID {vocab.index(target)}', 23, 'muted')
        evidence.append({'tokens': words, 'ids': [0]*(64-len(words))+ids,
                         'target': target, 'target_id': vocab.index(target)})
    body += t(580, 341, 'Two examples: X [2, 64] → predict the two targets y [2].', 27, 'ink', 'middle')
    return svg(body, 380, 'Two observed next-token targets, outside their input windows'), evidence


def dataset_figure(split):
    story = read('notebooks/wordlm/story_examples.json')['examples'][0]
    opening = '. '.join(story['text'].split('. ')[:2])+'.'
    body = t(20, 30, 'An actual TinyStories opening · The old hotel', 23, 'muted')
    for i, line in enumerate(textwrap.wrap(opening, 88)):
        body += t(20, 77+i*34, line, 26)
    for i, (key, name, purpose) in enumerate([
            ('train','Train','Learn parameters'),
            ('validation','Validation','Choose a checkpoint'),
            ('test','Test','Report held-out loss')]):
        x=20+i*390
        body += '<g>'+rect(x,170,340,110,'e')
        body += t(x+170,208,name,27,'e','middle',650)
        body += t(x+170,251,f'{split[key]:,} stories',28,'ink','middle')+'</g>'
        body += t(x+170,326,purpose,25,'muted','middle')
    return svg(body,358,'A source story opening and the disjoint document split')


def curves(result):
    body = t(20, 32, 'Validation cross-entropy · seed 11 · lower is better', 25)
    x0, y0, w, h = 80, 62, 1000, 245
    for loss in [3, 4, 5, 6, 7, 8, 9]:
        y = y0+h-(loss-3)/6*h
        body += path(f'M{x0} {y} H{x0+w}', 'line', 1)+t(63, y+7, loss, 20, 'muted', 'end')
    for step in [0, 1500, 3000, 4500, 6000]:
        x = x0+step/6000*w
        body += t(x, 337, f'{step:,}', 20, 'muted', 'middle')
    for k, color in zip(LABELS, ['e', 'k', 'd']):
        trace = next(r for r in result['runs'][k] if r['seed'] == 11)['trace']
        coords = [(x0+p['step']/6000*w, y0+h-(p['validation_loss']-3)/6*h) for p in trace]
        body += f'<g data-curve="{k}">' + path('M'+' L'.join(f'{x} {y}' for x,y in coords), color, 3)
        for point, (x,y) in zip(trace, coords):
            body += f'<circle cx="{x}" cy="{y}" r="4" fill="{COLORS[color]}" data-step="{point["step"]}" data-loss="{point["validation_loss"]}"/>'
        body += '</g>'
    body += t(580, 371, 'Optimizer updates', 22, 'muted', 'middle')
    for i,(kind,color) in enumerate(zip(LABELS,['e','k','d'])):
        body += t(65+i*370, 418, LABELS[kind], 24, color, weight=650)
    return svg(body, 443, 'Saved validation traces for all three models, seed 11')


def generations(prompt, samples):
    body = ''
    y = 33
    for line in textwrap.wrap('Prompt: '+prompt, 90):
        body += t(20, y, line, 25, 'e'); y += 33
    y += 25
    for kind in LABELS:
        sample = next(s for s in samples if s['model'] == kind and s['prompt_id'] == 'heldout')
        body += t(20, y, LABELS[kind], 24, 'd' if kind == 'multihead' else 'muted', weight=650)
        y += 32
        for line in textwrap.wrap(sample['continuation'], 95):
            body += t(20, y, line, 24); y += 30
        y += 23
    return svg(body, y, 'Unedited saved continuations for the same held-out story prefix')


def classroom(stage, reference):
    result = read('notebooks/wordlm/artifacts/heads/comparison.json')
    vocab = read('word-lab/models/vocab.json')['itos']
    examples = read('word-lab/examples.json')
    samples = read('word-lab/saved-examples.json')['examples']
    records = {s['key']: s for s in reference}
    parts = []

    def s(key, title, figure='', body='', **kw):
        return stage(key, title, figure, body, **kw)

    def reuse(key):
        old = records[key]
        return s(key, old['title'], old['figure'], old['body'],
                 companion=old.get('companion','').replace('href="#', 'href="part3-worked.html#'))

    parts.append(('Why more than one head?', [
        reuse('s01-v-prefix'), reuse('s01-v-coat-many'),
        reuse('s01-v-shared'), reuse('s01-v-independent'),
    ]))
    parts.append(('The complete multi-head operation', [
        s('s02-cap-roles', 'The same roles in the river example',
          rows(['Receiver: final “the”', 'Useful source', 'Message could carry'],
               [['Head 1', 'river', 'Setting clues'], ['Head 2', 'fisherman', 'Person clues']], [390,310,420]),
          'Same sentence, separate readings. Each head learns its own queries, keys and values.',
          companion='<p>These are possible roles, not assigned jobs or measured explanations of the trained heads. The prefix is “The fisherman sat beside the river bank and watched the ___”. Each head sees every allowed source.</p>'),
        s('s02-cap-combine', 'Keep both messages, then combine them',
          flow([('Two messages', 'each 1 × 2', 'v'), ('Concatenate', 'one 1 × 4 row', 'v'),
                ('Project with W_O', 'update: 1 × 4', 'd'), ('Add original row', 'e′ = e + Δe', 'e')]),
          'Concatenation joins the coordinates. The output projection learns how both heads contribute to the update.'),
        s('s02-cap-map', 'The complete multi-head model', full_map(),
          'Both heads read the same E. Their messages combine; the final updated row predicts the next token.',
          companion='<p>H = AV is the message matrix. This diagram keeps the two-head worksheet dimensions: 10 tokens, width 4, two heads of width 2. Next we switch to the trained TinyStories model: 64 slots, width 64, four heads of width 16.</p><p><a href="part3-worked.html">Optional: every worked calculation and the interactive head explorer</a> · <a href="notebooks/wordlm/07_multihead_step_by_step.html">Executable Notebook 7</a>.</p>'),
    ]))
    window_figure, evidence = windows(vocab)
    split = result['audit']['documents']
    parts.append(('TinyStories: text becomes training examples', [
        s('s03-cap-stories', 'Now learn from TinyStories',
          dataset_figure(split),
          'Our subset has 6,000 short, synthetic stories. Split whole stories before making next-token windows.',
          companion='<p><a href="https://huggingface.co/datasets/roneneldan/TinyStories">TinyStories by Ronen Eldan and Yuanzhi Li</a>, CDLA-Sharing-1.0. This is a 6,000-document subset, not the complete dataset. The saved split audit reports zero document overlap. <a href="notebooks/wordlm/05_training_and_inference_maps.html#story-complete">Read a complete source story and its provenance</a>.</p>'),
        s('s03-cap-tokens', 'Use one vocabulary for all three models',
          flow([('Story text', 'lowercase + tokenize', 'e'), ('Token IDs', '4,000 entries', 'e'),
                ('Known prefix', 'last 64 token slots', 'q'), ('Next token', 'observed target', 'd')]),
          '<strong>PAD</strong>: padding · <strong>BOS</strong>: beginning of story · <strong>EOS</strong>: end of story · <strong>UNK</strong>: unknown token.',
          companion='<p>The vocabulary is fitted on training stories only. The tokenizer separates punctuation too; “word” is shorthand for a vocabulary token. BOS/EOS wrap a complete story, not each sentence. Left-pad short prefixes, crop long prefixes to their last 64 tokens, and never cross a story boundary.</p>'),
        s('s03-cap-windows', 'The next word is the target, not an input', window_figure,
          'Slide the boundary one token forward to make another example. Each example still occupies 64 slots.',
          companion='<p>Lily’s sentence is an authored teaching example, not a claimed source story. The IDs here use the actual 4,000-item benchmark vocabulary. PAD is ID 0. These two rows illustrate batching; the saved training runs use batches of 512. Position IDs are 0–63 within each padded/cropped window.</p>'),
    ]))
    parts.append(('One quick pass through the four-head model', [
        s('s04-cap-embed', '1 · Look up rows, add positions, make Q/K/V',
          flow([('IDs X', '[B, 64]', 'e'), ('Token + position', 'E [B, 64, 64]', 'e'),
                ('Project Q, K, V', 'each [B, 64, 64]', 'q'), ('Split into 4 heads', 'each [B, 4, 64, 16]', 'q')]),
          'Now use 64 slots, width 64 and four heads of width 16. B counts examples: two above, 512 per training batch.',
          code='E = token_embedding(X) + position_embedding(positions)\nQ, K, V = (split_heads(layer(E)) for layer in (W_Q, W_K, W_V))'),
        s('s04-cap-weights', '2 · Match, mask, then normalize',
          flow([('Q × Kᵀ / √16', '[B, 4, 64, 64]', 'q'), ('Mask sources', 'future tokens + PAD', 'k'),
                ('Softmax over sources', 'A [B, 4, 64, 64]', 'a')],
               'One row = one receiver’s source weights, in one head. Each row sums to 1.'),
          'The four heads form four different attention patterns. Targets never enter this calculation.',
          companion='<p>This is the full causal teaching view for real receiver rows. The implementation keeps PAD-only receiver rows numerically defined; they do not feed the loss. The saved training and browser forward path computes only the final receiver’s query, since this task predicts one next token per window. It produces the same final logits with smaller [B,4,1,64] weight tensors.</p>'),
        s('s04-cap-mix', '3 · Mix values, join heads, add the update',
          flow([('A × V', '[B, 4, 64, 16]', 'v'), ('Join coordinates', '[B, 64, 64]', 'v'),
                ('Output projection', 'ΔE [B, 64, 64]', 'd'), ('Residual addition', 'E′ = E + ΔE', 'e')]),
          'Each head sends its own message. Joining four 16-coordinate messages restores width 64.',
          code='messages = A @ V\njoined = messages.transpose(1, 2).reshape(B, 64, 64)\nupdated = E + W_O(joined)'),
        s('s04-cap-predict', '4 · Predict the target from the final row',
          flow([('Final updated row', '[B, 64]', 'e'), ('Hidden + ReLU', '[B, 256]', 'd'),
                ('Vocabulary logits', '[B, 4,000]', 'e'), ('Cross-entropy', 'one scalar loss', 'a')]),
          '“BOS lily found” → target <strong>a</strong>. “BOS lily found a” → target <strong>red</strong>. The model supplies scores; the story supplies the answers.',
          code='logits = vocab_head(relu(hidden_layer(updated[:, -1])))\nloss = cross_entropy(logits, y)  # raw logits, observed next-token IDs'),
        s('s04-cap-train', '5 · One loss trains the whole model',
          flow([('Prediction loss', 'compare with y', 'a'), ('Backpropagation', 'compute gradients', 'd'),
                ('Optimizer step', 'update parameters', 'e'), ('Next batch', 'repeat', 'q')]),
          'Token rows, position rows, all heads, the output projection and the prediction MLP learn together.',
          code='optimizer.zero_grad()\nloss.backward()\noptimizer.step()',
          companion='<p>We do not assign head-specific targets such as “setting” or “person”. All heads learn through the same next-token task. These are one-block teaching models, not a full stack of normalized Transformer blocks.</p>'),
        s('s04-cap-generate', '6 · Generate with fixed parameters',
          flow([('Prompt → IDs', 'crop / pad to 64', 'e'), ('Four heads + MLP', 'vocabulary scores', 'q'),
                ('Choose next token', 'greedy or sampled', 'a'), ('Append and repeat', 'stop at EOS / limit', 'd')]),
          'Training changes the weights. Generation keeps the weights fixed and grows the prefix.',
          companion='<p>The app excludes PAD, BOS and UNK as generated outputs; EOS may end the story. The same tokenizer and vocabulary are used for training and generation. Attention softmax chooses source positions; vocabulary softmax supplies next-token probabilities.</p>'),
    ]))
    scores = [[LABELS[k], f'{result["aggregate"][k]["test_loss"]["mean"]:.3f} ± {result["aggregate"][k]["test_loss"]["sample_std"]:.3f}',
               f'{result["aggregate"][k]["test_perplexity"]["mean"]:.2f} ± {result["aggregate"][k]["test_perplexity"]["sample_std"]:.2f}'] for k in LABELS]
    costs = [[LABELS[k], f'{result["runs"][k][0]["parameter_count"]:,}',
              f'{result["aggregate"][k]["runtime_seconds"]["mean"]:.1f} s'] for k in LABELS]
    improvement = (1-result['aggregate']['multihead']['test_perplexity']['mean']/result['aggregate']['attention']['test_perplexity']['mean'])*100
    parts.append(('What did the trained models achieve?', [
        s('s05-cap-protocol', 'Change the model, keep the task fixed',
          rows(['Model', 'How it reads the 64-token window'],
               [['MLP', 'Flatten 64 × 64 → 4,096 inputs'],
                ['Single head', 'One source-weight pattern, width 64'],
                ['Four heads', 'Four source-weight patterns, width 16 each']], [380,740]),
          'Same 6,000 stories, vocabulary, context and 6,000-update budget. Three seeds: 11, 29, 47. Select by validation loss; evaluate on test stories.',
          companion='<p>Both attention models add learned position rows, use the same total width and prediction MLP, and have equal parameter counts. The MLP has no separate position table: concatenation encodes slot order. Four heads reuse the single-head optimizer settings. This is not a position-encoding ablation or a parameter-matched MLP comparison.</p>'),
        s('s05-cap-curves', 'Watch validation loss fall', curves(result),
          'Saved measurements from seed 11. Choose the best validation checkpoint—not simply the last update.',
          companion='<p>These are fixed selection-slice validation measurements, not test losses. The table on the next slide aggregates separately selected test checkpoints across three seeds. <a href="word-lab/models/comparison.json">All traces, checkpoint choices and protocol</a>.</p>'),
        s('s05-cap-scores', 'Four heads win this held-out comparison',
          rows(['Model', 'Test loss ↓', 'Test perplexity ↓'], scores, [450,335,335]),
          f'Mean ± sample SD across three seeds. Four heads reduce mean perplexity by {improvement:.1f}% versus one head in this experiment.',
          companion='<p>Loss is next-token cross-entropy in nats; each run’s perplexity is exp(loss), then the table averages the run perplexities. Lower means more probability assigned to observed tokens. It is not an accuracy percentage or a guarantee for every prompt.</p>'),
        s('s05-cap-cost', 'Better prediction has a cost',
          rows(['Model', 'Parameters', 'Training / seed'], costs, [450,335,335]),
          'Mean training time, including validation, on Apple M2 Max/MPS. The attention models have equal parameter counts; four heads train slightly slower.',
          companion='<p>Mean wall-clock training time, including validation, on Apple M2 Max / MPS. Times are device-specific. For full-sequence attention at fixed width D, matching and mixing cost O(T²D) for either head count; storing all weights costs O(hT²). The optimized next-token path used here computes only the final query. <a href="word-lab/#results">Open measured size, loss and timing results</a> · <a href="part2b.html#s16-cost-break">Optional complexity derivation</a>.</p>'),
    ]))
    heldout = next(p for p in examples['prompts'] if p['id'] == 'heldout')
    parts.append(('Finish by comparing their stories', [
        s('s06-cap-samples', 'The same prompt, three actual continuations', generations(heldout['prompt'], samples),
          'Saved seed-11 checkpoints · greedy decoding · 24-token limit. Four heads keep “owl” in view here, but all three outputs still have flaws.',
          companion='<p>Unedited saved outputs for the app’s held-out preset. It was chosen as the shortest test document, not selected for an attractive output. Aggregate loss is stronger evidence than one sample. The app lets students inspect more prompts and generate fresh continuations.</p>'),
        s('s06-cap-app', 'Try the models; compare the evidence',
          flow([('Choose one prompt', 'training / held-out / new', 'e'), ('Run all three', 'same decoding settings', 'q'),
                ('Read + measure', 'continuations and speed', 'd')]),
          '<strong><a href="word-lab/" target="_blank" rel="noopener">Open the live three-model comparison ↗</a></strong><br><a href="word-lab/#results" target="_blank" rel="noopener">Loss, perplexity, parameter counts and training time</a> · <a href="notebooks/wordlm/06_multihead_comparison.html">Experiment notebook</a>',
          companion='<p>Start with the held-out owl story and greedy decoding, then try an outside-story prompt. Compare coherence, repetition and topic consistency as well as local generation timings. Live inference runs on your device with WebGPU or WASM; it is not replaying the saved slide outputs. <a href="part3-worked.html">Optional worked arithmetic</a> · <a href="notebooks/wordlm/wordlm-notebooks.zip">Download code and notebooks</a>. This closes the text-attention series. Vision is a separate continuation.</p>'),
    ]))
    (ROOT/'figures/multihead/capstone-evidence.json').write_text(json.dumps({
        'windows': evidence, 'protocol': result['protocol'], 'scores': scores, 'costs': costs,
        'perplexity_improvement_percent': improvement,
        'heldout_prompt': heldout['prompt'],
        'samples': [s for s in samples if s['prompt_id'] == 'heldout'],
    }, indent=2)+'\n')
    return parts
