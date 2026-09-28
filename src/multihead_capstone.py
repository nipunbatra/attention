"""Brief head intuition, then the complete TinyStories classroom exercise.

The longer arithmetic story remains in part3-worked.html and Notebook 7.
All reported results and continuations below come from saved experiment files.
"""
from html import escape
import json
from pathlib import Path
import textwrap

from multihead_story import svg, t, rect, arrow, path, full_map, COLORS
from tinystories_setup import setup_frames
from tinystories_map import pipeline_svg, setup_checkpoint, MODEL_FOCUS

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


def flow(items, footer='', operators=None):
    """One left-to-right operation sequence; one label and shape per node."""
    width = (1120 - 38*(len(items)-1))/len(items)
    body = ''
    for i, (name, shape, color) in enumerate(items):
        x = 20+i*(width+38)
        body += '<g>' + rect(x, 40, width, 95, color)
        body += t(x+width/2, 77, name, 26, color, 'middle', 650)
        body += t(x+width/2, 113, shape, 24, 'muted', 'middle') + '</g>'
        if i < len(items)-1:
            if operators and operators[i]:
                body += t(x+width+19, 97, operators[i], 30, 'ink', 'middle')
            else:
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
        if key.startswith('s04-cap-'):
            name = key.removeprefix('s04-cap-')
            figure = pipeline_svg(MODEL_FOCUS[name], key)
            kw['companion'] = '<p>' + body + '</p>' + kw.get('companion', '')
            body = ''
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
    parts.append(('TinyStories: text becomes training examples', [
        *setup_frames(stage),
        setup_checkpoint(stage, 'scale', 'The same preparation at the experiment’s scale'),
        s('s03-cap-scale', 'The same preparation at the experiment’s scale',
          rows(['Quantity', 'Lily teaching example', 'Trained experiment'],
               [['Vocabulary C', '10 token IDs', '4,000 token IDs'],
                ['Context width T', '4 input slots', '64 input slots'],
                ['Batch size B', '2 examples', '512 examples']], [390,365,365]),
          'The rules stay the same: tokenize, look up IDs, add story boundaries, then create padded or cropped input–target pairs.',
          companion='<p>The toy vocabulary was chosen to expose every ID. The experimental vocabulary is frequency-ranked from training stories, so ordinary words receive different IDs. The four special IDs remain 0–3. The next slide remaps the same two prefixes into that real vocabulary.</p>'),
        setup_checkpoint(stage, 'real-windows', 'Our two examples in the real vocabulary'),
        s('s03-cap-windows', 'Our two examples in the real vocabulary', window_figure,
          'Same prefixes and answers; new ordinary-word IDs and 64 slots. We will follow this B = 2 batch through the trained model’s architecture.',
          companion='<p>Lily’s sentence is an authored teaching example, not a claimed source story. The IDs here use the actual 4,000-item benchmark vocabulary. PAD is ID 0. These two rows illustrate batching; the saved training runs use batches of 512. Position IDs are 0–63 within each padded/cropped window.</p>'),
    ]))
    parts.append(('TinyStories: the multi-head model, training and generation', [
        s('s04-cap-model', 'The four-head model used in this exercise',
          code='from multihead import MultiHeadAttentionLM\nmodel = MultiHeadAttentionLM(C, T, d_model=64, hidden=256, heads=4)\nB, T = X.shape',
          companion='<p>C=4,000 vocabulary items, T=64 input slots, model width 64, four heads of width 16 and a 256-unit prediction MLP. B=2 for our displayed batch. The highlighted boxes contain learned parameters. Other boxes reshape tensors or carry out fixed operations. The class is defined in <a href="notebooks/wordlm/multihead.py">multihead.py</a>.</p>'),
        s('s04-cap-lookup', 'Each input ID selects a learned row',
          flow([('Input IDs X', '[B, 64]', 'e'), ('Embedding table', '[4,000, 64]', 'e'),
                ('Selected token rows', '[B, 64, 64]', 'e')]),
          'The ID for “found” selects one row of 64 learned numbers. All occurrences share that token row. B counts examples, not heads.',
          code='token_rows = model.token_embedding(X)',
          companion='<p>The PAD token row is initialized to zero and receives no embedding-lookup gradient here, using padding_idx=0. BOS and UNK can learn when used in inputs. EOS is a target rather than an input in these windows, so its input row receives no task gradient. The separate classifier learns an EOS output score. Padding still needs an attention mask.</p>'),
        s('s04-cap-position', 'Add a learned position row in each slot',
          flow([('Token rows', '[B, 64, 64]', 'e'), ('Position rows', '[64, 64]', 'd'),
                ('Add by slot', 'E [B, 64, 64]', 'e')],
               'Token row says which item; position row distinguishes slots 0 through 63.', operators=['+', '=']),
          'This experiment uses learned absolute position embeddings, not sinusoidal encodings. The 64 × 64 table has one trainable row for each slot 0–63 within the padded or cropped window. Add that row to the token embedding. Both tables learn from the prediction loss.',
          code='# Learned 64 x 64 position table, not sinusoidal.\n# Trained together with the token embeddings.\npositions = torch.arange(T, device=X.device)\nE = token_rows + model.position_embedding(positions)[None]'),
        s('s04-cap-embed', 'Project Q, K and V, then split into four heads',
          flow([('Input E', '[B, 64, 64]', 'e'), ('Project Q, K, V', 'each [B, 64, 64]', 'q'),
                ('Reshape + transpose', 'each [B, 4, 64, 16]', 'q')]),
          'Every head reads the full input row through learned projections. Split the projected coordinates into four groups of 16.',
          code='Q = model.W_Q(E).reshape(B, T, 4, 16).transpose(1, 2)\nK = model.W_K(E).reshape(B, T, 4, 16).transpose(1, 2)\nV = model.W_V(E).reshape(B, T, 4, 16).transpose(1, 2)',
          companion='<p>For the two examples, each projected tensor is [2,4,64,16]: examples, heads, token slots, coordinates per head. Packed W_Q, W_K and W_V each map 64 input coordinates to 64 output coordinates.</p>'),
        s('s04-cap-scores', 'Each head compares queries with source keys',
          flow([('One head’s Q', '[B, 64, 16]', 'q'), ('One head’s Kᵀ', '[B, 16, 64]', 'k'),
                ('Divide by √16', '[B, 64, 64]', 'a')],
               'All four heads together: scores [B, 4, 64, 64].', operators=['×', None]),
          'For “BOS lily found”, follow the query at “found”. Its scores compare that receiver with the keys of the available source tokens.',
          code='scores = (Q @ K.transpose(-2, -1)) / math.sqrt(16)'),
        s('s04-cap-weights', 'Mask unused sources, then apply softmax',
          flow([('Scaled scores', '[B, 4, 64, 64]', 'a'), ('Mask sources', 'future tokens + PAD', 'k'),
                ('Softmax over sources', 'A [B, 4, 64, 64]', 'a')]),
          'At “found”, allow BOS, lily and found. PAD and future slots receive zero weight. Each real receiver’s source weights sum to one.',
          code='future = torch.ones(T, T, dtype=torch.bool, device=E.device).triu(1)\nreal = X.ne(0)\nmask = future[None] | ((~real[:, None, :]) & real[:, :, None])\nA = torch.softmax(scores.masked_fill(mask[:, None], float("-inf")), dim=-1)',
          companion='<p>This is the full causal teaching view for real receiver rows. The implementation keeps PAD-only receiver rows numerically defined; they do not feed the loss. The saved training and browser forward path computes only the final receiver’s query, since this task predicts one next token per window. It produces the same final logits with smaller [B,4,1,64] weight tensors.</p>'),
        s('s04-cap-mix', 'Each head mixes its value rows',
          flow([('Source weights A', '[B, 4, 64, 64]', 'a'), ('Value rows V', '[B, 4, 64, 16]', 'v'),
                ('Messages A × V', '[B, 4, 64, 16]', 'v')], operators=['×', '=']),
          'A weight scales a complete 16-number value row. Add those weighted rows to make one message per receiver, per head.',
          code='messages = A @ V'),
        s('s04-cap-join', 'Join the messages and add the update',
          flow([('Four messages', '4 × 16 coordinates', 'v'), ('Concatenate', '[B, 64, 64]', 'v'),
                ('Project + residual', 'E′ [B, 64, 64]', 'e')]),
          'Joining four 16-coordinate messages restores width 64. W_O learns how to combine them; the residual adds this update to the original E.',
          code='joined = messages.transpose(1, 2).reshape(B, T, 64)\nupdated = E + model.W_O(joined)'),
        s('s04-cap-predict', 'The final row predicts the observed next token',
          flow([('Final updated row', '[B, 64]', 'e'), ('Hidden + ReLU', '[B, 256]', 'd'),
                ('Vocabulary logits', '[B, 4,000]', 'e'), ('Cross-entropy', 'one scalar loss', 'a')]),
          '“BOS lily found” → target <strong>a</strong>. “BOS lily found a” → target <strong>red</strong>. The model supplies scores; the story supplies the answers.',
          code='hidden = torch.relu(model.hidden_layer(updated[:, -1]))\nlogits = model.vocab_head(hidden)\nloss = F.cross_entropy(logits, y)  # raw logits and target IDs'),
        s('s04-cap-train', 'One loss trains the whole model',
          flow([('Prediction loss', 'compare with y', 'a'), ('Backpropagation', 'compute gradients', 'd'),
                ('Optimizer step', 'update parameters', 'e'), ('Next batch', 'repeat', 'q')]),
          'Repeat over batches. The loss trains token rows, position rows, all four heads, W_O and the prediction MLP. Validation selects a checkpoint; test measures that frozen choice.',
          code='optimizer.zero_grad()\nloss.backward()\noptimizer.step()',
          companion='<p>We do not assign head-specific targets such as “setting” or “person”. All heads learn through the same next-token task. These are one-block teaching models, not a full stack of normalized Transformer blocks.</p>'),
        s('s04-cap-prompt', 'Generation begins with a known prompt',
          flow([('“Lily found”', 'same tokenizer', 'e'), ('BOS lily found', 'IDs [1, 23, 110]', 'e'),
                ('Pad / crop', 'input [1, 64]', 'e')]),
          'Use the training vocabulary and prepend BOS. Do not append EOS to an unfinished prompt. Load the chosen checkpoint and keep its parameters fixed.',
          code='history = [real_stoi["<BOS>"]] + [real_stoi.get(t, 3) for t in tokenize("Lily found")]\nvisible = history[-T:]\nX = torch.tensor([[0] * (T-len(visible)) + visible])',
          companion='<p>Left-pad this short prefix with 61 PAD IDs. After the history grows past 64 tokens, retain its last 64 IDs. There is no observed next-token target during free generation.</p>'),
        s('s04-cap-decode', 'Choose one token from the vocabulary scores',
          flow([('Frozen model', 'logits [1, 4,000]', 'q'), ('Vocabulary softmax', 'token probabilities', 'a'),
                ('Choose a token', 'greedy or sample', 'd')]),
          'Greedy decoding takes the highest score; sampling draws from the probability distribution. Exclude PAD, BOS and UNK as outputs. EOS can stop the story.',
          code='with torch.no_grad():\n    logits = model(X)\n    logits[:, [0, 1, 3]] = float("-inf")  # exclude PAD, BOS, UNK\n    next_id = logits.argmax(dim=-1).item()  # greedy choice',
          companion='<p>Attention softmax chooses source positions. Vocabulary softmax supplies next-token probabilities. Cross-entropy uses raw logits during training; we do not first apply this generation softmax before that loss.</p>'),
        s('s04-cap-generate', 'Append the chosen token and run the model again',
          flow([('BOS lily found', 'known prefix', 'e'), ('Suppose choice = a', 'illustrative choice', 'd'),
                ('BOS lily found a', 'next input prefix', 'e')]),
          'Keep generating until EOS or a token limit. Training used observed previous words; generation now feeds back the model’s own choices.',
          code='finished = next_id == real_stoi["<EOS>"]\nif not finished:\n    history.append(next_id)  # crop/pad history, then run the same model again',
          companion='<p>“a” here is an illustrative choice, not a claimed checkpoint output. The saved continuations later show the actual outputs. An early mistake can alter every later input. The weights stay fixed throughout generation.</p>'),
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
