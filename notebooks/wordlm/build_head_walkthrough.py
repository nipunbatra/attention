"""Notebook 7 shares Part III's diagrams and literal, paced code snippets."""
import json
import re
import shutil
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
FIG=REPO/'figures/multihead'


def build():
    shutil.copy2(REPO/'examples/multihead_from_scratch.py', ROOT/'multihead_from_scratch.py')
    md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    cells=[md('''# Multi-head attention, step by step

Keep the river-bank example from Part II. First follow two different reading
patterns through the lecture's figures. Then run the detailed implementation
lab, including the batch axes and a numerical comparison with PyTorch.

[Visual story](#visual-story) · [Executable lab](#executable-lab)

The toy uses hand-chosen parameters. The final links lead to genuinely trained
TinyStories models, not this worksheet. As in Part II, E stores embedding rows,
M is the mask, H = AV stores message rows, and E′ = E + ΔE. Superscripts label
heads; subscripts label tokens.

[Part III](../../part3.html) · [Live models](../../word-lab/) · [Download code and notebooks](wordlm-notebooks.zip)

Run all cells from this directory. No data download or training run is needed.
The single optimizer step demonstrates learning; it does not create the models
used in the benchmark.'''),code('''import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from IPython.display import display, HTML
from multihead_from_scratch import (TinyMultiHeadLM, ScratchMultiHead,
                                   load_worksheet_weights, copy_to_pytorch)

torch.manual_seed(7)
worksheet = json.loads(Path('multihead-worksheet.json').read_text())
word_to_id = {word: i for i, word in enumerate(worksheet['vocab'])}
river_ids = [word_to_id[w.lower()] for w in worksheet['sentences']['river']]
cheque_ids = [word_to_id[w.lower()] for w in worksheet['sentences']['cheque']]
print('River tokens:', worksheet['sentences']['river'])
print('River IDs:', river_ids)''')]
    cells.append(md('<a id="visual-story"></a>\n# The visual story\n\nThe same figures appear in the lecture. Short code excerpts here are explained visually; the executable lab below builds their inputs and runs each operation.'))
    steps=json.loads((FIG/'manifest.json').read_text())
    section_counts={}
    for step in steps:
        section=step['key'].split('-')[0]
        section_counts[section]=section_counts.get(section,0)+1
        link=f'../../part3.html?present#{section}/{section_counts[section]}/0'
        figure_path=FIG/(step['key']+'.svg')
        figure=figure_path.read_text() if figure_path.exists() else ''
        body = step['body']
        body = re.sub(r'href="(?!https?:|#)([^"]+)"', r'href="../../\1"', body)
        if step['key'] == 's03-v-explore':
            body = 'Switch contexts in the matching slide to see both reading patterns change. The executable lab below computes the river and cheque examples from the same parameters.'
            figure = (FIG/'s01-v-both.svg').read_text()
        companion=re.sub(r'href="(?!https?:|#)([^\"]+)"',r'href="../../\1"',step.get('companion',''))
        cells.append(md(f'<a id="{step["key"]}"></a>\n## {step["title"]}\n\n[Matching slide]({link})\n\n{figure}\n\n{body}'+(f'\n\n```python\n{step["code"]}\n```' if step['code'] else '')+'\n\n'+companion))
        if step['key']=='s01-v-independent':
            cells.append(code('''# Separate two-source illustration, before the ten-token worksheet.
values = torch.tensor([[10., 1.], [2., 8.]])  # river, fisherman
setting_weights = torch.tensor([0.8, 0.2])
person_weights = torch.tensor([0.2, 0.8])
one_head = setting_weights @ values
two_heads = torch.stack([setting_weights @ values[:, 0],
                         person_weights @ values[:, 1]])
print('One shared mixture:', one_head.tolist())
print('Two separate mixtures:', two_heads.tolist())
torch.testing.assert_close(one_head, torch.tensor([8.4, 2.4]))
torch.testing.assert_close(two_heads, torch.tensor([8.4, 6.6]))'''))
        if step['key']=='s02-v-sources':
            cells.append(code('''# Every projected row is computed from an input embedding.
case = worksheet['headsLesson']['cases']['river']
E = torch.tensor(case['E'])
for h, projection in enumerate(worksheet['headsLesson']['projections']):
    Q, K, V = [E @ torch.tensor(projection[kind], dtype=torch.float32)
               for kind in ['Q', 'K', 'V']]
    source = 5 if h == 0 else 1  # river or fisherman, zero-based index
    print(f'Head {h+1}: final query', Q[-1].tolist())
    print('Source:', case['tokens'][source],
          'key:', K[source].tolist(), 'value:', V[source].tolist())
    for kind, actual in [('Q', Q), ('K', K), ('V', V)]:
        torch.testing.assert_close(actual, torch.tensor(case['heads'][h][kind]))'''))
        if step['key']=='s02-v-separate':
            cells.append(code('''# Same projected coordinates, one wide softmax or two narrow softmaxes.
case = worksheet['headsLesson']['cases']['river']
Q_wide, K_wide, V_wide = [
    torch.cat([torch.tensor(head[kind]) for head in case['heads']], dim=-1)
    for kind in ['Q', 'K', 'V']
]
scores_wide = Q_wide[-1] @ K_wide.T / math.sqrt(4)
weights_wide = scores_wide.softmax(-1)  # all ten sources allowed at row 10
message_wide = weights_wide @ V_wide
torch.testing.assert_close(weights_wide, torch.tensor(case['wide']['A'][-1]))
torch.testing.assert_close(message_wide, torch.tensor(case['wide']['messages'][-1]))
for label, row in [('one wide head', weights_wide)] + [
    (f'head {h+1}', torch.tensor(head['A'][-1]))
    for h, head in enumerate(case['heads'])
]:
    print(label, 'river:', round(row[5].item(), 3),
          'fisherman:', round(row[1].item(), 3))'''))
        if step['key']=='s05-v-bias-layers':
            cells.append(code('''# A learned offset is broadcast to each token row.
Q_no_bias = torch.tensor(case['heads'][0]['Q'])
b_Q = torch.tensor([0.2, -0.1])  # illustrative, not a fitted parameter
Q_with_bias = Q_no_bias + b_Q
print('Receiver 10:', Q_no_bias[-1].tolist(), '->', Q_with_bias[-1].tolist())
torch.testing.assert_close(Q_with_bias[-1], torch.tensor([2.5, 2.2]))

without_bias = nn.MultiheadAttention(4, 2, bias=False)
with_bias = nn.MultiheadAttention(4, 2, bias=True)
count = lambda layer: sum(p.numel() for p in layer.parameters())
print('Projection parameters:', count(without_bias), 'vs', count(with_bias))
print('Packed Q/K/V bias:', tuple(with_bias.in_proj_bias.shape))
print('Output bias:', tuple(with_bias.out_proj.bias.shape))
assert count(with_bias) - count(without_bias) == 16
assert without_bias.in_proj_bias is None
assert without_bias.out_proj.bias is None'''))
    cells.append(md('''<a id="executable-lab"></a>
# The executable lab

The lecture has now shown the whole idea. This optional lab slows down the code:
we create two examples, project the embeddings, expose the head axis, calculate
the messages and check the results. Here `D` means the same model width as
`d_model` in the lecture. The head axis has size 2; it is not the message matrix H.

The expanded figures below accompany the code rather than add new lecture slides.'''))
    for step in json.loads((FIG/'lab-manifest.json').read_text()):
        figure_path=FIG/(step['key']+'.svg')
        figure=figure_path.read_text() if figure_path.exists() else ''
        body=re.sub(r'href="(?!https?:|#)([^"]+)"',r'href="../../\1"',step['body'])
        if step['key'] == 's04-live':
            body = 'The matching slide lets you switch contexts and inspect either head. Here, compute both sentences and print their final-query messages. These are hand-chosen worksheet parameters, not trained results.'
            figure = (FIG/'s03-weights1.svg').read_text()
        # SVGs are embedded, not fetched: the saved notebook keeps its figures offline.
        cells.append(md(f'<a id="{step["key"]}"></a>\n## {step["title"]}\n\n{figure}\n\n{body}'))
        if step['key'] == 's04-live':
            cells.append(code('''for name in ['river', 'cheque']:
    ids = torch.tensor([[word_to_id[w.lower()] for w in worksheet['sentences'][name]]])
    demo = load_worksheet_weights(TinyMultiHeadLM(len(word_to_id)), worksheet)
    E_demo = demo.token_embedding(ids) + demo.position_embedding(torch.arange(10))
    with torch.no_grad():
        _, weights = demo.attention(E_demo)
        values = demo.attention.split_heads(demo.attention.W_V(E_demo))
        messages = weights @ values
    print(name, 'final messages by head:', messages[0, :, -1].tolist())'''))
        if step['key']=='s05-model':
            cells.append(md('### The complete implementation\n\nThis is the same source file imported above. Read the small snippets that follow alongside the diagram, then return here to see how they fit together. `load_worksheet_weights` copies the printed parameters so our outputs match the figures. It is not part of an ordinary training loop.'))
            source=(ROOT/'multihead_from_scratch.py').read_text()
            cells.append(code(source))
        if step['code']:
            cells.append(code(step['code']))
        if step['key']=='s05-softmax':
            cells.append(code('''expected = torch.tensor([[h['A'] for h in worksheet['headsLesson']['cases'][name]['heads']]
                         for name in ['river', 'cheque']])
torch.testing.assert_close(A, expected)
torch.testing.assert_close(A.sum(-1), torch.ones(2, 2, 10))
assert not A.triu(1).any()
print('All 400 head weights match the worksheet; no future source receives weight.')'''))
        if step['key']=='s05-output':
            cells.append(code('''expected = torch.tensor([worksheet['headsLesson']['cases'][name]['logits'][-1]
                         for name in ['river', 'cheque']])
torch.testing.assert_close(logits, expected)
torch.testing.assert_close(model(X), logits)
print('All 40 vocabulary logits match the independent worksheet.')'''))
        if step['key']=='s06-update':
            cells.append(code('''assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
print('All learned tables, head projections and prediction layers received finite gradients.')'''))
        if step['key']=='s07-check':
            cells.append(code('''torch.testing.assert_close(A_api, A_scratch)
print('Scratch and PyTorch updates agree:', tuple(delta_api.shape))
print('Individual head weights agree:', tuple(A_api.shape))'''))
    cells.append(md('''## Continue with the trained experiment

[Notebook 6](06_multihead_comparison.html) inspects the four-head TinyStories
checkpoint, per-head weights and all three-seed results. The browser demo loads
the actual exported checkpoints and measures generation on your device.

Visual teaching references: [3Blue1Brown](https://www.3blue1brown.com/lessons/attention/)
and [Jay Alammar](https://jalammar.github.io/illustrated-transformer/).
Formula and API references: [Attention Is All You Need, §3.2.2](https://arxiv.org/abs/1706.03762)
and [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
The diagrams and worked numbers here are original adaptations of our Part II example.'''))
    notebook=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    nbf.write(notebook,ROOT/'07_multihead_step_by_step.ipynb')
    print('Wrote notebook 7:',len(cells),'cells')


if __name__=='__main__':build()
