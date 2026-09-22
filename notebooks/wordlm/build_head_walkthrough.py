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
        cells.append(md(f'<a id="{step["key"]}"></a>\n## {step["title"]}\n\n[Matching slide]({link})\n\n{figure}\n\n{body}'+(f'\n\n```python\n{step["code"]}\n```' if step['code'] else '')))
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
