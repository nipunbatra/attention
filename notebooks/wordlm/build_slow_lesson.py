#!/usr/bin/env python3
"""Execute one source of truth, then export its notebook, slides and study guide.

python build_slow_lesson.py --lecture-dir /path/to/attention
The export never uploads data or changes the separate private Site.
"""
from __future__ import annotations
import argparse
import contextlib
from html import escape
import io
import json
from pathlib import Path
import shutil
import zipfile

import nbformat as nbf
from slow_walkthrough import STAGES, initial_namespace, render_figure, map_mode
from pipeline_maps import pipeline_svg
from code_display import PYTHON_CSS, highlight_python, validate_python

ROOT = Path(__file__).resolve().parent
BOOK = '05_training_and_inference_maps'
LIVE = 'https://nipunbatra.github.io/attention/'
STORY_STAGES = {'story-complete', 'story-excerpts'}
TOKENIZATION_STAGES = {'tokenization-intro', 'tokenization-choices', 'tokenization-rules'}
TOKENIZATION_SOURCES = (
    ('Hugging Face tokenizer overview', 'https://huggingface.co/docs/transformers/tokenizer_summary'),
    ('Sennrich et al., 2016: subword units', 'https://aclanthology.org/P16-1162/'),
)
TOKENIZATION_LINKS = ' · '.join(f'<a href="{url}">{label}</a>' for label,url in TOKENIZATION_SOURCES)
STORY_CREDIT = ('TinyStories · Ronen Eldan &amp; Yuanzhi Li · '
    '<a href="https://huggingface.co/datasets/roneneldan/TinyStories">source dataset</a> · '
    '<a href="https://cdla.dev/sharing-1-0/">CDLA-Sharing-1.0</a>')

# Every frame has an explicit excerpt or None for a conceptual/summary frame.
# Never slice code by line count: that can remove closing delimiters or loop bodies.
SLIDE_CODE = {
 'data':None,
 'story-complete':None,
 'story-excerpts':None,
 'split':None,
 'sentence':"sentence = 'Lily found a red ball.'",
 'tokenization-intro':None,
 'tokenization-choices':None,
 'tokenization-rules':None,
 'tokenize':'pieces = tokenize(sentence)',
 'vocabulary':'for token_id, word in enumerate(words):\n    print(word, token_id)',
 'special':"unknown = vocab.encode_tokens(['blue'], boundaries=False)",
 'boundaries':'ids = vocab.encode_tokens(pieces)',
 'shapes':'B, w, C = 2, 4, 10\nd, h, d_k, d_v = 4, 8, 3, 2',
 'positions':'token_rows = attention.token_embedding(X)\nposition_rows = attention.position_embedding(torch.arange(w))\nE = token_rows + position_rows[None, :, :]',
 'story-indices':'w = 4\ntarget_position = 4',
 'one-pair':'context_ids = ids[target_position-w:target_position]\ntarget_id = ids[target_position]',
 'pair-lists':'contexts = []\ntargets = []',
 'pair-first':'t = 1\nvisible = ids[max(0, t-w):t]\ncontext_ids = [vocab.pad_id] * (w-len(visible)) + visible\ntarget_id = ids[t]',
 'pair-second':'t = 2\nvisible = ids[max(0, t-w):t]\ncontext_ids = [vocab.pad_id] * (w-len(visible)) + visible\ntarget_id = ids[t]',
 'pair-append':'contexts.append(context_ids)\ntargets.append(target_id)',
 'pair-append-second':'contexts.append(context_ids)\ntargets.append(target_id)',
 'pairs-loop':'for t in range(3, len(ids)):\n    visible = ids[max(0, t-w):t]\n    context_ids = [vocab.pad_id] * (w-len(visible)) + visible\n    target_id = ids[t]\n    contexts.append(context_ids)\n    targets.append(target_id)',
 'windows-first':'for row in range(4):\n    print(row, contexts[row], targets[row])',
 'windows-last':'for row in range(4, 7):\n    print(row, contexts[row], targets[row])',
 'pairs-tensors':'all_X = torch.tensor(contexts, dtype=torch.long)\nall_y = torch.tensor(targets, dtype=torch.long)\nN = len(all_y)',
 'counts-story':'ordinary_tokens = len(pieces)\nexamples_in_story = ordinary_tokens + 1',
 'counts-train':'train_tokens = 964_338\ntrain_stories = 4_822\ntrain_examples = train_tokens + train_stories',
 'counts':None,
 'context':'history = ids[:6]  # BOS lily found a red ball\nfor width in [2, 4, 6]:\n    print(history[-width:])',
 'batch':'selected = torch.tensor([2, 3])\nX, y = all_X[selected], all_y[selected]\nB = X.shape[0]',
 'batch-ids':'print(X.shape, X.dtype)\nprint(y.shape, y.dtype)',
 'batches':'for start in range(0, N, B):\n    batch_X = all_X[start:start+B]\n    print(len(batch_X))',
 'mlp-map':None,
 'lookup':"found_id = vocab.stoi['found']\nfound_vector = mlp.token_embedding.weight[found_id]",
 'lookup-flow':'selected_rows = embedding_table[X[1]]',
 'embedding-batch':'E_mlp = mlp.token_embedding(X)',
 'flatten':'flat = E_mlp.flatten(start_dim=1)',
 'hidden-affine':'pre_hidden = mlp.hidden_layer(flat)',
 'relu':'hidden_mlp = torch.relu(pre_hidden)',
 'vocab-head':'logits_mlp = mlp.vocab_head(hidden_mlp)',
 'attention-map':None,
 'query':'q = attention.W_Q(E[:, -1:, :])',
 'keys':'K = attention.W_K(E)',
 'values':'V = attention.W_V(E)',
 'scores':'raw_scores = q @ K.transpose(-2, -1)\nscores = raw_scores / math.sqrt(d_k)',
 'mask':"pad_mask = X[:, None, :].eq(vocab.pad_id)\nmasked_scores = scores.masked_fill(pad_mask, float('-inf'))",
 'attention-softmax':'source_exp = (masked_scores - masked_scores.amax(-1, keepdim=True)).exp()\nA = source_exp / source_exp.sum(-1, keepdim=True)',
 'mix':'message = A @ V',
 'output-map':'update = attention.W_O(message).squeeze(1)',
 'residual':'final = E[:, -1, :] + update\nhidden_att = torch.relu(attention.hidden_layer(final))\nlogits_att = attention.vocab_head(hidden_att)',
 'word-softmax':'word_exp = (logits_mlp[1] - logits_mlp[1].max()).exp()\np = word_exp / word_exp.sum()',
 'target':'guess_id = int(p.argmax())\ntarget_id = int(y[1])',
 'loss':"losses = F.cross_entropy(logits_mlp, y, reduction='none')\nloss = losses.mean()",
 'gradient':'mlp.zero_grad(set_to_none=True)\nloss.backward()\ngrad = mlp.vocab_head.weight.grad[9, 0]',
 'update':'optimizer = torch.optim.SGD(mlp.parameters(), lr=0.1)\noptimizer.step()  # every trainable parameter',
 'training-loop':None,
 'evaluation':'mlp.eval()\nwith torch.inference_mode():\n    frozen_loss = F.cross_entropy(mlp(X), y)',
 'mlp-inference':None,
 'attention-inference':None,
 'prompt':"prompt = 'Lily found'\nprompt_ids = vocab.encode_tokens(tokenize(prompt), boundaries=False)\nhistory = [vocab.bos_id] + prompt_ids",
 'prompt-window':'kept = history[-w:]\ncontext_ids = [vocab.pad_id] * (w-len(kept)) + kept\ninference_X = torch.tensor([context_ids])',
 'generation-logits':'with torch.inference_mode():\n    next_logits = mlp(inference_X)[0]',
 'generation-probabilities':"blocked_ids = [vocab.pad_id, vocab.bos_id, vocab.unk_id]\nwith torch.inference_mode():\n    next_logits[blocked_ids] = float('-inf')\n    next_p = next_logits.softmax(-1)",
 'decode':'greedy_id = int(next_p.argmax())\ncdf = next_p.cumsum(0)\nsample_id = int(torch.searchsorted(cdf, torch.tensor(0.8)))',
 'generation-append':'chosen = greedy_id\nif chosen != vocab.eos_id:\n    history.append(chosen)',
 'append':None,
 'benchmark':None,
 'next':None,
}
SLIDE_CODE.update({stage['id']:None for stage in STAGES if stage.get('map_checkpoint')})


def slide_code(stage):
    """An explicit choice is mandatory, even when a slide needs no code."""
    source = SLIDE_CODE[stage['id']]
    if source is not None:
        validate_python(source, stage['id'])
        limit = 6 if stage['id'] == 'pairs-loop' else 4
        if len(source.splitlines()) > limit or any(len(line) > 88 for line in source.splitlines()):
            raise ValueError(f"{stage['id']}: split this operation into smaller slides")
    return source

def notebook_cells(md, code, setup=None):
    cells=[md('''# 5 · From a story to a trained predictor

Trace the same two examples through an MLP and a one-layer attention model. Start
with data. Full-map checkpoints highlight the operation before its worked
example. Every numbered section links to its lecture slide. Run the cells in order.

The small calculation uses **B=2, w=4, C=10**. Its random weights are not the
trained TinyStories model. The final comparison is explicitly labelled saved
experimental evidence. Download the complete folder so the helper modules are
beside this notebook. No GPU, dataset download or retraining is needed for this
walkthrough.

[Readable study guide](https://nipunbatra.github.io/attention/notebooks/wordlm/05_training_and_inference_maps.html)
· [Part II](https://nipunbatra.github.io/attention/attention.html?present#s19/4/0)
'''),code('''from slow_walkthrough import initial_namespace, show_figure
from IPython.display import HTML, display
from pathlib import Path
globals().update(initial_namespace())
display(HTML('<style>' + Path('lesson.css').read_text() + '</style>'))''')]
    for index,s in enumerate(STAGES):
        link=LIVE+f"attention.html?present#s19/{index+5}/0"
        text=f"<a id='{s['id']}'></a>\n\n## {index+1:02d} · {s['title']}\n\n**{s['chapter']}** · [Matching slide]({link})\n\n{s['body']}"
        if s['id'] in STORY_STAGES:
            text+='\n\n'+STORY_CREDIT+'\n\nThree unchanged source texts are included in `story_examples.json`, with revision, row IDs and checksums. Display labels are ours; […] marks omitted text in the figure. The code below prints the complete text and recomputes the lengths.'
        if s['id'] in TOKENIZATION_STAGES:
            text+='\n\nBackground: '+TOKENIZATION_LINKS+'. The notebook rules come from `normalize_text` and `tokenize` in `wordlm.py`. The subword example illustrates a possible split, not a trained tokenizer output.'
        if s.get('check'):
            q,a=s['check'];text+=f'\n\n<details><summary>Check yourself: {q}</summary>{a}</details>'
        cells.append(md(text))
        cell=code(s['code']+f"\n\nshow_figure('{s['id']}', globals())")
        cell.metadata['lesson_stage']=s['id']
        cells.append(cell)
    return cells

def build(lecture):
    assert set(SLIDE_CODE) == {stage['id'] for stage in STAGES}
    for stage in STAGES:
        slide_code(stage)
    lecture=Path(lecture).resolve()
    out=lecture/'notebooks'/'wordlm';out.mkdir(parents=True,exist_ok=True)
    figures=ROOT/'figures'/'slow-walkthrough';figures.mkdir(parents=True,exist_ok=True)
    slide_figures=lecture/'figures'/'wordlm-pipeline'/'steps';slide_figures.mkdir(parents=True,exist_ok=True)
    md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    nb=nbf.v4.new_notebook(cells=notebook_cells(md,code),metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
    ns=initial_namespace();results=[]
    for index,s in enumerate(STAGES):
        captured=io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile(s['code'],s['id'],'exec'),ns)
        svg=render_figure(s,ns)
        for directory in (figures,slide_figures):
            (directory/(s['id']+'.svg')).write_text(svg)
        stdout=captured.getvalue()
        c=nb.cells[3+index*2];c.execution_count=index+2
        c.outputs=([nbf.v4.new_output('stream',name='stdout',text=stdout)] if stdout else [])+[nbf.v4.new_output('display_data',data={'image/svg+xml':svg},metadata={})]
        results.append(dict(s,svg=svg,stdout=stdout,index=index+1,slide=index+5))
    nbf.write(nb,ROOT/(BOOK+'.ipynb'))
    css=(ROOT/'lesson.css').read_text()+'\n'+PYTHON_CSS
    chunks=[];chapters={}
    for s in results:
        chapters.setdefault(s['chapter'],s['id'])
        checks=''
        if s.get('check'):
            q,a=s['check'];checks=f'<aside class="check"><strong>Pause and predict</strong><p>{escape(q)}</p><details><summary>Show explanation</summary><p>{escape(a)}</p></details></aside>'
        map_html=''
        if s['focus'] and not s.get('map_checkpoint'):
            focused=pipeline_svg(s['kind'],map_mode(s),s['focus'],uid='locate-'+s['id'])
            map_html=f'<details class="route-map"><summary>Where are we on the full {s["kind"].upper()} map?</summary><div class="figure-wrap">{focused}</div></details>'
        result_html=f'<div class="code-label">Printed output</div><pre class="output">{escape(s["stdout"])}</pre>' if s['stdout'] else ''
        caption=('Same figure as the lecture. Full-story lengths are computed below. '+STORY_CREDIT+'. Display labels added; […] marks omissions.' if s['id'] in STORY_STAGES else 'Same figure as the lecture. Numeric values are computed from the code below.')
        if s['id'] in TOKENIZATION_STAGES:
            caption+=' Background: '+TOKENIZATION_LINKS+'. Notebook rules: <a href="wordlm.py">wordlm.py</a>.'
        if s.get('map_checkpoint'):
            caption='The highlighted boxes locate the next worked example. The layout stays the same as we move through the model.'
        code_html='' if s.get('map_checkpoint') else f'<div class="code-label">Python · run after the previous step</div><pre>{highlight_python(s["code"])}</pre>'
        chunks.append(f'''<section class="lesson-step" id="{s['id']}">
<div class="step-meta"><span>{escape(s['chapter'])} · Step {s['index']:02d} / {len(STAGES)}</span><a href="../../attention.html?present#s19/{s['slide']}/0">Open matching slide ↗</a></div>
<h2>{escape(s['title'])}</h2><p>{escape(s['body'])}</p>
<figure><div class="figure-wrap">{s['svg']}</div><figcaption>{caption}</figcaption></figure>
{map_html}{code_html}{result_html}{checks}</section>''')
    nav=''.join(f'<li><a href="#{key}">{escape(ch.split(". ",1)[1])}</a></li>' for ch,key in chapters.items())
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>From a story to a trained predictor · Part II companion</title><style>{css}</style></head><body>
<header class="book-header"><div class="eyebrow">Attention and language · Notebook 5</div><h1>From a story to a trained predictor</h1><p>One sentence. Seven training examples. Two models. Follow the data, every tensor and one learning step before generating a new token.</p><div class="notebook-links"><a href="wordlm-notebooks.zip" download>Download all notebooks + support files</a><a href="{BOOK}.ipynb" download>Download this notebook</a><a href="../../attention.html?present#s19/4/0">Lecture slides ↗</a></div></header>
<div class="book-layout"><nav class="chapters" aria-label="Chapters"><strong>Work through the example</strong><ol>{nav}</ol><div class="download"><a href="#run">Run it yourself</a><a href="01_words_to_probabilities.html">1 · Train the MLP</a><a href="03_causal_attention_from_scratch.html">3 · Train attention</a><a href="04_what_did_the_models_learn.html">4 · Compare trained models</a></div></nav><main>
<section id="run" class="run-guide"><h2>Before you start</h2><p>The small example uses <code>B=2</code> examples, <code>w=4</code> context slots and <code>C=10</code> vocabulary items. Every number is generated by the supplied code, not drawn by hand. The final measured comparison uses a separate 6,000-story corpus and trained models.</p><p>Unzip the download, open a terminal in that folder and run:</p><pre>python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab {BOOK}.ipynb</pre><p>Then choose <strong>Run → Run All Cells</strong>. This notebook needs no GPU or data download. Notebooks 1 and 3 explain the real corpus and training runs. Keep <code>wordlm.py</code> and the other helpers beside the notebooks.</p></section>
{''.join(chunks)}<footer class="footer">{STORY_CREDIT}. Three selected stories are reproduced unchanged in the download; the figures add labels and mark omissions with […]. The full 6,000-story corpus is not bundled. Benchmark source and checksums are included. The measured device was Apple MPS, not DGX Spark.</footer></main></div></body></html>'''
    (out/(BOOK+'.html')).write_text(html)
    local_html=html.replace('../../attention.html',LIVE+'attention.html')
    for download in ['wordlm-notebooks.zip',BOOK+'.ipynb']:
        local_html=local_html.replace('href="'+download+'"','href="'+LIVE+'notebooks/wordlm/'+download+'"')
    (ROOT/'html').mkdir(exist_ok=True)
    (ROOT/'html'/(BOOK+'.html')).write_text(local_html)
    # Put numeric diagrams in the slides and keep an expandable master map nearby.
    fragments=['''<style>
#s19 .pipeline-lesson .step-figure{margin:12px 0;overflow-x:auto}
#s19 .pipeline-lesson .step-figure svg{width:100%;height:auto;max-height:275px;display:block}
#s19 .pipeline-lesson .step-figure.master svg{max-height:410px}
#s19 #s19-pipeline-lookup-flow .step-figure svg{max-height:350px}
#s19 .pipeline-lesson.story-sample .step-figure svg{max-height:350px}
#s19 .pipeline-lesson.tokenization-lesson .step-figure svg{max-height:310px}
#s19 #s19-pipeline-pairs-tensors .step-figure svg{max-height:310px}
#s19 #s19-pipeline-pairs-tensors .step-figure{margin:6px 0}
#s19 #s19-pipeline-pairs-tensors pre{margin:4px 0}
#s19 .tokenization-lesson.lecture-topic-break h3{margin-bottom:20px}
#s19 .tokenization-lesson.lecture-topic-break .step-meta{max-width:none}
#s19 .tokenization-mobile{display:none}
#s19 .pipeline-lesson .story-credit{font-size:18px;color:var(--ink-2);margin:10px 0 0}
#s19 .pipeline-lesson .step-copy{font-size:24px;line-height:1.4;margin:8px 0 12px}
#s19 .pipeline-lesson .step-meta{font-size:18px;color:var(--ink-2);margin:8px 0}
#s19 .pipeline-lesson .step-meta a{float:right}
#s19 .pipeline-lesson pre{font-size:21px;line-height:1.4;padding:10px 16px;margin:10px 0;background:var(--paper-2,#eef0f4);white-space:pre-wrap;max-width:100%;overflow-x:auto}
#s19 .pipeline-lesson .step-code{display:block}
#s19 .pipeline-lesson .step-code a{font-size:19px}
body:not(.present) #s19 .pipeline-lesson{padding:30px 0;border-bottom:1px solid var(--line)}
@media(max-width:650px){body:not(.present) #s19 .step-figure svg{min-width:900px}body:not(.present) #s19 .pipeline-lesson .step-code{grid-template-columns:1fr}body:not(.present) #s19 .pipeline-lesson .step-copy{font-size:18px}}
@media(max-width:650px){body:not(.present) #s19-pipeline-tokenization-intro .step-figure{overflow:visible}body:not(.present) #s19-pipeline-tokenization-intro .step-figure svg{display:none}body:not(.present) #s19 .tokenization-mobile{display:block;font-size:20px;line-height:1.4}#s19 .tokenization-mobile strong{display:block;font-size:26px}#s19 .tokenization-mobile .tokenization-example{display:block;font-size:40px;color:var(--c-e,#245EDB);margin:24px 0}}
</style>
<!--PIPELINE_TEMPLATES-->
<div class="frame pipeline-lesson lecture-topic-break" id="s19-pipeline-break" data-title="From a story to a trained predictor" data-autobuild="off">
<script type="text/x-notes">Use this as a slow, optional lab walkthrough over multiple sessions. The seven chapters connect real corpus provenance to a tiny, fully executable batch. Every figure also lives beside its code in Notebook 5. Distinguish toy arithmetic from saved benchmark evidence.</script><h3>From a story to a trained predictor</h3><p class="topic-recap">We have followed one attention calculation.</p><p class="topic-question">Now follow the data, each tensor, one learning step and the generation loop. <a href="notebooks/wordlm/05_training_and_inference_maps.html">Open the illustrated companion ↗</a></p></div>''']
    manifest=[]
    for s in results:
        master=s.get('map_checkpoint') or s['id'] in {'mlp-map','attention-map','mlp-inference','attention-inference'}
        story_sample=s['id'] in STORY_STAGES
        tokenization=s['id'] in TOKENIZATION_STAGES
        excerpt=slide_code(s)
        code_block='' if excerpt is None else f'<div class="step-code"><pre>{highlight_python(excerpt)}</pre></div>'
        if story_sample:
            code_block=f'<p class="story-credit">{STORY_CREDIT}</p>'
        if s['id']=='next':
            code_block=f'<p class="step-meta"><a href="notebooks/wordlm/{BOOK}.html">Read the illustrated guide</a> · <a href="notebooks/wordlm/wordlm-notebooks.zip" download>Download all five notebooks</a></p>'
        body='. '.join(s['body'].split('. ')[:1 if s['id']=='training-loop' else 2]).rstrip('.')+'.'
        if s['id']=='training-loop':
            body='Repeat this training step on batches of 512 windows.'
        if story_sample or tokenization or s['id']=='batches':
            body=s['body']
        extra_class=(' story-sample' if story_sample else ' tokenization-lesson' if tokenization else ' map-checkpoint' if s.get('map_checkpoint') else '')
        intro=s['id']=='tokenization-intro'
        if intro: extra_class+=' lecture-topic-break topic-midpoint'
        heading='<h3>Tokenization</h3>' if intro else ''
        mobile=f'<div class="tokenization-mobile"><strong>Does “next token” always mean “next word”?</strong><span class="tokenization-example">{escape(ns["tokenization_text"])}</span><span>The same text can become different sequences of tokens.</span></div>' if intro else ''
        sources=' Background: '+'; '.join(label+': '+url for label,url in TOKENIZATION_SOURCES)+'. Notebook rules: notebooks/wordlm/wordlm.py.' if tokenization else ''
        fragments.append(f'''<div class="frame pipeline-lesson{extra_class}" id="s19-pipeline-{s['id']}" data-title="{escape(s['title'],quote=True)}" data-autobuild="off"><script type="text/x-notes">{escape(s['body']+sources)} Notebook step {s['index']}. {escape(s['code'])}</script>{heading}<p class="step-meta">{escape(s['chapter'])} · Step {s['index']} / {len(STAGES)} <a href="notebooks/wordlm/{BOOK}.html#{s['id']}">Notebook step ↗</a></p><div class="step-figure{' master' if master else ' topic-question' if intro else ''}">{s['svg']}{mobile}</div><p class="step-copy">{escape(body)}</p>{code_block}</div>''')
        manifest.append(dict({k:s[k] for k in ['id','title','chapter','index','slide']},
                             map_checkpoint=bool(s.get('map_checkpoint'))))
    (lecture/'src'/'sections'/'sec19_pipeline.html').write_text('\n\n'.join(fragments))
    (out/'lesson-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    export_bundle(out)
    print(json.dumps({'steps':len(STAGES),'notebook':str(ROOT/(BOOK+'.ipynb')),'study_guide':str(out/(BOOK+'.html'))}))

def export_bundle(out):
    from nbconvert import HTMLExporter
    css=(ROOT/'lesson.css').read_text()
    # Explicit allow-list. No private Site files, raw corpus, credentials or work/.
    selected=['wordlm.py','pipeline_maps.py','slow_walkthrough.py','build_slow_lesson.py','code_display.py','make_notebooks.py','walkthrough_cells.py','lesson_evidence.json','story_examples.json','lesson.css','requirements.txt','prepare_data.py','run_experiments.py','README.md']
    selected += [p.name for p in ROOT.glob('0[1-5]_*.ipynb')]
    selected += ['artifacts/'+p.name for p in (ROOT/'artifacts').iterdir() if p.suffix in {'.json','.npz','.csv'} or p.name=='SHA256SUMS']
    selected += ['tests/'+p.name for p in (ROOT/'tests').glob('test_*.py')]
    for name in selected:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name=='README.md':
            # Machine-specific operations notes stay in the private working copy.
            readme=(ROOT/name).read_text()
            if '## DGX Spark launch and resume' in readme:
                begin=readme.index('## DGX Spark launch and resume')
                end=readme.index('## Verification',begin)
                readme=readme[:begin]+'## GPU training\n\nRun the same scripts in a CUDA-enabled PyTorch environment and pass `--device cuda`. No particular host or management setup is required.\n\n'+readme[end:]
            target.write_text(readme)
        elif (ROOT/name).resolve()!=target.resolve():
            shutil.copy2(ROOT/name,target)
    exporter=HTMLExporter(template_name='lab')
    for p in ROOT.glob('0[1-4]_*.ipynb'):
        nb=nbf.read(p,as_version=4)
        body,_=exporter.from_notebook_node(nb)
        body=body.replace('</head>','<style>'+css+'</style></head>').replace('<body>','<body><header class="book-header"><a href="05_training_and_inference_maps.html">← Step-by-step illustrated guide</a> · <a href="wordlm-notebooks.zip" download>Download runnable notebooks</a></header>',1)
        (out/(p.stem+'.html')).write_text(body)
    with zipfile.ZipFile(out/'wordlm-notebooks.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in selected:archive.write(out/name,arcname=name)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lecture-dir',required=True)
    build(parser.parse_args().lecture_dir)
