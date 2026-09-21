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
from slow_walkthrough import STAGES, initial_namespace, render_figure
from pipeline_maps import pipeline_svg

ROOT = Path(__file__).resolve().parent
BOOK = '05_training_and_inference_maps'
LIVE = 'https://nipunbatra.github.io/attention/'

# Slides show only the operation being discussed; the notebook retains all checks.
SLIDE_CODE = {
 'shapes':'B, w, C = 2, 4, 10\nd, h, d_k, d_v = 4, 8, 3, 2',
 'positions':'token_rows = attention.token_embedding(X)\nposition_rows = attention.position_embedding(torch.arange(w))\nE = token_rows + position_rows[None, :, :]',
 'windows-first':'visible = ids[max(0, t-w):t]\nx = [PAD] * (w-len(visible)) + visible\ny = ids[t]',
 'gradient':'loss.backward()\ngrad = mlp.vocab_head.weight.grad[9, 0]',
 'update':'optimizer = torch.optim.SGD(mlp.parameters(), lr=0.1)\noptimizer.step()  # every trainable parameter',
 'decode':'next_p = next_logits.softmax(-1)\ngreedy_id = next_p.argmax()\nsample_id = torch.searchsorted(next_p.cumsum(0), torch.tensor(0.8))',
 'append':'if chosen == vocab.eos_id: break\nhistory.append(chosen)\nkept = history[-w:]  # prepare the next window',
 'training-loop':'optimizer.zero_grad()\nloss = F.cross_entropy(model(X), y)\nloss.backward()\noptimizer.step()',
}

def notebook_cells(md, code, setup=None):
    cells=[md('''# 5 · From a story to a trained predictor

Trace the same two examples through an MLP and a one-layer attention model. Start
with data, not equations. Every numbered section has one calculation, an actual
numeric figure, and a link to its lecture slide. Run the cells in order.

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
        if s.get('check'):
            q,a=s['check'];text+=f'\n\n<details><summary>Check yourself: {q}</summary>{a}</details>'
        cells.append(md(text))
        cell=code(s['code']+f"\n\nshow_figure('{s['id']}', globals())")
        cell.metadata['lesson_stage']=s['id']
        cells.append(cell)
    return cells

def build(lecture):
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
    css=(ROOT/'lesson.css').read_text()
    chunks=[];chapters={}
    for s in results:
        chapters.setdefault(s['chapter'],s['id'])
        checks=''
        if s.get('check'):
            q,a=s['check'];checks=f'<aside class="check"><strong>Pause and predict</strong><p>{escape(q)}</p><details><summary>Show explanation</summary><p>{escape(a)}</p></details></aside>'
        map_html=''
        if s['focus']:
            mode='inference' if s['chapter'].startswith('6.') and s['id']!='evaluation' else 'training'
            focused=pipeline_svg(s['kind'],mode,s['focus'])
            map_html=f'<details class="route-map"><summary>Where are we on the full {s["kind"].upper()} map?</summary><div class="figure-wrap">{focused}</div></details>'
        result_html=f'<div class="code-label">Printed output</div><pre class="output">{escape(s["stdout"])}</pre>' if s['stdout'] else ''
        chunks.append(f'''<section class="lesson-step" id="{s['id']}">
<div class="step-meta"><span>{escape(s['chapter'])} · Step {s['index']:02d} / {len(STAGES)}</span><a href="../../attention.html?present#s19/{s['slide']}/0">Open matching slide ↗</a></div>
<h2>{escape(s['title'])}</h2><p>{escape(s['body'])}</p>
<figure><div class="figure-wrap">{s['svg']}</div><figcaption>Same figure as the lecture. Numeric values are computed from the code below.</figcaption></figure>
{map_html}<div class="code-label">Python · run after the previous step</div><pre><code>{escape(s['code'])}</code></pre>{result_html}{checks}</section>''')
    nav=''.join(f'<li><a href="#{key}">{escape(ch.split(". ",1)[1])}</a></li>' for ch,key in chapters.items())
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>From a story to a trained predictor · Part II companion</title><style>{css}</style></head><body>
<header class="book-header"><div class="eyebrow">Attention and language · Notebook 5</div><h1>From a story to a trained predictor</h1><p>One sentence. Seven training examples. Two models. Follow the data, every tensor and one learning step before generating a new token.</p><div class="notebook-links"><a href="wordlm-notebooks.zip" download>Download all notebooks + support files</a><a href="{BOOK}.ipynb" download>Download this notebook</a><a href="../../attention.html?present#s19/4/0">Lecture slides ↗</a></div></header>
<div class="book-layout"><nav class="chapters" aria-label="Chapters"><strong>Work through the example</strong><ol>{nav}</ol><div class="download"><a href="#run">Run it yourself</a><a href="01_words_to_probabilities.html">1 · Train the MLP</a><a href="03_causal_attention_from_scratch.html">3 · Train attention</a><a href="04_what_did_the_models_learn.html">4 · Compare trained models</a></div></nav><main>
<section id="run" class="run-guide"><h2>Before you start</h2><p>The small example uses <code>B=2</code> examples, <code>w=4</code> context slots and <code>C=10</code> vocabulary items. Every number is generated by the supplied code, not drawn by hand. The final measured comparison uses a separate 6,000-story corpus and trained models.</p><p>Unzip the download, open a terminal in that folder and run:</p><pre>python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab {BOOK}.ipynb</pre><p>Then choose <strong>Run → Run All Cells</strong>. This notebook needs no GPU or data download. Notebooks 1 and 3 explain the real corpus and training runs. Keep <code>wordlm.py</code> and the other helpers beside the notebooks.</p></section>
{''.join(chunks)}<footer class="footer">TinyStories: <a href="https://huggingface.co/datasets/roneneldan/TinyStories">dataset and CDLA-Sharing-1.0 terms</a>. Only a short source excerpt is shown. No raw story corpus is redistributed in this download. Benchmark source and checksums are included. The measured device was Apple MPS, not DGX Spark.</footer></main></div></body></html>'''
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
#s19 .pipeline-lesson .step-copy{font-size:24px;line-height:1.4;margin:8px 0 12px}
#s19 .pipeline-lesson .step-meta{font-size:18px;color:var(--ink-2);margin:8px 0}
#s19 .pipeline-lesson .step-meta a{float:right}
#s19 .pipeline-lesson pre{font-size:21px;line-height:1.4;padding:10px 16px;margin:10px 0;background:var(--paper-2,#eef0f4);white-space:pre-wrap;max-width:100%;overflow-x:auto}
#s19 .pipeline-lesson .step-code{display:block}
#s19 .pipeline-lesson .step-code a{font-size:19px}
body:not(.present) #s19 .pipeline-lesson{padding:30px 0;border-bottom:1px solid var(--line)}
@media(max-width:650px){body:not(.present) #s19 .step-figure svg{min-width:900px}body:not(.present) #s19 .pipeline-lesson .step-code{grid-template-columns:1fr}body:not(.present) #s19 .pipeline-lesson .step-copy{font-size:18px}}
</style>
<!--PIPELINE_TEMPLATES-->
<div class="frame pipeline-lesson lecture-topic-break" id="s19-pipeline-break" data-title="From a story to a trained predictor" data-autobuild="off">
<script type="text/x-notes">Use this as a slow, optional lab walkthrough over multiple sessions. The seven chapters connect real corpus provenance to a tiny, fully executable batch. Every figure also lives beside its code in Notebook 5. Distinguish toy arithmetic from saved benchmark evidence.</script><h3>From a story to a trained predictor</h3><p class="topic-recap">We have followed one attention calculation.</p><p class="topic-question">Now follow the data, each tensor, one learning step and the generation loop. <a href="notebooks/wordlm/05_training_and_inference_maps.html">Open the illustrated companion ↗</a></p></div>''']
    manifest=[]
    for s in results:
        master=s['id'] in {'mlp-map','attention-map','mlp-inference','attention-inference'}
        excerpt=SLIDE_CODE.get(s['id'],'\n'.join(s['code'].splitlines()[:3]))
        code_block='' if master else f'<div class="step-code"><pre><code>{escape(excerpt)}</code></pre></div>'
        if s['id']=='next':
            code_block=f'<p class="step-meta"><a href="notebooks/wordlm/{BOOK}.html">Read the illustrated guide</a> · <a href="notebooks/wordlm/wordlm-notebooks.zip" download>Download all five notebooks</a></p>'
        body='. '.join(s['body'].split('. ')[:1 if master or s['id']=='training-loop' else 2]).rstrip('.')+'.'
        if s['id']=='training-loop':
            body='Repeat this training step on batches of 512 windows.'
        fragments.append(f'''<div class="frame pipeline-lesson" id="s19-pipeline-{s['id']}" data-title="{escape(s['title'],quote=True)}" data-autobuild="off"><script type="text/x-notes">{escape(s['body'])} Notebook step {s['index']}. {escape(s['code'])}</script><p class="step-meta">{escape(s['chapter'])} · Step {s['index']} / {len(STAGES)} <a href="notebooks/wordlm/{BOOK}.html#{s['id']}">Notebook step ↗</a></p><div class="step-figure{' master' if master else ''}">{s['svg']}</div><p class="step-copy">{escape(body)}</p>{code_block}</div>''')
        manifest.append({k:s[k] for k in ['id','title','chapter','index','slide']})
    (lecture/'src'/'sections'/'sec19_pipeline.html').write_text('\n\n'.join(fragments))
    (out/'lesson-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    export_bundle(out)
    print(json.dumps({'steps':len(STAGES),'notebook':str(ROOT/(BOOK+'.ipynb')),'study_guide':str(out/(BOOK+'.html'))}))

def export_bundle(out):
    from nbconvert import HTMLExporter
    css=(ROOT/'lesson.css').read_text()
    # Explicit allow-list. No private Site files, raw corpus, credentials or work/.
    selected=['wordlm.py','pipeline_maps.py','slow_walkthrough.py','build_slow_lesson.py','make_notebooks.py','walkthrough_cells.py','lesson_evidence.json','lesson.css','requirements.txt','prepare_data.py','run_experiments.py','README.md']
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
