"""Replace the static attention-map sequence with one trained-patch explorer."""
from html import escape
import json
import re
import numpy as np
from vision1_focus_common import Figures
from vision1_explorer_tour import EXAMPLES


def enhance(b, sections):
    f = Figures(b)
    t, line, arrow = f.t, f.line, f.arrow
    folder = b['ASSETS'] / 'attention-explorer'
    meta = json.loads((folder / 'manifest.json').read_text())
    query, block, head = 74, 4, 1
    values = np.fromfile(folder / f'block-{block:02}.f32', dtype='<f4')
    q = values[:37824].reshape(3, 197, 64)
    k = values[37824:75648].reshape(3, 197, 64)
    scores = q[head-1, query] @ k[head-1].T / 8
    weights = np.exp(scores - scores.max()); weights /= weights.sum()
    top = np.argsort(-weights[1:])[:3] + 1
    body = t(25,30,'1 · Choose a query patch',27) + t(405,30,'2 · Follow its attention weights',27)
    body += f.image(25,65,280,280,f.photo) + f.image(405,65,280,280,f.photo)
    for j in range(196):
        x, y = (j % 14)*20, (j//14)*20
        alpha = float(weights[j+1] / weights[1:].max()) * .9
        body += f'<rect x="{405+x}" y="{65+y}" width="20" height="20" fill="#ffb11e" opacity="{alpha:.4f}"/>'
    r, c = divmod(query-1,14)
    body += f.rect(25+c*20,65+r*20,20,20,'c-q','transparent',0)
    body += arrow(330,205,382,205,'c-q')
    body += t(25,380,'P74 · the dog’s left-side ear patch',22,'c-q')
    body += t(405,380,f'Gold: 0 → {100*weights[1:].max():.2f}% weight',22,'c-k')
    body += t(750,83,'Saved trained-model example',25,'c-e')
    body += t(750,125,'Block 4 · Head 1 · Query P74',24)
    for i,j in enumerate(top):
        body += t(750,185+i*39,f'P{j}: {100*weights[j]:.2f}%',26,'c-k')
    body += t(750,325,f'CLS source: {100*weights[0]:.2f}%',23,'ink-2')
    body += t(25,426,'All 197 source weights sum to 100%. The whole image is available; no causal mask.',25)
    title = 'Explore a trained ViT, one example at a time'
    caption = 'Follow nine guided examples on the trained model. Each preset shows what to notice and one takeaway. Then choose Free exploration to select any patch, block, head or view. Each square is a 16×16 patch.'
    notes = ('What does the gold region mean in this example?\n'
             'Start with the background preset and use Next example. Examples 1–6 compare patch features; '
             '7–9 show attention weights. Read the observation before revealing the takeaway in discussion. '
             'The four corner features nearly coincide; this is not a segmentation guarantee. '
             'After example 9, use Free exploration. The presets retain exact queries, blocks and heads.')
    prose = ('<p>These are saved activations from the same pretrained '
             '<a href="https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k">ViT-Tiny classifier</a> '
             'used throughout this lecture. It predicts Newfoundland for this image with 95.73% probability. '
             'No training runs in the browser. One block loads at a time; the browser computes the selected attention row from saved Q and K.</p>'
             '<p><strong>Attention:</strong> one head’s query compares with all 197 keys. Softmax produces 196 patch weights plus one CLS weight. '
             'The map shows the patch weights without renormalizing them; the CLS weight is reported separately. Gold contrast adapts to each map, '
             'so compare numeric weights across blocks. Clicking a source shows the score, weight, and where its value vector enters the weighted sum.</p>'
             '<p><strong>Feature similarity:</strong> compare the 192-dimensional patch representations after the selected block’s attention, '
             'MLP, and residual additions, before the next normalization. Cosine uses a fixed −1 to 1 scale. The selected patch’s self-match is omitted '
             'from the top-three list. This uses the DINO-style interaction idea with our supervised classifier; '
             '<a href="https://github.com/facebookresearch/dino">DINO</a> learns its features with a different, self-supervised objective. '
             'Neither view establishes which pixels causally determined the class. The next slides intervene on the image to ask a different question.</p>'
             '<p>Keyboard: tab to either image grid, use arrow keys to move between patches, then Enter or Space to select. '
             'Play blocks runs once from block 1 to 12 and stops when you leave the slide. '
             '<a href="notebooks/vision/ATTENTION_EXPLORER.md">Data provenance and reproduction</a>.</p>')
    tour_rows = ''.join('<tr><td>'+str(i+1)+'. '+escape(e['title'])+'</td><td>'+
                        escape(('Feature similarity' if e['mode']=='similarity' else f'Attention, head {e["head"]}')+
                               f' · block {e["block"]} · '+('CLS' if e['query']==0 else f'P{e["query"]}'))+
                        '</td><td>'+escape(e['look'])+'</td><td>'+escape(e['takeaway'])+'</td></tr>'
                        for i,e in enumerate(EXAMPLES))
    prose = ('<p><strong>Start here:</strong> keep the preset fixed, locate the purple query, read the gold map, '
             'then discuss the takeaway. Next example changes the settings for you. Free exploration exposes '
             'the full controls; Guided examples returns to the last preset.</p>'
             '<table class="vp-table"><thead><tr><th>Example</th><th>Settings</th><th>Notice</th><th>Takeaway</th></tr></thead>'
             '<tbody>'+tour_rows+'</tbody></table>')+prose
    examples_json = json.dumps(EXAMPLES, ensure_ascii=False).replace('<', '\\u003c')
    tour_options = ''.join(f'<option value="{i}">{i+1}. {escape(e["title"])}</option>' for i,e in enumerate(EXAMPLES))
    for item in b['FRAMES']:
        if item['id'] == 'read-attention-map': item.update(title=title,caption=caption,notes=notes)
    original = b['frame']('read-attention-map',title,body,caption,notes,prose)
    static = b['svg'](body,title)
    options = ''.join(f'<option value="{i}"{" selected" if i==block else ""}>Block {i}</option>' for i in range(1,13))
    ui = f'''<div id="vit-explorer" class="vix is-guided" data-present="manual" data-keep-state data-base="figures/vision1/attention-explorer" data-image="{f.photo}">
<script type="application/json" data-examples>{examples_json}</script>
<div class="vix-tour-nav">
<div data-tour-controls class="vix-tour-controls"><label>Guided example<select data-example>{tour_options}</select></label>
<button type="button" data-previous disabled>Previous</button><button type="button" data-next>Next example →</button></div>
<button type="button" data-explore>Free exploration</button></div>
<div class="vix-tour-help" data-tour-help>Next example sets everything for you. Gold = more similar features.</div>
<div class="vix-controls" data-free-controls hidden>
<label>View<select data-control="mode"><option value="attention">Attention to keys</option><option value="similarity">Patch feature similarity</option></select></label>
<label>Transformer block<select data-control="block">{options}</select></label>
<label>Attention head<select data-control="head"><option value="1">Head 1</option><option value="2">Head 2</option><option value="3">Head 3</option></select></label>
<button type="button" data-play aria-pressed="false">▶ Play blocks</button></div>
<div class="vix-status"><span data-status role="status">Trained ViT · choose a patch to explore</span><button type="button" data-retry hidden>Retry</button></div>
<div class="vix-live">
<div class="vix-columns">
<div><h4 data-query-title>1 · Locate the purple patch</h4>
<div class="vix-photo"><img src="{f.photo}" alt="The Newfoundland photograph, divided into 196 patches" width="224" height="224"><div class="vix-grid" data-query-grid role="group" aria-label="Choose query patch. Arrow keys move; Enter selects."></div></div>
<div class="vix-selected"><span class="vix-crop" data-query-crop></span><span data-query-name></span></div>
<div class="vix-presets"><button type="button" data-preset="74">Ear</button><button type="button" data-preset="63">Nose</button><button type="button" data-preset="15">Tree</button><button type="button" data-preset="0">CLS</button></div>
</div>
<div><h4 data-map-title>2 · Where does this query read?</h4>
<div class="vix-photo"><img src="{f.photo}" alt="Source patches with a measured heatmap" width="224" height="224"><canvas width="280" height="280" aria-hidden="true"></canvas><div class="vix-grid" data-key-grid role="group" aria-label="Inspect source patch. Arrow keys move; Enter selects."></div></div>
<div class="vix-scale" data-scale></div><div class="vix-legend" data-scale-label></div>
<button type="button" data-cls-weight></button><div class="vix-accounting" data-accounting></div>
</div>
<aside class="vix-tour-panel" data-tour-panel aria-live="polite">
<div class="vix-tour-progress" data-tour-progress></div><h4 data-tour-title></h4>
<p class="vix-observe" data-observe></p>
<div class="vix-tour-fact"><span class="vix-crop" data-tour-crop></span><span data-tour-fact></span></div>
<h5>Takeaway</h5><p data-takeaway></p></aside>
<div class="vix-detail" data-free-detail hidden><h4 data-ranking-label>Strongest patch sources</h4><div data-top></div>
<h4>3 · Inspect one connection</h4>
<div class="vix-selected"><span class="vix-crop" data-source-crop></span><span data-source-name></span></div>
<p data-query-readout></p><p data-calculation></p><p data-result></p><p data-source-note></p></div>
</div>
<div class="vix-meaning" data-meaning></div><div class="vix-guide" data-guide></div>
<div class="vix-provenance">Saved pretrained ViT-Tiny · 224×224 RGB · 14×14 patches · 16×16 pixels per selection · Newfoundland: 95.73%</div>
</div><div class="vix-fallback">{static}<noscript>Enable JavaScript for patch selection. This is the saved block 4, head 1 example.</noscript></div></div>'''
    original = original.replace('<div class="vp-figure">'+static+'</div>',ui)
    assert 'id="vit-explorer"' in original
    result = []
    for section, frames in sections:
        out = []
        for fr in frames:
            key = re.search(r'class="frame[^\"]*" id="([^\"]+)"',fr).group(1)
            if key in {'real-heads','real-depth','real-patch-query'}: continue
            out.append(original if key == 'read-attention-map' else fr)
        result.append((section,out))
    return result
