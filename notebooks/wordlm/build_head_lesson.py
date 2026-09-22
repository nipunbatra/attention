"""Build the sixth runnable notebook and evidence-led closing lecture frames."""
from pathlib import Path
from html import escape
import json
import nbformat as nbf

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
LABELS={'mlp':'Fixed-window MLP','attention':'One-head attention','multihead':'Four-head attention'}


def build_notebook():
    md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    cells=[md('''# One head and four heads

This notebook extends the Part II attention model. It changes the number of heads, while retaining one attention block, learned absolute positions, the residual update and the prediction MLP. [Part III](../../part3.html) and [Notebook 7](07_multihead_step_by_step.html) walk through multi-head attention from scratch. The broader Transformer material is kept in [optional Part 2B](../../part2b.html).

[Run the browser demo](../../word-lab/) · [Part II slides](../../attention.html?present#s19) · [Download all notebooks](wordlm-notebooks.zip)

Run all cells to inspect the saved experiment on CPU. Training is optional. Keep the support files and artifacts from the ZIP beside this notebook.'''),
    code('''import math
from pathlib import Path
import torch
from IPython.display import display, HTML
from wordlm import read_json, load_model_npz, count_parameters, tokenize
from run_head_comparison import build_model

ROOT = Path.cwd()
assert (ROOT / 'wordlm.py').exists(), 'Open this notebook beside wordlm.py'
results = read_json(ROOT / 'artifacts/heads/comparison.json')
vocab_data = read_json(ROOT / 'artifacts/vocab.json')
itos = vocab_data['itos']
stoi = {word: i for i, word in enumerate(itos)}
display(HTML('<style>body{font-family:"Avenir Next",sans-serif} .jp-Notebook{max-width:1050px;margin:auto} h1,h2{letter-spacing:-.03em} table{font-variant-numeric:tabular-nums} td,th{padding:10px!important}</style>'))'''),
    md('''## The comparison

The MLP reads 64 embeddings in order. Attention instead forms a weighted message from the same context. Both attention models use a 64-dimensional representation. One head uses all 64 coordinates for one weight row. Four heads divide the projections into four groups of 16, each with its own weight row.

Four heads do not quadruple the parameter count here: the Q, K, V and output matrices retain shape 64 × 64. Adding heads at fixed **per-head** width would be a different experiment.'''),
    code('''models = {}
for kind in ['mlp', 'attention', 'multihead']:
    model = build_model(kind)
    metadata, _ = load_model_npz(ROOT / f'artifacts/heads/{kind}_seed11.npz', model)
    model.eval()
    models[kind] = model
    print(kind, count_parameters(model), 'parameters; selected step', metadata['selected_step'])
assert count_parameters(models['attention']) == count_parameters(models['multihead'])'''),
    md('''## Implementation

The initializer reuses the Part II embedding tables, projection matrices and prediction MLP from `CausalAttentionLM`. `split_heads` makes the head axis explicit. Each head scales by the square root of **its own width**, 16 here, rather than the total width 64.

`forward_details` computes every causal query row for inspection. The training loss supervises only the final query, so `forward` computes that row directly. The two results must agree.'''),
    code((ROOT/'multihead.py').read_text()),
    md('''## One prompt, four weight rows

Tokenization produces strings, vocabulary lookup produces IDs, and embedding lookup produces vectors. We add one BOS to this new story, crop to 64 IDs and left-pad unused slots. PAD keys receive zero attention weight.'''),
    code('''prompt = 'a little dog found a red hat'
tokens = tokenize(prompt)
history = [1] + [stoi.get(t, 3) for t in tokens]
context_ids = history[-64:]
context_ids = [0] * (64 - len(context_ids)) + context_ids
X = torch.tensor([context_ids])
print('Tokens:', tokens)
print('Non-padding IDs:', history)
print('Input shape:', tuple(X.shape))
model = models['multihead']
with torch.inference_mode():
    details = model.forward_details(X)
    torch.testing.assert_close(model(X), details['logits'])
for name in ['E', 'Q', 'K', 'V', 'weights', 'messages', 'contextual', 'logits']:
    print(name, tuple(details[name].shape))'''),
    md('''The dimensions are B = 1 example, T = 64 slots, H = 4 heads and d_head = 16. Q/K/V have shape [B,H,T,d_head]. The attention weights have shape [B,H,T,T]. After weighted sums, concatenating the four messages restores width 64.

The following table shows the final query's weights over the real input tokens. Different heads can weight different sources. A weight alone does not tell us which semantic feature a head learned.'''),
    code('''from html import escape
real = X[0].ne(0)
words = [itos[int(i)] for i in X[0, real]]
A = details['weights'][0, :, -1, real]
torch.testing.assert_close(A.sum(-1), torch.ones(4))
table = '<table><tr><th>Head</th>' + ''.join('<th>'+escape(w)+'</th>' for w in words) + '</tr>'
for head, weights in enumerate(A):
    table += '<tr><th>'+str(head+1)+'</th>' + ''.join(f'<td>{w:.3f}</td>' for w in weights) + '</tr>'
display(HTML(table+'</table>'))'''),
    md('''## Cross-entropy and perplexity

For each held-out target, read the probability assigned to the **observed next token**, not the generated token. Its loss is −ln p(target). Average these losses over all targets to obtain cross-entropy in nats per token.

Perplexity is exp(cross-entropy). It is the reciprocal of the geometric mean target probability. If a model assigned equal probability to four alternatives at every step, its perplexity would be four. It is not the literal number of words the model considers, and it is not an accuracy percentage. Compare it only with the same tokenizer and evaluation targets.'''),
    code('''target_probabilities = torch.tensor([0.5, 0.25, 0.125], dtype=torch.float64)
loss_per_target = -target_probabilities.log()
cross_entropy = loss_per_target.mean()
perplexity = cross_entropy.exp()
print('Individual losses:', loss_per_target.tolist())
print('Mean cross-entropy:', round(float(cross_entropy), 4))
print('Perplexity:', round(float(perplexity), 4))
assert abs(float(perplexity) - 4) < 1e-10'''),
    md('''## Measured results

All runs use 6,000 stories split by document, a training-only vocabulary, a 64-token context, 512 examples per update and a 6,000-update ceiling. The three seeds are 11, 29 and 47. Each run restores its best validation checkpoint before scoring the 120,393 test targets.

The four-head model reuses the single-head learning rate and weight decay. These choices were frozen before any test scoring. This is a fixed-settings head-count comparison, not an exhaustive tuning study. The original two-model benchmark remains in `artifacts/benchmark.json`; this repeat is stored separately.'''),
    code('''for kind, summary in results['aggregate'].items():
    print(kind)
    for metric in ['test_loss', 'test_perplexity', 'runtime_seconds']:
        m = summary[metric]
        print(f"  {metric}: {m['mean']:.4f} ± {m['sample_std']:.4f}")
print('Device:', results['environment'])
print('Per-seed checkpoint steps:')
for kind, runs in results['runs'].items():
    print(kind, [r['selected_step'] for r in runs])'''),
    md('''## Training the four-head model

Set `RUN_TRAINING = True` for a short training exercise. `prepare_data.py --profile dgx` downloads the pinned subset if it is not already cached. The download checks the source revision and checksum. A smoke run verifies the loop; it does not reproduce the published scores.

For the full three-seed comparison, run:

```sh
python run_head_comparison.py --data-dir work/data --output-dir work/head-repeat --device auto
```

The script uses CUDA if available, then Apple MPS, then CPU. It saves each completed run so the same command can resume. Full training replaces no bundled checkpoint when a separate output directory is supplied.'''),
    code('''RUN_TRAINING = False
if RUN_TRAINING:
    from wordlm import build_corpus, make_all_windows, TrainConfig, train_model
    corpus = build_corpus(ROOT / 'work/data', 'dgx')
    windows = make_all_windows(corpus, 64)
    fresh_model = build_model('multihead', len(corpus.vocab.itos), seed=11)
    config = TrainConfig(steps=100, batch_size=512, learning_rate=0.003,
                         weight_decay=0.001, seed=11, eval_every=50)
    training_result, _ = train_model(fresh_model, *windows['train'][:2],
                                    *windows['validation'][:2], config)
    print(training_result['validation_loss'])
else:
    print('Training is off. The earlier cells inspect the completed experiment.')'''),
    md('''## Generation and timing

Choose a training-story prefix, a held-out story or an authored outside-domain prompt in the [browser demo](../../word-lab/). Read the actual source story when using a dataset prompt. Unknown words become UNK, so an unfamiliar scientific prompt also tests vocabulary coverage.

The browser downloads the real FP32 ONNX checkpoints, warms them up, then computes a fresh forward pass after each chosen token. It shows actual new-token counts, stopping reason, total generation seconds and milliseconds per model call. An EOS decision is a model call but adds no text token. Downloads and warm-up are excluded. Hardware, runtime, context length, output length and decoding settings all affect time.

More heads may improve held-out loss without improving every continuation. Compare the complete test-set result as well as examples. These small models can repeat words, lose the plot and produce incorrect claims.''')]
    notebook=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    nbf.write(notebook,ROOT/'06_multihead_comparison.ipynb')


def frame(key,title,body):
    body=body.replace('<table','<div class="lab-table-wrap"><table').replace('</table>','</table></div>')
    return f'<div class="frame lab-comparison" id="s19-lab-{key}" data-title="{escape(title)}" data-autobuild="off"><script type="text/x-notes">Measured evidence: notebooks/wordlm/artifacts/heads/comparison.json and word-lab/models/metadata.json. Three fresh runs per model, frozen settings, same test targets. Live browser inference, not replayed text.</script>{body}</div>'


def build_slides():
    result=json.loads((ROOT/'artifacts/heads/comparison.json').read_text())
    meta=json.loads((REPO/'word-lab/models/metadata.json').read_text())
    rows=''
    for k,label in LABELS.items():
        a=result['aggregate'][k];m=meta['models'][k]
        rows+=f'<tr><th>{label}</th><td>{m["parameters"]:,}</td><td>{m["bytes"]/1e6:.2f} MB</td><td>{a["runtime_seconds"]["mean"]:.1f} ± {a["runtime_seconds"]["sample_std"]:.1f} s</td></tr>'
    score_rows=''
    for k,label in LABELS.items():
        a=result['aggregate'][k]
        score_rows+=f'<tr><th>{label}</th><td>{a["test_loss"]["mean"]:.3f} ± {a["test_loss"]["sample_std"]:.3f}</td><td>{a["test_perplexity"]["mean"]:.2f} ± {a["test_perplexity"]["sample_std"]:.2f}</td></tr>'
    frames=[frame('intro','Three trained models', '<p>All three read the same 64-token window.</p><table><tr><th>Model</th><th>How it reads the context</th></tr><tr><td>MLP</td><td>Concatenate 64 embeddings in slot order</td></tr><tr><td>One head</td><td>One attention weight row, 64 matching coordinates</td></tr><tr><td>Four heads</td><td>Four weight rows, 16 coordinates per head</td></tr></table><p>Both attention models add learned absolute position embeddings. Each has one attention block and the same prediction MLP.</p><p class="small">A preview of Part III. These models do not yet include a full Transformer’s normalization, block FFN or stacked blocks.</p>'),
      frame('ce','Cross-entropy: probability of the observed next token','<p>For each test prefix, read the probability assigned to the word that actually comes next.</p><table><tr><th>Target probability</th><th>Loss: −ln p(target)</th></tr><tr><td>0.500</td><td>0.693</td></tr><tr><td>0.250</td><td>1.386</td></tr><tr><td>0.125</td><td>2.079</td></tr></table><p>Cross-entropy = (0.693 + 1.386 + 2.079) / 3 = <strong>1.386 nats per token</strong>.</p><p>Higher probability for the correct target gives lower loss.</p>'),
      frame('ppl','Perplexity: putting the loss on another scale','<p class="eq">$\\mathrm{PPL}=\\exp(L)$, where $L$ is cross-entropy.</p><p>In our three-target example: exp(1.386) = <strong>4</strong>.</p><p>This is the same loss as assigning probability 1/4 to every observed target. Perplexity is the reciprocal of the geometric mean target probability.</p><p>It is not an accuracy percentage or a literal count of candidate words. Lower is better when the tokenizer and evaluation targets are the same.</p>'),
      frame('scores','Test loss and perplexity',f'<table><tr><th>Model</th><th>Cross-entropy ↓</th><th>Perplexity ↓</th></tr>{score_rows}</table><p>Mean ± sample SD over three seeds, on the same 120,393 held-out targets. Each run restores its best validation checkpoint.</p><p class="small">6,000 stories, training-only vocabulary, 64-token context, batch size 512, 6,000-update ceiling. Four heads reuse the one-head optimizer settings. No test-set tuning.</p>'),
      frame('costs','Parameters, model files and training time',f'<table><tr><th>Model</th><th>Parameters</th><th>FP32 ONNX</th><th>Training per seed</th></tr>{rows}</table><p>Four heads split the same 64 coordinates. The attention models therefore have equal parameter counts.</p><p class="small">Apple M2 Max / MPS. Training time includes validation checks, excludes data preparation and final test scoring. Mean ± sample SD, three seeds. File sizes use decimal MB and include graph metadata.</p>')]
    prompts=json.loads((REPO/'word-lab/examples.json').read_text())['prompts']
    samples=json.loads((REPO/'word-lab/saved-examples.json').read_text())['examples']
    for pid,title in [('train-588307','A prefix from a training story'),('instructions','A prompt outside the story task')]:
        p=next(p for p in prompts if p['id']==pid)
        sample_rows=''.join(f'<tr><th>{LABELS[k]}</th><td>{escape(next(s["continuation"] for s in samples if s["prompt_id"]==pid and s["model"]==k))}</td></tr>' for k in LABELS)
        frames.append(frame(pid,title,f'<p><strong>Prompt:</strong> {escape(p["prompt"])}</p><table class="lab-samples">{sample_rows}</table><p class="small">Recorded greedy outputs from seed 11, up to 24 new tokens. {escape(p["note"])} All presets were chosen before inspecting model outputs.</p>'))
    timing_path=REPO/'word-lab/browser-timing.json'
    if timing_path.exists():
        timing=json.loads(timing_path.read_text());tr=''
        for k in LABELS:
            r=timing['models'][k]
            tr+=f'<tr><th>{LABELS[k]}</th><td>{r["new_tokens"]}</td><td>{r["generation_ms_median"]/1000:.3f} s</td><td>{r["ms_per_call_median"]:.2f} ms</td></tr>'
        frames.append(frame('timing','Generation time in a real browser',f'<table><tr><th>Model</th><th>New tokens</th><th>Generation</th><th>Per model call</th></tr>{tr}</table><p>Median of {timing["repetitions"]} warm runs. {escape(timing["device"])}. {escape(timing["backend"])}. Greedy decoding, 24-token limit.</p><p class="small">Prompt: {escape(timing["prompt"])}. Downloads and warm-up excluded. EOS can shorten an output. The live demo reports timings on your own device.</p>'))
    frames.append(frame('play','The models in your browser','<p><a class="lab-live-link" href="word-lab/" target="_blank" rel="noopener">Open the live model comparison ↗</a></p><p>Choose a training story, a held-out story or an outside-domain prompt. The browser computes new predictions with WebGPU or WASM.</p><p>Compare the outputs, unknown words, actual token counts and generation timings. Greedy decoding keeps the demonstration repeatable.</p><p><a href="notebooks/wordlm/06_multihead_comparison.html">Notebook 6: one head and four heads ↗</a></p>'))
    css='<style>.lab-comparison table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;margin:1.1em 0}.lab-comparison th,.lab-comparison td{text-align:left;padding:.45em .65em;border-bottom:1px solid var(--line,#d8dce5)}.lab-comparison .lab-samples{font-size:.86em}.lab-comparison .lab-samples th{width:23%}.lab-comparison .small{font-size:.75em;color:#50586a}.lab-live-link{font-size:1.35em}</style>'
    css+='<style>.lab-table-wrap{max-width:100%;overflow-x:auto}.lab-table-wrap table{min-width:620px}body.present .lab-table-wrap{overflow:visible}body.present .lab-table-wrap table{min-width:0}</style>'
    (REPO/'src/sections/sec19_lab.html').write_text(css+'\n'+'\n'.join(frames))


if __name__=='__main__':
    import sys
    if '--slides-only' not in sys.argv:build_notebook()
    build_slides()
