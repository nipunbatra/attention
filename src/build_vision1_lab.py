"""Generate and execute the companion lab; the expensive training has its own script."""
from pathlib import Path
import contextlib
import io
import json
ROOT=Path(__file__).resolve().parents[1]

def build_lab():
    cells=[];scope={'__file__':str(ROOT/'src/vision1_worksheet.py')};counter=0
    def md(text): cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
    def code(text):
        nonlocal counter
        counter+=1;stream=io.StringIO()
        with contextlib.redirect_stdout(stream):exec(text,scope)
        cells.append({'cell_type':'code','metadata':{},'source':text.splitlines(keepends=True),'execution_count':counter,
          'outputs':[{'output_type':'stream','name':'stdout','text':stream.getvalue().splitlines(keepends=True)}] if stream.getvalue() else []})
    md('''# Vision I — from pixels to a trainable image classifier

Companion to [the complete lecture](../../vision1.html). Work in order: predict, calculate, then run.

**Three settings:** hand-chosen four-patch arithmetic; a full small ViT trained on generated noisy images; a pretrained ViT processing real Oxford-IIIT Pet photographs. Keep their parameters and claims separate.

Needs NumPy and PyTorch. The real-image scripts additionally need timm and Pillow, and download a public checkpoint if it is not cached. Nothing plays audio.
''')
    code('''from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# Works when opened from the repository root, src/, or notebooks/vision/.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "src/vision1_worksheet.py").exists())
''')
    md('## 1. Every worksheet parameter\nFirst inspect W_patch and both heads. Which coordinates do the nonzero entries select?')
    source=(ROOT/'src/vision1_worksheet.py').read_text().split("if __name__ == '__main__':")[0]
    code(source)
    code('''p = parameters()
for key in ['W_patch', 'b_patch', 'positions', 'W_O', 'W_class']:
    print(key, np.array(p[key]), sep='\n')
for number, head in enumerate(p['heads'], 1):
    print('Head', number)
    for key, value in head.items(): print(key, np.array(value), sep='\n')
'''.replace("sep='\n'", "sep='\\n'"))
    md('## 2. Trace the entire forward pass\nCalculate P1’s projected row, one score, and one weighted value before revealing the output.')
    code('''r = forward(p['images']['horizontal'])
for key in ['pixels','patches','content','E']:
    print(key, np.round(r[key], 6), sep='\\n')
for i, head in enumerate(r['heads'], 1):
    print('HEAD', i)
    for key in ['Q','K','V','scores','A','contributions','H']:
        print(key, np.round(head[key], 6), sep='\\n')
for key in ['joined','delta','updated','logits','probability']:
    print(key, np.round(r[key], 6), sep='\\n')
print('Loss', -np.log(r['probability'][0]))
''')
    md('## 3. Independent PyTorch comparison\nPack the two heads into MultiheadAttention. PyTorch Linear stores output-by-input weights, hence the transposes.')
    code('''layer = nn.MultiheadAttention(4, 2, bias=False, batch_first=True,
                              dtype=torch.float64).eval()
with torch.no_grad():
    packed = np.concatenate([np.concatenate([h[k] for h in p['heads']], axis=1).T
                              for k in ['W_Q','W_K','W_V']], axis=0)
    layer.in_proj_weight.copy_(torch.tensor(packed))
    layer.out_proj.weight.copy_(torch.tensor(np.array(p['W_O']).T))
    E = torch.tensor(r['E']).unsqueeze(0)
    delta, weights = layer(E,E,E,average_attn_weights=False)
np.testing.assert_allclose(delta[0], r['delta'], atol=1e-12)
for h in range(2):
    np.testing.assert_allclose(weights[0,h],r['heads'][h]['A'],atol=1e-12)
print('Both head matrices and the projected update agree to 1e-12.')
''')
    md('## 4. Position control\nPredict all four results. Moving contents to different positions differs from reordering complete positioned rows.')
    code('''for name, pixels in p['images'].items():
    for positions in [True,False]:
        result = forward(pixels,positions)
        print(name, 'positions:',positions, 'P(class):',result['probability'])
        if not positions: np.testing.assert_allclose(result['probability'],[.5,.5])
''')
    md('## 5. One learning step that fits on the board\nFreeze the worksheet; add a zero class bias and update only it using SGD with learning rate 0.5.')
    code('''logits = torch.tensor(r['logits'], dtype=torch.float64)
bias = torch.zeros(2, dtype=torch.float64, requires_grad=True)
loss = F.cross_entropy((logits+bias)[None],torch.tensor([0]))
loss.backward()
expected = r['probability']-np.array([1.,0.])
np.testing.assert_allclose(bias.grad,expected,atol=1e-12)
print('Gradient',bias.grad.numpy())
with torch.no_grad(): bias -= .5*bias.grad
print('New bias',bias.detach().numpy())
print('P(class)',(logits+bias).softmax(-1).detach().numpy())
print('Loss before/after',float(loss.detach()),float(F.cross_entropy((logits+bias)[None],torch.tensor([0])).detach()))
''')
    md('## 6. Restore the rest of the block\nThese independent examples explain LayerNorm and the row MLP; they are not inserted into the earlier hand calculation.')
    code('''z = torch.tensor([1.,0.,0.,1.], dtype=torch.float64)
print('LayerNorm', F.layer_norm(z,(4,),eps=1e-5).numpy())
x = torch.tensor([1.,-1.],dtype=torch.float64)
W1 = torch.tensor([[1.,0.,1.],[0.,1.,1.]],dtype=torch.float64)
W2 = torch.tensor([[1.,0.],[0.,1.],[0.,0.]],dtype=torch.float64)
hidden = F.gelu(x@W1,approximate='none')
print('GELU hidden',hidden.numpy())
print('MLP message',(hidden@W2).numpy())
print('Residual',(x+hidden@W2).numpy())
''')
    md('## 7. A complete small ViT\nThe following classes are exactly those used in the actual training experiment. Inspect both residual paths and the final CLS readout.')
    train=(ROOT/'notebooks/vision/train_small_vit.py').read_text()
    code(train[train.index('class Block'):train.index('\ndef evaluate')])
    code('''torch.manual_seed(7)
model = SmallViT(positions=True)
images, labels = dataset(2, seed=99)
logits = model(images)
F.cross_entropy(logits, labels).backward()
for name, parameter in model.named_parameters():
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(),name
print('Input',tuple(images.shape),'logits',tuple(logits.shape))
print('Parameters',sum(p.numel() for p in model.parameters()))
print('Gradients reach every trainable parameter.')
''')
    md('## 8. Conv2d really is the shared patch projection\nCheck flatten-and-linear against the convolution with exactly the same weights. Unfold uses channel-major pixel order.')
    code('''conv = nn.Conv2d(1,16,kernel_size=2,stride=2)
x = torch.randn(3,1,8,8)
patch_rows = F.unfold(x,kernel_size=2,stride=2).transpose(1,2)
a = patch_rows @ conv.weight.flatten(1).T + conv.bias
b = conv(x).flatten(2).transpose(1,2)
torch.testing.assert_close(a,b)
print('Conv2d == shared Linear on unfolded patches:',tuple(a.shape))
''')
    md('''## 9. Reproduce the trained checkpoint results

The training script already ran 80 epochs on 512 training images. Validation selected a checkpoint; these saved weights are evaluated below on the independently generated test split.

To rerun training from initialization:
```bash
python notebooks/vision/train_small_vit.py
```
This overwrites only the two saved small-model checkpoints and their report. The script includes the optimizer, minibatches, validation selection and final test evaluation. Test results below are **one-seed synthetic results**, not a benchmark of natural-image ability.
''')
    code('''test_images, test_labels = dataset(128, seed=33)
for positions in [True,False]:
    fitted = SmallViT(positions).eval()
    filename = 'small-vit-positions.pt' if positions else 'small-vit-no-positions.pt'
    fitted.load_state_dict(torch.load(ROOT/'notebooks/vision'/filename,weights_only=True,map_location='cpu'))
    with torch.inference_mode():
        logits = fitted(test_images)
        correct = int((logits.argmax(-1)==test_labels).sum())
        paired_gap = float((logits.softmax(-1)[0::2]-logits.softmax(-1)[1::2]).abs().max())
    print('positions:',positions,'correct:',correct,'/ 256','max paired probability gap:',paired_gap)
    assert correct == (256 if positions else 128)
''')
    md('''## 10. Real photographs, real measurements

The lecture uses the exact two saved Oxford-IIIT Pet images and the ImageNet checkpoint `timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`. Preprocessing comes from that checkpoint. The 14×14 maps show raw source probabilities for a specified block, head and query; CLS-source mass is reported separately.

Run the scripts to regenerate real measurements:
```bash
uv run --with timm --with pillow python notebooks/vision/run_real_images.py
uv run --with timm --with pillow python notebooks/vision/inspect_real_vit.py
```
The fixed quadrant interventions fill with normalized RGB zero. They measure sensitivity on this input, not causal importance or dataset accuracy.
''')
    code('''results = json.loads((ROOT/'figures/vision1/real-inference.json').read_text())
for result in results['results']: print(result['image_id'],result['top3'])
inspection = json.loads((ROOT/'figures/vision1/inspection.json').read_text())
for record in inspection['attention']:
    total = np.sum(record['patch_weights'])+record['cls_weight']
    np.testing.assert_allclose(total,1.,atol=2e-6)
print('Every recorded attention row sums to one including CLS.')
for result in inspection['occlusion']:
    print(result['region'],result['target_probability'])
''')
    md('''## 11. Predict first; then run the answers

1. Scores after scaling are `[ln 2, 0]`, values are `[2,0]` and `[0,3]`. What arrives?
2. For a 128×128 RGB image, P=16, D=64 and four heads, find N, Q and A shapes.
3. Halve patch width at fixed resolution. How do token count and score count change?
4. Why does permuting already-positioned rows preserve CLS, while moving image contents can change it?
5. Why does a bright attention cell not alone explain a class prediction?
''')
    code('''weights = softmax(np.array([np.log(2),0]))
print('Weights',weights,'message',weights@np.array([[2,0],[0,3]]))
N = (128//16)**2+1
print('E:',(N,64),'Q/head:',(N,16),'A/head:',(N,N))
for P in [32,16,8]:
    N=(224//P)**2+1
    print('Patch size',P,'tokens',N,'scores/head',N*N)
print('Position answer: keep content-position associations when reordering complete rows.')
print('Interpretation answer: values, output projection, residuals and later layers also affect logits.')
''')
    md('''## References and attribution

- [Original ViT paper](https://arxiv.org/abs/2010.11929)
- [D2L: Vision Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html)
- [UvA: Vision Transformer notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html)
- [Stanford CS231n 2025, Lecture 8](https://cs231n.stanford.edu/slides/2025/lecture_8.pdf)
- [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) via the [timm mirror](https://huggingface.co/datasets/timm/oxford-iiit-pet); image attribution and provenance are in figures/vision1/images.json. Images retain CC BY-SA 4.0 attribution and their owners' copyright.

The diagrams and hand worksheet are original to this teaching series. [Parts I](../../part1.html), [II](../../attention.html) and [III](../../part3.html) provide the earlier examples and notation.
''')
    for i,c in enumerate(cells):c['id']=f'vision-lab-{i}'
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':cells}
    (ROOT/'notebooks/vision/03_vision_transformer_lab.ipynb').write_text(json.dumps(book,indent=1)+'\n')
    print(f'Executed {counter} notebook code cells.')
if __name__=='__main__':build_lab()
