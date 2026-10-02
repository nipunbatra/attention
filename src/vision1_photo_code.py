"""SVG code and shapes for the same 224×224 RGB classifier."""
import re
from html import escape
from textwrap import dedent, indent
from vision1_focus_common import Figures


def build(b):
    f=Figures(b)
    t,g,box,arrow,line=f.t,f.g,f.box,f.arrow,f.line
    source=(b['ROOT']/'notebooks/vision/vit_image_classifier.py').read_text()
    def snippet(name):
        return dedent(re.search(r'# slide:'+name+r'\n(.*?)\s*# endslide',source,re.S).group(1)).strip()
    link='<a href="notebooks/vision/vit_image_classifier.py">Complete teaching implementation</a>.'
    premise='The code keeps the lecture architecture: patch size 16, D=192, three 64-wide heads, 12 blocks and 1,000 outputs. It is randomly initialized; the saved dog prediction uses a separate pretrained checkpoint. '
    def add(key,title,code,diagram,caption,question,point,prose='',wide=False):
        body=(f.code(code,x=35,y=45,width=1090,size=24,spacing=37) if wide
              else f.code(code,size=21,spacing=34))+diagram
        mobile_code=code
        if 'return ' in code:
            method='embed' if key.startswith('code-photo-tokens') else 'forward'
            mobile_code=f'def {method}(self, x):\n'+indent(code,'    ')
        return f.add(key,title,body,caption,question,point,premise+prose+link,
                     '<pre><code>'+escape(mobile_code)+'</code></pre><p>'+escape(point)+'</p>')

    code='# rgb: (224, 224, 3), RGB per pixel\nx = rgb.permute(2, 0, 1)  # 3, 224, 224\nx = x.unsqueeze(0)       # 1, 3, 224, 224\n# Patch order: all R, then G, then B'
    diagram=f.image(815,33,210,210,f.photo)+t(920,280,'Height × width × RGB',23,'ink-2','middle')
    diagram+=box(760,317,370,['PyTorch: B × C × H × W','B × 3 × 224 × 224'],size=24)
    add('code-photo-input','Keep the same photograph and add the batch axis',code,diagram,
        'The prepared photograph is 224×224 with three RGB channels. PyTorch puts channels before height and width, then adds a batch axis. B counts images; it does not count patches.',
        'Which axis says how many photographs are in the batch?', 'The leading axis B. One photograph uses B=1; a batch of photographs uses the same model parameters for each.',
        'Here rgb is already a resized/cropped, float, checkpoint-normalized H×W×3 tensor. permute changes axis order, not pixel values. '
        'Use the checkpoint’s prescribed normalization for pretrained inference. The diagram shows the saved model-input crop for continuity; this snippet does not run inference.')

    from vision1_patch_projection_code import build as patch_projection
    f.frames.update(patch_projection(b))

    def shape_flow(labels, colors=None):
        colors = colors or ['c-e'] * len(labels)
        width = 1090 / len(labels) - 40
        body = ''
        for i, (label, color) in enumerate(zip(labels, colors)):
            x = 35 + i * (width + 40)
            body += box(x, 338, width, label, color, h=80, size=25)
            if i < len(labels)-1:
                body += arrow(x+width+7, 378, x+width+32, 378, color)
        return body

    embed=snippet('embed').splitlines()
    add('code-photo-tokens','First turn the feature grid into patch rows',
        '\n'.join(embed[:5]),
        shape_flow([['Feature grid','B × 192 × 14 × 14'], ['Flatten spatial axes','B × 192 × 196'], ['Features last','B × 196 × 192']]),
        'Flatten combines the 14×14 spatial grid into 196 locations. Transpose puts the 192 features last. Each row now describes one patch.',
        'Did flatten mix images or merge their features?',
        'No. flatten(2) starts at axis 2. The batch and 192 feature channels remain separate.',
        'These are the first lines of embed. The next slide continues from rows with shape B×196×192.',wide=True)

    code='# Continue with rows: (B, 196, 192)\n'+ '\n'.join(embed[5:])
    code+='\n\n# self.cls: (1, 1, 192)   self.pos: (1, 197, 192)'
    add('code-photo-tokens-cls','Then prepend CLS and add position',code,
        shape_flow([['Patch rows','B × 196 × 192'], ['Prepend one CLS','B × 197 × 192'], ['Add position','B × 197 × 192']], ['c-e','c-q','c-q']),
        'Expand reuses the learned CLS start across the batch. Concatenation adds one row. Position addition changes the numbers while keeping 197 rows and 192 features.',
        'Which operation changes the number of rows?',
        'torch.cat adds CLS along dim=1. Adding positions leaves the shape unchanged.',
        'The complete implementation registers cls and pos as nn.Parameter tensors. Their size-1 batch axes broadcast across images. expand does not create B independently learned CLS parameters.',wide=True)

    code='B, N, D = x.shape                         # B, 197, 192\nqkv = self.qkv(x)                        # B, 197, 576\nqkv = qkv.reshape(B, N, 3, 3, 64)        # B, N, QKV, heads, features\nq, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)\n# q, k, v: each (B, 3, 197, 64)'
    add('code-photo-attention','Make queries, keys and values for three heads',code,
        shape_flow([['Input rows','B × 197 × 192'], ['QKV projection','B × 197 × 576'], ['Split Q, K, V','each B × 3 × 197 × 64']], ['c-e','c-q','c-q']),
        'Project each row once to produce all queries, keys and values. Split the 576 outputs into three roles, each with three heads of width 64.',
        'What do the two size-3 axes mean?',
        'One selects Q, K or V. The other selects the attention head. B always remains a separate image axis.',
        'This is the first part of Attention.forward, written with a named qkv intermediate. self.qkv is Linear(192,576). N=197 includes CLS.',wide=True)

    code='scores = (q @ k.transpose(-2, -1)) / 8  # B, 3, 197, 197\nweights = scores.softmax(dim=-1)       # B, 3, 197, 197\nmessages = weights @ v                # B, 3, 197, 64'
    add('code-photo-attention-messages','Compute one message for every query in every head',code,
        shape_flow([['Query–key scores','197 × 197'], ['Softmax over sources','197 × 197'], ['Weighted value sums','197 × 64']], ['c-k','c-k','c-v']),
        'For each image and head, every query scores all 197 sources. Softmax normalizes each query row. Multiplying by V combines the sources into one 64-feature message per query.',
        'Along which axis do the weights sum to one?',
        'The last axis: source keys. Every receiving query gets its own distribution.',
        'Shapes in the diagram show one image and one head; the code retains B and all three heads. The scale is sqrt(64)=8. No causal mask is needed.',wide=True)

    code='joined = messages.transpose(1, 2)     # B, 197, 3, 64\njoined = joined.reshape(B, N, D)      # B, 197, 192\nreturn self.proj(joined)             # B, 197, 192'
    add('code-photo-attention-join','Join the head messages and project back to 192',code,
        shape_flow([['Three messages per row','3 × 64'], ['Concatenate features','192'], ['Output projection','192']], ['c-v','c-v','c-e']),
        'Move the head axis beside its features, then join 3×64 into 192. A learned output projection mixes these features before the residual addition.',
        'Do we concatenate different images or different query rows?',
        'Neither. Each image and query keeps its own three head messages. Only head features are joined.',
        'This completes Attention.forward. self.proj is Linear(192,192). The output is the attention update; the block adds the original row on the next slides.',wide=True)

    add('code-photo-block-layers','Build the layers inside one Transformer block',snippet('block-init'),
        shape_flow([['Attention','mix information across rows'], ['MLP','192 → 768 → 192']], ['c-q','c-v']),
        'Create two normalization layers, one attention module and one MLP. The MLP expands and then restores the feature width for each row.',
        'Does the 768-wide hidden layer change the number of patches?',
        'No. Only the feature axis expands. B and N stay unchanged.',
        'These assignments belong in Block.__init__. Attention contains the QKV and output projections just shown. MLP weights are shared across rows within this block.',wide=True)

    code='# x: (B, 197, 192)\n'+snippet('block-forward')
    add('code-photo-block','Use the two residual paths in order',code,
        shape_flow([['Input x','B × 197 × 192'], ['Attention + input','B × 197 × 192'], ['MLP + input','B × 197 × 192']]),
        'Normalize, compute the attention update, and add the input. Then normalize the updated rows, compute the MLP update, and add them again.',
        'Which x enters the second line?',
        'The x already updated by attention. Each branch returns the same shape as the input it is added to.',
        'These lines are Block.forward. LayerNorm and MLP act separately on each row. Attention mixes information between rows of the same image.',wide=True)

    add('code-photo-stack','Create twelve blocks with separate learned parameters',snippet('stack'),
        shape_flow([['Block 1','B × 197 × 192'], ['Blocks 2–11','B × 197 × 192'], ['Block 12','B × 197 × 192']]),
        'ModuleList creates twelve distinct blocks. The final normalization and class head read the result of the entire stack.',
        'Are these twelve calls to one shared set of weights?',
        'No. The list comprehension creates twelve separate Block objects.',
        'These assignments belong in ImageClassifier.__init__. num_classes defaults to 1000 for the ImageNet architecture. Changing the label vocabulary changes the head output width.',wide=True)

    code='rows = self.embed(x)                 # B, 197, 192\nfor block in self.blocks:\n    rows = block(rows)               # B, 197, 192\nsummary = self.norm(rows)[:, 0]      # B, 192: final CLS\nreturn self.head(summary)           # B, 1000 logits'
    add('code-photo-readout','Run the stack, then read the final CLS',code,
        shape_flow([['All rows through 12 blocks','B × 197 × 192'], ['Normalize; select CLS','B × 192'], ['Class head','B × 1000']], ['c-e','c-q','c-e']),
        'Feed each block’s output into the next. Normalize the final rows and select CLS for each image. The class head returns 1,000 scores.',
        'Why is there no softmax inside this forward method?',
        'Cross-entropy accepts logits directly. During inference, logits.softmax(-1) converts scores into class probabilities.',
        'This is ImageClassifier.forward after the input assertion. A new two-class head, introduced in the next section, instead produces B×2 scores.',wide=True)

    diagram=box(790,31,310,['Images B × 3 × 224 × 224','labels B'],size=21)+arrow(945,115,945,147)
    diagram+=box(790,161,310,['Forward → cross-entropy','one scalar loss'],'c-a',size=23)+arrow(945,244,945,276,'c-a')
    diagram+=box(790,289,310,['backward: gradients','step: parameters change'],'c-a',size=22)
    add('code-photo-training','Connect the loss diagram to one training step',snippet('training'),diagram,
        'Cross-entropy consumes logits and class-index labels. Backward computes gradients; the optimizer updates the parameters it owns. Clear previous gradients before the next batch. This code shows the procedure without claiming a trained result.',
        'Should we apply softmax before cross_entropy?', 'No. It already includes log-softmax. labels is a length-B integer tensor; the model returns B×1000 logits for this example.',
        'This function is provided for teaching and is not called while generating the slides. Create an optimizer for the selected trainable parameters before calling it. '
        'For inference use model.eval() and torch.no_grad(); omit labels, backward and optimizer.step. '+
        '<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html">PyTorch cross_entropy</a>.')
    return list(f.frames.values())
