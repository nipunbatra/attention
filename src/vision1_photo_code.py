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
    def add(key,title,code,diagram,caption,question,point,prose=''):
        body=f.code(code,size=21,spacing=34)+diagram
        mobile_code=code
        if key in ['code-photo-block','code-photo-readout']:
            init,forward=code.split('\n\n',1)
            mobile_code='def __init__(self'+(', num_classes=1000' if key.endswith('readout') else '')+'):\n'+indent(init,'    ')
            mobile_code+='\n\ndef forward(self, x):\n'+indent(forward,'    ')
        elif 'return ' in code:
            method='embed' if key=='code-photo-tokens' else 'forward'
            mobile_code=f'def {method}(self, x):\n'+indent(code,'    ')
        return f.add(key,title,body,caption,question,point,premise+prose+link,
                     '<pre><code>'+escape(mobile_code)+'</code></pre><p>'+escape(point)+'</p>')

    code='# Prepared RGB image: 224 × 224 × 3\nx = rgb.permute(2, 0, 1)\nx = x.unsqueeze(0)\n# x.shape == (1, 3, 224, 224)'
    diagram=f.image(815,33,210,210,f.photo)+t(920,280,'Height × width × RGB',23,'ink-2','middle')
    diagram+=box(760,317,370,['PyTorch: B × C × H × W','B × 3 × 224 × 224'],size=24)
    add('code-photo-input','Keep the same photograph and add the batch axis',code,diagram,
        'The prepared photograph is 224×224 with three RGB channels. PyTorch puts channels before height and width, then adds a batch axis. B counts images; it does not count patches.',
        'Which axis says how many photographs are in the batch?', 'The leading axis B. One photograph uses B=1; a batch of photographs uses the same model parameters for each.',
        'Here rgb is already a resized/cropped, float, checkpoint-normalized H×W×3 tensor. permute changes axis order, not pixel values. '
        'Use the checkpoint’s prescribed normalization for pretrained inference. The diagram shows the saved model-input crop for continuity; this snippet does not run inference.')

    # One complete slide opens Conv2d: a learned filter bank, stride and output grid.
    body=f.code(snippet('conv'),x=40,y=40,width=1070,size=24,spacing=32)
    body+=f.image(40,162,196,196,f.photo)
    for i in range(1,14):
        body+=line(40+i*14,162,40+i*14,358,'card',.6)+line(40,162+i*14,236,162+i*14,'card',.6)
    body+=f.rect(124,218,14,14,'c-q','transparent',0)+t(138,396,'14 × 14 locations',24,'ink-2','middle')
    body+=arrow(244,244,301,244,'c-q')+box(315,180,260,['One patch: 3 × 16 × 16','768 input numbers'],'c-q',size=22)
    body+=arrow(585,222,631,222,'c-v')+box(645,180,460,['192 learned filters, each 3 × 16 × 16','one weighted sum + bias per filter'],'c-v',size=22)
    body+=t(650,303,'At every location: 768 inputs → 192 features',24,'c-v')
    body+=t(315,348,'Stride 16 moves by one patch; the filters are shared.',26)
    body+=t(315,402,'Output: B × 192 × 14 × 14',31,'c-e')
    f.add('code-photo-conv','Use nn.Conv2d for the shared patch projection',body,
        'A 16×16 kernel spans all three channels. Each of 192 filters computes one feature, including a bias. Stride 16 gives non-overlapping patches. This is the same shared affine patch projection, evaluated across the image.',
        'Does Conv2d merely cut the image into patches?', 'It also learns the projection: 192 filters, each with 768 weights and one bias. The output grid has one 192-feature vector at each patch location.',
        premise+'The weight tensor has shape [192,3,16,16] and the bias has shape [192]. There are 192×768+192=147,648 parameters. '
        'With padding=0 and dilation=1, each spatial output is (224−16)/16+1=14. '
        'Flatten each filter in the same channel/pixel order as its input patch: F.linear(patches, conv.weight.flatten(1), conv.bias) equals the Conv2d result after reshaping. '
        'This is patchification plus a learned projection, not a deep convolutional feature extractor. '
        'If flattening RGB-interleaved pixels instead, permute the matching weight columns too. '
        '<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html">PyTorch Conv2d documentation</a> · '
        '<a href="https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py">timm PatchEmbed uses Conv2d with kernel and stride equal to patch size</a>. '+link,
        '<pre><code>'+escape(snippet('conv'))+'</code></pre><p>Input: B×3×224×224. Weights: 192×3×16×16. '
        'Output: B×192×14×14. Each location contains 192 learned patch features; stride 16 keeps patches non-overlapping.</p>')

    code=snippet('embed')
    diagram=box(770,18,355,['Convolution output','B × 192 × 14 × 14'])+arrow(947,100,947,135)
    diagram+=box(770,149,355,['flatten spatial; transpose','B × 196 × 192'])+arrow(947,230,947,268)
    diagram+=box(770,281,355,['prepend CLS; add position','B × 197 × 192'],'c-q',size=23)
    code+='\n\n# cls parameter: (1, 1, 192)\n# pos parameter: (1, 197, 192)'
    add('code-photo-tokens','Turn the feature grid into the 197 input rows',code,diagram,
        'Flatten the two grid axes, then put features last. Prepend the shared CLS row and add the learned position table. Broadcasting reuses these parameters across images; every image keeps its own activations.',
        'Which operation increases the row count from 196 to 197?', 'Concatenating CLS along dim=1 adds one row. Adding positions changes values without changing the shape.',
        'The complete implementation registers cls and pos as nn.Parameter tensors. flatten(2) combines only the two spatial axes. '
        'It does not flatten the batch into the token axis. transpose(1,2) changes B×192×196 into B×196×192.')

    diagram=box(770,22,355,['Q, K, V — each','B × 3 × 197 × 64'],'c-q')+arrow(947,104,947,132)
    diagram+=box(770,144,355,['scores and weights','B × 3 × 197 × 197'],'c-k')+arrow(947,226,947,254)
    diagram+=box(770,266,355,['messages → joined → project','B × 197 × 192'],'c-v',size=22)
    diagram+=t(947,391,'No causal mask: the whole image is available.',19,'ink-2','middle')
    add('code-photo-attention','Write the three-head computation with explicit axes',snippet('attention'),diagram,
        'Each image and head has its own attention matrix. Softmax runs over source rows on the last axis. Join the three 64-feature messages and project to 192 features. Images in the batch never attend to one another.',
        'What do the two 3s in reshape mean?', 'The first selects Q, K or V; the second selects one of three heads. B stays separate. N=197 includes CLS.',
        'qkv has shape B×N×(3×192) before reshape. permute makes the Q/K/V selector the first axis, so unbind returns three tensors of shape B×3×N×64. '
        'The divisor is sqrt(64)=8. There is no mask or cross-image attention. This explicit implementation omits dropout. '
        'The two equal-sized axes mean receiving query and source key, not height and width.')

    code=snippet('block-init')+'\n\n'+snippet('block-forward')
    diagram=box(787,24,320,['Input E','B × 197 × 192'])+arrow(947,105,947,138)
    diagram+=box(787,151,320,['LN → attention → add E','B × 197 × 192'],size=22)+arrow(947,233,947,263)
    diagram+=box(787,276,320,['LN → MLP → add input','B × 197 × 192'],size=22)
    diagram+=t(947,408,'MLP features: 192 → 768 → 192',23,'c-v','middle')
    add('code-photo-block','Match the two residual branches to code',code,diagram,
        'LayerNorm and the MLP operate on each row’s features. Attention mixes rows within each image. Both branches return 192 features, so each residual addition preserves the full B×197×192 shape.',
        'Why can the MLP hidden layer be wider than the residual?', 'Its final linear layer returns to 192 features before addition. The batch and token axes remain unchanged.',
        'These are the block constructor assignments followed by its forward body, extracted from the complete implementation. '
        'Different blocks have their own copies of these learned modules; all rows within one block share them.')

    code=snippet('stack')+'\n\n'+snippet('readout')
    diagram=box(782,24,330,['12 distinct blocks','B × 197 × 192'])+arrow(947,108,947,140)
    diagram+=box(782,153,330,['Final norm; select row 0','B × 192'],'c-q',size=23)+arrow(947,235,947,266)
    diagram+=box(782,278,330,['Linear(192, 1000)','B × 1000 scores'])
    diagram+=t(947,408,'num_classes = 1000 for this architecture',21,'ink-2','middle')
    add('code-photo-readout','Run the stack and read one CLS per image',code,diagram,
        'Each block consumes the previous block’s output. After final normalization, select CLS independently for each image. The class head produces 1,000 logits per image, with no softmax inside the model’s forward method.',
        'Does the loop call the same block 12 times?', 'No. ModuleList holds 12 separate Block objects with different parameters. The activation from one becomes the input to the next.',
        'The constructor creates blocks, norm and head. The forward body below calls embed, visits those blocks in order, '
        'then normalizes and selects row zero. For prediction, logits.softmax(-1) gives class probabilities. '
        'For a new two-class task, num_classes=2 changes the head, as discussed in the following adaptation section.')

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
