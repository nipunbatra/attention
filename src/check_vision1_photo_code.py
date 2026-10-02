"""Check the lecture implementation: shapes, Conv2d equivalence and gradients.

Uses synthetic inputs, no checkpoint downloads or optimizer/training loop.
"""
from pathlib import Path
import json
import sys
import textwrap
import xml.etree.ElementTree as ET
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'notebooks/vision'))
from vit_image_classifier import ImageClassifier

torch.manual_seed(17)
torch.set_num_threads(2)
model=ImageClassifier().eval()
images=torch.randn(2,3,224,224)
with torch.no_grad():
    grid=model.patch(images)
    patches=F.unfold(images,kernel_size=16,stride=16).transpose(1,2)
    projected=F.linear(patches,model.patch.weight.flatten(1),model.patch.bias)
    torch.testing.assert_close(grid.flatten(2).transpose(1,2),projected,rtol=1e-5,atol=2e-6)
    # Unique values make a channel/pixel permutation visible (RGB 0/1 repeats
    # can conceal an indexing error). This verifies PyTorch's actual ordering.
    distinct=torch.arange(1,13,dtype=torch.float64).reshape(1,3,2,2)
    channel_row=F.unfold(distinct,kernel_size=2,stride=2).transpose(1,2)
    torch.testing.assert_close(channel_row[0,0],torch.arange(1,13,dtype=torch.float64))
    pixel_row=distinct.permute(0,2,3,1).reshape(1,1,12)
    torch.testing.assert_close(pixel_row[0,0],torch.tensor(
        [1,5,9,2,6,10,3,7,11,4,8,12],dtype=torch.float64))
    small_order=torch.arange(12).reshape(3,2,2).permute(1,2,0).flatten()
    weights=torch.arange(1,13,dtype=torch.float64).reshape(1,12)
    expected=F.linear(channel_row,weights)
    torch.testing.assert_close(F.linear(pixel_row,weights[:,small_order]),expected)
    assert not torch.allclose(F.linear(pixel_row,weights),expected)
    # The same permutation rule holds for all 196 real-size patch rows.
    pixel_order=torch.arange(768).reshape(3,16,16).permute(1,2,0).flatten()
    pixel_projected=F.linear(patches[:,:,pixel_order],
        model.patch.weight.flatten(1)[:,pixel_order],model.patch.bias)
    torch.testing.assert_close(pixel_projected,projected,rtol=1e-5,atol=2e-6)
    rows=model.embed(images)
    assert rows.shape==(2,197,192)
    torch.testing.assert_close(rows[0,0],rows[1,0])
    logits=model(images)
    assert logits.shape==(2,1000)
    separate=torch.cat([model(x[None]) for x in images])
    torch.testing.assert_close(logits,separate,rtol=1e-5,atol=2e-6)
    # Independently check the explicit attention against PyTorch's fused operation.
    a=model.blocks[0].attn
    q,k,v=a.qkv(rows).reshape(2,197,3,3,64).permute(2,0,3,1,4).unbind(0)
    fused=F.scaled_dot_product_attention(q,k,v,is_causal=False)
    reference=a.proj(fused.transpose(1,2).reshape(2,197,192))
    torch.testing.assert_close(a(rows),reference,rtol=1e-5,atol=2e-6)
    # Run the visible SVG snippets, including the newly split slides. This also
    # checks that syntax-colour spans preserve exact, executable Python text.
    def slide_code(name):
        root=ET.parse(ROOT/'figures/vision1'/f'{name}.svg').getroot()
        return '\n'.join(''.join(node.itertext()) for node in root.iter('{http://www.w3.org/2000/svg}text')
                         if 'monospace' in node.get('font-family',''))
    def run_slides(names, layer, inputs):
        code='\n'.join(slide_code(name) for name in names)
        scope={'torch':torch,'nn':torch.nn}
        exec('def shown(self, x):\n'+textwrap.indent(code,'    '),scope)
        return scope['shown'](layer,inputs)
    torch.testing.assert_close(run_slides(['code-photo-tokens','code-photo-tokens-cls'],model,images),rows)
    torch.testing.assert_close(run_slides(['code-photo-attention','code-photo-attention-messages',
        'code-photo-attention-join'],a,rows),a(rows))
    torch.testing.assert_close(run_slides(['code-photo-block'],model.blocks[0],rows),model.blocks[0](rows))
    torch.testing.assert_close(run_slides(['code-photo-readout'],model,images),logits)

assert len({id(block.attn.qkv.weight) for block in model.blocks})==12
loss=F.cross_entropy(model(images),torch.tensor([256,283]))
loss.backward()
for name,param in model.named_parameters():
    assert param.grad is not None and torch.isfinite(param.grad).all(),name
    assert param.grad.abs().sum()>0,name
report={'input':[2,3,224,224],'patch_grid':list(grid.shape),
        'input_rows':list(rows.shape),'logits':list(logits.shape),
        'conv2d_equals_unfold_linear':True,'separate_images_match_batch':True,
        'unfold_channel_order_checked_with_12_distinct_values':True,
        'joint_pixel_and_weight_permutation_preserves_outputs':True,
        'permuting_only_pixels_changes_outputs':True,
        'attention_matches_pytorch_sdpa':True,'distinct_parameter_sets':12,
        'visible_split_code_matches_model':True,
        'all_parameter_gradients_finite_and_nonzero':True,
        'optimizer_steps':0,'checkpoint_downloads':0}
(ROOT/'figures/vision1/photo-code-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
