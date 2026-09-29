"""Check the lecture implementation: shapes, Conv2d equivalence and gradients.

Uses synthetic inputs, no checkpoint downloads or optimizer/training loop.
"""
from pathlib import Path
import json
import sys
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

assert len({id(block.attn.qkv.weight) for block in model.blocks})==12
loss=F.cross_entropy(model(images),torch.tensor([256,283]))
loss.backward()
for name,param in model.named_parameters():
    assert param.grad is not None and torch.isfinite(param.grad).all(),name
    assert param.grad.abs().sum()>0,name
report={'input':[2,3,224,224],'patch_grid':list(grid.shape),
        'input_rows':list(rows.shape),'logits':list(logits.shape),
        'conv2d_equals_unfold_linear':True,'separate_images_match_batch':True,
        'attention_matches_pytorch_sdpa':True,'distinct_parameter_sets':12,
        'all_parameter_gradients_finite_and_nonzero':True,
        'optimizer_steps':0,'checkpoint_downloads':0}
(ROOT/'figures/vision1/photo-code-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
