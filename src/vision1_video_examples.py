"""Exact experiments for the video-inspired implementation teaching steps."""
import torch
from torch import nn
from torch.nn import functional as F


def examples():
    dtype=torch.float64
    x=torch.tensor([[[[1.,1.,1.,0.],[0.,0.,1.,0.],
                      [0.,0.,1.,1.],[1.,1.,1.,1.]]]],dtype=dtype)
    conv=nn.Conv2d(1,2,kernel_size=2,stride=2,dtype=dtype)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([[[[1.,1.],[-1.,-1.]]],
                                       [[[1.,-1.],[1.,-1.]]]],dtype=dtype))
        conv.bias.zero_()
    patch_rows=F.unfold(x,kernel_size=2,stride=2).transpose(1,2)
    linear=patch_rows@conv.weight.flatten(1).T+conv.bias
    convolved=conv(x).flatten(2).transpose(1,2)
    torch.testing.assert_close(linear,convolved)
    torch.testing.assert_close(convolved,torch.tensor([[[2.,0.],[0.,2.],[-2.,0.],[0.,0.]]],dtype=dtype))
    convolved.sum().backward()
    assert conv.weight.requires_grad and conv.bias.requires_grad
    assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all()

    # Q=K=0 makes attention uniform; V and the output projection are identities.
    # This removes random-weight effects from the batch-axis counterexample.
    X=torch.tensor([[[2.,0.],[4.,0.],[6.,0.]],
                    [[0.,2.],[0.,4.],[0.,6.]]],dtype=dtype)
    correct=nn.MultiheadAttention(2,1,bias=False,batch_first=True,dtype=dtype).eval()
    with torch.no_grad():
        correct.in_proj_weight.zero_()
        correct.in_proj_weight[4:6].copy_(torch.eye(2,dtype=dtype))
        correct.out_proj.weight.copy_(torch.eye(2,dtype=dtype))
    wrong=nn.MultiheadAttention(2,1,bias=False,batch_first=False,dtype=dtype).eval()
    wrong.load_state_dict(correct.state_dict())
    with torch.no_grad():
        good_solo=correct(X[:1],X[:1],X[:1],need_weights=False)[0]
        good_batch=correct(X,X,X,need_weights=False)[0][:1]
        bad_solo=wrong(X[:1],X[:1],X[:1],need_weights=False)[0]
        bad_batch=wrong(X,X,X,need_weights=False)[0][:1]
    torch.testing.assert_close(good_solo,good_batch)
    torch.testing.assert_close(good_batch[0,0],torch.tensor([4.,0.],dtype=dtype))
    torch.testing.assert_close(bad_solo[0,0],torch.tensor([2.,0.],dtype=dtype))
    torch.testing.assert_close(bad_batch[0,0],torch.tensor([1.,1.],dtype=dtype))
    assert not torch.allclose(bad_solo,bad_batch)
    return {'patch_projection':{'image':x[0,0].tolist(),'weights':conv.weight[:,0].detach().tolist(),
             'patches':patch_rows[0].tolist(),'output':convolved[0].detach().tolist(),
             'parameters':sum(p.numel() for p in conv.parameters()),'trainable':True,'finite_gradient':True},
            'batch_axis':{'input':X.tolist(),'correct_alone':good_solo[0].tolist(),
             'correct_batched':good_batch[0].tolist(),'wrong_alone':bad_solo[0].tolist(),
             'wrong_batched':bad_batch[0].tolist(),'correct_invariant':True,'wrong_detected':True}}
