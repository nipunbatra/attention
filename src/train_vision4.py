#!/usr/bin/env python3
"""Reproducible image-conditioned language toy; --check never writes files.

The image encoder is the disclosed hand-chosen attention worksheet from Vision I.
It includes CLS during attention, then keeps only the four updated patch rows.
It is frozen. A learned 2-to-3 connector
and a one-head prefix decoder fit only two image/answer pairs. This is a test
of conditioning and causal computation, not a general counter or pretrained VLM.
The numerical model omits LayerNorm, FFN and dropout.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
VOCAB = ['<bos>', 'count', '?', 'one', 'two', 'block', 'blocks', '<eos>']
PROMPT = ['<bos>', 'count', '?']
IMAGES = {
    'two': [[1,1,0,0],[1,1,0,0],[0,0,2,2],[0,0,2,2]],
    'one': [[0,0,0,0],[0,0,0,0],[0,0,2,2],[0,0,2,2]],
}
ANSWERS = {'two':['two','blocks','<eos>'], 'one':['one','block','<eos>']}
WP = np.array([[1.,0],[0,1],[1,0],[0,1]])
POS = np.array([[0.,0],[0,1],[1,0],[1,1]])
D = 3

def sm(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e/e.sum(axis=-1, keepdims=True)

def vision(name):
    image = np.array(IMAGES[name], dtype=float)
    patches = np.stack([image[y:y+2,x:x+2].reshape(-1) for y in (0,2) for x in (0,2)])
    e = np.concatenate([np.array([[1.,1.]]), patches @ WP + POS])
    q, k, v = .5*e, .5*e, e.copy()
    scores = q @ k.T / math.sqrt(2)
    a = sm(scores)
    g = (e + a @ v)[1:]
    return dict(image=image, patches=patches, W_patch=WP, positions=POS,
                embedded=patches@WP, E=e, Q=q, K=k, V=v, scores=scores, A=a, G=g)

def initial():
    rng = np.random.default_rng(34)
    p = {'W_bridge':rng.normal(0,.2,(2,D)), 'b_bridge':np.zeros(D),
         'E_tok':rng.normal(0,.4,(len(VOCAB),D)), 'P':rng.normal(0,.05,(12,D))}
    for key in ('W_Q','W_K','W_V','W_O'):
        p[key] = rng.normal(0,.3,(D,D))
    p['W_vocab'] = rng.normal(0,.35,(D,len(VOCAB)))
    p['b_vocab'] = np.zeros(len(VOCAB))
    return p

def forward(p, name='two', prefix=PROMPT):
    if name not in IMAGES or not 1 <= len(prefix) <= 8:
        raise ValueError('Unknown image or unsupported prefix length.')
    vi = vision(name)
    ids = [VOCAB.index(t) for t in prefix]
    bridged = vi['G'] @ p['W_bridge'] + p['b_bridge']
    e = np.concatenate([bridged, p['E_tok'][ids]]) + p['P'][:4+len(ids)]
    q,k,v = (e@p['W_'+role] for role in ('Q','K','V'))
    raw = q@k.T
    n = len(e)
    # Images read images only. Text reads every image and its known text prefix.
    allowed = np.zeros((n,n), dtype=bool)
    allowed[:4,:4] = True
    for i in range(4,n):
        allowed[i,:i+1] = True
    scores = np.where(allowed,raw/math.sqrt(D),-np.inf)
    a = sm(scores)
    msg = a@v
    delta = msg@p['W_O']
    out = e+delta
    z = out@p['W_vocab']+p['b_vocab']
    probs=sm(z)
    return dict(image=name, prefix=prefix, ids=ids, vision=vi, bridged=bridged,
                E=e,Q=q,K=k,V=v,raw=raw,scores=scores,allowed=allowed,A=a,
                message=msg,delta=delta,out=out,logits=z,probs=probs)

def loss_grad(p,name='two'):
    ans=ANSWERS[name]
    f=forward(p,name,PROMPT+ans[:-1])
    rows=np.arange(6,9)
    targets=np.array([VOCAB.index(t) for t in ans])
    z=f['logits'][rows]
    peak=z.max(axis=1)
    losses=peak+np.log(np.exp(z-peak[:,None]).sum(axis=1))-z[np.arange(3),targets]
    dz=np.zeros_like(f['logits'])
    dz[rows]=f['probs'][rows]
    dz[rows,targets]-=1
    dz/=3
    g={'W_vocab':f['out'].T@dz,'b_vocab':dz.sum(axis=0)}
    dout=dz@p['W_vocab'].T
    g['W_O']=f['message'].T@dout
    dm=dout@p['W_O'].T
    da=dm@f['V'].T
    dv=f['A'].T@dm
    ds=f['A']*(da-(da*f['A']).sum(axis=1,keepdims=True))/math.sqrt(D)
    dq=ds@f['K']; dk=ds.T@f['Q']
    g['W_Q']=f['E'].T@dq; g['W_K']=f['E'].T@dk; g['W_V']=f['E'].T@dv
    de=dout+dq@p['W_Q'].T+dk@p['W_K'].T+dv@p['W_V'].T
    g['W_bridge']=f['vision']['G'].T@de[:4]
    g['b_bridge']=de[:4].sum(axis=0)
    g['E_tok']=np.zeros_like(p['E_tok']); np.add.at(g['E_tok'],f['ids'],de[4:])
    g['P']=np.zeros_like(p['P']); g['P'][:len(de)]=de
    return float(losses.mean()),losses,g,f

def generate(p,name='two',limit=5):
    prefix=PROMPT[:]; trace=[]; reason='limit'
    for _ in range(min(limit,6)):
        f=forward(p,name,prefix)
        row=len(f['E'])-1
        chosen=VOCAB[int(f['probs'][row].argmax())]
        trace.append(dict(prefix=prefix[:],query=f['Q'][row],weights=f['A'][row],
                          logits=f['logits'][row],probs=f['probs'][row],chosen=chosen,row=row))
        prefix.append(chosen)
        if chosen=='<eos>': reason='eos'; break
    return dict(tokens=prefix[len(PROMPT):],trace=trace,stoppedBy=reason)

def native(x):
    if isinstance(x,np.ndarray): return native(x.tolist())
    if isinstance(x,np.generic): return native(x.item())
    if isinstance(x,dict): return {k:native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [native(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):
        if x==-math.inf:return None
        raise ValueError('Nonfinite model value.')
    return x

def build():
    p=initial(); hist=[]
    first={k:np.zeros_like(v) for k,v in p.items()}
    second={k:np.zeros_like(v) for k,v in p.items()}
    for it in range(6000):
        g={k:np.zeros_like(v) for k,v in p.items()}; losses=[]
        for name in IMAGES:
            l,_,gi,_=loss_grad(p,name); losses.append(l)
            for k in p:g[k]+=gi[k]/2
        if it%100==0:hist.append(dict(step=it,loss=float(np.mean(losses))))
        if it>100 and max(losses)<.09 and all(generate(p,n)['tokens']==ANSWERS[n] for n in IMAGES):break
        for k in p:
            first[k]=.9*first[k]+.1*g[k]
            second[k]=.999*second[k]+.001*g[k]**2
            p[k]-=.005*(first[k]/(1-.9**(it+1)))/(np.sqrt(second[k]/(1-.999**(it+1)))+1e-8)
    else: raise RuntimeError('Toy did not fit.')
    before=copy.deepcopy(p)
    l,ls,g,f=loss_grad(p)
    after={k:v-.02*g[k] for k,v in p.items()}
    al,als,_,af=loss_grad(after)
    checks=[]
    for k in p:
        for ix in np.ndindex(p[k].shape):
            old=p[k][ix]
            p[k][ix]=old+1e-5; hi=loss_grad(p)[0]
            p[k][ix]=old-1e-5; lo=loss_grad(p)[0]
            p[k][ix]=old
            numeric=(hi-lo)/2e-5
            checks.append(dict(parameter=k,index=list(ix),analytic=g[k][ix],numeric=numeric,error=abs(numeric-g[k][ix])))
    err=max(c['error'] for c in checks)
    assert err<1e-7 and al<l
    return native(dict(d_model=D,d_k=D,d_v=D,axes={},vlm=dict(
        vocab=VOCAB,prompt=PROMPT,images=IMAGES,answers=ANSWERS,
        snapshots=dict(before=before,after=after),frozenVision=dict(W_patch=WP,positions=POS),
        defaultSnapshot='before',reference=dict(before=f,after=af),
        generation={s:{n:generate(p0,n) for n in IMAGES} for s,p0 in [('before',before),('after',after)]},
        training=dict(seed=34,steps=it,optimizer='Adam',rate=.005,history=hist,overfit=True),
        update=dict(rate=.02,lossBefore=l,lossAfter=al,perTokenBefore=ls,perTokenAfter=als,
                    gradients=g,gradientChecks=checks,maxGradientError=err),
        notes=['Two hand-drawn grids only; not a general visual counter.',
               'Frozen hand-chosen four-patch encoder; learned connector and prefix decoder.',
               'One attention head; FFN, LayerNorm and dropout omitted.',
               'Loss on answer tokens only. Positions are full-width added vectors.'])))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    result=build(); dest=HERE/'toy8.json'
    if args.check:
        assert result==json.loads(dest.read_text()),'Saved toy differs from deterministic computation.'
    else:dest.write_text(json.dumps(result,indent=2)+'\n')
    m=result['vlm']; print('Vision4',m['training']['steps'],'steps; loss',m['update']['lossBefore'],'->',m['update']['lossAfter'])
    print('gradient max error',m['update']['maxGradientError'])
    print({n:g['tokens'] for n,g in m['generation']['before'].items()})
