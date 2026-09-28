"""Analytic reverse pass for the existing hand-chosen four-patch worksheet."""
import numpy as np
from vision1_worksheet import parameters, softmax, serializable

def evaluate(p):
    pixels=np.array(p['images']['horizontal'],dtype=float)
    patches=pixels.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)
    E=np.vstack([p['cls'],patches@p['W_patch']+p['b_patch']])+p['positions']
    heads=[]
    for h in p['heads']:
        Q,K,V=[E@h[k] for k in ['W_Q','W_K','W_V']]
        A=softmax(Q@K.T/np.sqrt(2)); H=A@V
        heads.append(dict(Q=Q,K=K,V=V,A=A,H=H))
    J=np.concatenate([h['H'] for h in heads],axis=1)
    U=E+J@p['W_O']; z=U[0]@p['W_class']; prob=softmax(z)
    return dict(patches=patches,E=E,heads=heads,J=J,U=U,z=z,p=prob,loss=-np.log(prob[0]))

def backward(p):
    r=evaluate(p); dz=r['p']-np.array([1.,0.])
    dU=np.zeros_like(r['U']); dU[0]=np.array(p['W_class'])@dz
    dJ=dU@np.array(p['W_O']).T; dE=dU.copy(); grads=[]
    for i,h in enumerate(r['heads']):
        dH=dJ[:,i*2:i*2+2]; dV=h['A'].T@dH; dA=dH@h['V'].T
        dS=h['A']*(dA-(dA*h['A']).sum(axis=1,keepdims=True))
        dQ=dS@h['K']/np.sqrt(2); dK=dS.T@h['Q']/np.sqrt(2)
        for g,k in [(dQ,'W_Q'),(dK,'W_K'),(dV,'W_V')]:dE+=g@np.array(p['heads'][i][k]).T
        grads.append(dict(dH=dH,dV=dV,dA=dA,dS=dS,dQ=dQ,dK=dK,
                          W_Q=r['E'].T@dQ,W_K=r['E'].T@dK,W_V=r['E'].T@dV))
    return r,dict(dz=dz,dU=dU,dJ=dJ,dE=dE,heads=grads,
                  W_class=np.outer(r['U'][0],dz),W_O=r['J'].T@dU,
                  cls=dE[0],positions=dE,W_patch=r['patches'].T@dE[1:],b_patch=dE[1:].sum(axis=0))

def checked_trace():
    import copy
    p=parameters();r,g=backward(p);errors=[]
    for name in ['W_class','W_O','cls','positions','W_patch','b_patch']:
        a=np.array(p[name],dtype=float)
        for index in np.ndindex(a.shape):
            plus=copy.deepcopy(p);minus=copy.deepcopy(p);ap=a.copy();am=a.copy();ap[index]+=1e-5;am[index]-=1e-5
            plus[name]=ap.tolist();minus[name]=am.tolist()
            diff=(evaluate(plus)['loss']-evaluate(minus)['loss'])/2e-5
            errors.append(abs(diff-g[name][index]))
    for hi in range(2):
        for name in ['W_Q','W_K','W_V']:
            a=np.array(p['heads'][hi][name],dtype=float)
            for index in np.ndindex(a.shape):
                plus=copy.deepcopy(p);minus=copy.deepcopy(p);ap=a.copy();am=a.copy();ap[index]+=1e-5;am[index]-=1e-5
                plus['heads'][hi][name]=ap.tolist();minus['heads'][hi][name]=am.tolist()
                diff=(evaluate(plus)['loss']-evaluate(minus)['loss'])/2e-5
                errors.append(abs(diff-g['heads'][hi][name][index]))
    assert max(errors)<1e-7,max(errors)
    updated=copy.deepcopy(p)
    updated['heads'][0]['W_Q'][3][0]-=.1*g['heads'][0]['W_Q'][3,0]
    after=evaluate(updated)
    return serializable(dict(scope='Chosen four-patch worksheet; analytic arithmetic, no model training run',forward=r,gradients=g,
      single_update=dict(parameter='head1.W_Q[3,0]',before=1.,gradient=g['heads'][0]['W_Q'][3,0],rate=.1,
                         after=updated['heads'][0]['W_Q'][3][0],loss_before=r['loss'],loss_after=after['loss']),
      verification=dict(finite_difference_coordinates=len(errors),max_absolute_error=max(errors))))

if __name__=='__main__':
    import json
    x=checked_trace();print(json.dumps({k:x[k] for k in ['single_update','verification']},indent=2))
