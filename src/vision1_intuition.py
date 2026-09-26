"""Small, independently checkable examples used in the expanded Vision I lesson."""
import numpy as np
from vision1_worksheet import parameters, forward, softmax


def calculations():
    p=parameters();r=forward(p['images']['horizontal'])
    edge=np.array([[1,1],[1,-1],[-1,1],[-1,-1]],dtype=float)
    top=np.array([1,1,0,0]);left=np.array([1,0,1,0])
    q=np.array([1.,0.]);h=r['heads'][0]
    scores=q@h['K'].T/np.sqrt(2)
    weights=softmax(scores)
    row=np.array([1.,0.,0.,1.]);centered=row-row.mean()
    joined=r['joined'][0];W=np.array(p['W_O'])
    return {
      'edge_projection':{'W':edge.tolist(),'top':(top@edge).tolist(),'left':(left@edge).tolist(),
                         'top_mean':float(top.mean()),'left_mean':float(left.mean())},
      'softmax_examples':[{'scores':s,'weights':softmax(np.array(s)).tolist()}
                          for s in [[0.,0.],[0.,float(np.log(3))],[-100.,-100.]]],
      'changed_query':{'query':q.tolist(),'scores':scores.tolist(),'weights':weights.tolist(),
                       'message':(weights@h['V']).tolist()},
      'layernorm':{'input':row.tolist(),'mean':float(row.mean()),'centered':centered.tolist(),
                   'squares':(centered**2).tolist(),'variance':float(row.var()),
                   'denominator':float(np.sqrt(row.var()+1e-5)),
                   'normalized':(centered/np.sqrt(row.var()+1e-5)).tolist()},
      'output_projection':{'joined':joined.tolist(),'columns':W.T.tolist(),
                            'terms':(joined[:,None]*W).T.tolist(),'delta':(joined@W).tolist()},
      'scale':{'patch_pixels':16*16*3,'patch_count':14*14,'tokens':197,
               'per_head_scores':197**2,'scores_3_heads_12_blocks':197**2*3*12},
    }
