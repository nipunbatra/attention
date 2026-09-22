"""Small, independently computed two-head worksheet using Part II input rows."""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def mm(A,B):
    return [[sum(a*b for a,b in zip(row,col)) for col in zip(*B)] for row in A]


def softmax(row):
    e=[math.exp(x-max(row)) for x in row]
    return [x/sum(e) for x in e]


def worksheet():
    base=json.loads((ROOT/'toy.json').read_text())
    projections=[
        dict(label='Setting',Q=[[0,0],[0,0],[0,0],[1,1]],
             K=[[1,0],[0,1],[0,0],[0,0]],V=[[1,0],[0,1],[0,0],[0,0]]),
        dict(label='Person',Q=[[0,0],[0,0],[0,0],[1,0]],
             K=[[0,0],[0,0],[1,0],[0,1]],V=[[0,0],[0,0],[1,0],[0,1]])]
    WO=[[1,0,0,0],[0,1,0,0],[.25,0,1,0],[0,0,0,1]]
    result=dict(projections=projections,W_O=WO,cases={})
    for name,tokens in base['sentences'].items():
        E=[[a+b for a,b in zip(base['tok_emb'][t.lower()],base['pos_emb'][i])] for i,t in enumerate(tokens)]
        heads=[]
        for p in projections:
            Q,K,V=[mm(E,p[k]) for k in ['Q','K','V']]
            scores=[[sum(a*b for a,b in zip(q,k))/math.sqrt(2) if j<=i else float('-inf')
                     for j,k in enumerate(K)] for i,q in enumerate(Q)]
            A=[softmax(row) for row in scores]
            heads.append(dict(Q=Q,K=K,V=V,scores=[[s if math.isfinite(s) else None for s in row] for row in scores],A=A,messages=mm(A,V)))
        joined=[heads[0]['messages'][i]+heads[1]['messages'][i] for i in range(len(E))]
        # Keep the same projection coordinates, but normalize once as one wide
        # head. This isolates separate softmaxes from simply making Q/K/V wider.
        wide={kind:[heads[0][kind][i]+heads[1][kind][i] for i in range(len(E))]
              for kind in ['Q','K','V']}
        wide_scores=[[sum(a*b for a,b in zip(q,k))/2 if j<=i else float('-inf')
                      for j,k in enumerate(wide['K'])] for i,q in enumerate(wide['Q'])]
        wide['A']=[softmax(row) for row in wide_scores]
        wide['messages']=mm(wide['A'],wide['V'])
        wide['scores']=[[s if math.isfinite(s) else None for s in row] for row in wide_scores]
        delta=mm(joined,WO)
        updated=[[a+b for a,b in zip(e,d)] for e,d in zip(E,delta)]
        hidden=[[max(0,a+b) for a,b in zip(row,base['b_hidden'])] for row in mm(updated,base['W_hidden'])]
        logits=[[a+b for a,b in zip(row,base['b_vocab'])] for row in mm(hidden,base['W_vocab'])]
        result['cases'][name]=dict(tokens=tokens,E=E,heads=heads,wide=wide,joined=joined,delta=delta,updated=updated,
                                   hidden=hidden,logits=logits,probabilities=[softmax(row) for row in logits])
    return base,result


if __name__=='__main__':
    _,data=worksheet()
    for name,c in data['cases'].items():
        print(name,'query',c['heads'][0]['Q'][-1], 'messages',c['joined'][-1], 'update',c['delta'][-1])
