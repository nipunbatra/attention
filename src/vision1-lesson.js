/* Pure numerical operations are also loadable by Node for parity checks. */
(function () {
  'use strict';
  const transpose = A => A[0].map((_, j) => A.map(row => row[j]));
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const mul = (A, B) => A.map(row => transpose(B).map(col => dot(row, col)));
  const add = (A, B) => A.map((row,i) => row.map((x,j) => x+B[i][j]));
  const softmax = row => {
    const m = Math.max(...row), ex = row.map(x => Math.exp(x-m));
    const sum = ex.reduce((a,b) => a+b,0); return ex.map(x => x/sum);
  };
  function forward(p, name, positions) {
    const image=p.images[name], patches=[];
    for(let r=0;r<4;r+=2) for(let c=0;c<4;c+=2)
      patches.push([image[r][c],image[r][c+1],image[r+1][c],image[r+1][c+1]]);
    const content=mul(patches,p.W_patch).map(row=>row.map((x,j)=>x+p.b_patch[j]));
    let E=[p.cls.slice(),...content];
    if(positions) E=add(E,p.positions);
    const heads=p.heads.map(w=>{
      const Q=mul(E,w.W_Q),K=mul(E,w.W_K),V=mul(E,w.W_V);
      const scores=mul(Q,transpose(K)).map(row=>row.map(x=>x/Math.sqrt(p.d_k)));
      const A=scores.map(softmax),H=mul(A,V);
      return {Q,K,V,scores,A,H};
    });
    const joined=E.map((_,i)=>heads.flatMap(h=>h.H[i]));
    const delta=mul(joined,p.W_O),updated=add(E,delta);
    const logits=mul([updated[0]],p.W_class)[0];
    return {patches,content,E,heads,joined,delta,updated,logits,probability:softmax(logits)};
  }
  if(typeof module!=='undefined'&&module.exports) { module.exports={forward}; return; }
  document.addEventListener('DOMContentLoaded',()=>{
    const arrangement=document.getElementById('vp-arrangement');
    const position=document.getElementById('vp-positions');
    const output=document.getElementById('vp-result');
    if(!arrangement||!position||!output) return;
    const fixed=x=>x.toFixed(3).replace('-', '−');
    const update=()=>{
      const r=forward(window.__TOY__,arrangement.value,position.checked);
      const messages=r.heads.map((h,i)=>`Head ${i+1}: [${h.H[0].map(fixed).join(', ')}]`).join(' · ');
      output.replaceChildren();
      const first=document.createElement('div'); first.textContent=messages;
      const second=document.createElement('strong');
      second.textContent=`Across: ${(100*r.probability[0]).toFixed(1)}% · Down: ${(100*r.probability[1]).toFixed(1)}%`;
      output.append(first,second);
    };
    arrangement.addEventListener('change',update);position.addEventListener('change',update);update();
  });
})();

if (typeof document !== 'undefined') document.addEventListener('DOMContentLoaded',()=>{
 const resolution=document.getElementById('vl-resolution'),patch=document.getElementById('vl-patch'),out=document.getElementById('vl-cost');
 if(!resolution||!patch||!out)return;
 const update=()=>{const side=Number(resolution.value)/Number(patch.value),N=side*side+1;out.textContent=`${side} × ${side} = ${N-1} patches · ${N} tokens with CLS · ${(N*N).toLocaleString()} scores per head`;};
 resolution.addEventListener('change',update);patch.addEventListener('change',update);update();
});
