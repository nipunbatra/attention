(function () {
  'use strict';
  const AT = window.AT, model = AT.model, data = model.headsLesson;
  const style = document.createElement('style');
  style.textContent = `.mh-frame{min-width:0}.mh-figure{max-width:100%;overflow-x:auto}.mh-figure svg{display:block;width:100%;height:auto;min-width:670px}body.present .mh-figure{overflow:visible}body.present .mh-figure svg{min-width:0;max-height:430px}body.present .mh-frame:has(pre) .mh-figure svg{max-height:265px}body.present .mh-frame pre{font-size:22px;line-height:1.35;padding:14px 18px;margin:12px 0}body.present .mh-frame p{margin:12px 0;line-height:1.4}body:not(.present) .mh-frame{padding:24px 0;border-bottom:1px solid var(--line)}body:not(.present) .mh-frame:before{content:attr(data-title);display:block;font-weight:700;font-size:1.25em;margin-bottom:18px}.mh-controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap}.mh-controls select{font:inherit;padding:8px;border:1px solid var(--line);border-radius:5px;background:var(--card)}.mh-readout{font-size:.85em}.mh-live-svg{max-width:100%;overflow-x:auto}.mh-live-svg svg{min-width:800px;width:100%;display:block}body.present .mh-live-svg{overflow:visible}body.present .mh-live-svg svg{min-width:0}`;
  document.head.appendChild(style);
  const mm = (a,b) => a.map(row => b[0].map((_,j) => row.reduce((s,x,k) => s+x*b[k][j],0)));
  const softmax = row => {const e=row.map(x=>Math.exp(x-Math.max(...row)));const z=e.reduce((a,b)=>a+b,0);return e.map(x=>x/z);};
  function compute(name) {
    const tokens=model.sentences[name];
    const E=tokens.map((t,i)=>model.tok_emb[t.toLowerCase()].map((v,c)=>v+model.pos_emb[i][c]));
    const heads=data.projections.map(p=>{
      const Q=mm(E,p.Q),K=mm(E,p.K),V=mm(E,p.V);
      const scores=Q.map((q,i)=>K.map((k,j)=>j<=i?q.reduce((s,x,c)=>s+x*k[c],0)/Math.sqrt(2):-Infinity));
      const A=scores.map(softmax);return {Q,K,V,A,scores,messages:mm(A,V)};
    });
    const joined=E.map((_,i)=>heads.flatMap(h=>h.messages[i]));
    const delta=mm(joined,data.W_O),updated=E.map((e,i)=>e.map((x,c)=>x+delta[i][c]));
    const hidden=mm(updated,model.W_hidden).map(row=>row.map((x,c)=>Math.max(0,x+model.b_hidden[c])));
    const logits=mm(hidden,model.W_vocab).map(row=>row.map((x,c)=>x+model.b_vocab[c]));
    return {tokens,E,heads,joined,delta,updated,logits,probabilities:logits.map(softmax)};
  }
  AT.multiheadWorksheet={compute};
  document.addEventListener('DOMContentLoaded',()=>{
    const host=document.getElementById('s04-head-explorer');if(!host)return;
    host.innerHTML='<div class="mh-controls"><label>Context <select id="s04-context"><option value="river">River bank</option><option value="cheque">Cheque at the bank</option></select></label><label>Inspect <select id="s04-head"><option value="0">Head 1</option><option value="1">Head 2</option></select></label></div><div class="mh-live-svg" id="s04-live-weights"></div><p class="mh-readout" id="s04-live-message"></p><p class="mh-readout" id="s04-live-prediction"></p>';
    function draw(){
      const F=compute(host.querySelector('#s04-context').value),head=Number(host.querySelector('#s04-head').value);
      const words=F.tokens.map((t,j)=>`<text x="${155+j*98}" y="40" text-anchor="middle" font-size="18">${t}</text>`).join('');
      const rows=F.heads.map((h,i)=>`<text x="8" y="${112+i*73}" font-size="23">Head ${i+1}</text>`+h.A.at(-1).map((a,j)=>`<rect x="${110+j*98}" y="${80+i*73}" width="90" height="48" fill="var(--c-a)" fill-opacity="${(i === head ? .10 : .03)+a*.65}"/><text x="${155+j*98}" y="${112+i*73}" text-anchor="middle" font-size="25" fill="var(--c-a)">${a.toFixed(3)}</text>`).join('')).join('');
      host.querySelector('#s04-live-weights').innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 220" role="img" aria-label="Computed attention rows for the final query" style="font-family:inherit;fill:var(--ink)">${words}${rows}</svg>`;
      const vector=x=>'['+x.map(v=>v.toFixed(3)).join(', ')+']';
      host.querySelector('#s04-live-message').textContent=`Head ${head+1}: q₁₀ = ${vector(F.heads[head].Q.at(-1))}; message = ${vector(F.heads[head].messages.at(-1))}.`;
      const top=F.probabilities.at(-1).map((p,i)=>({word:model.vocab[i],p})).sort((a,b)=>b.p-a.p).slice(0,3);
      host.querySelector('#s04-live-prediction').textContent='After both messages, W_O, residual and the prediction MLP: '+top.map(x=>x.word+' '+(100*x.p).toFixed(1)+'%').join(' · ');
    }
    host.querySelectorAll('select').forEach(s=>s.addEventListener('change',draw));draw();
  });
})();
