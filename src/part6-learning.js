/* A small masked-image model. All parameters and predictions are inspectable.
   Two width-two attention blocks, no LayerNorm/MLP, masked-pixel MSE.
   The encoder receives visible patches only; the target is used only by loss. */
(function () {
  'use strict';
  const A=window.AT, copy=x=>JSON.parse(JSON.stringify(x));
  const map=(x,f)=>Array.isArray(x)?x.map(v=>map(v,f)):f(x);
  const zip=(a,b,f)=>Array.isArray(a)?a.map((v,i)=>zip(v,b[i],f)):f(a,b);
  const add=(a,b)=>zip(a,b,(x,y)=>x+y), mul=(a,s)=>map(a,x=>x*s);
  const zeros=(n,d)=>Array.from({length:n},()=>Array(d).fill(0));
  const tr=a=>a[0].map((_,j)=>a.map(r=>r[j]));
  const mm=(a,b)=>a.map(r=>b[0].map((_,j)=>r.reduce((s,v,k)=>s+v*b[k][j],0)));
  const sum=r=>r.reduce((s,v)=>s+v,0), sumRows=a=>a.reduce(add,a[0].map(()=>0));
  const sm=r=>{const m=Math.max(...r),ex=r.map(v=>Math.exp(v-m)),z=sum(ex);return ex.map(v=>v/z);};
  const initial={W_patch:[[.2,-.1],[.1,.2],[.2,.1],[-.1,.2]],b_patch:[0,0],
    P_enc:[[-.15,-.15],[-.15,.15],[.15,-.15],[.15,.15]],
    encQ:[[.3,.1],[0,.3]],encK:[[.2,0],[.1,.3]],encV:[[.5,0],[0,.5]],encO:[[.5,0],[0,.5]],
    mask:[0,0],P_dec:[[-.15,-.15],[-.15,.15],[.15,-.15],[.15,.15]],
    decQ:[[.3,0],[.1,.3]],decK:[[.3,.1],[0,.3]],decV:[[.5,0],[0,.5]],decO:[[.5,0],[0,.5]],
    W_pixels:[[.2,.1,.2,.1],[.1,.2,.1,.2]],b_pixels:[0,0,0,0]};
  const names=Object.keys(initial);
  function patches(image){if(!Array.isArray(image)||image.length!==4||!image.every(r=>r.length===4&&r.every(Number.isFinite)))throw Error('Use a finite 4 by 4 image.');return [0,1,2,3].map(j=>[0,1,2,3].map(k=>image[2*Math.floor(j/2)+Math.floor(k/2)][2*(j%2)+k%2]));}
  function unpatch(rows){const image=zeros(4,4);rows.forEach((r,j)=>r.forEach((v,k)=>{image[2*Math.floor(j/2)+Math.floor(k/2)][2*(j%2)+k%2]=v;}));return image;}
  function block(E,p,prefix){const Q=mm(E,p[prefix+'Q']),K=mm(E,p[prefix+'K']),V=mm(E,p[prefix+'V']);const scores=mul(mm(Q,tr(K)),1/Math.sqrt(2)),weights=scores.map(sm),message=mm(weights,V),out=add(E,mm(message,p[prefix+'O']));return {E,Q,K,V,scores,weights,message,out};}
  function blockBackward(f,up,p,prefix,g){g[prefix+'O']=mm(tr(f.message),up);const dm=mm(up,tr(p[prefix+'O'])),dv=mm(tr(f.weights),dm),da=mm(dm,tr(f.V));const ds=f.weights.map((r,i)=>{const center=sum(r.map((v,j)=>v*da[i][j]));return r.map((v,j)=>v*(da[i][j]-center)/Math.sqrt(2));});const dq=mm(ds,f.K),dk=mm(tr(ds),f.Q);g[prefix+'Q']=mm(tr(f.E),dq);g[prefix+'K']=mm(tr(f.E),dk);g[prefix+'V']=mm(tr(f.E),dv);return add(up,add(mm(dq,tr(p[prefix+'Q'])),add(mm(dk,tr(p[prefix+'K'])),mm(dv,tr(p[prefix+'V'])))));}
  function forward(p,image,hidden=[3]){
    if(!hidden.length||hidden.length>=4||new Set(hidden).size!==hidden.length||hidden.some(j=>!Number.isInteger(j)||j<0||j>3))throw Error('Hide one to three different patches.');
    const R=patches(image),visible=[0,1,2,3].filter(j=>!hidden.includes(j));
    const embed=mm(visible.map(j=>R[j]),p.W_patch).map((r,i)=>add(add(r,p.b_patch),p.P_enc[visible[i]]));
    const encoder=block(embed,p,'enc');
    const restored=[0,1,2,3].map(j=>add(hidden.includes(j)?p.mask:encoder.out[visible.indexOf(j)],p.P_dec[j]));
    const decoder=block(restored,p,'dec'),prediction=mm(decoder.out,p.W_pixels).map(r=>add(r,p.b_pixels));
    const terms=hidden.map(j=>prediction[j].map((v,k)=>(v-R[j][k])**2)),loss=sum(terms.flat())/(4*hidden.length);
    return {image:copy(image),R,visible,hidden:hidden.slice(),embed,encoder,restored,decoder,prediction,terms,loss,completion:unpatch(R.map((r,j)=>hidden.includes(j)?prediction[j]:r))};
  }
  function backward(p,image,hidden=[3]){const f=forward(p,image,hidden),g={},dr=zeros(4,4);hidden.forEach(j=>{dr[j]=f.prediction[j].map((v,k)=>2*(v-f.R[j][k])/(4*hidden.length));});g.W_pixels=mm(tr(f.decoder.out),dr);g.b_pixels=sumRows(dr);const dd=blockBackward(f.decoder,mm(dr,tr(p.W_pixels)),p,'dec',g);g.P_dec=copy(dd);g.mask=sumRows(hidden.map(j=>dd[j]));const de=blockBackward(f.encoder,f.visible.map(j=>dd[j]),p,'enc',g);g.W_patch=mm(tr(f.visible.map(j=>f.R[j])),de);g.b_patch=sumRows(de);g.P_enc=zeros(4,2);f.visible.forEach((j,i)=>{g.P_enc[j]=de[i];});return {forward:f,loss:f.loss,grads:g};}
  function texture(kind,brightness){const tile=kind==='vertical'?[brightness,0,brightness,0]:[brightness,brightness,0,0];return unpatch([tile,tile,tile,tile]);}
  const training=['vertical','horizontal'].flatMap(kind=>[.6,1,1.4,1.8].map(brightness=>({kind,brightness,image:texture(kind,brightness)})));
  const heldout=['vertical','horizontal'].flatMap(kind=>[.8,1.2,1.6].map(brightness=>({kind,brightness,image:texture(kind,brightness)})));
  const cases=training.flatMap(ex=>[0,1,2,3].map(j=>({...ex,hidden:[j]})));
  function batch(p,examples=cases){const grads={};names.forEach(k=>{grads[k]=map(p[k],()=>0);});let loss=0;
    examples.forEach(ex=>{const b=backward(p,ex.image,ex.hidden);loss+=b.loss;names.forEach(k=>{grads[k]=add(grads[k],b.grads[k]);});});names.forEach(k=>{grads[k]=mul(grads[k],1/examples.length);});return {loss:loss/examples.length,grads};}
  function sgd(p,g,lr=.03){const next={};names.forEach(k=>{next[k]=zip(p[k],g[k],(v,d)=>v-lr*d);});return next;}
  function evaluate(p,examples=heldout){return sum(examples.flatMap(ex=>[0,1,2,3].map(j=>forward(p,ex.image,[j]).loss)))/(4*examples.length);}
  function encode(p,image){const E=mm(patches(image),p.W_patch).map((r,j)=>add(add(r,p.b_patch),p.P_enc[j]));return mul(sumRows(block(E,p,'enc').out),.25);}
  function probe(p){const train=training.map(ex=>({x:encode(p,ex.image),y:ex.kind==='vertical'?1:0}));let w=[0,0],bias=0;const sigmoid=x=>1/(1+Math.exp(-x));for(let step=0;step<600;step++){let dw=[0,0],db=0;train.forEach(ex=>{const d=sigmoid(sum(ex.x.map((v,k)=>v*w[k]))+bias)-ex.y;dw=add(dw,mul(ex.x,d));db+=d;});w=add(w,mul(dw,-.2/train.length));bias-=.2*db/train.length;}
    const rows=heldout.map(ex=>{const x=encode(p,ex.image),prob=sigmoid(sum(x.map((v,k)=>v*w[k]))+bias),pred=prob>=.5?'vertical':'horizontal';return {...ex,x,prob,pred,correct:pred===ex.kind};});return {w,bias,rows,accuracy:sum(rows.map(r=>+r.correct))/rows.length};}
  function run(steps=800,lr=.05){let p=copy(initial);const history=[],snapshots={initial:copy(p)};const first=batch(p);
    for(let step=0;step<=steps;step++){if(step%20===0||step===1||step===steps){history.push({step,train:evaluate(p,training),test:evaluate(p)});if([0,1,20,100,steps].includes(step))snapshots['step'+step]=copy(p);}if(step===steps)break;p=sgd(p,step===0?first.grads:batch(p).grads,lr);}
    return {history,snapshots,first,steps,lr,probeBefore:probe(initial),probeAfter:probe(p)};
  }
  A.mae={initial:copy(initial),names,copy,map,zip,mm,tr,add,mul,sum,patches,unpatch,block,forward,backward,batch,sgd,evaluate,encode,probe,texture,training,heldout,cases,run};
})();
