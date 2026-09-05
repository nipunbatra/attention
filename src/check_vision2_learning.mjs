// Independent scalar finite differences and leakage checks for the teaching model.
import fs from 'node:fs';import vm from 'node:vm';import assert from 'node:assert/strict';
const window={AT:{}},dir=new URL('./',import.meta.url);
vm.runInNewContext(fs.readFileSync(new URL('part6-learning.js',dir),'utf8'),{window});
const T=window.AT.mae,D=JSON.parse(fs.readFileSync(new URL('mae6.json',dir),'utf8'));
let checks=0,maxError=0;
const near=(a,b,tol=1e-8)=>{assert(Number.isFinite(a)&&Math.abs(a-b)<tol,`${a} != ${b}`);};
function cells(x,path=[]){return Array.isArray(x)?x.flatMap((v,i)=>cells(v,path.concat(i))):[path];}
function get(o,path){return path.reduce((x,k)=>x[k],o);}function set(o,path,v){const r=path.slice(0,-1).reduce((x,k)=>x[k],o);r[path.at(-1)]=v;}
for(const snapshot of ['initial','step800'])for(const hidden of [[3],[0,2,3]]){
 const p=T.copy(D.snapshots[snapshot]),image=T.texture('vertical',1.2),b=T.backward(p,image,hidden);
 for(const name of T.names)for(const path of cells(p[name])){const plus=T.copy(p),minus=T.copy(p),value=get(p[name],path),eps=1e-5;set(plus[name],path,value+eps);set(minus[name],path,value-eps);const fd=(T.forward(plus,image,hidden).loss-T.forward(minus,image,hidden).loss)/(2*eps),an=get(b.grads[name],path);maxError=Math.max(maxError,Math.abs(fd-an));near(fd,an,2e-7);checks++;}
 // Alter every hidden target pixel: the forward predictions must not change.
 const rows=T.patches(image);hidden.forEach(j=>{rows[j]=[4,-2,3,0];});const changed=T.forward(p,T.unpatch(rows),hidden),f=T.forward(p,image,hidden);
 assert.deepEqual(JSON.parse(JSON.stringify(changed.prediction)),JSON.parse(JSON.stringify(f.prediction)));
 assert.notEqual(changed.loss,f.loss);
 const manual=f.hidden.flatMap(j=>f.prediction[j].map((v,k)=>(v-f.R[j][k])**2));near(f.loss,manual.reduce((s,v)=>s+v,0)/manual.length);
}
assert(D.history.at(-1).test<D.history[0].test/100);
const initialCopy=JSON.stringify(T.initial),p=T.sgd(T.initial,D.first.grads,D.lr);
T.names.forEach(k=>cells(p[k]).forEach(path=>near(get(p[k],path),get(D.snapshots.step1[k],path))));
assert.equal(JSON.stringify(T.initial),initialCopy);
const trainKeys=new Set(T.training.map(ex=>JSON.stringify(ex.image)));
assert(T.heldout.every(ex=>!trainKeys.has(JSON.stringify(ex.image))));
assert.equal(D.probeBefore.accuracy,1);assert.equal(D.probeAfter.accuracy,1);
for(const bad of [[],[0,1,2,3],[3,3],[-1]])assert.throws(()=>T.forward(T.initial,T.texture('vertical',1),bad));
console.log(JSON.stringify({gradientChecks:checks,maxError,hiddenPixelLeakage:false,heldoutMSE:D.history.at(-1).test,probeBefore:D.probeBefore.accuracy,probeAfter:D.probeAfter.accuracy}));
