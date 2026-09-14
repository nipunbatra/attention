// Independent arithmetic for the Part II MLP, including genuinely nonlinear probes.
// Usage: node src/check_mlp_head.mjs
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {forward, baseline, head} from './toy_ref.mjs';
const toy=JSON.parse(readFileSync(new URL('toy.json',import.meta.url),'utf8'));
const old=JSON.parse(readFileSync(new URL('toy3.json',import.meta.url),'utf8'));
const near=(a,b)=>{
  if(Array.isArray(a)){assert.equal(a.length,b.length);a.forEach((x,i)=>near(x,b[i]));}
  else assert(Math.abs(a-b)<1e-12,`${a} != ${b}`);
};
const affine=(x,W,b)=>b.map((bias,j)=>bias+x.reduce((s,v,i)=>s+v*W[i][j],0));
assert.equal(toy.d_hidden,8);
assert.equal(toy.W_hidden.length,4);assert(toy.W_hidden.every(r=>r.length===8));
assert.equal(toy.W_vocab.length,8);assert(toy.W_vocab.every(r=>r.length===20));
assert.equal(toy.b_hidden.length,8);assert.equal(toy.b_vocab.length,20);
const probes=[[0,0,0,2.3],[-1,.5,-.3,2],[3.1,-.1,0,.1],[.7,.7,.1,.8],[0,0,0,0],[1,0,0,0],[2,0,0,0]];
for(const x of probes){
  const pre=affine(x,toy.W_hidden,toy.b_hidden),h=pre.map(v=>Math.max(0,v));
  const z=affine(h,toy.W_vocab,toy.b_vocab),peak=Math.max(...z),exp=z.map(v=>Math.exp(v-peak)),sum=exp.reduce((a,b)=>a+b,0);
  const actual=head(toy,[x]);near(actual.HeadPre[0],pre);near(actual.HeadHidden[0],h);
  near(actual.logits[0],z);near(actual.probs[0],exp.map(v=>v/sum));
}
// A threshold-crossing probe must differ from a single affine transformation.
const z0=head(toy,[[0,0,0,0]]).logits[0],z1=head(toy,[[1,0,0,0]]).logits[0],z2=head(toy,[[2,0,0,0]]).logits[0];
assert(z1.some((v,j)=>Math.abs(v-(z0[j]+z2[j])/2)>1e-3));
for(const tokens of Object.values(toy.sentences)){
  const f=forward(toy,tokens),legacy=forward(old,tokens);
  // Adding a prediction-layer hidden state must not change attention or residuals.
  for(const field of ['E','Q','K','V','Sraw','A','Mmsg','Delta','Enew'])near(f[field],legacy[field]);
  assert(f.HeadHidden.at(-1).some(v=>v>0),'Context activates the predictor.');
  const b=baseline(toy,tokens);near(b.HeadHidden.at(-1),Array(8).fill(0));
  near(b.logits.at(-1),toy.b_vocab);
}
near(baseline(toy,toy.sentences.river).probs.at(-1),baseline(toy,toy.sentences.cheque).probs.at(-1));
console.log('PASS: 4→8→20 shapes; seven independent ReLU/head probes; nonlinear threshold; identical baseline; unchanged attention/residuals and separate Part III readout.');
