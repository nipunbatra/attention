export const SPECIAL = {pad:0, bos:1, eos:2, unk:3};
export const KINDS = ['mlp','attention','multihead'];
export const LABELS = {mlp:'Fixed-window MLP',attention:'One-head attention',multihead:'Four-head attention'};
// Match Python wordlm.TOKEN_PATTERN, including its English-only word rule.
export function tokenize(text){return text.normalize('NFKC').toLowerCase().trim().replace(/\s+/gu,' ').match(/[a-z]+(?:'[a-z]+)?|[0-9]+|[^\p{L}\p{N}_\s]/giu)||[];}
export function detokenize(tokens){return tokens.join(' ').replace(/\s+([.,!?;:%)\]}])/gu,'$1').replace(/([(\[{])\s+/gu,'$1').trim();}
export function makeContext(history,width=64){const context=new Int32Array(width);const tail=history.slice(-width);context.set(tail,width-tail.length);return context;}
export function randomGenerator(seed){let state=seed>>>0;return()=>{state=(state+0x6d2b79f5)|0;let v=Math.imul(state^(state>>>15),1|state);v^=v+Math.imul(v^(v>>>7),61|v);return((v^(v>>>14))>>>0)/4294967296;};}
export function chooseToken(logits,mode,temperature,random){
  const allowed=i=>i===SPECIAL.eos||i>SPECIAL.unk;
  let max=-Infinity,best=-1;
  for(let i=0;i<logits.length;i++)if(allowed(i)&&logits[i]>max){max=logits[i];best=i;}
  if(best<0||!Number.isFinite(max))throw new Error('The model returned invalid scores.');
  if(mode==='greedy')return best;
  if(!(temperature>0))throw new Error('Temperature must be positive.');
  const weights=Float64Array.from(logits,(x,i)=>allowed(i)?Math.exp((x-max)/temperature):0);
  let threshold=random()*weights.reduce((a,b)=>a+b,0);
  for(let i=0;i<weights.length;i++){threshold-=weights[i];if(weights[i]>0&&threshold<=0)return i;}
  return best;
}
