/* The diagram reads the same live model as the article; no copied arithmetic. */
(function () {
  if (!window.AT || !AT.model || !AT.model.W_Q) return;
  var tokens = AT.sentences.river.slice(), F = AT.forward(tokens);
  function row(i) {
    var alpha=F.A[i], keys=F.K, values=F.V;
    var raw=keys.map(function(k){return AT.dot(F.Q[i],k);});
    var scaled=raw.map(function(s){return s/Math.sqrt(AT.d_k);});
    var mixture=AT.zeros(AT.d_v);
    values.forEach(function(v,j){mixture=AT.add(mixture,AT.scale(v,alpha[j]));});
    var top=AT.vocab.map(function(t,j){return {token:t,index:j,logit:F.logits[i][j],probability:F.probs[i][j]};}).sort(function(a,b){return b.probability-a.probability;});
    return {index:i,position:i+1,token:tokens[i],e:F.E[i],q:F.Q[i],keys:keys,values:values,
      rawScores:raw,scaledScores:scaled,maskedScores:scaled.map(function(s,j){return j>i?null:s;}),
      alpha:alpha,mixture:mixture,delta:F.Delta[i],updated:F.Enew[i],logits:F.logits[i],probabilities:F.probs[i],topVocabulary:top.slice(0,6)};
  }
  var pi=AT.axes.short.e.indexOf('pos');
  var positionIgnored=pi>=0 && ['W_Q','W_K','W_V','W_vocab'].every(function(k){return AT.model[k][pi].every(function(x){return x===0;});});
  window.ATTENTION_PREVIEW_DATA={tokens:tokens,vocabulary:AT.vocab,axes:AT.axes,
    dims:{T:tokens.length,dModel:AT.d_model,dKey:AT.d_k,dValue:AT.d_v,vocabSize:AT.vocab.length},
    provenance:{source:'AT.model / AT.forward',handDesigned:true,trained:false,positionIgnored:positionIgnored},bank:row(6),last:row(tokens.length-1)};
})();
