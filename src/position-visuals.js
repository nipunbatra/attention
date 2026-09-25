/* Original, inspectable position diagrams. Content vectors stay fixed in the
   RoPE shift experiment so only the positional operation is being compared. */
document.addEventListener('DOMContentLoaded',()=>{
  const NS='http://www.w3.org/2000/svg';
  const C={e:'var(--c-e)',q:'var(--c-q)',k:'var(--c-k)',v:'var(--c-v)',a:'var(--c-a)',d:'var(--c-d)',line:'var(--line)',ink:'var(--ink)'};
  function element(tag,attrs={},text){const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;}
  function canvas(host,label,h=280){const s=element('svg',{viewBox:`0 0 1120 ${h}`,role:'img','aria-label':label});s.append(element('title',{},label));host.replaceChildren(s);return s;}
  function text(s,x,y,value,color=C.ink,size=24){s.append(element('text',{x,y,style:`fill:${color};font-size:${size}px`},value));}
  function line(s,x1,y1,x2,y2,color=C.line,width=2){s.append(element('line',{x1,y1,x2,y2,stroke:color,'stroke-width':width}));}
  function arrow(s,x1,y1,x2,y2,color){line(s,x1,y1,x2,y2,color,3);const a=Math.atan2(y2-y1,x2-x1),l=10;line(s,x2,y2,x2-l*Math.cos(a-.45),y2-l*Math.sin(a-.45),color,3);line(s,x2,y2,x2-l*Math.cos(a+.45),y2-l*Math.sin(a+.45),color,3);}
  function dot(s,x,y,color,r=6){s.append(element('circle',{cx:x,cy:y,r,fill:color}));}
  function axes(s,x,y,r){line(s,x-r,y,x+r,y);line(s,x,y-r,x,y+r);s.append(element('circle',{cx:x,cy:y,r,fill:'none',stroke:C.line,'stroke-width':2}));}
  function rotate(v,a){return [v[0]*Math.cos(a)-v[1]*Math.sin(a),v[0]*Math.sin(a)+v[1]*Math.cos(a)];}
  const fmt=v=>'['+v.map(n=>(Math.abs(n)<.0005?0:n).toFixed(3)).join(', ')+']';
  function vector(s,x,y,r,v,color){arrow(s,x,y,x+r*v[0],y-r*v[1],color);}
  const lesson=AT.positionLesson;
  const swap=lesson.sequences.map(tokens=>lesson.experiment(tokens,false));
  const fixed=v=>'['+v.map(n=>n.toFixed(1)).join(', ')+']';
  function cell(tr,value,kind=''){
    const td=document.createElement('td');td.className=kind;
    td.textContent=typeof value==='number'?value.toFixed(3):value;
    if(typeof value==='number')td.dataset.value=String(value);
    tr.append(td);return td;
  }
  for(const [index,suffix]of ['a','b'].entries()){
    const r=swap[index],tokens=lesson.sequences[index];
    const body=document.querySelector('#position-score-'+suffix+' tbody');
    const contributions=document.querySelector('#position-contribution-'+suffix+' tbody');
    tokens.forEach((token,j)=>{
      const tr=document.createElement('tr');cell(tr,(j+1)+' '+token);
      cell(tr,r.rows[j].reduce((sum,x,c)=>sum+x*r.q[c],0),'numbers score');
      cell(tr,r.scores[j],'numbers score');cell(tr,Math.exp(r.scores[j]),'numbers');cell(tr,r.weights[j],'numbers weight');body.append(tr);
      const mix=document.createElement('tr');mix.dataset.token=token;cell(mix,token);
      const value=r.rows[j].map(x=>x*r.weights[j]);const td=cell(mix,fmt(value),'numbers message');td.dataset.vector=JSON.stringify(value);contributions.append(mix);
    });
    const foot=document.createElement('tr');cell(foot,'Sum');cell(foot,fmt(r.message),'numbers message');
    document.querySelector('#position-contribution-'+suffix+' tfoot').append(foot);
  }
  const denominator=swap[0].scores.reduce((sum,s)=>sum+Math.exp(s),0);
  document.getElementById('position-softmax-work').textContent='Sum of exponentials ≈ '+denominator.toFixed(3)+'. Maya’s weight ≈ '+Math.exp(swap[0].scores[0]).toFixed(3)+' / '+denominator.toFixed(3)+' ≈ '+swap[0].weights[0].toFixed(3)+'.';
  document.getElementById('position-follow-weight').textContent='Maya keeps weight '+swap[0].weights[0].toFixed(3)+' in slot 3. Ravi keeps weight '+swap[0].weights[2].toFixed(3)+' in slot 1.';
  document.getElementById('position-maya-contribution').textContent='For Maya: '+swap[0].weights[0].toFixed(3)+' × [1, 0] = '+fmt([swap[0].weights[0],0])+'.';
  const added=document.querySelector('#position-added-rows tbody');
  lesson.sequences[0].forEach((token,j)=>{
    const tr=document.createElement('tr');cell(tr,(j+1)+' '+token);
    cell(tr,fixed(lesson.embeddings[token]),'numbers');cell(tr,fixed(lesson.positions[j]),'numbers');
    cell(tr,fixed(lesson.embeddings[token].map((x,c)=>x+lesson.positions[j][c])),'numbers message');added.append(tr);
  });
  const mayaSlots=document.querySelector('#position-maya-slots tbody');
  lesson.sequences.forEach(tokens=>{
    const r=lesson.experiment(tokens,true),j=tokens.indexOf('Maya'),tr=document.createElement('tr');
    cell(tr,'Slot '+(j+1));cell(tr,fixed(lesson.positions[j]),'numbers');cell(tr,fixed(r.rows[j]),'numbers score');cell(tr,r.scores[j],'numbers score');mayaSlots.append(tr);
  });
  // Reuse the same word rows in a separate appended-position experiment.
  lesson.sequences.forEach((tokens,index)=>{
    const r=lesson.appendedExperiment(tokens),suffix=index?'b':'a';
    const body=document.querySelector('#position-append-'+suffix+' tbody');
    tokens.forEach((token,j)=>{
      const tr=document.createElement('tr');cell(tr,token);
      const td=cell(tr,'','numbers');td.dataset.vector=JSON.stringify(r.rows[j]);
      const slot=document.createElement('span');slot.className='append-index';slot.textContent=String(j+1);
      td.append('['+r.rows[j].slice(0,2).map(x=>x.toFixed(1)).join(', ')+', ',slot,']');
      body.append(tr);
    });
  });
  const short=value=>Number(value.toFixed(3)).toString();
  const dotFormula=(q,k)=>q.map((value,c)=>short(value)+' × '+short(k[c])).join(' + ');
  let appendScale;
  // One scale setting follows the last receiver through keys, scores and values.
  document.querySelectorAll('[data-append-control]').forEach((host,index)=>{
    const sentence=document.createElement('span');sentence.textContent=host.dataset.appendControl+'.';
    const label=document.createElement('label');label.htmlFor=index?'append-detail-scale-'+index:'position-append-scale';label.textContent='Position scale c =';
    const select=document.createElement('select');select.id=label.htmlFor;select.dataset.appendScaleControl='';
    for(const value of ['1','0.1']){const option=document.createElement('option');option.value=value;option.textContent=value;select.append(option);}
    if(index===0)appendScale=select;
    host.append(sentence,label,select);
    select.addEventListener('change',()=>{appendScale.value=select.value;drawAppended();});
  });
  function augmentedVector(host,row,prefix=''){
    host.replaceChildren();host.dataset.vector=JSON.stringify(row);
    const position=document.createElement('span');position.className='append-index';position.textContent=row[2].toFixed(1);
    host.append(prefix+'['+row.slice(0,2).map(x=>x.toFixed(1)).join(', ')+', ',position,']');
  }
  function drawCalculationTables(r,index){
    const letter=index?'b':'a',suffix=index?'-b':'',tokens=lesson.sequences[index];
    const keys=document.querySelector('#position-append-keys-'+letter+' tbody');keys.replaceChildren();
    const scores=document.querySelector('#position-append-scores'+suffix+' tbody');scores.replaceChildren();
    const softmax=document.querySelector('#position-append-softmax'+suffix+' tbody');softmax.replaceChildren();
    const values=document.querySelector('#position-append-values-'+letter+' tbody');values.replaceChildren();
    const exps=r.scores.map(Math.exp),denominator=exps.reduce((a,b)=>a+b,0);
    tokens.forEach((token,j)=>{
      const label=(j+1)+' '+token;
      const keyRow=document.createElement('tr');cell(keyRow,label);
      augmentedVector(cell(keyRow,'','numbers score'),r.K[j]);keys.append(keyRow);
      const tr=document.createElement('tr');cell(tr,label);
      const word=cell(tr,r.wordTerms[j],'numbers score');
      word.textContent=dotFormula(r.q.slice(0,2),r.K[j].slice(0,2))+' = '+short(r.wordTerms[j]);
      const position=cell(tr,r.positionTerms[j],'numbers append-index');
      position.textContent=short(r.q[2])+' × '+short(r.K[j][2])+' = '+short(r.positionTerms[j]);
      cell(tr,r.rawScores[j],'numbers append-totals');scores.append(tr);
      const row=document.createElement('tr');cell(row,label);
      const scaled=cell(row,r.scores[j],'numbers score');scaled.textContent=short(r.rawScores[j])+' / √3 = '+r.scores[j].toFixed(3);
      const exponential=cell(row,exps[j],'numbers');exponential.textContent=exps[j].toFixed(2);
      const weight=cell(row,r.weights[j],'numbers weight');weight.textContent=exps[j].toFixed(2)+' / '+denominator.toFixed(2)+' ≈ '+(100*r.weights[j]).toFixed(1)+'%';softmax.append(row);
      const mix=document.createElement('tr');mix.dataset.source=String(j);cell(mix,label);
      cell(mix,r.weights[j],'numbers weight');
      const value=cell(mix,fixed(r.V[j]),'numbers message');value.dataset.vector=JSON.stringify(r.V[j]);
      const contribution=cell(mix,fmt(r.contributions[j]),'numbers message');contribution.dataset.vector=JSON.stringify(r.contributions[j]);values.append(mix);
      if(j===3){row.className='append-self';mix.className='append-self';}
    });
    const sum=document.createElement('tr'),label=cell(sum,'Sum: message m₄');label.colSpan=3;
    const total=cell(sum,fmt(r.message),'numbers message');total.dataset.vector=JSON.stringify(r.message);
    document.querySelector('#position-append-values-'+letter+' tfoot').replaceChildren(sum);
    document.getElementById('position-append-denominator'+suffix).textContent='Sum: '+exps.map(x=>x.toFixed(2)).join(' + ')+' ≈ '+denominator.toFixed(2);
    document.getElementById('position-append-normalization'+suffix).textContent='Today’s own value gets '+(100*r.weights[3]).toFixed(1)+'% of the weight. All four sources are allowed. Values come next.';
    const maya=tokens.indexOf('Maya'),ravi=tokens.indexOf('Ravi');
    document.querySelector('[data-append-score-result="'+letter+'"]').textContent=r.rawScores[maya]<r.rawScores[ravi]
      ?'Maya has the stronger word match (0.8 versus 0.2). Ravi’s larger slot reverses that comparison.'
      :'Maya’s word match is 0.8; Ravi’s is 0.2. The slot term '+(Number(appendScale.value)===1?'reinforces this order.':'is now small enough to keep that order.');
    document.querySelector('[data-append-message-result="'+letter+'"]').textContent=Number(appendScale.value)===1
      ?'Today supplies '+(100*r.weights[3]).toFixed(1)+'% of the mixture. Each value has three coordinates. Calculations use unrounded weights.'
      :'The smaller slot feature spreads the weights more evenly. Each contribution uses the unrounded weight.';
  }
  // A fixed before/after makes the scale effect visible without remembering a toggle.
  const compared=[1,.1].map(scale=>lesson.appendedExperiment(lesson.sequences[0],scale));
  const comparison=document.querySelector('#position-append-scale-comparison tbody');
  lesson.sequences[0].forEach((token,j)=>{
    const tr=document.createElement('tr');cell(tr,(j+1)+' '+token);
    compared.forEach((r,index)=>{const td=cell(tr,r.weights[j],'numbers weight');td.dataset.scale=String(index?.1:1);td.textContent=(100*r.weights[j]).toFixed(1)+'%';});
    if(j===3)tr.className='append-self';comparison.append(tr);
  });
  const effectHost=document.querySelector('[data-append-scale-effect]');
  const effect=canvas(effectHost,'Same today-to-Ravi dot product with position scales 1 and 0.1',390);
  [1,.1].forEach((scale,index)=>{
    const r=lesson.appendedExperiment(lesson.sequences[0],scale),y=38+index*175;
    text(effect,20,y,'c = '+scale,C.ink,30);
    text(effect,200,y,'q₄ = '+fixed(r.q),C.q,29);
    text(effect,740,y,'k₃ (Ravi) = '+fixed(r.K[2]),C.k,29);
    text(effect,200,y+53,'Word: 0.8 × 0 + 0.2 × 1 = 0.2',C.ink,27);
    text(effect,740,y+53,'Position: '+short(r.q[2])+' × '+short(r.K[2][2])+' = '+short(r.positionTerms[2]),C.d,27);
    text(effect,200,y+106,'Total: 0.2 + '+short(r.positionTerms[2])+' = '+short(r.rawScores[2]),C.ink,27);
    text(effect,740,y+106,'Score: '+short(r.rawScores[2])+' / √3 ≈ '+r.scores[2].toFixed(3),C.k,27);
  });
  text(effect,200,375,'10× smaller slot features give a 100× smaller position product.',C.d,27);
  function drawAppended(){
    const scale=Number(appendScale.value);
    const results=lesson.sequences.map(tokens=>lesson.appendedExperiment(tokens,scale));
    document.querySelectorAll('[data-append-scale-control]').forEach(select=>{select.value=String(scale);});
    document.querySelectorAll('[data-append-query]').forEach(host=>augmentedVector(host,results[0].q,'Today’s query q₄ = '));
    results.forEach(drawCalculationTables);
  }
  drawAppended();
  document.querySelectorAll('[data-position-visual]').forEach(host=>{
    const kind=host.dataset.positionVisual,s=canvas(host,kind+' positional encoding illustration',kind.startsWith('swap')?330:kind==='shift'?310:280);
    if(kind==='swap-rows'){
      lesson.sequences.forEach((tokens,panel)=>{
        const x=20+panel*565;
        text(s,x,30,tokens.join(' '),C.ink,27);
        text(s,x,75,'Slot / word',C.ink,23);text(s,x+255,75,'eⱼ = kⱼ = vⱼ',C.e,25);
        tokens.forEach((token,j)=>{const y=117+j*43;text(s,x,y,(j+1)+' '+token,token==='today'?C.q:C.ink,26);text(s,x+255,y,fixed(lesson.embeddings[token]),token==='today'?C.q:C.e,27);});
      });
      text(s,20,316,'Query from today: q₄ = [0.8, 0.2] in both sentences',C.q,29);
    }else if(kind==='swap-result'){
      lesson.sequences.forEach((tokens,j)=>{
        const y=30+j*180;
        text(s,20,y,tokens.join(' '),C.ink,27);
        text(s,20,y+48,'Old row: '+fixed(lesson.embeddings.today),C.e,27);
        text(s,20,y+93,'Message: '+fmt(swap[j].message),C.v,27);
        arrow(s,490,y+65,595,165,C.line);
      });
      text(s,625,115,'Same output projection',C.ink,27);
      text(s,625,158,'+ same old row',C.e,27);
      line(s,625,181,1080,181);
      text(s,625,222,'Same updated row',C.d,29);
      text(s,625,270,'Same head, same probabilities',C.ink,26);
    }else if(kind==='shift'){
      [0,1].forEach(panel=>{
        const ox=70+panel*560,oy=245,scale=74,token=[1,1],p=panel?[0,1]:[1,0],result=token.map((v,j)=>v+p[j]);
        text(s,ox,30,panel?'Maya at slot B':'Maya at slot A',C.e,27);
        line(s,ox,oy,ox+205,oy);line(s,ox,oy,ox,oy-185);
        const tx=ox+scale,ty=oy-scale,rx=ox+scale*result[0],ry=oy-scale*result[1];
        dot(s,tx,ty,C.e);arrow(s,tx,ty,rx,ry,C.d);dot(s,rx,ry,C.d);
        text(s,ox+250,105,'e = [1, 1]',C.e);text(s,ox+250,155,'p = '+(panel?'[0, 1]':'[1, 0]'),C.d);text(s,ox+250,205,'e + p = '+(panel?'[1, 2]':'[2, 1]'),C.d);
        for(let tick=0;tick<=2;tick++){text(s,ox+tick*scale-5,oy+25,String(tick),C.ink,21);if(tick)text(s,ox-28,oy-tick*scale+7,String(tick),C.ink,21);}
        text(s,ox+20,294,'coordinate 1',C.ink,22);text(s,ox-48,62,'coord. 2',C.ink,22);
      });
    }else if(kind==='clock'){
      const x=170,y=145,r=85;axes(s,x,y,r);
      for(let i=0;i<4;i++){const v=rotate([1,0],i*Math.PI/2);vector(s,x,y,r,v,C.d);text(s,x+v[0]*113-18,y-v[1]*107+8,String(i),C.d);}
      text(s,400,50,'index i      [cos(iω), sin(iω)]',C.ink,26);
      ['0           [ 1,  0]','1           [ 0,  1]','2           [−1,  0]','3           [ 0, −1]','4           [ 1,  0]  ← repeats index 0'].forEach((v,i)=>text(s,400,93+i*36,v,C.d,24));
    }else if(kind==='rates'){
      text(s,25,30,'Position',C.ink,26);text(s,220,30,'Fast pair: π/2 rad / slot',C.d,26);text(s,665,30,'Slow pair: π/6 rad / slot',C.v,26);
      [0,4].forEach((i,row)=>{
        const y=90+row*106;text(s,30,y+8,'index '+i,C.ink,27);
        [[260,Math.PI/2,C.d],[705,Math.PI/6,C.v]].forEach(([x,rate,color])=>{const v=rotate([1,0],i*rate);axes(s,x,y,37);vector(s,x,y,37,v,color);text(s,x+66,y+8,fmt([v[1],v[0]]),color,27);});
      });
      text(s,220,264,'Fast pair repeats',C.d,24);text(s,665,264,'Slow pair differs',C.v,24);
    }else if(kind==='rope-queries'){
      ropeSentence(s,0);
      text(s,65,132,'Source: red (j = 2)',C.k,27);
      text(s,600,132,'Receiver: flowers (i = 3)',C.q,27);
      subscriptText(s,65,185,'k₂ = e₂ W_{K}',C.k,31);
      subscriptText(s,600,185,'q₃ = e₃ W_{Q}',C.q,31);
      text(s,65,234,'Key: matching features it offers',C.k,25);
      text(s,600,234,'Query: matching features it seeks',C.q,25);
      text(s,65,267,'Project the word representations first. Then RoPE rotates these vectors.',C.ink,24);
    }else if(kind==='rotate'){
      [[0,'red: key at slot 2',2,C.k,'k′₂'],[560,'flowers: query at slot 3',3,C.q,'q′₃']].forEach(([x,label,slot,color,name])=>{
        text(s,x+20,32,label,color,27);axes(s,x+110,149,65);
        const initial=element('g',{'stroke-dasharray':'5 4',opacity:.6});s.append(initial);vector(initial,x+110,149,65,[1,0],color);
        const rotated=rotate([1,0],slot*Math.PI/6);vector(s,x+110,149,65,rotated,color);
        text(s,x+205,106,slot+' × 30° = '+slot*30+'°',color,27);
        text(s,x+205,158,'[cos'+slot*30+'°, sin'+slot*30+'°]',color,26);
        text(s,x+205,213,name+' = '+fmt(rotated),color,26);
      });
      text(s,20,267,'Dashed: before rotation [1, 0]. Solid: after rotation.',C.ink,24);
    }else if(kind==='rope-match'){
      s.setAttribute('viewBox','0 0 1120 185');
      text(s,35,34,'flowers: rotated query',C.q,27);text(s,610,34,'red: rotated key',C.k,27);
      text(s,35,90,'q′₃ = [0.000, 1.000]',C.q,32);text(s,610,90,'k′₂ = [0.500, 0.866…]',C.k,32);
      text(s,500,90,'·',C.ink,35);
      text(s,35,167,'Raw dot product: 0 × 0.500 + 1 × 0.866… ≈ 0.866',C.ink,29);
    }else if(kind==='pairs'){
      text(s,20,45,'Query, width 4',C.q,29);text(s,370,45,'Pair 0',C.q,29);text(s,740,45,'Pair 1',C.q,29);
      text(s,20,120,'[1, 0, 0.6, 0.8]',C.q,30);arrow(s,280,112,330,112,C.line);
      text(s,370,105,'[1, 0]',C.q,29);text(s,740,105,'[0.6, 0.8]',C.q,29);
      text(s,370,160,'rotate by i × 1 rad',C.d,27);text(s,740,160,'rotate by i × 0.01 rad',C.d,27);
      text(s,20,242,'At i=3:',C.ink,27);text(s,370,242,fmt(rotate([1,0],3)),C.q,29);text(s,740,242,fmt(rotate([.6,.8],.03)),C.q,29);
    }
  });
  // One held circle: sine alone loses left/right information, the pair keeps it.
  const whyClock=document.querySelector('[data-position-clock-why]');
  const whyBase=whyClock.querySelector('[data-clock-why-base]'),whyPair=whyClock.querySelector('[data-clock-why-pair]');
  axes(whyBase,235,175,110);
  line(whyBase,95,175,380,175,C.line);line(whyBase,235,46,235,301,C.line);
  text(whyBase,180,36,'sin θ (height)',C.ink,25);
  text(whyBase,365,211,'cos θ',C.ink,25);
  text(whyBase,246,84,'1',C.ink,23);text(whyBase,241,287,'−1',C.ink,23);
  text(whyBase,520,36,'Same toy rate: ω = π/2 radians per slot.',C.ink,24);
  text(whyBase,520,94,'Index, angle',C.ink,26);text(whyBase,726,94,'Sine only',C.ink,26);
  text(whyPair,908,94,'[sin θ, cos θ]',C.d,25);
  [0,2].forEach((i,row)=>{
    const angle=i*Math.PI/2,p=[Math.cos(angle),Math.sin(angle)],x=235+110*p[0],y=175-110*p[1],baseline=153+row*64;
    arrow(whyBase,235,175,x,y,C.d);
    whyBase.append(element('circle',{cx:x,cy:y,r:7,fill:C.d,'data-clock-why-index':i,'data-angle':angle,'data-vector':JSON.stringify(p)}));
    text(whyBase,i===0?327:80,144,'i = '+i,C.d,26);
    text(whyPair,x-12,211,String(Math.round(p[0])).replace('-','−'),C.d,26);
    text(whyBase,520,baseline,i+', '+(i===0?'0':'π')+' rad',C.ink,29);
    text(whyBase,759,baseline,String(Math.round(p[1])),C.d,30);
    arrow(whyPair,818,baseline-10,887,baseline-10,C.d);
    const label=element('text',{x:910,y:baseline,'data-clock-why-offset':i,style:`fill:${C.d};font-size:29px`},'['+[p[1],p[0]].map(n=>String(Math.round(n)).replace('-','−')).join(', ')+']');whyPair.append(label);
  });
  text(whyBase,520,270,'Same height: sine cannot distinguish them.',C.ink,25);
  text(whyPair,520,315,'Different sides: cosine separates this pair.',C.d,25);

  // An isolated training illustration: never update the attention toy's tables.
  function learnedExample(){
    const word=lesson.embeddings.Maya.slice(),position=lesson.positions[2].slice();
    const gradient=[.2,-.4],rate=.1;
    const nextWord=word.map((v,j)=>v-rate*gradient[j]);
    const nextPosition=position.map((v,j)=>v-rate*gradient[j]);
    return {word,position,gradient,rate,nextWord,nextPosition,
      input:word.map((v,j)=>v+position[j]),nextInput:nextWord.map((v,j)=>v+nextPosition[j])};
  }
  const learned=learnedExample();
  const decimalRow=v=>'['+v.map(n=>n.toFixed(2).replace('-', '−')).join(', ')+']';
  function subscriptText(s,x,y,value,color,size){
    const label=element('text',{x,y,style:`fill:${color};font-size:${size}px`});
    value.split(/(_\{[^}]+\})/).forEach(part=>{
      label.append(part.startsWith('_{')
        ?element('tspan',{'baseline-shift':'sub','font-size':size*.7},part.slice(2,-1))
        :document.createTextNode(part));
    });
    s.append(label);
  }
  const lookup=document.querySelector('[data-learned-lookup]');
  const lookupBase=lookup.querySelector('[data-lookup-base]'),lookupSwap=lookup.querySelector('[data-lookup-swap]');
  function tableRow(s,x,y,label,values,color,selected=false){
    s.append(element('rect',{x,y:y-26,width:485,height:35,fill:selected?color:'none','fill-opacity':.08,stroke:selected?color:C.line}));
    text(s,x+12,y,label,C.ink,25);
    values.forEach((v,j)=>text(s,x+265+j*110,y,typeof v==='number'?v.toFixed(1).replace('-','−'):v,color,27));
  }
  subscriptText(lookupBase,20,36,'Word table E_{tok} [C × d]',C.e,30);
  text(lookupBase,20,72,'C vocabulary items, d coordinates',C.ink,24);
  subscriptText(lookupBase,610,36,'Position table P [N_{max} × d]',C.d,30);
  subscriptText(lookupBase,610,72,'N_{max} slots, d coordinates',C.ink,24);
  ['Maya','Ravi','helps'].forEach((token,j)=>tableRow(lookupBase,20,111+j*38,token,lesson.embeddings[token],C.e,j===0));
  tableRow(lookupBase,20,225,'⋮',['⋮','⋮'],C.e);
  lesson.positions.forEach((row,j)=>tableRow(lookupBase,610,111+j*38,'Slot '+(j+1),row,C.d,j===0));
  lookupSwap.append(element('rect',{x:610,y:161,width:485,height:35,fill:C.d,'fill-opacity':.08,stroke:C.d}));
  function lookupSum(s,slot,position,y){
    const result=learned.word.map((v,j)=>v+position[j]);
    const g=element('g',{'data-learned-slot':slot,'data-word-row':JSON.stringify(learned.word),'data-position-row':JSON.stringify(position),'data-sum-row':JSON.stringify(result)});s.append(g);
    text(g,20,y,'Maya, slot '+slot,C.ink,27);
    text(g,284,y,fixed(learned.word),C.e,29);text(g,458,y,'+',C.ink,29);
    text(g,510,y,fixed(position),C.d,29);text(g,725,y,'=',C.ink,29);
    text(g,785,y,(slot===1?'e₁':'e₃')+' = '+fixed(result),C.e,29);
  }
  lookupSum(lookupBase,1,lesson.positions[0],278);
  lookupSum(lookupSwap,3,learned.position,332);

  const update=document.querySelector('[data-learned-update]');
  const forward=update.querySelector('[data-update-forward]'),backward=update.querySelector('[data-update-gradient]'),step=update.querySelector('[data-update-step]');
  subscriptText(forward,20,36,'E_{tok}[Maya]',C.e,26);text(forward,20,81,fixed(learned.word),C.e,29);
  text(forward,20,151,'P[3]',C.d,26);text(forward,20,195,fixed(learned.position),C.d,29);
  line(forward,207,73,283,73,C.e);line(forward,283,73,283,120,C.e);arrow(forward,283,120,306,120,C.e);
  line(forward,237,186,283,186,C.d);line(forward,283,186,283,120,C.d);
  forward.append(element('circle',{cx:330,cy:120,r:22,fill:'var(--paper)',stroke:C.ink,'stroke-width':2}));
  text(forward,320,130,'+',C.ink,30);arrow(forward,354,120,378,120,C.ink);
  text(forward,397,99,'e₃',C.e,27);text(forward,397,143,fixed(learned.input),C.e,29);
  arrow(forward,650,120,702,120,C.ink);
  forward.append(element('rect',{x:715,y:78,width:196,height:86,rx:3,fill:'none',stroke:C.line,'stroke-width':2}));
  text(forward,735,111,'Attention',C.ink,27);text(forward,735,146,'Prediction',C.ink,27);
  arrow(forward,920,120,972,120,C.ink);
  forward.append(element('rect',{x:984,y:78,width:115,height:86,rx:3,fill:'none',stroke:C.a,'stroke-width':2}));
  text(forward,1013,111,'Loss',C.a,27);text(forward,1035,147,'L',C.a,30);
  line(backward,1041,175,1041,220,C.a);line(backward,1041,220,512,220,C.a);arrow(backward,512,220,512,162,C.a);
  line(backward,397,164,371,164,C.a);arrow(backward,371,164,351,145,C.a);
  line(backward,313,102,313,48,C.a);arrow(backward,313,48,214,48,C.a);text(backward,247,36,'g',C.a,27);
  line(backward,313,142,313,229,C.a);arrow(backward,313,229,214,229,C.a);text(backward,247,254,'g',C.a,27);
  text(backward,565,202,'Backpropagation',C.a,25);
  text(backward,480,257,'Suppose g = ∂L/∂e₃ = '+decimalRow(learned.gradient),C.a,25);
  text(step,20,295,'SGD: stored row − 0.1 × gradient',C.ink,24);
  [['word','E_{tok}[Maya]',learned.word,learned.nextWord,C.e],['position','P[3]',learned.position,learned.nextPosition,C.d]].forEach(([kind,label,before,after,color],j)=>{
    const g=element('g',{'data-learned-sgd':kind,'data-before':JSON.stringify(before),'data-gradient':JSON.stringify(learned.gradient),'data-after':JSON.stringify(after),'data-rate':learned.rate});step.append(g);
    const y=333+j*42;
    subscriptText(g,20,y,label,color,25);text(g,211,y,'←',C.ink,26);text(g,259,y,decimalRow(before),color,26);
    text(g,480,y,'− 0.1 ×',C.ink,26);text(g,590,y,decimalRow(learned.gradient),C.a,26);
    text(g,824,y,'=',C.ink,26);text(g,871,y,decimalRow(after),color,26);
  });

  function ropeSentence(s,shift){
    const words=[...(shift?['At','the','park','after','lunch']:[]),'Maya','carries','red','flowers'];
    const g=element('g',{'data-rope-sentence':String(shift)});s.append(g);
    words.forEach((word,index)=>{
      const role=word==='flowers'?'query':word==='red'?'key':'context',color=role==='query'?C.q:role==='key'?C.k:C.ink;
      const x=20+index*122+(shift?0:280),token=element('g',{'data-rope-word':word,'data-position':index,'data-role':role});g.append(token);
      text(token,x,24,String(index),color,22);text(token,x,60,word,color,27);
      line(token,x,73,x+108,73,role==='context'?C.line:color,role==='context'?1:3);
    });
  }

  const selector=document.getElementById('rope-shift');
  function shifted(shift){const i=3+shift,j=2+shift,q=rotate([1,0],i*Math.PI/6),k=rotate([1,0],j*Math.PI/6);return {i,j,q,k,dot:q[0]*k[0]+q[1]*k[1]};}
  function draw(){
    const shift=Number(selector.value),r=shifted(shift);
    const s=canvas(document.getElementById('rope-shift-picture'),'The same red–flowers pair before or after a five-word prefix',330);
    ropeSentence(s,shift);
    axes(s,180,221,76);vector(s,180,221,76,r.q,C.q);vector(s,180,221,76,r.k,C.k);
    text(s,330,135,'flowers query: i = '+r.i+' → '+r.i*30+'°',C.q,26);
    text(s,330,180,'q′ = '+fmt(r.q),C.q,27);
    text(s,740,135,'red key: j = '+r.j+' → '+r.j*30+'°',C.k,26);
    text(s,740,180,'k′ = '+fmt(r.k),C.k,27);
    text(s,330,240,'Gap: '+r.i+' − '+r.j+' = 1. Angle gap: 30°.',C.ink,28);
    text(s,330,293,shift?'Both rotate an extra 5 × 30° = 150°.':'Add five prefix words: both slots will move by 5.',C.ink,26);
    document.getElementById('rope-shift-result').textContent='Raw dot product stays '+r.dot.toFixed(3)+'; scaled score stays '+(r.dot/Math.sqrt(2)).toFixed(3)+'.';
  }
  if(selector){selector.addEventListener('change',draw);draw();}
  window.AT.positionVisuals={rotate,shifted,learnedExample};
});
