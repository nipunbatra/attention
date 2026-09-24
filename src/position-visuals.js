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
  document.querySelectorAll('[data-appended-projections]').forEach(host=>{
    const index=host.dataset.appendedProjections==='a'?0:1,tokens=lesson.sequences[index];
    const r=lesson.appendedExperiment(tokens);
    const s=canvas(host,tokens.join(' ')+': input times identity gives the complete query and key matrices',380);
    function matrix(name,values,x,y,width,color,label){
      const group=element('g',{'data-projection-matrix':name});s.append(group);
      text(group,x,32,label,color,29);
      text(group,x,65,values.length+' × '+values[0].length,C.ink,23);
      values.forEach((row,j)=>row.forEach((value,c)=>{
        const cx=x+c*width,cy=y+j*52;
        group.append(element('rect',{x:cx,y:cy,width,height:52,fill:color,'fill-opacity':name==='Q'&&j===3?.14:.035,stroke:C.line,'stroke-width':1}));
        group.append(element('text',{x:cx+width/2,y:cy+34,'text-anchor':'middle','data-row':j,'data-column':c,'data-value':value,style:`fill:${name==='input'&&c===2?C.d:color};font-size:29px`},name==='identity'||c===2?String(value):value.toFixed(1)));
      }));
      if(name==='Q')group.append(element('rect',{x,y:y+3*52,width:3*width,height:52,fill:'none',stroke:C.q,'stroke-width':3,'data-query-row':'3'}));
    }
    text(s,12,65,'Slot / word',C.ink,23);
    tokens.forEach((token,j)=>s.append(element('text',{x:12,y:112+j*52+34,'data-source-row':j,style:`fill:${token==='today'?C.q:C.ink};font-size:26px`},(j+1)+' '+token)));
    matrix('input',r.rows,160,112,66,C.e,'Input Ẽ');
    matrix('identity',r.WQ,420,138,44,C.ink,'I₃');
    matrix('Q',r.Q,628,112,66,C.q,'Queries Q');
    matrix('K',r.K,890,112,66,C.k,'Keys K');
    text(s,378,231,'×',C.ink,32);text(s,583,231,'=',C.ink,32);text(s,849,231,'=',C.ink,32);
    text(s,160,362,'Two word coordinates + slot',C.ink,25);
    text(s,628,362,'q₄ = ['+r.q.map((value,c)=>c===2?String(value):value.toFixed(1)).join(', ')+']',C.q,29);
  });
  const appendScale=document.getElementById('position-append-scale');
  const short=value=>Number(value.toFixed(3)).toString();
  const dotFormula=(q,k)=>q.map((value,c)=>short(value)+' × '+short(k[c])).join(' + ');
  // One experimental setting follows the student through all calculation stages.
  document.querySelectorAll('[data-append-control]').forEach((host,index)=>{
    const sentence=document.createElement('span');sentence.textContent=host.dataset.appendControl+'.';
    const label=document.createElement('label');label.htmlFor='append-detail-scale-'+index;label.textContent='Position scale c =';
    const select=document.createElement('select');select.id=label.htmlFor;select.dataset.appendScaleControl='';
    for(const value of ['1','0.1']){const option=document.createElement('option');option.value=value;option.textContent=value;select.append(option);}
    host.append(sentence,label,select);
    select.addEventListener('change',()=>{appendScale.value=select.value;drawAppended();});
  });
  function numericMatrix(s,name,values,x,y,cw,color,format,highlight={}){
    const group=element('g',{'data-append-matrix':name});s.append(group);
    values.forEach((row,i)=>row.forEach((value,j)=>{
        const marked=highlight.row===i||highlight.column===j,blocked=!Number.isFinite(value)||(highlight.causal&&j>i);
        group.append(element('rect',{x:x+j*cw,y:y+i*48,width:cw,height:48,fill:blocked?C.line:color,'fill-opacity':blocked?.6:marked?.13:.025,stroke:C.line}));
      group.append(element('text',{x:x+(j+.5)*cw,y:y+i*48+32,'text-anchor':'middle','data-row':i,'data-column':j,'data-value':String(value),style:`fill:${color};font-size:25px`},format(value)));
    }));
    if(highlight.row!==undefined)group.append(element('rect',{x,y:y+highlight.row*48,width:values[0].length*cw,height:48,fill:'none',stroke:color,'stroke-width':2}));
  }
  function drawProduct(host,r,tokens){
    const s=canvas(host,tokens.join(' ')+': Q times K transpose gives every raw dot product',390);
    text(s,80,30,'Q',C.q,29);text(s,80,61,'4 × 3',C.ink,23);
    text(s,340,30,'Kᵀ',C.k,29);text(s,340,61,'3 × 4',C.ink,23);
    text(s,728,30,'QKᵀ',C.ink,29);text(s,728,61,'4 × 4',C.ink,23);
    numericMatrix(s,'Q',r.Q,80,105,54,C.q,short,{row:3});
    const kt=r.K[0].map((_,c)=>r.K.map(row=>row[c]));
    numericMatrix(s,'KT',kt,340,129,54,C.k,short,{column:2});
    numericMatrix(s,'raw',r.rawMatrix,728,105,84,C.ink,value=>value.toFixed(2),{row:3});
    s.append(element('rect',{x:728+2*84,y:105+3*48,width:84,height:48,fill:'none',stroke:C.ink,'stroke-width':4,'data-selected-product':'3,2'}));
    for(let i=0;i<4;i++){
      text(s,43,137+i*48,String(i+1),C.ink,23);
      text(s,360+i*54,114,String(i+1),C.k,23);
      text(s,701,137+i*48,String(i+1),C.q,23);
      text(s,763+i*84,91,String(i+1),C.k,23);
    }
    text(s,278,215,'×',C.ink,32);text(s,633,215,'=',C.ink,32);
    text(s,30,336,'Slots: '+tokens.map((token,j)=>(j+1)+' '+token).join(', '),C.ink,24);
    text(s,30,378,'q₄ · k₃ = '+dotFormula(r.q,r.K[2])+' = '+short(r.rawScores[2]),C.ink,27);
  }
  function drawMatrixPair(host,results){
    const attention=host.dataset.appendMatrices==='attention';
    const s=canvas(host,attention?'Causal attention weights for both sentences':'Scaled, causally masked scores for both sentences',365);
    results.forEach((r,index)=>{
      const left=index*560,tokens=lesson.sequences[index],color=attention?C.a:C.k;
      text(s,left+12,30,tokens.join(' '),C.ink,25);
      text(s,left+145,69,(attention?'A':'S')+' · 4 × 4',color,26);
      text(s,left+145,98,'Source slot j',C.k,22);
      for(let j=0;j<4;j++)text(s,left+178+j*78,124,String(j+1),C.k,23);
      numericMatrix(s,(attention?'attention':'masked')+'-'+index,attention?r.attentionMatrix:r.maskedMatrix,left+145,139,78,color,value=>Number.isFinite(value)?value.toFixed(3):'−∞',{row:3,causal:true});
      tokens.forEach((token,i)=>text(s,left+12,171+i*48,(i+1)+' '+token,C.q,23));
    });
    text(s,12,355,'Rows: receiver i. The highlighted row belongs to today.',C.ink,24);
  }
  function drawCalculationTables(r,index){
    const suffix=index?'-b':'',tokens=lesson.sequences[index];
    const scores=document.querySelector('#position-append-scores'+suffix+' tbody');scores.replaceChildren();
    const softmax=document.querySelector('#position-append-softmax'+suffix+' tbody');softmax.replaceChildren();
    const exps=r.scores.map(Math.exp),denominator=exps.reduce((a,b)=>a+b,0);
    tokens.forEach((token,j)=>{
      const tr=document.createElement('tr');cell(tr,(j+1)+' '+token);
      const word=cell(tr,r.wordTerms[j],'numbers score');
      word.textContent=dotFormula(r.q.slice(0,2),r.K[j].slice(0,2))+' = '+short(r.wordTerms[j]);
      const position=cell(tr,r.positionTerms[j],'numbers append-index');
      position.textContent=short(r.q[2])+' × '+short(r.K[j][2])+' = '+short(r.positionTerms[j]);
      cell(tr,r.rawScores[j],'numbers append-totals');scores.append(tr);
      const row=document.createElement('tr');cell(row,token);
      const scaled=cell(row,r.scores[j],'numbers score');scaled.textContent=short(r.rawScores[j])+' / √3 = '+r.scores[j].toFixed(3);
      const exponential=cell(row,exps[j],'numbers');exponential.textContent=exps[j].toFixed(2);
      const weight=cell(row,r.weights[j],'numbers weight');weight.textContent=exps[j].toFixed(2)+' / '+denominator.toFixed(2)+' ≈ '+(100*r.weights[j]).toFixed(1)+'%';softmax.append(row);
    });
    document.getElementById('position-append-denominator'+suffix).textContent='Sum: '+exps.map(x=>x.toFixed(2)).join(' + ')+' ≈ '+denominator.toFixed(2);
    document.getElementById('position-append-normalization'+suffix).textContent='Normalize this sentence’s row using that sum. Displayed numbers are rounded.';
  }
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
  text(effect,200,375,'10× smaller in Q and in K gives a 100× smaller position product.',C.d,27);
  function drawAppended(){
    const scale=Number(appendScale.value);
    const results=lesson.sequences.map(tokens=>lesson.appendedExperiment(tokens,scale));
    document.querySelectorAll('[data-append-scale-control]').forEach(select=>{select.value=String(scale);});
    document.querySelectorAll('[data-append-query]').forEach(host=>{host.textContent='Today’s query q₄ = '+fixed(results[0].q);});
    document.querySelectorAll('[data-append-product]').forEach(host=>{
      const index=host.dataset.appendProduct==='a'?0:1;drawProduct(host,results[index],lesson.sequences[index]);
    });
    document.querySelectorAll('[data-append-matrices]').forEach(host=>drawMatrixPair(host,results));
    results.forEach(drawCalculationTables);
    results.forEach((r,index)=>{
      const suffix=index?'b':'a',body=document.querySelector('#position-append-weights-'+suffix+' tbody');
      body.replaceChildren();
      lesson.sequences[index].forEach((token,j)=>{
        const tr=document.createElement('tr');cell(tr,token);
        cell(tr,r.scores[j],'numbers score');
        const weight=cell(tr,r.weights[j],'numbers weight');
        weight.textContent=(100*r.weights[j]).toFixed(1)+'%';body.append(tr);
      });
    });
    document.getElementById('position-append-query').textContent='q₄ = '+fixed(results[0].q);
    document.getElementById('position-append-outcome').textContent=scale===1
      ?'Today gets '+results.map(r=>(100*r.weights[3]).toFixed(1)+'%').join(' and ')+'. The raw slot index dominates in this toy.'
      :'Today gets '+results.map(r=>(100*r.weights[3]).toFixed(1)+'%').join(' and ')+'. Scaling by 0.1 made the slot term 100× smaller.';
  }
  appendScale.addEventListener('change',drawAppended);drawAppended();
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
      text(s,25,30,'Position',C.ink,26);text(s,220,30,'Fast pair: 90° / slot',C.d,26);text(s,665,30,'Slow pair: 30° / slot',C.v,26);
      [0,4].forEach((i,row)=>{
        const y=90+row*106;text(s,30,y+8,'index '+i,C.ink,27);
        [[260,Math.PI/2,C.d],[705,Math.PI/6,C.v]].forEach(([x,rate,color])=>{const v=rotate([1,0],i*rate);axes(s,x,y,37);vector(s,x,y,37,v,color);text(s,x+66,y+8,fmt(v),color,27);});
      });
      text(s,220,264,'Fast pair repeats',C.d,24);text(s,665,264,'Slow pair differs',C.v,24);
    }else if(kind==='rotate'){
      axes(s,190,150,95);vector(s,190,150,95,[1,0],C.q);vector(s,190,150,95,rotate([1,0],Math.PI/3),C.k);
      text(s,320,160,'q = [1, 0]',C.q,28);text(s,300,62,'R₆₀°q = [0.500, 0.866]',C.k,28);
      text(s,650,112,'0.500 = 1 × cos60° − 0 × sin60°',C.k,23);text(s,650,175,'0.866 = 1 × sin60° + 0 × cos60°',C.k,23);
      text(s,80,270,'Rotate content; keep its length.',C.ink,26);
    }else if(kind==='pairs'){
      text(s,20,45,'Query, width 4',C.q,29);text(s,370,45,'Pair 0',C.q,29);text(s,740,45,'Pair 1',C.q,29);
      text(s,20,120,'[1, 0, 0.6, 0.8]',C.q,30);arrow(s,280,112,330,112,C.line);
      text(s,370,105,'[1, 0]',C.q,29);text(s,740,105,'[0.6, 0.8]',C.q,29);
      text(s,370,160,'rotate by i × 1 rad',C.d,27);text(s,740,160,'rotate by i × 0.01 rad',C.d,27);
      text(s,20,242,'At i=3:',C.ink,27);text(s,370,242,fmt(rotate([1,0],3)),C.q,29);text(s,740,242,fmt(rotate([.6,.8],.03)),C.q,29);
    }
  });
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

  const limits=document.querySelector('[data-learned-limits]');
  const limitsBase=limits.querySelector('[data-limits-base]'),unseen=limits.querySelector('[data-limits-unseen]'),missing=limits.querySelector('[data-limits-missing]');
  subscriptText(limitsBase,20,34,'Example: N_{max} = 4. Training uses only slots 1–3.',C.ink,28);
  for(let slot=1;slot<=5;slot++){
    const s=slot<4?limitsBase:slot===4?unseen:missing,y=83+(slot-1)*44,color=slot<4?C.d:slot===4?C.ink:C.a;
    const g=element('g',{'data-learned-coverage':slot<4?'visited':slot===4?'unvisited':'missing','data-slot':slot});s.append(g);
    text(g,25,y+9,'Slot '+slot,C.ink,26);
    g.append(element('rect',{x:165,y:y-21,width:235,height:40,fill:'none',stroke:color,'stroke-width':2,'stroke-dasharray':slot<4?'none':'7 5'}));
    text(g,185,y+9,slot===5?'No P[5]':'P['+slot+']',color,27);
    if(slot===2){arrow(g,416,y,469,y,C.d);text(g,490,y+9,'Training adapts these rows',C.d,28);}
    if(slot===4){arrow(g,416,y,469,y,C.ink);text(g,490,y+9,'Row exists, but gets no task gradient',C.ink,28);}
    if(slot===5){arrow(g,416,y,469,y,C.a);text(g,490,y+9,'No allocated row for this slot',C.a,28);}
  }
  text(missing,25,320,'Independent rows have no built-in distance rule.',C.ink,26);
  const selector=document.getElementById('rope-shift');
  function shifted(shift){const i=3+shift,j=2+shift,q=rotate([1,0],i*Math.PI/6),k=rotate([1,0],j*Math.PI/6);return {i,j,q,k,dot:q[0]*k[0]+q[1]*k[1]};}
  function draw(){
    const r=shifted(Number(selector.value));const s=canvas(document.getElementById('rope-shift-picture'),'Rotate queries and keys while preserving their relative angle',250);
    axes(s,210,125,95);vector(s,210,125,95,r.q,C.q);vector(s,210,125,95,r.k,C.k);
    text(s,390,60,'Query slot '+r.i+': '+fmt(r.q),C.q,28);text(s,390,112,'Key slot '+r.j+': '+fmt(r.k),C.k,28);
    text(s,390,173,'Dot product: '+r.dot.toFixed(3),C.ink,31);text(s,390,226,'Slot gap = 1 → angle gap = 30°',C.ink,27);
    document.getElementById('rope-shift-result').textContent='Both vectors moved, but their match stays cos(30°) ≈ '+r.dot.toFixed(3)+'.';
  }
  if(selector){selector.addEventListener('change',draw);draw();}
  window.AT.positionVisuals={rotate,shifted,learnedExample};
});
