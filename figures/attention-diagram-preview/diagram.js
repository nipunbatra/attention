/* Standalone SVG authoring: no DOM, libraries, fetches or external fonts needed. */
(function (root) {
  'use strict';
  const D = root.ATTENTION_PREVIEW_DATA;
  const C = { ink:'#243b39', muted:'#65756f', line:'#d8e1dc', paper:'#fbfcf9', q:'#8b2cde', k:'#aa4e08', v:'#0f766e', a:'#be123c', m:'#0f766e', d:'#147737', e:'#245edb' };
  const {dModel:dm,dKey:dk,dValue:dv,T,vocabSize:nv}=D.dims;
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  const f = (n, digits=3) => Number(n).toFixed(digits).replace(/^-/,'−');
  const sub = n => String(n).split('').map(x => '₀₁₂₃₄₅₆₇₈₉'[Number(x)]).join('');
  const vec = (a, digits=3) => '(' + a.map(v => f(v,digits)).join('   ') + ')';
  const txt = (x,y,s,cls='body',more='') => `<text x="${x}" y="${y}" class="${cls}" ${more.replace(/fill="([^"]+)"/g,'style="fill:$1"')}>${esc(s)}</text>`;
  const line = (x1,y1,x2,y2,more='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${more}/>`;
  const rect = (x,y,w,h,fill='white',stroke=C.line,r=8,more='') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" ${more}/>`;
  const stages = [
    {id:'input',short:'Input',title:'Start with the same input rows',lines:['The sentence is already given. We are first updating bank,','at position 7—not predicting the word bank.'],focus:'e',code:['i = 6  # position 7, using zero-based indexing','e = E[i:i+1]  # keep a row: [1, 4]'],note:['E stores the current token representations as rows.','Here: token embedding + position information.']},
    {id:'query',short:'Query',title:'Bank makes a query',lines:['Transform bank’s current representation into a request.','Its three numbers can match the three coordinates of a key.'],focus:'q',code:['q = e @ W_Q  # [1, 4] @ [4, 3] → [1, 3]'],note:['The query is a vector, not an English question.','This toy asks for setting clues; the numbers are hand-designed.']},
    {id:'keys',short:'Keys',title:'Every token offers a matching key',lines:['Each source describes when it may be useful.','Bank’s one query will be compared with all these keys.'],focus:'k',code:['K = E @ W_K  # [10, 4] @ [4, 3] → [10, 3]'],note:['A key helps determine how much attention a source receives.','It is not the information payload we will mix.']},
    {id:'scores',short:'Scores',title:'Compare the query with every key',lines:['Multiply matching coordinates, then add.','We get one raw score per source position.'],focus:'score',code:['scores = q @ K.T  # [1, 3] @ [3, 10] → [1, 10]'],note:['A large score is a strong match for this query.','It need not belong to bank itself.']},
    {id:'scale',short:'Scale',title:'Scale the scores before softmax',lines:['There are dₖ = 3 matching coordinates. Divide by √3.','Scaling controls score spread; it does not change their order.'],focus:'scale',code:['scores = scores / math.sqrt(q.shape[-1])'],note:['With independent unit-variance coordinates, raw dot products','have variance dₖ. The divisor keeps their typical spread stable.']},
    {id:'mask',short:'Mask',title:'Bank cannot read the future',lines:['For receiver 7, source positions 8–10 are not allowed.','Set their scores to −∞ before applying softmax.'],focus:'mask',code:['future = torch.arange(E.shape[0], device=E.device) > i','scores = scores.masked_fill(future, float("-inf"))'],note:['Causal means j ≤ i: earlier positions and the current position.','The future words are shown only to make the mask visible.']},
    {id:'weights',short:'Weights',title:'Turn scores into mixing weights',lines:['Softmax acts across the source-token columns.','Allowed weights sum to 1; masked positions get exactly 0.'],focus:'weights',code:['alpha = scores.softmax(dim=-1)  # [1, 10]'],note:['These are attention weights over input positions.','They are not next-token probabilities.']},
    {id:'values',short:'Values',title:'Values carry what each token sends',lines:['Project each source into a value vector.','Values take a separate route: they do not set the scores.'],focus:'v',code:['V = E @ W_V  # [10, 4] @ [4, 3] → [10, 3]'],note:['This snapshot happens to use W_K = W_V. Their roles differ;','in a learned model the two matrices can differ independently.']},
    {id:'mixture',short:'Mix',title:'Collect a message for this query',lines:['Multiply each value by its attention weight, then add.','The result m₇ is a 3-number message specifically for bank.'],focus:'mix',code:['m = alpha @ V  # [1, 10] @ [10, 3] → [1, 3]'],note:['The mixture lives in value space. It is not yet the updated','4-number representation, and it is not a predicted word.']},
    {id:'output',short:'Map back',title:'Map the message back to four coordinates',lines:['W_O maps the value-space message into representation space.','Now Δe₇ has the right shape to be added to e₇.'],focus:'output',code:['delta_e = m @ W_O  # [1, 3] @ [3, 4] → [1, 4]'],note:['This output projection bridges the two coordinate spaces.','The 3-number mixture and 4-number update are different objects.']},
    {id:'residual',short:'Add',title:'Keep bank, and add its context update',lines:['The long bypass carries the original e₇ unchanged.','Add Δe₇ to obtain the context-enriched representation e′₇.'],focus:'residual',code:['e_prime = e + delta_e  # [1, 4] + [1, 4] → [1, 4]'],note:['This changes the representation of this occurrence of bank.','It does not overwrite the shared embedding table.']},
    {id:'prediction',short:'Predict',title:'Now switch to the last token: the',lines:['Run the same path again with receiver 10, the final the.','Its updated representation predicts the token at position 11.'],focus:'prediction',code:['logits = e_prime @ W_vocab + b_vocab  # [1, 20]','p_next = logits.softmax(dim=-1)  # over vocabulary tokens'],note:['All projections still read the original E in this layer.','The new query does not read the just-computed e′₇.']}
  ];
  // Keep every shape tied to the same model used by the arithmetic.
  stages[0].code[1]=`e = E[i:i+1]  # keep a row: [1, ${dm}]`;
  if(D.provenance.positionIgnored) stages[0].note=['E includes token embedding + position. In this toy,','the projections ignore position; the causal mask still applies.'];
  stages[1].lines[1]=`Its ${dk} numbers match the ${dk} coordinates of each key.`;
  stages[1].code=[`q = e @ W_Q  # [1, ${dm}] @ [${dm}, ${dk}] → [1, ${dk}]`];
  stages[2].code=[`K = E @ W_K  # [${T}, ${dm}] @ [${dm}, ${dk}] → [${T}, ${dk}]`];
  stages[3].code=[`scores = q @ K.T  # [1, ${dk}] @ [${dk}, ${T}] → [1, ${T}]`];
  stages[4].lines[0]=`There are dₖ = ${dk} matching coordinates. Divide by √${dk}.`;
  stages[4].note=['For independent, zero-mean, unit-variance coordinates,','dot-product variance is dₖ; after scaling it is 1.'];
  stages[6].code=[`alpha = scores.softmax(dim=-1)  # [1, ${T}]`];
  stages[7].code=[`V = E @ W_V  # [${T}, ${dm}] @ [${dm}, ${dv}] → [${T}, ${dv}]`];
  stages[7].note=['Keys describe matching; values carry the message to retrieve.',`Here a key has ${dk} coordinates and a value has ${dv}.`];
  stages[8].lines[1]=`The result m₇ is a ${dv}-number message specifically for bank.`;
  stages[8].code=[`m = alpha @ V  # [1, ${T}] @ [${T}, ${dv}] → [1, ${dv}]`];
  stages[8].note=['The mixture lives in value space. It is not yet the updated',`${dm}-number representation, and it is not a predicted word.`];
  stages[9].title=`Map the message back to ${dm} coordinates`;
  stages[9].code=[`delta_e = m @ W_O  # [1, ${dv}] @ [${dv}, ${dm}] → [1, ${dm}]`];
  stages[9].note=['This output projection bridges the two coordinate spaces.',`The ${dv}-number mixture and ${dm}-number update are different objects.`];
  stages[10].code=[`e_prime = e + delta_e  # [1, ${dm}] + [1, ${dm}] → [1, ${dm}]`];
  stages[11].code[0]=`logits = e_prime @ W_vocab + b_vocab  # [1, ${nv}]`;
  function node(x,y,w,h,title,caption,role,show,active,prefix) {
    if (!show) return '';
    const color=C[role]||C.ink;
    return `<g class="node" data-node="${esc(prefix)}">${rect(x,y,w,h,active?color+'15':'#ffffff',active?color:C.line,8,`stroke-width="${active?2.5:1.5}"`)}${active?rect(x,y,4,h,color,color,2):''}${txt(x+w/2,y+h*.45,title,'node-title',`text-anchor="middle" fill="${color}"`)}${txt(x+w/2,y+h*.76,caption,'node-caption','text-anchor="middle"')}</g>`;
  }
  function arrow(path,role,show,active,prefix,label) {
    if(!show) return '';
    return `<path d="${path}" fill="none" stroke="${C[role]||C.ink}" stroke-width="${active?2.8:1.8}" opacity="${active?1:.65}" marker-end="url(#${prefix}-${role})"/>`+(label||'');
  }
  function vectorRow(x,y,arr,axes,role,label,w=620) {
    const color=C[role]||C.ink, cell=(w-12*(arr.length-1))/arr.length;
    let s=txt(x,y,label,'smallcap',`fill="${color}"`);
    arr.forEach((a,j)=>{
      const xx=x+j*(cell+12);
      s+=txt(xx+cell/2,y+30,axes[j],'axis','text-anchor="middle"');
      s+=rect(xx,y+42,cell,60,color+'0c',color+'55',5);
      s+=txt(xx+cell/2,y+81,f(a),'number',`text-anchor="middle" fill="${color}"`);
    });
    return s;
  }
  function table(rows,headers,widths,y=326,colors=[]) {
    let s='',x=60;
    headers.forEach((h,j)=>{s+=txt(x+(j?widths[j]/2:0),y,h,'table-head',j?'text-anchor="middle"':'');x+=widths[j];});
    s+=line(60,y+16,700,y+16,`stroke="${C.line}"`);
    rows.forEach((row,i)=>{
      const yy=y+44+i*30, muted=row.future, color=muted?'#a9b4af':C.ink;
      if(row.highlight)s+=rect(48,yy-22,664,29,'#eef5ef','none',3);
      let xx=60;
      row.cells.forEach((value,j)=>{
        s+=txt(xx+(j?widths[j]/2:0),yy,String(value),j?'table-number':'table-label',`${j?'text-anchor="middle" ':''}fill="${muted?color:colors[j]||color}"`);xx+=widths[j];
      });
    });
    return s;
  }
  function evidence(stage, d) {
    const er=D.axes.short.e;
    const kr=D.axes.short.qk;
    const vr=D.axes.short.v.map(s=>s.replace(/^→/,'')+' scene');
    let s='';
    if(stage===0) {
      s+=vectorRow(60,343,d.e,er,'e','e₇ · BANK’S INPUT REPRESENTATION');
      s+=txt(60,508,`${T} input positions × ${dm} coordinates = shape [${T}, ${dm}]`,'body');
      s+=txt(60,548,'Only the receiver row is highlighted; the input stays fixed.','detail');
    } else if(stage===1) {
      s+=txt(60,345,`e₇ = ${vec(d.e,2)}`,'formula');
      s+=txt(60,402,`e₇ [1 × ${dm}]   ·   W_Q [${dm} × ${dk}]   →   q₇ [1 × ${dk}]`,'body');
      s+=vectorRow(60,460,d.q,kr,'q','q₇ · THE REQUEST');
    } else if(stage===2||stage===7) {
      const vals=stage===2?d.keys:d.values, role=stage===2?'k':'v';
      const axes=stage===2?kr:vr, widths=[245,...axes.map(()=>396/axes.length)];
      s+=table(vals.map((a,j)=>({cells:[`${j+1} · ${D.tokens[j]}`,...a.map(x=>f(x,2))],highlight:j===5,future:j>d.index})),['source token',...axes],widths,358,['',...axes.map(()=>C[role])]);
      s+=txt(60,696,stage===2?`Each key has ${dk} matching coordinates.`:`Each value has ${dv} payload coordinates.`,'detail');
    } else if(stage===3) {
      const j=5;
      s+=txt(60,345,'One match, worked out: bank’s query · river’s key','smallcap',`fill="${C.q}"`);
      s+=txt(60,397,`${vec(d.q,2)} · ${vec(d.keys[j],2)}`,'formula');
      s+=txt(60,445,`= ${f(d.rawScores[j])}   (before scaling)`,'formula');
      s+=table([1,5,6,8].map(j=>({cells:[`${j+1} · ${D.tokens[j]}`,f(d.rawScores[j])],highlight:j===5,future:j>d.index})),['some source positions','raw score'],[380,260],504,['',C.q]);
    } else if(stage===4) {
      s+=txt(60,345,`dₖ = ${dk}      √dₖ ≈ ${f(Math.sqrt(dk))}`,'formula');
      s+=table([0,1,5,6,8].map(j=>({cells:[`${j+1} · ${D.tokens[j]}`,f(d.rawScores[j]),`÷ √${dk}`,f(d.scaledScores[j])],highlight:j===5,future:j>d.index})),['source token','raw score','scale','scaled'],[245,132,132,132],402,['',C.q,C.muted,C.q]);
      s+=txt(60,641,'The same positive divisor is used for every score.','detail');
    } else if(stage===5) {
      s+=table(d.maskedScores.map((n,j)=>({cells:[`${j+1} · ${D.tokens[j]}`,f(d.scaledScores[j]),n===null?'−∞':f(n),j>d.index?'future':'allowed'],highlight:j===6,future:j>d.index})),['source token','scaled','after mask','status'],[245,132,132,132],358);
    } else if(stage===6) {
      s+=txt(60,355,'source token','table-head');s+=txt(655,355,'weight','table-head','text-anchor="end"');
      d.alpha.forEach((a,j)=>{
        const y=389+j*30, color=j>d.index?'#acb5b0':j===5?C.k:C.a;
        s+=txt(60,y,`${j+1} · ${D.tokens[j]}`,'table-label',`fill="${color}"`);
        s+=rect(247,y-15,335,17,'#f0f3ef','none',3);
        if(a>0)s+=rect(247,y-15,335*a,17,color,'none',3);
        s+=txt(655,y,f(a),'table-number',`text-anchor="end" fill="${color}"`);
      });
      s+=txt(60,701,'Sum = 1.000 · computed before display rounding','detail');
    } else if(stage===8) {
      s+=table([1,5,6].map(j=>({cells:[`${j+1} · ${D.tokens[j]}`,f(d.alpha[j]),vec(d.values[j].map(v=>v*d.alpha[j]),3)],highlight:j===5})),['three contributions','weight','weighted value'],[220,100,321],358,['',C.a,C.v]);
      s+=vectorRow(60,509,d.mixture,vr,'m','m₇ · SUM OF ALL SEVEN ALLOWED CONTRIBUTIONS');
    } else if(stage===9) {
      s+=vectorRow(60,337,d.mixture,vr,'m',`m₇ · ${dv} COORDINATES IN VALUE SPACE`);
      s+=txt(370,512,`↓  W_O [${dv} × ${dm}]`,'body','text-anchor="middle"');
      s+=vectorRow(60,550,d.delta,er,'d',`Δe₇ · ${dm} COORDINATES IN REPRESENTATION SPACE`);
    } else if(stage===10) {
      s+=table(er.map((name,j)=>({cells:[name,f(d.e[j]),f(d.delta[j]),f(d.updated[j])]})),['coordinate','e₇','+ Δe₇','= e′₇'],[245,132,132,132],347,['',C.e,C.d,C.d]);
      s+=txt(60,558,'bank keeps its original representation and receives context.','detail');
      s+=txt(60,606,`e′₇ ≈ ${vec(d.updated)}`,'formula',`fill="${C.d}"`);
    } else {
      s+=txt(60,354,'NEW RECEIVER: 10 · the','smallcap',`fill="${C.q}"`);
      s+=txt(60,399,`q₁₀ = ${vec(d.q)}`,'formula',`fill="${C.q}"`);
      s+=txt(60,451,`e′₁₀ ≈ ${vec(d.updated)}`,'formula',`fill="${C.d}"`);
      s+=table(d.topVocabulary.slice(0,4).map((x,i)=>({cells:[x.token,f(x.logit),f(x.probability)],highlight:i===0})),['next-token candidate','logit','probability'],[320,160,160],480,['','',C.d]);
      s+=txt(60,679,`This softmax is over ${nv} vocabulary tokens, not ${T} inputs.`,'detail');
    }
    return s;
  }
  function render(stage=0,prefix='attention') {
    const si=Math.max(0,Math.min(stages.length-1,stage));
    const st=stages[si],d=si===11?D.last:D.bank,n=d.position,sn=sub(n);
    const is = id => st.focus===id || si===11;
    let out=`<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="960" viewBox="0 0 1600 960" role="img" aria-labelledby="${prefix}-title ${prefix}-desc"><title id="${prefix}-title">Stage ${si+1}: ${esc(st.title)}</title><desc id="${prefix}-desc">${esc(st.lines.join(' '))} Q and K create weights; V carries information. A separate residual path preserves the input representation. Receiver: ${n}, ${esc(d.token)}.</desc><defs>`;
    for(const role of ['q','k','v','a','m','d','e'])out+=`<marker id="${prefix}-${role}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L8 4 L0 8 Z" fill="${C[role]}"/></marker>`;
    out+=`</defs><style>text{font-family:'Avenir Next','Segoe UI',Arial,sans-serif;fill:${C.ink}}.eyebrow{font-size:14px;font-weight:700;letter-spacing:1.8px}.heading{font-size:33px;font-weight:600;letter-spacing:-.6px}.stage-title{font-size:27px;font-weight:600;letter-spacing:-.4px}.body{font-size:21px}.detail{font-size:17px;fill:${C.muted}}.axis{font-size:16px;fill:${C.muted}}.smallcap{font-size:15px;font-weight:700;letter-spacing:.6px}.formula{font-family:'STIX Two Text',Georgia,serif;font-size:26px}.number{font-family:'SFMono-Regular',Consolas,monospace;font-size:24px}.table-number{font-family:'SFMono-Regular',Consolas,monospace;font-size:18px}.table-label{font-size:18px}.table-head{font-size:15px;font-weight:600;fill:${C.muted}}.node-title{font-size:20px;font-weight:600}.node-caption{font-size:13px;fill:${C.muted}}.code{font-family:'SFMono-Regular',Consolas,monospace;font-size:17px}.token{font-size:19px}.position{font-size:12px;fill:${C.muted}}.tiny{font-size:13px;fill:${C.muted}}</style>`;
    out+=rect(0,0,1600,960,C.paper,'none',0);
    out+=txt(48,44,'PART 2 / A SINGLE ATTENTION HEAD','eyebrow',`fill="${C.muted}"`);
    out+=txt(48,87,'How context reaches a token','heading');
    out+=txt(1538,43,`${String(si+1).padStart(2,'0')} / ${stages.length}`,'eyebrow','text-anchor="end"');
    out+=txt(1538,82,'One input sequence · causal reads · one residual update','detail','text-anchor="end"');
    let tx=48;
    D.tokens.forEach((t,j)=>{
      const w=[70,146,65,100,70,81,87,72,127,70][j];
      const active=j===d.index, future=j>d.index;
      out+=rect(tx,117,w,42,active?C.q:future?'#f0f2ee':'white',active?C.q:C.line,5);
      out+=txt(tx+w/2,144,t,'token',`text-anchor="middle" fill="${active?'white':future?'#a2ada6':C.ink}"`);
      out+=txt(tx+w/2,178,`${j+1}${active?' · receiver':future?' · future':''}`,'position',`text-anchor="middle"${active?` fill="${C.q}"`:''}`);
      tx+=w+9;
    });
    out+=txt(tx+15,146,'___','token',`fill="${si===11?C.v:'#a2ada6'}"`);
    out+=txt(tx+30,178,'11 · predict','position','text-anchor="middle"');
    out+=line(48,202,1552,202,`stroke="${C.line}"`);
    out+=txt(48,249,`${String(si+1).padStart(2,'0')}   ${st.title}`,'stage-title');
    st.lines.forEach((t,j)=>out+=txt(60,287+j*28,t,'body'));
    out+=evidence(si,d);
    out+=rect(48,735,685,101,'#eef3ee','none',6);
    out+=txt(66,760,'PYTORCH · THE ACTIVE OPERATION','smallcap',`fill="${C.muted}"`);
    st.code.forEach((t,j)=>out+=txt(66,790+j*25,t,'code'));
    st.note.forEach((t,j)=>out+=txt(60,873+j*25,t,'detail'));
    out+=line(768,224,768,910,`stroke="${C.line}"`);
    /* Shared source at the top; one fixed graph for every reveal. */
    out+=node(1030,224,210,60,'Input E',`${T} × ${dm} · unchanged snapshot`,'e',true,is('e'),'input');
    out+=txt(1474,241,`receiver ${n} · ${d.token}`,'smallcap',`fill="${C.q}" text-anchor="end"`);
    const qx=915,kx=1135,vx=1390;
    out+=arrow(`M1080 284 V299 H${qx} V319`,'q',si>=1,is('q'),prefix);
    out+=arrow(`M${kx} 284 V319`,'k',si>=2,is('k'),prefix);
    out+=arrow(`M1190 284 V299 H${vx} V319`,'v',si>=7,is('v'),prefix);
    if(si>=1)out+=rect(859,292,112,21,C.paper,'none',0)+txt(915,308,`W_Q · ${dm} × ${dk}`,'tiny','text-anchor="middle"');
    if(si>=2)out+=txt(1203,314,`W_K · ${dm} × ${dk}`,'tiny');
    if(si>=7)out+=rect(1334,292,112,21,C.paper,'none',0)+txt(1390,308,`W_V · ${dm} × ${dv}`,'tiny','text-anchor="middle"');
    out+=node(qx-80,319,160,58,`Query q${sn}`,`1 × ${dk} · request`,'q',si>=1,is('q'),'q');
    out+=node(kx-80,319,160,58,'Keys K',`${T} × ${dk} · matching`,'k',si>=2,is('k'),'k');
    out+=node(vx-80,319,160,58,'Values V',`${T} × ${dv} · payload`,'v',si>=7,is('v'),'v');
    if(si>=3) {
      out+=rect(901,403,344,330,'none','#c6d3cb',10,'stroke-dasharray="4 5"');
      out+=txt(922,395,'SCALED DOT-PRODUCT ATTENTION','tiny');
    }
    out+=arrow(`M${qx} 377 V411 H1015 V426`,'q',si>=3,is('score'),prefix);
    out+=arrow(`M${kx} 377 V411 H1130 V426`,'k',si>=3,is('score'),prefix);
    out+=node(941,426,263,46,`q${sn} Kᵀ`,`raw scores · 1 × ${T}`,'q',si>=3,is('score'),'score');
    const cx=1072;
    out+=arrow(`M${cx} 472 V484`,'a',si>=4,is('scale'),prefix);
    out+=node(971,484,203,42,`Divide by √${dk}`,`scaled scores · 1 × ${T}`,'a',si>=4,is('scale'),'scale');
    out+=arrow(`M${cx} 526 V538`,'a',si>=5,is('mask'),prefix);
    out+=node(971,538,203,42,'Causal mask',`allow source j ≤ ${n}`,'a',si>=5,is('mask'),'mask');
    out+=arrow(`M${cx} 580 V592`,'a',si>=6,is('weights'),prefix);
    out+=node(971,592,203,46,'Softmax',`weights α${sn} · 1 × ${T}`,'a',si>=6,is('weights'),'weights');
    out+=arrow(`M${cx} 638 V669`,'a',si>=8,is('mix'),prefix);
    out+=arrow(`M${vx} 377 V692 H1204`,'v',si>=8,is('mix'),prefix);
    if(si>=8){out+=txt(1475,475,'Values bypass','tiny','text-anchor="middle"');out+=txt(1475,494,'score + softmax','tiny','text-anchor="middle"');}
    out+=node(941,669,263,46,`α${sn} V = m${sn}`,`query-specific mixture · 1 × ${dv}`,'m',si>=8,is('mix'),'mix');
    out+=arrow(`M${cx} 715 V750`,'m',si>=9,is('output'),prefix);
    out+=node(971,750,203,46,`× W_O [${dv} × ${dm}]`,`update Δe${sn} · 1 × ${dm}`,'d',si>=9,is('output'),'output');
    if(si>=10) {
      out+=arrow(`M1030 254 H807 V852 H1048`,'e',true,is('residual'),prefix);
      out+=txt(827,825,`keep original e${sn}`,'tiny');
      out+=arrow(`M${cx} 796 V828`,'d',true,is('residual'),prefix);
      out+=`<circle cx="${cx}" cy="852" r="24" fill="#edf5ef" stroke="${C.d}" stroke-width="2"/>`;
      out+=txt(cx,860,'+','formula','text-anchor="middle"');
      out+=arrow(`M1096 852 H1147`,'d',true,is('residual'),prefix);
      out+=node(1147,824,171,58,`e′${sn}`,`updated row · 1 × ${dm}`,'d',true,is('residual'),'updated');
    }
    if(si>=11) {
      out+=arrow('M1318 852 H1373','d',true,true,prefix);
      out+=node(1373,824,175,58,'Vocabulary head',`${nv} logits → probabilities`,'d',true,true,'prediction');
      out+=txt(1460,909,`${d.topVocabulary[0].token} · p ≈ ${f(d.topVocabulary[0].probability)}`,'detail',`text-anchor="middle" fill="${C.d}"`);
    }
    out+=line(48,924,1552,924,`stroke="${C.line}"`);
    out+=txt(48,949,'SELF: Q, K and V come from the same E.   CAUSAL: receiver i reads only positions j ≤ i.','tiny');
    out+=txt(1552,949,'Hand-designed toy · decimals rounded · no FFN or normalization shown','tiny','text-anchor="end"');
    // Inline SVG styles must not recolor other diagrams in the article.
    out=out.replace('<svg xmlns=',`<svg id="${prefix}-svg" xmlns=`);
    out=out.replace(/<style>([\s\S]*?)<\/style>/,(_,css)=>'<style>'+css.replace(/(^|})([^{}]+)\{/g,(_,end,selector)=>end+'#'+prefix+'-svg '+selector+'{')+'</style>');
    return out+'</svg>';
  }
  root.ATTENTION_PREVIEW={stages,render};
})(globalThis);
