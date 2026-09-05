/* Vision IV: exact image-conditioned prefix decoder and SVG teaching figures. */
(function(){
  'use strict';
  const A=window.AT, data=window.__TOY__.vlm, cp=x=>JSON.parse(JSON.stringify(x));
  const mm=A.matmul, tr=A.transpose, add=(a,b)=>a.map((r,i)=>r.map((v,j)=>v+b[i][j]));
  const sm=A.softmax, fmt=(v,n=3)=>A.fmt(v,n), vec=r=>'['+r.map(v=>fmt(v,2)).join(', ')+']';
  function params(s='before'){if(!data.snapshots[s])throw Error('Unknown VLM snapshot.');return cp(data.snapshots[s]);}
  function vision(name='two'){
    if(!data.images[name])throw Error('Unknown image.');
    const image=cp(data.images[name]),patches=[];
    for(const y of [0,2])for(const x of [0,2])patches.push([image[y][x],image[y][x+1],image[y+1][x],image[y+1][x+1]]);
    const embedded=mm(patches,data.frozenVision.W_patch),E=[[1,1]].concat(add(embedded,data.frozenVision.positions));
    const Q=E.map(r=>r.map(v=>v*.5)),K=cp(Q),V=cp(E),scores=mm(Q,tr(K)).map(r=>r.map(v=>v/Math.sqrt(2))),weights=scores.map(sm);
    return {image,patches,W_patch:data.frozenVision.W_patch,positions:data.frozenVision.positions,embedded,E,Q,K,V,scores,A:weights,G:add(E,mm(weights,V)).slice(1)};
  }
  function forward(name='two',prefix=data.prompt,options={}){
    const p=params(options.snapshot),vi=vision(name);
    if(!Array.isArray(prefix)||!prefix.length||prefix.length>8||prefix.some(t=>!data.vocab.includes(t)))throw Error('Unsupported VLM text prefix.');
    const ids=prefix.map(t=>data.vocab.indexOf(t));
    const bridged=mm(vi.G,p.W_bridge).map(r=>r.map((v,j)=>v+p.b_bridge[j]));
    const E=add(bridged.concat(ids.map(i=>p.E_tok[i])),p.P.slice(0,4+ids.length));
    const Q=mm(E,p.W_Q),K=mm(E,p.W_K),V=mm(E,p.W_V),raw=mm(Q,tr(K));
    const allowed=E.map((r,i)=>E.map((_,j)=>i<4?j<4:j<=i));
    const scores=raw.map((r,i)=>r.map((v,j)=>allowed[i][j]?v/Math.sqrt(3):-Infinity)),weights=scores.map(sm);
    const message=mm(weights,V),delta=mm(message,p.W_O),out=add(E,delta);
    const logits=mm(out,p.W_vocab).map(r=>r.map((v,j)=>v+p.b_vocab[j])),probs=logits.map(sm);
    return {image:name,prefix:prefix.slice(),ids,vision:vi,bridged,E,Q,K,V,raw,scores,allowed,A:weights,message,delta,out,logits,probs};
  }
  function teacher(name='two',options={}){
    const targets=data.answers[name],f=forward(name,data.prompt.concat(targets.slice(0,-1)),options);
    const losses=targets.map((t,i)=>-Math.log(f.probs[i+6][data.vocab.indexOf(t)]));
    return Object.assign(f,{targets:targets.slice(),losses,loss:losses.reduce((s,v)=>s+v,0)/losses.length});
  }
  // Exact bookkeeping for this single attention block followed by a linear head.
  // These are source-row terms in one forward pass, not raw-pixel causal effects.
  function contributions(name='two',prefix=data.prompt,options={}){
    const f=forward(name,prefix,options),p=params(options.snapshot),i=f.E.length-1;
    const a=data.vocab.indexOf('two'),b=data.vocab.indexOf('one'),direction=p.W_vocab.map(r=>r[a]-r[b]);
    const dot=(r,s)=>r.reduce((total,v,j)=>total+v*s[j],0),sources=f.V.map((v,j)=>{
      const weighted=v.map(x=>x*f.A[i][j]),update=mm([weighted],p.W_O)[0];
      return {label:j<4?'image '+(j+1):prefix[j-4],weight:f.A[i][j],value:v.slice(),weighted,update,contrast:dot(update,direction)};
    });
    const residual=dot(f.E[i],direction)+p.b_vocab[a]-p.b_vocab[b];
    return {f,row:i,sources,residual,image:sources.slice(0,4).reduce((s,r)=>s+r.contrast,0),text:sources.slice(4).reduce((s,r)=>s+r.contrast,0),total:f.logits[i][a]-f.logits[i][b]};
  }
  function generate(name='two',options={}){
    const prefix=data.prompt.slice(),trace=[];let stoppedBy='limit';
    const limit=options.limit==null?5:options.limit;
    if(!Number.isInteger(limit)||limit<1||limit>6)throw Error('Generation limit must be 1 through 6.');
    for(let i=0;i<limit;i++){
      const f=forward(name,prefix,options),row=f.E.length-1,chosen=data.vocab[A.argmax(f.probs[row])];
      trace.push({prefix:prefix.slice(),query:f.Q[row],weights:f.A[row],logits:f.logits[row],probs:f.probs[row],chosen,row});
      prefix.push(chosen);if(chosen==='<eos>'){stoppedBy='eos';break;}
    }
    return {tokens:prefix.slice(data.prompt.length),trace,stoppedBy};
  }
  let serial=0;
  const ns='http://www.w3.org/2000/svg';
  function el(tag,attrs={},text){const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!=null)n.textContent=text;return n;}
  function canvas(title,height=280){
    const id='vlm-'+(++serial),svg=el('svg',{viewBox:'0 0 1100 '+height,role:'img','aria-labelledby':id+'-t '+id+'-desc',class:'vlm-diagram'});
    svg.append(el('title',{id:id+'-t'},title),el('desc',{id:id+'-desc'},title+'. Schematic shapes show computation; numerical values come from the disclosed toy.'));
    const defs=el('defs');svg.append(defs);
    for(const c of ['e','q','k','v','a','d','neutral']){const m=el('marker',{id:id+'-'+c,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'});m.append(el('path',{d:'M0 0L10 5L0 10Z',fill:c==='neutral'?'var(--ink-2)':'var(--c-'+c+')'}));defs.append(m);}
    function text(x,y,t,c='neutral',size=22,anchor='middle'){const n=el('text',{x,y,'text-anchor':anchor,'dominant-baseline':'middle',fill:c==='neutral'?'var(--ink)':'var(--c-'+c+')','font-family':'var(--font-ui)','font-size':size},t);svg.append(n);return n;}
    function box(x,y,w,h,t,sub='',c='e'){
      svg.append(el('rect',{x:x-w/2,y:y-h/2,width:w,height:h,rx:8,fill:c==='neutral'?'var(--card)':'var(--t-'+c+')',stroke:c==='neutral'?'var(--line)':'var(--c-'+c+')','stroke-width':1.6}));
      text(x,y-(sub?13:0),t,c,26);if(sub)text(x,y+18,sub,'neutral',22);
    }
    function arrow(path,c='neutral',dashed=false){svg.append(el('path',{d:path,fill:'none',stroke:c==='neutral'?'var(--ink-2)':'var(--c-'+c+')','stroke-width':2,'marker-end':'url(#'+id+'-'+c+')',...(dashed?{'stroke-dasharray':'5 5'}:{})}));}
    return {svg,text,box,arrow};
  }
  function pixels(c,name,cx,cy,size=36){
    const image=data.images[name];
    image.forEach((r,y)=>r.forEach((v,x)=>{const shade=A.imageShade(v);c.svg.append(el('rect',{x:cx-2*size+x*size,y:cy-2*size+y*size,width:size-2,height:size-2,rx:2,'data-pixel-value':v,fill:`rgb(${shade},${shade},${shade})`,stroke:'var(--line)'}));c.text(cx-1.5*size+x*size,cy-1.5*size+y*size,String(v),'neutral',18).setAttribute('fill',shade<118?'#FFFFFF':'#000000');}));
  }
  function diagram(stage,options={}){
    const name=options.image||'two',snapshot=options.snapshot||'before';let c;
    if(stage==='task'){
      c=canvas('Image plus question predicts an answer',220);pixels(c,name,115,95,37);
      c.box(430,85,310,80,'Question: count?', 'Same prompt','q');
      c.box(880,85,280,65,'Answer tokens','one block / two blocks','d');c.arrow('M590 85L735 85','q');
      c.arrow('M190 140L690 140L735 110','e');
      c.text(115,196,'4 × 4 grid','neutral',22);c.text(715,164,'Pixels must affect the next-token distribution.','neutral',22);
    }else if(stage==='scene-bridge'){
      c=canvas('Mug-scene patches become language-model context',305);
      c.box(145,65,270,82,'Image patches','mugs, book, background','e');
      c.box(510,65,280,82,'Vision encoder','contextual visual rows G','e');
      c.box(910,65,340,82,'Connector','G → decoder-width rows','d');
      c.arrow('M283 65L366 65','e');c.arrow('M653 65L736 65','d');
      c.box(310,230,510,82,'Known text','How many mugs?','e');
      c.box(910,230,340,82,'Language decoder','image + known text rows','a');
      c.arrow('M910 109L910 185','d');c.arrow('M568 230L736 230','e');
    }else if(stage==='pipeline'){
      const level=options.level==null?3:options.level;c=canvas('Image rows become language-model input',level<3?130:250);
      c.box(125,65,200,65,'Four patches','4 rows × 4 pixels','e');
      if(level>=1){c.box(410,65,240,65,'Vision encoder G','4 × 2 encoded rows','e');c.arrow('M228 65L287 65','e');}
      if(level>=2){c.box(785,65,350,80,'Visual connector','G W_bridge + b: 4 × 3','d');c.arrow('M533 65L607 65','d');}
      if(level>=3){c.box(265,185,380,80,'Text lookup + positions','3 prompt rows, width 3','e');c.box(785,185,350,80,'Prefix decoder','7 rows → token logits','a');c.arrow('M785 108L785 142','d');c.arrow('M458 185L607 185','e');}
    }else if(stage==='routes'){
      c=canvas('Two ways to connect images and language',310);
      c.text(275,22,'Projected visual prefix','e',26);c.text(825,22,'Separate cross-attention','q',26);
      c.box(150,105,220,66,'Image rows','project to LM width','e');c.box(410,105,230,66,'Text rows','known prompt / answer','e');
      c.box(280,235,420,80,'Joint self-attention','image + text keys and values','a');c.arrow('M150 140L150 176L280 176L280 192','e');c.arrow('M410 140L410 176L280 176','e');
      c.box(685,105,230,66,'Vision memory','separate K and V','v');c.box(957,105,230,66,'Text decoder','supplies Q','q');
      c.box(825,235,420,80,'Read image values','text states receive the update','d');c.arrow('M685 140L685 177L825 177L825 192','v');c.arrow('M957 140L957 177L825 177','q');
    }else if(stage==='mask'){
      const f=teacher(name,{snapshot}),n=f.E.length;c=canvas('Visual-prefix attention mask',365);
      const cell=31,left=240,top=52;
      for(let i=0;i<n;i++){c.text(left-15,top+(i+.5)*cell,String(i+1),'neutral',22);for(let j=0;j<n;j++)c.svg.append(el('rect',{x:left+j*cell+1,y:top+i*cell+1,width:cell-2,height:cell-2,fill:f.allowed[i][j]?'var(--t-a)':'var(--card)',stroke:f.allowed[i][j]?'var(--c-a)':'var(--line)','data-allowed':f.allowed[i][j]}));}
      for(let j=0;j<n;j++)c.text(left+(j+.5)*cell,top-15,String(j+1),'neutral',22);
      c.text(785,83,'Rows 1–4: image tokens','e',26);c.text(785,119,'Read all image rows; no text.','neutral',22);
      c.text(785,188,'Rows 5–9: known text','q',26);c.text(785,224,'Read every image row','neutral',22);c.text(785,253,'and current / earlier text.','neutral',22);
      c.text(385,350,'Columns supply keys and values.','neutral',22);
    }else if(stage==='generation'){
      const g=generate(name,{snapshot}),i=options.step||0,s=g.trace[i];if(!s)throw Error('Invalid generation step');
      c=canvas('The image stays available at every generation step',340);
      pixels(c,name,102,87,26);c.text(102,163,'Same image','e',22);
      c.box(470,75,460,82,'Known text prefix',s.prefix.join(' '),'e');
      c.text(470,141,'Last known token: '+s.prefix.at(-1)+' (row '+(s.row+1)+')','q',22);
      c.box(890,75,320,82,'Current query','['+s.query.map(v=>fmt(v)).join(', ')+']','q');c.arrow('M703 75L727 75','q');
      c.box(455,240,430,82,'Decoder reads image + text','same learned parameters','a');
      c.box(890,240,320,82,'Next token: '+s.chosen,'p = '+fmt(s.probs[data.vocab.indexOf(s.chosen)]),'d');
      c.arrow('M102 182L102 240L237 240','e');c.arrow('M455 171L455 195','e');c.arrow('M890 119L890 166L610 166L610 195','q');c.arrow('M673 240L727 240','d');
      c.text(550,318,s.chosen==='<eos>'?'Stop at the end marker.':'Append the chosen token. It supplies the next query.','neutral',22);
    }else if(stage==='image-weights'){
      const f=forward(name),weights=f.A[6],mass=weights.slice(0,4).reduce((s,v)=>s+v,0);c=canvas('Attention on image slots, without renormalizing away the text',315);
      pixels(c,name,160,143,43);c.text(160,265,'Original grid','neutral',22);
      for(let j=0;j<4;j++){const x=450+(j%2)*260,y=75+Math.floor(j/2)*115;c.box(x,y,230,85,'image '+(j+1),(100*weights[j]).toFixed(2)+'% of all weight','a');}
      c.box(960,132,245,112,'Text rows',(100*(1-mass)).toFixed(2)+'% of all weight','e');
      c.text(700,280,'Four image slots together: '+(100*mass).toFixed(2)+'%.','a',24);
    }else if(stage==='contrast'){
      const rows=['two','one'].map(n=>contributions(n)),labels=['residual + bias','image value terms','text value terms','total: two − one'];
      c=canvas('Exact signed contributions to the two versus one logit difference',340);
      const groups=[rows.map(r=>r.residual),rows.map(r=>r.image),rows.map(r=>r.text),rows.map(r=>r.total)],max=Math.max(...groups.flat().map(Math.abs)),scale=195/max;
      for(let j=0;j<2;j++){const x=440+j*395;c.text(x,22,j?'One-block image':'Two-block image','neutral',25);c.svg.append(el('path',{d:`M${x} 52V290`,stroke:'var(--line)','stroke-width':2}));}
      labels.forEach((label,k)=>{const y=75+k*65;c.text(8,y,label,'neutral',22,'start');for(let j=0;j<2;j++){const value=groups[k][j],x=440+j*395,w=Math.abs(value)*scale;c.svg.append(el('rect',{x:value<0?x-w:x,y:y-13,width:w,height:26,fill:k===3?'var(--c-d)':'var(--c-v)',opacity:k===3?1:.55}));c.text(x+(value<0?-1:1)*(w+8),y,(value>0?'+':'')+fmt(value),'neutral',22,value<0?'end':'start');}});
      c.text(665,320,'Positive favors two. Negative favors one.','neutral',22);
    }else if(stage==='training'){
      c=canvas('Response loss trains the connector and decoder',315);
      c.box(150,66,260,80,'Fixed vision encoder','no parameter updates','e');
      c.box(495,66,300,68,'Learned connector','W_bridge, b_bridge','d');c.arrow('M283 66L341 66','d');
      c.box(150,207,260,80,'Known text','E_tok + P','e');c.box(495,207,300,80,'Decoder + head','Attention + vocabulary','a');c.arrow('M283 207L341 207','e');c.arrow('M495 103L495 164','d');
      c.box(885,207,300,80,'Answer-token loss','mean −log p(target)','neutral');c.arrow('M648 207L732 207');
      c.box(885,66,300,68,'Observed response','two · blocks · <eos>','neutral');c.arrow('M885 103L885 170');
      c.arrow('M1038 207L1080 207L1080 289L493 289L493 250','d',true);c.text(790,272,'autograd','d',22);
    }else if(stage==='thermal'){
      c=canvas('The same colour can encode different temperatures',265);
      for(const [cx,low,high] of [[270,30,50],[825,60,100]]){
        for(let i=0;i<8;i++)c.svg.append(el('rect',{x:cx-160+i*40,y:70,width:40,height:60,fill:'var(--c-e)',opacity:.2+.1*i}));
        c.text(cx,29,'Palette range '+low+' to '+high+' °C','neutral',26);c.arrow('M'+cx+' 173L'+cx+' 134','neutral');
        c.text(cx,205,'Midpoint = '+((low+high)/2)+' °C','neutral',26);
      }
      c.text(550,244,'Same normalized colours; different calibration ranges.','neutral',22);
    }else throw Error('Unknown VLM diagram stage.');
    return c.svg;
  }
  const notation=[
    ['matrix','\\ve{G}','Encoded visual rows before the connector','N\\times d_{\\rm vision}','4×2'],
    ['matrix','\\ve{B}=\\ve{G}W_{\\rm bridge}+b_{\\rm bridge}','Visual rows after the connector, before decoder positions','N\\times d_{\\rm model}','4×3'],
    ['matrix','P','Same-width position vectors added to image and text rows','(N+T)\\times d_{\\rm model}','7×3 for the prompt'],
    ['sizes','W_{\\rm bridge},b_{\\rm bridge}','Maps each visual row to the decoder width','d_{\\rm vision}\\times d_{\\rm model}','2×3; bias 1×3'],
    ['matrix','\\ve{E}','Visual and text rows after adding positions','(N+T)\\times d_{\\rm model}','7×3 for the prompt'],
    ['matrix','\\vq{Q}=\\ve{E}W_Q,\\quad\\vk{K}=\\ve{E}W_K','Matching projections; query/key widths must agree','(N+T)\\times d_k','7×3'],
    ['matrix','\\vv{V}=\\ve{E}W_V','Information to send; V is not G or E','(N+T)\\times d_v','7×3'],
    ['matrix','M','Allowed visual-prefix information flow','(N+T)\\times(N+T)','7×7 for the prompt'],
    ['token','m_i=\\sum_j\\va{\\alpha_{ij}}\\vv{v_j}','Message retrieved by the current query','1\\times d_v','1×3'],
    ['token','\\vd{\\Delta e_i}=m_iW_O,\\quad\\vp{e_i^\\prime}=\\ve{e_i}+\\vd{\\Delta e_i}','Output projection followed by the residual addition','1\\times d_{\\rm model}','1×3'],
    ['token','z_i=\\vp{e_i^\\prime}W_{\\mathrm{vocab}}+b','Vocabulary logits, not attention weights','1\\times|\\mathcal V|','1×8'],
    ['sizes','E_{\\rm tok}','Learned text-token lookup table','|\\mathcal V|\\times d_{\\rm model}','8×3'],
    ['sizes','W_{\\mathrm{vocab}},\\;b','Vocabulary prediction weights and bias','d_{\\rm model}\\times|\\mathcal V|,\\;1\\times|\\mathcal V|','3×8; bias 1×8']
  ];
  notation.forEach(([g,sym,mean,shape,d])=>A.notation.push({g,sym,mean,shape,dims:()=>d,parts:['vision4']}));
  A.axes.named=false;
  A.vlm={data,params,vision,forward,teacher,generate,contributions,diagram,vec,fmt};
})();
