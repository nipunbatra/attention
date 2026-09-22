/* Diagram-first position lesson. All plotted additions read the same toy data
   as the score tables. Authored SVG build groups exist before presenter boot. */
document.addEventListener('DOMContentLoaded',()=>{
  const NS='http://www.w3.org/2000/svg';
  const C={e:'var(--c-e)',p:'var(--c-d)',q:'var(--c-q)',k:'var(--c-k)',v:'var(--c-v)',a:'var(--c-a)',ink:'var(--ink)',muted:'var(--ink-2)',line:'var(--line)',paper:'var(--card)'};
  const L=AT.positionLesson;
  const vec=(a,n=1)=>'['+a.map(x=>(Math.abs(x)<.5*10**-n?0:x).toFixed(n)).join(', ')+']';
  const add=(a,b)=>a.map((x,j)=>x+b[j]);
  function el(tag,attrs={},content){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(content!==undefined)e.textContent=content;return e;}
  function text(s,x,y,value,color=C.ink,size=24,anchor='start'){
    const t=el('text',{x,y,'text-anchor':anchor,style:`fill:${color};font-size:${size}px`});
    for(const part of value.split(/(_(?:vocab|last|tok|[TQKVOi]))/g)){
      if(part.startsWith('_')){
        t.append(el('tspan',{dy:5,style:`font-size:${size*.75}px`},part.slice(1)));
        t.append(el('tspan',{dy:-5},'\u200b'));
      }
      else t.append(document.createTextNode(part));
    }
    s.append(t);return t;
  }
  function line(s,x1,y1,x2,y2,color=C.line,width=2,extra={}){s.append(el('line',{x1,y1,x2,y2,stroke:color,'stroke-width':width,...extra}));}
  function arrow(s,x1,y1,x2,y2,color=C.p){line(s,x1,y1,x2,y2,color,3);const a=Math.atan2(y2-y1,x2-x1),r=9;line(s,x2,y2,x2-r*Math.cos(a-.5),y2-r*Math.sin(a-.5),color,3);line(s,x2,y2,x2-r*Math.cos(a+.5),y2-r*Math.sin(a+.5),color,3);}
  function point(s,x,y,color,hollow=false,attrs={}){const p=el('circle',{cx:x,cy:y,r:6,fill:hollow?C.paper:color,stroke:color,'stroke-width':2.5,...attrs});s.append(p);return p;}
  function canvas(host,title,h=360){const s=el('svg',{viewBox:`0 0 1120 ${h}`,role:'img','aria-label':title});s.append(el('title',{},title));host.replaceChildren(s);return s;}
  function plane(s,ox,oy,scale){
    const xy=a=>[ox+a[0]*scale,oy-a[1]*scale];
    for(const t of [0,.5,1,1.5]){const x=ox+t*scale;line(s,x,oy-1.12*scale,x,oy+.3*scale);text(s,x,oy+.3*scale+26,String(t),C.muted,21,'middle');}
    for(const t of [0,.5,1]){const y=oy-t*scale;line(s,ox-.1*scale,y,ox+1.6*scale,y);text(s,ox-17,y+7,String(t),C.muted,21,'end');}
    line(s,ox-.1*scale,oy,ox+1.6*scale,oy,C.muted);line(s,ox,oy+.3*scale,ox,oy-1.12*scale,C.muted);
    text(s,ox+1.7*scale,oy+8,'c₁',C.muted,23);text(s,ox-8,oy-1.12*scale-12,'c₂',C.muted,23);
    return xy;
  }
  const originalOffsets={Maya:[16,5],Ravi:[16,6],helps:[-8,-16],today:[-18,-18]};
  function plotOriginal(s,xy){
    for(const[token,a]of Object.entries(L.embeddings)){const[x,y]=xy(a),[dx,dy]=originalOffsets[token];point(s,x,y,C.e,true,{'data-original':token,'data-vector':JSON.stringify(a)});text(s,x+dx,y+dy,token,C.e,23);}
  }
  for(const s of document.querySelectorAll('[data-position-moves]')){
    const index=Number(s.dataset.positionMoves),base=s.querySelector('[data-move-base]'),tokens=L.sequences[index];
    const xy=plane(base,85,270,200);plotOriginal(base,xy);
    text(base,535,28,'Word row + slot offset = new input',C.ink,27);
    text(base,535,63,'Green point numbers identify the slots.',C.muted,23);
    tokens.forEach((token,j)=>{
      const g=s.querySelector(`[data-move-slot="${j}"]`),a=L.embeddings[token],p=L.positions[j],b=add(a,p),[x,y]=xy(a),[xx,yy]=xy(b);
      if(j)arrow(g,x,y,xx,yy);
      point(g,xx,yy,C.p,false,{'data-endpoint':token,'data-slot':j+1,'data-vector':JSON.stringify(b)});
      const dy=index===1&&token==='Maya'?24:-9;
      text(g,xx+10,yy+dy,String(j+1),C.p,23);
      text(g,535,105+j*72,`${j+1}  ${token}${j===0?' (stays here)':''}`,C.p,25);
      text(g,560,136+j*72,`${vec(a)} + ${vec(p)} = ${vec(b)}`,C.ink,24);
    });
  }
  function wave(s,{y,rate,end,label,color}){
    const left=170,w=880,amp=44,mid=y+65;
    text(s,15,y+14,label,color,25);text(s,15,y+47,`ω = ${rate}`,color,23);
    text(s,15,y+78,'rad / slot',C.muted,21);
    for(const v of [-1,0,1]){line(s,left,mid-v*amp,left+w,mid-v*amp);text(s,left-18,mid-v*amp+7,String(v),C.muted,21,'end');}
    for(let t=0;t<=4;t++){const x=left+w*t/4;line(s,x,mid-amp,x,mid+amp);text(s,x,mid+amp+28,String(end*t/4),C.muted,22,'middle');}
    for(const[fn,dash,name]of [[Math.sin,'','sine'],[Math.cos,'8 5','cosine']]){
      const pts=Array.from({length:1201},(_,i)=>{const t=end*i/1200;return `${left+w*t/end},${mid-amp*fn(t*rate)}`;}).join(' ');
      s.append(el('polyline',{points:pts,fill:'none',stroke:color,'stroke-width':3,'stroke-dasharray':dash,'data-wave':name,'data-rate':rate,'data-end':end}));
    }
  }
  function ring(s,x,y,r,angle,color){
    s.append(el('circle',{cx:x,cy:y,r,fill:'none',stroke:C.line,'stroke-width':2}));
    line(s,x-r,y,x+r,y);line(s,x,y-r,x,y+r);
    point(s,x+r,y,C.muted,true);arrow(s,x,y,x+r*Math.cos(angle),y-r*Math.sin(angle),color);
  }
  function absolutePair(i,j){const p=n=>[Math.cos(n*Math.PI/6),Math.sin(n*Math.PI/6)],q=add([1,0],p(i)),k=add([1,0],p(j));return {q,k,score:q[0]*k[0]+q[1]*k[1]};}
  function overview(s,focus){
    // One map changes its highlight without changing node locations. Background paths
    // remain legible; only the current stage has the stronger outline.
    function group(stage){const g=el('g',{'data-map-stage':stage});s.append(g);return g;}
    const input=group('input'),attn=group('attention'),out=group('output');
    function box(g,x,y,w,h,title,caption,role,id){
      const active=focus==='all'||focus===g.dataset.mapStage,color=C[role];
      g.append(el('rect',{x,y,width:w,height:h,rx:8,fill:active&&focus!=='all'?`color-mix(in srgb, ${color} 9%, ${C.paper})`:C.paper,stroke:active?color:C.line,'stroke-width':active?2.5:1.5,'data-map-node':id}));
      text(g,x+w/2,y+26,title,color,24,'middle');text(g,x+w/2,y+53,caption,C.muted,21,'middle');
    }
    function route(g,coords,color){coords.slice(1,-1).forEach((p,i)=>line(g,...coords[i],...p,color,2.5));arrow(g,...coords.at(-2),...coords.at(-1),color);}
    box(input,10,15,178,65,'Word IDs','E_tok[ids] · T × 4','e','tokens');
    box(input,10,125,178,65,'Slots 0…T−1','P[slots] · T × 4','p','positions');
    box(input,235,70,148,72,'Add rows','E · T × 4','e','input');
    route(input,[[188,47],[210,47],[210,95],[235,95]],C.e);route(input,[[188,157],[210,157],[210,120],[235,120]],C.p);
    box(input,435,5,172,65,'q_T = e_T W_Q','1 × 3','q','query');
    box(input,435,100,172,65,'K = EW_K','T × 3','k','keys');
    box(input,435,210,172,65,'V = EW_V','T × 2','v','values');
    route(input,[[383,95],[405,95],[405,37],[435,37]],C.q);
    route(input,[[383,108],[414,108],[414,132],[435,132]],C.k);
    route(input,[[383,124],[397,124],[397,242],[435,242]],C.v);
    box(attn,655,32,172,70,'q_T Kᵀ / √3','scores · 1 × T','a','scores');
    route(attn,[[607,37],[630,37],[630,53],[655,53]],C.q);route(attn,[[607,132],[637,132],[637,84],[655,84]],C.k);
    box(attn,885,32,220,70,'Mask, then softmax','weights α_T · 1 × T','a','weights');
    arrow(attn,827,67,885,67,C.a);
    box(attn,885,210,220,70,'m_T = α_T V','message · 1 × 2','v','message');
    arrow(attn,995,102,995,210,C.a);text(attn,1013,161,'α_T',C.a,24);
    arrow(attn,607,242,885,242,C.v);text(attn,653,224,'Values bypass scoring',C.v,22);
    box(out,885,350,220,70,'Δe_T = m_T W_O','W_O: 2 × 4','p','projection');
    arrow(out,995,280,995,350,C.v);
    box(out,655,350,172,70,'e′_T = e_T + Δe_T','updated · 1 × 4','p','residual');
    arrow(out,885,385,827,385,C.p);
    route(out,[[309,142],[309,310],[741,310],[741,350]],C.e);text(out,334,301,'Residual keeps e_T (already includes position)',C.e,22);
    box(out,435,350,172,70,'ReLU hidden','4 → 8 units','q','hidden');arrow(out,655,385,607,385,C.p);
    box(out,10,350,373,70,'Logits → vocabulary softmax','hW_vocab + b · 1 × C','a','prediction');arrow(out,435,385,383,385,C.q);
  }
  for(const host of document.querySelectorAll('[data-position-journey]')){
    const kind=host.dataset.positionJourney,s=canvas(host,kind+' positional encoding diagram',kind==='overview'?440:kind==='period'?235:kind==='absolute-shift'?280:kind==='updated'?285:360);
    if(kind==='initial'||kind==='positioned'){
      const positioned=kind==='positioned';
      L.sequences.forEach((tokens,panel)=>{
        const x=30+panel*560;text(s,x,28,tokens.join(' '),C.ink,27);
        const xy=plane(s,x+55,268,165);
        if(!positioned)plotOriginal(s,xy);
        else tokens.forEach((token,j)=>{
          const a=add(L.embeddings[token],L.positions[j]),[px,py]=xy(a);
          point(s,px,py,C.p,false,{'data-positioned-token':token,'data-sentence':panel,'data-vector':JSON.stringify(a)});
          const offsets=panel?{Ravi:[14,6],helps:[15,-5],Maya:[14,28],today:[14,-8]}:{Maya:[14,6],helps:[15,-5],Ravi:[14,6],today:[14,6]};
          const[dx,dy]=offsets[token];text(s,px+dx,py+dy,token,C.p,23);
        });
      });
    }else if(kind==='updated'){
      L.sequences.forEach((tokens,j)=>{
        const r=L.experiment(tokens,true),y=35+j*139,updated=add(r.q,r.message);
        text(s,20,y,tokens.join(' '),C.ink,27);
        text(s,20,y+47,'Message m₄ = '+vec(r.message,3),C.v,28);
        arrow(s,462,y+40,545,y+40,C.v);
        text(s,572,y+13,'e′₄ = e₄ + Δe₄',C.p,28);
        const result=text(s,572,y+58,vec(r.q,3)+' + '+vec(r.message,3),C.ink,25);
        result.dataset.message=JSON.stringify(r.message);
        const t=text(s,572,y+97,'= '+vec(updated,3),C.p,29);t.dataset.updatedVector=JSON.stringify(updated);
      });
    }else if(kind==='waves'){
      wave(s,{y:18,rate:1,end:20,label:'Fast pair',color:C.p});wave(s,{y:188,rate:.01,end:20,label:'Slow pair',color:C.v});
      text(s,1080,348,'index i',C.muted,22,'end');
    }else if(kind==='period'){
      wave(s,{y:20,rate:.01,end:650,label:'Slow pair',color:C.v});
      const x=170+880*(2*Math.PI/.01)/650;line(s,x,36,x,155,C.v,2,{'stroke-dasharray':'3 4'});
      text(s,170,211,'One full turn at i ≈ 628.32 (continuous index)',C.v,27);
    }else if(kind==='absolute-shift'){
      [[3,2],[8,7]].forEach(([i,j],panel)=>{
        const r=absolutePair(i,j),x=20+560*panel;
        text(s,x,34,`Query index ${i}, key index ${j}`,C.ink,27);
        text(s,x,89,'Query after addition: '+vec(r.q,3),C.q,25);
        text(s,x,136,'Key after addition:    '+vec(r.k,3),C.k,25);
        const t=text(s,x,208,'Raw dot product = '+r.score.toFixed(3),C.a,30);t.dataset.absoluteScore=String(r.score);
        text(s,x,261,'Same relative gap: one slot',C.muted,24);
      });
    }else if(kind==='overview')overview(s,host.dataset.focus);
  }
  const mapFocus=document.getElementById('position-map-focus');
  mapFocus.addEventListener('change',()=>{
    const host=document.getElementById('position-overview-map');
    host.dataset.focus=mapFocus.value;
    overview(canvas(host,'overview positional encoding diagram',440),mapFocus.value);
  });
  const clock=document.getElementById('position-clock-index');
  function clockDraw(){
    const i=Number(clock.value),a=i*Math.PI/2,p=[Math.cos(a),Math.sin(a)],s=canvas(document.getElementById('position-clock-picture'),'A position feature moves around a circle',275);
    ring(s,165,136,94,a,C.p);
    text(s,300,70,'Angle = '+i+' × 90° = '+i*90+'°',C.p,29);
    text(s,300,123,'Offset pᵢ = '+vec(p,0),C.p,30);
    text(s,300,176,'Record the arrow tip’s x and y coordinates.',C.ink,26);
    text(s,300,233,i===4?'Index 4 repeats index 0: [1, 0].':'Index 0 starts at the hollow marker [1, 0].',C.muted,25);
    document.getElementById('position-clock-label').textContent=String(i);
  }
  clock.addEventListener('input',clockDraw);clock.addEventListener('change',clockDraw);clockDraw();
  const pairs=document.getElementById('position-pair-index');
  function pairDraw(){
    const i=Number(pairs.value),s=canvas(document.getElementById('position-pair-picture'),'Fast and slow position clocks',290),code=[];
    [[Math.PI/2,90,4,C.p],[Math.PI/6,30,12,C.v]].forEach(([rate,degrees,period,color],j)=>{
      const x=140+j*560,a=i*rate,p=[Math.cos(a),Math.sin(a)];code.push(...p);
      text(s,x-100,30,`${degrees}° / slot · period ${period}`,color,26);ring(s,x,139,78,a,color);
      text(s,x+110,127,vec(p,3),color,26);text(s,x+110,166,`index ${i}`,C.muted,24);
    });
    text(s,40,280,'Combined position row: '+vec(code,3),C.p,28);
    const result=document.getElementById('position-pair-result');
    result.textContent=i===0?'Starting code: both clocks point to [1, 0].':i===12?'At 12, both clocks repeat. This four-number toy code has a collision.':'The fast pair repeats, but the slow pair keeps this position distinct from 0.';
    result.dataset.vector=JSON.stringify(code);
  }
  pairs.addEventListener('change',pairDraw);pairDraw();
  AT.positionJourney={absolutePair};
});
