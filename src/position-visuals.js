/* Original, inspectable position diagrams. Content vectors stay fixed in the
   RoPE shift experiment so only the positional operation is being compared. */
document.addEventListener('DOMContentLoaded',()=>{
  const NS='http://www.w3.org/2000/svg';
  const C={e:'var(--c-e)',q:'var(--c-q)',k:'var(--c-k)',v:'var(--c-v)',d:'var(--c-d)',line:'var(--line)',ink:'var(--ink)'};
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
  document.querySelectorAll('[data-position-visual]').forEach(host=>{
    const kind=host.dataset.positionVisual,s=canvas(host,kind+' positional encoding illustration');
    if(kind==='shift'){
      [0,1].forEach(panel=>{
        const ox=70+panel*560,oy=245,scale=74,token=[1,1],p=panel?[0,1]:[1,0],result=token.map((v,j)=>v+p[j]);
        text(s,ox,30,panel?'Maya at slot B':'Maya at slot A',C.e,27);
        line(s,ox,oy,ox+205,oy);line(s,ox,oy,ox,oy-185);
        const tx=ox+scale,ty=oy-scale,rx=ox+scale*result[0],ry=oy-scale*result[1];
        dot(s,tx,ty,C.e);arrow(s,tx,ty,rx,ry,C.d);dot(s,rx,ry,C.d);
        text(s,ox+250,105,'e = [1, 1]',C.e);text(s,ox+250,155,'p = '+(panel?'[0, 1]':'[1, 0]'),C.d);text(s,ox+250,205,'e + p = '+(panel?'[1, 2]':'[2, 1]'),C.d);
        text(s,ox+10,270,'coordinate 1',C.ink,22);
      });
    }else if(kind==='clock'){
      const x=170,y=145,r=85;axes(s,x,y,r);
      for(let i=0;i<4;i++){const v=rotate([1,0],i*Math.PI/2);vector(s,x,y,r,v,C.d);text(s,x+v[0]*113-18,y-v[1]*107+8,String(i),C.d);}
      text(s,400,50,'index i      [cos(iω), sin(iω)]',C.ink,26);
      ['0           [ 1,  0]','1           [ 0,  1]','2           [−1,  0]','3           [ 0, −1]','4           [ 1,  0]  ← repeats index 0'].forEach((v,i)=>text(s,400,93+i*36,v,C.d,24));
    }else if(kind==='rates'){
      text(s,25,35,'Position',C.ink,27);text(s,220,35,'Fast pair: 90° / slot',C.d,27);text(s,650,35,'Slow pair: 30° / slot',C.v,27);
      [0,4].forEach((i,row)=>{const y=105+row*90;text(s,30,y,'index '+i,C.ink,28);text(s,220,y,fmt(rotate([1,0],i*Math.PI/2)),C.d,31);text(s,650,y,fmt(rotate([1,0],i*Math.PI/6)),C.v,31);});
      text(s,220,267,'Same fast coordinates',C.d,25);text(s,650,267,'Different slow coordinates',C.v,25);
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
  window.AT.positionVisuals={rotate,shifted};
});
