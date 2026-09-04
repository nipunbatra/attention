/* Vision series II: exact illustrative losses and offline architecture drawings.
   No pretrained model runs here. Predictions are explicit user/data inputs. */
(function(){
  'use strict';
  const AT=window.AT, D=window.__TOY__.visionSSL;
  const clone=x=>JSON.parse(JSON.stringify(x));
  function patchify(image=D.image){
    if(!Array.isArray(image)||image.length!==4||!image.every(r=>Array.isArray(r)&&r.length===4&&r.every(Number.isFinite))) throw new Error('Expected a finite 4 by 4 image.');
    const out=[];
    for(let y=0;y<4;y+=2)for(let x=0;x<4;x+=2)out.push([image[y][x],image[y][x+1],image[y+1][x],image[y+1][x+1]]);
    return out;
  }
  function reconstruction(mask=D.maskPresets[0].indices,guess=D.defaultGuess){
    if(!Array.isArray(mask)||!mask.length||mask.length>=4||new Set(mask).size!==mask.length||!mask.every(i=>Number.isInteger(i)&&i>=0&&i<4)) throw new Error('Choose one to three distinct masked patches.');
    if(!Number.isFinite(guess))throw new Error('The guess must be finite.');
    const patches=patchify(), rows=patches.map((target,i)=>{
      const hidden=mask.includes(i), prediction=target.map(p=>hidden?guess:p), errors=target.map((p,c)=>(prediction[c]-p)**2);
      return {index:i,hidden,target,prediction,errors,sse:errors.reduce((a,b)=>a+b,0)};
    });
    const sse=rows.filter(r=>r.hidden).reduce((s,r)=>s+r.sse,0),count=mask.length*4;
    return {rows,mask:mask.slice(),guess,sse,count,loss:sse/count,visible:4-mask.length};
  }
  function softmax(row){const m=Math.max(...row),exp=row.map(x=>Math.exp(x-m)),sum=exp.reduce((a,b)=>a+b,0);return exp.map(x=>x/sum);}
  function dino(first=D.dino.studentLogits[0]){
    if(!Number.isFinite(first))throw new Error('Student logit must be finite.');
    const cfg=D.dino,student=cfg.studentLogits.slice();student[0]=first;
    const centered=cfg.teacherLogits.map((x,i)=>x-cfg.center[i]);
    const teacherScaled=centered.map(x=>x/cfg.teacherTemperature),studentScaled=student.map(x=>x/cfg.studentTemperature);
    const target=softmax(teacherScaled),prediction=softmax(studentScaled),terms=target.map((p,i)=>-p*Math.log(prediction[i]));
    return {teacherLogits:cfg.teacherLogits.slice(),center:cfg.center.slice(),centered,teacherScaled,student,studentScaled,target,prediction,terms,loss:terms.reduce((a,b)=>a+b,0)};
  }
  function ema(old=D.ema.oldTeacher,student=D.ema.newStudent,beta=D.ema.momentum){
    if(![old,student,beta].every(Number.isFinite)||beta<0||beta>1)throw new Error('EMA needs finite operands and momentum in [0,1].');
    return beta*old+(1-beta)*student;
  }
  function transform(kind,image=D.image){
    if(kind==='flip')return image.map(r=>r.slice().reverse());
    if(kind==='dim')return image.map(r=>r.map(p=>p*.75));
    if(kind==='identity')return clone(image);
    throw new Error('Unknown view transformation.');
  }
  const NS='http://www.w3.org/2000/svg';let serial=0;
  function svgElement(tag,attrs={},text){const e=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,String(v)));if(text!=null)e.textContent=text;return e;}
  function scene(height,title){
    const id='vssl-'+(++serial),svg=svgElement('svg',{viewBox:'0 0 1100 '+height,role:'img','aria-labelledby':id+'-title','data-vision-ssl-diagram':title});
    svg.style.cssText='display:block;width:100%;height:auto;max-width:1100px;margin:0 auto';
    svg.appendChild(svgElement('title',{id:id+'-title'},title));
    const defs=svgElement('defs'),marker=svgElement('marker',{id:id+'-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});
    marker.appendChild(svgElement('path',{d:'M0 0L10 5L0 10Z',fill:'var(--ink-3)'}));defs.appendChild(marker);svg.appendChild(defs);
    function text(x,y,t,size=23,anchor='middle',color='var(--ink)'){svg.appendChild(svgElement('text',{x,y,'text-anchor':anchor,'dominant-baseline':'middle','font-size':size,'font-family':'var(--font-ui)',fill:color},t));}
    function box(x,y,w,h,title,sub='',rep=false){svg.appendChild(svgElement('rect',{x,y,width:w,height:h,rx:10,fill:rep?'var(--t-e)':'var(--card)',stroke:rep?'var(--c-e)':'var(--line)','stroke-width':2}));text(x+w/2,y+h/2-(sub?13:0),title,24,'middle',rep?'var(--c-e)':'var(--ink)');if(sub)text(x+w/2,y+h/2+17,sub,20);}
    function arrow(x1,y1,x2,y2,label='',dashed=false){svg.appendChild(svgElement('path',{d:`M${x1} ${y1} L${x2} ${y2}`,stroke:'var(--ink-3)','stroke-width':2,fill:'none','marker-end':`url(#${id}-arrow)`,'stroke-dasharray':dashed?'7 5':'none'}));if(label)text((x1+x2)/2,(y1+y2)/2-16,label,20);}
    function image(x,y,size=156,mask=[],pixels=D.image,label=''){
      const cell=size/4;
      pixels.forEach((row,ry)=>row.forEach((p,cx)=>{const k=Math.floor(ry/2)*2+Math.floor(cx/2),hidden=mask.includes(k),shade=245-Math.max(0,Math.min(2,p))*90;
        svg.appendChild(svgElement('rect',{x:x+cx*cell,y:y+ry*cell,width:cell,height:cell,fill:hidden?'var(--paper)':`rgb(${shade},${shade},${shade})`,stroke:'var(--line)'}));
        if(hidden||cell>=35)text(x+(cx+.5)*cell,y+(ry+.5)*cell,hidden?'×':Number(p.toFixed(2)),20,'middle',!hidden&&p>1.3?'white':'var(--ink)');
      }));
      for(let i=0;i<4;i++){const px=x+(i%2)*size/2,py=y+Math.floor(i/2)*size/2;svg.appendChild(svgElement('rect',{x:px,y:py,width:size/2,height:size/2,fill:'none',stroke:'var(--ink-3)','stroke-width':2}));}
      if(label)text(x+size/2,y+size+21,label,21);
    }
    return {svg,text,box,arrow,image};
  }
  function diagram(host,kind,options={}){
    const h={image:240,'mae-encoder':230,'mae-decoder':260,'mae-transfer':200,views:240,contrastive:240,dino:300,'dino-update':220,jepa:330,probe:250}[kind];
    if(!h)throw new Error('Unknown SSL diagram '+kind);
    const s=scene(h,kind+' teaching illustration'),{box,text,arrow,image}=s;
    if(kind==='image'){
      image(80,26,164,[],D.image,'4 × 4 grayscale image');arrow(280,107,410,107,'patchify');
      patchify().forEach((p,i)=>box(445+(i%2)*285,25+Math.floor(i/2)*100,250,73,'patch '+(i+1),p.join('  ')));
    }else if(kind==='mae-encoder'){
      image(25,20,150,[1,2,3],D.image,'only p1 remains');arrow(210,97,300,97);
      box(315,60,190,80,'input row e₁','projection + position',true);arrow(510,97,580,97);
      box(595,60,205,80,'ViT encoder','visible patches only');arrow(805,97,870,97);box(885,60,190,80,'encoded e₁','one output row',true);
      text(655,190,'No mask tokens enter this encoder.',23);
    }else if(kind==='mae-decoder'){
      box(20,70,170,82,'encoded e₁','visible source',true);arrow(195,110,255,110);
      box(270,35,310,140,'restore patch positions','e₁   [M]   [M]   [M]',true);text(425,200,'add decoder position vectors',22);
      arrow(585,110,645,110);box(660,70,190,82,'small decoder','all four positions');arrow(855,110,915,110);box(930,70,150,82,'pixel guesses','4 × 4 numbers');
    }else if(kind==='mae-transfer'){
      box(35,48,225,92,'unmasked image','all patch rows');arrow(265,94,345,94);box(360,48,270,92,'pretrained encoder','keep learned weights',true);arrow(635,94,715,94);box(730,48,310,92,'downstream task head','train with task labels');
    }else if(kind==='views'){
      image(70,28,152,[],D.image,'original');arrow(252,100,410,100,'two transforms');
      image(475,28,152,[],transform('flip'),'horizontal flip');image(810,28,152,[],transform('dim'),'brightness × 0.75');
    }else if(kind==='contrastive'){
      box(20,20,190,70,'view A','same image');box(20,130,190,70,'view B','same image');
      arrow(215,55,275,55);arrow(215,165,275,165);box(290,20,245,70,'encoder + head','shared weights',true);box(290,130,245,70,'encoder + head','shared weights',true);
      arrow(540,55,650,80);arrow(540,165,650,115);box(665,55,205,80,'positive pair','encourage agreement');
      box(650,171,420,58,'other images supply alternatives');arrow(1010,165,875,112,'contrast against');
    }else if(kind==='dino'){
      image(12,12,100,[],transform('flip'),'view A');image(12,158,100,[],transform('dim'),'view B');
      arrow(127,63,202,63);arrow(127,208,202,208);box(215,22,220,84,'student θ','encoder + head');box(215,166,220,84,'teacher φ','encoder + head');
      arrow(440,63,525,63);arrow(440,208,525,208);box(540,22,220,84,'student pₛ','temperature softmax');box(540,166,220,84,'target pₜ','center + sharpen');
      arrow(765,63,940,116);arrow(765,208,940,160,'stop gradient');box(945,105,145,70,'CE loss');
      arrow(325,109,325,160,'EMA',true);
    }else if(kind==='dino-update'){
      box(30,50,245,90,'student θ','gradient update');arrow(280,95,470,95,'EMA, not gradient',true);box(485,50,245,90,'teacher φ','moving parameters');
      arrow(735,95,805,95);box(820,50,250,90,'teacher targets','stop gradient in loss');
    }else if(kind==='jepa'){
      image(10,12,106,[3],D.image,'p4 hidden');image(10,184,106,[],D.image,'full image');
      arrow(125,65,195,65);arrow(125,236,195,236);box(205,26,205,80,'context encoder θ','visible patches',true);box(205,196,205,80,'target encoder φ','all patches',true);
      arrow(415,65,500,65);arrow(415,236,500,236);box(515,26,245,80,'predictor ψ','target position: p4');box(515,196,245,80,'select target row e₄','after encoding',true);
      arrow(765,65,930,116);text(845,45,'predicted ê₄',20);arrow(765,236,930,176);text(855,247,'stop gradient',20);box(940,110,145,80,'feature loss');
      arrow(307,111,307,190,'EMA',true);text(637,145,'Predict a representation, not four pixels.',22);
    }else if(kind==='probe'){
      box(20,32,235,82,'pretrained encoder','frozen',true);arrow(260,73,380,73);box(395,32,260,82,'linear classifier','learn head weights');arrow(660,73,790,73);box(805,32,270,82,'held-out task labels','evaluate predictions');
      box(20,150,235,82,'pretrained encoder','trainable',true);arrow(260,192,380,192);box(395,150,260,82,'task head','train together');arrow(660,192,790,192);box(805,150,270,82,'fine-tuning','different evaluation');
    }
    AT.clear(host);host.appendChild(s.svg);return s.svg;
  }
  AT.visionSSL={data:clone(D),patchify,reconstruction,dino,ema,transform,diagram};
  if(AT.axes)AT.axes.named=false;
  if(Array.isArray(AT.objects)){const e=AT.objects.find(o=>o.cls==='e');if(e){e.name='visual representation';e.def=e.tip='e is a current visual representation row; E stacks the rows. Raw pixels and loss probabilities use different symbols.';}}
  if(Array.isArray(AT.notation)){
    const rows=[
      ['r_j','raw pixel row for patch j','1\\times4'],
      ['\\ve{e_j}=r_jW_{\\mathrm{patch}}+b_{\\mathrm{patch}}+p_j','projected patch row plus bias and a same-width position vector','1\\times d'],
      ['\\ve{E}','stack of current patch representation rows','N\\times d'],
      ['\\mathcal M','indices of hidden patches',''],
      ['\\hat r_j','decoder prediction of the hidden pixels in patch j','1\\times4'],
      ['p_s,\\;p_t','student and teacher output distributions, not class labels','1\\times C'],
      ['\\theta,\\;\\phi','student/context parameters and EMA teacher/target parameters',''],
      ['\\operatorname{sg}[\\cdot]','stop gradient through this target branch',''],
      ['\\hat e_j,\\;e_j^{\\mathrm{target}}','predicted and target representation at patch j','1\\times d']
    ];
    rows.forEach(([sym,mean,shape])=>AT.notation.push({g:'token',sym,mean,shape,dims:()=>'',parts:['vision2']}));
  }
})();
