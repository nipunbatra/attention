/* Vision I: the image stays attached to its numbers.
   All arithmetic comes from AT.vision.forward(); this file draws, never trains.
   API: AT.vision.story(host, {stage, forward?, patch?, receiver?, source?,
                              coordinate?, step?, focus?, backward?}).
   patch is 0..3. receiver/source are input rows 0..4, with 0 reserved for CLS.
   Keep important stages in separate frames so ordinary slide PDFs retain them. */
(function () {
  'use strict';
  var AT = window.AT;
  if (!AT || !AT.vision) throw new Error('Load part5.js before part5-diagrams.js.');
  var T = AT.vision, NS = 'http://www.w3.org/2000/svg', serial = 0;
  var C = {e:'#2563EB',q:'#9333EA',k:'#D97706',v:'#0D9488',a:'#E11D48',d:'#16A34A',ep:'#2563EB',ink:'#202530',muted:'#586474',line:'#cbd2dc',paper:'#ffffff',param:'#6b7280'};
  // Reuse the article's semantic palette, including blue + green for updated e.
  var theme=getComputedStyle(document.documentElement);
  function themeColor(name,fallback) {
    var v=theme.getPropertyValue(name).trim(),short=/^#([0-9a-f])([0-9a-f])([0-9a-f])$/i.exec(v);
    if(short)return'#'+short.slice(1).map(function(x){return x+x;}).join('');
    return /^#[0-9a-f]{6}$/i.test(v)?v:fallback;
  }
  ['e','q','k','v','a','d'].forEach(function(role){C[role]=themeColor('--c-'+role,C[role]);});
  C.ep=C.e;C.ink=themeColor('--ink',C.ink);C.muted=themeColor('--ink-2',C.muted);C.line=themeColor('--line',C.line);C.paper=themeColor('--card',C.paper);
  var subs = '₀₁₂₃₄₅₆₇₈₉';
  function sub(n) { return String(n).split('').map(function(x){return subs[+x] || x;}).join(''); }
  function fmt(n, digits) { return Number(n).toFixed(digits == null ? 3 : digits).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1').replace(/^-/, '−'); }
  function fixed(n) { return Number(n).toFixed(3).replace(/^-/, '−'); }
  function vec(a, digits) { return '(' + a.map(function(n){return fmt(n,digits);}).join(', ') + ')'; }
  function clone(a) { return JSON.parse(JSON.stringify(a)); }
  function sv(tag, attrs, text) { var el=document.createElementNS(NS,tag);Object.keys(attrs||{}).forEach(function(k){el.setAttribute(k,String(attrs[k]));});if(text!=null)el.textContent=String(text);return el; }
  function node(parent, tag, attrs, text) { var el=sv(tag,attrs,text);parent.appendChild(el);return el; }
  function frame(stage,height,title,desc) {
    var id='vision-story-'+(++serial), svg=sv('svg',{viewBox:'0 0 1100 '+height,width:'100%',style:'height:auto',preserveAspectRatio:'xMidYMid meet',role:'img','aria-labelledby':id+'-title '+id+'-desc','data-vision-story':stage,class:'vision-story'});
    node(svg,'title',{id:id+'-title'},title);node(svg,'desc',{id:id+'-desc'},desc);
    var defs=node(svg,'defs');
    Object.keys(C).forEach(function(role){var marker=node(defs,'marker',{id:id+'-'+role,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto'});node(marker,'path',{d:'M0 0 L10 5 L0 10Z',fill:C[role]});});
    function text(x,y,s,role,size,anchor,extra) {var readableSize=size?(size<=24?22:size===25?26:size):26;return node(svg,'text',Object.assign({x:x,y:y,fill:C[role||'ink'],'font-family':'var(--font-ui,"Avenir Next",sans-serif)','font-size':readableSize,'text-anchor':anchor||'middle','dominant-baseline':'middle'},extra||{}),s);}
    function rect(x,y,w,h,role,param) {return node(svg,'rect',{x:x,y:y,width:w,height:h,rx:7,fill:role?C[role]+'0c':C.paper,stroke:role==='ep'?C.d:C[role||'line'],'stroke-width':2,'stroke-dasharray':param?'7 5':'none'});}
    function arrow(x1,y1,x2,y2,role,extra) {return node(svg,'path',Object.assign({d:'M'+x1+' '+y1+' L'+x2+' '+y2,fill:'none',stroke:C[role||'muted'],'stroke-width':2.5,'marker-end':'url(#'+id+'-'+(role||'muted')+')'},extra||{}));}
    function path(d,role,extra) {return node(svg,'path',Object.assign({d:d,fill:'none',stroke:C[role||'muted'],'stroke-width':2.5,'marker-end':'url(#'+id+'-'+(role||'muted')+')'},extra||{}));}
    function box(x,y,w,h,label,detail,role,param) {rect(x,y,w,h,role,param);text(x+w/2,y+h/2-(detail?14:0),label,role,26);if(detail)text(x+w/2,y+h/2+21,detail,'muted',24);}
    return {svg:svg,text:text,rect:rect,arrow:arrow,path:path,box:box,id:id};
  }
  function pixelShade(v) { var s=AT.imageShade(v);return'rgb('+s+','+s+','+s+')'; }
  function pixelGrid(c, image, x,y,size,opts) {
    opts=opts||{};var n=image.length,cell=size/n;
    image.forEach(function(row,i){row.forEach(function(v,j){node(c.svg,'rect',{x:x+j*cell,y:y+i*cell,width:cell,height:cell,'data-pixel-value':v,fill:pixelShade(v),stroke:'#aab0b9','stroke-width':.8});if(opts.numbers)c.text(x+(j+.5)*cell,y+(i+.5)*cell,fmt(v,2),'ink',Math.min(28,cell*.55),'middle',{fill:AT.imageShade(v)<118?'#fff':'#000000','font-family':'var(--font-mono,monospace)'});});});
    if(opts.patches){for(var i=0;i<2;i++)for(var j=0;j<2;j++){var selected=i*2+j===opts.selected;node(c.svg,'rect',{x:x+j*size/2+2,y:y+i*size/2+2,width:size/2-4,height:size/2-4,fill:'none',stroke:selected?C.e:'#f8fafc','stroke-width':selected?5:2});}}
    if(opts.label)c.text(x+size/2,y+size+25,opts.label,null,24);
  }
  function patchImage(F,p) {var row=F.patches[p];return[row.slice(0,2),row.slice(2,4)];}
  function crop(c,F,p,x,y,size,numbers,label) {pixelGrid(c,patchImage(F,p),x,y,size,{numbers:numbers,label:label});}
  function source(c,F,j,x,y,size) {if(j===0){c.rect(x,y,size,size,'e',true);if(size>=48)c.text(x+size/2,y+size/2,'CLS','e',24);}else crop(c,F,j-1,x,y,size,false);}
  function rowLabel(j) {return j===0?'CLS':'patch '+j;}
  function vector(c,a,x,y,width,role,label,digits,highlight) {
    if(label)c.text(x+width/2,y-22,label,role,24);
    var gap=8,w=(width-gap*(a.length-1))/a.length;
    a.forEach(function(v,j){c.rect(x+j*(w+gap),y,w,48,role);if(highlight!=null&&highlight!==j)node(c.svg,'rect',{x:x+j*(w+gap),y:y,width:w,height:48,rx:7,fill:'#ffffffa6'});c.text(x+j*(w+gap)+w/2,y+24,fmt(v,digits),role,26,'middle',{'font-family':'var(--font-mono,monospace)'});});
  }
  function matrix(c,M,x,y,width,role,column) {
    var cols=M[0].length,cw=width/cols,rh=42;
    M.forEach(function(row,i){row.forEach(function(v,j){node(c.svg,'rect',{x:x+j*cw,y:y+i*rh,width:cw-3,height:rh-3,rx:4,fill:column===j?C[role]+'20':'#fff',stroke:column===j?C[role]:C.line,'stroke-width':column===j?2:1});c.text(x+j*cw+(cw-3)/2,y+i*rh+(rh-3)/2,fmt(v,3),column===j?role:'muted',24);});});
  }
  function reduceTerms(a,b) {return a.map(function(v,j){return fmt(v)+' × '+fmt(b[j]);}).join(' + ');}
  function diagram(host,opts) {
    opts=opts||{};var stage=opts.stage||'task',F=opts.forward||T.forward();
    if(!F.image||!F.patches)F=Object.assign({},T.forward(),F);
    var D=Object.assign({},T.data,F.params||{},opts.params||{});
    var p=opts.patch==null?3:opts.patch,r=opts.receiver==null?0:opts.receiver,j=opts.source==null?p+1:opts.source,k=opts.coordinate==null?0:opts.coordinate,step=opts.step==null?99:opts.step,c;
    if(p<0||p>3||r<0||r>4||j<0||j>4)throw new Error('patch must be 0..3; receiver and source must be 0..4.');
    if(stage==='image'){
      c=frame(stage,325,'The image is a grid of numbers','The same sixteen grayscale pixels can be viewed as an image or read as numerical inputs.');
      pixelGrid(c,F.image,100,22,245,{numbers:false,patches:!!opts.patches});c.arrow(394,145,646,145,'e');pixelGrid(c,F.image,705,22,245,{numbers:true,patches:!!opts.patches});
      c.text(223,303,'what we see',null,27);c.text(827,303,'the pixel values',null,27);
    }else if(stage==='task'){
      c=frame(stage,320,'Same kind of image, two possible labels','The left image has two occupied 2 by 2 blocks. Removing the upper-left block leaves one occupied block. The classifier should distinguish them.');
      var one=clone(F.image);one[0][0]=one[0][1]=one[1][0]=one[1][1]=0;
      pixelGrid(c,F.image,115,35,205,{numbers:!!opts.numbers});pixelGrid(c,one,715,35,205,{numbers:!!opts.numbers});
      c.text(217,280,'two occupied blocks','e',28);c.text(817,280,'one occupied block','e',28);
      c.text(550,110,'Change these pixels.',null,25);c.arrow(350,160,680,160);c.text(550,210,'The label changes.',null,25);
    }else if(stage==='patches'){
      c=frame(stage,330,'One image becomes four patch inputs','Each numbered crop is a literal 2 by 2 region of the same image. The four patches retain all sixteen pixels.');
      pixelGrid(c,F.image,25,45,230,{numbers:!!opts.numbers,patches:true,selected:p});c.arrow(285,160,350,160);
      for(var a=0;a<4;a++){var x=385+a*175;crop(c,F,a,x,84,135,true,'patch '+(a+1));c.text(x+67.5,55,['top left','top right','bottom left','bottom right'][a],'muted',24);}
      c.text(710,295,'Four patches. No pixel has been discarded.',null,26);
    }else if(stage==='flatten'){
      c=frame(stage,335,'Keep the pixel values; change their arrangement','Read the selected patch from left to right across its first row, then its second row. Draw those four numbers as one raw row r.');
      pixelGrid(c,F.image,22,80,140,{patches:true,selected:p});c.arrow(180,150,230,150);
      crop(c,F,p,258,50,180,true);c.text(348,266,'patch '+(p+1),null,25);c.arrow(466,146,535,146);
      vector(c,F.patches[p],565,110,495,'e','raw row r'+sub(p+1),2);
      F.patches[p].forEach(function(v,a){
        var ox=278+(a%2)*90,oy=70+Math.floor(a/2)*90;
        node(c.svg,'circle',{cx:ox,cy:oy,r:15,fill:'#ffffff',stroke:C.e,'stroke-width':2});c.text(ox,oy,a+1,'e',22);
        c.text(565+a*(503/4)+58,186,'pixel '+(a+1),'e',24);
      });
      c.text(810,256,'2 × 2 pixels → 1 × 4 numbers',null,27);
      if(step>=1){
        var pixel=opts.pixel==null?0:Math.max(0,Math.min(3,opts.pixel)),sx=278+(pixel%2)*90,sy=70+Math.floor(pixel/2)*90,tx=565+pixel*(503/4)+58;
        // One trace at a time: four simultaneous crossing wires obscure the crop.
        if(pixel<2)c.path('M'+sx+' '+(sy-16)+' V26 H'+tx+' V105','e');
        else c.path('M'+sx+' '+(sy+16)+' V238 H'+tx+' V163','e');
        c.text(550,311,'Pixel '+(pixel+1)+' keeps its value and moves to slot '+(pixel+1)+'.','e',25);
      }
    }else if(stage==='project'){
      c=frame(stage,355,'One output coordinate is a dot product','Select one column of the shared patch projection. Multiply each pixel by the corresponding entry and add. Repeat for the other output coordinate.');
      crop(c,F,p,25,61,95,true,'patch '+(p+1));
      vector(c,F.patches[p],155,105,385,'e','r'+sub(p+1)+' · four pixel values',2);
      c.text(585,127,'×',null,30);matrix(c,D.W_patch,635,40,160,'e',k);c.text(715,235,'W_patch · 4 × 2','muted',24);
      c.arrow(823,127,875,127,'e');vector(c,F.embeddings[p],900,105,180,'e','embedding',3,k);
      var col=D.W_patch.map(function(row){return row[k];}),sum=F.patches[p].reduce(function(s,v,a){return s+v*col[a];},0),bias=(D.b_patch||[])[k]||0;
      c.text(550,285,'coordinate '+(k+1)+': '+reduceTerms(F.patches[p],col)+(bias?' + '+fmt(bias):'')+' = '+fmt(sum+bias),'e',26);
      c.text(550,333,'This column says how the four pixels contribute to one number.','muted',24);
    }else if(stage==='patchRows'){
      c=frame(stage,350,'Four crops, four embeddings, one shared projection','Every crop is transformed by the same patch matrix. Equal pixel patches give equal content embeddings.');
      for(var a=0;a<4;a++){var x=45+a*270;crop(c,F,a,x+55,25,130,true,'patch '+(a+1));c.arrow(x+120,189,x+120,220,'e');vector(c,F.embeddings[a],x+15,242,210,'e',null,3);}
      c.text(550,327,'same W_patch at all four locations','muted',24);
    }else if(stage==='arrangements'){
      c=frame(stage,330,'The same patch contents can occupy different places','Swap the first and fourth patch. The inventory of pixel patches is unchanged, but their spatial arrangement changes.');
      var order=opts.order||[3,1,2,0],im=clone(F.image);
      order.forEach(function(q,a){for(var dy=0;dy<2;dy++)for(var dx=0;dx<2;dx++)im[Math.floor(a/2)*2+dy][(a%2)*2+dx]=F.patches[q][dy*2+dx];});
      pixelGrid(c,F.image,110,30,225,{patches:true});pixelGrid(c,im,760,30,225,{patches:true});
      c.text(222,293,'original locations',null,26);c.text(872,293,'same tiles, new locations',null,26);
      c.arrow(370,145,725,145,'e');c.text(550,92,'Move patches 1 and 4.',null,25);c.text(550,220,'Contents alone lose this distinction.','muted',24);
    }else if(stage==='position-check'){
      c=frame(stage,345,'Does swapping patches change the updated CLS row?','The same two arrangements produce equal CLS outputs without position additions and different outputs with position additions. All model weights stay fixed.');
      var order=[3,1,2,0],original=T.forward(),swapped=T.forward({order:order}),off=T.forward({positions:false}),offSwap=T.forward({positions:false,order:order}),im=clone(original.image);
      order.forEach(function(q,a){for(var dy=0;dy<2;dy++)for(var dx=0;dx<2;dx++)im[Math.floor(a/2)*2+dy][(a%2)*2+dx]=original.patches[q][dy*2+dx];});
      c.text(445,21,'original arrangement',null,25);c.text(890,21,'patches 1 and 4 swapped',null,25);
      pixelGrid(c,original.image,367,52,156);pixelGrid(c,im,812,52,156);
      c.text(140,255,'positions OFF',null,24);c.text(445,255,vec(off.updated[0],3),'ep',28);c.text(890,255,vec(offSwap.updated[0],3),'ep',28);
      c.text(140,315,'positions ON',null,24);c.text(445,315,vec(original.updated[0],3),'ep',28);c.text(890,315,vec(swapped.updated[0],3),'ep',28);
    }else if(stage==='generalization'){
      c=frame(stage,350,'Move one block without changing its count','The learned model assigns the original one-block training image to one block with high probability. Moving exactly the same block to the top-left makes it wrongly prefer two blocks.');
      var L=T.learning,params=L.experiment.afterTraining.params,known=L.images[1],moved=known.map(function(row){return row.map(function(){return 0;});});
      for(var y=0;y<2;y++)for(var x=0;x<2;x++)moved[y][x]=known[y+2][x+2];
      var first=L.forward(params,known,1),second=L.forward(params,moved,1);
      c.text(270,22,'training image',null,26);c.text(830,22,'new arrangement',null,26);
      pixelGrid(c,known,185,56,170);pixelGrid(c,moved,745,56,170);c.arrow(400,135,700,135);c.text(550,90,'move the same block',null,23);
      c.text(270,262,'correct label: one block',null,25);c.text(830,262,'correct label: one block',null,25);
      c.text(270,306,'p(one) = '+first.probs[1].toFixed(4),null,29);c.text(830,306,'p(two) = '+second.probs[0].toFixed(4),null,29);
      c.text(270,336,'correct prediction',null,22);c.text(830,336,'wrong prediction',null,22);
    }else if(stage==='positions'){
      c=frame(stage,340,'Same pixels, different positions','The top-right and bottom-left patches contain identical zero pixels. Their content embeddings match. Different full-width position vectors are added, not appended.');
      c.text(210,30,'content embedding','e',24);c.text(565,30,'position vector','muted',24);c.text(925,30,'input row','e',24);
      [2,3].forEach(function(s,a){var y=85+a*150;source(c,F,s,20,y-15,65);c.text(52,y+83,'patch '+s,null,24);vector(c,F.content[s],115,y,235,'e',null,2);c.text(408,y+24,'+',null,30);vector(c,F.positions[s],460,y,210,'param',null,2);c.text(736,y+24,'=',null,30);vector(c,F.E[s],820,y,230,'e','e'+sub(s),2);});
    }else if(stage==='cls'){
      c=frame(stage,340,step<1?'Four patch rows, one image decision':'An extra row will collect information for the answer',step<1?'Four patch rows represent four regions. The classification task needs one answer about the whole image. Which row should supply that answer?':'CLS is an extra learned input vector with no pixel crop. Attention updates that row and the classifier reads its output.');
      for(var a=0;a<4;a++){var x=30+a*184;crop(c,F,a,x+26,25,88,false,'patch '+(a+1));vector(c,F.E[a+1],x,175,140,'e',null,2);if(step>=1)c.arrow(x+70,234,820,270,'a');}
      if(step>=1){c.rect(830,25,230,105,'e',true);c.text(945,52,'CLS: no pixels','e',25);c.text(945,96,'c = '+vec(D.cls),'e',28);c.arrow(945,144,945,224,'e');c.box(820,238,255,74,'updated CLS','for this image','ep');}
      else{c.box(815,83,260,116,'one image answer','which patch row?',null);c.text(945,249,'?',null,42);}
      c.text(370,300,'Four regions → one classification decision.',null,25);
    }else if(stage==='query'){
      c=frame(stage,335,'CLS makes a query from its current representation','The query comes from the receiving CLS row through W_Q. It is not the target label. Its coordinates will be compared with source-key coordinates.');
      source(c,F,r,25,92,86);vector(c,F.E[r],150,104,190,'e','e'+sub(r)+' · input',3);c.arrow(355,128,424,128,'q');
      matrix(c,D.W_Q,442,82,155,'q',k);c.text(519,204,'W_Q','q',25);c.arrow(622,128,704,128,'q');vector(c,F.Q[r],735,104,330,'q','q'+sub(r)+' · matching request',3);
      var qcol=D.W_Q.map(function(row){return row[k];});c.text(550,272,'query coordinate '+(k+1)+': '+reduceTerms(F.E[r],qcol)+' = '+fmt(F.Q[r][k]),'q',27);
      c.text(550,314,'The same first-layer CLS query can read different image-dependent keys.','muted',24);
    }else if(stage==='key'){
      c=frame(stage,335,'A patch makes its matching key','The source patch supplies an input row. W_K transforms that row into the key that other queries compare against.');
      source(c,F,j,25,92,86);vector(c,F.E[j],150,104,190,'e','e'+sub(j)+' · '+rowLabel(j),3);c.arrow(355,128,424,128,'k');
      matrix(c,D.W_K,442,82,155,'k',k);c.text(519,204,'W_K','k',25);c.arrow(622,128,704,128,'k');vector(c,F.K[j],735,104,330,'k','k'+sub(j)+' · matching description',3);
      var kc=D.W_K.map(function(row){return row[k];});c.text(550,272,'key coordinate '+(k+1)+': '+reduceTerms(F.E[j],kc)+' = '+fmt(F.K[j][k]),'k',27);
      c.text(550,314,'This key belongs to '+rowLabel(j)+'. It is compared with the receiving query.','muted',24);
    }else if(stage==='keys'){
      c=frame(stage,350,'Each source row supplies a matching key','Each image crop retains its identity next to its key. Keys are produced by W_K; they will be compared with the selected query.');
      for(var a=0;a<5;a++){var x=15+a*220;source(c,F,a,x+68,22,70);c.text(x+103,120,rowLabel(a)+' · e'+sub(a),'e',24);vector(c,F.E[a],x+5,151,195,'e',null,2);c.arrow(x+102,205,x+102,232,'k');vector(c,F.K[a],x+5,280,195,'k','k'+sub(a),3);}
      for(var a=0;a<5;a++)c.text(178+a*220,226,'W_K','k',24);
    }else if(stage==='score'){
      c=frame(stage,350,'Compare one query with one key','Multiply matching coordinates, add the products, then divide the dot product by the square root of the matching width.');
      source(c,F,r,22,24,72);c.text(160,59,rowLabel(r)+' query','q',24,'start');vector(c,F.Q[r],420,30,300,'q',null,3);
      source(c,F,j,22,121,72);c.text(160,156,rowLabel(j)+' key','k',24,'start');vector(c,F.K[j],420,128,300,'k',null,3);
      if(step>=1){var products=F.Q[r].map(function(v,a){return v*F.K[j][a];});c.text(550,229,reduceTerms(F.Q[r],F.K[j])+' = '+fmt(F.raw[r][j]),null,29);}
      if(step>=2)c.text(550,298,'scaled score = '+fmt(F.raw[r][j])+' ÷ √'+F.Q[r].length+' ≈ '+fixed(F.S[r][j]),'a',29);
      c.text(902,83,'one pair',null,26);c.text(902,127,'one score','a',26);c.text(902,169,'not a probability','muted',24);
    }else if(stage==='weights'){
      c=frame(stage,360,'One mixing weight per source position','A query is compared against all five sources. Softmax exponentiates each scaled score, then divides by one shared total. The resulting weights sum to one.');
      var ex=F.S[r].map(Math.exp),z=ex.reduce(function(s,v){return s+v;},0);
      c.text(180,25,'source',null,24);c.text(420,25,'scaled score','a',24);if(step>=1)c.text(620,25,'exp(score)',null,24);if(step>=2){c.text(824,25,'weight α','a',24);c.text(1000,25,'amount','a',24);}
      for(var a=0;a<5;a++){var y=65+a*49;source(c,F,a,34,y-17,34);c.text(100,y,rowLabel(a),null,24,'start');c.text(420,y,fixed(F.S[r][a]),'a',25);if(step>=1)c.text(620,y,fixed(ex[a]),null,25);if(step>=2){c.text(824,y,fixed(F.A[r][a]),'a',25);c.rect(925,y-12,147,24);node(c.svg,'rect',{x:925,y:y-12,width:147*F.A[r][a],height:24,rx:5,fill:C.a});}}
      if(step===1)c.text(550,329,'Total Z = '+fixed(z)+'. The next step divides every row by this same total.',null,24);
      if(step>=2)c.text(550,329,'α'+sub(r)+sub(j)+' = '+fixed(ex[j])+' ÷ '+fixed(z)+' ≈ '+fixed(F.A[r][j])+'     |     sum of weights = 1','a',25);
    }else if(stage==='values'){
      c=frame(stage,330,'A source value carries the information to be mixed','The selected source creates a value using W_V, separately from its matching key. The value is not an attention weight.');
      source(c,F,j,25,77,105);c.text(77,218,rowLabel(j),null,24);vector(c,F.E[j],180,96,235,'e','input e'+sub(j),3);
      c.arrow(435,122,493,122,'v');matrix(c,D.W_V,515,78,160,'v',k);c.text(595,203,'W_V','v',25);c.arrow(700,122,769,122,'v');vector(c,F.V[j],795,96,270,'v','value v'+sub(j),3);
      c.text(550,265,'key k'+sub(j)+' = '+vec(F.K[j])+' helps choose the weight.','k',26);
      c.text(550,312,'value v'+sub(j)+' = '+vec(F.V[j])+' supplies the numbers that weight multiplies.','v',26);
    }else if(stage==='mix'){
      c=frame(stage,360,'Weight values from the corresponding sources','Every alpha multiplies the value from the same source. Sum each coordinate of the five contributions to form a message for the receiving row.');
      for(var a=0;a<5;a++){var x=15+a*220;source(c,F,a,x+75,12,56);c.text(x+103,93,rowLabel(a),null,24);c.text(x+103,133,'v = '+vec(F.V[a]),'v',25);c.text(x+103,175,'× '+fixed(F.A[r][a]),'a',26);
        if(step>=1){c.text(x+103,218,vec(F.V[a].map(function(v){return v*F.A[r][a];}),3),'v',24);c.arrow(x+103,242,550,290,'v');}}
      if(step>=2)c.box(355,296,390,56,'message m'+sub(r)+' = '+vec(F.message[r],3),null,'v');
      else c.text(550,323,step===0?'One weight scales both coordinates of its source value.':'Rounded for display; computations keep the unrounded weights.',null,25);
    }else if(stage==='residual'){
      c=frame(stage,350,'Add the message-derived update to the original row','The original representation bypasses attention. The weighted message is projected by W_O to representation width and added to that original row.');
      vector(c,F.E[r],30,42,205,'e','original e'+sub(r),3);c.path('M235 66 H995 V229','e');c.text(565,43,'Keep this original row.','e',25);
      vector(c,F.message[r],30,175,205,'v','message m'+sub(r),3);c.arrow(255,198,331,198,'v');c.box(350,158,180,80,'W_O',F.message[r].length+' → '+F.E[r].length+' numbers','param',true);c.arrow(550,198,620,198,'d');vector(c,F.delta[r],641,175,205,'d','update Δe'+sub(r),3);c.arrow(865,198,920,252,'d');c.text(970,260,'+',null,36);c.arrow(939,283,890,308,'ep');
      c.text(370,312,'e'+sub(r)+'′ = '+vec(F.E[r],3)+' + '+vec(F.delta[r],3),'ep',28);c.text(823,323,'= '+vec(F.updated[r],3),'ep',28);
    }else if(stage==='head'){
      c=frame(stage,345,'Convert the updated CLS row into class scores','Each classifier column produces one logit. A class softmax produces probabilities over the two candidate labels, not over image patches.');
      var names=D.classes&&D.classes[0]!=='class A'?D.classes:['two blocks','one block'];
      vector(c,F.updated[0],35,120,250,'ep','updated CLS e₀′',3);c.arrow(310,144,388,144,'ep');matrix(c,D.W_class,420,91,210,'param',k);c.text(525,219,'W_class','param',25);c.arrow(655,144,730,144);
      for(var a=0;a<2;a++){var y=54+a*128;c.text(906,y,names[a],null,26);c.text(906,y+40,'logit '+fixed(F.logits[a]),null,27);if(step>=1)c.text(906,y+78,'p = '+fixed(F.probs[a]),null,28);}
      var hc=D.W_class.map(function(row){return row[k];}),b=(D.b_class||[])[k]||0;c.text(550,306,reduceTerms(F.updated[0],hc)+(b?' + '+fmt(b):'')+' ≈ '+fixed(F.logits[k]),null,26);
    }else if(stage==='pipeline'){
      c=frame(stage,360,'The same image, from pixels to loss','Pixels are turned into patch rows, positions and CLS are added, attention creates an update, and the updated CLS produces a prediction. Backward arrows show gradients reaching learned parameters.');
      var ids=['pixels','patch','input','attention','residual','head','loss'];
      var titles=[['pixels'],['patch','projection'],['add CLS','& position'],['self-','attention'],['residual','addition'],['class','head'],['loss']];
      var details=['4 × 4','4 × 2','+ CLS: 5 × 2','Q, K, V → m','E′ = E + ΔE','read CLS','−log p(label)'];
      var xx=[15,172,329,486,643,800,957],ww=128;
      for(var a=0;a<7;a++){if(step<a)break;var role=ids[a]===opts.focus?'q':a<3?'e':a<5?'v':a===5?'ep':'ink';
        if(a===0)pixelGrid(c,F.image,xx[a]+21,104,86,{numbers:false});else{c.rect(xx[a],100,ww,98,role);titles[a].forEach(function(s,b){c.text(xx[a]+ww/2,150+(b-(titles[a].length-1)/2)*34,s,role,25);});}
        c.text(xx[a]+ww/2,226,details[a],a===0?'muted':role,24);if(a>0)c.arrow(xx[a-1]+ww+3,147,xx[a]-4,147,'muted');}
      if(step>=3){c.rect(396,20,310,51,'param',true);c.text(551,46,'W_Q · W_K · W_V','param',24);c.arrow(550,74,550,94,'param');}
      if(opts.backward){c.path('M1020 250 V290 H235 V210','k',{'stroke-dasharray':'8 5'});c.path('M551 290 V251 H712 V45 H683','k',{'stroke-dasharray':'8 5'});c.path('M865 289 V207','k',{'stroke-dasharray':'8 5'});c.text(550,332,'loss.backward() follows the graph to the learned parameters.','k',25);}
      else c.text(550,320,'The image supplies the input. Its label is used only at the loss.',null,25);
    }else if(stage==='backward'){
      c=frame(stage,390,'Forward computations and the parameters that learn','The forward graph starts with image pixels and ends with a supervised loss. Dashed orange arrows trace gradients in reverse, including into patch weights, position and CLS parameters, attention projections, the output projection, and the class head.');
      var bx=[15,295,575,855],bw=230;
      pixelGrid(c,F.image,bx[0]+74,65,82,{numbers:false});
      c.box(bx[1],65,bw,82,'patch vectors','pixel projection','e');
      c.box(bx[2],65,bw,82,'input E','CLS + patches + P','e');
      c.box(bx[3],65,bw,82,'Q, K, V','project E','q');
      c.box(bx[3],268,bw,76,'message m','m = A V','v');
      c.box(bx[2],268,bw,76,'updated rows','E′ = E + m W_O','ep');
      c.box(bx[1],268,bw,76,'logits → softmax','classification head',null);
      c.box(bx[0],268,bw,76,'loss L','−log p(label)','ink');
      for(var a=0;a<3;a++){c.arrow(bx[a]+bw+4,108,bx[a+1]-7,108);c.arrow(bx[a+1]-7,306,bx[a]+bw+4,306);}
      c.arrow(970,151,970,165,'q');c.box(855,174,230,70,'scores → softmax','Q Kᵀ / √d_k → A','a');c.arrow(970,249,970,263,'a');
      // Values bypass score/softmax; the original E bypasses the attention update.
      c.path('M852 133 H833 V285 H850','v');c.text(816,227,'V','v',24);
      c.path('M572 133 H554 V285 H570','e');c.text(541,243,'E','e',24);
      var params=[{x:295,y:7,w:230,label:'W_patch, b_patch',down:true},{x:575,y:7,w:230,label:'CLS c, positions P',down:true},{x:855,y:7,w:230,label:'W_Q, W_K, W_V',down:true},{x:592,y:209,w:196,label:'W_O',down:true},{x:295,y:209,w:230,label:'W_class, b_class',down:true}];
      params.forEach(function(z){c.rect(z.x,z.y,z.w,39,'param',true);c.text(z.x+z.w/2,z.y+20,z.label,'param',24);var targetY=z.y<100?59:262;c.arrow(z.x+z.w/2-14,z.y+43,z.x+z.w/2-14,targetY,'param');c.arrow(z.x+z.w/2+14,targetY,z.x+z.w/2+14,z.y+43,'k',{'stroke-dasharray':'6 4'});});
      c.path('M868 158 H237','k',{'stroke-dasharray':'7 5'});c.path('M132 350 H961','k',{'stroke-dasharray':'7 5'});
      c.text(174,185,'solid: forward',null,24);c.text(482,185,'dashed: gradients','k',24);
      c.text(550,367,'The optimizer changes the dashed parameter boxes, not the cached Q, K, V or A.',null,23);
    }else if(stage==='rgb'){
      c=frame(stage,320,'Color changes the raw patch width, not the learned representation width','An RGB pixel contains three channel numbers. A P by P patch contains P squared times three numbers before projection.');
      ['R','G','B'].forEach(function(label,a){var x=75+a*165;node(c.svg,'rect',{x:x,y:65,width:132,height:132,rx:5,fill:['#c74545','#479054','#497dc2'][a]});c.text(x+66,228,label,null,28);});
      c.arrow(598,130,688,130,'e');c.box(716,72,335,113,'P × P × 3 pixel numbers','project to d_model numbers','e');
      c.text(550,290,'For 16 × 16 RGB patches: 16 × 16 × 3 = 768 raw numbers.',null,27);
    }else throw new Error('Unknown Vision I story stage: '+stage);
    c.svg.setAttribute('data-stage',stage);c.svg.setAttribute('data-step',step);c.svg.setAttribute('data-patch',p);c.svg.setAttribute('data-receiver',r);c.svg.setAttribute('data-source',j);
    if(host){AT.clear(host);host.appendChild(c.svg);}return c.svg;
  }
  T.story=diagram;
  T.storyStages=['image','task','patches','flatten','project','patchRows','arrangements','positions','cls','query','key','keys','score','weights','values','mix','residual','head','pipeline','backward','rgb'];
})();
