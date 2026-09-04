/* Vision Part I. Exact offline worksheet, with no trained-model claims.
   All numerical displays call AT.vision.forward; the base data is toy5.json.
   patchify accepts P=1,2,4. The supplied attention model is for P=2 only. */
(function () {
  'use strict';
  var AT = window.AT, data = (window.__TOY__ || {}).vision;
  if (!AT || !data) throw new Error('Vision Part I data is missing.');
  function copy(x) { return JSON.parse(JSON.stringify(x)); }
  function matmul(a,b) { return a.map(function(r){return b[0].map(function(_,j){return r.reduce(function(s,x,k){return s+x*b[k][j];},0);});}); }
  function transpose(a) { return a[0].map(function(_,j){return a.map(function(r){return r[j];});}); }
  function add(a,b) { return a.map(function(r,i){return r.map(function(x,j){return x+b[i][j];});}); }
  function softmax(r) { var m=Math.max.apply(null,r), ex=r.map(function(x){return Math.exp(x-m);}), z=ex.reduce(function(a,b){return a+b;},0);return ex.map(function(x){return x/z;}); }
  function validateImage(image) {
    if (!Array.isArray(image) || image.length!==4 || !image.every(function(r){return Array.isArray(r)&&r.length===4&&r.every(Number.isFinite);})) throw new Error('Use a finite 4 × 4 grayscale image.');
  }
  function patchify(image, size) {
    validateImage(image);
    if ([1,2,4].indexOf(size)<0) throw new Error('Patch size must be 1, 2, or 4.');
    var patches=[], origins=[];
    for(var y=0;y<4;y+=size) for(var x=0;x<4;x+=size){
      var row=[];
      for(var dy=0;dy<size;dy++) for(var dx=0;dx<size;dx++) row.push(image[y+dy][x+dx]);
      patches.push(row);origins.push([y,x]);
    }
    return {patches:patches,origins:origins,size:size,count:patches.length,width:size*size};
  }
  function attention(E, overrides) {
    var weights=Object.assign({},data,overrides||{});
    var Q=matmul(E,weights.W_Q), K=matmul(E,weights.W_K), V=matmul(E,weights.W_V);
    var raw=matmul(Q,transpose(K)), S=raw.map(function(r){return r.map(function(x){return x/Math.sqrt(2);});}), A=S.map(softmax);
    var message=matmul(A,V), delta=matmul(message,weights.W_O), updated=add(E,delta);
    return {E:E,Q:Q,K:K,V:V,raw:raw,S:S,A:A,message:message,delta:delta,updated:updated};
  }
  function forward(options) {
    options=options||{};
    var image=copy(options.image||data.image), p=patchify(image,2);
    var patches=p.patches;
    if(options.order){
      if(!Array.isArray(options.order)||options.order.length!==4||new Set(options.order).size!==4||!options.order.every(function(x){return Number.isInteger(x)&&x>=0&&x<4;})) throw new Error('Patch order must be a permutation of 0,1,2,3.');
      patches=options.order.map(function(i){return p.patches[i].slice();});
    }
    var embeddings=matmul(patches,data.W_patch).map(function(r){return r.map(function(x,j){return x+data.b_patch[j];});});
    var content=[data.cls.slice()].concat(embeddings), positions=options.positions===false?content.map(function(){return[0,0];}):copy(data.positions);
    var f=attention(add(content,positions));
    var logits=matmul([f.updated[0]],data.W_class)[0].map(function(x,j){return x+data.b_class[j];});
    f.image=image;f.patches=patches;f.embeddings=embeddings;f.content=content;f.positions=positions;
    f.logits=logits;f.probs=softmax(logits);f.loss=-Math.log(f.probs[data.target]);
    return f;
  }
  function editImage(value) {var image=copy(data.image);image[2][2]=value;return image;}
  function blockImage(value) {var image=copy(data.image);for(var y=2;y<4;y++)for(var x=2;x<4;x++)image[y][x]=value;return image;}
  var labels=['CLS','patch 1 · top left','patch 2 · top right','patch 3 · bottom left','patch 4 · bottom right'];
  var NS='http://www.w3.org/2000/svg', serial=0;
  var color={e:'var(--c-e)',q:'var(--c-q)',k:'var(--c-k)',v:'var(--c-v)',a:'var(--c-a)',d:'var(--c-d)',ep:'var(--c-d)',ink:'var(--ink)',muted:'var(--ink-3)'};
  function sv(tag,attrs,text){var e=document.createElementNS(NS,tag);Object.keys(attrs||{}).forEach(function(k){e.setAttribute(k,String(attrs[k]));});if(text!=null)e.textContent=text;return e;}
  function canvas(height,title,description){
    var id='vision1-diagram-'+(++serial), root=sv('svg',{viewBox:'0 0 1100 '+height,role:'img','aria-labelledby':id+'-title '+id+'-desc','data-vision1-diagram':title});
    root.appendChild(sv('title',{id:id+'-title'},title));root.appendChild(sv('desc',{id:id+'-desc'},description));
    var defs=sv('defs');Object.keys(color).forEach(function(k){var m=sv('marker',{id:id+'-'+k,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});m.appendChild(sv('path',{d:'M0 0 L10 5 L0 10 Z',fill:color[k]}));defs.appendChild(m);});root.appendChild(defs);
    function text(x,y,value,role,size,anchor){root.appendChild(sv('text',{x:x,y:y,fill:color[role||'ink'],'font-size':size||24,'font-family':'var(--font-ui)','text-anchor':anchor||'middle','dominant-baseline':'middle'},value));}
    function box(x,y,w,h,label,sub,role){root.appendChild(sv('rect',{x:x-w/2,y:y-h/2,width:w,height:h,rx:7,fill:role?'var(--t-'+(role==='ep'?'d':role)+')':'var(--card)',stroke:color[role||'muted'],'stroke-width':2}));text(x,sub?y-12:y,label,role,23);if(sub)text(x,y+20,sub,'muted',20);}
    function arrow(x1,y1,x2,y2,role){root.appendChild(sv('path',{d:'M'+x1+' '+y1+' L'+x2+' '+y2,fill:'none',stroke:color[role||'muted'],'stroke-width':2.5,'marker-end':'url(#'+id+'-'+(role||'muted')+')'}));}
    return{svg:root,text:text,box:box,arrow:arrow};
  }
  function grid(c,image,x,y,size,patchSize,selected){
    var cell=size/4;
    image.forEach(function(row,iy){row.forEach(function(value,ix){
      var shade=Math.round(62+Math.max(0,Math.min(3,value))*62);
      c.svg.appendChild(sv('rect',{x:x+ix*cell,y:y+iy*cell,width:cell,height:cell,fill:'rgb('+shade+','+shade+','+shade+')',stroke:'var(--ink-3)','stroke-width':.8}));
      c.svg.appendChild(sv('text',{x:x+(ix+.5)*cell,y:y+(iy+.5)*cell,fill:value<1?'#FFFFFF':'var(--ink)','font-size':Math.min(25,cell*.43),'font-family':'var(--font-mono)','text-anchor':'middle','dominant-baseline':'middle'},value));
    });});
    if(patchSize){for(var gy=0;gy<4;gy+=patchSize)for(var gx=0;gx<4;gx+=patchSize){var i=(gy/patchSize)*(4/patchSize)+gx/patchSize;c.svg.appendChild(sv('rect',{x:x+gx*cell+2,y:y+gy*cell+2,width:cell*patchSize-4,height:cell*patchSize-4,fill:'none',stroke:i===selected?color.q:color.k,'stroke-width':i===selected?5:2.5}));}}
  }
  function vector(row,dec){return '('+row.map(function(x){return dec==null?String(x):x.toFixed(dec);}).join(', ')+')';}
  function diagram(host,kind,options){
    options=options||{};var f=forward(options), c, i;
    if(kind==='image'){
      c=canvas(270,'One 4 by 4 grayscale image','Pixel values are 1 in the top-left block, 2 in the bottom-right block, and 0 elsewhere.');
      grid(c,f.image,70,12,224,0);c.arrow(345,120,475,120,'e');c.box(650,120,320,96,'image classifier','two blocks or one block','ep');
      c.text(195,247,'4 rows × 4 columns',null,22);c.text(650,205,'We will compute its two scores.',null,24);
    }else if(kind==='patches'){
      var p=patchify(f.image,options.size||2), selected=options.selected||0;c=canvas(270,'Divide the image into patches','Nonoverlapping square patches in raster order. The selected patch is read left to right, then top to bottom.');
      grid(c,f.image,30,15,224,p.size,selected);c.arrow(285,126,365,126,'e');
      c.text(710,36,p.count+' patches; '+p.width+' numbers in each raw patch',null,26);
      var chosen=p.patches[selected];c.box(710,120,630,80,'patch '+(selected+1),vector(chosen),'e');
      c.text(710,205,'Read each patch left to right, then top to bottom.',null,23);
      c.text(140,250,'P = '+p.size,null,22);
    }else if(kind==='projection'){
      c=canvas(210,'The same projection is used on every patch','A raw four-number patch multiplies a four by two weight matrix, producing a two-number embedding.');
      c.box(180,88,290,82,'raw patch '+((options.selected||0)+1),vector(f.patches[options.selected||0]),'e');
      c.arrow(340,88,430,88);c.box(550,88,210,82,'W_patch','4 × 2','e');c.arrow(665,88,760,88);
      c.box(925,88,290,82,'patch embedding',vector(f.embeddings[options.selected||0]),'e');
      c.text(550,175,'One shared matrix, four applications.',null,25);
    }else if(kind==='sequence'){
      c=canvas(195,'Five input rows including CLS','Prepend the class token, then add one two-coordinate position vector to each of five content rows.');
      for(i=0;i<5;i++){c.box(110+i*220,62,192,76,i===0?'CLS':'patch '+i,vector(f.content[i]),'e');c.text(110+i*220,128,'+ '+vector(f.positions[i]),'muted',22);c.text(110+i*220,175,'= '+vector(f.E[i]),'e',24);}
    }else if(kind==='roles'){
      c=canvas(270,'Three projections of the same image-token rows','Input rows E supply all queries, keys, and values. Queries and keys choose weights; values supply the message.');
      c.box(155,132,265,85,'E: five input rows','5 × 2','e');
      ['q','k','v'].forEach(function(role,j){var yy=45+j*85;c.arrow(288,132,515,yy,role);c.text(410,yy+(j===0?-5:17),'W_'+role.toUpperCase(),role,22);c.box(705,yy,340,68,role.toUpperCase()+': 5 × 2',j<2?'for matching':'information to mix',role);});
    }else if(kind==='attention'){
      c=canvas(260,'The selected query reads every input position','An encoder attention row assigns positive weights to all five known input positions. Arrows show mixing weights, not causal importance.');
      var receiver=options.receiver||0;
      c.box(150,125,270,78,receiver===0?'CLS query':'patch '+receiver+' query',vector(f.Q[receiver]),'q');
      for(i=0;i<5;i++){var yy=25+i*51;c.arrow(288,125,595,yy,'a');c.box(815,yy,395,43,(i===0?'CLS':'patch '+i)+'   α = '+f.A[receiver][i].toFixed(3),null,'k');}
    }else if(kind==='mix'){
      c=canvas(255,'A query-specific weighted value mixture','Each value row is multiplied by the CLS attention weight at the same source position. The five vectors are added.');
      for(i=0;i<5;i++){var xx=110+i*220;c.box(xx,45,194,70,i===0?'CLS value':'value '+i,vector(f.V[i]),'v');c.text(xx,105,'× '+f.A[0][i].toFixed(3),'a',24);c.arrow(xx,125,550,181,'a');}
      c.box(550,217,460,64,'CLS message '+vector(f.message[0],3),null,'d');
    }else if(kind==='pipeline'){
      c=canvas(245,'One complete image-classification computation','Pixels become patch embeddings and positioned rows; self-attention updates the class row; a class projection produces logits and probabilities.');
      var nodes=[['pixels','4 × 4','e'],['patch rows','4 × 2','e'],['+ CLS, + positions','5 × 2','e'],['attention + residual','5 × 2','d'],['take CLS','1 × 2','ep'],['class probabilities','1 × 2','ep']];
      nodes.forEach(function(n,j){var row=Math.floor(j/3),col=j%3,xx=180+col*370,yy=55+row*130;c.box(xx,yy,320,78,n[0],n[1],n[2]);if(col<2)c.arrow(xx+161,yy,xx+208,yy);});
      c.svg.appendChild(sv('path',{d:'M920 96 V119 H180 V145',fill:'none',stroke:color.muted,'stroke-width':2.5,'marker-end':'url(#vision1-diagram-'+serial+'-muted)'}));
    }else if(kind==='block'){
      c=canvas(315,'A pre-norm ViT encoder block','LayerNorm precedes multi-head attention and the feed-forward sublayer. Each sublayer has a residual addition. A final LayerNorm follows the stack.');
      // A horizontal layout avoids a tall diagram in the classroom frame.
      var xs=[65,220,430,640,830,1000];
      c.box(xs[0],85,115,58,'E',null,'e');c.box(xs[1],85,160,58,'LayerNorm',null);c.box(xs[2],85,230,58,'multi-head attention',null,'d');c.box(xs[3],85,125,58,'add E',null,'ep');
      c.arrow(124,85,137,85);c.arrow(302,85,312,85);c.arrow(547,85,575,85);c.arrow(705,85,778,85);c.box(890,85,215,58,'LayerNorm',null);
      c.arrow(890,115,890,174);c.box(810,215,400,72,'MLP: linear → GELU → linear','same operation on each row','ep');c.arrow(607,215,510,215);c.box(390,215,232,72,'add previous row',null,'ep');c.arrow(271,215,190,215);c.box(100,215,170,72,'next block',null,'ep');
      c.svg.appendChild(sv('path',{d:'M65 54 V25 H640 V54 M640 116 V155 H390 V178',fill:'none',stroke:color.e,'stroke-width':2,'stroke-dasharray':'6 4','marker-end':'url(#vision1-diagram-'+serial+'-e)'}));
      c.text(550,294,'After the last block: final LayerNorm → CLS row → class head.',null,23);
    }else throw new Error('Unknown Vision Part I diagram: '+kind);
    AT.clear(host);host.appendChild(c.svg);return c.svg;
  }
  function table(host,rows,options){return AT.ui.table(rows,Object.assign({cols:['coordinate 1','coordinate 2'],decimals:3,into:host},options||{}));}
  function patchExplorer(host){
    var root=AT.h('div'), controls=AT.h('div',{class:'vision1-controls'}), plot=AT.h('div'), result=AT.h('p'), size=2, selected=0;
    function draw(){var p=patchify(data.image,size);selected=Math.min(selected,p.count-1);diagram(plot,'patches',{size:size,selected:selected});result.textContent=p.count+' patch tokens + 1 CLS = '+(p.count+1)+' rows; '+((p.count+1)**2)+' attention scores per head.';}
    var select=AT.h('select',{'aria-label':'Patch side length',on:{change:function(){size=+select.value;selected=0;draw();}}},[1,2,4].map(function(p){return AT.h('option',{value:p,selected:p===2},p+' × '+p);}));
    controls.appendChild(AT.h('label',{},'Patch size ',select));
    controls.appendChild(AT.h('button',{type:'button',class:'btn btn-quiet',on:{click:function(){selected=(selected+1)%patchify(data.image,size).count;draw();}}},'Inspect next patch'));
    root.appendChild(controls);root.appendChild(plot);root.appendChild(result);host.appendChild(root);draw();
  }
  function pixelExplorer(host){
    var control=AT.h('div'), plot=AT.h('div'), out=AT.h('p',{class:'callout callout-key','aria-live':'polite'});host.appendChild(control);host.appendChild(plot);host.appendChild(out);
    function draw(v){var f=forward({image:editImage(v)});diagram(plot,'projection',{image:editImage(v),selected:3});out.textContent='Pixel (row 3, column 3) = '+v.toFixed(1)+'. Final CLS = '+vector(f.updated[0],3)+'. p('+data.classes[0]+') = '+f.probs[0].toFixed(3)+'.';}
    AT.ui.slider({into:control,label:'Change one pixel in patch 4',min:0,max:3,step:.25,value:2,format:function(v){return v.toFixed(2);},onInput:draw});draw(2);
  }
  function attentionExplorer(host){
    var select=AT.h('select',{'aria-label':'Receiving query'}), plot=AT.h('div');labels.forEach(function(label,i){select.appendChild(AT.h('option',{value:i},label));});select.addEventListener('change',function(){diagram(plot,'attention',{receiver:+select.value});});host.appendChild(AT.h('label',{},'Receiving row ',select));host.appendChild(plot);diagram(plot,'attention');
  }
  AT.vision={data:copy(data),labels:labels,forward:forward,attention:attention,patchify:patchify,editImage:editImage,blockImage:blockImage,diagram:diagram,table:table,vector:vector,patchExplorer:patchExplorer,pixelExplorer:pixelExplorer,attentionExplorer:attentionExplorer,math:{matmul:matmul,transpose:transpose,softmax:softmax}};
  AT.axes.named=false;
  var defs={e:['image-token representation','A projected image patch or CLS row, with position added at the input.'],q:['query','A receiving image-token row projected for matching.'],k:['key','A source image-token row projected for matching.'],v:['value','A source image-token row projected into the information used in the mixture.'],a:['attention weight','A mixing amount for one receiving row and one source row.'],d:['attention update','The weighted value message after the output projection W_O.'],ep:['updated representation','The input row plus its attention update.']};
  AT.objects.forEach(function(o){if(defs[o.cls]){o.name=defs[o.cls][0];o.def=defs[o.cls][1];o.tip=o.def;}});
  function notation(g,s,m,shape,dims){AT.notation.push({g:g,sym:s,mean:m,shape:shape,dims:function(){return dims||'';},parts:['vision1']});}
  notation('token','r_j','Raw flattened pixel values of patch j','1\\times(P^2C)','1×4');
  notation('token','\\ve{e_j^{\\mathrm{patch}}}=r_jW_{\\mathrm{patch}}+b_{\\mathrm{patch}}','Patch embedding before position is added','1\\times d_{\\mathrm{model}}','1×2');
  notation('token','\\ve{e_j}','Input row after adding its position vector; row 0 is CLS','1\\times d_{\\mathrm{model}}','1×2');
  notation('token','\\vq{q_i},\\vk{k_j},\\vv{v_j}','Query from receiver i; key and value from source j','1\\times2','');
  notation('token','\\vd{\\Delta e_i}=(\\sum_j\\alpha_{ij}v_j)W_O','Attention update for receiving row i','1\\times d_{\\mathrm{model}}','1×2');
  notation('token','\\vp{e_i^\\prime}=e_i+\\Delta e_i','Row after residual addition','1\\times d_{\\mathrm{model}}','1×2');
  notation('matrix','\\ve{E}','Stacked input rows: CLS first, then four patches','(N+1)\\times d_{\\mathrm{model}}','5×2');
  notation('matrix','\\vq{Q},\\vk{K},\\vv{V}','Separate projections of those same input rows','(N+1)\\times2','5×2 each');
  notation('matrix','\\va{A}=\\operatorname{softmax}_{\\mathrm{row}}(QK^\\top/\\sqrt{d_k})','One row of mixing weights per receiving token','(N+1)\\times(N+1)','5×5');
  notation('sizes','N,P,C','Number of patches, patch side length, channel count','','4, 2, 1');
  notation('sizes','d_{\\mathrm{model}},d_k,d_v','Representation width, matching width, and value width','','2, 2, 2');
  notation('sizes','W_{\\mathrm{patch}}','Shared pixel-to-embedding projection','P^2C\\times d_{\\mathrm{model}}','4×2');
  notation('sizes','W_Q,W_K,W_V,W_O','Four separate parameters; initially W_Q = W_K and W_V = W_O, but these pairs are not tied during learning','2\\times2','');
})();
