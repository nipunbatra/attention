// Editable teaching diagrams for the Part 1 representation detour.
// Character points use toy1.json. Word-map positions and domain flows are schematic.
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const ink = '#4A5160', blue = '#245EDB', amber = '#AA4E08';
  let serial = 0;
  function el(tag, attrs = {}, content) {
    const n = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => n.setAttribute(key, value));
    if (content !== undefined) n.textContent = content;
    return n;
  }
  function put(parent, tag, attrs, content) { const n = el(tag, attrs, content); parent.append(n); return n; }
  function base(name, description, height = 330) {
    const id = 'ep-' + (++serial), s = el('svg', {viewBox:`0 0 1000 ${height}`, role:'img',
      'aria-labelledby':id+'-title '+id+'-desc', 'data-embedding-diagram':name, width:'100%', style:'height:auto'});
    put(s,'title',{id:id+'-title'},name.replaceAll('-', ' '));
    put(s,'desc',{id:id+'-desc'},description);
    const defs=put(s,'defs',{}), marker=put(defs,'marker',{id:id+'-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'});
    put(marker,'path',{d:'M0 0 L10 5 L0 10 Z',fill:ink});
    s.dataset.arrow=id+'-arrow'; return s;
  }
  function text(s,x,y,value,{size=25,color='#14171F',anchor='middle',bold=false}={}) {
    return put(s,'text',{x,y,fill:color,'font-size':size,'font-family':'var(--font-ui, sans-serif)',
      'font-weight':bold?650:400,'text-anchor':anchor,'dominant-baseline':'middle'},value);
  }
  function arrow(s,x1,y1,x2,y2) { put(s,'line',{x1,y1,x2,y2,stroke:ink,'stroke-width':2.3,'marker-end':`url(#${s.dataset.arrow})`}); }
  function node(s,x,y,w,lines,{color=ink,fill='#FFFFFF',h=76}={}) {
    put(s,'rect',{x,y,width:w,height:h,rx:7,fill,stroke:color,'stroke-width':2});
    lines.forEach((l,i)=>text(s,x+w/2,y+h/2+(i-(lines.length-1)/2)*29,l,{color,size:24}));
  }
  function row(s,x,y,label,values) {
    text(s,x,y,label,{anchor:'start',bold:true,color:blue});
    text(s,x,y+39,'['+values.map(v=>v.toFixed(2).replace('-','−')).join(',  ')+']',{anchor:'start',color:blue,size:30});
  }
  function points(many=false) {
    const toy=window.__TOY__, tokens=many?['a','b','i']:['a'];
    const s=base(many?'three-character-points':'one-character-point',
      'Actual saved character embeddings. A row of two numbers is a point. Dashed guides locate the a row on the two axes.',390);
    const X=v=>690+v*65, Y=v=>183-v*65;
    [-2,-1,0,1,2].forEach(v=>{
      put(s,'line',{x1:X(v),y1:Y(2),x2:X(v),y2:Y(-2),stroke:'#D9DDE5'});
      put(s,'line',{x1:X(-2),y1:Y(v),x2:X(2),y2:Y(v),stroke:'#D9DDE5'});
      text(s,X(v),Y(-2)+22,String(v),{size:18,color:ink});
      text(s,X(-2)-20,Y(v),String(v),{size:18,color:ink});
    });
    arrow(s,X(-2.2),Y(0),X(2.4),Y(0)); arrow(s,X(0),Y(-2.2),X(0),Y(2.4));
    text(s,700,365,'coordinate 1',{size:22});text(s,748,30,'coordinate 2',{size:22,anchor:'start'});
    tokens.forEach((c,j)=>{
      const v=toy.E[toy.vocab.indexOf(c)],cx=X(v[0]),cy=Y(v[1]);
      row(s,70,80+j*98,c,v);
      if(!many){
        put(s,'path',{d:`M${X(0)} ${cy} H${cx} V${Y(0)}`,fill:'none',stroke:blue,'stroke-width':2,'stroke-dasharray':'5 5'});
      }
      put(s,'circle',{cx,cy,r:7,fill:blue,'data-token':c,'data-coordinates':JSON.stringify(v)});
      text(s,cx+(c==='b'?-17:17),cy+(c==='i'?19:-19),c,{bold:true,color:blue});
    });
    return s;
  }
  function learning() {
    const s=base('learning-the-coordinates','The three known characters select trainable embedding rows. A predictor produces a next-character distribution. The observed next character supplies the loss. Training changes the table and predictor.',300);
    node(s,20,100,170,['known context','a a b']);
    node(s,250,100,180,['embedding','table'],{color:blue,fill:'#E4ECFF'});
    node(s,490,100,200,['next-character','predictor']);
    node(s,800,100,170,['prediction','loss'],{color:'#BE123C'});
    arrow(s,190,138,250,138);arrow(s,430,138,490,138);arrow(s,690,138,800,138);
    text(s,885,38,'observed i',{color:amber});arrow(s,885,57,885,97);
    put(s,'path',{d:'M885 182 V245 H340 V183',fill:'none',stroke:ink,'stroke-width':2,'stroke-dasharray':'7 5','marker-end':`url(#${s.dataset.arrow})`});
    text(s,585,278,'training updates both the table and the predictor',{size:24});return s;
  }
  function wordLearning() {
    const s=base('word2vec-learning','Look up the vector for king, predict a nearby word such as ruled, and use the error to train the table and predictor.',300);
    node(s,20,100,135,['king']);
    node(s,220,100,195,['lookup its','word vector'],{color:blue,fill:'#E4ECFF'});
    node(s,480,100,205,['predict a','nearby word']);
    node(s,790,100,180,['prediction','loss'],{color:'#BE123C'});
    arrow(s,158,138,217,138);arrow(s,418,138,477,138);arrow(s,688,138,787,138);
    text(s,880,38,'observed: ruled',{color:amber,size:23});arrow(s,880,57,880,97);
    put(s,'path',{d:'M880 182 V239 H318 V183',fill:'none',stroke:ink,'stroke-width':2,'stroke-dasharray':'7 5','marker-end':'url(#'+s.dataset.arrow+')'});
    text(s,585,277,'Train on many word pairs; keep the learned word vectors.',{size:24});
    return s;
  }
  function wordMap(offsets=false) {
    const s=base(offsets?'word-analogy-offsets':'four-word-points',
      'Hand-chosen two-dimensional coordinates: man [1,1], woman [3,1], king [1,3], queen [3,3].'+
      (offsets?' The equal upward arrows show the man-to-king and woman-to-queen offsets.':' These illustrate vector positions, not measured Word2vec results.'),350);
    const X=v=>390+v*105,Y=v=>307-v*75;
    arrow(s,X(0),Y(0),X(4.5),Y(0));arrow(s,X(0),Y(0),X(0),Y(3.8));
    text(s,965,327,'coordinate 1',{anchor:'end',size:21});
    text(s,415,30,'coordinate 2',{size:21,anchor:'start'});
    [1,2,3,4].forEach(v=>text(s,X(v),Y(0)+20,String(v),{size:18,color:ink}));
    [1,2,3].forEach(v=>text(s,X(0)-22,Y(v),String(v),{size:18,color:ink}));
    const rows=[['man',1,1],['woman',3,1],['king',1,3],['queen',3,3]];
    if(offsets){
      [1,3].forEach(x=>arrow(s,X(x),Y(1)-12,X(x),Y(3)+12));
      text(s,X(1)-22,Y(2),'[0, 2]',{anchor:'end',color:ink,size:23});
      text(s,X(3)-22,Y(2),'[0, 2]',{anchor:'end',color:ink,size:23});
    }
    rows.forEach(([word,x,y],i)=>{
      row(s,50,37+i*77,word,[x,y]);
      put(s,'circle',{cx:X(x),cy:Y(y),r:7,fill:blue,'data-word':word,'data-vector':JSON.stringify([x,y])});
      text(s,X(x)+14,Y(y)+(y===3?-21:26),word,{anchor:'start',color:blue,bold:true});
    });
    return s;
  }
  function backToName() {
    const s=base('back-to-character-embeddings','The known context a a b selects the trained a row twice and the trained b row once. Each position receives two coordinates.',260);
    ['a','a','b'].forEach((token,i)=>{
      const x=55+i*330, v=window.__TOY__.E[window.__TOY__.vocab.indexOf(token)];
      node(s,x,22,230,[token+'  /  position '+(i+1)],{h:55});
      arrow(s,x+115,80,x+115,127);
      node(s,x,133,230,['['+v.map(n=>n.toFixed(2).replace('-','−')).join(',  ')+']'],{color:blue,fill:'#E4ECFF',h:65});
    });
    text(s,500,237,'The same a row is looked up twice.',{size:24});
    return s;
  }
  function documentFlow() {
    const s=base('document-embedding','A passage is encoded into a fixed-width document representation for a task such as retrieval. This is a schematic, not a measured document embedding.',300);
    node(s,20,70,290,['The bank lowered','its loan rate.','More customers applied.'],{h:135});
    node(s,385,97,240,['text encoder','and aggregation'],{h:85});
    node(s,712,97,260,['document vector','e_doc'],{color:blue,fill:'#E4ECFF',h:85});
    arrow(s,312,137,382,137);arrow(s,627,137,709,137);
    text(s,505,260,'A passage with many tokens can yield one fixed-width row.',{size:24});return s;
  }
  function imageFlow() {
    const s=base('image-embedding','The recurring two-mug scene is input to an image encoder, which produces an image representation. No model output for this scene is claimed.',330);
    put(s,'image',{href:window.__EMBEDDING_SCENE__,x:12,y:15,width:390,height:260,preserveAspectRatio:'xMidYMid meet'});
    node(s,475,100,210,['image encoder','CNN or ViT'],{h:90});
    node(s,775,100,210,['image vector','e_image'],{color:blue,fill:'#E4ECFF',h:90});
    arrow(s,405,145,472,145);arrow(s,688,145,772,145);
    text(s,210,305,'pixels: height × width × channels',{size:22});return s;
  }
  function wave(s,values,x,y,w,h,color=blue) {
    const d=values.map((v,i)=>(i?'L':'M')+(x+i*w/(values.length-1))+' '+(y+h/2-v*h/2)).join(' ');
    put(s,'line',{x1:x,y1:y+h/2,x2:x+w,y2:y+h/2,stroke:'#D9DDE5'});
    put(s,'path',{d,fill:'none',stroke:color,'stroke-width':3,'data-samples':JSON.stringify(values)});
  }
  function timeFlow() {
    const s=base('time-series-embedding','A synthetic measured window is encoded into one fixed-width time-series vector for forecasting or classification.',320);
    wave(s,[0,.4,.9,.4,0,-.5,-.9,-.4,0,.5,1,.5],20,70,320,130);
    text(s,180,238,'time within the observed window',{size:21});
    node(s,422,92,235,['temporal encoder','and aggregation'],{h:88});
    node(s,755,92,225,['window vector','e_time'],{color:blue,fill:'#E4ECFF',h:88});
    arrow(s,344,136,419,136);arrow(s,660,136,752,136);return s;
  }
  function timePatterns() {
    const s=base('time-series-patterns','Two schematic repeating signals and one isolated spike. A representation useful for recognizing repeating patterns should preserve that distinction.',335);
    const signals=[[0,.8,0,-.8,0,.8,0,-.8,0],[.2,.9,.2,-.5,.2,.9,.2,-.5,.2],[0,0,0,0,1,0,0,0,0]];
    signals.forEach((v,j)=>{wave(s,v,240,16+j*100,610,75);text(s,210,54+j*100,['repeating A','repeating B','single spike'][j],{anchor:'end',size:23});});return s;
  }
  window.AT.embeddingPrimer={points,learning,wordLearning,wordMap,backToName,documentFlow,imageFlow,timeFlow,timePatterns};
})();
