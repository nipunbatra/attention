/* One photographic illustration across the vision series. All crops come
   from the embedded asset; annotations are teaching prompts, not predictions. */
(function () {
  'use strict';
  const AT = window.AT;
  const assets = window.__VISION_SCENES__;
  const W = 1536, H = 1024;
  const patch = [512, 512, 256, 256];
  let serial = 0;
  const style = AT.h('style', {}, `
    .vision-scene{margin:12px 0;max-width:100%}
    .vision-scene > svg{display:block;width:100%;height:auto;overflow:visible}
    .vision-scene svg svg{overflow:hidden}
    .vision-scene figcaption{font-size:14px;color:var(--ink-3);margin:8px 0 0;line-height:1.4}
    .vision-scene-mobile{display:none}
    .vision-scene-mobile svg{width:100%;height:auto;overflow:hidden}
    .vision-scene-mobile p{margin:10px 0 20px;line-height:1.5}
    .vision-scene-mobile .scene-label{display:block;font-size:.85em;font-weight:600;color:var(--ink-2);margin-bottom:5px}
    .vision-scene-mobile .scene-path{border-left:2px solid var(--line);padding-left:16px;margin:16px 0}
    body.present .vision-scene{margin:8px 0}
    body.present .vision-scene figcaption{font-size:18px;line-height:1.25}
    @media(max-width:760px){body:not(.present) .vision-scene.has-mobile-scene > svg{display:none}body:not(.present) .vision-scene-mobile{display:block}}
  `);
  document.head.appendChild(style);

  function mount(host, options = {}) {
    if (!host) throw new Error('The vision scene needs a host element.');
    const mode = options.mode || 'scene', variant = options.variant || 'two';
    if (!assets[variant]) throw new Error('Unknown scene variant: ' + variant);
    const height = {scene:380, patches:400, 'patch-crop':425, masked:410, crops:410, targets:410, captions:410, compare:400, evidence:410}[mode];
    if (!height) throw new Error('Unknown scene mode: ' + mode);
    const id = 'vision-scene-' + (++serial);
    const title = options.alt || ({scene:'Two ceramic mugs, a blue book and a plant on a table', patches:'A regular patch grid cuts across the mugs and the book', 'patch-crop':'The highlighted patch contains a mug edge, tabletop and part of the book', masked:'A masked patch and the original pixels withheld from the input', crops:'Three crops of the same image contain two, one and zero mugs', targets:'An image supplies the target for a self-supervised task', captions:'Several descriptions of the same image', compare:'Two illustrative scenes: two mugs and one mug', evidence:'Questions with and without visible evidence'}[mode]);
    const fig = AT.h('figure', {class:'vision-scene', 'data-vision-scene':mode});
    const svg = AT.svg('svg', {viewBox:`0 0 1100 ${height}`, role:'img', 'aria-labelledby':id+'-title'});
    svg.appendChild(AT.svg('title', {id:id+'-title'}, title));
    const defs = AT.svg('defs');
    const marker = AT.svg('marker', {id:id+'-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto'});
    marker.appendChild(AT.svg('path', {d:'M0 0L10 5L0 10Z',fill:'var(--ink-3)'}));
    defs.appendChild(marker);svg.appendChild(defs);
    function text(x,y,t,size=26,anchor='start',color='var(--ink)',weight=400) {
      const el=AT.svg('text',{x,y,'font-family':'var(--font-ui)','font-size':size,'text-anchor':anchor,'dominant-baseline':'middle',fill:color,'font-weight':weight},t);
      svg.appendChild(el);return el;
    }
    function lines(x,y,t,width=38,size=26) {
      const words=t.split(/\s+/);let row='',n=0;
      words.forEach(word=>{if(row && (row+' '+word).length>width){text(x,y+n*34,row,size);row=word;n++;}else row+=(row?' ':'')+word;});
      if(row)text(x,y+n*34,row,size);return n+1;
    }
    function image(x,y,w,h,source=[0,0,W,H],v=variant) {
      const nested=AT.svg('svg',{x,y,width:w,height:h,viewBox:source.join(' '),preserveAspectRatio:'none',overflow:'hidden'});
      nested.appendChild(AT.svg('image',{href:assets[v],x:0,y:0,width:W,height:H}));
      svg.appendChild(nested);return nested;
    }
    function rect(x,y,w,h,fill='none',stroke='var(--ink)',sw=2) {
      svg.appendChild(AT.svg('rect',{x,y,width:w,height:h,fill,stroke,'stroke-width':sw}));
    }
    function arrow(x1,y1,x2,y2) {svg.appendChild(AT.svg('path',{d:`M${x1} ${y1}L${x2} ${y2}`,fill:'none',stroke:'var(--ink-3)','stroke-width':2,'marker-end':`url(#${id}-arrow)`}));}
    function box(x,y,w,h,words,sub) {
      svg.appendChild(AT.svg('rect',{x,y,width:w,height:h,rx:8,fill:'var(--card)',stroke:'var(--line)','stroke-width':2}));
      text(x+w/2,y+h/2-(sub?16:0),words,26,'middle');
      if(sub)text(x+w/2,y+h/2+20,sub,22,'middle','var(--ink-2)');
    }
    function grid(x,y,w,h,cols=6,rows=4,selected=true) {
      // White under-stroke keeps the grid visible against dark ceramic.
      for(let c=0;c<=cols;c++){const xx=x+c*w/cols;svg.appendChild(AT.svg('path',{d:`M${xx} ${y}V${y+h}`,stroke:'#fff','stroke-width':3,opacity:.85}));svg.appendChild(AT.svg('path',{d:`M${xx} ${y}V${y+h}`,stroke:'var(--ink)','stroke-width':1,opacity:.65}));}
      for(let r=0;r<=rows;r++){const yy=y+r*h/rows;svg.appendChild(AT.svg('path',{d:`M${x} ${yy}H${x+w}`,stroke:'#fff','stroke-width':3,opacity:.85}));svg.appendChild(AT.svg('path',{d:`M${x} ${yy}H${x+w}`,stroke:'var(--ink)','stroke-width':1,opacity:.65}));}
      if(selected)rect(x+2*w/6,y+2*h/4,w/6,h/4,'none','var(--c-e)',5);
    }
    if(mode==='scene') {
      image(265,0,570,380);
      svg.setAttribute('viewBox','265 0 570 380');fig.style.maxWidth='570px';fig.style.marginInline='auto';
    } else if(mode==='patches') {
      if(options.square){image(350,0,400,400,[256,0,1024,1024]);grid(350,0,400,400,options.grid||14,options.grid||14,false);svg.setAttribute('viewBox','350 0 400 400');fig.style.maxWidth='400px';}
      else {image(250,0,600,400);grid(250,0,600,400);svg.setAttribute('viewBox','250 0 600 400');fig.style.maxWidth='600px';}
      fig.style.marginInline='auto';
    } else if(mode==='patch-crop'||mode==='masked') {
      image(20,35,510,340);grid(20,35,510,340);
      const px=20+2*510/6,py=35+2*340/4;
      if(mode==='masked')rect(px,py,85,85,'var(--paper)','var(--ink)',2);
      arrow(px+88,py+42,750,172);
      image(770,55,240,240,patch);
      rect(770,55,240,240,'none','var(--c-e)',3);
      text(275,height-20,mode==='masked'?'Input: this patch is hidden':'One image, a regular grid',24,'middle');
      text(890,326,mode==='masked'?'Target: original pixels':'The same patch, enlarged',24,'middle');
    } else if(mode==='crops') {
      const crops=[[0,0,W,H],[110,285,540,480],[500,0,510,300]];
      const labels=['Wide view · two mugs','Close view · one mug','Plant view · no mugs'];
      crops.forEach((crop,i)=>{
        const x=20+i*365;const scale=Math.min(330/crop[2],230/crop[3]);
        const w=crop[2]*scale,h=crop[3]*scale;
        image(x+(330-w)/2,65+(230-h)/2,w,h,crop);
        text(x+165,338,labels[i],25,'middle');
      });
      text(550,390,'All three are crops of the same source image.',22,'middle','var(--ink-2)');
    } else if(mode==='targets') {
      if(options.target==='features') {
        image(15,30,330,220);text(180,280,'Original image',24,'middle');
        arrow(355,140,425,140);box(435,90,250,100,'Target encoder','outputs patch features');
        arrow(695,140,765,140);box(775,90,300,100,'Selected target row','hidden-patch position');
        text(925,225,'stop gradient',22,'middle','var(--ink-2)');
        text(550,345,'The masked-input branch predicts this vector.',26,'middle');
      } else {
        image(20,35,510,340);rect(190,205,85,85,'var(--paper)','var(--ink)',2);
        arrow(285,247,740,175);image(770,55,240,240,patch);
        text(275,390,'Input: patch withheld',24,'middle');
        text(890,330,'Target: its RGB pixels',24,'middle');
        text(890,365,'Kept from the original image',22,'middle','var(--ink-2)');
      }
    } else if(mode==='compare') {
      image(20,5,510,340,[0,0,W,H],'two');image(570,5,510,340,[0,0,W,H],'one');
      const labels=options.labels||['Scene A: two mugs','Scene B: one mug'];
      labels.forEach((label,i)=>text(275+i*550,382,label,26,'middle'));
    } else if(mode==='captions'||mode==='evidence') {
      image(0,36,510,340);
      const entries=mode==='captions'?(options.captions||[
        {label:'Fits',text:'Two mugs in front of a plant.'},
        {label:'Also fits',text:'A blue book beside two mugs.'},
        {label:'Wrong count',text:'One mug beside a blue book.'}
      ]):(options.questions||[
        {label:'Visible',text:'How many mugs? Two.'},
        {label:'Visible',text:'What color is the book? Blue.'},
        {label:'Not visible',text:'How hot are the mugs?'}
      ]);
      entries.slice(0,3).forEach((entry,i)=>{
        const y=55+i*125;
        text(565,y,entry.label||'',22,'start','var(--ink-2)',600);
        lines(565,y+37,entry.text,35,26);
      });
    }
    fig.appendChild(svg);
    // A narrow article stacks the same image windows and labels. The classroom
    // figure stays one SVG; no independently authored mobile lesson is kept.
    if(!['scene','patches'].includes(mode)) {
      const mobile=AT.h('div',{class:'vision-scene-mobile'});
      const photo=(source=[0,0,W,H],label='',v=variant,masked=false,highlight=false)=>{
        const s=AT.svg('svg',{viewBox:source.join(' '),role:'img','aria-label':label||title});
        s.appendChild(AT.svg('image',{href:assets[v],x:0,y:0,width:W,height:H}));
        if(highlight){
          for(let x=0;x<=W;x+=256)s.appendChild(AT.svg('path',{d:`M${x} 0V${H}`,stroke:'#fff','stroke-width':4,opacity:.8}));
          for(let y=0;y<=H;y+=256)s.appendChild(AT.svg('path',{d:`M0 ${y}H${W}`,stroke:'#fff','stroke-width':4,opacity:.8}));
          s.appendChild(AT.svg('rect',{x:patch[0],y:patch[1],width:patch[2],height:patch[3],fill:'none',stroke:'var(--c-e)','stroke-width':10}));
        }
        if(masked)s.appendChild(AT.svg('rect',{x:patch[0],y:patch[1],width:patch[2],height:patch[3],fill:'var(--paper)',stroke:'var(--ink)','stroke-width':5}));
        mobile.appendChild(s);
        if(label)mobile.appendChild(AT.h('p',{},label));
      };
      if(mode==='patch-crop'||mode==='masked'||(mode==='targets'&&options.target!=='features')){
        photo([0,0,W,H],mode==='patch-crop'?'The outlined patch is at row 3, column 3.':'Input with one patch withheld.',variant,mode!=='patch-crop',mode==='patch-crop');
        photo(patch,mode==='patch-crop'?'The same patch: mug edge, table and book.':'Target: the original withheld pixels.');
      }else if(mode==='crops'){
        photo([0,0,W,H],'Wide view: two mugs.');photo([110,285,540,480],'Close view: one mug.');photo([500,0,510,300],'Plant view: no mugs.');
      }else if(mode==='compare'){
        const labels=options.labels||['Scene A: two mugs','Scene B: one mug'];
        photo([0,0,W,H],labels[0],'two');photo([0,0,W,H],labels[1],'one');
      }else if(mode==='targets'){
        photo([0,0,W,H],'The target encoder reads the original image.');
        mobile.appendChild(AT.h('div',{class:'scene-path'},AT.h('p',{},'Target encoder → select the hidden-patch row → target features (stop gradient).'),AT.h('p',{},'The masked-input branch predicts this vector.')));
      }else{
        photo();
        const entries=mode==='captions'?(options.captions||[
          {label:'Fits',text:'Two mugs in front of a plant.'},{label:'Also fits',text:'A blue book beside two mugs.'},{label:'Wrong count',text:'One mug beside a blue book.'}
        ]):(options.questions||[
          {label:'Visible',text:'How many mugs? Two.'},{label:'Visible',text:'What color is the book? Blue.'},{label:'Not visible',text:'How hot are the mugs?'}
        ]);
        entries.slice(0,3).forEach(entry=>mobile.appendChild(AT.h('p',{},AT.h('span',{class:'scene-label'},entry.label||''),entry.text)));
      }
      fig.classList.add('has-mobile-scene');fig.appendChild(mobile);
    }
    // Asset provenance lives in the article notes and figures/vision-scene/README.md.
    // Keep the classroom figure focused on its task and observable content.
    host.replaceChildren(fig);
    return {el:fig,svg};
  }
  AT.visionScene={mount,width:W,height:H,patch:patch.slice()};
})();
