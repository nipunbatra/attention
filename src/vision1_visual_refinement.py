"""A conservative visual pass: retain derivations, add a clean presentation layer.

The large reference map, original paper and two CLS parameter digressions remain
in reading notes. All nine interactive examples also get a standalone lab page.
"""
from html import escape
from pathlib import Path
import json
import re
import numpy as np
from vision1_focus_common import Figures
from vision1_palette import CSS

READING_ONLY = {'paper-vision-transformer', 'photo-label-loss',
                'cls-parameter-origin', 'cls-parameter-learning',
                'read-attention-map', 'cover-2', 'cover-3', 'cover-4'}
LOCAL_ROLES = {
    'patch-context', 'patch-context-weights', 'patch-context-update',
    'bridge-image-query', 'bridge-image-key', 'bridge-image-value',
    'real-patch-qkv', 'real-cls-attention', 'real-attention-product',
    'real-attention-cls-zoom', 'real-attention-weights', 'real-attention-mask',
    'real-message-text-analogy', 'real-cls-values-origin', 'real-cls-value-scaling',
    'real-cls-value-contributions', 'real-cls-value-sum', 'real-cls-message-destination',
    'real-attention-values', 'real-cls-residual',
    'code-photo-attention', 'code-photo-attention-messages', 'code-photo-attention-join',
}


def key(markup):
    return re.search(r'class="frame[^\"]*" id="([^\"]+)"', markup).group(1)


def refine_visuals(b, sections):
    f = Figures(b)
    t, box, arrow, line = f.t, f.box, f.arrow, f.line
    metadata = {item['id']: item for item in b['FRAMES']}
    existing = {key(fr): fr for _, frames in sections for fr in frames}

    def add(k, title, body, caption, question, point, prose='', height=440):
        return f.add(k, title, body, caption, question, point, prose, height=height)

    # Reuse the actual three-family summary from Transformers beyond next-token
    # prediction. Keep a local SVG so this lecture remains independently buildable.
    recap = (b['ASSETS'] / 'prior-lecture-model-families.svg').read_text()
    body = recap.replace('<svg ', '<svg x="0" y="0" width="1160" height="440" ', 1)
    old = metadata['prior-encoder-recap']
    add('prior-encoder-recap', old['title'], body,
        'Today we reuse the encoder on the left: image patches become tokens, and the final image representation predicts a label.',
        'Which architecture reads a complete supplied input and returns contextual representations?',
        'The encoder, shown on the left. The full attention square lets every token read every token. '
        'The decoder uses causal attention and repeats next-token prediction. The encoder–decoder '
        'adds a source pathway into the target decoder through cross-attention. For ViT, replace '
        'text tokens with image patches, then use a classification head.',
        '<p>Summary diagram reused from <em>Transformers beyond next-token prediction</em> '
        '(“Encoder, decoder-only and encoder-decoder models”). The same token rows, attention '
        'patterns and source-to-target pathway connect this lecture to that recap.</p>')

    body=''
    for y,label,tokens,color in [(34,'TEXT',['CLS','Raghav','goes','to','school'],'language'),
                                 (213,'IMAGE',['CLS','P1','P2','…','P196'],'vision')]:
        body+=t(35,y,label,28,color,weight=650)
        for i,token in enumerate(tokens):
            body+=box(35+i*106,y+25,95,token,'special' if i==0 else color,h=47,size=23)
        body+=arrow(573,y+50,615,y+50,color)
        body+=box(625,y+7,263,['Encoder blocks','whole-input attention'],'mixing',h=86,size=23)
        body+=arrow(900,y+50,938,y+50,color)
        body+=box(948,y+7,182,['Read final CLS','→ classifier'],color,h=86,size=22)
    body+=t(580,379,'Same encoder idea. Different tokens.',36,'ink','middle',650)
    body+=t(580,429,'Blue: vision states     Purple: language states     Amber: learned CLS / position',23,'ink-2','middle')
    old=metadata['vit-same-encoder']
    add('vit-same-encoder',old['title'],body,'Attention builds context in either sequence. The input representation and the task head determine how we use the encoder.',*old['notes'].split('\n',1))

    body=t(250,40,'TEXT',30,'language','middle',650)+t(870,40,'IMAGE',30,'vision','middle',650)
    for x,color,labels in [(35,'language',[['“bank” → token ID'],['Embedding table','look up a learned row'],['token content row','D features']]),
                           (655,'vision',[['RGB patch: 16 × 16 × 3','flatten → 768 values'],['Shared Linear(768, D)','compute a learned projection'],['patch content row','D features']])]:
        for i,pair in enumerate(labels):
            y=67+i*120
            body+=box(x,y,440,pair,'special' if i==1 else color,h=83,size=26)
            if i<2:body+=arrow(x+220,y+90,x+220,y+112,color)
    body+=t(580,426,'Add learned position → send the rows into the encoder',29,'special','middle')
    old=metadata['vit-token-inputs']
    add('vit-token-inputs',old['title'],body,old['caption'],*old['notes'].split('\n',1))

    # Both streams in translation are language. Color encodes modality, labels
    # distinguish source/target; Q/K/V have their own small local-role legend.
    body=t(30,35,'TRANSLATION · cross-attention',28,'neutral',weight=650)
    for x,label,out in [(30,'Target-prefix states','Q'),(610,'Encoded source states','K, V')]:
        body+=box(x,64,290,label,'language',h=70,size=26)
        body+=arrow(x+302,99,x+340,99,'mixing')+box(x+350,64,175,out,'mixing',h=70,size=30)
    body+=t(580,175,'Two language streams meet at attention.',26,'language','middle')
    body+=line(30,207,1130,207)
    body+=t(30,251,'ViT · self-attention',28,'neutral',weight=650)
    body+=box(30,281,430,['Current CLS + patch states','one image sequence'],'vision',h=86,size=26)
    body+=arrow(475,324,537,324,'mixing')+box(550,281,575,['Project the same rows into Q, K and V','then mix their information'],'mixing',h=86,size=26)
    body+=t(580,427,'ViT has one stream. Translation had two streams.',34,'ink','middle',650)
    old=metadata['vit-self-vs-cross']
    add('vit-self-vs-cross',old['title'],body,'Self-attention uses one input sequence. Cross-attention takes queries from one stream and keys and values from another.',*old['notes'].split('\n',1))

    # House-style replacement for the historical raster, with model choices
    # named rather than suggesting all ViTs use these particular dimensions.
    body=f.image(35,55,180,180,f.photo)
    for j in range(1,14):
        body+=line(35+j*180/14,55,35+j*180/14,235,'card',.7)+line(35,55+j*180/14,215,55+j*180/14,'card',.7)
    body+=t(125,274,'224 × 224 RGB',25,'vision','middle')
    body+=arrow(230,144,274,144,'vision')
    body+=box(287,91,278,['Shared patch projection','196 rows · 192 features'],'vision',h=104,size=25)
    body+=arrow(578,144,625,144,'vision')
    body+=box(638,91,235,['+ learned CLS','+ learned position'],'special',h=104,size=26)
    body+=arrow(887,144,932,144,'mixing')+box(945,91,181,['Encoder','× 12'],'mixing',h=104,size=28)
    body+=line(1035,210,1035,319,'mixing')+arrow(1035,319,914,319,'mixing')
    body+=box(625,283,274,['Final LayerNorm','read CLS: 192 features'],'vision',h=83,size=24)
    body+=arrow(610,325,552,325)+box(292,283,247,['Linear class head','1,000 class scores'],'neutral',h=83,size=25)
    body+=t(35,426,'ViT: Dosovitskiy et al., 2020 · dimensions shown for our ViT-Tiny checkpoint',23,'ink-2')
    add('vit-house-architecture','The image classifier, drawn as one encoder pipeline',body,
        'Patches become feature rows. Learned CLS and position prepare the input. Encoder blocks build context; the final CLS feeds the classifier.',
        'Where does the image become a sequence?',
        'The shared patch projection creates 196 rows. CLS adds row 197; position adds location information without changing the shape.',
        '<p>Architecture adapted from <a href="https://arxiv.org/abs/2010.11929">Dosovitskiy et al., 2020</a>. '
        'The original paper figure is retained in the optional reference material below.</p>')

    body=t(35,28,'ONE IMAGE · batch axis omitted',24,'ink-2')
    steps=[('Pixels','3 × 224 × 224','vision'),('Patch projection','196 × 192','vision'),
           ('Add CLS + position','197 × 192','special'),('12 encoder blocks','197 × 192','mixing'),
           ('Final LN → read CLS','192','vision'),('Class head','1,000 logits','neutral')]
    for i,(label,shape,color) in enumerate(steps):
        x=35+(i%3)*382;y=64+(i//3)*238
        body+=f.rect(x,y,325,122,color,'card',6)
        body+=t(x+162,y+37,label,25,color,'middle')+t(x+162,y+89,shape,39,color,'middle',650)
        if i%3!=2:body+=arrow(x+338,y+61,x+368,y+61)
    body+=line(961,199,961,244)+line(961,244,197,244)+arrow(197,244,197,290)
    old=metadata['vit-shape-trace']
    add('vit-shape-trace',old['title'],body,'Blocks preserve the shape. They change what each row represents.',
        *old['notes'].split('\n',1),height=450)

    # This reusable block has one row per residual branch, large labels, no
    # arbitrary head role colors. The detailed multihead reference remains.
    body=t(35,32,'ONE PRE-LAYERNORM BLOCK · input and output: 197 × 192',28,'vision',weight=650)
    for y,label,branch,detail in [(105,'E','Self-attention','3 heads · 64 features each'),
                                (330,'U','MLP','192 → 768 → 192 · GELU')]:
        body+=box(35,y,136,label,'vision',h=72,size=32)
        body+=arrow(180,y+36,222,y+36,'vision')+box(235,y,208,'LayerNorm','neutral',h=72,size=26)
        body+=arrow(455,y+36,497,y+36)+box(510,y-8,367,[branch,detail],'mixing' if label=='E' else 'neutral',h=88,size=25)
        body+=arrow(890,y+36,955,y+36,'mixing')
        body+=f'<circle cx="980" cy="{y+36}" r="25" fill="var(--card)" stroke="var(--vision)" stroke-width="2"/>'
        body+=t(980,y+47,'+',34,'vision','middle')
        body+=line(198,y+36,198,y-45,'vision')+line(198,y-45,980,y-45,'vision')+arrow(980,y-45,980,y+1,'vision')
        if label=='U':body+=t(580,y-58,'Carry the input along the skip path',23,'vision','middle')
        body+=arrow(1016,y+36,1065,y+36,'vision')+t(1110,y+47,'U' if label=='E' else 'E′',31,'vision','middle',650)
    body+=line(1110,168,1110,239,'vision')+line(1110,239,103,239,'vision')+arrow(103,239,103,317,'vision')
    body+=t(580,460,'Attention mixes rows. The MLP transforms each row. Both add a residual update.',28,'ink','middle')
    add('vit-canonical-block','Inside each block: mix, transform, keep the residual',body,
        'Repeat this structure 12 times, with different learned weights in each block. All 197 rows are updated in parallel.',
        'Which operation exchanges information between rows?',
        'Self-attention mixes value rows. The MLP is applied independently to each row. LayerNorm precedes each branch; residual additions preserve the current representation.',
        '<p><a href="figures/vision1/vit-canonical-block.svg">Reusable block diagram (SVG)</a>. '
        'The full model reference below expands the attention heads, classifier and label loss.</p>',height=480)

    body=box(35,70,290,['Final image CLS','h · 192 features'],'vision',h=104,size=26)
    for i,label in enumerate(['Newfoundland','Persian cat','… 998 other labels']):
        y=55+i*113
        body+=box(415,y,322,[label,'learned class vector wₖ'],'neutral',h=82,size=25)
        body+=arrow(750,y+41,815,y+41)+t(835,y+50,'hᵀwₖ + bₖ',31,'neutral')
        body+=line(337,122,376,122,'vision')+line(376,122,376,y+41,'vision')+arrow(376,y+41,404,y+41,'vision')
    body+=t(35,402,'The head stores one learned vector and one bias per label.',31)
    old=metadata['vision-fixed-class-vectors'];add(old['id'],old['title'],body,old['caption'],*old['notes'].split('\n',1))

    # Match CLIP's canonical two-tower sequence and u/v notation: encoder,
    # readout, projection, unit normalization, then a cross-modal comparison.
    body=''
    for y,color,name,h,w,v in [(46,'vision','Image encoder','hᵢ','Wᵢ','u'),(244,'language','Text encoder','hₜ','Wₜ','v')]:
        if color=='vision':body+=f.image(25,y,114,114,f.photo)
        else:body+=box(25,y,177,['“a photo','of a dog”'],'language',h=114,size=25)
        body+=arrow(211,y+57,245,y+57,color)+box(259,y+12,238,[name,'→ final readout '+h],color,h=90,size=24)
        body+=arrow(510,y+57,545,y+57,color)
        body+=box(557,y+12,245,[h+' × '+w,'learned projection'],'special',h=90,size=24)
        body+=arrow(815,y+57,850,y+57,color)+t(886,y+47,v,44,color,'middle',650)
        body+=t(887,y+86,'unit length',20,color,'middle')
        body+=line(940,y+57,990,y+57,'mixing')
    body+=line(990,103,990,301,'mixing')+arrow(990,202,1024,202,'mixing')
    body+=t(1090,191,'u · v',39,'mixing','middle',650)+t(1090,230,'cosine',23,'mixing','middle')
    body+=t(580,413,'Separate encoders. Aligned vectors. Compare in one shared space.',32,'ink','middle',650)
    old=metadata['vision-language-handoff'];add(old['id'],old['title'],body,
        'CLIP trains image and text representations to match. Learned projections and unit normalization make their vectors comparable; prompts can then describe candidate classes.',
        *old['notes'].split('\n',1),prose='<p>The next lecture uses the same branches and notation: hᵢ → Wᵢ → normalized u; hₜ → Wₜ → normalized v; then u · v. CLIP also learns a score scale for its training objective.</p>')

    add_heroes(f,b)
    add_occlusion(f,b)
    write_explorer(b)

    # Keep the twenty implementation frames, but make the optional layer clear.
    body=t(48,65,'IMPLEMENTATION LAB · OPTIONAL',31,'special',weight=650)
    body+=t(48,146,'Build the same ViT in PyTorch',44,'ink',weight=700)
    body+=box(48,213,310,['One image','B × 3 × 224 × 224'],'vision',h=96,size=27)
    body+=arrow(373,261,428,261)+box(445,213,310,['Code + tensor shapes','Linear and Conv2d'],'special',h=96,size=26)
    body+=arrow(770,261,825,261)+box(841,213,270,['The same prediction','B × 1,000'],'vision',h=96,size=26)
    body+=t(48,402,'Use this lab in class, or work through it after the conceptual lecture.',29)
    add('vision-topic-06','Implementation lab · optional',body,'All implementation details are retained. Every operation is paired with shapes, diagrams or a numerical equivalence check.',
        'What should both patch implementations compute?', 'Exactly the same affine map with the same weights and biases, up to floating-point roundoff.')
    f.frames['vision-topic-06']=f.frames['vision-topic-06'].replace('class="frame vp-frame"','class="frame vp-frame vp-topic-break vp-lab-break"',1)

    result=[]
    for n,(title,frames) in enumerate(sections,1):
        kept=[];reading=[]
        for fr in frames:
            k=key(fr)
            if k in READING_ONLY:
                # Preserve original material and its speaker notes for reading.
                item=re.sub(r'class="frame[^\"]*"','class="vp-optional-practice"',fr,count=1)
                item=re.sub(r' data-build="\d+"','',item)
                if k=='read-attention-map':
                    item='<p><a href="vision1-explorer.html">Open the nine-example interactive lab</a></p>'+item
                reading.append('<h3>'+escape(metadata[k]['title'])+'</h3>'+item)
                if k=='paper-vision-transformer':kept.append(f.frames['vit-house-architecture'])
                if k=='photo-label-loss':kept.append(f.frames['vit-canonical-block'])
                if k=='read-attention-map':kept.extend(f.frames[v] for v in ['interpret-similarity','interpret-heads','interpret-cls'])
                continue
            replacement=f.frames.get(k,fr)
            # A previous transformation can attach optional notes after a frame.
            # Preserve those companions when replacing its diagram.
            if k in f.frames and '<div class="companion"><details>' in fr:
                replacement+=fr[fr.index('<div class="companion"><details>'):]
            kept.append(replacement)
        if reading:
            kept[-1]+='<div class="companion"><details><summary>Optional reference and extra examples</summary>'+''.join(reading)+'</details></div>'
        if n==1:
            kept[0]+='<div class="companion"><p><a href="pdf/vision1.pdf">Download the lecture PDF (151 pages · 36 MB)</a> · <a href="pdf/vision1-transcript.md">Searchable transcript</a> · <a href="vision1-explorer.html">Interactive lab</a></p></div>'
        if n==6:title='Implementation lab · optional'
        result.append((title,[semantic_frame(b,fr) for fr in kept]))
    return result


def saved_row(b,block,query,head=None):
    d=np.fromfile(b['ASSETS']/f'attention-explorer/block-{block:02}.f32',dtype='<f4').astype(np.float64)
    if head is None:
        features=d[75648:].reshape(197,192)
        return features@features[query]/(np.linalg.norm(features,axis=1)*np.linalg.norm(features[query]))
    q,k=d[:37824].reshape(3,197,64),d[37824:75648].reshape(3,197,64)
    scores=q[head-1,query]@k[head-1].T/8
    e=np.exp(scores-scores.max());return e/e.sum()


def map_photo(f,x,y,size,values=None,query=None,source=None,attention=False,maximum=None):
    out=f.image(x,y,size,size,f.photo); cell=size/14
    if values is not None:
        maximum=maximum if maximum is not None else (max(values[1:]) if attention else 1)
        for j,value in enumerate(values[1:]):
            color='#178F82' if attention else ('#B98224' if value>=0 else '#3478E5')
            alpha=min(abs(float(value))/maximum,1)*.88
            out+=f'<rect x="{x+j%14*cell:.3f}" y="{y+j//14*cell:.3f}" width="{cell:.3f}" height="{cell:.3f}" fill="{color}" opacity="{alpha:.4f}"/>'
    for j in range(1,14):
        out+=f.line(x+j*cell,y,x+j*cell,y+size,'card',.4)+f.line(x,y+j*cell,x+size,y+j*cell,'card',.4)
    for index,color,dash in [(source,'#FFFFFF',''),(query,'#8B5BB5','3 2' if values is not None else '')]:
        if index:
            r,c=divmod(index-1,14)
            out+=f'<rect x="{x+c*cell}" y="{y+r*cell}" width="{cell}" height="{cell}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>'
    return out


def lab_link(t,y=448):
    return '<a href="vision1-explorer.html" target="_blank">'+t(1130,y,'Explore all nine examples ↗',22,'vision','end')+'</a>'


def add_heroes(f,b):
    t=f.t
    sim=saved_row(b,12,74)
    body=t(35,30,'Query · P74',29,'language')+t(411,30,'Feature similarity · block 12',29,'special')
    body+=map_photo(f,35,65,308,query=74)+map_photo(f,411,65,308,sim,query=74,source=82)
    body+=t(765,108,'Similar features',35,'special',weight=650)+t(765,153,'across the dog',35,'special',weight=650)
    body+=t(765,224,f'P74 ↔ P82: {sim[82]:.3f}',32,'ink')
    body+=t(765,280,'Compare the final',27)+t(765,317,'192-number patch vectors.',27)
    body+=t(35,406,'Violet outline: selected patch',23,'language')+t(411,406,'Gold: positive cosine · fixed −1 to 1',22,'special')
    body+=t(35,448,'Feature similarity ≠ attention weight',29,'ink',weight=650)+lab_link(t)
    f.add('interpret-similarity','Similar patch features can connect distant image regions',body,
        'The selected ear-side patch matches patches on the other side of the dog. This measures representation similarity, rather than which values attention mixes.',
        'Does a similar representation imply a large attention weight?',
        'No. This compares contextual patch features after block 12 using cosine similarity. P82 is a measured high-similarity patch; a correspondence is not a guaranteed segmentation.',
        '<p>Saved pretrained ViT-Tiny, block 12, query P74, selected source P82. Gold indicates positive cosine similarity; blue indicates negative similarity. Self-similarity is 1. '
        '<a href="vision1-explorer.html">Nine guided examples and free exploration</a>.</p>',height=470)
    a,c=saved_row(b,4,74,1),saved_row(b,4,74,2);maxval=max(max(a[1:]),max(c[1:]))
    body=''
    for x,title,row,source in [(35,'Query · P74',None,None),(421,'Head 1',a,60),(807,'Head 2',c,38)]:
        body+=t(x,30,title,30,'language' if row is None else 'mixing',weight=650)
        body+=map_photo(f,x,62,288,row,74,source,True,maxval)
        if row is not None:body+=t(x+144,389,f'P{source}: {row[source]:.2%}',29,'mixing','middle')
        else:body+=t(x,389,'Same query · block 4',26)
    body+=t(35,445,'Different heads gather different mixtures.',31,'ink',weight=650)+lab_link(t)
    f.add('interpret-heads','Keep the query fixed; change only the attention head',body,
        f'Teal shows source weights. Both maps use the same 0–{maxval:.2%} scale. Within each head, all 197 source weights—including CLS—sum to 100%.',
        'Why can the maps differ although the image and query location stay fixed?',
        'Each head has its own learned query and key projections. These lead to different weights on the source value rows. Head 2’s weaker peak is shown on the same color scale.',
        '<p>The two maps are calculated from saved Q and K in block 4. Each softmax includes CLS and all 196 patches; the displayed patch values are not renormalized. '
        '<a href="vision1-explorer.html">Change the head or block in the interactive lab</a>.</p>',height=470)
    cls=saved_row(b,12,0,1)
    body=f.box(35,125,250,['Query = CLS','current image state'],'vision',h=115,size=28)
    body+=f.arrow(301,182,356,182,'mixing')
    body+=t(380,30,'Attention · block 12 · head 1',28,'mixing')+map_photo(f,380, 62,322,cls,source=64,attention=True)
    body+=t(751,110,'CLS reads',36,'mixing',weight=650)+t(751,155,'patch information',36,'mixing',weight=650)
    body+=t(751,225,f'P64: {cls[64]:.2%}',36,'mixing')+t(751,275,'of this head’s weight',27)
    body+=t(380,419,f'Teal: 0 → {max(cls[1:]):.2%}',24,'mixing')
    body+=t(35,465,'One attention head is one part of the classifier.',28,'ink',weight=650)+lab_link(t,465)
    f.add('interpret-cls','CLS gathers a message for the image summary',body,
        'The query is CLS. Its weights select a mixture of all source value rows. Later operations and the classifier turn the final CLS features into class scores.',
        'Is this a complete explanation of the Newfoundland prediction?',
        f'No. It is the last block’s first attention head. P64 receives {cls[64]:.2%} of this query’s source weight; CLS itself receives {cls[0]:.2%}. Other heads, residuals, MLPs and the class head also contribute.',
        '<p>This is measured attention, not a segmentation mask or a causal importance map. '
        '<a href="vision1-explorer.html">Open all nine guided examples</a>.</p>',height=490)
    measurements={'checkpoint':'vit_tiny_patch16_224.augreg_in21k_ft_in1k',
                  'similarity':{'block':12,'query':74,'source':82,'value':float(sim[82])},
                  'head_comparison':{'block':4,'query':74,'common_scale_max':float(maxval),
                    'heads':[{'head':1,'source':60,'weight':float(a[60])},{'head':2,'source':38,'weight':float(c[38])}]},
                  'cls':{'block':12,'head':1,'query':0,'source':64,'weight':float(cls[64])}}
    (b['ASSETS']/'interpretation-heroes.json').write_text(json.dumps(measurements,indent=2)+'\n')


def add_occlusion(f,b):
    data=json.loads((b['ASSETS']/'inspection.json').read_text());t=f.t
    body=t(35,35,f'Original P(Newfoundland): {data["baseline_probability"]:.2%}',30,'vision',weight=650)
    worst=min(data['occlusion'],key=lambda item:item['target_probability'])
    for i,r in enumerate(data['occlusion']):
        x=35+i*284;y=80;size=235
        body+=f.image(x,y,size,size,f.photo)
        body+=f'<rect x="{x+r["column"]/224*size}" y="{y+r["row"]/224*size}" width="{size/2}" height="{size/2}" fill="#808080"/>'
        color='loss' if r==worst else 'ink'
        body+=t(x+size/2,350,r['region'],27,color,'middle')
        body+=t(x+size/2,395,f'{r["target_probability"]:.2%}',36,color,'middle',650)
        body+=t(x+size/2,433,f'−{100*(data["baseline_probability"]-r["target_probability"]):.2f} points',24,color,'middle')
    f.add('occlusion','Four covers, four new forward passes',body,
        'Top-right covering causes the largest drop: 16.11 percentage points. All four images still predict Newfoundland. Attention shows internal mixing; occlusion tests how a changed input changes the prediction.',
        'Which intervention changes this probability most?',
        'The top-right cover gives the largest drop among these four tests. Keep the target class, fill value and model fixed. This does not isolate a semantic object part; the next slide uses smaller covers.',
        '<p>Each trial starts from the original normalized input and independently replaces one 112×112 quadrant with zero (gray RGB). '
        '<a href="figures/vision1/inspection.json">Saved measurements</a>. Probability drops are in percentage points and should not be added.</p>',height=455)


def write_explorer(b):
    root=b['ASSETS'].parents[1];src=root/'src'
    style=CSS+(src/'vision1-inspector.css').read_text()+'''
+body{margin:0;background:#FAFAF8;color:#30343B;font-family:"Avenir Next",system-ui,sans-serif}
+main{max-width:1180px;margin:40px auto;padding:0 28px}h1{font-size:38px;line-height:1.2}
+a{color:#3478E5}button,select{font:inherit}.vix{margin-top:32px;font-size:18px}
+.vix h4{font-size:23px}.vix-tour-panel h4{font-size:29px}.vix-takeaway{font-size:25px}
+.vix-status{font-size:16px}.vix-photo{max-width:320px}.vix-tour-panel p{font-size:21px}
+.vix-tour-controls select{min-width:330px}.vix-provenance{font-size:15px;margin-top:24px}
+@media(max-width:750px){main{padding:0 16px;margin-top:22px}h1{font-size:29px}}
+'''.replace('\n+','\n')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Explore a trained ViT</title><link rel="icon" href="data:,"><style>'+style+'</style><main><a href="vision1.html?present#s09/2/0">← Back to the lecture</a><h1>Explore a trained ViT, one example at a time</h1><p>Gold compares patch features. Teal shows attention weights. Choose Next example to follow the nine-step tour.</p>'+b['EXPLORER_UI']+'</main><script>'+(src/'vision1-inspector.js').read_text()+'</script></html>'
    (root/'vision1-explorer.html').write_text(page)


def semantic_frame(b,markup):
    """Reconcile older SVG role names without touching code syntax colors.

    The detailed Q/K/V calculation sequence keeps its familiar local palette,
    explicitly labelled below every figure. Outside it, purple never means ViT.
    """
    k=key(markup)
    match=re.search(r'<div class="vp-figure[^"]*">\s*(<svg\b)',markup,re.S)
    if not match:return markup
    # SVGs may contain nested crop SVGs; find the complete root by balancing tags.
    start=match.start(1);depth=0;end=None
    for token in re.finditer(r'</?svg\b[^>]*>',markup[start:]):
        depth+= -1 if token.group().startswith('</') else 1
        if depth==0:end=start+token.end();break
    if end is None:return markup
    fig=markup[start:end]
    if k.startswith('real-heads-') or k in {'heads-visual-roles','real-cls-message'}:
        for token in ['c-q','c-k','c-v']:
            fig=fig.replace('var(--'+token+')','var(--mixing)')
        fig=fig.replace('var(--t-q)','var(--t-mixing)')
    elif k in LOCAL_ROLES:
        view=re.search(r'viewBox="0 0 1160 ([\d.]+)"',fig);height=float(view.group(1))
        legend=b['t'](35,height+31,'LOCAL ROLES',18,'ink-2',weight=650)
        for x,text,color in [(200,'Q / receiver','c-q'),(450,'K / source','c-k'),(700,'V / message','c-v')]:
            legend+=b['t'](x,height+31,text,22,color)
        fig=fig.replace(view.group(),f'viewBox="0 0 1160 {height+46:g}"',1)
        fig=fig[:-6]+legend+'</svg>'
    elif k not in {'prior-encoder-recap','vit-same-encoder','vit-token-inputs','vit-self-vs-cross','vit-house-architecture',
                    'vit-shape-trace','vit-canonical-block','vision-fixed-class-vectors','vision-language-handoff',
                    'interpret-similarity','interpret-heads','interpret-cls'}:
        # Older diagrams used purple for both parameters and image activations.
        parameter=k.startswith('position-') or k in {'real-patch-position','cls-detour','real-cls-purpose','real-cls-sequence','code-photo-tokens-cls'}
        target='special' if parameter else 'vision'
        fig=fig.replace('var(--c-q)',f'var(--{target})').replace('var(--t-q)','var(--t-special)' if parameter else 'var(--t-e)')
        if k.startswith(('real-mlp','real-block','real-cls-depth','real-cls-readout','real-classifier','real-cls-prediction')):
            fig=fig.replace('var(--c-k)','var(--special)').replace('var(--c-v)','var(--mixing)')
        if k=='photo-two-softmaxes':fig=fig.replace('var(--c-k)','var(--mixing)')
    overrides = {
        'task-image-label': {'c-a':'neutral'},
        'patch-activation-location': {'c-v':'neutral'},
        'real-patch-shared': {'c-k':'vision'},
        'position-learning': {'c-v':'neutral'},
        'real-cls-mlp': {'c-v':'neutral'},
        'real-mlp-network': {'mixing':'vision'},
        'real-classifier-softmax': {'mixing':'vision'},
        'real-cls-prediction': {'mixing':'vision'},
        'pets-original-task': {'c-v':'vision'},
        'pets-new-domain': {'c-v':'ink'},
        'pets-frozen': {'c-k':'loss'},
        'pets-fine-tune': {'c-k':'loss'},
        'pets-learning-stages': {'c-k':'loss'},
        'pets-inference': {'c-v':'vision'},
        'pets-evaluation': {'c-v':'neutral','c-a':'ink'},
        'patch-cost': {'c-a':'mixing'},
        'real-work-count': {'c-a':'mixing'},
    }
    for old,new in overrides.get(k,{}).items():
        fig=fig.replace('var(--'+old+')','var(--'+new+')')
    (b['ASSETS']/(k+'.svg')).write_text(fig)
    return markup[:start]+fig+markup[end:]
