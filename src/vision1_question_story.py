"""Question-led main lecture; the existing detailed deck remains a reference.

All numeric examples come from the checked-in ViT-Tiny traces. No model runs
or new dependencies are needed to rebuild the slides.
"""
from pathlib import Path
from html import escape
import base64
import json
import re
from vision1_focus_common import Figures
from vision1_palette import CSS

STAGES = ['Image', 'Patches', 'Projection', 'Prepare rows', 'Attention', 'MLP', 'Read summary', 'Class scores']


def make_story(b):
    f = Figures(b)
    t, box, arrow, line, rect, image, g = f.t, f.box, f.arrow, f.line, f.rect, f.image, f.g
    assets = b['ASSETS']
    photo = f.photo
    cat = 'data:image/png;base64,' + base64.b64encode((assets/'cat-model-input.png').read_bytes()).decode() if (assets/'cat-model-input.png').exists() else b['CAT']
    patch = json.loads((assets/'real-patch-path.json').read_text())
    prediction = json.loads((assets/'real-inference.json').read_text())['results'][0]['top3']
    groups = []
    records = []
    current = None

    def section(title):
        nonlocal current
        current = []
        groups.append((title, current))

    def ribbon(stage):
        drawing = ''
        for i, label in enumerate(STAGES):
            x = 8 + i*144
            active = i in stage
            color = 'vision' if active else 'ink-3'
            drawing += rect(x, 6, 132, 38, color, 't-e' if active else 'transparent', 4)
            drawing += t(x+66, 31, label, 18, color, 'middle', 650 if active else 500)
            if i < 7: drawing += arrow(x+134, 25, x+143, 25, 'ink-3')
        return '<div class="vp-pathbar">'+b['svg'](drawing, 'Route: '+', '.join(STAGES[i] for i in stage), height=54)+'</div>'

    def add(key, title, body, caption, point, stage=(), builds=False):
        key = 'story-'+key
        markup = f.add(key, title, body, caption, title, point,
                       '<p>'+escape(point)+'</p>', height=440)
        # Keep the diagram in reading mode as well as classroom mode.
        markup = markup.replace(' vp-desktop', '')
        markup = re.sub(r'<div class="vp-mobile">.*?</div>', '', markup, flags=re.S)
        if stage:
            markup = markup.replace('class="frame vp-frame"', 'class="frame vp-frame vp-with-route"', 1)
            markup = markup.replace('<div class="vp-figure">', ribbon(stage)+'<div class="vp-figure">', 1)
        current.append(markup)
        records.append(dict(id=key, title=title, caption=caption, notes=title+'\n'+point,
                            section=f's{len(groups):02}', frame=len(current)))

    def grid(x, y, side, n=14, selected=None):
        out = image(x, y, side, side, photo)
        for k in range(1,n):
            out += line(x+k*side/n,y,x+k*side/n,y+side,'card',.8)
            out += line(x,y+k*side/n,x+side,y+k*side/n,'card',.8)
        if selected:
            r,c = divmod(selected-1,n)
            out += rect(x+c*side/n,y+r*side/n,side/n,side/n,'special','transparent',0).replace('stroke-width="2"','stroke-width="3"')
        return out

    def crop(x, y, side, p=63):
        r,c = divmod(p-1,14)
        return f'<svg x="{x}" y="{y}" width="{side}" height="{side}" viewBox="{c*16} {r*16} 16 16" overflow="hidden">'+image(0,0,224,224,photo)+'</svg>'

    def rows(x,y,w,labels,color='vision',h=48,gap=14):
        return ''.join(box(x,y+i*(h+gap),w,label,color,h=h,size=26) for i,label in enumerate(labels))

    def pipeline(focus=None):
        # One canonical geometry, used unchanged for all seven stage views.
        labels=[['IMAGE'],['PATCHES'],['PATCH','PROJECTION'],['+ POSITION','+ CLS'],['ENCODER','× 12'],['FINAL','CLS'],['CLASS','HEAD']]
        shapes=['3 × 224 × 224','196 crops','196 × 192','197 × 192','197 × 192','192','1,000 logits']
        out=''
        for i,(label,shape) in enumerate(zip(labels,shapes)):
            x=10+i*164
            active = focus is None or i==focus
            color='vision' if active else 'ink-3'
            h=116 if active else 90
            y=148 if active else 161
            node=box(x,y,148,label,color,h=h,size=22)
            node+=t(x+74,305,shape,21,color,'middle')
            if focus is not None and i<focus:node='<g opacity="0.45">'+node+'</g>'
            out+=node
            if i<6:out+=arrow(x+151,206,x+161,206,'ink-3')
        if focus is not None:out+=t(10+focus*164+74,122,'HERE',24,'vision','middle',700)
        return out

    (assets/'vit-canonical-pipeline.svg').write_text(b['svg'](pipeline(), 'Canonical ViT pipeline: image, patches, projection, position and CLS, 12 encoder blocks, final CLS, class head'))
    for i in range(7):
        (assets/f'vit-canonical-pipeline-stage-{i}.svg').write_text(b['svg'](pipeline(i), f'ViT pipeline, stage {i+1}'))

    # 1. Establish the known machinery, one task and the five unanswered questions.
    section('The question we want to answer')
    recap=(assets/'prior-lecture-model-families.svg').read_text().replace('<svg ','<svg x="0" y="0" width="1160" height="440" ',1)
    add('recall','We already know how an encoder builds context',recap,
        'Today we reuse the encoder: the new input is an image.',
        'Recall the three-family summary from Transformers beyond next-token prediction. The full attention square on the left reads a complete supplied input. Today we derive how an image can supply that input.')
    body=image(40,60,275,275,photo)+arrow(330,190,400,190)+box(415,120,305,['Pretrained ViT-Tiny','ImageNet-1k'],'neutral',h=135,size=30)+arrow(734,190,788,190)
    body+=box(802,120,320,['1,000 possible classes','Newfoundland: 95.73%'],'vision',h=135,size=27)
    body+=t(175,382,'Our input photograph',28,'vision','middle')+t(765,331,'One measured forward pass',26,'ink-2','middle')
    add('task','What should this photograph produce?',body,
        'The model predicts one of 1,000 ImageNet classes; Oxford-IIIT Pet supplies the photograph.',
        'Use newfoundland_31 from Oxford-IIIT Pet throughout. The checkpoint is vit_tiny_patch16_224.augreg_in21k_ft_in1k. Its saved probability is 0.9572675228 for Newfoundland. The dataset is the image source, not this classifier’s output vocabulary. Dog/cat adaptation belongs to the optional extension.',(0,))
    body=image(45,110,210,210,photo)+arrow(272,215,332,215)+box(350,140,210,'?','neutral',h=140,size=80)+arrow(578,215,640,215)+rows(660,90,450,['vector 1','vector 2','…','vector N'])
    add('contract','An encoder expects vectors. How can an image supply them?',body,
        'We need a sequence of vectors before we can reuse the encoder.',
        'Do not introduce Q, K and V here. The problem is the input representation: pixels are a spatial RGB array, while the encoder consumes rows of features.',(0,))
    body=''
    questions=[('1','Make tokens','Pixels → vectors'),('2','Keep location','Which patch goes where?'),('3','Choose a readout','Many rows → one label'),('4','Share information','Use the known encoder'),('5','Predict a class','Summary → class scores')]
    for i,(n,label,detail) in enumerate(questions):
        y=20+i*83
        body+=box(40,y,60,n,'vision',h=57,size=30)+t(130,y+36,label,32,'ink',weight=650)+t(580,y+36,detail,28,'ink-2')
        if i<4:body+=arrow(70,y+60,70,y+78,'vision')
    add('questions','Five questions build the architecture',body,
        'Each new component will solve one problem in this chain.',
        'Ask each question before naming its architectural solution. The main lecture ends when this one image becomes a class prediction and we identify the limit of a fixed class vocabulary.')

    # 2. Tokens: nine short frames, with only one measured projection preview.
    section('How can an image become a token sequence?')
    body=grid(35,92,220,56)+t(145,352,'Pixel tokens',30,'vision','middle')+box(290,107,290,['224 × 224','50,176 tokens'],'neutral',h=115,size=31)
    body+=grid(650,92,220,14)+t(760,352,'16 × 16 patches',30,'vision','middle')+box(903,107,220,['14 × 14','196 tokens'],'vision',h=115,size=31)
    body+=t(580,417,'Attention compares token pairs: its score matrix grows as N².',29,'ink','middle')
    add('pixel-budget','Why not make every pixel a token?',body,
        'Patching is also a computational choice.',
        'Pixel tokens would give 50,176 spatial positions, each initially holding RGB. The fine grid is schematic. Patch tokens give 196 positions before CLS. Attention score counts scale quadratically with sequence length. This comparison ignores CLS to emphasize the scale, not memory implementation details.',(1,))
    body=grid(65,48,315)+arrow(408,208,514,208)+box(540,106,550,['224 ÷ 16 = 14 patches per side','14 × 14 = 196 patches'],'vision',h=170,size=34)
    add('patch-grid','Split the same photograph into 196 patches',body,
        'Each non-overlapping crop is 16 × 16 pixels, with all three colour channels.',
        'The photograph has already been resized and centre-cropped using the checkpoint transform. We now work with its 224 by 224 RGB input. Patches are ordered row by row.',(1,))
    body=grid(40,70,270,selected=63)+arrow(330,198,404,198,'vision')+crop(420,110,170)+t(505,327,'P63',28,'vision','middle')+arrow(610,198,687,198)+box(711,109,403,['16 × 16 × 3','768 input numbers'],'vision',h=180,size=38)
    add('patch-768','What is inside one patch?',body,
        'One patch contains 768 pixel values; its position in the image is a separate fact.',
        'Patch P63 lies at zero-based grid row 4, column 6. The crop illustration is magnified. Before projection, use the checkpoint normalization: RGB to [0,1], subtract 0.5 and divide by 0.5 per channel.',(1,))
    body=''
    for i,(name,color) in enumerate([('R','loss'),('G','mixing'),('B','vision')]):
        x=35+i*377
        body+=box(x,75,320,[name+' channel','16 × 16 values'],color,h=106,size=30)+arrow(x+160,195,x+160,254,color)
        body+=box(x,270,320,'256 numbers',color,h=62,size=29)
    body+=t(580,400,'xᵢ = [ all R pixels | all G pixels | all B pixels ] ∈ ℝ⁷⁶⁸',31,'vision','middle')
    add('flatten','Flatten one channel at a time: R, then G, then B',body,
        'Flattening only rearranges the values; it does not learn features.',
        'Read pixels row by row within each channel, then concatenate R, G and B. This is channel-major ordering, matching PyTorch Unfold and flattened Conv2d kernels. Pixel-major ordering is also valid if the weight columns are permuted to match.',(1,))
    body=box(30,140,230,['xᵢ','768 values'],'vision',h=100,size=32)+arrow(275,190,354,190)+box(370,117,385,['Shared learned projection','W_E: 768 × 192','b_E: 192'],'special',h=145,size=28)+arrow(770,190,851,190)+box(870,140,258,['cᵢ','192 features'],'vision',h=100,size=32)
    body+=t(580,367,'cᵢ = xᵢ W_E + b_E',44,'ink','middle')
    add('projection','How do 768 pixel values become 192 features?',body,
        'A learned linear projection computes the patch representation.',
        'Use row-vector notation: x_i is 1 by 768, W_E is 768 by 192, and b_E has 192 entries. This affine map has no following activation in the patch layer. PyTorch stores the Linear weight transposed, as 192 by 768.',(2,))
    body=''
    for i,p in enumerate([1,63,196]):
        y=35+i*133
        body+=crop(45,y,88,p)+box(170,y+14,255,f'x{p}: 768 values','vision',h=60,size=27)
        body+=arrow(441,y+44,496,y+44)+box(512,y+5,290,'same W_E, b_E','special',h=78,size=30)+arrow(817,y+44,868,y+44)+box(888,y+14,234,f'c{p}: 192','vision',h=60,size=27)
    add('shared-projection','Should every patch get a different projection?',body,
        'The same W_E and b_E are reused at all 196 locations.',
        'Each patch supplies different inputs to one shared projection. There are not 196 separately learned pixel-to-feature layers. This sharing makes the next question unavoidable: where does location enter?',(2,))
    body=rows(70,58,300,['x₁: 768','x₂: 768','…','x₁₉₆: 768'])+arrow(395,193,492,193)+box(510,126,230,['W_E, b_E','shared'],'special',h=130,size=29)+arrow(760,193,837,193)+rows(852,58,255,['c₁: 192','c₂: 192','…','c₁₉₆: 192'])
    body+=t(215,388,'X: 196 × 768',33,'vision','middle')+t(980,388,'C: 196 × 192',33,'vision','middle')
    add('stack','Stack the patches into one feature matrix',body,
        'Projection changes the feature width; it preserves the number of patch rows.',
        'X contains all flattened patches. C = X W_E + b_E broadcasts the same bias across every row. The batch axis is omitted in the main lecture.',(2,))
    values=', '.join(f'{v:.3f}'.replace('-','−') for v in patch['content'][:3])
    body=crop(65,90,190)+t(160,332,'Measured P63',29,'vision','middle')+arrow(277,187,365,187)+box(385,128,310,['768 inputs','trained projection'],'special',h=115,size=29)+arrow(710,187,782,187)+box(801,118,323,['192 output features','c₆₃'],'vision',h=133,size=30)
    body+=t(760,352,'['+values+', …]',36,'vision','middle')
    add('measured-patch','What does the trained projection actually return?',body,
        'P63 becomes a measured 192-number feature row.',
        'These first three coordinates are read from real-patch-path.json for the saved checkpoint. Dots omit 189 coordinates. This is the content representation before positional embedding; no semantic meaning is assigned to individual coordinates.',(2,))
    add('pipeline-patches','We have solved the image-to-token problem',pipeline(2),
        'The image now supplies 196 rows of 192 learned features.',
        'Use this canonical pipeline throughout. The early stages fade once completed; the current stage is enlarged and highlighted. Next ask how these shared patch features retain their locations.',(2,))

    # 3. Position supplies location, without reopening optimization.
    section('How does the model know where a patch came from?')
    body=grid(70,68,290,selected=63)+arrow(391,208,479,208)+box(505,147,280,'content c₆₃','vision',h=95,size=35)+t(970,192,'Where?',45,'special','middle')+t(970,246,'row 5, column 7',27,'special','middle')
    add('where','The projection is shared. Where does location enter?',body,
        'Content describes the crop; a positional vector supplies its place in the image.',
        'P63 is row 5, column 7 using one-based classroom coordinates. Shared projection alone contains no explicit position index. Image content may incidentally suggest position, but learned positional embeddings provide the architectural location signal.',(3,))
    body=box(65,113,315,['content cᵢ','WHAT'],'vision',h=130,size=37)+t(438,192,'+',58,'ink','middle')+box(495,113,315,['position pᵢ','WHERE'],'special',h=130,size=37)+arrow(831,180,888,180)+box(903,132,207,'eᵢ','vision',h=92,size=46)
    body+=t(580,359,'eᵢ = cᵢ + pᵢ     ·     192 features',39,'ink','middle')
    add('content-position','Add WHAT and WHERE',body,
        'Adding position preserves the 196 × 192 shape.',
        'Content vectors are blue and learned positional vectors amber. We add corresponding coordinates, not concatenate them. The vectors remain 192 features wide. The CLS position is added when CLS is included.',(3,))
    body=rows(45,55,320,['P1 content','P63 content','P196 content'])+t(401,163,'+',48,'ink','middle')+rows(440,55,290,['position 1','position 63','position 196'],'special')+arrow(748,137,806,137)+rows(825,55,288,['P1 + location','P63 + location','P196 + location'])
    body+=t(580,349,'Learned table: one 192-number row per token position',30,'special','middle')
    add('position-table','Which positional vector goes with each row?',body,
        'The same learned position table is reused for every image on this grid.',
        'During pretraining the table is optimized with the model. At inference its parameters stay fixed. The chosen checkpoint stores 197 rows including the separate CLS slot. No SGD arithmetic is needed here.',(3,))

    # 4. Reuse the known CLS idea, distinguishing stored start from computed state.
    section('We have 196 rows. Which one should we read?')
    body=rows(70,48,310,['patch 1','patch 63','…','patch 196'])+arrow(400,180,490,180)+box(510,118,250,'Which readout?','neutral',h=118,size=31)+arrow(780,180,851,180)+box(870,118,230,'one image label','vision',h=118,size=28)
    body+=t(580,398,'Mean pooling is another valid readout; this checkpoint was trained with CLS.',27,'ink','middle')
    add('readout-question','Many patch representations, one image label',body,
        'Reuse the CLS readout we already know from text.',
        'Averaging final patch features is a valid alternative when the model is trained for that readout. We keep the actual checkpoint’s CLS design, rather than substituting a different architecture.',(3,))
    body=''
    for y,label,tokens,color in [(65,'TEXT',['CLS','sentence tokens'],'language'),(245,'IMAGE',['CLS','patch tokens'],'vision')]:
        body+=t(35,y-21,label,28,color,weight=650)+box(35,y,105,tokens[0],'special',h=70,size=26)+box(154,y,254,tokens[1],color,h=70,size=27)
        body+=arrow(422,y+35,479,y+35)+box(495,y,213,'encoder','mixing',h=70,size=30)+arrow(721,y+35,767,y+35)+box(783,y,198,'final CLS',color,h=70,size=27)+arrow(991,y+35,1020,y+35)+t(1075,y+44,'class',28,'ink','middle')
    add('cls-analogy','Same CLS idea, different input tokens',body,
        'The image supplies patch tokens where the text model supplied word tokens.',
        'The starting CLS is a learned special parameter in amber; the final CLS is an input-dependent representation, coloured by modality. This is a direct reminder, not a new derivation of CLS.',(3,))
    body=box(395,32,370,['One learned CLS start','192 stored numbers'],'special',h=104,size=31)
    body+=image(75,220,155,155,photo)+image(885,220,155,155,cat)
    body+=arrow(463,151,336,256,'special')+arrow(697,151,807,256,'special')
    body+=box(247,267,270,'same CLS start','special',h=70,size=29)+box(579,267,270,'same CLS start','special',h=70,size=29)
    add('cls-start','Every image starts with the same learned CLS vector',body,
        'The stored CLS parameter is shared across photographs.',
        'Both images use exactly the same trained model and starting CLS parameter, with the same CLS positional vector. Their patch inputs differ. The cat illustrates input dependence, not a new two-class task.',(3,))
    body=''
    for y,uri,label in [(37,photo,'dog image summary'),(235,cat,'cat image summary')]:
        body+=image(35,y,140,140,uri)+box(206,y+35,216,'same CLS start','special',h=70,size=26)+arrow(437,y+70,484,y+70)+box(500,y+19,242,['same encoder','+ these patch rows'],'mixing',h=104,size=25)+arrow(756,y+70,808,y+70)+box(825,y+30,296,label,'vision',h=80,size=27)
    add('cls-dependent','After reading the image, CLS becomes image-dependent',body,
        'Shared parameters can produce different representations for different inputs.',
        'Arrows depict the computation, not numerical equality. The dog and cat receive distinct contextual CLS states from the same model. Measured two-image traces remain in the reference deck; no new probabilities are invented.',(3,))
    body=''
    for j,x in enumerate([35,452,870]):
        body+=box(x,85,235,['CLS','state '+str(j)],'vision',h=83,size=30)+box(x,252,235,['196 patch rows','state '+str(j)],'vision',h=83,size=28)
    for x in [278,695]:
        body+=arrow(x,126,x+154,126,'mixing')+arrow(x,293,x+154,293,'mixing')+arrow(x,158,x+154,266,'mixing')+arrow(x,265,x+154,159,'mixing')
    body+=t(347,51,'block',26,'mixing','middle')+t(764,51,'next block',26,'mixing','middle')
    add('cls-reads','CLS reads the current patch states at every block',body,
        'CLS and patch rows can exchange information in both directions.',
        'Each block computes all updates from its incoming states, in parallel. The next block reads the updated CLS and patch states. Arrows show allowed dependencies, including patch-to-CLS and CLS-to-patch, not guaranteed large weights.',(3,))
    body=pipeline(3)+t(580,390,'[ CLS ; P1 ; P2 ; … ; P196 ]  +  position  →  197 × 192',30,'vision','middle')
    add('prepared','We now have the token sequence the encoder needs',body,
        '196 projected patches plus one CLS row, each with 192 features.',
        'The checkpoint prepends CLS and adds the positional table to all 197 rows. There is now one image sequence. Detailed Q/K/V comes only after this input has been constructed.',(3,))

    # 5. Visual Q/K/V intuition, the full attention mask, then the familiar block.
    section('How do the patches exchange information?')
    body=rows(40,58,275,['CLS','P1','…','P196'])+arrow(333,190,425,190)+box(447,115,323,['The encoder','we already know'],'mixing',h=143,size=35)+arrow(790,190,862,190)+rows(880,58,242,['CLS updated','P1 updated','…','P196 updated'])
    add('reuse-encoder','From here, reuse the encoder we already know',body,
        'The new work was turning an image into tokens.',
        'All following operations act on feature rows: normalization, self-attention, residual additions and the row-wise MLP. They do not operate on cropped pictures displayed as icons.',(4,))
    body=t(35,40,'TRANSLATION · cross-attention',29,'neutral',weight=650)
    body+=box(35,70,400,['target-prefix states → Q'],'language',h=80,size=28)+box(613,70,510,['source-encoder states → K, V'],'language',h=80,size=28)
    body+=line(35,195,1123,195)+t(35,248,'ViT · self-attention',29,'neutral',weight=650)
    body+=box(35,278,530,['CLS + patch states','one image sequence'],'vision',h=100,size=30)+arrow(585,328,650,328,'mixing')+box(672,278,450,'Q, K and V','mixing',h=100,size=39)
    add('self-cross','Where do Q, K and V come from?',body,
        'ViT has one stream; translation cross-attention had two.',
        'In ViT, three learned projections read the same normalized image sequence. In translation cross-attention, queries come from target decoder states and keys/values from source encoder states. Translation also contains self-attention; the comparison concerns its cross-attention operation.',(4,))
    body=grid(35,82,270,selected=74)+t(170,391,'Receiver: P74',29,'c-q','middle')
    body+=arrow(325,214,396,214,'c-q')+box(415,174,246,['current patch state','192 features'],'vision',h=88,size=27)
    for y,label,meaning,color in [(36,'QUERY · Q','Context I seek','c-q'),(173,'KEY · K','What can match','c-k'),(310,'VALUE · V','Information I send','c-v')]:
        body+=arrow(675,216,752,y+43,color)+box(771,y,345,[label,meaning],color,h=87,size=24)
    add('qkv-roles','One patch representation has three jobs',body,
        'Every token makes a query, a key and a value through three learned projections.',
        'Recall the same three roles from text. In this head q_i = LN(e_i) W_Q, k_i = LN(e_i) W_K and v_i = LN(e_i) W_V (with the checkpoint biases). Each is 64 features wide. These are learned numeric vectors; the questions are an analogy for their roles. P74 can receive information using its query and send information using its key and value.',(4,))
    body=t(580,32,'Illustrative questions — the model uses vectors, not sentences',27,'ink-2','middle')
    examples=[(74,'A patch on the dog','Which other regions help interpret this texture?','Other fur / face regions'),(1,'A background patch','Which other regions provide scene context?','Branches / bright background'),(0,'CLS','Which image features help classify the whole image?','Useful parts across the photograph')]
    for i,(p,label,question,sources) in enumerate(examples):
        y=70+i*119
        body+=(crop(35,y,80,p) if p else box(35,y,80,'CLS','special',h=80,size=23))
        body+=t(143,y+23,label,27,'c-q',weight=650)+t(143,y+62,question,26,'ink')
        body+=t(1107,y+98,sources,24,'c-k','end')
    add('query-examples','What might different queries try to gather?',body,
        'A different receiver can ask for different context from the same image.',
        'These are hypothetical interpretations, not decoded meanings of the trained vectors. A key exposes features that can match a query; its corresponding value supplies a projected feature vector from that same source state. Do not promise that an ear always attends to another ear or that a head has a fixed semantic role. Next show the actual mechanics and a measured map.',(4,))
    body=box(32,37,275,['P74 query q₇₄','64 features'],'c-q',h=87,size=29)
    body+=crop(440,30,98,60)+t(488,164,'Source P60',27,'vision','middle')
    body+=box(620,36,239,['key k₆₀','64 features'],'c-k',h=85,size=29)+box(900,36,225,['value v₆₀','64 features'],'c-v',h=85,size=29)
    body+=line(169,140,169,265,'c-q')+arrow(169,265,388,265,'c-q')+arrow(550,82,604,82,'c-k')+arrow(740,137,740,201,'c-k')
    body+=box(403,215,455,['q₇₄ · k₆₀ / √64','one score → row softmax'],'mixing',h=100,size=29)+arrow(880,267,1012,267,'mixing')
    body+=arrow(1012,135,1012,239,'c-v')+arrow(1012,275,1012,311,'c-v')+box(899,322,229,'a₇₄,₆₀ × v₆₀','c-v',h=65,size=29)
    body+=t(35,419,'Q and K set a scalar weight. V supplies the vector being weighted.',30,'ink')
    add('key-value-pair','A source supplies both a key and a value',body,
        'The weight for P60 scales all 64 coordinates of P60’s value vector.',
        'The pictured source provides both k_60 and v_60, using different projections of its current normalized state. q_74 dot k_60 divided by 8 is one score. Its attention weight comes from softmax against every source score in row P74, including CLS. The weight is a scalar; the value and weighted contribution are 64-feature vectors. This diagram shows roles, not invented measured weights.',(4,))
    body=t(580,30,'Toy example · one receiver · three sources · two value features',27,'ink-2','middle')
    for i,(label,weight,value,product) in enumerate([('source A','0.6','[2, 0]','[1.2, 0]'),('source B','0.3','[0, 1]','[0, 0.3]'),('source C','0.1','[1, 1]','[0.1, 0.1]')]):
        y=65+i*82
        rowbody=box(35,y,175,label,'vision',h=57,size=25)+box(232,y,120,weight,'mixing',h=57,size=29)+t(383,y+40,'×',33,'ink','middle')+box(414,y,175,value,'c-v',h=57,size=29)+arrow(609,y+28,665,y+28,'mixing')+box(688,y,228,product,'c-v',h=57,size=29)
        body+=rowbody if i==0 else g(rowbody,i)
    body+=g(box(355,337,534,['Add the contributions: m = [1.3, 0.4]'],'c-v',h=68,size=30)+t(580,435,'This message goes to the receiver whose query chose the weights.',27,'c-q','middle'),3)
    add('value-mixture','Several source values form one receiver’s message',body,
        'Next, next, next: weight each source vector, then add the contributions.',
        'These chosen toy weights sum to one; they are not from the dog checkpoint. Reveal the second source, third source, then their sum: 0.6[2,0] + 0.3[0,1] + 0.1[1,1] = [1.3,0.4]. The real head sums 197 source contributions with 64 features each. It produces one message per receiving query. The multihead output projection converts the joined message into the update added to that receiver’s original row.',(4,))
    body=''
    for x,title,causal in [(40,'TEXT DECODER · causal',True),(640,'ViT ENCODER · full',False)]:
        body+=t(x+220,30,title,29,'neutral','middle',650)+t(x+245,80,'source keys →',24,'c-k','middle')
        body+=t(x+43,213,'queries',21,'c-q','middle')+t(x+43,247,'↓',27,'c-q','middle')
        for r in range(7):
            for c in range(7):body+=rect(x+103+c*38,103+r*32,35,29,'line','card' if causal and c>r else 't-mixing',0)
        body+=t(x+230,376,'Earlier + current tokens' if causal else 'CLS + every patch token',27,'ink','middle')
    body+=t(580,430,'Filled = allowed. ViT sees the complete image: no causal mask.',30,'mixing','middle')
    add('full-attention','ViT uses full attention, not causal attention',body,
        'Rows are receiving queries; columns are source keys. Allowed does not mean equally weighted.',
        'The small matrices show permitted query–key pairs, not learned weights. In a next-token decoder, future source positions are masked; a ViT encoder receives the complete image, so all 197 by 197 pairs are allowed, including CLS and self-pairs. Actual scores and row-softmax weights are generally asymmetric because Q and K differ. Patch order in the sequence does not impose a temporal or reading direction.',(4,))
    body=box(35,42,270,['CLS query','one receiver'],'vision',h=94,size=30)+arrow(320,89,419,89,'mixing')+box(438,42,304,['compare with 197 keys','197 scores'],'mixing',h=94,size=28)+arrow(758,89,833,89)+box(851,42,271,['row softmax','197 weights'],'mixing',h=94,size=29)
    body+=arrow(987,152,987,254,'mixing')+box(818,269,304,['weight the 197 values','sum their contributions'],'mixing',h=104,size=27)+arrow(800,321,620,321,'mixing')+box(285,269,315,['one CLS message','64 features in this head'],'vision',h=104,size=28)
    body+=g(t(35,425,'Other query rows receive their own messages in parallel.',28,'ink'),1)
    add('cls-message','How does one CLS query collect one message?',body,
        'The message belongs to the query’s row: here, CLS.',
        'This is a conceptual recap of one head. All 197 weighted value rows contribute to the 64-feature message. Then heads are joined, projected to 192 features, and added to the original CLS row. Detailed source-by-source arithmetic is in the optional attention appendix.',(4,))
    body=box(30,160,225,['same sequence','197 × 192'],'vision',h=105,size=29)
    for i in range(3):
        y=25+i*132
        body+=arrow(272,211,384,y+42,'mixing')+box(403,y,255,['Head '+str(i+1),'197 × 64'],'mixing',h=84,size=29)+arrow(675,y+42,755,211,'mixing')
    body+=box(773,118,349,['concatenate messages','197 × 192','then output projection'],'mixing',h=183,size=28)
    add('multihead','Three heads form three views of the same sequence',body,
        'Join each row’s three 64-feature messages, then project them back to 192 features.',
        'All heads run in parallel on the same normalized input, with separate learned projections. Different messages are allowed; fixed meanings such as “ear head” are not assigned. Concatenation extends the feature axis, not the row axis.',(4,))
    from vision1_visual_refinement import saved_row, map_photo
    measured=saved_row(b,4,74,1)
    body=t(188,31,'Query: P74',29,'c-q','middle')+map_photo(f,35,65,290,query=74)
    body+=arrow(347,202,401,202,'mixing')+t(577,31,'Measured source weights',29,'mixing','middle')+map_photo(f,432,65,290,measured,74,60,True)
    body+=crop(818,73,105,60)+t(967,119,'P60',32,'vision')+t(818,230,f'Weight: {measured[60]:.2%}',37,'mixing')+t(818,279,'P60’s value contributes',25,'ink')+t(818,316,'to P74’s message.',25,'ink')
    body+=t(35,408,'Trained ViT · block 4 · head 1',26,'ink-2')+t(718,408,f'Teal: 0 → {max(measured[1:]):.2%}',25,'mixing','end')
    body+='<a href="vision1-explorer.html">'+t(1120,429,'Explore the nine examples ↗',23,'vision','end')+'</a>'
    add('measured-attention','Where does the trained query look?',body,
        'Fix one query and inspect its source weights; other heads and blocks can gather different mixtures.',
        f'The maps use saved Q/K from trained ViT-Tiny block 4, head 1, query P74. P60 receives {measured[60]:.6%} and CLS receives {measured[0]:.6%}. All 197 weights sum to one; the displayed 196 patch weights are not renormalized. Teal intensity is scaled to this map’s maximum. The violet outline marks the query and white marks source P60. An attention map visualizes mixing, not a segmentation or a complete explanation of the class prediction.',(4,))
    def plus(x,y):return f'<circle cx="{x}" cy="{y}" r="24" fill="var(--card)" stroke="var(--mixing)" stroke-width="2"/>'+t(x,y+10,'+',34,'mixing','middle')
    body=box(25,150,205,['P74 input','192 features'],'vision',h=105,size=28)+arrow(243,202,277,202,'vision')
    body+=box(290,150,230,['Attention','gather messages'],'mixing',h=105,size=28)+arrow(533,202,567,202,'mixing')
    body+=box(580,150,230,['Project message','192 features'],'mixing',h=105,size=27)+arrow(823,202,852,202,'mixing')+plus(880,202)+arrow(909,202,938,202,'vision')
    body+=box(951,150,187,['Updated P74','192 features'],'vision',h=105,size=27)
    body+=line(127,150,127,65,'vision')+line(127,65,880,65,'vision')+arrow(880,65,880,172,'vision')+t(505,43,'Keep the original embedding',28,'vision','middle')
    body+=t(580,346,'original embedding + projected message = updated embedding',31,'ink','middle')+t(580,410,'Every patch gets its own update. So does CLS.',30,'vision','middle')
    add('block','Add the attention update to the original embedding',body,
        'Gather a message, project it to 192 features, then add it to the receiving token’s original embedding.',
        'Follow P74 as one receiver. Its attention heads gather weighted values from the current image sequence. Concatenate the three 64-feature messages and apply the learned output projection to obtain a 192-feature update. Add that update to P74’s incoming embedding; the message alone is not the new embedding. All other patch rows and CLS receive their own updates in parallel. Normalization is omitted from this conceptual diagram; the actual checkpoint uses U = E + MSA(LN(E)). The complete pre-LN block remains in the optional reference deck.',(4,))
    body=box(35,159,250,['After attention','192 features'],'vision',h=108,size=30)+arrow(302,213,356,213,'vision')
    body+=box(373,159,343,['MLP','192-feature update'],'mixing',h=108,size=32)+arrow(734,213,811,213,'mixing')+plus(840,213)+arrow(869,213,901,213,'vision')
    body+=box(914,159,223,['New embedding','192 features'],'vision',h=108,size=27)
    body+=line(160,159,160,65,'vision')+line(160,65,840,65,'vision')+arrow(840,65,840,183,'vision')+t(500,43,'Keep the embedding after attention',28,'vision','middle')
    body+=t(580,347,'The MLP processes each token separately.',32,'mixing','middle')+t(580,410,'This new embedding is ready for the next block.',30,'vision','middle')
    add('mlp','The MLP adds one more update to each embedding',body,
        'Attention gathers context from other rows; the MLP then processes each row’s features.',
        'The MLP receives the attention-updated representation and computes another 192-feature update. Add it to that same attention-updated representation, not to the original input from before attention. The same MLP parameters are used independently for every patch and CLS. This completes one block. Normalization and the internal 192 → 768 → 192 layers with GELU are omitted from this conceptual figure; the exact formula is E_next = U + MLP(LN(U)). Implementation details remain in the reference deck.',(5,))
    body=t(580,38,'Each block: attention update, then MLP update.',31,'mixing','middle')
    for i in range(12):
        row,col=divmod(i,6);x=35+col*185;y=100+row*156
        body+=box(x,y,153,'Block '+str(i+1),'mixing',h=96,size=30)
        if col<5:body+=arrow(x+158,y+48,x+177,y+48,'mixing')
    body+=line(1112,148,1140,148,'mixing')+line(1140,148,1140,228,'mixing')+line(1140,228,21,228,'mixing')+line(21,228,21,304,'mixing')+arrow(21,304,33,304,'mixing')
    body+=t(580,412,'197 token embeddings · still 192 features each',31,'vision','middle')
    add('depth','Every block updates the patches and CLS again',body,
        'The next block starts from these new embeddings. After block 12, we read CLS.',
        'All 197 rows continue through the stack: 196 patch embeddings and one CLS embedding. Each block reads the states produced by the preceding block and performs attention and MLP updates. Blocks have their own learned weights. The feature width remains 192; what each row represents changes. After the last block, the checkpoint’s final normalization and CLS readout give one image representation for classification.',(4,5))
    body=t(292,43,'LEARNED PARAMETERS',29,'special','middle',650)+t(865,43,'COMPUTED FOR EACH IMAGE',29,'vision','middle',650)
    left=['Patch projection W, b','Position table + starting CLS','Transformer weights, including W_Q/K/V','Classifier weights + biases']
    right=['Patch embeddings','Q, K, V and attention weights','Contextual patch states + final CLS','Logits + probabilities']
    for i,(a,c) in enumerate(zip(left,right)):
        y=78+i*79
        body+=box(35,y,535,a,'special',h=61,size=25)+box(598,y,525,c,'vision',h=61,size=26)
    add('stored-computed','What is stored, and what changes with the image?',body,
        'W_Q, W_K and W_V are learned; attention weights are computed separately for each input.',
        'For inference the learned parameters remain fixed. Activations are recomputed for each image. During training gradients update the parameters. Avoid conflating the learned key/query projections with the input-dependent attention matrix.',(4,))

    # 6. Finish the prediction, then test its sensitivity to a changed input.
    section('What does the model predict, and what changes its answer?')
    body=t(260,36,'After all 12 blocks',30,'ink','middle')+box(80,72,360,['Updated CLS','192 features'],'vision',h=93,size=31)
    body+=box(80,200,360,'Updated patch embeddings','ink-3',h=75,size=26)+t(260,317,'196 rows still exist',26,'ink-2','middle')
    body+=arrow(459,118,610,118,'vision')+box(630,64,470,['Read final CLS','one image embedding','192 features'],'vision',h=132,size=29)
    body+=arrow(865,212,865,244,'vision')+box(676,264,378,'Next: the class head','special',h=83,size=32)
    body+=t(580,420,'The classifier uses the image summary carried by CLS.',30,'ink','middle')
    add('final-cls','After the blocks, read the updated CLS embedding',body,
        'One final CLS embedding summarizes the image for the classifier.',
        'This is a readout, not another update step. The checkpoint applies its final LayerNorm before selecting the CLS row, yielding a 192-feature image embedding. The diagram omits normalization to emphasize the readout. The final 196 patch embeddings also exist, but the trained ImageNet class head reads CLS. The next slide converts its features into 1,000 class scores.',(6,))
    body=box(35,130,226,['h_CLS','192 features'],'vision',h=121,size=31)+arrow(276,190,330,190)+box(347,130,337,['Linear(192, 1000)','1,000 logits'],'special',h=121,size=30)+arrow(700,190,750,190)+box(768,130,352,['softmax across classes','1,000 probabilities'],'neutral',h=121,size=28)
    body+=t(580,362,'One logit and one probability for every ImageNet class',30,'ink','middle')
    add('head','How do 192 features score 1,000 classes?',body,
        'The class head learns how each image feature contributes to each class score.',
        'The head is a learned affine map. Softmax here normalizes across class logits; it is distinct from attention softmax across source tokens. Prediction uses the highest class score. The main lecture does not repeat softmax arithmetic.',(7,))
    body=image(40,48,280,280,photo)
    labels=['Newfoundland','Tibetan mastiff','briard']
    for i,(label,entry) in enumerate(zip(labels,prediction)):
        y=82+i*112;p=100*entry['probability']
        body+=t(378,y,label,31,'ink')+t(1107,y,f'{p:.2f}%',36,'vision','end',650)+rect(378,y+17,730,23,'line','card',3)+rect(378,y+17,730*p/100,23,'vision','t-e',3)
    add('prediction','What does this checkpoint predict for our photograph?',body,
        'Newfoundland receives 95.73% of the probability across the 1,000 classes.',
        'These are saved measured predictions from real-inference.json: Newfoundland 95.726752%, Tibetan mastiff 1.625724%, briard 0.674269%. They are not performance or accuracy estimates. The remaining probability belongs to the other ImageNet classes.',(7,))

    inspection=json.loads((assets/'inspection.json').read_text())
    baseline=inspection['baseline_probability']
    def covered_photo(x,y,size,record):
        return image(x,y,size,size,photo)+f'<rect x="{x+record["column"]/224*size}" y="{y+record["row"]/224*size}" width="{record["size"]/224*size}" height="{record["size"]/224*size}" fill="#808080"/>'
    first=inspection['occlusion'][0]
    body=image(75,52,285,285,photo)+t(217,395,'Original: 95.73%',32,'vision','middle')+arrow(389,193,467,193,'loss')
    body+=covered_photo(496,52,285,first)+t(638,395,'Cover one quarter',31,'loss','middle')
    body+=t(978,120,'Same model.',32,'ink','middle')+t(978,180,'Changed pixels.',32,'ink','middle')+t(978,272,'Same label?',35,'loss','middle')+t(978,327,'Same probability?',30,'loss','middle')
    add('cover-question','What happens if we hide one quarter of the image?',body,
        'Predict the effect before we run the changed image through the same trained model.',
        'Introduce a controlled input intervention. The target is always the ImageNet Newfoundland class. Compare the original photograph with a fresh copy whose top-left 112 by 112 region is filled with mid-gray. This is not an attention mask or token removal. The question concerns the final prediction, unlike the previous internal attention map.',(7,))
    code='''covered = x.clone()              # x: (1, 3, 224, 224), normalized
covered[:, :, :112, :112] = 0     # gray RGB; 112 × 112 pixels
model.eval()
with torch.inference_mode():
    before = model(x).softmax(-1)[0, 256]
    after = model(covered).softmax(-1)[0, 256]'''
    body=f.code(code,x=35,y=28,width=1090,size=26,spacing=39)
    body+=image(40,291,110,110,photo)+arrow(167,344,216,344)+box(233,310,224,'same ViT','mixing',h=67,size=30)+arrow(472,344,508,344)+t(525,354,f'{baseline:.2%}',31,'vision')
    body+=covered_photo(694,291,110,first)+arrow(820,344,850,344)+box(865,310,100,'ViT','mixing',h=67,size=29)+arrow(979,344,1008,344)+t(1025,354,f'{first["target_probability"]:.2%}',29,'loss')
    add('cover-code','Replace pixels, then recompute the whole forward pass',body,
        'Weights stay fixed; all patch features, attention weights and class probabilities are recomputed.',
        'The checkpoint transform normalizes each channel as (RGB − 0.5) / 0.5, so normalized zero means gray RGB 0.5, not black. Index 256 is Newfoundland. Both calls use the same frozen model in evaluation mode. Image shape, all 196 patch tokens and CLS remain; 49 patch inputs now contain gray pixels. Each test uses a fresh clone, so covers never accumulate.',(7,))
    body=t(580,36,f'Original P(Newfoundland) = {baseline:.2%}',32,'vision','middle')
    worst=min(inspection['occlusion'],key=lambda item:item['target_probability'])
    for i,item in enumerate(inspection['occlusion']):
        x=35+i*284
        body+=covered_photo(x,77,235,item)
        color='loss' if item==worst else 'ink'
        body+=t(x+117,354,item['region'],27,color,'middle')+t(x+117,396,f'{item["target_probability"]:.2%}',36,color,'middle',650)+t(x+117,434,f'−{100*(baseline-item["target_probability"]):.2f} points',25,color,'middle')
    add('cover-results','Four independent covers, four measured predictions',body,
        'Top right causes the largest drop: 16.11 points. All four images still predict Newfoundland.',
        'Choose the four quadrants before viewing results. Run each covered copy separately and measure the same target-class probability. Top-right drops most among these four tests. This shows sensitivity to a specified pixel replacement on this photograph, not unique importance of an ear, face or other semantic part. The covers add gray edges and change the input distribution. Probability drops are not additive.',(7,))
    fine=json.loads((assets/'patch-occlusion.json').read_text());largest=fine['largest_drop']
    body=covered_photo(45,63,275,largest)+t(182,33,f'Cover one patch: P{largest["patch_index"]}',27,'loss','middle')
    body+=image(403,63,275,275,photo)+t(540,33,'196 separate tests',27,'mixing','middle')
    for item in fine['trials']:
        drop=item['drop_percentage_points'];color='#D45555' if drop>=0 else '#3478E5'
        body+=f'<rect x="{403+item["column"]/224*275}" y="{63+item["row"]/224*275}" width="{275/14}" height="{275/14}" fill="{color}" opacity="{min(abs(drop)/4,1)*.9:.4f}"/>'
    body+=t(731,101,'Largest local drop',30,'ink',weight=650)+t(731,157,f'{baseline:.2%} → {largest["target_probability"]:.2%}',36,'loss')+t(731,221,f'P{largest["patch_index"]}: −{largest["drop_percentage_points"]:.2f} points',29,'loss')
    body+=t(731,289,'Red: probability falls',25,'loss')+t(731,326,'Blue: probability rises',25,'vision')+t(731,366,'Scale: −4 to +4 points',24,'ink-2')
    body+=t(580,428,'All 196 tests keep the same label. Smaller covers probe more local sensitivity.',27,'ink','middle')
    add('cover-small','Would smaller covers tell us more?',body,
        'Each coloured square records a separate forward pass with one 16 × 16 patch covered.',
        f'The model and its patch size are unchanged; only the covered area shrinks. A fresh input is used for each of 196 tests. The largest drop is P{largest["patch_index"]}, {largest["drop_percentage_points"]:.6f} points. This is a probability-change map, not attention. Small drops can occur because other regions preserve related clues; they do not prove a patch is useless. Detailed experimental protocol and all measurements remain in the reference deck.',(7,))

    # 7. Architecture tradeoffs, two summary assets, optional routes, then CLIP.
    section('What have we built, and what is still missing?')
    body=t(295,39,'CNN',35,'neutral','middle',650)+t(865,39,'ViT',35,'neutral','middle',650)+grid(160,72,260)+grid(735,72,260)
    for size in [55,115,210]:body+=rect(290-size/2,202-size/2,size,size,'mixing','transparent',0)
    for px,py in [(750,85),(979,90),(975,315),(750,314),(859,95)]:body+=line(865,202,px,py,'mixing',2)
    body+=t(290,386,'Nearby first → wider context',28,'ink','middle')+t(865,386,'Direct access to distant patches',28,'ink','middle')
    add('cnn-context','Two ways to gather image context',body,
        'CNN context grows through local layers; global ViT attention can connect distant patches within one block.',
        'The receptive fields and attention edges are schematic, not measured. Both architecture families can ultimately use whole-image context. This comparison concerns ordinary local CNNs and this global-attention ViT; it does not claim one always performs better.')
    body=''
    table=[('','CNN','ViT'),('Basic units','Local feature maps','Patch-token rows'),('Mixing rule','Shared local convolution','Content-dependent attention'),('Locality','Built into each filter','Less spatial structure built in'),('Position','Grid structure','Explicit position embeddings')]
    for i,row in enumerate(table):
        y=20+i*80
        for j,(x,w) in enumerate([(30,250),(280,395),(675,455)]):
            body+=rect(x,y,w,80,'line','t-e' if i==0 else 'card',0)
            body+=t(x+20,y+49,row[j],29 if i==0 else 25,'neutral' if j==0 else ('mixing' if j==1 else 'vision'),weight=650 if i==0 else 500)
    add('cnn-bias','Which assumptions are built into the architecture?',body,
        'Inductive bias is the structure supplied before learning begins.',
        'Convolution builds in locality and shared filters. A patch-based ViT also has biases, including patchification and shared projections, but its global mixing is content-dependent and positions are supplied explicitly. Data, training and pretrained weights affect outcomes; selection advice remains optional.')
    body=t(290,35,'CNN: reuse a local detector',29,'cnn','middle',650)+t(870,35,'ViT: choose context from content',29,'vision','middle',650)
    body+=grid(160,77,255)+grid(744,77,255,selected=74)
    for x,y in [(202,133),(309,193)]:body+=rect(x,y,46,46,'cnn','transparent',0)
    body+=arrow(248,156,304,214,'cnn')+t(287,377,'Same filter, different locations',27,'cnn','middle')
    for x,y in [(833,108),(966,140),(791,300)]:body+=arrow(799,176,x,y,'mixing')
    body+=t(870,377,'Different weights for each query',27,'mixing','middle')
    body+=t(580,432,'A built-in image prior can help with limited data; pretraining changes the comparison.',26,'ink','middle')
    add('cnn-example','What does an inductive bias buy us?',body,
        'Locality and filter sharing make useful image assumptions before the CNN sees any examples.',
        'The boxes and arrows are schematic. A learned local CNN filter is reused across locations; standard convolution mixes a fixed neighborhood with the same kernel weights, although its activations depend on the image. ViT query–key matching computes different source weights for each receiver and image, using shared projection parameters. With limited task data, useful priors or pretrained representations can help; there is no universal winner. Compare models under the actual data, accuracy and compute constraints.')
    body=''
    for i,(p,n) in enumerate([(32,7),(16,14),(8,28)]):
        x=50+i*382
        body+=grid(x,67,255,n)+t(x+128,38,f'{p} × {p} patches',30,'vision','middle')+t(x+128,383,f'{n*n} patch tokens',33,'vision','middle')
    add('patch-sizes','How much detail should one token cover?',body,
        'Smaller patches preserve a finer spatial grid and create more tokens.',
        'For a 224 by 224 image: 32-pixel patches give 49 patch tokens, 16 gives 196, and 8 gives 784. Add one CLS token for this architecture. These are architectural comparisons, not a claim that the saved checkpoint accepts all three patch sizes without adaptation.',(1,))
    body=box(45,99,460,['16 × 16 patches','196 patch tokens','197 tokens with CLS'],'vision',h=180,size=32)+arrow(530,187,621,187)+box(650,99,460,['8 × 8 patches','784 patch tokens','785 tokens with CLS'],'vision',h=180,size=32)
    body+=t(580,349,'Patch tokens × 4  →  attention scores ≈ × 16',38,'mixing','middle')+t(580,407,'38,809 scores  →  616,225 scores per head',28,'ink-2','middle')
    add('patch-cost','What does a finer grid cost?',body,
        'The pairwise score count grows quadratically with token count.',
        'Including CLS, 197 squared is 38,809 and 785 squared is 616,225, about 15.88 times larger. The approximately 16-fold statement is exact for the patch-only counts. This concerns attention scores, not a 16-fold claim about total runtime or all model operations.',(4,))
    # Three visual stages echo the opening encoder-family recap.
    body=''
    for x,w,color,title in [(20,330,'vision','1 · Make tokens'),
                            (405,350,'mixing','2 · Build context'),
                            (810,330,'special','3 · Read CLS')]:
        body+=t(x+w/2,38,title,32,color,'middle',650)
        body+=rect(x,67,w,339,'line','card',8)
        body+=line(x+18,68,x+w-18,68,color,4)
    body+=grid(42,98,120)+arrow(167,158,187,158,'vision')
    body+=box(196,112,136,['Shared','projection'],'vision',h=87,size=25)
    body+=t(102,247,'224 × 224 RGB',22,'ink-2','middle')
    body+=arrow(264,207,264,267,'vision')+t(184,295,'+ CLS + position',28,'special','middle')
    for i,label in enumerate(['CLS','P1','P2','…']):
        body+=box(39+i*77,317,62,label,'special' if i==0 else 'vision',h=38,size=24)
    body+=t(185,387,'197 rows × 192 features',25,'vision','middle')
    body+=arrow(357,238,396,238,'vision')
    body+=t(580,105,'12 encoder blocks',28,'mixing','middle',650)
    body+=box(431,127,298,['Full attention','gather messages + add'],'mixing',h=90,size=25)
    body+=arrow(580,225,580,244,'mixing')
    body+=box(431,253,298,['MLP','update each row + add'],'mixing',h=90,size=25)
    body+=t(580,387,'197 updated rows × 192',25,'mixing','middle')
    body+=arrow(763,238,802,238,'mixing')
    body+=box(831,96,250,'Final CLS · 192','special',h=48,size=27)
    body+='<g opacity="0.42">'+box(831,156,250,'Updated patch rows','vision',h=42,size=24)+'</g>'
    body+=line(1090,120,1118,120,'special')+line(1118,120,1118,226,'special')+line(1118,226,975,226,'special')+arrow(975,226,975,254,'special')
    body+=box(831,264,288,['Linear class head','1,000 class scores'],'special',h=82,size=26)
    body+=arrow(975,351,975,367,'special')+t(975,394,'Newfoundland',28,'ink','middle',650)
    add('summary-pipeline','Once an image becomes tokens, the encoder is familiar',body,
        'Pixels become token rows. Every row gains context. Final CLS supplies the image summary for classification.',
        'Read the three panels from left to right, as in the opening model-family recap. Split the image into 196 RGB patches and apply one shared projection; prepend CLS and add position embeddings. All 197 rows pass through 12 encoder blocks. Within each block, full attention gathers messages, joins and projects the head outputs, and adds an update to each incoming row; the MLP then adds its own update to that attention-updated row. Each block has its own parameters. The shape remains 197 by 192. After final normalization, select only the CLS row for the trained 1,000-class head. The patch rows still exist. LayerNorm and the output projection are omitted from this overview; the nearby code and the detailed block slides make them explicit. The final label is the measured prediction for our photograph.')
    body=''
    items=[('Pixels','3 × 224 × 224'),('Patch projection','196 × 192'),('+ CLS + position','197 × 192'),('12 encoder blocks','197 × 192'),('Final LN → read CLS','192'),('Class head','1,000 logits')]
    for i,(label,shape) in enumerate(items):
        x=30+(i%3)*383;y=33+(i//3)*160
        body+=box(x,y,334,[label,shape],'mixing' if i==3 else 'vision',h=103,size=28)
        if i%3<2:body+=arrow(x+343,y+52,x+372,y+52,'ink-3')
    body+=line(1120,144,1120,167,'ink-3')+line(1120,167,17,167,'ink-3')+line(17,167,17,245,'ink-3')+arrow(17,245,29,245)
    body+=t(580,365,'BLOCKS PRESERVE THE SHAPE;',35,'mixing','middle',700)+t(580,413,'THEY CHANGE WHAT EACH ROW REPRESENTS.',34,'mixing','middle',700)
    add('summary-shapes','Follow the whole model through its shapes',body,
        'Batch dimension omitted; all shapes refer to one photograph.',
        'This is the conceptual climax. Projection sets the feature width; adding CLS changes 196 rows to 197. All twelve encoder blocks preserve 197 by 192. Only the readout reduces to one row and the classifier changes 192 features to 1,000 logits.')
    code='''# Project each image patch to 192 features.
x = patch_embed(image)        # (B, 196, 192)
# Add CLS at index 0 and position to every token.
x = prepend_cls(x) + pos      # (B, 197, 192)
# Update all patch embeddings and CLS in each block.
for block in blocks:
    x = block(x)              # (B, 197, 192)
# Normalize, then extract token 0: final CLS.
h = norm(x)[:, 0]             # (B, 192)
# Turn the image summary into 1,000 class scores.
logits = head(h)              # (B, 1000)'''
    body=f.code(code,x=65,y=54,width=1040,size=28,spacing=34)
    add('six-lines','The whole ViT in six lines',body,
        'B is the batch size. Token 0 is CLS; the head returns scores for the 1,000 ImageNet classes.',
        'This is readable pseudocode for the same pre-LN encoder classifier. The input image tensor has shape B by 3 by 224 by 224. patch_embed includes patch extraction, shared projection and conversion to patch rows. pos has shape 1 by 197 by 192 and broadcasts across the batch. Starting CLS is shared across the batch and prepended at token index zero. Each block updates every patch row and CLS through attention and MLP residual branches. norm(x) normalizes all final rows; [:, 0] selects CLS for every image in the batch, yielding B by 192. The linear head maps that image summary to 1,000 unnormalized class scores. No softmax is performed in these six executable lines.')
    body=t(580,45,'OPTIONAL LABS & EXTENSIONS',32,'special','middle',650)
    choices=[('Implementation lab',['Linear ↔ Conv2d','PyTorch shapes + code']),('Extensions',['Transfer + evaluation','Similarity · attention · occlusion']),('Open the boxes',['Patch arithmetic','Real attention numbers'])]
    for i,(title,lines_) in enumerate(choices):
        x=35+i*378
        route=['s06/1/0','s07/1/0','s02/1/0'][i]
        body+=f'<a href="vision1-reference.html?present#{route}">'
        body+=box(x,99,334,title,'neutral',h=76,size=27)
        for j,s in enumerate(lines_):body+=t(x+167,238+j*57,s,26,'ink-2','middle')
        body+='</a>'
    body+=t(580,412,'The conceptual model is complete; these are optional deeper dives.',28,'ink','middle')
    add('optional-routes','Choose a deeper dive when you need it',body,
        'The reference deck preserves the detailed diagrams, code, measurements and exercises.',
        'Open vision1-reference.html for the full detail. Section 6 is the implementation lab; sections 7–9 sit behind OPTIONAL EXTENSIONS. Optional reading also retains initialization, optimization and CLS-versus-pooling material. Feature similarity, attention and causal importance are different measurements.')
    current[-1]+='<div class="companion"><p><a href="vision1-reference.html?present#s06/1/0">Implementation lab</a> · <a href="vision1-reference.html?present#s07/1/0">Optional extensions</a> · <a href="vision1-reference.html?present#s02/1/0">Opening the patch-projection box</a> · <a href="vision1-reference.html?present#s03/1/0">Opening attention again with real ViT numbers</a> · <a href="vision1-explorer.html">Interactive lab</a></p></div>'
    body=box(35,149,270,['h_CLS','192 features'],'vision',h=105,size=33)+arrow(323,201,440,201)
    for i,label in enumerate(['w_Newfoundland','w_Persian cat','w_sports car','…']):
        y=26+i*98
        body+=box(461,y,323,label,'special',h=64,size=28)+arrow(800,y+32,863,y+32)+box(882,y,246,'class score','neutral',h=64,size=27)
    body+=t(580,430,'s_c = h_CLSᵀ w_c + b_c',34,'ink','middle')
    add('fixed-vocabulary','The classifier stores a learned vector per known class',body,
        'Its 1,000 output classes form a fixed vocabulary.',
        'The 192-feature CLS is compared with a learned weight vector for every ImageNet class, with one bias per class. Persian cat and sports car name actual ImageNet classes. The equation uses column-vector notation for h. The classifier cannot accept an arbitrary new class description without changing its readout or model.',(7,))
    body=box(55,85,440,'“a photo of a dog”','language',h=97,size=38)+arrow(517,133,611,133,'language')+box(634,79,177,'? ? ?','neutral',h=110,size=44)+arrow(833,133,916,133,'language')+box(940,87,175,'v_dog','language',h=93,size=39)
    body+=t(580,302,'What if a class vector came from language?',39,'ink','middle',650)+t(580,408,'NEXT: CLIP',43,'language','middle',700)
    add('clip-question','What if our class vocabulary could come from words?',body,
        'The next lecture begins with this question.',
        'End on the missing mapping. This checkpoint does not already convert arbitrary language to compatible class vectors. The next lecture explains the aligned image/text representation needed for that idea; do not teach CLIP’s training objective here.')
    assert len(records)==54, len(records)
    return groups, records


def refactor_main(b, legacy_sections, legacy_config, legacy_manifest):
    """Save the complete teaching library, then write the short main route."""
    root, src, assets = b['ROOT'], b['SRC'], b['ASSETS']
    meta = {item['id']: item for item in b['FRAMES']}
    reference_dir = src/'sections-vision1-reference'
    reference_dir.mkdir(exist_ok=True)
    ref_sections = [(title, list(frames)) for title,frames in legacy_sections]
    refs = Figures(b)
    def divider(key,title,lines,caption):
        body=''.join(b['t'](580,120+i*100,line,39,'special' if i==0 else 'neutral','middle',650 if i==0 else 500) for i,line in enumerate(lines))
        return refs.add(key,title,body,caption,title,caption,height=440).replace(' vp-desktop','')
    ref_sections[0][1].insert(0,divider('optional-library-start','Optional labs, worked examples and reference diagrams',
        ['OPTIONAL REFERENCE DECK','Open the boxes when you need the details.'],
        'The main 54-slide lecture is complete without this material.'))
    ref_sections[6][1].insert(0,divider('optional-extensions-start','Optional extensions',
        ['OPTIONAL EXTENSIONS','Transfer · evaluation · interpretation'],
        'These experiments extend the core ImageNet classification story.'))
    meta.update({item['id']:item for item in b['FRAMES']})
    # Preserve the four reference figures formerly appended to the audit PDF.
    appendix=[]
    for key in ['photo-label-loss','paper-vision-transformer','cls-parameter-origin','cls-parameter-learning']:
        item=meta[key]
        raw=(assets/(key+'.svg')).read_text()
        body=raw.replace('<svg ','<svg x="0" y="0" width="1160" height="440" ',1)
        appendix.append(refs.add('reference-'+key,item['title'],body,item['caption'],*item['notes'].split('\n',1),height=440).replace(' vp-desktop',''))
    ref_sections.append(('Optional reference diagrams',appendix))
    titles=['Optional recap: earlier intuition','Opening the patch-projection box',
            'Opening attention again with real ViT numbers','Optional whole-model walkthrough',
            'Optional CNN comparison','Implementation lab · optional',
            'Optional extensions: transfer and evaluation','Optional extensions: new-image inference',
            'Optional extensions: interpreting the model','Optional cost exercises and references',
            'Optional reference diagrams']
    css='<style>'+CSS+(src/'vision1-lesson.css').read_text()+(src/'vision1-inspector.css').read_text()+'</style>'
    meta={item['id']:item for item in b['FRAMES']}
    reference_manifest=[]
    for n,(_,frames) in enumerate(ref_sections,1):
        title=titles[n-1]
        text=f'<section id="s{n:02}" class="sec" data-title="{escape(title)}" data-lit=""><header class="sec-head"><span class="sec-num">{n:02}</span><div><h2>{escape(title)}</h2></div></header>'+''.join(frames)+'</section>'
        # PDF link/count in the old reading companion must refer to this deck.
        text=text.replace('pdf/vision1.pdf','pdf/vision1-reference.pdf').replace('149 pages','151 pages')
        text=text.replace('pdf/vision1-transcript.md','pdf/vision1-reference-transcript.md')
        if n==1:text=css+text
        (reference_dir/f'sec{n:02}.html').write_text(text)
        for j,markup in enumerate(frames,1):
            key=re.search(r'class="frame[^\"]*" id="([^\"]+)"',markup).group(1)
            reference_manifest.append(dict(meta[key],section=f's{n:02}',frame=j))
    ref_config=dict(legacy_config,title='Vision Transformer · optional labs and references',
                    subtitle='Detailed arithmetic, PyTorch implementation, transfer learning and interpretation.',
                    hook='Which part of the model would you like to open?',durationLabel='Optional material',
                    sectionDirectory='sections-vision1-reference',output='vision1-reference.html',
                    sections=[dict(id=f's{i+1:02}',title=title,lit='') for i,title in enumerate(titles)],
                    chain=[dict(section=f's{i+1:02}',label=title) for i,title in enumerate(titles)],
                    prev={'label':'Main Vision I lecture','href':'vision1.html'},
                    next={'label':'Main Vision I lecture','href':'vision1.html'})
    (src/'vision1-reference.json').write_text(json.dumps(ref_config,indent=2)+'\n')
    (assets/'reference-frame-manifest.json').write_text(json.dumps(reference_manifest,indent=2)+'\n')
    # Preserve the old route as a stable source for its numerical validation.
    (assets/'detailed-frame-manifest.json').write_text(json.dumps(legacy_manifest,indent=2)+'\n')

    sections,main_manifest=make_story(b)
    for path in b['OUT'].glob('sec[0-9][0-9].html'):
        if int(path.stem[3:])>len(sections):path.unlink()
    for n,(title,frames) in enumerate(sections,1):
        if n==1:frames[0]+='<div class="companion"><p><a href="pdf/vision1.pdf">Main lecture PDF · 55 pages</a> · <a href="pdf/vision1-transcript.md">Searchable transcript</a> · <a href="vision1-reference.html">Optional labs and references</a> · <a href="pdf/vision1-reference.pdf">Reference PDF · 151 pages</a></p></div>'
        b['section'](n,title,frames)
    config=dict(legacy_config,subtitle='Five questions turn one photograph into tokens, context and an ImageNet prediction.',
                hook='An encoder expects vectors. How can an image supply them?',
                durationLabel='54 conceptual slides · optional labs available separately',
                centralLabel='Derive the architecture',
                central=r'\text{image}\to\text{patch tokens}\to\text{encoder}\to\text{class}',
                sections=[dict(id=f's{i+1:02}',title=title,lit='') for i,(title,_) in enumerate(sections)],
                chain=[dict(section=f's{i+1:02}',label=title) for i,(title,_) in enumerate(sections)],
                objectSections={key:'s05' for key in ['e','q','k','v','a','d','ep']},
                provenance='One saved forward pass: pretrained ViT-Tiny, 1,000 ImageNet classes, Newfoundland 95.73%. Oxford-IIIT Pet supplies the photograph. Optional labs and reference examples are in a separate deck.',
                next={'label':'Next: CLIP','href':'vision3.html'},
                footer='Once the image is a token sequence, reuse the encoder. What if class vectors came from language?')
    (src/'part5.json').write_text(json.dumps(config,indent=2)+'\n')
    (assets/'frame-manifest.json').write_text(json.dumps(main_manifest,indent=2)+'\n')
    return len(main_manifest)
