"""Keep one architecture map and finish the real image before the worksheet."""
import base64
import json
import re
from html import escape

STAGES = ['Image', 'Patches', 'Projection', 'Prepare rows', 'Attention', 'MLP', 'Read summary', 'Class scores']


def connect_journey(b, sections):
    t,g,rect,arrow,line,image,frame,mobile_rows = (b[k] for k in
        ['t','g','rect','arrow','line','image','frame','mobile_rows'])
    data=json.loads((b['ASSETS']/'real-classifier-path.json').read_text())
    photo='data:image/png;base64,'+base64.b64encode((b['ASSETS']/'model-input.png').read_bytes()).decode()
    additions={}
    active={}
    evidence=(' <a href="figures/vision1/real-classifier-path.json">Measured full-model trace</a> · '
              '<a href="notebooks/vision/trace_real_classifier.py">Reproduce every operation</a>.')

    def vec(name):
        return '['+', '.join(f'{x:.3f}'.replace('-','−') for x in data['previews'][name][:2])+', …]'

    def box(x,y,w,labels,color='c-e',h=80):
        if isinstance(labels,str): labels=[labels]
        out=rect(x,y,w,h,color,'t-e' if color=='c-e' else 'transparent',6)
        for j,label in enumerate(labels):
            out+=t(x+w/2,y+h/2+(j-(len(labels)-1)/2)*36+9,label,25,color,'middle')
        return out

    def add(key,title,body,caption,question,point,prose,mobile,stage):
        additions[key]=frame(key,title,body,caption,question+'\n'+point,prose+evidence,mobile)
        active[key]=stage

    def map_body(focus):
        """Name each job before opening the block's internal operations."""
        out=''
        descriptions=[['one','photograph'],['small','image crops'],['pixels into','features'],
                      ['location +','summary'],['share','information'],['transform','features'],
                      ['one image','vector'],['score each','image class']]
        for i,(name,detail) in enumerate(zip(STAGES,descriptions)):
            x=8+i*144
            selected=i in focus
            color='c-e' if selected else 'ink-2'
            out+=rect(x,50,132,108,color,'t-e' if selected else 'transparent',5)
            out+=t(x+66,78,name,19,color,'middle',600)
            for j,s in enumerate(detail):out+=t(x+66,112+j*25,s,17,color,'middle')
            if i<7:out+=arrow(x+134,104,x+143,104,'ink-3')
            if selected:out+=t(x+66,32,'HERE',17,'c-e','middle',700)
        out+=line(584,173,860,173,'c-q',2)+line(584,162,584,173,'c-q')+line(860,162,860,173,'c-q')
        out+=t(722,201,'Inside one block',21,'c-q','middle')
        out+=t(1082,197,'softmax → label',18,'c-e','middle')
        out+=box(28,260,332,['First: make the input rows','pixels → patch features'])
        out+=arrow(370,300,401,300)
        out+=box(414,260,332,['Then: build context','patches share information'])
        out+=arrow(756,300,787,300)
        out+=box(800,260,332,['Finally: make a prediction','image summary → label'])
        out+=t(28,414,'The photograph stays fixed. The rows of features change as we follow the arrows.',26,'ink-2')
        return out

    map_mobile=('<ol class="vp-map-list">'+''.join('<li>'+escape(s)+'</li>' for s in STAGES)+'</ol>'
                '<p>Make patch features and prepare the rows. Inside a block, attention shares information and the MLP transforms each row.</p>'
                '<p>Read one image summary and score the classes. We will open each box as we reach it.</p>')
    add('model-journey-overview','The whole route: photograph to prediction',map_body([0,1,2]),
        'First turn pixels into patch features. Prepare those rows, let them exchange information, then read one image summary to score the classes. We will open each box as we reach it.',
        'Where do pixels become features, and where do features become class scores?',
        'Trace left to right. For now, call the extra row an image summary. Explain its name and how it works in the dedicated detour before attention.',
        'This map describes the actual ViT-Tiny checkpoint used for the dog photograph. It has D=192, three heads of width 64, 12 blocks and 1,000 ImageNet outputs. '
        'CLS means classification token. It is a learned extra input row; its final contextual representation is used to classify the image. '
        'The top route is reused on the detailed slides, with the current operation highlighted. The MLP hidden width of 768 happens to equal the patch pixel count; these are separate choices. '
        'This is a pre-LayerNorm architecture, so normalization precedes each branch. Both attention and the MLP have residual additions. '
        '<a href="https://arxiv.org/abs/2010.11929">Original ViT paper</a>.',map_mobile,[0,1,2])
    body=image(25,112,165,165,photo)+t(107,317,'Our dog photograph',22,'ink-2','middle')
    body+=arrow(200,190,236,190)+box(250,150,205,['Prepared rows','197 × 192'])
    body+=rect(493,80,420,205,'c-q','transparent',7)
    body+=t(703,119,'Transformer block 1',29,'c-q','middle')
    body+=arrow(461,190,509,190)+box(523,150,168,['Attention','share context'])
    body+=g(arrow(697,190,724,190)+box(736,150,163,['MLP','refine features']),1)
    body+=g(arrow(919,190,952,190)+box(965,150,173,['Updated rows','197 × 192']),2)
    body+=t(250,350,'Each row keeps 192 features. Its numbers are updated.',28)
    body+=t(250,407,'First, let’s follow this one block.',32,'c-q')
    add('model-journey-checkpoint','Start with one Transformer block',body,
        'Our 196 patch rows and one CLS row enter block 1. Attention shares information across rows; the MLP transforms each row. The output still has 197 rows and 192 features per row.',
        'What goes into one block, and what comes out?',
        'Keep the scope to block 1. Follow prepared rows through attention and then the MLP. Same matrix shape, updated feature values. Explain the stack only after these operations.',
        'The input E has shape 197×192, with positions already added and CLS already present. '
        'This introductory diagram groups each branch’s normalization and residual addition with its named operation; '
        'the upcoming attention and MLP slides open those branches. We next inspect Q/K/V for patch 63, then '
        'the CLS query and its message, all within block 1. After completing that block, we will pass its output '
        'to block 2 and introduce the full stack.',
        '<img src="figures/vision1/model-input.png" alt="The dog photograph whose feature rows enter block 1" width="160">'
        '<p>Prepared rows: 196 patches + one CLS → <strong>197 × 192</strong>.</p>'
        '<p>Block 1: attention shares information across rows → MLP transforms each row.</p>'
        '<p>Output: <strong>197 × 192</strong>, with updated feature values. First follow this one block.</p>',[4,5])

    body=image(25,80,235,235,photo)+t(142,361,'The same photograph',25,'ink-2','middle')
    body+=box(390,75,270,['196 patch rows','192 features each'])+arrow(660,115,760,115)
    body+=box(775,75,335,['One image label','Which row do we read?'])
    body+=g(box(390,246,270,['Add a CLS row','a summary slot'],'c-q')+arrow(660,286,760,286,'c-q')
            +box(775,246,335,['Read its final version','after all 12 blocks'],'c-q'),1)
    body+=g(t(390,403,'CLS = classification token',31,'c-q'),1)
    add('real-cls-purpose','Give the classifier a place to collect an image summary',body,
        'We need one prediction for the image. Add one learned row, called the classification token or CLS. Attention lets that row gather information from the patches; the classifier later reads its updated features.',
        'We have 196 patch representations. Where could one image summary live?',
        'Keep the image fixed. Reveal the extra row only after stating the need for a single image-level representation.',
        'CLS is one learned 192-coordinate parameter vector, shared as the starting content row across images. It is not a patch, a class label or a one-hot vector. '
        'The final CLS row becomes image-dependent because the blocks mix information from the image into it. A mean of final patch rows is another readout choice, compared during this detour.',
        '<p>196 patch rows → add one CLS row → all rows enter the blocks → read final CLS → one image prediction.</p><p>CLS means classification token. Its 192 starting coordinates are learned.</p>',[3])

    body=t(28,40,'The full sequence entering block 1',31,'ink',weight=600)
    for j,(label,content,position,result,color) in enumerate([
        ('CLS','learned token','p₀','e₀','c-q'),('P1','c₁','p₁','e₁','c-e'),('P63','c₆₃','p₆₃','e₆₃','c-e'),('P196','c₁₉₆','p₁₉₆','e₁₉₆','c-e')]):
        y=77+j*75
        row=t(28,y+35,label,27,color)+box(147,y,235,content,color,53)+t(422,y+34,'+',31)
        row+=box(470,y,150,position,color,53)+t(660,y+34,'=',31)+box(708,y,160,result,color,53)
        row+=t(910,y+34,'1 × 192',27,color)
        body+=row if j==0 else g(row,1)
    body+=g(t(28,414,'Stack: [e₀; e₁; …; e₁₉₆] → E has shape 197 × 192',31,'c-e'),2)
    add('real-cls-sequence','Add the summary row: 196 + 1 = 197',body,
        'Each patch keeps its content-plus-position row. CLS gets its own learned position too. Stack the summary and patch rows into E: 197 rows, each with 192 features. Batch size one is omitted here.',
        'Why do we now have 197 rows, while the width is still 192?',
        'Identify the extra row at the top. Position is added coordinate by coordinate. The subscript is the row’s identity, not its width.',
        'The full input is [CLS; C]+P, with 197×192 entries. We already calculated the 196 patch rows C+P_patch. Prepending CLS+p₀ gives exactly the same sequence. '
        'The trace verifies this equality against the checkpoint. Its initial CLS row begins '+vec('cls_input')+'. The ellipsis indicates omitted patch rows; all 196 are present.',
        mobile_rows(['Row','Content + position','Shape'],[['CLS','learned token + p₀','1 × 192'],['P1','c₁ + p₁','1 × 192'],['…','…','…'],['P196','c₁₉₆ + p₁₉₆','1 × 192']])+'<p>Stack all rows: E has shape 197 × 192.</p>',[3])

    from vision1_attention_matrices import build_attention_matrices
    matrix_frames = build_attention_matrices(b)
    additions.update(matrix_frames)
    active.update({key: [4] for key in matrix_frames})

    from vision1_multihead_journey import build_multihead_journey
    head_frames = build_multihead_journey(b)
    additions.update(head_frames)
    active.update({key: [4] for key in head_frames if key != 'real-heads-intro'})

    from vision1_block_journey import build_block_journey
    block_frames = build_block_journey(b)
    additions.update(block_frames)
    active.update({key: [5] if key.startswith('real-mlp-') else [4, 5] for key in block_frames})

    from vision1_classifier_journey import build_classifier_journey
    classifier_frames = build_classifier_journey(b)
    additions.update(classifier_frames)
    active.update({key: [6] if key=='real-cls-readout' else [7] for key in classifier_frames})

    # Move the position motivation to the point where position first enters.
    def key(markup): return re.search(r'class="frame[^"]*" id="([^"]+)"',markup).group(1)
    position_ids=['position-photo-layout','position-photo-content','position-photo-add']
    old_three={key(m):m for m in sections[2][1][1:]}
    second=[]
    for m in sections[1][1]:
        k=key(m)
        if k!='real-patch-position':
            second.append(m)
        if k=='vision-topic-02': second.append(additions['model-journey-overview'])
        if k=='real-patch-position':
            second.extend(old_three[x] for x in position_ids)
            second.append(m)
    second.append(additions['model-journey-checkpoint'])
    sections[1]=(sections[1][0],second)
    continuation=['real-cls-purpose','real-cls-sequence',*matrix_frames,*head_frames,
                  *block_frames,*classifier_frames]
    sections[2]=(sections[2][0],[sections[2][1][0]]+[additions[k] for k in continuation])
    # The small worksheet is now an explicitly introduced, separate calculation.
    worksheet=[m for k,m in old_three.items() if k not in position_ids]
    sections[3]=(sections[3][0],[sections[3][1][0]]+worksheet+sections[3][1][1:])

    # Reuse the same ordered SVG on the detailed slides; no numbers compete with steps.
    def ribbon(focus,label):
        drawing=''
        for i,name in enumerate(STAGES):
            x=8+i*144; selected=i in focus
            color='c-e' if selected else 'ink-3'
            drawing+=rect(x,8,132,35,color,'t-e' if selected else 'transparent',4)
            drawing+=t(x+66,31,name,17,color,'middle',700 if selected else 500)
            if i<7:drawing+=arrow(x+134,25,x+143,25)
        drawing+=line(584,49,860,49,'ink-3',1)+t(722,67,'inside one block',15,'ink-3','middle')
        return ('<div class="vp-pathbar"><p class="vp-path-label">'+escape(label)+'</p>'
                '<svg viewBox="0 0 1160 72" role="img" aria-label="Model route. Current operation: '
                +escape(', '.join(STAGES[i] for i in focus))+'">'+drawing+'</svg></div>')

    from vision1_photo_walkthrough import ORDER
    photo_stages=[[0],[1],[1],[1],[1],[1],[2],[2],[2],[2],[3],[4]]
    for n,k in enumerate(ORDER):active[k]=photo_stages[n]
    for k in position_ids:active[k]=[3]
    # Preserve the worksheet scope explicitly when returning to the shared map.
    result=[]
    for index,(title,frames) in enumerate(sections):
        out=[]
        for m in frames:
            k=key(m)
            if 'vision-topic-' in k:
                out.append(m);continue
            focus=active.get(k)
            label='Model route · highlighted operation'
            if k in ORDER:
                n=ORDER.index(k)+1
                label=f'Photo walkthrough · Step {n} of 12'
                # Step count is an eyebrow, not part of the main heading.
                old_title=re.search(r'data-title="([^"]+)"',m).group(1)
                new_title=re.sub(r'^\d+ · ','',old_title)
                m=m.replace('data-title="'+old_title+'"','data-title="'+new_title+'"',1)
                for meta in b['FRAMES']:
                    if meta['id']==k:meta['title']=new_title
            elif index in [3,4,5,6]:
                label='Four-patch worksheet' if index<6 else 'Full model · restore normalization and MLP'
                if k.startswith('qkv-') or k=='task-mask-reason':label='Visual intuition · connect the attention roles'
                focus=[4] if index==3 else [4] if index==4 else [6,7] if index==5 else [4,5]
                if k in old_three:
                    focus=[2] if any(s in k for s in ['projection','patch-matrix','empty','mean','edge']) else [3]
                if k=='s02-small':focus=[0,1]
            if focus is not None and not k.startswith('model-journey-'):
                m=m.replace('class="frame vp-frame','class="frame vp-frame vp-with-route',1)
                m=m.replace('<div class="vp-figure',ribbon(focus,label)+'<div class="vp-figure',1)
            out.append(m)
        result.append((title,out))
    return result
