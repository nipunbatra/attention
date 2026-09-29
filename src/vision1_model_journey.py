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

    body=t(25,48,'One head’s message into CLS',31,'c-v')
    body+=box(25,97,245,['197 source weights','a₀₀ … a₀,₁₉₆'],'c-k')+t(300,149,'×',38)
    body+=box(350,97,285,['197 value rows','each 64 features'],'c-v')+arrow(635,137,730,137,'c-v')
    body+=box(750,97,365,['h₀ = Σⱼ a₀ⱼvⱼ','one row · 1 × 64'],'c-v')
    body+=g(t(25,258,'Three heads: 64 + 64 + 64 = 192 features',31,'c-v'),1)
    body+=g(box(25,302,290,['join the messages','197 × 192'],'c-v')+arrow(315,342,382,342)
            +box(400,302,315,['output Linear(192,192)','197 × 192'])+arrow(715,342,782,342)
            +box(800,302,315,['add the input E','197 × 192']),2)
    add('real-cls-message','Mix the values, join the heads, and add the input',body,
        'A weighted sum produces one message per query. Join the three heads and apply the output projection. Add this attention update to E. Every row receives an update, including the summary row.',
        'How many coordinates does the joined message have?',
        'Read 64+64+64. Then distinguish joining head messages, projecting them, and adding the original row.',
        'Per head H=AV has shape (197,64). Joining three heads gives (197,192); the output projection preserves that width. '
        'The first CLS message begins '+vec('cls_head1_message')+'. After all heads, projection and the residual, the CLS row begins '+vec('cls_after_attention')+'. '
        'The script explicitly evaluates AV and checks both the attention result and residual against the checkpoint.',
        '<p>H = AV: (197 × 197) × (197 × 64) → 197 × 64 per head.</p><p>Join three heads → 197 × 192 → Linear(192,192) → add E.</p><p>All 197 rows are updated.</p>',[4])

    body=t(25,44,'After the attention residual, follow just the CLS row',29)
    body+=box(25,96,200,['current CLS','1 × 192'])+arrow(225,136,275,136)+box(290,96,205,['LayerNorm','1 × 192'])
    body+=g(arrow(495,136,545,136)+box(560,96,245,['Linear(192,768)','1 × 768'])
            +arrow(805,136,858,136)+box(875,96,245,['GELU','1 × 768'],'c-v'),1)
    body+=g(box(25,292,255,['Linear(768,192)','1 × 192'])+arrow(280,332,347,332)
            +box(365,292,325,['add current CLS','output: 1 × 192'])
            +line(1000,176,1000,236,'ink-3')+line(1000,236,152,236,'ink-3')+arrow(152,236,152,291),2)
    body+=g(t(760,310,'The same MLP processes',27,'ink-2')+t(760,355,'every row separately.',27,'ink-2'),2)
    body+=g(t(25,424,'Whole sequence: 197 × 192 → 197 × 768 → 197 × 192',29,'c-e'),2)
    add('real-cls-mlp','Transform each updated row with the MLP',body,
        'The MLP expands one row to 768 features, applies GELU, and returns to 192 features. Add that result to the row entering this branch. Attention exchanges information across rows; this MLP works on each row separately.',
        'Which operation here changes the number of rows?',
        'None does. Follow one summary row through both linear layers and GELU, then apply the same MLP to every patch row.',
        'For the complete matrix U after the attention residual, block output is U+MLP(LayerNorm(U)). '
        'This checkpoint uses Linear(192,768), GELU, Linear(768,192); its row count stays 197. The hidden width is a design choice, independent of the earlier 768 input pixel values. '
        'After block 1 the CLS row begins '+vec('cls_after_block1')+'. Both residual additions were verified against the checkpoint block.',
        '<p>For each current row: LayerNorm → Linear(192,768) → GELU → Linear(768,192) → add the original current row.</p><p>All rows: 197 × 192 → 197 × 768 → 197 × 192.</p><p>This MLP shares its weights across rows.</p>',[5])

    body=image(25,97,180,180,photo)+t(115,320,'The same dog',24,'ink-2','middle')
    body+=t(270,54,'We have finished block 1. Keep its updated feature rows.',28)
    body+=box(270,115,270,['Output of block 1','197 × 192'])
    body+=arrow(546,155,601,155,'c-q')
    body+=box(618,100,215,['Block 2','attention + MLP'],'c-q',110)
    body+=g(arrow(839,155,894,155,'c-q')+box(910,115,225,['Output of block 2','197 × 192']),1)
    body+=t(270,291,'Block 1’s output is block 2’s input.',34,'c-q')
    body+=g(t(270,356,'Block 2 makes new Q, K and V from these updated rows.',27)
            +t(270,410,'It has its own learned attention and MLP weights.',27),2)
    add('real-block-handoff','Pass block 1’s updated rows into block 2',body,
        'Block 2 receives the features produced by block 1. It computes attention and an MLP update using its own learned weights. The same 197 row positions continue forward, carrying new feature values.',
        'What is the input to block 2?',
        'Point to the arrow from block 1’s output. The current CLS and all 196 contextual patch rows continue together. Block 2 calculates its own queries, keys and values.',
        'The original image was patchified and projected once. The matrix leaving block 1 is exactly the matrix '
        'entering block 2. In this pre-LayerNorm model, block 2 normalizes that matrix before making its Q/K/V '
        'projections, then performs attention, a residual update, and its own normalized MLP branch and residual. '
        'It uses a distinct set of learned parameters from block 1. All 197 rows continue through both blocks. '
        'The photograph identifies the ongoing example; there is no second image-to-patch operation here.',
        '<img src="figures/vision1/model-input.png" alt="The same dog continues through the model" width="160">'
        '<p><strong>Block 1 output = block 2 input.</strong></p>'
        '<p>197 × 192 updated features → block 2 → 197 × 192 updated features.</p>'
        '<p>Block 2 computes new Q, K and V from its input. It has its own attention and MLP weights.</p>',[4,5])

    body=box(25,125,175,['Input rows','197 × 192'])
    for x,name in [(245,'Block 1'),(525,'Block 2'),(920,'Block 12')]:
        body+=box(x,102,215,[name,'attention + MLP','197 × 192'],'c-e',126)
    body+=arrow(200,165,234,165)+arrow(460,165,514,165)+t(824,180,'…',43,'ink-2','middle')+arrow(860,165,910,165)
    body+=t(25,302,'12 distinct blocks, connected in sequence, in one forward pass.',29,'c-e')
    body+=g(t(25,365,'Same operations and shape; each block has its own learned weights.',27)
            +t(25,423,'After block 12: read the final CLS to classify the dog.',29,'c-q'),1)
    add('real-cls-depth','Continue through all 12 blocks, then read CLS',body,
        'The updated rows pass from block 1 to block 2, then onward to block 12. Each block has its own weights. All 197 rows stay 192 features wide. After the final block, read CLS for classification.',
        'Does 12 blocks mean twelve separate predictions for this dog?',
        'Follow one forward pass through 12 distinct blocks. There is one class prediction at the end. Each arrow carries the previous block’s updated feature rows.',
        'All blocks have their own attention and MLP parameters. They do not share one set of weights across depth. '
        'Within each block the projections and MLP are shared across rows. Twelve is this checkpoint’s chosen depth; '
        'other models can use a different number of blocks. The input photograph is encoded once, the feature matrix '
        'flows through the stack, and the class head is applied after the final block and final normalization. '
        'The trace evaluates all 12 distinct blocks in order.',
        '<p>Input rows → block 1 → block 2 → … → block 12 → read CLS → classify the dog.</p>'
        '<p>One forward pass through <strong>12 distinct blocks</strong>. Each block has its own learned weights.</p>'
        '<p>Every block receives and returns 197 × 192 features. Their values change as the rows move forward.</p>',[4,5])

    body=box(25,75,280,['after block 12','197 × 192'])+arrow(305,115,366,115)+box(385,75,280,['final LayerNorm','197 × 192'])
    body+=g(arrow(665,115,726,115)+box(745,75,360,['select the CLS row','1 × 192'],'c-q'),1)
    body+=g(box(25,285,300,['CLS features',vec('cls_final')],'c-q')+arrow(325,325,396,325)
            +box(415,285,300,['nn.Linear(192,1000)','+ 1,000 biases'])+arrow(715,325,786,325)
            +box(805,285,300,['1,000 class scores','1 × 1,000']),2)
    add('real-cls-readout','Read the final CLS row and score the classes',body,
        'Normalize the final sequence, select CLS, and apply the classification layer. Its 192 features become 1,000 scores for ImageNet labels. We read this one row after it has interacted with the whole image.',
        'Why are 192 features enough to produce 1,000 class scores?',
        'The linear layer has one weighted sum and bias for each output class. Connect this to the class scores in text Part I.',
        'This pretrained model uses a linear ImageNet head with 192×1,000 weights and 1,000 biases. '
        'The opening cat/dog question was motivation; this checkpoint predicts 1,000 ImageNet labels, including dog breeds. '
        'A model trained for just cat versus dog could use Linear(192,2). Its weights would need to be trained for those labels. '
        'Final normalization is applied before selecting CLS; no new image patch is generated.',
        '<p>Final sequence (197 × 192) → LayerNorm → select CLS (1 × 192) → Linear(192,1000) → class scores (1 × 1,000).</p><p>For a trained cat/dog head, the output width could instead be 2.</p>',[6,7])

    body=image(25,72,270,270,photo)+t(160,392,'Our original photograph',24,'ink-2','middle')
    for j,item in enumerate(data['top3']):
        y=70+j*104
        label=item['label'].split(',')[0]
        body+=t(390,y,label,29)+t(760,y,f'{item["logit"]:.3f}',27,'ink-2','end')
        body+=g(rect(815,y-25,200*item['probability'],32,'c-e','t-e')
                +t(1128,y,f'{100*item["probability"]:.2f}%',24,'c-e','end'),1)
    body+=t(760,27,'score',22,'ink-2','end')+t(960,27,'probability',22,'ink-2','middle')
    body+=g(t(390,392,'Softmax over all 1,000 labels; only the top 3 are shown.',23,'ink-2'),1)
    add('real-cls-prediction','The same photograph now has a measured prediction',body,
        'Softmax converts the 1,000 scores to class probabilities. The top label is Newfoundland, at 95.73%. We have followed this photograph from pixels to that prediction. Next, use smaller numbers to calculate attention ourselves.',
        'What are the alternatives in this softmax, compared with attention’s softmax?',
        'They are image classes here, source rows inside attention. Reveal the measured probabilities and close the real-image route before switching examples.',
        'These are measured checkpoint outputs, reproduced by the explicit full-model trace and matching the previously saved inference. '
        'The 95.73% value is the model probability for this photograph, not test accuracy. The visible three bars do not exhaust the 1,000 labels.',
        mobile_rows(['Label','Logit','Probability'],[[v['label'].split(',')[0],f'{v["logit"]:.3f}',f'{v["probability"]*100:.2f}%'] for v in data['top3']])+'<p>Softmax uses all 1,000 scores. These are measured outputs on one image.</p>',[7])

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
    continuation=['real-cls-purpose','real-cls-sequence',*matrix_frames,'real-cls-message',
                  'real-cls-mlp','real-block-handoff','real-cls-depth','real-cls-readout','real-cls-prediction']
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
