"""Ground the CLS parameter, learning signal and activations in the dog photo."""
import base64
import json


def build_cls_story(b):
    t, g, rect, arrow, line, image, frame, mobile_rows = (b[k] for k in
        ['t', 'g', 'rect', 'arrow', 'line', 'image', 'frame', 'mobile_rows'])
    photo = 'data:image/png;base64,' + base64.b64encode(
        (b['ASSETS']/'model-input.png').read_bytes()).decode()
    data = json.loads((b['ASSETS']/'real-classifier-path.json').read_text())
    previews = data['previews']
    result = {}
    evidence = (' <a href="figures/vision1/real-classifier-path.json">Saved model values</a> · '
                '<a href="notebooks/vision/trace_real_classifier.py">How these values were checked</a>.')

    def vec(name):
        return '['+', '.join(f'{v:.3f}'.replace('-', '−') for v in previews[name])+', …]'

    def box(x, y, w, labels, color='c-e', h=82, size=25):
        out = rect(x, y, w, h, color, 't-q' if color=='c-q' else 't-e', 5)
        for j, label in enumerate(labels):
            out += t(x+w/2, y+34+j*31, label, size, color, 'middle')
        return out

    def grid(x, y, side):
        out = image(x, y, side, side, photo)
        cell = side/14
        for k in range(1,14):
            out += line(x+k*cell,y,x+k*cell,y+side,'card',.65)
            out += line(x,y+k*cell,x+side,y+k*cell,'card',.65)
        return out + rect(x+6*cell,y+4*cell,cell,cell,'c-q','transparent',0)

    def patch(x, y, side):
        return (f'<svg x="{x}" y="{y}" width="{side}" height="{side}" '
                'viewBox="96 64 16 16" overflow="hidden">'+image(0,0,224,224,photo)+'</svg>'
                +rect(x,y,side,side,'c-q','transparent',0))

    def add(key, title, body, caption, notes, prose, mobile):
        # Existing IDs retain their links while their drawings and metadata change.
        for meta in b['FRAMES']:
            if meta['id']==key:
                meta.update(title=title, caption=caption, notes=notes)
        result[key] = frame(key,title,body,caption,notes,prose,mobile)

    body = t(35,42,'The photograph still has exactly 196 patches.',30)
    body += grid(35,87,230) + t(150,356,'The same dog photo',25,'ink-2','middle')
    body += patch(315,95,80) + t(355,211,'P63',25,'c-q','middle')
    body += line(150,161,280,161,'card',6)+line(150,161,280,161,'c-e',2.5)
    body += arrow(280,161,309,135,'c-e') + arrow(401,135,440,135)
    body += box(455,95,280,['768 pixel values','Linear(768, 192)']) + arrow(741,135,788,135)
    body += box(805,95,315,['Patch content c₆₃','192 features'])
    body += g(box(455,263,280,['Create a parameter','192 trainable numbers'],'c-q',size=24)
              +arrow(741,303,788,303,'c-q')+box(805,263,315,['CLS row','192 features'],'c-q'),1)
    body += g(t(455,397,'196 image rows + 1 summary row = 197 rows',29,'c-q'),1)
    add('real-cls-purpose','Add a summary row beside the dog’s patch rows',body,
        'A patch row comes from pixels. CLS comes from a separate set of model parameters. No extra crop is cut from the photograph. Its 192 coordinates match the width of every patch row.',
        'Where are the pixels for CLS?\nThere are none. Trace the two different origins: a crop through the patch layer, and a separately stored parameter row.',
        'The figure compares the origins of the content rows, before position is added. P63 is a real 16×16 RGB crop, '
        'projected from 768 values to 192 features. CLS is a separately stored vector of 192 trainable parameters. '
        'It bypasses the patch-pixel projection. The model appends no region to the photograph: it prepends a row to the sequence of representations. '
        'As the preceding slide showed for patch rows, position is then added; CLS also has its own position vector.',
        '<p>The same dog photograph still gives <strong>196 real patches</strong>.</p>'
        '<p>P63: 16 × 16 × 3 pixels → 768 values → Linear(768,192) → 192 content features.</p>'
        '<p>CLS: a separately stored parameter → 192 coordinates. It has no pixels and no patch projection.</p>'
        '<p><strong>196 patch rows + 1 CLS row = 197 rows.</strong></p>')

    body = image(35,25,105,105,photo)
    body += t(175,58,'The model uses 192 features per row.',31)
    body += t(175,108,'Give the extra row the same width.',30,'special')
    for j,(label,index) in enumerate([('s₁','1'),('s₂','2'),('s₃','3'),('…','…'),('s₁₉₂','192')]):
        x=120+j*195
        body += rect(x,175,140,68,'special','t-special',5)+t(x+70,221,label,33,'special','middle')
        body += t(x+70,276,index,23,'ink-2','middle')
    body += g(t(35,335,'Initialize once with small random values; mark them trainable.',28)
              +t(35,391,'cls = nn.Parameter(torch.randn(1, 1, 192) * 0.02)',29,'special'),1)
    add('cls-parameter-origin','Create 192 trainable numbers for CLS',body,
        'Before training, initialize one 192-number vector. It is stored in the model, like a learned token embedding in text. The numbers do not come from this photograph. Training will adjust them.',
        'Why 192, and who supplies the initial numbers?\nThe chosen embedding width is 192. The initialization routine supplies the numbers once, before training; no image pixels are needed.',
        'The code is one illustrative initialization, not a reconstruction of the checkpoint’s original training run. '
        'nn.Parameter registers values that can receive gradients and be updated by the optimizer. '
        'The tensor shape (1,1,192) means a singleton batch axis, one CLS token and 192 features. Expand the leading singleton axis for a batch; '
        'this shares the same parameter across all images. The row width is a model choice, not the number of pixels, patches or classes. '
        'Some implementations initialize CLS differently. At inference, load the trained values instead of drawing new random numbers.',
        '<p>Choose the same width as each patch row: <strong>192 features</strong>.</p>'
        '<p>Create s = [s₁, s₂, …, s₁₉₂] once, before training.</p>'
        '<pre><code>cls = nn.Parameter(\n    torch.randn(1, 1, 192) * 0.02\n)</code></pre>'
        '<p>This illustrative code initializes a trainable parameter with small random values. It does not read pixels.</p>'
        '<p>Shape: one shared batch entry × one token × 192 features. Reuse the same starting parameter across images.</p>')

    body = t(35,38,'Training idea · imagine this is a labelled training example',27,'ink-2')
    body += image(35,80,165,165,photo)+t(118,284,'Dog photograph',24,'ink-2','middle')
    body += arrow(205,158,265,158)+box(280,116,250,['Patch rows + CLS','Transformer blocks'])
    body += arrow(536,157,587,157)+box(603,116,210,['Class scores','prediction'])
    body += arrow(819,157,870,157)+box(886,116,230,['Compare with label','loss'],h=82,size=23)
    body += box(875,280,252,['Known label','Newfoundland'],size=24)+arrow(1001,276,1001,203,'c-e')
    body += box(280,302,390,['192 starting CLS parameters','adjusted by the optimizer'],'special')
    body += arrow(400,297,400,202,'special')
    body += g(line(1120,157,1143,157,'c-a')+line(1143,157,1143,402,'c-a')
              +line(1143,402,475,402,'c-a')+arrow(475,402,475,389,'c-a')
              +t(695,386,'backward through classifier + blocks',22,'c-a'),1)
    body += g(t(35,429,'The loss trains CLS along with the other trainable model parameters.',28,'c-a'),1)
    add('cls-parameter-learning','The image label teaches the starting CLS numbers',body,
        'The label supplies a loss on the prediction. Backpropagation reaches the starting CLS vector through the classifier and blocks. The optimizer adjusts its 192 parameters, along with other trainable weights, across many labelled images.',
        'Do we need 192 target numbers for CLS?\nNo. The image’s class label provides the loss; the chain rule supplies gradients for the starting vector.',
        'This is a teaching illustration of supervised training, not a claim that this specific photo was used to train the saved checkpoint. '
        'Use a labelled Newfoundland photograph as an example training pair for the 1,000-class model. '
        'The known label enters the loss, not CLS or the forward input. Gradients travel from the loss through the class head '
        'and all Transformer blocks to the shared starting parameter. An optimizer then updates that parameter. '
        'There is no separate target vector or hand-written dog code for CLS. This diagram highlights one parameter group; '
        'other trainable parameters receive gradients too. We are describing the procedure, not running training.',
        '<p>Illustrative training pair: dog photograph with the label <strong>Newfoundland</strong>.</p>'
        '<p>Patch rows + CLS → blocks → class scores → compare with the known label → loss.</p>'
        '<p>Loss → backward through the model → CLS gradient → optimizer updates its 192 parameters.</p>'
        '<p>The label is used only in the loss. Repeat across many labelled images; other trainable parameters also learn.</p>')

    body = t(35,42,'Return to the pretrained model used for this dog.',30)
    for j,(label,name,color) in enumerate([
        ('Stored CLS parameter s','cls_parameter','special'),
        ('Its learned position p₀','cls_position','c-e'),
        ('Starting input row e₀ = s + p₀','cls_input','special')]):
        y=93+j*95
        body += t(35,y+25,label,27,color)
        body += t(615,y+25,vec(name),30,color)
        if j==1:body += t(568,y+25,'+',31)
        if j==2:body += line(600,y-17,1125,y-17,'line',2)+t(568,y+25,'≈',31)
    body += t(35,407,'First 3 of 192 coordinates · loaded from the saved model · rounded',26,'ink-2')
    add('cls-stored-start','Load the learned CLS for our dog example',body,
        'The saved model already contains the learned CLS vector and its position vector. Add them to form its starting input row. Load these same parameters for every photograph; no pixel values are needed for this row.',
        'Are these numbers freshly sampled for this photograph?\nNo. They are saved trained parameters. Add the CLS position vector, then combine this row with the dog’s patch rows.',
        'All three previews come from the existing verified checkpoint trace. Each full vector has 192 coordinates. '
        'The displayed values are rounded to three decimals; calculate with unrounded numbers. '
        'The CLS parameter and its position vector are separate learned parameter tensors. Their sum is the CLS input activation. '
        'These parameters stay fixed during inference. The activation will change as it passes through the blocks and reads this image.'+evidence,
        mobile_rows(['192-coordinate row','First three coordinates'],[
            ['Stored CLS parameter s',vec('cls_parameter')],['+ position p₀',vec('cls_position')],['= input e₀',vec('cls_input')]])
        +'<p>Saved trained parameters, reused for every image. Rounded previews; 189 coordinates are omitted.</p>')

    # A single conceptual bridge before the measured forward pass: successive
    # states of both CLS and the patch rows, with the readout defining their job.
    def state_rows(x, heading, cls_label, patch_label, revision):
        color = 'special' if revision == 0 else 'vision'
        tint = 't-special' if revision == 0 else 't-e'
        out = t(x+105,77,heading,25,'ink-2','middle')
        out += rect(x,102,210,65,color,tint,5)
        out += t(x+105,130,cls_label,23,color,'middle')
        # These marks stand for coordinates, not measured values or named parts.
        for k in range(8):
            width = 11 + (k*7+revision*11)%12
            out += rect(x+16+k*23,144,width,7,color,color,0)
        out += rect(x,203,210,84,'c-e','t-e',5)
        out += t(x+105,232,patch_label,21,'c-e','middle')
        for r in range(2):
            for k in range(8):
                width = 9 + (r*3+k*5+revision*7)%14
                out += rect(x+16+k*23,247+r*17,width,6,'c-e','c-e',0)
        return out

    body = t(35,32,'CLS and patch rows can read one another’s current features.',29)
    body += image(35,172,112,112,photo)+t(91,315,'Same photo',22,'ink-2','middle')
    body += arrow(153,244,193,244,'c-e')
    body += state_rows(205,'Input','Shared CLS start','196 patch rows',0)
    next_state = state_rows(560,'After block 1','Updated CLS','Updated patch rows',1)
    next_state += arrow(421,134,548,134,'c-q')
    next_state += arrow(421,245,548,245,'c-e')
    next_state += arrow(421,224,548,151,'c-e')
    # Both diagonal dependencies matter. The light underlay makes their
    # crossing legible without suggesting a junction between the messages.
    next_state += line(421,153,548,224,'card',7)
    next_state += arrow(421,153,548,224,'c-q')
    next_state += t(487,104,'Block 1',23,'ink-2','middle')
    body += g(next_state,1)
    final_state = state_rows(915,'After block 2','Updated CLS','Updated patch rows',2)
    final_state += arrow(776,134,903,134,'c-q')
    final_state += arrow(776,245,903,245,'c-e')
    final_state += arrow(776,224,903,151,'c-e')
    final_state += line(776,153,903,224,'card',7)
    final_state += arrow(776,153,903,224,'c-q')
    final_state += t(841,104,'Block 2',23,'ink-2','middle')
    final_state += t(660,316,'197 × 192 at each stage. All updates use that block’s input rows.',24,'ink-2','middle')
    body += g(final_state,2)
    reason = rect(35,346,1090,82,'c-a','card',5)
    reason += t(60,378,'Continue through block 12. The classifier then reads final CLS.',25,'c-a')
    reason += t(60,412,'The class loss trains this row to carry useful image information.',26,'ink')
    body += g(reason,3)
    add('cls-summary-refinement','What makes CLS an image summary?',body,
        'CLS can read all patch rows; each patch can read CLS and every patch. All updates use the rows entering that block. Block 2 reads the updated CLS and patches produced by block 1.',
        'Can patch rows read CLS too, and when do they see its updated version?\n'
        'Yes. Follow both diagonals: follow the upward arrows from patch rows to CLS and the downward arrows from CLS to patch updates. '
        'Each attention operation computes all its queries, keys and values from the same incoming rows. '
        'It does not first update CLS and then let patches read that new CLS in the same attention operation. '
        'Block 2 receives both updated CLS and updated patch rows from block 1. '
        'Continue through block 12: the classifier reads final CLS, so the class loss trains that row to be useful.',
        'Follow the same dog through the stack. At the input, CLS is the shared learned start plus its position vector. '
        'In block 1, attention brings image-dependent value messages into CLS. The block also updates the patch rows. '
        'The purple diagonal makes the reverse information path explicit: patch queries can read the incoming CLS key and value. '
        'Within one attention operation, every row’s Q, K and V is computed from the same normalized input sequence. '
        'All attention outputs are computed together; a query does not read another row’s newly computed output from that same operation. '
        'Block 2 receives all of block 1’s updated rows: its CLS and patch queries, keys and values are computed again from those representations. '
        'This continues through 12 successive blocks, each with its own learned parameters. The arrows summarize complete blocks, '
        'including attention, residual additions and the per-row MLP; their detailed computation comes later. '
        'Both diagonals show possible information flow, not equal attention weights or a guarantee of a large contribution. '
        'The horizontal arrows include the within-group dependencies: CLS can read itself, and each patch can read every patch, including itself. '
        'There is no causal mask here. All 197 incoming rows are available to every query. '
        'All 197 rows retain 192 coordinates. The marks stand for changing features; they are not measured activations or attention weights. '
        'The class head reads final CLS after final normalization. During training, the class loss sends gradients through this readout and the blocks, '
        'adjusting their weights and the shared starting CLS. There is no target summary vector or instruction assigning “fur”, “eyes” or “breed” to a coordinate or layer. '
        'A summary here means a learned representation useful for the classification objective, not a sentence describing the photograph or a copy of every pixel. '
        'The objective encourages useful features; it does not guarantee that every block increases confidence. '
        'During this saved forward pass, model parameters remain fixed and only the image-dependent activations change. '
        '<a href="https://arxiv.org/html/2010.11929v2#S3.SS1">ViT: class token, encoder and classification head</a>.',
        '<p>Same dog image → 196 patch rows. Add the shared starting CLS.</p>'
        '<ol><li><strong>Block 1:</strong> CLS can read CLS and all patches. Every patch can also read CLS and all patches. '
        'All attention outputs use the same incoming rows.</li>'
        '<li><strong>Block 2:</strong> its attention reads both the updated CLS and the updated patch rows produced by block 1. '
        'The new outputs become the next block’s inputs.</li>'
        '<li><strong>Continue through block 12:</strong> every block has its own learned parameters. All 197 rows remain 192 features wide.</li></ol>'
        '<p><strong>Why a summary?</strong> The classifier reads final CLS. The class loss trains the network to make its features useful for predicting the image’s class.</p>'
        '<p>We do not supply a target summary vector or assign a meaning to each coordinate. The diagram is conceptual; the next slide shows saved values.</p>')

    body = grid(35,25,155)+t(112,214,'This dog’s pixels',23,'ink-2','middle')
    body += arrow(196,128,295,128)+box(310,90,285,['196 patch input rows','pixels → features + position'],h=76,size=22)
    body += box(310,0,285,['CLS input e₀','same learned start'],'c-q',h=76,size=24)
    body += line(601,38,629,38,'c-q')+line(629,38,629,128,'c-q')+arrow(629,128,678,128,'c-q')
    body += arrow(601,128,623,128,'c-e')
    body += box(694,90,190,['12 blocks','all 197 rows'],h=76,size=24)+arrow(890,128,927,128)
    body += box(943,90,187,['Read final CLS','192 features'],'c-q',h=76,size=23)
    body += g(t(310,205,'Starting CLS input',25,'c-q')+t(750,205,'Final CLS for this dog',25,'c-q'),1)
    body += g(t(310,250,vec('cls_input'),26,'c-q')+arrow(690,240,734,240,'c-q')
              +t(750,250,vec('cls_final'),26,'c-q'),1)

    # Open the class head in place: representative input and output neurons.
    # The saved scores use all 192 features; the ellipses abbreviate the drawing.
    readout = json.loads((b['ASSETS']/'classifier-readout-trace.json').read_text())
    classes = readout['selected_classes']
    def neuron(x, y, label, color, radius=24):
        return (f'<circle cx="{x}" cy="{y}" r="{radius}" fill="var(--card)" '
                f'stroke="var(--{color})" stroke-width="2"/>'
                +t(x,y+7,label,20,color,'middle'))
    head = line(35,271,1130,271,'line')
    head += t(35,318,'Class head',28,'c-e')+t(35,360,'Linear',25,'c-e')
    head += t(35,398,'192 → 1,000',25,'c-e')
    head += t(350,305,'Final CLS features',25,'c-q','middle')
    head += t(650,305,'Class scores',25,'c-e','middle')
    head += '<g opacity=".45">'
    for yi in [341,408]:
        for yo in [341,408]:
            head += line(376,yi,621,yo,'c-e',1.6)
    head += '</g>'
    head += neuron(350,341,'h₁','c-q')+neuron(350,408,'h₁₉₂','c-q')
    head += t(350,379,'⋮',24,'c-q','middle')
    for y, item in zip([341,408],classes[:2]):
        head += neuron(650,y,f'{item["logit"]:.2f}','c-e',27)
    head += t(650,380,'⋮',24,'c-e','middle')
    head += t(497,329,'weights',22,'c-e','middle')
    head += t(497,438,'+ one bias per class',21,'c-e','middle')
    head += line(679,341,711,341,'c-e')+line(679,408,711,408,'c-e')
    head += line(711,341,711,408,'c-e')+arrow(711,374,745,374,'c-e')
    head += box(760,334,155,['softmax','all 1,000'],'c-v',h=76,size=23)
    head += arrow(921,374,952,374,'c-v')
    head += t(1044,354,classes[0]['label'],24,'c-v','middle')
    head += t(1044,398,f'{100*classes[0]["probability"]:.2f}%',32,'c-v','middle')
    body += g(head,2)
    add('cls-collect','The dog’s patches turn CLS into this image’s summary',body,
        'Here is the learned summary for this dog: 192 features read by the class head. Each class neuron combines all 192 features with its learned weights and bias. Softmax gives Newfoundland the highest probability.',
        'Which numbers change while we classify this photo?\nThe CLS activation changes through the blocks; the stored parameter does not. Then open the class head: each output reads all 192 features. The neurons show scores; softmax across all 1,000 scores gives the probability.',
        'This is the same saved forward pass used elsewhere in the lecture. Initial e₀ includes the CLS position; '
        'final CLS includes all 12 blocks and the checkpoint’s final normalization. Each preview shows three of 192 coordinates. '
        'The blocks update patch rows as well as CLS. The class head then maps the final 192-coordinate summary '
        'to 1,000 ImageNet scores with one affine layer, nn.Linear(192,1000). '
        'Each class neuron has 192 learned weights and one bias. The sketch shows the first and last CLS input neurons '
        'and two representative class outputs; the omitted 190 input neurons and 998 output neurons are also fully connected. '
        'The displayed scores are 15.48 for Newfoundland and 11.40 for Tibetan mastiff, rounded from the saved trace. '
        'There is no hidden layer or GELU in this checkpoint’s class head. Softmax uses all 1,000 scores, including the omitted outputs. '
        'The 95.73% value is the saved probability for Newfoundland on this image, not test accuracy. '
        'The original learned CLS parameter and classifier weights stay fixed during inference. '
        'We are not changing the checkpoint or running new inference or training. The upcoming detailed slides explain how Q/K matching and value mixtures produce the updates. '
        '<a href="figures/vision1/classifier-readout-trace.json">Saved class-head weights, scores and probabilities</a>.'+evidence,
        '<p>The same dog image → 196 patch input rows. Add the shared CLS input row → 197 rows through 12 blocks.</p>'
        +mobile_rows(['CLS activation','First three of 192 coordinates'],[
            ['At the input',vec('cls_input')],['After the blocks and final normalization',vec('cls_final')]])
        +'<p>Final CLS (192 features) → fully connected Linear(192,1000) → 1,000 class scores → softmax → class probabilities.</p>'
        +mobile_rows(['Class neuron','Measured score'],[[r['label'],f'{r["logit"]:.2f}'] for r in classes[:2]])
        +'<p>Each class neuron reads all 192 features with its own learned weights and bias. Softmax uses all 1,000 scores. '
        'Newfoundland has the highest probability: 95.73% for this photograph.</p>'
        '<p>The activation changes with the image. The stored starting CLS parameter remains fixed during inference.</p>')

    comparison = json.loads((b['ASSETS']/'cls-two-image-trace.json').read_text())
    pair = comparison['results']
    comparison_evidence = (' <a href="figures/vision1/cls-two-image-trace.json">All 192 coordinates for both photographs</a> · '
                           '<a href="notebooks/vision/trace_cls_comparison.py">Reproduce this comparison</a>.')

    def numbers(values):
        return '['+', '.join(f'{v:.3f}'.replace('-', '−') for v in values[:3])+', …]'

    body = t(35,38,'Shared learned CLS s = '+numbers(comparison['cls_parameter']),29,'special')
    body += t(35,84,'Add the same position vector before either image enters attention.',26,'ink-2')
    body += t(145,132,'Starting CLS input e₀',25,'special')
    body += t(790,132,'After attention + residual',25,'c-q')
    for j,(label,uri,row) in enumerate(zip(['Dog','Cat'],[b['PHOTO'],b['CAT']],pair)):
        y=156+j*132
        content = image(35,y,90,90,uri)+t(80,y+107,label,22,'ink-2','middle')
        content += t(145,y+50,numbers(row['cls_input']),24,'special')
        content += arrow(478,y+41,515,y+41)
        content += box(530,y,214,['Attention','same parameters'],'mixing',size=23)
        content += arrow(750,y+41,784,y+41)
        content += t(800,y+50,numbers(row['cls_after_attention1_residual']),24,'c-q')
        content += line(125,y+62,135,y+62,'c-e')+line(135,y+62,135,y+112,'c-e')
        content += line(135,y+112,637,y+112,'c-e')+arrow(637,y+112,637,y+86,'c-e')
        content += t(225,y+104,'196 '+label.lower()+' patch rows',22,'c-e')
        body += content if j==0 else g(content,1)
    body += t(35,430,'First 3 of 192 coordinates shown. Both CLS rows remain 192 numbers wide.',25,'ink-2')
    add('cls-shared-start','The same CLS start reads two different photographs',body,
        'Both images use the same starting CLS and the same model parameters. Their patch rows differ, so attention produces different updates. These are measured CLS values after the first attention update and residual addition.',
        'The starting numbers match exactly. Why do the two outputs differ?\n'
        'Point to the different photos. Their patches supply different keys and values. The starting CLS query is the same; the resulting attention message can differ.',
        'Each displayed CLS input is the same stored parameter plus the same CLS position vector. The 196 patch rows differ with the photograph. '
        'The trace checks equality of all 192 CLS input coordinates and the first-block CLS queries for all three heads. '
        'Attention compares these queries with image-dependent source keys and mixes image-dependent source values. '
        'The displayed output is E + Attention(LayerNorm(E)), at the CLS row, before the first block’s MLP. '
        'LayerNorm is included in the measured computation; its details remain in the later complete-block section. '
        'All model parameters are held fixed. Values are rounded to three decimals; dots omit 189 coordinates.'+comparison_evidence,
        '<p>Shared learned CLS parameter: '+numbers(comparison['cls_parameter'])+'. Add its shared position vector.</p>'
        +''.join('<h4>'+label+'</h4>'+mobile_rows(['CLS row','First three coordinates'],[
            ['Input',numbers(row['cls_input'])],['After first attention + residual',numbers(row['cls_after_attention1_residual'])]])
            for label,row in zip(['Dog','Cat'],pair))
        +'<p>Same 192-coordinate input and same model parameters. Different patch rows produce different updates.</p>'
        '<p>First three coordinates shown. This is the first attention update, before its MLP.</p>')

    body = t(35,43,'Continue both images through the same 12 blocks and classifier.',28)
    for j,(uri,row) in enumerate(zip([b['PHOTO'],b['CAT']],pair)):
        y=106+j*150
        content = image(35,y-10,120,120,uri)+arrow(161,y+41,199,y+41)
        content += box(215,y,410,['Final CLS · 192 features',numbers(row['cls_final_after_norm'])],'c-q',size=25)
        content += arrow(632,y+41,667,y+41)
        content += box(683,y,192,['Same class head','192 → 1,000'],size=22)+arrow(881,y+41,921,y+41)
        label = row['top_label'].split(',')[0]
        content += t(940,y+30,label,24,'c-e')
        content += t(940,y+69,f"{100*row['top_probability']:.2f}%",28,'c-e')
        body += content if j==0 else g(content,1)
    body += t(35,420,'The stored CLS parameter stays fixed. Each image produces its own final summary.',26,'ink-2')
    add('cls-two-image-readout','Read the two final summaries to classify the photographs',body,
        'After all 12 blocks and final normalization, the two CLS representations differ. The same classifier reads each 192-number summary. It predicts Newfoundland for this dog and Persian cat for this cat.',
        'Did we learn separate CLS parameters for these two photos?\n'
        'No. One shared starting parameter led to two image-dependent activations. The class head also uses the same weights for both images.',
        'The previews are measured final CLS representations after all blocks and final LayerNorm. Each has 192 coordinates, with three shown. '
        'The shared Linear(192,1000) class head outputs ImageNet scores, and softmax yields the displayed top-class probabilities. '
        'The reproducible trace matches both previously published image predictions and the earlier dog CLS trace. '
        'These are probabilities for individual photographs, not dataset accuracy. No optimizer step runs. '
        'Different inputs are not guaranteed to have distinct representations in general; this measured pair illustrates how context changes CLS.'+comparison_evidence,
        ''.join('<h4>'+label+'</h4><p>Final CLS: '+numbers(row['cls_final_after_norm'])+'</p><p>'
            +row['top_label'].split(',')[0]+f" · {100*row['top_probability']:.2f}%"+'</p>'
            for label,row in zip(['Dog','Cat'],pair))
        +'<p>Same starting parameter, same 12 blocks, same class head. Different image-dependent final summaries.</p>'
        '<p>Every summary has 192 coordinates. The class head produces 1,000 scores.</p>')
    return result
