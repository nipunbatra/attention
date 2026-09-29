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
    body += t(175,108,'Give the extra row the same width.',30,'c-q')
    for j,(label,index) in enumerate([('s₁','1'),('s₂','2'),('s₃','3'),('…','…'),('s₁₉₂','192')]):
        x=120+j*195
        body += rect(x,175,140,68,'c-q','t-q',5)+t(x+70,221,label,33,'c-q','middle')
        body += t(x+70,276,index,23,'ink-2','middle')
    body += g(t(35,335,'Initialize once with small random values; mark them trainable.',28)
              +t(35,391,'cls = nn.Parameter(torch.randn(1, 1, 192) * 0.02)',29,'c-q'),1)
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
    body += box(280,302,390,['192 starting CLS parameters','adjusted by the optimizer'],'c-q')
    body += arrow(400,297,400,202,'c-q')
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
        ('Stored CLS parameter s','cls_parameter','c-q'),
        ('Its learned position p₀','cls_position','c-e'),
        ('Starting input row e₀ = s + p₀','cls_input','c-q')]):
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

    body = grid(35,90,205)+t(137,340,'This dog’s pixels',24,'ink-2','middle')
    body += arrow(246,190,295,190)+box(310,149,285,['196 patch input rows','pixels → features + position'],size=22)
    body += box(310,22,285,['CLS input e₀','same learned start'],'c-q',size=24)
    body += line(601,63,629,63,'c-q')+line(629,63,629,190,'c-q')+arrow(629,190,678,190,'c-q')
    body += arrow(601,190,623,190,'c-e')
    body += box(694,149,190,['12 blocks','all 197 rows'],size=24)+arrow(890,190,927,190)
    body += box(943,149,187,['Read final CLS','192 features'],'c-q',size=23)
    body += g(t(310,302,'Starting CLS input',25,'c-q')+t(750,302,'Final CLS for this dog',25,'c-q'),1)
    body += g(t(310,347,vec('cls_input'),26,'c-q')+arrow(690,337,734,337,'c-q')
              +t(750,347,vec('cls_final'),26,'c-q'),1)
    body += g(t(310,419,'Class head → 1,000 scores → top label: Newfoundland (95.73%)',25,'c-e'),2)
    add('cls-collect','The dog’s patches turn CLS into this image’s summary',body,
        'The dog’s patch rows bring the image information. Attention lets CLS gather it, and the blocks update its features. Read the final CLS to predict Newfoundland. The stored starting parameters stay fixed during this forward pass.',
        'Which numbers change while we classify this photo?\nThe CLS activation changes through the blocks; the stored parameter does not. Point to the dog’s patch rows as the source of image-specific information.',
        'This is the same saved forward pass used elsewhere in the lecture. Initial e₀ includes the CLS position; '
        'final CLS includes all 12 blocks and the checkpoint’s final normalization. Each preview shows three of 192 coordinates. '
        'The blocks update patch rows as well as CLS. The class head then maps the final 192-coordinate summary '
        'to 1,000 ImageNet scores. The 95.73% value is the saved probability for Newfoundland on this image, not test accuracy. '
        'We are not changing the checkpoint or running new inference or training. The upcoming detailed slides explain how Q/K matching and value mixtures produce the updates.'+evidence,
        '<p>The same dog image → 196 patch input rows. Add the shared CLS input row → 197 rows through 12 blocks.</p>'
        +mobile_rows(['CLS activation','First three of 192 coordinates'],[
            ['At the input',vec('cls_input')],['After the blocks and final normalization',vec('cls_final')]])
        +'<p>Final CLS → class head → 1,000 scores → Newfoundland, 95.73% probability for this photograph.</p>'
        '<p>The activation changes with the image. The stored starting CLS parameter remains fixed during inference.</p>')
    return result
