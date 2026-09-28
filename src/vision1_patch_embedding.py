"""A small RGB-to-embedding calculation, followed by the real model's shapes."""
import json
import re
from html import escape


def example():
    # Pixel-major RGB: A (top-left), B, C, D. These are teaching parameters.
    import torch
    from torch import nn
    patches = torch.tensor([
        [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]],
        [[0.,0.,1.],[0.,0.,1.],[0.,1.,0.],[0.,1.,0.]],
    ], dtype=torch.float64)
    weight = torch.tensor([[1.,0.,0.]*4,
                           [0.,1.,0., 0.,1.,0., 0.,-1.,0., 0.,-1.,0.]], dtype=torch.float64)
    bias = torch.tensor([.5,-.5], dtype=torch.float64)
    proj = nn.Linear(12, 2, dtype=torch.float64)
    with torch.no_grad():
        proj.weight.copy_(weight)
        proj.bias.copy_(bias)
    X = patches.reshape(2,12)
    C = proj(X)
    expected = torch.tensor([[2.5,-.5],[.5,-2.5]],dtype=torch.float64)
    torch.testing.assert_close(C, expected)
    torch.testing.assert_close(C, X @ weight.T + bias)
    torch.testing.assert_close(proj(X[:1]), C[:1])
    torch.testing.assert_close(proj(X[1:]), C[1:])
    return {'patches':patches.tolist(), 'X':X.tolist(), 'weight':weight.tolist(),
            'W':weight.T.tolist(), 'bias':bias.tolist(), 'C':C.detach().tolist(),
            'input_shape':list(X.shape), 'output_shape':list(C.shape),
            'parameter_count':sum(p.numel() for p in proj.parameters()),
            'shared_layer_matches':True, 'activation':None}


def expand(b, sections):
    t,g,rect,arrow,line,crop,frame,mobile_rows = (b[k] for k in
        ['t','g','rect','arrow','line','crop','frame','mobile_rows'])
    data = json.loads((b["ASSETS"]/"patch-embedding-example.json").read_text()) if b.get("SLIDES_ONLY") else example()
    (b['ASSETS']/'patch-embedding-example.json').write_text(json.dumps(data,indent=2)+'\n')
    additions = {}
    old_meta = {f['id']:f for f in b['FRAMES']}
    linear_ref = '<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html">PyTorch Linear</a>'
    embedding_ref = '<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html">PyTorch Embedding</a>'
    source = ' <a href="figures/vision1/patch-embedding-example.json">All inputs, weights and verified outputs</a>.'

    def add(key,title,body,caption,question,point,prose,mobile):
        additions[key]=frame(key,title,body,caption,question+'\n'+point,prose,mobile)
        if key in old_meta:
            old_meta[key].update(title=title,caption=caption,notes=question+'\n'+point)

    def vec(values):
        return '['+', '.join(f'{v:g}' for v in values)+']'

    def box(x,y,w,label,size=30):
        return rect(x,y,w,65,'c-e','transparent')+t(x+w/2,y+43,label,size,'c-e','middle')

    def patch(x,y,size=80,which=0):
        out=''
        for i,rgb in enumerate(data['patches'][which]):
            px=x+(i%2)*size; py=y+(i//2)*size
            color='rgb('+','.join(str(int(v*255)) for v in rgb)+')'
            out+=f'<rect x="{px}" y="{py}" width="{size}" height="{size}" fill="{color}" stroke="var(--line)" stroke-width="2"/>'
            out+=t(px+size/2,py+size*.64,'ABCD'[i],28,'ink' if sum(v*w for v,w in zip(rgb,[.2126,.7152,.0722]))>.5 else 'card','middle')
        return out

    def mobile_patch(which=0):
        return '<svg viewBox="0 0 200 190" role="img" aria-label="Four RGB pixels labeled A, B, C, D in row order">'+patch(20,10,80,which)+'</svg>'

    pixels = data['patches'][0]
    body=patch(50,100,100)+t(150,350,'2 × 2 × 3',31,'c-e','middle')
    for x,label in [(395,'pixel'),(610,'R'),(780,'G'),(950,'B')]:body+=t(x,50,label,28,'ink-2')
    for i,rgb in enumerate(pixels):
        marks=t(395,115+i*67,'ABCD'[i],31)
        for x,value in zip([610,780,950],rgb):marks+=t(x,115+i*67,f'{value:g}',33,'c-e')
        body+=marks if i==0 else g(marks,i)
    body+=g(t(395,410,'4 pixels × 3 channels = 12 values',32,'c-e'),3)
    add('rgb-flatten-step-1','Read the RGB values of each pixel',body,
        'For this calculation, divide RGB values by 255. Read A, B, C, D: top-left, top-right, bottom-left, bottom-right.',
        'How many numbers does pixel A contribute?',
        'Point to the red pixel, read its RGB triple, then reveal one pixel at a time.',
        'This small color patch is an illustrative input. A, B, C and D label pixel positions within one patch, not image tokens. '
        'Each pixel contributes three channel values. We scale 8-bit values by 255 here; the real pretrained model uses its supplied channel normalization.'+source,
        mobile_patch()+mobile_rows(['Pixel','RGB / 255'],[[letter,vec(rgb)] for letter,rgb in zip('ABCD',pixels)]))

    body=patch(35,105,90)+t(125,350,'one RGB patch',28,'ink','middle')
    for i,rgb in enumerate(pixels):
        x=365+i*195
        part=t(x,115,'ABCD'[i],29,'ink-2','middle')+t(x,195,', '.join(f'{v:g}' for v in rgb),33,'c-e','middle')
        part+=line(x-72,220,x+72,220,'c-e')+t(x,275,'R, G, B',25,'ink-2','middle')
        body+=part if i==0 else g(part,i)
    body+=g(t(270,197,'[',45,'c-e')+t(1050,197,']',45,'c-e')
            +t(665,385,'x₁: one pixel row, shape (1, 12)',32,'c-e','middle'),3)
    add('rgb-flatten','Put the four RGB triples in one row',body,
        'Keep RGB together for each pixel. x₁ contains 12 values from patch 1; the subscript identifies the patch.',
        'Which three entries came from the bottom-right pixel?',
        'Follow A, B, C and D into their matching groups, then count the twelve scalar entries.',
        'We use pixel-major RGB order: R_A, G_A, B_A, R_B, G_B, B_B, and so on. '
        'x₁ is a row with shape (1,12); 1 is the number of patches shown and 12 is the number of input features. '
        'Flattening changes the arrangement of the values and has no trainable parameters. '
        'A channel-major implementation uses a corresponding permutation of the projection weights.'+source,
        mobile_rows(['Pixel group','Entries'],[[letter,vec(rgb)] for letter,rgb in zip('ABCD',pixels)])
        +'<p><strong>x₁ = '+escape(vec(data['X'][0]))+'</strong></p><p>Shape: (1, 12).</p>')

    body=t(35,55,'Text: a token ID',30,'ink-2')+t(655,55,'Image: pixel values',30,'ink-2')
    body+=box(35,110,440,'bank → its vocabulary ID')+box(655,110,440,'x₁: twelve values')
    body+=g(arrow(255,175,255,220)+box(35,220,440,'nn.Embedding(V, 4)',29)
            +arrow(875,175,875,220)+box(655,220,440,'nn.Linear(12, 2)',29),1)
    body+=g(arrow(255,285,255,330)+t(255,370,'look up 4 coordinates',31,'c-e','middle')
            +arrow(875,285,875,330)+t(875,370,'2 coordinates · no activation',27,'c-e','middle'),2)
    body+=t(35,420,'V = vocabulary size; the two models may choose different embedding widths.',23,'ink-2')
    add('s01-rows-step-1','How does this connect to text embeddings?',body,
        'Here the patch embedding is x₁W + b, with no activation afterward. Text looks up a learned row; the image layer computes one from pixels.',
        'Do we have a vocabulary ID for every possible image patch?',
        'Compare lookup on the left with computation on the right. Keep the number of input values separate from the embedding width.',
        'The Part II text toy used four embedding coordinates. Our RGB warm-up chooses two coordinates so both can be calculated by hand. '
        'These dimensions are choices for different models. The image output is a patch-content embedding, called cᵢ in the next slides. '
        'Both the text embedding table and the patch layer have trainable parameters. Their output widths do not have to equal the number of classes. '
        'nn.Linear includes a bias by default and does not apply an activation. We add position information to the content embedding afterwards. '
        +embedding_ref+' describes the lookup operation; '+linear_ref+' describes the affine map.',
        mobile_rows(['Text','Image'],[['Token ID','12 pixel values'],['nn.Embedding(V, 4)','nn.Linear(12, 2)'],
                                   ['Look up 4 coordinates','Compute 2 coordinates; no activation']])
        +'<p>The patch layer computes x₁W + b. No ReLU or GELU follows this projection.</p>')

    body=t(35,38,'Patch embedding · at the image input',29,'c-e')
    body+=box(35,75,245,'12 pixel values',27)+arrow(280,108,395,108)
    body+=box(395,75,350,'nn.Linear(12, 2)',30)+arrow(745,108,860,108)+box(860,75,265,'2 coordinates',27)
    body+=g(t(35,191,'c₁ = x₁W + b. Bias makes this affine; no ReLU or GELU follows.',28),1)
    body+=g(line(35,226,1125,226,'line')+t(35,268,'Later, after attention · the block MLP',29,'ink-2'),2)
    body+=g(box(35,303,185,'one row',27)+arrow(220,336,275,336)
            +box(275,303,260,'Linear(D, H)',29)+arrow(535,336,600,336)
            +rect(600,303,155,65,'c-v','transparent')+t(677.5,346,'GELU',30,'c-v','middle')
            +arrow(755,336,820,336)+box(820,303,275,'Linear(H, D)',29),2)
    body+=g(t(127.5,411,'D features',25,'ink-2','middle')+t(405,411,'H features',25,'ink-2','middle')
            +t(677.5,411,'H features',25,'c-v','middle')+t(957.5,411,'D features',25,'ink-2','middle'),3)
    add('patch-activation-location','Do we apply an activation after the patch layer?',body,
        'Our patch embedding uses one affine layer. The later block MLP puts GELU between two linear layers. D is the embedding width; H is the MLP’s hidden width.',
        'Which part of these two paths applies an activation function?',
        'Follow the direct pixel-to-embedding path first. Then reveal the separate block MLP and point to GELU between its two linear layers.',
        'In this lecture’s patch layer, cᵢ=xᵢW+b is the complete content projection: no activation is applied to cᵢ. '
        'With a bias, this is mathematically an affine transformation; the library calls the layer Linear. '
        'The content embedding then receives position information before the Transformer blocks. '
        'After attention in our pre-LN block, a normalized current row enters Linear(D,H), GELU, and Linear(H,D); the output participates in a residual addition. '
        'This diagram isolates the MLP branch. Our trained small model chooses D=16 and H=32, while the RGB hand calculation chooses D=2. '
        'The full Transformer also includes operations such as attention softmax and LayerNorm; GELU is the activation inside its MLP. '
        'Adding an activation to the patch projection would define a different input module. '
        +linear_ref+' · <a href="notebooks/vision/train_small_vit.py">The lecture’s executable model</a>.',
        '<p><strong>At the image input:</strong> 12 pixel values → nn.Linear(12, 2) → 2 embedding coordinates.</p>'
        '<p>c₁ = x₁W + b. This affine map has no ReLU or GELU afterward.</p>'
        '<p><strong>Later, after attention:</strong> the block MLP uses Linear(D, H) → GELU → Linear(H, D).</p>'
        '<p>Feature widths: D → H → H → D. D is the embedding width; H is the hidden width.</p>')

    body=t(35,60,'Input: x₁',28,'ink-2')+t(865,60,'Output: c₁',28,'ink-2')
    body+=box(35,110,245,'12 values')+arrow(280,143,405,143)+box(405,110,350,'nn.Linear(12, 2)',30)
    body+=g(arrow(755,143,865,143)+box(865,110,260,'2 coordinates'),1)
    body+=t(155,220,'(1, 12)',29,'c-e','middle')+g(t(995,220,'(1, 2)',29,'c-e','middle'),1)
    body+=g(t(35,325,'c₁ = x₁ W + b',41,'c-e')+t(35,410,'(1, 2) = (1, 12) × (12, 2) + (1, 2)',32),2)
    add('patch-linear-shapes','What does “projection” mean here?',body,
        'A projection forms weighted sums of the pixel values and adds a bias. This one linear layer turns 12 inputs into a 2-coordinate patch embedding.',
        'How many weighted sums do we need to produce two output coordinates?',
        'Match 12 to in_features and 2 to out_features, then follow the matrix dimensions.',
        'Here projection means a learned affine transformation, not a geometric camera projection or the whole forward pass. '
        'We write row vectors, so W has shape (12,2) and c₁=x₁W+b. The bias has two entries, shown as a (1,2) row for addition. '
        'No activation is attached to this layer. This is the patch embedding operation; the Transformer block MLP comes later. '+linear_ref+'.',
        mobile_rows(['Quantity','Shape'],[['x₁: pixel row','(1,12)'],['W: weights','(12,2)'],['b: bias row','(1,2)'],['c₁: content embedding','(1,2)']])
        +'<p><code>proj = nn.Linear(12, 2)</code></p><p>c₁ = x₁ W + b.</p>')

    input_names=[f'{pixel}.{channel}' for pixel in 'ABCD' for channel in 'RGB']

    def network(active=None, mobile=False):
        """Keep the same 12-to-2 layer visible while focusing one weighted sum."""
        ix,ox,first,step,ir,orr=(80,285,65,46,17,27) if mobile else (225,620,52,32,14,29)
        output_y=[205,435] if mobile else [145,315]
        body=t(ix,24,'12 inputs',19 if mobile else 25,'ink-2','middle')
        body+=t(ox,24,'2 outputs',19 if mobile else 25,'ink-2','middle')
        if not mobile:body+=t(422,24,'nn.Linear(12, 2)',25,'c-e','middle')
        # All 24 trainable connections stay present, including zero-valued weights.
        for out,yout in enumerate(output_y):
            for i in range(12):
                yin=first+step*i
                body+=f'<path class="patch-weight" data-input="{i}" data-output="{out}" data-weight="{data["weight"][out][i]}" d="M{ix+ir} {yin} L{ox-orr} {yout}" stroke="var(--line)" stroke-width="1.4" fill="none"/>'
        if active is not None:
            highlights=''
            for i,weight in enumerate(data['weight'][active]):
                if not weight:continue
                yin=first+step*i;yout=output_y[active]
                highlights+=line(ix+ir,yin,ox-orr,yout,'c-e',2.5)
                lx=ix+(62 if mobile else 88)
                ly=yin+(yout-yin)*(lx-ix-ir)/(ox-orr-ix-ir)
                label=f'{weight:+g}'.replace('-','−')
                highlights+=rect(lx-19,ly-24,38,24,'transparent','card',2)+t(lx,ly-5,label,19 if mobile else 23,'c-e','middle')
            body+=highlights if mobile else g(highlights,1)
        for i,value in enumerate(data['X'][0]):
            y=first+step*i
            body+=t(ix-ir-15,y+7,input_names[i],18 if mobile else 21,'ink-2','end')
            body+=f'<circle class="patch-input" cx="{ix}" cy="{y}" r="{ir}" fill="var(--card)" stroke="var(--c-e)" stroke-width="2"/>'
            body+=t(ix,y+7,f'{value:g}',21,'c-e','middle')
        for out,y in enumerate(output_y):
            color='c-e' if active in [None,out] else 'ink-3'
            body+=f'<circle class="patch-output" cx="{ox}" cy="{y}" r="{orr}" fill="var(--card)" stroke="var(--{color})" stroke-width="2.5"/>'
            body+=t(ox,y+9,'y₁' if out==0 else 'y₂',27,color,'middle')
        return body

    def mobile_network(active=None):
        return '<svg viewBox="0 0 360 610" role="img" aria-label="Twelve RGB input nodes connected to two embedding output nodes">'+network(active,True)+'</svg>'

    body=network()+t(755,95,'One patch in.',34,'c-e')+t(755,146,'Two features out.',34,'c-e')
    body+=g(t(755,230,'Each line has a weight.',27)+t(755,276,'Each output adds a bias.',27),1)
    body+=g(t(755,350,'c₁ = [y₁, y₂]',35,'c-e')+t(755,405,'24 weights + 2 biases',26,'ink-2'),2)
    add('patch-linear-weights','12 input numbers, 2 output numbers',body,
        'Each input node holds one RGB value. Both outputs read all 12 inputs. Together, y₁ and y₂ form the embedding for this one patch.',
        'How many connections enter each output node?',
        'Count the twelve actual pixel values. Follow their connections into each of the two outputs, then reveal the weights and biases.',
        'A.R means the red value of pixel A; each pixel supplies three consecutive input nodes. '
        'This drawing is exactly nn.Linear(12,2): a fully connected affine layer with 24 weights and two biases. '
        'There is no hidden layer or activation in this patch projection. The outputs are embedding coordinates, not dog/cat scores. '
        'The following slides keep the same network and highlight the nonzero weights for one output at a time. '
        'The small weights are chosen for arithmetic. PyTorch stores them as proj.weight with shape (2,12). '+linear_ref+'.'+source,
        mobile_network()+'<p>A.R is the red value of pixel A. Each pixel contributes three input nodes.</p>'
        '<p>Every input connects to both outputs: 24 weights, plus one bias for each output. c₁ = [y₁, y₂].</p>')

    for coord,key,title in [(0,'patch-linear-first','Follow the connections into output 1'),(1,'patch-linear-second','Now follow the connections into output 2')]:
        body=network(coord)+t(755,70,'Output '+str(coord+1),32,'c-e')
        formula=['A.R + B.R','+ C.R + D.R'] if coord==0 else ['A.G + B.G','− C.G − D.G']
        arithmetic='1 + 0 + 0 + 1' if coord==0 else '0 + 1 − 0 − 1'
        bias=data['bias'][coord]
        bias_text=('+' if bias>=0 else '−')+f' {abs(bias):g} (bias)'
        body+=g(t(755,133,formula[0],31,'c-e')+t(755,177,formula[1],31,'c-e')
                +t(755,219,'Other incoming weights: 0',23,'ink-2'),1)
        body+=g(t(755,277,arithmetic,31)+t(755,323,bias_text,31),2)
        body+=g(t(755,397,'= '+f'{data["C"][0][coord]:g}'.replace('-','−'),47,'c-e'),3)
        add(key,title,body,
            'Multiply each input by its connection weight, add the contributions, then add the bias. The highlighted connections show the nonzero weights for this output.',
            'What does this output receive before we add its bias?',
            'Reveal the highlighted edges and their weights. Read the connected input values, compute the sum, then add the bias and reveal the answer.',
            'This is output '+str(coord+1)+' of the same nn.Linear(12,2) layer. '
            'For this selected output, the four highlighted weights are nonzero and its eight remaining weights are zero. '
            'The other output remains in the diagram so the architecture stays visible. '
            'Output 1 sums the four red values and adds 0.5. Output 2 adds the two top green values, subtracts the two bottom green values, and adds −0.5. '
            'These are chosen teaching weights. The computed output is '+f'{data["C"][0][coord]:g}'+', with no activation afterwards.'+source,
            mobile_network(coord)+'<p>Highlighted edges carry the nonzero weights for output '+str(coord+1)+'. Its other incoming weights are zero.</p>'
            +'<p>'+escape(' '.join(formula))+'</p><p>'+escape(arithmetic)+' '+bias_text+' = <strong>'+f'{data["C"][0][coord]:g}'+'</strong>.</p>')

    body=patch(35,90,80)+arrow(235,170,340,170)+box(360,137,340,'nn.Linear(12, 2)',29)
    body+=g(arrow(700,170,820,170)+t(835,183,'[2.5, −0.5]',40,'c-e'),1)
    body+=t(115,310,'x₁: (1, 12)',27,'c-e','middle')+g(t(945,310,'c₁: (1, 2)',27,'c-e','middle'),1)
    body+=g(t(360,365,'No activation: −0.5 stays −0.5.',29,'ink')+t(35,420,'Later block MLP: Linear → GELU → Linear',28,'ink-2'),2)
    add('patch-linear-result','These two numbers are the patch embedding',body,
        'c₁ represents the content of patch 1. Its two coordinates are features for the model; the image classifier comes later.',
        'Should the negative output become zero?',
        'Collect the two calculated coordinates and point to the absence of an activation after the layer.',
        'The patch projection is one affine layer. Calling it a forward pass just means evaluating that layer on its input. '
        'The Transformer block MLP studied later contains Linear(D,H), GELU, and Linear(H,D), where H is its hidden width. '
        'It is a separate component, applied after attention within the block. Our chosen patch width D=2 is unrelated to the number of class labels. '
        'The current content embedding will have position information added before the attention block.'+source,
        mobile_patch()+'<p><code>c1 = proj(x1)</code></p><p>(1,12) → (1,2)</p><p><strong>c₁ = [2.5, −0.5]</strong></p>'
        '<p>No activation: −0.5 stays −0.5.</p><p>Later block MLP: Linear → GELU → Linear.</p>')

    body=t(35,40,'Create once: proj = nn.Linear(12, 2)',32,'c-e')
    for i,y in enumerate([85,255]):
        marks=patch(35,y,55,i)+t(90,y+143,'P'+str(i+1),25,'ink-2','middle')
        marks+=t(200,y+65,'x'+str(i+1)+' · (1, 12)',27,'c-e')
        marks+=arrow(415,y+55,500,y+55)+arrow(765,y+55,850,y+55)
        marks+=t(870,y+65,vec(data['C'][i]),35,'c-e')
        body+=marks if i==0 else g(marks,1)
    body+=rect(500,95,265,265,'c-e','transparent')+t(632,190,'same W',33,'c-e','middle')+t(632,255,'same b',33,'c-e','middle')
    body+=g(t(460,425,'C = proj(X):   (2, 12) → (2, 2)',30,'c-e'),2)
    add('patch-shared-code','Apply the very same layer to another patch',body,
        'The pixel rows differ. Both use the same 26 parameters, so we compute both embeddings with one call: C = proj(X).',
        'Do we create a new nn.Linear layer when we move to patch 2?',
        'Reveal the second patch. Follow both arrows through the single weight-and-bias box, then read the stacked input and output shapes.',
        'Patch 2 has two blue pixels on top and two green pixels below. The same chosen weights produce c₂=[0.5,−2.5]. '
        'A matrix X with two patch rows has shape (2,12), and proj(X) returns (2,2). The first axis counts patches; the last axis counts features. '
        'Instantiate proj once outside any patch loop. Calling proj on each row separately and on both rows together gives identical results. '
        'Changing the number of patches changes the number of evaluations, not the 26 parameters.'+source,
        mobile_rows(['Patch','RGB pixels A, B, C, D','Content embedding'],[
            ['P1','red, green, blue, white','[2.5, −0.5]'],['P2','blue, blue, green, green','[0.5, −2.5]']])
        +'<p><code>proj = nn.Linear(12, 2)<br>C = proj(X)</code></p><p>(2,12) → (2,2); one shared set of 26 parameters.</p>')

    from vision1_real_patch_path import add_real_path
    add_real_path(b, add)

    result=[]
    new_ids=['patch-linear-shapes','patch-linear-weights','patch-linear-first','patch-linear-second','patch-linear-result','patch-shared-code']
    for title,frames in sections:
        out=[]
        for html in frames:
            key=re.search(r'class="frame[^\"]*" id="([^\"]+)"',html).group(1)
            if key=='s01-rows':
                from vision1_photo_walkthrough import ORDER
                out.extend(additions[k] for k in new_ids + ORDER)
                continue
            if key=='projection-size':continue
            out.append(additions.get(key,html))
            if key=='s01-rows-step-1':out.append(additions['patch-activation-location'])
        result.append((title,out))
    return result
