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
    data = example()
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
            +arrow(875,285,875,330)+t(875,370,'compute 2 coordinates',31,'c-e','middle'),2)
    body+=t(35,420,'V = vocabulary size; the two models may choose different embedding widths.',23,'ink-2')
    add('s01-rows-step-1','How does this connect to text embeddings?',body,
        'Both produce a vector to represent one token. Text selects a learned table row; the image layer computes a row from the patch pixels.',
        'Do we have a vocabulary ID for every possible image patch?',
        'Compare lookup on the left with computation on the right. Keep the number of input values separate from the embedding width.',
        'The Part II text toy used four embedding coordinates. Our RGB warm-up chooses two coordinates so both can be calculated by hand. '
        'These dimensions are choices for different models. The image output is a patch-content embedding, called cᵢ in the next slides. '
        'Both the text embedding table and the patch layer have trainable parameters. Their output widths do not have to equal the number of classes. '
        +embedding_ref+' describes the lookup operation; '+linear_ref+' describes the affine map.',
        mobile_rows(['Text','Image'],[['Token ID','12 pixel values'],['nn.Embedding(V, 4)','nn.Linear(12, 2)'],
                                   ['Look up 4 coordinates','Compute 2 coordinates']]))

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

    body=t(35,45,'Chosen weights for this calculation',27,'ink-2')
    for i,letter in enumerate('ABCD'):body+=t(420+i*185,100,letter+' · RGB',25,'ink-2','middle')
    for r,values in enumerate(data['weight']):
        marks=t(35,170+r*95,'output '+str(r+1),28,'c-e')
        for j in range(4):marks+=t(420+j*185,170+r*95,vec(values[j*3:j*3+3]),29,'c-e','middle')
        body+=marks if r==0 else g(marks,1)
    body+=g(t(35,340,'bias = [0.5, −0.5]',31,'c-e'),2)
    body+=g(t(35,420,'proj.weight = Wᵀ: (2, 12)',27,'ink-2')+t(680,420,'proj.bias: (2,)',27,'ink-2'),2)
    add('patch-linear-weights','Which weights will we multiply by?',body,
        'Each output has twelve weights and one bias. We choose small values here; training normally learns these 26 parameters.',
        'Which input channels contribute to the first output?',
        'Read one weight per RGB entry. Point to the two output rows, then distinguish stored weight shape from W in the row-vector equation.',
        'These are all the weights, grouped by pixel. The first output sums red values; the second adds top-row green values and subtracts bottom-row green values. '
        'These interpretable filters are chosen for arithmetic, not claimed properties of a trained ViT. '
        'PyTorch stores proj.weight as (out_features,in_features)=(2,12), and evaluates X @ proj.weight.T + proj.bias. '
        'The slide equation uses W=proj.weight.T, shape (12,2). There are 12×2+2=26 trainable scalars. '+linear_ref+'.'+source,
        mobile_rows(['Pixel','Output 1 weights','Output 2 weights'],[[letter,vec(data['weight'][0][i*3:i*3+3]),vec(data['weight'][1][i*3:i*3+3])] for i,letter in enumerate('ABCD')])
        +'<p>Bias: [0.5, −0.5]. Stored weight shape: (2,12).</p>')

    for coord,key,title in [(0,'patch-linear-first','Calculate the first embedding coordinate'),(1,'patch-linear-second','Calculate the second embedding coordinate')]:
        body=t(35,45,'Same patch, output '+str(coord+1),27,'ink-2')
        for y,label in [(125,'RGB inputs'),(225,'weights'),(320,'dot product')]:body+=t(35,y,label,26,'ink-2')
        contributions=[]
        for i,rgb in enumerate(pixels):
            x=420+i*185; weights=data['weight'][coord][i*3:i*3+3]
            value=sum(a*w for a,w in zip(rgb,weights));contributions.append(value)
            body+=t(x,60,'ABCD'[i],28,'ink-2','middle')+t(x,125,vec(rgb),29,'c-e','middle')
            body+=g(t(x,225,vec(weights),29,'c-e','middle'),1)
            body+=g(arrow(x,245,x,280)+t(x,320,f'{value:g}',36,'c-e','middle'),2)
        terms=' + '.join(f'({v:g})' if v<0 else f'{v:g}' for v in contributions)
        bias=data['bias'][coord]
        body+=g(t(35,425,terms+(' + ' if bias>=0 else ' − ')+f'{abs(bias):g} = {data["C"][0][coord]:g}',37,'c-e'),3)
        add(key,title,body,
            'Multiply matching entries, add the four pixel contributions, then add the bias '+f'{bias:g}'+'.',
            'What does each pixel contribute before we add the bias?',
            'Multiply the three RGB entries by their three weights at each column, then add the four results and bias.',
            'This is output coordinate '+str(coord+1)+' of the same nn.Linear(12,2) layer. '
            'The corresponding row of proj.weight contains twelve entries. The displayed columns group those entries by pixel, not by separate layers. '
            'The resulting coordinate is '+f'{data["C"][0][coord]:g}'+'.'+source,
            mobile_rows(['Pixel','RGB','Weights','Product sum'],[[letter,vec(rgb),vec(data['weight'][coord][i*3:i*3+3]),f'{contributions[i]:g}'] for i,(letter,rgb) in enumerate(zip('ABCD',pixels))])
            +'<p>'+escape(terms)+(' + ' if bias>=0 else ' − ')+f'{abs(bias):g} = <strong>{data["C"][0][coord]:g}</strong>.</p>')

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

    body=t(35,48,'patch',25,'ink-2')
    for j,idx in enumerate([5,6]):
        y=90+j*175
        marks=crop(35,y,165,110,idx,'embedding-names')+t(118,y+145,'P'+str(idx+1),27,'ink','middle')
        marks+=arrow(210,y+55,315,y+55)+t(350,y+68,'c'+str(idx+1).translate(str.maketrans('67','₆₇')),37,'c-e')
        marks+=t(525,y+68,'+ p'+str(idx+1).translate(str.maketrans('67','₆₇')),37,'c-e')+arrow(705,y+55,790,y+55)+t(830,y+68,'e'+str(idx+1).translate(str.maketrans('67','₆₇')),37,'c-e')
        body+=marks if j==0 else g(marks,1)
    body+=t(350,48,'content',25,'ink-2')+t(525,48,'position',25,'ink-2')+t(830,48,'input to attention',25,'ink-2')
    body+=g(t(350,430,'Every cᵢ, pᵢ and eᵢ is a row of D coordinates.',29,'c-e'),2)
    add('s01-rows','What do c₆ and e₆ refer to?',body,
        'The subscript names the patch. c₆ is its content embedding; adding its position row p₆ gives e₆, the row that enters attention.',
        'Does the 6 mean six coordinates, or patch number 6?',
        'Point to P6 in the original illustration. Follow content plus position to e6, then repeat for P7.',
        'The earlier 4×4 grid is a coarse illustration with numbered patches. The subscripts 6 and 7 identify locations in that illustration, not embedding widths. '
        'cᵢ is the content row produced by the shared patch layer; pᵢ is a position row; eᵢ=cᵢ+pᵢ has the same shape (1,D). '
        'In the tiny RGB warm-up D=2; in our pretrained model D=192. '
        'This notation matches the text recap: attention receives an embedding plus position. We will calculate position addition in the next section.',
        mobile_rows(['Patch','Content','Position','Attention input'],[['P6','c₆','p₆','e₆ = c₆ + p₆'],['P7','c₇','p₇','e₇ = c₇ + p₇']])
        +'<p>Each row has shape (1,D). The subscript is a patch index.</p>')

    body=t(35,60,'One real-model patch',29,'ink-2')+box(35,110,285,'16 × 16 × 3',34)
    body+=g(arrow(320,143,435,143)+box(435,110,270,'768 values',33),1)
    body+=g(t(480,260,'nn.Linear(768, 192)',35,'c-e')+arrow(570,175,570,215)
            +arrow(820,248,910,248)+t(930,260,'cᵢ',40,'c-e'),2)
    body+=g(t(35,365,'xᵢ: (1, 768)',32,'c-e')+t(745,365,'cᵢ: (1, 192)',32,'c-e'),2)
    body+=g(t(35,420,'192 is the chosen embedding width; the input has 768 pixel values.',27,'ink-2'),3)
    add('patch-real-dimensions','Scale the same operation to a 16 × 16 RGB patch',body,
        'The pretrained ViT used later takes 768 values per patch and returns 192 embedding coordinates. The operation is the same weighted sum plus bias.',
        'Where do 768 and 192 come from?',
        'Calculate 16×16×3, then distinguish that count from the model designer’s choice of embedding width.',
        'Our saved checkpoint is vit_tiny_patch16_224.augreg_in21k_ft_in1k. Its 224×224 input is split into 16×16 patches. '
        'Each normalized RGB patch has 768 input values, mapped to D=192 output coordinates. '
        'Its Conv2d implementation is equivalent to a shared linear patch map with matching flattening order; section 8 derives that equivalence. '
        'This slide describes the real checkpoint shapes, while the preceding numerical weights belong only to the hand calculation.',
        mobile_rows(['Quantity','Size'],[['One patch','16×16×3'],['Flattened input','768 values'],['Shared layer','nn.Linear(768,192)'],['Content row','192 coordinates']]))

    body=t(35,55,'224 × 224 image / 16 × 16 patches → 196 patches',31)
    body+=t(35,140,'X',40,'c-e')+t(35,205,'(196, 768)',34,'c-e')
    body+=g(arrow(285,178,395,178)+box(395,143,375,'nn.Linear(768, 192)',29)
            +arrow(770,178,870,178)+t(895,140,'C',40,'c-e')+t(895,205,'(196, 192)',32,'c-e'),1)
    body+=g(t(35,315,'W: (768, 192)',31,'c-e')+t(650,315,'bias: (192,)',31,'c-e'),2)
    body+=g(t(35,415,'768 × 192 + 192 = 147,648 parameters, shared by all patches',29),3)
    add('projection-size','One layer produces all 196 patch embeddings',body,
        'The layer changes the feature width from 768 to 192. It keeps one output row per patch and reuses the same weights and bias everywhere.',
        'Do 196 patches require 196 different weight matrices?',
        'Follow the two axes: the patch count stays 196 while the feature width changes. Count the parameters once.',
        'C=XW+b broadcasts the 192-entry bias across the 196 rows. PyTorch stores the layer’s weight as (192,768), transposed relative to W here. '
        'All patches use this same learned map. A batch adds a leading dimension: (B,196,768) becomes (B,196,192). '
        'Neither patch position nor class labels are additional inputs to this linear layer. Position information is added afterwards. '
        'The next section shrinks to grayscale patches and D=4 so the complete attention computation fits on the board.',
        mobile_rows(['Array','Shape'],[['X: pixel rows','(196,768)'],['W: shared weights','(768,192)'],['C: content embeddings','(196,192)']])
        +'<p><code>proj = nn.Linear(768, 192)<br>C = proj(X)</code></p><p>147,648 parameters total.</p>')

    result=[]
    new_ids=['patch-linear-shapes','patch-linear-weights','patch-linear-first','patch-linear-second','patch-linear-result','patch-shared-code']
    for title,frames in sections:
        out=[]
        for html in frames:
            key=re.search(r'class="frame[^\"]*" id="([^\"]+)"',html).group(1)
            if key=='s01-rows':out.extend(additions[k] for k in new_ids)
            if key=='projection-size':out.append(additions['patch-real-dimensions'])
            out.append(additions.get(key,html))
        result.append((title,out))
    return result
