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

    # Keep the patch size, patch count, and embedding width on separate axes.
    grid=rect(65,90,240,240,'ink-3','transparent',0)
    for k in range(1,16):
        grid+=line(65+15*k,90,65+15*k,330,'line',1)
        grid+=line(65,90+15*k,305,90+15*k,'line',1)
    grid+=rect(65,90,15,15,'c-e','t-e',0)
    body=grid+t(185,55,'16 columns',28,'ink-2','middle')+t(185,380,'16 rows',28,'ink-2','middle')
    body+=g(t(420,100,'16 × 16 = 256 pixels',35,'c-e'),1)
    body+=g(t(420,185,'Each pixel contributes:',28,'ink-2')
            +box(420,215,120,'R')+box(570,215,120,'G')+box(720,215,120,'B'),2)
    body+=g(t(420,365,'256 × 3 = 768 values',37,'c-e'),3)
    add('patch-real-dimensions','Where do the 768 input values come from?',body,
        'One patch has 16 × 16 pixels. Each pixel contributes three channel values: red, green and blue. That gives 768 values in one patch.',
        'Are there 768 pixels in this patch, or 768 channel values?',
        'Count rows and columns in the patch grid. Pick one pixel, reveal its three channels, then multiply 256 by 3.',
        'The patch contains 256 spatial pixels, with three RGB numbers at each location. Thus 16×16×3=768 scalar input features. '
        'A channel value is one number within a pixel; it is not another image patch. '
        'Flattening keeps all these values in a chosen order, as in the earlier four-pixel example. '
        'Our pretrained checkpoint uses 16×16 RGB patches; the patch size and channel count determine this input width.',
        mobile_rows(['Count','Calculation'],[['Pixels per patch','16 rows × 16 columns = 256'],['Values per pixel','R, G, B = 3'],['Values per patch','256 × 3 = 768']]))

    body=t(35,125,'xᵢ =',33,'c-e')
    for x,label,values in [(300,'first pixel','R₁, G₁, B₁'),(575,'second pixel','R₂, G₂, B₂'),(965,'pixel 256','R₂₅₆, G₂₅₆, B₂₅₆')]:
        body+=t(x,65,label,25,'ink-2','middle')+t(x,125,values,28,'c-e','middle')
    body+=t(170,128,'[',39,'c-e')+t(780,125,'…',33,'c-e')+t(1130,128,']',39,'c-e')
    body+=g(t(365,240,'1',62,'c-e','middle')+t(505,240,'×',48,'ink-2','middle')+t(715,240,'768',62,'c-e','middle'),1)
    body+=g(arrow(365,260,365,315)+t(365,365,'one patch row',31,'ink','middle')
            +arrow(715,260,715,315)+t(795,365,'values in that row',31,'ink','middle'),2)
    add('patch-one-row-shape','What does the 1 in 1 × 768 count?',body,
        'Here we are processing one patch, written as one row. The second dimension counts its 768 input values. A whole image contributes many patch rows.',
        'If we put a second patch underneath this row, which dimension changes?',
        'Trace the long RGB row, then point to 1 for the row count and 768 for the number of entries in that row.',
        'We retain a two-dimensional matrix for this explanation: number of patch rows × values per patch. '
        'One patch has shape (1,768), two patches have shape (2,768), and all patches of this image will have shape (196,768). '
        'The leading 1 on this slide is a patch count, not the number of images in a batch. '
        'Batching adds a separate leading axis later. The tuple (1,768) and the written dimensions 1×768 describe the same shape here.',
        '<p><strong>1 × 768</strong></p>'+mobile_rows(['Dimension','Meaning'],[['1','One patch row'],['768','Channel values within that patch']])
        +'<p>Two patches: 2 × 768. A whole image will give 196 × 768.</p>')

    body=t(35,60,'pixel row',26,'ink-2')+t(380,60,'weights W',26,'ink-2')+t(710,60,'bias b',26,'ink-2')+t(930,60,'embedding',26,'ink-2')
    body+=t(35,155,'1 × 768',35,'c-e')+t(275,155,'×',32)
    body+=g(box(340,112,300,'768 × 192',33)+t(675,155,'+',32)+t(710,155,'1 × 192',29,'c-e'),1)
    body+=g(arrow(855,143,930,143)+t(945,155,'1 × 192',35,'c-e'),2)
    body+=g(t(35,285,'nn.Linear(768, 192)',38,'c-e')
            +t(35,355,'Affine map: weighted sums + bias; no activation.',28,'ink-2'),2)
    body+=g(t(35,425,'One patch stays one row. Its feature width changes: 768 → 192.',29,'c-e'),3)
    add('patch-one-row-projection','Does the embedding need 768 coordinates too?',body,
        'The model chooses 192 output coordinates here. Every patch uses that same output width; the input and output widths can differ.',
        'Does preserving one row require preserving all 768 input coordinates?',
        'Follow the row count 1 to the output. Then separately follow the feature width from 768 to the chosen 192.',
        'Input width is fixed by patch geometry and channels. Output width D is a model design choice: here D=192. '
        'A different model could choose another D; the operation does not require D=768. '
        'W has 768 rows and 192 columns in our row-vector notation, so each output coordinate combines 768 inputs and has one bias. '
        'We draw the bias as a (1,192) row for addition; PyTorch stores it as a 192-entry vector and broadcasts it. '
        'PyTorch stores W transposed as weight of shape (192,768). '
        'With a bias this is an affine map. nn.Linear does not add ReLU or GELU; the nonlinear block MLP is a separate component. '+linear_ref+'.',
        mobile_rows(['Quantity','Shape'],[['Input xᵢ','1 × 768'],['W','768 × 192'],['Bias','192 entries'],['Output cᵢ','1 × 192']])
        +'<p><code>nn.Linear(768, 192)</code>: weighted sums plus bias, with no activation.</p>'
        '<p>All patch embeddings have 192 coordinates in this model. That does not make 192 equal to the 768 input values.</p>')

    grid=rect(45,90,224,224,'ink-3','transparent',0)
    for k in range(1,14):
        grid+=line(45+16*k,90,45+16*k,314,'line',1)
        grid+=line(45,90+16*k,269,90+16*k,'line',1)
    grid+=rect(45,90,16,16,'c-e','t-e',0)
    body=grid+t(157,50,'224 × 224 image',27,'ink-2','middle')+t(157,360,'16 × 16 per patch',26,'c-e','middle')
    body+=g(t(405,100,'224 ÷ 16 = 14 patches across',31),1)
    body+=g(t(405,180,'224 ÷ 16 = 14 patches down',31),1)
    body+=g(t(405,270,'14 × 14 = 196 patch rows',35,'c-e'),2)
    body+=g(t(405,380,'X: (196, 768) → C: (196, 192)',31,'c-e'),3)
    add('projection-size','How many rows come from the whole image?',body,
        '196 counts patches in one image. Each patch supplies 768 input values and produces 192 embedding coordinates. The shared layer changes width while keeping one row per patch.',
        'Which number counts patches, and which numbers count values within a patch row?',
        'Count patches across and down the grid, then track the unchanged 196 in the input and output shapes.',
        'A 224×224 image divided into non-overlapping 16×16 patches gives (224/16)×(224/16)=196 patches. '
        'X has 196 rows and 768 pixel-value columns. C=proj(X) has the same 196 rows, with 192 embedding columns. '
        'These are the patch-content rows. Position information is added afterwards. '
        'For B images, shapes become (B,196,768) and (B,196,192); B is a separate batch axis.',
        mobile_rows(['Count','Calculation'],[['Across','224 ÷ 16 = 14'],['Down','224 ÷ 16 = 14'],['Patch rows','14 × 14 = 196'],
                                   ['Input X','196 × 768'],['Embeddings C','196 × 192']]))

    body=t(35,55,'nn.Linear(768, 192)',36,'c-e')+t(850,55,'parameter count',25,'ink-2')
    body+=t(35,145,'weights W',29,'ink-2')+t(400,145,'768 × 192',35,'c-e')
    body+=g(t(850,145,'147,456',35,'c-e'),1)
    body+=g(t(35,245,'bias b',29,'ink-2')+t(400,245,'1 per output',30)+t(850,245,'192',35,'c-e'),2)
    body+=g(line(35,285,1120,285,'ink-3')+t(35,345,'total parameters',29,'ink-2')+t(850,345,'147,648',38,'c-e'),3)
    body+=g(t(35,425,'Every one of the 196 patches reuses this same parameter set.',29),3)
    add('patch-projection-parameters','How many parameters does this one layer learn?',body,
        'Each of the 192 outputs has 768 weights and one bias. The parameters are shared across patches, so we count them once.',
        'If we add more patch rows, do we need another copy of these weights?',
        'Count weights, then biases, then add them. Point back to the single nn.Linear layer used for the whole matrix.',
        'There are 768×192=147,456 weights and 192 biases, giving 147,648 trainable parameters. '
        'Equivalently, each output has 768+1 parameters and there are 192 outputs. '
        'These parameters are used for each patch; 196 applications do not create 196 parameter sets. '
        'The next section shrinks to grayscale patches and D=4 so the complete attention calculation fits on the board.',
        mobile_rows(['Parameter','Count'],[['Weights','768 × 192 = 147,456'],['Biases','192'],['Total','147,648']])
        +'<p>All 196 patch rows reuse this same set of weights and biases.</p>')

    result=[]
    new_ids=['patch-linear-shapes','patch-linear-weights','patch-linear-first','patch-linear-second','patch-linear-result','patch-shared-code']
    for title,frames in sections:
        out=[]
        for html in frames:
            key=re.search(r'class="frame[^\"]*" id="([^\"]+)"',html).group(1)
            if key=='s01-rows':out.extend(additions[k] for k in new_ids)
            if key=='projection-size':
                out.extend(additions[k] for k in ['patch-real-dimensions','patch-one-row-shape','patch-one-row-projection'])
            out.append(additions.get(key,html))
            if key=='s01-rows-step-1':out.append(additions['patch-activation-location'])
            if key=='projection-size':out.append(additions['patch-projection-parameters'])
        result.append((title,out))
    return result
