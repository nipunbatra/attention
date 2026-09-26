"""Complete lecture additions, using the series' existing whiteboard primitives."""
import base64
import json
import math
import re
from html import escape
import numpy as np


def build_full(b):
    globals().update({k:b[k] for k in ['ROOT','SRC','ASSETS','OUT','DATA','R','REAL','PHOTO','CAT','FRAMES','f','v','t','g','line','arrow','rect','image','crop','pixels','source_icon','svg','frame','mobile_rows']})
    original_section=b['section']
    groups={}
    b['section']=lambda n,title,frames,lit='':groups.update({n:frames})
    for name in ['motivation','worksheet_intro','routing','prediction','full_model','real_result']:
        b[name]()
    b['section']=original_section
    training=json.loads((ASSETS/'training.json').read_text())
    inspection=json.loads((ASSETS/'inspection.json').read_text())
    processed='data:image/png;base64,'+base64.b64encode((ASSETS/'model-input.png').read_bytes()).decode()
    extras={}

    def add(key,title,body,caption,question,point,prose='',mobile=''):
        extras[key]=frame(key,title,body,caption,question+'\n'+point,prose,mobile)
        return extras[key]

    def matrix(x,y,rows,color='ink',cell=53,size=27):
        result=''; rh=43
        for i,row in enumerate(rows):
            for j,item in enumerate(row): result+=t(x+j*cell,y+i*rh,str(item),size,color,'middle')
        left=x-cell*.48;right=x+(len(rows[0])-.52)*cell;top=y-30;bottom=y+(len(rows)-1)*rh+12
        return f'<path d="M{left+8} {top} H{left} V{bottom} H{left+8} M{right-8} {top} H{right} V{bottom} H{right-8}" fill="none" stroke="var(--{color})" stroke-width="2"/>'+result

    def code_frame(key,title,code,diagram,caption,question,point,prose):
        fr=frame(key,title,diagram,caption,question+'\n'+point,prose)
        fr=fr.replace('<div class="vp-figure">','<div class="vl-code"><pre><code>'+escape(code.strip())+'</code></pre><div class="vp-figure">',1)
        fr=fr.replace('</svg></div>','</svg></div></div>',1)
        # Small diagram uses its own 400px-wide coordinate system.
        fr=fr.replace('viewBox="0 0 1160 440"','viewBox="0 0 400 440"')
        extras[key]=fr

    # REAL MOTIVATION AND CONTINUITY.
    body=t(35,70,'Part I',26,'ink-2')+t(275,70,'Part II',26,'ink-2')+t(560,70,'Part III',26,'ink-2')+t(900,70,'Vision I',26,'ink-2')
    body+=t(35,150,'a a b i d',35,'c-e')+t(275,150,'river → bank',31,'c-q')+t(560,150,'red / wool → coat',29,'c-v')
    body+=image(885,105,220,147)
    body+=g(t(35,290,'represent',28,'c-e')+t(275,290,'read context',28,'c-q')+t(560,290,'several messages',28,'c-v')+t(900,290,'read an image',28,'c-e'),1)
    body+=g(line(35,334,1110,334,'ink-3')+t(575,397,'Represent → read → update → predict',35,'ink','middle'),2)
    add('bridge-text','We already know the operation. What changes?',body,'Carry the same objects and colours from Parts I–III into the photograph.',
        'What stays the same when the input is a photograph?','Point to representation, contextual reading, and multiple head messages in the earlier examples.',
        '<p><a href="part1.html">Part I</a> built a character predictor: embeddings, scores, probabilities, loss and learning. <a href="attention.html">Part II</a> gave “bank” a contextual representation. <a href="part3.html">Part III</a> let “coat” read multiple kinds of information. We reuse their row-vector notation, seven colours, and calculation sequence.</p>')

    body=crop(30,85,260,174,9,'query')+t(160,305,'this patch',26,'c-q','middle')
    for j,idx in enumerate([5,6,10]):
        x=495+j*215;body+=crop(x,90,180,120,idx,'context')+t(x+90,245,f'P{idx+1}',25,'c-k','middle')
        body+=g(arrow(x+90,270,300,345,'c-v'),1)
    body+=g(t(400,392,'a new representation of the same patch',31,'c-e'),2)
    add('patch-context','What information could this patch borrow?',body,'The receiver stays a patch. Attention can bring it information from other locations.',
        'Which visible regions might help interpret the dark crop?','Point to face, fur and nearby background; then trace messages back to the receiver.',
        'These arrows pose a question about useful context; they are not measured attention weights. Self-attention updates every patch row. Image classification will also use a separate summary row. No patch is assigned a human semantic label before attention.')

    # PATCHIFY AND A REAL RGB CALCULATION.
    rgb=[['255','0','0'],['0','255','0'],['0','0','255'],['255','255','255']]
    body=''
    colors=['#ed4545','#45ae61','#397cdb','#ffffff']
    for j,color in enumerate(colors):
        x=50+(j%2)*105;y=65+(j//2)*105
        body+=f'<rect x="{x}" y="{y}" width="100" height="100" fill="{color}" stroke="var(--line)"/>'
        body+=t(x+50,y+57,str(j+1),28,'ink','middle')
    body+=t(150,340,'2 × 2 RGB',30,'ink','middle')
    body+=g(arrow(315,165,435,165)+t(480,100,'R G B · R G B · R G B · R G B',28,'c-e')+t(480,165,'12 numbers → one row',34,'c-e'),1)
    body+=g(t(480,250,'[1,0,0, 0,1,0, 0,0,1, 1,1,1]',30,'c-e'),2)
    body+=g(t(480,330,'divide by 255 in this illustration',25,'ink-2'),2)
    add('rgb-flatten','What exactly is inside one RGB patch?',body,'Flattening rearranges numbers; it does not learn a representation.',
        'How many numbers describe these four coloured pixels?','Read RGB inside each pixel, then concatenate the four triples in row order.',
        'The ordering convention here is pixel-major RGB. PyTorch Unfold commonly returns channel-major order; a corresponding permutation of the projection weights gives the same linear operation. The real checkpoint uses its supplied normalization, rather than only dividing by 255.',
        mobile_rows(['Pixel','RGB / 255'],[['Red','[1,0,0]'],['Green','[0,1,0]'],['Blue','[0,0,1]'],['White','[1,1,1]']]))

    body=t(30,75,'one patch',26,'ink-2')+t(30,145,'1 × 768',40,'c-e')
    body+=g(t(345,75,'shared W_patch',26,'ink-2')+rect(335,112,250,75,'c-e','t-e')+t(460,158,'768 × 192',33,'c-e','middle')+t(275,158,'×',34),1)
    body+=g(t(635,155,'+ bias',29)+arrow(785,145,865,145)+t(900,145,'1 × 192',36,'c-e'),2)
    body+=g(t(30,290,'196 patches → 196 applications of the same layer',32)+t(30,380,'768 × 192 + 192 = 147,648 parameters',33,'c-e'),3)
    add('projection-size','How many weights turn pixels into a patch row?',body,'One learned projection is shared across locations. Positions enter afterwards.',
        'Does doubling the number of patches double the projection parameters?','Trace one row through the matrix and bias; then reuse that operation for all patches.',
        'For 16×16 RGB patches and D=192, the patch projection has 768×192 weights and 192 biases. The number of image patches changes how often the projection is applied, not this parameter count. The positional table can depend on the grid size.')

    body=image(30,55,390,260)+t(225,360,'overlapping local neighbourhoods',26,'ink-2','middle')
    body+=g(rect(147,122,117,78,'c-v','transparent',0)+t(525,105,'Convolution',32,'c-v')+t(525,160,'shared local filters',29),1)
    body+=g(t(525,250,'Self-attention',32,'c-q')+t(525,305,'input-dependent weights across rows',29),2)
    body+=g(t(525,397,'Both can learn useful visual features.',29),3)
    add('cnn-context','Could a convolution use context too?',body,'A ViT changes the way spatial information is mixed.',
        'Does attention have a monopoly on context?','Point to a local neighbourhood; describe how stacked convolutions widen the receptive field.',
        'Convolutions encode locality and spatial weight sharing; deeper CNNs can use broad context. Global self-attention permits direct interactions between any two patch rows in one layer, with weights depending on the input. This is an architectural comparison, not a claim that ViTs always outperform CNNs. The ViT paper and D2L discuss the role of training data and inductive bias.')

    body=''
    for offset,label,causal in [(70,'next-token text',True),(705,'image classification',False)]:
        for row in range(5):
            for col in range(5):
                allowed=not causal or col<=row
                body+=rect(offset+col*52,85+row*52,48,48,'line','t-e' if allowed else 'card',0)
                if not allowed:body+=t(offset+col*52+24,119+row*52,'×',24,'ink-3','middle')
        body+=t(offset+130,395,label,29,'ink','middle')
    body+=g(t(375,160,'receiver rows',25,'c-q')+t(375,230,'source columns',25,'c-k'),1)
    add('image-mask','Should the top-left patch be forbidden to read the bottom-right?',body,'All image patches are observed before we predict the class. Standard ViT classification uses no causal mask.',
        'What future information would leak from another patch in this task?','Compare a lower-triangular next-token mask with all-to-all image attention.',
        'A raster ordering lets us store image patches as rows; it does not turn image classification into next-patch generation. Every receiver may read every source, including itself and CLS. Autoregressive image generation and masked-image training are different tasks with different masking choices.')

    # POSITION AND ALL PROJECTION PARAMETERS.
    body=pixels(20,70,DATA['images']['horizontal'],54,False,True)+t(130,340,'two filled + two empty',25,'ink','middle')
    body+=g(arrow(280,175,425,175)+t(470,135,'{ filled, filled, empty, empty }',31,'c-e'),1)
    body+=g(t(470,225,'Without location:',28)+t(470,287,'which arrangement did we start with?',29),2)
    body+=g(t(470,390,'A bag of patches loses their layout.',31,'c-a'),3)
    add('position-question','Can a bag of patches tell these images apart?',body,'Predict what happens before we add position information.',
        'What information was discarded when we made this bag?','Ask students to reconstruct the original layout; reveal that both layouts fit.',
        'This claim concerns shared patch operations and a permutation-invariant readout such as CLS, without another source of location. Reordering rows after adding position keeps the position attached to its patch and is a different experiment.')

    body=pixels(30,115,[[1,1],[1,1]],65,True)+t(95,320,'P1',27,'ink','middle')
    body+=t(240,200,'[1 1 1 1]',30,'c-e')+g(t(460,90,'W_patch',27,'c-e')+matrix(490,146,[['¼',0,0,0]]*4,'c-e',55),1)
    body+=g(t(740,200,'+ [0 0 0 1]',29,'c-e'),2)
    body+=g(t(325,400,'= [1, 0, 0, 1]',38,'c-e'),3)
    add('patch-matrix','Can we calculate the entire patch projection?',body,'Each output coordinate is a dot product with one column of W_patch.',
        'Which column computes ink, and which columns are zero?','Calculate all four coordinates, then add the bias.',
        'The worksheet fixes every parameter to expose the arithmetic. In a trained ViT, patch projection weights and bias are learned. Reusing the same matrix for P1–P4 is essential to the position experiment.',
        '<p>W_patch has four identical rows [¼,0,0,0]. Bias = [0,0,0,1]. P1 [1,1,1,1] projects to [1,0,0,1]; an empty patch projects to [0,0,0,1].</p>')

    body=''
    for j,idx in enumerate([5,6,9,10]):
        x=35+j*190;body+=crop(x,80,145,97,idx,'cls-source')
        body+=g(arrow(x+72,200,940,260,'c-v'),1)
    body+=g(rect(860,260,230,85,'c-e','t-e')+t(975,312,'CLS',38,'c-e','middle'),1)
    body+=g(t(35,395,'one extra row that learns to collect an image summary',32),2)
    add('why-cls','Where should a single image answer come from?',body,'Add a summary row, then let it read the same patch rows.',
        'We have many contextual patch rows. How could we produce one image label?','Gather their messages into CLS; mention average pooling as another valid readout.',
        'CLS is a learned initial vector shared across examples. After attention, it depends on the image. It is neither an image patch nor an extra known label. Some ViT variants use mean pooling instead. In the original ViT formulation the final normalized CLS representation feeds the classifier.')

    body=t(30,55,'input coordinate',23,'ink-2')+t(460,55,'W_Q¹',28,'c-q')+t(700,55,'W_K¹',28,'c-k')+t(950,55,'W_V¹',28,'c-v')
    for i,name in enumerate(['ink','row','column','constant']):body+=t(30,138+43*i,name,27,'c-e')
    body+=matrix(465,138,[[0,0],[0,0],[0,0],[1,1]],'c-q')
    body+=g(matrix(705,138,[['√2',0],[0,'√2'],[0,0],[0,0]],'c-k'),1)
    body+=g(matrix(955,138,[[1,0],[0,1],[0,0],[0,0]],'c-v'),2)
    body+=g(t(30,385,'[ink, row, column, 1] → q, k, v: two coordinates each',30),3)
    add('all-qkv','Which weights create the query, key and value?',body,'One input row; three separate learned projections. These worksheet weights are chosen for arithmetic.',
        'What does each nonzero matrix entry copy or scale?','Follow the constant into Q, then ink and row into K and V.',
        'These are 4×2 matrices in row-vector notation. Head 2 keeps the same W_Q, moves the second nonzero row of W_K and W_V from row-coordinate to column-coordinate, and has its own softmax. The full numerical notebook prints every Q, K and V row.',
        '<p>q=[1,1]; k=[√2×ink, √2×row]; v=[ink,row]. Head 2 replaces row with column.</p>')

    body=t(30,65,'Head 1',30,'c-v')+t(610,65,'Head 2',30,'c-v')
    body+=pixels(30,125,DATA['images']['horizontal'],43,False,True)+pixels(610,125,DATA['images']['horizontal'],43,False,True)
    body+=g(t(245,165,'ink + row',31,'c-k')+t(245,235,'values: [ink,row]',28,'c-v'),1)
    body+=g(t(830,165,'ink + column',29,'c-k')+t(830,235,'values: [ink,col]',27,'c-v'),2)
    body+=g(t(30,395,'Two independent mixtures of the same five source rows.',32),3)
    add('heads-question','Could one patch mixture hide a useful distinction?',body,'Part III’s coat example returns: preserve more than one reading of the input.',
        'What two kinds of information might help distinguish these layouts?','Point to row and column. Keep the same patches on both sides.',
        'The names “row head” and “column head” describe these hand-set matrices. Trained heads are not guaranteed to have neat human roles. This exercise demonstrates separate mixtures; it is not a proof that two heads are necessary for the task.')

    body=t(30,65,'Head 1 · all receiver rows',30,'c-a')
    for i,name in enumerate(['CLS','P1','P2','P3','P4']):
        body+=t(40,133+54*i,name,25,'c-e')
        body+=g(t(190,133+54*i,v(R['heads'][0]['A'][i]),28,'c-a'),1)
    body+=g(t(840,135,'Why identical?',27)+t(840,215,'All queries',27,'c-q')+t(840,260,'are [1, 1].',30,'c-q'),2)
    add('all-receivers','What about the patch rows themselves?',body,'Every row receives a message. Our chosen queries make all five attention rows identical.',
        'Do identical messages imply identical updated rows?','Compare the queries, then remind students that each residual starts from a different E row.',
        'This is a simplification of this worksheet, not a property of self-attention. In the later trained model, queries usually depend on image content, and attention patterns differ across receivers. A has shape 5×5 per head: receiver rows, source columns.',
        mobile_rows(['Receiver','Source weights'],[['CLS' if i==0 else f'P{i}',v(R['heads'][0]['A'][i])] for i in range(5)]))

    body=t(30,80,'same weight: 0.229',33,'c-a')+g(t(35,185,'P1 value [1,0]')+t(600,185,'P3 value [0,1]'),1)
    body+=g(arrow(185,215,185,280,'c-v')+arrow(765,215,765,280,'c-v')+t(35,335,'[0.229, 0]',36,'c-v')+t(600,335,'[0, 0.229]',36,'c-v'),2)
    add('weight-message','Does the same attention weight mean the same message?',body,'The value vector determines what travels along an attention weight.',
        'What do these two patches contribute to the ink coordinate?','Multiply both coordinates and compare the resulting messages.',
        'P1 and P3 have equal Head 1 weights in the Across example. P1 contributes ink; P3 contributes row information. Attention weight alone does not specify a patch’s effect on the eventual class score.')

    # FULL BLOCK: CONCRETE NORMALIZATION AND MLP.
    z=np.array([1.,0.,0.,1.]); zn=(z-z.mean())/np.sqrt(z.var()+1e-5)
    body=t(30,90,'one row',27,'ink-2')+t(300,90,'[1, 0, 0, 1]',38,'c-e')
    body+=g(t(30,190,'mean = 0.5',31)+t(550,190,'variance = 0.25',31),1)
    body+=g(t(30,290,'(x − mean) / √(variance + ε)',34),2)
    body+=g(t(30,395,'≈ [1, −1, −1, 1]',38,'c-e')+t(660,395,'then learned scale + bias',26),3)
    add('layernorm','What does LayerNorm do to one row?',body,'Normalize across the coordinates of each token, independently of other tokens.',
        'Which numbers participate in the mean and variance?','Use the four coordinates of one row; do not average across patches or the batch.',
        'This example uses ε=10⁻⁵ and initially scale γ=1, bias β=0. Exact normalized values are about ±0.99998. PyTorch LayerNorm uses the population variance across the normalized dimensions. γ and β are trainable. These are separate full-block examples; they do not retroactively alter the worksheet outputs.')

    body=t(30,80,'one row: [1, −1]',36,'c-e')+g(t(30,170,'linear → [1, −1, 0]',33),1)
    body+=g(t(30,260,'GELU → [0.841, −0.159, 0]',33,'c-v'),2)
    body+=g(t(30,350,'linear → [0.841, −0.159]',33,'c-d'),3)
    body+=g(t(30,418,'residual → [1.841, −1.159]',31,'c-e'),4)
    add('mlp-row','What can the MLP change without talking to another patch?',body,'Shared weights transform each row separately; the nonlinearity adds expressive power.',
        'Where is information mixed here: between rows or between coordinates?','Trace one row through two linear layers and GELU, then add the residual.',
        'For this separate 2D illustration W₁=[[1,0,1],[0,1,1]] and W₂=[[1,0],[0,1],[0,0]], with zero biases. Thus [1,−1]W₁=[1,−1,0]. GELU(x)=xΦ(x). A real block normally expands D to a wider hidden dimension and returns to D. The notebook checks these numbers with PyTorch.')

    body=''
    for j,(label,col) in enumerate([('raw patch rows','c-e'),('contextual rows','c-v'),('richer contextual rows','c-e')]):
        x=40+j*385
        for r in range(4):body+=rect(x,90+r*54,255,34,col,'t-e')
        body+=t(x+127,365,label,24,col,'middle')
        if j<2:body+=g(arrow(x+280,190,x+353,190)+t(x+318,160,'block',22,'ink-2','middle'),j+1)
    body+=g(t(30,420,'A later query can depend on context read in earlier blocks.',29),3)
    add('depth','What can a second block ask that the first could not?',body,'Every block receives the updated representations from the previous block.',
        'Has the query in block two already seen other patches indirectly?','Trace the first residual update into the next block’s query projection.',
        'In the first block the CLS query comes from a shared learned starting vector (plus its position). In later blocks its query depends on previous image-specific updates. Global attention gives direct access in one step, but does not make depth redundant.')

    # EXECUTABLE CODE, SAME ACTUAL MODEL AS THE TRAINING EXPERIMENT.
    simple=rect(75,50,245,70,'c-e','t-e')+t(197,93,'B × 1 × 8 × 8',26,'c-e','middle')+arrow(197,140,197,215)+rect(75,235,245,70,'c-e','t-e')+t(197,279,'B × 16 × 16',26,'c-e','middle')+t(197,387,'rows × width',24,'ink-2','middle')
    code_frame('code-patch','How do image pixels become rows in code?', '''patch = nn.Conv2d(1, 16,
    kernel_size=2, stride=2)

x = patch(images)    # B,16,4,4
x = x.flatten(2)     # B,16,16
x = x.transpose(1,2) # B,16,16''',simple,
        'A non-overlapping convolution implements the shared patch projection.',
        'Why does this convolution produce exactly one output per patch?','Follow kernel size 2 and stride 2 across an 8×8 image.',
        'The two 16s in the final shape mean different things: 16 patch rows and D=16 coordinates. Conv2d applies the same kernel weights at every patch. Flatten+Linear is equivalent when pixel order and weights are matched. For the real RGB checkpoint, use C=3, patch size 16, D=192. Full executable classes are in <a href="notebooks/vision/train_small_vit.py">train_small_vit.py</a>.')
    code_frame('code-attention','Can we write attention in five lines?', '''Q = E @ W_Q
K = E @ W_K
V = E @ W_V
A = (Q @ K.transpose(-2,-1)
     / math.sqrt(d_k)).softmax(-1)
H = A @ V''',t(50,90,'E: 5 × 4',30,'c-e')+g(t(50,175,'Q, K, V: 5 × 2',27,'c-q'),1)+g(t(50,260,'A: 5 × 5',30,'c-a'),2)+g(t(50,345,'H: 5 × 2',30,'c-v'),3),
        'Softmax over the last axis means a distribution over source rows for each receiver.',
        'Which axis must sum to one?','Match the source dimension in A to the row dimension in V.',
        'This is one worksheet head without a causal mask. The notebook independently compares both heads with torch.nn.MultiheadAttention and the browser’s JavaScript calculation. Softmax is implemented stably by the tensor library.')
    code_frame('code-block','What is the complete pre-LayerNorm block?', '''def forward(self, x):
    z = self.norm1(x)
    message, _ = self.attn(
        z, z, z, need_weights=False)
    x = x + message
    z = self.norm2(x)
    return x + self.mlp(z)''',t(50,75,'x',32,'c-e')+arrow(65,100,65,175)+t(50,215,'+ attention',28,'c-d')+arrow(65,245,65,300)+t(50,342,'+ row MLP',28,'c-d'),
        'MultiheadAttention includes concatenation and W_O. Both residual additions keep width D.',
        'Where did the output projection go in this code?','Open the accompanying Block class: PyTorch’s attention module owns out_proj.',
        'The executable Block initializes LayerNorm(D), MultiheadAttention(D,2,batch_first=True), another LayerNorm(D), and Linear(D,2D)→GELU→Linear(2D,D). Dropout is zero in this controlled experiment. All components receive gradients during training.')
    code_frame('code-model','How does the full model produce image logits?', '''x = self.patch(images)
x = x.flatten(2).transpose(1,2)
cls = self.cls.expand(x.size(0),-1,-1)
x = torch.cat([cls, x], dim=1)
x = x + self.pos
x = self.norm(self.blocks(x))
logits = self.head(x[:,0])''',t(35,60,'16 patch rows',28,'c-e')+g(t(35,145,'+ CLS → 17 rows',26,'c-e'),1)+g(t(35,230,'2 blocks → 17 rows',24),2)+g(t(35,315,'read CLS → 2 logits',25,'c-a'),3),
        'Only the final readout selects CLS. Every transformer block processes all rows.',
        'At what point do we discard the patch rows for classification?','Keep all 17 rows through both blocks, then select x[:,0].',
        'SmallViT owns a learned CLS vector, a learned 17×16 position table, two complete blocks, final LayerNorm, and a 16→2 classifier. The position control skips the addition and freezes the unused position parameter. The listing shows the positions-on path.')
    code_frame('code-train','Which call makes this model learn?', '''model.train()
logits = model(images)
loss = F.cross_entropy(logits, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()''',t(30,65,'image + known label',26,'c-e')+arrow(180,92,180,160)+t(30,207,'logits → loss',30,'c-a')+arrow(180,232,180,300,'c-d')+t(30,347,'gradients → weights',27,'c-d'),
        'Pass raw logits to cross_entropy. It combines log-softmax with the label loss.',
        'Which line computes gradients, and which one changes the weights?','Distinguish backward from step. Point back to Part I’s learning loop.',
        'The complete script uses AdamW, batches of 64, learning rate 0.003, weight decay 0.01, and 80 epochs. Validation selects the checkpoint. model.eval() and torch.inference_mode() are used for evaluation; neither trains the model.')

    # LEARNING: ONE CHECKABLE STEP, THEN MANY SAMPLES.
    probability=np.array(R['probability']); logit=np.array(R['logits']); gradient=probability-np.array([1.,0.]); newlogit=logit-.5*gradient
    newprob=np.exp(newlogit-newlogit.max());newprob/=newprob.sum()
    body=t(30,60,'target: Across the top',30,'c-e')+t(30,145,'p − y = '+v(gradient),36,'c-a')
    body+=g(t(30,240,'class bias b: [0,0] → '+v(-.5*gradient),33,'c-d'),1)
    body+=g(t(30,335,f'P(Across): {probability[0]:.3f} → {newprob[0]:.3f}',38,'c-e'),2)
    body+=g(t(30,416,f'Loss: {-np.log(probability[0]):.3f} → {-np.log(newprob[0]):.3f}',31,'c-a'),3)
    add('one-update','Can we make the correct answer a little more likely?',body,'Freeze the worksheet and update a newly added two-number class bias with learning rate 0.5.',
        'What sign should the gradient have for the correct class?','Compute p minus one-hot y, subtract half the gradient, then recompute the softmax.',
        'For one cross-entropy example ∂L/∂logits=p−y. Adding a class bias initialized at zero leaves the original worksheet unchanged; updating only that bias makes this step fully calculable. It is one SGD illustration, not the AdamW procedure in the larger training run. The notebook verifies the gradient with autograd.')

    body=''
    for j,arr in enumerate(training['examples'][:4]):
        x=25+j*285
        for r in range(8):
            for c in range(8):
                gray=int(255*(1-arr[r][c]));body+=f'<rect x="{x+c*29}" y="{78+r*29}" width="29" height="29" fill="rgb({gray},{gray},{gray})"/>'
        body+=t(x+116,355,'horizontal' if j%2==0 else 'vertical',27,'ink','middle')
    body+=g(t(30,421,'512 train     ·     128 validation     ·     256 test',31,'c-e'),1)
    add('training-data','Will it recognize a new noisy image?',body,'Randomize stripe location, intensity and pixel noise. Generate independent splits before training.',
        'What would memorizing our original two clean images fail to demonstrate?','Compare noisy samples, then identify the roles of the three splits.',
        'Images are 8×8 grayscale. Each has a bright horizontal or vertical stripe one patch wide; location is sampled from four rows or columns. Each horizontal/vertical pair has exactly the same 2×2 patch multiset. Train, validation and test use seeds 11,22,33 with 512,128,256 examples. Noise and intensity vary independently across pairs. These test examples share the training distribution; they do not establish natural-image or distribution-shift robustness.')

    hist=training['results'][0]['history'];x0,y0,w,h=100,345,950,270
    body=line(x0,y0,x0+w,y0,'ink-3')+line(x0,y0,x0,y0-h,'ink-3')+t(30,50,'accuracy',23,'ink-2')
    for val in [0,.5,1]:body+=t(85,y0-val*h+7,f'{val*100:.0f}%',21,'ink-2','end')+line(x0,y0-val*h,x0+w,y0-val*h,'line')
    for field,col in [('train','c-e'),('validation','c-v')]:
        points=' '.join(f'{x0+r["epoch"]/80*w},{y0-r[field]["accuracy"]*h}' for r in hist)
        body+=g(f'<polyline points="{points}" fill="none" stroke="var(--{col})" stroke-width="4"/>',1 if field=='train' else 2)
    for epoch in [0,20,40,60,80]:body+=t(x0+epoch/80*w,381,str(epoch),23,'ink-2','middle')
    body+=t(580,430,'epoch',23,'ink-2','middle')+t(650,40,'train',25,'c-e')+t(835,40,'validation',25,'c-v')
    add('learning-curves','Does improvement carry over to unseen examples?',body,'Track training and validation during learning; reserve test images for the selected checkpoint.',
        'What gap between these curves would worry you?','Read both curves at the same epoch, then explain minimum-validation-loss checkpoint selection.',
        'These are actual recorded accuracies from one deterministic run, sampled at epochs 0,1,5,10,20,40,60,80; lines connect those samples. Checkpoints were compared using validation loss each epoch. The selected positions-on checkpoint is epoch 80. The full logs include losses and exact counts. No claim of a multi-seed benchmark is made.',
        mobile_rows(['Epoch','Train','Validation'],[[str(z['epoch']),f"{z['train']['accuracy']:.1%}",f"{z['validation']['accuracy']:.1%}"] for z in hist]))

    body=t(30,70,'positions on',31,'c-e')+t(30,220,'positions off',31,'ink-2')
    for i,result in enumerate(training['results']):
        y=105+i*150;acc=result['test']['accuracy']
        body+=g(rect(340,y,650*acc,52,'c-e' if i==0 else 'line','t-e' if i==0 else 'card')+t(1080,y+36,f'{result["test"]["correct"]} / 256',30,'c-e','end'),1)
    body+=g(t(30,419,'Without position, each opposite-label pair has the same prediction.',27),2)
    add('trained-position-control','What happens when a trained ViT loses position?',body,'Same generated splits and training recipe. Only the position mechanism is disabled.',
        'Can any amount of training recover layout from the patch bag in this setup?','Reveal both test counts; connect the 50% result to paired permutation invariance.',
        'Positions on: 256/256 correct. Positions off: 128/256 correct. Without positions the maximum paired probability difference is below 10⁻⁵. Each opposite-label pair has identical patch contents, so a permutation-invariant classifier cannot distinguish the pair. This exact control supports the structural argument; 100% here reflects a deliberately simple synthetic task, not broad visual competence. Checkpoint selection is performed separately for each model.')

    # PREPROCESSING, PRETRAINING AND REAL PREDICTIONS.
    body=image(25,55,455,304)+g(arrow(505,190,600,190)+image(640,55,305,305,processed),1)
    body+=g(t(30,419,'original photograph → supplied resize / crop → normalize RGB',29),2)
    add('real-input','Which pixels does the checkpoint actually receive?',body,'Show the exact 224×224 crop used for the measured prediction.',
        'Are we feeding the original rectangular photograph directly to the network?','Match the original to the evaluation crop, then mention normalization.',
        'The script uses timm.data.resolve_model_data_config and create_transform(is_training=False) for the checkpoint, including its interpolation, resize, center crop, mean and standard deviation. The right image reverses normalization for display. The attention maps below refer to this crop, not the original rectangular image.')

    cat=REAL['results'][1]
    body=image(30,30,345,400,CAT)+t(520,70,'Same checkpoint. Another photograph.',29)
    for j,item in enumerate(cat['top3']):
        y=153+j*105
        body+=g(t(520,y,item['label'].split(',')[0],28)+t(1110,y,f"{item['probability']:.2%}",29,'c-e','end'),1)
    add('real-cat','Does the same model recognize the other photograph?',body,'Run the identical preprocessing and learned weights on a Persian cat.',
        'Do we change model weights when a different test image arrives?','Reveal the top three ImageNet labels and probabilities.',
        'This second saved inference result is not used to train or choose a checkpoint. The two photographs demonstrate actual model inputs and outputs, not an estimate of accuracy. Images may overlap unknown pretraining sources; no claim of complete pretraining holdout is made.')

    body=''
    entries=[('pretrain','many images','learn representations'),('fine-tune','new task + labels','update selected weights'),('infer','a new image','keep weights fixed')]
    for j,(name,inp,act) in enumerate(entries):
        y=75+j*130;body+=t(30,y,name,32,'c-e')+g(t(335,y,inp,28)+arrow(655,y-10,735,y-10)+t(775,y,act,25),j+1)
    add('three-phases','If our labels are “cat” and “dog”, what must change?',body,'Choose a task vocabulary and learn a suitable classifier from labelled training examples.',
        'Is an ImageNet breed score already a fitted two-class Pets classifier?','Separate the checkpoint’s history, adapting a classifier, and running inference.',
        'The model name identifies ImageNet-21k pretraining and ImageNet-1k fine-tuning. For a new cat/dog task, use an explicit class mapping and separate train/validation/test data. Training only a new head on frozen embeddings is a linear probe; updating the backbone is fine-tuning. This lecture runs full small-model training on the synthetic task and pretrained real-photo inference. It does not claim a completed Pets fine-tuning benchmark.')

    # MEASURED ATTENTION. COMMON SCALE WITH RAW PATCH MASS.
    def heatmap(x,y,size,record,maxweight,label,overlay=False):
        out=('<g opacity=".28">'+image(x,y,size,size,processed)+'</g>') if overlay else ''
        for rr,row in enumerate(record['patch_weights']):
            for cc,value in enumerate(row):
                strength=value/maxweight
                if overlay:fill=f'rgba(190,18,60,{strength*.8:.5f})'
                else:
                    end=(190,18,60);start=(247,248,250)
                    fill='rgb('+','.join(str(round(start[i]+strength*(end[i]-start[i]))) for i in range(3))+')'
                out+=f'<rect x="{x+cc*size/14}" y="{y+rr*size/14}" width="{size/14+.2}" height="{size/14+.2}" fill="{fill}"/>'
        out+=t(x+size/2,y+size+35,label,26,'ink','middle')
        return out
    maps=[m for m in inspection['attention'] if m['block']==12 and m['query_index']==0]
    mx=max(m['maximum_patch_weight'] for m in maps)
    body=t(30,25,f'Source weight: transparent 0 → rose {mx:.3f}',23,'ink-2')
    for i,m in enumerate(maps):
        body+=heatmap(30+i*380,45,300,m,mx,f'head {m["head"]}',overlay=True)
        body+=t(180+i*380,420,f'CLS self-weight {m["cls_weight"]:.3f}',23,'ink-2','middle')
    add('real-heads','Do real heads put weight on the same places?',body,'Measured block 12, CLS query, heads 1–3. All three maps use the same colour scale.',
        'Which similarities and differences can you see across heads?','Read the 14×14 source grid and the separately reported CLS self-weight.',
        f'These are raw attention probabilities A[CLS,patch], before value mixing, from the exact model input. Transparent→rose spans 0→{mx:.4f} in every map on this frame. No per-head rescaling or renormalization over patches is applied. The missing source is CLS itself, reported below each map. Head numbers are one-based. Maps are not class-specific explanations.',
        '<p>Block 12, query CLS, heads 1–3. Read-only data: <a href="figures/vision1/inspection.json">full attention arrays and metadata</a>.</p>')
    first=next(m for m in inspection['attention'] if m['block']==1 and m['head']==1 and m['query_index']==0)
    last=maps[0];mx=max(first['maximum_patch_weight'],last['maximum_patch_weight'])
    body=heatmap(80,45,305,first,mx,'block 1 · head 1',overlay=True)+heatmap(745,45,305,last,mx,'block 12 · head 1',overlay=True)+arrow(435,205,680,205)+t(560,255,'11 more blocks',23,'ink-2','middle')
    add('real-depth','Does the summary read the same way at every depth?',body,'Keep the image and CLS query identity fixed; compare two layers on one shared scale.',
        'What changed between the two query representations?','Recall that later CLS queries already contain image-dependent context.',
        f'Raw probabilities, both plotted from 0 to {mx:.4f}. Block 1 head 1 and block 12 head 1 are different learned projections; equal head indices do not imply the same learned function across layers. Changes in these weights alone do not identify a class explanation.')
    patch=next(m for m in inspection['attention'] if m['block']==12 and m['head']==1 and m['query_index']==104)
    body=image(60,55,320,320,processed)+rect(60+5*320/14,55+7*320/14,320/14,320/14,'c-q','transparent',0)+arrow(435,205,630,205,'c-q')+heatmap(735,55,320,patch,patch['maximum_patch_weight'],'sources for this patch',overlay=True)
    add('real-patch-query','What if the receiver is a patch instead of CLS?',body,'Measured block 12, head 1. Query: patch row 7, column 5, counting from zero.',
        'Which row of the attention matrix did we select this time?','Locate the purple receiver square, then read the source heatmap.',
        f'Token index 104 = 1 + 7×14 + 5. Its 196 patch-source weights are plotted, with scale 0→{patch["maximum_patch_weight"]:.4f}; CLS-source weight is {patch["cls_weight"]:.4f}. The query patch itself is also a permitted source. This frame returns to the opening question: a patch can read context from the rest of the image.')

    body=image(25,85,280,280,processed)+line(165,85,165,365,'card',2)+line(25,225,305,225,'card',2)
    body+=t(355,45,'mask one quadrant with mean RGB',26,'ink-2')
    for i,item in enumerate(inspection['occlusion']):
        y=115+i*80
        body+=t(355,y,item['region'],26)+g(rect(590,y-29,400*item['target_probability'],35,'c-e','t-e')+t(1105,y,f"{item['target_probability']:.1%}",27,'c-e','end'),1)
    body+=g(t(30,423,'Unmasked P(Newfoundland) = 95.7%',29,'c-e'),2)
    add('occlusion','Does removing a region change the prediction?',body,'Four predetermined quadrant interventions; report the same target-class probability each time.',
        'Which quadrant would you expect to affect this prediction most?','Collect predictions before revealing the measured bars.',
        'Each quadrant is 112×112 in the actual 224×224 model crop. Normalized input zero fills the quadrant with the model’s RGB mean. All four masked images still predict Newfoundland, with lower confidence. The experiment measures sensitivity to this specific intervention on one image; masking also changes the input distribution. It does not prove that a region causes a semantic concept.',
        mobile_rows(['Removed quadrant','P(Newfoundland)'],[[a['region'],f"{a['target_probability']:.3%}"] for a in inspection['occlusion']]))

    # RESOLUTION, EXERCISES AND BRIDGE.
    body=''
    for j,P in enumerate([32,16,8]):
        x=35+j*380;n=(224//P)**2;tokens=n+1
        body+=t(x,65,f'{P} × {P} patches',30,'c-e')
        size=238
        for r in range(224//P+1):
            pos=r*size/(224//P)
            body+=line(x+pos,100,x+pos,100+size,'line')+line(x,100+pos,x+size,100+pos,'line')
        body+=g(t(x,385,f'{n} patches + CLS',26)+t(x,429,f'{tokens:,}² = {tokens*tokens:,} scores / head',23,'c-a'),1)
    add('patch-cost','What changes when the patch size is halved?',body,'At fixed image size, half the patch width gives four times as many patch rows.',
        'Will attention-score count grow by four or roughly sixteen?','Count rows first, then square the total including CLS.',
        'For a 224×224 image, P=32 gives 50 tokens and 2,500 scores; P=16 gives 197 and 38,809; P=8 gives 785 and 616,225. These counts describe dense attention coefficients per head and layer, not total model FLOPs or actual memory allocated by a fused kernel. Projection and MLP work also contribute. The controls below explore sequence counts only; they do not rerun the pretrained model with incompatible patch sizes.')
    body=t(30,75,'image width / height',30)+t(600,75,'patch width / height',30)+t(30,165,'Predict the token count first.',34,'c-e')
    controls='''<div class="vp-experiment" data-present="manual" data-keep-state><div class="vp-controls"><label>Image size <select id="vl-resolution"><option>224</option><option>384</option></select></label><label>Patch size <select id="vl-patch"><option>32</option><option selected>16</option><option>8</option></select></label></div><output id="vl-cost" aria-live="polite"></output></div>'''
    fr=frame('cost-control','Can you predict the cost before changing the resolution?',body,'Count tokens, then count all query–key pairs.',
        'What happens to the 197-token sequence at resolution 384?\nAsk for a count, change the control, and compare with the prediction.',
        'This calculator concerns shapes, not model accuracy. Changing resolution in an existing ViT may require resizing the position embeddings and compatible preprocessing. Patch-size changes generally require different projection weights or a compatible adaptation.')
    fr=fr.replace('class="frame vp-frame"','class="frame vp-frame vp-experiment-frame"',1).replace('viewBox="0 0 1160 440"','viewBox="0 0 1160 210"')
    extras['cost-control']=fr.replace('<p class="vp-caption">',controls+'<p class="vp-caption">',1)

    body=t(30,70,'q = [1, 0]',36,'c-q')+t(470,70,'already-scaled logits = [ln 2, 0]',31,'c-k')
    body+=g(t(30,185,'values: [2, 0] and [0, 3]',33,'c-v'),1)
    body+=g(t(30,295,'weights = [2/3, 1/3]',35,'c-a'),2)
    body+=g(t(30,402,'message = (2/3)[2,0] + (1/3)[0,3] = [4/3, 1]',30,'c-v'),3)
    add('exercise-message','Your turn: which message arrives?',body,'Pause before revealing the weights and the weighted sum.',
        'Can you calculate the softmax without a calculator?','Exponentiate ln 2 and zero; normalize two and one; multiply the value vectors.',
        'The logits are explicitly supplied after any attention scaling, so do not divide them by √d_k again. Their exponentials are 2 and 1, yielding [2/3,1/3]. The output is a vector, not the index of the most attended source.')
    body=t(30,70,'RGB image: 128 × 128',34,'c-e')+t(30,155,'patch: 16 × 16      D = 64      heads = 4',31)
    body+=g(t(30,255,'64 patch rows + CLS → E is 65 × 64',32,'c-e'),1)
    body+=g(t(30,340,'Q per head: 65 × 16      A per head: 65 × 65',29,'c-q'),2)
    body+=g(t(30,423,'W_patch: 768 × 64      W_O: 64 × 64',30),3)
    add('exercise-shapes','Your turn: trace every important shape',body,'Keep patch count, embedding width and per-head width separate.',
        'What is the difference between a patch’s 768 pixel values and the 64 embedding coordinates?','Let students write all shapes before revealing one line at a time.',
        'There are (128/16)²=64 patches. Each patch has 16×16×3=768 inputs. With CLS, N=65. D=64 is split into four heads of 16 coordinates. The class head shape additionally depends on the number of labels. Batch dimensions are omitted here.')
    body=t(30,70,'1. Shuffle patch contents; keep locations fixed.',29)+t(30,155,'2. Shuffle complete (content + position) rows.',29)
    body+=g(t(30,260,'1 can change the image prediction.',33,'c-a'),1)
    body+=g(t(30,345,'2 preserves CLS under shared self-attention.',31,'c-e'),2)
    body+=g(t(30,421,'Which experiment did our position control perform?',29),3)
    add('exercise-position','Your turn: are these two shuffles equivalent?',body,'Distinguish moving image content from merely reordering the same positioned tokens.',
        'Which shuffle changes what is located at the top left?','Track one patch and its position vector through each operation.',
        'In the second experiment, leave CLS fixed and permute only the already-positioned patch rows. Standard shared self-attention plus a CLS readout is invariant to this permutation (assuming deterministic evaluation and no additional order-dependent mechanism). In the first, content is paired with different locations, so predictions can change.')

    body=image(25,60,340,227)+t(540,90,'CLS → one image label',32,'c-e')
    body+=g(t(540,185,'patch rows → labels over regions',30,'c-v'),1)
    body+=g(t(540,280,'masked patches → learn from images',29,'c-q'),2)
    body+=g(t(540,375,'image rows + text rows → joint tasks',28,'c-e'),3)
    add('next-vision','What can we do with these visual rows next?',body,'Keep the image representation; change the prediction task and training objective.',
        'Which output would need one label, and which would need many spatial answers?','Trace CLS classification, then patch-level readouts and image–text interaction.',
        '<p><a href="vision2.html">Vision II</a> studies learning representations without class labels; <a href="vision3.html">Vision III</a> learns image–text matching with CLIP; <a href="vision4.html">Vision IV</a> connects vision and language. Dense outputs need appropriate heads and spatial reconstruction. A classification ViT does not become a complete detector or segmenter just by exposing patch rows.</p>')
    body=t(30,80,'pixels → shared patch projection + location',34,'c-e')
    body+=g(t(30,185,'Q,K choose weights · V supplies messages',33,'c-v'),1)
    body+=g(t(30,290,'multiple heads → W_O → residual + row MLP',32,'c-d'),2)
    body+=g(t(30,395,'updated CLS → class scores → loss → learning',32,'c-a'),3)
    add('closing','Can you now explain the whole photograph-to-answer path?',body,'The same attention story from Parts I–III, now attached to pixels we can inspect.',
        'Which step is new, and which steps did we already know?','Ask students to narrate the computation from input to loss without reading equations.',
        '<p>Student materials: <a href="notebooks/vision/03_vision_transformer_lab.ipynb">complete worked lab</a>, <a href="notebooks/vision/train_small_vit.py">train the full small ViT</a>, <a href="notebooks/vision/inspect_real_vit.py">reproduce attention and occlusion</a>, and <a href="figures/vision1/training.json">training results</a>. The lecture, figures, numeric worksheet and notebook are generated from the same parameters and saved measurements.</p>')

    # Put the new figures into one deliberate lecture sequence.
    e=extras
    sections=[
      ('What clues let us recognize the animal?',[groups[1][0],groups[1][1],e['bridge-text'],e['patch-context'],e['cnn-context']]),
      ('How does a photograph become a sequence?',[groups[1][2],e['rgb-flatten'],groups[1][3],e['projection-size']]),
      ('Can the same patches make different images?',[groups[2][0],e['position-question'],groups[2][1],e['patch-matrix'],groups[2][2]]),
      ('What should the summary row read?',[e['why-cls'],e['image-mask'],e['heads-question'],e['all-qkv'],groups[3][0],groups[3][1],groups[3][2],e['weight-message']]),
      ('What does a second head contribute?',[groups[3][3],groups[3][4],e['all-receivers'],groups[4][0]]),
      ('How do messages become an image answer?',[groups[4][1],groups[4][2],e['one-update'],groups[4][3]]),
      ('What else belongs in a full ViT?',[groups[5][0],e['layernorm'],e['mlp-row'],e['depth'],groups[5][1]]),
      ('Can we build the model ourselves?',[e['code-patch'],e['code-attention'],e['code-block'],e['code-model'],e['code-train']]),
      ('Will it learn beyond two clean images?',[e['training-data'],e['learning-curves'],e['trained-position-control']]),
      ('What does the actual photograph produce?',[e['real-input'],groups[6][0],e['real-cat'],e['three-phases']]),
      ('What can we measure inside the real model?',[e['real-heads'],e['real-depth'],e['real-patch-query'],e['occlusion']]),
      ('How much does a finer patch grid cost?',[e['patch-cost'],e['cost-control']]),
      ('Can you work it out without the slides?',[e['exercise-message'],e['exercise-shapes'],e['exercise-position']]),
      ('Where can we take visual attention next?',[e['next-vision'],e['closing']])]
    from vision1_pedagogy import expand
    sections=expand(b,sections)
    from vision1_connections import connect
    sections=connect(b,sections)
    from vision1_video_lessons import augment
    sections=augment(b,sections)
    ordered=[]
    for n,(title,frames) in enumerate(sections,1):
        original_section(n,title,frames)
        for number,markup in enumerate(frames,1):
            key=re.search(r'class="frame[^"]*" id="([^"]+)"',markup).group(1)
            meta=next(x for x in FRAMES if x['id']==key).copy()
            meta.update(section=f's{n:02}',frame=number)
            ordered.append(meta)
    config={'part':1,'partLabel':'Vision I','series':'Vision to language',
      'title':'An image becomes a sequence',
      'subtitle':'Real photographs, a complete four-patch calculation, and a Vision Transformer we can train and inspect.',
      'audience':'Students who have completed the text attention Parts I–III.',
      'durationLabel':'A step-by-step lecture sequence with a worked lab.',
      'hook':'What can one patch borrow from the rest of its image?',
      'centralLabel':'Follow one computation','central':r'\text{pixels}\to E\to Q,K,V\to A\to H\to E\prime\to p(\text{class})',
      'sections':[{'id':f's{i+1:02}','title':s[0],'lit':''} for i,s in enumerate(sections)],
      'chain':[{'section':f's{i+1:02}','label':s[0]} for i,s in enumerate(sections)],
      'objects':['e','q','k','v','a','d','ep'],
      'objectSections':{'e':'s03','q':'s04','k':'s04','v':'s04','a':'s04','d':'s05','ep':'s06'},
      'provenance':'We use small, chosen numbers for the four-patch calculation. Then we train a complete small ViT on noisy images. Finally, we run a pretrained ViT on the photographs. The code and saved results keep these three examples separate.',
      'prev':{'label':'Part 4: Cross-attention: translate one phrase','href':'part4.html'},
      'next':{'label':'Vision II: Learn visual representations without class labels','href':'vision2.html'},
      'index':{'label':'Series home','href':'index.html'},
      'notation':'vision1','footer':'Pixels become rows; attention adds context; a learned readout answers the image question.',
      'sectionDirectory':'sections-vision1','toyFile':'vision1-worksheet.json','runtimeFile':'vision1-lesson.js',
      'legacyVisionRuntime':False,'syntaxHighlighting':True,'output':'vision1.html'}
    (SRC/'part5.json').write_text(json.dumps(config,indent=2)+'\n')
    (ASSETS/'frame-manifest.json').write_text(json.dumps(ordered,indent=2)+'\n')
    return len(ordered)
