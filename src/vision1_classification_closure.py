"""Finish image classification with CLS choices, multiple heads and a reverse pass.

All new numbers are from the existing hand-chosen worksheet. No training job runs.
"""
import json
import re
from html import escape
from vision1_gradients import checked_trace


def refine(b, sections):
    t,g,rect,arrow,line,image,crop,frame,mobile_rows=[b[k] for k in
        ['t','g','rect','arrow','line','image','crop','frame','mobile_rows']]
    trace=checked_trace(); r=trace['forward']; d=trace['gradients']; update=trace['single_update']
    (b['ASSETS']/'backward-worksheet.json').write_text(json.dumps(trace,indent=2)+'\n')
    new={}
    from vision1_model_journey import STAGES
    def ribbon(key):
        focus=([7] if key=='backward-scores' else [6,7] if key=='backward-cls'
               else [2,3] if key=='backward-patches' else [4,5] if key=='backward-full-block' else [4])
        drawing=''
        for i,name in enumerate(STAGES):
            x=8+i*144;selected=i in focus;c='c-a' if selected else 'ink-3'
            drawing+=rect(x,8,132,35,c,'transparent',4)+t(x+66,31,name,17,c,'middle',700 if selected else 500)
            if i<7:drawing+=arrow(x+142,25,x+133,25,'c-a')
        return '<div class="vp-pathbar"><p class="vp-path-label">Backward pass · follow the loss gradient toward the inputs</p><svg viewBox="0 0 1160 50" role="img" aria-label="Backward model route">'+drawing+'</svg></div>'
    def num(x):return f'{x:.3f}'.replace('-','−')
    def vec(a):return '['+', '.join(num(x) for x in a)+']'
    def box(x,y,w,labels,c='c-e',h=74):
        labels=[labels] if isinstance(labels,str) else labels
        out=rect(x,y,w,h,c,'t-e' if c=='c-e' else 'transparent',6)
        for j,s in enumerate(labels):out+=t(x+w/2,y+h/2+(j-(len(labels)-1)/2)*30+8,s,23,c,'middle')
        return out
    def flow(labels,y=115):
        w=1080/len(labels)-38;out=''
        for i,label in enumerate(labels):
            x=35+i*(w+38);out+=box(x,y,w,label)
            if i<len(labels)-1:out+=arrow(x+w+4,y+37,x+w+32,y+37)
        return out
    def add(key,title,body,caption,question,notes,prose='',mobile=''):
        if not mobile:mobile='<p>'+escape(caption)+'</p><p>'+escape(question)+'</p>'
        for meta in b['FRAMES']:
            if meta['id']==key:meta.update(title=title,caption=caption,notes=question+'\n'+notes)
        new[key]=frame(key,title,body,caption,question+'\n'+notes,prose,mobile)
        if key.startswith('backward-') and key!='backward-route':
            new[key]=new[key].replace('class="frame vp-frame','class="frame vp-frame vp-with-route',1).replace('<div class="vp-figure',ribbon(key)+'<div class="vp-figure',1)
        return new[key]

    body=box(35,65,250,['learned starting CLS','same 192 numbers'],'c-q')
    body+=image(425,35,210,140,b['PHOTO'])+image(425,240,210,140,b['CAT'])
    body+=arrow(285,100,400,100,'c-q')+line(330,100,330,310,'c-q')+arrow(330,310,400,310,'c-q')
    body+=box(725,60,370,['blocks read dog patches','dog-dependent final CLS'],'c-q')+arrow(635,103,716,103)
    body+=box(725,268,370,['blocks read cat patches','cat-dependent final CLS'],'c-q')+arrow(635,310,716,310)
    body+=t(35,425,'One learned starting vector; a different final summary for each image.',27)
    add('cls-shared-start','Does every image start with the same CLS?',body,
        'Yes. CLS starts as one shared learned vector. The patches differ, so attention produces a different final summary for each image.',
        'If the starting CLS is identical, where does the image-specific information enter?',
        'Trace both branches from the same starting parameter to different patch contexts. Training changes the shared parameter; inference updates the activations.',
        'CLS means classification token. It is neither the dog label nor a different learned parameter for every photograph. '
        'Its first-block query is shared across images in this model, while patch keys and values depend on the image. Later CLS queries can also become image-dependent.',
        '<p>Same learned starting CLS + dog patches → dog-dependent final CLS.</p><p>Same learned starting CLS + cat patches → cat-dependent final CLS.</p>')

    body=t(30,38,'Two valid ways to produce one image representation',30)
    body+=flow([['196 patches','+ 1 CLS'],['197 rows','through blocks'],['read final CLS','1 × 192'],['class head','1 × C']],95)
    body+=g(flow([['196 patches','no CLS'],['196 rows','through blocks'],['mean of final rows','1 × 192'],['class head','1 × C']],285),1)
    add('cls-without','What would happen if we built a ViT without CLS?',body,
        'The patch rows still exchange information. Average their final representations to obtain one image vector, then classify it. Train the model with this readout from the start.',
        'Which dimension disappears when we average 196 rows?',
        'Trace both complete paths. The row axis is averaged; the feature width remains 192. C is the number of classes.',
        'A classification token is an architectural choice. Global average pooling is another. Normalization placement follows the chosen model. '
        'Deleting CLS from an existing pretrained checkpoint changes both its attention context and its readout; it does not guarantee equivalent predictions. '
        '<a href="https://arxiv.org/abs/2010.11929">The original ViT paper discusses CLS and global average pooling in its appendix.</a>',
        mobile_rows(['Readout','Rows in blocks','Image vector'],[['CLS','197','final CLS: 192 values'],['Mean pooling','196','mean final patch row: 192 values']])+'<p>Train for the selected readout. Removing CLS from a pretrained CLS model is not a prediction-preserving edit.</p>')

    body=t(35,48,'Illustrative possibilities · learned heads are not assigned these jobs',26,'ink-2')
    for j,(label,indices,c,question) in enumerate([
        ('Nearby texture',[9,10],'c-q','Does this fur continue nearby?'),
        ('Animal parts',[5,6],'c-v','Which face clues belong together?'),
        ('Wider context',[0,3],'c-k','How does this region fit the scene?')]):
        x=35+j*380
        body+=t(x,105,label,29,c)
        for k,ix in enumerate(indices):body+=crop(x+k*162,145,144,96,ix,'head-intuition')
        body+=arrow(x+153,264,x+153,305,c)+box(x,313,330,['a different weighted','message'],c)
        body+=t(x+165,430,question,20,c,'middle')
    add('heads-visual-roles','What might different heads look for in this photograph?',body,
        'Different heads can learn different matching patterns and different messages. Local texture, relationships between parts and wider context are possible intuitions. These are hypotheses, not measured head labels.',
        'Could one mixture preserve all the different clues that matter for this image?',
        'Connect to coat reading colour and material in Part III. Each visual role is a possible interpretation, not a hand-written rule or guaranteed specialization.',
        'Every head receives the same input rows. Its own Q, K and V projections can extract different matching and message features. '
        'Heads need not have neat names: they may overlap, mix several cues, or be redundant. We later inspect measured maps without treating them as proof of a semantic role.',
        mobile_rows(['Possible pattern','Example question'],[['Nearby texture','Does this fur continue nearby?'],['Animal parts','Which face clues belong together?'],['Wider context','How does this region fit the scene?']])+'<p>Illustrative hypotheses. The classification loss learns the head parameters.</p>')

    body=box(35,155,180,['same E','197 × 192'])
    for j,(y,c) in enumerate([(45,'c-q'),(178,'c-v'),(311,'c-k')]):
        body+=line(215,192,255,192)+line(255,192,255,y+37)+arrow(255,y+37,290,y+37,c)
        body+=box(300,y,295,[f'Head {j+1}: its own Q, K, V','197 × 64 message'],c)
        body+=arrow(595,y+37,715,y+37,c)+line(715,y+37,715,216,c)
    body+=box(755,178,345,['concatenate → W_O','197 × 192'])+arrow(715,216,746,216)
    body+=t(770,335,'3 × 64 = 192',32,'c-e')
    add('heads-independent','Three heads read the same rows in three learned ways',body,
        'Each head produces its own attention matrix and message. Join the messages, then mix them with the output projection. The number of heads is independent of the number of image classes.',
        'Would a two-class dog/cat task require exactly two heads?',
        'No. Compare three 64-wide heads with two class scores. Keep head count, feature width and class count separate.',
        'For the real checkpoint, each head has Q,K,V of shape 197×64 and an attention matrix of shape 197×197. '
        'A class head can later output two scores or 1,000 scores independently of the three attention heads.',
        '<p>Same E (197×192) → three separately projected heads.</p><p>Each message: 197×64. Concatenate: 197×192. Output projection: 192→192.</p><p>Three heads does not mean three classes.</p>')

    # A reverse pass through the exact worksheet already used on the slides.
    body=flow([['patches + CLS','input rows E'],['attention + residual','updated rows'],['read CLS','class scores'],['class softmax','label loss']],105)
    body+=g(t(580,270,'Backward: how would a small change affect the loss?',30,'c-a','middle'),1)
    for i in range(3):body+=g(arrow(1010-i*280,330,805-i*280,330,'c-a'),1)
    body+=g(t(580,412,'Gradients flow back; the optimizer changes parameters afterward.',29,'c-a','middle'),2)
    add('backward-route','Run the same computation backward from the label loss',body,
        'The forward pass made a prediction. The backward pass measures how each parameter influenced its loss. Use the same four-patch worksheet and the same known label: Across the top.',
        'Does backward itself change any learned weight?',
        'No. It computes derivatives. The later optimizer step uses those derivatives to change parameters. The worksheet omits LayerNorm and MLP; we restore them in the full-block diagram.',
        'All following values use the exact existing worksheet, with chosen weights and cross-entropy for Across the top. '
        '<a href="figures/vision1/backward-worksheet.json">Complete analytic gradient trace</a>. '
        'Central finite differences check every one of its 116 parameter coordinates. No model-training experiment is run.',
        '<p>Forward: patches and CLS → attention → residual → read CLS → scores → loss.</p><p>Backward follows the dependencies in reverse. Optimizer step then changes parameters.</p>')

    body=t(35,60,'Class order: [Across the top, Down the left]',29,'ink-2')
    for j,(name,val,c) in enumerate([('probabilities p',vec(r['p']),'c-e'),('known target y','[1, 0]','c-e'),('score gradient p − y',vec(d['dz']),'c-a')]):
        body+=g(t(35,142+88*j,name,29,c)+t(555,142+88*j,val,34,c),max(0,j))
    body+=g(t(35,416,'Raise the correct score; lower the competing score.',30,'c-a'),2)
    add('backward-scores','First ask how the two class scores should change',body,
        'For softmax followed by cross-entropy, the score gradient is p − y. A negative gradient means increasing that score would locally reduce this example’s loss.',
        'Why is the gradient of the correct score negative?',
        'A parameter update subtracts its gradient. Separate a derivative from the later update. Both class scores receive a signal.',
        'For z=logits and L=−log p_correct, ∂L/∂z=p−y. These numbers use full-precision probabilities; the slides round to three decimals.',
        mobile_rows(['Quantity','Two numbers'],[['p',vec(r['p'])],['y','[1, 0]'],['∂L/∂z',vec(d['dz'])]]))

    u=d['dU'][0][0]; first=d['dz'][0]; second=d['dz'][1]
    body=t(35,52,'In our worksheet, only CLS coordinate 1 scores the classes.',28)
    body+=box(35,116,250,['CLS coordinate u₁',num(r['U'][0][0])],'c-q')
    body+=arrow(285,153,465,99)+t(365,105,'× (−4)',24,'ink-2','middle')+box(485,61,250,['Across score','gradient '+num(first)])
    body+=arrow(285,153,465,241)+t(365,250,'× 4',24,'ink-2','middle')+box(485,203,250,['Down score','gradient '+num(second)])
    body+=g(t(795,138,'Add both paths',29,'c-a')+t(795,199,'(−4) × '+num(first),27,'c-a')+t(795,245,'+ 4 × '+num(second),27,'c-a')+t(795,300,'= '+num(u),34,'c-a'),1)
    body+=g(t(35,412,'Gradient of the final CLS row: '+vec(d['dU'][0]),30,'c-a'),1)
    add('backward-cls','The class loss sends a gradient into the final CLS row',body,
        'Each score contributes through its classifier weight. Add those contributions to find the gradient of the CLS representation. The other coordinates have zero gradients here because our chosen classifier ignores them.',
        'Can an image label train the summary without a separate CLS target?',
        'Follow the two score branches backward into the same coordinate. The image label already supplies the learning signal.',
        'For row-vector z=uW_class, the gradient into u is W_class·(p−y) when represented as a column. In the implementation dU[0]=W_class@dz. '
        'A realistic classifier generally uses all coordinates; the zero entries belong only to this hand-chosen worksheet.',
        '<p>∂L/∂u₁ = (−4) × '+num(first)+' + 4 × '+num(second)+' = '+num(u)+'.</p><p>Full CLS gradient: '+vec(d['dU'][0])+'.</p>')

    h1=r['heads'][0]['H'][0][0];h2=r['heads'][1]['H'][0][0]
    body=t(35,60,'Forward: u₁ = starting CLS₁ + head₁ message₁ − head₂ message₁',28)
    for x,label,value,grad,c in [(35,'starting CLS₁',r['E'][0][0],u,'c-q'),(410,'head₁ message₁',h1,u,'c-v'),(785,'head₂ message₁',h2,-u,'c-k')]:
        body+=box(x,125,325,[label,num(value)],c)
        body+=g(arrow(x+163,312,x+163,215,'c-a')+t(x+163,366,'gradient '+num(grad),29,'c-a','middle'),1)
    body+=g(t(35,428,'The residual gives a direct path. Both head messages also receive a signal.',26),2)
    add('backward-heads','The residual and both heads receive the learning signal',body,
        'Our output projection adds Head 1’s first coordinate and subtracts Head 2’s. Backpropagation follows those signs. The residual also passes a gradient directly to the starting CLS row.',
        'Why does Head 2 receive the opposite sign?',
        'Point to the minus sign in the forward expression. Reverse the existing W_O operation; do not invent an independent target for either head.',
        'The direct gradient to E is only one contribution. The Q, K and V branches will add more. Gradients from all downstream paths must be summed.',
        mobile_rows(['Path','Gradient'],[['direct residual',num(u)],['Head 1 first coordinate',num(u)],['Head 2 first coordinate',num(-u)]]))

    a=r['heads'][0]['A'][0];vals=r['heads'][0]['V'];grad=d['heads'][0]
    body=t(35,40,'Head 1: message = Σⱼ aⱼ vⱼ       incoming gradient: ['+num(u)+', 0]',28,'c-v')
    headers=['source','aⱼ','vⱼ,₁','∂L/∂vⱼ,₁','∂L/∂aⱼ']
    xs=[35,260,460,700,985]
    for x,h in zip(xs,headers):body+=t(x,106,h,25,'ink-2')
    for j in range(5):
        y=170+j*48
        data=['CLS' if j==0 else f'P{j}',num(a[j]),num(vals[j][0]),num(grad['dV'][j][0]),num(grad['dA'][0][j])]
        for k,(x,s) in enumerate(zip(xs,data)):body+=g(t(x,y,s,27,'c-a' if k>2 else 'ink'),1 if k>2 else 0)
    add('backward-values','Learning can change what a source sends and how much we read',body,
        'For message Σⱼ aⱼvⱼ, the value gradient is aⱼ times the incoming gradient. The attention-weight gradient is its dot product with vⱼ. The loss reaches source patches through both routes.',
        'Do patch values receive gradients even though the classifier reads only CLS?',
        'Calculate one P1 product. Then notice the gradients in patch rows that have no direct classification readout.',
        'For this single-block worksheet only the final CLS row is directly supervised. Its attention uses patch keys and values, giving those source rows nonzero gradients. '
        'Patch query gradients are zero here because their output rows do not feed another block or the head. With more blocks they can affect later CLS states.',
        mobile_rows(headers,[['CLS' if j==0 else f'P{j}',num(a[j]),num(vals[j][0]),num(grad['dV'][j][0]),num(grad['dA'][0][j])] for j in range(5)]))

    body=t(35,50,'Attention weights came from a softmax over source scores.',29)
    body+=box(35,116,285,['source scores s','q · k / √2'],'c-k')+arrow(320,153,425,153)
    body+=box(445,116,285,['weights a','sum to one'],'c-a')+arrow(730,153,830,153)
    body+=box(850,116,275,['weighted values','message'],'c-v')
    body+=g(t(35,264,'g = gradient of the attention weights',27,'c-a')+t(35,327,'∂L/∂sⱼ = aⱼ (gⱼ − Σₖ aₖgₖ)',34,'c-a'),1)
    body+=g(t(35,407,'Head 1 score gradients: '+vec(grad['dS'][0]),27,'c-a'),2)
    add('backward-softmax','The source scores compete for attention weight',body,
        'Softmax couples the source weights: increasing one score changes all weights. Subtract the weighted mean gradient, then multiply by each source weight to get the score gradients.',
        'Why can we not update every attention weight independently?',
        'They must stay normalized. The score gradients sum to zero; adding the same constant to every score leaves softmax unchanged.',
        'Here g denotes ∂L/∂a, not a new parameter. The softmax Jacobian gives ∂L/∂sⱼ=aⱼ(gⱼ−Σₖaₖgₖ). '
        'Use the computed dS for both the query and key branches. Attention weights are activations; the optimizer updates projection parameters.',
        '<p>Let g = ∂L/∂a. Then ∂L/∂sⱼ = aⱼ(gⱼ − Σₖaₖgₖ).</p><p>Head 1 score gradients: '+vec(grad['dS'][0])+'.</p>')

    body=flow([['starting rows E','5 × 4'],['Q and K','5 × 2 each'],['scores QKᵀ / √2','5 × 5'],['attention softmax','5 × 5']],80)
    body+=g(t(35,235,'For CLS: ∂L/∂q₀ = Σⱼ (∂L/∂s₀ⱼ) kⱼ / √2',29,'c-a'),1)
    body+=g(t(35,299,'Head 1 query gradient: '+vec(grad['dQ'][0]),32,'c-a'),1)
    body+=g(t(35,362,'Each key also receives: (∂L/∂s₀ⱼ) q₀ / √2',29,'c-k'),2)
    body+=g(t(35,427,'Through Q = EW_Q, gradients reach both E and the learned W_Q.',27),2)
    add('backward-qk','Follow the score gradients into queries and keys',body,
        'Each dot product depends on both its query and its key. Reverse those products, then reverse the shared linear projections. The loss can improve which sources match the receiver.',
        'What teaches a head which visual clues are useful?',
        'The class loss trains the projections through this chain. No person labels a head as the eye head or fur head.',
        'Matrix forms: dQ=dS K/√d_k; dK=dSᵀ Q/√d_k. For Q=EW_Q, dW_Q=EᵀdQ and the contribution to dE is dQ W_Qᵀ. '
        'Add the K and V contributions and the residual contribution to obtain the complete gradient into E.',
        '<p>dQ = dS K / √2; dK = dSᵀ Q / √2.</p><p>Head 1 CLS-query gradient: '+vec(grad['dQ'][0])+'.</p><p>dW_Q = EᵀdQ. Gradients into E also arrive through K, V and the residual.</p>')

    body=t(35,50,'Change one real worksheet parameter; keep all other weights fixed.',28)
    body+=box(35,105,340,['Head 1 W_Q[4th row, 1st col]','starting value = 1'],'c-q')
    body+=g(t(445,138,'gradient = '+num(update['gradient']),33,'c-a')+t(445,199,'learning rate = 0.1',30,'ink-2'),1)
    body+=g(t(35,286,'new weight = 1 − 0.1 × '+num(update['gradient'])+' = '+num(update['after']),34,'c-q'),2)
    body+=g(t(35,377,'Run forward again: loss '+num(update['loss_before'])+' → '+num(update['loss_after']),33,'c-a'),3)
    add('backward-one-weight','Use the gradient to change one query weight',body,
        'We update one query-projection weight with a small SGD step, then repeat the same forward pass. This hand calculation changes the attention mechanism itself and lowers this example’s loss.',
        'Which quantity must be recomputed after the weight changes?',
        'Q, scores, attention weights, messages, class scores and loss. This one-example improvement says nothing about held-out accuracy.',
        'Array index W_Q[3,0] refers to the fourth row and first column of Head 1’s query matrix. The original value 1 becomes '+str(update['after'])+'. '
        'This is a deterministic arithmetic illustration, not a training run or fitted model. Every derivative is checked independently with central finite differences.',
        '<p>w = 1; gradient = '+num(update['gradient'])+'; rate = 0.1.</p><p>w_new = '+num(update['after'])+'.</p><p>Recomputed loss: '+num(update['loss_before'])+' → '+num(update['loss_after'])+'.</p>')

    body=t(35,45,'Input row for a patch: eᵢ = xᵢW_patch + b_patch + pᵢ',31)
    for j in range(4):
        x=35+j*280
        body+=box(x,102,240,[f'patch P{j+1}', 'row gradient dEᵢ'],'c-a')
        body+=arrow(x+120,181,580,278,'c-a')
    body+=g(box(290,290,580,['one shared projection gradient','dW_patch = Σᵢ xᵢᵀ dEᵢ'],'c-a',88),1)
    body+=g(t(35,430,'CLS gets dE₀. Each position gets its row gradient. Pixels are fixed input data.',26),2)
    add('backward-patches','Every patch contributes to the same projection weights',body,
        'The patch projection is shared, so add its gradient contributions across patches. Position vectors and the starting CLS vector also receive gradients. The usual training step changes parameters, while the photograph stays fixed.',
        'Do we now learn 196 separate patch projections?',
        'No. Contributions accumulate into one parameter matrix. In a batch, contributions also accumulate across images before the optimizer step.',
        'For the worksheet, dW_patch=XᵀdE_patches and db_patch is the sum of patch-row gradients. dCLS=dE₀; dP=dE. '
        'Computing pixel derivatives is possible, but model training normally optimizes model parameters rather than modifying the dataset photographs.',
        '<p>eᵢ=xᵢW_patch+b_patch+pᵢ.</p><p>dW_patch=ΣᵢxᵢᵀdEᵢ; db_patch=ΣᵢdEᵢ. dCLS=dE₀; each position receives its row gradient.</p>')

    body=t(35,48,'Full pre-LayerNorm block · the same reverse rule on every branch',28)
    nodes=[(35,155,'E'),(226,172,'LN → attention'),(460,90,'+'),(623,177,'LN → MLP'),(960,90,'+')]
    for x,w,label in nodes:body+=box(x,154,w,label,'c-e')
    for (x,w,_),(nx,_,__) in zip(nodes,nodes[1:]):body+=arrow(x+w+4,190,nx-7,190)
    body+=line(105,154,105,97,'c-v')+line(105,97,505,97,'c-v')+arrow(505,97,505,151,'c-v')
    body+=line(583,190,583,97,'c-v')+line(583,97,1005,97,'c-v')+arrow(1005,97,1005,151,'c-v')
    body+=g(arrow(1075,291,690,291,'c-a')+arrow(540,291,150,291,'c-a'),1)
    body+=g(t(35,355,'At each +: send the gradient down both incoming paths.',29,'c-a'),1)
    body+=g(t(35,411,'Reverse Linear → GELU → Linear, LayerNorm, and attention; add contributions.',25),2)
    add('backward-full-block','The full block sends gradients through both residual branches',body,
        'Restore the operations omitted from the worksheet. Backpropagate through every block, including its MLP and LayerNorm. At an addition, both input paths receive the incoming gradient; contributions meeting upstream are added.',
        'Does a residual connection prevent the attention or MLP weights from learning?',
        'No. It supplies a direct path alongside the learned branch. Trace the MLP branch and direct path, then the attention branch and its direct path.',
        'For Y=X+F(X), dX=dY+J_F(X)ᵀdY in column notation. The MLP forward order is Linear→GELU→Linear; backward reverses that dependency order. '
        'LayerNorm is differentiable and its learned scale/bias can receive gradients. Final LayerNorm, the readout and classifier are also included before entering the last block backward.',
        '<p>Forward: U=E+Attention(LN(E)); Y=U+MLP(LN(U)).</p><p>Backward: split at each residual addition, reverse both branches and sum contributions. Continue through earlier blocks to patch projection, positions and CLS.</p>')

    # CNN comparison after students know the full Transformer block.
    body=t(290,45,'Ordinary 3 × 3 convolution',29,'c-v','middle')+t(875,45,'Global self-attention',29,'c-q','middle')
    body+=image(60,92,430,287)+image(660,92,430,287)
    for k,size in enumerate([64,108,152]):body+=rect(275-size/2,230-size/2,size,size,'c-v','transparent',0)
    body+=t(285,420,'3 × 3 → 5 × 5 → 7 × 7 view',27,'c-v','middle')
    body+=rect(830,215,38,38,'c-q','transparent',0)
    for x,y in [(694,124),(1055,126),(708,339),(1035,342)]:body+=arrow(x,y,847,234,'c-q')
    body+=t(875,420,'One row can read distant rows',27,'c-q','middle')
    add('cnn-receptive-field','How does information travel through a CNN or a ViT?',body,
        'A stack of local convolutions expands the receptive field. Global self-attention can connect distant patch rows within one block. Both can use broad image context and learn useful features for classification.',
        'Can a CNN eventually use information from the whole image?',
        'Yes. The left diagram shows three stride-one 3×3 convolutions, giving nominal receptive fields 3,5,7. Pooling or strides change the calculation. Right arrows are illustrative.',
        'The left rectangles are a schematic receptive-field illustration, not pixel-exact filters on the resized display photograph. '
        'Ordinary convolution applies learned kernels shared over locations; nonlinear stacks have input-dependent overall responses. Attention computes its mixing weights from the current query and source keys.',
        mobile_rows(['Mechanism','Context'],[['Local convolution','Receptive field grows through layers'],['Global self-attention','One layer can compare every patch row']])+'<p>Both architectures can use the whole image.</p>')

    body=t(35,47,'Same classification goal; different choices about spatial structure',29)
    rows=[('Spatial mixing','local kernel at each location','query–key weights across rows'),('Shared parameters','kernel weights across locations','Q/K/V and MLP across rows'),('Image summary','often global average pooling','CLS or mean pooling'),('Position','grid neighborhoods are explicit','position signals supplement rows'),('Learning signal','image-label cross-entropy','image-label cross-entropy')]
    for x,h in [(35,'Question'),(380,'CNN'),(780,'ViT')]:body+=t(x,112,h,29,'c-e')
    for j,row in enumerate(rows):
        y=180+j*55
        for x,text in zip([35,380,780],row):body+=t(x,y,text,21,'ink')
    add('cnn-classifier-parallel','Both models turn visual features into class scores',body,
        'A CNN and a ViT can share the same dataset, labels, class head shape and loss. Their main difference here is how they build and exchange visual features before the image readout.',
        'Which parts of our training diagram would stay the same with a CNN encoder?',
        'Keep dataset, classifier, label loss and optimizer. Replace the visual feature extractor and explain its spatial assumptions.',
        'This comparison uses a conventional convolutional classifier and a plain global-attention ViT. Many hybrids exist. '
        'Neither head count nor feature-channel count equals the class count. Architecture choice does not guarantee better accuracy; data, pretraining, computation and evaluation matter.',mobile_rows(['Question','CNN','ViT'],rows))

    body=t(35,46,'The same classification model: forward to a loss, backward to parameters',28)
    names=[['Image','224 × 224 × 3'],['Patch layer','196 × 192'],['CLS + position','197 × 192'],['12 blocks','197 × 192'],['LN, read CLS','1 × 192'],['Class head','1 × C']]
    for i,labels in enumerate(names):
        x=25+i*188;body+=box(x,109,168,labels,h=82)
        if i<5:body+=arrow(x+171,151,x+183,151)
    body+=g(box(965,298,175,['class loss','one number'],'c-a')+arrow(1050,193,1050,290,'c-a'),1)
    body+=g(t(915,244,'known label y',24,'c-a','end')+arrow(943,236,1010,289,'c-a'),1)
    for i in range(5):body+=g(arrow(937-i*180,338,798-i*180,338,'c-a'),2)
    body+=g(t(35,414,'Optimizer: update trainable weights, positions and CLS using their gradients.',27,'c-a'),2)
    add('classification-training-map','One diagram connects the prediction and the learning',body,
        'Forward follows the blue route to class scores. A known label defines the loss. Backward follows the dependencies to trainable parameters; the optimizer then updates them. C is the chosen number of classes.',
        'Which part of this diagram is absent when we only want a prediction?',
        'Inference needs no target label, backward pass or optimizer step. Training uses the same forward computation before adding the loss and reverse path.',
        'For the saved ImageNet checkpoint C=1,000. In the proposed dog/cat adaptation C=2 and the new class head requires training. '
        'In a batch, each displayed activation shape gains a leading batch dimension. The backward arrows summarize the chain rule through the exact forward dependencies.',
        '<p>Image → patch layer → CLS and positions → 12 Transformer blocks → final normalization and CLS readout → C class scores.</p><p>Known label → class loss → backward through the model → optimizer updates trainable parameters.</p>')

    # A procedure for the pet task replaces a lengthy synthetic training detour.
    body=image(35,72,265,177,b['PHOTO'])+image(35,262,265,177,b['CAT'])
    body+=box(400,105,305,['pretrained ViT encoder','final CLS: 192 features'])+arrow(300,158,390,158)
    body+=line(334,350,334,158)+arrow(300,350,334,350)
    body+=g(box(790,105,340,['new nn.Linear(192, 2)','scores: [dog, cat]'])+arrow(705,143,780,143),1)
    body+=g(t(410,293,'384 weights + 2 biases = 386 parameters',28,'c-e')+t(410,352,'Keep the learned visual features to begin with.',27),2)
    add('pets-head','Return to the pets: give the encoder a two-class head',body,
        'For our dog/cat task, replace the ImageNet head with a new two-output linear layer. The encoder supplies a 192-number image representation. The new 386 head parameters must learn from labelled pet images.',
        'Why can we not simply rename two of the old 1,000 outputs?',
        'We have changed the label vocabulary. The new head learns a new mapping; the existing pretrained probabilities do not measure this proposed pet classifier.',
        'This is a proposed adaptation workflow, not a completed Pets experiment. The shown photo trace still belongs to the existing 1,000-class pretrained model. '
        'For B images, readout shape B×192 maps to B×2 logits. The number of patches and attention heads need not change.',
        '<p>Pet image → pretrained ViT → CLS features (192) → Linear(192,2) → dog/cat scores.</p><p>192×2+2=386 new trainable parameters. This is a proposed workflow, with no claimed benchmark results.</p>')

    body=t(35,45,'Start simple: train the new head',30)
    body+=flow([['photograph','fixed data'],['ViT encoder','FROZEN'],['192 features','fixed encoder output'],['dog/cat head','TRAINABLE']],100)
    body+=g(arrow(1090,280,880,280,'c-a')+t(860,325,'loss → head gradients',28,'c-a','end'),1)
    body+=g(t(35,392,'Next option: unfreeze selected blocks and fine-tune them with the head.',28),2)
    add('pets-frozen','Choose which parameters the optimizer may change',body,
        'First freeze the encoder and fit the new class head. If needed, unfreeze selected blocks and fine-tune. The forward path stays recognizable; the set of trainable parameters changes.',
        'When only the head is trained, do the Q/K/V weights change?',
        'No. Highlight the frozen encoder. Then describe unfreezing the final block as an optional next step, with validation guiding the decision.',
        'Head-only training is often called a linear probe. With a frozen encoder, its features may be cached using a fixed preprocessing transform. '
        'Fine-tuning lets the chosen encoder parameters receive updates. Use a held-out validation set to choose settings; no training is run for these slides.',
        mobile_rows(['Mode','Updated parameters'],[['Head only','Linear(192,2) weights and biases'],['Fine-tuning','Head plus selected encoder parameters']]))

    body=flow([['batch of images','B × 3 × 224 × 224'],['ViT + new head','B × 2 scores'],['known labels','B targets'],['mean cross-entropy','one loss']],100)
    body+=g(t(35,280,'zero gradients → forward → loss → backward → optimizer step',29,'c-a'),1)
    body+=g(t(35,365,'Repeat on training batches. Evaluate separately with weights fixed.',28),2)
    add('pets-training-step','One batch follows the same forward and backward paths',body,
        'Pair every image with its dog/cat label. Compute the average loss for the batch, backpropagate once, and update the trainable parameters. Repeat this procedure over training batches.',
        'What is shared across the images in a batch?',
        'The model parameters are shared. Each image has its own activations and attention matrix; attention does not mix different images in the batch.',
        'The criterion consumes logits and class-index targets; a standard cross-entropy implementation includes log-softmax internally. '
        'zero_grad clears accumulated parameter gradients. backward computes gradients. step changes parameters. Evaluation uses fixed parameters and no optimizer step.',
        '<p>Images B×3×224×224 → logits B×2; labels B → mean loss.</p><p>zero_grad → forward → loss → backward → step.</p><p>Attention stays within each image.</p>')

    body=t(35,48,'Before fitting: separate training, validation and test images',29)
    for j,(label,purpose,c) in enumerate([('TRAIN','update weights','c-e'),('VALIDATION','choose settings','c-v'),('TEST','final held-out check','c-q')]):
        x=35+380*j;body+=box(x,108,330,[label,purpose],c)
    body+=g(t(35,270,'Inspect: dog → dog    dog → cat    cat → dog    cat → cat',29),1)
    body+=g(t(35,345,'Accuracy = correct predictions / number of test images',30,'c-e'),2)
    body+=g(t(35,417,'Keep mistakes beside correct examples. A confident score can still be wrong.',25,'c-a'),2)
    add('pets-evaluation','How would we check whether the classifier learned?',body,
        'Use held-out images to measure accuracy and the two kinds of confusion. Inspect successes and mistakes, including confident errors. A probability on one photograph is not the classifier’s test accuracy.',
        'Could we choose the best epoch by repeatedly looking at test accuracy?',
        'Use validation for selection. Reserve the test set for the chosen procedure; related or near-duplicate images must not leak across splits.',
        'The slide describes an evaluation procedure and reports no new accuracy. The original saved synthetic learning curves remain available as optional lab evidence. '
        '<a href="notebooks/vision/03_vision_transformer_lab.ipynb">Optional worked lab</a> · <a href="figures/vision1/training.json">Existing synthetic results</a>.',
        mobile_rows(['Split','Use'],[['Train','Change weights'],['Validation','Select settings/checkpoints'],['Test','Evaluate the selected procedure']])+'<p>Inspect a 2×2 confusion table and representative failures. No new measured results are claimed.</p>')

    body=t(35,55,'One image: 224 × 224 RGB. Patches: 16 × 16. Width: 192.',29)
    for j,q in enumerate(['How many patch rows? What changes when we add CLS?',
                          'Three heads: what is the width of one head’s message?',
                          'Dog/cat head: how many outputs and trainable parameters?',
                          'If the model is wrong, how does the loss reach the patch layer?',
                          'If we remove CLS, how will we form one image representation?']):
        body+=t(35,138+j*59,str(j+1)+'. '+q,26)
    add('classification-exit','Explain one classifier from pixels to learning',body,
        'Use the architecture diagram to answer each question. Name the operation, its input and output shapes, and how its parameters receive a learning signal. Explain a mean-pooling alternative to CLS.',
        'Can you explain this without assuming every head has a fixed human-readable job?',
        'Answers: 196 patch rows,197 with CLS;64 features per head;2 outputs,386 head parameters;reverse classifier and blocks to patch projection;mean final patch rows then classify.',
        'Exit check: 14×14=196; 196+1=197; 192/3=64; 192×2+2=386. CLS and pooling are readout choices. '
        'Class-label gradients train features through the complete differentiable model, including the learned summary token when present.',
        '<ol>'+''.join('<li>'+escape(q)+'</li>' for q in ['Count patch and CLS rows.','Find one head’s width.','Count two-class head parameters.','Trace the backward path.','Explain the no-CLS alternative.'])+'</ol>')

    body=image(35,105,300,200,b['PHOTO'])+arrow(335,205,400,205)
    body+=box(420,162,280,['image representation','192 features'])+arrow(700,205,765,205)
    body+=box(785,162,335,['trained class head','fixed label vocabulary'])
    body+=g(t(35,390,'Next question: could the class descriptions supply the comparison vectors?',28,'c-q'),1)
    add('next-vision','Next: classify an image by writing the labels',body,
        'We now understand a complete image classifier. CLIP will connect its image representation with text representations, so we can compare a photograph with candidate descriptions.',
        'What changes if the candidate labels are descriptions we can write?',
        'Keep this as a closing motivation. The next lecture will explain both encoders, shared embedding space and the image–text training objective.',
        'The next lecture builds on this image encoder and on the text encoders from earlier parts. It will distinguish matching a description from generating an answer. '
        'The existing self-supervised lecture remains available as an optional extension. '
        '<a href="notebooks/vision/CLASSIFICATION_TEACHING_GUIDE.md">Main teaching route and optional calculations</a>.',
        '<p>Current lecture: image → representation → trained class head → fixed label choices.</p><p>Next: compare image features with features computed from candidate text descriptions.</p>')

    def key(m):return re.search(r'class="frame[^\"]*" id="([^\"]+)"',m).group(1)
    inserts={
      'real-cls-purpose':['cls-shared-start','cls-without'],
      'vision-topic-05':['heads-visual-roles','heads-independent'],
      'readout-choice':['backward-route','backward-scores','backward-cls','backward-heads','backward-values','backward-softmax','backward-qk','backward-one-weight','backward-patches'],
      'depth':['backward-full-block','classification-training-map','cnn-receptive-field','cnn-classifier-parallel'],
      'vision-topic-13':['classification-exit'],
    }
    remove={'vision-tasks','find-animal','image-caption','photo-search','cnn-context','cls-two-images'}
    result=[]
    for i,(title,frames) in enumerate(sections):
        if i==8:
            opener=frames[0]
            # The section opener is regenerated below with the new purpose.
            title='Adapt and evaluate the image classifier'
            frames=[opener]+[new[k] for k in ['pets-head','pets-frozen','pets-training-step','pets-evaluation']]
        out=[]
        for m in frames:
            k=key(m)
            if k in remove:continue
            out.append(new['next-vision'] if k=='next-vision' else m)
            out.extend(new[x] for x in inserts.get(k,[]))
        result.append((title,out))
    # Remove the synthetic-results claim from the path back to the real photos.
    for section_id,title,previous,question,caption in [
        (9,'Adapt and evaluate the classifier','We can trace the forward and backward passes.','How would we train this model for our dog/cat task?','Choose trainable parameters, follow one batch, then evaluate on held-out images. We will describe the procedure without running a new experiment.'),
        (10,'Return to the real photographs','We have described how adaptation would work.','What does our existing pretrained classifier predict?','Return to the saved dog and cat predictions. These outputs come from the ImageNet checkpoint, not from a newly trained pet classifier.')]:
        k=f'vision-topic-{section_id:02}'
        body=t(48,58,'SECTION',23,'c-e',weight=600)+t(48,177,f'{section_id:02}',110,'c-e',weight=600)+line(220,35,220,405,'c-e')
        from textwrap import wrap
        for j,s in enumerate(wrap(title,31)):body+=t(264,78+j*60,s,43,'ink',weight=700)
        for j,s in enumerate(wrap(previous,59)):body+=t(264,232+j*34,s,26,'ink-2')
        for j,s in enumerate(wrap(question,49)):body+=t(264,329+j*36,s,29)
        fr=add(k,f'Section {section_id} · {title}',body,caption,question,'Connect the preceding computation to the next concrete procedure.',mobile='<p>Section '+str(section_id)+'</p><h3>'+title+'</h3><p>'+previous+'</p><p>'+question+'</p>')
        for meta in b['FRAMES']:
            if meta['id']==k:meta.update(title=f'Section {section_id} · {title}',caption=caption,notes=question+'\nConnect the preceding computation to the next concrete procedure.')
        fr=fr.replace('class="frame vp-frame"','class="frame vp-frame vp-topic-break"',1)
        result[section_id-1]=(title,[fr]+result[section_id-1][1][1:])
    return result
