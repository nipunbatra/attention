"""Build the paced Part III lesson, shared SVGs and its executable notebook."""
from pathlib import Path
from html import escape
import json
import math
import textwrap
import nbformat as nbf
from head_worksheet import worksheet

SRC=Path(__file__).resolve().parent
REPO=SRC.parent
OUT=SRC/'sections3-heads'
FIG=REPO/'figures/multihead'
BOOK=REPO/'notebooks/wordlm'
BASE,DATA=worksheet()
R=DATA['cases']['river']
STEPS=[]
COLORS='--c-e:#245EDB;--c-q:#8B2CDE;--c-k:#AA4E08;--c-v:#0F766E;--c-a:#BE123C;--c-d:#147737;--ink:#14171F;--muted:#50586a;--line:#D9DFE9'


def fmt(x,n=3):
    return f'{x:.{n}f}'.replace('-','−')


def vec(xs):
    return '['+', '.join(fmt(x) for x in xs)+']'


def text(x,y,value,size=25,color='ink',anchor='start',weight=500):
    return f'<text x="{x}" y="{y}" fill="var(--{color})" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>'


def box(x,y,w,h,title,sub='',color='e',active=True):
    stroke='c-'+color if active else 'line'
    label_color='c-'+color if active else 'muted'
    return f'<g><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="white" stroke="var(--{stroke})" stroke-width="2"/>'+text(x+w/2,y+32,title,24,label_color,'middle',650)+(text(x+w/2,y+61,sub,21,'muted','middle') if sub else '')+'</g>'


def arrow(x1,y1,x2,y2,color='muted'):
    a=math.atan2(y2-y1,x2-x1)
    points=[(x2-10*math.cos(a-s),y2-10*math.sin(a-s)) for s in [-.4,.4]]
    return f'<path d="M{x1} {y1} L{x2} {y2} M{points[0][0]} {points[0][1]} L{x2} {y2} L{points[1][0]} {points[1][1]}" fill="none" stroke="var(--{color})" stroke-width="2.5"/>'


def svg(body,h=360,label=''):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 {h}" role="img" aria-label="{escape(label)}" style="font-family:Avenir Next,Segoe UI,sans-serif;{COLORS}"><title>{escape(label)}</title>{body}</svg>'


def flow(labels,subs=None,colors=None):
    n=len(labels);w=(1120-35*(n-1))/n
    body=''
    for i,label in enumerate(labels):
        x=20+i*(w+35)
        body+=box(x,70,w,86,label,(subs or ['']*n)[i],(colors or ['e']*n)[i])
        if i<n-1:body+=arrow(x+w,113,x+w+33,113)
    return svg(body,225,' → '.join(labels))


def master(focus='all'):
    body=''
    nodes=[('input',20,18,200,'Known tokens','IDs [B,T]','e'),
           ('embed',280,18,235,'Lookup + position','E [B,T,D]','e'),
           ('one',310,160,205,'Head 1','Q₁ · K₁ · V₁','q'),
           ('two',310,280,205,'Head 2','Q₂ · K₂ · V₂','q'),
           ('join',590,218,200,'Concatenate','messages [B,T,D]','v'),
           ('project',900,218,220,'W_O + residual','E′ = E + ΔE','d'),
           ('predict',900,365,220,'Last row → MLP','next-token logits','e')]
    for key,x,y,w,label,sub,color in nodes:
        body+=box(x,y,w,82,label,sub,color,focus=='all' or key==focus or (focus=='heads' and key in ['one','two']))
    body+=arrow(220,59,280,59)
    body+=arrow(397,100,397,158)+arrow(290,100,290,320)+arrow(290,320,308,320)
    body+=arrow(515,201,590,245)+arrow(515,321,590,275)+arrow(790,259,900,259)+arrow(1010,300,1010,365)
    body+=f'<path d="M515 59 H1010 V216" fill="none" stroke="var(--c-e)" stroke-width="2" stroke-dasharray="7 6"/>'
    body+=text(680,45,'Keep E for the residual',22,'c-e')
    return svg(body,460,'Two heads inside the same next-token model')


def table(headers,rows,widths=None,colors=None,rowheight=51):
    widths=widths or [1120/len(headers)]*len(headers)
    body='';y=20
    for r,row in enumerate([headers]+rows):
        x=20
        for c,val in enumerate(row):
            body+=f'<g data-cell-left="{x}" data-cell-right="{x+widths[c]}">'+text(x+10,y+30,val,23 if r==0 else 26,(colors[c] if colors and r else 'muted' if r==0 else 'ink'),weight=650 if r==0 else 500)+'</g>'
            x+=widths[c]
        body+=f'<path d="M20 {y+rowheight-8} H1140" stroke="var(--line)"/>'
        y+=rowheight
    return svg(body,y+12,'; '.join(headers))


def sentence(question=False):
    body=''
    widths=[80,150,80,110,80,110,110,80,135,80]
    x=20
    centers=[]
    for j,t in enumerate(R['tokens']):
        centers.append(x+widths[j]/2)
        body+=box(x,55,widths[j],88,t,str(j+1),'q' if j==9 else 'e')
        x+=widths[j]+12
    body+=text(580,325 if question else 208,'Predict the next token after the final “the”.',30,'ink','middle')
    if question:
        body+=text(centers[5],245,'Which setting?',30,'c-q','middle')+text(centers[1],245,'Who is in the scene?',30,'c-q','middle')
        body+=arrow(centers[5],220,centers[5],145,'c-q')+arrow(centers[1],220,centers[1],145,'c-q')
    return svg(body,370 if question else 330,'The river sentence from Part II')


def weight_row(head,detail=False):
    H=R['heads'][head];body=text(15,33,f'Head {head+1} · final query at position 10',26,'c-q')
    for j,t in enumerate(R['tokens']):
        x=120+j*101
        body+=text(x+45,85,t,18,'ink','middle')
        if detail:body+=text(x+45,137,fmt(H['scores'][-1][j],2),23,'c-k','middle')
        y=188 if detail else 145
        a=H['A'][-1][j]
        body+=f'<rect x="{x}" y="{y-34}" width="93" height="48" fill="var(--c-a)" fill-opacity="{.08+.6*a}"/>'
        body+=text(x+45,y,fmt(a),24,'c-a','middle')
    body+=text(15,137 if detail else 145,'score' if detail else 'weight',22,'muted')
    if detail:body+=text(15,188,'weight',22,'muted')
    body+=text(580,270,'One softmax over the ten sources. Row sum = 1.',26,'ink','middle')
    return svg(body,310,'Attention weights for head '+str(head+1))


def contribution(head):
    H=R['heads'][head];j=5 if head==0 else 1
    a=H['A'][-1][j];v=H['V'][j]
    rows=[[R['tokens'][j],fmt(a),vec(v),vec([a*x for x in v])],
          ['All other sources','', '',vec([m-a*x for m,x in zip(H['messages'][-1],v)])],
          ['Total message','','',vec(H['messages'][-1])]]
    return table(['Source','Weight','Value row','Weighted contribution'],rows,[235,145,330,410],['ink','c-a','c-v','c-v'])


def stage(key,title,figure,body='',code='',notes='',companion=''):
    STEPS.append(dict(key=key,title=title,figure=figure,body=body,code=code))
    FIG.mkdir(parents=True,exist_ok=True)
    if figure:(FIG/(key+'.svg')).write_text(figure)
    extra=f'<pre class="pytorch"><code>{escape(code)}</code></pre>' if code else ''
    copy=body if body.startswith('<div') else f'<p>{body}</p>'
    return f'''<div class="frame mh-frame" id="{key}" data-title="{escape(title)}" data-autobuild="off">
<script type="text/x-notes">{escape(notes or 'Ask students to name the next operation before revealing it. Point to the same input, head and output rows as on the previous frame.')}</script>
{f'<div class="mh-figure">{figure}</div>' if figure else ''}{copy}{extra}</div>
{f'<div class="companion">{companion}</div>' if companion else ''}'''


def divider(key,title,question):
    return stage(key,title,flow(['Known inputs','Two heads','One prediction'],colors=['e','q','d']),question)


def section(number,title,frames):
    return f'<section id="s{number:02}" class="sec" data-title="{title}" data-lit=""><header class="sec-head"><span class="sec-num">{number:02}</span><div><p class="eyebrow">Multi-head attention · a worked continuation</p><h2>{title}</h2></div></header>'+''.join(frames)+'</section>'


def build():
    OUT.mkdir(exist_ok=True)
    sections=[]
    def add(n,title,frames):
        sections.append(dict(id=f's{n:02}',title=title,lit=''))
        (OUT/f'sec{n:02}.html').write_text(section(n,title,frames))
    add(1,'The same prediction, more than one clue',[
        stage('s01-name','Which earlier characters might help?',flow(['a  b  i','Embeddings','Read context','Next character'],['3 known IDs','3 learned rows','MLP or attention','for example, d']),
              'Part I predicted the next character in a name such as aabid. The question stays the same; we are changing how the model reads the known context.'),
        stage('s01-river','Back to the river bank',sentence(True),'One query can benefit from several clues at once. What setting are we in? Who is there?'),
        stage('s01-scope','Keep the two examples separate',table(['Worksheet','Trained experiment'],[['2 heads × 2 coordinates','4 heads × 16 coordinates'],['Hand-chosen projections','Learned from TinyStories'],['Part II token + position rows','Same 64-token benchmark window']],[560,560]),
              'The worksheet explains the arithmetic. Held-out results later tell us whether the trained models improved.',companion='<p>The original Part II toy used query/key width 3 and value width 2. This new worksheet keeps its four-coordinate input rows but chooses two heads with width 2 each. The head labels describe our hand-chosen projections, not roles assigned during training.</p>')])
    add(2,'Give each head its own view',[
        divider('s02-break','From one weight row to two','What changes when the same input passes through two sets of projections?'),
        stage('s02-one','One head produces one message',flow(['Full E','Q, K, V','Weights A','Message AV'],['all coordinates','learned projections','one row per query','one weighted sum'],['e','q','a','v']),
              'One head can already read several sources. Its value coordinates share the same attention weights.'),
        stage('s02-two','Two heads keep two messages',master('heads'),'Each head has its own Q, K and V projections and its own softmax. Both receive the full E.'),
        stage('s02-full','Project first; split the projected coordinates',flow(['E [B,T,4]','W_Q [4,4]','Q [B,T,4]','2 heads × 2'],['full input rows','learned mixing','projected coordinates','one view per head'],['e','q','q','q']),
              'We split Q, K and V after projection. We do not give head 1 the first half of the raw embedding and head 2 the second half.'),
        stage('s02-columns','Separate heads can use different input features',table(['Input coordinate','W_Q: head 1','W_Q: head 2'],[[axis,str(a),str(b)] for axis,a,b in zip(BASE['axes']['e'],DATA['projections'][0]['Q'],DATA['projections'][1]['Q'])],[350,385,385]),
              'These are two 4×2 matrices placed side by side. Every head can learn from every input coordinate.',companion='<p>For compact arithmetic, this particular worksheet uses sparse projection matrices. Real learned projections are generally dense. Separate parameters allow different behavior; they do not guarantee distinct or human-readable roles.</p>')])
    h1,h2=R['heads']
    add(3,'Follow both heads through the numbers',[
        divider('s03-break','Two heads, one receiving token','Keep the final “the” at position 10 as the query throughout.'),
        stage('s03-input','Start with the same position-aware rows',table(['Position / token','water','finance','person','glue'],[[f'{i+1} {R["tokens"][i]}']+[fmt(x,1) for x in R['E'][i]] for i in [1,5,9]],[330,195,195,195,205]),
              'E already includes token embedding + position embedding. Token IDs are not embedding coordinates.'),
        stage('s03-q1','The first query asks about the setting',flow(['e₁₀','W_Q¹','q₁₀¹'],[vec(R['E'][-1]),'4 × 2',vec(h1['Q'][-1])],['e','q','q']),
              'Our chosen projection copies the glue coordinate into both query coordinates: [2.3, 2.3].'),
        stage('s03-q2','The second query uses a different projection',flow(['e₁₀','W_Q²','q₁₀²'],[vec(R['E'][-1]),'4 × 2',vec(h2['Q'][-1])],['e','q','q']),
              'The same input row now produces [2.3, 0.0]. The query changes because the projection matrix changes.'),
        stage('s03-keys','The source keys are different too',table(['Source','Key in head 1','Key in head 2'],[[R['tokens'][i],vec(h1['K'][i]),vec(h2['K'][i])] for i in [1,5,6,9]],[340,390,390]),
              'Head 1 exposes water/finance features. Head 2 exposes person/glue features. These labels belong to this designed worksheet.'),
        stage('s03-dot1','Head 1 scores the river key',table(['Calculation','Value'],[['q₁₀¹','[2.3, 2.3]'],['k₆¹ for river','[3.1, −0.1]'],['Dot product','2.3 × 3.1 + 2.3 × (−0.1) = 6.9'],['Divide by √2',fmt(h1['scores'][-1][5])]], [400,720]),
              'The scaling uses the head width: two coordinates, not four.'),
        stage('s03-dot2','Head 2 scores the fisherman key',table(['Calculation','Value'],[['q₁₀²','[2.3, 0.0]'],['k₂² for fisherman','[2.1, 0.0]'],['Dot product','2.3 × 2.1 + 0.0 × 0.0 = 4.83'],['Divide by √2',fmt(h2['scores'][-1][1])]], [400,720]),
              'We compare each head’s query only with keys from that same head.'),
        stage('s03-weights1','Head 1: scores become a weight row',weight_row(0,True),'Each weight is exp(score) divided by the sum of exp(scores) in this row. All ten sources are allowed for the final query.'),
        stage('s03-weights2','Head 2 has its own softmax',weight_row(1,True),'Normalize over sources within head 2. Do not normalize across heads or average their scores.'),
        stage('s03-value1','Head 1 sends water and finance information',contribution(0),'The river contribution is its weight × its value row. Add the contributions from every source to get the two-coordinate message.'),
        stage('s03-value2','Head 2 sends a different kind of message',contribution(1),'The second head mixes its own values with its own weights. Matching features choose where to read; values carry the information.')])
    add(4,'Combine the messages and predict',[
        divider('s04-break','Bring the two messages back together','How do two small message rows become one update to the original row?'),
        stage('s04-join','Concatenate the messages, not the tokens',table(['Head','Message coordinates','Message'],[['1','water / finance',vec(h1['messages'][-1])],['2','person / glue',vec(h2['messages'][-1])],['Joined','head 1, then head 2',vec(R['joined'][-1])]],[160,390,570]),
              'Two 2-coordinate rows become one 4-coordinate row. Concatenation preserves both feature groups; it does not average them.'),
        stage('s04-wo','W_O can mix information across heads',table(['Joined coordinate','Output weights'],[[str(i+1),str(row)] for i,row in enumerate(DATA['W_O'])],[400,720]),
              f'Water update = {fmt(R["joined"][-1][0])} + 0.25 × {fmt(R["joined"][-1][2])} = {fmt(R["delta"][-1][0])}. W_O maps the joined message back to model width.'),
        stage('s04-residual','Add the update to the original row',table(['Row','Four coordinates'],[['Original e₁₀',vec(R['E'][-1])],['Attention update Δe₁₀',vec(R['delta'][-1])],['Updated e′₁₀',vec(R['updated'][-1])]],[450,670]),
              'The residual is the same addition as in Part II. There is still one updated row per input token.'),
        stage('s04-predict','The prediction MLP stays in place',flow(['e′₁₀','Hidden + ReLU','20 logits','Softmax'],['4 coordinates','8 activations','one per vocabulary item','next-token probabilities'],['e','d','e','a']),
              'Choose a token only after vocabulary softmax. Attention weights choose source positions; vocabulary probabilities choose possible next tokens.'),
        stage('s04-live','Change the context; inspect both heads','',
              '<div id="s04-head-explorer"></div>',companion='<p>This explorer recomputes all projections, scores, softmaxes, messages and vocabulary probabilities when the sentence changes. It uses the worksheet matrices, not the trained TinyStories checkpoints. Try both contexts and both heads.</p>')])
    add(5,'Implement the same calculation',[
        divider('s05-break','From the drawing to tensors','Keep B = 2 examples, T = 10 tokens, D = 4 coordinates and H = 2 heads.'),
        stage('s05-model','Create the tables, projections and prediction MLP',flow(['Token + position','ScratchMultiHead','Hidden → vocabulary'],['20 × 4 and 10 × 4','4 coordinates, 2 heads','4 → 8 → 20']),
              'The download defines every layer. Load the hand-chosen worksheet weights to reproduce the printed numbers; ordinary training starts from random parameters.',code='model = TinyMultiHeadLM(vocab_size=20, context=10,\n                        width=4, heads=2, hidden=8)\nload_worksheet_weights(model, worksheet)'),
        stage('s05-batch','Two input windows form one batch',table(['Example','Ten token IDs','Observed next token'],[['river',str([BASE['vocab'].index(t.lower()) for t in BASE['sentences']['river']]),'water'],['cheque',str([BASE['vocab'].index(t.lower()) for t in BASE['sentences']['cheque']]),'teller']],[170,700,250]),
              'These are integer IDs, not embeddings. Each example has one observed target.',code='X = torch.tensor([river_ids, cheque_ids])\ny = torch.tensor([word_to_id["water"], word_to_id["teller"]])'),
        stage('s05-embed-map','Find the embedding lookup on the map',master('embed'),'We have token IDs. Next, look up their learned rows and add the position rows. Both heads will read the result.'),
        stage('s05-embed','Look up token rows and add position rows',flow(['Token IDs [2,10]','Token + position rows','E [2,10,4]'],['integers','lookup, then add','floating-point vectors']),'The embedding tables are learned parameters. E contains floating-point vectors with shape [2,10,4].',code='positions = torch.arange(X.shape[1])\nE = model.token_embedding(X) + model.position_embedding(positions)'),
        stage('s05-project','Project the full input three times',flow(['E [2,10,4]','Three matrices','Q, K, V'],['same input','each 4 × 4','each [2,10,4]'],['e','q','q']),
              'Different learned matrices give matching queries, matching keys and transmitted values.',code='Q = model.attention.W_Q(E)\nK = model.attention.W_K(E)\nV = model.attention.W_V(E)'),
        stage('s05-split','Make the head axis explicit',flow(['[2,10,4]','[2,10,2,2]','[2,2,10,2]'],['B, T, D','B, T, H, d_head','B, H, T, d_head'],['q','q','q']),
              'reshape groups projected coordinates; transpose puts heads before tokens. Apply the same operation to K and V.',code='Q = Q.reshape(2, 10, 2, 2).transpose(1, 2)\nK = K.reshape(2, 10, 2, 2).transpose(1, 2)\nV = V.reshape(2, 10, 2, 2).transpose(1, 2)'),
        stage('s05-scores','Compute every query–key pair within each head',flow(['Q [2,2,10,2]','Kᵀ [2,2,2,10]','Scores [2,2,10,10]'],['query rows','source columns','two 10 × 10 grids'],['q','k','a']),
              'The first two axes keep examples and heads separate. Matrix multiplication contracts only the matching-coordinate axis.',code='scores = Q @ K.transpose(-2, -1)\nscores = scores / math.sqrt(2)'),
        stage('s05-mask','Block future sources in every head',table(['Receiving position','Allowed source positions'],[['1','1'],['2','1, 2'],['3','1, 2, 3'],['10','1 through 10']],[450,670]),
              'True marks a blocked entry. The same causal mask broadcasts across both examples and both heads.',code='future = torch.ones(10, 10, dtype=torch.bool).triu(1)\nscores = scores.masked_fill(future, float("-inf"))'),
        stage('s05-softmax','Each row becomes a distribution over sources',weight_row(0),'The final axis indexes source tokens. Every allowed row sums to one; masked entries have weight zero.',code='A = scores.softmax(dim=-1)\nassert A.shape == (2, 2, 10, 10)'),
        stage('s05-mix','Multiply the weights by the values',flow(['A [2,2,10,10]','V [2,2,10,2]','Messages [2,2,10,2]'],['source weights','source content','one row per query/head'],['a','v','v']),
              'The source-token axis is summed out. Each head keeps its own two-coordinate message.',code='messages = A @ V'),
        stage('s05-join','Put each token’s head messages side by side',flow(['[2,2,10,2]','[2,10,2,2]','[2,10,4]'],['B, H, T, d_head','B, T, H, d_head','B, T, D'],['v','v','v']),
              'Transpose before reshaping. A direct reshape of the original layout would mix token rows with head rows.',code='joined = messages.transpose(1, 2).contiguous()\njoined = joined.reshape(2, 10, 4)'),
        stage('s05-output-map','Return to the shared prediction path',master('project'),'The two head messages are joined. W_O mixes them, and the residual adds that update to the original E.'),
        stage('s05-output','Project, add, and keep the final row',flow(['Joined messages','W_O + residual','Last row → MLP'],['[2,10,4]','updated E′ [2,10,4]','logits [2,20]'],['v','d','e']),'W_O combines the heads. The residual keeps E. Only the last updated row feeds this next-token loss.',code='delta = model.attention.W_O(joined)\nupdated = E + delta\nlogits = model.readout(F.relu(model.hidden(updated[:, -1])))')])
    add(6,'Train and generate with the same model',[
        divider('s06-break','The rest of the learning loop is familiar','More heads change the context update, not the definition of the next-token target.'),
        stage('s06-loss','Score the two observed targets',flow(['Input X','Two heads + MLP','Logits [2,20]','Loss'],['2 × 10 token IDs','one prediction/example','targets: water, teller','one scalar']),
              'Cross-entropy reads logits and observed token IDs. Do not sample a generated token to make the training target.',code='logits = model(X)\nloss = F.cross_entropy(logits, y)'),
        stage('s06-update','One loss trains all the projections',flow(['Loss','Gradients','Optimizer','Updated weights'],['from the same targets','all learned parameters','one update','used by the next batch'],['a','d','d','e']),
              'Heads are not assigned jobs or separate labels. They receive gradients from the same prediction loss.',code='optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)\noptimizer.zero_grad()\nloss.backward()\noptimizer.step()'),
        stage('s06-prompt','Start generation from known token IDs',flow(['River prefix','Vocabulary lookup','History [1,10]'],['the ten known words','same vocabulary','no observed next token']),
              'We already encoded the river sentence. A real application tokenizes and looks up a new prompt with the same vocabulary.',code='history = torch.tensor([river_ids])\nmodel.eval()'),
        stage('s06-infer','Generate one token, then repeat',flow(['Known IDs','Crop to context','Model logits','Choose + append'],['no future target','latest 10 in this toy','final row only','new known prefix']),
              'At inference, weights stay fixed. The current prefix becomes longer; we keep only the configured context window.',code='with torch.no_grad():\n    logits = model(history[:, -model.context:])\nnext_id = logits.argmax(dim=-1, keepdim=True)\nhistory = torch.cat([history, next_id], dim=1)'),
        stage('s06-map','Trace one request through the full model',master(),'Start at the known IDs. Name the shape at each arrow. Where does the observed target enter during training? It enters only at the loss.',companion='<p>The worksheet uses ten known tokens with no PAD rows. The browser models use a 64-slot window with left padding and PAD-source masking. No layer normalization or block FFN is needed to explain this one-block teaching model. A production Transformer adds further components; see the optional reference.</p>')])
    add(7,'Use the PyTorch block',[
        divider('s07-break','The same operation, packaged by PyTorch','Which boxes does nn.MultiheadAttention replace?'),
        stage('s07-api-map','Replace the head calculation, not the whole model',master('heads'),'The PyTorch layer replaces both heads, their concatenation and W_O. We still add the residual and use our prediction MLP.'),
        stage('s07-api','Create the multi-head attention layer',flow(['Input E','nn.MultiheadAttention','Projected update ΔE'],['[B,T,4]','2 heads × 2 coordinates','[B,T,4]'],['e','q','d']),'batch_first=True means [B,T,D]. bias=False and dropout=0 match our scratch implementation.',code='mha = nn.MultiheadAttention(\n    embed_dim=4, num_heads=2, bias=False,\n    dropout=0.0, batch_first=True)'),
        stage('s07-call','Pass E as query, key and value input',flow(['E, E, E','nn.MultiheadAttention','ΔE and A'],['unprojected input rows','Q/K/V, softmax, mix, W_O','not a residual yet'],['e','q','d']),
              'PyTorch performs the learned projections internally. average_attn_weights=False preserves the head axis in the returned weight tensor.',code='delta, A = mha(E, E, E, attn_mask=future,\n               average_attn_weights=False)\nupdated = E + delta'),
        stage('s07-check','Compare the same weights, not two random layers',table(['Quantity','Expected shape'],[['Projected update ΔE','[2,10,4]'],['Separate head weights A','[2,2,10,10]'],['Residual E + ΔE','[2,10,4]']],[560,560]),
              'The notebook copies the scratch Q/K/V and W_O parameters into PyTorch, then checks both outputs numerically.',code='mha = copy_to_pytorch(model.attention)\ndelta_scratch, A_scratch = model.attention(E)\ndelta_api, A_api = mha(E, E, E, attn_mask=future,\n                      average_attn_weights=False)\ntorch.testing.assert_close(delta_api, delta_scratch)'),
        stage('s07-boundary','What the attention block does not include',flow(['Token + position','Multi-head attention','Residual','Prediction MLP'],['outside the API','returns projected ΔE','add E ourselves','outside the API'],['e','q','d','e']),
              'nn.MultiheadAttention does not add position embeddings, the residual, a vocabulary head or a training loop.',companion='<p><a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html">PyTorch API reference</a>. A boolean attn_mask uses True for blocked query–source pairs. For padded real batches, also use a correctly shaped padding mask; the worked ten-token batch has no padding. Do not assume that boolean mask conventions are identical across different APIs.</p>')])
    result=json.loads((BOOK/'artifacts/heads/comparison.json').read_text())
    labels={'mlp':'MLP','attention':'One head','multihead':'Four heads'}
    summary=[[labels[k],fmt(a['test_loss']['mean']),fmt(a['test_perplexity']['mean'],2)] for k,a in result['aggregate'].items()]
    costs=[[labels[k],f'{result["runs"][k][0]["parameter_count"]:,}',fmt(result['aggregate'][k]['runtime_seconds']['mean'],1)+' s'] for k in labels]
    add(8,'Compare the trained models',[
        divider('s08-break','Do more heads help this experiment?','Move from the small worksheet to the actual TinyStories checkpoints.'),
        stage('s08-width','More heads need not mean more parameters',table(['Setting','Total width','Heads','Width per head'],[['One head','64','1','64'],['Four heads','64','4','16']],[420,230,200,270]),
              'Q, K, V and W_O remain 64×64. The number of attention patterns grows, while each pattern uses fewer matching coordinates.'),
        stage('s08-scores','Four heads improved held-out prediction here',table(['Model','Test cross-entropy ↓','Test perplexity ↓'],summary,[410,355,355]),
              'Three-seed means. Same stories, tokenizer, context and update budget; validation-selected checkpoints. Perplexity = exp(cross-entropy). Lower is better. This does not guarantee better text for every prompt.'),
        stage('s08-costs','Compare the cost as well as the score',table(['Model','Parameters','Training per seed'],costs,[400,350,370]),
              'Mean training time on Apple M2 Max/MPS, including validation checks. Both attention models have equal parameter counts; the MLP is larger.'),
        stage('s08-demo','Try the same prompt with all three models',flow(['Training prefix','Held-out prefix','Outside-domain prompt'],['familiar text','new story, same task','different kind of text']),
              '<a href="word-lab/" target="_blank" rel="noopener">Open the live browser demo ↗</a> Compare continuations, vocabulary coverage and generation time. This runs real checkpoints with WebGPU or WASM.'),
        stage('s08-notebooks','Run the calculation yourself',table(['Resource','What to do'],[['Notebook 7','Follow every operation and check PyTorch parity'],['Notebook 6','Inspect the trained four-head checkpoint and results'],['Part 2B · optional','Gradients, normalization, full blocks and context cost']],[420,700]),
              '<a href="notebooks/wordlm/07_multihead_step_by_step.html">Step-by-step notebook</a> · <a href="notebooks/wordlm/06_multihead_comparison.html">Measured experiment</a> · <a href="part2b.html">Optional reference</a>'),
        stage('s08-next','Next: read a different sequence',flow(['French decoder row','Query','English keys + values'],['the next output token','what do I need?','where should I read?'],['e','q','v']),
              'So far, Q, K and V came from the same sequence. <a href="part4.html">Part IV changes that: cross-attention reads another sequence.</a>')])
    config=dict(part=3,series='Attention and language',title='Multi-head attention, step by step',
                subtitle='Keep the same next-token problem. Follow two heads, combine their messages, and write the code.',
                audience='Deep-learning students who completed Parts I–II',minutes=75,
                centralLabel='The same update, several heads',central=r'\Delta E=\operatorname{Concat}(A^{(1)}V^{(1)},\ldots,A^{(H)}V^{(H)})W_O',
                chain=[dict(label=s['title'],section=s['id']) for s in sections],sections=sections,
                objects=['e','q','k','v','a','d','ep'],legendTitle='The same objects as Part II',
                provenance='The two-head worksheet reuses Part II’s exact token and position rows. Its projections are hand-chosen. The separate TinyStories comparison uses trained checkpoints and three seeds.',
                prev=dict(label='Part 2: Self-attention, from first principles',href='attention.html'),
                next=dict(label='Part 4: Cross-attention: translate one phrase',href='part4.html'),
                index=dict(label='Series home',href='index.html'),notation='multihead',
                footer='Several learned views, separate attention rows, one contextual update. Optional training and cost reference: Part 2B.',
                objectSections=dict(e='s02',q='s03',k='s03',v='s03',a='s03',d='s04',ep='s04'),
                hook='The final “the” may need both the river setting and the person in the scene. How can two heads retrieve both?',
                sectionDirectory='sections3-heads',runtimeFile='part3-heads.js',toyFile='toy3-heads.json',syntaxHighlighting=True)
    (SRC/'part3.json').write_text(json.dumps(config,indent=2)+'\n')
    toy=dict(BASE,headsLesson=DATA)
    (SRC/'toy3-heads.json').write_text(json.dumps(toy,indent=2)+'\n')
    (FIG/'worksheet.json').write_text(json.dumps(toy,indent=2)+'\n')
    (BOOK/'multihead-worksheet.json').write_text(json.dumps(toy,indent=2)+'\n')
    (FIG/'manifest.json').write_text(json.dumps([{k:v for k,v in s.items() if k!='figure'} for s in STEPS],indent=2)+'\n')
    print(f'{len(STEPS)} paced frames, {len(sections)} sections')


if __name__=='__main__':build()
