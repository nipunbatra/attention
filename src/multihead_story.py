"""Part III: a held drawing, two reading patterns, then the matrices.

Rows, semantic colours and e → Δe → e′ follow Part II. The little numbers
come from head_worksheet, never from hand-typed display approximations.
"""
from html import escape
import math
import re

COLORS={'e':'#245edb','q':'#8b2cde','k':'#aa4e08','v':'#0f766e',
        'a':'#be123c','d':'#147737','ink':'#14171f','muted':'#586174','line':'#d9dfe9'}


def f(x,n=2):
    return f'{x:.{n}f}'.replace('-','−')


def t(x,y,s,size=26,c='ink',anchor='start',weight=500):
    label=re.sub(r'([Wb])_([QKVO])',r'\1<tspan baseline-shift="sub" font-size="70%">\2</tspan>',escape(str(s)))
    label=re.sub(r'α_\{([0-9,j]+)\}',r'α<tspan baseline-shift="sub" font-size="70%">\1</tspan>',label)
    for mark,head in [('¹','1'),('²','2')]:
        label=label.replace(mark,f'<tspan baseline-shift="super" font-size="65%">({head})</tspan>')
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{COLORS.get(c,c)}" text-anchor="{anchor}" font-weight="{weight}">{label}</text>'


def rect(x,y,w,h,c='line',fill='white',width=1.5):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{COLORS.get(c,c)}" stroke-width="{width}"/>'


def path(d,c='muted',width=2.5,dash=''):
    return f'<path d="{d}" fill="none" stroke="{COLORS[c]}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'


def arrow(x1,y1,x2,y2,c='muted',width=2.5):
    angle=math.atan2(y2-y1,x2-x1)
    p=[(x2-10*math.cos(angle+a),y2-10*math.sin(angle+a)) for a in [-.4,.4]]
    return path(f'M{x1} {y1} L{x2} {y2} M{p[0][0]} {p[0][1]} L{x2} {y2} L{p[1][0]} {p[1][1]}',c,width)


def svg(body,h=410,label=''):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 {h}" role="img" aria-label="{escape(label)}" style="font-family:Avenir Next,Segoe UI,Arial,sans-serif"><title>{escape(label)}</title>{body}</svg>'


def cells(x,y,values,c='v',w=105,h=50,labels=None):
    out=''
    for i,value in enumerate(values):
        xx=x+i*w
        out+=rect(xx,y,w,h,c,COLORS[c]+'0b')+t(xx+w/2,y+h*.65,value,26,c,'middle')
        if labels:out+=t(xx+w/2,y-12,labels[i],20,'muted','middle')
    return out


def matrix(x,y,rows,cols,name,c='e',cw=28,ch=22,values=None,selected=None,causal=False):
    """Real row/column counts, with an optional receiver-row highlight."""
    out=t(x+cols*cw/2,y-42,name,28,c,'middle',650)
    out+=t(x+cols*cw/2,y-14,f'{rows} × {cols}',22,'muted','middle')
    for i in range(rows):
        for j in range(cols):
            a=values[i][j] if values else None
            opacity=.09 if a is None else min(.8,.07+abs(a)*.65)
            color=COLORS[c]
            if causal and j>i:color='#9da3ae';opacity=.2
            out+=rect(x+j*cw,y+i*ch,cw,ch,'line',color+f'{int(opacity*255):02x}',.6)
            if causal and j>i:out+=t(x+(j+.5)*cw,y+(i+.68)*ch,'×',14,'muted','middle')
    if selected is not None:out+=rect(x,y+selected*ch,cols*cw,ch,c,'none',2.5)
    return out


def strip(tokens,y=54,weights=None):
    widths=[69,137,65,101,64,88,84,68,119,70]
    out='';x=20;centers=[]
    for j,word in enumerate(tokens):
        w=widths[j];centers.append(x+w/2)
        out+=rect(x,y,w,48,'q' if j==9 else 'line',COLORS['q']+'0b' if j==9 else 'white')
        out+=t(x+w/2,y+31,word,23,'q' if j==9 else 'ink','middle',650 if j==9 else 500)
        out+=t(x+w/2,y-12,str(j+1),18,'muted','middle')
        if weights is not None:out+=t(x+w/2,y+77,f(weights[j],3),21,'a','middle')
        x+=w+13
    out+=t(1120,y+32,'___',25,'d','middle')
    return out,centers


def reading(case,head=None,both=False):
    if head is None and not both:
        body,_=strip(case['tokens'],90)
        body+=t(560,238,'The known prefix ends here.',32,'ink','middle')
        body+=t(560,288,'The updated final “the” will predict the next word.',28,'muted','middle')
        return svg(body,345,'The river-bank prefix from Part II')
    body=''
    heads=[0,1] if both else [head]
    for lane,h in enumerate(heads):
        y=69+lane*237
        weights=case['heads'][h]['A'][-1]
        words,centers=strip(case['tokens'],y)
        body+=t(20,y-37,('Head 1: setting clues' if h==0 else 'Head 2: person clues'),26,'ink',weight=650)+words
        # Source-to-receiver arrows: all sources are present, with width encoding weight.
        for j,a in enumerate(weights):
            body+=arrow(centers[j],y+55,830,y+139,'a',.55+9*a)
        main=5 if h==0 else 1
        body+=t(centers[main],y+83,f(weights[main],3),22,'a','middle',650)
        body+=rect(747,y+145,165,45,'q',COLORS['q']+'08')+t(830,y+175,'10 · the',24,'q','middle')
        body+=t(22,y+161,'Thicker arrow = more attention weight',24,'muted')
        body+=t(946,y+174,'receiver',22,'muted')
    return svg(body,505 if both else 287,'Two different reading patterns' if both else f'Head {head+1} reads the known prefix')


def bottleneck(two=False):
    """A separate two-source illustration, before the full ten-token worksheet."""
    body=t(24,32,'Two-source illustration: invented values and weights',23,'muted')
    for x,label in [(24,'Source'),(350,'Setting feature'),(675,'Person feature')]:
        body+=t(x,91,label,25,'muted',weight=600)
    for row,(word,values) in enumerate([('river',[10,1]),('fisherman',[2,8])]):
        y=140+row*52
        body+=t(24,y,word,29,'ink')+t(390,y,values[0],29,'v')+t(715,y,values[1],29,'v')
    body+=path('M24 215 H1120','line',2)
    if two:
        body+=t(24,273,'Head 1: setting',26,'ink',weight=650)
        body+=t(390,273,'0.8 × 10 + 0.2 × 2 = 8.4',30,'v')
        body+=t(24,333,'Head 2: person',26,'ink',weight=650)
        body+=t(390,333,'0.2 × 1 + 0.8 × 8 = 6.6',30,'v')
        body+=t(580,399,'river gets 0.8 in one head; fisherman gets 0.8 in the other.',26,'a','middle')
    else:
        body+=t(24,267,'One head: river gets 0.8, fisherman gets 0.2',27,'a',weight=600)
        body+=t(580,327,'0.8 × [10, 1] + 0.2 × [2, 8] = [8.4, 2.4]',33,'v','middle')
        body+=t(580,399,'Both output coordinates use the same 80% / 20% mixture.',27,'ink','middle')
    return svg(body,435,'Two heads weight the sources independently' if two else 'One weight per source multiplies every value coordinate')


def divider(question,sub):
    body=t(35,155,question,43,'ink',weight=650)+t(35,230,sub,29,'muted')
    return svg(body,325,question)


def head_example(kind,one=False,washed=False):
    """Possible semantic readings, explicitly not measured attention maps."""
    examples={
        'coat':(['Maya','wore','a','hooded','red','wool','coat'],[
            ('Which colour?', 'red', 'colour: red'),
            ('Which material?', 'wool', 'material: wool'),
            ('Which detail?', 'hooded', 'detail: hood')]),
        'grammar':(['The','dogs','near','the','gate','usually'],[
            ('Who is the subject?', 'dogs', 'plural subject'),
            ('Where are they?', 'gate', 'location clue')]),
        'event':(['Ravi','washed' if washed else 'dropped','the','glass','.','It'],[
            ('What does “It” refer to?', 'glass', 'object: glass'),
            ('What happened to it?', 'washed' if washed else 'dropped',
             'event: washing' if washed else 'event: dropping')])}
    tokens,roles=examples[kind]
    body=t(24,29,'Possible head roles — an illustration, not a trained model’s attention',23,'muted')
    widths=[max(72,21+len(word)*17) for word in tokens]
    x=(1160-sum(widths)-12*(len(tokens)-1))/2
    for i,(word,w) in enumerate(zip(tokens,widths)):
        receiver=i==len(tokens)-1
        body+=t(x+w/2,68,str(i+1),18,'muted','middle')
        body+=rect(x,79,w,47,'q' if receiver else 'line',COLORS['q']+'0b' if receiver else 'white')
        body+=t(x+w/2,110,word,25,'q' if receiver else 'ink','middle',650 if receiver else 500)
        if receiver:body+=t(x+w/2,152,'receiver',19,'q','middle')
        x+=w+12
    for x,label,c in [(24,'Head / query: what to look for','q'),(491,'Key: a matching source','k'),(866,'Value: what it can send','v')]:
        body+=t(x,195,label,21,c,weight=600)
    for i,(question,source,message) in enumerate(roles):
        if one and i>0:continue
        y=215+i*76
        row=t(24,y+33,f'{i+1} · {question}',25,'q')
        row+=arrow(382,y+25,474,y+25,'a')
        row+='<g>'+rect(491,y,232,49,'k',COLORS['k']+'08')+t(607,y+33,source,26,'k','middle')+'</g>'
        row+=arrow(736,y+25,845,y+25,'v')
        row+='<g>'+rect(866,y,267,49,'v',COLORS['v']+'08')+t(999,y+33,message,24,'v','middle')+'</g>'
        body+=f'<g data-example-head="{i+1}">{row}</g>'
    if kind=='coat':
        body+=t(580,475,'Still to retrieve: material and detail.' if one else 'Keep colour, material and detail as separate messages.',27,'ink','middle')
    elif kind=='grammar':
        body+=t(24,409,'Possible next words: “bark at visitors”',29,'ink',weight=600)
        body+=t(24,460,'“dogs” controls agreement; the nearer noun “gate” is singular.',26,'muted')
    else:
        body+=t(24,409,'Possible continuation: “gleamed”' if washed else 'Possible continuation: “broke”',29,'ink',weight=600)
        body+=t(24,460,'Same object. A different event gives a different clue.' if washed else 'The object alone does not tell us what happened.',26,'muted')
    return svg(body,510,'Illustrative '+kind+' head readings'+(' after changing the action' if washed else ''))


def coupled_outputs():
    body=t(24,38,'Keep the same two value rows: river [10, 1], fisherman [2, 8].',27,'v')
    body+=t(24,94,'Let α be the river weight; the fisherman weight is 1 − α.',27,'a')
    for y,label,equation,answer in [(157,'Want setting = 8.4','10α + 2(1 − α) = 8.4','α = 0.8'),
                                     (264,'Want person = 6.6','1α + 8(1 − α) = 6.6','α = 0.2')]:
        body+=t(24,y+30,label,28,'ink',weight=650)+t(411,y+30,equation,30,'v')
        body+=arrow(832,y+20,914,y+20,'a')+t(939,y+30,answer,29,'a')
    body+=path('M24 332 H1133','line',2)
    body+=t(580,385,'One shared weight cannot be both 0.8 and 0.2.',31,'ink','middle',650)
    body+=t(580,437,'Two heads can choose these mixtures independently.',28,'muted','middle')
    return svg(body,475,'A fixed-value example where one source mixture cannot provide both desired outputs')


def vision_bridge(classifier=False):
    """Shape illustration: no invented learned predictions or attention maps."""
    if not classifier:
        body=t(24,37,'A tiny 8 × 8 grayscale image',27,'ink',weight=650)
        for r in range(8):
            for c in range(8):
                body+=rect(45+c*26,93+r*26,26,26,'line','#245edb' if c in (3,4) else 'white',.6)
        for r in range(2):
            for c in range(2):body+=rect(45+c*104,93+r*104,104,104,'q','none',2.5)
        body+=t(149,335,'Four 4 × 4 patches',24,'q','middle')
        body+=arrow(275,196,383,196,'e')
        body+=t(564,107,'Flatten each patch',27,'ink','middle',650)
        for i in range(4):
            for j in range(16):
                # Row-major flattening of each patch, preserving the visible stripe.
                lit=j%4==(3 if i%2==0 else 0)
                body+=rect(408+j*19,133+i*43,19,27,'line',COLORS['e'] if lit else 'white',.6)
            body+=t(397,153+i*43,str(i+1),18,'muted','end')
        body+=t(560,335,'4 rows × 16 pixel values',24,'e','middle')
        body+=arrow(729,196,837,196,'e')+t(785,151,'Wpatch',23,'q','middle')+t(785,251,'16 × 4',23,'muted','middle')
        for i in range(4):
            for j in range(4):body+=rect(876+j*42,133+i*43,42,27,'e',COLORS['e']+'0c',.8)
        body+=t(960,107,'Patch embeddings',26,'ink','middle',650)
        body+=t(960,335,'Epatch: 4 × 4',24,'e','middle')
        body+=t(580,418,'The shared projection makes one embedding row per patch.',29,'ink','middle')
        return svg(body,461,'Image patches become embedding rows through one shared projection')
    body=t(24,40,'Four patch rows + one learned [CLS] row + position rows',29,'e',weight=650)
    for i in range(5):
        body+=t(25,112+i*47,'[CLS]' if i==0 else f'patch {i}',23,'q' if i==0 else 'e')
        for j in range(4):body+=rect(124+j*31,84+i*47,31,33,'q' if i==0 else 'e',COLORS['q' if i==0 else 'e']+'0c',1)
    body+=t(186,354,'E: 5 × 4',26,'e','middle')
    body+=arrow(264,197,357,197,'e')
    body+=rect(375,100,365,187,'q',COLORS['q']+'06')
    body+=t(557,140,'Transformer encoder',29,'ink','middle',650)
    body+=t(557,181,'MHA: 2 heads × 2 coordinates',24,'q','middle')
    body+=t(557,219,'then a per-row MLP',25,'d','middle')
    body+=t(557,258,'residuals + normalization inside',22,'muted','middle')
    body+=arrow(757,197,848,197,'d')
    for i in range(5):
        for j in range(4):body+=rect(883+j*31,84+i*47,31,33,'d',COLORS['d']+('18' if i==0 else '06'),1)
    body+=t(945,354,'Updated rows: 5 × 4',24,'d','middle')
    body+=t(580,406,'All image patches are visible: no future-token mask.',27,'ink','middle')
    body+=path('M1007 101 H1137 V449 H1110','d')
    body+=t(1100,473,'Final [CLS]: 1 × 4',26,'d','end')+arrow(853,464,738,464,'d')
    body+=rect(457,437,263,54,'e')+t(588,472,'Classifier: 4 → 2',25,'e','middle')
    body+=arrow(439,464,356,464,'e')+t(24,472,'Two class logits',26,'e')
    body+=t(580,534,'Softmax → vertical-stripe / horizontal-stripe probabilities',26,'muted','middle')
    return svg(body,568,'A CLS row gathers image information before a two-class prediction')


def wide_match(case):
    body=t(580,36,'Join the same query and key coordinates into one wider head.',27,'ink','middle')
    body+=t(70,92,'q₁₀ =',28,'q')+cells(228,57,[f(v,1) for v in case['wide']['Q'][-1]],'q',145)
    body+=path('M518 53 V113','ink',2)+t(374,144,'setting coordinates',22,'muted','middle')+t(663,144,'person coordinates',22,'muted','middle')
    for row,j in enumerate([5,1]):
        y=214+row*105
        dots=[sum(q*k for q,k in zip(h['Q'][-1],h['K'][j])) for h in case['heads']]
        body+=t(25,y,case['tokens'][j],28,'k',weight=650)
        body+=t(264,y,f(dots[0],2),30,'a')+t(403,y,'+',30)+t(464,y,f(dots[1],2),30,'a')
        body+=t(619,y,'= '+f(sum(dots),2),30,'a')
        body+=arrow(782,y-10,904,y-10,'a')+t(842,y-29,'÷ √4',23,'muted','middle')
        body+=t(945,y,f(case['wide']['scores'][-1][j],2),30,'a')
    body+=t(325,385,'setting dot',22,'muted','middle')+t(515,385,'person dot',22,'muted','middle')+t(990,385,'one score',22,'muted','middle')
    return svg(body,420,'A four-coordinate dot product adds both matching contributions before one softmax')


def wide_weights(case):
    body='';xs=[22,350,570,790,1010]
    for x,label in zip(xs,['Receiver 10 reads','river','fisherman','others','value width']):
        body+=t(x,48,label,23,'muted',weight=600)
    rows=[('One wide head',case['wide']['A'][-1],4),
          ('Head 1: setting',case['heads'][0]['A'][-1],2),
          ('Head 2: person',case['heads'][1]['A'][-1],2)]
    for i,(label,a,width) in enumerate(rows):
        y=101+i*94
        body+=t(22,y+26,label,27,'ink')
        for x,value in zip(xs[1:4],[a[5],a[1],1-a[5]-a[1]]):
            body+=rect(x-4,y-10,158,56,'a',COLORS['a']+f'{int(15+160*value):02x}')+t(x+75,y+26,f(value,3),27,'a','middle')
        body+=t(1060,y+26,str(width),29,'v','middle')
    body+=path('M22 166 H1125','line',2)
    body+=t(580,390,'One shared mixture of four coordinates, or two separate mixtures of two.',27,'ink','middle')
    return svg(body,423,'The same projection coordinates yield one attention row or two separately normalized rows')


def bias_example(case):
    q=case['heads'][0]['Q'][-1];offset=[.2,-.1]
    body=t(580,43,'One head: qᵢ = eᵢ W_Q + b_Q',32,'q','middle')
    body+=t(20,119,'bias=False',27,'ink',weight=650)+t(340,119,'e₁₀ W_Q',24,'q')+cells(653,83,[f(v,1) for v in q],'q',145)
    body+=t(20,242,'bias=True',27,'ink',weight=650)
    body+=cells(245,203,[f(v,1) for v in q],'q',105)+t(479,239,'+',32)
    body+=cells(526,203,[f(v,1) for v in offset],'q',105)+t(764,239,'=',32)
    body+=cells(815,203,[f(a+b,1) for a,b in zip(q,offset)],'q',145)
    body+=t(630,299,'b_Q: learned offset [2]',23,'q','middle')
    body+=t(580,369,'The same b_Q is added to every token row in this head.',29,'ink','middle')
    return svg(body,409,'A bias adds a learned offset after the projection; the displayed offset is illustrative')


def bias_locations():
    body=t(23,40,'Inside nn.MultiheadAttention',29,'ink',weight=650)
    for i,(kind,c) in enumerate([('Q','q'),('K','k'),('V','v')]):
        x=23+i*378
        body+=t(x,114,f'{kind} = E W_{kind} + b_{kind}',29,c)
        body+=t(x,156,'offset after input projection',22,'muted')
    body+=t(23,241,'ΔE = Concat(H¹, H²) W_O + b_O',30,'d')
    body+=t(23,282,'offset after output projection',23,'muted')
    body+=path('M23 312 H1125','line',2)
    body+=t(23,353,'Our worksheet and trained attention layers: bias=False.',28,'ink')
    body+=t(23,395,'The separate prediction MLP still has its own biases.',27,'muted')
    return svg(body,429,'The bias flag controls all query, key, value and output projection biases, not the separate prediction MLP')


def role_recap():
    body=t(24,36,'Maya cycled home in the rain. Cold and tired, Maya reached',27,'ink')
    body+=t(24,76,'for a hooded red wool coat. She …',27,'ink')
    rows=[('q: what is needed?','She × W_Q','Which earlier person?','q'),
          ('k: what can match?','Maya × W_K','Person candidate','k'),
          ('v: what is sent?','Maya × W_V','Cold, tired; cycled in rain','v')]
    for i,(role,source,meaning,c) in enumerate(rows):
        y=148+i*83
        body+=t(24,y,role,27,c,weight=650)+t(400,y,source,26,c)
        body+=arrow(637,y-9,691,y-9,c)+t(717,y,meaning,25,c)
    body+=t(580,399,'“Maya” and “She” here mean their current embedding rows.',25,'muted','middle')
    return svg(body,435,'Recall Part II: the query asks, the key matches, the value supplies content')


def reading_roles():
    body=t(580,36,'Receiver: the final “the” in the river-bank prefix',28,'e','middle')
    for x,label in [(24,'Head'),(230,'Query asks about'),(575,'A useful source'),(894,'Value carries')]:
        body+=t(x,109,label,23,'muted',weight=600)
    rows=[('1','the setting','river','setting clues'),('2','the person','fisherman','person clues')]
    for i,(h,q,k,v) in enumerate(rows):
        y=179+i*100
        body+=t(24,y,h,32,'ink')+t(230,y,q,29,'q')+t(575,y,k,29,'k')+t(894,y,v,27,'v')
    body+=t(580,365,'Each head computes q, k and v for every token.',28,'ink','middle')
    body+=t(580,406,'We follow one receiver and two useful sources.',25,'muted','middle')
    return svg(body,442,'The familiar query, key and value roles, now repeated in two heads')


def queries(case,data,h=0):
    sup='¹' if h==0 else '²'
    body=t(24,43,'Receiver: final “the”',27,'ink',weight=650)
    body+=t(620,43,'q₁₀'+sup+' = e₁₀ W_Q'+sup,31,'q')
    body+=t(24,122,'e₁₀: four input coordinates',25,'e')
    body+=cells(24,167,[f(v,1) for v in case['E'][-1]],'e',91,
                labels=['water','finance','person','glue'])
    body+=t(419,202,'×',37)
    body+=t(678,92,'W_Q'+sup+'  [4 × 2]',27,'q','middle')
    labels=['water','finance'] if h==0 else ['person','glue']
    for j,label in enumerate(labels):body+=t(626+j*104,131,label+'?',23,'q','middle')
    for i,row in enumerate(data['projections'][h]['Q']):
        body+=cells(574,149+i*39,[str(v) for v in row],'q',104,39)
    body+=arrow(803,216,869,216,'q')
    body+=t(1000,132,'q₁₀'+sup+'  [1 × 2]',27,'q','middle')
    body+=cells(890,167,[f(v,1) for v in case['heads'][h]['Q'][-1]],'q',110)
    body+=t(580,359,'First query coordinate: 0×0 + 0×0 + 0×0 + 2.3×1 = 2.3',27,'q','middle')
    last=data['projections'][h]['Q'][-1][1]
    body+=t(580,403,f'Second query coordinate: 0×0 + 0×0 + 0×0 + 2.3×{last} = {f(2.3*last,1)}',27,'q','middle')
    return svg(body,445,f'Head {h+1}: multiply the same four-coordinate embedding by its own query projection')


def source_rows(case):
    body=t(24,35,'Source input eⱼ: [water, finance, person, glue]',26,'e')
    for h,j in [(0,5),(1,1)]:
        y=102+h*164;sup='¹' if h==0 else '²';H=case['heads'][h]
        body+=t(24,y,f'Head {h+1}: {case["tokens"][j]}',28,'ink',weight=650)
        body+=t(24,y+44,str([round(v,1) for v in case['E'][j]]).replace('-','−'),27,'e')
        selected='water, finance' if h==0 else 'person, glue'
        body+=t(465,y,'W_K'+sup+' selects '+selected,24,'k')
        body+=t(465,y+44,'W_V'+sup+' selects '+selected,24,'v')
        body+=t(884,y,'k = '+str([round(v,1) for v in H['K'][j]]).replace('-','−'),25,'k')
        body+=t(884,y+44,'v = '+str([round(v,1) for v in H['V'][j]]).replace('-','−'),25,'v')
    body+=t(580,409,'kⱼ = eⱼ W_K sets the match.  vⱼ = eⱼ W_V supplies the message.',27,'ink','middle')
    return svg(body,445,'Each source embedding supplies a key for matching and a value for the weighted message')


def numeric_cell(x,y,w,h,value,c,selected=False):
    return '<g>'+rect(x,y,w,h,c if selected else 'line',COLORS[c]+('15' if selected else '06'),2 if selected else .7)+t(x+w/2,y+h*.7,value,25,c,'middle')+'</g>'


def head_matrices(case,h):
    H=case['heads'][h];sup='¹' if h==0 else '²';source=5 if h==0 else 1
    axes=['water','finance'] if h==0 else ['person','glue']
    body=t(24,109,'Token and position',25,'ink')
    for x,kind,c in [(354,'Q','q'),(788,'K','k')]:
        body+=t(x+110,38,kind+sup+' = E W_'+kind+sup,28,c,'middle')
        body+=t(x+110,75,'[10 × 4] × [4 × 2] = [10 × 2]',22,'muted','middle')
        for col,name in enumerate(axes):body+=t(x+55+col*110,109,name+('?' if kind=='Q' else ''),22,c,'middle')
        for row,values in enumerate(H[kind]):
            for col,value in enumerate(values):
                body+=numeric_cell(x+col*110,126+row*30,110,30,f(value,1),c,row==(9 if kind=='Q' else source))
    for j,word in enumerate(case['tokens']):body+=t(24,147+j*30,f'{j+1:2d}  {word}',25,'ink')
    body+=t(580,476,f'Follow query row 10 (“the”) and key row {source+1} (“{case["tokens"][source]}”).',27,'ink','middle')
    return svg(body,505,f'Head {h+1}: all ten query rows and key rows, with the receiver and example source highlighted')


def matching(case,h):
    H=case['heads'][h];sup='¹' if h==0 else '²';source=5 if h==0 else 1
    q=H['Q'][-1];key=H['K'][source]
    raw=[sum(a*b for a,b in zip(q,k)) for k in H['K']]
    body=t(24,49,'q₁₀'+sup+' =',29,'q')+cells(185,15,[f(v,1) for v in q],'q',100)
    body+=t(710,49,'Receiver 10: final “the”',27,'ink')
    body+=t(580,107,'q [1 × 2]  ×  Kᵀ [2 × 10]  =  raw scores r [1 × 10]',27,'ink','middle')
    widths=[68,124,67,92,67,80,79,67,110,67];x=155
    axes=['water','finance'] if h==0 else ['person','glue']
    body+=t(25,222,'Kᵀ',30,'k')
    for c,name in enumerate(axes):body+=t(140,210+c*44,name,21,'k','end')
    body+=t(25,323,'raw r',26,'a')
    for j,(word,w) in enumerate(zip(case['tokens'],widths)):
        body+=t(x+w/2,146,str(j+1),20,'muted','middle')+t(x+w/2,173,word,22,'ink','middle')
        for c in range(2):body+=numeric_cell(x,185+c*44,w,44,f(H['K'][j][c],1),'k',j==source)
        body+=numeric_cell(x,292,w,47,f(raw[j],2),'a',j==source)
        x+=w+9
    body+=t(580,399,f'Column {source+1}: query · {case["tokens"][source]} key',28,'ink','middle')
    def factor(v):return '('+f(v,1)+')' if v<0 else f(v,1)
    products=[a*b for a,b in zip(q,key)]
    body+=t(580,452,f'{factor(q[0])} × {factor(key[0])} + {factor(q[1])} × {factor(key[1])} = {f(products[0],2)} + ({f(products[1],2)}) = {f(raw[source],2)}',30,'a','middle')
    return svg(body,488,f'Head {h+1}: one query times the transposed key matrix yields ten dot products')


def weights(case,h):
    H=case['heads'][h];source=5 if h==0 else 1;q=H['Q'][-1]
    raw=[sum(a*b for a,b in zip(q,k)) for k in H['K']]
    scores=H['scores'][-1];exps=[math.exp(s) for s in scores];total=sum(exps)
    body=t(580,34,'Ten matching scores, normalized within this head',27,'ink','middle')
    xs=[24,352,579,807,1044]
    for x,label in zip(xs,['Source','Raw dot r','s = r / √2','exp(s)','Weight α']):body+=t(x,86,label,24,'muted','start' if x==24 else 'middle',600)
    for j,word in enumerate(case['tokens']):
        y=115+j*30
        if j==source:body+=rect(16,y-23,1118,30,'a',COLORS['a']+'0b',1)
        body+=t(24,y,f'{j+1:2d}  {word}',24,'ink')
        for x,value in zip(xs[1:],[f(raw[j],2),f(scores[j],3),f(exps[j],3),f(H['A'][-1][j],3)]):body+=t(x,y,value,25,'a','middle')
    body+=t(580,438,f'Shared total: Σ exp(s) = {f(total,3)}',27,'a','middle')
    body+=t(580,483,f'{case["tokens"][source]} weight: {f(exps[source],3)} / {f(total,3)} ≈ {f(H["A"][-1][source],3)}',29,'a','middle')
    return svg(body,515,f'Head {h+1}: scale all ten dot products and divide each exponential by the sum of all ten exponentials')


def message_products(case,h,phase='sum'):
    """Hold source rows in place while revealing weights, values, products, sum."""
    phases=['alpha','values','products','sum']
    level=phases.index(phase)
    H=case['heads'][h];sup='¹' if h==0 else '²';source=5 if h==0 else 1
    word=case['tokens'][source]
    alpha=f'α_{{10,{source+1}}}'
    body=t(580,37,'Receiver: the final “the”, at position 10',27,'ink','middle')
    columns=[('sources',24,'Source j','muted'),('weights',332,'Weight α_{10,j}','a')]
    if level>=1:columns.append(('values',575,'Value row vⱼ','v'))
    if level>=2:columns.append(('contributions',942,'Contribution α_{10,j} vⱼ','v'))
    # Every revealed column retains identical coordinates in later frames.
    for name,x,label,c in columns:
        body+=f'<g data-values-part="{name}">'
        body+=t(x,76,label,25,c,'start' if name=='sources' else 'middle',600)
        left,right={'sources':(16,218),'weights':(248,416),'values':(444,706),'contributions':(777,1138)}[name]
        body+=rect(left,99+source*30,right-left,30,c,COLORS['a' if name=='weights' else 'v']+'0b',1)
        if name in ['values','contributions']:
            body+=path(f'M{left+9} 100 H{left} V405 H{left+9} M{right-9} 100 H{right} V405 H{right-9}','v',1.5)
        for j,(token,a,v) in enumerate(zip(case['tokens'],H['A'][-1],H['V'])):
            y=122+j*30
            if name=='sources':value=f'{j+1:2d}  {token}'
            elif name=='weights':value=f(a,3)
            else:value='['+', '.join(f(vv,1) if name=='values' else f(a*vv,3) for vv in v)+']'
            body+=f'<g data-source-row="{j+1}">'+t(x,y,value,24 if name=='sources' else 25,'ink' if name=='sources' else c,'start' if name=='sources' else 'middle')+'</g>'
        body+='</g>'
    if level>=2:
        body+=t(430,122+source*30,'×',24,'ink','middle')+t(741,122+source*30,'=',24,'ink','middle')
    if phase=='alpha':
        body+=t(740,213,alpha+' ≈ '+f(H['A'][-1][source],3),36,'a','middle')
        body+=t(740,265,f'Receiver 10 reads source {source+1}: {word}',28,'ink','middle')
        body+=t(740,315,f'{f(100*H["A"][-1][source],1)}% of this head’s total weight',27,'a','middle')
        body+=t(580,460,'These are the softmax weights we just calculated.',28,'ink','middle')
    elif phase=='values':
        body+=t(928,233,'V = E W_V',31,'v','middle')
        body+=t(928,279,'10 source rows × 2',25,'v','middle')
        body+=t(580,460,f'{word}: its weight pairs with its own two-number value row.',28,'ink','middle')
    elif phase=='products':
        a=H['A'][-1][source];v=H['V'][source]
        body+=t(580,460,f'{word}: [{alpha} × ({f(v[0],1)}), {alpha} × ({f(v[1],1)})] ≈ [{f(a*v[0],3)}, {f(a*v[1],3)}]',28,'v','middle')
    else:
        body+='<g data-values-part="sum">'+path('M778 417 H1138','v',2)
        body+=t(24,458,'Sum each contribution coordinate:',27,'ink')
        body+=t(782,458,'m₁₀'+sup+' =',28,'v','end')
        for i,value in enumerate(H['messages'][-1]):
            body+=numeric_cell(813+i*152,425,152,50,f(value,3),'v',True)
        body+=t(580,510,'weight row [1 × 10]  ×  V [10 × 2]  =  message [1 × 2]',26,'ink','middle')+'</g>'
    return svg(f'<g data-value-phase="{phase}" data-head="{h+1}">{body}</g>',540,
               f'Head {h+1}: '+{'alpha':'attention weights for each source','values':'pair each source weight with its value row','products':'multiply both value coordinates by the same source weight','sum':'sum all ten weighted value rows to obtain the message'}[phase])


def messages(case):
    body=''
    for h in range(2):
        H=case['heads'][h];sup='¹' if h==0 else '²';y=73+h*212
        body+=t(24,y-33,f'Head {h+1}',29,'ink',weight=650)
        body+=rect(24,y,244,98,'q',COLORS['q']+'08')+t(146,y+36,'q₁₀'+sup+' K'+sup+'ᵀ / √2',27,'q','middle')+t(146,y+72,'ten scores',24,'muted','middle')
        body+=arrow(268,y+49,370,y+49,'a')+t(319,y+24,'softmax',21,'a','middle')
        body+=rect(370,y,245,98,'a',COLORS['a']+'08')+t(492,y+36,'Own weight row',25,'a','middle')
        j=5 if h==0 else 1
        body+=t(492,y+73,case['tokens'][j]+': '+f(H['A'][-1][j],3),25,'a','middle')
        body+=arrow(615,y+49,740,y+49,'v')+t(676,y+25,'× V'+sup,24,'v','middle')
        body+=cells(740,y+25,[f(v,3) for v in H['messages'][-1]],'v',194)
        body+=t(934,y-11,'m₁₀'+sup+' [1 × 2]',28,'v','middle')
    body+=t(580,466,'Both heads read the same input E. Their score lists and value sums stay separate.',26,'ink','middle')
    return svg(body,502,'Two complete head calculations produce two messages before concatenation and output projection')


def head_walkthrough(stage,case,h):
    n=h+1
    return [
        stage(f's02-v-head{n}-matrices',f'Head {n}: the query and key matrices',head_matrices(case,h),
              'Every token has a query and a key. We follow the last query row, for the final “the”, and compare it with all ten key rows.',
              notes='Which row is the receiver? Which rows are possible sources?\nPoint to query row 10, then all ten key rows. The highlighted key is only one example.'),
        stage('s02-v-match' if h==0 else 's02-v-head2-dots',f'Head {n}: one dot product per key',matching(case,h),
              r'Transposing \(K\) puts each source key in a column. Multiplying the query row by \(K^\top\) gives one raw dot product per source.',
              notes='Why are there ten outputs from a two-coordinate query?\nEach of the ten columns is a different key. Expand the highlighted column coordinate by coordinate.'),
        stage('s02-v-weights' if h==0 else 's02-v-head2-softmax',f'Head {n}: turning scores into weights',weights(case,h),
              'Divide each dot product by √2, then apply softmax across these ten sources. The final query can see all ten tokens. The unrounded weights sum to one.',
              companion='<p>Every source contributes to the denominator, including the receiver itself. All displayed numbers are rounded; calculations use full precision. At receiver row 10, the causal mask permits source positions 1 through 10. Earlier query rows cannot read later sources. The notebook also computes the full masked attention matrix. A numerically stable softmax subtracts the maximum score before exponentiating; this leaves the normalized weights unchanged.</p>',
              notes='What goes into the shared denominator?\nAdd all ten exponentials, not just the largest ones. Each head has its own denominator.'),
        stage(f's02-v-head{n}-alpha',f'Head {n}: the weights for each source',message_products(case,h,'alpha'),
              r'\(\alpha_{10,j}\) tells us how much the final “the” reads from source \(j\). Keep these weights as we bring in the values.',
              companion='<p>These are the ten entries of row 10 of this head’s attention matrix A. We display them vertically so each weight can sit beside its source word and value row. The receiver stays fixed at token 10. Only the source index j changes.</p>',
              notes='Which source has the largest weight?\nRead its source index and word, then the weight in that same row.'),
        stage(f's02-v-head{n}-values',f'Head {n}: the value beside each weight',message_products(case,h,'values'),
              'Each source supplies a two-number value row from this head’s V matrix. The source words and weights stay in place, so we can see which weight belongs to which value.',
              companion='<p>V = E W_V has one row per source. These are projected value vectors, not token IDs or attention weights. Our hand-chosen worksheet gives W_K and W_V equal entries, but they remain separate parameters with different roles. The keys helped compute the weights. V supplies the content to mix.</p>',
              notes='Which value row belongs to the highlighted weight?\nTrace straight across from the word. Keep both coordinates of that value row.'),
        stage(f's02-v-head{n}-products',f'Head {n}: each source’s contribution',message_products(case,h,'products'),
              'Multiply both value coordinates by the same source weight. The new column shows exactly what each source contributes. Displayed numbers are rounded.',
              companion='<p>The calculation below the rows expands the highlighted source’s scalar–vector multiplication coordinate by coordinate. A small weight scales down the entire value vector. The sign of each coordinate comes from the value. All products use the unrounded weights.</p>',
              notes='Does one source get a different weight for its second coordinate?\nUse the highlighted row to multiply both coordinates by the same alpha, then point to all ten contributions.'),
        stage(f's02-v-head{n}-sum',f'Head {n}: adding the contributions',message_products(case,h,'sum'),
              'Add the first coordinates to get the first message coordinate. Add the second coordinates to get the second. All ten sources contribute. The sum uses unrounded products.',
              companion='<p>This is one row of H = AV: m₁₀ = Σⱼ α₁₀,ⱼ vⱼ. The original weight row has shape [1,10] and V has shape [10,2], giving a [1,2] message. We arranged the weights vertically only to make the source correspondence visible. The displayed products are rounded, but the sums use full precision.</p>',
              notes='How do ten source rows become one message row?\nSum down each contribution coordinate separately. Keep the two coordinates in their original order.')]


def join(case):
    body=t(130,59,'m₁₀¹ · setting message',27,'v')+t(718,59,'m₁₀² · person message',27,'v')
    body+=cells(130,87,[f(x,3) for x in case['heads'][0]['messages'][-1]],'v',135)
    body+=cells(718,87,[f(x,3) for x in case['heads'][1]['messages'][-1]],'v',135)
    body+=arrow(265,144,440,223,'v')+arrow(853,144,728,223,'v')
    body+=cells(290,232,[f(x,3) for x in case['joined'][-1]],'v',145)
    body+=path('M580 227 V290','ink',3)
    body+=t(580,333,'2 coordinates + 2 coordinates → 4 coordinates',29,'ink','middle')
    return svg(body,375,'Concatenate two messages; do not average them')


def output(case,data):
    body=t(98,64,'joined message',25,'v')+cells(20,99,[f(x,3) for x in case['joined'][-1]],'v',112)
    body+=t(491,136,'×',36)
    body+=t(730,29,'W_O · 4 × 4',27,'d','middle')
    for i,row in enumerate(data['W_O']):
        for j,v in enumerate(row):
            body+=rect(566+j*75,50+i*47,75,47,'d',COLORS['d']+'09')
            body+=t(604+j*75,81+i*47,str(v),24,'d','middle')
    body+=path('M560 144 H872','ink',2.5)
    body+=t(914,103,'head 1 rows',21,'muted')+t(914,199,'head 2 rows',21,'muted')
    body+=t(126,312,'Δe₁₀ =',32,'d')+cells(300,277,[f(x,3) for x in case['delta'][-1]],'d',156)
    body+=t(580,377,f'First coordinate: {f(case["joined"][-1][0],3)} + 0.25 × {f(case["joined"][-1][2],3)} = {f(case["delta"][-1][0],3)}',26,'ink','middle')
    return svg(body,416,'Map the joined message back to the four input coordinates')


def residual(case):
    body=''
    for r,(label,values,c) in enumerate([('original e₁₀',case['E'][-1],'e'),('+ update Δe₁₀',case['delta'][-1],'d'),('= updated e′₁₀',case['updated'][-1],'d')]):
        y=54+r*116
        body+=t(23,y+35,label,30,c)
        body+=cells(360,y,[f(x,3) for x in values],c,181)
        if r==0:
            for j,name in enumerate(['water','finance','person','glue']):body+=t(450+j*181,y-17,name,23,'muted','middle')
    return svg(body,377,'Keep the original embedding row and add the context update')


def full_map(focus='all',one=False):
    """Fixed positions across recap, second-head reveal, and prediction."""
    body=''
    def node(x,y,w,label,sub,c,key):
        active=focus=='all' or key in focus
        return rect(x,y,w,69,c if active else 'line',COLORS[c]+'08' if active else 'white')+t(x+w/2,y+28,label,25,c if active else 'muted','middle',600)+t(x+w/2,y+53,sub,19,'muted','middle')
    body+=node(23,136,177,'Input E','10 rows × 4','e','e')
    for h,y in [(1,55),(2,229)]:
        if one and h==2:continue
        body+=path(f'M200 170 H238 V{y+35} H285','e')
        body+=node(285,y,235,f'Head {h}','Q, K, V → A → AV','q','heads')
        body+=arrow(520,y+35,602,y+35,'v')
        body+=node(604,y,182,f'H{chr(0x00b9) if h==1 else chr(0x00b2)}','10 messages × 2','v','heads')
        body+=path(f'M786 {y+35} H821 V149 H867','v')
    body+=node(867,115,265,'Concatenate','two [10×2] → [10×4]','v','join')
    body+=arrow(1000,184,1000,222,'v')
    body+=node(867,226,265,'Project with W_O','[10×4] × [4×4]','d','join')
    body+=path('M110 136 V40 H1148 V379 H1046','e',2,'7 6')+arrow(1046,379,1025,379,'e')
    body+=t(980,25,'keep original E',20,'e','middle')
    body+=arrow(1000,295,1000,353,'d')+t(986,329,'ΔE [10×4]',21,'d','end')
    body+='<circle cx="1000" cy="379" r="24" fill="white" stroke="'+COLORS['d']+'" stroke-width="2"/>'+t(1000,388,'+',30,'d','middle')
    body+=arrow(973,379,822,379,'d')+node(608,345,210,'E′ = E + ΔE','same 10 × 4 shape','d','residual')
    body+=arrow(608,380,520,380,'d')+node(285,345,235,'last row → MLP','next-token prediction','e','predict')
    if 'train' in focus:
        body+=arrow(285,380,200,380,'a')+node(23,345,177,'Loss','observed target y','a','train')
    if one:body+=t(560,254,'Part II: one message per receiving token',27,'muted','middle')
    return svg(body,455,'The same next-token path, with two parallel heads' if not one else 'The one-head path from Part II')


def projection_map(case,kind='Q',h=0):
    c={'Q':'q','K':'k','V':'v'}[kind]
    body=matrix(135,93,10,4,'E','e',42,24,selected=9)
    body+=t(365,221,'×',39)+matrix(440,130,4,2,f'W_{kind}{"¹" if h==0 else "²"}',c,66,44)
    body+=arrow(622,216,745,216,c)
    body+=matrix(837,93,10,2,f'{kind}{"¹" if h==0 else "²"}',c,63,24,selected=9)
    body+=t(330,379,'Every token uses the same projection matrix within this head.',26,'ink')
    return svg(body,420,'Project ten four-coordinate input rows into ten two-coordinate rows')


def attention_grid(case,h=0):
    H=case['heads'][h];sup='¹' if h==0 else '²'
    body=matrix(36,92,10,2,'Q'+sup,'q',31,22,selected=9)
    body+=t(133,211,'×',31)+matrix(186,166,2,10,'K'+sup+'ᵀ','k',24,27)
    body+=arrow(449,197,514,197,'a')
    body+=t(490,279,'÷ √2',22,'a','middle')+t(490,310,'mask',22,'a','middle')+t(490,341,'softmax',22,'a','middle')
    body+=matrix(555,92,10,10,'A'+sup,'a',24,22,H['A'],9,True)
    body+=t(821,207,'×',31)+matrix(863,92,10,2,'V'+sup,'v',31,22)
    body+=arrow(943,197,983,197,'v')+matrix(1021,92,10,2,'H'+sup,'v',31,22,selected=9)
    body+=t(580,395,'Each highlighted row follows receiver 10. Columns of A are source positions.',25,'ink','middle')
    return svg(body,437,'One head: query-key scores, a causal weight grid, then weighted values')


def stacked_maps(case):
    body=matrix(51,138,10,4,'same E','e',29,21,selected=9)
    for h,y in [(0,93),(1,327)]:
        body+=path(f'M167 243 H218 V{y+63} H269','e')
        body+=t(283,y-27,f'Head {h+1}',28,'q',weight=650)
        body+=rect(270,y,194,94,'q',COLORS['q']+'08')+t(367,y+39,'own Q, K, V',24,'q','middle')+t(367,y+71,'own projections',20,'muted','middle')
        body+=arrow(464,y+47,532,y+47,'a')
        body+=matrix(569,y-7,10,10,'A¹' if h==0 else 'A²','a',15,14,case['heads'][h]['A'],9,True)
        body+=arrow(729,y+61,800,y+61,'v')+t(761,y+27,'× V',21,'v','middle')
        body+=matrix(858,y-7,10,2,'H¹' if h==0 else 'H²','v',40,14,selected=9)
    return svg(body,495,'Two independent attention grids computed from the same input snapshot')


def join_matrices():
    body=matrix(27,108,10,2,'H¹','v',34,23,selected=9)+matrix(171,108,10,2,'H²','v',34,23,selected=9)
    body+=t(144,232,';',32,'muted','middle')+arrow(258,224,320,224,'v')
    body+=matrix(361,108,10,4,'Concat(H¹, H²)','v',31,23,selected=9)+path('M423 108 V338','ink',2.5)
    body+=t(529,233,'×',34)+matrix(589,157,4,4,'W_O','d',31,32)
    body+=arrow(745,225,813,225,'d')+matrix(855,108,10,4,'ΔE','d',31,23,selected=9)
    body+=t(580,398,'10 message rows → 10 update rows. The token count stays fixed.',28,'ink','middle')
    return svg(body,440,'Concatenate columns, project, and preserve the token axis')


def widths():
    body=''
    for row,(n,label) in enumerate([(1,'1 head × 64 coordinates'),(4,'4 heads × 16 coordinates')]):
        y=74+row*151;body+=t(22,y+34,label,29,'ink')
        for i in range(n):
            body+=rect(460+i*640/n,y,640/n,62,'q',COLORS['q']+'0c',2)
            body+=t(460+(i+.5)*640/n,y+39,str(64//n),28,'q','middle')
    body+=t(580,391,'W_Q, W_K, W_V and W_O stay 64 × 64.',28,'ink','middle')
    return svg(body,430,'More heads divide a fixed total width into smaller per-head projections')


def results_table(rows,headers):
    body='';x=[25,434,801];width=[390,348,339]
    for i,head in enumerate(headers):body+=t(x[i]+12,47,head,25,'muted',weight=650)
    for r,row in enumerate(rows):
        y=83+r*71
        for c,value in enumerate(row):body+=t(x[c]+12,y+33,value,29,'ink')
        body+=path(f'M25 {y+51} H1120','line',1.5)
    return svg(body,327,'; '.join(headers))


def story(stage,base,data):
    R=data['cases']['river'];parts=[]
    def add(title,frames):parts.append((title,frames))
    def s(key,title,figure,body='',**kw):return stage(key,title,figure,body,**kw)
    add('What else could this token read?',[
        s('s01-v-prefix','Back to the river bank',reading(R),
          'Which parts of this prefix would help you choose a continuation?',
          notes='Which earlier words would help you continue this sentence?\nPoint to river and fisherman, then to the final the—not the blank.'),
        s('s01-v-coat-one','One token can need several kinds of detail',head_example('coat',one=True),
          'At “coat”, colour is useful—but so are material and shape. One head can read several words. The question is whether we want the same mixture for every feature.',
          companion='<p>These next examples illustrate possible readings, not attention measured in our trained language model. A query is a numerical vector, not an English question. Each source also has a numerical key and value. The arrows highlight plausible useful sources; they do not mean all other sources are masked or have exactly zero weight. In deeper layers, source rows can already contain context.</p>',
          notes='What else do we know about the coat?\nAsk for material and shape before showing the next frame. Keep the token row and colour reading fixed.'),
        s('s01-v-coat-many','Three heads can keep three readings separate',head_example('coat'),
          'One head could favour red, another wool, another hooded. Each receives the same input rows but can return a different weighted message. The output projection combines those messages.',
          companion='<p>This is a possible division of work, not three manually assigned jobs. All three heads see all seven rows, including coat. Head-specific projections can expose different matching and value features. A single head may also encode several attributes; the extra capability is independent source weighting, not a rule that each head can understand only one concept.</p>',
          notes='What stays fixed from the previous frame?\nThe sentence, the receiver and the colour reading. Add two separate ways of reading that same sentence.'),
        s('s01-v-grammar','The nearest noun is not always the subject',head_example('grammar'),
          'The plural subject and the location are both useful clues. Separate heads can preserve them without having to favour the same source.',
          companion='<p>The receiver is usually, the last known token—not the as-yet-unknown word bark. “The dogs near the gate usually bark at visitors” is one possible continuation. Dogs supports bark, not barks; the nearer singular noun gate does not control subject agreement. Both useful source words occur before the receiver. These semantic roles are illustrative, not measured trained heads.</p>',
          notes='Which word controls “bark” versus “barks”?\nContrast dogs with the nearer noun gate. What other information could still be useful for the continuation?'),
        s('s01-v-event','A reference and an event are different clues',head_example('event'),
          'At “It”, one reading could retrieve the object; another could retrieve what happened to it. Combining glass and dropping makes “broke” a plausible continuation, not a certainty.',
          companion='<p>We treat punctuation as a token in this illustration: It is receiver 6. Both useful sources are in its known prefix. The two heads run in parallel, so the event head does not wait for the reference head’s result. Contextual input rows from earlier layers can help make these matches.</p>',
          notes='Would knowing only “glass” be enough to choose a continuation?\nName the event as a separate clue, then change only that event.'),
        s('s01-v-event-change','Change the event; keep the object',head_example('event',washed=True),
          'The glass is unchanged. “Washed” supports a different continuation, such as “gleamed”. Separate messages carry both the object clue and the event clue.',
          companion='<p>The arrows are a schematic of useful information, not a promise that a trained attention map remains unchanged after a word substitution. All head projections learn jointly from prediction loss. Interpretable roles sometimes emerge, but some heads overlap or can be pruned; see <a href="https://aclanthology.org/P19-1580/">Voita et al. (2019)</a>. More heads are a capacity choice to test, not an automatic improvement.</p>',
          notes='What changed, and what stayed the same?\nThe object clue stayed glass. The event clue changed. We have not measured a probability for either continuation.'),
        s('s01-v-shared','One head mixes the value rows',bottleneck(),
          'Back to river and fisherman. Suppose we want setting clues from one and person clues from the other. A single head uses one source weight for every coordinate of each value.',
          companion='<p>This two-source illustration uses invented values and normalized weights, separately from the ten-token worksheet that follows. A weight of 0.8 on river scales both of its value coordinates. A head can attend to multiple sources, but cannot choose a separate attention weight for each value coordinate.</p>',
          notes='What happens to both numbers in the river value?\nMultiply both by 0.8. The setting and person outputs use the same mixture.'),
        s('s01-v-independent','Two heads can choose different mixtures',bottleneck(two=True),
          'Let one head return the setting feature and another return the person feature. Each head can choose its own source weights. Their outputs stay separate until the output projection.',
          companion='<p>Both heads receive both source rows. For this illustration, one value projection selects the setting coordinate and the other selects the person coordinate. The total output width stays two. The example demonstrates independent weighting, not a guarantee that two trained heads outperform every one-head model. The following slides return to our ten-token, four-coordinate worksheet.</p>',
          notes='Which source should contribute more to each feature?\nFollow the different weights: 0.8 on river for setting, 0.8 on fisherman for person.'),
        s('s01-v-coupled','Why can’t one mixture give both results?',coupled_outputs(),
          'With these fixed values, choosing the best mixture for one output changes the other output too. Two heads remove that coupling by learning separate source-weight rows.',
          companion='<p>This is a claim about the displayed weighted-sum operation with fixed V, not a proof that every one-head network fails. A one-head model can learn other values and downstream transformations. Making Q/K wider still yields one score per source and one softmax row; making V wider still applies that row to every value coordinate. Later we compare one wide head and two narrow heads using the same projection entries.</p>',
          notes='Can one river weight satisfy both equations?\nSolve each separately: 0.8 for setting, 0.2 for person. Point out the fixed-value assumption.'),
        s('s01-v-one','Setting clues in the full sentence',reading(R,0),
          'Back to all ten tokens. Our hand-chosen head 1 gives river a large weight. Its values carry setting information to the receiver.',
          companion='<p>These are hand-chosen two-head parameters applied to Part II’s exact input rows. “Setting” and “person” name the intended behaviour of this example, not jobs assigned to trained heads. Arrows show information moving from source to receiver; their widths encode computed attention weights.</p>'),
        s('s01-v-two','Another head can read who is there',reading(R,1),
          'The sentence and receiver stay fixed. A different head gives fisherman a large weight.',
          notes='What changed: the words, the receiver, or how we read?\nPoint to the same token positions; compare the two thick arrows.'),
        s('s01-v-both','Keep both readings',reading(R,both=True),
          'Two heads can retain different source mixtures at the same token. They run in parallel, not one after the other.')])
    add('How does a second head get a different view?',[
        s('s02-v-break','How does each head choose what to read?',divider('Queries, keys and values in each head','Each head learns its own W_Q, W_K and W_V.'),
          '',companion='<p>Visual inspiration: <a href="https://www.3blue1brown.com/lessons/attention/">3Blue1Brown’s attention lesson</a> and <a href="https://jalammar.github.io/illustrated-transformer/">Jay Alammar’s Illustrated Transformer</a>. We keep our own river-bank example, numbers and Part II row-vector convention. A head-specific superscript labels a head; a subscript still labels a token.</p>'),
        s('s02-v-recall','Recall the Maya example: query, key and value',role_recap(),
          'In Part II’s illustrative later-layer example, the query asks for a person. Maya’s key can match that request. Her value supplies useful details. A head uses separate projections for these roles.',
          companion='<p>This is the verbal example from <a href="attention.html#s11-frame-separate-maps">Part II</a>. It illustrates possible later-layer representations, not measured model outputs. Earlier layers may have gathered the preceding facts into the second Maya row. Every token has a query, key and value, even though this example follows only She’s query and Maya’s key/value. The actual vectors contain numbers, not written questions or records.</p>',
          notes='Why do we need a value as well as a key?\nPerson-candidate features help match Maya; cold and tired supplies useful content for the continuation.'),
        s('s02-v-plan','The two heads and the output projection',full_map('heads join'),
          r'Both heads read \(E\). Concatenation joins their message coordinates. \(W_O\) projects the joined messages into embedding space before we add the update to \(E\).',
          companion='<p>H¹ and H² each have shape [10, 2]. Concatenation gives [10, 4]. Multiplying by W_O [4, 4] gives ΔE [10, 4]. Here the joined width already equals the embedding width. The projection still learns how to mix head outputs into embedding coordinates; it need not change the width. In general W_O has shape [n_heads × d_v, d_model]. It is not an inverse of the input projections.</p>',
          notes='What does the second head receive as input?\nTrace both branches from E. Neither branch starts at the other head’s output.'),
        s('s02-v-roles','The same roles in the river example',reading_roles(),
          'Head 1 can ask about the setting; head 2 can ask about the person. These are interpretations of our chosen numbers. Training learns the projections without assigning these jobs.',
          notes='What changes between the heads?\nThe query, matching features and value content can differ. The receiver and available source tokens stay fixed.'),
        s('s02-v-query','Head 1: computing the setting query',queries(R,data,0),
          r'In this toy, the final “the” has only a nonzero glue coordinate. The two columns of \(W_Q\) turn that input into water and finance matching features.',
          companion='<p>E contains position-aware embedding rows, not token IDs. The water, finance, person and glue axes are invented for teaching; glue is our toy feature for function words such as “the”. W_Q reads all four input coordinates. Each query coordinate is a dot product with one column of W_Q. These sparse matrices are hand-chosen; learned projections need not be sparse or interpretable. The head superscripts label heads, not powers. Query/key width is two per head here; Part II’s single-head toy used three.</p>',
          notes='Where does the first 2.3 in the query come from?\nMultiply the four input entries by the first W_Q column, then add. Repeat for the second column.'),
        s('s02-v-query-person','Head 2: computing the person query',queries(R,data,1),
          r'The input embedding stays [0, 0, 0, 2.3]. Head 2 uses its own \(W_Q\). Its query requests the person feature and gives zero weight to the glue matching feature.',
          notes='Did we change the input token or its embedding?\nKeep the blue input fixed. Compare the last rows of the two query matrices.'),
        s('s02-v-sources','Source rows supply keys and values',source_rows(R),
          r'Each head has its own \(W_K\) and \(W_V\), both \(4\times2\) here. For easy arithmetic, they select the same coordinates. Keys determine matching; values carry the numbers we mix.',
          companion='<p>Head 1 uses W_K = W_V = [[1,0],[0,1],[0,0],[0,0]], selecting water and finance. Head 2 uses W_K = W_V = [[0,0],[0,0],[1,0],[0,1]], selecting person and glue. Each matrix is [4,2] and acts on every source row, including sources not shown here. W_K and W_V are distinct parameters with equal numerical entries in this worksheet. They need not be equal in a trained model. In Part II’s Maya example they served different roles too.</p>',
          notes='Why are k and v equal here?\nWe chose equal projection entries for arithmetic. Point to the different uses of these same numbers on the next score and message slides.'),
        *head_walkthrough(s,R,0),
        *head_walkthrough(s,R,1),
        s('s02-v-message','The two complete head calculations',messages(R),
          'Each head has now computed its own scores, weights and message. We keep both messages, then concatenate and project them into the embedding update.',
          companion='<p>As in Part II, αᵢⱼ is one attention weight and A stores the full attention matrix. The walkthrough follows its final row, A[10, :], using one-based token positions. Q/K choose weights and V supplies the weighted content. All ten source contributions enter each message. The calculations are independent across heads and can run in parallel.</p>',
          notes='Where do the heads first combine?\nTheir score lists, softmaxes and value sums stay separate. Concatenation and W_O combine the resulting messages in the next section.'),
        s('s02-v-wide','What if we made one head wider?',wide_match(R),
          'A wider query/key can compare more features. Their products still add into one score per source, followed by one softmax.',
          companion='<p>For this controlled comparison, concatenate the two existing Q, K and V projections into width-four projections. A single wide head scales its dot products by √4. The two narrower heads each scale by √2 and normalize separately. The example changes no projection entries and does not compare separately trained models.</p>',
          notes='Could a four-coordinate query read both kinds of clues?\nYes. Point to the two dot-product contributions adding into a single score for each source.'),
        s('s02-v-separate','Two heads keep separate source preferences',wide_weights(R),
          'A wider value vector carries more features with one shared weight row. Two heads can favour river for setting features and fisherman for person features.',
          companion='<p>Every displayed attention row sums to one over all ten sources; “others” combines the remaining eight. One wide head uses its single row for all four value coordinates. The two heads use different rows for their two-coordinate values, preserving both mixtures before W_O combines them. This is a useful architectural choice, not proof that one wide head cannot learn useful relationships or that more heads always improve accuracy. See <a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need, §3.2.2</a>.</p>',
          notes='Which word receives the largest weight in each row?\nCompare river and fisherman. Total value width stays four; the number of separately normalized mixtures changes.')])
    add('How do the two messages update the token?',[
        s('s03-v-break','What reaches the receiving token?',divider('Combining the head messages','Concatenate, project into embedding space, then add the update.'),''),
        s('s03-v-join','Put the messages side by side',join(R),
          'Concatenation keeps the two messages separate. It does not add them coordinate by coordinate.'),
        s('s03-v-output','Map the messages back to embedding space',output(R,data),
          r'\(W_O\) learns how the head messages contribute to the embedding update. Four joined coordinates become four update coordinates.',
          companion='<p>Equivalently, split W_O into a two-row block per head. Then Δe = m¹ W_O¹ + m² W_O²: each head contributes a four-coordinate update. Concatenation followed by one matrix multiplication computes exactly that sum. We retain the standard concat notation from the original Transformer paper.</p>'),
        s('s03-v-residual','Add context to the original embedding',residual(R),
          r'Exactly as in Part II: \(e_i\) is the input embedding row, \(\Delta e_i\) is the context update, and \(e_i^{\prime}=e_i+\Delta e_i\).'),
        s('s03-v-map','Return to the next-token prediction',full_map(),
          'The last updated row still feeds the prediction MLP. More heads change how it reads context—not what the target means.'),
        s('s03-v-explore','Try the other bank', '', '<div id="s04-head-explorer"></div>',
          companion='<p>Change to the cheque sentence. These controls recalculate Q, K, V, both attention rows, the messages and the vocabulary prediction from the hand-chosen parameters. This is a worked arithmetic explorer, not a trained language model.</p>')])
    add('What are the matrix shapes?',[
        s('s04-v-break','Can every token do this at once?',divider('The same calculation for every token','Trace the receiver row through the full matrices.'),''),
        s('s04-v-project','Project every row with the same head matrix',projection_map(R),
          'Ten input rows × four coordinates. Multiplying by a 4 × 2 projection gives ten query rows × two coordinates.'),
        s('s04-v-grid','One head makes one attention grid',attention_grid(R),
          r'Keep Part II’s notation: \(M\) is the causal mask, \(A=\operatorname{softmax}(QK^\top/\sqrt{d_k}+M)\), and \(H=AV\) stores the message rows.'),
        s('s04-v-parallel','Two heads make two attention grids',stacked_maps(R),
          'Both heads receive the same E. Each uses its own projections, attention grid and values. Neither reads the other head’s output.'),
        s('s04-v-joined','Join coordinates, not token rows',join_matrices(),
          r'\(\Delta E=\operatorname{Concat}(H^{(1)},H^{(2)})W_O\), then \(E^{\prime}=E+\Delta E\). The superscript labels the head; each matrix still has ten token rows.')])
    add('Write the familiar operations twice',[
        s('s05-v-scratch','One head is still the Part II calculation',svg(t(580,55,'E → Q, K, V → scores → weights → message',31,'q','middle'),100,'One head in code'),
          'No new attention rule is needed. We reuse the same calculation with different learned matrices.',code='''def head(E, W_Q, W_K, W_V):
    Q, K, V = E @ W_Q, E @ W_K, E @ W_V
    scores = Q @ K.T / math.sqrt(Q.shape[-1])
    future = torch.ones(len(E), len(E), dtype=torch.bool).triu(1)
    A = scores.masked_fill(future, -torch.inf).softmax(-1)
    return A @ V'''),
        s('s05-v-combine','Call it with two sets of parameters',svg(t(580,54,'H¹, H² → concatenate columns → W_O → ΔE',31,'v','middle'),100,'Combine two head outputs in code'),
          'This unbatched example keeps one row per token. The notebook checks these outputs against the printed numbers.',code='''H1 = head(E, W_Q1, W_K1, W_V1)
H2 = head(E, W_Q2, W_K2, W_V2)
joined = torch.cat([H1, H2], dim=-1)  # [10, 4]
delta_E = joined @ W_O
E_prime = E + delta_E'''),
        s('s05-v-train','The same loss trains both heads',full_map('heads join predict train'),
          'Next-token cross-entropy trains all the projections together. We do not label training examples “setting head” or “person head”.',
          companion='<p>At training time, the observed next-token ID is the target for the vocabulary logits. At inference time, weights remain fixed: tokenize a prefix, run the model, choose a next token and append it. These steps are unchanged from Part II; the full executable loop remains in Notebook 7.</p>'),
        s('s05-v-bias','A bias adds a learned offset',bias_example(R),
          'The offset [0.2, −0.1] is illustrative. With bias=True, training learns it along with the projection weights. We use bias=False so the code matches the worksheet.',
          companion='<p>A projection bias is shared across token positions (and batches). Position embeddings depend on the position, so these are different operations. A bias can shift the projection even for a zero input. Omitting it keeps this example simpler; it is not a general claim that bias-free attention is better.</p>',
          notes='Where does the offset enter?\nAfter the matrix multiply. Compare [2.3, 2.3] with [2.5, 2.2]; the two-coordinate shape is unchanged.'),
        s('s05-v-bias-layers','Which projections use the bias flag?',bias_locations(),
          'PyTorch defaults to bias=True. In this lesson, bias=False removes the Q, K, V and output offsets. It leaves the mask, position embeddings and separate prediction layers unchanged.',
          companion='<p>With total width four, bias=True adds four query offsets, four key offsets, four value offsets and four output offsets: 16 additional trainable scalars. Within each head the Q/K/V bias slice has width two. PyTorch stores the three input offsets together in in_proj_bias and the output offset in out_proj.bias. The distinct add_bias_kv option is not the bias flag discussed here. <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html">Official API documentation</a>. Both our scratch implementation and trained attention variants omit attention projection biases; their prediction MLP layers retain biases.</p>'),
        s('s05-v-pytorch','PyTorch packages the head calculation',svg(t(580,53,'E → nn.MultiheadAttention → ΔE',32,'q','middle'),100,'PyTorch returns the projected update, before our residual'),
          'Here E has shape [10, 4]: one unbatched sequence. PyTorch handles the projections and mixing. A contains two 10 × 10 weight grids.',code='''mha = nn.MultiheadAttention(embed_dim=4, num_heads=2, bias=False)
future = torch.ones(10, 10, dtype=torch.bool).triu(1)
delta_E, A = mha(E, E, E, attn_mask=future,
                 average_attn_weights=False)
E_prime = E + delta_E''',
          companion='<p>The library layer does not add position embeddings, the residual or the vocabulary classifier. attn_mask=future blocks future sources. average_attn_weights=False returns each head’s weight grid separately; it changes the returned diagnostic weights, not how the head messages combine. The notebook copies identical weights from our scratch implementation to PyTorch and checks both ΔE and the per-head attention weights. See the <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html">PyTorch documentation</a>.</p>')])
    import json
    from pathlib import Path
    result=json.loads((Path(__file__).resolve().parents[1]/'notebooks/wordlm/artifacts/heads/comparison.json').read_text())
    labels={'mlp':'Embedding → MLP','attention':'One attention head','multihead':'Four attention heads'}
    scores=[[labels[k],f(result['aggregate'][k]['test_loss']['mean'],3),f(result['aggregate'][k]['test_perplexity']['mean'],2)] for k in labels]
    costs=[[labels[k],f'{result["runs"][k][0]["parameter_count"]:,}',f(result['aggregate'][k]['runtime_seconds']['mean'],1)+' s'] for k in labels]
    add('Do more heads help our trained model?',[
        s('s06-v-width','Keep the total width fixed',widths(),
          'The trained comparison uses width 64. Four smaller heads give four attention patterns with the same attention parameter count as one wide head.',
          companion='<p>Implementations often concatenate the per-head W_Q matrices into one large W_Q (and similarly for K and V). They project E first and reshape the projected coordinates into heads. They do not assign disjoint slices of raw E to different heads. Our two-head arithmetic toy uses width 4; this experiment uses width 64.</p>'),
        s('s06-v-scores','Compare held-out next-token prediction',results_table(scores,['Model','Cross-entropy ↓','Perplexity ↓']),
          'For each run, perplexity = exp(cross-entropy using natural logs). Lower means higher probability for observed tokens. The table averages three seeds on the same test stories.',
          companion='<p>Same 6,000-document TinyStories subset, tokenizer, 64-token context, 6,000-update budget and validation selection. Seeds 11, 29 and 47. More heads helped this experiment; this is not a guarantee for every dataset, head count or generated continuation. <a href="notebooks/wordlm/06_multihead_comparison.html">Full experiment and variation across seeds</a>.</p>'),
        s('s06-v-cost','Compare size and training time too',results_table(costs,['Model','Parameters','Training / seed']),
          'Mean training times on Apple M2 Max/MPS, including validation. The two attention models have equal parameter counts; four heads took slightly longer.'),
        s('s06-v-demo','Give all three models the same prompt',svg(t(580,75,'Training prefix → held-out story → a different kind of prompt',29,'ink','middle')+t(580,155,'Compare the continuation and generation time on your device.',27,'muted','middle'),212,'Compare real language models in the browser'),
          '<a href="word-lab/" target="_blank" rel="noopener">Open the live models ↗</a> · <a href="notebooks/wordlm/07_multihead_step_by_step.html">Work through the code</a> · <a href="notebooks/wordlm/06_multihead_comparison.html">Inspect the trained experiment</a>',
          companion='<p>The demo runs actual checkpoints with WebGPU or WebAssembly. It includes training, held-out and outside-domain prompts and measures generation locally. Normalization, full Transformer blocks and complexity remain in the <a href="part2b.html">optional Part 2B reference</a>. Formula source: <a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need, §3.2.2</a>. Visual teaching references: <a href="https://www.3blue1brown.com/lessons/attention/">3Blue1Brown</a> and <a href="https://jalammar.github.io/illustrated-transformer/">Jay Alammar</a>.')])
    add('Next: classify an image with the same attention operation',[
        s('s07-v-patches','Image patches can supply the embedding rows',vision_bridge(),
          'Replace word tokens with image patches. Flatten each patch and apply the same learned projection. Attention still receives a sequence of embedding rows.',
          companion='<p>This is a tiny shape illustration, not a trained vision model: an 8×8 grayscale image split into four 4×4 patches. Each patch has 16 pixel values; a shared 16×4 projection produces four-dimensional rows. For RGB images the flattened width is P²×3. The general sequence length is (H/P)×(W/P). The projection is learned from the image task.</p>',
          notes='What changes from text, and what stays the same?\nThe source of the rows changes. Q/K/V projections, separate head weights and message mixing remain.'),
        s('s07-v-classifier','One updated row can classify the whole image',vision_bridge(classifier=True),
          'The [CLS] row gathers image information. Its final representation feeds a classifier trained with the image label. <a href="vision1.html">Next: Vision I, from pixels to an image class ↗</a>',
          companion='<p>For this two-class shape example, prepend one learned CLS token to the four patch embeddings, then add position embeddings to the five rows. A Transformer encoder updates all five rows. The final CLS row feeds a two-logit classifier; softmax produces class probabilities and cross-entropy uses the observed image label. Image classification does not need a causal future-token mask. The full encoder also includes per-row MLPs, residuals and normalization, which are grouped here rather than silently omitted. This is a roadmap, not a benchmark or a completed classifier. See <a href="https://arxiv.org/abs/2010.11929">An Image is Worth 16×16 Words</a>. <a href="part4.html">Cross-attention remains a separate optional continuation</a>.</p>',
          notes='What is the prediction target now?\nAn image class, rather than the next token. Which row goes to the classifier? The updated CLS row—not the initial learned vector.')])
    return parts
