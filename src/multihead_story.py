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


def bottleneck(case):
    H=case['heads'][0];body=''
    body+=t(700,28,'source position: 1 → 10',23,'muted','middle')
    for row,(name,index) in enumerate([('water coordinate',0),('finance coordinate',1)]):
        y=60+row*139
        body+=t(20,y+28,name,25,'v')
        for j in range(10):
            xx=298+j*81
            body+=rect(xx,y,75,55,'a',COLORS['a']+f'{int(20+150*H["A"][-1][j]):02x}')
            body+=t(xx+37.5,y+35,f(H['A'][-1][j],2),21,'a','middle')
        body+=t(700,y+85,'the same ten source weights',22,'muted','middle')
    body+=t(580,352,'A second head can use a different row of weights.',31,'ink','middle')
    return svg(body,390,'One head shares its source weights across value coordinates')


def divider(number,question,sub):
    body=t(35,93,number,27,'muted')+t(35,181,question,43,'ink',weight=650)+t(35,250,sub,29,'muted')
    return svg(body,325,question)


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


def queries(case,data):
    body=t(580,31,'Same input row e₁₀ = [0, 0, 0, 2.3]',29,'e','middle')
    labels=['water','finance','person','glue']
    for h in range(2):
        x=90+h*585
        body+=t(x+226,93,f'Head {h+1}',29,'ink','middle',650)
        body+=t(x+226,130,'W_Q¹ · 4 × 2' if h==0 else 'W_Q² · 4 × 2',25,'q','middle')
        for j,label in enumerate(['water?','finance?'] if h==0 else ['person?','glue?']):
            body+=t(x+176+j*100,159,label,19,'muted','middle')
        for i,row in enumerate(data['projections'][h]['Q']):
            body+=t(x+108,196+i*38,labels[i],21,'muted','end')
            for j,value in enumerate(row):
                body+=rect(x+126+j*100,170+i*38,100,38,'q',COLORS['q']+'08')
                body+=t(x+176+j*100,196+i*38,str(value),24,'q','middle')
        body+=arrow(x+226,327,x+226,350,'q')
        body+=t(x+80,387,'q₁₀¹' if h==0 else 'q₁₀²',26,'q','end')
        body+=cells(x+105,355,[f(v,1) for v in case['heads'][h]['Q'][-1]],'q',100)
    return svg(body,430,'Two different query matrices read all four input coordinates')


def matching(case):
    body=''
    for h,j in [(0,5),(1,1)]:
        y=52+187*h;H=case['heads'][h];q=H['Q'][-1];k=H['K'][j]
        body+=t(24,y,f'Head {h+1}',29,'ink',weight=650)
        body+=t(225,y,'query',22,'q')+t(455,y,f'{case["tokens"][j]} key',22,'k')
        body+=cells(224,y+17,[f(v,1) for v in q],'q',86)
        body+=t(421,y+51,'·',31,'ink','middle')
        body+=cells(451,y+17,[f(v,1) for v in k],'k',86)
        body+=t(665,y+48,'÷ √2',28,'muted')+arrow(756,y+40,817,y+40)
        body+=t(846,y+48,f(H['scores'][-1][j],3),30,'a')
        body+=t(224,y+111,f'({f(q[0],1)} × {f(k[0],1)} + {f(q[1],1)} × {f(k[1],1)}) / √2',24,'muted')
    return svg(body,416,'Compute a query-key score separately within each head')


def weights(case):
    body=''
    for h in range(2):
        y=75+h*165;H=case['heads'][h]
        body+=t(15,y+25,f'Head {h+1}',25,'ink',weight=650)
        x=155
        widths=[70,137,70,101,70,88,84,70,119,70]
        for j,word in enumerate(case['tokens']):
            w=widths[j]
            body+=t(x+w/2,y-20,word,19,'ink','middle')
            body+=rect(x,y,w,49,'line',COLORS['a']+f'{int(18+165*H["A"][-1][j]):02x}')
            body+=t(x+w/2,y+33,f(H['A'][-1][j],3),23,'a','middle')
            x+=w+10
        body+=t(590,y+91,'row sum = 1',23,'muted','middle')
    return svg(body,381,'Each head normalizes its own scores over the same source positions')


def messages(case):
    body=''
    for h,j in [(0,5),(1,1)]:
        y=36+h*195;H=case['heads'][h];a=H['A'][-1][j];v=H['V'][j]
        body+=t(24,y+2,f'Head {h+1} · {case["tokens"][j]} contributes',26,'ink',weight=650)
        body+=t(43,y+68,f(a,3),29,'a')+t(150,y+68,'×',30)
        body+=cells(186,y+33,[f(x,1) for x in v],'v',97)
        body+=t(420,y+69,'+',30)+t(470,y+55,'other weighted',23,'muted')+t(470,y+85,'value rows',23,'muted')
        body+=arrow(679,y+61,753,y+61,'v')
        body+=cells(788,y+33,[f(x,3) for x in H['messages'][-1]],'v',147)
        body+=t(938,y+2,'m₁₀¹' if h==0 else 'm₁₀²',26,'v','middle')
        labels=['water','finance'] if h==0 else ['person','glue']
        for k,label in enumerate(labels):
            body+=t(234+k*97,y+111,label,22,'muted','middle')
            body+=t(861+k*147,y+111,label,22,'muted','middle')
    return svg(body,393,'Each head weights its own value rows to obtain its message')


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
        body+=path(f'M786 {y+35} H821 V170 H867','v')
    body+=node(867,136,265,'Join → W_O','ΔE · 10 × 4','d','join')
    body+=path('M110 136 V40 H1148 V379 H1046','e',2,'7 6')+arrow(1046,379,1025,379,'e')
    body+=t(980,25,'keep original E',20,'e','middle')
    body+=arrow(1000,205,1000,353,'d')
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
        s('s01-v-one','One head can read the setting',reading(R,0),
          'In this hand-chosen two-head example, the first head gives river a large weight. Its values carry setting information to the receiver.',
          companion='<p>These are hand-chosen two-head parameters applied to Part II’s exact input rows. “Setting” and “person” name the intended behaviour of this example, not jobs assigned to trained heads. Arrows show information moving from source to receiver; their widths encode computed attention weights.</p>'),
        s('s01-v-two','Another head can read who is there',reading(R,1),
          'The sentence and receiver stay fixed. A different head gives fisherman a large weight.',
          notes='What changed: the words, the receiver, or how we read?\nPoint to the same token positions; compare the two thick arrows.'),
        s('s01-v-shared','One head shares one set of source weights',bottleneck(R),
          'A single head can mix many words. But every coordinate of its message uses the same attention-weight row.'),
        s('s01-v-both','Keep both readings',reading(R,both=True),
          'Two heads can retain different source mixtures at the same token. They run in parallel, not one after the other.')])
    add('How does a second head get a different view?',[
        s('s02-v-break','How do we make the readings different?',divider('01 → 02','Same embeddings. Different projections.','Each head learns its own W_Q, W_K and W_V.'),
          '',companion='<p>Visual inspiration: <a href="https://www.3blue1brown.com/lessons/attention/">3Blue1Brown’s attention lesson</a> and <a href="https://jalammar.github.io/illustrated-transformer/">Jay Alammar’s Illustrated Transformer</a>. We keep our own river-bank example, numbers and Part II row-vector convention. A head-specific superscript labels a head; a subscript still labels a token.</p>'),
        s('s02-v-plan','Give both heads the same input',full_map('heads'),
          'We will zoom into these two branches. Each starts with the same embedding rows and sends back its own message.',
          notes='What does the second head receive as input?\nTrace both branches from E. Neither branch starts at the other head’s output.'),
        s('s02-v-query','The same embedding makes two queries',queries(R,data),
          'Both heads read all four input coordinates. Superscripts label the heads. We choose two query/key coordinates per head here; Part II’s toy used three.',
          companion='<p>E contains position-aware embedding rows, not token IDs. The labelled water, finance, person and glue axes are invented for this teaching example. The displayed W_Q matrices are deliberately sparse; learned projections need not be. Superscripts 1 and 2 identify the two heads, not exponentiation.</p>'),
        s('s02-v-match','Each query meets keys from its own head',matching(R),
          'Head 1 exposes water and finance in its keys. Head 2 exposes person and glue. Score every source with the matching query.'),
        s('s02-v-weights','Each head gets its own weight row',weights(R),
          r'Softmax runs across source positions, separately for each head. As before, \(\alpha_{ij}\) is one weight and \(A\) is the whole weight matrix.'),
        s('s02-v-message','Values turn those weights into messages',messages(R),
          'Keys set the weights; values carry the content. Here we chose equal key/value numbers, but their learned projections are separate.',
          companion='<p>In this particular worksheet W_K and W_V have equal numerical entries within each head, so the displayed key and value numbers coincide. They remain separate parameters with different roles: keys set scores; values are mixed into messages. All ten source contributions, including those not expanded on the slide, enter each result.</p>'),
        s('s02-v-wide','What if we made one head wider?',wide_match(R),
          'A wider query/key can compare more features. Their products still add into one score per source, followed by one softmax.',
          companion='<p>For this controlled comparison, concatenate the two existing Q, K and V projections into width-four projections. A single wide head scales its dot products by √4. The two narrower heads each scale by √2 and normalize separately. The example changes no projection entries and does not compare separately trained models.</p>',
          notes='Could a four-coordinate query read both kinds of clues?\nYes. Point to the two dot-product contributions adding into a single score for each source.'),
        s('s02-v-separate','Two heads keep separate source preferences',wide_weights(R),
          'A wider value vector carries more features with one shared weight row. Two heads can favour river for setting features and fisherman for person features.',
          companion='<p>Every displayed attention row sums to one over all ten sources; “others” combines the remaining eight. One wide head uses its single row for all four value coordinates. The two heads use different rows for their two-coordinate values, preserving both mixtures before W_O combines them. This is a useful architectural choice, not proof that one wide head cannot learn useful relationships or that more heads always improve accuracy. See <a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need, §3.2.2</a>.</p>',
          notes='Which word receives the largest weight in each row?\nCompare river and fisherman. Total value width stays four; the number of separately normalized mixtures changes.')])
    add('How do the two messages update the token?',[
        s('s03-v-break','What reaches the receiving token?',divider('02 → 03','Two messages become one context update.','Keep both messages, map back, then add to the original embedding.'),''),
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
        s('s04-v-break','Can every token do this at once?',divider('03 → 04','Stack the token rows into matrices.','Trace the same receiver row through the full calculation.'),''),
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
    return parts
