"""Build the complete Vision I lecture using the existing series shell.

Run with Python + NumPy. No browser, training, network, or publication occurs.
"""
from pathlib import Path
from html import escape
import base64
import json
import math
import subprocess
import sys
from vision1_worksheet import export

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
ASSETS = ROOT / 'figures/vision1'
OUT = SRC / 'sections-vision1'
OUT.mkdir(exist_ok=True)
DATA = export()
R = DATA['cases']['horizontal_positions']
REAL = json.loads((ASSETS / 'real-inference.json').read_text())
PHOTO = 'data:image/jpeg;base64,' + base64.b64encode((ASSETS / 'newfoundland_31.jpg').read_bytes()).decode()
CAT = 'data:image/jpeg;base64,' + base64.b64encode((ASSETS / 'Persian_98.jpg').read_bytes()).decode()
FRAMES = []


def f(x, n=3):
    return f'{x:.{n}f}'.replace('-', '−')


def v(xs, n=3):
    return '[' + ', '.join(f(x, n) for x in xs) + ']'


def t(x, y, content, size=27, color='ink', anchor='start', weight=500):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="var(--{color})" text-anchor="{anchor}" font-weight="{weight}">{escape(str(content))}</text>'


def g(content, step):
    return f'<g data-build="{step}">{content}</g>'


def line(x1, y1, x2, y2, color='line', width=2, dash=''):
    return f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="var(--{color})" stroke-width="{width}" stroke-dasharray="{dash}"/>'


def arrow(x1, y1, x2, y2, color='ink-3'):
    a = math.atan2(y2-y1, x2-x1)
    p = [(x2-11*math.cos(a+s), y2-11*math.sin(a+s)) for s in [-.4, .4]]
    return line(x1,y1,x2,y2,color,2.5) + line(*p[0],x2,y2,color,2.5) + line(x2,y2,*p[1],color,2.5)


def rect(x,y,w,h,color='line',fill='card',rx=4):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="var(--{fill})" stroke="var(--{color})" stroke-width="2"/>'


def image(x,y,w,h,uri=PHOTO):
    return f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{uri}" preserveAspectRatio="xMidYMid meet"/>'


def crop(x,y,w,h,index,uid):
    # A genuine crop of the same original 500 x 334 photograph.
    col,row=index%4,index//4
    return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="{col*125} {row*83.5} 125 83.5" overflow="hidden">'+image(0,0,500,334)+'</svg>'


def pixels(x,y,values,size=34,labels=False,patch_lines=False):
    b=''; rows=len(values); cols=len(values[0])
    for r in range(rows):
        for c in range(cols):
            val=values[r][c]
            b+=rect(x+c*size,y+r*size,size,size,'line','ink' if val else 'card',0)
            if labels:b+=t(x+(c+.5)*size,y+(r+.68)*size,int(val),22,'card' if val else 'ink','middle')
    if patch_lines:
        b+=line(x+2*size,y-5,x+2*size,y+rows*size+5,'ink-3',3,'5 4')
        b+=line(x-5,y+2*size,x+cols*size+5,y+2*size,'ink-3',3,'5 4')
    return b


def source_icon(x,y,j):
    if j==0:return rect(x,y,54,36,'c-e','t-e')+t(x+27,y+25,'CLS',19,'c-e','middle')
    return pixels(x,y,[R['patches'][j-1][:2],R['patches'][j-1][2:]],18)+t(x+48,y+27,f'P{j}',25)


def svg(body,label):
    colors='--ink:#14171F;--ink-2:#4A5160;--ink-3:#6B7280;--line:#D9DDE5;--card:#FFFFFF;--transparent:transparent;--c-e:#245EDB;--t-e:#E4ECFF;--c-q:#8B2CDE;--t-q:#F1E5FC;--c-k:#AA4E08;--c-v:#0F766E;--c-a:#BE123C;--c-d:#147737;'
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 440" role="img" aria-label="{escape(label)}" style="font-family:Avenir Next,Segoe UI,sans-serif;{colors}"><title>{escape(label)}</title>{body}</svg>'


def frame(key,title,body,caption,notes,companion='',mobile=''):
    markup=svg(body,title)
    # Standalone figures also preserve the original photograph for inspection.
    (ASSETS/(key+'.svg')).write_text(markup)
    FRAMES.append({'id':key,'title':title,'caption':caption,'notes':notes})
    return f'''<div class="frame vp-frame" id="{key}" data-title="{escape(title)}" data-autobuild="off">
<script type="text/x-notes">{escape(notes)}</script>
<div class="vp-figure{' vp-desktop' if mobile else ''}">{markup}</div>
{f'<div class="vp-mobile">{mobile}</div>' if mobile else ''}
<p class="vp-caption">{caption}</p></div>
{f'<div class="companion">{companion}</div>' if companion else ''}'''


def mobile_rows(headers, rows):
    return '<table class="vp-table"><thead><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(vv))+'</td>' for vv in row)+'</tr>' for row in rows)+'</tbody></table>'


def section(n,title,frames,lit=''):
    text=f'<section id="s{n:02}" class="sec" data-title="{escape(title)}" data-lit="{lit}"><header class="sec-head"><span class="sec-num">{n:02}</span><div><h2>{escape(title)}</h2></div></header>'+''.join(frames)+'</section>'
    if n==1:text='<style>'+ (SRC/'vision1-lesson.css').read_text()+'</style>'+text
    (OUT/f'sec{n:02}.html').write_text(text)


def motivation():
    frames=[]
    b=image(25,35,590,394)+t(715,115,'One photograph.',35)+t(715,170,'One image label.',35)
    b+=g(t(715,275,'What clues did you use?',28)+line(715,302,1110,302,'ink'),1)
    frames.append(frame('s01-photo','What is in this image?',b,'Start with the same task as Part I: produce scores for possible answers.',
        'What animal is this, and which parts made you decide?\nPoint to the face, fur, and silhouette before mentioning an architecture.',
        'In Part I, a prefix led to scores for possible next characters. Here a photograph leads to scores for possible image labels. The input and label vocabulary change; the idea of a classifier remains. This is a real Oxford-IIIT Pet image, newfoundland_31.'))
    b=crop(25,100,290,194,9,'isolated')+t(170,335,'P10 · one crop',26,'ink-2','middle')
    b+=g(arrow(345,200,445,200)+image(470,35,600,401)+rect(620,235.5,150,100.25,'c-e','transparent',0),1)
    frames.append(frame('s01-context','What can this patch tell us on its own?',b,'Part II: “bank” needed context. A patch can need context too.',
        'Could this dark crop be fur, a shadow, or something else?\nReveal where it came from; keep pointing to the same crop.',
        'The analogy is about contextual representation. A patch is a fixed piece of the input, not a word or an object with a ready-made semantic label. Its neighbours and distant patches can supply useful information. The outlined crop is an actual region of the opening photograph.'))
    b=image(25,65,500,334)
    grid=''
    for j in range(1,4):grid+=line(25+125*j,65,25+125*j,399,'card',2)+line(25,65+83.5*j,525,65+83.5*j,'card',2)
    b+=g(grid,1)
    for j in range(4):
        x=625+(j%2)*235;y=55+(j//2)*185
        idx=[5,6,9,10][j]
        b+=g(crop(x,y,200,133.6,idx,'patch')+t(x+100,y+163,f'P{idx+1}',24,'ink-2','middle'),2)
    frames.append(frame('s01-patches','Cut on a grid; keep the pieces identifiable',b,'The grid can cut across a face, fur, and background.',
        'Does each patch contain one whole object?\nReveal the grid, then match each enlarged crop back to its location.',
        'This drawing uses a 4×4 grid so the crops remain large. The real model used at the end takes a 224×224 image and uses 16×16-pixel patches, giving a 14×14 grid. Patch boundaries are chosen before recognition.'))
    b=''
    for j,idx in enumerate([5,6,9,10]):
        y=20+j*101;b+=crop(35,y,126,84,idx,'strip')+t(190,y+48,f'P{idx+1}',25)
        b+=g(arrow(250,y+43,325,y+43)+rect(345,y+20,270,47,'c-e','t-e')+t(480,y+52,'pixel values',25,'c-e','middle'),1)
        b+=g(arrow(645,y+43,775,y+43)+rect(800,y+20,300,47,'c-e','t-e')+t(950,y+52,f'e{idx+1} · same width',25,'c-e','middle'),2)
    b+=g(t(710,430,'one shared projection',23,'ink-2','middle'),2)
    frames.append(frame('s01-rows','An image supplies the embedding rows',b,'Part I learned a vector for a character. Here, a shared linear layer maps each patch to a vector.',
        'Do we need a different projection for every location?\nFollow the four rows through the same operation. Locations enter separately.',
        'At realistic scale, each RGB patch has 16×16×3 = 768 pixel values. The tiny pretrained ViT in this lecture maps each patch to 192 coordinates. The hand worksheet next uses four pixel values and four coordinates so the entire calculation is visible.',
        mobile_rows(['Source','Shared operation'],[[f'P{i+1}','pixel row → linear projection → embedding'] for i in [5,6,9,10]])))
    section(1,'What does a patch need from its image?',frames)


def worksheet_intro():
    frames=[]
    b=pixels(90,45,DATA['images']['horizontal'],67,False,True)+t(224,365,'Across the top',29,'ink','middle')
    b+=g(pixels(720,45,DATA['images']['vertical'],67,False,True)+t(854,365,'Down the left',29,'ink','middle'),1)
    b+=g(arrow(420,180,650,180)+t(535,235,'move the same patches',24,'ink-2','middle'),2)
    frames.append(frame('s02-small','Same patches. Can the answer change?',b,'Shrink the problem: four patches, two possible arrangements, every number visible.',
        'How many filled and empty patches are in each image?\nReveal the second arrangement. Count two of each, then ask what changed.',
        'Our arithmetic exercise distinguishes these two arrangements: across the top and down the left. A filled pixel is 1 and an empty pixel is 0. We will choose small parameters rather than fit them. The worksheet isolates attention and a residual; LayerNorm and the block MLP return when we assemble the full ViT.'))
    raw=R['patches'][0]
    b=pixels(25,72,[raw[:2],raw[2:]],80,True)+t(105,282,'P1',26,'ink','middle')
    b+=g(arrow(225,150,300,150)+t(340,150,v(raw,0),34,'c-e'),1)
    b+=g(t(340,225,'ink = ¼·1 + ¼·1 + ¼·1 + ¼·1',29,'c-e')+t(340,275,'= 1',36,'c-e'),2)
    b+=g(t(340,370,'content row = [1, 0, 0, 1]',34,'c-e'),3)
    frames.append(frame('s02-projection','Turn one patch into four coordinates',b,'A shared patch projection; its last coordinate is a constant bias.',
        'What is the mean of these four pixels?\nUnroll P1 in row order. Compute the first coordinate, then reveal the full content row.',
        'The worksheet uses W_patch with four identical rows [¼,0,0,0] and bias [0,0,0,1]. The first coordinate is ink (the fraction of black pixels); two empty coordinates will receive location; the final 1 makes the chosen queries easy to inspect. These are engineered teaching coordinates, not claimed meanings of learned ViT features.',
        '<p>P1 pixels: [1, 1, 1, 1]</p><p>Ink = ¼ + ¼ + ¼ + ¼ = 1.</p><p>Content row: [1, 0, 0, 1].</p>'))
    b=t(20,35,'Source',24,'ink-2')+t(220,35,'content',25,'c-e')+t(560,35,'location',25,'ink-2')+t(870,35,'input row E',25,'c-e')+line(20,52,1130,52)
    for j in range(5):
        y=76+66*j; b+=source_icon(25,y,j)
        content=DATA['cls'] if j==0 else R['content'][j-1]
        b+=t(220,y+27,v(content,0),27,'c-e')
        b+=g(t(488,y+27,'+',28)+t(560,y+27,v(DATA['positions'][j],0),27,'ink-2'),1)
        b+=g(t(810,y+27,'=',28)+t(870,y+27,v(R['E'][j],0),27,'c-e'),2)
    frames.append(frame('s02-positions','Add where the patch came from',b,'The row and column coordinates distinguish locations. CLS starts as a learned summary row in a real ViT.',
        'What distinguishes two identical filled patches now?\nPoint to the row and column coordinates. CLS is an extra row; it is not an image crop.',
        'Rows and columns are numbered 0 and 1. The five inputs are CLS followed by P1–P4 in raster order. This worksheet fixes CLS = [0,0,0,1]. A real ViT learns its initial CLS and positional vectors. The initial CLS is shared across images; its first query has not already seen this photograph.',
        mobile_rows(['Source','Input [ink,row,col,1]'],[['CLS' if j==0 else f'P{j}',v(row,0)] for j,row in enumerate(R['E'])])))
    section(2,'What changes when we move the patches?',frames,'e')


def routing():
    frames=[]
    b=rect(25,65,330,90,'c-e','t-e')+t(190,92,'CLS input',25,'c-e','middle')+t(190,139,v(R['E'][0],0),30,'c-e','middle')
    b+=g(arrow(385,110,490,110)+t(435,85,'W_Q¹',25,'ink-2','middle')+rect(515,65,250,90,'c-q','t-q')+t(640,92,'q₀¹',28,'c-q','middle')+t(640,139,'[1, 1]',31,'c-q','middle'),1)
    b+=g(t(25,230,'Head 1 keys keep ink and row:',28)+t(25,305,'kⱼ¹ = [√2 × ink, √2 × row]',32,'c-k'),2)
    b+=g(t(25,385,'P1: [1, 0, 0, 1]  →  k₁¹ = [√2, 0]',31,'c-k'),3)
    frames.append(frame('s03-query','Which features does this head compare?',b,'Part II again: the query and keys choose where to read.',
        'What does multiplying this input by W_Q do?\nPoint to the fixed last coordinate supplying [1,1]. Then construct P1’s key.',
        'Both heads receive the same five E rows. Their matrices are independent parameters. For readable arithmetic we choose both CLS queries as [1,1]; Head 1 keys carry ink and row, while Head 2 keys carry ink and column. Every matrix is shown in this lecture and included in the notebook.',
        '<p>CLS input: [0,0,0,1]</p><p>Query: [1,1]</p><p>Head 1 key = [√2 × ink, √2 × row].</p><p>P1 key: [√2,0].</p>'))
    h=R['heads'][0]
    b=t(25,35,'Source',24,'ink-2')+t(200,35,'key kⱼ¹',25,'c-k')+g(t(475,35,'q₀¹ · kⱼ¹ / √2',25),1)+g(t(760,35,'exp(score)',25),2)+g(t(1005,35,'weight',25,'c-a'),3)+line(25,54,1130,54)
    for j in range(5):
        y=74+j*55;b+=source_icon(25,y,j)
        key=['√2' if abs(x-math.sqrt(2))<1e-8 else '0' for x in h['K'][j]]
        b+=t(200,y+27,'['+', '.join(key)+']',28,'c-k')
        b+=g(t(520,y+27,f(h['scores'][0][j],0),28),1)
        b+=g(t(795,y+27,f(math.exp(h['scores'][0][j])),28),2)
        b+=g(t(1020,y+27,f(h['A'][0][j]),28,'c-a'),3)
    denom=sum(math.exp(x) for x in h['scores'][0])
    b+=g(t(25,392,'P1: (1×√2 + 1×0) / √2 = 1',27),1)
    b+=g(t(650,392,f'normalizer = 1 + 4e = {f(denom)}',26,'c-a'),3)
    frames.append(frame('s03-weights','One query; all five source weights',b,'Softmax normalizes across sources. The five weights sum to one.',
        'Which keys give the same score?\nRead the five rows, then reveal scores, exponentials, and normalized weights in that order.',
        'The denominator includes CLS and all four patches, including any patch used as receiver. There is no causal mask for this image-classification task. The displayed values are rounded; all later computations use unrounded values.',
        mobile_rows(['Source','Score','Weight'],[['CLS' if j==0 else f'P{j}',f(h['scores'][0][j],0),f(h['A'][0][j])] for j in range(5)])+f'<p>Denominator: 1 + 4e = {f(denom)}.</p>'))
    frames.append(mixing_frame(0,'s03-values','Keep each weight beside its own value'))
    h=R['heads'][1]
    b=t(25,35,'Source',24,'ink-2')+t(235,35,'key kⱼ²',25,'c-k')+g(t(575,35,'score',25),1)+g(t(825,35,'weight',25,'c-a'),2)+line(25,54,1130,54)
    for j in range(5):
        y=74+j*56;b+=source_icon(25,y,j)
        key=['√2' if abs(x-math.sqrt(2))<1e-8 else '0' for x in h['K'][j]]
        b+=t(235,y+27,'['+', '.join(key)+']',28,'c-k')
        b+=g(t(590,y+27,f(h['scores'][0][j],0),29),1)+g(t(835,y+27,f(h['A'][0][j]),29,'c-a'),2)
    denom=sum(math.exp(x) for x in h['scores'][0])
    b+=g(t(25,392,'q₀² = [1, 1] · keys use ink and column',26,'c-q'),1)
    b+=g(t(650,432,f'normalizer = 2 + 2e + e² = {f(denom)}',25,'c-a'),2)
    frames.append(frame('s03-second','A second head can choose a different mixture',b,'Part III: the same input rows, separate projections and a separate softmax.',
        'Which patch now receives the greatest weight?\nPoint to P2’s ink and column, then reveal its score 2 and the new weights.',
        'This deliberately chosen head reads ink and column instead of ink and row. A trained model is not assigned these jobs. The two patterns demonstrate independent mixtures; they do not prove that a one-head network cannot solve this task.',
        mobile_rows(['Source','Score','Weight'],[['CLS' if j==0 else f'P{j}',f(h['scores'][0][j],0),f(h['A'][0][j])] for j in range(5)])))
    frames.append(mixing_frame(1,'s03-second-values','The second head sends its own message'))
    section(3,'What should the summary row read?',frames,'q k a v')


def mixing_frame(head,key,title):
    h=R['heads'][head]
    b=t(25,35,'Source',24,'ink-2')+t(240,35,'weight',25,'c-a')+g(t(465,35,'value',25,'c-v'),1)+g(t(775,35,'weight × value',25,'c-v'),2)+line(25,54,1130,54)
    for j in range(5):
        y=74+j*55;b+=source_icon(25,y,j)+t(240,y+27,f(h['A'][0][j]),28,'c-a')
        b+=g(t(465,y+27,v(h['V'][j],0),29,'c-v'),1)
        b+=g(t(775,y+27,v(h['contributions'][j]),29,'c-v'),2)
    b+=g(line(750,356,1110,356,'c-v')+t(25,407,f'Head {head+1} message H₀{["¹","²"][head]}',29,'c-v')+t(775,407,v(h['H'][0]),32,'c-v'),3)
    return frame(key,title,b,'Multiply both value coordinates by the same source weight, then add down the columns.',
        'What exactly does a source weight multiply?\nReveal values, then products. Add each coordinate before revealing the message.',
        f'Head {head+1} uses values [ink, {"row" if head==0 else "column"}]. A large attention weight is only a multiplier: the value determines the information sent. An empty patch can still send location information.',
        mobile_rows(['Source','Weight','Value'],[['CLS' if j==0 else f'P{j}',f(h['A'][0][j]),v(h['V'][j],0)] for j in range(5)])+f'<p>Sum of weighted values: {v(h["H"][0])}.</p>')


def prediction():
    frames=[]
    a,b=R['heads'][0]['H'][0],R['heads'][1]['H'][0]
    body=t(25,45,'Head 1',26,'c-v')+t(200,45,v(a),31,'c-v')+t(660,45,'Head 2',26,'c-v')+t(825,45,v(b),31,'c-v')
    body+=g(arrow(320,70,460,135,'c-v')+arrow(930,70,755,135,'c-v')+t(250,185,v(R['joined'][0]),34,'c-v'),1)
    body+=g(t(25,280,'Output projection W_O',29)+t(570,280,'first update coordinate',27,'c-d')+t(570,335,f'{f(a[0])} − {f(b[0])} = {f(R["delta"][0][0])}',33,'c-d'),2)
    for row in range(4):
        for col in range(4):
            body+=g(t(105+col*49,318+35*row,f(DATA['W_O'][row][col],0),26,'ink-2','middle'),2)
    body+=g('<path d="M82 293 H72 V433 H82 M276 293 H286 V433 H276" fill="none" stroke="var(--ink-3)" stroke-width="2"/>',2)
    body+=g(t(570,410,'Δe₀ = '+v(R['delta'][0]),28,'c-d'),3)
    frames.append(frame('s04-join','Join the messages; then combine their coordinates',body,'Concatenation preserves both messages. The output projection mixes them.',
        'Which entries of W_O contribute to the first output coordinate?\nTrace the first coordinate from both head messages through +1 and −1.',
        'The concatenated row has width 4. Multiplying by the displayed 4×4 W_O produces an update of width 4. Although these widths happen to match, concatenation and projection are different operations. Both head messages contribute to the first update coordinate used by the classifier.',
        f'<p>Head 1: {v(a)}.</p><p>Head 2: {v(b)}.</p><p>Joined: {v(R["joined"][0])}.</p><p>First update: {f(a[0])} − {f(b[0])} = {f(R["delta"][0][0])}.</p><p>Full update: {v(R["delta"][0])}.</p>'))
    body=t(20,65,'original CLS',27,'c-e')+t(385,65,v(R['E'][0]),32,'c-e')
    body+=g(t(20,145,'+ contextual update',27,'c-d')+t(385,145,v(R['delta'][0]),32,'c-d')+line(380,172,1110,172,'ink-3'),1)
    body+=g(t(20,225,'updated CLS',27,'c-e')+t(385,225,v(R['updated'][0]),32,'c-e')+line(385,238,1100,238,'c-d',3),2)
    body+=g(t(20,325,'Across: −4 × '+f(R['updated'][0][0])+' = '+f(R['logits'][0]),30)+t(20,385,'Down:    4 × '+f(R['updated'][0][0])+' = '+f(R['logits'][1]),30),3)
    frames.append(frame('s04-residual','Keep the original row; add what it has read',body,'Part II’s residual update, followed by Part I’s class scores.',
        'Does attention replace the original representation?\nAdd the vectors coordinate by coordinate. Then use the first updated coordinate for both logits.',
        'The class matrix has first row [−4,4] and three zero rows. It acts on the updated CLS row. Both head messages affect that first coordinate through W_O. This linear classifier is chosen for the two named arrangements, not fitted as a general orientation recognizer.',
        f'<p>Original CLS: {v(R["E"][0])}.</p><p>Update: {v(R["delta"][0])}.</p><p>Updated CLS: {v(R["updated"][0])}.</p><p>Class logits: {v(R["logits"])}.</p>'))
    body=pixels(25,60,DATA['images']['horizontal'],65,False,True)+t(155,370,'Across the top',29,'ink','middle')
    for j,label in enumerate(DATA['classes']):
        y=90+140*j;body+=t(465,y,label,29)+g(t(1100,y,f(R['logits'][j]),30,'ink','end'),1)
        body+=g(rect(465,y+28,620*R['probability'][j],36,'c-e','t-e')+t(1110,y+56,f'{100*R["probability"][j]:.1f}%',29,'c-e','end'),2)
    loss=-math.log(R['probability'][0])
    body+=g(t(465,385,f'Loss = −log({f(R["probability"][0])}) = {f(loss)}',29,'c-a'),3)
    frames.append(frame('s04-probability','Turn the two scores into an answer',body,'This softmax runs over class labels. Attention’s softmax ran over source rows.',
        'What are the alternatives in this softmax?\nPoint to the two labels, then reveal the probabilities and the loss for the known answer.',
        f'The class probability for “Across the top” is exp({f(R["logits"][0])}…)/(exp({f(R["logits"][0])}…)+exp({f(R["logits"][1])}…)). The same unrounded logits generate the displayed bars and the negative-log-probability loss. A class probability is not an attention weight.',
        mobile_rows(['Class','Logit','Probability'],[[label,f(R['logits'][j]),f(R['probability'][j])] for j,label in enumerate(DATA['classes'])])+f'<p>Loss: {f(loss)}.</p>'))
    body=pixels(70,20,DATA['images']['horizontal'],43,False,True)+arrow(310,106,455,106)+pixels(510,20,DATA['images']['vertical'],43,False,True)
    body+=t(800,85,'Positions off?',31)+t(800,135,'What can change?',29,'ink-2')
    fr=frame('s04-experiment','Move the same patches, then remove position',body,'Change one thing at a time. Compare the two head messages and the class probabilities.',
        'Will the class probabilities change if we permute patches without position information?\nTake a prediction, choose Down the left, then turn positions off.',
        'Without position information, shared self-attention is permutation equivariant and the fixed CLS output is invariant to reordering patch rows. This control moves patch contents while location vectors stay fixed; it does not simply reorder already-positioned rows.',
        '<p>Predict the result, then use the controls below. Both arrangements have two filled and two empty patches.</p>')
    # Manual controls are deliberately available in both classroom and reading modes.
    controls='''<div class="vp-experiment" data-present="manual" data-keep-state>
<div class="vp-controls"><label>Arrangement <select id="vp-arrangement"><option value="horizontal">Across the top</option><option value="vertical">Down the left</option></select></label>
<label><input id="vp-positions" type="checkbox" checked> Include positions</label></div>
<output id="vp-result" aria-live="polite"></output></div>'''
    fr=fr.replace('<p class="vp-caption">',controls+'<p class="vp-caption">',1)
    # Make room for the controls: the experiment uses its own compact drawing.
    fr=fr.replace('class="frame vp-frame"','class="frame vp-frame vp-experiment-frame"',1)
    fr=fr.replace('viewBox="0 0 1160 440"','viewBox="0 0 1160 230"')
    frames.append(fr)
    section(4,'How do two messages become one answer?',frames,'d ep')


def full_model():
    frames=[]
    b=''
    for y,names in [(90,['E','LayerNorm','Multi-head attention','+ E']),
                    (265,['E₁','LayerNorm','Tokenwise MLP','+ E₁'])]:
        for j,name in enumerate(names):
            x=25+j*285;w=220
            b+=rect(x,y,w,75,'c-e' if j==0 else 'line','t-e' if j==0 else 'card')+t(x+w/2,y+45,name,24,'c-e' if j==0 else 'ink','middle')
            if j<3:b+=arrow(x+w+7,y+38,x+278,y+38)
        b+=f'<path d="M135 {y} V{y-42} H{25+3*285+110} V{y}" fill="none" stroke="var(--c-e)" stroke-width="2"/>'
    b+=g(arrow(1090,182,1090,218)+line(1090,218,135,218,'ink-3')+arrow(135,218,135,258),1)
    b+=g(t(25,420,'Repeat blocks → final normalization → CLS row → class scores',29),2)
    frames.append(frame('s05-block','Put the familiar attention inside a full block',b,'Attention mixes information across rows. The MLP transforms each row separately.',
        'Which operation communicates between patches, and which works on one row at a time?\nPoint to attention, then the MLP; trace both residual paths.',
        'The hand worksheet omitted normalization and the block MLP to isolate the arithmetic. A standard pre-LayerNorm ViT includes both. This diagram restores them explicitly; its full-block outputs are not the same numbers as the simplified worksheet. All patch rows update, even when the class head reads only CLS.',
        '<p>E₁ = E + attention(LayerNorm(E))</p><p>E′ = E₁ + MLP(LayerNorm(E₁))</p><p>Repeat blocks, normalize, read CLS, then classify.</p>'))
    b=image(25,45,270,180)+g(arrow(320,135,415,135)+rect(435,90,280,90,'c-e','t-e')+t(575,121,'196 patch rows + CLS',24,'c-e','middle')+t(575,164,'197 × 192',29,'c-e','middle'),1)
    b+=g(arrow(740,135,815,135)+rect(835,90,290,90,'line','card')+t(980,126,'12 transformer blocks',24,'ink','middle')+t(980,159,'3 heads per block',24,'ink-2','middle'),2)
    b+=g(t(35,315,'224 × 224 RGB pixels',29)+t(435,315,'16 × 16 × 3 → 192',30,'c-e')+t(835,315,'CLS → 1,000 scores',26),3)
    b+=g(t(35,405,'Same operations. More patches, coordinates, and layers.',31),4)
    frames.append(frame('s05-scale','Return to the photograph with a real ViT',b,'The pretrained model supplies learned representations and weights.',
        'Which numbers change when we scale up?\nTrace pixels, patch count, embedding width, heads, blocks, and the ImageNet label vocabulary.',
        'The measured endpoint uses timm’s vit_tiny_patch16_224.augreg_in21k_ft_in1k. The supplied evaluation transform resizes and center-crops the original photograph, normalizes RGB values, and produces a 224×224 tensor. It is an ImageNet classifier with 1,000 labels, including dog breeds and cat categories. It was not trained in this lecture.',
        mobile_rows(['Quantity','Real model'],[['Input','224×224 RGB'],['Patch pixels','16×16×3 = 768'],['Patch embeddings','196 rows × 192 coordinates'],['With CLS','197 rows'],['Blocks / heads','12 / 3'],['Class labels','1,000 ImageNet categories']])))
    section(5,'How does this become a Vision Transformer?',frames)


def real_result():
    dog=REAL['results'][0]
    b=image(25,45,535,357)+t(625,55,'Top 3 of 1,000 labels',23,'ink-2')
    for j,item in enumerate(dog['top3']):
        y=110+j*103
        label=item['label'].split(',')[0]
        b+=g(t(625,y,label,29)+t(1125,y,f'{100*item["probability"]:.2f}%',28,'c-e','end')+rect(625,y+22,500*item['probability'],24,'c-e','t-e'),1)
    fr=frame('s06-answer','The opening photograph goes through the actual model',b,'A measured prediction on this image. The worksheet explained the operations used inside.',
        'What answer should the model assign to our opening image?\nReveal the saved top-three predictions. Compare the model’s label with the students’ original answer.',
        'These are measured ImageNet probabilities for the exact Newfoundland photograph. The comparison Persian image is also included below. Two correct examples are an inference demonstration, not an accuracy estimate, a Pets benchmark, or evidence of robustness. No claim is made that these images were absent from all pretraining data.',
        f'<img src="{PHOTO}" alt="The same black Newfoundland dog from the opening photograph">'+mobile_rows(['ImageNet label','Probability'],[[x['label'].split(',')[0],f'{100*x["probability"]:.2f}%'] for x in dog['top3']]))
    cat=REAL['results'][1]
    companion=f'''<div class="companion"><h3>Try the same learned model on another photograph</h3>
<img class="vp-cat" src="{CAT}" alt="A white Persian cat resting against a knitted cushion">
<p>Persian_98: the top prediction is <strong>{escape(cat['top3'][0]['label'])}</strong>, with probability {cat['top3'][0]['probability']:.4f}.</p>
<p><a href="notebooks/vision/run_real_images.py">Reproduce real-image inference</a> · <a href="src/vision1_worksheet.py">Inspect every worksheet parameter and calculation</a> · <a href="notebooks/vision/03_vision_transformer_lab.ipynb">Open the worked notebook</a>.</p>
<h3>Keep the earlier lessons beside this one</h3><p><a href="part1.html">Part I</a>: representations, scores, probabilities, and loss. <a href="attention.html#s08">Part II</a>: queries/keys choose weights; values supply the messages. <a href="part3.html#s03">Part III</a>: separate head messages, concatenation, output projection, and residual.</p>
<h3>Images and teaching references</h3><p>Photographs: Oxford-IIIT Pet dataset, Parkhi, Vedaldi, Zisserman and Jawahar, via the <a href="https://huggingface.co/datasets/timm/oxford-iiit-pet">timm mirror</a>. Original files newfoundland_31 and Persian_98, test partition. <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/">Dataset and image license</a>: CC BY-SA 4.0. Original copyright remains with the image owners. The cropped views above retain this attribution.</p>
<p>Explanatory references: <a href="https://cs231n.stanford.edu/slides/2025/lecture_8.pdf">Stanford CS231n, Lecture 8</a>, <a href="https://jalammar.github.io/illustrated-transformer/">Jay Alammar</a>, <a href="https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html">D2L</a>, <a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html">UvA</a>, and the <a href="https://arxiv.org/abs/2010.11929">original ViT paper</a>. Diagrams and the four-patch calculation are original to this series.</p></div>'''
    section(6,'Can we answer the question we started with?',[fr+companion])


def notebook():
    code=(SRC/'vision1_worksheet.py').read_text().split("if __name__ == '__main__':")[0]
    cells=[{'cell_type':'markdown','metadata':{},'source':['# Four patches, two heads, one image answer\n','The same hand-chosen worksheet as the [Vision I lecture](../../vision1.html). The two labels name the two supplied arrangements; this is not a trained general-purpose recognizer.\n']}]
    snippets=[code,'''p = parameters()
r = forward(p['images']['horizontal'])
for name in ['patches', 'content', 'E']:
    print(name, '\\n', r[name])
''', '''for i, head in enumerate(r['heads'], 1):
    print('HEAD', i)
    for name in ['Q', 'K', 'V', 'scores', 'A', 'H']:
        print(name, '\\n', np.round(head[name], 6))
    print('CLS contributions', np.round(head['contributions'], 6))
''', '''for name in ['joined', 'delta', 'updated', 'logits', 'probability']:
    print(name, '\\n', np.round(r[name], 6))
print('Loss', -np.log(r['probability'][0]))
''', '''for name, image in p['images'].items():
    for position in [True, False]:
        print(name, 'positions:', position, forward(image, position)['probability'])
''']
    import contextlib,io
    scope={'__file__':str(SRC/'vision1_worksheet.py')}
    for i,snippet in enumerate(snippets,1):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):exec(snippet,scope)
        cells.append({'cell_type':'code','metadata':{},'source':snippet.splitlines(keepends=True),'execution_count':i,
                      'outputs':[{'output_type':'stream','name':'stdout','text':out.getvalue().splitlines(keepends=True)}] if out.getvalue() else []})
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':cells}
    for i,cell in enumerate(cells):cell['id']=f'four-patch-{i}'
    (ROOT/'notebooks/vision/02_four_patch_walkthrough.ipynb').write_text(json.dumps(book,indent=1)+'\n')


def main():
    from vision1_lesson_content import build_full
    count=build_full(globals())
    notebook()
    from build_vision1_lab import build_lab
    build_lab()
    subprocess.run([sys.executable,str(SRC/'assemble.py'),'--part','5','--out',str(ROOT/'vision1.html')],check=True)
    print(f'{count} teaching frames plus cover; complete lecture and executed lab')

if __name__ == '__main__': main()
