"""Smaller teaching steps and concrete examples for the Vision I lecture.

The existing figures stay in place across related slides. Each new slide has
its own question, caption and presenter note; it is not a screenshot of a build.
"""
import copy
import json
import math
import re
import xml.etree.ElementTree as ET
from html import escape
import numpy as np
from vision1_intuition import calculations

ET.register_namespace('', 'http://www.w3.org/2000/svg')
SVG='{http://www.w3.org/2000/svg}'


def expand(b, sections):
    gbl=['ROOT','SRC','ASSETS','DATA','R','REAL','PHOTO','CAT','FRAMES','f','v','t','g','line','arrow','rect','image','crop','pixels','frame','mobile_rows']
    globals().update({k:b[k] for k in gbl})
    calc=calculations()
    (ASSETS/'intuition-calculations.json').write_text(json.dumps(calc,indent=2)+'\n')
    old={re.search(r'class="frame[^\"]*" id="([^\"]+)"',html).group(1):html for _,frames in sections for html in frames}
    new={}
    metadata={r['id']:r for r in FRAMES}

    def add(key,title,body,caption,question,point,prose='',mobile=''):
        html=frame(key,title,body,caption,question+'\n'+point,prose,mobile)
        new[key]=html
        return html

    def row(y,label,values,color='c-e'):
        return t(35,y,label,29,'ink-2')+t(440,y,values,29,color)

    def math_steps(key,title,lines,caption,question,point,prose=''):
        body=''
        for i,(label,value,color) in enumerate(lines):
            mark=row(95+105*i,label,value,color)
            body+=mark if i==0 else g(mark,i)
        return add(key,title,body,caption,question,point,prose)

    def matrix(x,y,rows,color='c-e',dx=62,dy=49,size=28):
        out=''
        for r,values in enumerate(rows):
            for c,value in enumerate(values):out+=t(x+c*dx,y+r*dy,value,size,color,'middle')
        left=x-dx*.5;right=x+(len(rows[0])-.5)*dx;top=y-31;bottom=y+(len(rows)-1)*dy+14
        return out+f'<path d="M{left+9} {top} H{left} V{bottom} H{left+9} M{right-9} {top} H{right} V{bottom} H{right-9}" stroke="var(--{color})" fill="none" stroke-width="2"/>'

    def split(key,specs):
        """Retain one drawing, with a separately authored question for each step."""
        original=old[key];root=ET.fromstring((ASSETS/(key+'.svg')).read_text())
        out=[]
        for i,(limit,title,caption,question,point) in enumerate(specs):
            drawing=copy.deepcopy(root)
            def prune(parent):
                for child in list(parent):
                    build=int(child.get('data-build','0'))
                    if build>limit:parent.remove(child);continue
                    if build:
                        if build==limit:child.set('data-build','1')
                        else:child.attrib.pop('data-build',None)
                    prune(child)
            prune(drawing)
            drawing.set('aria-label',title)
            drawing.find(SVG+'title').text=title
            new_svg=ET.tostring(drawing,encoding='unicode')
            ident=key if i==len(specs)-1 else key+'-step-'+str(i+1)
            html=original.replace('id="'+key+'"','id="'+ident+'"',1)
            html=re.sub(r'data-title="[^"]*"','data-title="'+escape(title)+'"',html,count=1)
            # The root SVG is inside this figure div; nested photo crops stay intact.
            html=re.sub(r'(<div class="vp-figure[^\"]*">).*?(</div>)',lambda m:m[1]+new_svg+m[2],html,count=1,flags=re.S)
            notes=question+'\n'+point
            html=re.sub(r'<script type="text/x-notes">.*?</script>',lambda _: '<script type="text/x-notes">'+escape(notes)+'</script>',html,count=1,flags=re.S)
            html=re.sub(r'<p class="vp-caption">.*?</p>',lambda _: '<p class="vp-caption">'+escape(caption)+'</p>',html,count=1,flags=re.S)
            if i<len(specs)-1:
                # Early slides show only the current drawing. The final slide retains
                # the complete accessible mobile table and the detailed reading notes.
                html=re.sub(r'<div class="vp-mobile">.*?</div>','',html,count=1,flags=re.S)
                html=html.replace('vp-figure vp-desktop','vp-figure',1)
                html=html.split('<div class="companion">')[0]
            (ASSETS/(ident+'.svg')).write_text(new_svg)
            entry={'id':ident,'title':title,'caption':caption,'notes':notes}
            if ident==key:metadata[key].update(entry)
            else:FRAMES.append(entry)
            out.append(html)
        return out

    def edit(key,title=None,caption=None,notes=None):
        html=old[key];meta=metadata[key]
        if title:
            html=re.sub(r'data-title="[^"]*"','data-title="'+escape(title)+'"',html,count=1)
            meta['title']=title
        if caption:
            html=re.sub(r'<p class="vp-caption">.*?</p>',lambda _: '<p class="vp-caption">'+escape(caption)+'</p>',html,count=1,flags=re.S)
            meta['caption']=caption
        if notes:
            html=re.sub(r'<script type="text/x-notes">.*?</script>',lambda _: '<script type="text/x-notes">'+escape(notes)+'</script>',html,count=1,flags=re.S)
            meta['notes']=notes
        old[key]=html
        return html

    # Wording follows the earlier lessons: a direct question, a concrete action,
    # and one sentence that says what the calculation just showed.
    revisions={
      's01-photo':('What animal do you see?','Which parts of the photograph helped you decide?'),
      's01-context':('Would you recognize this crop on its own?','A dark crop may be fur, shadow or background. The full photo gives us clues.'),
      'bridge-text':('What can we carry over from our text models?','We still turn inputs into rows, read useful information, and predict an answer.'),
      'patch-context':('Is this dark region fur or background?','Face clues could make a dark crop easier to interpret. The final prediction is one label for the whole image.'),
      's01-patches':('Where do the patch boundaries go?','The grid cuts through the photograph before the model knows where the dog is.'),
      's02-small':('Same pieces, different picture?','Count the filled patches in each image. Then look at where they are.'),
      'position-question':('Could you put the picture back together?','Knowing which patches we have does not tell us where each one belongs.'),
      'patch-matrix':('Where does each number in the patch row come from?','Multiply the pixel row by each column, then add the bias.'),
      'why-cls':('We have several patch rows. Where does the answer go?','The classification token (CLS) is an extra learned row used to collect information for the image label.'),
      'image-mask':('Can the top-left patch read the bottom-right?', 'We already have the whole image when we predict its label, so all patches can read one another.'),
      'heads-question':('Would a second way of reading the image help?','As with colour and material in the coat example, we can keep two different messages.'),
      'all-receivers':('Do the patch rows get updated too?','Yes. Here every query is [1,1], so the messages match. Each row still has its own starting vector.'),
      'weight-message':('Do equal weights send equal information?','Compare the values beside the weights. They tell us what each patch sends.'),
      's04-experiment':('Move the patches. Does the answer change?','Make a prediction, change the arrangement, then turn positions off.'),
      'depth':('What does the next block get to see?','The next block starts from rows that have already read the image.'),
      'code-attention':('Can you match each line to our calculation?','Each row of A sums to one across its sources. Multiplying by V adds the weighted messages.'),
      'training-data':('Will the model recognize a new noisy stripe?','Move the stripe and vary the pixel values. Keep separate images for training, validation and testing.'),
      'trained-position-control':('Can training make up for missing positions?','Train on the same images with the same recipe, then compare the two test results.'),
      'real-input':('Which pixels are we giving the real model?','The model receives the square crop on the right, after RGB normalization.'),
      's06-answer':('What did the model call our dog?','These probabilities come from running this photograph through the pretrained model.'),
      'real-cat':('What happens when we give it the cat?','Keep the same model and preprocessing. Change only the photograph.'),
      'real-heads':('Do the heads read the same places?','Block 12, CLS query, heads 1–3. The colour scale is the same in all three images.'),
      'real-depth':('Does CLS read differently in a later block?','Compare the first and last blocks on the same image and colour scale.'),
      'real-patch-query':('What does this particular patch read?','The purple square is the receiver. The shading shows its source weights in block 12, head 1.'),
      'occlusion':('Which covered region changed the answer most?','Compare the dog probability after each of the four covers.'),
      'cost-control':('What happens if we use a larger image?','Work out the token count before you change the controls.'),
      'exercise-message':('Your turn: work out the message','Find the weights first. Then multiply each value by its own weight.'),
      'exercise-position':('Did we move the image, or just reorder its rows?','Follow one patch and its position vector in each experiment.'),
      'next-vision':('What else could we ask the image model to do?','The output we want determines which rows we read and how we train them.'),
      'closing':('Can you talk us through the whole model?','Start with the pixels. Explain what each step adds before moving to the next one.')}
    for key,(title,caption) in revisions.items():edit(key,title,caption)

    # Bridge the visual motivation to the representation we will construct next.
    body=image(25,75,310,207)+t(180,335,'whole photograph',27,'c-e','middle')
    grid=''
    for j in range(1,4):
        grid+=line(25+77.5*j,75,25+77.5*j,282,'card',2)
        grid+=line(25,75+51.75*j,335,75+51.75*j,'card',2)
    body+=g(grid,1)
    rows=arrow(365,180,455,180,'c-e')
    for j in range(3):
        rows+=rect(485,105+65*j,245,43,'c-e','t-e')
        rows+=t(607,134+65*j,'numbers for a patch',23,'c-e','middle')
    rows+=t(607,305,'⋮',26,'c-e','middle')+t(607,350,'one row per patch',27,'c-e','middle')
    body+=g(rows,2)
    body+=g(arrow(760,180,840,180,'c-q')+rect(875,138,240,83,'c-q','transparent')+t(995,189,'attention',34,'c-q','middle')+t(995,335,'share information',26,'c-v','middle'),3)
    add('image-to-rows','How can we give this photograph to attention?',body,'Attention works on rows of numbers. Next, we choose image patches and turn their pixels into those rows.',
        'In the text lessons, what did attention receive as its input?','Keep the photograph visible. Reveal its pieces, one row per piece, then the familiar attention operation.',
        'The previous slide motivated using clues from other image regions. Now we need a numerical representation so attention can combine those clues. Text supplied one row per token; our image will supply one row per fixed-size patch. This drawing previews the construction rather than calculating it. The next section first chooses the patch grid, then reads the pixels and applies a shared learned projection. Position information comes after that.')

    # More examples before architecture, as in the opening of text Part I.
    body=image(35,50,400,267)+image(600,40,225,300,CAT)
    body+=g(t(235,390,'dog',36,'c-e','middle')+t(712,390,'cat',36,'c-e','middle'),1)
    add('photo-folder','Find the dog photos in a folder',body,'Each photograph gets one label. Decide what the labels mean before choosing the model.',
        'What would you want the program to return for each photo?','Reveal dog and cat. Connect one image label to the spam/not-spam example in Part I.',
        'This is a proposed two-class task, like the email classification task in Part I. The pretrained model later in this lesson has a different label set: 1,000 ImageNet categories. Dog and cat here are human example labels, not outputs from a fitted two-class classifier.')
    body=image(25,40,535,357)+g(rect(145,81,311,300,'c-e','transparent',0),1)+g(t(655,130,'What animal?',31)+t(655,210,'Where is the animal?',31),2)
    add('find-animal','Suppose we also want to crop out the animal',body,'A class label tells us what is present. A box also tells us where to look.',
        'Is the word dog enough for an automatic crop?','Reveal the illustrative box and ask which extra numbers describe it.',
        'The box is drawn by hand to illustrate a localization task. It is not a prediction from our classification ViT. This parallels Part I: a sentence label and labels on individual words require different outputs.')
    body=image(35,45,350,234)+g(t(540,140,'“a black dog outdoors”',34,'c-e'),1)+g(t(540,270,'“a white cat on a cushion”',32,'ink-2'),2)
    add('photo-search','Find the photo that matches this description',body,'Here we need to compare an image with words.',
        'Which of our two photographs fits the first description?','Match the words to visible evidence, then connect this task to the later CLIP lesson.',
        'These are human-written descriptions for motivation. Vision III covers learning an image–text similarity model. We first need an image representation that such a model can use.')

    # Pixel, projection and position examples at a size students can calculate.
    math_steps('one-rgb','How can a red pixel be three numbers?',[
      ('red','[255, 0, 0]','c-e'),('divide by 255','[1, 0, 0]','c-e'),('white','[255, 255, 255] → [1,1,1]','c-e')],
      'Each pixel records a red, green and blue channel.',
      'What would [0,0,0] look like?','Name the channels, then ask for black, red and white before revealing the rows.',
      'This example uses 8-bit channel values and simple division by 255. The pretrained model also centers and scales each channel using its supplied preprocessing settings.')
    body=pixels(45,70,[[1,0],[0,1]],77,True)+t(120,310,'one patch',29,'ink','middle')
    body+=g(arrow(250,145,360,145)+t(425,150,'[1, 0, 0, 1]',39,'c-e'),1)
    body+=g(t(425,290,'top-left, top-right,',30)+t(425,345,'bottom-left, bottom-right',30),2)
    add('flatten-order','Which pixel goes first in the row?',body,'Choose an order and use it for every patch.',
        'Where does the last 1 in this row come from?','Trace the four pixel positions in raster order.',
        'Flattening preserves these four numbers and changes their arrangement in memory. It does not average them. The projection is the later operation that combines coordinates.')
    body=pixels(25,65,[[1,1],[0,0]],80,True)+pixels(425,65,[[1,0],[1,0]],80,True)
    body+=g(t(105,315,'mean = 0.5',30,'c-e','middle')+t(505,315,'mean = 0.5',30,'c-e','middle'),1)
    body+=g(t(775,120,'Same mean.',33)+t(775,190,'Different edge.',33)+t(775,290,'What did we lose?',28,'c-a'),2)
    add('mean-loses-edge','Would the mean pixel value tell these patches apart?',body,'A patch can have the same average and a different pattern.',
        'Which patch has a horizontal edge? Which has a vertical edge?','Count two ones in each, then compare their positions.',
        'Our four-patch worksheet uses mean ink because its patches are all filled or all empty. That chosen projection would lose the difference between these two mixed patches. A learned projection can use several filters to retain more useful structure.')
    body=t(35,55,'pixels',26,'ink-2')+t(465,55,'two columns of weights',28,'c-e')
    body+=t(35,170,'[1, 1, 0, 0]',33,'c-e')+t(35,285,'[1, 0, 1, 0]',33,'c-e')
    body+=matrix(530,120,[[1,1],[1,-1],[-1,1],[-1,-1]])
    body+=g(t(830,170,'→ [2, 0]',38,'c-v')+t(830,285,'→ [0, 2]',38,'c-v'),1)
    body+=g(t(35,420,'top − bottom                 left − right',28,'ink-2'),2)
    add('edge-filters','Could two projection columns keep that difference?',body,'Each column combines the same pixels in a different way.',
        'What does the first column subtract from what?','Calculate the top-edge patch against each column, then try the left-edge patch.',
        'The columns [1,1,−1,−1] and [1,−1,1,−1] compute top-minus-bottom and left-minus-right for this flattening order. The weights are chosen examples, not filters inspected in the pretrained ViT. Learning adjusts many such weighted combinations.')
    body=pixels(40,95,[[0,0],[0,0]],73,True)+g(t(370,135,'[0,0,0,0] W_patch = [0,0,0,0]',28,'c-e'),1)
    body+=g(t(370,240,'+ bias [0,0,0,1]',34,'c-e'),2)+g(t(370,350,'= [0,0,0,1]',38,'c-e'),3)
    add('empty-patch','What row does an empty patch get?',body,'The bias adds a learned offset even when every input pixel is zero.',
        'Will multiplying zeros by W_patch produce the final 1?','Calculate the product first, then add the bias as a separate step.',
        'In this worksheet the bias is fixed at [0,0,0,1]. That last coordinate makes the chosen queries easy to calculate. A real model learns its bias; it need not produce a constant coordinate with this interpretation.')
    body=rect(50,75,280,150,'c-e','t-e')+t(190,125,'P1',34,'c-e','middle')+t(190,185,'[1,0,0,1]',32,'c-e','middle')
    body+=rect(765,75,280,150,'c-e','t-e')+t(905,125,'P2',34,'c-e','middle')+t(905,185,'[1,0,0,1]',32,'c-e','middle')
    body+=g(t(50,310,'+ [0,0,0,0]',31,'ink-2')+t(765,310,'+ [0,0,1,0]',31,'ink-2'),1)
    body+=g(t(50,410,'= [1,0,0,1]',32,'c-e')+t(765,410,'= [1,0,1,1]',32,'c-e'),2)
    add('two-identical-patches','These patches look the same. How do we tell them apart?',body,'Their content rows match. Their positions do not.',
        'Which coordinate changes for the top-right patch?','Keep the filled patch content fixed and add its column location.',
        'The worksheet gives row and column their own named coordinates. A learned ViT position vector is generally a full D-dimensional row without these fixed coordinate meanings.')

    # Q/K/V and softmax, with short examples before the full table.
    body=crop(30,85,270,180,9,'request')+t(165,330,'receiver patch',28,'c-q','middle')
    body+=g(t(430,100,'Q: what would help here?',31,'c-q'),1)
    body+=g(t(430,220,'K: which sources match?',31,'c-k'),2)
    body+=g(t(430,340,'V: what do those sources send?',31,'c-v'),3)
    add('qkv-roles','Why do we make three versions of each row?',body,'Matching a source and reading its information are two different jobs.',
        'Which two vectors decide the weight, and which vector gets multiplied by it?','Recall bank’s request and river’s information from Part II. Keep the photo crop visible.',
        'The questions are an analogy for numeric projections. Q and K determine a compatibility score. V supplies the numbers to mix. They are not strings stored inside the model. Each image row produces all three vectors, even though we follow one receiver at a time.')
    math_steps('q-dot','Where does the CLS query [1,1] come from?',[
      ('CLS input','[0, 0, 0, 1]','c-e'),('first query coordinate','0·0 + 0·0 + 0·0 + 1·1 = 1','c-q'),('second query coordinate','0·0 + 0·0 + 0·0 + 1·1 = 1','c-q')],
      'Each query coordinate is a dot product with one column of W_Q.',
      'Which input coordinate supplies both ones?','Use the W_Q columns shown on the previous slide and multiply all four terms.',
      'This is the chosen CLS vector and W_Q from the worksheet. It follows exactly the row-times-matrix convention used in Parts II and III.')
    math_steps('one-key-dot','How does P1 get the key [√2,0]?',[
      ('P1 input','[1, 0, 0, 1]','c-e'),('first key coordinate','1·√2 + 0·0 + 0·0 + 1·0 = √2','c-k'),('second key coordinate','1·0 + 0·√2 + 0·0 + 1·0 = 0','c-k')],
      'Head 1 keeps ink and row in its keys.',
      'Would the constant 1 in the input affect this key?','Point to the zero in the last row of each key column.',
      'The √2 entries are chosen to cancel the attention scaling in our two-coordinate head. A trained model does not generally have this special pattern.')
    math_steps('one-score','Can we work out one score before filling the table?',[
      ('query and P1 key','[1,1] · [√2,0]','c-q'),('dot product','1·√2 + 1·0 = √2','c-k'),('divide by √d_k','√2 / √2 = 1','c-a')],
      'The head has two query/key coordinates, so d_k is 2.',
      'Should the denominator use 2, 4 or 5 here?','Distinguish head width, embedding width and number of source rows.',
      'The usual scaling divides by √d_k. Under a simple model of independent unit-variance coordinates, dot-product variance grows with d_k; this scaling keeps its typical size steadier. The exact arithmetic here uses d_k=2.')
    examples=calc['softmax_examples']
    body=''
    for i,label in enumerate(['[0, 0]','[0, ln 3]','[−100, −100]']):
        y=80+145*i;weights=examples[i]['weights']
        body+=t(30,y,label,34,'ink')
        for j,w in enumerate(weights):
            body+=g(rect(435+j*325,y-35,280*w,39,'c-a','t-e')+t(435+j*325,y+43,f'{100*w:.0f}%',28,'c-a'),1)
    add('softmax-relative','What does softmax do when the scores tie?',body,'Equal scores get equal shares. Only score differences affect the shares.',
        'Are two very negative scores less than 100% in total?','Compare the first and third rows, then explain the middle row using exponentials 1 and 3.',
        'The three rows are independent examples. [0,0] and [−100,−100] both give [0.5,0.5]. [0,ln 3] gives [0.25,0.75]. Subtracting the largest score before exponentiating preserves these probabilities and improves numerical stability.')
    math_steps('weight-denominator','Where does the 0.229 beside P1 come from?',[
      ('all exponentials','[1, e, e, e, e]','c-a'),('their sum','1 + 4e = 11.873','c-a'),('P1’s share','e / (1 + 4e) = 0.229','c-a')],
      'Every source appears in the same denominator, including CLS.',
      'What would go wrong if each patch used a different denominator?','Sum the five source contributions first, then divide one exponential by that sum.',
      'The displayed values are rounded to three decimals. Every later calculation uses full precision. CLS gets 1/(1+4e)=0.084, and each image patch gets approximately 0.229.')
    h=R['heads'][0]
    body=t(35,75,'P3 weight = '+f(h['A'][0][3]),34,'c-a')+t(35,165,'P3 value = [0,1]',34,'c-v')
    body+=g(t(35,285,'0.229 × [0,1] = [0,0.229]',38,'c-v'),1)
    body+=g(t(35,405,'no ink contribution; a row contribution',31,'ink-2'),2)
    add('one-value-product','Can an empty patch still send something?',body,'An empty patch still has a position, and its value can carry that information.',
        'Which coordinate receives the contribution from P3?','Multiply both coordinates by the same weight; keep the zero visible.',
        'Head 1 uses [ink,row] as its value. P3 is empty, but it is in row 1. A zero ink coordinate does not make its entire value row zero.')
    changed=calc['changed_query']
    body=t(35,70,'same keys and values',30,'ink-2')+t(35,150,'query: [1,1] → [1,0]',35,'c-q')
    body+=g(t(35,245,'scores: [0,1,1,0,0]',35,'c-a'),1)
    body+=g(t(35,330,'weights: '+v(changed['weights']),30,'c-a'),2)
    body+=g(t(35,420,'message: '+v(changed['message']),32,'c-v'),3)
    add('change-query','What if the query cared only about ink?',body,'Change the query and keep the sources fixed. The mixture changes.',
        'Which filled patches should get more weight now?','Set the second query coordinate to zero, then recompute all five scores.',
        'This is a separate what-if calculation using Head 1’s existing K and V. It is not a new trained checkpoint or a change to the main worksheet. It shows why real receivers with different queries can read the same sources differently.')

    # Combining heads: every column of W_O can be checked.
    proj=calc['output_projection']
    body=t(35,70,'joined row: '+v(proj['joined']),31,'c-v')
    for i,(label,expression) in enumerate([
      ('coordinate 2',f"{f(proj['joined'][1])} = {f(proj['delta'][1])}"),
      ('coordinate 3',f"{f(proj['joined'][0])} + {f(proj['joined'][2])} = {f(proj['delta'][2])}"),
      ('coordinate 4',f"{f(proj['joined'][3])} = {f(proj['delta'][3])}")]):
        body+=g(t(35,175+i*107,label,28,'ink-2')+t(390,175+i*107,expression,34,'c-d'),i+1)
    add('other-output-coordinates','What do the other columns of W_O produce?',body,'Use the same joined row for every output coordinate.',
        'Which column adds information from both heads?','Read W_O column by column, matching these results to the full update vector.',
        'The first column subtracts the heads’ ink coordinates. The second copies Head 1’s row coordinate; the third adds both ink coordinates; the fourth copies Head 2’s column coordinate. The notebook checks every product using the displayed matrix.')
    math_steps('residual-zero','What if attention sent a zero message?',[
      ('starting row','[1, 0, 1, 1]','c-e'),('attention update','[0, 0, 0, 0]','c-d'),('after addition','[1, 0, 1, 1]','c-e')],
      'The residual gives the next block a direct path from the starting row.',
      'Would the original row disappear if the attention output were zero?','Add coordinate by coordinate, then connect this to the nonzero update in our main example.',
      'This example explains the forward computation. Residual paths also help gradients flow through deep models, but adding a residual does not guarantee that every learned update improves the prediction.')
    body=t(35,70,'attention softmax',30,'c-a')+t(620,70,'class softmax',30,'c-a')
    for i,label in enumerate(['CLS','P1','P2','P3','P4']):body+=t(35,145+i*52,label,28,'c-e')
    body+=g(t(620,165,'Across the top',31,'c-e')+t(620,265,'Down the left',31,'c-e'),1)
    body+=g(t(35,427,'Which sources should I read?',27)+t(620,427,'Which label should I predict?',27),2)
    add('two-softmaxes','We used softmax twice. What changed?',body,'The formula is the same. The alternatives being compared are different.',
        'Should the five attention weights sum with the two class probabilities?','Circle the alternatives separately. Each softmax has its own denominator.',
        'Attention softmax normalizes source scores for one receiver and head. Class softmax normalizes output logits over labels. A large attention weight is not the probability that a patch belongs to a class.')
    math_steps('loss-comparison','How much does a confident wrong answer cost?',[
      ('P(correct) = 0.9','−log(0.9) = 0.105','c-a'),('P(correct) = 0.5','−log(0.5) = 0.693','c-a'),('P(correct) = 0.1','−log(0.1) = 2.303','c-a')],
      'Training lowers the loss by increasing the probability of the known label.',
        'Which prediction should receive the largest penalty?','Compare all three cases using the same correct label.',
        'These are separate cross-entropy examples with natural logarithms. A low loss on one example does not establish good performance on other images.')

    # Make normalization and the row MLP calculable, not just named boxes.
    ln=calc['layernorm']
    math_steps('ln-mean','What average are we subtracting?',[
      ('one token row','[1, 0, 0, 1]','c-e'),('mean','(1 + 0 + 0 + 1) / 4 = 0.5','ink'),('centered row','[0.5, −0.5, −0.5, 0.5]','c-e')],
      'LayerNorm works across the coordinates of this one row.',
        'Are the other patches included in this mean?','Keep the four coordinates together and calculate their mean.',
        'The batch and other token rows are not included in this LayerNorm mean. The normalization dimension is the row width D.')
    math_steps('ln-variance','How spread out is the centered row?',[
      ('squared differences','[0.25, 0.25, 0.25, 0.25]','c-e'),('variance','(0.25 + 0.25 + 0.25 + 0.25)/4','ink'),('standard deviation','√0.25 = 0.5','ink')],
      'Square the deviations, take their mean, then take the square root.',
        'Why do positive and negative deviations not cancel here?','Square each deviation before averaging. Mention epsilon on the next slide.',
        'This is the population variance used by LayerNorm, dividing by D. The actual denominator uses √(variance+ε), with ε=10⁻⁵ in this example, giving about 0.50001.')
    body=t(35,65,'input: [1, −1]',35,'c-e')+matrix(520,130,[[1,0,1],[0,1,1]],dx=65)
    body+=g(t(35,275,'[1, −1] W₁ = [1, −1, 0]',36,'c-v'),1)
    body+=g(t(35,405,'2 coordinates → 3 hidden coordinates',31),2)
    add('mlp-first-linear','How does the MLP make a wider row?',body,'The first linear layer mixes coordinates within the same token.',
        'Where does the zero in the hidden row come from?','Compute 1×1 + (−1)×1 using the third column.',
        'This small MLP is a separate arithmetic illustration with zero biases. The full small ViT in the training lab expands D=16 to 32 hidden coordinates.')
    body=t(35,65,'after GELU: [0.841, −0.159, 0]',32,'c-v')+matrix(570,150,[[1,0],[0,1],[0,0]],dx=65)
    body+=g(t(35,370,'message: [0.841, −0.159]',37,'c-d'),1)
    add('mlp-second-linear','How does the MLP return to the original width?',body,'The second linear layer produces an update that can be added to the starting row.',
        'Why must the output width be two in this example?','Match its width to the residual row [1,−1].',
        'Here W₂ copies the first two hidden coordinates. In a trained block it can mix all hidden coordinates. GELU supplies a nonlinearity between the two learned linear maps.')
    body=t(35,70,'2 × 2 × 1 pixels per patch',34,'c-e')+t(35,155,'16 patches per image',34,'c-e')
    body+=g(t(35,270,'D = 16 numbers per patch row',34,'c-e'),1)
    body+=g(t(35,395,'[batch, 16 rows, 16 coordinates]',37),2)
    add('two-sixteens','These two 16s mean different things',body,'Patch count comes from the image grid. Row width is a model choice.',
        'Which 16 changes if we use a larger image with the same patch size?','Point to the row axis, then to the coordinate axis.',
        'In the small model, 8×8 images split into 2×2 patches give 16 patch rows. D=16 is an independent design choice. After adding CLS there are 17 rows, still 16 coordinates wide.')

    def code(key,title,source,labels,caption,question,point,prose=''):
        body=''
        for i,(label,col) in enumerate(labels):
            body+=t(35,85+110*i,label,27,col)
            if i<len(labels)-1:body+=arrow(50,110+110*i,50,150+110*i)
        html=add(key,title,body,caption,question,point,prose)
        html=html.replace('<div class="vp-figure">','<div class="vl-code"><pre><code>'+escape(source)+'</code></pre><div class="vp-figure">',1)
        html=html.replace('</svg></div>','</svg></div></div>',1).replace('viewBox="0 0 1160 440"','viewBox="0 0 400 440"')
        new[key]=html
        return html
    code('code-add-cls','How do we add one CLS row per image?',
      'cls = self.cls.expand(\n    x.size(0), -1, -1)\nx = torch.cat([cls, x], dim=1)',
      [('B × 16 × 16','c-e'),('+ one CLS row','c-e'),('B × 17 × 16','c-e')],
      'Every image starts with the same learned CLS vector.',
      'Do we need a separate learned CLS parameter for each training image?','Expand the same parameter across the batch, then concatenate along the row axis.',
      'The parameter has shape [1,1,D]. expand supplies one view per image. Attention later makes its updated value depend on the image.')
    code('code-add-pos','Where does location enter the code?',
      '# pos has shape [1,17,16]\nx = x + self.pos\nx = self.blocks(x)',
      [('content + position','c-e'),('all 17 rows','c-e'),('two blocks','ink')],
      'Broadcast the same position table across the batch.',
      'Which position vector goes with the third image patch?','Point to CLS at index 0 and patch P3 at index 3.',
      'The position vectors are trainable parameters in the small ViT. Turning off the addition creates the later control experiment. Blocks preserve the number and width of rows.')
    code('code-cls-readout','Which row reaches the classifier?',
      'x = self.norm(x)\nsummary = x[:, 0]\nlogits = self.head(summary)',
      [('B × 17 × 16','c-e'),('CLS: B × 16','c-e'),('B × 2 logits','c-a')],
      'We keep all rows through the blocks, then select CLS for this image-label task.',
      'Why is the second index zero here?','Point to the row dimension; the batch dimension stays in place.',
      'The classifier maps the final normalized CLS row from D coordinates to the number of classes. Other tasks may read patch rows or use pooling instead.')
    code('code-backward','When are gradients computed?',
      'optimizer.zero_grad()\nloss.backward()',
      [('one scalar loss','c-a'),('one gradient per','c-d'),('trainable parameter','c-d')],
      'backward computes gradients. The parameter values stay where they are until the optimizer step.',
      'Would another forward pass already use new weights?','Separate computing a derivative from using it to change a parameter.',
      'PyTorch accumulates gradients unless they are cleared. zero_grad clears the previous batch’s gradients; backward differentiates the current computation graph.')
    code('code-step','Which line changes the weights?',
      'optimizer.step()\n\n# Next batch uses updated weights.\nlogits = model(next_images)',
      [('gradients','c-d'),('updated weights','c-e'),('next predictions','c-a')],
      'The optimizer uses the gradients to update the parameters.',
      'Are AdamW updates exactly minus learning-rate times gradient?','Recall the simple SGD bias example, then distinguish the optimizer used in this lab.',
      'AdamW also uses moving averages of gradients and decoupled weight decay. The earlier single-bias example used plain SGD so we could calculate one update by hand.')

    # Training and real-model interpretation need the same slow setup as the math.
    body=t(35,70,'train: 512 images',35,'c-e')+g(t(35,170,'validation: 128 different images',35,'c-v'),1)+g(t(35,270,'test: 256 different images',35,'c-a'),2)
    body+=g(t(35,405,'learn weights → choose checkpoint → evaluate once',29),3)
    add('three-splits','Which images are allowed to influence the weights?',body,'Give each split one job before starting the experiment.',
      'Why should we not keep trying checkpoints on the test set?','Follow the three arrows in order. Validation chooses the checkpoint; test reports its result.',
      'These splits use different generation seeds, fixed before training. Examples share the same synthetic distribution. The test labels never enter optimizer updates or checkpoint selection.')
    body=t(35,80,'512 training images',38,'c-e')+g(t(35,195,'64 images per batch',36),1)+g(t(35,310,'512 / 64 = 8 updates per epoch',35,'c-d'),2)
    body+=g(t(35,420,'80 epochs → 640 optimizer steps',31),3)
    add('one-epoch','What does one epoch mean in this experiment?',body,'An epoch visits every training example once.',
      'How many weight updates happen during one pass through the training set?','Divide 512 by 64; then distinguish an epoch from an optimizer step.',
      'The script shuffles the training examples each epoch, processes eight batches and evaluates validation loss after the epoch. The checkpoint with lowest validation loss is saved. This count applies to this script and batch size.')
    body=t(35,75,'known label: horizontal',34,'c-e')+g(t(35,185,'prediction: [0.30, 0.70]',35,'c-a'),1)+g(t(35,295,'loss = −log(0.30) = 1.204',35,'c-a'),2)
    body+=g(t(35,410,'Use the known label to correct the weights.',30),3)
    add('training-one-image','Suppose the model gets this training image wrong',body,'The loss uses the probability of the known label, even when the model prefers the other one.',
      'Which of the two probabilities belongs inside the logarithm?','Keep the label horizontal visible while students choose the correct probability.',
      'These probabilities are an illustrative training example, not a saved checkpoint measurement. The actual minibatch loss averages the per-example cross-entropies before computing gradients.')
    body=t(35,75,'196 patch rows + CLS',38,'c-e')+g(t(35,185,'197 × 197 = 38,809 scores per head',34,'c-a'),1)
    body+=g(t(35,295,'3 heads × 12 blocks',34)+t(35,400,'1,397,124 scores for one image',37,'c-a'),2)
    add('real-work-count','How much matching happens inside the tiny real model?',body,'Even this small ViT compares many pairs of rows.',
      'How can a 14-by-14 patch grid produce more than a million scores?','Count pairs, then heads, then blocks. Keep token count separate from pixel count.',
      'The count is 197²×3×12 dense attention coefficients across this model’s layers for one image. It does not include projection/MLP computation and does not state how many coefficients an optimized kernel stores simultaneously.')
    body=t(35,70,'probability of the known breed: 0.957',35,'c-e')+g(t(35,200,'one successful prediction',34,'ink'),1)+g(t(35,330,'How often does that happen on new photos?',32,'c-a'),2)
    add('one-photo-limit','Does one correct photograph tell us the accuracy?',body,'To estimate accuracy, count correct predictions over an appropriate test set.',
      'Can we report 95.7% accuracy from a 95.7% probability on one dog?','Separate a model’s probability on one image from a fraction correct over many images.',
      'The dog and cat examples show real inference. Their probabilities are not an accuracy estimate or proof that either image was absent from all pretraining data. A dataset evaluation needs a fixed label mapping, an appropriate held-out split and aggregate metrics.')
    body=t(35,70,'A[row, column]',38,'c-a')+g(t(35,180,'row = receiver: whose query?',34,'c-q'),1)+g(t(35,290,'column = source: whose key/value?',32,'c-k'),2)
    body+=g(t(35,410,'one CLS row → 196 patch weights + 1 CLS weight',28),3)
    add('read-attention-map','What is one coloured square actually showing?',body,'It is one source weight for a specified receiver, head and block.',
      'Where is the self-weight for CLS on a 14-by-14 patch map?','Account for the 197th source before comparing colours.',
      'The grid maps patch-source weights back to the input’s 14×14 patch locations. It omits the extra CLS source from the picture but reports that weight separately. The colours do not directly represent object labels or final class probabilities.')
    inspection=json.loads((ASSETS/'inspection.json').read_text())
    import base64
    processed='data:image/png;base64,'+base64.b64encode((ASSETS/'model-input.png').read_bytes()).decode()
    rgb=','.join(str(round(c*255)) for c in inspection['preprocessing']['mean'])
    for i,record in enumerate(inspection['occlusion']):
        x,y,size=35,45,345
        body=image(x,y,size,size,processed)
        bx=x+record['column']/224*size;by=y+record['row']/224*size
        body+=f'<rect x="{bx}" y="{by}" width="{size/2}" height="{size/2}" fill="rgb({rgb})"/>'
        body+=t(505,100,record['region']+' covered',32)+t(505,185,'Before: 95.7% Newfoundland',29,'ink-2')
        body+=g(t(505,300,f"After: {100*record['target_probability']:.1f}%",43,'c-e'),1)
        add('cover-'+str(i+1),'What happens if we cover the '+record['region'].lower()+'?',body,
          'Keep the image, model and target class fixed. Change this one region.',
          'Will the dog probability rise, fall, or stay close?','Take a prediction, reveal the measured probability, then compare with the uncovered image.',
          'The cover is 112×112 pixels in the 224×224 model input. Its colour is the model’s mean RGB, corresponding to zero after normalization. This visual shows the same intervention used by the saved occlusion experiment. Masking probes this intervention and also changes the input distribution.')

    # Authored split slides: new questions and captions for each operation.
    replacements={
      'rgb-flatten':split('rgb-flatten',[
        (1,'How many numbers are in four RGB pixels?','Four pixels, three channels each: twelve numbers.','Does one pixel contribute one number or three?','Count RGB channels before flattening.'),
        (2,'Write those twelve numbers in one row','Keep the same pixel order each time.','Which three entries came from the white pixel?','Trace each RGB triple back to its coloured pixel.')]),
      's01-rows':split('s01-rows',[
        (1,'Give each patch its own row of pixels','A row starts with the pixel values from one fixed part of the image.','Do we mix pixels from different patches at this step?','Follow one crop into its own row.'),
        (2,'Use the same projection on every patch','Every output row has the same width, so the attention block can process them together.','How many different projection matrices do we need?','Point to the shared operation between the four rows.')]),
      's02-projection':split('s02-projection',[
        (1,'Flatten P1 so we can multiply it','These are the same four pixel values, in a fixed order.','Where does each entry in the row come from?','Point to one pixel and its matching row entry.'),
        (2,'Compute the first coordinate of P1','This chosen column takes the mean of the four pixels.','What is one quarter of each pixel, added together?','Add all four terms before revealing 1.'),
        (3,'Write the whole content row','The first coordinate is ink; the final 1 comes from the bias.','What happened to the two middle coordinates?','Use the zero columns and then add the bias.')]),
      's02-positions':split('s02-positions',[
        (1,'Give each patch a location vector','Two patches can look alike and still come from different places.','Which entries record the row and column here?','Read each location from the small image grid.'),
        (2,'Add content and location, coordinate by coordinate','These five rows form E, the input to our attention calculation.','What distinguishes P1 from P2 now?','Compare the two filled patches after adding positions.')]),
      's03-weights':split('s03-weights',[
        (1,'Repeat the dot product for every source','Keep the query fixed. Only the source key changes.','Which of these keys give the same score?','Work down the five sources, including CLS.'),
        (2,'Exponentiate the five scores','Scores 0 and 1 become 1 and e.','Why are four exponentials equal?','Match each exponential to its score.'),
        (3,'Divide by one shared sum','The five source weights now add up to one.','What fraction belongs to CLS?','Divide its exponential 1 by the same sum used for every patch.')]),
      's03-values':split('s03-values',[
        (1,'Put each value beside its weight','Keep a patch’s weight and value on the same row.','Which value belongs to the weight for P3?','Trace the source label across the table.'),
        (2,'Multiply the value by its weight','One scalar scales both coordinates of the source value.','Does the second coordinate get a different weight?','Multiply both entries before moving to the next row.'),
        (3,'Add the contributions to get one message','Add down each coordinate column.','How do five source rows become a two-number message?','Sum each coordinate separately, then reveal the result.')]),
      's03-second':split('s03-second',[
        (1,'Head 2 compares ink and column','The same input rows now pass through a different key projection.','Which patch has both ink and column equal to 1?','Point to P2 before calculating its score.'),
        (2,'Give Head 2 its own softmax','Each head divides by its own sum of exponentials.','Can we reuse Head 1’s denominator here?','Compare the score lists before normalizing.')]),
      's03-second-values':split('s03-second-values',[
        (1,'What information does Head 2 send?','Its values carry ink and column.','Which value coordinate changed from Head 1?','Keep the same source labels while comparing value definitions.'),
        (2,'Calculate each Head 2 contribution','Use Head 2’s weight and Head 2’s value from the same source.','Which source contributes most to the ink coordinate?','Compute P2’s product, then compare the other sources.'),
        (3,'Add Head 2’s contributions','We now have a second two-number message.','Are the two heads’ messages equal?','Compare the sums with Head 1’s result.')]),
      's04-join':split('s04-join',[
        (1,'Keep both messages by putting them side by side','Concatenation gives us four numbers; it does not average the heads.','What would we lose by averaging these two rows immediately?','Follow all four coordinates into the joined row.'),
        (2,'Use W_O to make the first update coordinate','The first column subtracts Head 2’s ink value from Head 1’s.','Which two entries of W_O make that subtraction?','Trace the +1 and −1 from the joined row.'),
        (3,'Collect all four output coordinates','The update has the same width as the row we will add it to.','Why do we need four update coordinates?','Match the update width to E before moving to the residual.')]),
      's04-residual':split('s04-residual',[
        (2,'Add the message to the starting CLS row','Only the last starting coordinate is nonzero in this example.','Which coordinate changes because the original CLS contains a 1?','Add the two rows one coordinate at a time.'),
        (3,'Use the updated CLS row to score the two labels','The class matrix maps four coordinates to two scores.','Why does a negative first coordinate favour Across here?','Apply −4 and +4 to that same coordinate.')]),
      's04-probability':split('s04-probability',[
        (2,'Turn the class scores into probabilities','The alternatives here are Across and Down.','Which two exponentials belong in this denominator?','Normalize over the two labels, then read the bars.'),
        (3,'Use the known label to calculate the loss','For this image, the correct label is Across the top.','Which probability belongs inside the logarithm?','Keep the image and correct label visible while calculating the loss.')]),
      'one-update':split('one-update',[
        (1,'Use the gradient to change the class bias','Subtract half the gradient from the two bias values.','Why does the correct class bias increase?','Read the sign of p minus the one-hot target.'),
        (3,'Run the prediction again after that update','The correct-class probability rises and its loss falls on this example.','Did changing the bias alter the image or the attention weights?','Keep the frozen worksheet separate from the two bias values we changed.')]),
      'mlp-row':split('mlp-row',[
        (2,'Apply GELU to each hidden coordinate','GELU(1) is about 0.841, and GELU(−1) is about −0.159.','Does GELU send every negative input to zero?','Compare the two signs using these actual values.'),
        (4,'Add the MLP message to the starting row','Both vectors have width two, so we can add them coordinate by coordinate.','Where does the 1.841 come from?','Add 1 and 0.841, then repeat for the second coordinate.')]),
      's05-scale':split('s05-scale',[
        (1,'How many rows does the real photograph produce?','A 14-by-14 patch grid gives 196 patch rows, plus CLS.','Why is there one more row than image patches?','Count the grid, then add the summary row.'),
        (2,'How many blocks process those rows?','This checkpoint has 12 blocks and three heads in each block.','Does every head read the same input rows?','Recall the shared E and separate projections from our two-head calculation.'),
        (4,'What changed when we made the model larger?','There are more patches, wider rows and more blocks. The operations follow the same pattern.','Which dimensions are different from our four-patch worksheet?','Trace pixels, patch rows, blocks and the 1,000 class scores.')]),
      'learning-curves':split('learning-curves',[
        (1,'Is the model improving on its training images?','Training accuracy measures the examples used for the weight updates.','Could memorization alone make this curve look good?','Read the training curve before revealing validation.'),
        (2,'Does the improvement carry over to validation?','Validation images help us choose a checkpoint without using the test set.','What would a widening gap between the curves suggest?','Compare the curves at the same epoch.')]),
      'three-phases':split('three-phases',[
        (1,'Where did the checkpoint learn its visual features?','Pretraining uses many images to fit a starting model.','Are we starting the photo experiment with random weights?','Point to the pretraining phase and the supplied checkpoint.'),
        (2,'How would we adapt it to our own labels?','Fit a new classifier; choose whether to update the image encoder too.','What changes if our labels are only cat and dog?','Distinguish a frozen-encoder linear probe from fine-tuning.'),
        (3,'What happens when we classify a new photograph?','Inference uses the fitted weights without updating them.','Does the model train again for every new photo?','Follow a new image through the fixed network.')]),
    }

    def use(key):return replacements.get(key,[old[key]])
    def seq(*keys):return [html for key in keys for html in (use(key) if key in old else [new[key]])]
    # The same 14 section anchors remain useful for remote review.
    revised=[
      ('What would we like the image model to do?',seq('s01-photo','photo-folder','find-animal','photo-search','s01-context','bridge-text','patch-context','image-to-rows')),
      ('How do pixels become patch rows?',seq('s01-patches','one-rgb','rgb-flatten','flatten-order','s01-rows','projection-size')),
      ('What does each row know about its patch?',seq('s02-small','position-question','s02-projection','patch-matrix','empty-patch','mean-loses-edge','edge-filters','why-cls','two-identical-patches','s02-positions')),
      ('How does one row read the other rows?',seq('image-mask','qkv-roles','all-qkv','q-dot','one-key-dot','s03-query','one-score','softmax-relative','s03-weights','weight-denominator','s03-values','one-value-product','weight-message','change-query')),
      ('What does a second head add?',seq('heads-question','s03-second','s03-second-values','all-receivers','s04-join','other-output-coordinates')),
      ('How do the messages lead to a prediction?',seq('residual-zero','s04-residual','two-softmaxes','s04-probability','loss-comparison','one-update','s04-experiment')),
      ('What happens inside a complete block?',seq('s05-block','ln-mean','ln-variance','layernorm','mlp-first-linear','mlp-row','mlp-second-linear','depth','s05-scale')),
      ('Can we follow the same steps in code?',seq('two-sixteens','code-patch','cnn-context','code-attention','code-block','code-add-cls','code-add-pos','code-cls-readout','code-model','code-train','code-backward','code-step')),
      ('How do we know the model learned something?',seq('training-data','three-splits','training-one-image','one-epoch','learning-curves','trained-position-control')),
      ('What happens on the real photographs?',seq('real-input','s06-answer','one-photo-limit','real-cat','three-phases')),
      ('What can we learn by inspecting the model?',seq('read-attention-map','real-heads','real-depth','real-patch-query','cover-1','cover-2','cover-3','cover-4','occlusion')),
      ('How much work does a finer patch grid require?',seq('patch-cost','real-work-count','cost-control')),
      ('Can you work through a new example?',seq('exercise-message','exercise-shapes','exercise-position')),
      ('What can we build with visual rows next?',seq('next-vision','closing'))]
    # Put the second MLP linear calculation before its residual, preserving order.
    title,frames=revised[6]
    residual_id='mlp-row'
    residual=next(x for x in frames if f'id="{residual_id}"' in x)
    frames.remove(residual)
    position=next(i for i,x in enumerate(frames) if 'id="mlp-second-linear"' in x)
    frames.insert(position+1,residual)
    return revised
