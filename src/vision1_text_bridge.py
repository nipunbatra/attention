"""Recall the text computation, then ask for each image counterpart."""
import json
import re


def bridge(b, sections):
    frame, t, g, rect, arrow, image, crop, line, mobile_rows = (
        b[k] for k in ['frame', 't', 'g', 'rect', 'arrow', 'image', 'crop', 'line', 'mobile_rows'])
    toy = json.loads((b['SRC'] / 'toy.json').read_text())
    tokens = toy['sentences']['river']
    assert tokens[5:7] == ['river', 'bank'] and tokens[-1] == 'the'
    prefix = ' '.join(tokens)
    reference = ('<a href="attention.html#s16-flow-frame">Part II: the complete attention diagram</a> '
                 'and <a href="attention.html#s10">bank in two contexts</a>. '
                 'The sentence and token embedding are taken from <a href="src/toy.json">the original text toy</a>. ')
    slides = []

    def box(x, y, width, label, color='c-e', size=28):
        return rect(x, y, width, 60, color, 'transparent') + t(x+width/2, y+40, label, size, color, 'middle')

    def add(key, title, body, caption, question, point, prose, mobile):
        slides.append(frame(key, title, body, caption, question+'\n'+point, prose, mobile))

    body=t(35,45,prefix+' …',29,'ink')
    for x,label in [(35,'selected tokens'),(285,'embedding + position'),(660,'read context'),(940,'updated rows')]:
        body+=t(x,105,label,24,'ink-2')
    for y,pos in [(145,6),(225,7),(305,10)]:
        word=tokens[pos-1]
        subscript=str(pos).translate(str.maketrans('0123456789','₀₁₂₃₄₅₆₇₈₉'))
        body+=box(35,y,175,f'{word} ({pos})',size=28)
        body+=g(arrow(210,y+30,285,y+30)+box(285,y,230,'e'+subscript,size=33),1)
        body+=g(box(930,y,190,'e'+subscript+'′','c-d',33),3)
    body+=g(arrow(515,255,630,255)+rect(630,145,235,220,'c-q','transparent')
            +t(747,205,'Q, K, V',34,'c-q','middle')+t(747,265,'attention',31,'c-v','middle')
            +t(747,325,'+ original row',25,'ink','middle'),2)
    body+=g(arrow(865,255,930,255,'c-d'),3)
    body+=g(t(35,427,'bank reads positions 1–7; the final the reads positions 1–10.',27,'ink-2'),4)
    add('text-context-recap','Back to text: what did attention update?',body,
        'Each token starts with an embedding plus position. Attention supplies context; adding its update gives a new representation of the same token.',
        'Did attention replace the word bank, or change the numbers representing it?',
        'Trace bank through the middle row. Then point to the final the: that is the row used to predict the next word after this full prefix.',
        reference+'This is a compact redraw of the same text-attention path, with three of the ten rows shown. '
        'Every token has an initial embedding plus position, makes query/key/value projections, and receives a context-dependent update. '
        'The residual connection adds the projected attention message to the original row. '
        'Causal attention lets bank at position 7 read positions 1–7; the last the at position 10 can read all ten observed positions. '
        'The contextual bank row helps us understand the mechanism. Next-token prediction after the complete prefix reads the updated final row. '
        'During a forward pass these contextual rows change; that is different from updating the embedding-table parameters during training.',
        '<p>'+prefix+' …</p>'+mobile_rows(['Step','Text example'],[
            ['Token','bank at position 7'],['Initial row','token embedding + position'],
            ['Read context','Q and K set weights; V supplies information'],['Update','e7 + attention update → e7′'],
            ['Predict after the full prefix','Use the updated final the row, e10′']]))

    body=t(35,45,'Text: word tokens in Part II',29,'ink-2')+t(655,45,'Image: what should one token be?',29,'ink-2')
    for x,word in [(35,'the'),(205,'river'),(375,'bank')]:body+=box(x,125,140,word)
    body+=image(655,80,440,294)
    grid=''
    for j in range(1,4):
        grid+=line(655+j*110,80,655+j*110,374,'card',2)
        grid+=line(655,80+j*73.5,1095,80+j*73.5,'card',2)
    body+=g(grid+rect(765,227,110,73.5,'c-q','transparent',0),1)
    body+=g(t(275,315,'one word → one token row',28,'c-e','middle')
            +t(875,427,'one patch → one token row',28,'c-q','middle'),2)
    add('bridge-image-token','What could be the image equivalent of a token?',body,
        'We can use one fixed-size patch as an image token. Each patch gets its own row, just as each text token did.',
        'Should one image token be a pixel, an object, or a fixed-size piece of the image?',
        'Pause on the whole photo. Reveal the grid, then follow the outlined patch to the idea of one token row.',
        'Part I used character tokens and the Part II toy used word tokens. ViT uses a regular grid of fixed-size patches. '
        'The grid is chosen before the model knows which patches contain the animal. One object can span many patches, and a patch can mix object and background. '
        'This slide introduces the correspondence. The next section will calculate the patch count and read the pixel values.',
        mobile_rows(['Text','Image'],[['One token in the word-level toy','One fixed-size image patch'],['One row per token','One row per patch']])
        +'<svg viewBox="630 60 490 340" role="img" aria-label="Photo divided into fixed-size patches">'+image(655,80,440,294)+grid+rect(765,227,110,73.5,'c-q','transparent',0)+'</svg>')

    embedding=toy['tok_emb']['bank']
    vector='['+', '.join(f'{x:g}' for x in embedding)+']'
    body=t(35,45,'Text',30,'ink-2')+t(685,45,'Image',30,'ink-2')+box(170,100,170,'bank')
    body+=arrow(255,160,255,275)+t(290,225,'embedding lookup',26,'c-e')
    body+=box(35,275,475,vector,size=32)+crop(795,75,220,147,9,'embedding-bridge')
    body+=g(arrow(905,222,905,275)+t(880,255,'learned pixel projection',26,'c-e','end'),1)
    body+=g(box(685,275,440,'one learned patch row',size=29),2)
    body+=g(t(580,415,'Then add position information to both.',30,'c-e','middle'),3)
    add('bridge-image-embedding','What could be the image equivalent of an embedding?',body,
        'Text uses a learned lookup table. A patch projection produces D coordinates per patch. D is shared across patches and can differ from the number of pixel values.',
        'Where would a vector for this crop come from, if we have no word ID to look up?',
        'Read the actual bank embedding from Part II. Keep the image row unrevealed while students suggest how pixels could become numbers.',
        reference+'The displayed text vector is the token embedding before position is added. '
        'A ViT patch embedding is obtained by flattening the patch pixels and applying a shared learned linear projection, usually with a bias. '
        'Both models then add position information to form their initial rows. The embedding widths need not match between the two models. '
        'We leave the image row symbolic here because the upcoming slides derive its entries from pixels.',
        mobile_rows(['Text','Image'],[['bank → '+vector,'patch pixels → learned projection → patch row'],
                                   ['Add token position','Add patch position']]))

    def role_diagram(kind, token, index, top_reading, bottom_reading):
        color={'Q':'c-q','K':'c-k','V':'c-v'}[kind]
        role='receiver' if kind=='Q' else 'source'
        sub='dark' if kind=='Q' else 'face'
        body=t(35,40,f'Text · {role}',27,'ink-2')+box(35,75,165,token)
        body+=t(35,235,f'Image · {role}',27,'ink-2')+crop(35,260,165,110,index,'bridge-'+kind)
        for y,label_ in [(75,'e_'+token),(285,'e_'+sub)]:
            body+=g(arrow(200,y+30,255,y+30)+box(255,y,250,label_),1)
            body+=g(arrow(505,y+30,575,y+30,color)+box(575,y,160,'W_'+kind,color)
                    +arrow(735,y+30,825,y+30,color)+box(825,y,270,kind.lower()+'_'+(token if y==75 else sub),color),2)
        body+=g(t(575,190,top_reading,27,color)+t(575,405,bottom_reading,26,color),3)
        return body

    body=role_diagram('Q','bank',9,'Which context could help bank?','Which clues could help this texture?')
    add('bridge-image-query','What could a query be in the image?',body,
        'The row being updated makes a query. Its learned projection determines what kinds of source information it can match.',
        'Which patch should make the query if we want to update the dark patch?',
        'Follow the receiver row through W_Q in each case. Read the questions as intuition for a learned vector, not words typed by a user.',
        'For text, q_bank=e_bank W_Q. For vision, q_dark=e_dark W_Q. '
        'The operation has the same role in both models; their trained matrices are separate parameters. '
        'The plain-language questions illustrate what a query does. A ViT computes numerical queries from its current rows without a text prompt. '
        'In the full pre-normalized block, the projection uses the normalized version of the row. We will return to normalization in the complete-block section.',
        mobile_rows(['Text receiver','Image receiver'],[['bank row → W_Q → q_bank','dark-patch row → W_Q → q_dark'],
            ['Seek context for bank','Seek context for the dark texture']]))

    body=role_diagram('K','river',6,'q_bank · k_river → score','q_dark · k_face → score')
    add('bridge-image-key','What could a key be in the image?',body,
        'Each source row makes a key. Comparing the receiver’s query with source keys gives the scores used to choose attention weights.',
        'How could the dark patch compare a face patch with a branches patch?',
        'Keep the query on the receiver. Point to river and the face as sources, each with its own key.',
        'A key is a learned matching vector made from a source row. The text receiver bank can compare its query with river’s key; '
        'the dark image patch can compare its query with the face patch’s key. Other source rows have their own keys as well. '
        'Scaled query–key dot products are normalized together across allowed sources to form attention weights. '
        'The visual names identify crops for students; the model learns the key coordinates from data.',
        mobile_rows(['Text source','Image source'],[['river row → W_K → k_river','face-patch row → W_K → k_face'],
            ['q_bank compares with source keys','q_dark compares with source keys']]))

    body=role_diagram('V','river',6,'weight × v_river','weight × v_face')
    add('bridge-image-value','What information would a value send?',body,
        'Q and K choose the weights. V supplies the information to combine using those weights; the combined message updates the receiving representation.',
        'Once a source receives a weight, which vector actually contributes to the message?',
        'Reuse the same source crops from the key slide. Replace W_K with W_V, then multiply each value by its weight.',
        'The river token and the face patch each make a value vector from their current row, using W_V. '
        'The receiver sums weighted values from its allowed sources. The output projection maps that message to the width needed for the residual update. '
        'This is the same operation illustrated by the preceding source-to-receiver arrows. The original token or crop stays fixed while its representation gains context. '
        'The later worked examples calculate the weights and weighted sums explicitly.',
        mobile_rows(['Text','Image'],[['river row → W_V → v_river','face-patch row → W_V → v_face'],
            ['Sum weighted source values','Sum weighted source values']]))

    body=t(35,40,'Text generation',30,'c-e')+t(685,40,'Image classification',30,'c-e')
    body+=t(35,105,'The fisherman sat beside the',27)+t(35,150,'river bank and watched the ___',27)
    body+=image(795,70,220,147)
    body+=g(arrow(280,165,280,235)+box(35,235,475,'updated final token row',size=29)
            +arrow(272,295,272,335)+box(35,335,475,'vocabulary scores',size=30)
            +t(35,430,'next token: water, boats, …',27,'c-e'),1)
    body+=g(arrow(905,217,905,235)+box(685,235,440,'one image summary',size=29)
            +arrow(905,295,905,335)+box(685,335,440,'class scores',size=30)
            +t(685,430,'image label: dog or cat',27,'c-e'),2)
    add('bridge-image-target','What is the “next token” for this image?',body,
        'Here the target is an image label. We reuse attention to build representations; our classification task changes what we read out and predict.',
        'Are we predicting another patch, another word, or one label for this photograph?',
        'Pause before revealing the image answer. Contrast vocabulary scores with class scores, then return to how the patch rows are built.',
        'There is no next-token target in this image-classification task. The observed image leads to scores over the class vocabulary. '
        'For the text prefix, the updated final the row leads to scores over possible next words; bank’s internal row is not the final prediction row. '
        'Both models score possible answers, but the inputs, readout and training targets differ. '
        'Image captioning would bring back next-token prediction, conditioned on the image and preceding generated words. '
        'Now we have the correspondences. The next drawing assembles the image path, and the following section calculates the patch embeddings.',
        mobile_rows(['Text generation','Image classification'],[['Prefix tokens','Whole image'],['Updated final token row','One image summary'],
            ['Vocabulary scores → next token','Class scores → image label']]))

    revised=[]
    for title, frames in sections:
        out=[]
        for html in frames:
            if re.search(r'class="frame[^\"]*" id="image-to-rows"',html):
                out.extend(slides)
            out.append(html)
        revised.append((title,out))
    return revised
