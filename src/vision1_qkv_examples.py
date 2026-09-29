"""Photograph-based Q/K/V intuitions alongside the familiar text bridge."""
from html import escape


def examples(b):
    frame, t, g, rect, arrow, image, crop = (b[k] for k in
        ['frame', 't', 'g', 'rect', 'arrow', 'image', 'crop'])
    groups = {'bridge-image-query': [], 'bridge-image-key': [], 'bridge-image-value': []}
    sub = lambda n: str(n).translate(str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉'))
    analogy = ('These are human interpretations of possible learned behaviour. '
               'No attention head or semantic coordinate is measured on these slides. '
               'The classifier receives pixels, and its class-label loss trains the projection matrices. '
               'In a full pre-LN block, x denotes a normalized current row. '
               'At deeper layers, that row can already contain context from other patches. ')

    def add(after, key, title, body, caption, question, point, prose, mobile):
        groups[after].append(frame(key, title, body, caption, question+'\n'+point,
                                   analogy+prose, mobile))

    # Each question is grounded in a visible receiver and two actual source crops.
    query_cases = [
        ('dark', 9, 'Could this dark texture belong to the animal?',
         ('Which patches could connect', 'this texture to the animal?'),
         [(6, 'eye and muzzle'), (10, 'more of the coat')],
         'Face and coat clues could help this patch represent part of the animal. The classifier will still predict one label for the whole photo.'),
        ('face', 6, 'Where is the rest of this face?',
         ('Which patches could complete', 'the face around this eye?'),
         [(5, 'other side of the face'), (10, 'neck and coat')],
         'One patch contains only part of a face. Context from other patches could help its representation describe a larger visual structure.'),
        ('branch', 3, 'Where does this branch continue?',
         ('Which patches could continue', 'this branch beyond the crop?'),
         [(7, 'branch below'), (2, 'branch to the left')],
         'Background patches can also gather context. A branch representation could use nearby edge continuity; the task still predicts the image label.'),
    ]
    for name, receiver, title, question_lines, sources, caption in query_cases:
        n = receiver+1
        body=t(25,32,'Same task: one dog/cat label for the whole photograph.',24,'ink-2')
        body+=image(25,100,330,220.44)
        body+=rect(25+(receiver%4)*82.5,100+(receiver//4)*55.11,82.5,55.11,'c-q','transparent',0)
        body+=arrow(365,200,400,200,'c-q')
        body+=crop(415,125,210,140.28,receiver,'query-'+name)+rect(415,125,210,140.28,'c-q','transparent',0)
        body+=t(520,100,f'receiver P{n}',27,'c-q','middle')
        body+=g(t(695,98,'A possible query:',26,'c-q')
                +t(695,142,question_lines[0],25)+t(695,180,question_lines[1],25),1)
        body+=g(t(695,226,'Possible source clues',23,'ink-2'),2)
        for j,(idx,label) in enumerate(sources):
            x=715+j*245
            body+=g(crop(x,247,145,96.86,idx,'query-source')
                    +t(x+72.5,376,label,22,'c-k','middle'),2)
        body+=g(t(25,374,f'q{sub(n)} = x{sub(n)} W_Q',31,'c-q')
                +t(25,416,f'x{sub(n)} is the current row for P{n}.',24,'ink-2')
                +t(695,423,'The “question” is a learned vector.',23,'c-q'),3)
        mobile=f'<p><strong>Receiver P{n}</strong></p><svg viewBox="0 0 330 221" role="img" aria-label="Receiver location in the photograph">'
        mobile+=image(0,0,330,220.44)+rect((receiver%4)*82.5,(receiver//4)*55.11,82.5,55.11,'c-q','transparent',0)+'</svg>'
        mobile+='<p><strong>A possible query:</strong> '+escape(' '.join(question_lines))+'</p>'
        for idx,label in sources:
            mobile+=f'<figure class="vp-qkv-crop"><svg viewBox="0 0 210 141" role="img" aria-label="{escape(label)}">'+crop(0,0,210,140.28,idx,'mobile-query')+f'</svg><figcaption>Source P{idx+1}: {escape(label)}</figcaption></figure>'
        mobile+=f'<p>q{sub(n)} = x{sub(n)} W_Q. These questions express intuition; the model computes vectors.</p>'
        add('bridge-image-query','query-example-'+name,title,body,caption,
            'What is missing from this crop, and which other crop could help interpret it?',
            'Locate the purple receiver in the photograph. Reveal its possible question, then inspect the two source crops before showing the projection.',
            'All three examples use the same W_Q within one head and layer. Different receiver rows can produce different queries. '
            'The selected sources illustrate potential context; actual attention compares all allowed keys, including the receiver’s own key. '
            'No separate fur, face or branch label is supplied for a patch. Useful contextual features are learned through the whole-image objective.',mobile)

    sources=[(6,'eye / muzzle',('face-like shape', 'and appearance'),('shape around the', 'eye and muzzle')),
             (10,'coat',('fur-like texture', 'and appearance'),('coat texture', 'and colour')),
             (7,'branch',('branch-like edges', 'and appearance'),('background edges', 'and context'))]

    def mobile_source(idx, label, description, kind):
        n=idx+1
        return ('<figure class="vp-qkv-crop">'
                +f'<svg viewBox="0 0 210 141" role="img" aria-label="{escape(label)} source crop">'
                +crop(0,0,210,140.28,idx,'mobile-source')+f'</svg><figcaption><strong>P{n} · {escape(label)}</strong></figcaption></figure>'
                +f'<p>Possible {"matching features" if kind=="K" else "information to send"}: {escape(" ".join(description))}. '
                +f'{kind.lower()}{sub(n)} = x{sub(n)} W_{kind}.</p>')

    body=t(25,34,'Keep the dark receiver P10: seek clues that connect it to the animal.',26,'c-q')
    for j,(idx,label,key_reading,_) in enumerate(sources):
        x=30+j*390; mid=x+160; n=idx+1
        body+=crop(x+60,65,200,133.6,idx,'key-example')+t(mid,234,f'P{n} · {label}',26,'ink','middle')
        body+=g(t(mid,277,key_reading[0],25,'c-k','middle')+t(mid,310,key_reading[1],25,'c-k','middle'),1)
        body+=g(t(mid,362,f'k{sub(n)} = x{sub(n)} W_K',28,'c-k','middle')
                +t(mid,416,f'compare q₁₀ with k{sub(n)}',24,'c-q','middle'),2)
    add('bridge-image-key','key-example-sources','What could each source offer for matching?',body,
        'A source key can match one query well and another poorly. Comparing this receiver’s query with all source keys sets its attention weights.',
        'Could the branch key be useful to the branch receiver even if the dark receiver gives it little weight?',
        'Hold the dark query fixed and compare the three source keys. Then recall the branch query: a key’s relevance depends on the query.',
        'The crop descriptions give intuition for learned matching features; they are neither key vectors nor fixed semantic labels. '
        'Each source makes a key with the same W_K. Relevance belongs to a query–key pair: a source does not have one universal importance score. '
        'Scaled dot products followed by softmax choose weights over all allowed sources. The three shown here are a visual subset.',
        '<p><strong>Receiver:</strong> dark patch P10, with query q₁₀.</p>'
        +''.join(mobile_source(idx,label,k,'K') for idx,label,k,_ in sources)
        +'<p>A different receiver can find different keys relevant. The words describe intuition for numerical vectors.</p>')

    body=t(25,34,'Use the same source patches. Now ask what information each could send.',26,'c-v')
    for j,(idx,label,_,value_reading) in enumerate(sources):
        x=30+j*390; mid=x+160; n=idx+1
        body+=crop(x+60,65,200,133.6,idx,'value-example')+t(mid,234,f'P{n} · {label}',26,'ink','middle')
        body+=g(t(mid,279,value_reading[0],25,'c-v','middle')+t(mid,312,value_reading[1],25,'c-v','middle'),1)
        body+=g(t(mid,363,f'v{sub(n)} = x{sub(n)} W_V',28,'c-v','middle'),2)
    body+=g(t(580,423,'message for P10 = a₁₀,₇ v₇ + a₁₀,₁₁ v₁₁ + a₁₀,₈ v₈ + …',29,'c-v','middle'),3)
    add('bridge-image-value','value-example-messages','What information could these values carry?',body,
        'Each source supplies one value vector. Different receivers mix these values with different weights. Here a₁₀,₇ is the weight for P10 reading P7.',
        'If two receivers read the same source, does the source need a new value vector for each receiver?',
        'Keep the source crops fixed from the key slide. Follow W_V into the value vectors, then reveal the weighted sum; the dots include the other sources.',
        'The descriptions suggest possible visual information carried by learned features. They are not measured meanings of particular coordinates. '
        'Within a head and layer, a source has one value vector shared by all receivers; each receiver can assign a different weight to it. '
        'The sum includes every allowed source, including self and CLS where present. It produces a vector that contributes to the receiving row update. '
        'This operation neither pastes pixels nor directly chooses the dog/cat label. The classifier later reads the image summary. '
        'The dog walkthrough later draws the full Q/K/V matrices and calculates a measured CLS message.',
        ''.join(mobile_source(idx,label,v,'V') for idx,label,_,v in sources)
        +'<p>Message for P10 = a₁₀,₇ v₇ + a₁₀,₁₁ v₁₁ + a₁₀,₈ v₈ + …</p>'
        +'<p>a₁₀,₇ is the weight for P10 reading P7. Other source values also contribute. Every receiver uses its own weights.</p>')
    return groups
