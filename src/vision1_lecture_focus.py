"""Keep one photo forward pass; retain only new ideas in the continuation."""
import re
from textwrap import wrap
from vision1_focus_common import Figures


def consolidate(b, sections):
    f = Figures(b)
    t,g,box,arrow,line=f.t,f.g,f.box,f.arrow,f.line
    def key(markup):
        return re.search(r'class="frame[^\"]*" id="([^\"]+)"', markup).group(1)
    existing={key(fr):fr for _,frames in sections for fr in frames}

    body=t(35,36,'Two small grayscale patches: same mean, different arrangement',29)
    for i,(patch,flat,result,label) in enumerate([
        ([[1,1],[0,0]],'[1, 1, 0, 0]','[2, 0]','Horizontal edge'),
        ([[1,0],[1,0]],'[1, 0, 1, 0]','[0, 2]','Vertical edge')]):
        y=85+i*180
        body+=f.pixels(42,y,patch,48,True,False)
        body+=t(181,y+40,flat,27,'c-e')+t(181,y+84,'mean = 0.5',23,'ink-2')
        body+=arrow(382,y+44,450,y+44,'c-e')
        body+=box(470,y+4,380,['same two weighted sums','[1, 1, −1, −1]  ·  [1, −1, 1, −1]'],'c-v',size=21,h=88)
        body+=arrow(860,y+44,925,y+44,'c-v')
        body+=t(1018,y+42,result,34,'c-v','middle')+t(1018,y+81,label,21,'ink-2','middle')
    f.add('patch-filter-patterns','Weighted filters can see more than average brightness',body,
        'These two patches have the same mean. Two different weighted sums distinguish their spatial patterns. A learned projection can retain several useful features from the same pixels; averaging alone discards this distinction.',
        'Would one average-brightness number distinguish these patches?',
        'No. Apply the same two filters to each flattened patch: the outputs are [2,0] and [0,2]. Both biases are zero in this chosen example.',
        'This isolated 2×2 grayscale example explains what a projection can represent; it does not start another classifier walkthrough. '
        'With row-major flattening, the two columns of W are [1,1,−1,−1] and [1,−1,1,−1], and b=[0,0]. '
        'The real photo uses 768 inputs and 192 learned output features. These chosen edge weights are not measurements from that checkpoint.')

    body=t(35,40,'The same normalization, applied to two different questions',29)
    body+=box(35,103,325,['ATTENTION · one query','Which source rows help?'],'c-k',size=23)
    body+=arrow(374,141,452,141,'c-k')+box(471,103,650,['197 scores → softmax over sources → 197 weights','CLS + P1 + P2 + … + P196'],'c-k',size=24)
    body+=g(box(35,276,325,['CLASSIFIER · one image','Which image label fits?'],'c-e',size=23)
        +arrow(374,314,452,314,'c-e')+box(471,276,650,['1,000 logits → softmax over labels → probabilities','Newfoundland, Persian cat, …'],'c-e',size=24),1)
    f.add('photo-two-softmaxes','Two softmaxes, two different questions',body,
        'Attention normalizes across 197 source rows for each query and head. The classifier normalizes across 1,000 labels for each image. Both sum to one along their chosen axis; only the second distribution predicts the image label.',
        'Does a large attention weight mean a high probability of the dog class?',
        'No. It means that one source contributes strongly to that query’s value mixture. The class head later scores labels using the final CLS features.',
        'For a batch, attention weights have shape B×3×197×197 and class probabilities have shape B×1000. '
        'Softmax uses the last axis in both cases, but the axes mean different things. Cross-entropy in the training code accepts logits and includes log-softmax internally.')

    def insert_after(frames, anchor, extra):
        result=[]
        for fr in frames:
            result.append(fr)
            if key(fr)==anchor: result.append(extra)
        assert any(key(fr)==anchor for fr in frames), anchor
        return result
    first=list(sections[:3])
    first[1]=(first[1][0],insert_after(first[1][1],'patch-shared-code',f.frames['patch-filter-patterns']))
    title,frames=first[2]
    for anchor,extra in [
        ('real-heads-intro',existing['heads-visual-roles']),
        ('real-classifier-softmax',f.frames['photo-two-softmaxes'])]:
        frames=insert_after(frames,anchor,extra)
    first[2]=(title,frames)

    def opener(n,title,previous,question,caption):
        body=t(48,58,'SECTION',23,'c-e',weight=600)+t(48,177,f'{n:02}',110,'c-e',weight=600)
        body+=line(220,35,220,405,'c-e',2)
        for j,phrase in enumerate(wrap(title,32)):
            body+=t(264,78+j*60,phrase,43,'ink',weight=700)
        for j,phrase in enumerate(wrap(previous,59)):
            body+=t(264,235+j*34,phrase,26,'ink-2')
        for j,phrase in enumerate(wrap(question,49)):
            body+=t(264,335+j*36,phrase,29,'ink',weight=600)
        return f.add(f'vision-topic-{n:02}',f'Section {n} · {title}',body,caption,
                     question,'Connect the previous result to this new question. Keep the same image classifier as the reference.').replace(
                         'class="frame vp-frame"','class="frame vp-frame vp-topic-break"',1)

    from vision1_photo_learning import build as learning
    from vision1_cnn_comparison import build as comparison
    from vision1_photo_code import build as code
    new=[
        ('Whole model walkthrough',learning(b),
         'We have opened each part of the classifier.',
         'How do forward and backward fit together?',
         'One example: follow the full model to a label loss, then reverse the path to compute gradients and update the parameters.'),
        ('CNNs, ViTs and inductive bias',comparison(b),
         'Attention lets patches exchange information.',
         'How do they differ, and which should we try?',
         'Three visual comparisons: how each model uses an image, its starting assumptions, and practical choices for data and compute.'),
        ('The same classifier in PyTorch',code(b),
         'We know the forward path and the learning signal.',
         'How do these diagrams become code?',
         'Keep the original 224×224 RGB input. Match each code operation to its tensor shape, beginning with Conv2d patch embedding.')]
    result=first
    for n,(title,frames,previous,question,caption) in enumerate(new,4):
        result.append((title,[opener(n,title,previous,question,caption)]+frames))

    # Keep later adaptation, measured inspection, cost and transfer exercises.
    # Their section dividers must use the new consecutive numbering.
    for n,(title,old) in enumerate(sections[8:],7):
        content=[fr for fr in old if not key(fr).startswith('vision-topic-')]
        old_meta=next(x for x in b['FRAMES'] if x['id']==key(old[0]))
        questions={
            7:('Our code produces 1,000 ImageNet scores.','How would we adapt it to dog versus cat?'),
            8:('Training and inference have different jobs.','What do the saved photo predictions establish?'),
            9:('A prediction tells us the model’s answer.','What can we measure inside its attention blocks?'),
            10:('Every query compares all source rows.','What happens when we make the patches smaller?'),
            11:('We have followed one complete image classifier.','Can you reason about a new shape or intervention?'),
            12:('We can turn pixels into an image representation.','What could a different prediction task use it for?')}
        previous,question=questions[n]
        result.append((title,[opener(n,title,previous,question,old_meta['caption'])]+content))
    return result
