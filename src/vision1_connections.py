"""Task, Q/K/V and readout bridges to the earlier text-attention lessons."""
import re


def connect(b, sections):
    globals().update({k:b[k] for k in ['frame','t','g','line','arrow','rect','image','crop','CAT']})
    slides={}
    def add(key,title,body,caption,question,point,prose=''):
        slides[key]=frame(key,title,body,caption,question+'\n'+point,prose)
    def label(x,y,word,color='c-e',width=160):
        return rect(x,y,width,60,color,'transparent')+t(x+width/2,y+40,word,28,color,'middle')
    def path(y,words,colors):
        body=''
        for i,(word,color) in enumerate(zip(words,colors)):
            x=25+i*290
            body+=label(x,y,word,color,235)
            if i<len(words)-1:body+=arrow(x+245,y+30,x+280,y+30)
        return body

    body=t(30,65,'Text generation: continue the prefix',33)
    body+=path(125,['a a b','last row','letter scores','next letter'],['c-e','c-e','c-a','c-e'])
    body+=g(t(35,300,'Choose a letter. Append it. Run again.',34,'c-v'),1)
    body+=g(arrow(1080,205,1080,390)+arrow(1080,390,130,390)+arrow(130,390,130,205),2)
    add('task-next-token','What were we asking the text model to predict?',body,'A prefix supplies the context. The answer is a distribution over the next token.',
        'After predicting one letter, how would we generate a whole name?','Follow the loop back to the input, now with one more token.',
        'This recalls the autoregressive task in the text series. At generation time we read the final available token row, score the vocabulary, select a token and extend the prefix. During training we can score many positions in parallel using a causal mask and shifted targets. No particular next letter or probability is assumed here.')
    body=image(30,85,360,240)+g(arrow(420,205,490,205)+label(520,175,'image summary',width=250),1)
    body+=g(arrow(800,205,860,205)+t(900,165,'dog',34,'c-e')+t(900,240,'cat',34,'ink-2')+t(900,325,'class scores',26,'c-a'),2)
    add('task-image-label','What are we asking the image model to predict?',body,'We observe the whole photograph and predict one label for it.',
        'Are we trying to guess a missing patch in this task?','Point to the full observed image and the two possible class labels.',
        'This is supervised image classification. The training pair is an image and its class label. The two-label dog/cat example motivates the task; the later pretrained checkpoint uses 1,000 ImageNet labels. Predicting a future or missing patch would require a different objective.')
    body=t(30,45,'',25)+t(380,55,'Text generation',31,'c-e')+t(800,55,'Image classification',30,'c-e')
    for i,(name,left,right) in enumerate([
      ('given','prefix tokens','whole image'),('predict','next token','image label'),
      ('read out','last available row','CLS or pooled rows'),('loss','−log p(next token)','−log p(correct class)')]):
        y=140+87*i;mark=t(30,y,name,27,'ink-2')+t(380,y,left,29)+t(800,y,right,27)
        body+=mark if i==0 else g(mark,i)
    add('task-side-by-side','What changed, and what stayed the same?',body,'Both models score possible answers. The target, available context and readout differ.',
        'Which parts of our old calculation can we reuse unchanged?','Compare one row at a time, ending with cross-entropy.',
        'Both tasks can use the same attention operation and a linear classifier followed by softmax. In the autoregressive setup, each position predicts the next token from its allowed prefix. In the image-classification setup, one image-level representation predicts the supplied label. “Text” alone does not imply a causal mask: text encoders can read both directions too.')
    body=''
    for i,(x,title,causal) in enumerate([(70,'Predict the next token',True),(690,'Classify the whole image',False)]):
        body+=t(x,50,title,29,'c-e')
        for r in range(4):
            for c in range(4):
                allowed=not causal or c<=r
                body+=rect(x+c*72,100+r*65,64,56,'line','t-e' if allowed else 'card')
                body+=t(x+c*72+32,138+r*65,'✓' if allowed else '×',27,'c-e' if allowed else 'c-a','middle')
        body+=t(x,405,'future targets hidden' if causal else 'all patches already observed',26,'ink-2')
    add('task-mask-reason','Why did our text predictor hide later tokens?',body,'A causal mask prevents the model from reading the answer it is meant to predict.',
        'Would seeing the next training token make prediction unfairly easy?','Read one matrix row as a receiver and its permitted source columns.',
        'For next-token training, the input at a later position can reveal the target for an earlier one. The image label is supplied separately; another observed patch does not reveal a hidden future input. The mask follows the task. Autoregressive image models can use causal masks, while text classification encoders can use full attention.')

    body=crop(30,90,300,200,9,'receiver')+t(180,365,'dark receiver crop',27,'c-q','middle')
    body+=g(crop(690,60,220,147,5,'face')+t(945,135,'face crop',27,'c-k'),1)
    body+=g(crop(690,255,220,147,0,'branches')+t(945,325,'branches',27,'c-k'),2)
    body+=g(arrow(345,160,660,125,'c-q')+arrow(345,245,660,320,'c-q'),3)
    add('qkv-photo-question','Which part of the picture could help this dark crop?',body,'Imagine borrowing nearby face evidence to interpret a patch of dark fur.',
        'Would the face crop or the branches help more with this particular ambiguity?','Keep the receiver fixed and point to two possible sources.',
        'This is a human interpretation of actual crops from the opening photograph, not a claim about a specific learned head. Early patch vectors are projections of pixels. Later vectors can represent more abstract context. Like the bank example in Part II, the same local input can be easier to interpret when it receives relevant context.')
    body=label(40,170,'one image row',width=260)
    for i,(name,color,desc) in enumerate([('Q','c-q','what information to seek'),('K','c-k','how this row can be matched'),('V','c-v','what information to send')]):
        y=65+150*i
        body+=g(arrow(320,200,465,y+25,color)+label(500,y,name,color,90)+t(640,y+39,desc,28,color),i+1)
    add('qkv-three-roles','Each row makes a query, a key and a value',body,'Q and K decide the weights. V supplies the message that gets weighted.',
        'Can the same patch receive information and also send information?','Trace all three projections from the same row.',
        'Numerically, Q=E W_Q, K=E W_K and V=E W_V for the current input to attention (the normalized rows in our full pre-LN block). One receiver’s query is compared with every source key. Its weighted sum uses those sources’ value vectors. These roles match the text-attention calculation exactly. The projections are learned, and their coordinates need not have human-readable names.')
    # A tiny separate example with d_k=1 makes the key/value distinction visible.
    body=t(35,45,'Constructed example · one query/key coordinate',25,'ink-2')+t(35,115,'receiver q = 1',35,'c-q')
    body+=crop(40,190,180,120,5,'source-a')+crop(615,190,180,120,0,'source-b')
    body+=g(t(250,230,'k_face = ln 3',32,'c-k')+t(825,230,'k_branches = 0',29,'c-k'),1)
    body+=g(t(250,310,'score = ln 3',30,'c-a')+t(825,310,'score = 0',30,'c-a'),2)
    body+=g(t(250,410,'weight = 3/4',33,'c-a')+t(825,410,'weight = 1/4',33,'c-a'),3)
    add('qkv-match-numbers','How could the face crop get more weight?',body,'Exponentiate the two scores: 3 and 1. Their shares are ¾ and ¼.',
        'Where does the factor 3 come from?','Multiply q by each key, then normalize the two exponentials.',
        'The scalar keys are hand-chosen to illustrate routing. They are not measured features of these crops. Here d_k=1, so scaling by √d_k changes nothing. The values will have two coordinates; query/key width and value width do not have to match. Our main worksheet later uses two coordinates for both.')
    body=t(35,55,'Chosen value coordinates: [fur evidence, foliage evidence]',29,'c-v')
    body+=crop(35,110,165,110,5,'value-a')+t(235,150,'v_face = [2,0]',32,'c-v')+t(730,150,'¾ × [2,0]',34,'c-a')
    body+=crop(35,255,165,110,0,'value-b')+t(235,295,'v_branches = [0,2]',32,'c-v')+t(730,295,'¼ × [0,2]',34,'c-a')
    body+=g(t(490,420,'message = [1.5, 0.5]',38,'c-v'),1)
    add('qkv-read-numbers','What do we receive after choosing those weights?',body,'Multiply each source’s value by its weight, then add the two vectors.',
        'Does choosing the face with ¾ weight discard the other source?','Calculate both contributions before adding them.',
        'The evidence names and value vectors are invented teaching coordinates, not neuron interpretations from the pretrained model. The message is ¾[2,0]+¼[0,2]=[1.5,0.5]. Attention returns a vector of information, rather than a source index or a class label. The same weighted-sum operation supplied context to a token in Part II.')
    body=t(35,65,'Keep q = 1 and both values fixed.',31,'ink-2')
    body+=t(35,165,'keys: [ln 3, 0] → [0, ln 3]',36,'c-k')
    body+=g(t(35,275,'weights: [¾, ¼] → [¼, ¾]',36,'c-a'),1)
    body+=g(t(35,390,'message: [1.5, 0.5] → [0.5, 1.5]',36,'c-v'),2)
    add('qkv-change-key','Change only the keys. What happens?',body,'The values stay available, but their shares in the message change.',
        'Which source now gets three times the weight of the other?','Swap the two keys and recompute the mixture.',
        'This controlled intervention changes K while holding Q and V fixed. In an actual network, changing a source embedding can affect all of its projections. We separate them here to understand their different jobs.')
    body=t(35,65,'Keep q, keys and weights [¾, ¼] fixed.',31,'ink-2')
    body+=t(35,165,'v_face: [2,0] → [4,0]',37,'c-v')+t(35,250,'v_branches stays [0,2]',32,'c-v')
    body+=g(t(35,385,'message: [1.5, 0.5] → [3.0, 0.5]',36,'c-v'),1)
    add('qkv-change-value','Change only a value. What happens?',body,'The attention weights stay the same, but the received information changes.',
        'Would an attention heatmap show this change?', 'Keep the two weights visible while doubling the first value coordinate.',
        'The attention matrix depends on Q and K, so it is unchanged. Its product with V changes. This explains why an attention heatmap alone cannot tell us everything about the message or the final class prediction. Later we inspect real maps and separately measure prediction changes under occlusion.')
    body=image(35,90,300,200)+g(arrow(365,190,450,190)+label(480,160,'image rows',width=225),1)
    body+=g(arrow(735,190,825,190,'c-q')+t(855,202,'Q = E W_Q',33,'c-q'),2)
    body+=g(t(35,385,'The “question” is a vector produced by the model.',33),3)
    add('qkv-no-prompt','Do we type a question into this classifier?',body,'No text prompt is needed. The image rows generate their own Q, K and V.',
        'Where does the query come from if nobody types a question?','Follow the numerical projection from the input rows.',
        '<a href="https://cseweb.ucsd.edu/~mkchandraker/classes/CSE252D/Spring2024/Lectures/lec02_visiontransformers.pdf">UCSD CSE252D (slides 18–25)</a> and <a href="https://visionbook.mit.edu/transformers.html">MIT VisionBook (§26.6)</a> use an external question to motivate what a query does, then introduce self-attention. Here all three projections come from the same image-token sequence, including CLS. A text query attending to image keys and values would be cross-attention, which belongs to our later vision–language discussion. The plain-language questions on these slides are an explanatory analogy.')

    body=t(35,65,'Four patch rows',31,'c-e')
    for i in range(4):body+=label(35+i*275,125,f'P{i+1}',width=180)
    body+=g(label(35,285,'CLS',color='c-q',width=180)+t(260,325,'one learned vector, copied for every image',31,'c-q'),1)
    add('cls-start','Where does CLS come from?',body,'CLS is an extra learned input vector with the same width as a patch row.',
        'Which image pixels were used to create CLS?','Point to the separate parameter, then count five rows entering attention.',
        'CLS is not extracted from image pixels. It is a model parameter trained along with the projection weights; its position embedding is also learned in the usual ViT. At the start of a forward pass, the same vector is prepended to every image. Its name does not contain the correct class, and the target label is never inserted into the input.')
    body=image(30,35,250,167)+image(30,235,150,200,CAT)
    for i,y in enumerate([95,315]):
        body+=label(320,y,'same initial CLS','c-q',235)+arrow(580,y+30,650,y+30)
        body+=g(label(680,y,'reads this image','c-v',230)+arrow(935,y+30,995,y+30),1)
        body+=g(t(1020,y+40,'c_dog' if i==0 else 'c_cat',30,'c-e'),2)
    add('cls-two-images','How can the same starting CLS describe different pictures?',body,'Its starting vector is shared. The keys and values it reads come from the current image.',
        'Does a shared first-layer query force the output to be identical?','Follow each image through its own source keys, values and weighted sum.',
        'In deterministic evaluation, the initial CLS input and its first-block query are identical across these images. The patch keys and values differ, so the updated CLS can differ. Later CLS queries can already depend on the image. This is why a learned constant input can become an image-specific summary.')
    body=path(85,['image + CLS','blocks','class scores','class loss'],['c-e','c-v','c-a','c-a'])
    body+=g(arrow(1055,190,1055,310,'c-a')+arrow(1055,310,150,310,'c-a')+arrow(150,310,150,175,'c-a'),1)
    body+=g(t(35,410,'Backpropagation trains CLS and the weights that it uses.',31,'c-d'),2)
    add('cls-learns','Who teaches CLS what information to collect?',body,'The image-classification loss sends gradients through the classifier and the attention blocks.',
        'Is there a separate label telling CLS which patch to read?', 'Trace the loss backward through the computational path.',
        'The ordinary supervised class loss is enough to train the readout, initial CLS vector and attention parameters jointly. There is no required supervision on an attention map. The learned representation is useful insofar as it helps reduce the training objective and generalizes to new data.')
    body=t(35,55,'Suppose the final patch rows are:',30)
    for i,value in enumerate(['[2,0]','[0,2]','[1,1]','[1,1]']):
        body+=label(35+i*275,120,value,'c-e',200)
    body+=g(t(35,285,'mean = ([2,0] + [0,2] + [1,1] + [1,1]) / 4',33,'c-v'),1)
    body+=g(t(35,405,'= [1,1] → learned classifier → class scores',34,'c-a'),2)
    add('pooling-example','Could we classify the image without CLS?',body,'Yes. One option is to average the final patch rows and train a classifier on that vector.',
        'Can you calculate the two coordinates of the mean?', 'Add each column and divide by the number of patch rows.',
        'These are chosen final contextual rows, not raw pixels. Pooling is an alternative readout: build and train the network with that choice. Removing CLS from an already trained CLS-based checkpoint changes its computation and is not a guaranteed drop-in replacement. The task requires an image-level output; it does not require this particular summary token.')
    body=t(35,60,'CLS readout',33,'c-q')+label(35,140,'updated CLS','c-q',260)+arrow(325,170,460,170)+label(490,140,'class scores','c-a',245)
    body+=g(t(35,270,'Mean readout',33,'c-e')+label(35,330,'updated patches','c-e',260)+arrow(325,360,460,360)+label(490,330,'average','c-v',245)+arrow(765,360,840,360)+label(870,330,'class scores','c-a',245),1)
    add('readout-choice','So why use CLS in our ViT?',body,'It gives the model a dedicated row whose final representation is trained for the image label.',
        'What must both of these paths produce before the class head?', 'Follow each route to one fixed-width vector.',
        'CLS participates in the attention blocks, so it can gather a content-dependent summary at every layer. Mean pooling combines contextual patch rows at the end. Both are viable design choices. Our pretrained checkpoint and full small model use CLS, so we follow it consistently through the worked examples. Neither option is guaranteed to be best for every dataset.')

    def grid(x,y):
        out=''
        for r in range(4):
            out+=t(x-20,y+60*r+32,'P'+str(r+1),25,'c-e','end')
            for c in range(4):out+=rect(x+c*75,y+r*60,65,45,'c-e','t-e')
        return out
    body=grid(90,80)+g(arrow(470,103,735,215,'c-v')+arrow(470,163,735,215,'c-v')+arrow(470,223,735,215,'c-v')+arrow(470,283,735,215,'c-v')+label(765,185,'one new row','c-v',275),1)
    body+=g(t(35,420,'Attention: combine information across source rows.',32,'c-v'),2)
    add('mix-across-rows','Which operation lets one patch borrow from another?',body,'Attention mixes source value rows using weights chosen for the receiver.',
        'Which arrows cross from one patch to another?', 'Point to several source rows contributing to one destination row.',
        'This returns to the matrix view used in Part II: A V combines source rows. The learned Q/K/V projections first modify coordinates within each row; the attention-weighted sum is the part that moves information between rows.')
    body=grid(90,80)+g(arrow(470,163,670,163,'c-d')+label(700,133,'same row MLP','c-d',300),1)
    body+=g(t(35,380,'Each row is transformed separately.',32,'c-d')+t(35,430,'The MLP weights are shared across rows.',28,'ink-2'),2)
    add('mix-within-row','What does the MLP change?',body,'The MLP combines coordinates within each row and applies a nonlinearity.',
        'Does this MLP directly read a neighbouring patch row?', 'Keep one row highlighted as it passes through the two linear layers.',
        'The same positionwise MLP is applied independently to every row. Its input may already contain information from other patches, because attention ran first. This separation between mixing rows and modifying a row helps students read the complete block without treating it as one unexplained box.')

    insert_after={
      'photo-folder':['task-next-token','task-image-label','task-side-by-side'],
      'why-cls':['cls-start','cls-two-images'],
      'image-mask':['task-mask-reason'],
      'qkv-roles':['qkv-photo-question','qkv-three-roles','qkv-match-numbers','qkv-read-numbers','qkv-change-key','qkv-change-value','qkv-no-prompt'],
      's04-experiment':['cls-learns','pooling-example','readout-choice'],
      's05-block':['mix-across-rows','mix-within-row']}
    revised=[]
    for title,frames in sections:
        out=[]
        for html in frames:
            out.append(html)
            key=re.search(r'class="frame[^\"]*" id="([^\"]+)"',html).group(1)
            out.extend(slides[k] for k in insert_after.get(key,[]))
        revised.append((title,out))
    assert sum(map(len,insert_after.values()))==len(slides)
    return revised
