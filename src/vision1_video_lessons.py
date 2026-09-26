"""Small implementation lessons prompted by the reviewed Vizuara videos."""
import re
import json
from vision1_video_examples import examples


def augment(b,sections):
    globals().update({k:b[k] for k in ['frame','t','g','line','arrow','rect','pixels','ASSETS']})
    result=examples()
    (ASSETS/'video-examples.json').write_text(json.dumps(result,indent=2)+'\n')
    new={}
    def add(key,title,body,caption,question,point,prose):
        new[key]=frame(key,title,body,caption,question+'\n'+point,prose)
    def numbers(x,y,rows,color='c-e',step=68):
        body=''
        for r,row in enumerate(rows):
            for c,value in enumerate(row):
                body+=rect(x+c*step,y+r*step,step,step,'line','transparent',0)
                body+=t(x+(c+.5)*step,y+(r+.67)*step,str(value),31,color,'middle')
        return body
    video='<a href="https://www.youtube.com/watch?v=ZRo74xnN2SI&t=3085s">Vizuara’s implementation lecture, patch embedding</a>'
    body=pixels(35,95,[[1,1],[0,0]],85,True)+t(120,55,'one patch',29,'c-e','middle')
    body+=t(310,195,'×',39)+numbers(400,95,[[1,1],[-1,-1]],'c-k',85)+t(485,55,'one filter',29,'c-k','middle')
    body+=g(arrow(625,180,750,180)+t(800,195,'2',50,'c-e'),1)
    body+=g(t(35,365,'1·1 + 1·1 + 0·(−1) + 0·(−1) = 2',36,'c-e'),2)
    add('conv-one-patch','Why does our ViT code use Conv2d?',body,'One filter computes the same dot product as one column of the patch-projection matrix.',
        'Have we already calculated this number using a flattened patch?', 'Match the four pixel–weight pairs to the top-minus-bottom column in our earlier edge example.',
        'The example returns to the chosen edge projection from section 3. A 2×2 grayscale patch has four inputs. A filter with weights [1,1,−1,−1] and zero bias gives 2 for [1,1,0,0]. Conv2d stores this linear map as a spatial kernel. '+video+' motivates checking the relation between the picture and the code.')
    body=pixels(35,80,result['patch_projection']['image'],70,True,True)
    body+=rect(35,80,140,140,'c-q','transparent',0)+g(arrow(190,55,310,55,'c-q')+t(525,85,'kernel = 2 × 2',34,'c-k')+t(525,165,'stride = 2',34,'c-q'),1)
    body+=g(t(525,270,'four separate 2 × 2 patches',32,'c-e'),2)
    body+=g(t(525,375,'same filter at every location',31),3)
    add('conv-stride','How far should the filter move?',body,'A stride equal to the patch width gives non-overlapping patches.',
        'What would happen if the stride were one?', 'Move the 2×2 outline two columns, then two rows; keep the weights unchanged.',
        'This is a separate 4×4 edge-pattern illustration. With kernel size 2, stride 2 and no padding, the filter visits four patch locations. Stride 1 would produce overlapping windows. The learned projection shares weights over locations; global attention is the later operation that exchanges information between patch rows.')
    body=t(35,55,'two filters → two coordinates per patch',34,'c-e')
    for i,(name,row) in enumerate(zip(['top-left','top-right','bottom-left','bottom-right'],result['patch_projection']['output'])):
        y=140+70*i;body+=t(35,y,name,29,'ink-2')+t(400,y,str([int(v) for v in row]),34,'c-e')
    body+=g(t(700,145,'2 × (4 weights + 1 bias)',29,'c-k')+t(700,215,'= 10 learned parameters',29,'c-k'),1)
    body+=g(t(700,315,'[B,2,2,2] → [B,4,2]',31,'c-e')+t(700,390,'channels → row coordinates',26,'ink-2'),2)
    add('conv-trainable','Does this layer merely cut up the image?',body,'Conv2d both selects each patch and applies a learned linear projection.',
        'Where are the trainable numbers in this operation?', 'Count four weights and one bias per filter, then read each output as a two-coordinate row.',
        'We set two filters by hand for this worked example: top-minus-bottom and left-minus-right, both with zero bias. Their parameters remain trainable. The executed lab compares Conv2d with unfold-plus-linear, verifies all four outputs, and checks gradients. Our actual small ViT has 16 output channels, so its grayscale 2×2 patch layer has 16×4+16=80 parameters. See <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html">PyTorch Conv2d</a>.')
    body=t(35,60,'image A',33,'c-e')+t(670,60,'image B',33,'c-v')
    for i,(left,right) in enumerate(zip(result['batch_axis']['input'][0],result['batch_axis']['input'][1])):
        y=145+92*i;body+=t(35,y,'P'+str(i+1),28,'ink-2')+t(130,y,str([int(v) for v in left]),36,'c-e')
        body+=t(670,y,'P'+str(i+1),28,'ink-2')+t(765,y,str([int(v) for v in right]),36,'c-v')
    body+=g(t(35,425,'One batch: B = 2 images, N = 3 rows each, D = 2.',30),1)
    add('batch-boundary','Can a patch in image A read image B?',body,'Ordinary image self-attention gives each image its own source rows.',
        'Should putting a second photograph in the batch change the first image’s message?', 'Keep the two groups of rows separate; identify the image, token and coordinate axes.',
        'These are chosen numeric patch rows for a unit-sized example, with no CLS. A batch groups independent examples for efficient computation; it does not add their tokens to one shared sequence. The coding lecture’s <a href="https://www.youtube.com/watch?v=ZRo74xnN2SI&t=6346s">batch_first debugging segment</a> motivates this check.')
    body=t(35,55,'Input shape: [2 images, 3 rows, 2 coordinates]',33)
    body+=t(35,155,'batch_first=True',34,'c-e')+t(635,155,'wrong axis interpretation',29,'c-a')
    body+=g(t(35,255,'read A’s three rows',31,'c-e')+t(635,255,'read P1 in A and B',31,'c-a'),1)
    body+=g(t(35,345,'( [2,0]+[4,0]+[6,0] ) / 3',28,'c-v')+t(635,345,'( [2,0]+[0,2] ) / 2',29,'c-v'),2)
    body+=g(t(35,425,'= [4,0]',35,'c-e')+t(635,425,'= [1,1]',35,'c-a'),3)
    add('batch-axis','A tensor can have the right shape and the wrong meaning',body,'For [B,N,D] input, set batch_first=True so attention runs across N.',
        'Which axis should softmax normalize over in each image?', 'Use uniform attention to calculate the first output row under both interpretations.',
        'Here Q=K=0, so all allowed sources have equal weight. V and W_O are identity maps. PyTorch MultiheadAttention defaults to [sequence,batch,features]. Passing our [2,3,2] tensor without batch_first=True makes the image axis act as the sequence axis. The output shape still looks valid. The notebook reproduces both calculations; see <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html">the documented input layouts</a>.')
    body=t(35,60,'First output row for image A',33)
    body+=t(460,145,'A alone',30,'ink-2')+t(820,145,'A beside B',30,'ink-2')
    body+=g(t(35,260,'correct layout',30,'c-e')+t(460,260,'[4,0]',37,'c-e')+t(820,260,'[4,0]',37,'c-e'),1)
    body+=g(t(35,360,'wrong layout',30,'c-a')+t(460,360,'[2,0]',37,'c-a')+t(820,360,'[1,1]',37,'c-a'),2)
    add('batch-check','A quick check before training',body,'Run the same image alone and beside another image. Its attention output should agree.',
        'Why does the wrong version change when B is added?', 'Compare the first patch’s output under both layouts before looking at any accuracy curve.',
        'This deterministic attention example disables dropout and uses fixed weights. The lab checks all of A’s output rows, not only a class score. The correct version passes and the deliberately misconfigured version fails. This is one useful check, not a proof of the whole model: a broken CLS readout can be insensitive to every patch and still appear batch-independent. Check source axes and meaningful inputs too.')
    after={'code-patch':['conv-one-patch','conv-stride','conv-trainable'],
           'code-attention':['batch-boundary','batch-axis','batch-check']}
    revised=[]
    for title,frames in sections:
        out=[]
        for html in frames:
            out.append(html)
            key=re.search(r'class="frame[^\"]*" id="([^\"]+)"',html).group(1)
            out.extend(new[k] for k in after.get(key,[]))
        revised.append((title,out))
    return revised
