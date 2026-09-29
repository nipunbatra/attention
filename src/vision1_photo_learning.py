"""Continue the saved photograph forward pass into loss and learning."""
import json
import math
from vision1_focus_common import Figures


def build(b):
    f = Figures(b)
    t, g, box, arrow, line = f.t, f.g, f.box, f.arrow, f.line
    trace = json.loads((b['ASSETS']/'classifier-readout-trace.json').read_text())
    p = trace['selected_classes'][0]['probability']
    loss = -math.log(p)
    provenance = ('The probability is from the saved dog forward pass. Treat this as an illustrative labelled training example; '
                  'we do not claim that this photo was in the checkpoint training set. No training is run. ')

    body = f.image(35,76,220,220,f.photo)+t(145,337,'Label: Newfoundland',23,'ink','middle')
    body += arrow(264,182,329,182)+box(345,137,280,['Same saved forward pass',f'p(label) = {p:.5f}'],size=22)
    body += g(arrow(632,182,704,182)+box(720,137,385,['Cross-entropy: −log p(label)',f'loss = {loss:.4f}'],'c-a',size=25),1)
    body += g(t(345,306,'Known label enters here, after the prediction.',27,'c-a')
              +t(345,367,'Higher probability for the correct class → lower loss.',25),1)
    f.add('photo-label-loss','Attach a label loss to the same dog prediction',body,
        'For a labelled Newfoundland example, use the probability of Newfoundland to compute cross-entropy. This continues the saved forward pass. Training will use the loss to adjust parameters across many labelled images.',
        'Where does the known label enter?', 'At the loss. The pixels, patch inputs and CLS never receive the answer as an input.',
        provenance+'The full softmax still contains 1,000 classes. The loss uses the probability assigned to the known class. '
        'Use unrounded saved probabilities to compute the displayed loss. In PyTorch, pass logits directly to cross_entropy. '
        'A small loss on one image does not measure generalization.')

    body = t(35,45,'First reverse the class head: 192 features → 1,000 scores.',29)
    body += box(35,112,245,['Final CLS','192 features'],'c-q')+arrow(286,150,360,150)
    body += box(375,112,270,['Linear(192, 1000)','weights + biases'])+arrow(651,150,715,150)
    body += box(730,112,395,['Class scores → label loss','1,000 → 1'],'c-a')
    body += g(arrow(890,269,535,269,'c-a')+arrow(515,269,158,269,'c-a'),1)
    body += g(t(35,325,'Each score gets gradient: p(class) − 1[class is the label]',29,'c-a')
              +t(35,383,f'Newfoundland: {p:.4f} − 1 = {p-1:.4f} → increase its score',28,'c-a'),1)
    f.add('photo-backward-head','The class loss reaches the final CLS features',body,
        'Cross-entropy gives a gradient for every class score. The linear head passes gradients to its weights, biases and the final CLS features. A negative gradient asks a small gradient-descent step to increase that score.',
        'Does the loss only train the last layer?', 'No. The class head also passes a 192-coordinate gradient to the representation it reads.',
        provenance+'For one image, dlogits = p − one_hot(y). If h is a row and logits=hW+b, '
        'dW=hᵀdlogits, db=dlogits, and dh=dlogits Wᵀ. Final LayerNorm lies between the last block and this readout. '
        'Only CLS is read directly, but earlier attention connects it to the patch rows. Other final patch rows do not receive a direct classifier gradient.')

    body = t(35,40,'Reverse the stack: block 12 → block 11 → … → block 1',28,'c-a')
    body += box(35,135,115,'E')+arrow(157,173,192,173)
    body += box(205,135,240,['LayerNorm','Attention'])+arrow(451,173,490,173)
    body += box(505,135,66,'+')+arrow(577,173,625,173)
    body += box(642,135,240,['LayerNorm','MLP'])+arrow(888,173,930,173)
    body += box(945,135,66,'+')+arrow(1017,173,1050,173)+t(1085,184,'out',28,'c-e','middle')
    body += line(92,131,92,80,'c-e')+line(92,80,538,80,'c-e')+arrow(538,80,538,131,'c-e')
    body += line(602,173,602,80,'c-e')+line(602,80,978,80,'c-e')+arrow(978,80,978,131,'c-e')
    body += g(arrow(1090,257,775,257,'c-a')+arrow(775,257,330,257,'c-a')+arrow(330,257,80,257,'c-a'),1)
    body += g(t(35,337,'At +, the incoming gradient travels down both input paths.',28,'c-a')
              +t(35,398,'Reverse each learned branch; add gradients where paths meet.',28),1)
    f.add('photo-backward-block','Both residual paths carry the learning signal',body,
        'Start at the final readout and reverse the 12 blocks. At each residual addition, both inputs receive the incoming gradient. The attention and MLP branches learn while the skip paths also carry gradients upstream.',
        'Does the skip path stop the attention branch from learning?', 'Both paths receive gradients. When they reach the same earlier activation, add their contributions.',
        'Forward is U=E+Attention(LN(E)), then Y=U+MLP(LN(U)). Reverse Y first through the MLP and its residual, '
        'then reverse U through attention and its residual. Backward through the MLP follows the reverse dependency order of '
        'Linear → GELU → Linear. LayerNorm is differentiable; its scale and bias also receive gradients. '
        'Every block has separate parameters. The shapes remain 197×192 for block input and output activations and their gradients.')

    body = t(35,40,'Zoom into one head and the CLS message: h = a V',30,'c-v')
    body += box(780,95,340,['Gradient arriving at h','1 × 64'],'c-a')
    body += arrow(774,133,575,133,'c-a')+box(340,95,220,['Weighted sum','a V'],'c-v')
    body += g(arrow(400,177,240,256,'c-a')+box(35,271,390,['Value route','dV = aᵀ dh  ·  197 × 64'],'c-v',size=24),1)
    body += g(arrow(500,177,765,256,'c-a')+box(565,271,555,['Weight route: da = dh Vᵀ','softmax → scores → Q and K'],'c-q',size=24),1)
    body += g(t(35,412,'Patch rows learn because their keys and values helped CLS make the prediction.',26),2)
    f.add('photo-backward-attention','The loss teaches both what to read and what to send',body,
        'A message depends on attention weights and source values, so gradients follow both routes. Reverse softmax and query–key matching to reach Q and K. Reverse the value projection to reach V’s input rows and weights.',
        'How can patches learn when the classifier only reads CLS?', 'CLS used the source keys and values. Reverse those dependencies into the patch representations; earlier blocks also let patch queries affect the final summary.',
        'One CLS query has a with shape 1×197, V with shape 197×64 and h=aV with shape 1×64. '
        'With incoming gradient dh, dV=aᵀdh and da=dhVᵀ. For s=qKᵀ/8, '
        'ds=a⊙(da−sum(da⊙a)), dq=dsK/8, dK=dsᵀq/8. These are row-vector formulas. '
        'All query rows and all three heads follow the same chain rule, accumulating contributions. '
        'For any projection XW+b, dW=XᵀdY and dX=dYWᵀ. Reverse the output projection and split concatenated features by head before this calculation. '
        'Gradients for patch queries in the last block can be zero if their output rows never reach the CLS readout; earlier blocks can use those rows.')

    body = t(35,40,'After reversing block 1, gradients reach the prepared input rows.',27)
    for j,(name,c) in enumerate([('P1','c-e'),('P2','c-e'),('…','ink-2'),('P196','c-e')]):
        x=35+j*178
        body += box(x,90,145,[name,'row gradient'],c,size=20)
        body += arrow(x+72,174,372,258,'c-a')
    body += box(182,273,390,['Shared patch layer','sum all patch contributions'],'c-a',size=24)
    body += box(805,90,300,['Position table','197 × 192 parameters'],'c-q',size=24)
    body += box(805,273,300,['Starting CLS','192 parameters'],'c-q',size=24)
    body += g(t(35,413,'One shared patch layer; 197 position rows; one starting CLS. The pixels stay fixed.',25),1)
    f.add('photo-backward-inputs','Accumulate gradients into the shared input parameters',body,
        'Every patch uses the same projection, so its parameter gradients add across patches and images. Each position row and the shared starting CLS also receives gradients. Ordinary training updates model parameters while keeping the input photographs fixed.',
        'Do 196 patches require 196 separate patch layers?', 'No. Sum their contributions into one projection. The batch also shares these parameters.',
        'For patch rows e_i=x_i W_patch+b_patch+p_i, dW_patch=sum_i x_iᵀ de_i and db_patch=sum_i de_i. '
        'Each p_i receives de_i. The starting CLS receives the gradient of row zero, as does its position p_0. '
        'For a mean-reduced batch loss, these gradients include the batch averaging factor. Gradients can be computed with respect to pixels, '
        'but the optimizer in this classifier training procedure is given model parameters, not image pixels.')

    body = t(35,47,'One illustrative SGD update of a class bias',29)
    body += box(35,122,270,['Stored bias b','Newfoundland'])+arrow(313,160,377,160)
    body += box(395,122,325,[f'Gradient = {p-1:.4f}','learning rate = 0.1'],'c-a')+arrow(730,160,792,160)
    body += box(810,122,315,['New bias',f'b + {0.1*(1-p):.5f}'],'c-e')
    body += g(t(35,285,'backward(): compute gradients',31,'c-a')
              +t(35,346,'step(): update trainable parameters using those gradients',29,'c-e')
              +t(35,410,'Then run a new forward pass. The next activations will be different.',26),1)
    f.add('photo-optimizer-step','Only the optimizer changes the model parameters',body,
        'For a simple SGD step, subtract learning rate times gradient. This class-bias example would raise the Newfoundland score. In full training, the optimizer updates every selected parameter using gradients from the batch.',
        'Do weights change when we call backward?', 'Backward fills gradients. The optimizer step changes weights; zero_grad clears previous accumulated gradients before the next batch.',
        provenance+'The displayed arithmetic uses db=p(label)−1 and learning rate 0.1. It describes one SGD coordinate update, '
        'not an executed optimizer step or a full-model training result. Adam-like optimizers also maintain state, so their exact update formula differs. '
        'An improvement on a training example is not evidence of test accuracy.')
    return list(f.frames.values())
