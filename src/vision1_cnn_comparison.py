"""Three visual comparisons of conventional CNNs and a plain global ViT."""
from vision1_focus_common import Figures


def build(b, existing):
    f=Figures(b)
    t,g,box,line,arrow=f.t,f.g,f.box,f.line,f.arrow
    # Retain the useful receptive-field diagram, then explain its assumptions.
    frames=[existing['cnn-receptive-field']]
    body=t(35,37,'Inductive bias: structure the architecture supplies before learning.',29)
    for j,shift in enumerate([0,2]):
        x=60+j*555
        body+=t(x+205,89,'Edge here' if j==0 else 'Same edge, shifted',27,'c-e','middle')
        for r in range(5):
            for c in range(7):
                active=c in [1+shift,2+shift] and r in [1,2,3]
                body+=f.rect(x+c*29,119+r*29,29,29,'line','ink' if active else 'card',0)
        body+=f.rect(x+shift*29,148,87,87,'c-v','transparent',0)
        body+=arrow(x+214,192,x+280,192,'c-v')
        body+=box(x+292,145,180,['Same kernel','same response'],'c-v',size=20)
        body+=t(x+102,299,'3 × 3 local view',23,'c-v','middle')
    body+=g(t(35,352,'Shared weights let a learned detector work at other image locations.',29,'c-v')
            +t(35,405,'Shift the pattern → shift the feature map. This is translation equivariance.',26),1)
    f.add('cnn-inductive-bias','A CNN reuses a local detector across the image',body,
        'Local connections and shared kernels encode assumptions about images. With stride one away from boundaries, shifting the input shifts its convolutional feature map. The model can reuse a detector instead of learning it separately at each location.',
        'Must we learn a second kernel when the edge moves?',
        'No. The same weights slide over the grid. Its response moves with the matching pattern; the class prediction need not change just because the feature location changes.',
        'Inductive bias means preferences introduced by model structure or training, before fitting a particular dataset. '
        'Here the architecture imposes local connectivity and weight sharing. The edge and boxes are schematic, not measured activations. '
        'Equivariance means the output transforms with the input; invariance means the output does not change. '
        'Padding, finite boundaries, stride, pooling, and other operations limit exact translation equivariance. '
        'Convolutional weights are fixed during a forward pass, while activations and the whole nonlinear network response depend on the image. '
        '<a href="https://arxiv.org/html/2010.11929v2#S3.SS1">ViT paper: inductive bias</a>.')
    frames.append(f.frames['cnn-inductive-bias'])

    body=f.image(35,145,140,140,f.photo)
    body+=line(183,216,220,216)+line(220,114,220,320)
    body+=arrow(220,114,267,114)+arrow(220,320,267,320)
    body+=box(282,72,280,['CNN: conv layers','local shared kernels'],'c-v')
    body+=arrow(570,110,625,110,'c-v')+box(640,72,235,['Global pooling','one feature vector'],'c-v',size=22)
    body+=arrow(883,110,930,110,'c-v')+box(947,72,180,['Class head','scores'],'c-v')
    body+=box(282,278,280,['ViT: patch rows + position','global attention blocks'],'c-q',size=22)
    body+=arrow(570,316,625,316,'c-q')+box(640,278,235,['Read CLS','one feature vector'],'c-q',size=22)
    body+=arrow(883,316,930,316,'c-q')+box(947,278,180,['Class head','scores'],'c-q')
    body+=t(282,192,'CNN: nearby pixels interact first; deeper layers build a wider view.',24,'c-v')
    body+=t(282,398,'ViT: each query chooses weights over all patches from the start.',24,'c-q')
    f.add('cnn-vit-design','How CNNs and ViTs build an image representation',body,
        'A conventional CNN builds context through local shared kernels. A plain ViT learns global mixing between patch rows and uses position signals. Both produce one image representation, use class-label loss, and can learn from pretraining.',
        'Does ViT have no image-related bias at all?',
        'It still groups nearby pixels into patches, shares the patch projection, and uses a position representation. It imposes less local spatial structure inside its attention blocks than the conventional CNN shown here.',
        'Compare conventional convolutional classifiers with the plain global-attention ViT in this lecture; hybrids and local-attention variants also exist. '
        'The CNN encodes locality and spatially shared kernels. In ViT, learned query–key comparisons make mixing weights depend on the current image. '
        'ViT still has inductive biases: patch construction, a shared projection and a chosen positional representation. '
        'Locality can make learning image structure more data-efficient in some settings; flexible global mixing can benefit from larger training sets and pretraining. '
        'Architecture alone does not rank accuracy: compare data, augmentation, pretraining and compute. '
        'Global pooling and a class head do not guarantee exact translation invariance for a finite practical CNN. '
        '<a href="https://arxiv.org/html/2010.11929v2#S3.SS1">Original ViT architecture and inductive-bias discussion</a>.')
    frames.append(f.frames['cnn-vit-design'])
    return frames
