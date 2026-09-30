"""Motivate transfer learning before replacing the pretrained class head."""
from vision1_focus_common import Figures


def build(b):
    f = Figures(b)
    t, box, arrow, g = f.t, f.box, f.arrow, f.g
    reference = '<a href="https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k">Checkpoint model card and training history</a>.'
    body = t(35, 36, 'The checkpoint already learned from labelled photographs.', 29)
    body += box(35, 85, 425, ['Pretrain on ImageNet-21k', 'then fine-tune on ImageNet-1k'], size=27, h=95)
    body += arrow(475, 134, 565, 134)
    body += box(582, 85, 543, ['Current task: choose among 1,000 labels', 'animal breeds, objects and other categories'], size=25, h=95)
    body += f.image(35, 236, 150, 150, f.photo)
    body += arrow(199, 311, 275, 311)
    body += box(292, 267, 328, ['ViT encoder', '192 final CLS features'], size=25, h=89)
    body += arrow(635, 311, 692, 311)
    body += box(708, 267, 417, ['Trained 1,000-class head', 'Top label: Newfoundland'], 'c-v', size=26, h=89)
    body += t(292, 420, 'Our saved dog prediction used this existing task and head.', 27, 'c-v')
    f.add('pets-original-task', 'What was this model trained to predict?', body,
          'The checkpoint was pretrained on ImageNet-21k and fine-tuned on ImageNet-1k. Its current head predicts 1,000 categories. Newfoundland is one of those labels.',
          'Did our earlier dog example train a pet classifier?',
          'No. It ran inference with the existing ImageNet checkpoint. Its learned encoder and 1,000-class head were already available.',
          'The current checkpoint is vit_tiny_patch16_224.augreg_in21k_ft_in1k. '+reference)

    body = t(35, 37, 'Now our application asks a simpler question: dog or cat?', 29)
    for i, (photo, old, new) in enumerate([(f.photo, 'Newfoundland', 'dog'), (b['CAT'], 'Persian cat', 'cat')]):
        y = 100+i*160
        body += f.image(35, y, 125, 125, photo)
        body += box(225, y+15, 360, ['Old label vocabulary', old], size=27, h=96)
        body += arrow(602, y+63, 700, y+63, 'c-q')
        body += box(720, y+15, 405, ['Our new target', new], 'c-q', size=28, h=96)
    f.add('pets-new-task', 'Same photographs, a different label vocabulary', body,
          'Our pet application groups breeds into two categories. We want two scores for every image, trained and evaluated for this dog/cat task. Start by reusing the encoder’s visual features.',
          'Could we reuse the visual features even though the output labels changed?',
          'Yes. The encoder already represents visual patterns. A new head can learn how those features separate the two target classes.',
          'This lecture chooses a learned two-output head. One could also define a baseline by grouping appropriate ImageNet probabilities, '
          'but that mapping and its handling of other classes would need evaluation. Neither approach inherits a Pets accuracy claim from two example predictions. '
          'The Oxford-IIIT Pet dataset supplies dog/cat species labels as well as 37 breed categories; the chosen label task determines the output head size.')

    def sketch(x, y, animal):
        # Original line diagrams of a possible new input domain, not data samples.
        if animal == 'dog':
            path = 'M45 46 Q18 4 12 63 Q10 110 47 93 M119 46 Q145 4 152 63 Q155 110 121 93 M43 47 Q83 18 123 47 L126 107 Q121 145 83 149 Q42 145 39 107 Z M65 82 h1 M103 82 h1 M73 106 Q83 99 93 106 L83 117 Z M83 117 v10 M67 125 Q83 140 100 125'
        else:
            path = 'M35 66 L31 17 L65 45 Q84 39 101 45 L136 17 L132 69 Q151 119 115 141 Q82 159 49 141 Q13 119 35 66 Z M57 85 h1 M110 85 h1 M76 106 L91 106 L84 115 Z M84 115 v12 M69 126 Q84 136 99 126 M57 111 L12 102 M56 123 L12 131 M113 111 L156 102 M113 123 L156 131'
        return f'<g transform="translate({x} {y})" fill="none" stroke="var(--ink-2)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"><path d="{path}"/></g>'

    body = t(35, 34, 'A second kind of change: same labels, different-looking inputs.', 28)
    body += t(184, 89, 'Pet photographs', 27, 'c-e', 'middle')
    body += f.image(35, 121, 145, 145, f.photo)+f.image(203, 121, 145, 145, b['CAT'])
    body += arrow(389, 193, 511, 193, 'c-q')
    body += t(487, 119, 'New domain', 24, 'c-q', 'middle')
    body += sketch(568, 111, 'dog')+sketch(806, 111, 'cat')
    body += t(771, 89, 'Sketches · schematic examples', 27, 'c-q', 'middle')
    body += t(643, 297, 'dog', 28, 'c-q', 'middle')+t(887, 297, 'cat', 28, 'c-q', 'middle')
    body += g(t(35, 362, 'Fur texture and colour disappear; outlines become more important.', 28), 1)
    body += g(t(35, 415, 'Keep two outputs. Test on sketches; use labelled sketches to adapt if needed.', 26, 'c-v'), 1)
    f.add('pets-new-domain', 'What if our users supply sketches?', body,
          'The labels can stay dog and cat while the image domain changes. A photo classifier may struggle with sketches. Evaluate on the target domain, then compare head training with encoder fine-tuning.',
          'Would a different-looking image automatically require more output neurons?',
          'No. The two labels are unchanged. The features may need adaptation because sketches remove colour and texture cues.',
          'These line drawings illustrate a hypothetical application; no sketch inference or benchmark was run. '
          'First establish held-out target-domain performance. Use representative labelled sketches for adaptation and choose how much of the encoder to unfreeze using validation. '
          'We continue with the dog/cat photograph task below; sketches show why label changes and domain changes are different reasons to adapt.')

    def status(x, y, frozen):
        if frozen:
            paths=''.join(f'<path transform="rotate({angle})" d="M0 -12 V12 M-4 -10 L0 -6 L4 -10 M-4 10 L0 6 L4 10"/>' for angle in [0,60,120])
            color='c-e'
        else:
            paths='<path d="M0 -15 C0 -6 -10 -3 -10 5 C-10 17 11 17 11 5 C11 -1 7 -6 4 -8 C5 -1 1 0 0 -15 Z"/>'
            color='c-k'
        return (f'<g transform="translate({x} {y})" fill="none" stroke="var(--{color})" '
                f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round">{paths}</g>')

    def network(x, y, mode, show_status=True):
        # Three drawn layers stand for a few of the model's many neurons.
        out=''
        for layer in range(2):
            color='c-k' if mode=='pretrain' or (mode=='adapt' and layer==1) else 'c-e'
            for a in [0,30,60]:
                for z in [0,30,60]:
                    out+=f.line(x+layer*63+9,y+a,x+(layer+1)*63-9,y+z,color,1.5)
        for layer in range(3):
            color='c-k' if mode=='pretrain' or (mode=='adapt' and layer==2) else 'c-e'
            for a in [0,30,60]:
                out+=f'<circle cx="{x+layer*63}" cy="{y+a}" r="9" fill="var(--card)" stroke="var(--{color})" stroke-width="2"/>'
        if show_status:
            out+=status(x+165,y+30,mode=='infer')
        return out

    body=''
    stages=[('1 · Pretrain','Many labelled images','Learn visual features','pretrain'),
            ('2 · Adapt','Our images + labels','Learn the new task','adapt'),
            ('3 · Infer','One new image','Predict its label','infer')]
    for i,(name,inputs,output,mode) in enumerate(stages):
        y=20+i*137
        row=t(35,y+48,name,29,'c-e' if i==2 else 'c-k',weight=600)
        if i==0:
            for j in [2,1,0]:
                row+=f.rect(248+j*12,y+7-j*4,63,63,'line','card',2)
            row+=f.image(253,y+12,53,53,f.photo)
        elif i==1:
            row+=f.image(240,y+4,62,62,f.photo)+f.image(311,y+4,62,62,b['CAT'])
        else:
            row+=f.image(270,y+4,64,64,b['CAT'])
        row+=t(305,y+99,inputs,22,'ink-2','middle')
        row+=arrow(398,y+37,459,y+37)
        row+=network(485,y+7,mode)
        row+=arrow(683,y+37,744,y+37)
        row+=t(774,y+33,output,29,'c-e' if i==2 else 'c-k')
        row+=t(774,y+76,'Weights stay fixed' if i==2 else 'All weights learn' if i==0 else 'Chosen weights learn',25,'ink-2')
        body+=row if i==0 else g(row,i)
    f.add('pets-learning-stages','Three stages: learn, adapt, then predict',body,
          'Orange connections and flames mark learning. Blue connections and snowflakes mark fixed parameters. During inference, a new image changes the features while the stored weights stay fixed.',
          'During which stages can the optimizer change weights?',
          'Pretraining and adaptation use learning signals. Inference only computes outputs using the learned parameters. Reveal each stage and follow the image-to-network-to-output arrows.',
          'This is a schematic recap of supervised pretraining, adaptation and inference. The image stack represents many labelled examples; '
          'the pet thumbnails illustrate the task rather than document a training run. Orange connections are trainable; blue connections are fixed. '
          'Adaptation may train only a new head or also selected encoder weights. A few representative neurons are drawn. '
          'The photographs are reused as visual examples; no new pet classifier or unseen-image result is claimed.',
          '<ol><li>Pretrain: many labelled images → learn encoder and classifier weights.</li>'
          '<li>Adapt: target images and labels → update the selected weights for the new task.</li>'
          '<li>Infer: a new image → run the trained model → predict a label with weights fixed.</li></ol>')

    body=t(35,34,'One forward pass through the trained model',29)
    body+=f.image(35,124,170,170,b['CAT'])
    body+=t(120,338,'Input photograph',24,'ink-2','middle')
    body+=arrow(218,209,278,209,'c-e')
    body+=box(293,127,295,[],h=161)
    body+=t(429,164,'Trained ViT',27,'c-e','middle')
    body+=status(557,151,True)
    body+=network(354,194,'infer',show_status=False)
    features=t(706,143,'192 features',26,'c-q','middle')
    features+=arrow(601,209,642,209,'c-q')
    for j in range(6):
        features+=f.rect(658+j*17,175,12,67,'c-q','t-q',1)
    features+=t(706,282,'Final CLS',24,'c-q','middle')
    features+=t(706,316,'depends on this image',21,'ink-2','middle')
    body+=g(features,1)
    head=arrow(773,209,820,209,'c-q')+t(978,143,'Trained class head',26,'c-e','middle')
    head+=status(1120,137,True)
    for iy in [182,208,234]:
        for oy in [189,254]:
            head+=f.line(849,iy,965,oy,'c-e',1.5)
        head+=f'<circle cx="841" cy="{iy}" r="8" fill="var(--card)" stroke="var(--c-q)" stroke-width="2"/>'
    for y,label in [(189,'dog score'),(254,'cat score')]:
        head+=f'<circle cx="978" cy="{y}" r="13" fill="var(--card)" stroke="var(--c-e)" stroke-width="2"/>'
        head+=t(1003,y+8,label,23,'c-e')
    head+=t(978,316,'192 inputs → 2 scores',23,'ink-2','middle')
    head+=t(35,416,'Softmax → two probabilities → choose the larger one.',29,'c-v')
    body+=g(head,2)
    f.add('pets-inference', 'What happens when we classify a new photograph?', body,
          'The image produces its own CLS features and class scores. Both snowflakes mark fixed weights. Inference computes a prediction without a label, loss or optimizer update.',
          'Do we need to know the new image’s label to run inference?',
          'No. First reveal the image-dependent CLS features, then the trained head and its two scores. Labels are needed later to evaluate the prediction.',
          'This diagram assumes a model already adapted to two outputs; this deck does not train one. '
          'The cat thumbnail illustrates an input and the bars depict feature coordinates, not measured values. '
          'No score, probability or winning label is fabricated. The head reads all 192 features; only a few connections are drawn. '
          'The learned parameters stay fixed, while feature values and attention weights depend on the input image. '
          'eval changes module behaviour such as dropout. inference_mode disables gradient tracking. Neither call trains the new head. '
          '<pre><code>model.eval()\nwith torch.inference_mode():\n    probabilities = model(images).softmax(dim=-1)  # B, 2</code></pre>',
          '<pre><code>model.eval()\nwith torch.inference_mode():\n    probabilities = model(images).softmax(dim=-1)  # B, 2</code></pre>')
    return f.frames
