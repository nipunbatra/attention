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

    body = t(35, 44, 'Use the learned parameters without updating them.', 29)
    body += box(35, 113, 305, ['New pet photograph', 'no target label supplied'], size=25, h=100)
    body += arrow(350, 163, 405, 163)
    body += box(420, 113, 330, ['Adapted encoder + head', 'two logits: dog, cat'], 'c-q', size=25, h=100)
    body += arrow(765, 163, 815, 163, 'c-q')
    body += box(830, 113, 295, ['Softmax', 'two probabilities'], 'c-v', size=25, h=100)
    body += f.code('model.eval()\nwith torch.inference_mode():\n    probabilities = model(images).softmax(dim=-1)  # B, 2', x=65, y=310, width=1030, size=25, spacing=39)
    f.add('pets-inference', 'Training changes parameters; inference uses them', body,
          'Training uses labelled examples and an optimizer. Inference uses the adapted model to predict a new image’s label. No loss, backward pass or optimizer step is needed for that prediction.',
          'Do we need to know the new image’s label to run inference?',
          'No. Labels are needed later to evaluate whether predictions were correct. Every image still gets its own features and attention weights.',
          'The code assumes a model already adapted to two outputs; this deck does not train one. '
          'eval changes module behaviour such as dropout. inference_mode disables gradient tracking. Neither call trains the new head.',
          '<pre><code>model.eval()\nwith torch.inference_mode():\n    probabilities = model(images).softmax(dim=-1)  # B, 2</code></pre>')
    return f.frames
