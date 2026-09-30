"""One whole-model example: forward to the loss, backward to the parameters."""
import json
import math
from vision1_focus_common import Figures


def build(b):
    f = Figures(b)
    t, g, box, arrow, line = f.t, f.g, f.box, f.arrow, f.line
    trace = json.loads((b['ASSETS']/'classifier-readout-trace.json').read_text())
    p = trace['selected_classes'][0]['probability']
    loss = -math.log(p)
    provenance = ('The probability is from the saved dog forward pass. Treat this as an illustrative labelled example; '
                  'we do not claim that this photo was in the checkpoint training set. No training is run. ')
    source = ' <a href="figures/vision1/classifier-readout-trace.json">Saved prediction used in this example</a>.'

    def model_path(backward=False):
        color = 'c-a' if backward else 'c-e'
        out = box(35,105,132,['Image x','3 × 224 × 224'],'ink-2',h=90,size=18)
        out += box(199,105,217,['Input embedding','197 × 192'],color,h=90,size=23)
        out += box(448,105,239,['12 blocks','197 × 192'],color,h=90,size=25)
        out += box(719,105,211,['Read CLS + head','192 → 1,000'],color,h=90,size=23)
        out += box(962,105,163,['Label loss',f'L = {loss:.4f}'],'c-a',h=90,size=23)
        if backward:
            for right,left in [(958,935),(714,692),(443,421)]:
                out += arrow(right,150,left,150,'c-a')
            out += t(101,229,'Fixed data',23,'ink-2','middle')
        else:
            for left,right in [(173,191),(422,440),(693,711),(936,954)]:
                out += arrow(left,150,right,150,'c-e')
        return out

    body = t(35,38,'One labelled example → one prediction → one loss',30)
    body += model_path()
    body += line(199,215,930,215,'c-e')+t(564,249,'ViT classifier: scores = fθ(x)',27,'c-e','middle')
    body += g(box(840,285,285,['Known label y','Newfoundland'],'c-a',h=72,size=24)
              +arrow(1043,278,1043,204,'c-a'),1)
    body += g(t(35,322,f'p(Newfoundland) = {p:.5f}',30,'c-e')
              +t(35,369,f'L = −log p(y) = {loss:.4f}',31,'c-a'),1)
    body += t(35,433,'Forward: compute the prediction and loss with the current parameters.',27)
    f.add('photo-label-loss','Forward: follow the whole model to one label loss',body,
        'The image passes through the complete classifier. The known label enters at the loss, after the model produces class scores. Cross-entropy measures how much probability the model assigned to that label. Parameters stay fixed during this forward pass.',
        'Where does the known label enter?',
        'Only at the loss. Follow the blue arrows through input embeddings, all 12 blocks and the final CLS readout. Then reveal the label and calculate one scalar loss.',
        provenance+'The input embedding box includes the shared patch projection, learned starting CLS and positions. '
        'We use their established shapes without repeating the patch construction. Each block includes attention, '
        'its output projection and residual, then the MLP and its residual. Final LayerNorm is included in the CLS readout. '
        'The class head produces 1,000 logits; the displayed probability comes from their softmax. '
        'The label is Newfoundland in this example, so L=-log p(Newfoundland). '
        'Use the unrounded stored probability to calculate the loss. In PyTorch, cross_entropy takes logits and the label directly. '
        'This example illustrates the learning objective; one image’s loss does not establish generalization.'+source,
        '<p>Image x (3 × 224 × 224) → input embeddings (197 × 192) → 12 Transformer blocks '
        '(197 × 192) → final normalization, CLS readout and class head (1,000 scores) → label loss.</p>'
        '<p>The known label y enters only at the loss. Here y = Newfoundland, p(y) = '
        f'{p:.5f}, and L = −log p(y) = {loss:.4f}.</p>'
        '<p>Forward computes activations and loss. Model parameters remain fixed during this calculation.</p>')

    body = t(35,38,'Same model, reverse direction: loss → parameter gradients',29,'c-a')
    body += model_path(backward=True)
    body += t(199,244,'Gradients reach the head, all blocks, and the learned input parameters.',25,'c-a')
    body += g(box(35,295,310,['Compute gradients','θ.grad = ∂L/∂θ'],'c-a',h=84,size=28)
              +arrow(351,337,389,337,'c-a')
              +box(397,295,350,['Optimizer step','θ ← θ − η ∇θ L'],'c-e',h=84,size=28)
              +arrow(754,337,795,337,'c-e')
              +box(803,295,322,['Next forward pass','use updated parameters'],'c-e',h=84,size=24),1)
    body += g(line(199,181,181,181,'c-a')+line(181,181,181,267,'c-a')
              +line(181,267,190,267,'c-a')+arrow(190,267,190,288,'c-a'),1)
    body += t(35,433,'backward() computes gradients; step() changes the parameters.',29)
    f.add('photo-optimizer-step','Backward: compute gradients, then update the model',body,
        'Reverse the forward dependencies to compute gradients for the trainable parameters. The optimizer uses these gradients to update them. The next forward pass uses the updated model. The image and its label remain fixed.',
        'Does backward itself change the weights?',
        'No. Backward computes gradients through the same model. The optimizer step applies the update. The learned input parameters include the patch projection, positions and starting CLS; the photograph and label are fixed data.',
        provenance+'This is the backward pass for the same scalar cross-entropy loss shown on the preceding slide. '
        'Start with dlogits=p−one_hot(y). The head receives parameter gradients and passes a gradient to final CLS. '
        'Reverse final normalization and the 12 blocks. Residual additions send gradients along both paths, '
        'and shared inputs accumulate their contributions. Attention connects the CLS loss to patch keys and values; '
        'earlier patch updates can affect later CLS states. Input gradients reach the shared patch projection, '
        'the position table and starting CLS. The optimizer is given model parameters, not image pixels. '
        'The diagram summarizes these dependencies without opening each block again. '
        'The displayed update is ordinary SGD, with learning rate η; other optimizers use the same gradients '
        'with their own update rules. In a training loop, zero_grad clears accumulated gradients before backward, '
        'and step changes the selected parameters. The parameter update is illustrative, not an executed training result.'+source,
        '<p>Label loss → class head and final CLS → blocks 12 through 1 → learned input parameters.</p>'
        '<p>Gradients reach the head, attention and MLP weights, normalization parameters, patch projection, '
        'position table and starting CLS. Input image and label remain fixed.</p>'
        '<p><strong>backward():</strong> compute θ.grad. <strong>step():</strong> update parameters. '
        'For SGD, θ ← θ − η∇θL. Then run a new forward pass.</p>')
    return list(f.frames.values())
