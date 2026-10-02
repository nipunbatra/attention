"""One whole-model example: forward to the loss, backward to the parameters."""
import json
import math
from vision1_focus_common import Figures
from vision1_model_recap import model_recap


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

    body = model_recap(b, p, loss)
    f.add('photo-label-loss','The whole Vision Transformer in one figure',body,
        'Follow the numbered path. The expanded block shows three parallel heads, both residual additions, and the MLP. All 12 blocks keep every row. Final CLS feeds the classifier; the known label enters only at the loss.',
        'Can you trace one image through every operation without skipping a box?',
        'Start at pixels, then patch projection, CLS and positions. Trace all 12 blocks. Open block 1: normalize, three parallel Q/K/V → scores → row softmax → AV lanes, concatenate, output projection, add the original E. Normalize U, apply Linear → GELU → Linear, and add U. Continue through the remaining distinct blocks, final normalization, final CLS and the class head. Softmax gives the prediction; logits and the known label give cross-entropy.',
        provenance+'This complete reference diagram uses the parallel head lanes of the earlier TinyStories map. '
        'The single-image batch axis is omitted. Flatten each 3×16×16 patch to 768 numbers; the shared affine '
        'projection produces 196 rows of width 192. Prepend the learned 1×192 starting CLS, then add '
        'the learned 197×192 position table. All twelve block instances are shown explicitly. '
        'The large panel is an expanded view of block 1, not an extra block inserted into the sequence. '
        'Each block has its own learned weights and recomputes activations from its input. '
        'LN means LayerNorm. With X=LN₁(E), each of the three heads forms its own Q, K and V, each 197×64. '
        'Scores QKᵀ/√64 and attention weights A are 197×197; softmax runs across the source-key axis. '
        'The dashed V route bypasses scores and softmax. AV returns a 197×64 message matrix per head. '
        'Concatenate the three outputs to 197×192, apply the affine output projection, then add E to get U. '
        'The second sublayer applies LN₂(U), Linear(192,768), GELU and Linear(768,192), then adds U. '
        'Both linear layers have biases. The MLP is shared across rows and does not mix rows. '
        'Both residual paths preserve their own sublayer input. Every block returns all 197 rows. '
        'After block 12, apply final LayerNorm and select the CLS row. '
        'The class head produces 1,000 logits; the displayed probability comes from their softmax. '
        'The label is Newfoundland in this example, so L=-log p(Newfoundland). '
        'Use the unrounded stored probability to calculate the loss. In PyTorch, cross_entropy takes logits and the label directly. '
        'This example illustrates the learning objective; one image’s loss does not establish generalization.'+source
        +' <a href="figures/vision1/photo-label-loss.svg">Open the complete vector diagram</a>.',
        '<ol><li>RGB image (3 × 224 × 224) → 196 flattened patches (196 × 768) → shared projection (196 × 192).</li>'
        '<li>Prepend learned CLS and add learned positions: E⁰ is 197 × 192.</li>'
        '<li>Pass all rows through blocks 1–12. Each has separate learned weights.</li>'
        '<li>Inside each block: LN₁ → three parallel attention heads. Each forms Q, K, V (197 × 64), '
        'QKᵀ/√64 (197 × 197), row softmax, then AV (197 × 64). Join the heads, project to width 192, and add E to get U.</li>'
        '<li>LN₂(U) → Linear(192,768) → GELU → Linear(768,192) → add U.</li>'
        '<li>After block 12: final LayerNorm → select CLS (1 × 192) → class head (1,000 logits) → softmax → top label.</li></ol>'
        '<p>The known label y enters only at the loss. Here y = Newfoundland, p(y) = '
        f'{p:.5f}, and L = −log p(y) = {loss:.4f}.</p>'
        '<p>Forward computes activations and loss. Model parameters remain fixed during this calculation.</p>'
        '<p><a href="figures/vision1/photo-label-loss.svg">Open the complete vector diagram</a>.</p>', height=582)

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
        provenance+'This is the backward pass for the scalar cross-entropy loss in the optional complete-model reference. '
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
