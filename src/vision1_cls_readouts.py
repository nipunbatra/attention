"""Follow the same dog through CLS and mean-pooling classifier designs."""
import base64


def build_cls_readouts(b):
    t, g, rect, arrow, line, image, frame, mobile_rows = (b[k] for k in
        ['t', 'g', 'rect', 'arrow', 'line', 'image', 'frame', 'mobile_rows'])
    photo = 'data:image/png;base64,' + base64.b64encode(
        (b['ASSETS'] / 'model-input.png').read_bytes()).decode()
    result = {}
    source = (' <a href="https://arxiv.org/html/2010.11929v2#A4.SS3">'
              'ViT paper, Appendix D.3: class token and average pooling</a>.')

    def box(x, y, w, labels, color='c-e', h=84, size=24):
        out = rect(x, y, w, h, color, 't-q' if color == 'c-q' else 'card', 5)
        for j, label in enumerate(labels):
            out += t(x+w/2, y+h/2+(j-(len(labels)-1)/2)*30+8, label, size, color, 'middle')
        return out

    def crop(x, y, side, index, n=14):
        cell = 224/n
        left, top = ((index-1) % n)*cell, ((index-1)//n)*cell
        return (f'<svg x="{x}" y="{y}" width="{side}" height="{side}" '
                f'viewBox="{left} {top} {cell} {cell}" overflow="hidden">'
                + image(0, 0, 224, 224, photo) + '</svg>')

    def grid(x, y, side, n=14):
        out = image(x, y, side, side, photo)
        for k in range(1, n):
            out += line(x+k*side/n, y, x+k*side/n, y+side, 'card', .7)
            out += line(x, y+k*side/n, x+side, y+k*side/n, 'card', .7)
        return out

    def add(key, title, body, caption, notes, prose, mobile):
        for meta in b['FRAMES']:
            if meta['id'] == key:
                meta.update(title=title, caption=caption, notes=notes)
        result[key] = frame(key, title, body, caption, notes, prose, mobile)

    def dog_inputs(with_cls):
        out = t(35, 35, 'Same dog → patches → projection + positions', 28)
        out += grid(35, 117, 190) + t(130, 342, '224 × 224 × 3', 23, 'ink-2', 'middle')
        out += arrow(233, 214, 270, 214)
        out += t(410, 74, 'Input rows · 192 features each', 22, 'ink-2', 'middle')
        if with_cls:
            out += box(285, 90, 250, ['CLS · shared start'], 'c-q', 38, 23)
        else:
            out += t(410, 119, 'Only the patch rows', 24, 'c-v', 'middle')
        for y, index in [(147, 1), (205, 63), (299, 196)]:
            out += crop(285, y, 40, index)
            out += box(339, y, 196, [f'P{index} → e{str(index).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))}'], h=40, size=24)
        out += t(417, 278, '⋮', 32, 'ink-2', 'middle')
        out += t(410, 380, ('197' if with_cls else '196')+' × 192', 29,
                 'c-q' if with_cls else 'c-v', 'middle')
        out += arrow(544, 214, 588, 214)
        out += box(605, 151, 190, ['12 blocks', 'attention + MLP'], h=124, size=23)
        return out

    body = dog_inputs(True)
    body += t(995, 88, 'Read the final CLS row', 24, 'c-q', 'middle')
    body += g(arrow(801, 192, 852, 151, 'c-q')
              +box(870, 109, 250, ['Dog’s final CLS', '1 × 192'], 'c-q'), 1)
    body += g(arrow(995, 199, 995, 240, 'c-q')
              +box(870, 254, 250, ['Class head', 'Linear(192, 1000)'], size=23)
              +arrow(995, 344, 995, 365)
              +t(700, 393, 'Newfoundland · 95.73%', 29, 'c-e'), 2)
    body += t(35, 433, '197 rows are updated; the classifier reads the one reserved for the image summary.', 25)
    add('cls-without', 'With CLS, read the dog’s final summary row', body,
        'The dog supplies 196 patch rows. Add the shared CLS row, update all 197 rows through the blocks, then read the final CLS. Its 192 features produce the saved Newfoundland prediction.',
        'Which row does the classifier read, and how did the dog enter that row?\n'
        'Follow the purple CLS row. Attention mixes information from the dog’s patch rows into it. The blocks update patch rows too.',
        'This is the already measured classifier, shown again to locate its readout. Each 16×16 RGB crop is projected '
        'to 192 features and receives its position vector. The stored CLS parameter also receives its position vector. '
        'All 197 rows pass through 12 blocks. The classifier reads the final normalized CLS representation, then softmax '
        'gives the saved 95.73% Newfoundland probability for this photograph. Intermediate normalization is folded into '
        'this overview. The patch thumbnails identify rows; the network processes their feature vectors. '
        '<a href="figures/vision1/real-classifier-path.json">Saved forward-pass evidence</a>.',
        '<img src="figures/vision1/model-input.png" alt="The dog photograph used throughout the lecture" width="160">'
        '<p>196 dog patch rows + one CLS row → <strong>197 × 192</strong>.</p>'
        '<p>12 blocks update every row. Read the final CLS → <strong>1 × 192</strong>.</p>'
        '<p>Linear(192,1000) → class probabilities → Newfoundland, 95.73% for this photograph.</p>')

    body = dog_inputs(False)
    body += t(995, 88, 'Combine the final patch rows', 23, 'c-v', 'middle')
    body += g(arrow(801, 192, 852, 151, 'c-v')
              +box(870, 109, 250, ['Mean pooling', '196 × 192 → 1 × 192'], 'c-v', size=22), 1)
    body += g(arrow(995, 199, 995, 240, 'c-v')
              +box(870, 254, 250, ['Class head', 'Linear(192, 1000)'], size=23)
              +arrow(995, 344, 995, 365)
              +t(995, 393, '1,000 class scores', 26, 'c-e', 'middle'), 2)
    body += t(35, 433, 'Train this design with image labels, such as Newfoundland for a dog photograph like this.', 25)
    add('cls-pool-dog', 'Without CLS, combine the dog’s final patch rows', body,
        'Build a second design with 196 patch rows and no CLS. Attention still lets the patches share information. Average their final features to get one 192-number summary. Train the classifier and blocks with this readout.',
        'Without an extra summary row, where is the information about this dog?\n'
        'It remains in the 196 contextual patch representations. Average corresponding coordinates across these rows, then classify the resulting vector.',
        'This is an architectural alternative, not a measured run of a second model. The known breed label is a '
        'supervised training target, not a claimed prediction or an input to the forward pass. Use 196 positional '
        'vectors for 196 patch rows; there is no CLS row. The final patch features have already exchanged context through '
        'attention. Mean pooling reduces the row axis from 196 to one and preserves all 192 feature coordinates. '
        'The classifier has the same input/output dimensions as before, but its parameters and the encoder are trained '
        'for the chosen readout. Normalization follows that architecture and is omitted from this overview. '
        'No alternate-model accuracy or probability is claimed.'+source,
        '<img src="figures/vision1/model-input.png" alt="The same dog, now used to explain a model without CLS" width="160">'
        '<p>196 dog patch rows → <strong>196 × 192</strong> through the Transformer blocks.</p>'
        '<p>Mean pooling: average final patch features across the 196 rows → <strong>1 × 192</strong>.</p>'
        '<p>Linear(192,1000) → class scores. Train with the breed label Newfoundland for a labelled photo like this.</p>'
        '<p>This is a proposed model design, with no measured prediction shown.</p>')

    body = t(35, 35, 'Small calculation · 4 large patches, 2 final features each · chosen numbers', 26, 'ink-2')
    body += grid(35, 98, 224, 2)
    for j in range(4):
        x, y = 35+(j % 2)*112, 98+(j//2)*112
        body += rect(x+6, y+6, 37, 28, 'c-e', 'card', 3)+t(x+24, y+27, f'P{j+1}', 18, 'c-e', 'middle')
    body += t(147, 360, 'Same dog, smaller example', 22, 'ink-2', 'middle')
    body += arrow(267, 212, 308, 212)
    body += t(472, 80, 'After the attention blocks', 25, 'c-e', 'middle')
    toy_rows = [(2, 0), (4, 2), (2, 4), (0, 2)]
    for j, values in enumerate(toy_rows):
        y = 103+j*66
        body += crop(325, y, 49, j+1, 2)+t(393, y+33, f'P{j+1}', 24, 'ink-2')
        body += rect(463, y, 67, 49, 'c-e', 't-e', 4)+t(496, y+33, values[0], 28, 'c-e', 'middle')
        body += rect(546, y, 67, 49, 'c-v', 'card', 4)+t(579, y+33, values[1], 28, 'c-v', 'middle')
    body += g(t(708, 110, 'Average down each column', 28)
              +t(708, 171, '(2 + 4 + 2 + 0) / 4 = 2', 27, 'c-e')
              +t(708, 225, '(0 + 2 + 4 + 2) / 4 = 2', 27, 'c-v')
              +box(755, 279, 280, ['Image summary: [2, 2]', '4 × 2 → 1 × 2'], 'c-v', size=25), 1)
    body += t(35, 427, 'For our full-size design: average 196 rows, keeping all 192 feature coordinates.', 26)
    add('cls-pool-arithmetic', 'Mean pooling: average each feature across the patches', body,
        'Each row is a patch representation after it has read context. Average feature 1 across patches, then feature 2. Four rows become one row. In the full model, 196 rows become one 192-feature summary.',
        'Does pooling average the pixels, or the features after attention?\n'
        'The final features. Ask students to compute each column average. Each feature coordinate is retained; only the patch axis is reduced.',
        'This separate miniature illustration uses four 112×112 crops of the same 224×224 photograph and two feature '
        'coordinates per final row. It does not change the real model’s 16×16 patch size or D=192. The four output '
        'vectors [2,0], [4,2], [2,4], [0,2] are deliberately chosen arithmetic values, not measured embeddings or '
        'outputs from the other worksheet. Their coordinate-wise average is [2,2]. The thumbnails identify which '
        'patch each row belongs to; each final row can contain context from all patches. In the full pooling design, '
        'for every feature j, summary[j] = (H[1,j] + … + H[196,j]) / 196. In batched code, H.mean(dim=1) '
        'maps (B,196,192) to (B,192). Pooling has no trainable parameters; the encoder and classifier learn.',
        '<img src="figures/vision1/model-input.png" alt="The same dog is split into four large patches for a small arithmetic example" width="160">'
        '<p>Separate small example: four large patches and two final features. These numbers are chosen for calculation.</p>'
        +mobile_rows(['Patch row after the blocks', 'Final features'],
                     [[f'P{j+1}', str(list(v))] for j, v in enumerate(toy_rows)])
        +'<p>Feature 1: (2 + 4 + 2 + 0) / 4 = 2.</p><p>Feature 2: (0 + 2 + 4 + 2) / 4 = 2.</p>'
        '<p><strong>Image summary: [2, 2]. Shape 4 × 2 → 1 × 2.</strong></p>'
        '<p>For the full design: 196 × 192 → 1 × 192. Average final features, after attention.</p>')

    body = t(35, 32, '1 · A familiar convention', 29, weight=650)
    body += box(35, 59, 285, ['BERT: classification token'], 'c-q', h=52, size=23)
    body += arrow(328, 85, 398, 85, 'c-q')
    body += box(414, 59, 285, ['Original ViT adopts it'], 'c-q', h=52, size=23)
    body += arrow(707, 85, 777, 85, 'c-q')
    body += box(793, 59, 332, ['Our checkpoint uses CLS'], 'c-q', h=52, size=23)

    learned = t(35, 164, '2 · A dedicated summary, refined through the blocks', 29, weight=650)
    learned += box(35, 192, 240, ['Patch rows + CLS'], h=70, size=24)
    learned += arrow(286, 227, 448, 227, 'c-v')
    learned += t(368, 211, 'attention', 22, 'c-v', 'middle')
    learned += box(461, 192, 240, ['Updated CLS'], 'c-q', h=70, size=25)
    learned += arrow(712, 227, 835, 227, 'c-q')
    learned += t(773, 211, 'later blocks', 20, 'ink-2', 'middle')
    learned += box(848, 182, 277, ['Final CLS', '→ class head'], 'c-q', h=90, size=24)
    learned += t(35, 308, 'Attention learns how to weight sources; the label loss trains the summary.', 27)
    body += g(learned, 1)

    evidence = t(35, 364, '3 · Both readouts deserve a fair comparison', 29, weight=650)
    evidence += t(35, 404, 'Original ViT found similar performance after tuning each readout’s learning rate.', 25, 'c-v')
    evidence += t(35, 439, 'ViT paper · §3.1 and Appendix D.3, Figure 9', 20, 'ink-2')
    body += g(evidence, 2)
    add('cls-readout-return', 'Why is CLS a common choice if pooling also works?', body,
        'This diagram follows CLS; patch rows receive updates too. Keep the readout a checkpoint was trained to use, or adapt and evaluate the changed model.',
        'Does a learned summary token guarantee better accuracy than mean pooling?\n'
        'No. Explain the BERT heritage, then the dedicated readout. Original ViT Appendix D.3 found comparable results after tuning the learning rate. Use the checkpoint’s trained route; compare alternatives with suitable training and evaluation.',
        'The original ViT adopted the classification-token convention from BERT (§3.1). CLS provides a designated '
        'place for the classifier to read an image-level representation. In each block its attention weights can '
        'depend on the current query and source keys; later CLS queries already contain image context. The '
        'classification objective trains the initial CLS parameter and the shared encoder and classifier weights. '
        'The diagram follows the CLS readout only; patch rows also attend to all rows and receive updates. '
        'A learned readout is a useful architectural option, not proof of superior accuracy. Appendix D.3 and '
        'Figure 9 report comparable class-token and global-average-pooling results once learning rates were tuned. '
        'Mean pooling has fixed averaging coefficients over final rows, but those rows contain learned context; '
        'it does not make every raw pixel equally important. For an existing checkpoint, retain its readout unless '
        'you adapt and evaluate the changed model. We now return to our checkpoint’s 197-row CLS route.'+source,
        '<ol><li><strong>History:</strong> BERT’s classification token → original ViT → our checkpoint.</li>'
        '<li><strong>Role:</strong> CLS is a dedicated summary updated through attention; classification loss '
        'trains it to support the image label. Patch rows are updated too.</li>'
        '<li><strong>Evidence:</strong> the original ViT study found similar performance for CLS and mean pooling '
        'after tuning their learning rates. CLS is not a universal accuracy winner.</li></ol>'
        '<p>Keep the readout used by a pretrained checkpoint. Train or adapt and evaluate a different choice.</p>'+source)
    return result
