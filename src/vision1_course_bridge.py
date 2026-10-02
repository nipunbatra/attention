"""Connect the prior encoder lecture to ViT; trim repeated recaps conservatively.

Removed presentation frames remain in each section's optional reading notes.
The worked message, multihead, patch-projection and code sequences stay intact.
"""
import re
from html import escape
from vision1_focus_common import Figures

# These are duplicated introductions, not the worked computations.
OPTIONAL = {
    'paper-text-transformer', 'dataset-counts', 'dataset-dimensions', 'photo-folder',
    'task-next-token', 'task-bank-next-token', 'task-side-by-side', 's01-context',
    'bridge-text', 'text-context-recap', 'bridge-image-token', 'bridge-image-embedding',
    'query-example-dark', 'query-example-face', 'query-example-branch',
    'key-example-sources', 'value-example-messages', 'bridge-image-target',
    'one-rgb', 'flatten-order', 's01-rows-step-1',
    'cls-stored-start', 'cls-collect', 'cls-two-image-readout', 'cls-without',
    'cls-pool-arithmetic', 'cls-pool-dog', 'cls-readout-return',
}


def revise(b, sections):
    f = Figures(b)
    t, box, arrow, line, g = f.t, f.box, f.arrow, f.line, f.g
    key = lambda markup: re.search(r'class="frame[^\"]*" id="([^\"]+)"', markup).group(1)
    existing = {key(fr): fr for _, frames in sections for fr in frames}
    meta = {item['id']: item for item in b['FRAMES']}

    body = t(35, 42, 'Recall the three routes from the previous lecture', 31)
    for y, name, source, result, color in [
        (85, 'Encoder', 'supplied tokens', 'contextual rows', 'c-q'),
        (195, 'Decoder', 'a token prefix', 'next-token scores', 'c-e'),
        (305, 'Encoder–decoder', 'source + target prefix', 'next target-token scores', 'c-v'),
    ]:
        body += t(35, y+41, name, 27, color, weight=650)
        body += box(330, y, 310, source, color, h=70, size=27)
        body += arrow(655, y+35, 725, y+35, color)
        body += box(740, y, 385, result, color, h=70, size=27)
    f.add('prior-encoder-recap', 'We already know how an encoder builds context', body,
          'Today we reuse the encoder route. The new question is how pixels become tokens, and how the resulting image representation predicts a label.',
          'Which route gives a contextual representation of a complete supplied input?',
          'The encoder. Recall that the output head determines the task. We will supply image patches instead of text tokens.',
          '<p>This lecture follows <em>Transformers beyond next-token prediction</em>. '
          'Its encoder, decoder and encoder–decoder distinctions carry over directly. '
          'The main ViT example is an encoder-only image classifier.</p>')

    body = ''
    for y, label, tokens, color in [(38, 'TEXT ENCODER', ['CLS', 'Raghav', 'goes', 'to', 'school'], 'c-e'),
                                    (226, 'VISION TRANSFORMER', ['CLS', 'P1', 'P2', '…', 'P196'], 'c-q')]:
        body += t(35, y, label, 25, color, weight=650)
        for i, token in enumerate(tokens):
            x = 35+i*106
            body += box(x, y+25, 95, token, color, h=47, size=23)
        body += arrow(573, y+50, 622, y+50, color)
        body += box(632, y+7, 260, ['Encoder blocks', 'whole-input attention'], color, h=86, size=23)
        body += arrow(904, y+50, 942, y+50, color)
        body += box(952, y+7, 178, ['Read CLS', '→ classifier'], color, h=86, size=24)
    body += t(580, 401, 'Same encoder idea. Different tokens.', 36, 'ink', 'middle', 650)
    f.add('vit-same-encoder', 'A ViT is an encoder over image patches', body,
          'Our ViT uses encoder-only self-attention over the supplied image. CLS and all patches can exchange information. It has no text decoder or separate cross-attention stream.',
          'What changes when we replace the text sequence with patch tokens?',
          'The input embedding and task head change. The encoder machinery remains: attention, MLPs, residuals and normalization. Both pictured classifiers read final CLS.',
          '<p>The top row recalls the previous text-classification example; the bottom row gives today’s architecture. '
          'Full attention means all supplied rows are available, not that they receive equal weights. '
          'Our checkpoint uses twelve pre-LayerNorm blocks. See the <a href="https://arxiv.org/abs/2010.11929">original ViT paper</a>.</p>')

    body = t(250, 40, 'TEXT', 30, 'c-e', 'middle', 650)+t(870, 40, 'IMAGE', 30, 'c-q', 'middle', 650)
    for x, color, labels in [(35, 'c-e', [('“bank” → token ID', ''), ('Embedding table', 'look up a learned row'), ('token content row', 'D features')]),
                             (655, 'c-q', [('RGB patch: 16 × 16 × 3', 'flatten → 768 values'), ('Shared Linear(768, D)', 'compute a learned projection'), ('patch content row', 'D features')])]:
        for i, pair in enumerate(labels):
            y = 67+i*120
            body += box(x, y, 440, [v for v in pair if v], color, h=83, size=26)
            if i < 2: body += arrow(x+220, y+90, x+220, y+112, color)
    body += t(580, 426, 'Add position → send the sequence of rows into the encoder', 28, 'ink', 'middle')
    f.add('vit-token-inputs', 'Text looks up a row; an image patch computes one', body,
          'Both routes produce a row of D features. Text learns a vocabulary table; vision learns a shared pixel projection. Our image checkpoint chooses D = 192.',
          'Does an image patch need a vocabulary ID?',
          'No. Its pixel values go through a learned affine layer. Once position is added, the encoder works with feature rows in either case.',
          '<p>D denotes each model’s chosen embedding width; the two models need not use the same numerical D. '
          'The patch projection includes a bias and no following activation. '
          'Later we calculate its output by hand and implement the same map with Linear and Conv2d.</p>')

    body = t(35, 42, 'TRANSLATION · cross-attention', 29, 'c-v', weight=650)
    body += box(35, 83, 360, ['Target-prefix representations', 'from the decoder'], 'c-q', h=83, size=25)
    body += arrow(408, 124, 472, 124, 'c-q')+box(483, 96, 82, 'Q', 'c-q', h=57, size=28)
    body += box(620, 83, 310, ['Source representations', 'from the encoder'], 'c-v', h=83, size=25)
    body += arrow(942, 124, 994, 124, 'c-v')+box(1005, 96, 125, 'K, V', 'c-v', h=57, size=28)
    body += line(35, 210, 1125, 210)
    body += t(35, 254, 'ViT · self-attention', 29, 'c-e', weight=650)
    body += box(35, 298, 455, ['CLS + 196 patch representations', 'the same current input to this block'], 'c-e', h=89, size=25)
    body += arrow(505, 342, 589, 342, 'c-e')
    body += box(603, 298, 527, ['Three learned projections → Q, K, V', 'all come from these 197 image rows'], 'c-e', h=89, size=26)
    f.add('vit-self-vs-cross', 'Self-attention: Q, K and V share the same input', body,
          'ViT computes Q, K and V from the same image sequence. Translation cross-attention instead takes queries from the target stream and keys and values from the source stream.',
          'Does CLS query a separate image encoder through cross-attention?',
          'No. CLS is one row inside the image sequence. In this pre-LN model, all three projections read the same normalized current rows.',
          '<p>Self-attention describes the source of Q/K/V, not equality between their learned projections. '
          'Different W_Q and W_K make QKᵀ generally asymmetric even though both begin with the same rows. '
          'The following slides retain a worked CLS row so every score, value contribution and receiver update can be followed.</p>')

    body = t(35, 32, 'ONE IMAGE · batch axis omitted · the measured tiny ViT', 24, 'ink-2')
    labels = [(['RGB image', '3 × 224 × 224'], 'c-e'),
              (['Shared patch projection', '196 × 192'], 'c-e'),
              (['Prepend CLS; add position', '197 × 192'], 'c-q'),
              (['12 encoder blocks', '197 × 192'], 'c-v'),
              (['Final LayerNorm; read CLS', '192'], 'c-q'),
              (['Linear class head', '1,000 logits'], 'c-e')]
    for i, (words, color) in enumerate(labels):
        x = 35+(i%3)*382; y = 73+(i//3)*220
        body += box(x, y, 325, words, color, h=100, size=25)
        if i%3 != 2: body += arrow(x+337, y+50, x+367, y+50, color)
    # Wrapped path is explicit rather than implying three independent columns.
    body += line(961, 185, 961, 226, 'c-q')+line(961, 226, 197, 226, 'c-q')+arrow(197, 226, 197, 281, 'c-q')
    body += t(580, 440, 'All 197 rows evolve through the blocks; the head reads one final row.', 26, 'ink', 'middle')
    f.add('vit-shape-trace', 'One shape trace from pixels to class scores', body,
          'The patch layer changes feature width. CLS adds one row. Every block preserves 197 × 192. Final normalization and CLS selection give the 192-feature input to the class head.',
          'Which step changes the number of rows, and which step changes the final output width?',
          'Prepending CLS changes 196 rows to 197. The head maps the selected 192-feature row to 1,000 logits. Expand the next diagram to see inside every block.',
          '<p>The model input uses channel-first order (3,224,224), although the photograph may be described as 224×224×3. '
          'The shared patch map consumes 3×16×16=768 input values per patch. '
          'This small checkpoint uses D=192, three heads of width 64, MLP hidden width 768 and twelve blocks; these are model choices.</p>', height=460)

    body = box(35, 70, 290, ['Final CLS', 'h · 192 features'], 'c-q', h=104, size=25)
    for i, label in enumerate(['Newfoundland', 'Persian cat', '… 998 other labels']):
        y=55+i*113
        body += box(415, y, 322, [label, 'learned class vector wₖ'], 'c-e', h=82, size=25)
        body += arrow(750, y+41, 815, y+41, 'c-e')
        body += t(835, y+50, 'hᵀwₖ + bₖ', 31, 'c-e')
        body += line(337, 122, 376, 122, 'c-q')+line(376, 122, 376, y+41, 'c-q')+arrow(376, y+41, 404, y+41, 'c-q')
    body += t(35, 402, 'The head stores one learned vector and one bias per label.', 31)
    f.add('vision-fixed-class-vectors', 'Our classifier stores a vector for each known label', body,
          'A class score is a dot product with its learned weight vector, plus a bias. The current head has 1,000 fixed label slots. New tasks can train a replacement head.',
          'Where does the vector for the label Newfoundland come from?',
          'It is a learned row of the classifier weight matrix. The class name itself is not read by this classifier.',
          '<p>For this checkpoint, nn.Linear(192,1000) stores weights with shape (1000,192) and 1000 biases. '
          'The formula re-expresses the same class-head neurons seen earlier. These are unnormalized supervised class weights, not CLIP text embeddings.</p>')

    body = f.image(35, 70, 120, 120, f.photo)+arrow(166,130,211,130,'c-q')
    body += box(222, 87, 300, ['Image encoder', '+ projection'], 'c-q', h=85, size=27)
    body += arrow(534,130,589,130,'c-q')+box(601,99,234,'image vector','c-q',h=62,size=26)
    body += box(35, 271, 286, ['“a photo of a dog”', '“a photo of a cat”'], 'c-v', h=85, size=25)
    body += arrow(333,313,378,313,'c-v')+box(390,271,252,['Text encoder','+ projection'],'c-v',h=85,size=26)
    body += arrow(654,313,690,313,'c-v')+box(702,282,207,'text vectors','c-v',h=62,size=26)
    body += line(847,130,992,130,'c-q')+arrow(992,130,992,174,'c-q')
    body += line(921,313,992,313,'c-v')+arrow(992,313,992,266,'c-v')
    body += box(900,180,226,['Compare in a','shared space'],'c-e',h=80,size=23)
    body += t(580,423,'Next: train image and text representations to match.',31,'ink','middle',650)
    f.add('vision-language-handoff', 'What if a class vector could come from language?', body,
          'CLIP learns a shared space for image and text representations. A prompt can then describe a candidate class. The next lecture explains the matching objective and how these vectors are trained.',
          'Could we replace today’s class weights with arbitrary text embeddings?',
          'Not directly. Image and text encoders need compatible projections and joint alignment training. CLIP compares normalized image and text vectors, with a learned score scale.',
          '<p>This is a question for the next lecture, not a claim that our ImageNet classifier already understands prompts. '
          'The diagram is conceptual; it supplies no invented matching scores. '
          'See <a href="https://arxiv.org/abs/2103.00020">Learning Transferable Visual Models From Natural Language Supervision</a> '
          'for CLIP’s image–text training and transfer mechanism.</p>')

    result=[]
    for i,(title,frames) in enumerate(sections):
        kept=[]; optional=[]
        for markup in frames:
            k=key(markup)
            if k in OPTIONAL:
                reading=re.sub(r'class="frame[^\"]*"', 'class="vp-optional-practice"', markup, count=1)
                reading=re.sub(r' data-build="\d+"', '', reading)
                optional.append('<h3>'+escape(meta[k]['title'])+'</h3>'+reading)
                continue
            if k=='real-patch-qkv': kept.append(f.frames['vit-self-vs-cross'])
            if k=='photo-label-loss': kept.append(f.frames['vit-shape-trace'])
            kept.append(markup)
        if i==0:
            kept=[f.frames[k] for k in ['prior-encoder-recap','vit-same-encoder','vit-token-inputs']]+kept
            # Keep the primary paper diagram, after the task has been established.
            kept=[fr for fr in kept if key(fr)!='paper-vision-transformer']+[existing['paper-vision-transformer']]
        if i==9:
            kept += [f.frames['vision-fixed-class-vectors'], f.frames['vision-language-handoff']]
        if optional:
            kept[-1]+='<div class="companion"><details><summary>Optional recap and extra examples</summary>'+''.join(optional)+'</details></div>'
        result.append((title,kept))
    b['REVISION_OPTIONAL_IDS']=sorted(OPTIONAL)
    return result
