"""Introduce the two original architecture figures before the photograph walkthrough."""
import base64


def introduce_papers(b, sections):
    t, g, rect, image, frame = (b[k] for k in
        ['t', 'g', 'rect', 'image', 'frame'])
    assets = b['ASSETS']
    text_path = b['ROOT'] / 'figures/transformer-paper/architecture.png'
    vision_path = assets / 'vit-paper-architecture.png'
    text_uri = 'data:image/png;base64,' + base64.b64encode(text_path.read_bytes()).decode()
    vision_uri = 'data:image/png;base64,' + base64.b64encode(vision_path.read_bytes()).decode()
    text_source = '<a href="https://arxiv.org/abs/1706.03762">Vaswani et al. (2017), Attention Is All You Need, Figure 1</a>'
    vision_source = '<a href="https://arxiv.org/abs/2010.11929">Dosovitskiy et al. (2020; ICLR 2021), An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Figure 1</a>'

    def add(key, title, body, caption, notes, prose, mobile):
        markup = frame(key, title, body, caption, notes, prose, mobile)
        # These overview figures need more vertical space than an arithmetic row.
        # Keep both the standalone SVG and the inline copy identical.
        old, new = 'viewBox="0 0 1160 440"', 'viewBox="0 0 1160 530"'
        path = assets / (key + '.svg')
        path.write_text(path.read_text().replace(old, new, 1))
        return markup.replace(old, new, 1)

    body = image(30, 0, 339.44, 500, text_uri)
    body += t(450, 50, '2017 · a Transformer for translation', 29, 'ink-2')
    body += t(450, 116, 'Text tokens → embedding rows', 32, 'c-e')
    body += t(450, 160, 'Add their positions.', 28, 'ink-2')
    # The outline is a lecture annotation around the encoder in the original.
    encoder = rect(65, 177, 138, 186, 'c-e', 'transparent', 8)
    encoder += t(450, 234, 'Encoder: build context', 34, 'c-e')
    encoder += t(450, 277, 'Self-attention + feed-forward layers', 28)
    encoder += t(450, 333, 'Decoder: generate the output text', 28, 'ink-2')
    body += g(encoder, 1)
    body += g(t(450, 412, 'Next: give an encoder image patches.', 31, 'c-e')
              +t(450, 457, 'Use its output to classify the image.', 28), 2)
    body += t(35, 526, 'Vaswani et al. (2017), Figure 1 · arXiv:1706.03762', 20, 'ink-2')
    text_frame = add('paper-text-transformer', 'From text: Attention Is All You Need', body,
        'Follow the encoder: text tokens become rows, then attention gives them context.',
        'Which stack will connect to our image classifier?\nShow the complete original encoder–decoder diagram first. Highlight the encoder on the left. Recall that it contextualizes input rows; the decoder on the right generates text. Preview image patches as the encoder input.',
        text_source + '. The authors’ original architecture diagram is reproduced here, with a blue lecture outline added around the encoder. '
        'The 2017 paper introduced this encoder–decoder Transformer for sequence transduction, including translation. '
        'Both stacks use attention and position-wise feed-forward networks. The encoder reads the available input sequence; '
        'the autoregressive decoder uses masked self-attention and cross-attention to the encoder. '
        'For this image-classification lecture, the useful connection is the encoder: it updates a sequence of input representations. '
        'The next figure replaces text embeddings with projected image patches and adds an image-classification readout. '
        'These are related architectures, not identical blocks: the original figure uses normalization after residual addition; '
        'the ViT figure places normalization before attention and the MLP. We will unpack our ViT block later. '
        '<a href="figures/transformer-paper/architecture.png">Open the original figure at full resolution</a>.',
        '<img src="figures/transformer-paper/architecture.png" alt="Original Transformer figure: encoder on the left, autoregressive decoder on the right">'
        '<p><strong>Follow the encoder:</strong> text embeddings + positions → self-attention and feed-forward layers → contextual rows.</p>'
        '<p>Next: give an encoder image patches and classify the image.</p><p>' + text_source + '.</p>')

    # Preserve the entire original ViT figure, including its encoder detail.
    scale, x, y = 1010 / 1327, 75, 0
    body = image(x, y, 1010, 510, vision_uri)
    def outline(px, py, width, height, color):
        return rect(x+px*scale, y+py*scale, width*scale, height*scale, color, 'transparent', 7)
    body += g(outline(314, 439, 571, 62, 'c-e'), 1)
    body += g(outline(216, 215, 670, 128, 'c-q'), 2)
    body += g(outline(56, 42, 282, 191, 'c-v'), 3)
    body += t(35, 526, 'Dosovitskiy et al. (2020; ICLR 2021), Figure 1 · arXiv:2010.11929', 20, 'ink-2')
    vision_frame = add('paper-vision-transformer', 'To images: An Image is Worth 16 × 16 Words', body,
        'Image patches become tokens. The encoder builds a summary for classification.',
        'What changed between the two paper figures?\nRead this figure from the image upward. Reveal the patch projection, the encoder, and the classification readout in that order. Point to the expanded encoder on the right. Then begin our dog-photo example.',
        vision_source + '. The complete original Figure 1 is reproduced; colored outlines are lecture annotations. '
        'The picture at the bottom is cut into patches. A shared linear projection turns each flattened patch into a vector. '
        'Add position embeddings and a learned classification token, process the sequence with Transformer encoder blocks, '
        'then use the final classification-token representation to score image classes. '
        'The right-hand inset opens an encoder block: normalization, multi-head self-attention, an MLP and residual additions. '
        'This is an architecture preview; the following photograph walkthrough explains each component when it is needed. '
        'The title’s 16×16 refers to a patch’s pixel height and width in a /16 model, not the number of patches or words. '
        'The paper figure draws nine patches schematically. Our 224×224 image with 16×16 patches gives a 14×14 grid, or 196 patch tokens. '
        'The original figure labels the classifier MLP Head; the saved fine-tuned checkpoint used later in this lecture has a single linear classifier. '
        '<a href="figures/vision1/vit-paper-architecture.png">Open the original figure at full resolution</a>.',
        '<img src="figures/vision1/vit-paper-architecture.png" alt="Original ViT overview: image patches, linear projection, positions and class token, Transformer encoder, and classification head; an expanded encoder appears on the right">'
        '<p>Image → patches → projected rows + positions + CLS → encoder → image class scores.</p>'
        '<p>16×16 is the pixel size of one patch. The nine patches drawn in this figure are schematic.</p><p>' + vision_source + '.</p>')

    title, frames = sections[0]
    sections[0] = (title, [text_frame, vision_frame] + frames)
    return sections
