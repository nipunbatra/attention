"""Draw the real classifier's attention shapes and explain every matrix axis."""
import base64


def build_attention_matrices(b):
    t, g, rect, arrow, line, image, frame, mobile_rows = (b[k] for k in
        ['t', 'g', 'rect', 'arrow', 'line', 'image', 'frame', 'mobile_rows'])
    result = {}
    tokens = ['CLS', 'P1', '…', 'P63', '…', 'P196']
    short_tokens = ['CLS', 'P1', '…', 'P196']
    features = ['1', '2', '…', '64']
    photo = 'data:image/png;base64,' + base64.b64encode(
        (b['ASSETS']/'model-input.png').read_bytes()).decode()
    source = (' <a href="https://arxiv.org/html/2010.11929v2">ViT paper, Appendix A</a> · '
              '<a href="notebooks/vision/trace_real_classifier.py">This checkpoint’s verified attention computation</a>.')

    def matrix(x, y, w, h, rows, cols, color='c-e', row=None, col=None, cell=None,
               entry=None, labels=True, size=20, allowed=None, entries=None):
        """Schematic grid: labels specify omitted rows; dots are not data."""
        nr, nc = len(rows), len(cols)
        cw, rh = w/nc, h/nr
        out = ''
        for r in range(nr):
            for c in range(nc):
                selected = r == row or c == col or (r, c) == cell
                permit = allowed(r, c) if allowed else None
                tint = color if selected or permit is True else 'line'
                opacity = .18 if selected or permit is True else (.07 if allowed else .025)
                out += (f'<rect x="{x+c*cw}" y="{y+r*rh}" width="{cw}" height="{rh}" '
                        f'fill="var(--{tint})" fill-opacity="{opacity}" stroke="var(--line)" stroke-width="1"/>')
                mark = '×' if permit is False else ('•' if permit is True else '·')
                if entries is not None:
                    mark = entries.get((r, c), mark)
                if (r, c) == cell and entry:
                    mark = entry
                out += t(x+(c+.5)*cw, y+(r+.5)*rh+6, mark, 21,
                         color if selected or permit else 'ink-3', 'middle')
        out += line(x-5, y, x-10, y, color)+line(x-10, y, x-10, y+h, color)+line(x-10, y+h, x-5, y+h, color)
        out += line(x+w+5, y, x+w+10, y, color)+line(x+w+10, y, x+w+10, y+h, color)+line(x+w+10, y+h, x+w+5, y+h, color)
        if labels:
            for r, label in enumerate(rows):
                out += t(x-18, y+(r+.5)*rh+6, label, size, color if r == row else 'ink-2', 'end')
            for c, label in enumerate(cols):
                out += t(x+(c+.5)*cw, y-12, label, size, color if c == col else 'ink-2', 'middle')
        if cell is not None:
            r, c = cell
            out += rect(x+c*cw, y+r*rh, cw, rh, color, 'transparent', 0)
        return out

    def add(key, title, body, caption, question, notes, prose, mobile):
        full_notes = question+'\n'+notes
        for meta in b['FRAMES']:
            if meta['id'] == key:
                meta.update(title=title, caption=caption, notes=full_notes)
        result[key] = frame(key, title, body, caption, full_notes, prose+source, mobile)

    body = image(25, 10, 64, 64, photo)
    body += t(110, 35, 'Follow one head in block 1 · 192 features / 3 heads = 64 per head', 26)
    body += t(165, 110, 'X = LayerNorm(E)', 25, 'c-e', 'middle')
    body += matrix(92, 155, 180, 212, short_tokens, ['1', '2', '…', '192'], row=0)
    body += t(182, 406, '197 × 192', 29, 'c-e', 'middle')
    body += line(288, 259, 332, 259, 'c-e', 2.5)+line(332, 122, 332, 382, 'c-e', 2.5)
    for j, (name, color, y) in enumerate([('Q', 'c-q', 80), ('K', 'c-k', 207), ('V', 'c-v', 334)]):
        branch = arrow(332, y+36, 398, y+36, color)
        branch += matrix(431, y, 114, 69, ['', '', ''], ['', '', ''], color, labels=False)
        branch += t(488, y-12, '× W_'+name+' (192 × 64)', 22, color, 'middle')
        branch += t(582, y+43, '+ b_'+name, 25, color)
        branch += t(620, y+76, '64 values', 19, color, 'middle')
        branch += arrow(680, y+36, 759, y+36, color)
        branch += matrix(842, y-4, 160, 92, short_tokens, ['', '', '', ''], color, row=0, size=18)
        branch += t(1087, y+34, name, 31, color, 'middle')
        branch += t(1087, y+67, '197 × 64', 22, color, 'middle')
        body += branch if j == 0 else g(branch, j)
    add('real-cls-attention', 'The same input matrix feeds three learned projections', body,
        'One normalized matrix X feeds three different learned projections. Q, K and V each contain 197 rows of 64 features, in the same token order. This is one head; dots stand for omitted values.',
        'Do Q, K and V come from three different images?',
        'Trace all three arrows back to the same X. The learned weights differ. Each projection processes every row using the same weights within that projection.',
        'E is the 197×192 input from the previous slide; LayerNorm changes its values while preserving shape. '
        'Call the normalized matrix X. In row-vector notation Q=XW_Q+b_Q, K=XW_K+b_K, V=XW_V+b_V. '
        'Each W is 192×64, each bias has 64 coordinates and is broadcast to all 197 rows. These are three '
        'affine layers, equivalent to separate nn.Linear(192,64) operations for this head, without an added activation. '
        'PyTorch stores each corresponding weight as (64,192); the drawing uses the transposed matrix in XW notation. '
        'The checkpoint computes all three heads and Q/K/V jointly with a fused linear layer, then splits the result. '
        'Its three heads have distinct parameters. The matrix grids are schematic; dots omit numerical entries, '
        'and their equal dimensions do not imply equal values. CLS is first throughout, followed by P1 through P196.',
        '<img src="figures/vision1/model-input.png" alt="The same dog photograph" width="130">'
        '<p>X = LayerNorm(E), shape <strong>197 × 192</strong>. Rows: CLS, P1, …, P196.</p>'
        +mobile_rows(['Projection from the same X', 'Result'],
                     [['X W_Q + b_Q', 'Q: 197 × 64'], ['X W_K + b_K', 'K: 197 × 64'], ['X W_V + b_V', 'V: 197 × 64']])
        +'<p>Each W: 192 × 64; each bias: 64. One head, three different learned projections. Row identities stay the same.</p>')

    body = t(35, 33, 'Rows: receiving queries. Columns: source keys. CLS is first on both axes.', 26)
    body += t(164, 82, 'Q · 197 × 64', 27, 'c-q', 'middle')
    body += matrix(80, 138, 168, 192, tokens, features, 'c-q', row=0, size=18)
    body += t(280, 237, '×', 37)
    body += t(497, 82, 'Kᵀ · 64 × 197', 27, 'c-k', 'middle')
    body += matrix(353, 138, 288, 128, features, tokens, 'c-k', col=3, size=18)
    body += t(497, 310, 'Turn K’s rows into columns', 22, 'ink-2', 'middle')
    body += arrow(670, 225, 724, 225)+t(697, 191, '÷ 8', 23, 'ink-2', 'middle')
    body += t(959, 82, 'S · 197 × 197', 27, 'c-k', 'middle')
    body += matrix(815, 138, 288, 192, tokens, tokens, 'c-k', row=0, col=3, cell=(0,3), entry='s', size=18)
    body += g(t(35, 381, 'Highlighted cell: s(CLS, P63) = q(CLS) · k(P63) / 8', 29, 'c-q'), 1)
    body += t(35, 428, '197 queries × 197 keys = 38,809 matching scores in this head.', 27)
    add('real-attention-product', 'One query–key comparison fills one matrix cell', body,
        'Multiply Q by Kᵀ and divide by √64 = 8. Row CLS, column P63 compares the CLS query with patch 63’s key. Every query–key pair gets a score. These scores are not yet attention weights.',
        'Which query and which key produced the highlighted cell?',
        'Follow the highlighted row of Q and column of Kᵀ to row CLS, column P63 of S. The dot product sums 64 coordinate products. S can contain negative scores and need not be symmetric.',
        'The token order is CLS, P1, …, P196, so 196+1=197. Q has one 64-coordinate query per receiver; '
        'Kᵀ has one 64-coordinate key per source column. Contracting the shared 64 dimension produces '
        '197×197 entries, not a sum of the row counts. The highlighted entry uses all 64 coordinate products, '
        'scaled by 1/√64. For this dog, P63 is the crop in row 5, column 7 identified earlier. '
        'The entry is a learned query–key matching score, not a dog probability or a direct comparison of raw pixels. '
        'Q and K use different learned projections, so S(i,j) need not equal S(j,i). Dots and ellipses denote omitted entries.',
        '<p>Q (197 × 64) × Kᵀ (64 × 197) / 8 → <strong>S (197 × 197)</strong>.</p>'
        '<p>Rows are receiving queries; columns are source keys. Both axes follow CLS, P1, …, P196.</p>'
        '<p><strong>S[CLS,P63] = dot(q_CLS, k_P63) / 8.</strong> Sum 64 coordinate products.</p>'
        '<p>197 × 197 = 38,809 matching scores per head. They can be negative and are not probabilities.</p>')

    zoom_sources = ['CLS', 'P1', 'P2', '…', 'P63', '…', 'P196']
    zoom_indices = ['₀', '₁', '₂', None, '₆₃', None, '₁₉₆']

    def cls_strip(y, symbol, color, tint, labels=False):
        out = ''
        for j, (source_name, index) in enumerate(zip(zoom_sources, zoom_indices)):
            x, width = 510+j*610/7, 610/7
            out += rect(x, y, width, 55, color, tint, 0)
            out += t(x+width/2, y+36, symbol+index if index else '…', 30, color, 'middle')
            if labels:
                out += t(x+width/2, y-17, source_name, 23, 'ink-2', 'middle')
        return out

    body = t(225, 40, 'S · 197 × 197 scores', 29, 'c-k', 'middle')
    body += t(225, 89, 'source keys →', 23, 'ink-2', 'middle')
    body += matrix(90, 143, 270, 204, tokens, tokens, 'c-q', row=0, size=18)
    body += rect(90, 143, 270, 34, 'c-q', 'transparent', 0)
    zoom = t(815, 40, 'CLS row, enlarged', 29, 'c-q', 'middle')
    zoom += t(815, 78, 'We follow one query; every row gets an update.', 23, 'ink-2', 'middle')
    zoom += line(373, 143, 500, 143, 'c-q', 2.5)+line(373, 177, 500, 198, 'c-q', 2.5)
    zoom += t(433, 123, 'zoom', 21, 'c-q', 'middle')
    zoom += cls_strip(143, 's', 'c-q', 't-q', labels=True)
    body += g(zoom, 1)
    weights = arrow(691, 211, 691, 278, 'c-v')
    weights += t(721, 240, 'softmax', 29, 'c-v')
    weights += t(721, 269, 'across all 197 scores', 23, 'c-v')
    weights += cls_strip(290, 'a', 'c-v', 'transparent')
    weights += t(815, 382, 'a₀ + a₁ + … + a₁₉₆ = 1', 29, 'c-v', 'middle')
    body += g(weights, 2)
    body += t(225, 382, '1 CLS score + 196 patch scores', 23, 'c-q', 'middle')
    body += t(35, 435, 'Patch queries update patches → the next block’s CLS reads those updated patches.', 26, 'c-e')
    add('real-attention-cls-zoom', 'Follow the CLS row from scores to weights', body,
        'Earlier blocks update patches so later CLS queries can read their new features. In the final block, a CLS-only readout could compute just the CLS output, using all 197 incoming keys and values.',
        'If the classifier reads CLS, why compute the other score rows?',
        'We follow one row to explain the calculation. Each patch query needs its own scores and weights to update that patch. '
        'The next block computes its keys and values from these updated patches, which CLS can then read. '
        'For this one CLS message alone, only its 197 scores and all 197 value rows are needed. '
        'In the final block, if only CLS is used, patch outputs can be skipped in a specialized implementation. '
        'All incoming keys and values are still needed. The standard implementation shown computes every row. '
        'There are 196 patch scores plus the CLS self-score. Softmax normalizes all 197; there is no causal mask.',
        'The purple outline selects the first row of S, not its first column. The two connecting lines enlarge '
        'that same row without changing its contents or order. Here the receiving query is fixed to CLS, '
        'so s_j is shorthand for S[CLS,j] = dot(q_CLS,k_j)/8. Source 0 is CLS itself; sources 1 through 196 '
        'are the image patches P1 through P196. Ellipses omit scores from the drawing, but all 197 enter '
        'the softmax denominator: a_j = exp(s_j) / Σ_{k=0}^{196} exp(s_k). '
        'The resulting a_j is shorthand for A[CLS,j], the weight on source j’s 64-feature value row. '
        'To calculate this single message in isolation, q_CLS Kᵀ / 8 gives a 1×197 score row; '
        'its row-wise softmax multiplied by V gives a 1×64 message. Other query scores are not inputs to this row’s softmax. '
        'However, in blocks 1–11 of this model, the other query rows compute the patch updates needed by later blocks. '
        'After attention, residual additions and the per-row MLP, these updated patches supply the next block’s keys and values. '
        'Keeping only CLS at every block would change the model’s computation. '
        'A final-block optimization follows from the same equations: when the only readout is final CLS, '
        'one can compute only the CLS query and output in block 12, while still computing keys and values for all 197 incoming rows. '
        'The final attention projection, residual update, MLP and normalization can then operate on CLS alone. '
        'For the deterministic forward pass here, this preserves the class prediction up to numerical rounding. '
        'It does not apply unchanged to patch-level outputs or a readout that pools patch rows. '
        'The standard checkpoint implementation used in this lecture computes all rows, including in the final block; '
        'actual speed gains from a specialized path depend on the implementation and hardware. '
        'These are symbolic entries, not measured outputs. Every source is permitted because the full image '
        'is observed before its class label is predicted. The next slide applies the same operation to '
        'each of the remaining query rows; later we use the weights to mix V.',
        '<p><strong>Highlight row CLS in S (197 × 197), then enlarge that row.</strong></p>'
        +mobile_rows(['Source key', 'Score for the CLS query', 'After softmax'],
                     [['CLS itself', 's₀', 'a₀'], ['P1', 's₁', 'a₁'], ['P2', 's₂', 'a₂'],
                      ['…', '…', '…'], ['P63', 's₆₃', 'a₆₃'], ['…', '…', '…'], ['P196', 's₁₉₆', 'a₁₉₆']])
        +'<p>1 CLS self-score + 196 patch scores = <strong>197 scores</strong>. '
        'For this fixed CLS query, s_j = dot(q_CLS,k_j) / 8.</p>'
        '<p>Softmax across all 197 scores gives 197 weights: <strong>a₀ + a₁ + … + a₁₉₆ = 1</strong>. '
        'Each a_j weights the corresponding source value row.</p>'
        '<p><strong>Why compute the other rows?</strong> Each patch gets its own update. '
        'The next block’s CLS reads keys and values made from those updated patches.</p>'
        '<p><strong>Final-block exception:</strong> if the classifier uses only final CLS, '
        'a specialized implementation could compute only that output. It still needs all 197 incoming keys and values. '
        'The standard implementation here computes every row.</p>'
        '<p><strong>No causal mask:</strong> the whole image is available before predicting its class. '
        'The ellipses only shorten the drawing; no source is excluded from softmax.</p>')

    body = t(284, 37, 'S: matching scores', 29, 'c-k', 'middle')
    body += t(884, 37, 'A: attention weights', 29, 'c-v', 'middle')
    body += matrix(140, 114, 288, 204, tokens, tokens, 'c-k', row=0, cell=(0,3), entry='s', size=20)
    body += g(arrow(466, 209, 658, 209, 'c-v')+t(562, 166, 'softmax', 29, 'c-v', 'middle')
              +t(562, 251, 'across each row', 23, 'c-v', 'middle'), 1)
    body += g(matrix(740, 114, 288, 204, tokens, tokens, 'c-v', row=0, cell=(0,3), entry='a', size=20), 1)
    body += g(t(884, 369, 'Highlighted CLS row: sum = 1', 25, 'c-v', 'middle'), 1)
    body += t(35, 420, 'a(CLS, P63): how much weight CLS gives to P63’s value row.', 29, 'c-v')
    add('real-attention-weights', 'Turn each query’s 197 scores into 197 source weights', body,
        'Apply softmax across each score row. The CLS row becomes weights over CLS and all 196 patches, summing to one. Each other query gets its own weight row. These weights tell us how to mix the value rows.',
        'What are the alternatives in this softmax: animal classes or source rows?',
        'They are sources, including CLS itself. a(CLS,P63) is normalized against all 197 scores in the CLS row. Every row sums to one; columns need not.',
        'A has the same 197×197 shape and token order as S. For receiver i and source j, '
        'A[i,j] = exp(S[i,j]) / Σ_k exp(S[i,k]), where k runs over all 197 source rows. '
        'A is nonnegative, and each receiver row sums to one in the deterministic forward pass shown. '
        'Training-time attention dropout, when used, is a subsequent operation and is not drawn here. '
        'The highlighted CLS/P63 weight determines the coefficient on v(P63) in the CLS message. '
        'Row-wise softmax does not generally make A symmetric or make its columns sum to one. '
        'The later classifier softmax is a different operation over 1,000 image labels. Grid entries are schematic.',
        '<p>S (197 × 197) → softmax across each row → A (197 × 197).</p>'
        '<p>The CLS row contains weights for CLS, P1, …, P196. <strong>All 197 weights sum to 1.</strong></p>'
        '<p>A[CLS,P63] = exp(S[CLS,P63]) / sum(exp(S[CLS,k])) over all 197 source rows k.</p>'
        '<p>It is the coefficient on P63’s value row in the message to CLS. The sources are image rows, not animal classes.</p>')

    text_positions = ['t₁', 't₂', 't₃', 't₄', 't₅', 't₆']
    body = t(280, 36, 'Text: predict the next token', 28, 'c-q', 'middle')
    body += t(862, 36, 'ViT: classify the whole image', 28, 'c-v', 'middle')
    body += matrix(136, 107, 288, 204, text_positions, text_positions, 'c-q', allowed=lambda r,c:c<=r)
    body += matrix(718, 107, 288, 204, tokens, tokens, 'c-v', allowed=lambda r,c:True)
    body += t(280, 353, 'Read the current and earlier positions', 24, 'c-q', 'middle')
    body += t(862, 353, 'Read every row, including itself', 24, 'c-v', 'middle')
    body += t(35, 399, 'Filled cell = permitted connection. × = blocked. This shows access, not weight size.', 25)
    body += g(t(35, 439, 'No causal mask here: CLS can read P196, and P196 can read CLS.', 28, 'c-v'), 1)
    add('real-attention-mask', 'Every image row can read every image row', body,
        'Next-token prediction hides later text positions. Here the complete photograph is available before classification. Every query can use every key and value, including itself. The ViT attention matrix has no causal triangle removed.',
        'If CLS is the first row, what would a causal mask let it read?',
        'Only itself. That would prevent it from gathering patch information. This classifier uses full attention because all image patches are available for the image-label prediction.',
        'Rows are receiver positions and columns are sources in both access diagrams. t₁…t₆ illustrate six text '
        'positions in a causal decoder, where later source positions are masked before softmax. '
        'The image axes abbreviate 197 rows with ellipses; every entry in the full 197×197 matrix is allowed. '
        'The top-left image entry allows CLS to read itself, the top-right allows CLS to read P196, '
        'and the bottom-left allows P196 to read CLS. Allowed access does not imply equal or large attention weights. '
        'The fixed-size, fully populated image input used here also has no padding mask. Other architectures may use '
        'other masks, but no causal mask is part of this classifier’s saved forward pass.',
        '<p><strong>Text generation:</strong> receiving position t_i may read source positions t_j only when j ≤ i.</p>'
        '<p><strong>This image classifier:</strong> every receiving row may read CLS and P1 through P196.</p>'
        '<p>There is no causal mask. The whole photograph is available before predicting its label.</p>'
        '<p>CLS can read P196; P196 can read CLS. Every row can also read itself. Permitted access does not imply equal weights.</p>')

    from vision1_cls_value_message import build_cls_value_message
    result.update(build_cls_value_message(b, matrix))

    body = t(35, 33, 'Keep the source order aligned: CLS, P1, …, P196 in both A and V.', 27)
    body += t(221, 85, 'A · 197 × 197', 28, 'c-k', 'middle')
    body += matrix(80, 141, 282, 192, tokens, tokens, 'c-k', row=0, col=3, cell=(0,3), entry='a', size=18)
    body += t(405, 244, '×', 37)
    body += t(601, 85, 'V · 197 × 64', 28, 'c-v', 'middle')
    body += matrix(511, 141, 180, 192, tokens, features, 'c-v', row=3, size=18)
    body += arrow(723, 237, 819, 237, 'c-v')
    body += t(1000, 85, 'H · 197 × 64', 28, 'c-v', 'middle')
    body += matrix(910, 141, 180, 192, tokens, features, 'c-v', row=0, size=18)
    body += g(t(35, 384, 'h(CLS) = a(CLS, CLS) v(CLS) + … + a(CLS, P196) v(P196)', 26, 'c-v'), 1)
    body += t(35, 432, 'Q and K choose the weights. V supplies the features carried into each message.', 28)
    add('real-attention-values', 'Repeat for every query: H has 197 message rows', body,
        'Multiply A by V. For CLS, multiply each source value row by its attention weight, then add the 197 weighted rows. The result is one 64-feature message. Do this for every query to form H.',
        'Where does P63’s information enter the message to CLS?',
        'Match column P63 of A with row P63 of V. Its weight multiplies all 64 coordinates of that source’s value row. Add the contributions from all sources.',
        'A is (197,197), V is (197,64), and H=AV is (197,64). The contracted 197 dimension is the shared source axis. '
        'The remaining row axis identifies the receiving query. The highlighted A[CLS,P63] coefficient multiplies '
        'v(P63), and that weighted row contributes to h(CLS). Values come from the same normalized input matrix '
        'through their own learned projection; they are not pixel crops or class probabilities. '
        'Every query, including every patch query, receives a message. H is one head’s message matrix, not yet '
        'the residual-updated sequence. Next join all three heads, project back to 192 features, and add the input E.',
        '<p>A (197 × 197) × V (197 × 64) → <strong>H (197 × 64)</strong>.</p>'
        '<p>A[CLS,P63] multiplies all 64 features in V[P63]. Add the weighted value rows from all 197 sources to get H[CLS].</p>'
        '<p>Q and K determine the weights; V supplies the message features. Every query gets one 64-feature message.</p>'
        '<p>Next: join the three heads, project to 192 features, and add E.</p>')
    return result
