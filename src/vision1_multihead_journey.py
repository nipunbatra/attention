"""Finish one head, then expose three parallel paths and their join before the MLP."""
import json


def build_multihead_journey(b):
    t, g, rect, arrow, line, frame, mobile_rows = (b[k] for k in
        ['t', 'g', 'rect', 'arrow', 'line', 'frame', 'mobile_rows'])
    data = json.loads((b['ASSETS']/'multihead-cls-trace.json').read_text())
    result = {}
    colors = ['c-q', 'c-k', 'c-v']
    superscripts = ['¹', '²', '³']
    evidence = (' <a href="figures/vision1/multihead-cls-trace.json">Measured three-head CLS trace</a> · '
                '<a href="notebooks/vision/trace_multihead_messages.py">Reproduce the head projections and concatenation</a>.')

    def preview(values, count=3):
        return '['+', '.join(f'{v:.3f}'.replace('-', '−') for v in values[:count])+', …]'

    def box(x, y, w, lines, color='c-e', height=80, size=25):
        out = rect(x, y, w, height, color, 'transparent', 5)
        for j, text in enumerate(lines):
            out += t(x+w/2, y+height/2+8+(j-(len(lines)-1)/2)*30, text, size, color, 'middle')
        return out

    def grid(x, y, w, h, color, labels=False, cls_column=False):
        out = rect(x, y, w, h, color, 'transparent', 0)
        out += f'<rect x="{x}" y="{y}" width="{w/4 if cls_column else w}" height="{h if cls_column else h/4}" fill="var(--{color})" fill-opacity=".14"/>'
        for i in range(1,4):
            out += line(x+i*w/4, y, x+i*w/4, y+h, 'line', 1)
            out += line(x, y+i*h/4, x+w, y+i*h/4, 'line', 1)
        for r in range(4):
            for c in range(4):
                out += t(x+(c+.5)*w/4, y+(r+.5)*h/4+4, '·', 15, color, 'middle')
        if labels:
            for j, name in enumerate(['CLS','P1','…','P196']):
                out += t(x-10, y+(j+.5)*h/4+5, name, 17, color if j==0 else 'ink-2', 'end')
        return out

    def add(key, title, body, caption, question, point, prose, mobile):
        notes = question+'\n'+point
        for meta in b['FRAMES']:
            if meta['id'] == key:
                meta.update(title=title, caption=caption, notes=notes)
        result[key] = frame(key, title, body, caption, notes, prose+evidence, mobile)

    body = t(35, 65, 'Head 1 is complete.', 38, 'c-q', weight=600)
    body += t(35, 127, 'Q and K → source weights → mix V → a message for every row.', 29)
    body += box(35, 190, 485, ['197 input rows', '197 messages, each 64 features'], 'c-q', 102, 29)
    body += arrow(546, 241, 615, 241)
    body += box(644, 190, 480, ['Now use three heads in parallel', 'Each reads the same input rows'], 'c-v', 102, 29)
    body += t(35, 370, 'First combine their messages. Then continue to the MLP.', 33)
    add('real-heads-intro', 'From one completed head to three parallel heads', body,
        'We have finished one head’s attention calculation. This model has three heads. Each starts from the same input rows and computes its own messages. Next, follow those parallel paths and combine their outputs before the MLP.',
        'Have we finished head 1, or already updated the input E?',
        'We have its message matrix H¹. The output projection and residual addition still follow. Introduce the other heads only now; they run alongside head 1 and do not consume its output.',
        'This is a topic break within the same real-image forward pass. One head has produced H¹ of shape '
        '197×64, including a 1×64 message for CLS. The chosen checkpoint has three heads. To continue its '
        'actual forward pass, compute the other two messages, concatenate all three, project to 192 features '
        'and add E. The MLP follows this attention residual. Heads are parallel; Transformer blocks are sequential.',
        '<p><strong>Head 1 is complete:</strong> Q/K → weights → weighted V → H¹ (197 × 64).</p>'
        '<p>Now follow three parallel heads from the same 197 × 192 input. Combine their messages, project, add E, then continue to the MLP.</p>')
    result['real-heads-intro'] = result['real-heads-intro'].replace(
        'class="frame vp-frame', 'class="frame vp-frame vp-topic-break', 1)

    body = t(35, 38, '196 patch rows + one CLS row = 197 rows', 28)
    body += t(884, 38, 'Each Q/K/V: 197 × 64', 24, 'ink-2', 'middle')
    body += t(156, 151, 'X = LayerNorm(E)', 23, 'c-e', 'middle')
    body += grid(87, 180, 138, 160, 'c-e', labels=True)
    body += t(156, 377, '197 × 192', 27, 'c-e', 'middle')
    body += line(235, 260, 272, 260, 'c-e', 2.5)+line(272, 104, 272, 354, 'c-e', 2.5)
    for h, color in enumerate(colors):
        y = 55+h*125
        branch = arrow(272, y+49, 310, y+49, color)
        branch += box(320, y+15, 235, ['Head '+str(h+1), 'own Q/K/V projections'], color, 68, 21)
        branch += line(565, y+49, 583, y+49, color, 2)
        branch += line(583, y+49, 583, y+101, color, 2)+line(583, y+101, 1065, y+101, color, 2)
        for x, name in [(650,'Q'),(835,'K'),(1020,'V')]:
            branch += t(x+45, y+5, name+superscripts[h], 26, color, 'middle')
            branch += grid(x, y+19, 90, 62, color, labels=x==650)
            branch += arrow(x+45, y+101, x+45, y+84, color)
        body += branch if h==0 else g(branch, h)
    body += t(35, 436, 'One shared CLS input; each head projects it differently.', 29, 'c-e')
    add('real-heads-qkv', 'The same rows feed three sets of Q, K and V', body,
        'All three heads read the same normalized matrix X. Each has its own Q, K and V projection weights and biases. Every output matrix has 197 rows and 64 features. The highlighted first row is always CLS.',
        'Do we create three CLS tokens or divide the patches between the heads?',
        'Neither. Every head reads all 197 rows, including the same 192-feature CLS input. Each applies a different learned projection. Superscripts 1, 2 and 3 name heads.',
        'For head h, Qʰ=XW_Qʰ+b_Qʰ, Kʰ=XW_Kʰ+b_Kʰ and Vʰ=XW_Vʰ+b_Vʰ. '
        'Each matrix W has row-vector shape 192×64, and each bias has 64 coordinates. '
        'Every projection reads all 192 input features; the heads do not split the patch list or simply '
        'take separate 64-coordinate slices of X. Within each projection, weights are shared across rows. '
        'Across heads, the parameters differ. The checkpoint evaluates these projections with one fused '
        'Linear(192,576), then reshapes into Q/K/V and three heads. The drawing exposes the equivalent '
        'separate operations. The same CLS parameter and its position contribute one row of E, which '
        'becomes one row of X and is read by every head.',
        '<p>X = LayerNorm(E), <strong>197 × 192</strong>: CLS plus P1 through P196.</p>'
        +mobile_rows(['Head', 'Independent projections', 'Each result'],
                     [[str(h+1), 'Q'+superscripts[h]+', K'+superscripts[h]+', V'+superscripts[h], '197 × 64'] for h in range(3)])
        +'<p>Each projection uses its own W (192 × 64) and bias (64). All heads read the same CLS row and all patches.</p>')

    body = ''
    for h, color in enumerate(colors):
        y = h*132
        s = superscripts[h]
        lane = t(30, y+62, 'Head '+str(h+1), 25, color)
        for x, w, name, shape in [(145,65,'Q'+s,'197 × 64'),(250,82,'(K'+s+')ᵀ','64 × 197'),
                                  (403,75,'S'+s,'197 × 197'),(625,75,'A'+s,'197 × 197'),
                                  (775,75,'V'+s,'197 × 64'),(1002,110,'H'+s,'197 × 64')]:
            lane += t(x+w/2, y+23, name, 25, color, 'middle')+grid(x, y+35, w, 50, color, cls_column=x==250)
            lane += t(x+w/2, y+111, shape, 17, color, 'middle')
        lane += t(226, y+69, '×', 27)+arrow(343, y+60, 387, y+60, color)
        lane += t(365, y+31, '÷ 8', 18, color, 'middle')
        lane += arrow(493, y+60, 609, y+60, color)+t(551, y+37, 'softmax', 21, color, 'middle')
        lane += t(735, y+69, '×', 29)+arrow(870, y+60, 982, y+60, color)
        lane += t(926, y+37, 'mix values', 19, color, 'middle')
        body += lane if h==0 else g(lane, h)
    body += t(35, 435, 'Each head has its own matching scores, attention weights and messages.', 28)
    add('real-heads-messages', 'Each head repeats the complete attention calculation', body,
        'Within each head, compare Q with K, scale the scores, apply softmax across each row, then multiply by that head’s V. Each path returns 197 messages of 64 features. The heads operate in parallel.',
        'Does head 2 use head 1’s attention weights or its value matrix?',
        'No. Keep each colored path intact: Qʰ and Kʰ produce Aʰ, which mixes Vʰ to make Hʰ. Every head has its own 197×197 weight matrix.',
        'For each head h, Sʰ=Qʰ(Kʰ)ᵀ/√64, Aʰ=softmax(Sʰ) over source columns, and Hʰ=AʰVʰ. '
        'Every head uses full image attention without a causal mask. Each Hʰ has the same receiver-row '
        'order: CLS, P1, …, P196. Equal shapes do not imply equal values. Each drawing is schematic; '
        'its highlighted first row is CLS except in the transposed K diagram, where the first column '
        'is the CLS key. The next slide compares the three measured CLS message rows for the same dog.',
        mobile_rows(['Head', 'Score and weight matrices', 'Message matrix'],
                    [[str(h+1), 'Q'+superscripts[h]+'(K'+superscripts[h]+')ᵀ / 8 → softmax → A'+superscripts[h]+' (197 × 197)',
                      'A'+superscripts[h]+' V'+superscripts[h]+' → H'+superscripts[h]+' (197 × 64)'] for h in range(3)])
        +'<p>The three paths use different learned projections and run in parallel from the same X.</p>')

    body = t(35, 37, 'Follow the same dog and the same receiving CLS row.', 28)
    body += t(195, 136, 'CLS row in X · 1 × 192', 25, 'c-e', 'middle')
    body += box(35, 170, 320, [preview(data['cls_normalized']), 'same input to every head'], 'c-e', 82, 23)
    body += line(365, 211, 400, 211, 'c-e', 2.5)+line(400, 100, 400, 352, 'c-e', 2.5)
    for h, color in enumerate(colors):
        y = 71+h*126
        branch = arrow(400, y+29, 665, y+29, color)
        branch += t(533, y+6, 'Head '+str(h+1)+' attention', 24, color, 'middle')
        branch += box(682, y, 431, [preview(data['heads'][h]['cls_message'])], color, 60, 29)
        branch += t(897, y+92, 'CLS message · 1 × 64', 23, color, 'middle')
        body += branch if h==0 else g(branch, h)
    body += t(35, 339, 'One CLS input row.', 28, 'c-e')
    body += t(35, 385, 'Three sets of learned projections.', 24)
    add('real-heads-cls', 'One CLS input produces three different messages', body,
        'CLS is the same input row for all heads. Their learned projections produce different queries, keys and values, so their attention computations produce different messages. These measured messages each contain 64 features. There is still one CLS token.',
        'What is shared, and what differs, across these three CLS paths?',
        'The normalized CLS input is shared. The projection parameters, projected queries/keys/values, source weights and resulting messages differ. Each message also depends on the patch values and keys from this photograph.',
        'The displayed vectors are measured activations for the dog in block 1. Ellipses omit coordinates; '
        'the shared input has 192 features and each output has 64. A CLS message is not separately stored '
        'as a learned token parameter: it is computed anew from the current input. There is one stored '
        'CLS parameter vector for the model. After projection and residual addition there will be one '
        'updated CLS row, not three separate CLS rows. The trace independently verifies each head’s '
        'Q/K/V slices using the same normalized CLS row.',
        '<p>Shared normalized CLS row: '+preview(data['cls_normalized'])+' (1 × 192).</p>'
        +mobile_rows(['Head', 'Measured CLS message', 'Shape'],
                     [[str(h['head']), preview(h['cls_message']), '1 × 64'] for h in data['heads']])
        +'<p>One CLS token. Different head projections and attention computations produce three different messages.</p>')

    body = ''
    for h, color in enumerate(colors):
        x = 35+h*390
        body += t(x+155, 38, 'Head '+str(h+1)+' CLS message', 25, color, 'middle')
        body += box(x, 67, 310, [preview(data['heads'][h]['cls_message'], 2)], color, 72, 27)
        body += t(x+155, 176, '1 × 64', 28, color, 'middle')
    joined = ''
    for h, color in enumerate(colors):
        x = 75+h*336
        joined += arrow(190+h*390, 187, x+168, 265, color)
        joined += box(x, 281, 336, [preview(data['heads'][h]['cls_message'], 2)], color, 72, 28)
        joined += t(x+168, 390, 'features '+str(h*64+1)+'–'+str((h+1)*64), 24, color, 'middle')
    body += g(joined, 1)
    body += g(t(580, 438, 'One joined row · 1 × 192     (64 + 64 + 64)', 30, 'c-e', 'middle'), 1)
    add('real-heads-concat', 'Concatenate the three CLS messages', body,
        'Keep all three messages by placing their coordinates side by side. The joined CLS row has 192 features: 64 from head 1, then 64 from head 2, then 64 from head 3. Concatenation has no learned parameters.',
        'Does concatenation add corresponding numbers from different heads?',
        'No. Follow each vector into its own feature range. The coordinates retain their values and order. Joining happens across the feature dimension, keeping one receiver row.',
        'The operation is torch.cat([h_cls_head1,h_cls_head2,h_cls_head3], dim=-1). '
        'No softmax, averaging or learned layer is applied during concatenation itself. The trace verifies '
        'all 192 coordinates equal the three complete 64-coordinate segments in head order. '
        'For all queries, concatenate H¹, H² and H³ along the feature axis to produce J with shape '
        '197×192. The row count stays 197. CLS remains its first row.',
        mobile_rows(['Joined feature range', 'Source'], [['1–64','Head 1 message'],['65–128','Head 2 message'],['129–192','Head 3 message']])
        +'<p>Concatenation appends coordinates: (1 × 64), (1 × 64), (1 × 64) → <strong>1 × 192</strong>.</p>'
        '<p>For all queries: concatenate H¹, H² and H³ → J (197 × 192). No learned parameters in this step.</p>')

    # Match the text-attention diagram: retain E on a visible bypass while the
    # attention branch computes an update. All boxes here represent all rows.
    body = box(22, 174, 110, ['Original', 'E'], 'c-e', 92, 23)
    body += t(77, 295, '197 × 192', 21, 'c-e', 'middle')
    body += arrow(138, 220, 157, 220)
    body += box(165, 183, 120, ['LayerNorm', 'X'], 'c-e', 74, 23)
    body += line(285, 220, 300, 220, 'c-e', 2)
    body += line(300, 119, 300, 319, 'c-e', 2)
    for h, color in enumerate(colors):
        y = 84+h*100
        body += arrow(300, y+35, 325, y+35, color)
        body += box(334, y, 145, ['Head '+str(h+1), 'H'+superscripts[h]+' · 197 × 64'], color, 70, 21)
        body += arrow(486, y+35, 534, 190+h*30, color)
    body += t(611, 148, 'Concatenate', 24, 'c-v', 'middle')
    for h, color in enumerate(colors):
        body += box(542+h*46, 168, 46, ['H'+superscripts[h]], color, 104, 23)
    body += t(611, 306, 'J · 197 × 192', 22, 'c-v', 'middle')
    projection = arrow(688, 220, 708, 220)
    projection += box(716, 183, 146, ['Linear', '(192,192)'], 'c-v', 74, 23)
    projection += t(789, 298, '× W_O + b_O', 21, 'c-v', 'middle')
    projection += arrow(868, 220, 886, 220, 'c-v')
    projection += box(894, 183, 70, ['ΔE'], 'c-v', 74, 30)
    projection += t(929, 153, 'Update', 23, 'c-v', 'middle')
    body += g(projection, 1)
    residual = line(77, 174, 77, 48, 'c-e', 3, '7 5')
    residual += line(77, 48, 1006, 48, 'c-e', 3, '7 5')
    residual += arrow(1006, 48, 1006, 191, 'c-e')
    residual += t(550, 30, 'Skip path: carry the original E unchanged', 26, 'c-e', 'middle')
    residual += arrow(970, 220, 980, 220, 'c-v')
    residual += '<circle cx="1006" cy="220" r="24" fill="var(--card)" stroke="var(--c-e)" stroke-width="2.5"/>'
    residual += t(1006, 230, '+', 35, 'c-e', 'middle')
    residual += arrow(1032, 220, 1044, 220, 'c-e')
    residual += box(1052, 174, 87, ['U'], 'c-e', 92, 34)
    residual += t(1095, 151, 'Updated', 22, 'c-e', 'middle')
    residual += t(1040, 338, 'U = E + ΔE', 27, 'c-e', 'middle')
    residual += t(1040, 371, 'both 197 × 192', 22, 'c-e', 'middle')
    body += g(residual, 2)
    body += t(35, 386, 'Each H contains one message per row: CLS, P1, …, P196.', 25)
    body += t(35, 432, 'The same residual pattern as text: original embedding + context update.', 28)
    add('real-cls-message', 'Keep the embedding; add the context from attention', body,
        'Follow two paths from E. The heads produce messages; concatenate and project them to make ΔE. The skip path carries E directly to addition. U = E + ΔE is the contextualized representation passed to the MLP.',
        'Which operation learns to combine the head features, and which row gets the residual update?',
        'Linear(192,192) learns the combination. Add the result to E, before the attention branch’s LayerNorm. The MLP will receive this updated row through its own LayerNorm.',
        'For the full sequence, J=Concat(H¹,H²,H³) has shape 197×192. The output projection is '
        'Delta=JW_O+b_O, with W_O shaped 192×192 and a 192-coordinate bias. This affine layer '
        'has no additional activation; it can mix coordinates from every head into each output feature. '
        'The residual is U=E+Delta, preserving 197×192. The diagram shows the same residual pattern '
        'as the text-attention lecture: retain the original representations and add retrieved context. '
        'E contains all 196 patch rows and CLS, including position, for the same dog. At later blocks, '
        'E denotes that block’s input representations. The skip path bypasses LayerNorm as well as '
        'the head computations. Every row receives its own update, including CLS. '
        'Next the MLP branch computes U+MLP(LayerNorm(U)); it is separate from this attention output '
        'projection. The original photograph has not been re-encoded and the row count has not changed.',
        '<p>Concatenate heads → J (197 × 192) → Linear(192,192) → Delta (197 × 192).</p>'
        '<p><strong>U = E + Delta</strong>, still 197 × 192.</p>'
        '<p>The skip path carries the original E, before LayerNorm, directly to the addition.</p>'
        '<p>Next: LayerNorm(U) → MLP → add U. The output projection and MLP are different learned layers.</p>')

    body = b['image'](35, 14, 132, 88)
    body += t(190, 48, 'Same dog · block 1 · zoom in on the CLS row', 29)
    body += t(190, 91, 'Every vector below contains 192 numbers.', 26, 'ink-2')
    body += t(184, 198, 'Original CLS embedding', 25, 'c-e', 'middle')
    body += box(35, 218, 298, [preview(data['cls_input'], 2), 'e₀ · 1 × 192'], 'c-e', 90, 26)
    body += t(622, 132, 'After concatenation + projection', 24, 'c-v', 'middle')
    body += box(454, 150, 336, [preview(data['cls_projected_update'], 2), 'Δe₀ · attention update'], 'c-v', 76, 25)
    addition = arrow(622, 232, 622, 253, 'c-v')
    addition += arrow(345, 277, 587, 277, 'c-e')
    addition += t(457, 304, 'skip path', 22, 'c-e', 'middle')
    addition += '<circle cx="622" cy="277" r="25" fill="var(--card)" stroke="var(--c-e)" stroke-width="2.5"/>'
    addition += t(622, 288, '+', 35, 'c-e', 'middle')
    addition += arrow(653, 277, 819, 277, 'c-e')
    addition += t(978, 198, 'Contextualized CLS', 25, 'c-e', 'middle')
    addition += box(831, 218, 295, [preview(data['cls_after_residual'], 2), 'u₀ · 1 × 192'], 'c-e', 90, 26)
    body += g(addition, 1)
    body += g(t(580, 373, 'First coordinate: −0.704 + 1.167 ≈ 0.463', 30, 'c-e', 'middle'), 1)
    body += g(t(35, 432, 'Next: MLP → later blocks → final CLS → class scores.', 29), 2)
    add('real-cls-residual', 'The dog’s CLS keeps its input and gains context', body,
        'Add the projected attention update coordinate by coordinate to the original CLS embedding. This produces a contextualized CLS row of the same width. It continues through the MLP and later blocks before the classifier reads the final summary.',
        'Is the attention update already the contextualized embedding, or do we still add something?',
        'The update is Δe₀. The contextualized row is u₀=e₀+Δe₀. Add the original block input, before LayerNorm. The 192 output features are a representation; class scores are computed later.',
        'These are measured values from the same dog forward pass, rounded to three decimals. '
        'The first two residual additions are −0.704+1.167≈0.463 and −0.067−0.051≈−0.118. '
        'All 192 coordinates are added in this way. For a patch row, the same formula updates that '
        'patch’s representation using its own head messages; for CLS it updates the image-summary row. '
        'The illustration follows block 1. Its output is an intermediate representation, not the final '
        'class prediction. The next MLP takes LayerNorm(U), transforms features, and adds its own '
        'update to U. Subsequent blocks refine these representations before final CLS classification.',
        '<p>Original CLS: '+preview(data['cls_input'],2)+' (1 × 192).</p>'
        '<p>Projected attention update: '+preview(data['cls_projected_update'],2)+' (1 × 192).</p>'
        '<p><strong>Add coordinate by coordinate:</strong> '+preview(data['cls_after_residual'],2)+' (1 × 192).</p>'
        '<p>Same dog → MLP → later blocks → final CLS → class scores.</p>')
    return result
