"""Follow one CLS weight row through the value matrix to its 64-feature message."""
import base64
import json


def build_cls_value_message(b, matrix):
    t, g, rect, arrow, line, image, frame, mobile_rows = (b[k] for k in
        ['t', 'g', 'rect', 'arrow', 'line', 'image', 'frame', 'mobile_rows'])
    result = {}
    photo = 'data:image/png;base64,' + base64.b64encode(
        (b['ASSETS']/'model-input.png').read_bytes()).decode()
    patch = json.loads((b['ASSETS']/'real-patch-path.json').read_text())
    trace = json.loads((b['ASSETS']/'real-classifier-path.json').read_text())
    head_trace = json.loads((b['ASSETS']/'multihead-cls-trace.json').read_text())
    examples = head_trace['value_message_examples']
    a63 = trace['cls_head1_weights'][63]
    v63 = patch['head1_v']
    message = trace['previews']['cls_head1_message']
    tokens = ['CLS', 'P1', '…', 'P63', '…', 'P196']
    features = ['1', '2', '…', '64']
    evidence = (' <a href="figures/vision1/real-patch-path.json">Saved P63 value row</a> · '
                '<a href="figures/vision1/real-classifier-path.json">Saved CLS weights and message</a> · '
                '<a href="figures/vision1/multihead-cls-trace.json">Saved source contributions and receiver messages</a> · '
                '<a href="notebooks/vision/trace_multihead_messages.py">Verified forward computation</a>.')

    def number(value, places=3):
        return f'{value:.{places}f}'.replace('-', '−')

    def crop(x, y, side, index=63):
        row, col = divmod(index-1, 14)
        return (f'<svg x="{x}" y="{y}" width="{side}" height="{side}" '
                f'viewBox="{col*16} {row*16} 16 16" overflow="hidden">'
                + image(0, 0, 224, 224, photo) + '</svg>')

    def strip(x, y, w, entries, color='c-v', height=56, size=27, selected=None):
        out = ''
        for j, entry in enumerate(entries):
            cell = w/len(entries)
            out += rect(x+j*cell, y, cell, height, color, 'transparent', 0)
            if j == selected:
                out += (f'<rect x="{x+j*cell}" y="{y}" width="{cell}" height="{height}" '
                        f'fill="var(--{color})" fill-opacity=".14"/>')
            out += t(x+(j+.5)*cell, y+height/2+9, entry, size, color, 'middle')
        return out

    def add(key, title, body, caption, notes, prose, mobile):
        result[key] = frame(key, title, body, caption, notes, prose+evidence, mobile)

    from vision1_focus_common import Figures
    box = Figures(b).box

    # Recall the operation already used for text before following its image values.
    body = t(35, 34, 'Choose a receiver. Mix source values to make a message for that receiver.', 28)
    body += t(35, 102, 'Text · receiver: bank', 29, 'c-q')
    body += box(35, 130, 155, 'river', 'c-v', h=64, size=29)
    body += box(211, 130, 210, ['other allowed', 'tokens'], 'c-v', h=64, size=23)
    body += arrow(435, 162, 493, 162, 'c-v')
    body += box(505, 130, 250, ['Mix their values', 'using bank’s weights'], 'c-v', h=64, size=24)
    body += arrow(768, 162, 829, 162, 'c-v')
    body += box(843, 130, 280, 'Message for bank', 'c-q', h=64, size=28)
    vision = t(35, 270, 'Image · receiver: CLS', 29, 'c-q')
    for x, index in [(35, 1), (125, 63), (215, 196)]:
        vision += crop(x, 292, 64, index)+t(x+32, 383, 'P'+str(index), 23, 'c-v', 'middle')
    vision += t(303, 334, '…', 30, 'c-v')+box(350, 292, 71, 'CLS', 'c-v', h=64, size=24)
    vision += arrow(435, 324, 493, 324, 'c-v')
    vision += box(505, 292, 250, ['Mix their values', 'using CLS’s weights'], 'c-v', h=64, size=24)
    vision += arrow(768, 324, 829, 324, 'c-v')
    vision += box(843, 292, 280, 'Message for CLS', 'c-q', h=64, size=28)
    body += g(vision, 1)
    body += t(505, 431, 'Same weighted-sum operation. Different receiver and sources.', 25, 'ink-2', 'middle')
    add('real-message-text-analogy', 'A message for CLS works like a message for bank', body,
        'In text, bank receives a weighted mixture of allowed token values. Here, CLS receives a weighted mixture of image-token values, including its own. The query chooses the receiver; the values supply the message.',
        'What corresponds to bank in the calculation we are following?\nCLS is our selected receiver. River was one text source; P63 is one image source. Both calculations sum weighted value vectors. CLS also has its own value row.',
        'Recall <a href="attention.html#s16-flow-frame">the text-attention message diagram</a>: '
        'bank at position 7 uses its query to assign weights to positions 1–7, including river. '
        'Those weights mix value vectors into a message for bank. The dog calculation uses the '
        'same operation with CLS as the receiver and all 197 image-token rows as permitted sources. '
        'The photo crops identify source rows; we mix their projected value vectors, not their pixels. '
        'CLS has no crop but participates as both a possible receiver and a source. '
        'This diagram describes the computation, not a claim that any named source necessarily gets a large weight. '
        'A message is an intermediate vector. The output projection and residual addition will later use it '
        'to update the receiving token’s representation. The token’s identity stays the same.',
        mobile_rows(['Receiver', 'Sources', 'Result'], [
            ['bank', 'river and other causally allowed tokens', 'weighted value mixture for bank'],
            ['CLS', 'P1–P196 and CLS itself', 'weighted value mixture for CLS']])
        +'<p>Same attention operation: use the receiver’s weights to mix source value vectors.</p>')

    body = t(35, 34, 'Same dog · block 1 · head 1', 29)
    body += image(35, 106, 200, 200, photo)
    for j in range(1, 14):
        body += line(35+j*200/14, 106, 35+j*200/14, 306, 'card', .6)
        body += line(35, 106+j*200/14, 235, 106+j*200/14, 'card', .6)
    body += rect(35+6*200/14, 106+4*200/14, 200/14, 200/14, 'c-q', 'transparent', 0)
    body += t(135, 335, '196 patches + CLS', 22, 'ink-2', 'middle')
    body += arrow(245, 210, 300, 210)
    body += t(433, 82, 'X = LayerNorm(E)', 23, 'c-e', 'middle')
    body += matrix(368, 121, 130, 184, tokens, ['1', '2', '…', '192'], 'c-e', row=3, size=17)
    body += t(433, 335, '197 × 192', 25, 'c-e', 'middle')
    body += g(arrow(519, 210, 547, 210, 'c-v')+rect(558, 170, 174, 80, 'c-v', 'transparent')
              +t(645, 203, 'Linear(192,64)', 22, 'c-v', 'middle')
              +t(645, 234, '× W_V + b_V', 21, 'c-v', 'middle')
              +arrow(738, 210, 768, 210, 'c-v'), 1)
    body += g(t(947, 82, 'V: one value row per source', 23, 'c-v', 'middle')
              +matrix(825, 121, 244, 184, tokens, features, 'c-v', row=3, size=17)
              +t(947, 335, '197 × 64', 25, 'c-v', 'middle'), 1)
    body += g(crop(35, 360, 60)+t(124, 389, 'P63 → v₆₃ = ['
              +number(v63[0])+', '+number(v63[1])+', …, '+number(v63[-1])+']', 28, 'c-v')
              +t(124, 425, '64 learned features · rounded values from this dog', 23, 'ink-2'), 2)
    add('real-cls-values-origin', 'The dog’s feature rows become value rows', body,
        'Use the same normalized input X that produced Q and K. The value projection makes 64 features for each of its 197 rows. P63’s crop identifies one source; its value row contains learned features.',
        'Where does row P63 of V come from?\nFollow P63’s normalized 192-feature row through the shared value projection. It produces 64 features. CLS has a value row too, although it has no image crop.',
        'This is head 1 of block 1 in the saved dog forward pass. The image’s patch projection and positions '
        'already produced E; we do not flatten or project pixels again here. X=LayerNorm(E), then '
        'V=XW_V+b_V with W_V shaped 192×64 and a 64-coordinate bias, shared across source rows. '
        'There is no additional activation after this value projection. The diagram abbreviates matrix entries; '
        'the P63 preview is read from the saved head1_v array, whose length is 64. These coordinates are learned '
        'features rather than RGB channels. The displayed dog grid connects source identity to a crop; CLS '
        'comes from the extra learned row explained earlier.',
        '<img src="figures/vision1/model-input.png" alt="The same dog photograph" width="160">'
        '<p>X = LayerNorm(E): <strong>197 × 192</strong>. Apply the same Linear(192,64) to every row.</p>'
        '<p>V = X W_V + b_V: <strong>197 × 64</strong>, ordered CLS, P1, …, P196.</p>'
        '<p>P63’s value row begins ['+number(v63[0])+', '+number(v63[1])+', …] and has 64 learned features.</p>')

    sources = ['CLS', 'P1', 'P2', '…', 'P63', '…', 'P196']
    body = t(35, 34, 'Receiver: CLS · selected source: P63', 28, 'c-q')
    for j, label in enumerate(sources):
        body += t(270+(j+.5)*850/7, 74, label, 22, 'ink-2', 'middle')
    body += strip(270, 88, 850, ['a₀', 'a₁', 'a₂', '…', 'a₆₃', '…', 'a₁₉₆'], 'c-q', selected=4)
    body += t(150, 125, '1 × 197', 27, 'c-q', 'middle')
    body += crop(35, 203, 78)+t(74, 316, 'P63', 24, 'c-v', 'middle')
    body += t(194, 231, 'a₆₃', 31, 'c-q', 'middle')+t(194, 274, number(a63, 6), 25, 'c-q', 'middle')
    body += t(290, 242, '×', 36)
    body += t(685, 190, 'Value row v₆₃ · 1 × 64', 26, 'c-v', 'middle')
    body += strip(330, 210, 710, [number(v63[0]), number(v63[1]), '…', number(v63[-1])])
    scaled = arrow(685, 278, 685, 333, 'c-v')+t(711, 313, 'scale every feature', 24, 'c-v')
    scaled += t(194, 387, 'a₆₃ v₆₃', 30, 'c-q', 'middle')
    scaled += strip(330, 345, 710, [number(a63*v63[0], 6), number(a63*v63[1], 6), '…', number(a63*v63[-1], 6)], size=25)
    body += g(scaled, 1)
    body += t(330, 436, 'Contribution to CLS · 1 × 64 · P63 is the source', 26, 'c-v')
    add('real-cls-value-scaling', 'One weight scales all 64 features in its value row', body,
        'P63 supplies this value row; CLS receives its weighted contribution. The same weight scales all 64 features. Add contributions from all sources to complete the CLS message. P63’s own message uses P63’s query.',
        'Who receives this contribution: P63 or CLS?\nCLS is the receiver because we selected the CLS weight row. P63 is the source. The scalar scales all 64 features; the result is one contribution, before summing the other sources.',
        'For this fixed CLS query, a_j means A[CLS,j]. Source 0 is CLS itself. '
        'The saved weight a₆₃ is '+str(a63)+'. Its product with the first value feature is '
        +str(a63*v63[0])+', and the second product is '+str(a63*v63[1])+'. '
        'The displayed products use the full stored precision before rounding. The same scalar multiplies '
        'all 64 coordinates; no feature dimension is removed. Values may be negative, so weighted '
        'features can be negative even though the softmax weights are nonnegative. This crop does not '
        'supply a probability for a class; it supplies a learned value vector. Repeat this multiplication '
        'for all 197 sources, including CLS. Sending a contribution does not modify P63’s value row. '
        'P63’s own update is computed separately from the weights made by its query. '
        'The next slide shows four concrete contributions; the following slide adds all 197 sources.',
        '<p>From the CLS weight row (1 × 197), select a₆₃ = '+number(a63, 6)+'.</p>'
        +mobile_rows(['Feature of P63', 'Value', 'Weight × value'],
                     [[str(j+1), number(v63[j]), number(a63*v63[j], 6)] for j in [0, 1, 63]])
        +'<p><strong>a₆₃ × v₆₃ → one weighted row of shape 1 × 64.</strong> '
        'Every feature gets the same scalar weight. Repeat for all 197 sources.</p>'
        '<p><strong>Receiver: CLS. Source: P63.</strong> P63’s own message uses its own query and attention weights.</p>')

    body = t(35, 32, 'Keep the receiver fixed: every weight below comes from the CLS row.', 28, 'c-q')
    for x, label in [(97, 'Source'), (225, 'CLS weight'), (472, 'Value row · 64 features'), (899, 'Contribution to CLS')]:
        body += t(x, 85, label, 23, 'ink-2', 'middle')
    for j, item in enumerate(examples['sources']):
        y = 108+j*72
        lane = ''
        if item['index']:
            lane += crop(35, y, 48, item['index'])
        else:
            lane += box(35, y, 48, 'CLS', 'c-q', h=48, size=18)
        lane += t(108, y+32, item['token'], 23, 'c-v', 'middle')
        lane += t(225, y+32, number(item['weight'], 6), 24, 'c-q', 'middle')
        lane += t(300, y+33, '×', 31)
        lane += strip(330, y, 295, [number(item['value'][0]), number(item['value'][1]), '…'], height=48, size=24)
        lane += arrow(639, y+24, 677, y+24, 'c-v')
        lane += strip(691, y, 414, [number(item['weighted_value'][0], 6), number(item['weighted_value'][1], 6), '…'], height=48, size=24)
        body += lane if j < 2 else g(lane, 1)
    body += t(35, 426, 'Four source examples. Every contribution still has 64 features.', 29, 'c-v')
    add('real-cls-value-contributions', 'Each source contributes a weighted value row', body,
        'Repeat the P63 calculation for other sources. Each uses its own CLS weight and value row. These are four measured examples from the same head; every contribution is a 64-feature vector addressed to CLS.',
        'Are these messages for four different receivers?\nNo. All four weights come from the CLS attention row. These are four source contributions to one receiver. Each source supplies 64 features, scaled by its own weight.',
        'The sources shown are CLS, P1, P63 and P196. Each product uses A[CLS,j] times V[j]. '
        'Only two of the 64 coordinates are displayed; the ellipsis means all other features remain. '
        'CLS is source index 0 and has a projected value row even though it has no image pixels. '
        'These numbers are measured from block 1, head 1 of the trained dog model. '
        'The unusually large CLS self-weight in this particular head is a measured result; '
        'it is not a general rule for every head, block or image. Products are computed at full '
        'precision before rounding. A source contribution does not itself update the source. '
        'Next we add all 197 source contributions, keeping the receiving query fixed.',
        mobile_rows(['Source', 'CLS weight', 'First two value features', 'First two weighted features'],
                    [[v['token'],number(v['weight'],6),', '.join(number(x) for x in v['value'][:2]),
                      ', '.join(number(x,6) for x in v['weighted_value'][:2])] for v in examples['sources']])
        +'<p>Every product is a 1 × 64 contribution to CLS. These four examples are part of the 197-source sum.</p>')

    body = t(35, 32, 'Add the same feature across sources. Keep all 64 features.', 30)
    body += t(503, 80, 'Weighted contributions for CLS · each 1 × 64', 25, 'c-v', 'middle')
    sum_rows = [(v['token'], v['weighted_value']) for v in examples['sources']]
    sum_rows.append(('Other 193', examples['remaining_weighted_sum']))
    for j, (label, values) in enumerate(sum_rows):
        y = 100+j*55
        body += t(163, y+29, label, 26, 'c-v', 'end')
        if j:
            body += t(210, y+29, '+', 30, 'c-v', 'middle')
        body += strip(245, y, 515, [number(values[0], 6), number(values[1], 6), '…'], height=41, size=25)
    body += line(779,100,793,100,'c-v')+line(793,100,793,361,'c-v')+line(779,361,793,361,'c-v')
    summed = arrow(805,231,850,231,'c-v')
    summed += t(990,171,'Message for CLS',29,'c-q','middle')
    summed += strip(865,200,250,[number(message[0]),number(message[1]),'…'],height=62,size=25)
    summed += t(990,308,'1 × 64',31,'c-v','middle')
    body += g(summed, 1)
    body += t(35, 423, '197 source contributions → one message for the receiving CLS query.', 28, 'c-q')
    add('real-cls-value-sum', 'Add the contributions to make one CLS message', body,
        'Add corresponding coordinates from all 197 weighted value rows, including CLS itself. The result has 64 features: one message for CLS in this head. The other 193 sources are grouped to keep the addition readable.',
        'Why is the result one row with 64 features?\nFor feature 1, sum 197 weighted feature-1 values. Repeat for features 2 through 64. We sum over sources, keeping the feature dimension. Every term belongs to the fixed CLS query.',
        'The calculation is h(CLS) = Σ_j A[CLS,j] V[j]. The first four displayed vectors are '
        'the same four contributions from the preceding slide. The last vector is the measured '
        'sum of the remaining 193 source contributions; it is not a new token or an average. '
        'Adding all five displayed vectors gives the complete 64-coordinate message. '
        'The saved trace checks this equality at full precision; displayed values are rounded. '
        'The first three output features are ['+', '.join(number(v) for v in message)+']. '
        'Equivalently, (1 × 197) times (197 × 64) gives (1 × 64). '
        'This is the weighted-sum operation used for a receiving text token. It produces a '
        'context message for CLS, not a class score or the updated CLS embedding yet.',
        mobile_rows(['Source group', 'Weighted features 1 and 2'],
                    [[label,', '.join(number(v,6) for v in values[:2])] for label,values in sum_rows])
        +'<p>Add down each feature column. The complete message begins ['
        +', '.join(number(v) for v in message)+', …], shape <strong>1 × 64</strong>.</p>'
        '<p>Receiver: CLS. This is one head’s message; the embedding update comes next.</p>')

    # Preview only the destination; the following multi-head sequence opens this box.
    body = t(35, 32, 'Same pattern as text: current token row + context update → updated token row.', 27)
    body += box(35, 85, 250, ['Head 1 message', 'for CLS · 1 × 64'], 'c-v', h=88, size=27)
    body += arrow(300, 129, 363, 129, 'c-v')
    body += box(379, 75, 351, ['Combine 3 head messages', 'then project to 192 features'], 'c-v', h=108, size=25)
    body += t(555, 216, 'We open this box next.', 25, 'ink-2', 'middle')
    body += arrow(744, 129, 804, 129, 'c-v')
    body += box(820, 85, 300, ['CLS context update', '1 × 192'], 'c-v', h=88, size=28)
    destination = box(35, 292, 250, ['Current CLS row', '1 × 192'], 'c-q', h=88, size=28)
    destination += arrow(300, 336, 714, 336, 'c-q')
    destination += line(970,183,970,253,'c-v')+line(970,253,750,253,'c-v')
    destination += arrow(750,253,750,301,'c-v')
    destination += '<circle cx="750" cy="336" r="28" fill="var(--card)" stroke="var(--ink)" stroke-width="2"/>'
    destination += t(750,348,'+',40,'ink','middle')+arrow(791,336,815,336,'c-q')
    destination += box(830,292,290,['Updated CLS row','1 × 192'],'c-q',h=88,size=28)
    body += g(destination, 1)
    body += t(35, 434, 'The CLS representation changes. It is still the same receiving token.', 29, 'c-q')
    add('real-cls-message-destination', 'Where does the CLS message go?', body,
        'The CLS messages become a 192-feature context update after combining heads and projecting. Add this update to the CLS row entering the block. The result is a new representation of the same CLS token, as in text.',
        'Can we add the 64-feature message directly to the 192-feature CLS row?\nNo. First combine all three heads and project. Then add the 192-feature context update to the original 192-feature CLS row. We will open the middle box in the multi-head slides.',
        'This is a destination preview, not an additional attention computation. The middle box '
        'takes three messages for the same CLS receiver, one per head, concatenates them into '
        '192 features and applies the learned output projection with its bias. Only head 1’s '
        'message is expanded here; the next section shows the other two parallel paths. '
        'The current CLS row is the row entering this block, before LayerNorm and attention. '
        'The residual path preserves this row until addition: u_CLS = e_CLS + Δe_CLS. '
        'The result is an attention-updated activation, which will enter the MLP branch. '
        'The stored initial CLS parameter is not changed during this forward pass. '
        'As in the text example, contextualizing a token changes its representation while '
        'preserving its identity. This particular receiver is later used for image classification.',
        '<p>One head’s CLS message: <strong>1 × 64</strong>.</p>'
        '<p>Combine all three heads’ CLS messages, then apply the output projection: '
        '<strong>CLS context update, 1 × 192</strong>.</p>'
        '<p>Current CLS row + CLS context update → updated CLS row (all 1 × 192). '
        'This is the same residual pattern used for text tokens.</p>')

    body = t(35, 32, 'Keep the source values. Change the receiving query and its weights.', 29)
    body += t(105, 104, 'Receiver', 25, 'ink-2', 'middle')
    body += t(377, 104, 'Its own 197 weights', 25, 'c-q', 'middle')
    body += t(640, 104, 'Same V', 25, 'c-v', 'middle')
    body += t(956, 104, 'Its own message · 1 × 64', 25, 'c-v', 'middle')
    for j, receiver in enumerate(examples['receivers']):
        y = 137+j*133
        lane = box(35,y,145,receiver['token']+' query','c-q',h=64,size=25)
        lane += arrow(194,y+32,237,y+32,'c-q')
        lane += strip(251,y,251,[number(receiver['weights'][0],4),number(receiver['weights'][1],4),'…'],
                      'c-q',height=64,size=23)
        lane += t(532,y+43,'×',34)
        lane += box(570,y,140,'197 × 64','c-v',h=64,size=25)
        lane += arrow(724,y+32,777,y+32,'c-v')
        lane += strip(790,y,330,[number(receiver['message'][0]),number(receiver['message'][1]),'…'],height=64,size=28)
        body += lane if j == 0 else g(lane,1)
    body += t(35, 428, 'CLS’s messages update CLS. P63’s messages update P63. Every row has its own.', 27, 'c-q')
    add('real-attention-values', 'Each query gets its own message', body,
        'P63’s query mixes the same source values using P63’s weights, producing P63’s own message. CLS and every other patch do the same. Each receiver’s messages ultimately update its own input row through the residual path.',
        'Does the CLS message also update P63?\nNo. P63 uses its own query, its own attention weights and its own message. Both receivers read the same V in this head. All messages are computed from the rows entering this block.',
        'The two lanes are measured from the same dog, block 1, head 1. Weight previews '
        'show source CLS and source P1, followed by the other 195 sources. Both lanes mix '
        'all 197 value rows. The receiver is selected by the query, not by which source '
        'has the largest attention weight. The full operation H = A V produces 197 rows '
        'of 64 features, one message per receiver. Each input row later combines its own '
        'three head messages, projects and adds to its own input row. The output projection '
        'parameters are shared across receivers. All updates within a block use the same '
        'incoming sequence; the newly updated CLS row does not feed P63’s calculation '
        'inside this same attention operation. The next block receives the updated sequence.',
        mobile_rows(['Receiver', 'Weights over shared V', 'One-head message'],
                    [[r['token'],'A['+r['token']+', all 197 sources]',
                      '['+', '.join(number(v) for v in r['message'][:2])+', …] (1 × 64)']
                     for r in examples['receivers']])
        +'<p>H = A V contains one 64-feature message for each of the 197 receiving rows.</p>'
        '<p>CLS messages update CLS. P63 messages update P63. All receivers use the same incoming sequence.</p>')
    return result
