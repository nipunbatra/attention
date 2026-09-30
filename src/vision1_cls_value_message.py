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
        'The next slide shows four concrete contributions and adds all 197 sources.',
        '<p>From the CLS weight row (1 × 197), select a₆₃ = '+number(a63, 6)+'.</p>'
        +mobile_rows(['Feature of P63', 'Value', 'Weight × value'],
                     [[str(j+1), number(v63[j]), number(a63*v63[j], 6)] for j in [0, 1, 63]])
        +'<p><strong>a₆₃ × v₆₃ → one weighted row of shape 1 × 64.</strong> '
        'Every feature gets the same scalar weight. Repeat for all 197 sources.</p>'
        '<p><strong>Receiver: CLS. Source: P63.</strong> P63’s own message uses its own query and attention weights.</p>')

    body = t(35, 28, 'Receiver stays CLS · same dog, block 1, head 1 · four of 197 sources', 26)
    for x, label in [(92, 'Source'), (206, 'CLS weight'), (455, 'Value row · 1 × 64'), (859, 'Weighted contribution')]:
        body += t(x, 65, label, 23, 'ink-2', 'middle')
    for j, item in enumerate(examples['sources']):
        y = 83 + j*53
        if item['index']:
            body += crop(35, y, 38, item['index'])
        else:
            body += rect(35, y, 38, 38, 'c-q', 't-q', 3)
            body += t(54, y+26, '0', 22, 'c-q', 'middle')
        body += t(95, y+27, item['token'], 23, 'c-v', 'middle')
        body += t(206, y+27, number(item['weight'], 6), 23, 'c-q', 'middle')
        body += t(268, y+29, '×', 29)
        body += strip(300, y, 310, [number(item['value'][0]), number(item['value'][1]), '…'], height=42, size=23)
        body += arrow(622, y+21, 656, y+21, 'c-v')
        body += strip(674, y, 370, [number(item['weighted_value'][0], 6), number(item['weighted_value'][1], 6), '…'], height=42, size=22)
    summed = line(1062,83,1074,83,'c-v')+line(1074,83,1074,284,'c-v')+line(1062,284,1074,284,'c-v')
    summed += t(35,328,'Also include the other 193 source contributions.',25)
    summed += t(35,372,'Add all 197 rows, feature by feature.',27,'c-v')
    summed += t(859,325,'Message for CLS · 1 × 64',26,'c-v','middle')
    summed += line(1074,284,1074,367,'c-v')+arrow(1074,367,1048,367,'c-v')
    summed += strip(674,340,370,[number(message[0]),number(message[1]),'…'],height=54,size=27)
    body += g(summed,1)
    body += g(t(35,435,'Next: join heads → project to 192 → add to the CLS input row.',27,'c-q'),2)
    add('real-cls-value-sum', 'Many source contributions become one message for CLS', body,
        'Keep the CLS query fixed. Each source supplies a weighted value row. Add all 197 contributions to get one 64-feature message for CLS. The displayed four sources are examples; the other 193 also contribute.',
        'Do these four contributions finish the message?\nNo. Include all 197 sources, including CLS itself. Add corresponding feature coordinates. The receiver remains CLS because every displayed weight comes from the CLS attention row.',
        'The four rows show saved values for sources CLS, P1, P63 and P196 in block 1, head 1. '
        'For source j, multiply all 64 features of V[j] by A[CLS,j]. The crop identifies the source; '
        'the small zero box marks CLS at source index 0, which has no pixels. Only the first two '
        'feature coordinates are displayed. Products are computed at full stored precision before rounding. '
        'The sum is h(CLS) = Σ_j A[CLS,j] V[j], over all 197 sources. The saved trace verifies that '
        'the four shown weighted rows plus the remaining 193 weighted rows equal the complete message. '
        'The message begins ['+', '.join(number(v) for v in message)+']. '
        'It has 64 features and belongs to the receiving CLS query. This is one head’s message, '
        'before concatenation, output projection and residual addition. The three heads produce '
        'three messages for CLS; concatenate them to 192 features, project, and add to the 192-feature CLS input. '
        'This produces the attention-updated CLS row, which then enters the MLP branch. '
        'The following slide compares the separate message received by a patch query.',
        mobile_rows(['Source', 'CLS weight', 'First two value features', 'First two weighted features'],
                    [[v['token'],number(v['weight'],6),', '.join(number(x) for x in v['value'][:2]),
                      ', '.join(number(x,6) for x in v['weighted_value'][:2])] for v in examples['sources']])
        +'<p>Add these four rows and the other 193 source contributions. '
        '<strong>All 197 contributions → one message for CLS, shape 1 × 64.</strong></p>'
        '<p>Message begins ['+', '.join(number(v) for v in message)+', …]. '
        'Join the three heads, project to 192 features, then add to the original CLS input row.</p>')

    body = t(35,30,'Same dog · block 1 · head 1 · a different query gives a different message',26)
    body += t(104,68,'Receiver',24,'ink-2','middle')
    body += t(377,68,'Its own 197 weights',24,'c-q','middle')
    body += t(640,68,'Same V',24,'c-v','middle')
    body += t(943,68,'Message for this receiver · 1 × 64',24,'c-v','middle')
    for j, receiver in enumerate(examples['receivers']):
        y=88+j*86
        body += t(104,y+36,receiver['token']+' query',25,'c-q','middle')
        body += arrow(203,y+26,245,y+26,'c-q')
        body += strip(255,y,244,[number(receiver['weights'][0],4),number(receiver['weights'][1],4),'…'],
                      'c-q',height=52,size=21)
        body += t(530,y+37,'×',31)
        body += rect(570,y,140,52,'c-v','transparent')+t(640,y+33,'197 × 64',25,'c-v','middle')
        body += arrow(724,y+26,772,y+26,'c-v')
        body += strip(790,y,325,[number(receiver['message'][0]),number(receiver['message'][1]),'…'],height=52,size=25)
    body += t(35,272,'Next, complete the update for the receiving row (shown for CLS):',26)
    from vision1_focus_common import Figures
    box=Figures(b).box
    route=box(35,300,184,['CLS messages','3 × (1 × 64)'],'c-v',h=64,size=23)
    route+=arrow(225,332,247,332,'c-v')+box(255,300,206,['Concatenate','1 × 192'],'c-v',h=64,size=24)
    route+=arrow(468,332,485,332,'c-v')+box(493,300,187,['Project','1 × 192 update'],'c-v',h=64,size=22)
    route+=arrow(686,332,718,332,'c-v')
    route+='<circle cx="743" cy="332" r="23" fill="var(--card)" stroke="var(--ink)" stroke-width="2"/>'
    route+=t(743,342,'+',36,'ink','middle')+arrow(769,332,792,332,'c-e')
    route+=box(800,300,185,['Updated CLS','1 × 192'],'c-q',h=64,size=23)
    route+=arrow(992,332,1025,332)+box(1032,300,85,'MLP',h=64,size=24)
    route+=box(517,376,249,'CLS input · 1 × 192','c-q',h=37,size=22)
    route+=arrow(743,376,743,357,'c-q')
    body+=g(route,1)
    body+=g(t(35,439,'P63 follows the same route using P63’s messages and input.',26,'c-e'),1)
    add('real-attention-values','The query determines which row receives the message',body,
        'CLS and P63 use different weights over the same value rows, producing separate messages. For each receiver, join its three head messages, project to 192 features, then add to that receiver’s input. The MLP follows.',
        'Does the CLS message update P63 too?\nNo. P63’s query produces P63’s message. Each row combines its own three head messages, applies the shared output projection, and adds the result to its own 192-feature input row.',
        'Both rows are measured from the same dog, block 1, head 1. The weight previews show source CLS, '
        'source P1 and an ellipsis for the other 195 sources. All 197 value rows participate in each weighted sum. '
        'The same V appears in both lanes; only the receiving query and hence its attention weights differ. '
        'H=A V has shape 197×64. Row i of H is the message for row i of the input: '
        'h_i=Σ_j A[i,j]V[j]. Source identity j tells us where a contribution came from; '
        'receiver identity i tells us which row gets the message. '
        'The lower diagram previews the later multi-head explanation: Δe_i=Concat(h_i¹,h_i²,h_i³)W_O+b_O, '
        'then u_i=e_i+Δe_i. A single 64-feature head message is not directly added to a 192-feature input. '
        'The output projection is shared across receivers; each receiver supplies its own joined message. '
        'The residual input is the row entering this block. The MLP branch subsequently transforms u_i '
        'and adds its own residual update. Repeat for all 197 receiving rows; these updates use the same '
        'incoming sequence rather than running one receiver after another.',
        mobile_rows(['Receiver', 'Weights', 'One-head message'],
                    [[r['token'],'A['+r['token']+', all 197 sources]',
                      '['+', '.join(number(v) for v in r['message'][:2])+', …] (1 × 64)']
                     for r in examples['receivers']])
        +'<p>Same source value matrix V for every query. H = A V has 197 rows, one per receiver.</p>'
        '<p>For each receiver: concatenate its three head messages → 1 × 192 → output projection '
        '→ add its own input row → attention-updated row → MLP.</p>'
        '<p>CLS messages update CLS. P63 messages update P63. All other patch queries have their own messages too.</p>')
    return result
