"""One complete ViT reference figure, using the parallel lanes of TinyStories."""


def model_recap(b, probability, loss):
    t, rect, line, arrow = (b[k] for k in ['t', 'rect', 'line', 'arrow'])

    def node(x, y, w, title, detail, color='c-e', h=56):
        compact = h <= 48
        return (rect(x, y, w, h, color, 'card', 5)
                + t(x+w/2, y+(18 if compact else 22), title, 19 if compact else 21, color, 'middle', 600)
                + t(x+w/2, y+h-(7 if compact else 10), detail, 15 if compact else (16 if w < 150 else 18), 'ink-2', 'middle'))

    def route(points, color='ink-3', dashed=False):
        out = ''.join(line(*a, *c, color, 1.8, '5 4' if dashed else '')
                      for a, c in zip(points[:-2], points[1:-1]))
        return out + arrow(*points[-2], *points[-1], color)

    def plus(x, y):
        return (f'<circle cx="{x}" cy="{y}" r="16" fill="var(--card)" '
                'stroke="var(--c-e)" stroke-width="2"/>'
                + t(x, y+8, '+', 28, 'c-e', 'middle'))

    # Input construction: each operation changes the shape only as labelled.
    out = ''  # One image; the batch axis is omitted throughout.
    out += node(20, 12, 146, '1 · RGB image', '3 × 224 × 224')
    out += node(190, 12, 205, '16 × 16 patches', 'flatten → 196 × 768')
    out += node(419, 12, 209, 'Shared projection', '768 → 192 + bias')
    out += node(652, 12, 200, 'Prepend learned CLS', '196 + 1 = 197 rows', 'special').replace('font-size="21"', 'font-size="19"')
    out += node(876, 12, 264, '+ learned positions', 'E⁰: 197 × 192', 'special')
    for left, right in [(166, 190), (395, 419), (628, 652), (852, 876)]:
        out += arrow(left+4, 40, right-4, 40)

    # All twelve real block instances are shown, with one opened below.
    out += route([(1008, 68), (1008, 72), (35, 72), (35, 115), (89, 115)], 'c-e')
    out += t(590, 93, '2 · Blocks 1–12 · all 197 rows, each 192 features wide', 23, 'c-e', 'middle', 600)
    for i in range(12):
        x = 94+i*80
        out += rect(x, 99, 64, 32, 'mixing' if i == 0 else 'c-e', 't-mixing' if i == 0 else 't-e', 4)
        out += t(x+32, 122, str(i+1), 23, 'mixing' if i == 0 else 'c-e', 'middle', 600)
        if i < 11:
            out += arrow(x+67, 115, x+76, 115)
    out += arrow(1044, 115, 1077, 115, 'c-e')+t(1104, 122, 'E¹²', 26, 'c-e', 'middle')

    # The large panel is a zoom of block 1, not another block in sequence.
    out += line(98, 133, 25, 148, 'mixing', 1.6, '5 4')
    out += line(156, 133, 1135, 148, 'mixing', 1.6, '5 4')
    out += rect(20, 148, 1120, 286, 'mixing', 'transparent', 6)
    out += t(38, 170, 'INSIDE BLOCK 1', 21, 'mixing', weight=700)
    out += t(300, 170, 'Blocks 2–12 repeat these operations with their own weights.', 19, 'ink-2')

    # Attention sublayer. Separate V routes bypass scores and row softmax.
    out += route([(68, 242), (68, 182), (1092, 182), (1092, 247)], 'c-e')
    out += t(1056, 179, 'skip: E', 18, 'c-e', 'end')
    out += node(36, 242, 65, 'E', '197 × 192', h=46).replace('font-size="15"', 'font-size="12"')
    out += arrow(105, 265, 121, 265)
    out += node(125, 237, 98, 'LN 1', '197 × 192')
    out += arrow(227, 265, 246, 265)+line(246, 217, 246, 315, 'ink-3', 1.8)
    for i, y in enumerate([196, 245, 294]):
        cy = y+21
        out += arrow(246, cy, 272, cy)
        out += node(276, y, 122, f'Head {i+1}: QKV', 'each 197 × 64', 'mixing', 42).replace('font-size="19"', 'font-size="17"')
        out += node(422, y, 125, 'QKᵀ / √64', '197 × 197', 'mixing', 42)
        out += node(571, y, 116, 'A = softmax', 'over source keys', 'mixing', 42)
        out += node(711, y, 85, 'AV', '197 × 64', 'c-v', 42)
        out += arrow(401, cy, 418, cy, 'mixing')
        out += arrow(551, cy, 567, cy, 'mixing')
        out += arrow(691, cy, 707, cy, 'mixing')
        out += route([(350, y+42), (350, y+46), (754, y+46), (754, y+42)], 'c-v', True)
        out += t(553, y+44, 'V', 11, 'c-v', 'middle')
        out += line(796, cy, 807, cy, 'c-v', 1.8)
    out += line(807, 217, 807, 315, 'c-v', 1.8)+arrow(807, 265, 817, 265, 'c-v')
    out += node(821, 237, 101, 'Concat', '3 × 64 = 192', 'c-v')
    out += arrow(926, 265, 938, 265, 'c-v')
    out += node(942, 237, 97, 'W_O + b', '192 → 192', 'c-e')
    out += arrow(1043, 265, 1073, 265, 'c-e')+plus(1092, 265)
    out += t(1092, 306, 'U', 25, 'c-e', 'middle')
    out += t(840, 323, '3 parallel heads', 20, 'mixing')
    out += t(840, 343, 'No causal mask', 19, 'ink-2')

    # The next branch starts with U, the result of the attention residual.
    out += route([(1092, 281), (1120, 281), (1120, 350), (67, 350), (67, 360)], 'c-e')
    out += node(36, 362, 65, 'U', '197 × 192', h=44).replace('font-size="15"', 'font-size="12"')
    out += arrow(105, 384, 121, 384)
    out += node(125, 362, 98, 'LN 2', '197 × 192', h=44)
    out += arrow(227, 384, 266, 384)
    out += node(270, 362, 200, 'MLP: Linear + bias', '192 → 768', 'neutral', 44)
    out += arrow(474, 384, 511, 384)
    out += node(515, 362, 100, 'GELU', '197 × 768', 'neutral', 44)
    out += arrow(619, 384, 656, 384)
    out += node(660, 362, 200, 'Linear + bias', '768 → 192', 'neutral', 44)
    out += arrow(864, 384, 923, 384, 'c-v')+plus(942, 384)
    out += route([(67, 406), (67, 421), (942, 421), (942, 402)], 'c-e')
    out += t(142, 416, 'skip: U', 18, 'c-e')
    out += arrow(962, 384, 1002, 384, 'c-e')+node(1007, 362, 113, 'E¹', '197 × 192', h=44)

    # Actual depth path bypasses the explanatory zoom and resumes at readout.
    out += route([(1129, 115), (1152, 115), (1152, 447), (90, 447), (90, 480)], 'c-e')
    out += t(185, 475, '3 · After block 12: normalize → CLS → class scores → prediction / loss', 22, 'c-e', weight=600)
    out += node(20, 484, 141, 'Final LN', '197 × 192')
    out += node(185, 484, 134, 'Select CLS', '1 × 192', 'vision')
    out += node(343, 484, 186, 'Linear head + bias', '192 → 1,000 logits')
    out += node(553, 484, 142, 'Softmax', '1,000 probs', 'neutral')
    out += node(719, 484, 235, 'Newfoundland', f'top label · {probability:.2%}', 'vision')
    for left, right in [(161, 185), (319, 343), (529, 553), (695, 719)]:
        out += arrow(left+4, 512, right-4, 512)
    out += node(995, 484, 145, 'Label loss', f'L = {loss:.4f}', 'c-a')
    out += route([(436, 540), (436, 553), (1067, 553), (1067, 542)], 'c-a')
    out += t(759, 576, 'Cross-entropy(logits, y) · y = Newfoundland', 21, 'c-a', 'middle')
    return out
