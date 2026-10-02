"""One shared affine projection, drawn as neurons and as convolution filters."""
import json
import re
from html import escape
from textwrap import dedent
from vision1_focus_common import Figures


def build(b):
    f = Figures(b)
    t, box, line, arrow = f.t, f.box, f.line, f.arrow
    source = (b['ROOT'] / 'notebooks/vision/compare_patch_projections.py').read_text()
    report = json.loads((b['ASSETS'] / 'patch-projection-equivalence.json').read_text())

    def snippet(name):
        return dedent(re.search(r'# slide:'+name+r'\n(.*?)\s*# endslide', source, re.S).group(1)).strip()

    links = ('<a href="notebooks/vision/compare_patch_projections.py">Runnable dog/cat equivalence example</a> · '
             '<a href="figures/vision1/patch-projection-equivalence.json">Measured outputs and checks</a>. ')
    docs = ('<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html">PyTorch Linear</a> · '
            '<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html">PyTorch Conv2d</a> · '
            '<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html">PyTorch Unfold</a>. ')

    def add(key, title, body, caption, question, point, prose='', mobile=''):
        f.add(key, title, body, caption, question, point, prose+' '+links+docs, mobile)

    def neuron(x, y, label, color='c-e', radius=25):
        return (f'<circle cx="{x}" cy="{y}" r="{radius}" fill="var(--card)" '
                f'stroke="var(--{color})" stroke-width="2"/>'
                + t(x, y+7, label, 21, color, 'middle'))

    def photo_grid(x, y, size=196):
        out = f.image(x, y, size, size, f.photo)
        cell = size/14
        for i in range(1, 14):
            out += line(x+i*cell, y, x+i*cell, y+size, 'card', .6)
            out += line(x, y+i*cell, x+size, y+i*cell, 'card', .6)
        return out+f.rect(x+6*cell, y+4*cell, cell, cell, 'c-q', 'transparent', 0)

    # Start with the operation already established in the photo walkthrough.
    body = t(35, 34, 'Implement the same patch projection we used for the dog.', 29)
    body += photo_grid(35, 81)
    body += t(133, 313, '224 × 224 × 3', 25, 'ink-2', 'middle')
    body += t(133, 347, '196 patches', 25, 'ink-2', 'middle')
    body += arrow(247, 178, 310, 178, 'c-q')
    body += box(325, 109, 235, ['One RGB patch', '3 × 16 × 16', '= 768 numbers'], 'c-q', h=126, size=25)
    body += arrow(572, 178, 635, 178, 'c-q')
    body += box(650, 109, 235, ['Shared projection', '768 → 192'], 'c-v', h=126, size=26)
    body += arrow(895, 178, 950, 178, 'c-v')
    body += box(965, 109, 160, ['Patch row', '192 features'], 'c-e', h=126, size=23)
    body += t(325, 302, 'Apply the same learned weights and biases at every location.', 26)
    body += t(325, 347, 'One image → 196 rows, each with 192 features.', 29, 'c-e')
    body += t(35, 419, 'Two implementations: flatten + Linear     or     Conv2d', 29, 'c-v')
    add('code-patch-goal', 'Our target: turn every patch into 192 features', body,
        'Each RGB patch supplies 768 numbers. One shared projection turns it into 192 features. Reusing that projection across the 14×14 grid produces 196 rows. We will implement this exact operation in two ways.',
        'Do we learn a separate projection for each of the 196 patches?',
        'No. Each patch supplies different inputs to the same learned projection. The same parameters also serve the cat and every other image.',
        'This continues the existing 224×224 photo example. We are implementing the patch projection before CLS, '
        'position addition and attention. The highlighted location is P63. There is no attention or cross-patch mixing here.')

    # Draw an actual dense layer, with the bias visible on the selected neuron.
    body = t(35, 32, 'One affine layer: nn.Linear(768, 192, bias=True)', 29, 'c-e')
    body += t(142, 76, '768 inputs', 26, 'c-q', 'middle')+t(459, 76, '192 outputs', 26, 'c-v', 'middle')
    ys = [117, 183, 302]
    for ix, y in enumerate(ys):
        for oy in ys:
            body += line(168, y, 434, oy, 'c-v' if oy == 117 else 'line', 2 if oy == 117 else 1)
    body += t(283, 100, 'w₁,₁, w₁,₂, …, w₁,₇₆₈', 21, 'c-v', 'middle')
    for y, label in zip(ys, ['x₁', 'x₂', 'x₇₆₈']):
        body += neuron(142, y, label, 'c-q', 27)
    for y, label in zip(ys, ['c₁', 'c₂', 'c₁₉₂']):
        body += neuron(459, y, label, 'c-v', 27)
    body += t(142, 249, '⋮', 32, 'c-q', 'middle')+t(459, 249, '⋮', 32, 'c-v', 'middle')
    body += arrow(551, 117, 491, 117, 'c-k')+t(568, 124, '+ b₁', 25, 'c-k')
    body += t(680, 100, 'Open just the first output neuron', 27, 'c-v')
    body += t(680, 157, 'c₁ = w₁,₁x₁ + w₁,₂x₂ + …', 30, 'c-v')
    body += t(748, 198, '+ w₁,₇₆₈x₇₆₈ + b₁', 30, 'c-v')
    body += t(680, 260, 'Every output has its own', 27)
    body += t(680, 301, '768 weights + 1 bias.', 30, 'c-k')
    body += line(35, 343, 1125, 343)
    body += t(35, 382, '192 × 768 weights + 192 biases = 147,648 parameters', 31, 'c-e')
    body += t(35, 427, 'Patch projection: no hidden layer or activation. Later MLP: Linear → GELU → Linear.', 23, 'ink-2')
    add('code-patch-dense', 'See the patch projection as a layer of neurons', body,
        'All 768 inputs connect to every output neuron. Each neuron computes a weighted sum and adds its own bias. The patch projection is one dense layer; the later Transformer MLP has two layers and a nonlinear activation.',
        'Where does the 192 in the bias count come from?',
        'There is one bias per output feature, not one per input pixel or patch. Every output has 768 weights, so there are 147,456 weights and 192 biases in total.',
        'Only a few neurons and connections are drawn; dots stand for the omitted ones. '
        'In PyTorch the weight storage is [out_features,in_features]=[192,768], so the row-vector operation is xWᵀ+b. '
        'Earlier mathematical diagrams may store its transpose. An MLP can contain this dense operation, '
        'but adding a hidden layer or activation would change the patch embedding used by this model.',
        b['mobile_rows'](['Quantity', 'Count'], [['Inputs per patch', '3×16×16 = 768'],
          ['Output neurons', '192'], ['Weights', '192×768 = 147,456'], ['Biases', '192'], ['Total', '147,648']])
        + '<p>c₁ = w₁₁x₁ + … + w₁₇₆₈x₇₆₈ + b₁. This projection has no hidden layer or activation.</p>')

    code = snippet('linear-init')+'\n\ndef project_with_linear(images, linear):\n'+\
        '\n'.join('    '+s for s in snippet('linear-forward').splitlines())
    body = f.code(code, x=35, y=68, width=740, size=21, spacing=37)
    body += t(785, 36, 'images · B × 3 × 224 × 224', 24, 'c-q')
    body += arrow(951, 55, 951, 83, 'c-q')
    body += box(790, 95, 330, ['unfold: flattened columns', 'B × 768 × 196'], 'c-q', size=23)
    body += arrow(951, 181, 951, 202)
    body += box(790, 214, 330, ['transpose: patch rows', 'B × 196 × 768'], 'c-q', size=23)
    body += arrow(951, 299, 951, 319, 'c-v')
    body += box(790, 331, 330, ['Linear: project each row', 'B × 196 × 192'], 'c-v', size=23)
    body += t(35, 423, 'One patch row: 256 R values | 256 G values | 256 B values.', 23, 'ink-2')
    add('code-patch-linear', 'Implementation 1: extract patches, then use Linear', body,
        'Unfold extracts each non-overlapping patch as a column. Transpose makes patches into rows. Linear changes only the last axis, from 768 inputs to 192 features, reusing its weights for every patch and image.',
        'Does applying Linear to 196 rows create 196 copies of its weights?',
        'No. A Linear module broadcasts the same affine operation across the leading axes. Only activations grow with B and the number of patches.',
        'The upper line belongs in initialization; the remaining lines form project_with_linear(images, linear). '
        'F.unfold is the functional version of Unfold. Its patch order is row-major over the spatial grid; '
        'inside each patch it flattens channel, row, column. This is the same ordering as the 2×2 RGB example: '
        'the channel groups now contain 256 values each. Transpose swaps the patch and feature axes without changing the order within a patch. '
        'The modules are initialized randomly here; '
        'the equivalence example below copies pretrained weights into them before evaluating the dog and cat.',
        '<pre><code>'+escape(snippet('linear-init')+'\n\ndef project_with_linear(images, linear):\n'+
                               '\n'.join('    '+s for s in snippet('linear-forward').splitlines()))+'</code></pre>')

    # Same dot product arranged as three 16×16 channel slices.
    body = f.code(snippet('conv-init'), x=35, y=38, width=1090, size=25, spacing=32)
    body += photo_grid(35, 126)
    body += t(133, 357, '14 × 14 locations', 24, 'ink-2', 'middle')
    body += arrow(246, 218, 318, 218, 'c-q')
    body += t(455, 118, 'One filter = one output neuron', 25, 'c-v', 'middle')
    for j, color in enumerate(['c-a', 'c-v', 'c-e']):
        x, y = 354+j*22, 164-j*16
        body += f.rect(x, y, 116, 116, color, 'card', 0)
        for k in range(1, 4):
            body += line(x+k*29, y, x+k*29, y+116, color, .6)
            body += line(x, y+k*29, x+116, y+k*29, color, .6)
    body += t(590, 230, '+ bⱼ', 28, 'c-k')
    body += t(454, 315, '3 × 16 × 16 weights', 25, 'c-v', 'middle')
    body += t(454, 350, 'all 768 pixels contribute', 23, 'ink-2', 'middle')
    body += arrow(655, 218, 717, 218, 'c-v')
    body += box(733, 159, 386, ['192 filters → 192 features', 'at each patch location'], 'c-v', h=106, size=27)
    body += t(733, 319, 'Stride 16: move by one patch.', 25)
    body += t(733, 354, 'Output: B × 192 × 14 × 14', 26, 'c-e')
    body += t(35, 420, 'Same filters at every location. Kernel size = stride = 16; patches do not overlap.', 26)
    add('code-photo-conv', 'Implementation 2: the same projection with Conv2d', body,
        'Reshape each neuron’s 768 weights into a 3×16×16 filter. It reads the same pixels and adds the same bias. Conv2d applies all 192 filters at every patch location, giving a 192-channel feature grid.',
        'What does one output channel represent?',
        'One of the 192 learned patch features, computed at all 14×14 locations using one shared filter and bias. The filter spans all three RGB channels.',
        'The three drawn planes stand for channel slices of one filter; the coarse grid is schematic. '
        'Each plane actually holds 16×16 learned weights. Conv2d computes cross-correlation, so no kernel flip '
        'is needed to match the Linear weights. With padding=0, dilation=1, groups=1, kernel=stride=16, '
        'the output height and width are each (224−16)/16+1=14. There is no activation in this operation.',
        '<pre><code>'+escape(snippet('conv-init')+'\n\ndef project_with_conv(images, conv):\n'+
                                '\n'.join('    '+s for s in snippet('conv-forward').splitlines()))+'</code></pre>'
        '<p>Each of 192 filters has 3×16×16 weights and one bias. Output: B×192×14×14; flatten and transpose to B×196×192.</p>')

    body = t(35, 32, 'Identical numbers, stored with different shapes', 29)
    table = [['Parameter', 'Linear', 'Conv2d'],
             ['Weight shape', '192 × 768', '192 × 3 × 16 × 16'],
             ['Weights', '147,456', '147,456'],
             ['Biases', '192', '192'],
             ['Total', '147,648', '147,648']]
    for i, row in enumerate(table):
        for x, width, value, color in zip([35, 385, 750], [350, 365, 375], row, ['ink-2', 'c-e', 'c-v']):
            body += f.rect(x, 55+i*43, width, 43, 'line', 'card' if i == 0 else 'transparent', 0)
            body += t(x+17, 84+i*43, value, 25, color, weight=600 if i in [0, 4] else 500)
    body += t(52, 297, 'Match the input order: each filter flattens R, then G, then B.', 24, 'ink-2')
    body += f.code(snippet('copy-weights'), x=52, y=332, width=1073, size=25, spacing=27)
    add('code-patch-parameters', 'Reshape the weights; keep the same parameter count', body,
        'Both implementations have 147,456 weights and 192 biases. Copying the exact weights and biases makes their computations equivalent. More patches or images create more output values, while the learned parameter count stays 147,648.',
        'Would two independently initialized layers produce the same features?',
        'No. Equal parameter counts are not enough. They must use the same weights, biases and pixel order. Flatten each Conv2d filter to obtain the corresponding row of Linear.weight.',
        'For the comparison, the Conv2d module receives the saved pretrained patch-embedding weights. '
        'The displayed copy then transfers them to Linear. The dense module and convolution module are '
        'alternative implementations; a deployed classifier uses just one of them. Replacing only this '
        'operation leaves the full classifier parameter count unchanged. The bias is shared over patch '
        'locations, just like the weights. Changing to a different pixel flatten order requires the same '
        'permutation of weight columns.',
        b['mobile_rows'](table[0], table[1:])+'<pre><code>'+escape(snippet('copy-weights'))+'</code></pre>')

    # Corresponding input and weight coordinates, drawn in the same order.
    trace = report['pixel_weight_trace']
    def num(value):
        return f'{value:.6f}'.replace('-', '−')
    body = t(35, 31, 'Dog patch P63 · first output feature · indices below start at 0', 26)
    body += t(35, 77, 'Flatten order: R₀ … R₂₅₅  |  G₀ … G₂₅₅  |  B₀ … B₂₅₅', 26, 'ink-2')
    body += t(35, 119, 'Green pixel at (row 2, column 3) → flat index 256 + 2×16 + 3 = 291', 27, 'c-q')
    for x, title, color in [(35, 'Linear: one neuron', 'c-e'), (625, 'Conv2d: one filter, flattened', 'c-v')]:
        body += t(x, 157, title, 27, color, weight=600)
        for row, (y, values, label) in enumerate([
            (193, ['R₀', 'R₁', '…', 'G₃₅', '…', 'B₂₅₅'], 'same 768 pixels'),
            (284, ['w₀', 'w₁', '…', 'w₂₉₁', '…', 'w₇₆₇'], 'same 768 weights')]):
            if row == 1:
                label = 'W[0,291] selected' if x == 35 else 'W[0,1,2,3] selected'
            strip = t(x, y-7, label, 19, 'ink-2')
            for i, value in enumerate(values):
                c = 'c-q' if i == 3 else color
                strip += f.rect(x+i*83, y, 83, 40, c, 't-q' if i == 3 else 'transparent', 0)
                strip += t(x+i*83+41.5, y+28, value, 24, c, 'middle')
            if row == 1:
                for i in range(6):
                    strip += t(x+i*83+41.5, 263, '…' if i in [2,4] else '×', 26,
                               'c-q' if i == 3 else color, 'middle')
            body += strip if row == 0 else f.g(strip, 1)
    body += f.g(t(35, 369, f'Selected product in both:  {num(trace["pixel"])} × {num(trace["weight"])} = {num(trace["product"])}', 27, 'c-q'), 2)
    body += f.g(t(35, 419, f'Add all 768 products + the same bias ({num(trace["bias"])}) → {num(trace["sum_768_products_plus_bias"])}', 26, 'c-v'), 2)
    add('code-patch-same-products', 'Same pixel × same weight, in both implementations', body,
        'Both paths pair the same 768 pixels with the same 768 weights, sum their products, and add the same bias.',
        'Which Conv2d weight corresponds to Linear.weight[0,291]?',
        'conv.weight[0,1,2,3]: output 0, green channel, patch row 2, column 3. R, G and B each contain 256 pixels in row-major order.',
        'The pixel is the checkpoint-normalized green value images[0,1,66,99] inside dog patch P63. '
        'G35 is the 36th green pixel, since 2×16+3=35. Its flattened offset is 256+35=291. '
        'linear.weight[0,291] equals conv.weight[0,1,2,3]. These are saved measurements; displayed values are rounded. '
        'The Conv2d weight strip is a drawing of the flattened filter for comparison, not an extra runtime layer. '
        'All other 767 products correspond in the same way. Both add the same bias once. The script verifies this selected pixel, weight and full sum.',
        '<p>Flatten RGB channels in order, row-major within each channel. Green [2,3] has flat index 291.</p>'
        '<p>Linear.weight[0,291] = Conv2d.weight[0,1,2,3]. Both multiply the same input value by this same learned weight.</p>')

    # Actual checkpoint outputs, not invented feature values.
    body = t(35, 30, 'Same pretrained weights · same prepared dog and cat images', 28)
    body += t(244, 75, 'P63 · first 3 of 192 features', 24, 'ink-2')
    body += t(647, 75, 'Linear', 25, 'c-e', 'middle')+t(955, 75, 'Conv2d', 25, 'c-v', 'middle')
    mobile_results = []
    for i, (result, photo, label) in enumerate(zip(report['results'], [f.photo, b['CAT']], ['Dog', 'Cat'])):
        y = 92+i*90
        body += f.image(35, y, 78, 78, photo)+t(132, y+47, label, 28)
        for x, field, color in [(482, 'linear_first_features', 'c-e'), (810, 'conv_first_features', 'c-v')]:
            values = '['+', '.join(f'{v:.3f}'.replace('-', '−') for v in result[field])+', …]'
            body += t(x, y+46, values, 25, color)
        body += t(244, y+46, '768 → 192', 26, 'ink-2')
        mobile_results.append([label, str(result['linear_first_features']), str(result['conv_first_features'])])
    body += t(35, 307, 'Checked all 2 × 196 × 192 = 75,264 features.', 28, 'c-v')
    body += t(35, 347, f'Largest float32 difference: {report["max_absolute_error"]:.2g}. Same result within rounding.', 25, 'ink-2')
    body += t(35, 416, 'Both produce B × 196 × 192 rows for the same classifier.', 27, 'c-e')
    add('code-patch-equivalence', 'Verify it on both photographs: the features match', body,
        'Using identical pretrained weights, both paths produce the same dog and cat features within floating-point rounding. Conv2d packages patch extraction and projection into one image operation. Flatten and transpose its grid before adding CLS and positions.',
        'Why use Conv2d if Linear can compute the same thing?',
        'Conv2d accepts the image grid directly and uses optimized convolution implementations. It avoids explicitly constructing a patch matrix in our code. This is a choice of implementation, not a change in the learned model.',
        'Measured using '+escape(report['model'])+'. The photo thumbnails identify the two inputs; both were '
        'prepared with the checkpoint transform before computation. P63 means zero-based grid [4,6]. '
        'Displayed features are rounded to three decimals. All 75,264 values were checked with '
        'torch.testing.assert_close(rtol=1e-5,atol=2e-5), not only these previews. Exact arithmetic is '
        'equivalent; floating-point summation order can differ. The script also checks the saved '
        'PatchEmbed result, individual-patch pixel order and batch/individual consistency. It performs '
        'no training. Actual speed and memory depend on hardware and backend; no benchmark claim is made. '
        'Executable comparison:<pre><code>'+escape(snippet('compare'))+'</code></pre>',
        b['mobile_rows'](['Image', 'Linear P63 preview', 'Conv2d P63 preview'], mobile_results)
        +'<pre><code>'+escape(snippet('compare')+'\n\ndef project_with_conv(images, conv):\n'+
            '\n'.join('    '+s for s in snippet('conv-forward').splitlines()))+'</code></pre>')

    body = t(35, 34, 'One image operation applies all 192 filters at all 196 locations.', 28)
    body += t(35, 99, 'Linear route', 27, 'c-e', weight=600)
    body += box(35, 125, 280, ['Image grid', 'B × 3 × 224 × 224'], size=24)
    body += arrow(324, 163, 406, 163, 'c-e')
    body += box(417, 125, 300, ['Explicit patch rows', 'B × 196 × 768'], size=24)
    body += arrow(727, 163, 798, 163, 'c-e')
    body += box(809, 125, 315, ['Shared Linear', 'B × 196 × 192'], size=24)
    conv_route = t(35, 258, 'Conv2d route', 27, 'c-v', weight=600)
    conv_route += box(35, 283, 280, ['Image grid', 'B × 3 × 224 × 224'], 'c-v', size=24)
    conv_route += arrow(324, 321, 406, 321, 'c-v')
    conv_route += box(417, 283, 300, ['Conv2d', 'B × 192 × 14 × 14'], 'c-v', size=24)
    conv_route += arrow(727, 321, 798, 321, 'c-v')
    conv_route += box(809, 283, 315, ['Flatten + transpose', 'B × 196 × 192'], 'c-v', size=24)
    body += f.g(conv_route, 1)
    body += f.g(t(35, 420, 'Less reshaping code; optimized convolution backends. Same learned model.', 27, 'c-v'), 1)
    add('code-patch-why-conv', 'Why package patch projection as Conv2d?', body,
        'Conv2d accepts the image layout directly and handles the repeated patch computations. It avoids an explicit unfolded patch tensor in our code and uses optimized convolution backends. Actual speed and memory depend on the hardware.',
        'Does Conv2d learn fewer parameters or a different function?',
        'No. The same 147,648 parameters compute the same outputs. The advantage is a convenient image operation and backend support; benchmark actual speed and memory.',
        'Both implementations batch all images and patches; Linear does not require a Python patch loop. '
        'Conv2d packages patch access and projection without an explicit unfold result at the Python level. '
        'Backends can still allocate working buffers. With non-overlapping patches, the unfolded tensor has '
        '196×768 = 3×224×224 entries per image, so there is no overlap-induced expansion here. '
        'No universal speedup or memory saving is claimed. Shared weights, biases, stride, padding and '
        'flatten order are exactly those on the preceding slides.')
    return f.frames
