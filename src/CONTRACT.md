# CONTRACT — shared runtime (`shared.js` → `window.AT`), CSS catalogue, fragment rules

Read BRIEF.md first. This file is the API you build against. Everything here is implemented and tested in
`sections/sec00_demo.html` (the component gallery — open `test00.html` after assembling it to see every component live).

---
## 0. Section skeleton (copy this)

```html
<section id="s07" class="sec" data-title="Every token becomes a searchable record" data-lit="">
<style>
/* fragment-local CSS only, always scoped with #s07 … ; keep it minimal */
#s07 .my-thing{margin-top:8px}
</style>
<header class="sec-head">
  <span class="sec-num">07</span>
  <div>
    <p class="eyebrow">Why we are here: we have Q, K, V for search results — now every token becomes one.</p>
    <h2>Every token becomes a searchable record</h2>
  </div>
</header>

<div class="prose">
  <p>Static math is written inline: $\vk{k_j} = W_K \ve{e_j}$ and $\vv{v_j} = W_V \ve{e_j}$.</p>
  $$\vq{q_i} = W_Q \ve{e_i}$$
</div>

<div class="card">
  <h3 class="card-title">Pick a query token</h3>
  <div id="s07-chips"></div>
  <div id="s07-out" class="row"></div>
</div>

<script>
(function(){
  const S = document.getElementById('s07');
  const toks = AT.sentences.river;
  const F = AT.forward(toks);
  const out = S.querySelector('#s07-out');
  const chips = AT.ui.chips(toks, { active: 6, into: S.querySelector('#s07-chips'), onClick: (i) => { chips.setActive(i); draw(i); } });
  function draw(i){
    AT.clear(out);
    AT.ui.vec(F.E[i], { cls: 'e', label: '\\ve{e_{' + (i+1) + '}}', into: out });
    AT.ui.op('→', { into: out });
    AT.ui.vec(F.Q[i], { cls: 'q', label: '\\vq{q_{' + (i+1) + '}}', into: out });
  }
  draw(6);
})();
</script>
</section>
```

Rules recap (BRIEF §5): root `<section id="sNN" class="sec" data-title="…">`; **one** `<script>` with an IIFE; no globals;
all ids prefixed `sNN-`; query relative to `S`. `data-lit` = space-separated object keys this section *introduces*
(from the BRIEF §2 table: `e`, `q`, `k`, `v`, `a`, `d`, `ep`) — leave it `""` for sections that introduce nothing.
The strip lights chips cumulatively, so only list the objects your section is the first to introduce
(s02: `e`; s04: `a`; s05: `q k v`; s09: `d ep`; everyone else: `""`).

### Testing commands
```bash
cd /private/tmp/claude-501/-Users-nipun-git-attention/0546d9dc-78a5-49cb-9bbd-2ed5c6cbf59d/scratchpad
python3 assemble.py --only sections/sec07.html --out test07.html          # Part 2 (default); other parts: --part 1 --only sections1/sec07.html
node pres_test.mjs test00.html frames00                                    # present-mode contract test (the demo section); exits 1 on any failure
node walk.mjs test07.html frames07 1280 720 [--only s07]                   # present-mode walk: one screenshot per build of every frame, reports errors + frames that scroll
node qa.mjs test07.html --full --shot test07.png                           # desktop 1280 — pageErrors/consoleErrors/katexErrors must be [] and overflowX false
node qa.mjs test07.html --width 390 --full --shot test07m.png              # phone — same conditions
node qa.mjs test07.html --click '#s07-next' --eval "document.querySelector('#s07-out').textContent"   # interactions + assertions
```
Then **Read the PNGs** and fix what looks wrong (overlaps, clipped math, colours on the wrong object, unreadable numbers).
`node qa.mjs` counts console *warnings* as errors too — do not log anything. For close-ups use
`node shot.mjs test07.html crop.png --y 1200 --h 900` (clip of the full page) or `node zoom.mjs test07.html z.png '#s07 .card'` (2× element shot; prints the element's box).
For tooltips (`dotTable` / `mixTable` score and sum cells) use `node hovershot.mjs test07.html hov.png '#s07 .dt-comp.has-tip' [390]`: it hovers with a real mouse move
(the page scrolls smoothly, so Playwright's `hover()` right after a scroll can land on the wrong cell) and prints `{on, box}` for the `.dt-tip`.
`qa.mjs --eval` awaits a returned Promise, so `--eval "new Promise(r=>{…; setTimeout(()=>r(value),500)})"` lets you assert after animations/scroll.

---
## 1. Timing: what is rendered when

1. `<!--SHARED-->` is inserted before `<main>`: `window.__TOY__` then `shared.js`. **`AT` is available in every section script.**
2. Section scripts run in document order as the page parses (yours runs when your `<section>` is parsed).
3. The boot script at the end runs `renderMathInElement(document.body, …)` → all `$…$` / `$$…$$` present in the DOM
   at that moment get rendered, including markup your script created at parse time.
4. Anything you create **later** (on click / on slider) must render its own math: use `AT.tex(el, latex)` for a single
   formula, or build markup with `$…$` and call `AT.renderMath(el)`. All `AT.ui.*` components already do this for you
   (labels, captions, notes, callouts, reveals, stepper stages).
5. KaTeX auto-render ignores `script, style, textarea, pre, code, option` and anything with class `no-math`.
   Put `class="no-math"` on elements that legitimately contain a `$` character (rare). Rendering twice is harmless.

### How the notation strip reacts to your section
The boot script observes every `main .sec`. A section becomes *current* when it crosses the band 35 %–45 % down the viewport; the strip then
shows your `id` number and `data-title`, lights (cumulatively, never un-lights) the chips listed in `data-lit` of every section up to yours,
and rings the chips in *your* `data-lit` as "current". Sections scrolled past via a roadmap link are lit too. So: keep `data-title` short
(it is ellipsised on phones), and put in `data-lit` only the objects you are the first to introduce — listing more would light them early.
The strip chips are buttons that scroll to `data-target` (s02 for e, s05 for Q K V, s04 for α, s09 for Δe and e+Δe) — those ids must exist.

---
## 2. Colour + class catalogue

### Object colour scoping
Put `obj-X` on any element and its subtree gets `--oc` (object colour) and `--ot` (tint):
`obj-e` blue · `obj-q` purple · `obj-k` amber · `obj-v` teal · `obj-a` rose · `obj-d` green · `obj-ep` blue + `--oc2` green ring ·
`obj-m` teal (message `m_i`, drawn with dashed brackets) · `obj-neutral` grey.
Use them in fragment-local CSS as `color:var(--oc)` / `background:var(--ot)`. **Never** write a hex colour in a fragment.

### Math colour (KaTeX macros — defined in the shell and in `AT.katexOpts()`)
| macro | renders | use for |
|---|---|---|
| `\ve{e_i}` | blue | current representation `e`, `E`, `e_i^{(0)}` |
| `\vq{q_i}` | purple | query `q`, `Q` |
| `\vk{k_j}` | amber | key `k`, `K` |
| `\vv{v_j}` | teal | value `v`, `V` |
| `\va{\alpha_{ij}}` | rose | attention weight `α`, `A` |
| `\vd{\Delta e_i}` | green | contextual update `Δe`, `ΔE` |
| `\vp{e_i'}` | blue text, 2px green underline | updated representation `e'`, `E'`, `e+Δe` |
Colour the *object symbols only* — not operators, not `W_Q`, not `d_k`. Example:
`\va{\alpha_{ij}} = \operatorname{softmax}_j\!\left(\vq{q_i}^\top \vk{k_j}/\sqrt{d_k}\right)`, `\vp{e_i'} = \ve{e_i} + \vd{\Delta e_i}`.
Notation strings: BRIEF §3, exactly.

### Inline symbols in prose / SVG (no KaTeX): `<span class="sym sym-q">Q</span>` (`sym-e sym-q sym-k sym-v sym-a sym-d sym-ep`).
Italic KaTeX_Math font in the object colour. Use in headings, button labels, SVG-adjacent text. In `<svg><text>` use plain Unicode (`α`, `Δe`, `q_bank`).

### Layout classes
| class | what it is |
|---|---|
| `.sec` `.sec-head` `.sec-num` `.eyebrow` | section root, header row (top rule + big mono number), number, one-line "why we are here" |
| `.prose` | max-width 68ch text column; ≤ 60 words per block |
| `.companion` (+ `.after`, `data-kicker="…"`) | self-study prose block: 70ch, `--ink-2`, 17px, left rule. Place one after `.sec-head` and before the first widget; `.companion.after` at the end of a section gets the kicker "What to remember" automatically (override with `data-kicker`). Hidden in the strip's **lean** reading mode, so nothing a widget needs may live inside it |
| `.card` + `h3.card-title` | white panel, 1px border, radius 12, padding 28 (20 on phones), 24px below; put every widget in one. **One widget per row** is the default: stack cards, do not tile them |
| `.row` (+ `.top`, `.between`) | flex-wrap row, gap 16, centred (or top-aligned / space-between) |
| `.stack` | vertical flex, gap 14 |
| `.side-by-side` (alias `.grid-2`) `.grid-3` | equal columns, gap 24, collapse to 1 column ≤ 820px. Only for a deliberate comparison (before/after, river/cheque); never to save vertical space |
| `.dt-fig` `.dt-scroll` `table.dt` `.dt-corner` `.dt-ch` `.dt-rl` `.dt-num` `.dt-lead` `.dt-comp` `.dt-tint` `.is-heat` `.is-masked` `tr.is-hl` `tr.is-dim` `.dt-foot` `.dt-tip` `.dt-cap` `.dt-note` (+ `.lg` on `.dt-fig`) | spreadsheet table (use `AT.ui.table` / `dotTable` / `mixTable`) |
| `.notation-card` `.notation-group` `.notation-title` `.dt-notation` | the symbol / meaning / shape card (use `AT.ui.notationCard`); the shell shows one in the hero inside `details.notation-details` |
| `.scroll-x` | horizontal scroll container for anything wide (never let the body scroll sideways) |
| `.dotgrid` | faint dot-grid background for diagram areas (put on the div that holds your SVG) |
| `.center` `.spacer` `.muted` `.faint` `.small` `.mono` `.warn-text` | small helpers (`.warn-text` is the only allowed use of `--warn`) |
| `.btn` `.btn-primary` `.btn-big` `.btn-quiet` `.btn-row` | buttons — always a real `<button type="button">`; label says what it does |
| `.callout` `.callout-key` `.callout-note` `.callout-warn` | statement boxes with an automatic "Key idea / Note / Careful" kicker |
| `.reveal` | `<details>` question/answer (use `AT.ui.reveal`) |
| `.chips` `.chip` `.chip-i` `.chip-t` `.is-active` `.is-muted` `.is-hl-k/.is-hl-v/.is-hl-q` `.is-slot` | token chips (use `AT.ui.chips`) |
| `.vec` `.vec-label` `.vec-body` `.vec-axes` `.vec-ax` `.vec-cells` `.cell` `.is-hl` `.is-dim` `.vec-dim` `.has-axes` (+ `.sm`/`.lg` on `.vec`) | vector with its axis header (use `AT.ui.vec`) |
| `.calc` `.calc-scroll` `.calc-t` `.calc-ax` `.calc-rl` `.calc-term` `.calc-zero` `.calc-res` `.calc-line` `.calc-r` `.calc-outrow` | worksheets (use `AT.ui.dotCalc` / `AT.ui.matVecCalc`) |
| `.calc-pop` `.calc-pop-title` `.calc-pop-body` `.calc-close` `.has-calc` | the click popover on score cells (`dotTable`, `mat`/`heat` with `calc`) |
| `.dt-w` `td.is-zero` `tbody th.dt-tint` | weight-matrix table (use `AT.ui.wTable`) |
| `.mat` `.mat-grid` `.mat-cl` `.mat-rl` `.mat-lb/.mat-rb` `.mat-cap` `.mat-dim` `.cell.is-heat/.is-masked/.is-leak/.is-hl/.is-hl-row/.is-hl-col` (+ `.sm`) | matrix (use `AT.ui.mat` / `AT.ui.heat`) |
| `.mat-op` | big operator glyph between vectors/matrices in a `.row` (use `AT.ui.op('+')`) |
| `.bars` `.bl` `.bt` `.bf` `.bv` `.is-hl` `.is-dim` (+ `.lg`) | probability bars (use `AT.ui.bars`) |
| `.slider` `.slider-label` `.slider-val` | range input (use `AT.ui.slider`) |
| `.toggle` `.toggle-track` `.toggle-knob` `.toggle-text` `.toggle-state` | switch button with `aria-pressed` (use `AT.ui.toggle`) |
| `.stepper` `.stepper-bar` `.stepper-list` `.stepper-stage` `.is-current` `.is-done` `.stage-title` `.stage-tex` `.stage-note` | stepper (use `AT.ui.stepper`) |
| `.motif` (+ `.sm`) | pipeline SVG (use `AT.motif`) |
| `.flow-dot` | animated dots (use `AT.ui.flow`) |
| `.legend` | 7-object legend (use `AT.ui.legend`) |

Tokens available as CSS variables: `--paper --ink --ink-2 --ink-3 --line --card --grid --c-e --c-q --c-k --c-v --c-a --c-d --c-ep-grad --c-mask --warn --t-e --t-q --t-k --t-v --t-a --t-d --t-neutral --font-ui --font-mono --font-math --radius --radius-s --dur`.

---
### Layout rhythm (shell tokens, do not restyle in fragments)
Sections are 128px apart (96 on phones); `.sec-head` has 40px below it; `h2` is 44px on desktop; cards have 28px padding and 24px between them;
prose runs 68ch. The rhythm of a section is: `.sec-head` → `.companion` (self-study prose) → `.prose` (the short in-class text) → one `.card` per widget,
each followed by its own `.callout` / `.reveal` when needed → optional `.companion.after`. A student reading alone gets the companion blocks; the strip's
"Reading: full / lean" toggle (persisted in `localStorage['at-reading']`, default full) hides them for projection.

## 3. `AT` API — data & math

All functions are defensive: bad/empty input returns empty arrays / zeros / `'—'`, never throws.

| member | description |
|---|---|
| `AT.model` | the parsed `toy.json` (`window.__TOY__`): `d_model d_k d_v vocab tok_emb pos_emb W_Q W_K W_V W_O W_vocab b_vocab sentences candidates axes notes` |
| `AT.axes` | the named coordinates: `{ named: true, e: [4 names], qk: [3 names], v: [3 names], short: { e, qk, v } }` (`toy.json` → `axes`; index labels `'1'..` and `named:false` when the toy has none). Never retype the names: read them here |
| `AT.axisKind(cls, n)` | `'e'` for `e/ep/d` of width `d_model`, `'qk'` for `q/k` of width `d_k`, `'v'` for `v/m` of width `d_v`, else `null` |
| `AT.axesFor(kind, n, short=true) → [{label, title}]` | axis labels for `kind` = `'e' | 'qk' | 'v' | 'vocab' | cls | [labels]`; `null` when nothing matches. `AT.axisLabels(kind, n)` returns the labels only |
| `AT.vocab` | `string[20]` |
| `AT.sentences.river`, `AT.sentences.cheque` | token arrays (10 each, original casing; `bank` is index 6, prediction slot after index 9) |
| `AT.candidates.river/.cheque` | the four candidate next tokens per sentence |
| `AT.d_model AT.d_k AT.d_v AT.sqrt_dk` | dimensions (4, 3, 3, √3) |
| `AT.lower(tok)` | lowercase lookup key |
| `AT.embed(tokens) → E` | rows `e_i^{(0)} = tok_emb[lower(tok_i)] + pos_emb[i]` |
| `AT.forward(tokens, {mask=true, scale=true}) → R` | see below |
| `AT.baseline(tokens) → {E, logits, probs}` | output head applied directly to `e^{(0)}` (no attention) |
| `AT.head(vecRow) → {logits, probs}` | output head on any single row vector (e.g. a pooled context, or `Enew[i]`) |
| `AT.softmax(arr)` | handles `-Infinity` (→ 0); all `-Infinity` → zeros |
| `AT.dot(a,b) AT.add(a,b) AT.sub(a,b) AT.scale(v,s) AT.matmul(A,B) AT.transpose(M) AT.sum(a) AT.mean(rows) AT.norm(a) AT.argmax(a) AT.zeros(n) AT.range(n)` | row-vector linear algebra. `matmul(vec, M)` returns a vector. |
| `AT.topk(probRow, k=5) → [{tok, p, i}]` | sorted descending |
| `AT.probsFor(probRow, ['water','boats']) → [{tok,p,i}]` | probabilities for named tokens in that order |
| `AT.fmt(x, decimals=2)` | display string with U+2212 minus; `−∞` for `-Infinity`; `—` for NaN/undefined; `-0.001` → `0.00` |
| `AT.fmtSigned(x, d)` | same with a leading `+` on positives |
| `AT.heatColor(a)` | `0..1` → `rgb()` from white to `--c-a` |
| `AT.objColor(cls) AT.objTint(cls)` | `'e'|'q'|…` → `var(--c-e)` / `var(--t-e)` strings for SVG fills |

### `AT.forward` result (row-vector convention; BRIEF §4)
```js
const F = AT.forward(AT.sentences.river);          // causal mask on, scaling on
F.tokens  // the tokens        F.T // 10
F.E       // T×4  e_i^{(0)}    F.Q, F.K // T×3   F.V // T×3
F.Sraw    // T×T  q_i·k_j  (no scale, no mask)
F.Sfull   // T×T  q_i·k_j/√d_k, NO mask   ← "before the mask" view
F.S       // T×T  scaled scores with -Infinity where j>i (when mask on)
F.A       // T×T  softmax rows; masked entries are exactly 0; every row sums to 1
F.Mmsg    // T×3  m_i = Σ_j α_ij v_j
F.Delta   // T×4  Δe_i = m_i W_O
F.Enew    // T×4  e_i' = e_i + Δe_i
F.logits  // T×20 e_i' W_vocab + b       F.probs // T×20 softmax rows
F.mask, F.scale // the options this result was computed with (booleans)
// Options: AT.forward(toks, {mask:false})  → leakage demo (then F.S === F.Sfull and rows of F.A are non-zero everywhere);
//          AT.forward(toks, {scale:false}) → unscaled scores (F.S and F.Sfull are then q·k without /√d_k)
```
Verified: every number in `F` (both sentences, masked/unmasked/unscaled, `AT.baseline`, `AT.head`) equals `toy_ref.mjs` to 0 ulp
(`node cmp_shared.mjs` re-runs that comparison; keep it passing if you ever touch the math).
**Unknown tokens do not throw**: `AT.embed` silently uses a zero token vector (only the position survives) and `AT.probsFor`
returns `p: 0` for a token outside the vocabulary — so pass only tokens from `AT.vocab` (any casing). Positions come from
the index in the array you pass: a 7-token walkthrough uses `pos_emb[0..6]`, so slicing `AT.sentences.river.slice(0,7)` is exact.
`F.A[6]` is the `bank` row; `F.probs[9]` are the next-token probabilities after "watched the". These numbers agree with `toy_ref.mjs`.

---
## 4. `AT` API — KaTeX

| member | description |
|---|---|
| `AT.tex(el, latex, {display=false})` | renders into `el` (Element or id string) with the colour macros; returns `el` |
| `AT.texStr(latex, {display=false}) → html` | string for embedding in generated markup (`el.innerHTML = 'weight ' + AT.texStr('\\va{\\alpha_{ij}}')`) |
| `AT.renderMath(el)` | auto-render `$…$` / `$$…$$` inside `el` (for markup you create after boot) |
| `AT.katexOpts(extra)` | the shared options object (`throwOnError:false, trust:true, strict:false, macros`) |
| `AT.macros` | the macro table |

```js
AT.tex(S.querySelector('#s12-eq'), '\\va{\\alpha_{ij}} = \\operatorname{softmax}_j\\!\\left(\\frac{\\vq{q_i}^\\top \\vk{k_j}}{\\sqrt{d_k}}\\right)', { display: true });
```
Remember to double the backslashes inside JS strings (or use `String.raw`).
Never call `katex.render` / `renderMathInElement` yourself: the colour macros expand to `\htmlClass`, which only works with the
`trust:true` option that `AT.katexOpts()` sets — without it the macros render as errors. `AT.tex(el, …, {display:true})` also adds the
class `tex-display` (scrolls horizontally if the formula is wider than the card).

---
## 5. `AT` API — UI components

Every component returns an `HTMLElement` (stepper and motif return a small API object holding `.el`) and accepts
`into: element` in its options to append itself. Elements carry helper methods (`setActive`, `update`, …) listed below.
`AT.h(tag, attrs, ...children)` is a tiny element builder (`AT.h('div', {class:'row', id:'s07-x'}, childEl, 'text')`),
`AT.clear(el)` empties an element, `AT.append(el, child)` appends nodes/strings/arrays.

### `AT.ui.chips(tokens, opts)` → `.chips`
`opts: { active: i | [i…], muted: [i…], onClick(i, chipEl), size: 'md'|'lg', numbered: true, slot: '___'|string, into }`
Chips show position (1-based) + token. With `onClick` they are real `<button aria-pressed>`s. Methods: `el.setActive(i|[..])`,
`el.setMuted([i…])`, `el.setMark([i…], 'k'|'v'|'q')` (tints chips in an object colour; each kind is an independent class, so
`setMark([], 'k')` clears the amber marks without touching the teal ones), `el.chips` (the chip elements).
Chips are only `<button>`s when `onClick` is given; otherwise they are inert `<span>`s (use that for purely illustrative rows).
`slot:'___'` appends a dashed "prediction slot" chip.
```js
const chips = AT.ui.chips(AT.sentences.river, { active: 6, muted: [7,8,9], slot: '___', into: host,
  onClick: (i) => { chips.setActive(i); redraw(i); } });
```

### `AT.ui.vec(values, opts)` → `.vec`
`opts: { cls: 'e'|'q'|'k'|'v'|'a'|'d'|'m'|'ep'|'neutral', label: latex|null, decimals: 2, highlight: index, size: 'sm'|'lg', dims: true|'1×4', axes: undefined|false|'e'|'qk'|'v'|[labels], into }`
Bracketed row of monospace cells tinted in the object colour; label rendered with KaTeX. **By default a small header row of axis names sits above the cells**
whenever the width matches the object (`e`, `ep`, `d` of width 4 → the `e` axes; `q`, `k` of width 3 → the `q/k` axes; `v`, `m` of width 3 → the `v` axes; `a`, `neutral` and
other widths get no header). Short names are shown, the full name is in the `title` tooltip of each header and in the cell's `aria-label`. `axes:false` hides the header,
`axes:'e'` or an array forces one. Methods: `el.update(values)`, `el.highlight(i)`, `el.setLabel(latex)`, `el.setAxes(spec)`, `el.cells`, `el.axes` (the labels or `null`).
```js
AT.ui.vec(F.Delta[6], { cls: 'd', label: '\\vd{\\Delta e_7}', dims: true, into: row });
```

### `AT.ui.table(rows, opts)` → `figure.dt-fig` (wrapping a real `table.dt`)
**When to use which.** Use a table whenever the coordinates have names (topic axes in the search example, "coord 1..3" of a projection), whenever the point is a
column-by-column comparison (a query against several keys), or whenever a weighted sum is the point (α times value rows). Keep `AT.ui.vec` for a single anonymous
vector shown inline, and `AT.ui.mat` for a matrix seen as one object. Always label the coordinate columns, at least with indices (`'1','2','3','4'`): a bare row of
numbers with the axis names floating in a caption is what the instructor asked us to stop doing.

`rows`: array of number arrays, or of objects when `cols[j].key` is set.
**Default columns:** when `cols` is omitted the columns are named after the axes of the row object (`rowCls` / `cls`) if the width matches (`e/ep/d` × 4, `q/k` × 3, `v/m` × 3),
otherwise `'1'..'n'`; pass `axes:'e'|'qk'|'v'` to choose, `axes:false` for plain indices. Column objects accept `title` (shown as a tooltip on the header); the axis defaults carry the full names there.
`opts: { cols: [ label | { label, key?, title?, cls?, decimals?, heat?, heatMax? } ], axes?, rowLabels: [string|latex], rowCls: [cls per row] | cls, highlightRow: i, dimRows: [i…],
heatCols: [col indices] (white → --c-a by value/heatMax, default heatMax 1), lead: [ { label, values: number[] | fn(row,i), cls?, heat?, decimals? } ] (extra columns on the LEFT, after the label),
computed: [ { label, fn(row,i) → number|string, values?, cls?, decimals?, heat? } ] (extra columns on the RIGHT, separated by a rule),
footer: { label, values: number[] | string, lead?: [], computed?: [], cls?, decimals? } (a bottom row such as the weighted sum),
decimals: 2, caption: html-with-$math$, note: html (a quiet line under the caption), cornerLabel: text|latex (top-left header cell, e.g. 'item' or 'token'),
sticky: true (first column sticky while the table scrolls sideways), size: 'md'|'lg', cls: object class for the whole figure, into }`
Labels (`cols[].label`, `rowLabels`, `cornerLabel`, `footer.label`) are plain text, or latex when wrapped in `$…$` (rendered with `AT.tex`), or html with inline `$…$`.
Numbers are monospace, tabular, right aligned, 14px (`lg`: 16px); header cells are small caps in `--ink-2`; the header row and the label column carry the paper tint;
object tints apply per column (`cols[j].cls`) or per row (`rowCls`) with the same `--t-*` tokens as `vec`/`mat`; a string cell `'×'` renders as a hatched masked cell.
The table scrolls inside its own `.dt-scroll`; it never widens the body.
Methods: `fig.update(rows)`, `fig.setFooter(values, leadVals?, compVals?)`, `fig.setHighlightRow(i)`, `fig.setDim([i…])`, `fig.onRow(event, fn(i, tr))`, `fig.onCell(event, fn(i, j, td))`,
`fig.tipOn(td, text | () => text)` (hover/focus tooltip on any cell; the cell becomes keyboard focusable), `fig.showTip(td, text)`, `fig.hideTip()`.
Elements: `fig.table`, `fig.rowEls[i]`, `fig.cells[i][j]`, `fig.leadCells[i][k]`, `fig.computedCells[i][k]`, `fig.rowLabelEls[i]`, `fig.footer.{label, cells, lead, computed}`, `fig.cap`, `fig.note`, `fig.tip`, `fig.colLabels` (the column labels in use).
```js
// W_Q as a table: rows are e-coordinates, columns are q-coordinates, plus a computed column
AT.ui.table(AT.model.W_Q, { cols: [{ label: '$\\vq{q}$ coord 1', cls: 'q' }, { label: '$\\vq{q}$ coord 2', cls: 'q' }, { label: '$\\vq{q}$ coord 3', cls: 'q' }],
  rowLabels: ['$\\ve{e}$ coord 1', '$\\ve{e}$ coord 2', '$\\ve{e}$ coord 3', '$\\ve{e}$ coord 4'], cornerLabel: '$W_Q$',
  computed: [{ label: 'row sum', fn: (row) => AT.sum(row) }], caption: 'Rows are input coordinates, columns are output coordinates.', into: host });
// an attention block as a heat table with bank's row highlighted
AT.ui.table(F.A.slice(0, 7).map(r => r.slice(0, 7)), { cols: toks.slice(0, 7), rowLabels: toks.slice(0, 7).map((t, i) => (i + 1) + ' ' + t), cornerLabel: 'reads from →',
  rowCls: 'a', heatCols: AT.range(7), highlightRow: 6, size: 'lg', into: host });
```

### `AT.ui.dotTable(q, K, opts)` → `figure.dt-fig`
The query as one highlighted row (`cls q`) directly above the key rows (`cls k`) under the **same** column headers, with a computed right-hand column
**score** = `q·k_j` (`cls a`) for every key row. Hovering or focusing a score cell shows the arithmetic column by column
(`2.0×2.2 + 0.6×0.6 + 0.2×0.2 + 0.0×0.0 = 4.8`; with `scaled:true` it continues `, then 4.8 ÷ √3 = 2.77` and the header reads `score/√d_k`).
**Clicking (or Enter/Space on) a score cell opens a `dotCalc` worksheet popover** for that key (the two rows, the products, the sum, the scaling); Escape, the × button or a click elsewhere closes it (`fig.openCalc(j)` opens it from code, `fig.calcPop` is the popover).
`opts: { cols: [labels] (default: the q/k axis names when the width is d_k, else '1'..), axes?, rowLabels: [per key] (default $\vk{k_j}$), queryLabel: text|latex (default $\vq{q}$), cornerLabel, decimals: 2, scoreDecimals?,
scaled: false (divide by √d_k; d_k defaults to q.length, pass d_k: AT.d_k to be explicit), mask: [key indices that are masked] (row dimmed, score cell hatched, tooltip explains −∞ → weight 0),
scoreLabel?, caption, note (default 'Hover or focus a score…'; pass '' to drop it), size, sticky, into }`
Extra fields: `fig.scores` (numbers, `-Infinity` when masked), `fig.raw` (unscaled dot products), `fig.scoreCells[j]`, `fig.queryRow`, `fig.keyRows[j]`, plus every `table` method.
```js
// the search example (SPEC §5 numbers are illustrative; one decimal)
AT.ui.dotTable([2.0, 0.6, 0.2, 0.0], keys, { cols: ['gradients', 'backward', 'optimisation', 'architecture'], rowLabels: items, queryLabel: 'query', cornerLabel: 'item', decimals: 1, into: host });
// the real thing: bank's query against the first seven keys, scaled, nothing masked
AT.ui.dotTable(F.Q[6], F.K.slice(0, 7), { rowLabels: toks.slice(0, 7).map((t, j) => (j + 1) + ' ' + t), queryLabel: '7 bank', cornerLabel: 'token', scaled: true, d_k: AT.d_k, into: host });
```

### `AT.ui.mixTable(alpha, V, opts)` → `figure.dt-fig`
Value rows (`cls v`) with a left **α** column (`cls a`, heat-filled), optionally every cell shown as `α_j·v_j` (`weighted:true`), and a footer row
**Σ_j α_j v_j** (`cls m` by default, `footerCls:'d'` for a Δe-coloured footer) whose cells show the weighted terms row by row on hover/focus
(`0.05×1.03 + 0.18×(−1.08) + … = −0.85`). Rows whose weight is 0 (masked tokens) are dimmed automatically (`dimZero:false` to keep them, or pass `dimRows`).
`opts: { cols (default: the v axis names when the width is d_v, else indices), axes?, rowLabels (default $\vv{v_j}$ or $\va{\alpha_j}\vv{v_j}$), cornerLabel, decimals: 2, alphaDecimals?, weighted: false, footer: true|false (false = no footer row at all), footerLabel (default the Σ latex), footerCls: 'm', alphaLabel, heat: true, highlightRow, caption, note, size, sticky, into }`
Extra: `fig.sum` (the footer vector, equals `F.Mmsg[i]` when you pass `F.A[i]` and `F.V`), `fig.alpha`, `fig.setAlpha(newAlpha)` (stores a copy of the new weights and re-fills the α column, the weighted cells, the dims and the footer from them: use it under a slider; passing the creation array mutated in place also works).
```js
const mix = AT.ui.mixTable(F.A[6].slice(0, 7), F.V.slice(0, 7), { rowLabels: toks.slice(0, 7).map((t, j) => (j + 1) + ' ' + t), cornerLabel: 'token', weighted: true,
  caption: 'The footer is $m_7 = \\sum_j \\va{\\alpha_{7j}}\\,\\vv{v_j}$.', into: host });
mix.setAlpha(AT.softmax(scores));   // e.g. after a temperature slider
```

### `AT.ui.wTable(W, opts)` → `figure.dt-fig.dt-w`
A weight matrix as a table a student can read aloud: row headers are the **input axes**, column headers the **output axes**, zero cells greyed (`td.is-zero`), every cell carrying
a tooltip `water → setting: water?: 1.0`. `opts: { from: 'e'|'v'|'qk' (default 'e'), to: 'qk'|'v'|'e'|'vocab' (default 'qk'), cls: output object ('q'|'k'|'v'|'e'…, default from `to`),
decimals: 1, cornerLabel (default `$\ve{e} \rightarrow \vq{q}$`-style), caption, note, size, sticky, into }`. Extra: `fig.readings` (`['water → water? 1.0', …]` for the non-zero cells), `fig.inAxes`, `fig.outAxes`, plus every `table` method.
```js
AT.ui.wTable(AT.model.W_Q, { from: 'e', to: 'qk', cls: 'q', caption: '$W_Q$: an $\\ve{e}$ axis (row) feeds a $\\vq{q}$ axis (column).', into: host });
AT.ui.wTable(AT.model.W_K, { from: 'e', to: 'qk', cls: 'k', into: host });   AT.ui.wTable(AT.model.W_O, { from: 'v', to: 'e', into: host });
AT.ui.wTable(AT.model.W_vocab, { from: 'e', to: 'vocab', into: host });      // 4 × 20, scrolls sideways inside the figure
```

### `AT.ui.dotCalc(q, k, opts)` → `.calc.calc-dot`   (worksheet: the two vectors being dot-producted)
KaTeX-free, renders instantly, fits a card at 1280 and scrolls inside at 390. Row 1 the query values (`cls q`), row 2 the key values (`cls k`), row 3 the products `q_c × k_c`
(neutral, zero products greyed), each column headed by its axis name; then `sum = a + b + c = S` and, with `scale:true`, `S / √d_k = s` (rose result).
`opts: { axes: 'qk' (default; 'e' | 'v' | [labels] | false for indices), qLabel: 'q', kLabel: 'k' (plain text, e.g. 'q · 7 bank'), scale: true, d_k (default q.length), decimals: 2, resultDecimals?, into }`
Methods: `el.update(q, k, { qLabel?, kLabel? })` refills in place. Fields: `el.dot`, `el.score`, `el.rows.{q,k,prod}`, `el.axes`.
```js
const dc = AT.ui.dotCalc(F.Q[6], F.K[5], { axes: 'qk', qLabel: 'q · 7 bank', kLabel: 'k · 6 river', scale: true, d_k: AT.d_k, into: host });
dc.update(F.Q[6], F.K[1], { kLabel: 'k · 2 fisherman' });
```

### `AT.ui.matVecCalc(x, W, opts)` → `.calc.calc-mv`   (worksheet: a projection `x W`, one line per output coordinate)
The input row `x` with its axis names on top, then one line per **output** coordinate, `water? = 0.70×1.0 + 0.70×0.0 + (−0.10)×0.0 + 0.90×0.7 = 1.33`, aligned under the input
axes (terms whose `W` entry is 0 are greyed, every term has a tooltip `water → setting: water?: …`), then the resulting output row as a `vec` with the output axes.
`opts: { from: 'e' (input axes), to: 'qk'|'v'|'e', cls: output object (default 'q' for qk, 'v' for v, 'd' for e), xLabel: 'x', outLabel: 'x W' (plain text), decimals: 2, wDecimals: 1, into }`
Methods: `el.update(x, { xLabel?, outLabel? })`. Fields: `el.out` (the product), `el.outVec`, `el.lines[o].{terms, res}`, `el.xCells`.
```js
AT.ui.matVecCalc(F.E[6], AT.model.W_Q, { from: 'e', to: 'qk', cls: 'q', xLabel: 'e · 7 bank', outLabel: 'q · 7 bank', into: host });
```

### `AT.ui.popover(fig, { label })` → `{ el, body, show(anchorEl, title, content), hide(), isOpen() }`
The click popover used by `dotTable`, `mat` and `heat` (`.calc-pop`, appended to the figure so the scroll container cannot clip it, clamped to the viewport, closed by Escape,
the × button or a click outside). Use it if a fragment needs its own worksheet on click: `pop.show(td, 'title', AT.ui.dotCalc(q, k, {...}))`.

### `AT.ui.notationCard(opts)` → `.notation-card`
Small tables (symbol · meaning · shape): "One token at a time" (`e_i^{(0)}, e_i, q_i = e_i W_Q, k_j, v_j, s_ij, α_ij, m_i, Δe_i = m_i W_O, e_i' = e_i + Δe_i`),
"All tokens at once" (`E, Q, K, V, S, M, A, H, ΔE, E'`), "Sizes and learned weights" (`T, d_model, d_k, d_v, W_Q, W_K, W_V, W_O, W_vocab, b`) and, when the toy names its
coordinates, "Named coordinates (illustrative)" (the `e`, `q/k` and `v` axis names read from `AT.axes`, with a note that the toy was written by hand so the names are true). Shapes are latex plus the toy
value from `AT` (`1×d_model = 1×4`). `opts: { part: 'part1'|'part2'|'part3' (default: the part's `notation` field), groups: ['token','matrix','sizes','axes'] (Part 1 uses 'mlp','sizes','axes'; Part 3 adds 'train','block'), only: [latex symbols], into }`. The shell already places one in the hero
(`details#notation`, collapsed); use it again only if a section genuinely needs the reference in place (`groups:['matrix']` in s16, say). `AT.notation` is the row list.

### `AT.ui.mat(rows, opts)` → `figure.mat`
`opts: { cls, rowLabels: [...], colLabels: [...], axes: undefined|false|{rows, cols}, decimals: 2, heat: false, heatMax: 1, mask: true|'causal'|boolMatrix, leak: false, maskText: '×',
highlight: {r,c} | {row} | {col}, caption: html-with-$math$, dims: true|'T×d', size: 'sm', calc: { Q, K, scaled: true, d_k, axes }, into }`
**Default labels:** with `colLabels` omitted the columns are named after the axes when the width matches the object (`e/ep/d` × 4, `q/k` × 3, `v/m` × 3); with `rowLabels` omitted
the projection matrices get their input axes too (`W_Q`/`W_K`/`W_V` as `cls q/k/v`: rows are the `e` axes; `W_O` as `cls neutral` 3 × 4: rows are the `v` axes, columns the `e` axes).
Full names sit in the `title` of each label. `axes:false` keeps the old unlabelled look; `axes:{rows:'e', cols:'qk'}` forces a choice.
**`calc`:** for a score matrix pass `calc: { Q: F.Q, K: F.K, scaled: true, d_k: AT.d_k }` and every cell becomes focusable and **opens a `dotCalc` popover on click**
(`q_i · k_j` with the token labels; masked cells add the −∞ explanation, heat cells add the weight after the softmax). `fig.calcPop` is the popover.
Bracketed matrix; `heat:true` fills cells white→rose by value `v/heatMax` (clamped to 0..1 — negative values render white, so for a
score matrix `S` pass `heatMax` explicitly or use a `neutral` non-heat matrix); masked cells are hatched grey with `×`.
`leak:true` marks the *future* cells `j>i` for the leakage story in both states: with the mask **on** they are hatched rose
(“this is what would leak”); with the mask **off** their real values are shown inside a 2px `--warn` ring (`.cell.is-leak:not(.is-masked)`).
`fig.setMask(false)` / `fig.setMask(true)` switches between the two views in place (the mask toggle demo in s13).
Methods: `fig.update(rows, newOpts?)`, `fig.setHighlight(hl)`, `fig.setMask(mask)`, `fig.onCell('click'|'mouseenter', (i,j,cellEl)=>{})`, `fig.cells[i][j]`.
```js
AT.ui.mat(AT.model.W_Q, { cls: 'q', caption: '$W_Q$', dims: true, into: host });   // rows water/finance/person/glue, columns water?/finance?/who? by default
AT.ui.mat(F.Sfull, { cls: 'neutral', size: 'sm', rowLabels: toks, colLabels: toks, highlight: { r: 6, c: 5 }, caption: 'Scores before the mask. Click a cell for its worksheet.', calc: { Q: F.Q, K: F.K, scaled: true, d_k: AT.d_k } });
```

### `AT.ui.heat(A, opts)` → `figure.mat.heat`
`opts: { rows: tokens, cols: tokens (defaults to rows), mask: true (causal) | false | boolMatrix, leak, decimals: 2, onHover(i,j,cellEl), onClick(i,j,cellEl), highlightRow: i, caption, size, calc: { Q, K, scaled, d_k } (click → dotCalc popover, as for `mat`), into }`
Attention heatmap (`cls:'a'`, heat on, `heatMax` 1). Masked cells hatched with a cross. Methods as `mat` plus `fig.setHighlightRow(i)`.
With `onHover` every cell also becomes keyboard-focusable (`tabindex=0`) and `onHover` fires on focus, so the hover text is reachable without a mouse.
`leak:true` behaves as documented under `mat` (hatched rose when masked, `--warn` ring around the real future values when `mask:false`).
```js
const heat = AT.ui.heat(F.A, { rows: toks, mask: true, highlightRow: 6, onHover: (i,j) => msg.textContent = AT.fmt(F.A[i][j], 3), into: host });
```

### `AT.ui.bars(items, opts)` → `.bars`
`items: [{label, p}]` (or `AT.topk` / `AT.probsFor` output — `tok` is accepted as label). `opts: { max: 1, cls: 'neutral'|'a', highlight: label|[labels], decimals: 2, sorted: true, size: 'lg', into }`
Methods: `el.update(items)` (animates widths in place when the label order is unchanged; with `sorted:true` (default) the order can change
between updates, which rebuilds the rows — pass `sorted:false` when you want the in-place animation).
`max` defaults to 1 (full bar = probability 1). Toy probabilities top out around 0.37, so for a "which wins" view pass
`max: Math.max(...items.map(x => x.p))` as the demo does; keep `max:1` when the absolute scale matters (e.g. comparing baseline vs attention).
```js
const bars = AT.ui.bars(AT.probsFor(F.probs[9], AT.candidates.river), { cls: 'neutral', sorted: false, highlight: 'water', decimals: 3, into: host });
bars.update(AT.probsFor(AT.forward(other).probs[9], AT.candidates.river));
```

### `AT.ui.slider(opts)` → `.slider`
`opts: { label, labelTex?, min, max, step, value, format(v)→string, onInput(v), into }`. Real `<input type="range">` with a live `<output>`.
Methods: `el.value()`, `el.setValue(v, fire=true)`, `el.input`.
```js
AT.ui.slider({ label: 'Window size K', min: 1, max: 10, step: 1, value: 3, onInput: (K) => draw(K), into: host });
```

### `AT.ui.toggle(opts)` → `button.toggle[aria-pressed]`
`opts: { label, on: false, onChange(on), onText:'on', offText:'off', into }`. Methods: `btn.get()`, `btn.set(on, fire=true)`.
```js
AT.ui.toggle({ label: 'Causal mask', on: true, onChange: (on) => redraw(on), into: host });
```

### `AT.ui.stepper(opts)` → `{ el, stage, list, go(i), next(), prev(), reset(), index(), steps, buttons }` (the same object is reachable from the element as `el.stepperApi`, which present mode uses to drive the steps)
`opts: { steps: [{ title, tex?, note?, render(stageEl, ctx), keep?: false }], el?: existing element to fill, into, onChange(i, step), big: false, compact: false, hideList: false, start: 0|false, nextLabel: 'Next step' }`
"Previous / Next step / Reset" buttons (+ ← → keys), a numbered step list where only the current step is highlighted, and a stage area.
For each step the stage is cleared (unless `keep:true`), then `title`, `tex` (display math), `note` (html with `$…$`) are shown, then your `render(stageEl, ctx)` appends whatever you like (`ctx = {index, step, steps, stepper}`); math in the stage is auto-rendered afterwards.
`big:true` = the large prominent Next-step button (use it in s15).
Also: every list item is itself a control (click / Enter / Space jumps to that step), so students can go back to any stage; the stage is
**cleared** on every step unless `keep:true`, so build everything a step needs inside its own `render`; an exception thrown inside `render`
is caught and reported with `console.error` — which `qa.mjs` counts as a failure, so it will not go unnoticed.
```js
AT.ui.stepper({ big: true, into: host, steps: [
  { title: 'Tokens', render: (el) => AT.ui.chips(toks, { into: el }) },
  { title: 'Compute the query', tex: '\\vq{q_7} = W_Q \\ve{e_7}', note: 'Only token 7 needs a query right now.',
    render: (el) => AT.ui.vec(F.Q[6], { cls: 'q', label: '\\vq{q_7}', into: el }) }
]});
```

### `AT.ui.reveal(questionHTML, answerHTML, opts)` → `details.reveal`
`opts: { open: false, hint: 'Reveal', onToggle(open), into }`. Math in both parts is rendered on creation and on every toggle.
```js
AT.ui.reveal('Is $\\vq{q_i}$ the new representation of token $i$?', 'No. It is a temporary projection of $\\ve{e_i}$.', { into: host });
```

### `AT.ui.callout(html, opts)` → `.callout`
`opts: { kind: 'key'|'note'|'warn', into }`. Math inside is rendered.
```js
AT.ui.callout('Attention does not replace the existing representation. It computes context-dependent information that can be added to it.', { kind: 'key', into: S });
```

### `AT.ui.card(title|null, ...children)` → `.card`
Children: elements, HTML strings (math rendered), arrays, or a trailing `{ into, cls }` options object.
```js
host.appendChild(AT.ui.card('Retrieved mixture', AT.ui.vec(m, { cls: 'm', label: 'm' }), '<p class="muted">Rows of $A$ sum to 1.</p>'));
```

### `AT.ui.legend(opts)` → `ul.legend`   `opts: { only: ['q','k','v'], into }` — the seven-object legend (or a subset).

### `AT.ui.op(text='→', opts)` → `.mat-op`   big operator glyph to place between vectors/matrices inside a `.row` (`'+'`, `'='`, `'→'`, `'·'`).

### `AT.motif(el, opts)` → `{ el, svg, stages, setActive(stage) }`
`opts: { active: 'e'|'qkv'|'att'|'delta'|'add'|'ep'|null, labels: true, size: 'sm'|'md' }`
The recurring pipeline **e → (Q,K,V) → attention → Δe → ⊕ → e+Δe** with the dashed blue residual edge. One stage highlighted, others dimmed.
Use it in s09, s11, s15, s19 — same drawing every time; just change the active stage.
```js
const motif = AT.motif(S.querySelector('#s09-motif'), { active: 'delta' });
motif.setActive('add');
```

### `AT.ui.flow(container, opts)` → `Promise`
`opts: { from: [elements], to: element, weights: [0..1 …], color: 'var(--c-v)', duration: 900, stagger: 70, skipZero: true, onDone }`
Dots travel from each `from` element to `to` (dot size/opacity ∝ weight). `container` must contain both (it is made `position:relative`).
Instant no-op under `prefers-reduced-motion` (the promise still resolves and `onDone` still fires — put the "arrival" state change in `.then`,
never in a timer). Returns a promise that resolves when all dots have arrived.
Positions are measured with `getBoundingClientRect()` at call time, so call it only when both ends are in the DOM and laid out
(not inside a `display:none` stage), and make `container` the nearest ancestor that does not scroll (the dots are absolutely positioned in it).
Weights ≤ 0.001 send no dot (`skipZero`), so masked entries of an `A` row are naturally silent.
```js
AT.ui.flow(S.querySelector('#s08-arena'), { from: chips.chips.slice(0, 7), to: target.querySelector('.vec-cells'), weights: F.A[6].slice(0, 7), color: 'var(--c-v)' })
  .then(() => target.update(F.Mmsg[6]));
```

### Misc
`AT.productLine(a, b, decimals)` → `'a1×b1 + a2×b2 + … = Σ'` (the string the dotTable tooltip shows; negatives are parenthesised). `AT.T` → the toy sequence length (10).
`AT.onVisible(el, cb)` — run `cb(el)` once when `el` scrolls into view (use it to start an animation lazily).
`AT.debounce(fn, ms=120)`. `AT.reducedMotion()` → boolean. `AT.escape(str)` → HTML-escaped. `AT.svg(tag, attrs, ...children)` → SVG element builder. `AT.objects` → the 7 object definitions.

---
## 6. Don'ts
- **No new hues.** Only the seven object colours (via `obj-*`, `\v*` macros, `sym-*`, `var(--c-*)`), `--ink*`, `--line`, `--warn` (leakage text only). Never use an object colour for something that is not that object; buttons/links/headings stay ink.
- **No hand-written math.** No `<sub>`, `<i>x</i>`, `x_i` in HTML text. Static: `$…$`; dynamic: `AT.tex` / `AT.texStr` / `AT.renderMath`. Exception: SVG `<text>` labels and `.sym` spans may use plain Unicode.
- **No hard-coded toy numbers.** Everything numeric comes from `AT.model` / `AT.forward` / `AT.baseline`, except the SPEC's own illustrative examples (search table §5, `[2,5,9,1]` §12, 4×4 mask §13, labelled-illustrative multi-head patterns, Part 3).
- **No globals.** One IIFE; `const S = document.getElementById('sNN')`; ids prefixed `sNN-`; fragment CSS scoped `#sNN …`.
- **No external resources**: no URLs, fonts, images, fetch, CDN, libraries. No `console.log` (QA treats warnings as failures).
- **No layout that widens the body.** Wide things go inside `.scroll-x` / `.mat-scroll` / `.dt-scroll` (matrices, vectors and tables already scroll internally). Test at 1280 and 390.
- **No bare rows of numbers for named coordinates.** If the axes mean something, or the reader must compare column by column, or a weighted sum is the point, use `AT.ui.table` / `dotTable` / `mixTable` with labelled columns. The coordinates of `e`, `q`, `k`, `v` **have names** (`AT.axes`): every component uses them by default, so do not pass `'1','2','3'` for those objects and never retype the names. Show arithmetic with `AT.ui.dotCalc` / `AT.ui.matVecCalc` instead of describing it in prose.
- **One widget per row.** Cards stack; `.side-by-side` only for a deliberate comparison. Companion prose (`.companion`) is for self-study and may be hidden, so no widget may depend on it.
- **No motion for its own sake.** Only message-passing dots, stepper transitions, bar-width changes. Everything must also work with reduced motion.
- **Keep notation exactly as BRIEF §3** (`e`, never `x/h/z`; `W_{\text{vocab}}`; `\Delta e`), and keep the Δ-notation disclaimer where the BRIEF requires it (s09, s11, s16, s18).
- Every control: real `<button>` / `<input type=range>` / `<details>`, visible label, keyboard reachable, label says what it does ("Next step", "Reset", "Toggle mask").
- Do not edit `shell.html`, `shared.js`, `katex-bundle.html`, `assemble.py`, `qa.mjs`, `toy.json` (section builders). If you need a component that is missing, build it fragment-locally with `AT.h` and the catalogue classes.
- Every section is authored with `.frame` wrappers, `data-build` numbers and `text/x-notes` (section 7 below). No frame may depend on a control state left behind by an earlier frame.


---
## 7. Parts, frames and presentation

### 7.1 The multi-part build (PARTS.md)
```bash
python3 assemble.py --part 2 --out attention.html                  # sections/, toy.json, part2.json
python3 assemble.py --part 1 --out part1.html                      # sections1/, toy1.json, part1.json, + part1.js if it exists
python3 assemble.py --part 3 --out part3.html                      # sections3/, toy3.json, part3.json, + part3.js if it exists
python3 assemble.py --part 1 --only sections1/sec03.html --out t1_03.html   # one fragment with the right toy and config
```
Overrides: `--toy`, `--config`, `--runtime`, `--sections`, `--shell`, `--shared`. `assemble.py` injects `window.__TOY__` and `window.__PART__` before `shared.js`,
then the part runtime, sets the `<title>` from the config, and warns when the `sections` list of the config and the files on disk differ.

`partN.json` (everything the shell shows outside the sections; nothing part-specific is hard-coded in `shell.html`):
| field | what the shell does with it |
|---|---|
| `part`, `series` | "SERIES · PART N" line above the title (hero and footer) |
| `title`, `subtitle`, `audience`, `minutes` | `<title>`, `h1`, the two hero lines ("… About 60 minutes.") |
| `central`, `centralLabel` | the boxed display formula in the hero (latex with the colour macros; omit `central` to hide the box) |
| `chain: [{label (html allowed), section: "sNN"}]` | the roadmap; the number comes from the section id |
| `sections: [{id, title, lit}]` | the strip's chip targets (`lit` = the objects that section introduces, same as the fragment's `data-lit`) |
| `objects: ["e","q",...]` | which strip chips and legend rows exist (tips come from `AT.objects[].tip`) |
| `legendTitle`, `provenance` | legend heading and the provenance paragraph; `provenance` may use `{{d_model}}`, `{{vocab}}`, `{{axes}}` (filled from `AT`, never retyped) |
| `prev`, `next`, `index` | `{label, href}` links (hero and footer); `null` to omit |
| `notation` | which notation-card rows: `part1`, `part2`, `part3` |
| `footer` | html after the "self-contained file" sentence |
`AT.strip = { setCurrent(idx), lightThrough(idx), sections, chips }` is exposed by the shell after boot (present mode uses it; fragments do not need it).
`AT.objects[]` now carries `tip` (the strip tooltip) next to `def`. `AT.notation` holds the rows of all three parts; each row has `parts: ['part1'] | ['part2','part3'] | ['part3']` and the group `token | matrix | sizes | axes | mlp | train | block`.
The series landing page is `index_series.html` (published as `index.html`): static, no dependencies, same tokens.

### 7.2 Authoring frames and builds (PRESENT.md)
Read mode is the article: frames are invisible wrappers, builds are all visible. Present mode (`Present` button, key `P`, `?present`, or a hash `#sNN/f/b`) shows one frame at a time and reveals builds one per key press.

```html
<section id="s05" class="sec" data-title="A detour through search" data-lit="q k v">
<header class="sec-head">…</header>
<div class="companion">… self-study prose (hidden in present mode) …</div>

<div class="frame" data-title="Score the keys">                      <!-- one frame; 1 to 6 per section -->
  <script type="text/x-notes">
    Which column decides the winner here?                            <!-- first line = the question to ask before the reveal -->
    Point at the query row, then reveal one score at a time.
    Timing: three minutes.                                           <!-- blank line = paragraph -->
  </script>
  <div class="prose"><p>Build 0: visible at once.</p></div>          <!-- no data-build = build 0 -->
  <div id="s05-table" data-build="1"></div>                          <!-- shown on the first key press -->
  <div id="s05-stepper" data-build="2"></div>                        <!-- a stepper: its steps are walked by the arrow keys before build 3 -->
  <div class="callout callout-key" data-build="3">…</div>
  <div class="row" data-build="3">…</div>                            <!-- same number = same key press -->
  <div class="card" data-build="4" data-build-mode="collapse">…</div><!-- collapse: takes no space until revealed (default keeps the layout) -->
</div>

<div class="frame" data-title="The value table" data-notes="Key decides whether; value is what you get back.">   <!-- one-line notes -->
  …
</div>
<div class="frame" data-autobuild="off">…everything visible at once…</div>
<script>(function(){ … })();</script>
</section>
```
Rules the runtime enforces:
- A section with no `.frame` wrappers is one frame: at the first entry into present mode everything after `.sec-head` is moved into a generated `.frame.frame-auto` (the DOM order is unchanged, scripts keep their references). Write real frames anyway: the auto frame scrolls.
- Builds: `data-build="n"`, `n >= 1`, on any element inside the frame (nested is fine). Hidden builds keep their layout (`visibility:hidden`, 180 ms opacity fade) unless `data-build-mode="collapse"`. The frame's build count is the largest `n`.
- Auto builds: a frame with no `data-build` at all numbers its direct children that match `.card, .callout, .tex-display, .prose, p:not(.companion), .chips, table, .stepper, .reveal, .dt-fig, figure, .motif, .netsk, .row, .stack, .btn-row, .scroll-x, ul, ol, h3, blockquote` in order (the first is build 0; children of `.side-by-side / .grid-2 / .grid-3` count one by one). The runtime adds `data-build` + `data-build-auto` attributes. `data-autobuild="off"` on the frame keeps everything visible. Hand-written numbers always win.
- Steppers: when the current build contains a `.stepper` (made with `AT.ui.stepper`) the right/left arrows call its Next/Previous until it runs out, then navigation continues. A stepper as build 0 (s15, s16) is walked immediately. `data-present="manual"` on the stepper host excludes it. Steppers are reset to step 1 when the frame is left and set to their last step when a later build is shown by a deep link.
- Toggles, sliders and `<details>` inside a frame are snapshotted on the frame's first entry and restored when it is left, unless they or an ancestor carry `data-keep-state`. Other button state (a "switch context" button) is yours to reset: give it an idempotent redraw.
- `.companion` is hidden in present mode; nothing a frame needs may live in it. Every frame stands alone: repeat a held drawing with the same component call.
- Notes: `<script type="text/x-notes">` (plain text; the first line is its own paragraph, blank lines separate the rest) or `data-notes="…"`. Shown by the notes strip (`S`) and the presenter window.
- The section header is the frame's title bar (number + title); the frame's `data-title` is shown at the right of it. Keep both short.
- Present base font is `clamp(20px, 1.35vw, 30px)`; tables, vectors, chips, bars, buttons and steppers scale with it (em sizes under `body.present`). Fragment CSS that sets px sizes should use em or add a `body.present #sNN …` rule.

Keys in present mode: `→` `Space` `PageDown` `N` next build (then next frame) · `←` `PageUp` `Backspace` back · `Home` / `End` first / last frame · `O` overview · `B` or `.` blank · `S` notes strip · `?` key help · `Esc` exit (closes the overview or help first).
Arrow keys are left to a focused slider; use Space or PageDown after touching one. `P` enters present mode from read mode. The hash is kept as `#sNN/f/b` (frame 1-based, build 0-based) and honoured on load and on `hashchange`.

Presenter window: the strip button opens the same file with `#presenter` in a second window (three panes: now + build dots, notes, next frame; clock and elapsed time; Back / Next / Blank). The windows talk with `postMessage('*')` (works from `file://`): `{type:'at-presenter-ready'}`, `{type:'at-key', key}`, `{type:'at-go', id, f, b}` towards the presentation, `{type:'at-state', state}` towards the presenter.

Print (handout): `beforeprint` reveals every build, sets every stepper to its last step and leaves the present layout; `afterprint` restores it. The print stylesheet is A4 landscape, 16 px, one frame per page (`.frame` breaks before; the first frame of a section shares the page with the header), companion prose visible, controls hidden, a small uppercase frame title from `data-title`.

### 7.3 `AT.present` API
```js
AT.present.enter({ id:'s05', f:2, b:1 } | { index: 7 })   // optional target; default: hash, else the section in view
AT.present.exit(); AT.present.toggle(); AT.present.isActive()
AT.present.next(); AT.present.prev(); AT.present.first(); AT.present.last()
AT.present.go('s05', 2, 1); AT.present.go(7, 0); AT.present.setBuild(3)
AT.present.overview(true|false); AT.present.blank(bool); AT.present.notes(bool); AT.present.help(bool)
AT.present.openPresenter()                                   // opens #presenter in a new window and enters present mode
AT.present.prepareForPrint(); AT.present.restoreAfterPrint()   // what the beforeprint / afterprint handlers call
AT.present.frames()  // [{ id, num, secTitle, title, index, count, maxBuild, notes, el, sec }]  (discovers on first call)
AT.present.state()   // { active, fi, total, build, frame, next, stepper: {index, count} | null, hash, startedAt, part }
AT.present.on('change' | 'frame' | 'build' | 'step' | 'overview', fn(state))
AT.present.parseHash('#s05/2/1')   // { id:'s05', f:2, b:1 } | null
```
Runtime classes (for fragment CSS, if ever needed): `body.present`, `.sec.is-live`, `.frame.is-live`, `[data-build].is-pending`, `html.printing`, `body.presenter`.

### 7.4 `AT.netSketch(el, opts)` → `{ el, svg, setInputs(w), setHighlight(label), inputs() }`
A node-network sketch: input nodes `e_1 … e_w` (blue, class `e`, KaTeX labels), a neutral hidden column, and an output column labelled with the vocabulary (one highlighted), with thin full bipartite edges.
`opts: { inputs: 3, hidden: 5, outputs: [labels] (default AT.vocab), collapseAbove: 8, highlightOutput: 'water', labels: { input: (i) => latex (1-based; default '\\ve{e_{i}}'), output: (label) => text }, inputCaption: (w) => text (default 'w inputs'), hiddenCaption, outputCaption }`
When `w > collapseAbove` the input column shows `e_1, e_2, e_3`, an ellipsis node, `e_{w-1}, e_w` and the caption reads "w inputs"; the output column collapses the same way (keeping the highlighted entry). `setInputs(w)` animates the column growing or shrinking (nodes slide, new nodes and edges fade in; instant under reduced motion). The drawing height is fixed, so a slider can drive it without the page jumping.
```js
const net = AT.netSketch(S.querySelector('#s03-net'), { inputs: 3, hidden: 5, outputs: AT.vocab, highlightOutput: 'water' });
AT.ui.slider({ label: 'Window w', min: 1, max: 100, step: 1, value: 3, onInput: (w) => net.setInputs(w), into: host });
```
Classes: `.netsk`, `.node.e / .hid / .out`, `.node.out.is-hl`, `.ell`, `.edges line(.hl)`, `.cap`, `.out-l`, `.lab-in`.
