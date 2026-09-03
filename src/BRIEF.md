# BRIEF — architecture, design system, and rules for building `attention.html`

Read SPEC.md first (the instructor's verbatim brief). This file adds the engineering contract
so that several people can build sections in parallel and the result feels like ONE artifact.

## 0. Files (all in this scratchpad directory)
- `SPEC.md`            — verbatim pedagogical brief (authoritative on content/order/notation)
- `BRIEF.md`           — this file (authoritative on engineering + design tokens)
- `katex-bundle.html`  — KaTeX 0.17 JS+CSS+fonts inlined (~540 KB). Never edit. Inserted at `<!--KATEX-->`.
- `qa.mjs`             — headless Chromium QA harness (Playwright). `node qa.mjs page.html --shot x.png [--full] [--click "css"] [--eval "js"] [--width 1280]`
                          prints JSON {pageErrors, consoleErrors, katexErrors, overflowX, docHeight, evalResult}
- `assemble.py`        — builds a page: `python3 assemble.py --out attention.html` (all sections) or
                          `python3 assemble.py --only sections/sec07.html --out test07.html` (one/few fragments for testing)
- `toy.json`           — the ONE shared toy model (numbers) used by every section. Produced by the toy-model task.
- `toy_ref.mjs`        — reference forward pass in plain JS (source of truth for the math). shared.js must agree with it.
- `shell.html`         — full HTML document skeleton: <head> with all CSS, sticky notation strip, hero/roadmap,
                          placeholders `<!--KATEX-->`, `<!--SHARED-->`, `<!--SECTIONS-->`, then footer + boot script.
- `shared.js`          — the `AT` namespace: math, toy model forward pass, UI components, KaTeX helper, motif diagram.
- `CONTRACT.md`        — API documentation for shared.js + CSS class catalogue + copy-paste examples. Builders read this.
- `sections/secNN.html`— one fragment per SPEC section (NN = 01..19, zero padded). Each fragment = one `<section>` + one `<script>`.
- `attention.html`     — final single file, output of assemble.py.

## 1. Design direction (fixed — do not re-decide)
Subject: a 60-minute lecture artifact projected in a lecture hall, then re-read by students at home.
The page's single job: make the chain  e → Q,K,V → attention → Δe → e+Δe → next-token prediction  inevitable and unconfusable.

**Look**: light "lab notebook" — clean, high-contrast, projector-safe. Not cream/terracotta, not dark-mode-neon,
not newspaper hairlines. Cool near-white paper with a faint dot grid behind diagrams. Large type. Generous spacing.
The colour of the page is carried almost entirely by the SEVEN OBJECT COLOURS below; everything else is neutral.

**Signature element**: the *notation strip* — a sticky bar at the top holding seven object chips
(e · Q · K · V · α · Δe · e+Δe). Chips are dim until the section that introduces the object is reached, then stay lit.
Hovering a chip shows a one-line definition. The same seven colours are used for every vector, matrix, chip, arrow,
and KaTeX symbol on the page. The strip also shows a thin progress bar and the current section title.

**Recurring motif**: a compact SVG pipeline  e → (Q,K,V) → attention → Δe → ⊕ → e'  (`AT.motif`) that can highlight
one active stage. It appears in sections 09, 11, 15, 18 and 21. Same drawing every time.

**Type** (system fonts only, no webfonts):
- display/headings: `"Avenir Next", "Segoe UI Variable Display", "Segoe UI", "Helvetica Neue", system-ui, sans-serif`, weight 700, letter-spacing -0.01em
- body/UI: same stack, weight 400/500, base 17px (18px ≥1400px wide), line-height 1.5
- data/numbers: `"SF Mono", Menlo, Consolas, "Liberation Mono", monospace`, `font-variant-numeric: tabular-nums`
- math: KaTeX (serif). Inline math sizes with surrounding text.

**Colour tokens** (CSS custom properties, defined once in shell.html):
```
--paper:#F7F8FA  --ink:#14171F  --ink-2:#4A5160  --ink-3:#8A91A0  --line:#D9DDE5  --card:#FFFFFF  --grid:#E3E6EC
--c-e:#2563EB   (e   — current token representation; blue)
--c-q:#9333EA   (Q   — query; purple)
--c-k:#D97706   (K   — key; amber)
--c-v:#0D9488   (V   — value; teal)
--c-a:#E11D48   (α   — attention weight; rose. Heatmaps go white → --c-a)
--c-d:#16A34A   (Δe  — contextual update; green)
--c-ep: e' = e+Δe is drawn as BLUE FILL WITH A GREEN RING (2px) — never a new hue. Provide `--c-ep-grad: linear-gradient(90deg,var(--c-e),var(--c-d))` for arrows/labels.
--c-mask:#9CA3AF (masked cells, hatched)   --warn:#B45309 (leakage / warning text)
Each object colour also has a tint for backgrounds: --t-e:#E4ECFF --t-q:#F1E5FC --t-k:#FCEFD9 --t-v:#D9F2EF --t-a:#FDE2E7 --t-d:#DDF3E4
```
Semantic rule: NEVER use an object colour for anything that is not that object. Buttons, links, headings are --ink / --ink-2.
Focus rings: 2px solid --ink offset 2px. Respect `prefers-reduced-motion` (animations become instant).

**Layout**: single column, `max-width: 1120px`, side padding 24px (16px on phones). Sections are separated by
88px vertical space; each section starts with a header row: two-digit number in monospace (this IS a sequence —
the numbers encode the argument order) + title + one-line "why we are here" eyebrow. Interactive widgets live in
`.card` panels (white, 1px --line border, radius 12px, padding 20px). Wide content (matrices, heatmaps) scrolls
horizontally inside its own container; the body never scrolls horizontally. Projector mode: everything must be legible
at 1280×720 with ~17px base text; numbers in vectors ≥ 14px monospace.

**Motion**: purposeful only — message-passing animations (dots moving along edges), stepper highlight transitions
(200ms), chip lighting. No parallax, no scroll-jacking, no decorative particles.

## 2. Sections (ids are fixed; order is the argument)
| id  | title                                         | strip chips lit after this section |
| s01 | Predict the next token                        | —                                   |
| s02 | Baseline: only the last token                 | e                                   |
| s03 | Baseline: a fixed window of K tokens          |                                     |
| s04 | Weighted pooling: where should weights come from? | α                               |
| s05 | A detour through search: Query, Key, Value    | Q K V                               |
| s06 | Hard retrieval → soft retrieval               |                                     |
| s07 | Every token becomes a searchable record       |                                     |
| s08 | Two phases: read routing (Q,K) vs message passing (V) |                             |
| s09 | The contextual update Δe                      | Δe  e+Δe                            |
| s10 | "bank": same start, different context         |                                     |
| s11 | What exactly are Q, K, and V?                 |                                     |
| s12 | Why divide by √d_k?                           |                                     |
| s13 | Causal masking                                |                                     |
| s14 | From Δe to the next-token probabilities       |                                     |
| s15 | Full walkthrough, one step at a time          |                                     |
| s16 | The same thing in matrix form                 |                                     |
| s17 | Attention versus the alternatives             |                                     |
| s18 | Pause and think                               |                                     |
| s19 | Three summaries                               |                                     |
(Multiple heads and layer after layer belong to Part 3; seed fragments live in sections3_seed/.)
Fragment file for section NN is `sections/secNN.html` and its root element is `<section id="sNN" class="sec" data-title="...">`.

## 3. Notation (LaTeX strings — use EXACTLY these; never x, h, z for representations)
- current representation of token i: `e_i`; initial: `e_i^{(0)}`; after layer ℓ: `e_i^{(\ell)}`
- update: `\Delta e_i`; updated: `e_i' ` (also written `e_i^{\text{new}} = e_i^{\text{old}} + \Delta e_i` where SPEC does)
- projections: `q_i = W_Q e_i`, `k_j = W_K e_j`, `v_j = W_V e_j`   (column-vector convention for single tokens)
- scores: `s_{ij} = q_i^\top k_j / \sqrt{d_k}`; weights: `\alpha_{ij}`; message: `m_i = \sum_j \alpha_{ij} v_j`; `\Delta e_i = W_O m_i`
- matrices (row-per-token convention): `E \in \mathbb{R}^{T\times d_{\text{model}}}`, `Q = E W_Q`, `K = E W_K`, `V = E W_V`,
  `S = QK^\top/\sqrt{d_k}`, `A = \operatorname{softmax}(S+M)`, `H = AV`, `\Delta E = H W_O`, `E' = E + \Delta E`
- output head: `\ell = W_{\text{vocab}} e_t + b`, `p(x_{t+1}\mid x_{\le t}) = \operatorname{softmax}(\ell)` (SPEC §2 writes W_out; we use W_vocab everywhere and say "the output head")
- toy dimensions: `T = 10` (7 in the short walkthrough), `d_{\text{model}} = 4`, `d_k = d_v = 3`, vocabulary size 20.
- The Δ-notation disclaimer (SPEC "Central notation") must appear verbatim-in-spirit in s09 AND s11 AND s16 AND s18.
- Colour-coded math: KaTeX macros are pre-defined in the shell:  `\ve{…}` (e, blue) `\vq{…}` `\vk{…}` `\vv{…}` `\va{…}` (α) `\vd{…}` (Δe) `\vp{…}` (e′, blue text with green underline).
  Example: `\vp{e_i'} = \ve{e_i} + \vd{\Delta e_i}`,  `\va{\alpha_{ij}} = \operatorname{softmax}_j(\vq{q_i}^\top \vk{k_j}/\sqrt{d_k})`.
  Colour the OBJECT symbols, not operators or W matrices.

## 4. The shared toy model (`toy.json`) — exact schema
```
{
  "d_model": 4, "d_k": 3, "d_v": 3,
  "vocab": [20 lowercase strings],
  "tok_emb": { "<token>": [4 numbers] },          // token embedding rows
  "pos_emb": [[4 numbers] × 10],                   // positions 1..10 (index 0 = position 1)
  "W_Q": [[3]×4], "W_K": [[3]×4], "W_V": [[3]×4],  // d_model × d_k   (row-vector convention: q = e·W_Q where e is a row)
  "W_O": [[4]×3],                                  // d_v × d_model
  "W_vocab": [[20]×4], "b_vocab": [20],            // d_model × |V|
  "sentences": {
    "river":  ["The","fisherman","sat","beside","the","river","bank","and","watched","the"],
    "cheque": ["She","deposited","the","cheque","at","the","bank","and","watched","the"]
  },
  "candidates": { "river": ["water","boats","fish","ducks"], "cheque": ["teller","clerk","queue","money"] },
  "notes": "free text: what patterns the numbers were tuned to produce"
}
```
Tokens are looked up lowercase (`"The"` → `tok_emb["the"]`). Display uses the original casing.
`e_i^{(0)} = tok_emb[token_i] + pos_emb[i-1]`. Forward pass (single head, causal):
`q_i = e_i W_Q`, `k_j = e_j W_K`, `v_j = e_j W_V`, `s_ij = q_i·k_j/√d_k`, mask j>i, `α = softmax`, `m_i = Σ α_ij v_j`,
`Δe_i = m_i W_O`, `e_i' = e_i + Δe_i`, `logits = e_i' W_vocab + b`, `p = softmax(logits)`.
(Row-vector convention in code = column-vector convention in the single-token LaTeX. Both describe the same numbers; the
page shows W_Q as a d_model × d_k matrix and vectors as rows of numbers.)
Both sentences have **bank at position 7** and the prediction slot after **position 10** ("watched the ___").

## 5. Rules for every section fragment
1. One `<section id="sNN" class="sec" data-title="…">` root, then ONE `<script>` containing an IIFE: `(function(){ const S = document.getElementById('sNN'); ... })();`
   No global variables. All ids inside a fragment are prefixed `sNN-`. Query elements relative to `S`.
2. Use ONLY components/classes from CONTRACT.md plus small fragment-local CSS inside a `<style>` scoped with `#sNN …` selectors (keep it minimal).
3. All math via KaTeX: static math in the HTML as `$…$` / `$$…$$` (auto-rendered at boot); dynamic math via `AT.tex(el, latex, {display})`.
   Never hand-write math in HTML `<sub>`/`<i>` tags. SVG text labels may use plain Unicode (e.g. `q_bank`, `α`, `Δe`).
4. Numbers shown on the page must come from `toy.json` via `AT.model`/`AT.forward` etc. — never hard-code numbers that the toy computes,
   except the deliberately illustrative ones the SPEC dictates (search-engine table in §5, `[2,5,9,1]` in §12, 4×4 mask in §13, and any
   explicitly "illustrative" multi-head patterns in Part 3 which must be labelled illustrative).
5. Every interactive control must be keyboard reachable (real `<button>`, `<input type=range>`, `<details>`), have a visible label, and work when clicked.
6. Concise text: ≤ 60 words per prose block; prefer visual → intuition → equation. Use `.callout` for the key statements the SPEC quotes.
7. Test your fragment: `python3 assemble.py --only sections/secNN.html --out testNN.html && node qa.mjs testNN.html --full --shot testNN.png`,
   then Read the PNG and fix what looks wrong. pageErrors/consoleErrors/katexErrors must be empty; overflowX must be false at width 1280 AND 390.
8. Do not introduce LayerNorm, MLPs, full Transformer blocks, dropout, or training details. Do not mention other libraries or frameworks.
9. No external URLs, no images, no fonts, no fetch. Everything inline.
10. Copy voice: plain, active, second person ("you") sparingly, no hype. Sentence case for UI labels. Buttons say what they do ("Next step", "Reset", "Toggle mask").
