# REV2 task: position column + narrower values (see AXES.md "Revision 2")

Working directory: this scratchpad (absolute paths). Never touch /Users/nipun/git/attention. No network.
Read AXES.md (revision 2 at the end), PARTS.md, CONTRACT.md, toy_report.md, and skim sections/sec01.html, sec07.html, sec11.html.

1. Toy: edit make_toy2.py so d_model = 5 (axes water, finance, person, glue, position), pos_emb[i] = [0,0,0,0, 0.1*(i+1)],
   d_v = 2 (v axes "says: water scene", "says: finance scene"; short "→water", "→finance"), W_V 5x2, W_O 2x5, W_vocab 5x20 with a zero position row,
   zero position rows in W_Q, W_K, W_V. Keep every other number. Regenerate toy.json (schema: d_model 5, d_k 3, d_v 2, axes updated, notes updated:
   position carried but unused by this toy; values narrower on purpose), py_check.json and toy_report.md (gen_report.py). Make toy_ref.mjs generic in
   d_model/d_v (no hard-coded 3 or 4) and confirm "node toy_ref.mjs" exits 0 with all targets passing on the rounded numbers.
2. Runtime: in shared.js remove any assumption that d_model = 4 or d_v = d_k = 3 (AT.d_model, AT.d_k, AT.d_v must come from the model); axis headers by width:
   5 -> e axes, 3 -> qk axes, 2 -> v axes (and by cls when widths tie). Δe and e' use e axes; m uses v axes. The notation card and sec00_demo must show the
   new shapes (read them from AT). Check AT.ui.vec/mat/table/dotTable/mixTable/wTable/dotCalc/matVecCalc/netSketch with the new widths (assemble sec00_demo, qa, screenshot).
3. Sections: grep every sections/secNN.html and shell.html for shape and count text tied to the old sizes: "1×4", "1\\times 4", "4×3", "4 \\times 3", "3×4", "4×20",
   "four numbers", "4-number", "d_{\\text{model}} = 4", "d_v = 3", "three numbers" (only where it means d_v), "1×3" where it means a value or message,
   and fix them (prefer computing from AT.d_model / AT.d_v in scripts; in static prose write the new sizes). Do not touch numbers that are computed.
4. Add the columns note where q, k, v first sit together: s07 (records) and s11 (what exactly are Q, K, V): one short line each:
   "q and k share the same three columns because the dot product compares them column by column. v has its own two columns, and its own width, because it is
   only mixed and sent." Also one line in s01's token + position table caption: "position has its own column; the toy carries it but its attention does not use it."
5. Verify: python3 assemble.py --part 2 --out attention.html (if --part is not implemented yet in assemble.py, use python3 assemble.py --out attention.html),
   node qa.mjs attention.html --width 1280 --height 720 and --width 390 (zero errors, no overflow), node sweep.mjs attention.html (no problems),
   node mathdiff.mjs <f> <f> on every fragment (no humanizer flags), node toy_ref.mjs exit 0. Screenshot s01, s07, s09, s11 at 1280x720 (secshot.mjs) and READ them:
   the position column visible and separate in s01, v rows two columns wide, W_V 5x2 and W_O 2x5 tables readable, nothing clipped.
Report: what changed in the toy (attention rows and top probabilities before/after), every file edited, and the test results.

6. s05 mock search UI (instructor request; edit sections/sec05.html, and keep sections/sec06.html consistent since it reuses the same six items):
   Build a mock video site inside the first card: a search bar showing the query text (the two existing preset queries stay as buttons), a results list of six
   video cards that REORDER by score (animate with a short transform transition; respect prefers-reduced-motion). Each card: a small inline SVG thumbnail
   (draw simple icons in the page style, 96x54, one per topic: backpropagation = arrows going backwards through three boxes; gradient descent = a bowl with a
   ball; CNNs = a grid with a sliding window; transformers = stacked blocks with a crossing arrow; batch normalisation = a bell curve centred on zero;
   regularisation = a wiggly curve smoothed to a straight one), the title, a one-line summary (this is the VALUE's text), a fake duration, and a rank badge.
   Under each card a "why it ranks here" strip (open by default for the top result, collapsed for the rest, toggle on click): the KEY row on the four matching
   axes (gradient flow, optimisation, architecture, generalisation) with the dot product against the query written out (use AT.ui.dotCalc), the score, and
   the VALUE row on its own three axes (gradients, optimisation, architecture). Hovering a card highlights its row in the two spreadsheets below and vice versa.
   Keep the existing dotTable and value table below the mock UI as the "same thing as a spreadsheet" view. In s06, the same six cards appear with hard mode
   (one card highlighted, others greyed) and soft mode (each card's opacity and a small weight badge follow alpha; the mixed value row shown at the bottom).
   No external images, no fonts, everything inline; test at 1280 and 390 (cards stack on phones).
