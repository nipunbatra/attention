# HANDOVER — interactive self-attention teaching series (attention.html and parts)

Owner: Nipun Batra (nipun.batra@iitgn.ac.in). Repo: https://github.com/nipunbatra/attention  Live: https://nipunbatra.github.io/attention/
Written 2026-09-03 by Claude Code at the end of a long build session. Everything needed to continue lives in the repo under `src/`.
This file is the entry point for any successor: a later Claude session, Codex, Gemini, or a human.

## 1. What this is
A single-file, offline interactive lesson series for a deep-learning course:
- Part 2 (done, live): `attention.html` — self-attention from first principles for next-token prediction. 19 sections after the restructure.
- Part 1 (planned): `part1.html` — from characters to next-token prediction (the aabid name model). Plan: `src/GUIDE1.md`. Data: `src/names.csv`.
- Part 3 (planned): `part3.html` — learning (loss, autograd, one update, parallel causal training), multi-head, FFN, residual stream, LayerNorm,
  blocks, decoder-only model, generation, KV cache, model families. Plan: `src/GUIDE3.md`. Seed material: `src/sections3_seed/` (the old s17 heads and s18 layers).
- `index.html` is the series landing; `part2.html` redirects to attention.html.
The instructor's original brief is `src/SPEC.md`; his slide deck that the series mirrors is `/Users/nipun/git/dl-teaching/lecture11/` (Typst; outline in
`L11M-from-characters-to-transformers-outline.md`; PDF `slides-pdf/L11M.pdf`). Notation, colours and the sequence follow that deck (row vectors, e' = e + Delta e).

## 2. Layout of src/
- `SPEC.md` brief · `BRIEF.md` design tokens + engineering rules (row-vector convention) · `GUIDE.md` per-section design for Part 2 · `CONTRACT.md` the runtime API
  (AT.* components: chips, vec, mat, heat, bars, table, dotTable, mixTable, wTable, dotCalc, matVecCalc, notationCard, netSketch, motif, flow, stepper, reveal,
  callout, present mode authoring) · `PARTS.md` multi-part build contract · `PRESENT.md` presentation-mode design · `FRAMES.md` frame/build plan for Part 2 ·
  `AXES.md` the named-axes toy design (revision 2 at the end is NOT yet applied) · `GUIDE1.md`, `GUIDE3.md` plans for Parts 1 and 3 ·
  `CODEX_FEEDBACK.md` (applied), `CODEX_FEEDBACK_2.md` (assessed, NOT yet applied) · `REV2_TASK.md` the next task, ready to hand to an agent ·
  `humanizer/SKILL.md` the prose rules (blader/humanizer; installed as a Claude Code plugin `humanizer@humanizer`).
- `shell.html` (page skeleton + all CSS + hero/strip/footer + boot; reads window.__PART__), `shared.js` (the AT runtime), `assemble.py` (builder),
  `katex-bundle.html` (KaTeX 0.17 with fonts inlined; never edit), `part2.json` (Part 2 config), `toy.json` (Part 2 toy v2, named axes; `toy_v1.json` old),
  `make_toy2.py` (hand-designed toy generator; `python3 make_toy2.py` rewrites toy.json and checks targets), `gen_report.py` -> `toy_report.md` (every number
  on the page), `toy_ref.mjs` (reference forward pass; `node toy_ref.mjs` must exit 0), `py_check.json`.
- `sections/secNN.html` Part 2 fragments (one <section> + one <script> IIFE each; `sec00_demo.html` is the component gallery, not shipped).
- Test tools: `qa.mjs` (errors/overflow/screenshot), `sweep.mjs` (clicks every control, scans for NaN), `secshot.mjs`, `crop.mjs`, `hovershot.mjs`,
  `mathdiff.mjs` (structure diff + humanizer flags), `pres_test.mjs`, `walk.mjs` (present-mode walk). Playwright lives at
  `/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright` (see qa.mjs for the require pattern); Chromium is cached; no install needed.

## 3. Build and test (from src/)
    python3 assemble.py --part 2 --out ../attention.html            # full Part 2
    python3 assemble.py --part 2 --only sections/sec07.html --out t07.html   # one fragment for testing
    node qa.mjs ../attention.html --width 1280 --height 720 --shot x.png     # must print zero pageErrors/consoleErrors/katexErrors, overflowX false
    node qa.mjs ../attention.html --width 390                               # phones: no overflow
    node sweep.mjs ../attention.html                                        # clicks every control; problems must be {}
    node toy_ref.mjs                                                        # toy targets; exit 0
    node mathdiff.mjs sections/secNN.html sections/secNN.html               # lists humanizer flags (dashes, AI words) in a fragment
    node walk.mjs ../attention.html                                         # present mode: walks frames, screenshots
Publish: copy the assembled file(s) to the repo root, `git add -A && git commit && git push` (Pages serves main, root). Pages builds in about a minute.

## 4. Rules that must survive (decided with the instructor)
- One symbol family: e_i (current representation), e_i^{(0)} = token + position, q_i = e_i W_Q, k_j = e_j W_K, v_j = e_j W_V, s_ij, alpha_ij, m_i = sum_j alpha_ij v_j,
  Delta e_i = m_i W_O (pedagogical name for the attention output; the addition is the standard residual), e_i' = e_i + Delta e_i. Matrix form E, Q, K, V, S, A, H, DeltaE, E'.
  ROW-VECTOR convention everywhere (q_i = e_i W_Q; scores as a dot product q_i . k_j; S = QK^T). Never x, h, z for representations. Output head is W_vocab.
- Seven object colours (e blue, Q purple, K amber, V teal, alpha rose, Delta e green, e' blue with green ring) used only for those objects.
- Every number on the page is computed from the toy through AT.model/AT.forward; nothing is retyped. The toy is hand designed so that its named axes are true.
- Tables with named columns wherever coordinates mean something; worksheets (dotCalc, matVecCalc) wherever a product or a weighted sum is shown.
- Prose follows humanizer/SKILL.md: plain verbs, sentence case, no em or en dashes, no "not just X but Y", no forced triples, no AI vocabulary. Companion prose
  (80 to 120 words) per section for self-study; "Reading: full / lean" toggle in the strip.
- Causal examples only (bank never reads a later word). Do not introduce LayerNorm/MLP/blocks in Part 2. Heads and layers belong to Part 3.
- Multi-part: PARTS.md; each part is one file; frames authored per PRESENT.md so slide mode is a toggle.

## 5. State at handover (updated 2026-09-03, commit 12c5ab0)
Done and live: Part 2 with named axes on FIVE coordinates (water, finance, person, glue, position), keys 3 wide and values 2 wide, tables and
worksheets everywhere, the s05/s06 mock video search with per-card key and value strips, dot-product notation q_i . k_j, companion prose, notation card,
s13 background staircase, present-mode runtime, multi-part shell, series index, every round-2 review item (CODEX_FEEDBACK_2.md) applied.
Present-mode frames exist for s01 to s05 (s05 needs re-checking after the mock UI); s06 to s19 fall back to automatic one-card-per-build frames.
NOT done, in priority order:
1. Presentation prototype: frames for s05 (re-check) and s06 per FRAMES.md, walk all frames, fix layout (a Codex job was started for this on 2026-09-03;
   check `git status` in src/sections for its edits). Then decide with the instructor whether to frame s07 to s19.
2. Part 3 per GUIDE3.md (numbers agent: toy3.json = toy.json + training block via numpy autodiff, part3.js; builders by section group; gate). Start from
   sections3_seed/ (old heads and layers sections; they still use the OLD toy widths and ids s17/s18 and need renumbering and a re-check against the 5/3/2 toy).
3. Part 1 per GUIDE1.md (train the aabid MLP with train_names.py on names.csv, part1.js, sections1/), then update index.html cards and part2.json prev/next.
Known small issues: the roadmap in the hero lists s01 to s14 only; sections3_seed/sec17.html clips at 390px.
Codex note: the Codex sandbox cannot launch the bundled headless Chromium (Mach port error); it verifies in the system browser instead. Run qa.mjs and
sweep.mjs from a normal shell before pushing.

## 6. How to hand a task to an agent
Give it: this file, the task file (e.g. REV2_TASK.md), CONTRACT.md, and the acceptance commands in section 3. Ask for: files changed, tests run with results,
screenshots inspected, open issues. Keep edits inside src/ and the assembled outputs at the repo root. Push after every green build so Pages stays current.
