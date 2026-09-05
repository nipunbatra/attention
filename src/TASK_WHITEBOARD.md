# TASK_WHITEBOARD.md — the whiteboard pass, ready to rerun (one agent or one person per part)

Goal: make every presentation frame of Parts 1, 3, 4 and Vision I to IV teach the way the instructor's handwritten decks and Part 2 do.
Rules: STYLE_WHITEBOARD.md (authoritative). Worklist: FRAME_AUDIT.md (per part). Reference frames: sections/sec05.html, sec08.html, sec09.html,
sec15.html (Part 2). Feel: ~/git/ml-teaching/neural-networks/slides/next-token-prediction.pdf pages 1 to 20 and autograd.pdf pages 1 to 12.
Part 2 (sections/) is the reference and is not touched.

## Prompt to give an agent for part N (N in 1, 3, 4, 5, 6, 7, 8; sources sectionsN/, partN.js, partN.json; outputs part1/part3/part4/vision1..vision4)
"Rework the PRESENT-MODE frames of part N listed in FRAME_AUDIT.md (and any other frame that breaks STYLE_WHITEBOARD.md): split frames that carry more
than one idea; give each frame one held drawing (SVG diagram, coloured chips, a vector or table drawn as the object, netSketch, bars; for vision parts the
scene grid/overlay, the patch scatter, the routing tree with thumbnails, the circle, the triptych, the loss curve) and one new mark per build (2 to 6 builds,
data-build on the elements that appear); cut on-frame prose to at most 40 words (a question, the reading of a number, or one boxed idea) and move the rest
to the section's companion prose; put numbers on the drawing (arrow, cell, patch) not in sentences; presenter notes (script type=text/x-notes) whose first
line is the question to ask before the reveal. Keep every number computed from the runtime; do not change what a section teaches; keep read mode complete.
Do not edit shell.html, shared.js, vision-shared.js, assemble.py, the toy JSON, or other parts; add helpers to partN.js.
Held drawings per part: Part 3 = the computational graph drawn once and annotated with one coloured gradient arrow per build (upstream in red times local
equals downstream); Part 4 = English source row above, French target row below, the decoder query arrow reaching across, gaining scores, weights as arrow
thickness, the mixed value, the update, the prediction bars; Vision IV = the token row (16 thumbnails plus question chips) and the overlay per generated token.
VERIFY: python3 assemble.py --part N --out ../<output>.html; node qa.mjs at 1280x720 and 390 (zero errors, no overflow); node sweep.mjs (no problems);
node walk.mjs (no errors, no scrolling frames); a Playwright pass over every frame asserting #at-fit-warning stays empty; READ at least fifteen frame
screenshots and fix walls of text, boxed cards around drawings, empty frames; the part's check script (check_visionK.mjs, check_part1.mjs, check_training.py)
must pass. Report frames before/after per section, what was split or redrawn, test results, anything left undone."

## Budget note
Seven such agents in parallel exhausted a Claude session limit within minutes on 2026-09-05 before any of them wrote frames. Run ONE part at a time,
or hand a part to Codex CLI (codex exec) with the same prompt; verify with the commands above before pushing.
