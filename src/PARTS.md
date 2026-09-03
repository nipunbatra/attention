# PARTS.md — the three-part series: one design system, three single files

Files published on GitHub Pages (repo nipunbatra/attention):
- index.html   : series landing (title, one paragraph, three cards with "what you will build", links, prev/next).
- part1.html   : Part 1 — From characters to next-token prediction (GUIDE1.md).
- attention.html : Part 2 — Self-attention (the existing page; also reachable as part2.html via a redirect file).
- part3.html   : Part 3 — Learning, heads, and the Transformer (GUIDE3.md).
Each part is one self-contained file (KaTeX inlined, no dependencies) built by assemble.py from the shared shell + shared.js +
its own sections directory + its own toy JSON + an optional part runtime script.

## Build contract (assemble.py)
python3 assemble.py --part 2 --out attention.html      (Part 2: sections/, toy.json, part2.json)
python3 assemble.py --part 1 --out part1.html          (Part 1: sections1/, toy1.json, part1.json, runtime part1.js)
python3 assemble.py --part 3 --out part3.html          (Part 3: sections3/, toy3.json (a copy of toy.json plus training numbers), part3.json, runtime part3.js)
--only still works for testing single fragments of any part (pass --part so the right toy/config is used).
partN.json = { "part": N, "title", "subtitle", "audience", "minutes", "chain": [ {"label", "section": "sNN"} ... ] (the hero roadmap),
  "sections": [ {"id": "sNN", "title", "lit": "e q"} ... ], "objects": ["e","q","k","v","a","d","ep"] (which strip chips this part uses),
  "prev": {"label","href"} | null, "next": {"label","href"} | null, "notation": "part2" | "part1" | "part3" (which notation card rows) }
The shell reads window.__PART__ (injected by assemble.py next to window.__TOY__) and renders the title, subtitle, roadmap,
strip chips, prev/next links and the notation card from it. Nothing part-specific is hard-coded in shell.html.

## Runtime
shared.js stays part-agnostic (math utils, KaTeX, UI components incl. tables/worksheets, motif, netSketch, present mode).
AT.model / AT.forward / AT.baseline (the attention toy) exist only when window.__TOY__ has W_Q (Part 2 and Part 3).
partN.js (optional) adds part-specific runtime: Part 1 = AT.mlp (embedding lookup, concatenate, MLP forward, softmax, loss,
sampling/generation with temperature, tokenizer helpers); Part 3 = AT.train (precomputed before/after numbers, gradients),
AT.block (LayerNorm, FFN, residual demo numbers), AT.heads (illustrative head weights).

## Notation across parts (one card, three rows-sets; see CONTRACT.md notation card "part" option)
Part 1: t_i (token id), V (vocabulary), w (window), E (lookup table, |V| x d), e_i = E[t_i], a_0 = [e_1, ..., e_w], a_1 = sigma(a_0 W_1 + b_1),
z = a_1 W_2 + b_2, p = softmax(z), loss = -log p(target), d, d_h. Row vectors multiply matrices on the right (same as Part 2).
Part 2: as today (e_i^{(0)}, e_i, q_i = e_i W_Q, ..., Delta e_i = m_i W_O, e_i' = e_i + Delta e_i, E/Q/K/V/S/A/H/DeltaE/E', W_vocab).
Part 3: adds theta (all parameters), L (loss), gradients, eta (learning rate), M (mask), n_h heads, d_ff, LN, block, L layers, KV cache, T^2.

## Colours and objects: unchanged (BRIEF section 1). Part 1 uses e (blue) for embeddings, neutral for a_0/a_1/z, rose for probabilities? No:
probabilities stay neutral bars (rose is reserved for attention weights). Targets/loss use --warn.

## Frames: every section in every part is authored with .frame wrappers, data-build numbers and text/x-notes from the start (PRESENT.md).
