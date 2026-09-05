# VISION_AXES.md — one scene, named axes, shared figures (design for the vision rebuild)

Companion to VISION_FEEDBACK.md. This file fixes the decisions so that four parts can be rebuilt consistently. It plays the role AXES.md
played for the language parts: names first, numbers chosen so the names are true.

## 1. The scene (replaces the 4×4 block worksheet everywhere)
An 8×8 grayscale image, pixel values in {0, 1, 2, 3} (0 = table, darkest; 3 = brightest), 2×2 patches, so 16 patches in a 4×4 grid,
numbered 1..16 row-major. Objects are deliberately NOT aligned to the patch grid (a patch is a piece of the input, not an object).

Scene A, "two mugs" (rows top to bottom):
    0 0 0 0 1 0 0 0      ← the plant: one pixel of value 1 at row 0, col 4
    0 3 3 3 0 3 3 3      ← left mug (cols 1..3) and right mug (cols 5..7), rows 1..3, value 3
    0 3 3 3 0 3 3 3
    0 3 3 3 0 3 3 3
    0 0 0 0 0 0 0 0
    0 2 2 2 2 2 0 0      ← the book, rows 5..6, cols 1..5, value 2
    0 2 2 2 2 2 0 0
    0 0 0 0 0 0 0 0
Scene B, "one mug": Scene A with the right mug removed (rows 1..3, cols 5..7 set to 0).
Scene C, "one mug, moved": Scene B with the left mug moved to cols 4..6 (the generalisation probe; the model may fail on it, which is a lesson).
Scene D, "book only": no mugs, no plant (used by CLIP as the second pair). Scene E, "plant only": only the plant pixel (CLIP's third pair).

Region names, derived from pixel content (a helper `AT.vision.regionOf(patchIndex, scene)` returns the name):
the region with the most pixels in the patch, ties broken toward the object; all-zero → "table". Display labels: "patch 6 · left mug", "patch 14 · book",
"patch 3 · plant", "patch 16 · table". Patch 6 (rows 2..3, cols 2..3) is all mug ("left mug centre"); patch 5 has two mug pixels ("left mug edge").
The photo of the real tabletop stays only as the opening hook of Vision I. Every other picture on every vision page is the scene, the model can see it,
and the caption "AI-generated teaching illustration. No model output is shown." disappears.

One grayscale ramp for all parts, in shared code: value 0 → #3A3A3A, 1 → #6E6E6E, 2 → #A3A3A3, 3 → #D8D8D8, with white digits on 0..1 and dark digits on 2..3.

## 2. The Vision I toy (toy5.json v2): axes first
d_model = 4, axes e = ["brightness", "edge", "row", "col"] (short: bright / edge / row / col).
- brightness = mean of the four pixels (0..3); edge = (left column − right column) mean, so a patch whose left half is mug and right half is table has a
  positive edge and the mirror patch a negative one; row and col carry position: row = (patch row index)/3, col = (patch col index)/3, added as the
  position vector (zeros on brightness and edge). W_patch is therefore fixed by the names: column 1 = [1,1,1,1]/4, column 2 = [1,−1,1,−1]/2, columns
  3 and 4 zero; b_patch = 0. CLS starts at c = (0, 0, 0, 0) plus its own position row (−1, −1) on row/col so it is distinguishable.
d_k = 2, axes qk = ["bright region?", "edge here?"] (queries ask, keys offer). d_v = 2, axes v = ["sends: brightness", "sends: edge"].
Initial (before training) W_Q, W_K, W_V, W_O sparse and readable: W_K: brightness→bright? 1.0, edge→edge? 1.0; W_V: brightness→sends brightness 1.0,
edge→sends edge 1.0; W_O: sends brightness→brightness 1.0, sends edge→edge 1.0; W_Q: brightness→bright? 1.0 (CLS asks for bright regions). Position rows of
W_Q/W_K/W_V are zero initially (the toy carries position but the untrained attention does not use it; say so, as Part 2 does).
Head: two classes ("two mugs", "one mug") read the updated CLS row.
Training: fit on scenes A and B. Keep the axis names by keeping W_patch fixed (not trained) and training only positions, W_Q, W_K, W_V, W_O and the head;
report which learned entries became large and give them reading lines as Part 2 does (e.g. "after training CLS asks mostly for bright regions in rows 1..2").
Targets to verify: (T1) untrained CLS attention spreads by brightness (mug patches highest); (T2) trained CLS attention puts ≥ 0.6 of its mass on the six mug
patches of scene A; (T3) the head predicts two mugs for A and one mug for B with p ≥ 0.9; (T4) scene C is reported honestly (whatever happens);
(T5) all numbers one or two decimals.

## 3. Shared figure components (vision-shared.js, documented in CONTRACT.md)
- `AT.vision.grid(scene, {size, labels:'values'|'none', highlight:[patches], patchLines:true})` → SVG of the 8×8 image with the single ramp, optional patch grid.
- `AT.vision.thumb(scene, patchIndex, {size})` → a tiny SVG of one patch (a row label in tables, a leaf in the routing tree).
- `AT.vision.overlay(scene, alpha, {receiver, onHover(j), showValues})` → the grid with each patch tinted by the rose heat colour ∝ α_j, receiver outlined; hover
  shows the patch's name, α, k_j and v_j. THE key figure: Vision I (CLS before/after training), II (what visible patches read in MAE), IV (what the answer token reads).
- `AT.vision.scatter(points, {axes:['brightness','edge'], arrows:[{from,to,label}], labels})` → 2-D embedding plot with named axes, patch thumbnails as markers,
  optional Δe arrows (as Part 2 s10).
- `AT.vision.circle(points, {kind:'img'|'txt'})` → unit-circle (2-D) or projected-sphere (3-D) picture for CLIP embeddings, images as squares, captions as
  triangles, with an `animate(states)` method for the training trajectory.
- `AT.vision.triptych(masked, prediction, original)` → three grids side by side for MAE.
- `AT.vision.curve(values, {label, marks:[step]})` → a small loss curve with a marker for "you are here".
- Table conventions: row labels are thumbnails plus names; column headers are the axis names; masked pixels hatched as in Part 2.

## 4. The other toys (later jobs)
- toy6 (Vision II): the same encoder (frozen W_patch and names) plus a tiny decoder (2 → 4 pixels per patch) trained by MAE on scenes A, B, D with the
  right-mug patches hidden; DINO views = flip / brightness × 0.75 / crop of scene A through the same encoder plus a 3-slot head; a centering toggle that
  exposes collapse (loss → log 3); I-JEPA: predict the hidden patch's (brightness, edge) feature from the visible context.
- toy7 (Vision III, CLIP): image encoder = the frozen Vision I encoder's updated CLS row (brightness, edge, row, col) followed by a learned 4 × 3 map to the
  joint axes ["mug", "book", "plant"]; text encoder = bag of words over a small vocabulary whose rows are named by those axes; pairs: (A, "two mugs on a table"),
  (D, "a book on a table"), (E, "a small plant"); train 60 steps; zero-shot with prompts including a wrong candidate list.
- toy8 (Vision IV, VLM): frozen Vision I encoder's 16 patch rows → connector → decoder (d = 3) with vocabulary {<bos> how many mugs ? one two <eos> book plant};
  fit on A and B; the answer token's attention overlay must show mug patches; scene C probes generalisation and the page shows the result whatever it is.

## 5. What stays true everywhere
Row-vector convention; the seven object colours only for their objects; every displayed number computed by the toy; one caveat box per part after the hero
("What this toy can and cannot show"); no "coordinate N" labels anywhere; the recurring motif e → Q,K,V → attention → Δe → e+Δe at the top of Vision I and IV.
