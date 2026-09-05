# STYLE_WHITEBOARD.md — how the instructor teaches, and what every frame must do

Source: Nipun's own decks in ~/git/ml-teaching (next-token-prediction, autograd, mlp, CNN: handwritten one-idea pages) and Part 2 of this
series, which he likes most. This file is the rule set for improving every other part. It sits above BRIEF.md for presentation frames.

## What the handwritten decks do (and Part 2 does)
1. A concrete example before any machinery. Page 1 of next-token prediction is "a b b _ ?  What is the next character?", then "pose as a classification task"
   with a tiny table of p(c) values. No definitions first.
2. One held drawing, one new mark per page. The aabid training rows appear one line at a time; the embedding table stays put while rows get highlighted;
   the concatenated feature vector keeps each position's colour; the MLP is drawn as circles and arrows above the same coloured vector.
3. Exact small numbers, written where they are used. E[a] = (0.1, 0.3), the feature vector (0.1, 0.3, −0.1, 0.1, 0.6, 0.4). Never "some numbers".
4. Colour carries identity, and nothing else. Yellow/green/blue = which position a number came from; red = the gradient or the loss; the same colour on the
   token, the table row, and the vector slot. (Our seven object colours are the same idea.)
5. A drawing, not a card. Circles, arrows, boxes with one word in them, a table with a hand-ruled line. Cards, badges and captions are UI, not teaching.
6. Big type, lots of empty space, at most one sentence of prose per page; the sentence is either the question or the "important idea".
7. Assumptions and scope written in the margin, once ("Assume: 26 lowercase chars, _ marks the end, 4 < len < 10"), not repeated as caveats.
8. Analogies drawn, not described (king − man + woman; the crying child). Every abstract step gets one picture a student can redraw.
9. The computational graph is drawn once and then annotated in colour: upstream gradient (red) × local gradient = downstream gradient.
10. The "important idea" is boxed, rare, and stated in one line ("learn a vector representation for each character; similar characters end up closer").

## Rules for our frames (presentation mode)
- Every section starts with a frame that is a question or an example, never a definition, a recap, or a control panel.
- A frame has one held drawing (an SVG, an overlay, a scatter, a table drawn as a table, chips with colours) and adds one mark per build. If a frame needs
  more than 6 builds, split it. If a frame has no drawing, it is either a question frame (one sentence, big) or it is wrong.
- Prose on a frame: at most 40 words, and only as the question, the reading of a number ("bank asks bright? 1.00"), or the boxed idea. Everything else
  is companion prose in read mode.
- Numbers appear next to the thing they measure (on the arrow, in the cell, on the patch), rounded to what a student can copy.
- Tables on a frame: at most one, with thumbnails or coloured chips as row labels; two tables side by side only for a before/after.
- Cards are containers in read mode; in present mode they should not draw a box around a drawing. Prefer the drawing alone with a one-line caption.
- Scope and caveats: one box per part, after the hero; never at the end of a section frame.
- Colour: only the seven object colours for objects, red only for loss and gradients and forbidden cells, position colours only for positions.
- Presenter notes: first line is the question to ask before the reveal, second line what to point at. Not a restatement.
- Read mode keeps the companion prose and the full tables; present mode is the whiteboard.

## Acceptance for a rebuilt frame
Question or example first; a drawing on the frame; one new mark per build; ≤ 40 words of prose; numbers on the drawing; fits 1280×720 with no fit warning;
notes present; the frame could be redrawn by hand on a whiteboard in under a minute.
