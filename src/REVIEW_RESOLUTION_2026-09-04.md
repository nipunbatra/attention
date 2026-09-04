# Review resolution: 4 September 2026

This is the implementation response to `REVIEW_2026-09-04.md`, not a replacement for it. The original review is preserved. Findings were reproduced against the sources and browser; proposed fixes were checked independently before adoption.

## Repairs made

| Review area | Resolution |
| --- | --- |
| Vision II stale tables | Rebuild each calculator's table on input. Mask changes can change row count, so updating existing cells alone is insufficient. Tests compare the visible rows and loss after repeated changes. |
| Finite-difference diagnostics | Preserve diagnostic precision. The two saved checks use central differences with epsilon 1e-4, maximum error 6.10794e-10. The independent 260-gradient check uses epsilon 1e-5, maximum error 4.51e-11. The learned model arrays are unchanged. |
| Position table | Show the stored 20-by-5 table, the ten selected rows, and zero gradients for unused rows in this example. |
| Baseline ties | State that eight candidate logits are exactly tied, at probability about 0.094 each. Do not name a winner chosen by sorting order, or imply every context-free model is uniform. |
| Image brightness | Every vision renderer uses one clamped grayscale ramp, `AT.imageShade(v)`. Larger pixel values look lighter; the dimming experiment now looks dimmer. |
| Arithmetic popovers | Convert stage coordinates, stabilize popup width, flip/reposition and clamp to the actual clipping boundary. Escape dismisses the dialog before leaving presentation. |
| Notes and keyboard | Reserve space for notes and recenter the slide. Sliders retain their native arrow/Page keys; unused presenter shortcuts work. Escape returns focus to the slide. |
| Presenter cues | Add cues/support to all Vision IV frames; split substantial single-paragraph notes elsewhere. Keep useful imperatives, rather than requiring every cue to be a question. |
| Contrast | Darken existing semantic hues and muted ink. The six object text colours exceed 4.5:1 on their matching tints. Keep masked cells distinct through hatching and symbols too. |
| Hero and navigation | Place notation below both hero columns; group Vision II notation meaningfully; typeset CLIP symbols; show distinct frame titles in overview and presenter view. Part III chips have explicit useful destinations rather than all returning to its recap. |
| Part II pacing | Replace the 60-minute promise with a multi-session label and a first-pass route that reaches a complete next-token prediction. No unsupported 150-minute estimate. |
| Notation and rounding | Repair residual colours, distinguish keys from values, replace undefined `U` with the actual vocabulary head and its shapes, and place rounding caveats beside the affected classroom calculations. |
| Metadata and source builds | Align titles, series labels, roadmaps and links. Link availability follows complete published sources, not which output file happened to be built first. Clean-directory tests cover all eight lessons and planned-link cases. |
| Release checks | QA and sweep return failure statuses on errors. Sweeps exercise selects and repeat range inputs. The focused interaction regression covers popovers, notes, keyboard handling, variable-row tables, notation and chip destinations. |

## Teaching additions

- **Vision I, Section 5:** after computing CLS, compare the same swapped patches with positions off and on. Without positions the two CLS outputs are both (3.293, 3.293); with positions they are (4.075, 4.075) and (3.565, 3.565). These are computed from the disclosed encoder, not invented labels.
- **Vision I, Section 7:** move the trained one-block example from bottom-right to top-left. The fitted model goes from ONE with probability 0.9953 to the wrong answer TWO with probability 0.7915. This demonstrates a failure to generalize, not a new counting capability.
- **Vision III, Section 7:** offer one wrong caption and ask students to work out its softmax probability. The answer is 1 because it is the only candidate. The reveal is also shown in the PDF.

## Advice not adopted literally

- A zero finite-difference error is not inherently impossible; the actual defect was rounding these particular small nonzero errors to zero.
- More printed decimal places cannot guarantee that rounded addends sum to rounded totals. Calculations keep full precision, with local explanations of rounding.
- The proposed popup scale conversion and notes CSS change did not by themselves resolve clipping/overlap. The fixes account for the actual frame and separately reserved notes height.
- Native range-control keys are not broken navigation. Stealing those keys would damage keyboard use.
- CLS and patch 1 sharing an initial position vector is not a mathematical error. Their incoming content differs; changing the toy parameters merely to make positions unique is unnecessary.
- The learning examples remain small disclosed toys. Training loss on their pictured examples is not held-out accuracy, robust counting, or evidence of pretrained language understanding.

## Remaining limits

The full Part II walkthrough still needs multiple sessions; the route is guidance, not a separately authored short deck. Wide phone-reading tables may pan within their containers; classroom frames never scroll. Slide PDFs are appearance-faithful raster pages with answers shown, not searchable/selectable text. Tests cover Chromium and the stated viewports, not every browser, assistive technology or classroom projector. A future pass can test the teaching route with students and add measured timings.

Release evidence and reproduction commands are recorded in `CLASSROOM_QA.md`, `README.md`, and `HANDOVER.md`. Passing these checks is not a claim that no further improvements are possible.
