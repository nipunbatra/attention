# PRESENT.md — presentation mode for attention.html (technical design)

Goal: the same single file works as (1) a scrolling article for self-study and (2) a frame-by-frame lecture where
one new mark appears per build, driven by the arrow keys or a clicker. Content is authored once.

## Modes
- Read mode (default): today's page. Companion prose visible (or lean via the strip toggle).
- Present mode: body.present. Entered by the strip button "Present", key P, or URL `?present` / hash `#s05/2/1`.
  Exit with Esc or the button. Present mode persists nothing.

## Frames
- A frame is `<div class="frame" data-title="...">` inside a `section.sec`. A section with no .frame wrappers is one frame
  (the whole section). The section header (.sec-head) is shown as the frame's title bar in present mode (number + title),
  the frame's own data-title as a subtitle when present.
- Frames fill the viewport: min-height 100vh, flex column, vertically centred when short, scroll inside when tall
  (never cut content). Base font clamp(20px, 1.35vw, 30px); tables and vectors scale with it; .companion hidden.
- A frame counter (section.frame/total) bottom-right; the notation strip shrinks to one line; progress bar continues to work.

## Builds
- Elements carry `data-build="n"` (n >= 1). Build 0 (no attribute) is visible immediately. Right arrow reveals n = 1, 2, ...
  Hidden builds keep their layout (visibility:hidden; opacity transition 180ms) unless `data-build-mode="collapse"`.
- AUTO-BUILDS: if a frame has no data-build attributes, direct children matching
  `.card, .callout, .tex-display, p:not(.companion), .chips, table, .stepper, .side-by-side > *` get sequential build numbers
  at runtime (the first one is build 0). `data-autobuild="off"` on the frame disables this. Hand-authored numbers win.
- STEPPERS: a stepper element that is the current build captures the right/left arrow while it has steps left/back
  (calls its Next/Previous), then navigation continues to the next build. Reset when the frame is left.
- Toggles/sliders are never auto-driven; the presenter clicks them. Their state is reset when the frame is left (optional
  `data-keep-state`).

## Navigation
- Right / Space / PageDown / N: next build, then next frame. Left / PageUp / P... (P is Present; use Left/PageUp/Backspace).
- Home / End: first / last frame. O: overview (the roadmap as a grid of frame titles; click to jump). B: blank screen. Esc: exit.
- URL hash `#sNN/f/b` is kept current so a frame can be deep-linked or resumed after a break; on load, present mode with a hash
  jumps there.

## Presenter notes and presenter window
- Notes per frame: `<script type="text/x-notes">...</script>` inside the frame (plain text, blank line = paragraph), or
  `data-notes` for one-liners. Typical content: the question to ask before the next reveal; what to point at; a timing hint.
- Presenter view: strip button "Presenter view" opens the same file in a new window with `#presenter`; that window renders
  a three-pane layout (current frame title + build count, notes, next frame title) and a clock. Sync both ways with
  window.postMessage('*') (works from file://). The presenter window's keys drive the main window.
- Fallback (single screen): key S toggles a small notes strip at the bottom of the main window.

## Handout / print
- @media print: one frame per page (page-break-before), all builds visible, .companion visible, steppers set to their final
  step by a beforeprint handler, controls hidden. Landscape A4 with 16px base.

## Authoring contract for sections
- Wrap content in .frame blocks with data-title; keep 1 to 6 frames per section, 2 to 8 builds per frame.
- Put a `<script type="text/x-notes">` in each frame. First line = the question to ask before the reveal.
- Anything that needs to persist across frames in a section goes outside the frames? No: every frame must stand alone
  (repeat the held drawing if the next frame builds on it; use the same component call so it renders identically).
