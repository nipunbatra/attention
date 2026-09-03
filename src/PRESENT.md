# PRESENT.md — presentation mode for attention.html (technical design)

Goal: the same single file works as (1) a scrolling article for self-study and (2) a frame-by-frame lecture where
one new mark appears per build, driven by the arrow keys or a clicker. Content is authored once.

## Modes
- Read mode (default): today's page. Companion prose visible (or lean via the strip toggle).
- Present mode: body.present. Entered by the strip button "Present", key P, or URL `?present` / hash `#s05/2/1`.
  A bare article anchor such as `#s07` does **not** enter presentation, either on load or on an anchor change.
  Present/P starts at the section currently in view. An explicit URL starts at its requested section/frame/build.
  Exit with Esc or either Exit button. Exit restores reading at the current section, removes the `present` query
  parameter while preserving other query parameters, and reduces the hash to `#sNN`. Reload then stays in reading mode.
  Present mode persists nothing in storage.

## Frames
- A frame is `<div class="frame" data-title="...">` inside a `section.sec`. A section with no .frame wrappers is one frame
  (the whole section). The section header (.sec-head) is shown as the frame's title bar in present mode (number + title),
  the frame's own data-title as a subtitle when present.
- Authored frames fill the available dynamic viewport (`100dvh`), minus the measured notation strip and classroom
  controls. Short frames centre vertically; tall frames scroll internally. There is no scale-to-fit text shrinking.
  The legacy whole-section fallback is for unconverted content, not the classroom authoring target: split teaching
  content into explicit frames instead of putting an entire lesson in one auto frame.
  Base font is clamp(21px, 1.7vw, 30px); tables, vectors and worksheets scale with it; .companion is hidden.
- A persistent bottom toolbar has Previous, Next, Frames, Full screen and Exit, with a live frame/build/step status.
  Navigation boundary buttons are disabled. The notation strip and toolbar may wrap on narrow displays; their measured
  heights are reserved so neither obscures the frame. The progress bar continues to work.
- One- or two-line code examples use `<pre class="torch-snippet"><code>…</code></pre>`; `pre.pytorch` is an equivalent
  alias. Both are monospace, bounded by their container and horizontally scrollable in reading and presentation modes.

## Builds
- Elements carry `data-build="n"` (n >= 1). Build 0 (no attribute) is visible immediately. Right arrow reveals n = 1, 2, ...
  Hidden builds keep their layout (visibility:hidden; opacity transition 180ms) unless `data-build-mode="collapse"`.
  Pending builds are inert and aria-hidden; their controls cannot receive focus. Existing author inert/aria-hidden
  values are restored when revealed or when returning to reading. Nested elements wait for the highest enclosing
  build number, not merely their nearest data-build ancestor.
- AUTO-BUILDS: if a frame has no data-build attributes, direct children matching
  `.card, .callout, .tex-display, p:not(.companion), .chips, table, .stepper, .side-by-side > *` get sequential build numbers
  at runtime (the first one is build 0). `data-autobuild="off"` on the frame disables this. Hand-authored numbers win.
- STEPPERS: a stepper element in the current build captures next/back while it has steps left/back, then navigation
  continues to the next build. Multiple steppers in one build advance in DOM order and rewind in reverse DOM order.
  Entering a build forward starts its steppers at their first step; returning backward lands at their final step,
  including when returning from another frame. Managed steppers reset when the frame is left.
  A widget or ancestor marked `data-present="manual"` opts out: its own arrow controls continue to work and its state
  is retained when changing frames. `AT.ui.stepper` already registers the necessary `el.stepperApi`; SVG renderers can
  use its normal `steps[].render(stage, context)` hook without a separate presentation registration API.
- Toggles/sliders are never auto-driven; the presenter clicks them. Their state is reset when the frame is left (optional
  `data-keep-state` or `data-present="manual"` retains it). Range inputs keep native arrow/Home/End behavior.

## Navigation
- Right / Space / PageDown / N: next step, then build, then frame. Left / PageUp / Backspace: reverse that sequence.
  P enters presentation from reading; it is not the previous key. Buttons and editable widgets retain their normal Space behavior.
- Home / End: first / last frame. O or Frames: overview (a grid of frame titles; click or press Enter to jump).
  Overview traps Tab/Shift+Tab, moves focus with arrows/Home/End, makes background content inert, and restores focus on
  close. It has a visible Close overview button; O or Esc also closes it. Overview keys never move the underlying deck.
- B: blank screen. S: notes. ?: help. F or Full screen: request browser full screen. If unavailable or denied, a visible
  message explains that the browser's own full-screen command can be used and classroom navigation remains available.
  Esc or Exit returns to reading (native browser full-screen handling can consume the first Esc).
- URL hash `#sNN/f/b` is kept current so a frame can be deep-linked or resumed after a break; on load, present mode with a hash
  jumps there. The URL records build, not an internal stepper index; loading that build starts its current steppers at step 1.

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

## Focused runtime checks

From the project root, run `node src/pres_test.mjs`. It serves the existing component gallery from the current source
in memory; it neither rebuilds nor edits `attention.html`. Screenshots go to a printed temporary-directory path.
Alternatively pass an already assembled gallery and output directory: `node src/pres_test.mjs /path/to/test00.html /path/to/screenshots`.

The test resolves an existing local Playwright module, then existing npm-cache modules. It installs no dependencies.
Set `PLAYWRIGHT_MODULE` to an installed module path and, when needed, `PLAYWRIGHT_CHROMIUM_EXECUTABLE` to a browser binary.
Checks cover reading anchors, explicit deep links, forward/back builds and nested/multiple steppers, manual controls,
inert pending builds, modal focus, visible controls, full-screen fallback, presenter-window sync, print restoration,
exit/reload, mobile toolbar clearance, snippet overflow, rounded arithmetic labels, and fresh accessible mixture footers.
