# PRESENT.md — one lesson, two layouts

The source is **slide-first, not slide-only**. Authors compose a sequence of bounded teaching frames. Presentation mode
shows one frame on a fixed 16:9 stage; reading mode removes the stage and unfolds those same frames into a continuous
article, with optional companion prose between them. There is one DOM, one set of interactive widgets, and one notation
system—never a separately maintained slide deck.

## The layout contract

- A teaching frame is `<div class="frame" data-title="…">` inside a `section.sec`.
- Presentation uses a logical **1280 × 720** stage. JavaScript uniformly scales the complete stage into the viewport,
  with a quiet outer margin. There is no persistent header or footer. Browser shape and resolution change only the scale; they do not trigger a
  mobile reflow of the slide.
- Presentation typography is expressed in logical stage pixels through reusable CSS tokens:
  `--present-body: 28px`, `--present-title: 42px`, `--present-caption: 22px`, and `--present-math: 32px`.
  A lesson may override the tokens, but it must not use viewport-relative type inside a frame.
- In presentation, `data-title` is the single large slide heading. Repeated section labels and numbers are hidden.
  If `data-title` is absent (or repeats the section title), the section heading becomes the slide heading.
- A frame **never scrolls and never shrinks its own type to fit**. If content exceeds the stage, move the next idea to a
  continuation frame. The runtime outlines the overfull frame and displays an authoring warning with the excess width or
  height. This warning is absent for a valid frame.
- Reading mode is ordinary responsive document flow. `.frame` wrappers impose no fixed height, transform, or clipping;
  all frames and their builds are visible. `.companion` prose can appear before, between, or after frames and is available
  in full reading mode (hidden in lean reading and presentation).

This direction is deliberate: a frame is the smallest teachable claim. The article is the frames reopened, connected by
the prose needed for self-study.

## Modes and URLs

- Read mode is the default. The strip toggle switches between full reading and lean reading.
- Present mode is `body.present`. Enter through **Present**, key `P`, `?present`, or an explicit hash such as
  `#s05/2/1`. A bare article anchor such as `#s07` remains in reading mode.
- Present/P starts at the section currently in view. An explicit URL starts at its requested section/frame/build.
- Esc or Exit returns to reading at the current section, removes `present` from the query string, and reduces the hash to
  `#sNN`. Presentation state is not stored.

## Authoring a section

```html
<section class="sec" id="s07" data-title="The bank position needs context">
  <header class="sec-head">…<h2>The bank position needs context</h2></header>

  <p class="companion">Long-form bridge for a reader. Hidden while presenting.</p>

  <div class="frame" data-title="Ask the prefix for a useful clue" data-autobuild="off">
    <p>The sentence and the one question this frame answers.</p>
    <figure data-build="1">…</figure>
    <p class="callout callout-key" data-build="2">One conclusion.</p>
    <script type="text/x-notes">
      Ask: which earlier word tells us what kind of bank this is?

      Point to money only after students answer. Then reveal the conclusion.
    </script>
  </div>

  <p class="companion">A reader-only explanation connecting the two frames.</p>

  <div class="frame" data-title="Turn the clue into an update">…</div>
</section>
```

Use one frame for one visible argument:

- Prefer 2–6 purposeful builds and one visual centrepiece.
- Keep the final conclusion visible in the fully built state.
- Repeat a small amount of context when a continuation frame needs it; every frame should make sense when deep-linked.
- Put the frame’s question, pointing cue, and transition in `text/x-notes`, not in tiny slide text.
- Use `<pre class="torch-snippet"><code>…</code></pre>` for one- to four-line code. In reading it may scroll
  horizontally; in presentation it wraps and participates in the same fit check.
- Do not depend on viewport `@media (max-width: …)` rules for slide composition. Scope reading-only responsive rules to
  `body:not(.present)`, or override them under `body.present` with the canonical grid.

Sections without explicit frames still receive one legacy auto-frame at runtime. That fallback makes old content usable,
but it is not a classroom authoring target.

## Builds and interactive components

- `data-build="n"` reveals an element on build `n`; unnumbered content is visible on build 0. Pending builds retain layout
  with `visibility:hidden`, so a frame is composed for its fullest state. Use `data-build-mode="collapse"` only when the
  deliberate reflow is part of the explanation.
- If a frame has no build attributes, direct teaching units are assigned sequential builds at runtime. Set
  `data-autobuild="off"` for deliberate choreography.
- Managed steppers consume next/back while they have steps, then frame navigation continues. A control inside
  `data-present="manual"` keeps native keyboard behaviour and its own state.
- Presentation hides the duplicate local toolbar of a managed stepper; keyboard navigation or the on-demand Previous/Next controls
  drive it. Manual steppers retain their controls. `.diagram-step` uses full-width SVG stages and a current-step heading
  in presentation; reading mode shows its compact stage selector above the same drawing.
- Sliders and toggles are presenter-driven. Their state resets when leaving the frame unless the widget uses
  `data-keep-state` or `data-present="manual"`.
- Builds, steppers, values, IDs, and event listeners are the same nodes in both modes. The runtime does not clone or
  re-render article content.

## Tables and numerical worksheets

`AT.ui.table` distinguishes semantic cell kinds. Numbers stay right-aligned with tabular figures; prose is left-aligned
in the reading font. Column, `lead`, and `computed` definitions accept `kind: 'auto' | 'number' | 'text' | 'code'`.
The default `auto` recognizes numeric values/strings; use `code` for left-aligned literal tokens or code-like labels when
needed. Updates and footer changes reclassify cells, so an interactive column does not retain stale formatting.

```js
AT.ui.table(rows, {
  cols: [{label: 'source', kind: 'text'}, {label: 'weight', kind: 'number', decimals: 3}]
});
```

Keep one precision per numeric column unless separately labelled rows explicitly compare different quantities.
Repeat column headers and the receiving token/query when a table continues on another frame. Worksheet headers must
remain readable (at least 18 logical pixels in presentation). Long row labels may scroll away on a narrow article view;
they must not pin most of the phone viewport. Wide article diagrams can be panned on small screens, but presentation
frames must still fit without internal scrolling.

`node src/check_tables.mjs part1.html attention.html part3.html` inspects desktop reading, 390px reading, and every
presentation state for text/numeric styling, worksheet type size, cell text overlap, and table containment. It saves
JSON evidence and representative screenshots. Pair it with `frame_audit.mjs`; neither test replaces visual inspection.

## Fit preflight

The live frame is checked after entry, resize, build changes, step changes, images, and form changes:

```js
AT.present.fitReport()
// { overflow, horizontal, vertical, section, frame, title, ... }
```

For an authoring audit, run the asynchronous all-frame preflight in the browser console or a Playwright check:

```js
const report = await AT.present.preflight()
console.table(report.overflow)
```

Preflight reveals every build, samples every managed stepper state and opens native reveal panels on the canonical stage,
records the worst size, and then restores reading/presentation state. A release should have
`report.overflow.length === 0`. Do not silence the warning with CSS; split the frame.

The command-line `frame_audit.mjs` also reparses rendered KaTeX with strict error handling. This catches unknown commands that tolerant on-page rendering may display in red without a `.katex-error` element. It checks the article and every live managed state, with duplicate formulas cached for speed.

## Navigation and presenter tools

- Right / Space / PageDown / N: next step, build, then frame.
- Left / PageUp / Backspace: reverse that sequence. Home / End: first / last frame.
- O: overview. B: blank screen. S: notes. ?: help. F: browser full screen.
- Click **Controls** or press **C** for Previous, Next, Frames, Full screen, Presenter view, and Exit. The panel overlays
  the page without resizing the stage. It starts closed, and hidden controls cannot receive keyboard focus. Escape closes
  it first; Escape again returns to reading. Arrow-key navigation works with the panel closed.
- The URL hash `#sNN/f/b` follows the current frame and build for deep links and break-time resumption.
- Presenter view opens the same file with `#presenter`, showing current/next frame, notes, build status, and a clock. It is
  synchronized through `postMessage`, including from `file://`.

## Reading, handout, and print

In reading, frames are transparent structural wrappers, so the article remains responsive and accessible. Full reading
shows `.companion`; lean reading hides it. `@media print` removes the fixed stage, shows all builds and companions, places
one frame per landscape page, advances steppers to their final state, and opens answer reveals before printing.
After printing, the runtime restores the interactive state.

### Exact slide-view PDF

Browser Print is a handout. To export the actual 16:9 classroom stage, use the bundled exporter:

```sh
# One fully revealed PDF page per authored frame (default)
node src/export_slides.mjs attention.html output/pdf/attention-part2-slides.pdf

# Preserve every build and managed stepper state as a separate PDF page
node src/export_slides.mjs attention.html output/pdf/attention-part2-builds.pdf --builds all

# Optionally retain the exact 1280x720 source PNG for every exported page
node src/export_slides.mjs attention.html deck.pdf --frames output/slide-pngs

# Rasterize at 1x, 2x (the default), or 3x resolution
node src/export_slides.mjs attention.html deck.pdf --scale 3

# Optional question handout: preserve authored open/closed answer states
node src/export_slides.mjs attention.html deck.pdf --answers authored
```

The exporter uses the same live presentation runtime, hides browser chrome, checks every open reveal/build/step for fit,
and refuses to make a PDF while any frame overflows. It installs no package and resolves an existing Playwright runtime.
Each PDF page is the exact rendered 16:9 stage, stored as a high-resolution image so browser typography, diagrams, and
widget states match the classroom view. This means the exported text is not selectable. The default export advances
authored builds and managed steppers to their final state and opens every `details.reveal` answer in that completed frame.
The default is `--answers show`: PDF readers should not lose an explanation because they cannot click it. With `--builds all`,
earlier build pages remain unanswered; answers appear on the completed frame. `--answers authored` preserves the authored
open/closed state instead. Presenter-driven sliders and other manual controls keep their defaults.

## Focused checks

Run from the project root:

```sh
node src/pres_test.mjs
node src/export_test.mjs
node src/check_tables.mjs part1.html attention.html part3.html
node src/frame_audit.mjs attention.html --shots /tmp/attention-overflow
node src/export_slides.mjs attention.html /tmp/attention-slides.pdf
```

The test uses an already installed Playwright runtime and writes screenshots to a printed temporary path. It covers
reading anchors, deep links, builds, nested/multiple steppers, manual controls, modal focus, presentation-state restore,
presenter-window sync, print-state restoration, canonical-stage scaling, on-demand controls, no-scroll code snippets, and accessible
arithmetic. `frame_audit.mjs` walks the assembled lesson, checks the 1280×720 stage contract at every live state, invokes
the full preflight (including open reveals), and writes screenshots only for failures. Run it for every assembled part
before release.

`export_test.mjs` checks final/all-build exports with shown/authored answers. It verifies PDF page counts and actual answer
pixels in the exported PNGs, including a reveal created only in the final step of a managed widget.
