// Export the exact classroom stage to a 16:9 PDF.
//
// Usage:
//   node src/export_slides.mjs attention.html output/pdf/part2-slides.pdf
//   node src/export_slides.mjs attention.html output/pdf/part2-builds.pdf --builds all
//   node src/export_slides.mjs attention.html deck.pdf --frames output/slide-pngs
//   node src/export_slides.mjs attention.html deck.pdf --answers authored
//
// The default writes one fully revealed page per authored frame. `--builds all`
// writes each build and managed stepper state as its own page. The exporter uses
// the normal presentation runtime and refuses to export any overfull frame.
// Slides are 2x-resolution raster snapshots: live SVGs/widgets retain their exact
// appearance, but text in this PDF is not selectable. Use --scale 1 for smaller files.
import { createRequire } from 'module';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { pathToFileURL } from 'url';

const require = createRequire(import.meta.url);
const candidates = [process.env.PLAYWRIGHT_MODULE, 'playwright', 'playwright-core'].filter(Boolean);
const npmCache = path.join(os.homedir(), '.npm', '_npx');
if (fs.existsSync(npmCache)) {
  for (const dir of fs.readdirSync(npmCache).sort()) {
    candidates.push(path.join(npmCache, dir, 'node_modules', 'playwright'));
  }
}
let pw;
for (const candidate of candidates) {
  try { pw = require(candidate); break; } catch { /* use the next installed runtime */ }
}
if (!pw) throw new Error('No existing Playwright runtime found. Set PLAYWRIGHT_MODULE; the exporter installs nothing.');

const args = process.argv.slice(2);
const source = args[0];
const output = args[1];
const option = (flag, fallback) => {
  const i = args.indexOf(flag);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const buildMode = option('--builds', 'final');
const answerMode = option('--answers', 'show');
const requestedFrames = option('--frames', '');
const captureScale = Number(option('--scale', '2'));
if (!source || !output || !['final', 'all'].includes(buildMode) || !['show', 'authored'].includes(answerMode) || ![1, 2, 3].includes(captureScale)) {
  throw new Error('Usage: node src/export_slides.mjs page.html output.pdf [--builds final|all] [--answers show|authored] [--frames DIR] [--scale 1|2|3]');
}
if (path.extname(output).toLowerCase() !== '.pdf') throw new Error('Output must end in .pdf');
if (!fs.existsSync(source) || !fs.statSync(source).isFile()) throw new Error('Source HTML file not found: ' + source);

const outPath = path.resolve(output);
fs.mkdirSync(path.dirname(outPath), { recursive: true });
const ownTemp = !requestedFrames;
const frameDir = requestedFrames ? path.resolve(requestedFrames) : fs.mkdtempSync(path.join(os.tmpdir(), 'attention-slide-export-'));
fs.mkdirSync(frameDir, { recursive: true });

let browser;
try {
  browser = await pw.chromium.launch(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
    ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
    : {});
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 }, deviceScaleFactor: captureScale,
    colorScheme: 'light', reducedMotion: 'reduce'
  });
  const page = await context.newPage();
  const runtimeErrors = [];
  page.on('pageerror', error => runtimeErrors.push('PAGEERROR: ' + String(error.message || error).split('\n')[0]));
  page.on('console', message => {
    if (message.type() === 'error') runtimeErrors.push('CONSOLE: ' + message.text().slice(0, 240));
  });

  const url = new URL(pathToFileURL(path.resolve(source)).href);
  url.searchParams.set('present', '');
  await page.goto(url.href, { waitUntil: 'load' });
  await page.waitForFunction(() => window.AT && AT.present && AT.present.state().active);
  await page.addStyleTag({ content: `
  #strip,#at-controls-toggle,#at-controls,#at-counter,#at-announcement,#at-fit-warning,#at-notes,#at-overview,#at-blank,#at-help{display:none!important}
  html,body{width:1280px!important;height:720px!important;background:var(--paper)!important}
  body.present{--present-strip-h:0px!important;--present-controls-h:0px!important;--present-gutter:0px!important;--present-scale:1!important}
  body.present main{width:1280px!important;height:720px!important}
  body.present .sec.is-live{transform:translate(-50%,-50%) scale(1)!important}
` });
  // Wait for fonts, image decoding, and the stage's deferred SVG repaint before
  // measuring or capturing. A fixed timeout alone can capture a stale widget.
  const settle = () => page.evaluate(async () => {
    await document.fonts.ready;
    await Promise.all(Array.from(document.images, img => img.decode().catch(() => {})));
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  });
  await page.evaluate(() => window.dispatchEvent(new Event('resize')));
  await settle();

  const preflight = await page.evaluate(() => AT.present.preflight());
  if (preflight.overflow.length) {
    const names = preflight.overflow.slice(0, 12).map(r => `${r.section}.${r.frame} ${r.title} (+${r.horizontal}px × +${r.vertical}px)`).join('\n');
    throw new Error(`Refusing to export ${preflight.overflow.length} overfull frame(s). Split them first:\n${names}`);
  }
  await page.evaluate(() => AT.present.first());
  await settle();

  const images = [];
  const seen = new Set();
  let revealedAnswers = 0;
  let reachedEnd = false;
  for (let guard = 0; guard < 5000; guard++) {
    const { state, steps } = await page.evaluate(() => {
      const state = AT.present.state();
      const live = document.querySelector('.frame.is-live');
      // Include every managed stepper: two steppers can share a build and both
      // visit index 0, so the active stepper's index alone is not a unique state.
      const steps = Array.from(live.querySelectorAll('.stepper'))
        .filter(el => el.stepperApi && !el.closest('[data-present="manual"]'))
        .map(el => el.stepperApi.index() + 1);
      return { state, steps };
    });
    const key = state.hash + (steps.length ? ':steps-' + steps.join('-') : '');
    if (seen.has(key)) throw new Error('Presentation navigation stopped before the final frame: ' + key);
    seen.add(key);
    const terminal = state.build === state.frame.maxBuild && (!state.stepper || state.stepper.index === state.stepper.count - 1);
    if (buildMode === 'all' || terminal) {
      // A PDF cannot be clicked. On each fully built frame, open its authored
      // answer panels before capture. Earlier build pages remain questions.
      if (answerMode === 'show' && terminal) {
        const opened = await page.evaluate(() => {
          const live = document.querySelector('.frame.is-live');
          if (!live) return 0;
          const panels = Array.from(live.querySelectorAll('details.reveal')).filter(d => !d.closest('.is-pending,[hidden]'));
          let count = 0;
          panels.forEach(d => { if (!d.open) { d.open = true; count++; } });
          return count;
        });
        revealedAnswers += opened;
        if (opened) await settle();
      }
      const fit = await page.evaluate(() => AT.present.fitReport());
      if (fit && fit.overflow) throw new Error('Frame became overfull while exporting: ' + key);
      const frameName = String(images.length + 1).padStart(3, '0') + '-' + key.replace(/[^a-z0-9_-]+/gi, '_').replace(/^_+/, '') + '.png';
      const framePath = path.join(frameDir, frameName);
      await page.locator('.sec.is-live').screenshot({ path: framePath, animations: 'disabled' });
      images.push(framePath);
    }
    if (state.fi === state.total - 1 && terminal) { reachedEnd = true; break; }
    // Use the same navigation path as classroom controls, without depending on
    // browser focus remaining outside an interactive input or widget.
    await page.evaluate(() => AT.present.next());
    await settle();
  }

  if (!reachedEnd) throw new Error('Export exceeded 5,000 navigation states; no partial PDF was written.');
  if (!images.length) throw new Error('No slide frames were captured.');
  if (buildMode === 'final' && images.length !== preflight.total) throw new Error('Not every authored frame was captured.');
  if (runtimeErrors.length) throw new Error(runtimeErrors.join('\n'));

  const pdfPage = await context.newPage();
  const imageTags = images.map((file, i) => {
    const data = fs.readFileSync(file).toString('base64');
    return `<section class="page" aria-label="Slide ${i + 1}"><img alt="" src="data:image/png;base64,${data}"></section>`;
  }).join('');
  await pdfPage.setContent(`<!doctype html><html><head><meta charset="utf-8"><title>Classroom slides</title><style>
  @page{size:13.333333in 7.5in;margin:0}
  *{box-sizing:border-box}
  html,body{margin:0;padding:0;background:#fff}
  .page{width:1280px;height:720px;margin:0;break-after:page;page-break-after:always;overflow:hidden}
  .page:last-child{break-after:auto;page-break-after:auto}
  img{display:block;width:1280px;height:720px;object-fit:contain}
</style></head><body>${imageTags}</body></html>`, { waitUntil: 'load' });
  // Decode sequentially so a long 2x/3x deck does not request hundreds of large
  // decoded bitmaps at once. Chromium can evict earlier pages before printing.
  await pdfPage.evaluate(async () => {
    for (const img of document.images) {
      if (!img.complete || !img.naturalWidth) await img.decode();
    }
  });
  await pdfPage.pdf({ path: outPath, preferCSSPageSize: true, printBackground: true, margin: { top: '0', right: '0', bottom: '0', left: '0' } });
  const bytes = fs.statSync(outPath).size;
  console.log(JSON.stringify({ source: path.resolve(source), output: outPath, mode: buildMode, answerMode, revealedAnswers, pages: images.length, rasterScale: captureScale, bytes }, null, 2));
} finally {
  if (browser) await browser.close();
  // This directory is created by mkdtemp above; never remove a caller's --frames directory.
  if (ownTemp) fs.rmSync(frameDir, { recursive: true, force: true });
}
