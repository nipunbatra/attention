// Present-mode test. Usage: node src/pres_test.mjs [test00.html] [outdir]
// Without a file argument, serves the existing sec00 gallery from source in memory; no assembled file is written.
// Opens the page with ?present, walks every build with ArrowRight (screenshots into outdir), and checks:
// hash sync, hidden builds before their turn, stepper capture, overview, blank, notes, print preparation, Esc to read mode,
// the presenter window (postMessage both ways). Exits 1 on any failure or any page/console error.
import { createRequire } from 'module'; import path from 'path'; import fs from 'fs'; import os from 'os';
import { fileURLToPath, pathToFileURL } from 'url';
const require = createRequire(import.meta.url);
const candidates = [process.env.PLAYWRIGHT_MODULE, 'playwright', 'playwright-core'].filter(Boolean);
const npmCache = path.join(os.homedir(), '.npm', '_npx');
if (fs.existsSync(npmCache)) for (const dir of fs.readdirSync(npmCache).sort()) candidates.push(path.join(npmCache, dir, 'node_modules', 'playwright'));
let pw;
for (const candidate of candidates) { try { pw = require(candidate); break; } catch {} }
if (!pw) throw new Error('No existing Playwright runtime found. Set PLAYWRIGHT_MODULE to its installed module path. No dependency is installed by this test.');
const file = process.argv[2], src = path.dirname(fileURLToPath(import.meta.url));
const outdir = process.argv[3] || fs.mkdtempSync(path.join(os.tmpdir(), 'attention-present-'));
fs.mkdirSync(outdir, { recursive: true });
console.log('Screenshots: ' + outdir);
const fails = [], errs = [];
const check = (cond, msg) => { if (!cond) fails.push(msg); console.log((cond ? 'ok   ' : 'FAIL ') + msg); };
const b = await pw.chromium.launch(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE } : {});
const ctx = await b.newContext({ viewport: { width: 1280, height: 720 } });
let baseUrl;
if (file) baseUrl = pathToFileURL(path.resolve(file)).href;
else {
  const read = name => fs.readFileSync(path.join(src, name), 'utf8');
  const part = JSON.parse(read('part2.json'));
  if (part.prev) part.prev.available = false;
  if (part.next) part.next.available = false;
  const safeJSON = value => JSON.stringify(value).replace(/<\//g, '<\\/');
  const shared = '<script>window.__TOY__=' + safeJSON(JSON.parse(read('toy.json'))) + ';window.__PART__=' + safeJSON(part) + ';</script><script>' + read('shared.js') + '</script>';
  const html = read('shell.html').replace('<!--KATEX-->', () => read('katex-bundle.html')).replace('<!--SHARED-->', () => shared).replace('<!--SECTIONS-->', () => read('sections/sec00_demo.html'));
  baseUrl = 'http://attention.test/gallery.html';
  await ctx.route('http://attention.test/**', route => route.fulfill({ contentType: 'text/html', body: html }));
}
const hook = (p, tag) => { p.on('pageerror', e => errs.push(tag + ' PAGEERROR: ' + String(e.message || e).split('\n')[0])); p.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errs.push(tag + ' CONSOLE ' + m.type() + ': ' + m.text().slice(0, 200)); }); };
const p = await ctx.newPage(); hook(p, 'main');
await p.goto(baseUrl + '#s00', { waitUntil: 'load' });
check(await p.evaluate(() => !AT.present.isActive() && !document.body.classList.contains('present') && location.hash === '#s00'), 'a bare article anchor stays in reading mode');
check(await p.evaluate(() => AT.present.parseHash('#s07') === null && AT.present.parseHash('#s07/1/0').id === 's07'), 'only the explicit frame/build hash is a presentation link');
if (!file) check(await p.evaluate(() => !document.querySelector('.series-nav a[rel="next"]') && !!document.querySelector('.series-coming[aria-disabled="true"]')), 'unavailable sibling parts have Coming soon text instead of broken links');
await p.goto(baseUrl + '?keep=1&present#s00', { waitUntil: 'load' });
await p.waitForTimeout(700);
const st = async () => p.evaluate(() => { const s = AT.present.state(); return { active: s.active, fi: s.fi, build: s.build, hash: location.hash, stepper: s.stepper, total: s.total, max: s.frame && s.frame.maxBuild }; });
const vis = async sel => p.evaluate(sel => { const e = document.querySelector(sel); if (!e) return null; const cs = getComputedStyle(e); return cs.visibility !== 'hidden' && cs.display !== 'none' && +cs.opacity > 0.01; }, sel);
let s = await st();
check(s.active && await p.evaluate(() => document.body.classList.contains('present')), 'present mode entered from ?present');
check(s.hash === '#s00/1/0', 'hash is #s00/1/0 on entry (got ' + s.hash + ')');
check(await vis('#s00-f1-chips'), 'build 0 (chips) visible at once');
check((await vis('#s00-frame1 .card[data-build="1"]')) === false, 'build 1 hidden before its turn');
check((await vis('#s00-f1-stepper')) === false, 'build 2 (stepper) hidden before its turn');
check(await p.evaluate(() => { const el = document.querySelector('#s00-f1-stepper'); return el.inert && el.getAttribute('aria-hidden') === 'true'; }), 'pending builds are inert and hidden from assistive technology');
check(await p.evaluate(() => { const button = document.querySelector('#s00-f1-stepper button'); button.focus(); return document.activeElement !== button; }), 'a pending stepper cannot receive keyboard focus');
check(await p.locator('#at-controls').isVisible() && await p.locator('#at-prev').isDisabled() && await p.locator('#at-next').isEnabled(), 'visible classroom controls expose the correct beginning boundary');
check(await p.evaluate(() => document.querySelector('#at-counter').getAttribute('aria-label').includes('Frame 1 of 2')), 'classroom status includes a meaningful frame description');
check((await vis('#s00-frame1 .callout[data-build="3"]')) === false, 'build 3 hidden before its turn');
check((await vis('#s00-frame2')) === false, 'frame 2 not shown while frame 1 is live');
check(await p.evaluate(() => getComputedStyle(document.querySelector('.companion')).display === 'none'), 'companion prose hidden in present mode');
check(await p.evaluate(() => document.querySelector('#at-counter').textContent.startsWith('00.1/2')), 'frame counter reads 00.1/2');
await p.screenshot({ path: path.join(outdir, 'f1-b0.png') });
await p.keyboard.press('ArrowRight'); await p.waitForTimeout(250);
s = await st();
check(s.hash === '#s00/1/1', 'ArrowRight -> hash #s00/1/1 (got ' + s.hash + ')');
check(await vis('#s00-frame1 .card[data-build="1"]'), 'build 1 visible after one ArrowRight');
check((await vis('#s00-f1-stepper')) === false, 'build 2 still hidden after one ArrowRight');
await p.screenshot({ path: path.join(outdir, 'f1-b1.png') });
await p.keyboard.press('ArrowRight'); await p.waitForTimeout(250);
s = await st();
check(s.hash === '#s00/1/2' && s.stepper && s.stepper.index === 0, 'build 2 shows the stepper at step 1 (got ' + s.hash + ', stepper ' + JSON.stringify(s.stepper) + ')');
check((await vis('#s00-frame1 .callout[data-build="3"]')) === false, 'build 3 still hidden while the stepper runs');
await p.screenshot({ path: path.join(outdir, 'f1-b2-step1.png') });
for (let k = 1; k < 4; k++) { await p.keyboard.press('ArrowRight'); await p.waitForTimeout(250); s = await st(); check(s.build === 2 && s.stepper && s.stepper.index === k, 'ArrowRight drives the stepper to step ' + (k + 1) + ' (build stays 2)'); await p.screenshot({ path: path.join(outdir, 'f1-b2-step' + (k + 1) + '.png') }); }
await p.keyboard.press('ArrowRight'); await p.waitForTimeout(250);
s = await st();
check(s.hash === '#s00/1/3', 'after the last step the next ArrowRight reveals build 3 (got ' + s.hash + ')');
check(await vis('#s00-frame1 .callout[data-build="3"]'), 'build 3 visible');
await p.screenshot({ path: path.join(outdir, 'f1-b3.png') });
// going back steps the stepper backwards
await p.keyboard.press('ArrowLeft'); await p.waitForTimeout(200); s = await st();
check(s.build === 2 && s.stepper && s.stepper.index === 3, 'ArrowLeft hides build 3 and lands on the stepper at its last step');
await p.keyboard.press('ArrowLeft'); await p.waitForTimeout(200); s = await st();
check(s.build === 2 && s.stepper.index === 2, 'ArrowLeft steps the stepper back');
for (let k = 0; k < 2; k++) { await p.keyboard.press("ArrowRight"); await p.waitForTimeout(120); }
s = await st(); check(s.hash === '#s00/1/3', 'forward again to build 3');
// next frame
await p.keyboard.press('ArrowRight'); await p.waitForTimeout(300); s = await st();
check(s.fi === 1 && s.hash === '#s00/2/0', 'ArrowRight past the last build moves to frame 2 (got ' + s.hash + ')');
check(await vis('#s00-frame2 .prose'), 'auto build 0 (prose) visible');
check((await vis('#s00-net')) === false, 'auto build 1 (card) hidden before its turn');
check(await p.evaluate(() => document.querySelector('#s00-frame2 .card').getAttribute('data-build') === '1' && document.querySelector('#s00-frame2 .callout').getAttribute('data-build') === '2' && document.querySelector('#s00-frame2 p.muted').getAttribute('data-build') === '3'), 'auto build numbers 1, 2, 3 assigned to card, callout, paragraph');
await p.screenshot({ path: path.join(outdir, 'f2-b0.png') });
await p.keyboard.press('ArrowRight'); await p.waitForTimeout(250);
check(await vis('#s00-net'), 'auto build 1 (network card) visible');
// slider drives the sketch; state is reset when the frame is left
await p.evaluate(() => { const r = document.querySelector('#s00-net-slider input'); r.value = 5; r.dispatchEvent(new Event('input', { bubbles: true })); });
await p.waitForTimeout(400);
check(await p.evaluate(() => document.querySelectorAll('#s00-net .col-in .node').length === 5), 'netSketch grows to 5 input nodes');
await p.screenshot({ path: path.join(outdir, 'f2-b1-w5.png') });
await p.evaluate(() => { const r = document.querySelector('#s00-net-slider input'); r.value = 100; r.dispatchEvent(new Event('input', { bubbles: true })); });
await p.waitForTimeout(400);
check(await p.evaluate(() => document.querySelectorAll('#s00-net .col-in .node').length === 6 && !!document.querySelector('#s00-net .col-in .ell') && document.querySelector('#s00-net .cap').textContent === '100 inputs'), 'netSketch collapses at w = 100 (3 + ellipsis + 2, caption "100 inputs")');
await p.screenshot({ path: path.join(outdir, 'f2-b1-w100.png') });
let n = 0; let last = (await st()).hash;
while (n < 20) { await p.keyboard.press('ArrowRight'); await p.waitForTimeout(200); const h = (await st()).hash; if (h === last) break; last = h; n++; await p.screenshot({ path: path.join(outdir, 'f2-' + h.replace(/[#/]/g, '_') + '.png') }); }
s = await st();
check(s.fi === 1 && s.build === 3 && s.hash === '#s00/2/3', 'walked to the last build of the last frame (got ' + s.hash + ')');
check(await vis('#s00-frame2 p.muted'), 'last auto build visible');
check(await p.locator('#at-next').isDisabled(), 'Next is disabled at the final build of the final frame');
// overview
await p.click('#at-overview-btn'); await p.waitForTimeout(200);
check(await p.evaluate(() => document.querySelector('#at-overview').classList.contains('is-on') && document.querySelectorAll('#at-overview .ov-item').length === 2), 'O opens the overview with 2 frames');
check(await p.evaluate(() => document.querySelector('main').inert && document.querySelector('#at-controls').inert && document.querySelector('#at-overview').getAttribute('aria-modal') === 'true'), 'overview makes background content inert');
await p.keyboard.press('End'); await p.keyboard.press('Tab');
check(await p.evaluate(() => document.activeElement.id === 'at-overview-close'), 'overview Tab wraps from the last frame to Close');
await p.keyboard.press('Shift+Tab');
check(await p.evaluate(() => document.activeElement === document.querySelector('#at-overview .ov-item:last-child')), 'overview Shift+Tab wraps to its last frame');
await p.keyboard.press('ArrowLeft');
check((await st()).fi === 1 && await p.evaluate(() => document.activeElement === document.querySelector('#at-overview .ov-item:first-child')), 'overview arrow keys move focus without navigating the deck');
await p.keyboard.press('Escape');
check(await p.evaluate(() => !document.querySelector('main').inert && document.activeElement.id === 'at-overview-btn'), 'overview restores background and opening-button focus');
await p.keyboard.press('o');
await p.screenshot({ path: path.join(outdir, 'overview.png') });
await p.click('#at-overview .ov-item:first-child'); await p.waitForTimeout(300); s = await st();
check(!(await p.evaluate(() => document.querySelector('#at-overview').classList.contains('is-on'))) && s.hash === '#s00/1/0', 'clicking a frame in the overview jumps there and closes it (got ' + s.hash + ')');
check(await p.evaluate(() => document.querySelector('#s00-net-slider input').value === '3'), 'slider reset to 3 after leaving frame 2');
check(await p.evaluate(() => AT.present.state().stepper === null && document.querySelector('#s00-f1-stepper .stepper').stepperApi.index() === 0), 'stepper reset to step 1 after leaving frame 1');
// blank + notes + help
await p.keyboard.press('b'); await p.waitForTimeout(100);
check(await p.evaluate(() => document.querySelector('#at-blank').classList.contains('is-on')), 'B blanks the screen');
await p.keyboard.press('b'); await p.waitForTimeout(100);
check(await p.evaluate(() => !document.querySelector('#at-blank').classList.contains('is-on')), 'B again unblanks');
await p.keyboard.press('s'); await p.waitForTimeout(150);
check(await p.evaluate(() => document.querySelector('#at-notes').classList.contains('is-on') && document.querySelector('#at-notes p').textContent.startsWith('Ask: which token')), 'S shows the notes strip with the first line of the frame notes');
await p.screenshot({ path: path.join(outdir, 'notes.png') });
await p.keyboard.press('s');
// hash navigation while presenting
await p.evaluate(() => { location.hash = '#s00/2/2'; }); await p.waitForTimeout(300); s = await st();
check(s.fi === 1 && s.build === 2, 'setting the hash to #s00/2/2 jumps there');
check(await vis('#s00-frame2 .callout') && (await vis('#s00-frame2 p.muted')) === false, 'deep link shows builds 0..2 and hides build 3');
// Fullscreen denial is recoverable and does not change classroom navigation.
await p.evaluate(() => Object.defineProperty(document.documentElement, 'requestFullscreen', { configurable: true, value: () => Promise.reject(new Error('simulated full-screen denial')) }));
await p.click('#at-fullscreen');
await p.waitForFunction(() => document.querySelector('#at-announcement').textContent.includes('unavailable'));
check(await p.locator('#at-announcement').isVisible() && (await st()).active, 'full-screen denial shows a visible browser-command fallback');
await p.evaluate(() => delete document.documentElement.requestFullscreen);
// print preparation
await p.evaluate(() => { location.hash = '#s00/1/1'; }); await p.waitForTimeout(300);
await p.evaluate(() => window.dispatchEvent(new Event('beforeprint'))); await p.waitForTimeout(150);
check(await p.evaluate(() => document.querySelectorAll('[data-build].is-pending').length === 0 && !document.body.classList.contains('present')), 'beforeprint reveals every build and leaves the present layout');
check(await p.evaluate(() => document.querySelector('#s00-f1-stepper .stepper').stepperApi.index() === 3), 'beforeprint sets the stepper to its last step');
check(await vis('#s00-frame1 .callout[data-build="3"]') && await vis('#s00-frame2 p.muted'), 'all builds of all frames visible for the handout');
await p.evaluate(() => window.dispatchEvent(new Event('afterprint'))); await p.waitForTimeout(150); s = await st();
check(s.active && s.fi === 0 && s.build === 1 && await p.evaluate(() => document.body.classList.contains('present')) && (await vis('#s00-f1-stepper')) === false, 'afterprint restores the presentation at #s00/1/1');
check(await p.evaluate(() => document.querySelector('#s00-f1-stepper .stepper').stepperApi.index() === 0), 'afterprint restores the stepper step');
// presenter window
const [pop] = await Promise.all([ctx.waitForEvent('page'), p.click('#presenter-btn')]);
hook(pop, 'presenter');
await pop.waitForLoadState('load'); await pop.waitForTimeout(900);
check(await pop.evaluate(() => document.body.classList.contains('presenter') && !!document.querySelector('#at-presenter')), 'presenter window renders the presenter layout');
check(await pop.evaluate(() => document.querySelector('.pr-cur h2').textContent === 'Component gallery (not shipped)' && document.querySelector('.pr-sub').textContent === 'Builds, hand numbered'), 'presenter shows the current section and frame title');
check(await pop.evaluate(() => document.querySelector('.pr-notes p').textContent.startsWith('Ask: which token')), 'presenter shows the notes');
check(await pop.evaluate(() => document.querySelector('.pr-next h3').textContent.includes('00.2')), 'presenter shows the next frame');
check(await pop.evaluate(() => /^\d\d:\d\d$/.test(document.querySelector('.pr-clock').textContent)), 'presenter clock runs');
await pop.screenshot({ path: path.join(outdir, 'presenter.png') });
await pop.keyboard.press('ArrowRight'); await p.waitForTimeout(300); s = await st();
check(s.hash === '#s00/1/2', 'ArrowRight in the presenter window drives the main window (got ' + s.hash + ')');
check(await pop.evaluate(() => document.querySelector('.pr-build').textContent.startsWith('build 2 of 3')), 'presenter build count follows the main window');
await pop.close();
// Esc returns to read mode
await p.keyboard.press('Escape'); await p.waitForTimeout(400); s = await st();
check(!s.active && !(await p.evaluate(() => document.body.classList.contains('present'))), 'Esc exits present mode');
check(await p.evaluate(() => location.hash === '#s00' && document.querySelectorAll('[data-build].is-pending').length === 0), 'read mode: hash #s00, no pending builds');
check(await p.evaluate(() => !new URL(location.href).searchParams.has('present') && new URL(location.href).searchParams.get('keep') === '1'), 'exit removes the present query parameter and preserves unrelated query state');
check(await vis('#s00-frame2 p.muted') && await vis('#s00-frame1 .callout[data-build="3"]') && await p.evaluate(() => getComputedStyle(document.querySelector('.companion')).display !== 'none'), 'read mode shows everything, companion included');
await p.screenshot({ path: path.join(outdir, 'read.png') });
// P re-enters at the section in view
await p.keyboard.press('p'); await p.waitForTimeout(300); s = await st();
check(s.active && s.hash === '#s00/1/0', 'P enters present mode at the section in view');
check(await p.locator('#s00-frame1').isVisible(), 're-entering the same frame restores its visible presentation layout');
await p.keyboard.press('Escape');
await p.reload({ waitUntil: 'load' });
check(!(await st()).active, 'reloading after exit stays in reading mode');
await p.goto(baseUrl + '#s00/1/2', { waitUntil: 'load' });
check((await st()).active && (await st()).build === 2, 'an explicit frame/build link enters directly at its requested build');
await p.click('#at-prev');
check((await st()).build === 1, 'visible Previous follows build navigation');
await p.click('#at-next');
check((await st()).build === 2 && (await st()).stepper.index === 0, 'visible Next follows build navigation');
await p.click('#at-exit');
check(!(await st()).active, 'visible Exit returns to reading');

// Test nested builds, multiple steppers in a build, and opt-out widgets without changing authored sections.
await p.goto(baseUrl + '#s00', { waitUntil: 'load' });
await p.reload({ waitUntil: 'load' }); // Hash-only navigation does not reset the runtime's discovered frame list.
await p.evaluate(() => {
  const frame = document.createElement('div'); frame.className = 'frame'; frame.id = 'nested-frame'; frame.dataset.title = 'Nested stepper fixture'; frame.dataset.autobuild = 'off';
  frame.innerHTML = '<p>Start</p><p data-build="1">Before the worksheet</p><div id="nested-build" data-build="2"><div data-build="1" id="nested-steppers"></div></div><div data-present="manual" id="manual-widget"><label>Manual range <input aria-label="Manual range" type="range" min="0" max="10" value="5"></label></div><pre class="pytorch"><code>q = e @ W_Q\nalpha = scores.softmax(dim=-1)</code></pre>';
  document.querySelector('#s00').appendChild(frame);
  const make = id => { const el = document.createElement('div'); el.id = id; return AT.ui.stepper({ el, steps: [{ title: 'First' }, { title: 'Last' }], hideList: true, scrollList: false }); };
  window.fixtureSteppers = [make('nested-a'), make('nested-b'), make('manual-stepper')];
  document.querySelector('#nested-steppers').append(fixtureSteppers[0].el, fixtureSteppers[1].el);
  document.querySelector('#manual-widget').append(fixtureSteppers[2].el);
  const last = document.createElement('div'); last.className = 'frame'; last.dataset.autobuild = 'off'; last.innerHTML = '<p>Final frame</p>'; document.querySelector('#s00').appendChild(last);
  AT.present.enter({ id: 's00', f: 3, b: 1 });
});
check((await st()).stepper === null && await p.evaluate(() => document.querySelector('#nested-build').inert), 'nested steppers wait for the highest enclosing build number');
await p.click('#at-next'); await p.click('#at-next');
check(await p.evaluate(() => fixtureSteppers[0].index() === 1 && fixtureSteppers[1].index() === 0), 'same-build steppers advance in document order');
await p.click('#at-next');
check(await p.evaluate(() => fixtureSteppers[1].index() === 1), 'the second stepper completes before leaving the build');
await p.locator('#manual-stepper .btn-next').focus(); await p.keyboard.press('ArrowRight');
check(await p.evaluate(() => fixtureSteppers[2].index() === 1 && AT.present.state().frame.index === 2), 'manual stepper keys remain local');
await p.locator('#manual-widget input').focus(); await p.keyboard.press('Home');
check(await p.evaluate(() => document.querySelector('#manual-widget input').value === '0' && AT.present.state().frame.index === 2), 'range Home is not hijacked by classroom navigation');
await p.click('#at-next');
check((await st()).fi === 3, 'Next leaves only after all managed steppers finish');
await p.click('#at-prev');
check(await p.evaluate(() => fixtureSteppers[0].index() === 1 && fixtureSteppers[1].index() === 1), 'Back from another frame lands on the final step of the previous frame');
check(await p.evaluate(() => fixtureSteppers[2].index() === 1 && document.querySelector('#manual-widget input').value === '0'), 'manual widgets retain their chosen state across frames');
await p.click('#at-prev');
check(await p.evaluate(() => fixtureSteppers[0].index() === 1 && fixtureSteppers[1].index() === 0), 'same-build steppers rewind in reverse document order');
await p.click('#at-prev');
check(await p.evaluate(() => fixtureSteppers[0].index() === 0), 'Back reaches the first managed step');

// Numerical presentation: unrounded calculations, approximate displayed arithmetic, fresh accessible footer labels.
check(await p.evaluate(() => {
  const mix = AT.ui.mixTable([1 / 3, 2 / 3], [[1], [4]], { decimals: 2 });
  const old = mix.footer.cells[0].getAttribute('aria-label'); mix.setAlpha([1, 0]);
  const label = mix.footer.cells[0].getAttribute('aria-label');
  return old.includes('≈') && old !== label && label.startsWith(mix.footer.cells[0].textContent + ':') && !label.includes('0.33');
}), 'weighted/mean footer accessible arithmetic refreshes after changing weights');
check(await p.evaluate(() => {
  const calc = AT.ui.matVecCalc([1 / 3], [[0.7]], { decimals: 2, wDecimals: 1 });
  return AT.productLine([1 / 3], [0.7], 2).includes('≈') && calc.lines[0].res.textContent.startsWith('≈') && Math.abs(calc.out[0] - 0.7 / 3) < 1e-12 && AT.objects.find(o => o.cls === 'd').tip.includes(') W_O');
}), 'rounded worksheets mark approximations while retaining exact row-vector calculations');

// A narrow dynamic viewport must keep controls and frame scrolling available, not shrink the content to fit.
await p.setViewportSize({ width: 390, height: 760 });
await p.waitForTimeout(150);
check(await p.evaluate(() => {
  const frame = document.querySelector('.frame.is-live').getBoundingClientRect(), controls = document.querySelector('#at-controls').getBoundingClientRect();
  return frame.height > 150 && frame.bottom <= controls.top + 1 && controls.bottom <= innerHeight + 1 && document.documentElement.scrollWidth <= innerWidth;
}), 'mobile frame reserves the dynamic-height classroom toolbar without page-wide overflow');
check(await p.evaluate(() => { const pre = document.querySelector('#nested-frame pre.pytorch'); return getComputedStyle(pre).overflowX === 'auto' && pre.getBoundingClientRect().width <= document.querySelector('#nested-frame').clientWidth; }), 'PyTorch snippets retain horizontal scrolling within their frame');
await p.screenshot({ path: path.join(outdir, 'mobile-controls.png') });
await b.close();
console.log('\nerrors: ' + errs.length); errs.forEach(e => console.log('  ' + e));
console.log('failures: ' + fails.length + (fails.length ? '\n  ' + fails.join('\n  ') : ''));
process.exit(fails.length || errs.length ? 1 : 0);
