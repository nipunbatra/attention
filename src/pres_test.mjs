// Present-mode test. Usage: node pres_test.mjs [test00.html] [outdir]
// Opens the page with ?present, walks every build with ArrowRight (screenshots into outdir), and checks:
// hash sync, hidden builds before their turn, stepper capture, overview, blank, notes, print preparation, Esc to read mode,
// the presenter window (postMessage both ways). Exits 1 on any failure or any page/console error.
import { createRequire } from 'module'; import path from 'path'; import fs from 'fs';
const require = createRequire(import.meta.url);
const pw = require('/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const file = process.argv[2] || 'test00.html';
const outdir = process.argv[3] || 'frames00';
fs.mkdirSync(outdir, { recursive: true });
const fails = [], errs = [];
const check = (cond, msg) => { if (!cond) fails.push(msg); console.log((cond ? 'ok   ' : 'FAIL ') + msg); };
const b = await pw.chromium.launch();
const ctx = await b.newContext({ viewport: { width: 1280, height: 720 } });
const hook = (p, tag) => { p.on('pageerror', e => errs.push(tag + ' PAGEERROR: ' + String(e.message || e).split('\n')[0])); p.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errs.push(tag + ' CONSOLE ' + m.type() + ': ' + m.text().slice(0, 200)); }); };
const p = await ctx.newPage(); hook(p, 'main');
await p.goto('file://' + path.resolve(file) + '?present', { waitUntil: 'load' });
await p.waitForTimeout(700);
const st = async () => p.evaluate(() => { const s = AT.present.state(); return { active: s.active, fi: s.fi, build: s.build, hash: location.hash, stepper: s.stepper, total: s.total, max: s.frame && s.frame.maxBuild }; });
const vis = async sel => p.evaluate(sel => { const e = document.querySelector(sel); if (!e) return null; const cs = getComputedStyle(e); return cs.visibility !== 'hidden' && cs.display !== 'none' && +cs.opacity > 0.01; }, sel);
let s = await st();
check(s.active && await p.evaluate(() => document.body.classList.contains('present')), 'present mode entered from ?present');
check(s.hash === '#s00/1/0', 'hash is #s00/1/0 on entry (got ' + s.hash + ')');
check(await vis('#s00-f1-chips'), 'build 0 (chips) visible at once');
check((await vis('#s00-frame1 .card[data-build="1"]')) === false, 'build 1 hidden before its turn');
check((await vis('#s00-f1-stepper')) === false, 'build 2 (stepper) hidden before its turn');
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
// overview
await p.keyboard.press('o'); await p.waitForTimeout(200);
check(await p.evaluate(() => document.querySelector('#at-overview').classList.contains('is-on') && document.querySelectorAll('#at-overview .ov-item').length === 2), 'O opens the overview with 2 frames');
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
check(await vis('#s00-frame2 p.muted') && await vis('#s00-frame1 .callout[data-build="3"]') && await p.evaluate(() => getComputedStyle(document.querySelector('.companion')).display !== 'none'), 'read mode shows everything, companion included');
await p.screenshot({ path: path.join(outdir, 'read.png') });
// P re-enters at the section in view
await p.keyboard.press('p'); await p.waitForTimeout(300); s = await st();
check(s.active && s.hash === '#s00/1/0', 'P enters present mode at the section in view');
await p.keyboard.press('Escape');
await b.close();
console.log('\nerrors: ' + errs.length); errs.forEach(e => console.log('  ' + e));
console.log('failures: ' + fails.length + (fails.length ? '\n  ' + fails.join('\n  ') : ''));
process.exit(fails.length || errs.length ? 1 : 0);
