// Exercise the states a static frame walk cannot see: focused controls, changing
// masks, live arithmetic dialogs, and the optional presenter-notes strip.
// node src/interaction_test.mjs [output-directory]
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
const src = path.dirname(fileURLToPath(import.meta.url));
const out = process.argv[2] || fs.mkdtempSync(path.join(os.tmpdir(), 'attention-interactions-'));
fs.mkdirSync(out, { recursive: true });
const require = createRequire(import.meta.url);
const candidates = [process.env.PLAYWRIGHT_MODULE, 'playwright', 'playwright-core'].filter(Boolean);
const cache = path.join(os.homedir(), '.npm', '_npx');
if (fs.existsSync(cache)) for (const dir of fs.readdirSync(cache)) candidates.push(path.join(cache, dir, 'node_modules/playwright'));
let pw;
for (const candidate of candidates) { try { pw = require(candidate); break; } catch {} }
if (!pw) throw new Error('Set PLAYWRIGHT_MODULE to an existing Playwright installation.');
const files = {};
for (const part of [2, 3, 6, 7]) {
  files[part] = path.join(out, `part${part}.html`);
  execFileSync('python3', [path.join(src, 'assemble.py'), '--part', String(part), '--out', files[part]], { stdio: 'pipe' });
}
const failures = [], errors = [], checks = [];
const check = (ok, label, evidence) => { checks.push({ ok, label, ...(evidence ? { evidence } : {}) }); if (!ok) failures.push(label); };
const browser = await pw.chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
page.on('pageerror', e => errors.push(e.message));
page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errors.push(m.type() + ': ' + m.text()); });
const load = async (part, suffix = '') => { await page.goto(pathToFileURL(files[part]).href + suffix); await page.waitForTimeout(180); };
const state = () => page.evaluate(() => { const s = AT.present.state(); return [s.fi, s.build, s.stepper?.index]; });
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// A real search-score worksheet near the slide edge. At each projector scale,
// click both its leftmost and rightmost arithmetic cells and inspect clipping.
for (const [width, height] of [[1280, 720], [1920, 1080], [2560, 1440]]) {
  await page.setViewportSize({ width, height });
  await load(2, '?present#s05/4/0');
  await page.evaluate(() => AT.present.setBuild(AT.present.state().frame.maxBuild));
  const cells = page.locator('.frame.is-live [aria-haspopup="dialog"], .frame.is-live td.has-calc');
  const n = await cells.count();
  check(n > 0, `arithmetic controls exist at ${height}p`);
  for (const i of [...new Set([0, Math.max(0, n - 1)])]) {
    await cells.nth(i).click(); await page.waitForTimeout(120);
    const bounds = await page.evaluate(() => {
      const pop = [...document.querySelectorAll('.calc-pop')].find(x => !x.hidden);
      if (!pop) return { visible: false };
      const p = pop.getBoundingClientRect(), f = pop.closest('.frame').getBoundingClientRect();
      return { visible: true, contained: p.left >= f.left - 1 && p.right <= f.right + 1 && p.top >= f.top - 1 && p.bottom <= f.bottom + 1,
        warning: document.querySelector('#at-fit-warning').textContent, noScroll: pop.scrollHeight <= pop.clientHeight + 1 && pop.scrollWidth <= pop.clientWidth + 1,
        pop: [p.left, p.top, p.right, p.bottom], frame: [f.left, f.top, f.right, f.bottom] };
    });
    check(bounds.visible && bounds.contained && !bounds.warning && bounds.noScroll, `${height}p arithmetic cell ${i} stays inside its frame`, bounds);
    await page.screenshot({ path: path.join(out, `popover-${height}-${i}.png`) });
    await page.keyboard.press('Escape');
    check(await page.evaluate(() => AT.present.isActive()), 'Escape closes the arithmetic dialog without exiting');
  }
  await page.evaluate(() => AT.present.notes(true)); await page.waitForTimeout(180);
  const notes = await page.evaluate(() => {
    const stage = document.querySelector('.sec.is-live').getBoundingClientRect(), note = document.querySelector('#at-notes').getBoundingClientRect();
    return { separate: stage.bottom <= note.top + 1, notesAtBottom: Math.abs(note.bottom - innerHeight) < 1, ratio: stage.width / stage.height,
      warning: document.querySelector('#at-fit-warning').textContent };
  });
  check(notes.separate && notes.notesAtBottom && Math.abs(notes.ratio - 16 / 9) < .001 && !notes.warning, `${height}p notes reserve space and retain 16:9`, notes);
  await page.screenshot({ path: path.join(out, `notes-${height}.png`) });
  await page.evaluate(() => AT.present.notes(false));
}

await page.evaluate(() => {
  const frame = AT.present.frames()[AT.present.state().fi];
  frame.notes = ['Keep the target fixed. Change one score and ask what changed in the output.'];
  AT.present.notes(true);
});
check(await page.evaluate(() => getComputedStyle(document.querySelector('#at-notes p')).fontWeight === '400'), 'a one-line multi-sentence note is not all bold');
await page.evaluate(() => { AT.present.notes(false); AT.present.overview(true); });
check(await page.evaluate(() => {
  const state = AT.present.state(), f = state.frame;
  return document.querySelector('.ov-item.is-current .ov-t').textContent === (f.title && f.title !== f.secTitle ? f.title : f.secTitle);
}), 'overview prominently identifies the distinct projected frame');
await page.keyboard.press('Escape');

// Normal ranges keep native keyboard control; N is the non-conflicting slide
// shortcut. Escape explicitly returns focus, unlike blur-on-change hacks.
await page.setViewportSize({ width: 1280, height: 720 }); await load(2);
await page.evaluate(() => {
  const range = document.querySelector('main input[type="range"]:not([data-present])');
  const frame = range.closest('.frame'), section = frame.closest('.sec');
  const index = [...section.querySelectorAll(':scope > .frame')].indexOf(frame) + 1;
  AT.present.enter({ id: section.id, f: index, b: 999 });
  range.value = String((Number(range.min) + Number(range.max)) / 2);
  range.dispatchEvent(new Event('input', { bubbles: true })); range.focus();
});
const initialState = await state();
const oldValue = await page.evaluate(() => document.activeElement.value);
await page.keyboard.press('ArrowRight');
check(same(initialState, await state()) && await page.evaluate(v => document.activeElement.value !== v, oldValue), 'focused range ArrowRight changes its value, not the slide');
await page.keyboard.press('PageDown');
check(same(initialState, await state()), 'focused range PageDown remains native');
await page.keyboard.press('n');
check(!same(initialState, await state()), 'N advances after a normal range interaction');
await load(6, '?present#s03/2/0');
const manualRange = page.locator('#s03-slider input'); await manualRange.focus(); const manualState = await state();
await page.keyboard.press('n'); check(same(manualState, await state()), 'manual ranges opt out of slide shortcuts');
await page.keyboard.press('Escape');
check(await page.evaluate(() => AT.present.isActive() && document.activeElement.classList.contains('frame')), 'Escape returns focus from a manual range to the slide');
await page.keyboard.press('n'); check(!same(manualState, await state()), 'N navigates normally after returning focus');

// Reconstruction mask presets change row count; every redraw must show exactly
// one current table. Check displayed totals against the live model each time.
await load(6, '?present#s03/2/0');
for (const mask of ['0', '1', '2', '0']) for (const guess of [0, 1, 2]) {
  await page.selectOption('#s03-mask', mask);
  await page.locator('#s03-slider input').evaluate((r, value) => { r.value = String(value); r.dispatchEvent(new Event('input', { bubbles: true })); }, guess);
  await page.waitForTimeout(70);
  const evidence = await page.evaluate(({ mask, guess }) => {
    const T = AT.visionSSL, F = T.reconstruction(T.data.maskPresets[Number(mask)].indices, guess), host = document.querySelector('#s03-lab-table');
    return { tables: host.querySelectorAll('table').length, rows: host.querySelectorAll('tbody tr').length,
      expectedRows: F.rows.filter(r => r.hidden).length, lossMatches: document.querySelector('#s03-lab-loss').textContent.includes(F.loss.toFixed(4)), warning: document.querySelector('#at-fit-warning').textContent };
  }, { mask, guess });
  check(evidence.tables === 1 && evidence.rows === evidence.expectedRows && evidence.lossMatches && !evidence.warning, `mask ${mask}, guess ${guess}: one consistent table`, evidence);
}
await page.screenshot({ path: path.join(out, 'mask-after-interaction.png') });
await load(6, '?present#s05/3/0');
for (const value of [0, .75, 3, 1.5, 0]) {
  await page.locator('#s05-slider input').evaluate((r, value) => { r.value = String(value); r.dispatchEvent(new Event('input', { bubbles: true })); }, value);
  await page.waitForTimeout(70);
  const evidence = await page.evaluate(value => {
    const R = AT.visionSSL.dino(value), host = document.querySelector('#s05-ce-table');
    return { tables: host.querySelectorAll('table').length, rows: host.querySelectorAll('tbody tr').length,
      lossMatches: document.querySelector('#s05-ce-loss').textContent.includes(R.loss.toFixed(4)), warning: document.querySelector('#at-fit-warning').textContent };
  }, value);
  check(evidence.tables === 1 && evidence.rows === 3 && evidence.lossMatches && !evidence.warning, `student logit ${value}: table agrees with loss`, evidence);
}

// A recap can light all chips without making their navigation destinations
// identical. Explicit targets must not change the cumulative lighting rule.
await load(3);
check(await page.evaluate(() => {
  const targets = window.__PART__.objectSections;
  return targets && Object.entries(targets).every(([key, id]) => document.querySelector('.nchip[data-obj="' + key + '"]').dataset.target === id)
    && window.__PART__.sections[0].lit.split(/\s+/).length === 7;
}), 'Part 3 chip destinations respect objectSections while keeping recap lighting');
await page.locator('.nchip[data-obj="q"]').click();
await page.waitForFunction(() => Math.abs(document.getElementById(window.__PART__.objectSections.q).getBoundingClientRect().top) < 200);
check(await page.evaluate(() => !AT.present.isActive() && Math.abs(document.getElementById(window.__PART__.objectSections.q).getBoundingClientRect().top) < 200), 'query chip navigates to the projection section in reading mode');

// Wide notation has its own grid row. Formulas survive small screens without
// forcing a horizontal body scroll. TeX-labelled objects appear in both places.
for (const width of [390, 1280]) {
  await page.setViewportSize({ width, height: 850 }); await load(7);
  await page.locator('#notation summary').click();
  const card = await page.evaluate(() => ({ directChild: document.querySelector('#notation').parentElement.matches('.hero > .wrap'),
    columns: getComputedStyle(document.querySelector('#notation .notation-card')).gridTemplateColumns.split(' ').length,
    overflow: document.documentElement.scrollWidth > innerWidth + 1,
    texChips: document.querySelectorAll('#strip-chips .katex').length, texLegend: document.querySelectorAll('#hero-legend .katex').length }));
  check(card.directChild && card.columns === 1 && !card.overflow && card.texChips > 0 && card.texLegend > 0, `${width}px hero notation and mathematical chip labels`, card);
}
const katexErrors = await page.locator('.katex-error').count(); check(!katexErrors, 'no rendered math errors');
const contrasts = await page.evaluate(() => {
  const css = getComputedStyle(document.documentElement);
  function luminance(hex) {
    const rgb = hex.trim().slice(1).match(/.{2}/g).map(x => parseInt(x, 16) / 255).map(v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4);
    return rgb[0] * .2126 + rgb[1] * .7152 + rgb[2] * .0722;
  }
  function ratio(fg, bg) { return (luminance(bg) + .05) / (luminance(fg) + .05); }
  const pairs = ['e', 'q', 'k', 'v', 'a', 'd'].map(key => [key, css.getPropertyValue('--c-' + key), css.getPropertyValue('--t-' + key)]);
  pairs.push(['muted', css.getPropertyValue('--ink-3'), css.getPropertyValue('--paper')]);
  return pairs.map(([role, fg, bg]) => ({ role, ratio: ratio(fg, bg) }));
});
check(contrasts.every(x => x.ratio >= 4.5), 'small semantic labels meet AA on their role tints and paper', contrasts);
await browser.close();
const result = { checks: checks.length, failures, errors, output: out, evidence: checks };
fs.writeFileSync(path.join(out, 'interaction-report.json'), JSON.stringify(result, null, 2));
console.log(JSON.stringify({ checks: checks.length, failures, errors, output: out }, null, 2));
if (failures.length || errors.length) process.exitCode = 1;
