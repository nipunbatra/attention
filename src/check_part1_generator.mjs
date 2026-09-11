// Live generator controls, replay, real-model probabilities, and responsive fit.
// node src/check_part1_generator.mjs [part1.html]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const candidates = [process.env.PLAYWRIGHT_PATH, process.env.PLAYWRIGHT_MODULE, 'playwright', 'playwright-core'].filter(Boolean);
const cache = path.join(os.homedir(), '.npm', '_npx');
if (fs.existsSync(cache)) for (const dir of fs.readdirSync(cache).sort()) candidates.push(path.join(cache, dir, 'node_modules/playwright'));
let pw;
for (const candidate of candidates) { try { pw = require(candidate); break; } catch {} }
if (!pw) throw new Error('Set PLAYWRIGHT_PATH to an existing Playwright installation.');
const browser = await pw.chromium.launch(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
  ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE } : {});
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, reducedMotion: 'reduce' });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(pathToFileURL(path.resolve(process.argv[2] || 'part1.html')).href);
  await page.evaluate(() => document.fonts.ready);
  const original = await page.evaluate(() => JSON.stringify(window.__TOY__));
  await page.evaluate(() => { AT.present.enter(); AT.present.go('s10', 10, 0); });
  const read = () => page.evaluate(() => ({
    context: [...document.querySelectorAll('#s10-window .chip-t')].map(x => x.textContent),
    name: document.querySelector('#s10-name').textContent,
    chosen: document.querySelector('#s10-chosen').textContent,
    bars: [...document.querySelectorAll('#s10-bars .bl')].map(el => ({ label: el.textContent, p: Number(el.nextElementSibling.nextElementSibling.textContent) })),
    status: document.querySelector('#s10-status').textContent,
    seed: document.querySelector('#s10-seed').value,
    nextDisabled: document.querySelector('#s10-next').disabled,
    allDisabled: document.querySelector('#s10-all').disabled
  }));
  const click = id => page.locator(`#s10-${id}`).click();
  const fill = (id, value) => page.locator(`#s10-${id}`).fill(String(value));
  const temperature = value => page.locator('#s10-temp input').evaluate((el, t) => { el.value = t; el.dispatchEvent(new Event('input', { bubbles: true })); }, String(value));
  const expected = (prefix, seed, temp = 1, greedy = false, maxLength = 18) => page.evaluate(opts => {
    const context = ('---' + opts.prefix.toLowerCase()).slice(-3).split('');
    return AT.mlp.generate({ ids: context, seed: opts.seed, temperature: opts.temp, greedy: opts.greedy, maxLength: opts.maxLength });
  }, { prefix, seed, temp, greedy, maxLength });
  function checkBars(actual, p) {
    const shown = actual.bars.filter(x => !x.label.startsWith('other'));
    const vocab = '-abcdefghijklmnopqrstuvwxyz';
    for (const row of shown) assert.ok(Math.abs(row.p - p[vocab.indexOf(row.label)]) <= .00501, `wrong probability for ${row.label}`);
    const rest = 1 - shown.reduce((sum, row) => sum + p[vocab.indexOf(row.label)], 0);
    assert.ok(Math.abs(actual.bars.at(-1).p - rest) <= .00501, 'wrong grouped probability mass');
    assert.equal(actual.bars.at(-1).label, 'other ×23');
  }
  async function fits(label) {
    const report = await page.evaluate(() => {
      const card = document.querySelector('#s10-next').closest('.card'), box = card.getBoundingClientRect();
      return {
        fit: AT.present.fitReport(),
        escapes: [...card.querySelectorAll('input,button,.bars,.gen-name,.window,.small')].filter(el => {
          const r = el.getBoundingClientRect();
          return r.width && (r.left < box.left - 1 || r.right > box.right + 1 || r.bottom > box.bottom + 1);
        }).map(el => el.id || el.className),
        scrolls: [...card.querySelectorAll('*')].filter(el => {
          const s = getComputedStyle(el);
          return el.clientHeight && /auto|scroll/.test(s.overflowY) && el.scrollHeight > el.clientHeight + 1;
        }).map(el => el.id || el.className)
      };
    });
    assert.equal(report.fit.overflow, false, `${label}: ${JSON.stringify(report.fit)}`);
    assert.deepEqual(report.escapes, [], `${label}: escaped card`);
    assert.deepEqual(report.scrolls, [], `${label}: internal scrollbar`);
  }
  assert.deepEqual((await read()).context, ['-', '-', '-']);
  await fits('default');
  await temperature(1); await fill('seed', 3);
  await click('all');
  const first = await read();
  assert.equal(first.name, 'sam'); assert.equal(first.chosen, 'END'); assert.equal(first.nextDisabled, true);
  assert.equal(first.seed, '3');
  await click('all'); assert.deepEqual(await read(), first, 'Generate must not silently change the seed');
  await click('reset');
  const run = await expected('', 3);
  for (let i = 0; i < run.trace.length; i++) {
    await click('next');
    const actual = await read(), last = run.trace[i];
    assert.deepEqual(actual.context, last.context);
    assert.equal(actual.chosen, last.chosen === '-' ? 'END' : last.chosen);
    assert.ok(actual.bars.some(row => row.label === last.chosen), 'chosen token must remain visible');
    checkBars(actual, last.probabilities);
  }
  assert.deepEqual(await read(), first, 'stepping must equal Generate');
  for (const prefix of ['a', 'sa', 'aab', 'nipu', 'AbHi', 'abcdefghijklmnop']) {
    await fill('prefix', prefix); await fill('seed', 12);
    const normalized = prefix.toLowerCase(), initial = ('---' + normalized).slice(-3).split('');
    assert.deepEqual((await read()).context, initial, `initial window for ${prefix}`);
    const preview = await page.evaluate(ctx => AT.mlp.distribution(ctx, 1).p, initial);
    checkBars(await read(), preview);
    await click('all');
    const exp = await expected(prefix, 12), actual = await read(), last = exp.trace.at(-1);
    assert.equal(actual.name, normalized + exp.name, 'keep the whole supplied prefix in the output');
    assert.deepEqual(actual.context, last.context); checkBars(actual, last.probabilities);
    await fits(`prefix ${prefix}`);
    await click('reset');
    assert.equal((await read()).name, normalized);
    assert.equal((await read()).seed, '12');
  }
  // Guard against sampler's implicit uint32 wrapping and unknown-token fallback.
  for (const bad of ['', '-1', '1.2', '1e3', '4294967296', 'hello']) {
    await fill('seed', bad);
    assert.equal(await page.locator('#s10-seed').getAttribute('aria-invalid'), 'true');
    assert.equal((await read()).allDisabled, true); assert.deepEqual((await read()).bars, []);
    await fits(`invalid seed ${bad}`);
  }
  for (const good of [0, 4294967295]) {
    await fill('seed', good); await click('all');
    assert.equal((await read()).name, 'abcdefghijklmnop' + (await expected('abcdefghijklmnop', good)).name);
  }
  for (const bad of ['sa2', 'सा', '--s', 'ab cd', '<script>', 'abcdefghijklmnopq']) {
    await fill('prefix', bad);
    assert.equal(await page.locator('#s10-prefix').getAttribute('aria-invalid'), 'true');
    assert.equal((await read()).nextDisabled, true); await fits(`invalid prefix ${bad}`);
  }
  await fill('prefix', 'sa'); await fill('seed', 3);
  const preview = (await read()).bars;
  await click('new-seed');
  const newSeed = Number((await read()).seed);
  assert.ok(Number.isInteger(newSeed) && newSeed >= 0 && newSeed <= 4294967295 && newSeed !== 3);
  assert.deepEqual((await read()).bars, preview, 'seed affects draws, never probabilities');
  assert.equal((await read()).name, 'sa', 'New seed resets instead of silently generating');
  await click('greedy'); await click('all'); const greedy = (await read()).name;
  await fill('seed', 99); await temperature(.5); await click('all');
  assert.equal((await read()).name, greedy, 'greedy must ignore seed and positive temperature');
  await fits('greedy note');
  // Native typing must not activate slide shortcuts. Configuration survives navigation.
  await fill('prefix', ''); await page.locator('#s10-prefix').pressSequentially('sam');
  assert.equal(await page.evaluate(() => location.hash), '#s10/10/0');
  await page.locator('#s10-prefix').press('Enter'); const entered = await read();
  await page.evaluate(() => { AT.present.go('s10', 11, 0); AT.present.go('s10', 10, 0); });
  assert.deepEqual(await read(), entered); assert.equal(await page.locator('#s10-temp input').inputValue(), '0.5');
  await page.locator('#s10-seed').focus(); await page.keyboard.press('Escape');
  assert.equal(await page.evaluate(() => document.body.classList.contains('present')), true);
  for (const id of ['prefix', 'seed']) assert.ok(await page.locator(`label[for=s10-${id}]`).textContent());
  // Exercise the longest possible output, including an 18-call cap, not just short samples.
  await click('sample'); await temperature(1.5); await fill('prefix', 'abcdefghijklmnop');
  const cappedSeed = await page.evaluate(() => {
    for (let seed = 0; seed < 1000; seed++) {
      const r = AT.mlp.generate({ ids: ['n', 'o', 'p'], temperature: 1.5, seed, maxLength: 18 });
      if (r.name.length === 18) return seed;
    }
    throw new Error('No capped fixture found');
  });
  await fill('seed', cappedSeed); await click('all');
  assert.equal((await read()).name.length, 34); assert.equal((await read()).nextDisabled, true);
  assert.match((await read()).status, /safety limit/); await fits('34-letter capped output');
  await page.screenshot({ path: '/tmp/part1-generator-long.png' });
  await page.setViewportSize({ width: 993, height: 1041 }); await fits('portrait presentation');
  await page.screenshot({ path: '/tmp/part1-generator-portrait.png' });
  await page.evaluate(() => AT.present.exit()); await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await page.locator('#s10-prefix').scrollIntoViewIfNeeded();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'phone article width');
  assert.ok(await page.locator('#s10-prefix').isVisible());
  const card = page.locator('#s10-next').locator('xpath=ancestor::div[contains(@class,"card")]');
  assert.ok(await card.evaluate(el => el.scrollWidth <= el.clientWidth + 1), 'phone generator card width');
  await card.screenshot({ path: '/tmp/part1-generator-phone.png' });
  await page.emulateMedia({ media: 'print' });
  assert.equal(await page.locator('#s10-prefix').inputValue(), 'abcdefghijklmnop');
  assert.equal((await read()).name.length, 34, 'print retains current output');
  assert.equal(await page.evaluate(() => JSON.stringify(window.__TOY__)), original, 'controls must not mutate model');
  assert.deepEqual(errors, []);
  console.log('Generator PASS: exact seed replay, prefix padding/preservation, model probabilities, errors, stop/cap, keyboard, navigation, slide/phone/print fit.');
} finally { await browser.close(); }
