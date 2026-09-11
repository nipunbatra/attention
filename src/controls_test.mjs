// Real pointer, keyboard and touch regression checks for the presentation tray.
// node src/controls_test.mjs [part1.html] [screenshot-directory]
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
const require = createRequire(import.meta.url);
const candidates = [process.env.PLAYWRIGHT_MODULE, 'playwright', 'playwright-core'].filter(Boolean);
const cache = path.join(os.homedir(), '.npm', '_npx');
if (fs.existsSync(cache)) for (const dir of fs.readdirSync(cache)) candidates.push(path.join(cache, dir, 'node_modules/playwright'));
let pw;
for (const candidate of candidates) { try { pw = require(candidate); break; } catch {} }
if (!pw) throw new Error('Set PLAYWRIGHT_MODULE to an existing Playwright installation.');
const url = pathToFileURL(path.resolve(process.argv[2] || 'part1.html')).href + '?present#s00/1/0';
const out = process.argv[3] || fs.mkdtempSync(path.join(os.tmpdir(), 'attention-controls-'));
fs.mkdirSync(out, { recursive: true });
const browser = await pw.chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const failures = [], errors = [];
const hook = p => { p.on('pageerror', e => errors.push(e.message)); p.on('console', m => { if (m.type() === 'error') errors.push(m.text()); }); };
hook(page);
const check = (ok, label) => { console.log((ok ? 'ok   ' : 'FAIL ') + label); if (!ok) failures.push(label); };
const geometry = p => p.locator('.sec.is-live').boundingBox();
const idle = p => p.evaluate(() => document.body.classList.contains('present-ui-idle') &&
  getComputedStyle(document.querySelector('#at-controls-toggle')).opacity === '0' && document.querySelector('#at-controls').inert);
const pause = p => p.waitForTimeout(3350);
await page.goto(url);
await page.waitForTimeout(250);
const before = await geometry(page);
check(!(await page.locator('#at-controls').isVisible()), 'tray starts closed');
check(await page.evaluate(() => {
  const cover = document.querySelector('.sec-cover');
  return cover && !cover.querySelector('.katex, [data-build]') && !/minutes|by hand/.test(cover.innerText);
}), 'cover has no opening equations, extra reveals, duration, or by-hand promise');
await page.mouse.move(400, 200);
await pause(page);
check(await idle(page), 'idle presentation hides the tray and its affordance');
await page.mouse.move(600, 250);
check(await idle(page), 'pointing at slide content does not reveal navigation');
await page.screenshot({ path: path.join(out, 'cover-idle.png') });
await page.mouse.move(1200, 680);
await page.waitForTimeout(200);
check(!(await idle(page)) && !(await page.locator('#at-controls').isVisible()), 'bottom-right pointer movement reveals only the small affordance');
await page.click('#at-controls-toggle');
check(await page.locator('#at-controls').isVisible(), 'pointer opens the tray');
check((await page.locator('#at-controls').boundingBox()).height <= 64, 'desktop controls use one compact row');
check(JSON.stringify(before) === JSON.stringify(await geometry(page)), 'controls never move or resize the frame');
await page.locator('#at-next').hover();
await pause(page);
check(await page.locator('#at-controls').isVisible(), 'hovered controls remain available');
await page.screenshot({ path: path.join(out, 'controls-open.png') });
await page.click('#at-next');
await page.mouse.move(400, 200);
await pause(page);
check(await idle(page), 'pointer leaves: clicked controls hide after three seconds');
check(await page.evaluate(() => document.activeElement === document.querySelector('.frame.is-live')), 'idle close returns stale pointer focus to the slide');
await page.keyboard.press('c');
check(await page.locator('#at-controls').isVisible() && await page.evaluate(() => document.activeElement.id === 'at-next'), 'C opens controls and focuses Next');
await pause(page);
check(await page.locator('#at-controls').isVisible(), 'keyboard-focused controls never time out');
await page.keyboard.press('Tab');
check(await page.evaluate(() => document.activeElement.id === 'at-overview-btn'), 'Tab moves through controls');
await page.keyboard.press('Enter');
await pause(page);
check(await page.evaluate(() => document.querySelector('#at-overview').classList.contains('is-on') && document.querySelector('main').inert), 'idle timer leaves the overview focus trap intact');
await page.keyboard.press('Escape');
check(await page.evaluate(() => document.activeElement.id === 'at-overview-btn'), 'overview restores the focused control');
await page.keyboard.press('Escape');
check(await page.evaluate(() => AT.present.isActive()) && !(await page.locator('#at-controls').isVisible()), 'Escape closes navigation before leaving the lesson');
await page.keyboard.press('c');
await page.mouse.click(300, 200);
check(!(await page.locator('#at-controls').isVisible()), 'clicking the slide dismisses navigation');

for (const [width, height] of [[1280, 720], [2048, 1011], [3650, 1802], [1024, 768], [390, 844]]) {
  await page.setViewportSize({ width, height });
  await page.waitForTimeout(200);
  await page.evaluate(() => { AT.present.first(); });
  const fit = await page.evaluate(() => {
    const stage = document.querySelector('.sec.is-live').getBoundingClientRect();
    return { ratio: stage.width / stage.height, inside: stage.left >= 0 && stage.top >= 0 && stage.right <= innerWidth && stage.bottom <= innerHeight,
      noScroll: document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight,
      body: getComputedStyle(document.body).fontSize, report: AT.present.fitReport() };
  });
  check(Math.abs(fit.ratio - 16 / 9) < .001 && fit.inside && fit.noScroll && fit.body === '28px' && !fit.report.overflow, `${width}×${height}: complete 16:9 frame, unchanged type, no page scroll`);
  if (width === 2048) await page.screenshot({ path: path.join(out, 'wide-cover.png') });
}

const touch = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
hook(touch);
await touch.goto(url);
await pause(touch);
await touch.touchscreen.tap(160, 320);
await touch.waitForTimeout(200);
check(!(await idle(touch)) && !(await touch.locator('#at-controls').isVisible()), 'touch on slide restores the affordance without opening an overlay');
await touch.locator('#at-controls-toggle').tap();
const tray = await touch.locator('#at-controls').boundingBox();
check(await touch.locator('#at-controls').isVisible() && tray.x >= 0 && tray.x + tray.width <= 390 && tray.y + tray.height <= 844, 'touch navigation fits the phone viewport');
await touch.screenshot({ path: path.join(out, 'touch-controls.png') });
await pause(touch);
check(await idle(touch), 'touch controls auto-hide despite the last button retaining click focus');
await browser.close();
console.log(JSON.stringify({ screenshots: out, failures, errors }, null, 2));
if (failures.length || errors.length) process.exitCode = 1;
