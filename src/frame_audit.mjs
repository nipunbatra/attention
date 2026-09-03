// Audit every presentation state for the slide-first layout contract.
// Usage: node src/frame_audit.mjs page.html [--width 1280] [--height 720] [--shots DIR]
//
// A valid classroom frame:
//   - is authored on a logical 1280x720 stage,
//   - never overflows that stage or its frame body,
//   - never introduces a nested vertical scrollbar,
//   - remains valid at every build and stepper state.
//
// Reading mode is intentionally not constrained by this audit; the same frames
// unfold into the long-form article there.
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
  try { pw = require(candidate); break; } catch { /* try the next existing runtime */ }
}
if (!pw) throw new Error('No existing Playwright runtime found. Set PLAYWRIGHT_MODULE; this audit installs nothing.');

const args = process.argv.slice(2);
const file = args[0];
if (!file) throw new Error('Usage: node src/frame_audit.mjs page.html [--width 1280] [--height 720] [--shots DIR]');
const value = (flag, fallback) => {
  const i = args.indexOf(flag);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const width = Number(value('--width', 1280));
const height = Number(value('--height', 720));
const shots = value('--shots', '');
if (shots) fs.mkdirSync(shots, { recursive: true });

const errors = [];
const browser = await pw.chromium.launch(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
  ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
  : {});
const context = await browser.newContext({ viewport: { width, height } });
const page = await context.newPage();
page.on('pageerror', error => errors.push('PAGEERROR: ' + String(error.message || error).split('\n')[0]));
page.on('console', message => {
  if (message.type() === 'error') errors.push('CONSOLE: ' + message.text().slice(0, 240));
});

const url = new URL(pathToFileURL(path.resolve(file)).href);
url.searchParams.set('present', '');
await page.goto(url.href, { waitUntil: 'load' });
await page.waitForTimeout(500);

// The runtime preflight samples full builds, every managed stepper state, and
// native reveal panels opened. The state walk below independently checks live
// navigation, fit warnings, stage geometry, and nested scroll containers.
const preflight = await page.evaluate(() => AT.present.preflight());

const failures = [];
const visited = new Set();
let completed = false;
for (let guard = 0; guard < 5000; guard++) {
  const report = await page.evaluate(() => {
    const state = AT.present.state();
    const section = document.querySelector('.sec.is-live');
    const frame = document.querySelector('.frame.is-live');
    const stepState = frame ? Array.from(frame.querySelectorAll('.stepper')).filter(el => el.stepperApi)
      .map((el, i) => i + ':' + el.stepperApi.index()).join(',') : '';
    const key = location.hash + (stepState ? ':steps[' + stepState + ']' : '');
    const over = (el, axis) => axis === 'x'
      ? el.scrollWidth > el.clientWidth + 2
      : el.scrollHeight > el.clientHeight + 2;
    const nested = frame ? Array.from(frame.querySelectorAll('*')).flatMap(el => {
      if (el.closest('[data-present-scroll="allow"]')) return [];
      const css = getComputedStyle(el);
      if (!el.getClientRects().length || css.visibility === 'hidden') return [];
      const horizontal = /(auto|scroll)/.test(css.overflowX) && over(el, 'x');
      const vertical = /(auto|scroll)/.test(css.overflowY) && over(el, 'y');
      return horizontal || vertical ? [{ tag: el.tagName.toLowerCase(), id: el.id || '', cls: String(el.className || '').slice(0, 100), horizontal, vertical }] : [];
    }) : [];
    const sectionBox = section ? { width: section.offsetWidth, height: section.offsetHeight } : null;
    const warning = document.querySelector('#at-fit-warning, #at-overflow-warning, [data-fit-warning]');
    return {
      key,
      hash: location.hash,
      frameId: frame && (frame.id || frame.getAttribute('data-title')),
      total: state.total,
      final: state.fi === state.total - 1 && state.build === state.frame.maxBuild && (!state.stepper || state.stepper.index === state.stepper.count - 1),
      frameOverflowX: !!frame && over(frame, 'x'),
      frameOverflowY: !!frame && over(frame, 'y'),
      stageOverflowX: !!section && over(section, 'x'),
      stageOverflowY: !!section && over(section, 'y'),
      nested,
      sectionBox,
      bodyFont: parseFloat(getComputedStyle(document.body).fontSize),
      fitWarning: !!warning && getComputedStyle(warning).display !== 'none'
    };
  });
  if (visited.has(report.key)) {
    errors.push('Navigation repeated ' + report.key + ' before reaching the final frame.');
    break;
  }
  visited.add(report.key);

  const stageContract = !report.sectionBox || Math.abs(report.sectionBox.width - 1280) > 1 || Math.abs(report.sectionBox.height - 720) > 1;
  const bad = report.frameOverflowX || report.frameOverflowY || report.stageOverflowX || report.stageOverflowY || report.fitWarning || stageContract || report.nested.length;
  if (bad) {
    failures.push({
      state: report.key,
      frame: report.frameId,
      frameOverflow: [report.frameOverflowX, report.frameOverflowY],
      stageOverflow: [report.stageOverflowX, report.stageOverflowY],
      fitWarning: report.fitWarning,
      stageSize: report.sectionBox,
      stageContract,
      nestedScroll: report.nested
    });
  }
  if (shots && bad) {
    const name = report.key.replace(/[^a-z0-9_-]+/gi, '_').replace(/^_+/, '') + '.png';
    await page.screenshot({ path: path.join(shots, name) });
  }
  if (report.final) { completed = true; break; }
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(30);
}

if (!completed) errors.push('Navigation did not reach the final frame within the audit walk.');

await browser.close();
const result = {
  file: path.resolve(file),
  viewport: [width, height],
  states: visited.size,
  errors,
  preflightOverflow: preflight.overflow,
  overflowFailures: failures
};
console.log(JSON.stringify(result, null, 2));
process.exit(errors.length || preflight.overflow.length || failures.length ? 1 : 0);
