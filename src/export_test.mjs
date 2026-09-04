// Regression test for answer visibility in slide-PDF exports.
// Usage: node src/export_test.mjs [outdir]
// Builds a two-frame fixture from the shared source, then runs the real exporter.
// Retains its PDFs and PNGs for visual inspection; no article is assembled or edited.
import { createRequire } from 'module';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import { execFile } from 'child_process';
import { promisify } from 'util';

const require = createRequire(import.meta.url);
const candidates = [process.env.PLAYWRIGHT_MODULE, 'playwright', 'playwright-core'].filter(Boolean);
const npmCache = path.join(os.homedir(), '.npm', '_npx');
if (fs.existsSync(npmCache)) {
  for (const dir of fs.readdirSync(npmCache).sort()) candidates.push(path.join(npmCache, dir, 'node_modules', 'playwright'));
}
let pw;
for (const candidate of candidates) {
  try { pw = require(candidate); break; } catch { /* try another installed runtime */ }
}
if (!pw) throw new Error('No existing Playwright runtime found. Set PLAYWRIGHT_MODULE; this test installs nothing.');

const src = path.dirname(fileURLToPath(import.meta.url));
const outdir = process.argv[2] ? path.resolve(process.argv[2]) : fs.mkdtempSync(path.join(os.tmpdir(), 'attention-export-test-'));
fs.mkdirSync(outdir, { recursive: true });
const read = name => fs.readFileSync(path.join(src, name), 'utf8');
const safeJSON = value => JSON.stringify(value).replace(/<\//g, '<\\/');
const part = JSON.parse(read('part2.json'));
part.title = 'PDF answer regression';
part.subtitle = 'Two frames for checking the export policy.';
part.central = '';
part.chain = [];
part.sections = [{ id: 's00', title: 'Export fixture', lit: '' }];
part.objects = [];
part.prev = null;
part.next = null;
part.index = null;

// Distinct answer backgrounds let us test the actual PNG contents, independently
// of the exporter's reported answer count. A closed details element paints none
// of its answer background. Neither color appears elsewhere in the fixture.
const sections = `<section id="s00" class="sec" data-title="Export fixture" data-lit="">
<style>
#s00 .export-answer{padding:18px;color:#fff;font-size:24px;line-height:1.4}
#s00 .export-answer-step{background:rgb(19,143,83)}
#s00 .export-answer-quiz{background:rgb(173,37,151)}
</style>
<header class="sec-head"><span class="sec-num">00</span><div><h2>Export fixture</h2></div></header>
<div class="frame" id="export-step-frame" data-title="An answer created by a stepper" data-autobuild="off">
  <p>The answer is created only when the stepper reaches its last step.</p>
  <div id="export-stepper"></div>
</div>
<div class="frame" id="export-quiz-frame" data-title="A closed quiz answer" data-autobuild="off">
  <div id="export-quiz"></div>
  <p data-build="1">At this final build, a PDF reader should be able to read the answer.</p>
</div>
<script>
(function(){
  AT.ui.stepper({el:document.getElementById('export-stepper'),hideList:true,scrollList:false,steps:[
    {title:'Work it out',render:function(host){host.appendChild(AT.h('p',{},'What is 2 + 3? Think before advancing.'));}},
    {title:'Check the answer',render:function(host){AT.ui.reveal('What is 2 + 3?','<div class="export-answer export-answer-step">The stepper answer is 5.</div>',{open:false,into:host});}}
  ]});
  AT.ui.reveal('How many coordinates are in (1, 2, 3)?','<div class="export-answer export-answer-quiz">The quiz answer is 3 coordinates.</div>',{open:false,into:document.getElementById('export-quiz')});
})();
</script>
</section>`;
const shared = '<script>window.__TOY__=' + safeJSON(JSON.parse(read('toy.json'))) + ';window.__PART__=' + safeJSON(part) + ';</script><script>' + read('shared.js') + '</script>';
const html = read('shell.html')
  .replace('<!--KATEX-->', () => read('katex-bundle.html'))
  .replace('<!--SHARED-->', () => shared)
  .replace('<!--SECTIONS-->', () => sections);
const fixture = path.join(outdir, 'export-fixture.html');
fs.writeFileSync(fixture, html);

const cases = [
  { name: 'default-final', args: [], mode: 'final', answerMode: 'show', revealed: 2, expected: [[true, false], [false, true]] },
  { name: 'show-all-builds', args: ['--builds', 'all'], mode: 'all', answerMode: 'show', revealed: 2, expected: [[false, false], [true, false], [false, false], [false, true]] },
  { name: 'authored-final', args: ['--answers', 'authored'], mode: 'final', answerMode: 'authored', revealed: 0, expected: [[false, false], [false, false]] },
  { name: 'authored-all-builds', args: ['--builds', 'all', '--answers', 'authored'], mode: 'all', answerMode: 'authored', revealed: 0, expected: [[false, false], [false, false], [false, false], [false, false]] }
];
const failures = [];
const results = [];
const check = (condition, message) => {
  if (!condition) failures.push(message);
  console.log((condition ? 'ok   ' : 'FAIL ') + message);
};
const run = promisify(execFile);
console.log('Export fixture and screenshots: ' + outdir);
for (const item of cases) {
  const caseDir = path.join(outdir, item.name);
  fs.mkdirSync(caseDir, { recursive: true });
  const pdf = path.join(caseDir, 'slides.pdf');
  const frames = path.join(caseDir, 'frames');
  const args = [path.join(src, 'export_slides.mjs'), fixture, pdf, '--scale', '1', '--frames', frames, ...item.args];
  const { stdout, stderr } = await run(process.execPath, args, { cwd: path.dirname(src), timeout: 90000, maxBuffer: 4 * 1024 * 1024 });
  check(stderr.trim() === '', item.name + ': exporter has no stderr errors');
  const report = JSON.parse(stdout.trim());
  check(report.mode === item.mode && report.answerMode === item.answerMode, item.name + ': reported build and answer modes');
  check(report.revealedAnswers === item.revealed, item.name + ': opened answer count is ' + item.revealed);
  check(report.pages === item.expected.length, item.name + ': reported page count is ' + item.expected.length);
  check(report.rasterScale === 1 && report.bytes > 0, item.name + ': nonempty scale-1 PDF');
  const bytes = fs.readFileSync(pdf);
  check(bytes.subarray(0, 5).toString() === '%PDF-', item.name + ': output is a PDF');
  // Chromium emits plain page dictionaries; /Pages is deliberately excluded.
  const pdfPages = (bytes.toString('latin1').match(/\/Type\s*\/Page\b/g) || []).length;
  check(pdfPages === item.expected.length, item.name + ': actual PDF page dictionaries match the expected count');
  const images = fs.readdirSync(frames).filter(name => name.endsWith('.png')).sort().map(name => path.join(frames, name));
  check(images.length === item.expected.length, item.name + ': one PNG per expected page');
  results.push({ ...item, report, pdfPages, images });
}

let browser;
const browserErrors = [];
try {
  browser = await pw.chromium.launch(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
    ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
    : {});
  const page = await browser.newPage();
  page.on('pageerror', error => browserErrors.push(String(error.message || error)));
  page.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
  for (const item of results) {
    item.pixelChecks = [];
    for (let i = 0; i < item.images.length; i++) {
      const data = fs.readFileSync(item.images[i]).toString('base64');
      const pixels = await page.evaluate(async base64 => {
        const img = new Image();
        img.src = 'data:image/png;base64,' + base64;
        await img.decode();
        const canvas = document.createElement('canvas');
        canvas.width = img.width; canvas.height = img.height;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        let step = 0, quiz = 0;
        for (let p = 0; p < rgba.length; p += 4) {
          if (rgba[p] === 19 && rgba[p + 1] === 143 && rgba[p + 2] === 83 && rgba[p + 3] === 255) step++;
          if (rgba[p] === 173 && rgba[p + 1] === 37 && rgba[p + 2] === 151 && rgba[p + 3] === 255) quiz++;
        }
        return { width: img.width, height: img.height, step, quiz };
      }, data);
      check(pixels.width === 1280 && pixels.height === 720, item.name + ': page ' + (i + 1) + ' is a 16:9 stage');
      const expected = item.expected[i];
      check(expected && (expected[0] ? pixels.step > 1000 : pixels.step === 0), item.name + ': page ' + (i + 1) + ' has the expected stepper-answer visibility');
      check(expected && (expected[1] ? pixels.quiz > 1000 : pixels.quiz === 0), item.name + ': page ' + (i + 1) + ' has the expected quiz-answer visibility');
      item.pixelChecks.push({ file: item.images[i], ...pixels });
    }
  }
  check(browserErrors.length === 0, 'screenshot verification has no browser errors');
} finally {
  if (browser) await browser.close();
}

const report = { outdir, fixture, cases: results, browserErrors, failures };
fs.writeFileSync(path.join(outdir, 'report.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify({ outdir, cases: results.length, exportedPages: results.reduce((n, item) => n + item.images.length, 0), failures }, null, 2));
if (failures.length) process.exitCode = 1;
