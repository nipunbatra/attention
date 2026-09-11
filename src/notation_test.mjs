// Check that introductory Part 1 equations have nearby prose in matching colours.
// node src/notation_test.mjs [part1.html] [screenshot-directory]
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
const out = process.argv[3] || fs.mkdtempSync(path.join(os.tmpdir(), 'attention-notation-'));
fs.mkdirSync(out, { recursive: true });
const browser = await pw.chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const errors = [];
page.on('pageerror', e => errors.push(e.message));
await page.goto(pathToFileURL(path.resolve(process.argv[2] || 'part1.html')).href);
await page.waitForTimeout(250);
const checkGuides = () => {
  const issues = [], guides = [...document.querySelectorAll('[data-math-guide]')];
  const roles = ['input', 'target', 'param', 'prob', 'activation', 'score', 'loss', 'token', 'index', 'temperature'];
  for (const guide of guides) {
    for (const role of roles) {
      const cls = '.p1-' + role;
      const symbols = [...guide.querySelectorAll('.p1-math .katex-html ' + cls)];
      if (!symbols.length) continue;
      const prose = [...guide.querySelectorAll('.p1-key ' + cls + ', .p1-read' + cls + ', .p1-read ' + cls)]
        .filter(el => !el.closest('.katex') && /[A-Za-z]{3}/.test(el.textContent));
      if (!prose.length) issues.push(guide.closest('.sec').id + ': no prose for ' + role);
      for (const symbol of symbols) if (!prose.some(el => getComputedStyle(el).color === getComputedStyle(symbol).color)) {
        issues.push(guide.closest('.sec').id + ': colour mismatch for ' + role);
      }
    }
  }
  if (guides.length < 10) issues.push('Missing introductory equation guides');
  for (const row of document.querySelectorAll('#s06-shapes tbody tr, #s06-parameter-shapes tbody tr, #s16-input-notation tbody tr, #s16-output-notation tbody tr')) {
    const symbol = row.querySelector('th .katex-html [class*="p1-"]');
    const meaning = row.querySelector('td');
    if (!symbol || !meaning || getComputedStyle(symbol).color !== getComputedStyle(meaning).color) {
      issues.push(row.closest('[id]').id + ': table symbol/meaning colour mismatch');
    }
  }
  return { guides: guides.length, issues };
};
const article = await page.evaluate(checkGuides);
errors.push(...article.issues);
const lookup = await page.evaluate(() => {
  const issues=[], steps=[...document.querySelectorAll('[data-lookup-step]')];
  const order=steps.map(f=>f.dataset.lookupStep);
  if(order.join(',')!=='character,id,table,select,row,equation,code')issues.push('Lookup needs the paced symbol order before the complete equation');
  const ids=[...document.querySelectorAll('#s04-lookup-ids tbody tr')].map(r=>[r.querySelector('th').textContent,Number(r.querySelector('td').textContent)]);
  if(JSON.stringify(ids)!==JSON.stringify(['-','a','b','i'].map(c=>[c,AT.mlp.stoi[c]])))issues.push('Character IDs disagree with vocabulary');
  for(const id of ['s04-embed','s04-selected-row']){
    const rows=[...document.querySelectorAll('#'+id+' tbody tr')];
    for(const row of rows){
      const c=row.querySelector('th').textContent, cells=[...row.querySelectorAll('td')].map(n=>n.textContent);
      const expected=[String(AT.mlp.stoi[c]),...__TOY__.E[AT.mlp.stoi[c]].map(n=>AT.fmt(n,2))];
      if(JSON.stringify(cells)!==JSON.stringify(expected))issues.push('Lookup table values differ: '+id+'/'+c);
    }
  }
  const expected='['+__TOY__.E[AT.mlp.stoi.a].map(n=>AT.fmt(n,2)).join(', ')+']';
  if(document.querySelector('[data-lookup-values]').textContent!==expected)issues.push('Retrieved a vector differs');
  if(!/selected|hl|highlight/.test(document.querySelector('#s04-selected-row tbody tr').className))issues.push('Selected a row is not highlighted');
  return {steps:order,issues};
});
errors.push(...lookup.issues);
const shapes = await page.evaluate(() => {
  const issues=[], M=window.__TOY__, inputWidth=M.w*M.E[0].length;
  const rows=id=>[...document.querySelectorAll('#'+id+' tbody tr')].map(row=>[...row.querySelectorAll('td')].map(cell=>cell.textContent.trim()));
  const expected=[inputWidth,M.d_h,M.vocab.length].map((width,i)=>['1 example',width+' '+['input features','hidden activations','vocabulary scores'][i],'1 × '+width]);
  if(JSON.stringify(rows('s06-shapes'))!==JSON.stringify(expected))issues.push('Activation shapes must label examples and feature/score counts separately');
  const parameterShapes=rows('s06-parameter-shapes').map(row=>row[1]);
  if(JSON.stringify(parameterShapes)!==JSON.stringify([inputWidth+' × '+M.d_h,'1 × '+M.d_h,M.d_h+' × '+M.vocab.length,'1 × '+M.vocab.length]))issues.push('Parameter shapes differ from model dimensions');
  const explanation=document.getElementById('s06-parameter-shapes').closest('.frame').textContent;
  if(!explanation.includes('does not count examples')||!explanation.includes('Broadcasting'))issues.push('Bias-row exception and broadcasting need an explanation');
  return {rows:expected,parameterShapes,issues};
});
errors.push(...shapes.issues);
const probabilitySequence = await page.evaluate(() => {
  const issues=[], M=window.__TOY__, F=AT.mlp.forward(['a','a','b']);
  const top=AT.topk(F.p,5).map(item=>item.i), rest=F.p.map((_,i)=>i).filter(i=>!top.includes(i));
  const exp=F.z.map(Math.exp), total=exp.reduce((a,b)=>a+b,0);
  const expected=top.map(i=>[F.z[i],exp[i],F.p[i]].map(n=>AT.fmt(n,3)));
  expected.push(['',AT.fmt(rest.reduce((a,i)=>a+exp[i],0),3),AT.fmt(rest.reduce((a,i)=>a+F.p[i],0),3)]);
  const rows=[...document.querySelectorAll('#s07-softmax tbody tr')].map(row=>[...row.querySelectorAll('td')].map(cell=>cell.textContent.trim()));
  if(JSON.stringify(rows)!==JSON.stringify(expected))issues.push('Compact softmax table differs from saved logits/exponentials/probabilities');
  if(exp.some((n,i)=>Math.abs(n/total-F.p[i])>1e-12))issues.push('Direct and stable softmax disagree');
  if(document.querySelector('#s07-target-prob').textContent!==AT.fmt(F.p[AT.mlp.stoi.i],3))issues.push('Final displayed target probability differs');
  const frames=[...document.querySelectorAll('#s07 .frame')];
  if(frames.length!==3||document.querySelectorAll('#s06-worked-hidden,#s06-worked-logit').length!==2)issues.push('Expected one combined MLP frame and three probability frames');
  for(const id of ['s06-hidden-compact','s06-logit-compact'])if(!document.getElementById(id).closest('.companion'))issues.push('Detailed products must remain available in the article: '+id);
  if(!document.querySelector('[data-torch="stable-softmax"]').closest('.companion'))issues.push('Stability derivation/code should stay in full article mode');
  const hiddenSum=F.a0.reduce((sum,x,i)=>sum+x*M.W1[i][0],0);
  const target=AT.mlp.stoi.i, outputSum=F.a1.reduce((sum,x,i)=>sum+x*M.W2[i][target],0);
  for(const [id,terms] of [['s06-worked-hidden',[hiddenSum,M.b1[0],F.a1[0]]],['s06-worked-logit',[outputSum,M.b2[target],F.z[target]]]]){
    const math=document.querySelector('#'+id+' annotation')?.textContent||'';
    if(terms.some(n=>!math.includes(AT.fmt(n,3))))issues.push('Compact MLP arithmetic differs: '+id);
  }
  return {classroomFrames:1+frames.length, targetProbability:F.p[target], issues};
});
errors.push(...probabilitySequence.issues);
// Include the fully revealed notation frames and the tables adjacent to them.
for (const [section, title, name] of [
  ['s03', 'Writing the same question as a probability', 'probability'],
  ['s04', 'c stands for one character', 'lookup-character'],
  ['s04', "id(c) gives that character's row number", 'lookup-id'],
  ['s04', 'Look up a learned row', 'lookup'],
  ['s04', 'Square brackets select a row', 'lookup-select'],
  ['s04', 'Lowercase e names the retrieved vector', 'lookup-row'],
  ['s04', 'The complete lookup equation', 'lookup-equation'],
  ['s04', 'PyTorch: look up just the character a', 'lookup-code'],
  ['s05', 'Three rows become one row', 'concatenation'],
  ['s06', 'The hidden layer combines the input numbers', 'hidden-layer'],
  ['s06', 'The output layer makes one score per token', 'output-layer'],
  ['s06', 'Check the shapes before multiplying', 'shapes'],
  ['s06', 'Parameters are shared across examples', 'parameter-shapes'],
  ['s06', 'PyTorch: four examples, the same feature widths', 'batch-shapes'],
  ['s06', 'A hidden activation and an output score', 'worked-mlp'],
  ['s07', 'Exponentiate, then divide by the total', 'softmax'],
  ['s07', 'Softmax worksheet', 'softmax-table'],
  ['s07', 'A distribution over characters', 'probability-bars'],
  ['s08', 'One target probability', 'loss'],
  ['s08', 'Low target probability gives a large loss', 'loss-examples'],
  ['s09', 'What the optimizer changes', 'parameters'],
  ['s09', 'Only rows a and b receive embedding gradients', 'embedding-gradients'],
  ['s14', 'A larger window makes a larger weight matrix', 'window-shapes'],
  ['s16', 'The same model in symbols', 'summary'],
  ['s16', 'Notation: from tokens to the MLP input', 'input-notation'],
  ['s16', 'Notation: from the input to a probability', 'output-notation']
]) {
  await page.evaluate(({ section, title }) => {
    const frames = [...document.querySelectorAll('#' + section + ' .frame')];
    const index = frames.findIndex(frame => frame.dataset.title === title);
    if (index < 0) throw new Error('Missing notation frame: ' + title);
    AT.present.enter(); AT.present.go(section, index + 1, 99);
  }, { section, title });
  await page.waitForTimeout(80);
  const fit = await page.evaluate(() => AT.present.fitReport());
  if (fit.overflow) errors.push(name + ': frame overflow');
  await page.screenshot({ path: path.join(out, name + '.png') });
}
await page.evaluate(() => AT.present.exit());
await page.setViewportSize({ width: 390, height: 844 });
await page.evaluate(() => document.querySelectorAll('#s03 .frame')[1].scrollIntoView({ behavior: 'instant' }));
await page.waitForTimeout(80);
const phone = await page.evaluate(() => ({
  overflow: document.documentElement.scrollWidth > innerWidth,
  columns: getComputedStyle(document.querySelector('#s03 .p1-key')).gridTemplateColumns.split(' ').length
}));
if (phone.overflow || phone.columns !== 1) errors.push('Phone article legend does not fit one column');
await page.screenshot({ path: path.join(out, 'phone-probability.png') });
await page.emulateMedia({ media: 'print' });
const printed = await page.evaluate(checkGuides);
errors.push(...printed.issues.map(x => 'print: ' + x));
await browser.close();
console.log(JSON.stringify({ article, lookup, shapes, probabilitySequence, phone, print: printed, screenshots: out, errors }, null, 2));
if (errors.length) process.exitCode = 1;
