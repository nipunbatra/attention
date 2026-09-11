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
  const roles = ['input', 'target', 'param', 'prob', 'activation', 'score', 'loss', 'token', 'symbol', 'index', 'temperature'];
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
await page.locator('#s01-names-intro img').evaluate(img => img.decode());
const namesIntro = await page.evaluate(() => {
  const issues=[], frames=[...document.querySelectorAll('#s01 .frame')];
  const intro=document.getElementById('s01-names-intro'), index=frames.indexOf(intro);
  if(frames[index-1]?.dataset.title!=='Predict the next word first')issues.push('Name-generation introduction must follow next-word prediction');
  const img=intro.querySelector('img'), repo=intro.querySelector('.names-source > a');
  if(repo.href!=='https://github.com/balasahebgulave/Dataset-indian-names')issues.push('Screenshot must link to the source dataset repository');
  if(!intro.querySelector('figcaption a').href.endsWith('/80401358aaa609cbe30ae57afbea37654879d0ab/Indian_Names.csv'))issues.push('CSV link must identify the verified dataset revision');
  if(!img.src.startsWith('data:image/png;base64,')||img.naturalWidth!==1520||img.naturalHeight!==1072)issues.push('Dataset screenshot must load in standalone HTML');
  if(intro.querySelector('.katex, .pytorch'))issues.push('Name-generation introduction should stay free of maths and code');
  return {frame:index+1,screenshot:[img.naturalWidth,img.naturalHeight],issues};
});
errors.push(...namesIntro.issues);
const lookup = await page.evaluate(() => {
  const issues=[], steps=[...document.querySelectorAll('[data-lookup-step]')];
  const order=steps.map(f=>f.dataset.lookupStep);
  const character=steps.find(f=>f.dataset.lookupStep==='character');
  const symbol=character.querySelector('.katex-html .p1-symbol'),literal=character.querySelector('.katex-html .p1-token');
  if(symbol?.textContent!=='c'||literal?.textContent!=='"a"')issues.push('Character choice must distinguish placeholder c from the quoted literal "a"');
  if(symbol&&literal&&getComputedStyle(symbol).color===getComputedStyle(literal).color)issues.push('Placeholder and character literal need distinct colours');
  for(const stage of ['id','row']){
    if(steps.find(f=>f.dataset.lookupStep===stage).querySelector('.katex-html .p1-token')?.textContent!=='"a"')issues.push(stage+': retain the quoted character literal');
  }
  const equation=steps.find(f=>f.dataset.lookupStep==='equation');
  if(equation.querySelectorAll('.p1-math .katex-html .p1-symbol').length!==2)issues.push('Both generic lookup occurrences of c must use the placeholder colour');
  if(order.join(',')!=='character,id,table,select,row,equation,code')issues.push('Lookup needs the paced symbol order before the complete equation');
  const frames=[...document.querySelectorAll('#s04 .frame')];
  const single=frames.findIndex(f=>f.querySelector('[data-torch="single-character-lookup"]'));
  const context=frames.findIndex(f=>f.querySelector('[data-torch="embedding"]'));
  if(single<0||context!==single+1)issues.push('Single-character lookup must immediately precede the three-character context lookup');
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
  const exp=F.z.map(Math.exp), total=exp.reduce((a,b)=>a+b,0);
  const tokens=['-','a','b',null,'z'];
  const expected=tokens.map(token=>{
    if(token===null)return ['⋮','⋮','⋮'];
    const i=AT.mlp.stoi[token];
    return [AT.fmt(F.z[i],3),AT.fmt(exp[i],3),F.p[i]<0.001?'<0.001':AT.fmt(F.p[i],3)];
  });
  const rows=[...document.querySelectorAll('#s07-softmax tbody tr')].map(row=>[...row.querySelectorAll('td')].map(cell=>cell.textContent.trim()));
  if(JSON.stringify(rows)!==JSON.stringify(expected))issues.push('Compact softmax table differs from saved logits/exponentials/probabilities');
  const labels=[...document.querySelectorAll('#s07-softmax tbody th')].map(cell=>cell.textContent.trim());
  if(JSON.stringify(labels)!==JSON.stringify(tokens.map(token=>token===null?'⋮':token)))issues.push('Worksheet must keep vocabulary order with an explicit omission');
  const footer=document.querySelector('#s07-softmax tfoot');
  if(footer.querySelector('th').textContent!=='all 27 tokens')issues.push('Softmax total must be labelled as all 27 tokens');
  if(JSON.stringify([...footer.querySelectorAll('td')].map(cell=>cell.textContent.trim()))!==JSON.stringify(['',AT.fmt(total,3),'1.000']))issues.push('Softmax denominator must include the omitted vocabulary rows');
  if(!document.querySelector('#s07-softmax').textContent.includes('Dots omit c–y'))issues.push('Explain which rows the dots omit');
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
const registration = await page.evaluate(() => {
  const issues=[], M=window.__TOY__, V=M.vocab.length, D=M.d_model, H=M.d_h;
  const expected=[['embedding.weight',[V,D]],['hidden.weight',[H,M.w*D]],['hidden.bias',[H]],['output.weight',[V,H]],['output.bias',[V]]]
    .map(([name,shape])=>[name,'['+shape.join(', ')+']']);
  const rows=[...document.querySelectorAll('#s09-named-params tbody tr')].map(row=>[row.querySelector('th').textContent,row.querySelector('td').textContent]);
  if(JSON.stringify(rows)!==JSON.stringify(expected))issues.push('Named parameter output must show the five registered tensors with PyTorch shapes');
  const frames=[...document.querySelectorAll('#s09 .frame')];
  const inspection=frames.findIndex(f=>f.querySelector('[data-torch="named-parameters"]'));
  if(inspection<0||!frames[inspection+1]?.querySelector('[data-torch="optimizer"]'))issues.push('Inspect the class parameters before passing them to SGD');
  const explanation=frames[inspection]?.textContent||'';
  if(!explanation.includes('nn.Module')||!explanation.includes('registers them'))issues.push('Explain inheritance and layer registration');
  return {rows,issues};
});
errors.push(...registration.issues);
const temperatureSequence = await page.evaluate(() => {
  const issues=[], table=document.getElementById('s10-temperature-table');
  if(!table.closest('.companion')||table.closest('.frame'))issues.push('Temperature arithmetic must be article-only');
  if(!table.getBoundingClientRect().height)issues.push('Optional temperature arithmetic must remain visible in article mode');
  return {articleArithmeticVisible:table.getBoundingClientRect().height>0,issues};
});
errors.push(...temperatureSequence.issues);
if(await page.evaluate(()=>/attention/i.test(document.querySelector('#s12').textContent)))errors.push('Tokenization section must not introduce attention');
const unknownTokens = await page.evaluate(() => {
  const issues=[], words=['hyperhappiness','electrojoy','nanobotany','unbelievable'];
  const expectedChars=words.map(word=>Array.from(word));
  const expectedSubwords=[['h','y','p','e','r','h','a','p','p','in','e','s','s'],
    ['e','le','c','t','r','o','j','o','y'],Array.from('nanobotany'),['un','believ','able']];
  const specs=[['s12-unk',words.map(()=>['<UNK>']),word=>AT.mlp.tokenizeWords(word,{unknown:true})],
    ['s12-char-unk',expectedChars,AT.mlp.tokenizeChars],['s12-subword-unk',expectedSubwords,AT.mlp.tokenizeSubwords]];
  const counts={};
  for(const [id,expected,tokenize] of specs){
    const rows=[...document.querySelectorAll('#'+id+' tbody tr')];
    if(rows.length!==words.length)issues.push(id+': missing an example word');
    counts[id]=[];
    rows.forEach((row,i)=>{
      const tokens=[...row.querySelectorAll('.oov-piece')].map(piece=>piece.textContent);
      if(row.dataset.word!==words[i]||row.querySelector('th').textContent!==words[i])issues.push(id+': comparisons must use the same four words in order');
      if(JSON.stringify(tokens)!==JSON.stringify(expected[i])||JSON.stringify(tokens)!==JSON.stringify(tokenize(words[i])))issues.push(id+'/'+words[i]+': pieces must match the existing tokenizer');
      const count=Number(row.querySelector('.oov-count').textContent);counts[id].push(count);
      if(count!==tokens.length)issues.push(id+'/'+words[i]+': incorrect token count');
      if(id!=='s12-unk'&&tokens.join('')!==words[i])issues.push(id+'/'+words[i]+': lost the original spelling');
    });
  }
  if(AT.mlp.tokenizeWords('learning',{unknown:true})[0]!=='learning')issues.push('Known whole words should keep their own identity');
  return {words,counts,issues};
});
errors.push(...unknownTokens.issues);
const generationLoop = await page.evaluate(() => {
  const root=document.getElementById('s10-loop-code'), issues=[];
  const original=JSON.stringify(__TOY__), trace=AT.part1Diagrams.generationExample().trace;
  const lines=[1,2,3,4,5,3];
  if(root.stepperApi.steps.length!==6)issues.push('Expected six line-by-line generation states');
  if(trace.map(t=>t.chosen_id).join(',')!=='19,1,13,0')issues.push('Loop walkthrough must replay the earlier sam sample');
  const required=[
    ['tensor([[0, 19, 1]])','name = ["s", "a"]','_ = 2','0, 1, …, 17'],
    ['ctx = tensor([[0, 19, 1]])','next_id = tensor([[13]])','shape [1, 1]'],
    ['next_id.item() = 13','stoi["-"] = 0','13 == 0  is False'],
    ['vocab[13] = "m"','name = ["s", "a", "m"]'],
    ['ctx[:, 1:] = tensor([[19, 1]])','next_id    = tensor([[13]])','ctx = tensor([[19, 1, 13]])'],
    ['ctx = tensor([[19, 1, 13]])','next_id = tensor([[0]])','0 == 0  is True','name = ["s", "a", "m"]']
  ];
  const frame=root.closest('.frame'), frameIndex=[...document.querySelectorAll('#s10 .frame')].indexOf(frame)+1;
  AT.present.enter(); AT.present.go('s10',frameIndex,0);
  for(let i=0;i<6;i++){
    root.stepperApi.go(i);
    const active=frame.querySelectorAll('.loop-line.is-current');
    if(active.length!==1||Number(active[0].dataset.line)!==lines[i]||active[0].getAttribute('aria-current')!=='step')issues.push('Wrong active code line at step '+i);
    if(required[i].some(text=>!root.stepperApi.stage.textContent.includes(text)))issues.push('Incorrect input/output at step '+i);
    if(AT.present.fitReport().overflow)issues.push('Loop step '+i+' overflows');
  }
  const reference=document.getElementById('s10-loop-reference');
  if(reference.getBoundingClientRect().height)issues.push('Article reference should not appear in the presentation');
  AT.present.exit();
  if(reference.querySelectorAll('.loop-reference').length!==6||!reference.getBoundingClientRect().height)issues.push('Article should retain all six explanations together');
  if(JSON.stringify(__TOY__)!==original)issues.push('Walkthrough must not change model parameters');
  root.stepperApi.go(0);
  return {steps:6,highlightedLines:lines,issues};
});
errors.push(...generationLoop.issues);
for(let step=0;step<6;step++){
  await page.evaluate(step=>{
    const root=document.getElementById('s10-loop-code'),frames=[...document.querySelectorAll('#s10 .frame')];
    AT.present.enter();AT.present.go('s10',frames.indexOf(root.closest('.frame'))+1,0);
    root.stepperApi.go(step);
  },step);
  await page.waitForTimeout(80);
  await page.screenshot({path:path.join(out,'generation-loop-'+step+'.png')});
}
// Include the fully revealed notation frames and the tables adjacent to them.
for (const [section, title, name] of [
  ['s01', 'Generate Indian names, one character at a time', 'names-intro'],
  ['s03', 'Writing the same question as a probability', 'probability'],
  ['s04', 'c stands for one character', 'lookup-character'],
  ['s04', "id(c) gives that character's row number", 'lookup-id'],
  ['s04', 'Look up a learned row', 'lookup'],
  ['s04', 'Square brackets select a row', 'lookup-select'],
  ['s04', 'Lowercase e names the retrieved vector', 'lookup-row'],
  ['s04', 'The complete lookup equation', 'lookup-equation'],
  ['s04', 'PyTorch: look up just the character a', 'lookup-code'],
  ['s04', 'PyTorch: a learned embedding lookup', 'lookup-context'],
  ['s05', 'Three rows become one row', 'concatenation'],
  ['s06', 'The hidden layer combines the input numbers', 'hidden-layer'],
  ['s06', 'The output layer makes one score per token', 'output-layer'],
  ['s06', 'PyTorch: reuse the same layers as one model', 'model-sequential'],
  ['s06', 'PyTorch: the same model as a class', 'model-class'],
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
  ['s09', 'Where model.parameters() comes from', 'named-parameters'],
  ['s09', 'PyTorch: choose the parameters to update', 'optimizer'],
  ['s09', 'Only rows a and b receive embedding gradients', 'embedding-gradients'],
  ['s10', 'Temperature changes the sampling probabilities', 'temperature-bars'],
  ['s10', 'Divide the logits by temperature, then apply softmax', 'temperature-guide'],
  ['s10', 'PyTorch: choose the next character', 'sampling-code'],
  ['s12', 'Coverage and sequence length', 'tokenization-tradeoff'],
  ['s12', 'Word tokens: unseen words become <UNK>', 'unknown-words'],
  ['s12', 'Character tokens: known letters spell new words', 'unknown-characters'],
  ['s12', 'Subword tokens: pieces first, letters when needed', 'unknown-subwords'],
  ['s14', 'The same calculation at larger widths', 'window-shapes'],
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
  if(section==='s10' && await page.locator('#s10-temperature-table').isVisible())errors.push(name+': optional arithmetic must stay hidden in presentation mode');
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
await page.locator('#s01-names-intro').scrollIntoViewIfNeeded();
const namesPhone = await page.evaluate(() => ({
  columns:getComputedStyle(document.querySelector('#s01 .names-intro')).gridTemplateColumns.split(' ').length,
  overflow:document.documentElement.scrollWidth>innerWidth
}));
if(namesPhone.columns!==1||namesPhone.overflow)errors.push('Phone dataset introduction must fit one column');
await page.screenshot({path:path.join(out,'phone-names-intro.png')});
await page.emulateMedia({ media: 'print' });
const printed = await page.evaluate(checkGuides);
errors.push(...printed.issues.map(x => 'print: ' + x));
await browser.close();
console.log(JSON.stringify({ article, namesIntro, lookup, shapes, probabilitySequence, registration, temperatureSequence, unknownTokens, generationLoop, phone, namesPhone, print: printed, screenshots: out, errors }, null, 2));
if (errors.length) process.exitCode = 1;
