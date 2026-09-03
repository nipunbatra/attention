// Table-specific layout/semantic audit. Uses an existing Playwright runtime.
// node src/check_tables.mjs [page.html ...] [--out /private/tmp/table-audit]
import { createRequire } from 'module';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { pathToFileURL } from 'url';

const require = createRequire(import.meta.url);
const candidates = [process.env.PLAYWRIGHT_MODULE, 'playwright', 'playwright-core'].filter(Boolean);
const cache = path.join(os.homedir(), '.npm', '_npx');
if (fs.existsSync(cache)) for (const dir of fs.readdirSync(cache).sort()) candidates.push(path.join(cache, dir, 'node_modules/playwright'));
let pw;
for (const candidate of candidates) { try { pw = require(candidate); break; } catch {} }
if (!pw) throw new Error('No existing Playwright runtime; installs nothing.');
const args = process.argv.slice(2), outIndex = args.indexOf('--out');
const out = outIndex >= 0 ? args.splice(outIndex, 2)[1] : '/private/tmp/table-audit';
const files = args.length ? args : ['part1.html', 'attention.html', 'part3.html'];
fs.mkdirSync(out, { recursive: true });
const browser = await pw.chromium.launch(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE } : {});
const reports = [];

// Text excludes the MathML/annotation duplicate used by KaTeX.
function scanTables({ rootSelector = 'body', mode, state }) {
  const root = document.querySelector(rootSelector);
  if (!root) return [];
  const visible = el => el.getClientRects().length && getComputedStyle(el).visibility !== 'hidden' && !el.closest('[hidden]');
  const text = el => {
    if (!el) return '';
    const clone = el.cloneNode(true);
    clone.querySelectorAll('.katex-mathml, annotation, script, style').forEach(n => n.remove());
    return clone.textContent.trim().replace(/\s+/g, ' ');
  };
  const box = el => { const b = el.getBoundingClientRect(); return { x:b.x, y:b.y, width:b.width, height:b.height }; };
  const numeric = s => /^[-+−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:%|\s*×\s*10[−\d]+)?$/.test(s);
  const renderedTables=[...root.querySelectorAll('table, .mat-grid[role="table"]')].filter(visible);
  return renderedTables.map((table, index) => {
    const section = table.closest('.sec'), frame = table.closest('.frame');
    const fig = table.closest('.dt-fig, .mat, .calc') || table;
    const scroll = table.closest('.dt-scroll, .mat-scroll, .calc-scroll, .scroll-x') || table.parentElement;
    const tableBox = box(table), scrollBox = box(scroll);
    const scale = document.body.classList.contains('present') ? Number(getComputedStyle(document.body).getPropertyValue('--present-scale')) : 1;
    const cells = [...table.querySelectorAll('th, td, .cell, .mat-cl, .mat-rl')].filter(visible);
    const headers = table.matches('table') ? [...table.querySelectorAll('thead th')].map(text) : [...table.querySelectorAll('.mat-cl')].map(text);
    const rows = table.matches('table') ? [...table.querySelectorAll('tbody tr, tfoot tr')].map(tr => [...tr.children].map(text)) : [];
    const owner = table.closest('[id]');
    const issues = [];
    const proseNumeric = cells.filter(c => c.matches('td.dt-num') && !c.querySelector('.katex') && /[a-zA-Z]{2,}/.test(text(c)) && (getComputedStyle(c).textAlign === 'right' || /mono/i.test(getComputedStyle(c).fontFamily)));
    if (proseNumeric.length) issues.push({ kind:'text-styled-numeric', count:proseNumeric.length, examples:proseNumeric.slice(0,4).map(c => ({text:text(c), align:getComputedStyle(c).textAlign, font:getComputedStyle(c).fontFamily})) });
    const overlap = cells.filter(c => {
      if(!c.matches('th,td')||c.querySelector('.katex')||!text(c))return false;
      const range=document.createRange();range.selectNodeContents(c);
      const r=range.getBoundingClientRect(),b=c.getBoundingClientRect();
      // Tight headings may use their own padding; flag actual neighboring-cell
      // overlap, not a harmless few pixels of padding intrusion.
      return r.left < b.left-1 || r.right > b.right+1;
    });
    if(overlap.length)issues.push({kind:'cell-text-overlap',examples:overlap.slice(0,4).map(c=>({text:text(c),width:Math.round(c.getBoundingClientRect().width/scale)}))});
    const outside = tableBox.width > scrollBox.width + 2;
    const css = getComputedStyle(scroll);
    if (outside && !/(auto|scroll)/.test(css.overflowX)) issues.push({kind:mode==='presentation'?'column-spill':'uncontained-table', extra:Math.round((tableBox.width-scrollBox.width)/scale)});
    if(mode==='presentation'&&!table.closest('.calc-pop')) {
      const neighbors=renderedTables.filter(other=>other!==table&&!other.closest('.calc-pop')&&!table.contains(other)&&!other.contains(table)).flatMap(other=>{
        const b=other.getBoundingClientRect(),w=Math.min(tableBox.x+tableBox.width,b.right)-Math.max(tableBox.x,b.left),h=Math.min(tableBox.y+tableBox.height,b.bottom)-Math.max(tableBox.y,b.top);
        return w>2&&h>2?[{owner:other.closest('[id]')?.id||'',width:Math.round(w/scale),height:Math.round(h/scale)}]:[];
      });
      if(neighbors.length)issues.push({kind:'table-overlap',neighbors});
    }
    if (mode === 'phone') {
      const rowHeaders = cells.filter(c => c.matches('tbody th, tfoot th') && getComputedStyle(c).position === 'sticky');
      const maxLabel = rowHeaders.reduce((m,c) => Math.max(m,c.getBoundingClientRect().width),0);
      if (outside && maxLabel > scrollBox.width * .5) issues.push({kind:'phone-sticky-label-dominates', labelWidth:Math.round(maxLabel), available:Math.round(scrollBox.width), ratio:+(maxLabel/scrollBox.width).toFixed(2)});
      if (tableBox.width > window.innerWidth && !/(auto|scroll)/.test(css.overflowX)) issues.push({kind:'phone-page-overflow',width:Math.round(tableBox.width)});
    }
    const squeezed = cells.filter(c => {
      if (c.querySelector('.katex') || text(c).length < 6 || !/[a-zA-Z]{4}/.test(text(c))) return false;
      const cs=getComputedStyle(c), b=c.getBoundingClientRect();
      const available=b.width/scale-parseFloat(cs.paddingLeft)-parseFloat(cs.paddingRight);
      return cs.whiteSpace !== 'nowrap' && available < parseFloat(cs.fontSize)*4.5 && b.height/scale > parseFloat(cs.fontSize)*2.5;
    });
    if (squeezed.length) issues.push({kind:'narrow-prose-column',examples:squeezed.slice(0,4).map(c=>({text:text(c),width:Math.round(c.getBoundingClientRect().width/scale)}))});
    const emptyHeaders = headers.flatMap((s,i)=>!s && i>0?[i]:[]);
    if (emptyHeaders.length) issues.push({kind:'empty-data-headers',columns:emptyHeaders});
    const precision = headers.map((header,i)=> {
      const values = rows.map(r=>r[i]).filter(s=>numeric(s||''));
      const digits = [...new Set(values.map(s=>s.includes('.')?s.split('.')[1].replace(/\D/g,'').length:0))];
      return {header,digits,values:values.length};
    }).filter(p=>p.digits.length>1);
    const fonts = cells.map(c=>parseFloat(getComputedStyle(c).fontSize));
    const smallestFont = Math.min(...fonts.filter(n=>n>0));
    if (mode === 'presentation' && table.matches('.calc-t') && smallestFont < 18) {
      issues.push({kind:'small-presentation-table-text',fontSize:smallestFont,minimum:18});
    }
    return {
      mode,state,index,section:section?.id||'outside',sectionTitle:section?.dataset.title||'',
      frame:frame?.id||'',frameTitle:frame?.dataset.title||'',owner:owner?.id||'',
      family:table.className,headers,rows,caption:text(fig.querySelector('.dt-cap,.mat-cap,figcaption,caption')),
      note:text(fig.querySelector('.dt-note')),cellCount:cells.length,
      tableWidth:Math.round(tableBox.width/scale),viewportWidth:Math.round(scrollBox.width/scale),
      tableHeight:Math.round(tableBox.height/scale),horizontalScroll:outside&&/(auto|scroll)/.test(css.overflowX),
      smallestFont,largestFont:Math.max(...fonts),precision,issues,
      rowLabels:cells.filter(c=>c.matches('tbody th,tfoot th,.mat-rl')).map(text),
      location:{x:tableBox.x,y:tableBox.y,width:tableBox.width,height:tableBox.height}
    };
  });
}

for (const file of files) {
  const name = path.basename(file);
  const regressionOwners = {
    'part1.html': ['s11-table', 's12-trade', 's16-input-notation'],
    'attention.html': ['s11-wgrid', 's18-q7', 's18-notation', 's17-cmp-fixed'],
    'part3.html': ['s03-table', 's06-shift', 's15-table-rest', 's02-softmax']
  }[name] || [];
  const regressionFrames = {
    'part1.html': ['#s04/2/0:', '#s11/2/0:', '#s16/3/0:'],
    'attention.html': ['#s07/3/0:', '#s11/3/0:', '#s14/2/1:', '#s17/2/0:'],
    'part3.html': ['#s03/1/0:', '#s12/1/0:', '#s14/2/1:']
  }[name] || [];
  const context = await browser.newContext({viewport:{width:1280,height:720}});
  const page = await context.newPage();
  const pageErrors=[];
  page.on('pageerror', e=>pageErrors.push(String(e)));
  await page.goto(pathToFileURL(path.resolve(file)).href,{waitUntil:'load'});
  await page.waitForTimeout(250);
  const kindContract = await page.evaluate(() => {
    const host = document.createElement('div'); document.body.appendChild(host);
    const fig = AT.ui.table([['word', 'forced prose', 'aab', 42]], {
      into: host, rowCls: 'q',
      lead: [{ label: 'lead', fn: row=>row[0] }],
      cols: [{ label: 'auto' }, { label: 'forced number', kind: 'number' }, { label: 'forced code', kind: 'code' }, { label: 'forced text', kind: 'text' }],
      computed: [{ label: 'computed', fn: row=>row[0] }],
      footer: { label: 'footer', cls: 'q', lead: ['footer lead'], values: ['footer text', '2.50', 'footer code', 42], computed: ['footer computed'] }
    });
    const failures = [];
    const expect = (cell, kind, where) => {
      if (!cell || cell.dataset.kind !== kind || !cell.classList.contains('dt-' + (kind === 'number' ? 'num' : kind))) {
        failures.push(where + ': expected ' + kind + ', got ' + (cell && cell.dataset.kind));
      }
    };
    expect(fig.leadCells[0][0], 'text', 'lead initial');
    expect(fig.cells[0][0], 'text', 'auto text initial');
    expect(fig.cells[0][1], 'number', 'forced number initial');
    expect(fig.cells[0][2], 'code', 'forced code initial');
    expect(fig.cells[0][3], 'text', 'forced text initial');
    expect(fig.computedCells[0][0], 'text', 'computed initial');
    expect(fig.footer.lead[0], 'text', 'footer lead initial');
    expect(fig.footer.cells[0], 'text', 'footer auto text initial');
    expect(fig.footer.cells[1], 'number', 'footer number initial');
    expect(fig.footer.cells[2], 'code', 'footer code initial');
    expect(fig.footer.cells[3], 'text', 'footer forced text initial');
    expect(fig.footer.computed[0], 'text', 'footer computed initial');
    const tinted = [...fig.rowEls[0].querySelectorAll('td'), ...fig.table.querySelectorAll('tfoot td')];
    const background = getComputedStyle(fig.cells[0][0]).backgroundColor;
    if (tinted.some(cell=>getComputedStyle(cell).backgroundColor!==background)) failures.push('row/footer tint differs by cell kind');
    fig.update([['3.50', 'still forced', 'xyz', 42]]);
    expect(fig.cells[0][0], 'number', 'auto text-to-number update');
    expect(fig.leadCells[0][0], 'number', 'lead text-to-number update');
    expect(fig.computedCells[0][0], 'number', 'computed text-to-number update');
    expect(fig.table.tHead.rows[0].cells[2], 'number', 'auto header update');
    expect(fig.cells[0][1], 'number', 'forced number update');
    expect(fig.cells[0][2], 'code', 'forced code update');
    expect(fig.cells[0][3], 'text', 'forced text update');
    fig.update([['again', 42, 'xyz', 42]]);
    expect(fig.cells[0][0], 'text', 'auto number-to-text update');
    expect(fig.leadCells[0][0], 'text', 'lead number-to-text update');
    expect(fig.computedCells[0][0], 'text', 'computed number-to-text update');
    expect(fig.table.tHead.rows[0].cells[2], 'text', 'auto header text update');
    fig.setFooter(['9', 'not numeric', 'tail', 42], ['3.14'], ['8.25']);
    expect(fig.footer.cells[0], 'number', 'footer auto update');
    expect(fig.footer.cells[1], 'number', 'footer forced number update');
    expect(fig.footer.cells[2], 'code', 'footer forced code update');
    expect(fig.footer.cells[3], 'text', 'footer forced text update');
    expect(fig.footer.lead[0], 'number', 'footer lead update');
    expect(fig.footer.computed[0], 'number', 'footer computed update');
    host.remove();
    return { failures };
  });
  kindContract.failures.forEach(failure=>pageErrors.push('TABLE KIND CONTRACT: '+failure));
  await page.evaluate(()=>document.querySelectorAll('details').forEach(d=>d.open=true));
  const tables=[], seen=new Set(), screenshots=[];
  async function scan(mode,state,rootSelector='body') {
    const found=await page.evaluate(scanTables,{rootSelector,mode,state});
    for (const t of found) {
      const key=JSON.stringify([mode,t.section,t.frameTitle,t.owner,t.headers,t.rows,t.tableWidth,t.tableHeight]);
      if (seen.has(key)) continue;
      seen.add(key); tables.push(t);
    }
    return found;
  }
  async function reading(mode) {
    await scan(mode,'initial');
    const steps=await page.evaluate(()=>[...document.querySelectorAll('.stepper')].filter(s=>s.stepperApi).map((s,i)=>{s.dataset.tableAuditStepper=String(i);return {i,count:s.stepperApi.steps.length};}));
    for(const stepper of steps) {
      const selector=`[data-table-audit-stepper="${stepper.i}"]`;
      for(let i=0;i<stepper.count;i++) {
        await page.evaluate(({selector,i})=>document.querySelector(selector).stepperApi.go(i),{selector,i});
        await scan(mode,`stepper-${stepper.i}-${i}`,selector);
      }
      await page.evaluate(selector=>document.querySelector(selector).stepperApi.go(0),selector);
    }
  }
  await reading('reading');
  for (const owner of regressionOwners.slice(0,2)) {
    const locator=page.locator(`[id="${owner}"]`).first();
    if (!await locator.count()) continue;
    await locator.scrollIntoViewIfNeeded();
    const dest=path.join(out,`${path.basename(file,'.html')}-reading-${owner}.png`);
    await page.screenshot({path:dest}); screenshots.push(dest);
  }
  await page.setViewportSize({width:390,height:844});
  await reading('phone');
  // Capture a bounded sample of tables with quality warnings and dense tables.
  const shotTargets=tables.filter(t=>t.mode==='phone'&&(t.issues.length||t.cellCount>90||regressionOwners.includes(t.owner))).filter((t,i,a)=>a.findIndex(x=>x.owner===t.owner&&x.section===t.section)===i).slice(0,12);
  for (const t of shotTargets) {
    if(!t.owner||t.state!=='initial') continue;
    const locator=page.locator(`[id="${t.owner}"]`).first();
    await locator.scrollIntoViewIfNeeded();
    const dest=path.join(out,`${path.basename(file,'.html')}-phone-${t.owner}.png`);
    await page.screenshot({path:dest});screenshots.push(dest);
    if(regressionOwners.includes(t.owner)) {
      const scrolled=await locator.evaluate(el=>{
        const scroll=el.querySelector('.dt-scroll,.calc-scroll,.scroll-x')||el.closest('.scroll-x');
        if(!scroll||scroll.scrollWidth<=scroll.clientWidth+2)return false;
        scroll.scrollLeft=scroll.scrollWidth;return true;
      });
      if(scrolled){
        const right=dest.replace('.png','-right.png');await page.screenshot({path:right});screenshots.push(right);
        await locator.evaluate(el=>{const scroll=el.querySelector('.dt-scroll,.calc-scroll,.scroll-x')||el.closest('.scroll-x');if(scroll)scroll.scrollLeft=0;});
      }
    }
  }
  await page.setViewportSize({width:1280,height:720});
  await page.evaluate(()=>{ AT.present.enter(); AT.present.first(); });
  const states=new Set();
  for(let guard=0;guard<5000;guard++) {
    const st=await page.evaluate(()=>{const s=AT.present.state();return {key:location.hash+':'+(s.stepper?.index??''), final:s.fi===s.total-1&&s.build===s.frame.maxBuild&&(!s.stepper||s.stepper.index===s.stepper.count-1)};});
    if(states.has(st.key)) {pageErrors.push('Repeated presentation state: '+st.key);break;}
    states.add(st.key);
    const live=await scan('presentation',st.key,'.frame.is-live');
    // Presentation worksheet headers inherit fixed 11px rules unless explicitly
    // overridden; capture those too because they can pass geometry checks.
    if((live.some(t=>t.issues.length||t.smallestFont<18)||regressionFrames.includes(st.key))&&screenshots.filter(s=>s.includes('-present-')).length<16) {
      const shotKey=live.map(t=>t.owner+':'+t.frameTitle).join('|');
      if(!screenshots.includes(shotKey)) {
        const dest=path.join(out,`${path.basename(file,'.html')}-present-${st.key.replace(/[^a-z0-9_-]/gi,'_')}.png`);
        await page.waitForTimeout(250);
        await page.screenshot({path:dest});screenshots.push(dest,shotKey);
      }
    }
    if(st.final) break;
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(12);
  }
  const report={file:path.resolve(file),presentationStates:states.size,pageErrors,kindContract,tables,screenshots:screenshots.filter(s=>s.endsWith('.png'))};
  fs.writeFileSync(path.join(out,path.basename(file,'.html')+'.json'),JSON.stringify(report,null,2));
  const summary={file,presentationStates:states.size,pageErrors,tableSamples:tables.length,counts:Object.fromEntries(['reading','phone','presentation'].map(mode=>[mode,tables.filter(t=>t.mode===mode).length])),issues:tables.filter(t=>t.issues.length).map(({mode,state,section,frameTitle,owner,issues})=>({mode,state,section,frameTitle,owner,issues})),screenshots:report.screenshots};
  reports.push(summary);
  console.log(JSON.stringify({file, presentationStates:states.size, pageErrors, counts:summary.counts,
    issueCounts:Object.fromEntries([...new Set(tables.flatMap(t=>t.issues.map(i=>i.kind)))].map(kind=>[kind,tables.filter(t=>t.issues.some(i=>i.kind===kind)).length])),
    report:path.join(out,path.basename(file,'.html')+'.json')},null,2));
  await context.close();
}
await browser.close();
fs.writeFileSync(path.join(out,'summary.json'),JSON.stringify(reports,null,2));
const failureKinds = new Set(['cell-text-overlap', 'table-overlap', 'text-styled-numeric', 'small-presentation-table-text', 'phone-page-overflow']);
process.exitCode = reports.some(report=>report.pageErrors.length || report.issues.some(table=>table.issues.some(issue=>failureKinds.has(issue.kind)))) ? 1 : 0;
