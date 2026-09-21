// Four master maps, code-linked focus views, and presentation/reading layouts.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {spawnSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const browser=await chromium.launch();
const shots=fs.mkdtempSync('/private/tmp/wordlm-pipeline-');
try{
  const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const url=pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href;
  await page.goto(url);await page.evaluate(()=>document.fonts.ready);
  const original=await page.evaluate(()=>JSON.stringify(AT.model));
  const ids=await page.locator('.pipeline-lesson').evaluateAll(es=>es.map(e=>e.id));
  const manifest=JSON.parse(fs.readFileSync(new URL('../notebooks/wordlm/lesson-manifest.json',import.meta.url)));
  assert.equal(ids.length,manifest.length+1);
  assert.equal(await page.locator('template[id^="pipeline-"]').count(),4);
  assert.equal(await page.locator('.pipeline-lesson script[type="text/x-notes"]').count(),ids.length);
  assert.match(await page.locator('#s19-pipeline-boundaries').innerText(),/7 supervised targets/);
  for(const id of ['pair-append','pair-append-second','pairs-loop']){
    const code=await page.locator('#s19-pipeline-'+id+' pre').innerText();
    assert(code.includes('contexts.append(context_ids)'),id+' exposes input storage');
    assert(code.includes('targets.append(target_id)'),id+' exposes target storage');
  }
  assert.match(await page.locator('#s19-pipeline-one-pair pre').innerText(),/target_id = ids\[target_position\]/);
  assert.match(await page.locator('#s19-pipeline-pairs-tensors').innerText(),/torch.long/);
  const batchText=await page.locator('#s19-pipeline-batch svg text').allTextContents();
  assert.deepEqual(batchText.slice(0,6),['Data row','Batch row','X: input IDs','Input tokens','y: ID','Target token']);
  assert.deepEqual(batchText.slice(6,12),['2','0','[0, 1, 8, 7]','<PAD> <BOS> lily found','5','a']);
  assert.deepEqual(batchText.slice(12,18),['3','1','[1, 8, 7, 5]','<BOS> lily found a','9','red']);
  assert.match(await page.locator('#s19-pipeline-batch-ids').innerText(),/Tokenization and vocabulary lookup are complete/);
  assert.match(await page.locator('#s19-pipeline-batch-ids').innerText(),/X.shape = \(2, 4\).*y.shape = \(2,\)/s);
  assert.match(await page.locator('#s19-pipeline-batch-ids .step-copy').innerText(),/no embedding coordinates yet.*X goes through embedding lookup.*y stays as target IDs for the loss/s);
  assert.equal(await page.locator('#s19-pipeline-batches').getAttribute('data-title'),'Seven examples, four batches');
  assert.match(await page.locator('#s19-pipeline-batches .step-copy').innerText(),/With one update per batch.*four optimizer steps.*512 windows per update with replacement/s);
  const dimensionText=await page.locator('#s19-pipeline-shapes svg text').allTextContents();
  assert.deepEqual(dimensionText,[
    'Data dimensions','Symbol','Value','What it counts',
    'B','2','examples per batch','w','4','input slots per example','C','10','vocabulary items',
    'Model dimensions','Symbol','Value','What it counts',
    'd','4','embedding coordinates','h','8','prediction-head hidden units',
    'dₖ','3','query and key coordinates','dᵥ','2','value coordinates',
  ],'each dimension has its own symbol, value and definition');
  assert(!dimensionText.some(text=>text.includes('/')),'dimension entries do not look like division');
  assert.equal(await page.locator('#s19 .map-checkpoint').count(),20,'recurring maps are visible lecture frames');
  for(const [route,detail,focus] of [
    ['route-data','data','stories'],['route-split','split','split'],
    ['route-tokenize','tokenize','tokenize'],['route-boundaries','boundaries','ids'],
    ['route-batch','batch','windows'],['route-flatten','flatten','flatten'],
    ['route-hidden','hidden-affine','hidden'],['route-logits','vocab-head','logits'],
    ['route-message','mix','message'],['route-backward','gradient','backward'],
  ]){
    const selector='#s19-pipeline-'+route;
    assert.equal(await page.locator(selector+' .master svg').getAttribute('data-focus'),focus);
    assert.equal(await page.locator(selector).evaluate(el=>el.nextElementSibling.id),'s19-pipeline-'+detail);
    assert.equal(await page.locator(selector+' pre').count(),0,'map checkpoint has no dummy code');
  }
  assert.match(await page.locator('#s19-pipeline-lookup-flow').innerText(),/T: 10 rows × 4 coordinates/);
  assert.match(await page.locator('#s19-pipeline-lookup-flow').innerText(),/X \[2,4\] indexes T \[10,4\] to produce E \[2,4,4\]/);
  const markerProblems=await page.locator('.pipeline-lesson .master svg').evaluateAll(svgs=>svgs.flatMap(svg=>{
    const ids=new Set([...svg.querySelectorAll('marker')].map(m=>m.id));
    return [...svg.querySelectorAll('[marker-end]')].filter(p=>{
      const id=p.getAttribute('marker-end').slice(5,-1);
      return !ids.has(id)||document.querySelectorAll('[id="'+id+'"]').length!==1;
    }).map(p=>p.getAttribute('marker-end'));
  }));
  assert.deepEqual(markerProblems,[],'arrow markers belong uniquely to the visible SVG');
  assert(!((await page.locator('#s19-pipeline-windows-last pre').innerText()).includes('torch.tensor')),'window example does not jump to tensor conversion');
  const codeBlocks=page.locator('pre > code');
  assert.equal(await codeBlocks.count(),await page.locator('pre > code.python-code').count(),'all Part II Python blocks have offline highlighting');
  const dynamicProbe = await page.evaluate(() => {
    const source = 'name = "<script>&x</script>"\n# preserve < and >\nif name:\n    print(1.5, name)\n';
    const code = AT.pythonCode(source);
    return {same:code.textContent === source, scripts:code.querySelectorAll('script').length,
      classes:['keyword','string','number','call','comment','operator'].every(k=>code.querySelector('.py-'+k))};
  });
  assert.deepEqual(dynamicProbe,{same:true,scripts:0,classes:true},'live snippets preserve source safely and use the same syntax classes');
  const liveCode = await page.evaluate(() => {
    const snippets = [], missing = [];
    for (const id of ['s15-stepper','s16-flow-stepper']) {
      const root = document.getElementById(id), api = root.querySelector('.stepper').stepperApi;
      const previous = api.index();
      for (let i=0;i<api.steps.length;i++) {
        api.go(i);
        for (const pre of root.querySelectorAll('pre')) {
          snippets.push(pre.textContent);
          if (!pre.querySelector('code.python-code')) missing.push(id+':'+i);
        }
      }
      api.go(previous);
    }
    return {snippets,missing};
  });
  assert.deepEqual(liveCode.missing,[],'every live step retains its highlighting');
  assert(liveCode.snippets.length>=30,'inspect all token and matrix-flow steps');
  const syntax=spawnSync('python3',['-c','import ast,json,sys; [compile(ast.parse(s), "<live slide>", "exec") for s in json.load(sys.stdin)]'],{input:JSON.stringify(liveCode.snippets),encoding:'utf8'});
  assert.equal(syntax.status,0,syntax.stderr);
  assert.equal(await page.locator('#s19-pipeline-counts pre').count(),0,'totals summary does not introduce a comprehension');
  assert.equal(await page.locator('#s19-pipeline-counts-train pre').innerText(),
    'train_tokens = 964_338\ntrain_stories = 4_822\ntrain_examples = train_tokens + train_stories');
  assert(await page.locator('#s19-pipeline-counts-train .py-number').count(),'count literals are highlighted');
  const codeColors=await page.locator('#s19-pipeline-generation-append code').evaluate(el=>({
    base:getComputedStyle(el).color,
    keyword:getComputedStyle(el.querySelector('.py-keyword')).color,
    call:getComputedStyle(el.querySelector('.py-call')).color,
  }));
  assert.notEqual(codeColors.base,codeColors.keyword,'keywords use a visible syntax color');
  assert.notEqual(codeColors.base,codeColors.call,'function calls use a visible syntax color');
  assert.match(await page.locator('#s19-pipeline-mask script').textContent(),/no future columns/);
  for(const id of ['story-complete','story-excerpts']){
    const frame=page.locator('#s19-pipeline-'+id);
    assert.equal(await frame.locator('pre').count(),0,'story slides prioritize reading the data');
    assert.equal(await frame.locator('.story-credit a').count(),2,'dataset and license attribution');
  }
  assert.match(await page.locator('#s19-pipeline-story-complete').innerText(),/52 words · 60 tokens/);
  assert.match(await page.locator('#s19-pipeline-story-excerpts').innerText(),/248 words · 300 tokens/);
  assert.equal(await page.locator('#s19-pipeline-tokenization-intro.lecture-topic-break h3').innerText(),'Tokenization');
  assert.equal(await page.locator('.tokenization-lesson').count(),3);
  assert.equal(await page.locator('.tokenization-lesson pre').count(),0,'conceptual detour precedes implementation');
  assert.match(await page.locator('#s19-pipeline-tokenization-choices').innerText(),/Actual splits depend on the tokenizer/);
  assert.match(await page.locator('#s19-pipeline-tokenization-rules').innerText(),/training and generation/);
  const summary=JSON.parse(fs.readFileSync(new URL('../figures/wordlm-pipeline/benchmark-summary.json',import.meta.url)));
  for(const kind of ['mlp','attention'])assert.match(await page.locator('#s19-pipeline-benchmark').innerText(),new RegExp(summary.aggregate[kind].test_perplexity.mean.toFixed(2).replace('.','\\.')));
  await page.evaluate(()=>AT.present.enter());
  async function go(id){
    await page.evaluate(id=>{
      const el=document.getElementById(id),sec=el.closest('.sec');
      AT.present.go(sec.id,[...sec.querySelectorAll('.frame')].indexOf(el)+1,99);
    },id);
    await page.waitForTimeout(80);
  }
  for(const viewport of [{width:1280,height:720},{width:1024,height:768}]){
    await page.setViewportSize(viewport);
    for(const id of ids){
      await go(id);
      assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,id+' fits '+viewport.width);
      const outside=await page.locator('#'+id+' [data-stage]').evaluateAll(gs=>gs.flatMap(g=>{
        const r=g.querySelector('rect').getBBox();
        return [...g.querySelectorAll('text')].filter(t=>{const b=t.getBBox();return b.x<r.x-1||b.x+b.width>r.x+r.width+1;}).map(t=>t.textContent);
      }));
      assert.deepEqual(outside,[],id+' SVG labels stay inside their nodes');
      const clipped=await page.locator('#'+id+' .step-figure svg').evaluateAll(es=>es.flatMap(s=>{
        const v=s.viewBox.baseVal;
        return [...s.querySelectorAll('text')].filter(t=>{const b=t.getBBox();return b.x<v.x-1||b.y<v.y-1||b.x+b.width>v.x+v.width+1||b.y+b.height>v.y+v.height+1;}).map(t=>t.textContent);
      }));
      assert.deepEqual(clipped,[],id+' SVG text stays within the figure');
      const stage=manifest.find(s=>'s19-pipeline-'+s.id===id);
      if(stage)assert.equal(await page.evaluate(()=>AT.present.state().frame.index+1),stage.slide,'notebook link matches slide index');
      const button=page.locator('#'+id+' .pipeline-toggle');
      if(await button.count()){
        const svg=page.locator('#'+id+' .pipeline-map svg');const before=await svg.getAttribute('viewBox');
        await button.click();assert.equal(await button.getAttribute('aria-expanded'),'true');
        assert.notEqual(await svg.getAttribute('viewBox'),before);
        assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,id+' expanded');
        await button.click();assert.equal(await svg.getAttribute('viewBox'),before);
      }
      await page.screenshot({path:path.join(shots,id+'-'+viewport.width+'.png')});
    }
  }
  assert.equal(await page.evaluate(()=>JSON.stringify(AT.model)),original,'diagrams do not mutate the hand-chosen lecture model');
  assert.equal(await page.locator('.pipeline-lesson .katex-error').count(),0);
  await page.goto(url);await page.setViewportSize({width:390,height:844});
  await page.waitForTimeout(150);
  assert(await page.locator('#s19-pipeline-tokenization-intro .tokenization-mobile').isVisible(),'intro prose wraps on phones');
  assert(!(await page.locator('#s19-pipeline-tokenization-intro svg').isVisible()),'wide intro SVG has a readable phone equivalent');
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'mobile overflow stays inside diagrams');
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({ok:true,frames:ids.length,viewports:[1280,1024,390],screenshots:shots}));
}finally{await browser.close();}
