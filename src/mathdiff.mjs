// node mathdiff.mjs before.html after.html
// Verifies that a prose-only edit did not touch: <script> blocks, <style> blocks, $...$ / $$...$$ math, AT.tex(...) latex
// literals, numbers, ids/classes/data-attributes. Also counts humanizer red flags in the AFTER prose. Exit 1 on any violation.
import fs from 'fs';
const [a, b] = process.argv.slice(2);
const A = fs.readFileSync(a, 'utf8'), B = fs.readFileSync(b, 'utf8');
const grab = (s, re) => (s.match(re) || []).map(x => x.trim());
const same = (name, ra, rb) => { const ok = JSON.stringify(ra) === JSON.stringify(rb); if (!ok) { console.log('DIFF in ' + name + ': before ' + ra.length + ' items, after ' + rb.length + ' items'); const sa = new Set(ra); const sb = new Set(rb); [...sa].filter(x => !sb.has(x)).slice(0, 5).forEach(x => console.log('  - removed: ' + x.slice(0, 160))); [...sb].filter(x => !sa.has(x)).slice(0, 5).forEach(x => console.log('  + added:   ' + x.slice(0, 160))); } return ok; };
let ok = true;
ok &= same('script blocks', grab(A, /<script[\s\S]*?<\/script>/g), grab(B, /<script[\s\S]*?<\/script>/g));
ok &= same('style blocks', grab(A, /<style[\s\S]*?<\/style>/g), grab(B, /<style[\s\S]*?<\/style>/g));
const stripScripts = s => s.replace(/<script[\s\S]*?<\/script>/g, '').replace(/<style[\s\S]*?<\/style>/g, '');
const pa = stripScripts(A), pb = stripScripts(B);
ok &= same('display math', grab(pa, /\$\$[\s\S]*?\$\$/g), grab(pb, /\$\$[\s\S]*?\$\$/g));
ok &= same('inline math', grab(pa.replace(/\$\$[\s\S]*?\$\$/g, ''), /\$[^$\n]+?\$/g), grab(pb.replace(/\$\$[\s\S]*?\$\$/g, ''), /\$[^$\n]+?\$/g));
ok &= same('ids', grab(pa, /\sid="[^"]*"/g), grab(pb, /\sid="[^"]*"/g));
ok &= same('classes', grab(pa, /\sclass="[^"]*"/g), grab(pb, /\sclass="[^"]*"/g));
ok &= same('data attributes', grab(pa, /\sdata-[a-z-]+="[^"]*"/g), grab(pb, /\sdata-[a-z-]+="[^"]*"/g));
ok &= same('numbers', grab(pa, /(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])/g), grab(pb, /(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])/g));
// humanizer red flags in AFTER prose (outside math)
const prose = pb.replace(/\$\$[\s\S]*?\$\$/g, '').replace(/\$[^$\n]+?\$/g, '').replace(/<[^>]+>/g, ' ');
const flags = [
  ['em/en dash', /[—–]| -- /g],
  ['not just X, but Y', /\bnot (just|only|merely|simply)\b[^.]{0,80}\b(but|it'?s|rather)\b/gi],
  ['fake-deep phrase', /\b(at its core|the real question|what really matters|fundamentally|the heart of the matter)\b/gi],
  ['announcing', /\b(let'?s (dive|explore|break|look)|here'?s what you need to know|without further ado)\b/gi],
  ['AI vocabulary', /\b(delve|crucial|pivotal|testament|tapestry|landscape|showcase|underscore[sd]?|leverage|seamless|robust|elegant|intricate|vibrant|foster(s|ing)?|enhance[sd]?|garner)\b/gi],
  ['emoji', /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu],
  ['bold mini-heading list', /<(li|p)[^>]*>\s*<(b|strong)>[^<]{1,40}:<\/(b|strong)>/g],
];
const hits = [];
for (const [name, re] of flags) { const m = prose.match(re); if (m) hits.push(name + ' x' + m.length + ' (' + [...new Set(m)].slice(0, 4).join(' | ') + ')'); }
const scr = grab(B, /<script[\s\S]*?<\/script>/g).join('\n'); const sd = scr.match(/[—–]/g); if (sd) hits.push('em/en dash inside <script> string literals x' + sd.length + ' (rewrite the visible-text literals; code itself never needs a dash)');
if (hits.length) { console.log('HUMANIZER FLAGS remaining in prose: '); hits.forEach(h => console.log('  ' + h)); }
// also flag bold-mini-heading pattern in AFTER raw html
console.log(ok ? 'STRUCTURE OK: scripts, styles, math, ids, classes, data attributes and numbers unchanged' : 'STRUCTURE VIOLATION');
process.exit(ok ? 0 : 1);
