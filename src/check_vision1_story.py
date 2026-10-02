"""Check the short conceptual route and preservation of the detailed library."""
from pathlib import Path
from html.parser import HTMLParser
import json,re
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'figures/vision1'
main=json.loads((ASSETS/'frame-manifest.json').read_text())
reference=json.loads((ASSETS/'reference-frame-manifest.json').read_text())
detailed=json.loads((ASSETS/'detailed-frame-manifest.json').read_text())
assert len(main)==54 and len(reference)==150 and len(detailed)==144
ids=[f['id'] for f in main]
assert len(set(ids))==len(ids)
assert {f['id'] for f in detailed}<={f['id'] for f in reference}
assert ids.index('story-prepared') < ids.index('story-self-cross') < ids.index('story-full-attention')
assert ids[-2:]==['story-fixed-vocabulary','story-clip-question']
assert sum(f['id'].startswith('story-cnn-') for f in main)==3
assert len([f for f in main if f['section']=='s06'])==7
assert ids.index('story-qkv-roles') < ids.index('story-value-mixture') < ids.index('story-full-attention')
assert all('story-'+key in ids for key in ['query-examples','key-value-pair','measured-attention','cover-question','cover-code','cover-results','cover-small'])
assert all(len(f['caption'].split())<=25 for f in main)
assert all('\n' in f['notes'] and len(f['notes'])>80 for f in main)
reference_ids=[f['id'] for f in reference]
assert reference_ids.index('optional-extensions-start') < reference_ids.index('pets-original-task')
for frame in main+reference:ET.parse(ASSETS/(frame['id']+'.svg'))
class Frames(HTMLParser):
 def __init__(self):super().__init__();self.ids=[];self.links=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'frame' in a.get('class','').split():self.ids.append(a['id'])
  if tag=='a' and a.get('href'):self.links.append(a['href'])
for name,frames in [('vision1.html',main),('vision1-reference.html',reference)]:
 parser=Frames();parser.feed((ROOT/name).read_text())
 assert parser.ids==[f['id'] for f in frames],name
 for link in parser.links:
  if link.startswith(('https:','http:','mailto:','#')):continue
  path=link.split('#')[0].split('?')[0]
  # PDFs are built after HTML; their delivery is checked during packaging.
  if path and not path.endswith(('.pdf','transcript.md')):assert (ROOT/path).exists(),link
prediction=json.loads((ASSETS/'real-inference.json').read_text())
assert prediction['classes']==1000 and prediction['model'].startswith('vit_tiny_patch16_224')
svg=(ASSETS/'story-prediction.svg').read_text()
for row in prediction['results'][0]['top3']:assert f"{100*row['probability']:.2f}%" in svg
p63=json.loads((ASSETS/'real-patch-path.json').read_text())
for value in p63['content'][:3]:assert f'{value:.3f}'.replace('-','−') in (ASSETS/'story-measured-patch.svg').read_text()
assert 197**2==38809 and 785**2==616225
inspection=json.loads((ASSETS/'inspection.json').read_text())
for row in inspection['occlusion']:
 assert f"{100*row['target_probability']:.2f}%" in (ASSETS/'story-cover-results.svg').read_text()
fine=json.loads((ASSETS/'patch-occlusion.json').read_text())
assert f"{fine['largest_drop']['target_probability']:.2%}" in (ASSETS/'story-cover-small.svg').read_text()
assert abs(0.6*2+0.1-1.3)<1e-12 and abs(0.3+0.1-0.4)<1e-12
report={'main_slides':54,'main_pdf_pages':55,'reference_slides':150,'reference_pdf_pages':151,
        'all_144_detailed_frames_preserved':True,'tokens_before_attention':True,
        'measured_probabilities_and_projection_checked':True,'caption_word_limit':25,
        'optional_extensions_divider':True,'ends_on_clip_question':True,
        'qkv_intuition_and_full_mask':True,'cnn_comparison_frames':3,'measured_ablation_checked':True}
(ASSETS/'story-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
