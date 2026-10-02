"""Package both rendered decks, bookmarks and searchable audit transcripts.

After running export_slides.mjs for the main and reference HTML, run this with
an existing Python environment containing pypdf and Pillow. No model runs.
"""
from pathlib import Path
from html.parser import HTMLParser
import argparse,hashlib,json,re
from PIL import Image,ImageChops,ImageStat
from pypdf import PdfReader,PdfWriter
from vision1_explorer_tour import EXAMPLES

ROOT=Path(__file__).resolve().parents[1]
class Frames(HTMLParser):
    void={'img','image','input','br','hr','meta','link','source','wbr','area','base','col','embed','param','track'}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack=[];self.frames={};self.active=None;self.depth=None
    def handle_starttag(self,tag,attrs):
        a=dict(attrs);classes=a.get('class','').split()
        if 'frame' in classes:self.active=a.get('id');self.depth=len(self.stack);self.frames[self.active]=[]
        hidden=tag in {'script','style','title','defs'} or 'vp-mobile' in classes or 'vp-pathbar' in classes
        hidden=hidden or (self.stack and self.stack[-1][1])
        if self.active and not hidden and tag in {'p','li','pre','text','h3','h4','tr','br','div','button','option'}:self.frames[self.active].append('\n')
        if tag not in self.void:self.stack.append((tag,bool(hidden)))
    def handle_startendtag(self,tag,attrs):
        self.handle_starttag(tag,attrs)
        if tag not in self.void:self.handle_endtag(tag)
    def handle_endtag(self,tag):
        if tag in self.void:return
        for i in range(len(self.stack)-1,-1,-1):
            if self.stack[i][0]==tag:del self.stack[i:];break
        if self.active and len(self.stack)<=self.depth:self.active=None;self.depth=None
    def handle_data(self,text):
        if self.active and not (self.stack and self.stack[-1][1]):self.frames[self.active].append(text)


def package(name,manifest_name,config_name,rendered,frames_dir,expected):
    manifest=json.loads((ROOT/'figures/vision1'/manifest_name).read_text())
    config=json.loads((ROOT/'src'/config_name).read_text())
    html=(ROOT/(name+'.html')).read_text()
    parser=Frames();parser.feed(html)
    pages=[dict(id='cover',title=config['title'],section=None,frame=None,page=1)]+[dict(f,page=i+2) for i,f in enumerate(manifest)]
    images=sorted((ROOT/frames_dir).glob('*.png'))
    writer=PdfWriter(clone_from=ROOT/rendered)
    assert len(writer.pages)==len(pages)==len(images)==expected
    for page,path in zip(writer.pages,images):
        with Image.open(path) as im:page.images[0].replace(im.convert('RGB'),quality=90,optimize=True)
    section=None;parent=None
    for i,p in enumerate(pages):
        if p['section'] and p['section']!=section:
            section=p['section'];title=config['sections'][int(section[1:])-1]['title']
            parent=writer.add_outline_item(title,i)
        writer.add_outline_item(p['title'],i,parent=parent if p['section'] else None)
    writer.add_metadata({'/Title':config['title'],'/Author':'Nipun Batra','/Subject':f'{len(manifest)} content slides plus cover. Question-led Vision I revision.'})
    target=ROOT/'pdf'/(name+'.pdf');writer.write(target)
    reader=PdfReader(target,strict=True);max_error=0
    for i,(page,path) in enumerate(zip(reader.pages,images)):
        assert (float(page.mediabox.width),float(page.mediabox.height))==(960.,540.)
        actual=page.images[0].image.convert('RGB').resize((256,144))
        with Image.open(path) as im:expected_image=im.convert('RGB').resize((256,144))
        error=sum(ImageStat.Stat(ImageChops.difference(actual,expected_image)).mean)/3
        assert error<4,(i+1,error)
        max_error=max(max_error,error)
    text=['# '+config['title']+' — audit transcript','',
          f'Companion PDF: [{name}.pdf]({name}.pdf) · **{len(pages)} pages**.','',
          'Main lecture: 54 conceptual slides plus cover. The optional reference deck preserves the detailed material separately. Final reveals are captured in the PDF; progressive builds and interactions remain in HTML.','',
          'Diagram labels are listed in source order; use the PDF to judge spatial layout. Speaker notes contain the detail omitted from the projected slide.','',
          '## Page index','','| PDF page | Route | Title |','|---:|---|---|']
    for p in pages:
        route=f'#{p["section"]}/{p["frame"]}' if p['section'] else 'Cover'
        text.append(f'| {p["page"]} | {route} | {p["title"]} |')
    text+=['','## Transcript','']
    for p in pages:
        text += [f'### Page {p["page"]} — {p["title"]}','']
        if p['id']=='cover':text += [config['subtitle'],''];continue
        content=re.sub(r'\n[ \t]*\n+','\n',''.join(parser.frames[p['id']])).strip()
        text+=['```text',content,'```','','**Caption:** '+p['caption'],'','**Speaker notes**','',p['notes'],'']
    if name.endswith('reference'):
        text+=['## Interactive lab — all nine examples','']
        for i,e in enumerate(EXAMPLES,1):
            text += [f'### Example {i}: {e["title"]}',f'{e["mode"]} · block {e["block"]} · head {e["head"]} · query {e["query"]} · source {e["source"]}','',e['look'],'',e['takeaway'],'',e['detail'],'']
    transcript='\n'.join(line.rstrip() for line in '\n'.join(text).splitlines()).rstrip()+'\n'
    (ROOT/'pdf'/(name+'-transcript.md')).write_text(transcript)
    out=ROOT/'output/audit';out.mkdir(exist_ok=True,parents=True)
    (out/(name+'-page-index.json')).write_text(json.dumps(pages,indent=2,ensure_ascii=False)+'\n')
    report=dict(name=name,pages=len(pages),bytes=target.stat().st_size,max_mean_rgb_error=max_error,
                all_page_images_checked=True,html_sha256=hashlib.sha256(html.encode()).hexdigest(),
                pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (out/(name+'-verification.json')).write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--render-prefix',default='tmp/pdfs/vision1-question');args=ap.parse_args()
    reports=[]
    for name,manifest,config,suffix,count in [('vision1','frame-manifest.json','part5.json','main',55),
            ('vision1-reference','reference-frame-manifest.json','vision1-reference.json','reference',151)]:
        reports.append(package(name,manifest,config,args.render_prefix+'-'+suffix+'.pdf',args.render_prefix+'-'+suffix+'-frames',count))
    print(json.dumps(reports,indent=2))
