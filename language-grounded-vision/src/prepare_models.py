from pathlib import Path
import json
from huggingface_hub import snapshot_download
R=Path(__file__).resolve().parents[1]
info=[]
for name,rev in [('IDEA-Research/grounding-dino-tiny','a2bb814dd30d776dcf7e30523b00659f4f141c71'),('facebook/sam-vit-base','70c1a07f894ebb5b307fd9eaaee97b9dfc16068f')]:
 print('Downloading',name,rev,flush=True)
 dest=R/'output/model-cache'/name.split('/')[-1]
 snapshot_download(name,revision=rev,local_dir=dest,token=False,allow_patterns=['*.json','*.txt','*.safetensors','*.model'],max_workers=4)
 info.append({'id':name,'revision':rev,'local_dir':str(dest)})
 (R/'output/model-cache/models.json').write_text(json.dumps(info,indent=2))
print('Models ready',flush=True)
