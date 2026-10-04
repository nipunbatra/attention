"""Record real Grounding DINO + SAM predictions. No target annotation enters inference."""
from pathlib import Path
import json,time,hashlib,platform,datetime
import numpy as np
import torch,transformers
from PIL import Image
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection,SamProcessor,SamModel
R=Path(__file__).resolve().parents[1]
torch.set_num_threads(4);torch.manual_seed(0)
models=json.loads((R/'output/model-cache/models.json').read_text())
D,S=models
print('Loading public checkpoints on CPU',flush=True)
dp=AutoProcessor.from_pretrained(D['local_dir'],local_files_only=True)
dm=AutoModelForZeroShotObjectDetection.from_pretrained(D['local_dir'],local_files_only=True).eval()
sp=SamProcessor.from_pretrained(S['local_dir'],local_files_only=True)
sm=SamModel.from_pretrained(S['local_dir'],local_files_only=True).eval()
record={'schema_version':2,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':'recorded model inference; separate from authored references','models':[{k:v for k,v in m.items() if k!='local_dir'} for m in models],'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'platform':platform.platform(),'device':'cpu','dtype':'float32','threads':4,'seed':0},'settings':{'box_threshold':.25,'text_threshold':.25,'sam_multimask_output':False,'sam_mask_threshold':0.0,'box_format':'xyxy original-image pixels','timing':'wall seconds, CPU inference + postprocessing; model loading excluded; SAM encoder timing stored separately','notes':'All returned detections are retained. SAM receives each detected box. No authored target labels are supplied to either model.'},'images':[],'runs':[]}
record['preprocessing']={'grounding_dino':dp.image_processor.to_dict(),'sam':sp.image_processor.to_dict()}
def save():
 p=R/'results/grounded_vision_outputs.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(p)
def rle(mask):
 # Alternating background/foreground counts in row-major order, starting with background.
 a=mask.reshape(-1).astype(np.uint8);ch=np.flatnonzero(a[1:]!=a[:-1])+1;counts=np.diff(np.r_[0,ch,a.size]).tolist()
 if a[0]:counts=[0]+counts
 return {'size':list(mask.shape),'order':'C','counts':counts}
for scene,prompts in [('scene',['dog.','red ball.','ball beside the dog.','dog beside the red ball.','bicycle.']),('scene-swapped',['red ball.','ball beside the dog.'])]:
 p=R/'figures/inputs'/f'{scene}.png';im=Image.open(p).convert('RGB')
 img={'id':scene,'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'width':im.width,'height':im.height,'provenance':'authored fixed park illustration; clean image with no reference overlays'}
 start=time.perf_counter();si=sp(images=im,return_tensors='pt')
 with torch.inference_mode():emb=sm.get_image_embeddings(si['pixel_values'])
 img['sam_image_encoder_seconds']=time.perf_counter()-start;record['images'].append(img)
 for prompt in prompts:
  print(scene,prompt,flush=True);start=time.perf_counter()
  di=dp(images=im,text=prompt,return_tensors='pt')
  with torch.inference_mode():do=dm(**di)
  det=dp.post_process_grounded_object_detection(do,di.input_ids,threshold=.25,text_threshold=.25,target_sizes=[(im.height,im.width)])[0]
  boxes=det['boxes'].tolist();ds=time.perf_counter()-start
  run={'image_id':scene,'prompt':prompt,'status':'ok','grounding_seconds':ds,'detections':[],'segmentation_seconds':None,'raw_text_labels':det.get('text_labels',det.get('labels',[]))}
  if boxes:
   start=time.perf_counter();bi=sp(images=im,input_boxes=[boxes],return_tensors='pt')
   with torch.inference_mode():so=sm(image_embeddings=emb,input_boxes=bi['input_boxes'],multimask_output=False)
   masks=sp.image_processor.post_process_masks(so.pred_masks,bi['original_sizes'],bi['reshaped_input_sizes'])[0][:,0].cpu().numpy()
   run['segmentation_seconds']=time.perf_counter()-start
   for i,(box,score,mask) in enumerate(zip(boxes,det['scores'].tolist(),masks)):
    run['detections'].append({'box_xyxy':box,'text':str(run['raw_text_labels'][i]),'grounding_score':score,'score_kind':'maximum sigmoid text-token alignment score; not calibrated correctness probability','mask_rle':rle(mask),'sam_predicted_iou':so.iou_scores[0,i,0].item(),'sam_predicted_iou_kind':'model quality estimate, not measured overlap with a ground-truth mask'})
  record['runs'].append(run);save();print('saved',len(boxes),'detections',round(ds,2),'s',flush=True)
print('Recorded',len(record['runs']),'runs',flush=True)
