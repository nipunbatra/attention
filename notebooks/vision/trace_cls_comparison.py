"""Compare CLS activations for the lecture's dog and cat with fixed weights.

HF_HUB_OFFLINE=1 uv run --offline --with timm --with pillow python notebooks/vision/trace_cls_comparison.py
Uses the cached checkpoint. Inference only; no training or notebook execution.
"""
from pathlib import Path
import hashlib
import json
import torch
import timm
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT/'figures/vision1'
MODEL = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
torch.set_num_threads(4)
model = timm.create_model(MODEL, pretrained=True).eval()
transform = create_transform(**resolve_model_data_config(model), is_training=False)
parameter_before = model.cls_token.detach().clone()
references = json.loads((ASSETS/'real-inference.json').read_text())['results']
dog_trace = json.loads((ASSETS/'real-classifier-path.json').read_text())
rows = []

def values(tensor):
    return tensor.reshape(-1).tolist()

with torch.inference_mode():
    for ref in references:
        path = ASSETS/(ref['image_id']+'.jpg')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256']
        x = transform(Image.open(path).convert('RGB')).unsqueeze(0)
        E = model._pos_embed(model.patch_embed(x))
        torch.testing.assert_close(E[:,0], model.cls_token[:,0]+model.pos_embed[:,0])
        block = model.blocks[0]
        normalized = block.norm1(E)
        qkv = block.attn.qkv(normalized).reshape(1,197,3,3,64)
        cls_query = qkv[0,0,0]
        after_attention = E + block.attn(normalized)
        final = model.norm(model.blocks(E))
        logits = model.head(final[:,0])
        torch.testing.assert_close(logits,model(x),atol=5e-5,rtol=1e-5)
        probability = logits.softmax(-1)
        best = ref['top3'][0]
        assert int(probability.argmax()) == best['index']
        assert abs(float(probability[0,best['index']])-best['probability']) < 1e-5
        rows.append({
            'image_id':ref['image_id'], 'source_sha256':ref['sha256'],
            'cls_input':values(E[:,0]), 'first_cls_query_all_heads':values(cls_query),
            'cls_after_attention1_residual':values(after_attention[:,0]),
            'cls_final_after_norm':values(final[:,0]),
            'top_label':best['label'], 'top_probability':float(probability[0,best['index']]),
        })
    torch.testing.assert_close(model.cls_token,parameter_before,atol=0,rtol=0)

assert rows[0]['cls_input'] == rows[1]['cls_input']
assert rows[0]['first_cls_query_all_heads'] == rows[1]['first_cls_query_all_heads']
assert rows[0]['cls_after_attention1_residual'] != rows[1]['cls_after_attention1_residual']
assert rows[0]['cls_final_after_norm'] != rows[1]['cls_final_after_norm']
for field, old in [('cls_input','cls_input'),('cls_after_attention1_residual','cls_after_attention'),
                   ('cls_final_after_norm','cls_final')]:
    torch.testing.assert_close(torch.tensor(rows[0][field][:3]),
                               torch.tensor(dog_trace['previews'][old]),atol=2e-5,rtol=1e-5)

report = {
    'model':MODEL, 'torch':torch.__version__, 'timm':timm.__version__,
    'embedding_width':192, 'patch_rows':196, 'rows_with_cls':197,
    'cls_parameter':values(parameter_before), 'cls_position':values(model.pos_embed[:,0].detach()),
    'results':rows,
    'verification':{
        'source_photos_match_saved_hashes':True, 'same_starting_input':True,
        'same_first_query':True, 'different_after_attention':True, 'different_final_summary':True,
        'stored_parameter_unchanged':True, 'dog_matches_previous_trace':True,
        'class_predictions_match_saved_inference':True, 'training_run':False,
    },
}
(ASSETS/'cls-two-image-trace.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'verification':report['verification'],'previews':[
    {k:(v[:3] if isinstance(v,list) else v) for k,v in row.items() if k!='source_sha256'}
    for row in rows]},indent=2))
