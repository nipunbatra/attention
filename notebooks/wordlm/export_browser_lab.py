"""Export actual checkpoints, audited example prompts and numerical parity probes."""
import argparse
from pathlib import Path
import json
import shutil
import numpy as np
import torch
import onnx
import onnxruntime as ort
from run_head_comparison import build_model
from wordlm import (build_corpus, read_json, write_json, load_model_npz, sha256_file,
                    tokenize, detokenize)

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parents[1]/'word-lab'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    args=parser.parse_args()
    corpus=build_corpus(args.data_dir,'dgx')
    vocab=corpus.vocab
    results=read_json(ROOT/'artifacts/heads/comparison.json')
    assert 'aggregate' in results, 'Complete all benchmark runs before exporting'
    destination=LAB/'models';destination.mkdir(parents=True,exist_ok=True)
    prompts=[]
    # Choose by source row, not by model output. Reuse stories already shown in class.
    for row,label in [(588307,'Sara and Tom'),(1992490,'The old hotel')]:
        story=next(s for s in corpus.stories['train'] if s.row_idx==row)
        prompts.append(dict(id=f'train-{row}',label=f'Training story: {label}',
                            prompt=detokenize(story.tokens[:20]),source_text=story.text,
                            row_idx=row,split='train',token_count=len(story.tokens),
                            note='This complete story was in the training split. Continue its first 20 tokens. This is not a held-out quality test.'))
    heldout=min(corpus.stories['test'],key=lambda s:len(s.tokens))
    prompts.append(dict(id='heldout',label='Held-out story: shortest test document',
                        prompt=detokenize(heldout.tokens[:20]),source_text=heldout.text,
                        row_idx=heldout.row_idx,split='test',token_count=len(heldout.tokens),
                        note='This story was not used in training or checkpoint selection. It still resembles the training stories.'))
    for name,prompt,note in [
        ('new-story','a little dog found a red hat under the table', 'An authored story opening. Its exact token sequence is absent from all 6,000 source stories.'),
        ('instructions','first put the milk in the pot and then turn off the heat', 'An authored cooking instruction. It uses familiar words in a different kind of text.'),
        ('science','the quantum processor measures entanglement in superconducting qubits', 'An authored technical sentence. Several words are outside this vocabulary and become UNK.')]:
        tokens=tokenize(prompt)
        assert all(not any(s.tokens[i:i+len(tokens)]==tokens for i in range(len(s.tokens)-len(tokens)+1)) for split in corpus.stories.values() for s in split)
        prompts.append(dict(id=name,label={'new-story':'New story opening','instructions':'Outside stories: cooking instructions','science':'Outside stories: quantum computing'}[name],prompt=prompt,note=note))
    for p in prompts:
        p['tokens']=tokenize(p['prompt'])
        p['token_ids']=vocab.encode_tokens(p['tokens'],boundaries=False)
        p['unknown_tokens']=sorted(set(t for t in p['tokens'] if t not in vocab.stoi))
    write_json(LAB/'examples.json',dict(prompts=prompts,source=corpus.manifest,
               selection='Two pre-existing lecture stories, shortest test story and three authored prompts. No selection based on generated outputs.',license='CDLA-Sharing-1.0'))
    write_json(destination/'vocab.json',dict(itos=vocab.itos))
    shutil.copy2(ROOT/'artifacts/heads/comparison.json',destination/'comparison.json')
    probes=[]
    for ids in [[1],prompts[0]['token_ids'],list(range(4,100))]:
        ids=ids[-64:];probes.append([0]*(64-len(ids))+ids)
    input_ids=np.asarray(probes,dtype=np.int32)
    metadata=dict(context_length=64,vocabulary_size=len(vocab.itos),format='ONNX FP32',
                  model_seed=11,positional_encoding=results['protocol']['positional_encoding'],models={})
    saved=[]
    for kind in ['mlp','attention','multihead']:
        model=build_model(kind,len(vocab.itos));checkpoint=ROOT/f'artifacts/heads/{kind}_seed11.npz'
        checkpoint_meta,_=load_model_npz(checkpoint,model);model.eval()
        path=destination/f'{kind}.onnx'
        with torch.inference_mode():
            expected=model(torch.from_numpy(input_ids)).numpy()
        torch.onnx.export(model,(torch.from_numpy(input_ids),),path,
                          input_names=['input_ids'],output_names=['logits'],
                          dynamic_axes={'input_ids':{0:'batch'},'logits':{0:'batch'}},
                          opset_version=18,do_constant_folding=True,dynamo=False)
        onnx.checker.check_model(onnx.load(path))
        session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
        observed=session.run(['logits'],{'input_ids':input_ids})[0]
        error=float(np.max(np.abs(expected-observed)))
        assert error<2e-4,(kind,error)
        run=results['runs'][kind][0]
        metadata['models'][kind]=dict(file=f'./models/{kind}.onnx',bytes=path.stat().st_size,
              sha256=sha256_file(path),checkpoint_sha256=sha256_file(checkpoint),
              parameters=run['parameter_count'],selected_step=checkpoint_meta['selected_step'],
              test_cross_entropy=run['test_loss'],test_perplexity=run['test_perplexity'],
              onnx_cpu_max_absolute_error=error)
        # Full logits make the browser test stronger than a single argmax check.
        write_json(destination/f'{kind}-parity.json',dict(input_ids=probes,logits=expected.tolist()))
        for p in prompts:
            history=[1]+p['token_ids'];generated=[]
            for _ in range(24):
                row=history[-64:];row=[0]*(64-len(row))+row
                with torch.inference_mode():
                    logits=model(torch.tensor([row],dtype=torch.int32))[0]
                    logits[[0,1,3]]=float('-inf');chosen=int(logits.argmax())
                if chosen==2:break
                history.append(chosen);generated.append(chosen)
            saved.append(dict(prompt_id=p['id'],model=kind,decoding='greedy',max_new_tokens=24,
                              generated_ids=generated,continuation=detokenize(vocab.decode_ids(generated))))
    write_json(destination/'metadata.json',metadata)
    write_json(LAB/'saved-examples.json',dict(note='Recorded greedy continuations for lecture discussion. The demo computes fresh logits and does not load this file.',examples=saved))
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
