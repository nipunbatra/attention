"""Diagram-first companion cells; imported by make_notebooks.py."""


def add_maps(cells, kind, md, code):
    result=[]
    stages_mlp={"## 1.":("stories","split","tokenize","ids"),"## 2.":("windows",),
                "## 3.":("embedding","flatten","hidden","logits"),
                "## 4.":("loss","backward","optimizer"),"## 5.":("loss","backward","optimizer")}
    stages_att={"## 1.":("qkv","scores","weights","message","projection","residual"),
                "## 2.":("tokenize","ids","windows","embedding"),
                "## 3.":("embedding","qkv","scores","weights","message","projection","residual","readout","hidden","logits"),
                "## 4.":("weights","loss","backward"),"## 5.":("loss","backward","optimizer")}
    for cell in cells:
        result.append(cell)
        source=cell.source
        if cell.cell_type=="code" and 'print(f"torch=' in source:
            result.extend([md("""## The complete route

The two maps below are our table of contents. Training sends an observed target
to the loss. Inference appends a chosen token and leaves the model parameters fixed.
Later cells highlight the same map without rearranging it.

`B` = batch size, `w` = context slots, `d` = embedding width, `h` = hidden width,
`C` = vocabulary size. In generation, `B=1`. Open Notebook 5 for the window-by-window
trace, individual model operations, optimizer step and checkpoint inference.
"""),code(f"from pipeline_maps import show_pipeline\nshow_pipeline('{kind}', 'training')\nshow_pipeline('{kind}', 'inference')")])
        if cell.cell_type=="markdown":
            for heading,focus in (stages_mlp if kind=="mlp" else stages_att).items():
                if heading in source:
                    result.append(code(f"show_pipeline('{kind}', 'training', {focus!r})"));break
        if cell.cell_type=="code" and 'for temperature in' in source:
            result.pop()
            result.extend([md("""### The inference loop on the same map

Reuse the saved tokenizer and vocabulary. Crop long history to the last `w`
tokens, left-pad a shorter prefix, predict, choose a token and append it.
Stop on `<EOS>` or the token limit. This implementation recomputes the window;
it does not use a KV cache. No target, backward pass or optimizer step is involved.
"""),code(f"show_pipeline('{kind}', 'inference', ('windows','probabilities','choose','append','stop'))"),cell])
    return result


def notebook_5(md, code, setup):
    return [md("""# 5 · One example through training and inference

These are the same four maps used in the lecture. First see the whole route,
then keep its layout fixed while highlighting the code being executed.

We use a **short authored sentence** to expose every context/target pair.
That example is not evidence of language quality. At the end we load the real
TinyStories checkpoints and the frozen three-seed benchmark from Notebook 4.

The comparison has one observed target per window for both architectures.
`B` is the number of windows in a batch. `w` is the number of context slots.
`d`, `d_k`, `d_v`, `h`, and `C` are representation widths and vocabulary size.
"""),code(setup),code("""from pipeline_maps import show_pipeline
for architecture in ['mlp', 'attention']:
    print(architecture.upper(), 'TRAINING')
    show_pipeline(architecture, 'training')
    print(architecture.upper(), 'INFERENCE')
    show_pipeline(architecture, 'inference')
"""),md("""## 1. Text, tokenization and the split boundary

The real corpus workflow removes exact duplicates, splits **whole stories**, then
fits the vocabulary on training stories only. Overlapping windows stay inside
their story. Validation selects settings/checkpoints; test measures the frozen choice.

Here the single sentence is an authored tracing example. Its tiny vocabulary is
local to this example, not the vocabulary of either trained TinyStories model.
"""),code("""show_pipeline('mlp', 'training', ('stories','split','tokenize','ids'))
sentence = 'Lily found a red ball.'
pieces = tokenize(sentence)
toy_words = list(SPECIAL_TOKENS) + sorted(set(pieces))
toy_vocab = Vocabulary(toy_words, {t:i for i,t in enumerate(toy_words)}, {}, len(toy_words), 1)
ids = toy_vocab.encode_tokens(pieces)
print('text:', sentence)
print('tokens:', pieces)
print('vocabulary:', toy_vocab.stoi)
print('with boundaries:', toy_vocab.decode_ids(ids, skip_special=False))
print('IDs:', ids)
"""),md("""## 2. One observed next token per window

The target is **outside** its input context. A short history gets `<PAD>` on the
left; `<BOS>` is a real boundary token. A long history loses its oldest IDs.
`<EOS>` is a target the model can learn. `<PAD>` is not a training target.
"""),code("""show_pipeline('mlp', 'training', 'windows')
w = 4
contexts, targets = [], []
for t in range(1, len(ids)):
    visible = ids[max(0, t-w):t]
    context = [toy_vocab.pad_id] * (w-len(visible)) + visible
    contexts.append(context)
    targets.append(ids[t])
    print(toy_vocab.decode_ids(context, False), '->', toy_vocab.itos[ids[t]])
X = torch.tensor(contexts)
y = torch.tensor(targets)
assert X.shape == (len(pieces)+1, w)
assert not y.eq(toy_vocab.pad_id).any()
print('X:', X.shape, 'y:', y.shape)
"""),md("""### Short history, exactly full, and too long

The last `w` tokens form the **available context** of one prediction. The whole
story can be longer. Increasing `w` exposes more history, but cannot guarantee
the model uses it well. The MLP fixes `w*d` input connections when constructed.
Our attention model also has a configured maximum learned position table.
It does not acquire unlimited memory just because attention weights are dynamic.
"""),code("""for history in [ids[:2], ids[:4], ids[:6]]:
    kept = history[-w:]
    padded = [toy_vocab.pad_id] * (w-len(kept)) + kept
    print('history:', toy_vocab.decode_ids(history, False))
    print('dropped:', toy_vocab.decode_ids(history[:-w], False) if len(history)>w else [])
    print('model input:', toy_vocab.decode_ids(padded, False))
"""),md("""## 3. Embedding lookup and the MLP

Each token ID selects a row of the learned embedding table. Flattening keeps
each slot separate and in order. A ReLU hidden layer sits between that long
vector and the vocabulary logits. Logits are scores, not probabilities.
"""),code("""show_pipeline('mlp', 'training', ('embedding','flatten','hidden','logits'))
seed_everything(11)
mlp = FixedWindowMLP(len(toy_words), w, d_embed=4, hidden=8, pad_id=toy_vocab.pad_id)
embedded = mlp.token_embedding(X)        # [B,4,4]
joined = embedded.flatten(1)             # [B,16]
hidden = F.relu(mlp.hidden_layer(joined)) # [B,8]
logits = mlp.vocab_head(hidden)           # [B,C]
for name, value in [('IDs',X),('embeddings',embedded),('joined',joined),('hidden',hidden),('logits',logits)]:
    print(name, tuple(value.shape))
assert torch.allclose(logits, mlp(X))
"""),md("""## 4. The same IDs through one attention head

For this small trace choose `d=4`, `d_k=3`, `d_v=2`, and hidden width 8.
The final known token supplies a query. Every real context position supplies
a key and a value. Add position vectors before computing these projections.
"""),code("""show_pipeline('attention', 'training', ('embedding','qkv'))
attention = CausalAttentionLM(len(toy_words), w, 4, 3, 2, 8, toy_vocab.pad_id)
positions = torch.arange(w)
E = attention.token_embedding(X) + attention.position_embedding(positions)[None]
q = attention.W_Q(E[:, -1:, :])  # [B,1,3]
K, V = attention.W_K(E), attention.W_V(E) # [B,4,3], [B,4,2]
for name,value in [('E',E),('q',q),('K',K),('V',V)]: print(name, tuple(value.shape))
"""),md("""### Matching and masking

All real tokens in this window precede the target. For the **final query** there
are no future columns, but padded keys must still be blocked. If we compute all
query rows (`forward_details`), earlier rows also need the triangular causal mask.
Keeping `<BOS>` ensures every prediction has at least one allowed key.
"""),code("""show_pipeline('attention', 'training', ('scores','weights'))
scores = q @ K.transpose(-2,-1) / math.sqrt(attention.d_k)
scores = scores.masked_fill(X.eq(toy_vocab.pad_id)[:,None,:], float('-inf'))
A = scores.softmax(-1)
assert torch.allclose(A.sum(-1), torch.ones(len(X),1))
assert torch.equal(A.masked_select(X.eq(toy_vocab.pad_id)[:,None,:]),
                   torch.zeros_like(A.masked_select(X.eq(toy_vocab.pad_id)[:,None,:])))
print('first context:', toy_vocab.decode_ids(X[0],False))
print('its attention weights:', A[0,0].detach().numpy().round(3))
"""),md("""### Values, the output projection, and the residual

The weights mix **values**, not keys. The two-number message cannot be added
directly to the four-number input row. `W_O` maps it back to width four first.
The updated final row then enters the same kind of hidden-layer prediction MLP.
"""),code("""show_pipeline('attention', 'training', ('message','projection','residual','readout','hidden','logits'))
message = A @ V                              # [B,1,2]
update = attention.W_O(message).squeeze(1)    # [B,4]
final_row = E[:,-1,:] + update               # [B,4]
hidden = F.relu(attention.hidden_layer(final_row))
attention_logits = attention.vocab_head(hidden)
assert torch.allclose(attention_logits, attention(X), atol=1e-6)
assert torch.allclose(attention_logits, attention.forward_details(X)['logits'], atol=1e-6)
for name,value in [('message',message),('update',update),('final row',final_row),('hidden',hidden),('logits',attention_logits)]:
    print(name, tuple(value.shape))
"""),md("""## 5. The target meets the logits only at the loss

Cross-entropy uses raw logits and the observed next-token IDs. It contains a
stable log-softmax. We do not pass softmax probabilities to `cross_entropy`.
The following is **one teaching update**, not a trained language-model benchmark.
Autograd computes gradients; the optimizer changes the learned parameters.
"""),code("""show_pipeline('attention', 'training', ('loss','backward','optimizer','parameters'))
for name, model in [('MLP',mlp),('attention',attention)]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    z = model(X)
    loss = F.cross_entropy(z, y)
    manual_loss = -z.log_softmax(-1)[torch.arange(len(y)),y].mean()
    assert torch.allclose(loss, manual_loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    print(name, 'CE before this update:', round(loss.item(),4))
    for n,p in model.named_parameters():
        print(' ',n,'parameter change:',round((p.detach()-before[n]).norm().item(),6))
"""),md("""## 6. Test scoring is different from generation

Held-out **test scoring** still has observed context/target pairs. Run a frozen
checkpoint on those contexts, accumulate summed cross-entropy, then divide by
the number of targets. There is no optimizer step. Generated samples are not
inserted in place of the observed test contexts.

**Generation** has a prompt and no observed next-token target. It chooses a token,
appends it, and predicts again. The code below loads our saved TinyStories models.
They use a 64-token context, not the four-slot tracing example above.
"""),code("""saved_vocab = json.loads((HERE/'artifacts/vocab.json').read_text())
vocab = Vocabulary(saved_vocab['itos'], {t:i for i,t in enumerate(saved_vocab['itos'])},
                   saved_vocab['counts'], saved_vocab['max_vocab'], saved_vocab['min_freq'])
benchmark = json.loads((HERE/'artifacts/benchmark.json').read_text())
models = {}
for kind in ['mlp','attention']:
    cfg = benchmark['configs'][kind]
    args = (len(vocab.itos),cfg['context_len'],cfg['d_embed'])
    model = (FixedWindowMLP(*args,cfg['hidden'],vocab.pad_id) if kind=='mlp' else
             CausalAttentionLM(*args,cfg['d_embed'],cfg['d_embed'],cfg['hidden'],vocab.pad_id))
    load_model_npz(HERE/f'artifacts/{kind}_seed11.npz', model)
    models[kind] = model.eval()
    print(kind, 'context:', cfg['context_len'], 'parameters:', count_parameters(model))
"""),md("""## 7. One prompt through the complete inference loop

Use the saved tokenizer and vocabulary, prepend `<BOS>` once, and keep the last
`w` IDs. Left-pad short prompts exactly as during training. Attention uses the
same window-slot positions and excludes padded keys. This implementation
**recomputes the window without a KV cache**; cropping shifts the slot positions.

`eval()` selects evaluation behaviour. `inference_mode()` disables gradient
tracking. Omitting optimizer steps keeps parameters fixed. Here greedy decoding
makes the four-step trace reproducible. Sampling would use the softmax distribution.
"""),code("""for kind, model in models.items():
    show_pipeline(kind, 'inference')
    history = [vocab.bos_id] + vocab.encode_tokens(tokenize('once upon a time'), boundaries=False)
    saved = {n:p.detach().clone() for n,p in model.named_parameters()}
    with torch.inference_mode():
        for step in range(4):
            kept = history[-model.context_len:]
            context = [vocab.pad_id]*(model.context_len-len(kept)) + kept
            logits = model(torch.tensor([context]))[0]
            logits[[vocab.pad_id,vocab.bos_id,vocab.unk_id]] = float('-inf')
            probabilities = logits.softmax(-1)
            next_id = int(probabilities.argmax())
            print(kind, 'step',step+1, 'real context:',vocab.decode_ids(kept,False),
                  'PAD slots:',model.context_len-len(kept), 'next:',vocab.itos[next_id],
                  'probability:',round(float(probabilities[next_id]),4))
            if next_id == vocab.eos_id: break
            history.append(next_id)
    assert all(torch.equal(p,saved[n]) for n,p in model.named_parameters())
"""),md("""## 8. The measured TinyStories comparison

The existing frozen experiment uses the same documents, tokenizer, 64-token
context, widths, evaluation targets and maximum supervised-target budget.
Each architecture has validation-tuned optimization settings. Three seeds and
validation-selected checkpoints give attention about **39.1% lower test
perplexity**. This is a measured result for this corpus and budget, not a theorem
that attention always wins. It was run on Apple MPS, not the occupied DGX.

The small smoke runs in Notebooks 1 and 3 use different teaching defaults;
their losses are not the fair matched benchmark. Notebook 4 carries the actual
comparison, embedding changes, attention maps and controlled interventions.
"""),code("""print(json.dumps(benchmark['comparison'], indent=2))
print('Frozen settings:',json.dumps(benchmark['configs'],indent=2))
"""),md("""### References

- [TinyStories dataset and license](https://huggingface.co/datasets/roneneldan/TinyStories)
- [PyTorch cross-entropy: logits and class-index targets](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch embedding lookup and padding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

The SVG diagrams are generated by `pipeline_maps.py`. Call
`show_pipeline('attention', 'training', ('scores','weights'))` to reuse the same
layout with a different stage highlighted. The model code and benchmark artifacts
remain the source of truth for calculations.
""")]
