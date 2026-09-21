#!/usr/bin/env python3
"""Build the five teaching notebooks from reviewable source cells."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from walkthrough_cells import add_maps
from build_slow_lesson import notebook_cells as notebook_5


HERE = Path(__file__).resolve().parent


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def write(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    nbf.write(notebook, HERE / name)


COMMON_SETUP = r'''
from pathlib import Path
import json, math, os, sys, time

HERE = Path.cwd()
if not (HERE / "wordlm.py").exists():
    matches = list(Path.cwd().rglob("word-level-next-token/wordlm.py"))
    if not matches:
        raise FileNotFoundError("Run this notebook from its word-level-next-token directory.")
    HERE = matches[0].parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from wordlm import *

SEED = 11
seed_everything(SEED)
DEVICE = resolve_device(os.getenv("WORDLM_DEVICE", "auto"))
plt.style.use("seaborn-v0_8-whitegrid")
print(f"torch={torch.__version__}  device={DEVICE}")
'''


def notebook_1() -> list:
    return [
        md(r'''
        # 1 · Words to next-token probabilities

        **Question.** Can the embedding → concatenate → hidden-layer MLP from the name generator learn a real word-level next-token task?

        This notebook follows the implementation-first rhythm of [Karpathy's *makemore* sequence](https://karpathy.ai/zero-to-hero.html) and the inspectable style of [Nipun Batra's names notebook](https://nipunbatra.github.io/ml-teaching/notebooks/names.html), but changes the tokens from characters to lowercase words and punctuation. It aligns with [Attention and language · Part I](https://nipunbatra.github.io/attention/part1.html).

        **Learning path**

        ```text
        story → word/punctuation tokens → integer IDs → learned rows
              → concatenate a fixed window → ReLU hidden layer
              → one logit per vocabulary item → cross-entropy → next-token probabilities
        ```

        We split whole stories *before* making overlapping windows. Only training stories fit the vocabulary. The test split stays untouched until the architecture and training settings below are fixed.
        '''),
        code(COMMON_SETUP),
        md(r'''
        ## 1. A bounded, auditable corpus

        [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) contains synthetic short stories generated with GPT-3.5 and GPT-4 and is listed under **CDLA-Sharing-1.0**. The upstream repository is pinned to commit `f54c09f…`. The quick profile fetches 480 deterministic source rows through Hugging Face's documented row API—no dataset script or pickle is executed.

        Exact duplicates are removed first. A stable hash then assigns each complete story to train/validation/test (80/10/10 in expectation), so no window can cross a story or split boundary.
        '''),
        code(r'''
        PROFILE = os.getenv("WORDLM_PROFILE", "smoke")
        DATA_ROOT = HERE / "work" / "data"
        corpus = build_corpus(DATA_ROOT, PROFILE)

        print("source revision:", corpus.manifest["dataset_revision"])
        print("raw subset SHA-256:", corpus.manifest["raw_sha256"])
        print("documents:", corpus.audit["documents"])
        print("duplicates removed before split:", corpus.audit["duplicates_removed_before_split"])
        print("document overlap audit:", corpus.audit["document_overlap"])
        print("vocabulary size:", len(corpus.vocab.itos), "(fit on train only)")
        for split in ["validation", "test"]:
            print(f"{split} OOV rate: {corpus.audit['oov'][split]['rate']:.2%}")
        assert not any(corpus.audit["document_overlap"].values())
        '''),
        md(r'''
        **Interpretation.** Held-out words that miss the train-only vocabulary become `<UNK>`. This is a real limitation of word tokenization, not evidence leakage to be repaired by peeking at validation or test text.

        We normalize Unicode with NFKC, lowercase it, collapse whitespace, keep contractions such as `don't` together, and make punctuation separate tokens. The four special tokens have separate jobs:

        - `<PAD>` left-pads short contexts; `<BOS>` starts each story.
        - `<EOS>` is a learnable stopping target; `<UNK>` represents held-out or rare words.
        '''),
        code(r'''
        example = "Lily's little cat, Max, didn't sleep!"
        pieces = tokenize(example)
        rebuilt = detokenize(pieces)
        print("text:       ", example)
        print("tokens:     ", pieces)
        print("detokenized:", rebuilt)
        assert tokenize(rebuilt) == pieces

        first_story = corpus.stories["train"][0]
        print("\nOne story preview (first 32 tokens only):")
        print(first_story.doc_id, first_story.tokens[:32])
        '''),
        md(r'''
        ## 2. Context–target examples

        With context length $w=8$, every observed token (and the final `<EOS>`) is one target. The context is the preceding eight IDs, left-padded inside that story only.
        '''),
        code(r'''
        CONTEXT_LEN = 8
        windows = make_all_windows(corpus, CONTEXT_LEN)
        X_train, y_train, train_doc_ids = windows["train"]
        X_val, y_val, _ = windows["validation"]
        X_test, y_test, _ = windows["test"]

        def show_pair(row):
            context = corpus.vocab.decode_ids(X_train[row], skip_special=False)
            target = corpus.vocab.itos[int(y_train[row])]
            return " ".join(context), target, train_doc_ids[row]

        for row in range(6):
            context, target, doc_id = show_pair(row)
            print(f"{context:70s} → {target:12s}  doc={doc_id}")
        print("\nshapes:", tuple(X_train.shape), tuple(y_train.shape))
        assert all(len(set(ids)) == 1 for ids in [train_doc_ids[: min(6, len(train_doc_ids))]])
        '''),
        md(r'''
        ## 3. The fixed-window MLP

        ```text
        IDs [B,w] → lookup E[IDs] [B,w,d] → flatten [B,w·d]
                  → affine + ReLU [B,h] → affine [B,|V|]
        ```

        In row-vector notation,

        $$a_0=[e_1,\ldots,e_w],\qquad a_1=\operatorname{ReLU}(a_0W_1+b_1),\qquad z=a_1W_2+b_2.$$

        `cross_entropy(z, y)` applies log-softmax stably and scores the observed target. The embedding table, both weight matrices, and both biases are learned together.
        '''),
        code(r'''
        class WordMLP(nn.Module):
            def __init__(self, vocab_size, context_len, d_embed=32, hidden=128, pad_id=0):
                super().__init__()
                self.context_len, self.d_embed = context_len, d_embed
                self.token_embedding = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)
                self.hidden_layer = nn.Linear(context_len * d_embed, hidden)
                self.vocab_head = nn.Linear(hidden, vocab_size)

            def forward(self, context_ids):
                embedded = self.token_embedding(context_ids)                 # [B,w,d]
                a0 = embedded.reshape(context_ids.shape[0], -1)              # [B,w*d]
                a1 = F.relu(self.hidden_layer(a0))                            # [B,h]
                logits = self.vocab_head(a1)                                 # [B,|V|]
                return logits

        seed_everything(SEED)
        model = WordMLP(len(corpus.vocab.itos), CONTEXT_LEN, 32, 128, corpus.vocab.pad_id)
        logits = model(X_train[:4])
        print("lookup:", tuple(model.token_embedding(X_train[:4]).shape))
        print("logits:", tuple(logits.shape), "parameters:", f"{count_parameters(model):,}")
        print("initial cross-entropy:", float(F.cross_entropy(logits, y_train[:4])))
        assert logits.shape == (4, len(corpus.vocab.itos))
        '''),
        md(r'''
        **Interpretation.** The first layer sees $w d$ inputs. Doubling the context length doubles both that activation width and the $wd\times h$ first weight matrix. Notebook 2 measures this trade-off.
        '''),
        md(r'''
        ## 4. Debug before training: overfit 16 examples

        A small network should memorize a tiny batch. Failure here usually means a target shift, shape, optimizer, or gradient bug—not insufficient data.
        '''),
        code(r'''
        seed_everything(23)
        debug_model = WordMLP(len(corpus.vocab.itos), CONTEXT_LEN, 24, 96, corpus.vocab.pad_id)
        before, after = tiny_batch_overfit(debug_model, X_train, y_train, device=DEVICE, steps=180)
        print(f"tiny-batch loss: {before:.3f} → {after:.5f}")
        assert after < min(0.15, before / 10)
        '''),
        md(r'''
        ## 5. Train, validate, then touch test once

        AdamW and gradient clipping are practical training ingredients rather than new language-model architecture. The smoke profile is intentionally small; the measured DGX-profile comparison appears in Notebook 4.
        '''),
        code(r'''
        seed_everything(SEED)
        model = WordMLP(len(corpus.vocab.itos), CONTEXT_LEN, 32, 128, corpus.vocab.pad_id)
        config = TrainConfig(
            steps=300, batch_size=256, learning_rate=3e-3, weight_decay=1e-4,
            seed=SEED, eval_every=50, eval_examples=20_000,
        )
        result, _ = train_model(model, X_train, y_train, X_val, y_val, config, device=DEVICE)
        test_loss = evaluate_loss(model, X_test, y_test, device=DEVICE)
        print(f"validation CE={result['validation_loss']:.3f}, PPL={math.exp(result['validation_loss']):.2f}")
        print(f"test CE={test_loss:.3f}, PPL={math.exp(test_loss):.2f}  (first and only test use)")
        print(f"tokens seen={result['tokens_seen']:,}, runtime={result['runtime_seconds']:.1f}s")
        '''),
        code(r'''
        trace = result["trace"]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot([p["tokens_seen"] for p in trace], [p["train_batch_loss"] for p in trace], "o-", label="train batch")
        ax.plot([p["tokens_seen"] for p in trace], [p["validation_loss"] for p in trace], "o-", label="validation")
        ax.set(xlabel="supervised target tokens seen", ylabel="cross-entropy (nats)", title="Fixed-window MLP learning curve")
        ax.legend(); plt.show()
        '''),
        md(r'''
        The validation curve is the tuning signal. Perplexity is $\exp(\text{cross-entropy})$ and is comparable only when tokenizer, vocabulary, targets, and reduction are held fixed.

        ## 6. Generate with fixed parameters

        Temperature changes the sampling distribution $p=\operatorname{softmax}(z/\tau)$; it does not retrain the model. Lower temperature is sharper, higher temperature is more varied.
        '''),
        code(r'''
        for temperature in [0.7, 1.0, 1.3]:
            print(f"T={temperature:.1f}:", generate_text(
                model, corpus.vocab, "once upon a time", context_len=CONTEXT_LEN,
                max_new_tokens=45, temperature=temperature, seed=4, device=DEVICE,
            ))
        '''),
        md(r'''
        **What this notebook supports.** A word-level learned-embedding MLP can be trained end-to-end and sampled. Its outputs on the smoke profile may still be repetitive or locally plausible rather than coherent; that is an observed limitation, not a formatting failure. The next notebook gives this baseline a controlled tuning pass before attention is compared with it.
        '''),
    ]


def notebook_2() -> list:
    return [
        md(r'''
        # 2 · What changes when we change hyperparameters?

        The baseline deserves a real tuning pass. These experiments change **one variable at a time** around a common MLP configuration while holding the bounded TinyStories data, tokenizer, train-only vocabulary, seed, batch size, and target-token budget fixed.

        ```text
        context w ─┐
        embedding d ├─→ first-layer input width w·d ─→ parameters / compute
        hidden h ───┘
        learning rate + weight decay ─→ optimization / generalization
        ```

        The included artifact is a measured sweep, not illustrative numbers. Test data was not used for selection.
        '''),
        code(COMMON_SETUP),
        code(r'''
        sweep_path = HERE / "artifacts" / "mlp_sweep.json"
        comparison_path = HERE / "artifacts" / "comparison_tuning.json"
        if not sweep_path.exists() or not comparison_path.exists():
            raise FileNotFoundError("Run run_experiments.py --mode sweep or use the included artifacts.")
        sweep = read_json(sweep_path)
        comparison_tuning = read_json(comparison_path)
        records = sweep["records"]
        print("test set used during sweep:", sweep["protocol"]["test_set_used"])
        print("runs:", len(records), "selection seed:", sweep["protocol"]["selection_seed"])
        print("selected primary config:", sweep["selected_primary_mlp_config"])
        assert sweep["protocol"]["test_set_used"] is False
        '''),
        md(r'''
        ## 1. Make the growing input visible

        For vocabulary size $|\mathcal V|$, context $w$, embedding width $d$, and hidden width $h$:

        $$\text{MLP parameters}=|\mathcal V|d+(wd)h+h+h|\mathcal V|+|\mathcal V|.$$

        The context length affects the $wd\times h$ matrix directly. It does not change the embedding lookup width or vocabulary output width.
        '''),
        code(r'''
        baseline = sweep["protocol"]["baseline"]
        vocab_size = sweep["records"][0]["model_config"].get("vocab_size", 4000)
        print(f"{'w':>4} {'d':>4} {'w·d input':>10} {'W1 weights':>12}")
        for w in [8, 16, 32, 64]:
            d, h = baseline["d_embed"], baseline["hidden"]
            print(f"{w:4d} {d:4d} {w*d:10d} {w*d*h:12,d}")
        '''),
        md(r'''
        ## 2. Final validation loss, parameter count, and runtime

        Lower validation cross-entropy is better. Parameter count is an architectural cost; runtime is a measurement from this run and environment, not a hardware-independent law.
        '''),
        code(r'''
        ordered = sorted(records, key=lambda r: r["validation_loss"])
        print(f"{'axis':>14} {'value':>10} {'val CE':>8} {'gap':>8} {'params':>11} {'multiplies':>12} {'seconds':>8}")
        for r in ordered:
            cfg = r["model_config"]
            gap = r["validation_loss"] - r["final_train_batch_loss"]
            print(f"{cfg['sweep_axis']:>14} {str(cfg['sweep_value']):>10} {r['validation_loss']:8.3f} "
                  f"{gap:8.3f} {r['parameter_count']:11,d} {r['approximate_forward_multiplies']:12,d} {r['runtime_seconds']:8.1f}")
        '''),
        code(r'''
        axes = ["context_len", "d_embed", "hidden", "learning_rate", "weight_decay"]
        fig, panels = plt.subplots(2, 3, figsize=(12, 6.5))
        panels = panels.ravel()
        for ax, axis in zip(panels, axes):
            relevant = [r for r in records if r["model_config"]["sweep_axis"] in (axis, "baseline")]
            relevant = sorted(relevant, key=lambda r: float(r["model_config"][axis]))
            x = [float(r["model_config"][axis]) for r in relevant]
            y = [r["validation_loss"] for r in relevant]
            ax.plot(x, y, "o-", color="#007a75")
            ax.set(title=axis, xlabel=axis, ylabel="validation CE")
            if axis in ("learning_rate", "weight_decay"):
                ax.set_xscale("symlog", linthresh=1e-5)
        panels[-1].scatter(
            [r["parameter_count"] for r in records], [r["validation_loss"] for r in records],
            c=[r["runtime_seconds"] for r in records], cmap="viridis", s=55,
        )
        panels[-1].set(title="cost vs validation", xlabel="parameters", ylabel="validation CE")
        fig.tight_layout(); plt.show()
        '''),
        md(r'''
        **Interpretation.** Wider or longer is not automatically better under a fixed token budget. Extra capacity changes both optimization and overfitting pressure. Learning rate often moves the result more than a modest width change, which is why an intentionally weak baseline would make an architecture comparison meaningless.

        ## 3. Learning curves, not only final dots
        '''),
        code(r'''
        # Show the baseline plus the best and worst validation runs.
        baseline_run = next(r for r in records if r["model_config"]["sweep_axis"] == "baseline")
        chosen = [baseline_run, ordered[0], ordered[-1]]
        labels = ["baseline", "best validation", "worst validation"]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for r, label in zip(chosen, labels):
            trace = r["trace"]
            ax.plot([p["tokens_seen"] for p in trace], [p["validation_loss"] for p in trace], "o-", label=label)
        ax.set(xlabel="supervised target tokens seen", ylabel="validation cross-entropy", title="Validation trajectories under the same budget")
        ax.legend(); plt.show()
        '''),
        md(r'''
        ## 4. The selection rule

        The primary comparison in Notebook 4 fixes a **shared 64-word context** for both architectures. This is a deliberate long-context stress test: the MLP must concatenate all 64 embedding rows, while attention can learn a content-weighted summary. Learning rate and weight decay are tuned separately for each model on validation, with identical selection seeds, widths, batch size, and target-token budget. The test split remains sealed.
        '''),
        code(r'''
        print("comparison selection protocol:", comparison_tuning["protocol"])
        print("selected without test data:")
        for kind, config in comparison_tuning["selected_configs"].items():
            print(f"  {kind:>9s}: {config}")
        assert comparison_tuning["protocol"]["test_set_used"] is False
        assert comparison_tuning["protocol"]["same_context_length"] == 64
        '''),
        md(r'''
        **Limit.** This is a modest, controlled sweep—not exhaustive hyperparameter optimization. It establishes a credible baseline at a stated budget. The untouched test split is evaluated only after both MLP and attention configurations are frozen.
        '''),
    ]


def notebook_3() -> list:
    return [
        md(r'''
        # 3 · Causal attention from scratch

        This notebook stops exactly at [Attention and language · Part II](https://nipunbatra.github.io/attention/attention.html): **one** causal self-attention head, Q/K/V projections, scaled scores, causal mask, row-wise softmax, weighted values, output projection, residual addition, then a ReLU prediction MLP.

        ```text
        token + position rows X
          ├─→ Q and K → scaled matches → causal mask → row-wise weights A
          └─→ V ────────────────────────────────→ weighted messages A·V
        message → W_O → contextual update;  X + update → hidden MLP → logits
        ```

        There is no `nn.MultiheadAttention`, fused attention call, LayerNorm, multi-head split, stacked block, or pretrained embedding. A learned position lookup and AdamW are named practical ingredients used to make the experiment trainable.
        '''),
        code(COMMON_SETUP),
        md(r'''
        ## 1. Hand-worked tiny tensors

        Let rows be token positions. Here $T=4$, $d_{model}=3$, $d_k=2$, and $d_v=2$. Queries and keys choose **where** to read; the resulting weights mix values, which carry **what** is sent.
        '''),
        code(r'''
        X_tiny = torch.tensor([
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.1],
            [1.0, 1.0, 0.0],
            [0.5, 1.0, 0.4],
        ])                                                     # [T=4,d_model=3]
        W_Q = torch.tensor([[1.,0.], [0.,1.], [0.5,0.5]])       # [3,2]
        W_K = torch.tensor([[1.,0.], [0.,1.], [0.,0.]])         # [3,2]
        W_V = torch.tensor([[1.,0.], [0.,1.], [1.,-1.]])        # [3,2]
        W_O = torch.tensor([[1.,0.,0.5], [0.,1.,-0.5]])         # [2,3]

        Q, K, V = X_tiny @ W_Q, X_tiny @ W_K, X_tiny @ W_V
        raw_scores = Q @ K.T
        scaled_scores = raw_scores / math.sqrt(Q.shape[-1])
        future = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
        masked_scores = scaled_scores.masked_fill(future, float("-inf"))
        A = F.softmax(masked_scores, dim=-1)
        messages = A @ V
        updates = messages @ W_O
        contextual = X_tiny + updates

        for name, tensor in [("X",X_tiny),("Q",Q),("K",K),("V",V),("scores",scaled_scores),
                             ("A",A),("messages",messages),("updates",updates),("X'",contextual)]:
            print(f"{name:9s} shape={tuple(tensor.shape)}")
        print("\ncausal weights A:\n", A.round(decimals=3))
        assert torch.equal(A[future], torch.zeros_like(A[future]))
        assert torch.allclose(A.sum(-1), torch.ones(4))
        '''),
        md(r'''
        Each row of $A$ is a probability distribution over allowed **source positions**. A zero above the diagonal means a future value contributes nothing. It is not a zero probability for a vocabulary token.

        $$Q=XW_Q,\ K=XW_K,\ V=XW_V,$$
        $$S=QK^\top/\sqrt{d_k},\quad A=\operatorname{softmax}_{row}(S+M),$$
        $$H=AV,\quad \Delta X=HW_O,\quad X'=X+\Delta X.$$

        Scaling controls score spread; subtracting a softmax maximum handles numerical overflow. They solve different problems.
        '''),
        md(r'''
        ## 2. Word data and positional information

        Attention without position information cannot tell which identical multiset order it received. We add a learned position row of the **same width** as each word embedding. This is an architecture input, not a claim that a learned coordinate literally means “position.”
        '''),
        code(r'''
        PROFILE = os.getenv("WORDLM_PROFILE", "smoke")
        corpus = build_corpus(HERE / "work" / "data", PROFILE)
        CONTEXT_LEN = 12
        windows = make_all_windows(corpus, CONTEXT_LEN)
        X_train, y_train, _ = windows["train"]
        X_val, y_val, _ = windows["validation"]
        X_test, y_test, _ = windows["test"]
        print(corpus.audit["documents"], "vocab", len(corpus.vocab.itos), "train targets", len(y_train))
        '''),
        md(r'''
        ## 3. The complete one-head teaching model

        All intermediate tensors are returned by `forward_details` so shapes and claims can be tested. The training `forward` computes only the final causal query, which is exactly the final row of the full calculation and avoids producing unused vocabulary logits at earlier rows. The comparison still predicts exactly one target per window.
        '''),
        code(r'''
        class TeachingAttentionLM(nn.Module):
            def __init__(self, vocab_size, context_len, d_model=48, d_k=48, d_v=48, hidden=160, pad_id=0):
                super().__init__()
                self.context_len, self.d_k, self.pad_id = context_len, d_k, pad_id
                self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
                self.position_embedding = nn.Embedding(context_len, d_model)
                self.W_Q = nn.Linear(d_model, d_k, bias=False)
                self.W_K = nn.Linear(d_model, d_k, bias=False)
                self.W_V = nn.Linear(d_model, d_v, bias=False)
                self.W_O = nn.Linear(d_v, d_model, bias=False)
                self.hidden_layer = nn.Linear(d_model, hidden)
                self.vocab_head = nn.Linear(hidden, vocab_size)

            def forward_details(self, ids):
                B, T = ids.shape
                positions = torch.arange(T, device=ids.device)
                X = self.token_embedding(ids) + self.position_embedding(positions)[None, :, :]
                Q, K, V = self.W_Q(X), self.W_K(X), self.W_V(X)
                scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
                future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)
                real = ids.ne(self.pad_id)
                padded_key_for_real_query = (~real[:, None, :]) & real[:, :, None]
                attention_mask = future[None, :, :] | padded_key_for_real_query
                masked_scores = scores.masked_fill(attention_mask, float("-inf"))
                weights = F.softmax(masked_scores, dim=-1)
                messages = weights @ V
                updates = self.W_O(messages)
                contextual = X + updates
                hidden = F.relu(self.hidden_layer(contextual))
                logits_all = self.vocab_head(hidden)
                return dict(X=X,Q=Q,K=K,V=V,scores=scores,masked_scores=masked_scores,
                            weights=weights,messages=messages,updates=updates,contextual=contextual,
                            hidden=hidden,logits_all=logits_all,logits=logits_all[:,-1,:])

            def forward(self, ids):
                _, T = ids.shape
                positions = torch.arange(T, device=ids.device)
                X = self.token_embedding(ids) + self.position_embedding(positions)[None, :, :]
                q_last, K, V = self.W_Q(X[:, -1:, :]), self.W_K(X), self.W_V(X)
                scores_last = q_last @ K.transpose(-2, -1) / math.sqrt(self.d_k)
                scores_last = scores_last.masked_fill(ids.eq(self.pad_id)[:, None, :], float("-inf"))
                message_last = F.softmax(scores_last, dim=-1) @ V
                contextual_last = X[:, -1, :] + self.W_O(message_last).squeeze(1)
                return self.vocab_head(F.relu(self.hidden_layer(contextual_last)))
        '''),
        code(r'''
        seed_everything(SEED)
        model = TeachingAttentionLM(len(corpus.vocab.itos), CONTEXT_LEN, pad_id=corpus.vocab.pad_id)
        details = model.forward_details(X_train[:3])
        for name in ["X","Q","K","V","scores","weights","messages","updates","contextual","hidden","logits"]:
            print(f"{name:11s} {tuple(details[name].shape)}")
        assert torch.allclose(model(X_train[:3]), details["logits"], atol=1e-6)
        '''),
        md(r'''
        ## 4. Mechanical checks before interpretation

        A model that produces a pretty heatmap can still be wrong. We test the mask, normalization axis, causality, finite gradients, and ability to overfit 16 targets.
        '''),
        code(r'''
        checks = attention_invariants(model, X_train[:2])
        print(checks)
        assert checks["max_future_weight"] == 0.0
        assert checks["max_row_sum_error"] < 1e-6
        assert checks["max_earlier_representation_change_after_future_edit"] < 1e-6

        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(X_train[:8]), y_train[:8])
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        print("finite loss and gradients:", float(loss))
        '''),
        code(r'''
        seed_everything(31)
        debug_model = TeachingAttentionLM(len(corpus.vocab.itos), CONTEXT_LEN, d_model=32, d_k=32, d_v=32, hidden=96, pad_id=corpus.vocab.pad_id)
        before, after = tiny_batch_overfit(debug_model, X_train, y_train, device=DEVICE, steps=220)
        print(f"tiny-batch loss: {before:.3f} → {after:.5f}")
        assert after < min(0.15, before / 10)
        '''),
        md(r'''
        ## 5. Actually train the one-head model

        This smoke run uses one target per window—the same loss unit as the MLP. The larger, multi-seed, matched-context comparison is reserved for Notebook 4.
        '''),
        code(r'''
        seed_everything(SEED)
        model = TeachingAttentionLM(len(corpus.vocab.itos), CONTEXT_LEN, pad_id=corpus.vocab.pad_id)
        config = TrainConfig(steps=300, batch_size=256, learning_rate=3e-3, weight_decay=1e-4,
                             seed=SEED, eval_every=50, eval_examples=20_000)
        result, _ = train_model(model, X_train, y_train, X_val, y_val, config, device=DEVICE)
        test_loss = evaluate_loss(model, X_test, y_test, device=DEVICE)
        print(f"validation CE={result['validation_loss']:.3f}, PPL={math.exp(result['validation_loss']):.2f}")
        print(f"test CE={test_loss:.3f}, PPL={math.exp(test_loss):.2f} (after settings were fixed)")
        print(f"parameters={count_parameters(model):,}, tokens seen={result['tokens_seen']:,}, runtime={result['runtime_seconds']:.1f}s")
        '''),
        code(r'''
        trace = result["trace"]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot([p["tokens_seen"] for p in trace], [p["train_batch_loss"] for p in trace], "o-", label="train batch")
        ax.plot([p["tokens_seen"] for p in trace], [p["validation_loss"] for p in trace], "o-", label="validation")
        ax.set(xlabel="supervised target tokens seen", ylabel="cross-entropy (nats)", title="One-head causal attention learning curve")
        ax.legend(); plt.show()
        '''),
        code(r'''
        for temperature in [0.8, 1.0]:
            print(f"T={temperature:.1f}:", generate_text(
                model, corpus.vocab, "once upon a time", context_len=CONTEXT_LEN,
                max_new_tokens=45, temperature=temperature, seed=8, device=DEVICE,
            ))
        '''),
        md(r'''
        **Scope.** This model demonstrates learned causal routing and next-token training. It is not a complete Transformer: no multiple heads, block FFN, LayerNorm, stacked blocks, or pretrained representation has been silently added. Notebook 4 asks what the measured model—not an imagined semantic-axis story—actually learned.
        '''),
    ]


def notebook_4() -> list:
    return [
        md(r'''
        # 4 · What did the models learn?

        We now inspect the selected fixed-window MLP and one-head causal-attention models using their **actual saved parameters and activations**.

        ```text
        fair held-out metrics → embedding motion on one shared 2-D basis
        → cosine neighbours → same lookup / different contextual row
        → descriptive attention map → controlled token interventions
        ```

        Individual learned coordinates are not labelled “water,” “finance,” or “person.” The named axes in the Part II lecture are hand-chosen teaching aids, not properties to assume of trained embeddings.
        '''),
        code(COMMON_SETUP),
        code(r'''
        ARTIFACTS = HERE / "artifacts"
        benchmark = read_json(ARTIFACTS / "benchmark.json")
        vocab_data = read_json(ARTIFACTS / "vocab.json")
        itos = vocab_data["itos"]
        vocab = Vocabulary(itos, {token:i for i,token in enumerate(itos)}, vocab_data["counts"],
                           vocab_data["max_vocab"], vocab_data["min_freq"])
        print("source revision:", benchmark["source"]["dataset_revision"])
        print("profile:", benchmark["protocol"]["profile"], "seeds:", benchmark["protocol"]["seeds"])
        print("comparison protocol:", {k:v for k,v in benchmark["protocol"].items() if k.startswith("same_")})
        '''),
        md(r'''
        ## 1. Did attention improve the natural-corpus benchmark?

        Both models use the same documents, normalization, train-only vocabulary, **64-word context**, target windows, seed set, batch size, maximum supervised-token budget, and token-weighted loss. Both predict one target per window, avoiding a hidden all-positions versus one-position accounting advantage. Within the shared 6,000-step ceiling, each run restores its lowest-validation-loss checkpoint before the test is evaluated; the chosen step is reported rather than hidden.
        '''),
        code(r'''
        print(f"{'model':>10} {'params':>11} {'max tokens':>11} {'selected steps':>17} {'test CE mean±sd':>20} {'test PPL mean±sd':>20} {'runtime mean':>14}")
        for kind in ["mlp", "attention"]:
            runs = benchmark["runs"][kind]
            agg = benchmark["aggregate"][kind]
            selected_steps = ",".join(str(r["selected_step"]) for r in runs)
            print(f"{kind:>10} {runs[0]['parameter_count']:11,d} {runs[0]['tokens_seen']:11,d} {selected_steps:>17s} "
                  f"{agg['test_loss']['mean']:8.4f}±{agg['test_loss']['sample_std']:.4f} "
                  f"{agg['test_perplexity']['mean']:8.2f}±{agg['test_perplexity']['sample_std']:.2f} "
                  f"{agg['runtime_seconds']['mean']:12.1f}s")
        print("\nmeasured conclusion:", benchmark["comparison"])
        '''),
        md(r'''
        Interpret the sign of `attention_minus_mlp_test_loss` directly: negative favours attention. The claim is limited to this bounded subset and budget. Multiple seeds expose optimization variation; they do not turn one classroom experiment into a universal architecture result.

        ## 2. Load safe, compact NumPy checkpoints

        These `.npz` files contain arrays and JSON only; `allow_pickle=False` is enforced. We inspect seed 11 and keep the three-seed metric table above as the quality result.
        '''),
        code(r'''
        mlp_cfg = benchmark["configs"]["mlp"]
        attn_cfg = benchmark["configs"]["attention"]
        mlp = FixedWindowMLP(len(itos), mlp_cfg["context_len"], mlp_cfg["d_embed"], mlp_cfg["hidden"], vocab.pad_id)
        attention = CausalAttentionLM(len(itos), attn_cfg["context_len"], attn_cfg["d_embed"],
                                     attn_cfg["d_embed"], attn_cfg["d_embed"], attn_cfg["hidden"], vocab.pad_id)
        mlp_meta, mlp_trace = load_model_npz(ARTIFACTS / "mlp_seed11.npz", mlp)
        attn_meta, attn_trace = load_model_npz(ARTIFACTS / "attention_seed11.npz", attention)
        mlp.eval(); attention.eval()
        print("MLP checkpoints:", list(mlp_trace), "attention checkpoints:", list(attn_trace))
        '''),
        md(r'''
        ## 3. Embedding movement on one shared basis

        Fitting a separate PCA panel at each checkpoint can make a stationary point appear to rotate or reflect. We fit **one** global two-dimensional basis to all lexical embeddings from both models and all checkpoints, then reuse it everywhere.
        '''),
        code(r'''
        selected_words = [word for word in ["dog","cat","boy","girl","happy","sad","park","house","water","play"] if word in vocab.stoi]
        all_traces = list(mlp_trace.values()) + list(attn_trace.values())
        mean, basis = fit_common_pca([array[4:] for array in all_traces])
        projected = {
            "mlp": {name: project_with_basis(array, mean, basis) for name,array in mlp_trace.items()},
            "attention": {name: project_with_basis(array, mean, basis) for name,array in attn_trace.items()},
        }
        coords = np.concatenate([p for model_trace in projected.values() for p in model_trace.values()])
        lexical = coords[4:]
        xlim = np.percentile(lexical[:,0], [1,99]); ylim = np.percentile(lexical[:,1], [1,99])

        fig, panels = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        for ax, kind, trace in zip(panels, ["mlp","attention"], [mlp_trace, attn_trace]):
            names = list(trace)
            for word in selected_words:
                idx = vocab.stoi[word]
                path = np.array([projected[kind][name][idx] for name in names])
                ax.plot(path[:,0], path[:,1], "o-", alpha=.8)
                ax.text(path[-1,0], path[-1,1], word, fontsize=9)
                ax.set(title=f"{kind}: retained checkpoint states", xlabel="shared PC 1", ylabel="shared PC 2",
                   xlim=xlim, ylim=ylim)
        fig.suptitle("Saved word-embedding states on one jointly fitted basis")
        fig.tight_layout(); plt.show()
        '''),
        md(r'''
        The arrows are descriptive movement in a shared projection. They do not prove what any individual dimension “means,” and the two-dimensional view discards most of the learned space.

        ## 4. Cosine neighbours and similarities
        '''),
        code(r'''
        for kind, trace in [("mlp", mlp_trace), ("attention", attn_trace)]:
            final_name = list(trace)[-1]
            embedding = trace[final_name]
            print(f"\n{kind.upper()} final embedding neighbours")
            for word in [w for w in ["dog","happy","park","water"] if w in vocab.stoi]:
                print(f"{word:>7s}:", cosine_neighbors(embedding, vocab, word, top_k=6))
        '''),
        md(r'''
        Neighbours summarize similarity under the learned lookup geometry. They are corpus- and seed-dependent descriptions, not dictionary definitions.

        ## 5. Same word lookup, different contextual representations

        A lookup row is static. In the attention model, the contextual row also depends on the other words and positions in that occurrence.
        '''),
        code(r'''
        def context_tensor(text):
            ids = [vocab.bos_id] + vocab.encode_tokens(tokenize(text), boundaries=False)
            ids = ids[-attn_cfg["context_len"]:]
            ids = [vocab.pad_id] * (attn_cfg["context_len"] - len(ids)) + ids
            return torch.tensor([ids], dtype=torch.long)

        sentences = ["the children like to play", "the dog and the cat play"]
        contexts = [context_tensor(text) for text in sentences]
        assert all(vocab.itos[int(ctx[0,-1])] == "play" for ctx in contexts)
        lookup_a = attention.token_embedding(contexts[0])[0,-1]
        lookup_b = attention.token_embedding(contexts[1])[0,-1]
        with torch.no_grad():
            contextual = [attention.forward_details(ctx)["contextual"][0,-1] for ctx in contexts]
        print("lookup rows identical:", torch.equal(lookup_a, lookup_b))
        print("lookup cosine:", float(F.cosine_similarity(lookup_a[None], lookup_b[None])))
        print("contextual cosine:", float(F.cosine_similarity(contextual[0][None], contextual[1][None])))
        print("contextual L2 distance:", float(torch.linalg.vector_norm(contextual[0]-contextual[1])))
        '''),
        md(r'''
        The same word has one shared lookup row, but its occurrence-specific representation changes after attention and the residual update. That is the central representational difference from the baseline's static lookup table.

        ## 6. An attention heatmap is descriptive
        '''),
        code(r'''
        probe_text = "once upon a time there was a little dog"
        probe = context_tensor(probe_text)
        labels = vocab.decode_ids(probe[0], skip_special=False)
        with torch.no_grad():
            details = attention.forward_details(probe)
        A = details["weights"][0].numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(A, cmap="magma", vmin=0, vmax=A.max())
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
               xlabel="source / key position", ylabel="receiver / query position", title="One-head causal attention weights")
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
        fig.colorbar(image, ax=ax, label="attention weight"); fig.tight_layout(); plt.show()
        '''),
        md(r'''
        A bright cell says that this head assigned a large mixing weight on this forward pass. It does **not** by itself establish that the source token caused the prediction, that it is an explanation a human would endorse, or that another head/block agrees—there are no other heads or blocks here.

        ## 7. Controlled token interventions

        We now change exactly one earlier token to `<UNK>` and recompute the whole model. This is causal evidence about this intervention on this trained model, though still not a general linguistic explanation.
        '''),
        code(r'''
        final_weights = details["weights"][0,-1].clone()
        eligible = [i for i,tok in enumerate(labels[:-1]) if tok not in (PAD, BOS)]
        high = max(eligible, key=lambda i: float(final_weights[i]))
        low = min(eligible, key=lambda i: float(final_weights[i]))

        def intervene(model, position):
            changed = probe.clone()
            changed[0, position] = vocab.unk_id
            with torch.no_grad():
                return model(probe)[0], model(changed)[0]

        with torch.no_grad():
            base_distribution = F.softmax(attention(probe)[0], dim=-1)
            target_id = int(base_distribution.argmax())
        print("unmodified top prediction:", vocab.itos[target_id], float(base_distribution[target_id]))
        for label, position in [("high-attention", high), ("low-attention", low)]:
            before, after = intervene(attention, position)
            p, q = F.softmax(before, -1), F.softmax(after, -1)
            kl = float((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum())
            print(f"\n{label} source: position={position}, token={labels[position]!r}, weight={float(final_weights[position]):.3f}, KL={kl:.5f}")
            print(top_probability_changes(before, after, vocab, top_k=5))
        '''),
        md(r'''
        Compare the heatmap with the intervention: a larger attention weight need not imply a proportionally larger prediction change because values, output projection, residual addition, and the nonlinear vocabulary MLP all intervene. The ablation is more causally informative than the heatmap, but it is still a particular replacement (`<UNK>`) in one context.

        **Bottom line.** The held-out metric establishes which architecture performed better here; embedding and attention plots describe mechanisms and geometry; controlled interventions test specific causal sensitivities. None justifies naming individual trained coordinates or treating attention weights alone as explanations.
        '''),
    ]


def main() -> None:
    write("01_words_to_probabilities.ipynb", add_maps(notebook_1(), "mlp", md, code))
    write("02_hyperparameter_experiments.ipynb", notebook_2())
    write("03_causal_attention_from_scratch.ipynb", add_maps(notebook_3(), "attention", md, code))
    write("04_what_did_the_models_learn.ipynb", notebook_4())
    write("05_training_and_inference_maps.ipynb", notebook_5(md, code, COMMON_SETUP))
    print("Wrote five notebooks to", HERE)


if __name__ == "__main__":
    main()
