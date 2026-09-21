# Word-level next-token prediction: MLP to one-head attention

This directory is a five-notebook, executable companion to *Attention and
language* Parts I–II. It begins with a tuned learned-embedding MLP and stops at
one causal attention head, output projection, residual addition, and a hidden
prediction MLP. It does not silently introduce a multi-head/multi-block
Transformer, LayerNorm, a block FFN, or pretrained embeddings.

## Notebooks

1. [`01_words_to_probabilities.ipynb`](01_words_to_probabilities.ipynb) —
   bounded corpus, word/punctuation tokenizer, leakage-safe document split,
   train-only vocabulary, fixed-window MLP, validation, sampling, and readable
   generation.
2. [`02_hyperparameter_experiments.ipynb`](02_hyperparameter_experiments.ipynb)
   — controlled MLP sweeps over context, embedding width, hidden width,
   learning rate, and weight decay, including curves and cost accounting.
3. [`03_causal_attention_from_scratch.ipynb`](03_causal_attention_from_scratch.ipynb)
   — hand-worked tensors followed by a trained single-head causal model with
   explicit Q/K/V, scaling, masking, softmax, value mixing, output projection,
   residual, and vocabulary MLP.
4. [`04_what_did_the_models_learn.ipynb`](04_what_did_the_models_learn.ipynb)
   — matched multi-seed comparison, embedding trajectories on one shared PCA
   basis, cosine neighbours, contextual representations, attention maps, and
   controlled token interventions.
5. [`05_training_and_inference_maps.ipynb`](05_training_and_inference_maps.ipynb)
   — 67 small steps: actual stories, a brief tokenization detour, token IDs,
   individual pairs appended to lists, seven stored examples, an explicit B=2 batch,
   each MLP/attention operation, a real optimizer step, and generation with the
   tiny teaching model. The measured TinyStories results are clearly separate.
   Begin here for the visual walkthrough;
   return to notebooks 1 and 3 for the full data and training code.

Notebooks 1 and 3 show the complete training/inference maps first and highlight
the relevant stages beside the code. The editable diagram source is
[`pipeline_maps.py`](pipeline_maps.py). For example:

```python
from pipeline_maps import show_pipeline
show_pipeline('attention', 'training', ('scores', 'weights'))
```

Training uses one target per window in both models. Attention computes the
final query for that loss; `forward_details` can display all causal rows.
The inference maps deliberately show recomputation, not a KV cache.
Run `python make_notebooks.py` to rebuild notebook sources (this clears outputs),
then execute and export them as below. `python pipeline_maps.py figures` exports
the four master SVGs and their graph data for reuse in slides.

### Illustrated study guide and matching slides

[Read the chaptered guide](https://nipunbatra.github.io/attention/notebooks/wordlm/05_training_and_inference_maps.html).
Every step links to its exact Part II slide, displays the same generated SVG,
and includes the corresponding executable code. Expand the complete map to
locate the current operation. The guide works on phones; wide figures scroll
inside their panels. The companion is public, separately from the private lab Site.

`slow_walkthrough.py` is the shared source for the 67 explanations, calculations
and numeric figures. `lesson_evidence.json` records the saved corpus counts and
benchmark provenance; it does not contain the raw story corpus.
`story_examples.json` contains three unchanged TinyStories texts, source row IDs,
the dataset revision, full-text hashes and reproducible word/token counts. The
story slides show one complete text and two excerpts with `[…]` marking omissions;
the notebook prints all three full texts. These examples are from **TinyStories
by Ronen Eldan and Yuanzhi Li**, distributed under
[CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/). Display labels are added for the
lesson; the saved source texts are unchanged. The full 6,000-story corpus is not
bundled.

Three short tokenization slides precede the first sentence-tokenization step.
They compare word, character and illustrative subword splits of `redder!`, then
show the actual English tokenizer's casing, apostrophe, number and punctuation
rules. Token counts, context length and vocabulary IDs are distinct. Both models
retain the same tokenizer and training-fitted vocabulary at generation time.
The subword example is hand-chosen; no BPE tokenizer or new dependency is added.

The window walkthrough distinguishes a story position from a vocabulary ID.
`context_ids` and `target_id` name one input/answer pair. The first two pairs are
built and appended separately, showing the lists after each operation. A loop
adds the remaining five pairs before `torch.tensor` converts the complete lists
to `all_X` (7 × 4) and `all_y` (7). Selecting dataset rows 2 and 3 creates the
batch `X` (2 × 4) and `y` (2), preserving the input/target correspondence.
The batch table places the input tokens beside their IDs and labels both the
target ID and its next-token text. A separate frame shows the integer tensors
and explains their shapes. Tokenization and ID lookup are already complete at
this stage. Input embedding lookup follows, while `y` stays as IDs for the loss.

Counting has separate one-story and training-corpus examples before the totals
table. Generation separates prompt tokenization, window preparation, scoring,
candidate probabilities, selection and appending. Slide code is explicitly
chosen in `build_slow_lesson.py`, with no automatic line truncation. Most excerpts
have one to three lines. Complete loops keep their bodies. The build rejects
invalid Python, missing excerpt choices and oversized snippets. `code_display.py`
adds offline syntax colors without changing the copyable source. The readable
guide keeps the full code and assertions with the same highlighting.

After editing:

```sh
python build_slow_lesson.py --lecture-dir /path/to/attention
```

This executes all small calculations, exports the source notebook and reading
pages, and rebuilds `src/sections/sec19_pipeline.html` in the lecture repository.
Reassemble Part II with that repository's `src/assemble.py`. The export uses an
explicit file allow-list and excludes `work/`, the private Site, and the full corpus.

The compact experiment implementation is in [`wordlm.py`](wordlm.py). The
model equations remain visible in the notebook cells; the module centralizes
data, training, evaluation, and safe NumPy artifacts. Tests live in
[`tests/test_wordlm.py`](tests/test_wordlm.py).

## Dataset and split contract

The source is
[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories),
a synthetic short-story corpus listed under `CDLA-Sharing-1.0`. The scripts
require repository revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` before using the public Dataset
Viewer rows API. They never execute repository code or load dataset pickles.

Three bounded profiles are defined:

| Profile | Source stories | Maximum vocabulary | Intended use |
|---|---:|---:|---|
| `smoke` | 480 | 1,500 | laptop/CPU notebook checks |
| `classroom` | 3,000 | 3,000 | longer local demonstration |
| `dgx` | 6,000 | 4,000 | sweep and final comparison |

The checked DGX-profile raw subset has SHA-256
`ecaf3b99879170311c1fcde645251b9793620f1f5bf57159210fb7ab7180a7b3`.
Exact normalized duplicates are removed before a stable hash assigns whole
stories to train/validation/test. Windows never cross stories. The vocabulary
is fitted on train only; validation and test OOV rates are recorded. The test
split is not read by the sweep and is evaluated only after both model
configurations are frozen.

## Quick laptop setup

From this directory:

```sh
uv venv work/venv --python 3.11
uv pip install --python work/venv/bin/python -r requirements.txt
work/venv/bin/python prepare_data.py --profile smoke --data-dir work/data
work/venv/bin/python -m pytest -q tests/test_wordlm.py
```

Execute every notebook from a clean kernel and keep its outputs:

```sh
export JUPYTER_CONFIG_DIR="$PWD/work/jupyter-config"
for notebook in 01_words_to_probabilities.ipynb \
                02_hyperparameter_experiments.ipynb \
                03_causal_attention_from_scratch.ipynb \
                04_what_did_the_models_learn.ipynb \
                05_training_and_inference_maps.ipynb; do
  work/venv/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1200 "$notebook"
done
```

The first and third notebooks default to the `smoke` profile. Override with
`WORDLM_PROFILE=classroom`. Device selection is automatic (`CUDA`, then Apple
MPS, then CPU); set `WORDLM_DEVICE=cpu` for a CPU-only check.

Export readable HTML after execution:

```sh
mkdir -p html
export JUPYTER_CONFIG_DIR="$PWD/work/jupyter-config"
work/venv/bin/jupyter nbconvert --to html --output-dir html 0*.ipynb
```

## Reproduce the measured sweep and benchmark

The final protocol uses one target per window for both architectures. Context
length, documents, tokenizer/vocabulary, evaluation targets, batch size,
maximum supervised-token budget, and seeds are shared. The one-variable-at-a-
time sweep explains MLP behaviour; for the primary long-context comparison,
learning rate and weight decay are tuned separately for both architectures on
validation. The test set stays sealed until those choices are frozen. The final
benchmark uses seeds 11, 29, and 47 and restores each run's best validation
checkpoint within the shared step ceiling.

```sh
work/venv/bin/python prepare_data.py --profile dgx --data-dir work/data

work/venv/bin/python run_experiments.py \
  --profile dgx --mode sweep \
  --data-dir work/data --output-dir work/results-dgx-profile \
  --device auto --sweep-steps 600 --tune-steps 3000 --batch-size 512

work/venv/bin/python run_experiments.py \
  --profile dgx --mode benchmark \
  --data-dir work/data --output-dir work/results-dgx-profile \
  --device auto --benchmark-steps 6000 --batch-size 512 --seeds 3
```

The benchmark writes `benchmark_partial.json` after every completed seed. Run
the same command to resume; matching completed model/seed pairs are skipped.

## Measured result

The frozen three-seed benchmark was completed on Apple MPS after the DGX pilot
found its GPU occupied by VLLM. Each run used the same 64-token context,
embedding width 64, hidden width 256, batch size 512, 6,000-step ceiling, and
3,072,000 maximum supervised target tokens. Validation-selected checkpoints
were restored before the untouched test split was evaluated. Values below are
mean ± sample standard deviation.

| Model | Parameters | Test cross-entropy | Test perplexity | Runtime per seed |
|---|---:|---:|---:|---:|
| Fixed-window MLP | 2,332,832 | 3.9399 ± 0.0086 | 51.41 ± 0.44 | 67.77 ± 0.39 s |
| One-head causal attention | 1,321,120 | 3.4441 ± 0.0097 | 31.31 ± 0.30 | 71.46 ± 0.81 s |

On this bounded corpus and training budget, attention improved cross-entropy by
0.4958 and reduced perplexity by 39.1%. It used 43.4% fewer parameters and took
about 5.4% longer on this accelerator. This is a scoped long-context empirical
result, not a general claim that attention wins under every corpus, budget, or
hardware setting. The selected steps were 2,250 for all MLP seeds and
6,000/5,250/4,500 for attention; they are recorded rather than hidden. The
machine-readable evidence is in
[`artifacts/benchmark.json`](artifacts/benchmark.json); compact artifact hashes
are recorded in [`artifacts/SHA256SUMS`](artifacts/SHA256SUMS).

## GPU training

Run the same scripts in a CUDA-enabled PyTorch environment and pass `--device cuda`. No particular host or management setup is required.

## Verification

The required checks cover:

- tokenizer token-stable round trips and deterministic document splitting;
- MLP and attention tensor shapes;
- exactly zero future attention weights and normalized attention rows;
- invariance of earlier contextual rows to edited future tokens;
- finite loss/gradients and tiny-batch overfitting;
- exact final-query/full-causal-forward agreement, left-padding exclusion, and
  validation-checkpoint restoration;
- source revision/checksum, duplicate removal, cross-split document overlap,
  train-only vocabulary, and held-out OOV rates;
- token-weighted held-out cross-entropy/perplexity with identical targets and
  target-token budgets.

## References

- [Karpathy, Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Nipun Batra, Generating names using MLPs](https://nipunbatra.github.io/ml-teaching/notebooks/names.html)
- [Attention and language · Part I](https://nipunbatra.github.io/attention/part1.html)
- [Attention and language · Part II](https://nipunbatra.github.io/attention/attention.html)
- [TinyStories dataset card](https://huggingface.co/datasets/roneneldan/TinyStories)
