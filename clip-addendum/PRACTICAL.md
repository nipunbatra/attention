# Build the two ideas, one stage at a time

These are classroom implementations. The search baseline is a small text index. The word reader is a proposed supervised experiment on cached real features. Neither script is a reproduction of the Mac apps.

## 1. Search: indexing and lookup are different jobs

**Indexing:** choose files → extract pieces → encode each piece → normalize → save the vectors beside their source locations.

**Lookup:** encode the new question → normalize → compare with stored vectors → rank → open the corresponding source.

With N pieces and the model's native 768-dimensional output, the stored array is `N × 768`. A query is `768`. `X @ q` returns N cosine similarities. They are scores, not confidence probabilities. There is no training in these steps.

Start with a small JSONL file, one extracted text piece per line:

```json
{"piece_id":"clip-17","file":"clip/index.html","slide":"batch-encoding-job","text":"CLIP uses causal self-attention in its text branch. The final EOT state reads the preceding description."}
{"piece_id":"clip-96","file":"clip/index.html","slide":"main-row-denominator","text":"For image-to-text matching, normalize each row of the logit matrix independently with softmax."}
{"piece_id":"clip-103","file":"clip/index.html","slide":"row-probabilities","text":"Save the image-to-text probability matrix. Return to the original logits before computing text-to-image probabilities down columns."}
```

The passages above are authored examples. For an actual course index, extract the current slide text and preserve its real slide ID and lecture URL. Split longer documents into passages small enough to inspect. Preserve headings and source locations. Remove navigation, repeated footers and empty content before embedding.

Install in a fresh Python environment with a SentenceTransformers/Transformers release that supports EmbeddingGemma 2:

```sh
python -m pip install -U numpy torch sentence-transformers
python code/semantic_search.py index pieces.jsonl course-index
python code/semantic_search.py query course-index "Why can EOT represent the whole sentence?"
```

The first command may download the model; subsequent queries reuse its local cache. The script resolves the model revision to a commit, saves it with the index, and uses that same revision for queries. It uses the model card's `Document` and `SearchQuery` prompts, full 768D outputs and L2 normalization. Run indexing once, then query repeatedly. This minimal CLI loads the model on each invocation; a web app/server would keep it loaded.

This script does **text search only**. To add media, implement these extraction and encoding adapters using the official model runtime:

| Source | Piece sent to the encoder | Metadata to keep |
|---|---|---|
| Photo | Decoded image | File path, image ID |
| PDF | Page image and/or text passage | Document, page number, bounding region if available |
| Recording | Audio window, consistently sampled | File, start/end times, sample rate |
| Video | Sampled frames and audio windows | File, timestamp, track |
| Code | Function or coherent source passage | Repository, path, revision, start/end lines |

Use media inputs for media: a filename string is not the image or audio itself. Task prefixes apply to text only. For example, audio must follow the model's mono 16 kHz preprocessing. Do not claim a text-only checkpoint or a text transcript is a native audio encoder. All compared vectors must come from compatible model revisions and representation settings.

For ten or a few thousand pieces, exact matrix multiplication is a useful starting point. Larger collections may need an approximate-nearest-neighbor index. Keep the source metadata either way. Refresh changed pieces; rebuild when changing the embedding model. The companion script deliberately makes a new directory for every index; it does not implement an incremental file watcher.

DigUp adds SQLite storage, keyword/OCR search, modality calibration, per-file grouping and source opening. An exact identifier such as an invoice number is a good reason to combine embedding scores with lexical search. Write a small set of questions with known source locations and check top-k retrieval before tuning rankings.

## 2. Train a reader on frozen 768D features

The public speech demo describes a frozen audio encoder and a trained reader, but does not disclose the reader's exact input width, layers, alignment or loss. The following architecture and dataset are **our proposed teaching experiment**.

Start with [Speech Commands v0.02](https://arxiv.org/abs/1804.03209), restricted to eight words in this exact order:

```python
words = ["yes", "no", "up", "down", "left", "right", "stop", "go"]
```

Each example is a WAV clip and one word label. Preserve `validation_list.txt` and `testing_list.txt`; use the other selected examples for training. The speaker ID is the filename prefix before `_nohash_`. Check that no speaker appears in multiple splits. Torchaudio's [Speech Commands loader](https://github.com/pytorch/audio/blob/main/src/torchaudio/datasets/speechcommands.py) implements these split lists.

Prepare the cache **before** running the reader trainer:

1. Decode each selected WAV as mono 16 kHz. Record padding/cropping and other preprocessing; use the same policy for every split.
2. Run the frozen EmbeddingGemma 2 audio path on each clip and retain its native 768D output. Normalize it. Do not run the text encoder on the WAV filename. This feature-export adapter is runtime-specific and is not included in the trainer.
3. Encode the eight candidate word strings once with the same frozen model, using the text `Classification` prompt; normalize the resulting `8 × 768` matrix E.
4. Save the arrays and provenance to a NumPy `.npz` file. Do not mix features from different checkpoints or different normalization/projection settings.

| Cache key | Shape/type | Meaning |
|---|---|---|
| `x_train`, `x_val`, `x_test` | `(N_split, 768)`, float | Real frozen audio features |
| `y_train`, `y_val`, `y_test` | `(N_split,)`, integer | Word index 0 through 7 |
| `speakers_train`, `speakers_val`, `speakers_test` | `(N_split,)`, Unicode string | Speaker IDs for leakage checking |
| `E` | `(8, 768)`, float | Frozen candidate text vectors |
| `words` | `(8,)`, Unicode string | Exact order given above |
| `metadata_json` | Scalar Unicode string | JSON with `model`, `revision`, `audio_preprocessing`, `text_prompt`, `dataset` |

Save using `np.savez("commands-cache.npz", x_train=..., ..., metadata_json=json.dumps(provenance))` after extracting real features. The ellipses describe fields to fill; they are not an executable data-preparation script. The trainer rejects absent features and speaker overlap rather than substituting artificial embeddings.

For a minibatch of B clips:

```text
X: B × 768
   ↓ Linear(768, 128), ReLU
H: B × 128
   ↓ Linear(128, 768), L2 normalization
R: B × 768
   ↓ R @ E.T / 0.1
S: B × 8 word logits
   ↓ cross_entropy(S, y), averaged over B
one scalar training loss
```

There are 197,504 trainable parameters, including biases. Each linear unit forms a weighted sum of its inputs and adds a bias. The second layer outputs a vector in the comparison space; the word-vector table converts it to eight scores. The chosen temperature 0.1 changes softmax sharpness. It is a hyperparameter, not an extra label.

```sh
python code/train_reader.py commands-cache.npz --epochs 20 --output reader-result
```

The script shuffles minibatches, uses AdamW, selects the epoch with the lowest validation loss, then reports held-out word accuracy and saves the reader. It also evaluates the direct `X @ E.T` zero-shot baseline on the same test set. A `Linear(768, 8)` classifier is another useful baseline. Keep the test set out of hyperparameter selection.

For inference, preprocess a new clip, obtain its frozen 768D vector, apply the trained reader, compare with E, and return `words[argmax(scores)]`. This eight-word model will still choose one of those words for an unrelated sound: it has no learned “unknown” option. A useful extension adds unknown/background examples and tests rejection explicitly.

This is **head-only training**. The embedding encoder stays fixed, so caching features is valid. To fine-tune the encoder itself, feed raw clips through it during training, keep its computational graph, include the chosen encoder parameters in the optimizer, and do not use a detached feature cache. If the embedding space changes, recompute the affected candidate vectors and retrieval index. Start with the frozen-head experiment because its inputs and updates are easy to inspect.

## 3. From one command to a whole transcript

An isolated-word label is not a sentence transcript. A full recognizer needs time-indexed features, variable-length word sequences and an alignment/decoding method. A single pooled whole-recording vector does not provide a usable word-by-word timeline.

One proposed next dataset is [LibriSpeech](https://www.openslr.org/12): begin with `train-clean-100`, tune on `dev-clean`, and reserve `test-clean`/`test-other` for final word error rate. This is a suggestion for students, **not a claim about the demo author's training data**.

Two possible designs are a temporal feature sequence trained with CTC, or word-aligned segments obtained with a separate forced aligner. Both require additional implementation and evaluation. Neither is asserted to be the author's unpublished method. Word lookup alone does not specify how to place boundaries, merge repeated predictions, handle out-of-vocabulary words or restore punctuation.

Report WER = (substitutions + deletions + insertions) / reference words, runtime on the same audio/hardware, and performance on names, accents and noise. Success at eight-way command classification does not establish full transcription quality.

## Verification scope

```sh
python code/semantic_search.py check
python code/train_reader.py --check
```

These checks use illustrative arrays/random tensors to verify dimensions, dot products and gradient flow. They do not test pretrained encoding or establish model accuracy. No trained-reader accuracy or reproduced app benchmark is supplied with this addendum.

The API contract follows the [official model card](https://huggingface.co/google/embeddinggemma-2); detailed application provenance is in [SOURCES.md](SOURCES.md).
