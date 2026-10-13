# Sources and media credits

Prepared 13 October 2026. The two demos are independent applications. Neither is a WebGPU implementation of the course lab.

## DigUp

- [Abdur Rahim’s X demo, 10 October 2026](https://x.com/_ARahim_/status/2108649742217679257): original 59.6-second video, streamed from X’s media host in the HTML.
- [Repository at revision f9b8106](https://github.com/ARahim3/DigUp/tree/f9b8106ffb8308b4f48b76149d270f7c2e6b4918): platform, examples, readers, incremental indexing and model runtime.
- [Searcher.swift](https://github.com/ARahim3/DigUp/blob/f9b8106ffb8308b4f48b76149d270f7c2e6b4918/Sources/DigUpKit/Searcher.swift): matrix–vector product (lines 294–312), modality calibration (315 onward), hybrid ranking and file grouping (116–250).
- [IndexStore.swift](https://github.com/ARahim3/DigUp/blob/f9b8106ffb8308b4f48b76149d270f7c2e6b4918/Sources/DigUpKit/IndexStore.swift): segment locations, normalized 768D float16 vectors and FTS5.
- [LlamaEmbedder.swift](https://github.com/ARahim3/DigUp/blob/f9b8106ffb8308b4f48b76149d270f7c2e6b4918/Sources/LlamaRuntime/LlamaEmbedder.swift): local pretrained inference.
- [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2): model capabilities. Do not confuse it with the earlier text-only EmbeddingGemma 300M.

The deck simplifies the search pipeline. Actual DigUp adjusts cosine scores by modality, applies semantic/keyword eligibility rules, combines semantic salience with word coverage and file-kind hints, groups by file and chooses a source location. Its raw cosine is not the final ranking score and is not a probability.

## Embedding Gemma Transcribe

- [Shahir’s X demo, 12 October 2026](https://x.com/TslShahir/status/2109660368381395414): original 26.1-second video.
- [Author’s method thread](https://x.com/TslShahir/status/2109660369429954631): frozen base model, trained reader, nearest lookup among about 229,000 words; training in under two hours on 8 H100 GPUs.
- [Project site](https://eg2.haku.sh/): frozen audio encoder, reader into the text space, nearest English word, then punctuation/capitalization; Mac/Neural Engine runtime.
- [Author’s WER reply](https://x.com/TslShahir/status/2109732426591478238): 5.18% LibriSpeech test-clean; about 11.5–11.8% on the harder test set.
- [Accent caveat](https://x.com/TslShahir/status/2109699140196421903): author reports higher WER on their own speech.

Speed and accuracy are author-reported, not reproduced here. The site’s rounded “12.6 s / 288×” values are presented as about 13 seconds per hour in the lecture. The exact reader layers, distance function, slice timing, alignment and training objective are not disclosed in these public sources. The deck does not claim CTC, autoregressive decoding, or lossless recovery from a pooled embedding.

## Figures, toy numbers and video fallbacks

- `figures/digup-demo.jpg` and `figures/speech-demo.jpg` are unaltered frames at 12 seconds from the respective authors’ videos, used as attributed posters and static PDF figures. They show the author demonstrations, not outputs produced for this course. Rights remain with their creators.
- The HTML streams each original video with manual playback. No autoplay, installation or account access is required by the lecture. Network or X availability can affect playback; each figure includes the original post link.
- The PDF includes the static frames and clickable source links. It cannot play video.
- The numeric examples on slides 4 and 10 are hand-chosen unit vectors. Filenames, timestamps, word scores and rankings in those examples are illustrative, not measured DigUp/Transcribe outputs.
- Architecture diagrams are editable HTML/SVG and use the existing CLIP lecture’s colours and progressive-reveal conventions. Slides 2, 5, 7, 9 and 16 are high-level explanations of the cited mechanisms.

## Proposed classroom implementation

Slides 3 and 6 show a minimal index/query baseline. Slides 11–15 describe a separate teaching experiment, not a reconstruction of the speech demo.

- [Speech Commands paper](https://arxiv.org/abs/1804.03209) and [Torchaudio split implementation](https://github.com/pytorch/audio/blob/main/src/torchaudio/datasets/speechcommands.py): v0.02, eight selected command words, official split lists and speaker IDs.
- [LibriSpeech](https://www.openslr.org/12): proposed follow-on speech/transcript data, 16 kHz, CC BY 4.0. The author reports evaluation on LibriSpeech; their training corpus has not been established.
- [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2): native 768D output, text prompt names, media preprocessing and SentenceTransformers API.
- [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html): takes logits and class indices, with mean minibatch reduction.

The 768 → 128 → 768 reader, eight-word subset, temperature 0.1 and AdamW settings are teaching choices. Its 197,504 parameters are calculated from the two affine layers. The three-way probability example (0.70, 0.20, 0.10) is illustrative; −ln(0.70) ≈ 0.357. CTC and forced alignment are possible future implementation paths, not claims about the author's method. No speech training accuracy has been measured for this addendum.
