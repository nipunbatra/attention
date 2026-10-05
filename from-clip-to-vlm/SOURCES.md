# Sources and evidence boundaries

Prepared 1 October 2026; rebuilt 3 October 2026. This lecture follows the visual and pedagogical conventions developed in the chats **“Transformers beyond next-token prediction”** and **“CLIP — classify images with written…”** and the user-supplied 80-minute VLM outline. It is a new, isolated lecture; the existing lectures were not replaced.

## Architecture sources

- [CLIP, Radford et al. (2021)](https://arxiv.org/abs/2103.00020): independent image/text encoders and contrastive matching. CLIP is a VLM in the broader terminology; this lecture distinguishes it from *generative* image-conditioned language models.
- [LLaVA, Liu et al. (2023)](https://arxiv.org/abs/2304.08485): vision-to-language projection and visual instruction tuning. The freeze table is a simplified LLaVA-style recipe, not a universal training procedure.
- [Flamingo, Alayrac et al. (2022)](https://arxiv.org/abs/2204.14198): a Perceiver Resampler combined with gated cross-attention. Thus resampling and explicit cross-attention are not mutually exclusive.
- [BLIP-2, Li et al. (2023)](https://arxiv.org/abs/2301.12597): a Q-Former bridge and frozen pretrained components. Its first stage has representation-learning objectives; the deck does not equate it with the simplified caption-alignment recipe.

The step-by-step architecture drawings are original teaching diagrams. They omit some normalization, residual and implementation details where the omission is identified in the teaching notes. The visual-prefix causal mask is one valid construction; it is not claimed to be the attention mask of every VLM. Visual vectors have the language hidden width but are not vocabulary entries.

## Chat roles, embeddings and positions

- [Chat templates](https://huggingface.co/docs/transformers/chat_templating): formatting roles and adding an assistant generation prompt. USER and ASST are readable teaching labels; real templates may use dedicated special tokens or ordinary text with delimiters.
- [Embedding-table resizing and initialization](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings): new vocabulary items need initialized embedding rows. Existing rows are reused; embedding updates depend on which parameters are trainable.
- [LLaMA implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py): token embedding lookup, sequence position IDs, and RoPE applied to queries and keys inside attention. This differs from adding a positional vector to the input.

Checked 5 October 2026. The role-token walkthrough uses symbolic IDs and the existing three-feature teaching width. Its indices 0–18 follow from twelve image vectors, two single-token role markers and an illustrative four-token question; they are not measured tokenizer output. Inference loads trained embeddings, with no per-prompt reinitialization. A frozen language model can reuse its learned role embeddings without updating them during connector training.

## Browser implementation

- [SmolVLM-256M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct).
- [Official Transformers.js example](https://github.com/huggingface/transformers.js-examples/tree/main/smolvlm-webgpu), Apache-2.0. The worker is adapted from the loading, processing, streaming and generation pattern.
- Runtime: `@huggingface/transformers@3.8.1` from jsDelivr.
- Pinned model revision: `7e3e67edbbed1bf9888184d9df282b700a323964`.
- Precision: FP32 vision encoder and token embeddings, Q4 decoder. The three ONNX files total 574,420,817 bytes (about 574 MB decimal), excluding configuration/tokenizer/runtime files. Memory during execution is larger than file size.
- WebGPU; greedy generation; maximum 64 new tokens; repetition penalty 1.1; image splitting disabled. The latter reduces workload and may reduce fine-detail accuracy.

The lab accepts 11 application image presets plus three fixed-prompt counterfactual conditions and an editable question. It performs no server-side inference. It downloads runtime/model files, then sends no image to an inference API. Independent questions start fresh; the lab does not implement cross-turn KV reuse. Within a generation call, normal autoregressive computation reuse is handled by the library. Rebuild measurements are in `results/vlm_lab_outputs.json`; the original six-case run remains in `output/verification/live-model-results.json`; instructor reference answers are not model output.

## Representative benchmark families

- [OCRBench](https://arxiv.org/abs/2305.07895): text-centric visual tasks.
- [DocVQA](https://arxiv.org/abs/2007.00398): question answering over document images.
- [ChartQA](https://arxiv.org/abs/2203.10244): visual, logical and arithmetic reasoning over charts.
- [MathVista](https://arxiv.org/abs/2310.02255): mathematical reasoning in visual contexts.
- [MMMU](https://arxiv.org/abs/2311.16502): subject-specific multimodal understanding and reasoning across 30 subjects.
- [MMBench](https://arxiv.org/abs/2307.06281): broad multimodal capabilities.
- [MM-Vet](https://arxiv.org/abs/2308.02490): integrated vision-language capabilities.
- [HallusionBench](https://arxiv.org/abs/2310.14566): visual illusion and language hallucination.

The lecture gives no leaderboard rankings. Its invoice, chart, geometry and absence questions are authored classroom analogues, not official benchmark items. Six local examples cannot estimate broad model accuracy.

## Images and licensing

`figures/dog.jpg` is the unchanged `newfoundland_31.jpg` from the preceding ViT/CLIP lectures. Source: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/), O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar, *Cats and Dogs*, CVPR 2012. [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); original owners retain copyright. Original provenance was carried in `../clip/figures/ATTRIBUTION.md` and `../clip/figures/vision1-image-provenance.json`. SHA-256: `617dc6f6dcd26c69303b54985eadd5c75bc07b8ba22cdb1ea511307096098d0a`. Copies in the test suite retain this license. The dog is black and outdoors; there is no ball in this photograph.

`figures/no-bicycle.jpg` is the unchanged coffee photograph from `../clip/figures/coffee.jpg`, by Rachel Michetti, CC0, via [scikit-image sample data](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee). It contains no visible bicycle.

`figures/cat.jpg` is `Persian_98.jpg` reused from the preceding CLIP lecture, from the same Oxford-IIIT Pet dataset and under the same CC BY-SA 4.0 terms.

Apart from the three original paper figures credited below, the remaining figures are original deterministic SVG illustrations made with `src/figures.py`, `src/rebuild_diagrams.py`, `src/intuition_diagrams.py` and `src/projection_diagrams.py`. The desk and room scenes are drawings, the invoice is fictitious, the chart values and thermal measurements are synthetic, and the settings interface is fictional. No financial, personal or clinical records are used. Original SVGs and rendered classroom PNGs can be reused under CC BY 4.0 with attribution to Nipun Batra / ES 667.

## Numerical provenance

The projector and query/key/value matrices are hand-set teaching examples. The cross-attention walkthrough uses all twelve patch positions, two assumed text states, and explicit W_Q, W_K, W_V and W_O matrices. `src/cross_numbers.py` is the numeric source shared by the SVG diagrams; `src/check_math.py` independently recomputes every stage with NumPy. The executed arithmetic notebook reproduces both queries, all twelve weights, both contexts and the residual updates. These are illustrative activations, not measured image features or named semantic coordinates.

The main lecture uses symbolic vocabulary logits and exact measured responses. Its “A black dog” sequence is explicitly illustrative and contains no fabricated numerical token probabilities. The optional arithmetic notebook retains a separately labelled hand-set cross-entropy worksheet (sum 1.1672552213907732; mean 0.23345104427815463), not a checkpoint trace.

Mathematics is prerendered using [KaTeX](https://katex.org/docs/api.html). KaTeX CSS, fonts and its MIT license are included under `vendor/katex/` and embedded in the standalone lecture.

`src/check_math.py` and the executed notebook check the arithmetic and a connector gradient through a fixed linear readout. This final toy is a small classification graph, not a Transformer or a trained VLM. The 44.4× attention-pair reduction in teaching notes is the ratio 640²/96², not a latency or total-compute claim.

## Prefix input-path clarification

The simple patch-preserving interface was checked against the official LLaVA implementation on 3 October 2026: [CLIP patch feature selection](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_encoder/clip_encoder.py) removes the global CLS row in patch mode; [multimodal input preparation](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py) projects image features, embeds text IDs and concatenates image and text embeddings. The diagram omits chat-template markers, position handling, image tiling and implementation-specific sequence limits. N is the patch-row count for the simple single-grid example; compression and other variants are introduced separately.

## Separate-memory clarification

The [original Transformer, §3.2.3](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS3) defines encoder–decoder attention with queries from decoder states and keys/values from encoder outputs. The lecture applies that same connection to vision, then retains a vocabulary head for token scores. This is a generic teaching architecture. [Flamingo, §3.1.2](https://arxiv.org/html/2204.14198v1#S3.SS1.SSS2) uses visual keys/values and language queries too, but adds resampling and gated blocks with its own arrangement.

[ViT, §3.1](https://arxiv.org/html/2010.11929v2#S3.SS1) supplies the patch-sequence and CLS convention. The diagram’s count is conditional arithmetic: (224/16)² = 196 patch rows; retaining CLS adds one. B is batch size, N the retained memory length, and dV its feature width. Image preprocessing, special-token choices and later compression can change N.

## Complete compression reference diagram

[BLIP-2, §3.1 and §3.3](https://arxiv.org/html/2301.12597v3) describes learned queries extracting image features, followed by a projection whose outputs can precede the input text embeddings of a decoder-only LLM. [Flamingo, §3.1.1–3.1.2](https://arxiv.org/html/2204.14198v1) resamples image features and supplies the resulting visual representations to gated decoder cross-attention. The two cards on frame 43 are alternative generic connections; the shared one-block summarizer is a teaching abstraction. Full Q-Former/resampler implementations include additional operations. The summarizer’s query parameters are shared, while its summary outputs depend on the image. The three overview diagrams omit visible credit footers at the user’s request; this source record and the teaching notes retain attribution and implementation boundaries.

## Persistent projector example

Frames 52–62 and 65 reuse the credited Newfoundland photo and the authored question. All numeric image rows, text embeddings and projector weights are deliberately hand-set. The displayed image rows [2,1], [1,3] and [0,2] map to [2,1,3], [1,3,4] and [0,2,2] under the same matrix [[1,0,1],[0,1,1]]. `src/check_math.py` imports the figure data and verifies every displayed numeric row. N and T stay symbolic; ellipses omit middle rows from the drawing, not from the computation. The example illustrates row-wise width adaptation in the prefix interface described above, not a measurement from LLaVA or from the browser model.

## Where the causal mask applies

Pages 65 and 71 separate the encoder's full attention from the standard LLaVA/LLaMA decoder's causal mask. The [ViT method](https://arxiv.org/html/2010.11929v2#S3.SS1) supplies contextual image features. LLaVA's [multimodal input preparation](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py) inserts projected image features into the text-embedding sequence; its [language-model wrapper](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/language_model/llava_llama.py) forwards that sequence to LLaMA. The [Transformers 4.37.2 LLaMA implementation](https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py#L921-L964) prepares the causal mask using the full input-embedding sequence and passes it through the decoder layers. These sources support the lecture's lower-triangular mask including visual positions; this is not a claim that every VLM uses the same mask. The example omits preceding system/role tokens. Prefill computes supplied positions together with the mask; generation appends new answer tokens afterwards. V1–V3 in the small mask are the projected visual rows denoted z1…zN in the earlier numeric example.

## Original paper architecture slides

After the mechanisms have been derived, pages 74, 106 and 107 show the original LLaVA Figure 1, Flamingo Figure 3 and BLIP-2 Figure 3 respectively. Each includes the exact paper title, authors, venue/year, a paper link and three gradual explanations connecting the authors’ modules to the lecture. The original papers’ figure colors and notation are retained; speaker notes identify differences from our teaching diagrams. In particular, Flamingo uses an NFNet visual encoder; BLIP-2 shows both a decoder-only OPT route and an encoder–decoder FlanT5 route.

Figure files, source versions, preparation steps, hashes and rights are in [the paper-figure attribution record](figures/papers/ATTRIBUTION.md). LLaVA and BLIP-2 carry CC BY 4.0; the Flamingo source carries the arXiv nonexclusive distribution license and is not covered by the license for our original course artwork.

## Visual-prefix pseudocode recap

Page 75 is an authored high-level code summary, checked against LLaVA’s [image projection and multimodal input preparation](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py) and [language-model wrapper](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/language_model/llava_llama.py). It represents the simple visual-first prefix already derived in the lecture. Function names are conceptual, rather than executable package APIs; the batch dimension, image preprocessing, chat template, positional handling and caching are abstracted. All N patch rows are retained. The decoder call denotes the full causal Transformer stack; its final supplied state is mapped to vocabulary scores. Greedy argmax selects the next text token ID. Speaker notes explain the generation cache and the distinction between a text token ID and its display string.

## Opening CLIP revision

The four-slide revision reuses Nipun Batra’s [preceding CLIP lecture](https://nipunbatra.github.io/attention/clip/?present#closing-summary-encoders). `figures/clip-recap-{encoders,training,uses}.svg` are snapshots of its closing diagrams on 5 October 2026, with reveal attributes adapted to this deck. The original toy illustrations and hat-image pair remain embedded in the SVGs; their underlying provenance is recorded in the [CLIP sources](https://nipunbatra.github.io/attention/clip/SOURCES.md). No new measured scores are introduced.

## Two-model browser comparison

The lab adds [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct/tree/a7da5b986cb59b408707209984f360a5f4ad7e47), pinned alongside the existing 256M revision in `src/lab-models.json`. The [Hugging Face introduction](https://huggingface.co/blog/smolervlm) documents the two model sizes and browser demos. Both run with Transformers.js 3.8.1, fp32 embeddings and vision encoder, q4 merged decoder, no image splitting, greedy decoding, 64-token limit and repetition penalty 1.1. The 500M selected ONNX files total 811,545,504 bytes before tokenizer/config/runtime files (about 812 MB). `results/vlm_lab_500m_outputs.json` preserves all 14 new outputs; image SHA-256 hashes and decoding settings match the 256M saved runs. The two recording sessions were on different dates, so their timings are not a controlled speed benchmark.

## Concrete cross-attention example

The three-slide visual lookup walkthrough reuses the credited Newfoundland photo. The continuations dog and trees are authored reference answers. The twelve image rows come from the running 3 × 4 patch example; all twelve keys and values are shown. These are symbolic vectors, not extracted model activations. The weighted readout follows [Attention Is All You Need, §3.2](https://arxiv.org/html/1706.03762v7#S3.SS2). No numeric weights, attention heatmap or measured generation is asserted. The following thirteen worked diagrams keep the same twelve patch positions and introduce explicit toy numbers. Both questions are traced through their final supplied token states, Q/K/V, scores, softmax, all twelve value contributions, output projection and residual update. The two photo overlays label calculated toy weights; they are not measured attention maps or evidence of grounding. All displayed results are rounded from full precision. The new widths and feature values belong to this independent arithmetic example; they are not outputs of the earlier projector example.
