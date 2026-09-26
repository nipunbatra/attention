# Vision I: reference review and incorporated changes

Reviewed 26 September 2026. Review was silent. Both YouTube players were muted and paused; targeted transcripts and one paused coding-lecture frame were inspected. This was not uninterrupted viewing of either recording.

| Source | Review completed | Incorporated naturally |
|---|---|---|
| [UCSD CSE252D, lecture 2](https://cseweb.ucsd.edu/~mkchandraker/classes/CSE252D/Spring2024/Lectures/lec02_visiontransformers.pdf) | PDF text and rendered Q/K/V slide 21; attention/self-attention sequence on slides 18–25 | Keep real crops beside the receiver and sources. Separate choosing weights from reading values. Explicitly move from an imagined question to queries computed from image rows. |
| [MIT VisionBook, chapter 26](https://visionbook.mit.edu/transformers.html) | Chapter text, especially §§26.4, 26.6–26.9 | Draw attention across rows and the MLP within one row. Explain that the task determines the readout. Add mean pooling as a concrete alternative to CLS. |
| [Vizuara: Introduction to Vision Transformer (ViT)](https://www.youtube.com/watch?v=aIi5FsdURUA) | Full original-English auto-captions retrieved; targeted review at 12:38–17:47 (context), 40:44–45:35 (Q/K/V), and 1:05:30–1:20:57 (text-to-image bridge, architecture and classification) | Reinforces the existing crop/context motivation and task comparison. Keep a visible image as we change representations, and identify the class vocabulary and readout explicitly. |
| [Vizuara: Build Vision Transformer ViT From Scratch — Intuition and coding](https://www.youtube.com/watch?v=ZRo74xnN2SI) | Full original-English auto-captions retrieved; targeted review at 3:47–24:49 (architecture), 51:25–59:01 (patch projection), 1:17:21–1:25:16 (block/readout), and 1:43:50–1:50:24 (training/debugging) | Six new implementation slides: compute a filter dot product, choose stride, count learned parameters, separate images in a batch, calculate the wrong-axis result, and check the same image alone and batched. The executed lab includes a deliberately wrong layout as a negative control. |

## Adaptation choices

The new diagrams and all numerical examples are original to this lecture. The previous text series supplies the teaching rhythm: one question, a visible object, a prediction, then one calculation. The realistic photo motivates the two-source demonstration; its scalar keys and named evidence coordinates are explicitly hand-chosen. The measured pretrained-model maps remain a separate experiment.

A natural-language question can help explain a query, but an image-only classifier receives no text prompt. Every image row produces numerical Q, K and V. Matching and message coordinates can differ; the two-source example uses query/key width one and value width two.

CLS is trained through the image-label loss. It starts as the same parameter vector for every image and receives image-dependent messages. It is a readout choice, not a requirement of classification. A pooling-based architecture must be trained for that readout; removing CLS from the supplied checkpoint is not a guaranteed substitution.

The source chapter's preliminary unnormalized counting analogy is not used as a softmax counting recipe: normalized weights times constant-one values always sum to one. We retain the checked operation A @ V and distinguish permutation equivariance of rows from invariance of an image readout.

The full source shortlist and earlier article/video review are in [the research plan](VISION1_REDESIGN_PLAN.md). Current teaching navigation is in [the slide map](VISION1_SLIDE_MAP.md).

## YouTube retrieval and source checks

The original rate-limited attempts were resolved with yt-dlp 2026.08.19, curl-cffi HTTP impersonation and the Node JavaScript runtime. Public caption retrieval succeeded without browser cookies. YouTube’s browser transcript export also succeeded for both recordings. Full captions and transient download metadata remain local; this repository publishes original summaries, source links and numerical examples.

Reproducible public-caption command (no audio or video download):

```sh
uvx --from 'yt-dlp==2026.08.19' --with curl-cffi yt-dlp \
  --impersonate chrome --skip-download --write-auto-subs \
  --sub-langs en-orig --sub-format vtt --js-runtimes node \
  'https://www.youtube.com/watch?v=aIi5FsdURUA' \
  'https://www.youtube.com/watch?v=ZRo74xnN2SI'
```

Useful teaching moves still need technical checking:

- **Patch projection learns.** The coding transcript describes it in places as having no learned weights. Our example follows [PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): two grayscale 2×2 filters with biases have ten trainable parameters. The notebook checks the outputs, gradients and equivalence to a linear projection.
- **CLS is optional.** The coding lecture’s early comparison suggests it improves accuracy. [The original ViT paper, Appendix D.3](https://arxiv.org/html/2010.11929v2#A4.SS3) reports that its initial pooling gap was explained by a different learning rate. Keep the existing CLS-versus-pooling calculation and explain that either readout must be trained appropriately.
- **Raw patch length and embedding width differ.** A 16×16 RGB patch has 768 input numbers. The learned projection can produce a different width; our real checkpoint uses D=192.
- **Tensor shape alone cannot establish correctness.** The coding lecture’s [batch-first debugging segment](https://www.youtube.com/watch?v=ZRo74xnN2SI&t=6346s) motivates our original two-image arithmetic example. [PyTorch’s input layout documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) and an executed negative control establish the behavior. Check patch outputs as well as class scores: a broken CLS path can ignore every patch and still appear batch-independent.

New slides follow section 8’s patch code and attention code. The worked results and source provenance are saved in `figures/vision1/video-examples.json` and `figures/vision1/video-review.json`. No accuracy reported in a source video is presented as our result.
