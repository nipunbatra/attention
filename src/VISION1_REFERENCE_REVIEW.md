# Vision I: reference review and incorporated changes

Reviewed 26 September 2026. Everything was read silently; no audio or video playback.

| Source | Review completed | Incorporated naturally |
|---|---|---|
| [UCSD CSE252D, lecture 2](https://cseweb.ucsd.edu/~mkchandraker/classes/CSE252D/Spring2024/Lectures/lec02_visiontransformers.pdf) | PDF text and rendered Q/K/V slide 21; attention/self-attention sequence on slides 18–25 | Keep real crops beside the receiver and sources. Separate choosing weights from reading values. Explicitly move from an imagined question to queries computed from image rows. |
| [MIT VisionBook, chapter 26](https://visionbook.mit.edu/transformers.html) | Chapter text, especially §§26.4, 26.6–26.9 | Draw attention across rows and the MLP within one row. Explain that the task determines the readout. Add mean pooling as a concrete alternative to CLS. |
| [YouTube aIi5FsdURUA](https://www.youtube.com/watch?v=aIi5FsdURUA) | Not reviewed: webpage retrieval failed; text-only caption retrieval returned HTTP 429 and a sign-in/bot check | No content attributed to this recording. |
| [YouTube ZRo74xnN2SI](https://www.youtube.com/watch?v=ZRo74xnN2SI) | Not reviewed: English captions were listed but their retrieval returned HTTP 429 | No content attributed to this recording. |

## Adaptation choices

The new diagrams and all numerical examples are original to this lecture. The previous text series supplies the teaching rhythm: one question, a visible object, a prediction, then one calculation. The realistic photo motivates the two-source demonstration; its scalar keys and named evidence coordinates are explicitly hand-chosen. The measured pretrained-model maps remain a separate experiment.

A natural-language question can help explain a query, but an image-only classifier receives no text prompt. Every image row produces numerical Q, K and V. Matching and message coordinates can differ; the two-source example uses query/key width one and value width two.

CLS is trained through the image-label loss. It starts as the same parameter vector for every image and receives image-dependent messages. It is a readout choice, not a requirement of classification. A pooling-based architecture must be trained for that readout; removing CLS from the supplied checkpoint is not a guaranteed substitution.

The source chapter's preliminary unnormalized counting analogy is not used as a softmax counting recipe: normalized weights times constant-one values always sum to one. We retain the checked operation A @ V and distinguish permutation equivariance of rows from invariance of an image readout.

The full source shortlist and earlier article/video review are in [the research plan](VISION1_REDESIGN_PLAN.md). Current teaching navigation is in [the slide map](VISION1_SLIDE_MAP.md).
