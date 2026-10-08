# From CLIP to Vision-Language Models

How Images Become Context for Language Generation · ES 667 · Nipun Batra · IIT Gandhinagar

- [Read the HTML lecture](https://nipunbatra.github.io/attention/from-clip-to-vlm/)
- [Present](https://nipunbatra.github.io/attention/from-clip-to-vlm/?present#cover)
- [174-page PDF](from-clip-to-vlm.pdf)
- [VLM Lab](demo.html) · [recorded mode](recorded-demo.html)
- [Three connections in Colab](https://colab.research.google.com/github/nipunbatra/attention/blob/main/from-clip-to-vlm/notebooks/vlm-three-connections.ipynb) · [executed notebook](notebooks/vlm-three-connections.ipynb)
- [Sources and image credits](SOURCES.md) · [original-paper figure credits](figures/papers/ATTRIBUTION.md)

The opening four slides revise CLIP’s encoders, training and applications, then introduce the generation question. The VLM lab compares SmolVLM-256M and SmolVLM-500M on identical inputs, with live or recorded outputs.

This lecture follows [CLIP](https://nipunbatra.github.io/attention/clip/).
It derives visual prefixes, separate visual memory and learned-query compression,
then connects them to LLaVA, Flamingo and BLIP-2, training and generation.
Idea 1 includes four compact slides for the exact input, vocabulary scores, token
choice and embedding feedback. All projected patch vectors form a visual prefix.
The separate generation recap has been removed; compression leads directly
to training. Token IDs are verified; the answer and toy scores are illustrative.
The measured image-sensitivity check appears in optional evaluation reading.
Training leads directly to the closing summary. The ten-slide inference
refresher is optional reading after the multi-image extension; original links
for prefill, decoding, caching and multi-turn context remain valid.
The evaluation section follows inference in optional reading; its
false-premise example, image-sensitivity results and diagnostic suite are retained.

Three visual recap slides close the main lecture: ViT/CLIP/generation, the three
connection ideas, and a complete image-to-token path. The two-stage training
overview now opens a dedicated Captions to instructions subsection before the
summary: supplied reference data, next-token loss, selected weight updates, then
inference without a reference answer. The diagrams reuse Beyond Attention’s
token rows, attention patterns and model portraits.
A separate backup contents page groups all optional material: A multiple images,
B inference, C evaluation and metrics, D model details and further training,
E sources. Each group has a direct link and a heading in the lecture map.
The training sequence keeps the same labelled example: reference tokens →
probabilities → answer loss → projector gradient → six commented training statements.
The code slide reuses the prepared input and adds labels, loss and an update;
the four familiar preparation statements remain in its notes.
Immediately after the first target loss, a single side-by-side slide contrasts
teacher forcing (reference A enters the next prefix) with greedy inference
(model-selected The enters the next prefix). Sampling is also distinguished.
The following slides align all five reference targets and compute the mean loss.
Three diagrams finish training: caption alignment updates the projector;
the checkpoint carries forward to varied instruction examples; instruction tuning
updates both projector and language model while vision remains fixed.

Right / Space / N reveals the next part; Left / P goes back. F shows all or restarts
the current slide. R switches between reading and presentation. Open Controls
for the lecture map, questions, answers, notes and reading mode. The visual style
matches the preceding CLIP lecture. The PDF contains
the 174 completed slides; the HTML has 712 build states.
The main lecture has 138 slides, followed by 36 optional slides.
An optional [multi-image change-detection walkthrough](https://nipunbatra.github.io/attention/from-clip-to-vlm/?present#optional-multi-image)
follows the closing summary: two ordered images → per-image summaries → attention
across both groups → one generated answer. It also compares joint summarization
and image attention masks. The packing diagram distinguishes ordinary text labels from image-derived vectors.
The model-specific special-marker slide has been removed; its old link opens the code recap.
A commented pseudocode slide immediately follows the packing diagram and traces
the same ordered embedding sequence through one next-token prediction.
Its scenes, answer and weights are authored examples,
not measured model outputs. The detailed Flamingo and BLIP-2 paper diagrams
follow in backup. Open optional material from the slide map when needed.

The HTML lecture embeds its figures, fonts and mathematics. Recorded lab mode
works with the accompanying local files. Live inference requires WebGPU and an
initial model download of about 575 MB (256M) or 812 MB (500M). Recorded responses are actual model
outputs, including errors; they are distinct from the authored reference answers.
