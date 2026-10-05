# From CLIP to Vision-Language Models

How Images Become Context for Language Generation · ES 667 · Nipun Batra · IIT Gandhinagar

- [Read the HTML lecture](https://nipunbatra.github.io/attention/from-clip-to-vlm/)
- [Present](https://nipunbatra.github.io/attention/from-clip-to-vlm/?present#cover)
- [158-page PDF](from-clip-to-vlm.pdf)
- [VLM Lab](demo.html) · [recorded mode](recorded-demo.html)
- [Sources and image credits](SOURCES.md) · [original-paper figure credits](figures/papers/ATTRIBUTION.md)

The opening four slides revise CLIP’s encoders, training and applications, then introduce the generation question. The VLM lab compares SmolVLM-256M and SmolVLM-500M on identical inputs, with live or recorded outputs.

This lecture follows [CLIP](https://nipunbatra.github.io/attention/clip/).
It derives visual prefixes, separate visual memory and learned-query compression,
then connects them to LLaVA, Flamingo and BLIP-2, training and generation.

Right / Space / N reveals the next part; Left / P goes back. F shows all or restarts
the current slide. R switches between reading and presentation. Open Controls
for the lecture map, questions, answers, notes and reading mode. The visual style
matches the preceding CLIP lecture. The PDF contains
the 158 completed slides; the HTML has 528 build states.

The HTML lecture embeds its figures, fonts and mathematics. Recorded lab mode
works with the accompanying local files. Live inference requires WebGPU and an
initial model download of about 575 MB (256M) or 812 MB (500M). Recorded responses are actual model
outputs, including errors; they are distinct from the authored reference answers.
