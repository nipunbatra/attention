# Sources and provenance

Checked 4 October 2026. Original teaching adaptations of representative published designs; no leaderboard claim.

- [CLIP · 2021](https://arxiv.org/abs/2103.00020)
- [OWL-ViT · 2022](https://arxiv.org/abs/2205.06230)
- [GLIP · 2022](https://arxiv.org/abs/2112.03857)
- [DETR · 2020](https://arxiv.org/abs/2005.12872)
- [Grounding DINO · 2024](https://arxiv.org/html/2303.05499v5)
- [Segment Anything · 2023](https://arxiv.org/abs/2304.02643)
- [Grounded SAM · implementation](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [Kosmos-2 · 2023](https://arxiv.org/html/2306.14824v3)
- [Florence-2 · 2024](https://arxiv.org/abs/2311.06242)
- [Florence-2 · task parser](https://huggingface.co/microsoft/Florence-2-large/blob/main/processing_florence2.py)
- [LISA · 2024](https://arxiv.org/abs/2308.00692)
- [LISA · segmentation interface](https://github.com/JIA-Lab-research/LISA/blob/main/model/LISA.py)
- [GLaMM · 2024](https://arxiv.org/abs/2311.03356)
- [COCO · evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
- [SAM 2 · 2024](https://arxiv.org/abs/2408.00714)

## Source boundaries

OWL-ViT uses local image features with detection training; it is not pure global CLIP. Grounding DINO is a language-conditioned detector, not a chat VLM. Original released SAM consumes spatial prompts; Grounded SAM composes a language grounder with it. Kosmos-2 uses two corner location tokens on a 32×32 grid. The preceding four axis-token grammar is illustrative. Florence-2 task strings follow its official processor; referring segmentation returns polygons. LISA projects a segmentation-token hidden representation and supplies it alongside spatial image embeddings to a mask decoder. No hidden reasoning trace is displayed.

## Assets and numerical provenance

`src/rebuild_figures.py` is the canonical source for diagrams and the fixed park. Reused people, dogs, kiln and overlap drawings originate in `src/figures.py`. All are authored illustrations. The child shape is revised consistently in slides, clean model inputs and explorer. Main-slide overlays are authored references. Toy cosine, attention and quantization numbers are pedagogical arithmetic. Actual model outputs, when present, are stored separately in `results/grounded_vision_outputs.json`; all measured slide content is loaded from that file.

## Local style references

Recent `from-clip-to-vlm`, `clip`, `beyond-attention` and `beyond-boxes` lectures: stable staged geometry, mechanism before model name, semantic modality colors, KaTeX, final takeaways, questions, notes and optional route. No sibling files are modified.
