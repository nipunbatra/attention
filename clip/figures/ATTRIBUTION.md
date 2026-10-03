# Photograph attribution

Photographs from **Oxford-IIIT Pet**, by O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar, *Cats and Dogs*, CVPR 2012.

- [Original dataset and license](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [timm dataset mirror](https://huggingface.co/datasets/timm/oxford-iiit-pet), revision `089695c834a7deb60505b7cc506672db1c31a6aa`
- [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Original image owners retain copyright.

The original JPEG bytes were copied from the current Vision I reference assets. They are displayed without editing and with their original aspect ratios. The captions in the CLIP worksheet were written for this lesson; they are not claimed source annotations. Hand-chosen vectors attached to these photos are not measured image features.

| File | Original dimensions | Mirror test row | SHA-256 |
| --- | --- | --- | --- |
| `newfoundland_31.jpg` | 500 × 334 | 0 | `617dc6f6dcd26c69303b54985eadd5c75bc07b8ba22cdb1ea511307096098d0a` |
| `Persian_98.jpg` | 375 × 500 | 9 | `6071d201543c999e063615d6c2cc674ea004adb11711d34d29d3df11c0012292` |
| `pug_57.jpg` | 500 × 333 | 1 | `c3ffdc25566ae138be7d76121c7225fba6879568739338b240b60d97181d2609` |

The first two are the opening dog/cat photographs from Vision I; the pug is one of its additional dataset examples. Original provenance for the first two is preserved in `vision1-image-provenance.json`. All `.svg` diagrams in this folder are original teaching drawings generated from `src/diagrams.py` and `src/lesson.py`.

The paired-supervision and alignment drawings (`mot-requirements*.svg`, `mot-alignment*.svg`) are authored in `src/alignment_slides.py`. Neural-network icons and 2D unit-vector directions are schematic, not measured model outputs or checkpoint comparisons. The dog caption on these slides was written for the lesson. The photograph retains the attribution above; the learning objective is credited to Radford et al. (2021).

The classifier drawings (`classifier-vectors*.svg`, `text-weights*.svg`) are authored in `src/classifier_slides.py` and reuse the schematic network icon from `src/alignment_slides.py`. The photograph is the same credited Newfoundland image. The separate three-feature arithmetic example is hand-chosen; the CLIP text-vector comparison uses the existing measured embeddings.


## Motivation revision images

Astronaut portrait: NASA, Eileen Collins, public domain, via scikit-image. Coffee: Rachel Michetti, CC0; Chelsea: Stefan van der Walt, CC0; rocket: SpaceX, public domain; all via scikit-image. Chest X-ray: Stillwaterising, Wikimedia Commons, CC0. Six domain/pair figures were generated with the OpenAI image tool, 29 September 2026, and are marked synthetic. No video-creator photos are included. [Full provenance and prompts](https://nipunbatra.github.io/clip-lab/sources.html).

## ViT recap input crop

`vit-model-input.png` is copied unchanged from the Vision I reference's `figures/vision1/model-input.png` (checkout revision `1c57670`). It shows the exact evaluation crop of `newfoundland_31.jpg`, with channel normalization reversed for display. The same Oxford-IIIT Pet CC BY-SA 4.0 attribution applies. A CSS grid marks the 14×14 non-overlapping patch boundaries; it does not change the saved pixels. Crop checksum and original-photo checksum are recorded in `output/vit-recap-evidence.json`.


## Generated image interventions — 29 September 2026

Five images newly generated/edited with the built-in OpenAI image tool for this lecture:

- `bird-toucan-original.png`: synthetic toucan photograph-style illustration.
- `bird-toucan-small-beak.png`: edit of that image, replacing the large orange beak with a small dark beak.
- `bird-toucan-background.png`: background-only control edit of the original.
- `zebra-stripes-original.png`: synthetic zebra photograph-style illustration.
- `zebra-stripes-removed.png`: edit removing dark stripe markings and approximately retaining body shape, pose and scene.

Exact prompts and edit invariants: [bird](../output/bird-intervention/prompts.md), [stripe](../output/stripe-intervention/prompts.md). Image SHA-256 hashes are saved with each inference result. These are artificial teaching examples, not real wildlife observations or biological transformations. No existing creator images were reused.

The paired-data introduction (`src/contrastive_intro.py`) reuses the Newfoundland, Persian cat and coffee photos credited above. The captions, curation examples and network drawings are authored for teaching. They are not screenshots, quoted web captions or asserted WIT training samples. Vector previews and cosine values come from the existing recorded pretrained embeddings; the grid in `whole-map` illustrates the pairing targets.

## Main rebuild additions (3 October 2026)

- `clip-paper-thumbnail.png`: reduced thumbnail of page 1 of Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision*, [PMLR paper](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf). Used as a small bibliographic anchor; the teaching architecture is redrawn.
- `tutorial-cat.svg`, `tutorial-dog.svg`, `tutorial-car.svg`: original teaching illustrations reused from the user's **Build interactive CLIP loss tutorial**, `~/git/interactives/clip-loss/app.js`. [Public tutorial snapshot](https://nipunbatra.github.io/clip-lab/loss-tutorial/) and `../output/tutorial-example.json` record provenance and source hashes.
- `clip-canonical.html`: standalone rendering of the same canonical architecture component used in the rebuilt main lecture. Image/text branches and geometry are authored course diagrams of the CLIP method, credited to Radford et al. (2021) and the official implementation.

The current cover, schematic encoder/alignment diagrams and calculation sequence also reuse the full original inline illustrations captured in `../output/interactive-lecture.json` from **How CLIP Learns**, source commit `7b315b906ac13ef56588f05f2491a3564892daab`. These are the same toy cat, dog and car as in the companion interactive. Measured demos keep their original photographs.
