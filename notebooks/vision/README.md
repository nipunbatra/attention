# Vision I classification lab

[Open the executed notebook](01_classification_from_patches.ipynb) after reading [Vision I](../../vision1.html).

The notebook reads three 8×8 grayscale PNGs in `figures/vision-scene/`. Scenes A and B were used to fit the course's one-head toy; scene C was held out. The first code exercise builds two hand-chosen heads from scratch with NumPy. It then applies the fitted one-head parameters to C and checks the same result with PyTorch. The final cell runs an ImageNet-pretrained ViT on the existing JPEG illustrations. That model's output labels cannot answer the right-half question.

To regenerate the PNGs and execute every code cell, from the repository root run:

```sh
python notebooks/vision/build_classification_lab.py
```

This needs NumPy, Pillow, PyTorch and `timm`. The pretrained `timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k` weights are downloaded on first use. The generation script does not train or change the course toy. The two-head numbers are chosen for teaching, while the one-head parameters and their 1,500-step two-image fit come from `src/toy5.json` and `src/make_vision_toy.py`.
