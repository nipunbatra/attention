# Minimal CLIP applications lab

[Open in Google Colab](https://colab.research.google.com/github/nipunbatra/attention/blob/main/clip/notebooks/clip-applications.ipynb)

17 images and 17 supplied captions, reused from the lecture. Run the notebook top to bottom. It shows preprocessing and vector shapes, image–caption matching, text–image search, image neighbours, and normalized before/after differences. The actual OpenAI ViT-B/32 checkpoint is frozen throughout. No API key is needed.

Colab: select a T4 GPU if available, then Run all. CPU also works. The first run downloads about 350 MB of weights. The notebook installs the pinned OpenAI CLIP implementation and uses Colab's existing PyTorch stack. Saved outputs are from the original-model CPU run; they may differ slightly from the lecture's q8 browser measurements.

Locally, install `torch torchvision matplotlib Pillow jupyterlab` and `git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`, then open `clip-applications.ipynb` and Run all.

`clip-mini-gallery.zip` contains only the 17 images, their captions and credits. Its manifest records source-image and final-image hashes. Two-panel and four-panel source images were cropped into the same individual examples shown in the lecture; no image content was generated or retouched for the notebook. [Credits](CREDITS.md).
