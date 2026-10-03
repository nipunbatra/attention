# Supply before and after as PIL RGB images before running this snippet.
import torch, clip
import torch.nn.functional as F

model, preprocess = clip.load("ViT-B/32", device="cpu")
model.eval()
images = torch.stack([preprocess(before), preprocess(after)])
words = ["hat", "boat", "cup", "cat"]
with torch.no_grad():
    u = F.normalize(model.encode_image(images).float(), dim=-1)
    v = F.normalize(model.encode_text(clip.tokenize(words)).float(), dim=-1)
    delta = u[1] - u[0]                 # after minus before
    if delta.norm() < 1e-6:
        raise ValueError("No stable change direction")
    d = F.normalize(delta, dim=-1)      # normalize again
    scores = v @ d                     # one cosine per word
print(sorted(zip(words, scores.tolist()), key=lambda x: -x[1]))
