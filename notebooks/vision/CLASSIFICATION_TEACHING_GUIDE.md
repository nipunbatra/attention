# Vision I: image classification teaching route

The lecture explains one image classifier. Keep the same photograph as the anchor, use the small worksheet for arithmetic, then return to the real architecture. All new backward-pass numbers are chosen-parameter calculations, not a fitted model or benchmark.

## Main route

1. Sections 1–3: pet dataset, text parallels, image shapes, patches and shared projection. Section 2 ends with 196 content rows. Section 3 first locates P63 on the unchanged photograph and adds position. The purple CLS detour explains why the classifier needs one image summary, how attention fills that row, why the shared starting CLS produces image-dependent summaries, and how mean pooling can replace it. Resume the forward pass with all 197 rows present, then make Q/K/V and complete the measured classifier. Name LayerNorm here; save its details for section 7.
2. Sections 4–5: calculate one query, its scores, softmax and value message. Work the second head, concatenate and project. Return to possible visual head roles; explain that these are hypotheses rather than assigned jobs.
3. Section 6: readout, class probabilities, loss and learning. The new reverse sequence follows the exact earlier worksheet through the class head, residual, both heads, values, attention softmax, Q/K and patch projection. Show the single query-weight update as an arithmetic example. The two softmaxes have different axes and purposes.
4. Section 7: restore LayerNorm and the MLP. Follow both residual gradient paths, then return to the whole forward/loss/backward diagram. Compare information flow and readout with a conventional CNN.
5. Section 9: explain the proposed dog/cat adaptation. A Linear(192,2) head has 386 parameters. Distinguish frozen-encoder head training from fine-tuning. Show one batch and the train/validation/test procedure. No new training has been run.
6. Sections 10–11: use the saved real-photo predictions and measured attention/occlusion examples. Keep ImageNet outputs separate from the proposed two-class model. A confidence on one photo is not test accuracy.
7. Section 13: ask students to narrate the shapes and reverse path. The optional patch-rearrangement check belongs here: explicitly call it a thought experiment about location, not a preprocessing step. Its coarse 4×4 grid illustrates the idea; the model uses 14×14 patches. The animal label need not change. Section 14 closes classification and previews CLIP.

## Choose the depth for the audience

For a first pass through backward propagation, use `backward-route`, `backward-scores`, `backward-cls`, `backward-heads`, `backward-one-weight` and `backward-patches`. The detailed softmax/QK derivatives can be a calculation workshop after the main mechanism is understood. Sections 8 (code), 12 (cost) and the existing notebooks are optional extensions; the lecture does not require a live notebook.

## Keep the examples distinct

- Pet photographs motivate the real task.
- The exact checkpoint trace is a pretrained 1,000-class ImageNet model: 224×224 RGB, 16×16 patches, 196 patch rows, D=192, 3 heads, 12 blocks.
- The four-patch grayscale worksheet has engineered 4-wide embeddings, two heads and two arrangement labels. Its simplified block omits LayerNorm and the MLP.
- The new pet adaptation is a procedure students could run, with no claimed measured accuracy.
- Earlier noisy-stripe training results remain in the optional worked lab and saved `training.json`; they are not pet-classification evidence.

## Useful questions

- Is CLS an image patch, a label, an activation, or a learned starting parameter?
- Why does it become image-dependent? How does class loss train it?
- Without CLS, how could we obtain one image vector? Why is deleting it from a pretrained CLS model a different experiment?
- Why are three heads compatible with two classes? What exactly differs between heads?
- Can CNNs see the whole image? How do their neighborhoods grow?
- What receives a gradient if only CLS is read? Why do source patch values still receive gradients?
- Which operation computes gradients and which changes parameters?
- How do we check generalization without selecting on the test set?

## Rebuild without executing notebooks

```sh
python3 src/build_vision1_lesson.py --slides-only
PYTHONPATH=src python3 src/vision1_gradients.py
```

The slide-only build reuses saved experiment artifacts. The second command checks analytic gradients against central differences for all 116 worksheet parameter coordinates; it runs no model-training loop.
