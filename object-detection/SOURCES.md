# Sources and teaching assets

## Course material

- The user-provided Object Detection Lecture Blueprint and subsequent slide annotations.
- The user-provided two-lecture feedback, retained in `output/two-lecture-feedback.txt`.
- The existing object detection lecture in `lecture9/`, which drew on [Ayush's shared object detection slides](https://docs.google.com/presentation/d/1onxJtUt-ZeStGcVppns8bGp7GzjV75EVcYjYICU-cp4/edit).
- The HTML reading/presentation shell in `attention-followups/transformer-outputs/`, associated with the chat *Transformers beyond next-token prediction*.
- The existing [PyTorch companion notebook](object_detection_companion.ipynb). The new [offline parcel lab](lab.html) uses precomputed candidates; the [older live-model lab](https://nipunbatra.github.io/interactive-articles/object-detection/) remains a separate network-dependent resource.
- The major rebuild brief and later request to adopt Beyond Attention's visual style, 3 October 2026. Theme source: ../transformer-outputs/src/lecture.css and story_diagrams.py.
- KaTeX 0.16.45, bundled locally with its fonts; [MIT licence](assets/KATEX-LICENSE.txt).

## Technical references

- Redmon et al. [You Only Look Once](https://arxiv.org/abs/1506.02640). The reference slides distinguish YOLOv1's shared per-cell class probabilities and IoU-weighted confidence from the simplified objectness teaching head.
- Carion et al. [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), and the [official DETR repository](https://github.com/facebookresearch/detr). Learned object queries, one-to-one bipartite matching and original DETR inference without NMS.
- Lin et al. [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144).
- Ren et al. [Faster R-CNN](https://arxiv.org/abs/1506.01497).
- Rezatofighi et al. [Generalized Intersection over Union](https://arxiv.org/abs/1902.09630).
- [COCO detection evaluation](https://cocodataset.org/#detection-eval) and [official evaluator implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py). The main matching example omits crowd/ignore cases. The toy area of 5/6 is not presented as an exact COCO AP result.
- Torchvision [Faster R-CNN](https://docs.pytorch.org/vision/stable/models/faster_rcnn.html) and [batched_nms](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.batched_nms.html). Class IDs determine suppression groups; images must be handled separately.

## Image provenance

The JPEG assets are resized copies of course-owned generated photographs from `lecture9/figures/`. Original prompts and image hashes are in [the existing provenance file](IMAGE_PROVENANCE.md).

| HTML source asset | Original generated image |
| --- | --- |
| cat.jpg | imagen-cat-single.png |
| pets.jpg | imagen-cat-dog.png |
| dogs.jpg | imagen-two-dogs.png |
| crowd.jpg | imagen-campus-crowd.png |
| street.jpg | imagen-campus-scale.png |
| parcel.jpg | imagen-warehouse-parcel.png |
| detection.jpg | imagen-promise-detection.png |
| mask.jpg | imagen-promise-segmentation.png |

The generated photographs do not contain predicted annotations. Boxes, scores, grids, arrows, plots and the approximate silhouette mask are authored SVG. All scores and candidate coordinates are constructed examples. No object detector was run to produce them.
