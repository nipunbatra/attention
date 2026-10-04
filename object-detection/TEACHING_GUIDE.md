# Object detection: from one label to a set of objects

70 main concepts; 113 physical builds. Twenty optional reference slides.

## 1. Object detection: from one label to a set of objects

This lecture follows ViT, CLIP and VLMs. Reuse image features and cross-attention; the new problem is the structure of the output.

Question: What does an image-level label omit?

Answer: Individual instances, their locations and their count.

## 2. Why is classification not enough?

What would one image-level label leave out?

Question: What would one image-level label leave out?

Answer: Use the following examples to work through the question.

## 3. One label?

Assume a mutually exclusive cat/dog/other classification task.

Question: Which label would you select?

Answer: Cat.

## 4. One label now?

The input type has not changed, but the requested description now includes two classes.

Question: Would a new “both” class solve detection?

Answer: It would still omit instance counts and locations.

## 5. The winning class leaves the cat out

These illustrative scores sum to one. Selecting the largest score returns one image-level label.

Question: Is this just a training problem?

Answer: The selected-label output cannot describe two instances.

## 6. Which classes are present?

Independent class scores can describe co-occurrence. They do not allocate one record per instance.

Question: What happens to dog-present if another dog enters?

Answer: It stays positive. The class-presence vector does not count dogs.

## 7. Class presence and object instances

The distinction is about representation, not accuracy.

Question: What extra output do we need?

Answer: A separate class-and-location record for each object.

## 8. What should one object record contain?

First solve the simpler case: one designated dog.

Question: First solve the simpler case: one designated dog.

Answer: Use the following examples to work through the question.

## 9. One dog, one record

Class and geometry define the target record. Add an inferred score only after establishing that record. A score is not automatically a calibrated probability that the complete detection is correct.

Question: Why is the score absent from the target?

Answer: The annotation provides the class and box. The model supplies a score.

## 10. Centre and full box size

Use this one format in the main derivation. Width and height span the complete box. Other conventions and letterboxing remain in the reference route.

Question: Is w half the box width?

Answer: No. It is the full width.

## 11. One backbone, two heads

The feature extractor is shared. One box head with four outputs describes only one designated instance.

Question: What is the limitation?

Answer: There is no second object record.

## 12. Fixed tensor, variable object set

How many records should the model reserve?

Question: How many records should the model reserve?

Answer: Use the following examples to work through the question.

## 13. Zero, two, or eleven objects

Annotation policy defines the target instances. Empty images have an empty target set.

Question: Can the target have length zero?

Answer: Yes. Images without target objects still teach background.

## 14. The annotation order does not change the target

Rows can be permuted without changing the set. The class must stay paired with its own box.

Question: What must move together when rows are swapped?

Answer: The class and coordinates belonging to one instance.

## 15. A fixed tensor, a variable unordered set

For a fixed input shape, the head emits a fixed-capacity tensor. We need a representation and selection rule for a variable set.

Question: Must the network change its shape for every scene?

Answer: No. Reserve capacity, then return a subset.

## 16. First attempt: K global slots

K is a capacity, chosen before observing the particular scene. Empty slots need a no-object target.

Question: What do unused slots represent?

Answer: No object.

## 17. What if there are more objects than slots?

Every finite candidate design has a capacity limit.

Question: Would learned queries remove this limit?

Answer: No. K query outputs still have capacity K.

## 18. Which dog is slot 1?

The target set does not identify a first dog. This is an assignment problem; spatial addresses are one possible response.

Question: What changed between the two builds?

Answer: Only which slot represents which dog.

## 19. Can the image give candidates an address?

Retain the feature map instead of collapsing it immediately to one global vector.

Question: Which information survives in a feature map?

Answer: Its row and column structure.

## 20. Give candidates spatial addresses

Use a feature-map location to index each record.

Question: Use a feature-map location to index each record.

Answer: Use the following examples to work through the question.

## 21. A location can index a prediction

The grid indexes output records. Shared head weights operate across locations.

Question: Does each cell require a different network?

Answer: No. The head shares parameters.

## 22. From image features to candidate records

This diagram will recur as later operations are introduced. Faded stages are still unresolved.

Question: What does the prediction head add?

Answer: A record of existence, geometry and class at each spatial address.

## 23. One generic prediction record

This is a generic teaching abstraction. It is not the exact parameterization of every detector.

Question: Where does the 5 in C+5 come from?

Answer: One objectness value and four coordinates.

## 24. The dimensions of the dense output

Here S=4 and C=3: cat, dog and other. The toy head emits 16 records with 8 values each. The class term uses a softmax over the foreground classes for positive candidates.

Question: How many records are reserved?

Answer: 16.

## 25. Grid cell = output address

A cell is not an independent image crop. The dog box deliberately crosses cell boundaries.

Question: Must the box fit inside the cell?

Answer: No. Its centre selects an address under this teaching rule.

## 26. One dog, one normalized teaching box

The normalized box is a constructed teaching target. The generated dog photograph identifies the object; it is not a measured dataset annotation. The same values, target record and prediction recur throughout the numerical trace.

Question: Which cell contains the centre?

Answer: Work out the column from x, then the row from y.

## 27. Which cell owns the centre?

The normalized box is a constructed teaching target. The generated dog photograph identifies the object; it is not a measured dataset annotation. The same values, target record and prediction recur throughout the numerical trace. Row and column are zero based.

Question: Why is the first index not 1?

Answer: Array indices begin at zero in this example.

## 28. How far inside that cell is the centre?

The normalized box is a constructed teaching target. The generated dog photograph identifies the object; it is not a measured dataset annotation. The same values, target record and prediction recur throughout the numerical trace. The offsets are fractions of one cell. Width and height are fractions of the full image.

Question: Are all four values measured relative to a cell?

Answer: No. Only the centre offsets are cell-local in this teaching convention.

## 29. Construct the target record

The normalized box is a constructed teaching target. The generated dog photograph identifies the object; it is not a measured dataset annotation. The same values, target record and prediction recur throughout the numerical trace.

Question: Why is objectness 1?

Answer: The teaching assignment rule gives this cell the dog target.

## 30. The model emits a record at the same address

These are fixed illustrative predictions, not measured detector output. Every loss and decoding calculation uses exactly these values from the shared JSON.

Question: Is the predicted record identical to the target?

Answer: No. Compare its existence, geometry and class probability with the target separately.

## 31. Which candidate should learn which object?

A loss needs a target for each prediction.

Question: A loss needs a target for each prediction.

Answer: Use the following examples to work through the question.

## 32. Which of these candidates should learn the dog?

The simplified centre-cell policy chooses exactly one positive. Modern detectors use model-specific rules, often with several positives per object.

Question: Is this the assignment rule of every detector?

Answer: No. It is the declared rule for this worked example.

## 33. Positive, background, and ignored candidates

This toy policy has positives and background, with no ignore band. Other policies can exclude ambiguous candidates. Do not silently treat ignore as background.

Question: Does the grey cell receive the dog box?

Answer: No. It is not assigned to this object.

## 34. Objectness: is an object assigned here?

Binary cross-entropy for this positive target reduces to minus log p. Natural logarithms are used.

Question: What happens if the objectness approaches 1?

Answer: The positive-objectness loss approaches zero.

## 35. Class: how much probability went to dog?

Categorical cross-entropy selects the probability of the annotated class. This term supervises the assigned positive, not the background candidate.

Question: Why does the cat probability not appear as a separate term?

Answer: For a one-hot categorical target, only the target-class log probability contributes directly.

## 36. Box: compare one coordinate at a time

We sum absolute differences, rather than averaging. Geometry combines two local centre offsets and two image-normalized sizes in this deliberately simple pedagogical loss. It is not attributed to a YOLO version.

Question: Would the mean give the same numerical loss?

Answer: No. It would be the displayed sum divided by four.

## 37. Add the three supervised terms

The total is computed from unrounded component values; displayed components are rounded to three decimals. This is the positive-candidate loss, not the complete batch loss.

Question: What must happen before computing these terms?

Answer: Assign a target to the candidate.

## 38. A nearby unassigned candidate

Background does not learn the dog box from the adjacent positive cell.

Question: Does its predicted geometry need to be zero?

Answer: No. This candidate has no target box term in the toy loss.

## 39. Many background terms can dominate a sum

This separate 8×8 example motivates imbalance; it does not change the 4×4 numerical trace. Easy negative terms can dominate an unweighted sum.

Question: Does low average loss guarantee good object recall?

Answer: No. Many correctly classified background locations can obscure positive errors.

## 40. A teaching record, not a universal detector format

The record and losses are a generic dense-detector teaching abstraction. Original YOLOv1 confidence, anchor-free heads, distributional regression and DETR are not identical parameterizations.

Question: Must every detector predict an explicit objectness scalar?

Answer: No.

## 41. From candidates to final detections

At inference there are no annotations to consult.

Question: At inference there are no annotations to consult.

Answer: Use the following examples to work through the question.

## 42. Decode the same predicted box

Use the predicted offsets, not the target offsets. The model has not changed its record; decoding only interprets its parameterization.

Question: Why does the decoded centre differ from the ground truth?

Answer: The predicted local offsets differ from the target local offsets.

## 43. A score for this predicted dog

This product is the declared score for the toy factorization. Other heads use other definitions. It does not add another learned field to the prediction.

Question: Does a score of 0.68 prove the box is correct?

Answer: No. It is a ranking score; correctness needs evaluation.

## 44. Decoding interprets the fixed candidate tensor

We have followed one dog candidate through target creation, supervision and decoding. The parcel scene now isolates selection among several candidates.

Question: Which steps remain to choose what to return?

Answer: Score filtering and duplicate removal.

## 45. Now several candidates describe one parcel

The parcel photograph and A-E values are constructed teaching data. Candidate IDs, coordinates, scores and colours stay fixed in every slide and the lab. Colours identify candidates, not correctness.

Question: How many object instances do five candidates imply?

Answer: None by themselves. Several candidates may describe one instance.

## 46. Filter by score

Filtering tests the score, not spatial overlap. E remains faintly visible so the identity and geometry do not disappear from the explanation.

Question: Why do A, B and C all remain?

Answer: Each score exceeds the cutoff.

## 47. How much do A and B overlap?

All areas are in normalized image units and come from the displayed parcel boxes. B lies inside A in this example. IoU near 1 means strong overlap; near 0 means little overlap.

Question: What is the union when B lies wholly inside A?

Answer: The area of A.

## 48. Non-maximum suppression

Scores, geometry and ID colours do not change. Green KEEP and red suppression text encode state separately. NMS is greedy prediction-to-prediction comparison, not truth matching.

Question: Why does D survive?

Answer: It passes the score cutoff and does not overlap A.

## 49. The returned set has variable length

The reported count is the number of retained predictions, not a guarantee of the true count.

Question: Which quantity was fixed?

Answer: Candidate capacity. Selection chose the returned count.

## 50. Class-aware NMS separates class groups

The example deliberately aligns the two boxes so only class grouping changes. Same-class crowded objects can still suppress each other.

Question: Does class-aware NMS solve crowded same-class instances?

Answer: No. Their boxes can still overlap above the threshold.

## 51. Fixed capacity, scene-dependent returned set

This is the dense-detector answer to the opening question.

Question: Can the returned set be empty?

Answer: Yes, when no candidate survives selection.

## 52. Three different matching problems

The operands and purpose differ, even when all three use boxes and overlap. Training assignment is not NMS or evaluation matching.

Question: Which one requires annotations while measuring a test set?

Answer: Evaluation. Ordinary inference does not receive them.

## 53. How do we measure a detector?

Compare the returned predictions with annotated instances.

Question: Compare the returned predictions with annotated instances.

Answer: Use the following examples to work through the question.

## 54. When does a prediction count as a true positive?

In this simplified rule, process predictions in descending score order and choose the greatest-IoU available same-class truth. Official evaluators also handle crowd, ignore, area and detection limits.

Question: Why is a second accurate box on one object a false positive?

Answer: A higher-ranked prediction has already claimed that target.

Source: https://cocodataset.org/#detection-eval

## 55. Precision and recall use different denominators

The example has two matched predictions, one false positive, and no missed truth.

Question: Can one wrong box cause both FP and FN?

Answer: Yes. It can be unmatched while the actual object remains missed.

## 56. A deployment cutoff and AP serve different purposes

AP is not precision at one arbitrary deployment threshold. Apply the benchmark protocol, including its detection limits. An aggressive prefilter can truncate the ranking and reduce measured recall.

Question: Should we discard every prediction below a chosen deployment score before comparing AP?

Answer: Not arbitrarily. Follow the evaluation protocol and retain the needed ranked predictions.

## 57. A ranked prefix traces the PR curve

Each prediction appears with its point; future labels are hidden. The two true positives refer to distinct targets. Axes remain fixed in all builds.

Question: Which step adds a prediction without finding another object?

Answer: The FP at rank 2.

## 58. AP summarizes precision across recall

The green envelope uses the best precision at or beyond each recall. This toy continuous area differs from protocol-specific sampled AP such as COCO 101-point averaging.

Question: Can identical final counts produce different AP?

Answer: Yes. The ranking can differ.

Source: https://cocodataset.org/#detection-eval

## 59. Average AP across classes

Constructed per-class AP values use the same protocol. Their mean is not pooled class-agnostic AP.

Question: What can the mean hide?

Answer: Poor performance on one class.

## 60. The IoU protocol changes what counts as correct

COCO additionally specifies detection caps, area ranges, recall sampling and crowd/ignore handling. Evaluation thresholds do not rerun NMS.

Question: Which protocol demands tighter localization?

Answer: The average including high IoU thresholds.

Source: https://cocodataset.org/#detection-eval

## 61. Another answer: set prediction with queries

Return to the global slots that had no fixed identity.

Question: Return to the global slots that had no fixed identity.

Answer: Use the following examples to work through the question.

## 62. What if the slots could read the whole image?

Queries are learned vectors, not text prompts or fixed class labels. Their count still limits output capacity.

Question: Is query 1 permanently a dog query?

Answer: No. Its prediction depends on the image.

## 63. Reuse cross-attention from Beyond Attention

Distinguish K, the number of queries, from K in the attention notation for keys. In original DETR, the decoder combines self-attention among queries with cross-attention to encoded image features.

Question: What is read, and what asks the question?

Answer: Image features are the keys and values; the decoder query state supplies the query.

Source: https://arxiv.org/abs/2005.12872

## 64. Which prediction should match which target?

Do not assume the shown pairing follows from row order. This is an illustrative lowest-cost assignment; a model computes costs from class scores and box geometry.

Question: Can two queries both match the one dog target?

Answer: No. The assignment is one to one.

## 65. Now name the mechanism: bipartite matching

Matched queries learn class and geometry. Unmatched queries learn no-object. The mechanism and purpose precede the name. Original DETR trains a set predictor with this matching objective.

Question: Why does this avoid choosing an annotation order?

Answer: The assignment is optimized for the set of predictions and targets, rather than fixed target row indices.

Source: https://arxiv.org/abs/2005.12872

## 66. Two solutions to the same output problem

This compares the teaching dense model with original DETR. Neither is universally superior, and both retain finite capacity and can make errors.

Question: What problem do both solve?

Answer: Representing a variable unordered object set using fixed candidate capacity.

Source: https://arxiv.org/abs/2005.12872

## 67. Summary: fixed capacity to a variable set

Connect the representation, training and inference decisions.

Question: Connect the representation, training and inference decisions.

Answer: Use the following examples to work through the question.

## 68. The complete dense-detector path

Annotations assign supervision while training the fixed-capacity head. At inference, decoding and selection return a scene-dependent subset.

Question: Where do the annotations enter?

Answer: Assignment and training; later, evaluation. They are not an inference input.

## 69. Try the same parcel candidates

The offline lab embeds the same canonical JSON and pure numerical functions as this lecture. No detector or network download is needed. The optional class-ambiguity scenario changes a label only, not candidate geometry or score.

Question: At IoU threshold 1, which candidates survive NMS?

Answer: All score-retained candidates, because suppression uses IoU strictly greater than the threshold.

## 70. The network can stay fixed-shape. The scene does not have to.

Both routes use fixed capacity to describe a variable scene. The next representation question is what to predict when a box is insufficient.

Question: What changes from image to image?

Answer: The content and number of returned object records, not necessarily the shape of the candidate tensor.

## R1. Three coordinate formats

The format must accompany the data. Centre coordinates are the means of corresponding corners. Width and height are corner differences. Do not confuse full sizes with half sizes.

Question: What would you check when implementing this?

Answer: The format must accompany the data. Centre coordinates are the means of corresponding corners. Width and height are corner differences. Do not confuse full sizes with half sizes.

## R2. Worked coordinate conversion

Continuous coordinates are used. Centre = top-left + half-size. Divide horizontal values by image width and vertical values by image height.

Question: What would you check when implementing this?

Answer: Continuous coordinates are used. Centre = top-left + half-size. Divide horizontal values by image width and vertical values by image height.

## R3. Letterboxing transforms the boxes too

Scale the coordinates, then add the padding offsets. The example uses continuous geometry and symmetric vertical padding. Crops and flips also transform targets.

Question: What would you check when implementing this?

Answer: Scale the coordinates, then add the padding offsets. The example uses continuous geometry and symmetric vertical padding. Crops and flips also transform targets.

## R4. A minimal PyTorch target

Check image dimensions, class IDs, dtypes and coordinate format. Transform the image and target together. Many torchvision detection models reserve class 0 for background.

Question: What would you check when implementing this?

Answer: Check image dimensions, class IDs, dtypes and coordinate format. Transform the image and target together. Many torchvision detection models reserve class 0 for background.

Source: https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html

## R5. Target encoding and decoding are inverses

This round trip uses the target offsets. The main inference example uses different predicted offsets. Width and height remain normalized to the image. Other architectures use different box parameterizations.

Question: What would you check when implementing this?

Answer: This round trip uses the target offsets. The main inference example uses different predicted offsets. Width and height remain normalized to the image. Other architectures use different box parameterizations.

## R6. Why YOLOv1 used 7 × 7 × 30

Original YOLOv1 shares class probabilities between the two box predictors in a cell. Its confidence combines object presence and localization quality: Pr(object) × IoU. This is not the independent objectness term in our toy record.

Question: What would you check when implementing this?

Answer: Original YOLOv1 shares class probabilities between the two box predictors in a cell. Its confidence combines object presence and localization quality: Pr(object) × IoU. This is not the independent objectness term in our toy record.

Source: https://arxiv.org/abs/1506.02640

## R7. Two predictors do not guarantee two objects

The predictors specialize in geometry through training. Two independent objects with centres in the same cell are still difficult to represent. These original predictors are not predefined anchor templates.

Question: What would you check when implementing this?

Answer: The predictors specialize in geometry through training. Two independent objects with centres in the same cell are still difficult to represent. These original predictors are not predefined anchor templates.

Source: https://arxiv.org/abs/1506.02640

## R8. Anchors provide reference geometries

Anchor-based heads can emit several records at one location. Assignment is model-specific and can use IoU and heuristics. Anchor-free detectors do not require these reference boxes.

Question: What would you check when implementing this?

Answer: Anchor-based heads can emit several records at one location. Assignment is model-specific and can use IoU and heuristics. Anchor-free detectors do not require these reference boxes.

## R9. A compact dense prediction head

Illustrative code, not a complete detector. Define all dimensions consistently. Separate branches and extra regression channels are common.

Question: What would you check when implementing this?

Answer: Illustrative code, not a complete detector. Define all dimensions consistently. Separate branches and extra regression channels are common.

## R10. Feature pyramids support different object scales

Feature Pyramid Networks combine coarse semantic features with higher-resolution maps through a top-down pathway and lateral connections. Image resizing alone cannot guarantee small-object performance.

Question: What would you check when implementing this?

Answer: Feature Pyramid Networks combine coarse semantic features with higher-resolution maps through a top-down pathway and lateral connections. Image resizing alone cannot guarantee small-object performance.

Source: https://arxiv.org/abs/1612.03144

## R11. The full binary cross-entropy expression

Natural logarithms. A high probability is good for a positive and poor for background. Use numerically stable losses accepting logits, rather than taking logs manually.

Question: What would you check when implementing this?

Answer: Natural logarithms. A high probability is good for a positive and poor for background. Use numerically stable losses accepting logits, rather than taking logs manually.

## R12. Box losses measure geometric error

The choice depends on parameterization and failure modes. Ordinary IoU offers limited guidance for disjoint boxes. GIoU adds enclosing-region geometry. There is no universal ranking of losses.

Question: What would you check when implementing this?

Answer: The choice depends on parameterization and failure modes. Ordinary IoU offers limited guidance for disjoint boxes. GIoU adds enclosing-region geometry. There is no universal ranking of losses.

Source: https://arxiv.org/abs/1902.09630

## R13. An additional IoU calculation

Clamp intersection width and height at zero for non-overlap. Validate corner ordering before computing areas.

Question: What would you check when implementing this?

Answer: Clamp intersection width and height at zero for non-overlap. Validate corner ordering before computing areas.

## R14. Class-aware NMS in Torchvision

batched_nms compares only matching group indices. Do not mix images under the same class IDs without image-specific groups. Decoding and score filtering are still separate operations.

Question: What would you check when implementing this?

Answer: batched_nms compares only matching group indices. Do not mix images under the same class IDs without image-specific groups. Decoding and score filtering are still separate operations.

Source: https://docs.pytorch.org/vision/stable/generated/torchvision.ops.batched_nms.html

## R15. Duplicate, wrong-class and low-IoU predictions

Apply the declared benchmark protocol. One erroneous prediction and one missed target can occur at the same time.

Question: What would you check when implementing this?

Answer: Apply the declared benchmark protocol. One erroneous prediction and one missed target can occur at the same time.

## R16. Toy AP from two recall intervals

The envelope is 1 up to recall 0.5 and 2/3 from 0.5 to 1. COCO samples 101 recall levels; do not label this exact continuous toy area as COCO AP.

Question: What would you check when implementing this?

Answer: The envelope is 1 up to recall 0.5 and 2/3 from 0.5 to 1. COCO samples 101 recall levels; do not label this exact continuous toy area as COCO AP.

## R17. Final counts can hide a different ranking

Both complete lists have the same precision and recall. Putting both correct predictions first improves AP. This controlled example assumes the true positives match distinct targets.

Question: What would you check when implementing this?

Answer: Both complete lists have the same precision and recall. Putting both correct predictions first improves AP. This controlled example assumes the true positives match distinct targets.

## R18. A high-level API may already run NMS

The model includes proposals, decoding, internal score selection and NMS. This keep line adds a user cutoff. Inspect the API before applying another NMS stage.

Question: What would you check when implementing this?

Answer: The model includes proposals, decoding, internal score selection and NMS. This keep line adds a user cutoff. Inspect the API before applying another NMS stage.

Source: https://docs.pytorch.org/vision/stable/models/faster_rcnn.html

## R19. One-stage and two-stage detector families

This is a structural distinction, not a universal speed or accuracy ranking. Faster R-CNN is a two-stage example. Architecture, resolution, hardware and implementation determine practical performance.

Question: What would you check when implementing this?

Answer: This is a structural distinction, not a universal speed or accuracy ranking. Faster R-CNN is a two-stage example. Architecture, resolution, hardware and implementation determine practical performance.

Source: https://arxiv.org/abs/1506.01497

## R20. Sources and further practice

Photographs are generated teaching illustrations reused from the existing lecture. Their overlays, predictions and scores are constructed examples, not measured detector outputs. Original provenance remains in lecture9/figures/PROVENANCE.md.

Question: What would you check when implementing this?

Answer: Photographs are generated teaching illustrations reused from the existing lecture. Their overlays, predictions and scores are constructed examples, not measured detector outputs. Original provenance remains in lecture9/figures/PROVENANCE.md.
