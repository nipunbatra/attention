# Language-Grounded Vision

How Words Become Boxes, Regions, and Masks

55-minute main route: 48 physical slides (38 teaching screens + 10 dividers), with click-by-click builds. Eight backups are excluded from the default presentation route. PDF contains the completed state of all 56 physical slides.

## Timing

| Section | Minutes |
|---|---:|
| 00 Language changes target | 5 |
| 01 CLIP, but spatial | 7 |
| 02 Text-defined detection | 4 |
| 03 Category to phrase | 4 |
| 04 Language-guided queries | 8 |
| 05 Text to box to mask | 5 |
| 06 Generate space | 7 |
| 07 Reason into masks | 6 |
| 08 Architecture synthesis + explorer | 4 |
| 09 Training, evaluation + exit | 5 |

For 50 minutes, assign the attention worksheet, shorten the recorded-results discussion, and use only two exit responses. For 60 minutes, work the coordinate-handoff backup.

## Presentation

Right/Space/N advances one build. Left/P reverses one build. F completes or restarts the current slide. R toggles reading/presentation. Q/A/S show question/answer/notes. O opens map. Home/End follow the current route. Include backup adds the appendix. D toggles the pen, Z undoes ink; ink is saved locally per slide. Reading and print show complete diagrams.

## Evidence boundaries

The park is a fixed authored scene; the child visibly kicks the red ball. Drawn examples are reference targets, not predictions. Toy vectors, attention values and coordinate tokens are explicitly labelled. Recorded Grounding DINO/SAM outputs are separate, with model revisions, prompts, thresholds, timing and image hashes in results/grounded_vision_outputs.json. The explorer does not perform live inference. A one-dog success cannot demonstrate relation understanding.

## Slide notes

### 01. LANGUAGE-GROUNDED VISION

`cover` · divider · 1 reveal states

The fixed park scene is an authored illustration. All main teaching overlays are reference annotations; recorded model evidence is identified separately. The child now visibly kicks the red ball; this same drawing is used throughout. Ten dividers include this title slide.

**Ask:** 

**Reference answer:** 

### 02. A detector already gives us locations

`ordinary` · main · 5 reveal states

Recall the recent detector lecture. These are reference boxes, not measured predictions. Do not assign an invented score to the scene.

**Ask:** What does the image-level label “dog” leave out?

**Reference answer:** Location and the number of instances.

### 03. “Person” does not identify one person

`people` · main · 6 reveal states

Preserved three-person illustration. Point to all three before adding red. Ask for the selected instance before revealing B.

**Ask:** What information did “wearing red” add?

**Reference answer:** An attribute that distinguishes instances of the same class.

### 04. Change the phrase; change the target

`relation` · main · 8 reveal states

Two boxes remain at the completed state to compare target noun roles. During teaching first point at the ball, then dog. Nearness is obvious in this authored scene. This example alone cannot establish model relational understanding.

**Ask:** Which object should be returned for “dog beside the red ball”?

**Reference answer:** The dog. The ball is the reference object, not the requested target.

### 05. The words can ask for unfamiliar evidence

`domain` · main · 6 reveal states

Preserve kiln opening but compress it. The oval kiln and central chimney illustrate different regions. Recognizing a domain term still requires suitable image evidence and training.

**Ask:** Would a text encoder make an unresolved chimney visible?

**Reference answer:** No. The image resolution and domain evidence still limit localization.

### 06. CLIP, BUT SPATIAL

`section-01` · divider · 1 reveal states

Use this transition to restate the problem before introducing a mechanism.

**Ask:** 

**Reference answer:** 

### 07. IF is a different question from WHERE

`clip-where` · main · 5 reveal states

Connect to the CLIP matrix intuition. Global matching can support image retrieval. Localizing requires retaining spatial information and learning a geometric output.

**Ask:** What has pooling removed?

**Reference answer:** Explicit per-location output structure.

### 08. Replace the fixed label representation

`fixed-to-text` · main · 7 reveal states

The first row recalls a conventional linear class head. Then replace learned class weight vectors by encoded descriptions. Actual open-vocabulary models need local feature alignment and detection training.

**Ask:** Does changing the label list alone teach box geometry?

**Reference answer:** No. Geometry requires spatial features and a trained box predictor.

### 09. One dot product becomes one matrix cell

`one-cell` · main · 7 reveal states

Go slowly: multiply the first components, then second, then add. Step 5 moves the scalar into the amber outlined cell. The vectors both have unit norm.

**Ask:** What does 0.6 refer to?

**Reference answer:** Region 1 compared with the supplied dog vector; not a probability.

### 10. Build the matrix from comparisons

`matrix` · main · 8 reveal states

The full 3×2 matrix appears only after several cells. The r vectors are toy candidates, not semantic labels for the park. No claim is made that row 1 is a dog. Ask row and column interpretations separately.

**Ask:** For dog, which toy region scores highest?

**Reference answer:** Column dog: r₂ at 0.8. Row r₁ instead prefers ball at 0.8.

### 11. Similarity, logit and probability are distinct

`scores` · main · 7 reveal states

No new contrastive-loss derivation. Normalization choices matter. A probability over a supplied list changes if the list changes; it does not prove the correct object exists.

**Ask:** What happens when we add another text candidate?

**Reference answer:** The cosine values can remain fixed while the softmax probabilities change.

### 12. HOW DOES TEXT BECOME A DETECTION?

`section-02` · divider · 1 reveal states

Use this transition to restate the problem before introducing a mechanism.

**Ask:** 

**Reference answer:** 

### 13. Derive a text-defined detector

`open-architecture` · main · 8 reveal states

Six reveals: visual backbone; spatial features; box head; text embeddings; local text matching; boxes combined with scores. Do not name the exemplar before the mechanism is complete. Scores and geometry are different predictions.

**Ask:** Where does language enter this design?

**Reference answer:** Through text embeddings compared with local visual features.

### 14. Example: OWL-ViT-style detection

`owl-example` · main · 2 reveal states

Name OWL-ViT now. Simplification: image-token outputs, projections and detection heads stand in for implementation details. OWL-ViT transfers image-text pretraining and fine-tunes localization. Open vocabulary is a capability with limits, not an unlimited guarantee.

**Ask:** Which branch would fail if box training were missing?

**Reference answer:** The geometry branch would not have learned to predict useful enclosing boxes.

### 15. FROM CATEGORY TO REFERRING EXPRESSION

`section-03` · divider · 1 reveal states

Use this transition to restate the problem before introducing a mechanism.

**Ask:** 

**Reference answer:** 

### 16. A description can add constraints

`phrase-ladder` · main · 7 reveal states

All four phrases refer to the same dog here. That does not test whether a model uses relations. The next slide adds same-class alternatives. In a single-referent setting this is referring expression comprehension.

**Ask:** Does success on this one-dog image prove relation understanding?

**Reference answer:** No. The model could return the only dog for every prompt.

### 17. Make the alternatives plausible

`instances` · main · 6 reveal states

Retain the existing three-dog asset. Use it to distinguish detecting a category from resolving a referring expression. These are reference annotations, not a benchmark.

**Ask:** Why is this a stronger target-selection test?

**Reference answer:** Several regions satisfy the noun, so additional words must disambiguate.

### 18. “A child stands near a dog and a red ball.”

`phrase-links` · main · 5 reveal states

The child is standing on one leg while kicking. The sentence is an example of region-language correspondence, not a generated caption. Span colors identify corresponding objects on this slide; modal colors resume in architecture diagrams.

**Ask:** Is one whole-image similarity enough to supply these three links?

**Reference answer:** No. We need phrase-to-region alignment.

### 19. LANGUAGE-GUIDED OBJECT QUERIES

`section-04` · divider · 1 reveal states

Use this transition to restate the problem before introducing a mechanism.

**Ask:** 

**Reference answer:** 

### 20. Recall DETR: queries read the image

`detr` · main · 5 reveal states

Recall learned object queries, visual K,V and per-query predictions. The diagram omits query self-attention, positional encodings and repeated layers.

**Ask:** Do the learned slots begin as “dog query” and “ball query”?

**Reference answer:** No. Explicit semantic slot labels would be misleading.

### 21. The same operation, a spatial question

`beyond-attention` · main · 7 reveal states

Make the connection explicit rather than treating multimodal attention as a new primitive. Different sublayers may read visual and textual memories; they need not concatenate everything into one matrix.

**Ask:** What changes while the operation stays the same?

**Reference answer:** The query source, memory contents, training targets and output head.

### 22. One query mixes three values

`attention-worked` · main · 8 reveal states

Calculate raw dot products first, then divide by sqrt(2), then softmax, then weighted values. All vectors are invented for arithmetic, not measured activations. Explain that value coordinates have no assigned dog/red-ball meaning in a real model.

**Ask:** Is the attention distribution already a box?

**Reference answer:** No. It is a weighting used to update a representation.

### 23. Let text influence feature selection

`grounding-build` · main · 9 reveal states

Seven reveals: image branch, text branch, feature enhancer, language-guided query selection, decoder, both memory side paths, then box and phrase-score outputs. Query selection retains language-relevant visual positions. No chat interface is implied.

**Ask:** Which part changes the initial queries supplied to the decoder?

**Reference answer:** Language-guided query selection.

### 24. This is the Grounding DINO design

`grounding-name` · main · 2 reveal states

Only now attach the name Grounding DINO. Distinguish it from DINO self-supervised visual learning and from a generative chat VLM. The architecture supports language conditioning; robust relational understanding must still be evaluated.

**Ask:** Why can a longer phrase still fail?

**Reference answer:** The model may latch onto nouns, confuse relations or select the wrong instance.

### 25. TEXT → BOX → MASK

`section-05` · divider · 1 reveal states

Use this transition to restate the problem before introducing a mechanism.

**Ask:** 

**Reference answer:** 

### 26. Compose a grounder with a segmenter

`grounded-sam-build` · main · 8 reveal states

Six reveals correspond to the requested six frames: query and image; grounder and box; box prompt; cached SAM image embedding; mask decoder; reference mask. Use the same image in both models. Original released SAM accepts spatial prompts; text is handled upstream.

**Ask:** What passes from the grounder to SAM?

**Reference answer:** A spatial box after coordinate conversion.

### 27. The user asks for a mask

`text-to-mask` · main · 5 reveal states

Show the final text-to-mask view without intermediate box overlays. Grounding and shape quality are separable. The displayed mask is an authored reference. Coordinate bookkeeping is preserved in backup.

**Ask:** Can the segmenter fix a perfectly confident wrong-object box?

**Reference answer:** It can produce a precise mask of the wrong object; target selection remains a separate failure.

### 28. CAN A VLM GENERATE SPACE?

`section-06` · divider · 1 reveal states

Use this transition to restate the problem before introducing a mechanism.

**Ask:** 

**Reference answer:** 

### 29. A familiar VLM, a different answer grammar

`generative-build` · main · 7 reveal states

Build visual encoder and tokens; connector and instruction; decoder; first coordinate token; remaining spatial sequence. This is a generic VLM pattern, not a claim about a specific model’s exact token syntax. The displayed numeric box is a separate toy box.

**Ask:** Which part makes the output spatial?

**Reference answer:** Training the sequence model to emit an agreed spatial grammar and coordinates.

### 30. Continuous coordinate → bin → token

`quantization` · main · 7 reveal states

Reveal x1, y1, x2, y2 separately. This box demonstrates quantization and is not claimed to enclose the dog or ball. Centre decoding makes the maximum normalized error 0.0005 per coordinate, including clipped endpoints. Real models may use different tokenization.

**Ask:** Do four discrete indices retain infinite spatial precision?

**Reference answer:** No. Quantization imposes finite coordinate resolution.

### 31. Predict one spatial token at a time

`autoregressive` · main · 12 reveal states

Do not reveal a complete token sequence first. Each advance adds one token after the initial context pipeline. The toy coordinate grammar is not the literal Kosmos-2 vocabulary.

**Ask:** Could all tokens be syntactically valid but the box be wrong?

**Reference answer:** Yes. Syntax and spatial correctness are independent checks.

### 32. Example: Kosmos-2 grounds generated phrases

`kosmos` · main · 2 reveal states

Attach name after the generic mechanism. Kosmos-2 links phrase text with location tokens; two tokens specify top-left and bottom-right grid cells. Contrast with the preceding four axis-token toy scheme.

**Ask:** Why should we not copy the toy x/y token strings into its tokenizer?

**Reference answer:** Its location vocabulary and serialization convention differ.

### 33. Example: Florence-2 changes tasks through prompts

`florence` · main · 9 reveal states

The displayed task strings are actual processor keys, not the shorthand names from the brief. The referring-expression segmentation route produces polygons; rasterization gives a bitmap. This is an encoder–decoder sequence model.

**Ask:** Where is the mask bitmap in the segmentation sequence?

**Reference answer:** The sequence represents polygons; downstream parsing and rasterization produce a mask.

### 34. CAN A VLM REASON INTO A MASK?

`section-07` · divider · 1 reveal states

Use this transition to restate the problem before introducing a mechanism.

**Ask:** 

**Reference answer:** 

### 35. From naming to interpreting the target

`reasoning-queries` · main · 6 reveal states

The park scene stays fixed. The child’s foot visibly contacts the ball. Even with a clear authored cue, action interpretation is an inference, and real images may be ambiguous. These reference answers do not claim measured model reasoning.

**Ask:** Which noun in the final request directly says “ball”?

**Reference answer:** None. The model must resolve the described interaction.

### 36. The request can depend on world knowledge

`rain-query` · main · 5 reveal states

This supplementary scene makes the world-knowledge query answerable without inventing an absent park object. Do not display hidden chain-of-thought; just the instruction and target.

**Ask:** Would an excellent mask of the ball answer this instruction?

**Reference answer:** No. Correct shape is insufficient when target identity is wrong.

### 37. Turn a language representation into a mask

`seg-build` · main · 8 reveal states

First image and query enter a VLM. Then introduce SEG; expose its hidden representation; project it into a segmentation prompt; show spatial image features entering the mask decoder; finally emit a mask. The displayed hidden state is a symbol, not decoded thought.

**Ask:** Why must the mask decoder also receive image features?

**Reference answer:** The prompt specifies the target; spatial image features supply location and boundary detail.

### 38. Example: LISA-style reasoning segmentation

`lisa-name` · main · 2 reveal states

Name LISA after the mechanism. The original implementation projects the selected token hidden representation into a SAM-style prompt and combines it with image embeddings. No full paper architecture or private reasoning transcript is needed.

**Ask:** How does this differ from text → box → SAM?

**Reference answer:** It supplies a learned segmentation prompt from the VLM rather than requiring an intermediate box.

### 39. FOUR ARCHITECTURES FOR LANGUAGE-GROUNDED VISION

`section-08` · divider · 1 reveal states

Use this transition to restate the problem before introducing a mechanism.

**Ask:** 

**Reference answer:** 

### 40. Four ways to make language spatial

`four-families` · main · 6 reveal states

Reveal one complete family per click. The fourth row includes two distinct output paths: generated coordinates versus a segmentation representation plus pixel decoder. Categories describe patterns, not mutually exclusive taxonomic bins.

**Ask:** Which design naturally exposes an intermediate box?

**Reference answer:** The modular ground-and-segment system. A learned SEG interface need not expose one.

### 41. A recorded answer can miss the intended target

`explorer` · main · 6 reveal states

Measured content is loaded from results/grounded_vision_outputs.json. These are actual Grounding DINO tiny outputs on the authored scene at box/text thresholds 0.25. The detector returns noun detections, not a guaranteed single referred instance. Explore all five interfaces, color swap and absent bicycle; use reference mode for unmeasured generative/reasoning examples.

**Ask:** Which returned box answers the request?

**Reference answer:** The ball box; the dog is the reference object. Returning both does not resolve the referring expression.

### 42. TRAINING / EVALUATION SYNTHESIS

`section-09` · divider · 1 reveal states

Use this transition to restate the problem before introducing a mechanism.

**Ask:** 

**Reference answer:** 

### 43. What annotation teaches each behavior?

`supervision` · main · 8 reveal states

Three training slides total. The ladder represents different annotation types, not an automatic performance ranking. Mask supervision can be paired with explicit or implicit descriptions.

**Ask:** Can image-level class labels alone specify an exact mask?

**Reference answer:** No. They do not directly supply spatial boundary targets.

### 44. Loss follows the output representation

`loss-output` · main · 6 reveal states

Box objectives commonly combine alignment/classification and geometry such as L1/GIoU after assignment. Mask objectives compare pixels or overlap. Original SAM uses focal and Dice plus quality prediction; LISA uses LM with BCE/Dice. No full derivation in main route.

**Ask:** Does a coordinate token loss directly optimize IoU?

**Reference answer:** No. It penalizes next-token errors; spatial quality must still be evaluated.

### 45. A teacher can scale labels—and its mistakes

`teacher-data` · main · 7 reveal states

Third and final training slide. Link to GLIP-style data expansion without a long training recipe. Teacher confidence cannot replace instance-level validity checks.

**Ask:** What error can survive perfect mask boundaries?

**Reference answer:** A wrong region referred to by the instruction.

### 46. Evaluate the output and the target identity

`evaluate` · main · 6 reveal states

First of two evaluation slides. AP aggregates ranked detections under a defined matching protocol. Referring expression accuracy often uses a box-IoU threshold but also requires the intended instance. Mask IoU/Dice depend on aggregation choice. Arithmetic is in backup.

**Ask:** Why is valid token syntax insufficient?

**Reference answer:** The generated box can refer to the wrong object or be shifted.

### 47. Separate four failure modes

`failures` · main · 6 reveal states

Second and final evaluation slide. The small drawings intentionally illustrate errors, not real predictions. Shifted coordinates show both reference and displaced box; shape failure erases part of the dog mask.

**Ask:** Which component should we inspect for a consistent shift?

**Reference answer:** The coordinate transforms and box format before blaming language understanding.

### 48. Choose a route; explain the representation

`exit` · main · 5 reveal states

Close by asking students to propose either modular grounding+SAM or VLM+mask, then identify the loss and a failure case. Do not imply one design always wins. Main lecture ends here.

**Ask:** How would you tell whether a correct mask came from understanding the instruction?

**Reference answer:** Use counterfactual queries and same-class alternatives, inspect target identity and score spatial quality separately.

### 49. Coordinate handoff: the same box in two systems

`backup-coordinates` · backup · 6 reveal states

Example: original 800×500; detector resize s=.8 then pad top=120 to 640×640. Detector xyxy [96,296,272,440] → original [120,220,340,400]. SAM longest-side scale1.28 → [153.6,281.6,435.2,512]. Original SAM pads bottom/right, so no added top/left offset. Use library postprocessors.

**Ask:** 

**Reference answer:** 

### 50. Overlap arithmetic belongs in backup

`backup-overlap` · backup · 2 reveal states

Two 200×200 boxes offset by50 pixels: intersection30000, union50000. This binary evaluation identity is not a complete specification of a differentiable Dice training loss. Define empty-mask behavior and macro/micro aggregation.

**Ask:** 

**Reference answer:** 

### 51. Pseudo-labels need an error audit

`backup-teacher` · backup · 5 reveal states

Use a stratified audit: category, relation, small objects, absent query, new domain. Prevent train/evaluation leakage. Retain low-confidence cases for diagnosis instead of hiding every failure.

**Ask:** 

**Reference answer:** 

### 52. Optional: language can point into masks

`backup-glamm` · backup · 4 reveal states

One optional contextual slide. The sentence and masks are illustrative, not a measured output. No additional architecture is required.

**Ask:** 

**Reference answer:** 

### 53. Specify the evaluation protocol

`backup-protocol` · backup · 5 reveal states

Retained evaluation detail outside the main flow. Avoid comparing headline AP, mask mIoU and referring accuracy as if they measured the same task.

**Ask:** 

**Reference answer:** 

### 54. Optional: carry a spatial prompt through video

`backup-sam2` · backup · 5 reveal states

Preserves the prior optional extension. This is not needed for the 55-minute route.

**Ask:** 

**Reference answer:** 

### 55. Primary sources and implementation references

`references-1` · backup · 2 reveal states

Primary sources checked 4 October 2026. The lecture teaches representative published designs, not a leaderboard.

**Ask:** 

**Reference answer:** 

### 56. Primary sources and implementation references

`references-2` · backup · 2 reveal states

Primary sources checked 4 October 2026. The lecture teaches representative published designs, not a leaderboard.

**Ask:** 

**Reference answer:** 

