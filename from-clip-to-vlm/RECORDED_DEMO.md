# Recorded VLM outputs

Measured with the pinned SmolVLM-256M-Instruct WebGPU model. Answers below are unedited, including leading spaces, omissions, mistakes and unsupported claims. Full settings, revision, timestamps, timing, token IDs, stream events and image hashes are in [the JSON record](results/vlm_lab_outputs.json).

## dog

**Prompt:** What animal is this?

**Raw answer:**

```text
 Bear.
```

**Reference:** A dog (Newfoundland).

## desk

**Prompt:** Is the laptop plugged in?

**Raw answer:**

```text
 Yes.
```

**Reference:** Yes: a visible cable connects it to the socket. Whether power is flowing is not visible.

## shapes

**Prompt:** What is immediately left of the mug?

**Raw answer:**

```text
 A.
```

**Reference:** The red ball.

## count

**Prompt:** How many apples are outside the bowl?

**Raw answer:**

```text
 8
```

**Reference:** Three apples are outside the bowl.

## invoice

**Prompt:** What is the invoice number? What percentage of the subtotal is GST?

**Raw answer:**

```text
 The invoice number is ES667-042.
```

**Reference:** ES667-042. GST is 180 / 1000 × 100 = 18%.

## chart

**Prompt:** Which month has the highest value? How much higher is March than January?

**Raw answer:**

```text
 High.
```

**Reference:** March. The difference is 75 − 30 = 45 µg/m³.

## math

**Prompt:** What is x in the triangle?

**Raw answer:**

```text
 The length of the hypotenuse (3) is 4.0. Therefore, the length of the side opposite to the right angle is $\frac{4}{3}=\boxed{\frac{2}{2}}$.The answer is not valid. The answer is x.
```

**Reference:** x = 5, from the marked right angle and legs 3 and 4.

## change

**Prompt:** What changed from image A to image B?

**Raw answer:**

```text
 The first image shows a table with two chairs, one of which has a red cup on it.
```

**Reference:** The chair moved from left to right and the cup was removed. Two images are supplied as one composite.

## screen

**Prompt:** Where should I click to disable notifications?

**Raw answer:**

```text
 Account.
```

**Reference:** The blue toggle in the Notifications row.

## domain

**Prompt:** What is the hottest measured region? Give the row, column and temperature.

**Raw answer:**

```text
 Warmest region.
```

**Reference:** Row 3, column 4: 62 °C. Count from the top left.

## no-bicycle

**Prompt:** What color is the bicycle?

**Raw answer:**

```text
 The bicycle is red.
```

**Reference:** No bicycle is visible.

## cf-dog

**Prompt:** What animal is shown?

**Raw answer:**

```text
 There is a black dog in the foreground.
```

**Reference:** A dog.

## cf-cat

**Prompt:** What animal is shown?

**Raw answer:**

```text
 A cat is visible.
```

**Reference:** A cat.

## cf-blank

**Prompt:** What animal is shown?

**Raw answer:**

```text
 A squirrel.
```

**Reference:** No animal is visible; a blank white image is provided, not a missing image input.

The three counterfactuals use exactly “What animal is shown?” and the same greedy decoding settings. The blank condition is a white image, not missing input. No token probabilities are exposed.
