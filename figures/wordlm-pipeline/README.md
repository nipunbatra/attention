# Training and inference maps

Four editable SVGs cover the fixed-window MLP and one-head attention model,
each in training and generation. `pipeline_maps.py` is the diagram source;
the same module is used by the word-level TinyStories notebook companion.

```sh
python3 figures/wordlm-pipeline/pipeline_maps.py figures/wordlm-pipeline
python3 src/assemble.py --part 2 --out attention.html
node src/check_wordlm_pipeline.mjs attention.html
```

`src/sections/sec19_pipeline.html` embeds the master maps, preserving node
positions while highlighting and cropping stages. Each focused view can
return to the complete map. The assembled lecture embeds the SVGs and needs
no network request to render them.

The diagrams follow **one target per window**, not all-position supervision:
the optimized attention forward pass uses one final query and all available
keys/values. Full causal matrices remain an explanatory view in the notebooks.
PAD keys are masked; the target never enters its own context. Generation
recomputes the window without a KV cache and reuses saved parameters.

The authored sentence is only a tracing example. `benchmark-summary.json`
records the independent, frozen three-seed TinyStories result used on the
comparison slide. The saved run used Apple MPS, not DGX hardware. It is not a
claim that attention always beats an MLP.

The notebook companion adds `05_training_and_inference_maps.ipynb` and
highlights these same maps throughout notebooks 1 and 3. No trained model,
benchmark settings, or raw corpus was modified for this diagram extension.
