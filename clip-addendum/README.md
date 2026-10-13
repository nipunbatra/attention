# CLIP addendum: DigUp and speech from embeddings

Sixteen slides with progressive reveals: seven on search, three introducing the speech demo, five practical training slides, and a final comparison. Allow roughly 20–25 minutes including the two original author videos and the worked calculations. The main CLIP lecture is unchanged.

- [HTML presentation](index.html?present#digup-demo)
- [PDF, 16 pages](clip-addendum.pdf)
- [Practical build guide](PRACTICAL.md)
- [Sources and media credits](SOURCES.md)
- [Minimal text search](code/semantic_search.py)
- [Cached-feature reader trainer](code/train_reader.py)

Use arrows or Space to reveal each step. **Show all** completes the slide. **Read** displays complete slides and expandable teaching notes. PDF and reading views show full diagrams. Videos need internet; their posters and explanations work offline.

## Build and preview

```sh
python3 src/build.py
python3 -m http.server 8767 --bind 127.0.0.1
```

Run these commands inside this addendum directory. Open `http://127.0.0.1:8767/?present#digup-demo`. The local reveal helper is copied from the CLIP lecture. Building this deck does not modify the main lecture.

Export PDF using browser Print → Save as PDF, with background graphics enabled and CSS page size 1280 × 720. All 16 slides print with their full reveals and attributed video stills.

## What students build

1. Extract pieces; preserve page, timestamp or code-line locations.
2. Encode and normalize an N × 768 matrix; save it beside metadata.
3. Encode a query, compute cosine scores and retrieve the source.
4. Create a small labelled Speech Commands subset with speaker-separated splits.
5. Pass frozen 768D features through a 768 → 128 → 768 reader.
6. Compare with eight word vectors, compute cross-entropy and update the reader.
7. Understand why full transcription additionally needs temporal features and alignment.

The two author demos are source-grounded. The small reader architecture, dataset and objective are explicitly proposed teaching choices; they are not claimed to reproduce the unpublished speech app. The Python shape/gradient checks do not establish model accuracy. Real cached audio features are required to train the companion reader; audio feature export is a documented adapter step, not an included pipeline.
