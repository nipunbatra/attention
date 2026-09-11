# Indian names dataset screenshot

`github-dataset.png` is a real Chromium screenshot of the GitHub CSV preview,
captured on 2026-09-11. No page content was modified. The capture excludes the
global navigation bar and shows the repository, filename, and first data rows.

- Repository: https://github.com/balasahebgulave/Dataset-indian-names
- Screenshot URL: https://github.com/balasahebgulave/Dataset-indian-names/blob/master/Indian_Names.csv
- Dataset at the verified revision: https://github.com/balasahebgulave/Dataset-indian-names/blob/80401358aaa609cbe30ae57afbea37654879d0ab/Indian_Names.csv
- The bundled `src/names.csv` is byte-identical to this upstream CSV (85,538 bytes).
- SHA-256: `f7b2bd2a8ad89a0c865296f81aff9df7a8ab360f6c6b0a0bde996b5172b9c47a`

The source has 6,486 data rows and a `Name` column. `src/train_names.py` reads
that column, lowercases and keeps a–z, removes duplicates, and splits by name
before constructing training windows. The screenshot introduces the data,
not generated model outputs. The source repository's preprocessing script
documents how it assembled the list.

Capture settings: Chromium, viewport 760 × 630 CSS pixels, device scale 2,
clip `{ x: 0, y: 64, width: 760, height: 536 }`, light color scheme. Wait for
the CSV preview before capturing. The assembler embeds the image in Part 1
so the downloaded HTML also works offline.
