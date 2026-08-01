# 🌿 AgriVision Pro

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

Plant disease classification from leaf images: a fine-tuned CNN, a corrected
classical baseline to compare it against, and Grad-CAM to check that the model
is looking at the lesion rather than the background.

---

## What happened in v2

v1 was a Random Forest over ten hand-crafted features, reporting 79% accuracy.
Auditing that pipeline turned up two bugs that had quietly disabled most of it.

### Bug 1 — the shape feature was a constant

`area_ratio` was computed as `countNonZero()` on the k-means *reconstruction*.
After reconstruction the image contains exactly two colours, and both are
non-zero, so the ratio was always 1.0. Measured across synthetic leaves with
lesion coverage from 0% to 100%:

| Lesion coverage | v1 `area_ratio` | v2 `lesion_ratio` |
| --------------- | --------------- | ----------------- |
| 0%              | 1.0000          | 0.0000            |
| 25%             | 1.0000          | 0.0796            |
| 50%             | 1.0000          | 0.1572            |
| 75%             | 1.0000          | 0.2214            |
| 100%            | 1.0000          | 0.2826            |

One of ten features carried zero information. The classifier was training on
nine.

### Bug 2 — texture was measured on a two-colour image

GLCM ran *after* segmentation with `levels=256`. But the segmented image has
only **2** distinct grey levels, so the 256×256 co-occurrence matrix was 99.99%
empty and contrast/correlation/energy were close to meaningless. The original
image had 37 distinct levels; a real photograph has 200+.

Colour features had the same root cause — mean and standard deviation taken over
a two-colour reconstruction discard the actual colour distribution, which is the
strongest signal available for this task.

### Bug 3 — the segmentation wasn't segmenting what the README claimed

PlantVillage images sit on a uniform grey backdrop, so the dominant k=2 split is
**leaf vs. background**, not lesion vs. healthy tissue. And on a healthy leaf,
k=2 still forces a split, partitioning sensor noise into hundreds of speckle
"lesions".

v2 masks the background first (HSV saturation + largest connected component),
clusters *within* the leaf, picks the lesion cluster deterministically by
greenness, and applies morphological opening plus an area filter so only
spatially coherent regions survive.

**These fixes are in `baseline/features.py`.** The classical pipeline is kept
rather than deleted, because "the CNN beats the classical approach" only means
something if the classical approach was implemented correctly.

---

## Architecture

| Component | Implementation |
| --------- | -------------- |
| Main model | EfficientNet-B0, ImageNet-pretrained, two-stage fine-tune |
| Baseline | Random Forest over 60 corrected hand-crafted features |
| Explainability | Grad-CAM via forward/backward hooks (no extra dependency) |
| Interface | CustomTkinter desktop app |

Training freezes the backbone and trains the new head first, then unfreezes and
fine-tunes with a lower trunk learning rate. Model selection uses **macro-F1**,
not accuracy — on an imbalanced dataset, accuracy is dominated by the largest
classes and will happily pick a model that has given up on the rare ones.

---

## Quickstart

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src        # Windows: set PYTHONPATH=%CD%\src
```

Download [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
and point `--data-root` at the **`color/`** subdirectory (the archive also ships
`grayscale/` and `segmented/`; passing the parent reads as three classes).

```bash
# Train
python -m agrivision.train --data-root "dataset/plantvillage dataset/color"

# Evaluate on the held-out test split
python -m agrivision.evaluate --data-root "dataset/plantvillage dataset/color"

# Classify an image
python -m agrivision.predict path/to/leaf.jpg

# See where the model looked
python -m agrivision.explain path/to/leaf.jpg

# Classical baseline, on the same splits
python baseline/train_baseline.py --data-root "dataset/plantvillage dataset/color"

# Desktop app
python app_ui.py
```

---

## Why the headline accuracy isn't the interesting number

PlantVillage is laboratory data: every leaf sits on the same flat studio
backdrop. A network can score near-99% by reading that backdrop and the leaf
silhouette without ever attending to a lesion — and such a model collapses on
field photographs. High accuracy here is routine and does not, on its own,
indicate a useful model.

So `evaluate.py` re-scores the same test set with the background replaced by
random colour or noise, and reports the gap:

```
=== Randomised-background probe ===
  accuracy    0.9xxx   (-0.0xxx vs clean)
  Accuracy drop from background swap: 0.0xxx
```

A large drop means the score was resting on the backdrop. Training with
`--randomize-background 0.5` trades a little clean accuracy for a model that
generalises. Grad-CAM is the visual version of the same check: heatmap on the
lesion is good, heatmap on the background means the model is cheating no matter
what the metrics say.

---

## Results

| Model | Test accuracy | Macro-F1 | Accuracy w/ randomised background |
| ----- | ------------- | -------- | --------------------------------- |
| v1 Random Forest — 10 features, 2 inert | 0.79 | — | — |
| v2 baseline — 60 corrected features | _TBD_ | _TBD_ | _TBD_ |
| v2 EfficientNet-B0 fine-tuned | _TBD_ | _TBD_ | _TBD_ |

Splits are computed once, written to `artifacts/splits.json`, and reused by every
stage including the baseline — so the comparison is like-for-like rather than
across two different random partitions.

---

## Engineering notes

Fixes beyond the feature pipeline:

- **Three-way split.** v1 used a single 80/20 split and reported test accuracy on
  data that had also driven hyper-parameter choices. There is now a test set
  touched exactly once.
- **Checkpoints carry their schema.** v1 saved a bare `.pkl`; if the feature code
  changed, the old model kept predicting, just wrongly. Checkpoints now embed the
  class list, image size, normalisation and a schema version, and loading refuses
  on mismatch.
- **Stratified splits, balanced class weights, cross-validated baseline scores.**
- **Cached, parallel feature extraction** in the baseline, so re-tuning doesn't
  re-extract 54k images serially.
- **GUI:** `state("zoomed")` was Windows-only and raised `TclError` elsewhere;
  inference ran on the UI thread behind a blocking sleep; a missing model file
  failed silently until the user clicked Run; the placeholder advertised
  drag-and-drop that was never implemented. All fixed, and the app now shows
  runner-up classes and an explicit low-confidence warning.

---

## Project layout

```
app_ui.py              Desktop GUI
src/agrivision/
    data.py            Splits, transforms, background-randomisation probe
    model.py           Backbone factory, staged freezing, checkpoint bundling
    train.py           Two-stage fine-tuning
    evaluate.py        Test metrics, confusion matrix, calibration, robustness
    predict.py         Reusable Predictor + CLI
    explain.py         Grad-CAM
baseline/
    features.py        Corrected hand-crafted features
    train_baseline.py  Random Forest on shared splits
```

---

## Limitations

Trained on laboratory images with controlled lighting and uniform backgrounds;
field performance will be lower. Expected calibration error is reported because
the confidence percentage shown to the user is misleading if the model is
overconfident. This is a research prototype, not agronomic advice.

## License

MIT — see [LICENSE](LICENSE).
