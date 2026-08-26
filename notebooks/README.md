# Training and diagnostic notebooks

These are single-cell Colab scripts, not Jupyter notebooks: paste one into
a cell and run it. They are kept in the repository because the Layer 6
result reported in the [project README](../README.md#8-layer-6--ensemble-fusion)
is only meaningful if the path to it can be inspected, including the two
attempts that were rejected.

Run them in order. Each writes its output to Google Drive, so a recycled
Colab runtime never costs more than the step in progress.

| Script | Purpose | Cost |
|---|---|---|
| `train_pin_f1_colab.py` | Extracts the 54-feature design matrix for a corpus by running Layers 1, 2 and 4 over every image, then fits and evaluates a first meta-learner. | ~10 min per 3,000 images on an A100 |
| `diagnose_and_retrain_f1.py` | Measures each base detector per corpus, ranks features by how well they identify which corpus a sample came from, and compares five train/test configurations. | seconds |
| `train_f1_openfake.py` | Refits on the corpus where the detectors demonstrably carry signal, with a disjoint train/test split. | ~25 min extraction |
| `train_f1_cv.py` | **The script that produced the shipped model.** Pools the corpus and uses stratified 5-fold cross-validation so every prediction comes from a model that never saw that row. | ~1 min |

## Why four scripts rather than one

The first two attempts failed, and each failure changed the design:

1. **`train_pin_f1_colab.py`** fitted on `Hemg/deepfake-and-real-images` and
   scored 0.50, chance, on a cross-dataset holdout. Feature importance put
   `a4_max_face_area_ratio` far ahead of every forensic signal: the model had
   learned corpus framing, not evidence.
2. **`diagnose_and_retrain_f1.py`** established why. The base detectors are at
   chance on that corpus and PIN-B3 is *anti-correlated* with the label there,
   while all four carry real signal on `ComplexDataLab/OpenFake`. A stacked
   ensemble cannot manufacture signal its base learners lack.
3. **`train_f1_openfake.py`** refitted on OpenFake and still lost to PIN-B2
   alone, because provenance features, already adjudicated by the Layer 5
   rule calculus, dominated the fit and did not transfer across the split.
4. **`train_f1_cv.py`** enforces the documented Tier 3/4 scope and replaces the
   split with cross-validation. Fusion then beats the best single detector by
   +0.16 ROC-AUC and reduces calibration error nineteenfold.

Each script refuses to save an artefact that does not exceed the best single
detector and fall below an ECE ceiling. Two of the three were rejected by that
gate, which is why the reported figure is worth something.

## Prerequisites

- A Colab runtime with a GPU (A100 preferred, T4 workable)
- The Layer 2 weights in `MyDrive/DeepReality/models/`, see [models/README.md](../models/README.md)
- `REPO_URL` at the top of the extraction script pointing at your fork

`train_f1_cv.py` needs neither the repository nor the weights: it reads only
the cached feature matrices in Drive, so it survives a recycled runtime.
