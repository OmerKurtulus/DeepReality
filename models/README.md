# Model weights

Two artefacts ship with the repository; the rest are hosted separately
because they exceed GitHub's 100 MB per-file limit.

## Included in this repository

| File | Pin | Size | Notes |
|---|---|---|---|
| `pin_f1_xgboost.json` | PIN-F1 | 409 KB | Trained Layer 6 meta-learner, 400 trees over 33 features |
| `pin_f1_metadata.json` | PIN-F1 | 3 KB | Feature contract, Platt calibration, measured metrics |
| `blaze_face_short_range.tflite` | PIN-A4 | 224 KB | MediaPipe face detector |
| `pin_b4_ai_deepfake_real/config.json` | PIN-B4 | 4 KB | Architecture configuration |
| `pin_b4_ai_deepfake_real/preprocessor_config.json` | PIN-B4 | <1 KB | Input preprocessing |

The Layer 6 artefact is small enough to track directly, so the ensemble
stage works from a fresh clone with no download. It is stored as XGBoost
JSON, a tree ensemble is a set of split rules rather than a weight
matrix, so it serialises to readable text and is evaluated without the
xgboost runtime (see `layer6_ensemble/booster_eval.py`).

## Downloaded separately

> **Hugging Face:** https://huggingface.co/USERNAME/deepreality-models
> *(link to be updated once the upload is complete)*

| File | Pin | Size | Description |
|---|---|---|---|
| `pin_b1_clip_ln_tune_final.pt` | PIN-B1 | 1.6 GB | CLIP ViT-L/14, frozen backbone with LayerNorm tuning |
| `pin_b2_siglip2_finetune_final.pt` | PIN-B2 | 1.4 GB | SigLIP2-base-patch16-512, full fine-tune |
| `pin_b3_freq_cnn_final.pt` | PIN-B3 | 19 MB | Frequency CNN over DCT/DWT maps, trained from scratch |
| `pin_b4_ai_deepfake_real/model.safetensors` | PIN-B4 | 354 MB | Three-class SigLIP2 classifier |

PIN-B4's weights are a public third-party model and need no manual
upload:

```bash
cd models/pin_b4_ai_deepfake_real
curl -L -o model.safetensors \
  'https://huggingface.co/prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2/resolve/main/model.safetensors'
```

## Placement

```
models/
├── pin_b1_clip_ln_tune_final.pt        (download)
├── pin_b2_siglip2_finetune_final.pt    (download)
├── pin_b3_freq_cnn_final.pt            (download)
├── pin_f1_xgboost.json                 (in repository)
├── pin_f1_metadata.json                (in repository)
├── blaze_face_short_range.tflite       (in repository)
└── pin_b4_ai_deepfake_real/
    ├── config.json                     (in repository)
    ├── preprocessor_config.json        (in repository)
    └── model.safetensors               (download)
```

PIN-B1 and PIN-B2 additionally fetch their base architectures
(`openai/clip-vit-large-patch14`, `google/siglip2-base-patch16-512`) from
the Hugging Face Hub on first run. One internet connection is therefore
required initially; the system operates fully offline thereafter, apart
from Layer 5, which calls a language-model API by design.

## Training provenance

| Model | Training data | Reported performance |
|---|---|---|
| PIN-B1 (CLIP, LayerNorm tuning) | OpenDeepfake-Preview, 20 K images | Accuracy 99.77 %, ROC-AUC 0.9997 |
| PIN-B2 (SigLIP2, full fine-tune) | OpenDeepfake-Preview, 20 K images | Accuracy 99.97 %, ROC-AUC 1.0000 |
| PIN-B3 (frequency CNN) | OpenDeepfake-Preview, 20 K images | Accuracy 96.50 %, ROC-AUC 0.9923 |
| PIN-B4 (Independent Core) | AI-vs-Deepfake-vs-Real (pretrained) | Accuracy 99.05 % |
| PIN-F1 (meta-learner) | ComplexDataLab/OpenFake, 2,400 pooled | **ROC-AUC 0.8846, ECE 0.0203** (5-fold out-of-fold) |

The Layer 2 figures are held-out results on their own corpus and should be
read as in-distribution performance. They do not survive a change of
corpus: measured on `ComplexDataLab/OpenFake` the same detectors fall to
0.51–0.84, and on `Hemg/deepfake-and-real-images` all four sit at chance
with PIN-B3 anti-correlated. That gap is the reason the system exists and
is documented under *Known limitations* in the [project README](../README.md#11-known-limitations).

PIN-F1's figure is the only one measured by cross-validation on a corpus
none of the base detectors were trained on, which makes it the most
conservative number in this table and the one to quote.
