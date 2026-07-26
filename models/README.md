# Model Weights

Trained weights are distributed separately from this repository because their combined size (~3.4 GB) exceeds GitHub's file limits.

> **Download:** the weights are hosted on Hugging Face:
> **https://huggingface.co/USERNAME/deepreality-models** *(link to be updated once the upload is complete)*

## Required files

The following files must be present in this directory for the system to run:

| File | Pin | Size | Description |
|---|---|---|---|
| `pin_b1_clip_ln_tune_final.pt` | PIN-B1 | ~1.6 GB | CLIP ViT-L/14, frozen backbone with LayerNorm tuning |
| `pin_b2_siglip2_finetune_final.pt` | PIN-B2 | ~1.4 GB | SigLIP2-base-patch16-512, full fine-tune |
| `pin_b3_freq_cnn_final.pt` | PIN-B3 | ~19 MB | Frequency CNN (DCT/DWT, five blocks), trained from scratch |
| `pin_b4_ai_deepfake_real/model.safetensors` | PIN-B4 | ~354 MB | Three-class SigLIP2 classifier |
| `blaze_face_short_range.tflite` | PIN-A4 | ~224 KB | MediaPipe face detection model *(included in the repository)* |

The `config.json` and `preprocessor_config.json` files inside `pin_b4_ai_deepfake_real/` are tracked in the repository; only `model.safetensors` needs to be downloaded.

## Placement

Place the downloaded files directly in this directory:

```
models/
├── pin_b1_clip_ln_tune_final.pt
├── pin_b2_siglip2_finetune_final.pt
├── pin_b3_freq_cnn_final.pt
├── blaze_face_short_range.tflite
└── pin_b4_ai_deepfake_real/
    ├── config.json                  (in repository)
    ├── preprocessor_config.json     (in repository)
    └── model.safetensors            (download)
```

The PIN-B4 weights may alternatively be obtained from their original source:

```bash
cd models/pin_b4_ai_deepfake_real
curl -L -o model.safetensors \
  'https://huggingface.co/prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2/resolve/main/model.safetensors'
```

PIN-B1 and PIN-B2 additionally download their base architectures (`openai/clip-vit-large-patch14`, `google/siglip2-base-patch16-512`) from Hugging Face on first run. An internet connection is therefore required once; the system operates fully offline thereafter.

## Training summary

| Model | Training data | Test performance |
|---|---|---|
| PIN-B1 (CLIP, LayerNorm tuning) | OpenDeepfake-Preview (20 K images) | Accuracy 99.77 %, ROC-AUC 0.9997 |
| PIN-B2 (SigLIP2, full fine-tune) | OpenDeepfake-Preview (20 K images) | Accuracy 99.97 %, ROC-AUC 1.0000 |
| PIN-B3 (frequency CNN) | OpenDeepfake-Preview (20 K images) | Accuracy 96.50 %, ROC-AUC 0.9923 |
| PIN-B4 (Independent Core) | AI-vs-Deepfake-vs-Real (pretrained) | Accuracy 99.05 % |

These figures are held-out test results on the respective corpora and should be read as in-distribution performance. Cross-dataset generalisation — in particular on authentic smartphone photography — is discussed under *Known limitations* in the [project README](../README.md).
