# Model Ağırlıkları

Eğitilmiş model ağırlıkları, GitHub'ın dosya boyutu limitleri nedeniyle bu repoda tutulmamaktadır (toplam ~3.4 GB).

> 📦 **İndirme bağlantısı:** Model ağırlıkları HuggingFace üzerinde barındırılmaktadır:
> **https://huggingface.co/KULLANICI-ADI/deepreality-models** *(yükleme tamamlandığında bu bağlantı güncellenecektir)*

## Gerekli Dosyalar

Sistemin çalışması için aşağıdaki dosyaların bu klasörde (`models/`) bulunması gerekir:

| Dosya | Pin | Boyut | Açıklama |
|---|---|---|---|
| `pin_b1_clip_ln_tune_final.pt` | PIN-B1 | ~1.6 GB | CLIP ViT-L/14, frozen backbone + LayerNorm tuning |
| `pin_b2_siglip2_finetune_final.pt` | PIN-B2 | ~1.4 GB | SigLIP2-base-patch16-512, full fine-tune |
| `pin_b3_freq_cnn_final.pt` | PIN-B3 | ~19 MB | Frekans CNN (DCT/DWT, 5 blok), sıfırdan eğitildi |
| `pin_b4_ai_deepfake_real/model.safetensors` | PIN-B4 | ~354 MB | 3 sınıflı SigLIP2 sınıflandırıcı |
| `blaze_face_short_range.tflite` | PIN-A4 | ~224 KB | MediaPipe yüz tespit modeli *(repoda mevcut)* |

Not: `pin_b4_ai_deepfake_real/` klasöründeki `config.json` ve `preprocessor_config.json` dosyaları repoda mevcuttur; yalnızca `model.safetensors` indirilmelidir.

## Kurulum

HuggingFace'ten indirdiğiniz dosyaları doğrudan bu klasöre yerleştirin:

```
models/
├── pin_b1_clip_ln_tune_final.pt
├── pin_b2_siglip2_finetune_final.pt
├── pin_b3_freq_cnn_final.pt
├── blaze_face_short_range.tflite
└── pin_b4_ai_deepfake_real/
    ├── config.json                  (repoda mevcut)
    ├── preprocessor_config.json     (repoda mevcut)
    └── model.safetensors            (indirilecek)
```

PIN-B4'ün ağırlıkları alternatif olarak doğrudan kaynağından da indirilebilir:

```bash
cd models/pin_b4_ai_deepfake_real
curl -L -o model.safetensors \
  'https://huggingface.co/prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2/resolve/main/model.safetensors'
```

Ayrıca PIN-B1 ve PIN-B2, mimari iskeleti kurmak için ilk çalıştırmada HuggingFace'ten base modelleri (`openai/clip-vit-large-patch14`, `google/siglip2-base-patch16-512`) otomatik indirir — ilk çalıştırmada internet bağlantısı gereklidir, sonrasında tamamen yerel çalışır.

## Eğitim Bilgileri

| Model | Eğitim Verisi | Test Performansı |
|---|---|---|
| PIN-B1 (CLIP LN-tune) | OpenDeepfake-Preview (20K görsel) | Acc %99.77 · ROC-AUC 0.9997 |
| PIN-B2 (SigLIP2 fine-tune) | OpenDeepfake-Preview (20K görsel) | Acc %99.97 · ROC-AUC 1.0000 |
| PIN-B3 (Frekans CNN) | OpenDeepfake-Preview (20K görsel) | Acc %96.50 · ROC-AUC 0.9923 |
| PIN-B4 (Independent Core) | AI-vs-Deepfake-vs-Real (pretrained) | Acc %99.05 |

*Performans değerleri ilgili veri setlerinin test kümeleri üzerindedir; gerçek dünya görsellerinde (cross-dataset) genelleme performansı ayrıca değerlendirilmektedir.*
