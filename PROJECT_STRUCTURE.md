# DeepReality — Proje Yapısı ve Kaynak Kod Dokümantasyonu

## Projenin Amacı

DeepReality, görsellerin (fotoğraf/resim) yapay zeka (AI) tarafından üretilip üretilmediğini veya manipüle edilip edilmediğini tespit etmeye çalışan çok katmanlı bir analiz sistemidir. Python ile yazılmıştır. Şu an **Layer 1 (Ön İşlem / Preprocessing)**, **Layer 2 (Detection Core / Ana Tespit Motoru)** ve **Layer 4 (XAI / Açıklanabilirlik)** aktif olarak çalışmaktadır. Layer 3 (video temporal — bilinçli olarak ertelendi), Layer 5-6 klasörleri mevcuttur ancak henüz içleri boştur (gelecek geliştirme için planlanmış).

Sistem "PIN" adı verilen bağımsız analiz modülleri üzerine kuruludur. Her PIN farklı bir teknikle görseli analiz eder ve 0.0 (temiz/gerçek) ile 1.0 (kesin sahte/AI üretimi) arasında bir skor üretir.

**PARALEL ÇALIŞMA (PIN Architecture'ın özü):** Pinler `core/pipeline.py` içindeki bağımlılık grafiği (DAG) tabanlı orkestratör ile çalıştırılır. Birbirine bağımlı OLMAYAN tüm pinler (Layer 1'in 4 pini + Layer 2'nin 4 pini = 8 pin) aynı anda paralel başlar. Yalnızca başka bir pinin çıktısına ihtiyaç duyan pinler (XAI pinleri: PIN-D1 ← B1+B2+B3, PIN-D2 ← A3+B3+D1) bağımlılıkları bittiği anda çalışır.

**Layer 1** hafif metadata ve sinyal tabanlı ön analiz yaparken, **Layer 2** derin öğrenme modelleri (CLIP, SigLIP2, Frekans CNN, Independent Core) ile görselleri sınıflandırır. Layer 2'deki her model farklı bir perspektiften analiz yapar — spatial domain, frequency domain ve çok sınıflı (AI/Deepfake/Real) ayrımı. **Layer 4 (XAI)** modellerin kararlarını Grad-CAM ısı haritalarıyla görselleştirir (PIN-D1) ve ELA + Grad-CAM kanıtlarını birleştirerek manipülasyon bölgelerini lokalize eder (PIN-D2).

---

## Dizin Ağacı (Directory Tree)

```
DeepReality/
├── main.py                              # Ana çalıştırıcı — PIN'leri paralel orkestratörle çalıştırır (Layer 1+2+4)
├── Requirements.txt                     # Python bağımlılıkları
│
├── config/
│   ├── __init__.py                      # (boş)
│   └── settings.py                      # Global konfigürasyon — eşikler, ağırlıklar, yollar (XAI_CONFIG dahil)
│
├── core/
│   ├── __init__.py                      # (boş)
│   ├── base_pin.py                      # Tüm PIN'lerin miras aldığı soyut temel sınıf (BasePin) — context desteği
│   └── pipeline.py                      # PinPipeline: DAG tabanlı PARALEL pin orkestratörü
│
├── layer1_preprocessing/
│   ├── __init__.py                      # Layer 1 modül açıklaması
│   ├── pin_a1_metadata.py              # PIN-A1: EXIF/Metadata Analizi
│   ├── pin_a2_c2pa.py                  # PIN-A2: C2PA Provenance (Dijital İmza) Analizi
│   ├── pin_a3_ela.py                   # PIN-A3: ELA (Error Level Analysis)
│   └── pin_a4_face.py                  # PIN-A4: Yüz Tespiti & Kırpma
│
├── layer2_detection_core/               # Layer 2: Derin öğrenme tabanlı tespit motoru
│   ├── __init__.py                      # Layer 2 modül açıklaması
│   ├── pin_b1_clip.py                  # PIN-B1: CLIP ViT-L/14 (Frozen + LN-tune) Deepfake Detection
│   ├── pin_b2_siglip2.py              # PIN-B2: SigLIP2-base-512 (Fine-tuned) Deepfake Detection
│   ├── pin_b3_freq.py                 # PIN-B3: Frekans Analizi (DCT/DWT + CNN)
│   └── pin_b4_IndependentCore.py      # PIN-B4: Independent Core (AI vs Deepfake vs Real, 3-sınıf)
│
├── layer3_video_temporal/               # (boş — ertelendi: Video temporal analiz, ileride eklenecek)
│
├── layer4_xai/                          # Layer 4: Açıklanabilirlik (XAI) pinleri
│   ├── __init__.py                      # Layer 4 modül açıklaması
│   ├── pin_d1_gradcam.py               # PIN-D1: Grad-CAM Heatmap (CLIP + SigLIP2 + FreqCNN karar odakları)
│   └── pin_d2_anomaly.py               # PIN-D2: Anomaly Localization (ELA + Grad-CAM füzyonu)
│
├── layer5_llm_reasoning/                # (boş — gelecek: LLM tabanlı muhakeme)
├── layer6_ensemble/                     # (boş — gelecek: Tüm katmanların birleşik kararı)
├── tests/                               # (boş — gelecek: Test dosyaları)
│
├── models/
│   ├── blaze_face_short_range.tflite   # PIN-A4: MediaPipe yüz tespit modeli (binary, ~224 KB)
│   ├── pin_b1_clip_ln_tune_final.pt   # PIN-B1: CLIP ViT-L/14 eğitilmiş ağırlıklar (~1.6 GB)
│   ├── pin_b2_siglip2_finetune_final.pt # PIN-B2: SigLIP2 eğitilmiş ağırlıklar (~1.4 GB)
│   ├── pin_b3_freq_cnn_final.pt       # PIN-B3: Frequency CNN eğitilmiş ağırlıklar (~19 MB)
│   └── pin_b4_ai_deepfake_real/       # PIN-B4: HuggingFace pretrained model dizini (~354 MB toplam)
│       ├── config.json                 # Model konfigürasyonu (SiglipForImageClassification, 3 label)
│       ├── model.safetensors           # Model ağırlıkları (~354 MB)
│       └── preprocessor_config.json    # Görsel ön işleme konfigürasyonu (224x224, mean/std 0.5)
│
├── input/                               # Analiz edilecek görseller buraya konur
│   ├── test3.png                        # Test görseli (~6.4 MB)
│   ├── test5.png                        # Test görseli (2048x2048, ~8.3 MB)
│   ├── test8.png                        # Test görseli (1024x1024, ~1.1 MB)
│   ├── test9.png                        # Test görseli (~1.9 MB)
│   └── test11.png                       # Test görseli (~5.9 MB)
│
├── outputs/                             # Analiz sonuçları buraya yazılır (her görsel için 8 JSON + ek dosyalar)
│   ├── test3_PIN-A1.json ... test3_PIN-B4.json    # test3 Layer 1+2 sonuçları
│   ├── test3_ELA_heatmap.png                       # test3 ELA ısı haritası
│   ├── test3_face_0.png                             # test3 kırpılmış yüz
│   ├── test5_PIN-A1.json ... test5_PIN-B4.json    # test5 Layer 1+2 sonuçları
│   ├── test5_ELA_heatmap.png                       # test5 ELA ısı haritası
│   ├── test5_face_0.png                             # test5 kırpılmış yüz
│   ├── test8_PIN-A1.json ... test8_PIN-B4.json    # test8 Layer 1+2 sonuçları
│   ├── test8_ELA_heatmap.png                       # test8 ELA ısı haritası
│   ├── test8_face_0.png                             # test8 kırpılmış yüz
│   ├── test9_PIN-A1.json ... test9_PIN-B4.json    # test9 Layer 1+2 sonuçları
│   ├── test9_ELA_heatmap.png                       # test9 ELA ısı haritası
│   ├── test9_face_0.png                             # test9 kırpılmış yüz
│   ├── test11_PIN-A1.json ... test11_PIN-B4.json  # test11 Layer 1+2 sonuçları
│   ├── test11_ELA_heatmap.png                      # test11 ELA ısı haritası
│   └── test11_face_0.png                            # test11 kırpılmış yüz
│
└── venv/                                # Python sanal ortamı (pip paketleri)
```

---

## Çalışma Akışı

1. Kullanıcı `input/` klasörüne görseller koyar
2. `python3 main.py` çalıştırılır — `build_pipeline()` bağımlılık grafiğini kurar
3. Her görsel için **8 bağımsız pin AYNI ANDA (PARALEL)** başlar:
   - **PIN-A1**: EXIF/Metadata analizi → AI aracı imzası arar
   - **PIN-A2**: C2PA dijital imza doğrulaması → resmi provenance verisi çıkarır
   - **PIN-A3**: ELA (Error Level Analysis) → sıkıştırma farkları ile manipülasyon tespiti
   - **PIN-A4**: Yüz tespiti & kırpma → yüzleri bulur, hizalar, normalize eder
   - **PIN-B1**: CLIP ViT-L/14 → spatial domain'de deepfake/AI tespiti (224x224, 1024-dim feature)
   - **PIN-B2**: SigLIP2-base-512 → yüksek çözünürlüklü micro-anomaly tespiti (512x512, 768-dim feature)
   - **PIN-B3**: Frekans Analizi (DCT/DWT + CNN) → frequency domain'de yapay iz tespiti (4-kanal, 512-dim feature)
   - **PIN-B4**: Independent Core → 3 sınıflı ayrım: AI / Deepfake / Real
4. Bağımlılıkları biten **Layer 4 (XAI)** pinleri otomatik başlar:
   - **PIN-D1** (B1+B2+B3 bitince): Grad-CAM → her modelin karar odağı ısı haritası + birleşik harita
   - **PIN-D2** (A3+B3+D1 bitince): Anomali lokalizasyonu → ELA + Grad-CAM füzyonu, işaretli bölgeler
5. Sonuçlar `outputs/` klasörüne JSON dosyaları olarak yazılır (her görsel için 10 JSON + XAI PNG'leri)
6. Terminalde pin bazlı canlı ilerleme + katman özetleri + paralel hızlanma oranı gösterilir

---

## Dosya Detayları ve Kaynak Kodlar

---

### 1. `main.py` — Ana Çalıştırıcı (Paralel Orkestrasyon)

**Konum:** `/DeepReality/main.py`
**İşlev:** Ana entry point. `input/` klasöründeki görselleri bulur ve her biri için `core/pipeline.py` içindeki **paralel PIN orkestratörünü** çalıştırır. Pinleri artık sırayla DEĞİL, bağımlılık grafiğine (DAG) göre paralel çalıştırır.

**Önemli noktalar:**
- `build_pipeline()`: PIN Architecture bağımlılık grafiğini kurar:
  - **Bağımsız (paralel):** PIN-A1, A2, A3, A4, B1, B2, B3, B4 — 8 pin aynı anda başlar
  - **Bağımlı:** PIN-D1 ← [B1, B2, B3] (Grad-CAM model instance'larını paylaşır), PIN-D2 ← [A3, B3, D1] (ELA bölgeleri + ham CAM matrisi)
- `on_pin_complete` callback'i ile her pin bittiği anda terminale `[OK] PIN-XX  1.23s` satırı yazılır (canlı ilerleme)
- Tüm pinler bitince özetler `PIN_DISPLAY_ORDER` sabit sırasıyla yazdırılır (A1→A2→...→D2) — paralel çalışma çıktı düzenini bozmaz
- Görsel başına paralel süre / sıralı süre / hızlanma çarpanı raporlanır
- Her PIN için ayrı `print_pin_XX_summary()` fonksiyonu (A1-A4, B1-B4, D1-D2)
- HEIC/HEIF (iPhone) desteği `pillow_heif` ile; MediaPipe/TFLite uyarıları bastırılır

**Tam kaynak kodu için dosyanın kendisine bakınız** (`main.py`). Pipeline mantığı `core/pipeline.py` içindedir (bkz. bölüm 6b).

---

### 2. `Requirements.txt` — Bağımlılıklar

**Konum:** `/DeepReality/Requirements.txt`
**İşlev:** Projenin ihtiyaç duyduğu Python paketleri.

```
# DeepReality — Bağımlılıklar
# pip install -r Requirements.txt

# Layer 1 — Preprocessing
Pillow>=10.0.0          # PIN-A1, PIN-A3: Görsel okuma, EXIF çıkarma, JPEG dönüşüm
c2pa-python>=0.28.0     # PIN-A2: C2PA Content Credentials parse + doğrulama
opencv-python>=4.8.0    # PIN-A3, PIN-A4: ELA heatmap, görsel işleme, yüz kırpma
numpy>=1.24.0           # PIN-A3, PIN-A4: İstatistiksel analiz
pillow-heif>=0.16.0     # PIN-A3: iPhone HEIC/HEIF format desteği
mediapipe>=0.10.0       # PIN-A4: Yüz tespiti (Face Detection) + landmarks

# Layer 2 — Detection Core
torch>=2.0.0            # PIN-B1: PyTorch deep learning framework
transformers>=4.40.0    # PIN-B1: HuggingFace CLIP model loading
```

**Not:** Layer 2 ek olarak `scipy` (DCT dönüşümü) ve `pywt` (DWT wavelet dönüşümü) paketlerini de kullanır — bunlar PIN-B3 tarafından import edilir ancak Requirements.txt'te henüz listelenmemiştir.

---

### 3. `config/__init__.py`

**Konum:** `/DeepReality/config/__init__.py`
**İşlev:** Boş init dosyası — `config` klasörünü Python paketi yapar.

*(Dosya boştur)*

---

### 4. `config/settings.py` — Global Konfigürasyon

**Konum:** `/DeepReality/config/settings.py`
**İşlev:** Tüm PIN'lerin kullandığı paylaşılan ayarları, eşik değerleri, AI araç imzalarını, ağırlıkları ve yolları tanımlar. Projenin "beyni" denilebilir — tüm karar mekanizmaları buradaki değerlere bağlıdır.

**İçerdiği konfigürasyonlar (Layer 1):**
- `METADATA_CONFIG`: PIN-A1 için AI aracı imzaları (Stable Diffusion, Midjourney, DALL-E, Adobe Firefly, Leonardo AI, Flux, vb.), kamera EXIF alanları, GPS alanları, C2PA binary marker'ları, AI tipik boyutları, scoring ağırlıkları ve verdict eşikleri
- `ELA_CONFIG`: PIN-A3 için JPEG yeniden kaydetme kalitesi, amplifikasyon çarpanı, grid boyutu, hotspot/coldspot eşikleri, uniformity eşikleri
- `C2PA_CONFIG`: PIN-A2 için AI dijital kaynak tipleri (IPTC standardı), bilinen AI issuer'ları, bilinen AI yazılım ajanları, C2PA eylemleri
- `FACE_CONFIG`: PIN-A4 için MediaPipe model seçimi, minimum güvenilirlik, kırpma marjini, normalize boyutu, maksimum yüz sayısı

**İçerdiği konfigürasyonlar (Layer 2 — YENİ):**
- `CLIP_CONFIG`: PIN-B1 için HuggingFace CLIP model adı, eğitilmiş ağırlık yolu, label map, verdict eşikleri
- `SIGLIP_CONFIG`: PIN-B2 için HuggingFace SigLIP2 model adı, eğitilmiş ağırlık yolu, label map, verdict eşikleri
- `FREQ_CONFIG`: PIN-B3 için eğitilmiş model yolu, frekans dönüşüm parametreleri (boyut, wavelet, kanal sayısı), label map, verdict eşikleri
- `INDEPENDENT_CORE_CONFIG`: PIN-B4 için local model dizini, HuggingFace kaynak bilgisi, 3-sınıf label map (AI/Deepfake/Real), verdict eşikleri

```python
"""
DeepReality — Global Configuration
Tüm pinlerin kullandığı ortak ayarlar, eşik değerler ve yollar.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# PIN-A1: Metadata Analysis Thresholds
# ──────────────────────────────────────────────
METADATA_CONFIG = {
    # AI aracı metadata imzaları — her yeni araç keşfedildiğinde buraya ekle
    "ai_tool_signatures": {
        # Stable Diffusion ailesi
        "stable_diffusion": {
            "software_patterns": [
                "stable diffusion", "automatic1111", "comfyui",
                "invoke ai", "diffusionbee", "easy diffusion",
                "sd.next", "forge", "a1111"
            ],
            "parameter_fields": ["parameters", "prompt", "negative_prompt", "steps", "sampler", "cfg_scale", "seed"],
            "comment_patterns": ["stable diffusion", "sd model", "lora", "vae", "txt2img", "img2img"]
        },
        # Midjourney
        "midjourney": {
            "software_patterns": ["midjourney"],
            "comment_patterns": ["midjourney", "--ar ", "--v ", "--s ", "--q ", "--style"],
            "description_patterns": ["midjourney", "mj_"]
        },
        # DALL-E (OpenAI)
        "dalle": {
            "software_patterns": ["dall-e", "dalle", "openai"],
            "comment_patterns": ["dall-e", "dalle", "openai"],
            "xmp_patterns": ["openai", "dall-e"]
        },
        # Adobe Firefly
        "adobe_firefly": {
            "software_patterns": ["adobe firefly", "firefly"],
            "comment_patterns": ["adobe firefly", "firefly"],
            "xmp_patterns": ["firefly"]
        },
        # Leonardo AI
        "leonardo_ai": {
            "software_patterns": ["leonardo", "leonardo.ai"],
            "comment_patterns": ["leonardo"]
        },
        # Flux
        "flux": {
            "software_patterns": ["flux", "black forest labs"],
            "comment_patterns": ["flux"]
        },
        # Generic AI indicators
        "generic_ai": {
            "software_patterns": [
                "novelai", "runway", "pika", "kling", "suno",
                "deepai", "nightcafe", "artbreeder", "canva ai",
                "photoshop ai", "generative fill", "neural filters"
            ]
        }
    },

    # Kamera EXIF alanları — bu alanlar VARSA gerçek fotoğraf olma olasılığı artar
    "camera_fields": [
        "Make", "Model", "LensModel", "LensMake",
        "FocalLength", "FNumber", "ExposureTime",
        "ISOSpeedRatings", "ISO", "Flash",
        "ShutterSpeedValue", "ApertureValue",
        "MeteringMode", "WhiteBalance"
    ],

    # GPS alanları — gerçek fotoğraf kanıtı
    "gps_fields": [
        "GPSLatitude", "GPSLongitude", "GPSAltitude",
        "GPSLatitudeRef", "GPSLongitudeRef",
        "GPSInfo"
    ],

    # ── C2PA / JUMBF Binary Markers ──
    # ChatGPT, DALL-E, Sora gibi araçlar dosyaya C2PA metadata gömer.
    # Dosyanın binary içeriğinde bu byte pattern'ları aranır.
    "c2pa_binary_markers": [
        b"jumb",                    # JUMBF box marker
        b"c2pa",                    # C2PA manifest label
        b"caBX",                    # PNG C2PA chunk type
        b"c2pa.actions",            # C2PA action assertion
        b"c2pa.created",            # C2PA creation marker
        b"c2pa.hash.data",          # C2PA hash assertion
        b"trainedAlgorithmicMedia", # IPTC digital source type for AI
        b"c2pa-rs",                 # C2PA Rust SDK identifier (OpenAI uses this)
    ],
    # C2PA issuer → tool mapping
    "c2pa_issuers": {
        b"OpenAI": "openai_dalle",
        b"ChatGPT": "chatgpt",
        b"DALL": "dalle",
        b"GPT-4o": "gpt4o",
        b"Adobe": "adobe_firefly",
        b"Google": "google_ai",
        b"Microsoft": "microsoft_designer",
    },

    # ── AI Dimension Heuristic ──
    # AI araçları belirli boyutlarda çıktı verir (64'ün katları, kare, vb.)
    # Bu boyutlar kesin kanıt DEĞİL ama ek sinyal üretir.
    "ai_typical_dimensions": [
        (256, 256), (512, 512), (768, 768),
        (1024, 1024), (1536, 1536), (2048, 2048),
        # DALL-E / ChatGPT
        (1024, 1792), (1792, 1024),
        # Stable Diffusion
        (512, 768), (768, 512),
        # SDXL
        (1024, 576), (576, 1024),
        (1024, 768), (768, 1024),
        # Midjourney
        (1456, 816), (816, 1456),
    ],

    # Scoring weights (toplam = 1.0)
    "weights": {
        "ai_signature_detected": 0.30,   # Metadata'da AI imzası
        "c2pa_detected": 0.25,           # C2PA/JUMBF binary marker bulundu
        "no_camera_data": 0.12,          # Kamera verisi yok
        "no_gps_data": 0.03,             # GPS yok
        "no_datetime": 0.07,             # Tarih bilgisi yok
        "software_suspicious": 0.08,     # Yazılım alanı şüpheli
        "metadata_stripped": 0.05,       # Metadata tamamen silinmiş
        "ai_dimensions": 0.05,           # AI tipik boyutları
        "compression_anomaly": 0.05      # Sıkıştırma oranı anomalisi
    },

    # Verdict thresholds
    "thresholds": {
        "high_risk": 0.70,      # ≥ 0.70 → Yüksek AI üretimi şüphesi
        "medium_risk": 0.40,    # ≥ 0.40 → Orta şüphe
        "low_risk": 0.0         # < 0.40 → Düşük şüphe / muhtemelen gerçek
    }
}

# ──────────────────────────────────────────────
# PIN-A3: ELA (Error Level Analysis) Configuration
# ──────────────────────────────────────────────
ELA_CONFIG = {
    # JPEG yeniden kaydetme kalitesi (ELA referansı)
    # Düşük = daha fazla fark görünür, Yüksek = daha hassas analiz
    "resave_quality": 90,

    # ELA fark amplifikasyonu (görselleştirme için)
    # Piksel farkları küçük olduğundan görünürlük için çarpan
    "amplification_scale": 20,

    # Bölgesel analiz ızgara boyutu (grid NxN)
    "grid_size": 8,

    # Anomali tespit eşiği (MAD tabanlı, robust)
    "hotspot_std_threshold": 3.0,

    # Coldspot için ayrı σ eşiği
    "coldspot_std_threshold": 3.5,

    # Coldspot minimum absolute ELA farkı
    "coldspot_min_absolute_deviation": 20.0,

    # Uniformity eşikleri (AI tespiti için)
    "uniformity_thresholds": {
        "very_uniform": 3.0,    # < 3.0 → çok uniform (AI sinyali)
        "uniform": 6.0,         # < 6.0 → uniform (olası AI)
        "moderate": 12.0,       # < 12.0 → orta (belirsiz)
        # > 12.0 → yüksek varyasyon (gerçek veya manipüle)
    },

    # ELA heatmap kaydetme
    "save_heatmap": True,
    "heatmap_colormap": "jet",  # OpenCV colormap: jet, hot, inferno
}

# ──────────────────────────────────────────────
# JSON Output Standard Format
# ──────────────────────────────────────────────
OUTPUT_SCHEMA_VERSION = "1.0.0"
# ──────────────────────────────────────────────
# PIN-A2: C2PA Provenance Configuration
# ──────────────────────────────────────────────
C2PA_CONFIG = {
    # AI üretim dijital kaynak tipleri (IPTC standardı)
    "ai_digital_source_types": [
        "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
        "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicMedia",
        "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
    ],

    # Bilinen AI üretim araçları (C2PA issuer veya softwareAgent olarak görünür)
    "known_ai_issuers": {
        "OpenAI": "openai",
        "Google": "google_ai",
        "Adobe": "adobe",
        "Microsoft": "microsoft",
        "Stability AI": "stability_ai",
        "Midjourney": "midjourney",
    },
    "known_ai_software_agents": [
        "DALL-E", "DALL·E", "GPT-4o", "GPT-4", "ChatGPT",
        "Gemini", "Imagen", "ImageFX",
        "Adobe Firefly", "Firefly",
        "Stable Diffusion", "SDXL",
        "Midjourney",
        "Microsoft Designer", "Copilot",
    ],

    # AI olmayan eylemler (kamera, tarayıcı vb.)
    "non_ai_actions": [
        "c2pa.captured",       # Kamerayla çekilmiş
        "c2pa.scanned",        # Taranmış
    ],

    # AI üretim eylemleri
    "ai_creation_actions": [
        "c2pa.created",        # Oluşturulmuş (AI olabilir)
        "c2pa.generated",      # Üretilmiş
    ],

    # Düzenleme eylemleri (AI ile düzenlenmiş olabilir)
    "edit_actions": [
        "c2pa.edited",
        "c2pa.drawing",
        "c2pa.converted",
        "c2pa.cropped",
        "c2pa.resized",
        "c2pa.opened",
    ],
}

# ──────────────────────────────────────────────
# PIN-A4: Face Detection & Cropping
# ──────────────────────────────────────────────
FACE_CONFIG = {
    # MediaPipe model seçimi
    # 0 = kısa mesafe (≤2m, selfie kamerası)
    # 1 = tam menzil (≤5m, genel fotoğraflar)
    "model_selection": 1,

    # Minimum yüz tespit güvenilirliği (0.0 - 1.0)
    "min_detection_confidence": 0.5,

    # Yüz kırpma marjini (yüz boyutunun yüzdesi)
    "crop_margin": 0.30,

    # Normalize edilmiş yüz boyutu [genişlik, yükseklik]
    # 224×224: Standart CNN/ViT input boyutu
    "normalized_size": [224, 224],

    # Maksimum tespit edilecek yüz sayısı
    "max_faces": 10,
}

# ──────────────────────────────────────────────
# PIN-B1: CLIP ViT-L/14 Deepfake Detection
# ──────────────────────────────────────────────
CLIP_CONFIG = {
    # HuggingFace model adı (mimari iskeleti yüklemek için)
    "clip_model_name": "openai/clip-vit-large-patch14",

    # Eğitilmiş model ağırlıkları dosya yolu
    "model_path": str(PROJECT_ROOT / "models" / "pin_b1_clip_ln_tune_final.pt"),

    # Sınıf sayısı
    "num_labels": 2,

    # Etiket haritası
    "label_map": {0: "fake", 1: "real"},

    # Verdict eşikleri (0.0=real, 1.0=fake)
    "thresholds": {
        "high_risk": 0.70,
        "medium_risk": 0.40,
    },
}

# ──────────────────────────────────────────────
# PIN-B2: SigLIP2-base-512 Deepfake Detection
# ──────────────────────────────────────────────
SIGLIP_CONFIG = {
    # HuggingFace model adı (mimari iskeleti yüklemek için)
    "model_name": "google/siglip2-base-patch16-512",

    # Eğitilmiş model ağırlıkları dosya yolu
    "model_path": str(PROJECT_ROOT / "models" / "pin_b2_siglip2_finetune_final.pt"),

    # Sınıf sayısı
    "num_labels": 2,

    # Etiket haritası
    "label_map": {0: "fake", 1: "real"},

    # Verdict eşikleri (0.0=real, 1.0=fake)
    "thresholds": {
        "high_risk": 0.70,
        "medium_risk": 0.40,
    },
}

# ──────────────────────────────────────────────
# PIN-B3: Frequency Analysis (DCT/DWT + CNN)
# ──────────────────────────────────────────────
FREQ_CONFIG = {
    # Eğitilmiş model ağırlıkları dosya yolu
    "model_path": str(PROJECT_ROOT / "models" / "pin_b3_freq_cnn_final.pt"),

    # Sınıf sayısı
    "num_labels": 2,

    # Frekans dönüşüm parametreleri
    "freq_image_size": 224,           # Frekans haritası boyutu
    "dwt_wavelet": "haar",            # DWT wavelet ailesi
    "num_channels": 4,                # DCT + DWT-LH + DWT-HL + DWT-HH

    # Etiket haritası
    "label_map": {0: "fake", 1: "real"},

    # Verdict eşikleri (0.0=real, 1.0=fake)
    "thresholds": {
        "high_risk": 0.70,
        "medium_risk": 0.40,
    },
}

# ──────────────────────────────────────────────
# PIN-B4: Independent Core (AI vs Deepfake vs Real)
# ──────────────────────────────────────────────
INDEPENDENT_CORE_CONFIG = {
    # Local model dizini (HuggingFace'den indirilen dosyalar)
    # İçinde: config.json, preprocessor_config.json, model.safetensors
    "model_dir": str(PROJECT_ROOT / "models" / "pin_b4_ai_deepfake_real"),

    # Kaynak bilgisi
    "source": "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2",
    "base_model": "google/siglip2-base-patch16-224",

    # Sınıf sayısı ve etiket haritası
    "num_labels": 3,
    "label_map": {0: "AI", 1: "Deepfake", 2: "Real"},

    # Verdict eşikleri (fake_score = ai_prob + deepfake_prob)
    # 0.0 = kesinlikle gerçek, 1.0 = kesinlikle fake/AI
    "thresholds": {
        "high_risk": 0.70,
        "medium_risk": 0.40,
    },
}
```

---

### 5. `core/__init__.py`

**Konum:** `/DeepReality/core/__init__.py`
**İşlev:** Boş init dosyası — `core` klasörünü Python paketi yapar.

*(Dosya boştur)*

---

### 6. `core/base_pin.py` — Temel PIN Sınıfı (Abstract Base Class)

**Konum:** `/DeepReality/core/base_pin.py`
**İşlev:** Tüm PIN modüllerinin miras aldığı soyut temel sınıf. Standart JSON çıktı formatını, dosya hash hesaplama, çıktı kaydetme ve hata yönetimi mekanizmalarını sağlar. Her PIN bu sınıftan türer ve `analyze()` metodunu kendi mantığıyla doldurur.

**Standart JSON çıktı formatı:**
```json
{
    "schema_version": "1.0.0",
    "pin_id": "PIN-A1",
    "pin_name": "EXIF/Metadata Analysis",
    "layer": 1,
    "timestamp": "2026-02-16T...",
    "input_file": "image.jpg",
    "input_hash": "sha256...",
    "status": "success" | "error",
    "results": { ... },
    "score": 0.0 - 1.0,
    "verdict": "low_risk" | "medium_risk" | "high_risk",
    "details": "Türkçe açıklama",
    "errors": []
}
```

```python
"""
DeepReality — Base Pin Class
Tüm PIN modüllerinin miras alacağı temel sınıf.
Standart JSON çıktı formatı ve ortak yardımcı metodlar burada tanımlanır.
"""

import json
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from config.settings import OUTPUTS_DIR, OUTPUT_SCHEMA_VERSION


class BasePin(ABC):
    """
    Her PIN bu sınıftan türer.
    Standart çıktı formatı:
    {
        "schema_version": "1.0.0",
        "pin_id": "PIN-A1",
        "pin_name": "EXIF/Metadata Analysis",
        "layer": 1,
        "timestamp": "2026-02-16T...",
        "input_file": "image.jpg",
        "input_hash": "sha256...",
        "status": "success" | "error",
        "results": { ... },          # Pin'e özel sonuçlar
        "score": 0.0 - 1.0,          # 0 = temiz, 1 = kesin sahte
        "verdict": "low_risk" | "medium_risk" | "high_risk",
        "details": "Türkçe açıklama",
        "errors": []
    }
    """

    def __init__(self, pin_id: str, pin_name: str, layer: int):
        self.pin_id = pin_id
        self.pin_name = pin_name
        self.layer = layer
        self.errors: list[str] = []

    @abstractmethod
    def analyze(self, file_path: str) -> dict:
        """
        Her pin bu metodu kendi analiz mantığıyla doldurur.
        Returns: {"results": {...}, "score": float, "details": str}
        """
        pass

    def run(self, file_path: str) -> dict:
        """
        Ana çalıştırıcı. analyze() metodunu çağırır,
        standart JSON formatına sarar, dosyaya kaydeder.
        """
        self.errors = []
        file_path = Path(file_path)

        # Dosya kontrolü
        if not file_path.exists():
            return self._build_output(
                file_path=str(file_path),
                file_hash="",
                status="error",
                results={},
                score=0.0,
                verdict="error",
                details=f"Dosya bulunamadı: {file_path}"
            )

        # Dosya hash'i hesapla (tekrarlı analizleri önlemek ve takip için)
        file_hash = self._compute_hash(file_path)

        try:
            analysis = self.analyze(str(file_path))
            output = self._build_output(
                file_path=str(file_path),
                file_hash=file_hash,
                status="success",
                results=analysis.get("results", {}),
                score=analysis.get("score", 0.0),
                verdict=analysis.get("verdict", "low_risk"),
                details=analysis.get("details", "")
            )
        except Exception as e:
            self.errors.append(str(e))
            output = self._build_output(
                file_path=str(file_path),
                file_hash=file_hash,
                status="error",
                results={},
                score=0.0,
                verdict="error",
                details=f"Analiz hatası: {str(e)}"
            )

        # JSON dosyasına kaydet
        self._save_output(output, file_path.stem)
        return output

    def _build_output(self, file_path: str, file_hash: str,
                      status: str, results: dict, score: float,
                      verdict: str, details: str) -> dict:
        """Standart JSON çıktı formatını oluşturur."""
        return {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "pin_id": self.pin_id,
            "pin_name": self.pin_name,
            "layer": self.layer,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_file": str(file_path),
            "input_hash": file_hash,
            "status": status,
            "results": results,
            "score": round(score, 4),
            "verdict": verdict,
            "details": details,
            "errors": self.errors
        }

    def _save_output(self, output: dict, file_stem: str) -> Path:
        """JSON çıktısını outputs/ klasörüne kaydeder."""
        filename = f"{file_stem}_{self.pin_id}.json"
        output_path = OUTPUTS_DIR / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        return output_path

    @staticmethod
    def _compute_hash(file_path: Path) -> str:
        """Dosyanın SHA-256 hash'ini hesaplar."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
```

---

### 6b. `core/pipeline.py` — Paralel PIN Orkestratörü (YENİ)

**Konum:** `/DeepReality/core/pipeline.py`
**İşlev:** PIN Architecture'ın kalbi. Pinleri bağımlılık grafiği (DAG) üzerinden çalıştırır: bağımlılığı olmayan tüm pinler `ThreadPoolExecutor` ile AYNI ANDA başlar; bir pin, tüm üst pinleri bittiği anda (başkalarını beklemeden) başlar.

**Ana sınıflar:**
- `PinPipeline(max_workers=8)`: `add_pin(pin, depends_on=[...])` ile graf kurulur; `run(file_path, on_pin_complete)` tek görsel için tüm pinleri çalıştırır. Eksik bağımlılık ve döngü kontrolü (`_validate`) içerir.
- `PipelineRun`: `results` (pin_id→JSON), `durations` (pin_id→saniye), `total_time` (paralel süre), `sequential_time` (sıralı olsaydı), `speedup` (hızlanma çarpanı).

**Context geçişi:** Bağımlı pinlere üst pin sonuçları `context` dict'i ile verilir: `context["PIN-B1"]` → tam sonuç JSON'u, `context["_pins"]["PIN-D1"]` → pin instance'ı (PIN-D2'nin, PIN-D1'in ham CAM matrislerine `cam_cache` üzerinden erişmesi için — numpy matrisler JSON'a yazılmaz).

**Neden thread (process değil)?** PyTorch inference, NumPy/OpenCV ve dosya I/O GIL'i bıraktığı için thread'ler bu iş yükünde gerçek paralellik sağlar; ayrıca 3+ GB'lık modeller process kopyalama maliyeti olmadan bellekten paylaşılır.

---

### 7. `layer1_preprocessing/__init__.py`

**Konum:** `/DeepReality/layer1_preprocessing/__init__.py`
**İşlev:** Layer 1 modül açıklaması.

```python
"""
DeepReality — Layer 1: Ön İşlem Pinleri (Pre-Processing Pins)
Görsel/video sisteme girdiğinde, model analizi öncesinde çalışan hafif ama değerli sinyaller.

PIN-A1: EXIF/Metadata Analysis      → AI aracı tespiti, metadata analizi
PIN-A2: C2PA Provenance Analysis     → Dijital imza doğrulama, kaynak tespiti
PIN-A3: ELA (Error Level Analysis)   → Manipülasyon tespiti, sıkıştırma analizi
PIN-A4: Yüz Tespiti & Kırpma        → Yüz tespiti, hizalama, normalizasyon
"""
```

---

### 8. `layer1_preprocessing/pin_a1_metadata.py` — PIN-A1: EXIF/Metadata Analizi

**Konum:** `/DeepReality/layer1_preprocessing/pin_a1_metadata.py`
**İşlev:** Görselin metadata bilgilerini çıkarır ve analiz eder. AI üretim araçlarının bıraktığı metadata izlerini (Stable Diffusion parametreleri, DALL-E imzaları, C2PA binary marker'lar vb.) tespit eder. Kamera, GPS, tarih bilgisi varlığını kontrol ederek görselin gerçek mi yoksa yapay mı üretildiğine dair sinyal üretir.

**Teknoloji:** Pillow (PIL), struct (binary EXIF parsing)

**Analiz adımları:**
1. Metadata Extraction (EXIF, XMP, PNG tEXt, JPEG COM)
2. Signal Analysis (AI imza tespiti, C2PA binary tarama, kamera/GPS/tarih/yazılım analizi, boyut analizi, sıkıştırma oranı)
3. İki katmanlı skorlama: Evidence Floor (kesin kanıt kuralları) + Weighted Sum (ağırlıklı toplam)
4. Verdict belirleme ve Türkçe açıklama üretme

**Sınıf:** `PinA1Metadata(BasePin)` — ~1042 satır

**Ana metodlar:**
- `analyze()`: Ana analiz pipeline
- `_extract_exif()`: Pillow ile EXIF verisi çıkarma (IFD, GPS dahil)
- `_extract_raw_text()`: Binary'den XMP, PNG tEXt, JPEG COM çıkarma
- `_detect_ai_signatures()`: AI aracı imza tespiti
- `_detect_c2pa_binary()`: C2PA/JUMBF binary marker arama
- `_analyze_camera_data()`: Kamera bilgisi analizi
- `_analyze_gps_data()`: GPS koordinat analizi
- `_analyze_dimensions()`: AI tipik boyut kontrolü
- `_analyze_compression_ratio()`: Sıkıştırma oranı anomali tespiti
- `_calculate_score()`: İki katmanlı skorlama (evidence floor + weighted sum)

*(Kaynak kodu ~1042 satır olduğu için burada tam kodu vermek yerine yukarıda tam listesini verdim. Dosya doğrudan okunabilir.)*

---

### 9. `layer1_preprocessing/pin_a2_c2pa.py` — PIN-A2: C2PA Provenance Analizi

**Konum:** `/DeepReality/layer1_preprocessing/pin_a2_c2pa.py`
**İşlev:** C2PA Content Credentials (dijital provenance) verisini resmi `c2pa-python` kütüphanesi ile okur ve doğrular. PIN-A1'den farkı: A1 heuristik binary tarama yapar, A2 resmi kütüphane ile manifest parse + doğrulama yapar.

**Teknoloji:** c2pa-python (>= 0.28.0)

**Analiz adımları:**
1. c2pa.Reader ile dosyayı oku
2. Active manifest'ten bilgi çıkar (creator, tool, timestamp, actions, digitalSourceType, ingredients, validation)
3. Tüm manifest zincirini tara (OpenAI gibi sağlayıcılar çift manifest kullanır)
4. Eksik alanları parent manifest'lerden zenginleştir (chain enrichment)
5. Dinamik skorlama (baz skor 0.40 + artırıcı/azaltıcı sinyaller)

**Sınıf:** `PinA2C2pa(BasePin)` — ~1162 satır

**Önemli tasarım kararı:** OpenAI gibi bazı sağlayıcılar çift manifest zinciri kullanır — active manifest'te sadece "c2pa.opened" olabilir, gerçek AI bilgisi parent manifest'tedir. Bu yüzden tüm zincir taranır.

**Skorlama:**
- Baz: C2PA var → 0.40
- trainedAlgorithmicMedia → +0.35
- Bilinen AI issuer → +0.15
- Bilinen AI araç → +0.15
- AI eylemleri → +0.10
- Kamera eylemi (captured) → -0.30
- İmza hatası → -0.10

---

### 10. `layer1_preprocessing/pin_a3_ela.py` — PIN-A3: ELA (Error Level Analysis)

**Konum:** `/DeepReality/layer1_preprocessing/pin_a3_ela.py`
**İşlev:** JPEG sıkıştırma seviyesi farklarını analiz ederek manipülasyon bölgelerini ve AI üretim izlerini tespit eder.

**Teknoloji:** OpenCV, PIL (Pillow), NumPy

**Algoritma:**
1. Görseli JPEG Q=90 olarak bellekte yeniden kaydet
2. Orijinal ile yeniden kaydedilmiş versiyonu piksel piksel karşılaştır
3. Fark haritası (ELA map) üzerinde analiz:
   - Global istatistikler (mean, std, max, skewness, energy, percentiles)
   - Bölgesel grid analizi (8x8 grid)
   - Uniformity skoru (bölgesel ortalamaların std'si)
   - Hotspot tespiti (MAD tabanlı robust tespit)
   - Coldspot tespiti (ağır sıkıştırılmış yapıştırma)
4. ELA heatmap kaydet (JET colormap)
5. Format-bilinçli skorlama

**Sınıf:** `PinA3Ela(BasePin)` — ~886 satır

**Kritik tasarım kararı:**
- ELA uniformity tek başına AI/gerçek AYIRT EDEMEZ (iPhone hesaplamalı fotoğrafçılık gerçek foto bile uniform ELA verir)
- Uniformity → zayıf destekleyici sinyal (max 0.25)
- Hotspot → güçlü manipülasyon sinyali (max 0.85)
- Format cezası YOK (AI ve gerçek foto her formatta olabilir)

**Hotspot/Coldspot tespiti:** MAD (Median Absolute Deviation) tabanlı robust outlier detection. Coldspot'lar için ek minimum absolute fark kontrolü (doğal varyasyonu filtrelemek için).

---

### 11. `layer1_preprocessing/pin_a4_face.py` — PIN-A4: Yüz Tespiti & Kırpma

**Konum:** `/DeepReality/layer1_preprocessing/pin_a4_face.py`
**İşlev:** Görseldeki yüzleri tespit eder, kırpar, hizalar ve normalize eder. Layer 2'deki deepfake detection modelleri için hazır yüz görselleri üretir. Risk skoru ÜRETMEZ (score = 0.0) — preprocessing PIN'i olarak sadece veri hazırlar.

**Teknoloji:** MediaPipe Face Detection (BlazeFace), OpenCV, NumPy

**İki API desteği:**
- Yeni MediaPipe (≥0.10.8): `mp.tasks.vision.FaceDetector` — model dosyası gerektirir
- Eski MediaPipe (<0.10.8): `mp.solutions.face_detection` — dahili model

**Sınıf:** `PinA4Face(BasePin)` — ~636 satır

**Her yüz için çıktılar:**
- Bounding box (konum + güvenilirlik)
- 6 landmark (sağ göz, sol göz, burun ucu, ağız merkezi, sağ kulak, sol kulak)
- Hizalama bilgisi (roll açısı, önden mi/yandan mı)
- Kalite metrikleri (netlik/sharpness, bulanıklık, parlaklık, kontrast, çözünürlük kategorisi)
- Normalize edilmiş yüz görseli (224x224 PNG) — CNN/ViT standart input boyutu

**Model yönetimi:** Model dosyası (`blaze_face_short_range.tflite`) `models/` klasöründe aranır, yoksa otomatik indirilir.

---

### 12. `layer2_detection_core/__init__.py`

**Konum:** `/DeepReality/layer2_detection_core/__init__.py`
**İşlev:** Layer 2 modül açıklaması. Dört PIN modülünü listeler.

```python
"""
DeepReality — Layer 2: Detection Core Pins
Ana tespit motoru — farklı derin öğrenme mimarileri paralel çalışır.

PIN-B1: CLIP ViT-L/14 (Frozen + LN-tune)  → Generalist deepfake/AI detection
PIN-B2: SigLIP2-base-512 (Fine-tuned)     → High-resolution micro-anomaly detection
PIN-B3: Frekans Analizi (DCT/DWT + CNN)   → Frequency domain artifact detection
PIN-B4: Independent Core (3-sınıf)        → AI vs Deepfake vs Real classification
"""
```

---

### 13. `layer2_detection_core/pin_b1_clip.py` — PIN-B1: CLIP ViT-L/14 Deepfake Detection

**Konum:** `/DeepReality/layer2_detection_core/pin_b1_clip.py`
**İşlev:** OpenAI CLIP ViT-L/14 backbone ile deepfake/AI üretimi görsel tespiti. Backbone dondurulmuş (frozen), sadece LayerNorm parametreleri fine-tune edilmiş — bu teknik küçük veri setlerinde overfitting'i önler ve genelleme yeteneği sağlar.

**Teknoloji:** PyTorch, HuggingFace Transformers (CLIPModel, CLIPProcessor)

**Model bilgisi:**
- Mimari: CLIP ViT-L/14 (427M toplam parametre, 365K eğitilebilir)
- Eğitim: OpenDeepfake-Preview (20K görsel), 10 epoch, LN-tune only
- Performans: Test Acc %99.77, F1 %99.77, ROC-AUC %99.97
- Input: 224x224 piksel
- Model dosyası: `pin_b1_clip_ln_tune_final.pt` (~1.6 GB)

**Sınıflar:**
- `PINB1Model(nn.Module)` — CLIP backbone + sınıflandırma başlığı (LayerNorm → Dropout → Linear 1024→256 → GELU → Dropout → Linear 256→2)
- `PinB1Clip(BasePin)` — Ana PIN sınıfı, BasePin'den miras alır

**Çıktı alanları:**
- `clip_prob`: P(fake) olasılığı (0.0-1.0)
- `clip_verdict`: "FAKE" veya "REAL"
- `clip_confidence`: Modelin tahminine güveni
- `clip_features`: 1024 boyutlu feature vektörü (ensemble/LLM katmanları için)

```python
"""
DeepReality — PIN-B1: CLIP ViT-L/14 (Frozen + LN-tune)
═══════════════════════════════════════════════════════

Generalist deepfake/AI-generated image detection using OpenAI CLIP
backbone with frozen weights and only LayerNorm parameters fine-tuned.

Input:  Image file path (any format supported by PIL)
Output: Standard PIN JSON with:
    - clip_prob:     P(fake) probability (0.0 - 1.0)
    - clip_verdict:  "FAKE" or "REAL"
    - clip_confidence: Model confidence in its prediction
    - clip_features: 1024-dim feature vector (for ensemble/LLM layers)

Model: pin_b1_clip_ln_tune_final.pt (~1.6 GB)
Architecture: CLIP ViT-L/14 (427M params, 365K trainable)
Training: OpenDeepfake-Preview (20K images), 10 epochs, LN-tune only
Performance: Test Acc 99.77%, F1 99.77%, ROC-AUC 99.97%
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

from core.base_pin import BasePin
from config.settings import CLIP_CONFIG

# ---------------------------------------------------------------------------
# Lazy imports — torch and transformers are heavy, only load when needed
# ---------------------------------------------------------------------------
_model_instance = None
_processor_instance = None


class PINB1Model(nn.Module):
    """
    CLIP ViT-L/14 with frozen backbone + LayerNorm tuning + classification head.
    This is the same architecture used during training — required to load weights.
    """

    def __init__(self, clip_model_name: str, num_labels: int = 2):
        super().__init__()
        from transformers import CLIPModel

        self.clip = CLIPModel.from_pretrained(clip_model_name)
        hidden_size = self.clip.config.vision_config.hidden_size  # 1024

        # Freeze all parameters
        for param in self.clip.parameters():
            param.requires_grad = False

        # Unfreeze LayerNorm parameters in vision encoder
        for name, param in self.clip.vision_model.named_parameters():
            if "layer_norm" in name or "layernorm" in name or "LayerNorm" in name:
                param.requires_grad = True

        # Classification head (must match training architecture exactly)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(self, pixel_values):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        cls_embedding = vision_outputs.pooler_output  # (batch, 1024)
        logits = self.classifier(cls_embedding)        # (batch, 2)
        return logits, cls_embedding


def _get_device() -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    return torch.device("cpu")


def _load_model():
    """Load model and processor once, cache globally."""
    global _model_instance, _processor_instance

    if _model_instance is not None:
        return _model_instance, _processor_instance

    from transformers import CLIPProcessor

    model_path = Path(CLIP_CONFIG["model_path"])
    clip_model_name = CLIP_CONFIG["clip_model_name"]
    device = _get_device()

    if not model_path.exists():
        raise FileNotFoundError(
            f"PIN-B1 model not found: {model_path}\n"
            f"Please place 'pin_b1_clip_ln_tune_final.pt' in the models/ directory."
        )

    # Build model architecture
    model = PINB1Model(clip_model_name, num_labels=2)

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load processor
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    _model_instance = model
    _processor_instance = processor

    return model, processor


class PinB1Clip(BasePin):
    """
    PIN-B1: CLIP ViT-L/14 Frozen + LN-tune Deepfake Detector.
    Inherits from BasePin for standard JSON output format.
    """

    def __init__(self):
        super().__init__(
            pin_id="PIN-B1",
            pin_name="CLIP ViT-L/14 Deepfake Detection",
            layer=2
        )
        self._model = None
        self._processor = None
        self._device = None

    def _ensure_model_loaded(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            self._model, self._processor = _load_model()
            self._device = _get_device()

    def analyze(self, file_path: str) -> dict:
        """
        Analyze a single image for deepfake/AI-generation indicators.

        Args:
            file_path: Path to the image file.

        Returns:
            dict with keys: results, score, verdict, details
        """
        self._ensure_model_loaded()

        # Load and preprocess image
        image = Image.open(file_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)

        # Inference
        with torch.no_grad():
            logits, features = self._model(pixel_values)
            probs = torch.softmax(logits, dim=1)

        # Extract results
        fake_prob = probs[0, 0].item()   # P(fake)
        real_prob = probs[0, 1].item()   # P(real)
        pred_label = logits.argmax(dim=1).item()
        confidence = probs[0, pred_label].item()
        verdict_label = "FAKE" if pred_label == 0 else "REAL"

        # Feature vector for downstream layers (ensemble, LLM)
        feature_vector = features[0].cpu().numpy().tolist()

        # Score: 0.0 = clean/real, 1.0 = fake/AI-generated
        # We use fake_prob directly as the score
        score = fake_prob

        # Determine verdict based on thresholds
        thresholds = CLIP_CONFIG["thresholds"]
        if score >= thresholds["high_risk"]:
            verdict = "high_risk"
        elif score >= thresholds["medium_risk"]:
            verdict = "medium_risk"
        else:
            verdict = "low_risk"

        # Build details string
        details = (
            f"CLIP ViT-L/14 analizi: {verdict_label} "
            f"(fake olasılığı: {fake_prob:.4f}, "
            f"güven: {confidence:.4f})"
        )

        results = {
            "clip_prob": round(fake_prob, 6),
            "clip_real_prob": round(real_prob, 6),
            "clip_verdict": verdict_label,
            "clip_confidence": round(confidence, 6),
            "clip_features": feature_vector,
            "model_info": {
                "architecture": "CLIP ViT-L/14 (Frozen + LN-tune)",
                "trainable_params": 365314,
                "total_params": 427881475,
                "training_dataset": "OpenDeepfake-Preview",
                "input_resolution": "224x224",
            }
        }

        return {
            "results": results,
            "score": score,
            "verdict": verdict,
            "details": details,
        }
```

---

### 14. `layer2_detection_core/pin_b2_siglip2.py` — PIN-B2: SigLIP2 Deepfake Detection

**Konum:** `/DeepReality/layer2_detection_core/pin_b2_siglip2.py`
**İşlev:** Google SigLIP2-base-patch16-512 ile yüksek çözünürlüklü deepfake/AI üretimi tespiti. Vision encoder tamamen fine-tune edilmiş. PIN-B1'den temel farkı: 512x512 input ile mikro-anomalileri yakalar.

**Teknoloji:** PyTorch, HuggingFace Transformers (AutoModel, AutoProcessor)

**Model bilgisi:**
- Mimari: SigLIP2-base-patch16-512 (376M toplam parametre, 93.7M eğitilebilir)
- Eğitim: OpenDeepfake-Preview (20K görsel), 8 epoch, full fine-tune
- Performans: Test Acc %99.97, F1 %99.97, ROC-AUC %100.00
- Input: 512x512 piksel
- Model dosyası: `pin_b2_siglip2_finetune_final.pt` (~1.4 GB)

**Sınıflar:**
- `PINB2Model(nn.Module)` — SigLIP2 vision encoder + sınıflandırma başlığı (LayerNorm → Dropout 0.15 → Linear 768→256 → GELU → Dropout 0.1 → Linear 256→2)
- `PinB2Siglip(BasePin)` — Ana PIN sınıfı

**PIN-B1 vs PIN-B2 karşılaştırması:**
- B1: 224x224, frozen backbone, generalist (daha iyi zero-shot)
- B2: 512x512, full fine-tune, precision-focused (mikro-anomalileri yakalar)

**Çıktı alanları:**
- `siglip_prob`: P(fake) olasılığı (0.0-1.0)
- `siglip_verdict`: "FAKE" veya "REAL"
- `siglip_confidence`: Modelin tahminine güveni
- `siglip_features`: 768 boyutlu feature vektörü

```python
"""
DeepReality — PIN-B2: SigLIP2-base-patch16-512 (Fine-tuned)
════════════════════════════════════════════════════════════

High-resolution deepfake/AI-generated image detection using Google
SigLIP2 with full vision encoder fine-tuning.

Input:  Image file path (any format supported by PIL)
Output: Standard PIN JSON with:
    - siglip_prob:       P(fake) probability (0.0 - 1.0)
    - siglip_verdict:    "FAKE" or "REAL"
    - siglip_confidence: Model confidence in its prediction
    - siglip_features:   768-dim feature vector (for ensemble/LLM layers)

Model: pin_b2_siglip2_finetune_final.pt (~1.4 GB)
Architecture: SigLIP2-base-patch16-512 (376M params, 93.7M trainable)
Training: OpenDeepfake-Preview (20K images), 8 epochs, full fine-tune
Performance: Test Acc 99.97%, F1 99.97%, ROC-AUC 100.00%

Key difference from PIN-B1:
    - PIN-B1: 224x224, frozen backbone, generalist (better zero-shot)
    - PIN-B2: 512x512, full fine-tune, precision-focused (catches micro-anomalies)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

from core.base_pin import BasePin
from config.settings import SIGLIP_CONFIG

# ---------------------------------------------------------------------------
# Lazy imports — torch and transformers are heavy, only load when needed
# ---------------------------------------------------------------------------
_model_instance = None
_processor_instance = None


class PINB2Model(nn.Module):
    """
    SigLIP2-base-patch16-512 with full vision encoder fine-tuning
    + classification head.
    This is the same architecture used during training — required to load weights.
    """

    def __init__(self, model_name: str, num_labels: int = 2):
        super().__init__()
        from transformers import AutoModel

        self.siglip = AutoModel.from_pretrained(model_name)
        self.vision_model = self.siglip.vision_model
        hidden_size = self.siglip.config.vision_config.hidden_size  # 768

        # Freeze text model (not needed for classification)
        for param in self.siglip.text_model.parameters():
            param.requires_grad = False

        # Classification head (must match training architecture exactly)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(self, pixel_values):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        cls_embedding = vision_outputs.pooler_output  # (batch, 768)
        logits = self.classifier(cls_embedding)        # (batch, 2)
        return logits, cls_embedding


def _get_device() -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model():
    """Load model and processor once, cache globally."""
    global _model_instance, _processor_instance

    if _model_instance is not None:
        return _model_instance, _processor_instance

    from transformers import AutoProcessor

    model_path = Path(SIGLIP_CONFIG["model_path"])
    model_name = SIGLIP_CONFIG["model_name"]
    device = _get_device()

    if not model_path.exists():
        raise FileNotFoundError(
            f"PIN-B2 model not found: {model_path}\n"
            f"Please place 'pin_b2_siglip2_finetune_final.pt' in the models/ directory."
        )

    # Build model architecture
    model = PINB2Model(model_name, num_labels=2)

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)

    _model_instance = model
    _processor_instance = processor

    return model, processor


class PinB2Siglip(BasePin):
    """
    PIN-B2: SigLIP2-base-patch16-512 Fine-tuned Deepfake Detector.
    Inherits from BasePin for standard JSON output format.
    """

    def __init__(self):
        super().__init__(
            pin_id="PIN-B2",
            pin_name="SigLIP2-base-512 Deepfake Detection",
            layer=2
        )
        self._model = None
        self._processor = None
        self._device = None

    def _ensure_model_loaded(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            self._model, self._processor = _load_model()
            self._device = _get_device()

    def analyze(self, file_path: str) -> dict:
        """
        Analyze a single image for deepfake/AI-generation indicators.

        Args:
            file_path: Path to the image file.

        Returns:
            dict with keys: results, score, verdict, details
        """
        self._ensure_model_loaded()

        # Load and preprocess image
        image = Image.open(file_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)

        # Inference
        with torch.no_grad():
            logits, features = self._model(pixel_values)
            probs = torch.softmax(logits, dim=1)

        # Extract results
        fake_prob = probs[0, 0].item()   # P(fake)
        real_prob = probs[0, 1].item()   # P(real)
        pred_label = logits.argmax(dim=1).item()
        confidence = probs[0, pred_label].item()
        verdict_label = "FAKE" if pred_label == 0 else "REAL"

        # Feature vector for downstream layers
        feature_vector = features[0].cpu().numpy().tolist()

        # Score: 0.0 = clean/real, 1.0 = fake/AI-generated
        score = fake_prob

        # Determine verdict
        thresholds = SIGLIP_CONFIG["thresholds"]
        if score >= thresholds["high_risk"]:
            verdict = "high_risk"
        elif score >= thresholds["medium_risk"]:
            verdict = "medium_risk"
        else:
            verdict = "low_risk"

        details = (
            f"SigLIP2-base-512 analizi: {verdict_label} "
            f"(fake olasılığı: {fake_prob:.4f}, "
            f"güven: {confidence:.4f})"
        )

        results = {
            "siglip_prob": round(fake_prob, 6),
            "siglip_real_prob": round(real_prob, 6),
            "siglip_verdict": verdict_label,
            "siglip_confidence": round(confidence, 6),
            "siglip_features": feature_vector,
            "model_info": {
                "architecture": "SigLIP2-base-patch16-512 (Full Fine-tune)",
                "trainable_params": 93719044,
                "total_params": 376022788,
                "training_dataset": "OpenDeepfake-Preview",
                "input_resolution": "512x512",
            }
        }

        return {
            "results": results,
            "score": score,
            "verdict": verdict,
            "details": details,
        }
```

---

### 15. `layer2_detection_core/pin_b3_freq.py` — PIN-B3: Frekans Analizi (DCT/DWT + CNN)

**Konum:** `/DeepReality/layer2_detection_core/pin_b3_freq.py`
**İşlev:** Frekans domain'inde deepfake/AI üretimi tespiti. Görseli DCT (Discrete Cosine Transform) ve DWT (Discrete Wavelet Transform) ile frekans domain'ine çevirir, ardından özel bir CNN ile sınıflandırır. B1/B2 spatial domain'de çalışırken, B3 frequency domain'de GAN upsampling artefaktları ve diffusion model frekans izlerini yakalar.

**Teknoloji:** PyTorch, scipy (DCT), PyWavelets/pywt (DWT), OpenCV

**Model bilgisi:**
- Mimari: Özel 5-bloklu CNN (FreqCNNBlock x5), 4.8M parametre, sıfırdan eğitilmiş
- Eğitim: OpenDeepfake-Preview (20K görsel), 15 epoch
- Performans: Test Acc %96.50, F1 %96.58, ROC-AUC %99.23
- Input: 4-kanal 224x224 frekans haritası
- Model dosyası: `pin_b3_freq_cnn_final.pt` (~19 MB)

**Frekans dönüşüm pipeline:**
1. Görsel → Gri tonlama → 224x224 resize
2. Kanal 0: DCT log-magnitude spektrumu (tüm görsel)
3. Kanal 1: DWT LH alt-bandı (yatay detay — dikey kenarlar)
4. Kanal 2: DWT HL alt-bandı (dikey detay — yatay kenarlar)
5. Kanal 3: DWT HH alt-bandı (çapraz detay — köşe/doku artefaktları)
6. 4-kanal tensör → CNN → fake/real

**Sınıflar:**
- `FreqCNNBlock(nn.Module)` — Çift konvolüsyon bloğu (Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool2d)
- `PINB3_FreqCNN(nn.Module)` — 5 bloklu CNN (4→32→64→128→256→512) + AdaptiveAvgPool + Classifier
- `PinB3Freq(BasePin)` — Ana PIN sınıfı

**Fonksiyonlar:**
- `image_to_frequency_map()` — Görseli 4-kanallı frekans haritasına dönüştürür

**Çıktı alanları:**
- `freq_prob`: P(fake) olasılığı (0.0-1.0)
- `freq_verdict`: "FAKE" veya "REAL"
- `freq_confidence`: Modelin tahminine güveni
- `freq_features`: 512 boyutlu feature vektörü

```python
"""
DeepReality — PIN-B3: Frequency Analysis (DCT/DWT + CNN)
═══════════════════════════════════════════════════════

Frequency-domain deepfake/AI-generated image detection using DCT
and DWT transforms fed into a lightweight custom CNN.

Input:  Image file path (any format supported by PIL)
Output: Standard PIN JSON with:
    - freq_prob:       P(fake) probability (0.0 - 1.0)
    - freq_verdict:    "FAKE" or "REAL"
    - freq_confidence: Model confidence in its prediction
    - freq_features:   512-dim feature vector (for ensemble/LLM layers)

Model: pin_b3_freq_cnn_final.pt (~18.5 MB)
Architecture: Custom 5-block CNN (4.8M params), trained from scratch
Training: OpenDeepfake-Preview (20K images), 15 epochs
Performance: Test Acc 96.50%, F1 96.58%, ROC-AUC 99.23%

Frequency transform pipeline:
    Image → Grayscale → [DCT log spectrum, DWT-LH, DWT-HL, DWT-HH]
    → 4-channel 224x224 tensor → CNN → fake/real

Key difference from PIN-B1/B2:
    - B1/B2 work in spatial domain (pixels)
    - B3 works in frequency domain (DCT/DWT coefficients)
    - Catches GAN upsampling artifacts and diffusion model frequency traces
      that are invisible to spatial models
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

from core.base_pin import BasePin
from config.settings import FREQ_CONFIG

# ---------------------------------------------------------------------------
# Frequency transform functions
# ---------------------------------------------------------------------------

def image_to_frequency_map(image, target_size=224, wavelet="haar"):
    """
    Convert a PIL image to a 4-channel frequency representation.

    Channel 0: DCT log-magnitude spectrum (full image)
    Channel 1: DWT LH subband (horizontal detail — vertical edges)
    Channel 2: DWT HL subband (vertical detail — horizontal edges)
    Channel 3: DWT HH subband (diagonal detail — corner/texture artifacts)

    Args:
        image: PIL Image (any mode)
        target_size: Output spatial dimensions
        wavelet: Wavelet family for DWT decomposition

    Returns:
        numpy array of shape (4, target_size, target_size), float32, range [0, 1]
    """
    from scipy.fft import dctn
    import pywt
    import cv2

    # Convert to grayscale and resize
    if image.mode != "L":
        gray = image.convert("L")
    else:
        gray = image
    gray = gray.resize((target_size, target_size), Image.BILINEAR)
    gray_np = np.array(gray, dtype=np.float64)

    # --- Channel 0: DCT log-magnitude spectrum ---
    dct_coeffs = dctn(gray_np, type=2, norm="ortho")
    dct_log = np.log1p(np.abs(dct_coeffs))
    dct_max = dct_log.max()
    if dct_max > 0:
        dct_log = dct_log / dct_max
    dct_channel = dct_log.astype(np.float32)

    # --- Channels 1-3: DWT subbands ---
    coeffs = pywt.dwt2(gray_np, wavelet)
    cA, (cH, cV, cD) = coeffs  # cH=LH, cV=HL, cD=HH

    dwt_channels = []
    for subband in [cH, cV, cD]:
        sub_abs = np.abs(subband)
        sub_resized = cv2.resize(sub_abs, (target_size, target_size),
                                 interpolation=cv2.INTER_LINEAR)
        sub_max = sub_resized.max()
        if sub_max > 0:
            sub_resized = sub_resized / sub_max
        dwt_channels.append(sub_resized.astype(np.float32))

    # Stack all 4 channels: (4, H, W)
    freq_map = np.stack([dct_channel] + dwt_channels, axis=0)
    return freq_map


# ---------------------------------------------------------------------------
# Model architecture (must match training exactly)
# ---------------------------------------------------------------------------

class FreqCNNBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class PINB3_FreqCNN(nn.Module):
    """
    Lightweight CNN for frequency-domain deepfake detection.
    Input: 4-channel frequency map (DCT + DWT subbands), 224x224
    Output: Binary classification logits + 512-dim feature vector
    """

    def __init__(self, in_channels=4, num_labels=2):
        super().__init__()

        self.features = nn.Sequential(
            FreqCNNBlock(in_channels, 32),    # 224 → 112
            FreqCNNBlock(32, 64),              # 112 → 56
            FreqCNNBlock(64, 128),             # 56 → 28
            FreqCNNBlock(128, 256),            # 28 → 14
            FreqCNNBlock(256, 512),            # 14 → 7
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

    def forward(self, x):
        features = self.features(x)
        pooled = self.pool(features)
        flat = pooled.view(pooled.size(0), -1)
        logits = self.classifier(flat)
        return logits, flat


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_model_instance = None


def _get_device() -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model():
    """Load model once, cache globally."""
    global _model_instance

    if _model_instance is not None:
        return _model_instance

    model_path = Path(FREQ_CONFIG["model_path"])
    device = _get_device()

    if not model_path.exists():
        raise FileNotFoundError(
            f"PIN-B3 model not found: {model_path}\n"
            f"Please place 'pin_b3_freq_cnn_final.pt' in the models/ directory."
        )

    # Build model architecture
    model = PINB3_FreqCNN(
        in_channels=FREQ_CONFIG["num_channels"],
        num_labels=FREQ_CONFIG["num_labels"]
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _model_instance = model
    return model


# ---------------------------------------------------------------------------
# PIN class
# ---------------------------------------------------------------------------

class PinB3Freq(BasePin):
    """
    PIN-B3: Frequency Analysis (DCT/DWT + CNN) Deepfake Detector.
    Inherits from BasePin for standard JSON output format.
    """

    def __init__(self):
        super().__init__(
            pin_id="PIN-B3",
            pin_name="Frequency Analysis (DCT/DWT + CNN)",
            layer=2
        )
        self._model = None
        self._device = None

    def _ensure_model_loaded(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            self._model = _load_model()
            self._device = _get_device()

    def analyze(self, file_path: str) -> dict:
        """
        Analyze a single image for frequency-domain deepfake indicators.

        Args:
            file_path: Path to the image file.

        Returns:
            dict with keys: results, score, verdict, details
        """
        self._ensure_model_loaded()

        # Load and convert to frequency map
        image = Image.open(file_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        freq_map = image_to_frequency_map(
            image,
            target_size=FREQ_CONFIG["freq_image_size"],
            wavelet=FREQ_CONFIG["dwt_wavelet"]
        )
        freq_tensor = torch.from_numpy(freq_map).unsqueeze(0).to(self._device)

        # Inference
        with torch.no_grad():
            logits, features = self._model(freq_tensor)
            probs = torch.softmax(logits, dim=1)

        # Extract results
        fake_prob = probs[0, 0].item()
        real_prob = probs[0, 1].item()
        pred_label = logits.argmax(dim=1).item()
        confidence = probs[0, pred_label].item()
        verdict_label = "FAKE" if pred_label == 0 else "REAL"

        # Feature vector for downstream layers
        feature_vector = features[0].cpu().numpy().tolist()

        # Score: 0.0 = clean/real, 1.0 = fake/AI-generated
        score = fake_prob

        # Determine verdict
        thresholds = FREQ_CONFIG["thresholds"]
        if score >= thresholds["high_risk"]:
            verdict = "high_risk"
        elif score >= thresholds["medium_risk"]:
            verdict = "medium_risk"
        else:
            verdict = "low_risk"

        details = (
            f"Frekans analizi (DCT/DWT): {verdict_label} "
            f"(fake olasılığı: {fake_prob:.4f}, "
            f"güven: {confidence:.4f})"
        )

        results = {
            "freq_prob": round(fake_prob, 6),
            "freq_real_prob": round(real_prob, 6),
            "freq_verdict": verdict_label,
            "freq_confidence": round(confidence, 6),
            "freq_features": feature_vector,
            "model_info": {
                "architecture": "Custom FreqCNN (DCT+DWT, 5-block)",
                "trainable_params": 4846338,
                "total_params": 4846338,
                "training_dataset": "OpenDeepfake-Preview",
                "input_resolution": "224x224 (4-channel frequency map)",
                "channels": "DCT log spectrum + DWT-LH + DWT-HL + DWT-HH",
            }
        }

        return {
            "results": results,
            "score": score,
            "verdict": verdict,
            "details": details,
        }
```

---

### 16. `layer2_detection_core/pin_b4_IndependentCore.py` — PIN-B4: Independent Core (3-Sınıf)

**Konum:** `/DeepReality/layer2_detection_core/pin_b4_IndependentCore.py`
**İşlev:** Sistemdeki TEK 3-sınıflı detektör. Görseli AI üretimi (tamamen sentetik), Deepfake (gerçek içeriğin manipülasyonu) veya Real (orijinal fotoğraf) olarak sınıflandırır. Diğer Layer 2 PIN'leri binary (fake/real) ayrım yaparken, B4 AI ile Deepfake'i de birbirinden ayırır.

**Teknoloji:** PyTorch, HuggingFace Transformers (SiglipForImageClassification, AutoImageProcessor)

**Model bilgisi:**
- Mimari: google/siglip2-base-patch16-224 + SiglipForImageClassification
- Kaynak: prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2 (HuggingFace)
- Eğitim: AI-vs-Deepfake-vs-Real dataset (4000 test, %99.05 accuracy)
- Input: 224x224 piksel
- Model dizini: `pin_b4_ai_deepfake_real/` (~354 MB toplam)

**Skor hesaplama (ensemble uyumluluğu için):**
- `fake_score = ai_prob + deepfake_prob` (0.0 = kesinlikle gerçek, 1.0 = kesinlikle fake/AI)

**Sınıf:** `PinB4IndependentCore(BasePin)` — ~216 satır

**Çıktı alanları:**
- `ai_prob`: P(AI-generated) olasılığı
- `deepfake_prob`: P(Deepfake) olasılığı
- `real_prob`: P(Real) olasılığı
- `predicted_class`: "AI", "Deepfake" veya "Real"
- `fake_score`: ai_prob + deepfake_prob (ensemble için birleşik skor)
- `confidence`: En yüksek olasılığa sahip sınıfın güveni

```python
"""
DeepReality — PIN-B4: Independent Core (AI vs Deepfake vs Real)
═══════════════════════════════════════════════════════════════

External pretrained 3-class image classifier for distinguishing:
    - AI-generated images (fully synthetic, e.g. DALL-E, Midjourney, Stable Diffusion)
    - Deepfake images (manipulated real content, e.g. face swap, reenactment)
    - Real images (authentic, unaltered photographs)

This is the ONLY 3-class detector in the system. All other Layer 2 pins
are binary (fake/real). The 3-class distinction provides unique information
for the ensemble layer — specifically whether the image is AI-generated
from scratch or a manipulation of existing content.

Source: prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2 (HuggingFace)
Architecture: google/siglip2-base-patch16-224 + SiglipForImageClassification
Training: AI-vs-Deepfake-vs-Real dataset (4000 test, 99.05% accuracy)
License: Apache 2.0
Model runs 100% LOCAL — no internet required after initial download.

Input:  Image file path
Output: Standard PIN JSON with:
    - ai_prob:         P(AI-generated) probability
    - deepfake_prob:   P(Deepfake manipulation) probability
    - real_prob:       P(Real authentic) probability
    - predicted_class: "AI", "Deepfake", or "Real"
    - fake_score:      ai_prob + deepfake_prob (combined for ensemble, 0-1)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from core.base_pin import BasePin
from config.settings import INDEPENDENT_CORE_CONFIG

# ---------------------------------------------------------------------------
# Model loading (lazy, cached globally)
# ---------------------------------------------------------------------------
_model_instance = None
_processor_instance = None


def _get_device() -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model():
    """Load SiglipForImageClassification model from local directory."""
    global _model_instance, _processor_instance

    if _model_instance is not None:
        return _model_instance, _processor_instance

    from transformers import AutoImageProcessor, SiglipForImageClassification

    model_dir = Path(INDEPENDENT_CORE_CONFIG["model_dir"])
    device = _get_device()

    if not model_dir.exists():
        raise FileNotFoundError(
            f"PIN-B4 model directory not found: {model_dir}\n"
            f"Download the model files:\n"
            f"  mkdir -p {model_dir}\n"
            f"  cd {model_dir}\n"
            f"  curl -L -o config.json 'https://huggingface.co/prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2/resolve/main/config.json'\n"
            f"  curl -L -o preprocessor_config.json 'https://huggingface.co/prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2/resolve/main/preprocessor_config.json'\n"
            f"  curl -L -o model.safetensors 'https://huggingface.co/prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2/resolve/main/model.safetensors'"
        )

    required_files = ["config.json", "preprocessor_config.json", "model.safetensors"]
    for fname in required_files:
        if not (model_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing file: {model_dir / fname}\n"
                f"Download it from HuggingFace: prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2"
            )

    # Load model and processor from local directory (NO internet needed)
    model = SiglipForImageClassification.from_pretrained(
        str(model_dir), local_files_only=True
    )
    processor = AutoImageProcessor.from_pretrained(
        str(model_dir), local_files_only=True
    )

    model.to(device)
    model.eval()

    _model_instance = model
    _processor_instance = processor

    return model, processor


# ---------------------------------------------------------------------------
# PIN class
# ---------------------------------------------------------------------------

class PinB4IndependentCore(BasePin):
    """
    PIN-B4: Independent Core — 3-Class AI/Deepfake/Real Detector.
    The only multi-class detector in Layer 2.
    Inherits from BasePin for standard JSON output format.
    """

    def __init__(self):
        super().__init__(
            pin_id="PIN-B4",
            pin_name="Independent Core (AI vs Deepfake vs Real)",
            layer=2
        )
        self._model = None
        self._processor = None
        self._device = None

    def _ensure_model_loaded(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            self._model, self._processor = _load_model()
            self._device = _get_device()

    def analyze(self, file_path: str) -> dict:
        """
        Analyze a single image and classify as AI / Deepfake / Real.

        Score calculation for ensemble compatibility:
            fake_score = ai_prob + deepfake_prob
            (0.0 = definitely real, 1.0 = definitely fake/AI)

        Args:
            file_path: Path to the image file.

        Returns:
            dict with keys: results, score, verdict, details
        """
        self._ensure_model_loaded()

        # Load and preprocess image
        image = Image.open(file_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)

        # Inference
        with torch.no_grad():
            outputs = self._model(pixel_values)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        # Extract per-class probabilities
        # Class 0: AI, Class 1: Deepfake, Class 2: Real
        label_map = INDEPENDENT_CORE_CONFIG["label_map"]
        ai_prob = probs[0]
        deepfake_prob = probs[1]
        real_prob = probs[2]

        # Determine predicted class
        pred_idx = logits.argmax(dim=1).item()
        predicted_class = label_map[pred_idx]
        confidence = probs[pred_idx]

        # Combined fake score for ensemble (AI + Deepfake = not real)
        fake_score = ai_prob + deepfake_prob

        # Score: 0.0 = real, 1.0 = fake/AI
        score = fake_score

        # Determine verdict
        thresholds = INDEPENDENT_CORE_CONFIG["thresholds"]
        if score >= thresholds["high_risk"]:
            verdict = "high_risk"
        elif score >= thresholds["medium_risk"]:
            verdict = "medium_risk"
        else:
            verdict = "low_risk"

        details = (
            f"Independent Core: {predicted_class} "
            f"(AI={ai_prob:.4f}, Deepfake={deepfake_prob:.4f}, "
            f"Real={real_prob:.4f}, güven={confidence:.4f})"
        )

        results = {
            "ai_prob": round(ai_prob, 6),
            "deepfake_prob": round(deepfake_prob, 6),
            "real_prob": round(real_prob, 6),
            "predicted_class": predicted_class,
            "fake_score": round(fake_score, 6),
            "confidence": round(confidence, 6),
            "model_info": {
                "architecture": "SigLIP2-base-patch16-224 (SiglipForImageClassification)",
                "source": "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2",
                "num_classes": 3,
                "classes": ["AI", "Deepfake", "Real"],
                "training_accuracy": "99.05%",
                "input_resolution": "224x224",
                "license": "Apache 2.0",
            }
        }

        return {
            "results": results,
            "score": score,
            "verdict": verdict,
            "details": details,
        }
```

---

### 16b. `layer4_xai/pin_d1_gradcam.py` — PIN-D1: Grad-CAM Heatmap (YENİ)

**Konum:** `/DeepReality/layer4_xai/pin_d1_gradcam.py`
**İşlev:** Katman 2 modellerinin (B1 CLIP, B2 SigLIP2, B3 FreqCNN) "fake" kararını verirken görüntünün HANGİ bölgelerine baktığını Grad-CAM (Selvaraju et al., ICCV 2017) ile ısı haritası olarak görselleştirir. Harici paket KULLANMAZ — PyTorch forward hook + `torch.autograd.grad` ile implemente edilmiştir (transformers 5.x uyumlu). Gradyan tam backward yerine yalnızca hedef katmana kadar hesaplanır (hedef son encoder bloğu olduğundan ~20x hızlıdır).

**Teknik detaylar:**
- ViT modelleri için hedef katman: son encoder bloğunun `layer_norm1`'i (`_find_vit_target_layer` ile isimden otomatik bulunur)
- Token aktivasyonları patch ızgarasına çevrilir: B1 CLIP → 16×16 (CLS token'ı otomatik tespit edilip atılır), B2 SigLIP2 → 32×32 (CLS yok)
- B3 FreqCNN için klasik CNN Grad-CAM (`features[-1]`, 7×7). Frekans domain'inde çalıştığından uzamsal karşılığı YAKLAŞIKTIR — birleşik haritada %20 ağırlıkla kullanılır (XAI_CONFIG.combine_weights)
- Modeller Katman 2 pinlerinin global cache'inden (`_load_model()`) paylaşılır — ek bellek maliyeti yok. Bu yüzden pipeline'da B1+B2+B3'ten sonra çalışır
- CLIP-SigLIP odak uyumu IoU ile ölçülür (`model_agreement_iou`)

**Çıktılar:** `{dosya}_XAI_D1_clip.png`, `_siglip.png`, `_freq.png`, `_combined.png` (overlay'ler) + JSON'da `focus_regions` (bbox + aktivasyon). Ham CAM matrisleri `cam_cache` instance attribute'unda PIN-D2 için saklanır.

**Skor:** XAI bilgilendirme katmanıdır — skor üretmez (0.0, verdict `informational`).

---

### 16c. `layer4_xai/pin_d2_anomaly.py` — PIN-D2: Anomaly Localization (YENİ)

**Konum:** `/DeepReality/layer4_xai/pin_d2_anomaly.py`
**İşlev:** İki BAĞIMSIZ kanıt kaynağını birleştirerek manipüle edilmiş bölgeleri tek haritada işaretler: (1) PIN-A3 ELA anomali bölgeleri (sıkıştırma-fiziği kanıtı), (2) PIN-D1 birleşik Grad-CAM haritası (model-karar kanıtı).

**Füzyon mantığı:**
- Komşu ELA grid hücreleri tek bölgede birleştirilir (`_merge_adjacent_boxes`)
- Bir ELA bölgesi içindeki ortalama CAM aktivasyonu eşiği (0.50) aşarsa bölge "DOĞRULANMIŞ" (fused) sayılır — iki bağımsız yöntem aynı bölgeyi işaret ediyor demektir
- Sadece CAM'in işaretlediği bölgeler (üst %15 quantile maskesi) ayrıca raporlanır

**Skor semantiği (ÖNEMLİ):** Skor fake olasılığı DEĞİL, "lokalize manipülasyon kanıtı" gücüdür (ensemble için destekleyici sinyal): füzyon-doğrulamalı bölge → 0.80, sadece yüksek ELA → 0.55, sadece düşük/orta ELA → 0.35, sadece yoğun CAM odağı → 0.30, kanıt yok → 0.05 (XAI_CONFIG.evidence_scores).

**Çıktı:** `{dosya}_XAI_D2_anomaly.png` — KIRMIZI=ELA hotspot, MAVİ=ELA coldspot, SARI=CAM odağı, TURUNCU (kalın)=füzyon-doğrulamalı. JSON'da `marked_regions` (bbox + source + strength).

---

### 17. `models/blaze_face_short_range.tflite`

**Konum:** `/DeepReality/models/blaze_face_short_range.tflite`
**İşlev:** MediaPipe Face Detection için TFLite model dosyası (~224 KB). BlazeFace kısa mesafe modeli. PIN-A4 tarafından kullanılır.

*(Binary dosya — kaynak kod değil)*

---

### 18. `models/pin_b1_clip_ln_tune_final.pt` — PIN-B1 Model Ağırlıkları

**Konum:** `/DeepReality/models/pin_b1_clip_ln_tune_final.pt`
**İşlev:** CLIP ViT-L/14 + LN-tune sınıflandırma başlığı eğitilmiş PyTorch checkpoint (~1.6 GB). `model_state_dict` anahtarı altında model ağırlıklarını içerir. PIN-B1 tarafından kullanılır.

*(Binary dosya — kaynak kod değil)*

---

### 19. `models/pin_b2_siglip2_finetune_final.pt` — PIN-B2 Model Ağırlıkları

**Konum:** `/DeepReality/models/pin_b2_siglip2_finetune_final.pt`
**İşlev:** SigLIP2-base-patch16-512 full fine-tune PyTorch checkpoint (~1.4 GB). `model_state_dict` anahtarı altında model ağırlıklarını içerir. PIN-B2 tarafından kullanılır.

*(Binary dosya — kaynak kod değil)*

---

### 20. `models/pin_b3_freq_cnn_final.pt` — PIN-B3 Model Ağırlıkları

**Konum:** `/DeepReality/models/pin_b3_freq_cnn_final.pt`
**İşlev:** Özel FreqCNN (DCT+DWT, 5-blok) eğitilmiş PyTorch checkpoint (~19 MB). `model_state_dict` anahtarı altında model ağırlıklarını içerir. PIN-B3 tarafından kullanılır.

*(Binary dosya — kaynak kod değil)*

---

### 21. `models/pin_b4_ai_deepfake_real/` — PIN-B4 HuggingFace Model Dizini

**Konum:** `/DeepReality/models/pin_b4_ai_deepfake_real/`
**İşlev:** HuggingFace pretrained model dizini. `SiglipForImageClassification` ve `AutoImageProcessor` ile `local_files_only=True` modunda yüklenir. PIN-B4 tarafından kullanılır.

**İçerdiği dosyalar:**

#### `config.json` — Model Konfigürasyonu
```json
{
  "_name_or_path": "google/siglip2-base-patch16-224",
  "architectures": ["SiglipForImageClassification"],
  "id2label": {"0": "AI", "1": "Deepfake", "2": "Real"},
  "label2id": {"AI": 0, "Deepfake": 1, "Real": 2},
  "model_type": "siglip",
  "problem_type": "single_label_classification",
  "torch_dtype": "float32",
  "transformers_version": "4.50.0.dev0"
}
```

#### `preprocessor_config.json` — Görsel Ön İşleme
```json
{
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [0.5, 0.5, 0.5],
  "image_processor_type": "SiglipImageProcessor",
  "image_std": [0.5, 0.5, 0.5],
  "size": {"height": 224, "width": 224}
}
```

#### `model.safetensors` — Model Ağırlıkları (~354 MB)
*(Binary dosya — safetensors formatında model ağırlıkları)*

---

### 22. `input/` — Giriş Görselleri

**Konum:** `/DeepReality/input/`
**İçerik:**
- `test3.png` — ~6.4 MB test görseli
- `test5.png` — 2048x2048 piksel, ~8.3 MB, RGBA modunda PNG. Analiz sonucuna göre Google AI tarafından üretilmiş (C2PA imzalı, trainedAlgorithmicMedia).
- `test8.png` — 1024x1024 piksel, ~1.1 MB, RGBA modunda PNG. C2PA imzası yok, metadata boş, ancak bilinen AI boyutu (1024x1024).
- `test9.png` — ~1.9 MB test görseli
- `test11.png` — ~5.9 MB test görseli

---

### 23. `outputs/` — Çıktı Dosyaları

Her girdi görseli için **8 adet JSON** (Layer 1: PIN-A1~A4 + Layer 2: PIN-B1~B4) ve varsa ek görseller (ELA heatmap, yüz kırpmaları) üretilir.

**Çıktı dosya listesi (her görsel için):**
| Dosya | PIN | Layer | İçerik |
|-------|-----|-------|--------|
| `{dosya}_PIN-A1.json` | PIN-A1 | 1 | EXIF/Metadata analiz sonucu |
| `{dosya}_PIN-A2.json` | PIN-A2 | 1 | C2PA provenance analiz sonucu |
| `{dosya}_PIN-A3.json` | PIN-A3 | 1 | ELA analiz sonucu |
| `{dosya}_PIN-A4.json` | PIN-A4 | 1 | Yüz tespit sonucu |
| `{dosya}_PIN-B1.json` | PIN-B1 | 2 | CLIP ViT-L/14 deepfake tespit sonucu (~27 KB, 1024-dim feature dahil) |
| `{dosya}_PIN-B2.json` | PIN-B2 | 2 | SigLIP2 deepfake tespit sonucu (~21 KB, 768-dim feature dahil) |
| `{dosya}_PIN-B3.json` | PIN-B3 | 2 | Frekans analizi sonucu (~14 KB, 512-dim feature dahil) |
| `{dosya}_PIN-B4.json` | PIN-B4 | 2 | Independent Core 3-sınıf sonucu (~1.1 KB) |
| `{dosya}_ELA_heatmap.png` | PIN-A3 | 1 | ELA ısı haritası görseli |
| `{dosya}_face_0.png` | PIN-A4 | 1 | Kırpılmış yüz görseli (224x224) |

**Örnek Layer 1 çıktıları (test5.png için):**

#### `outputs/test5_PIN-A1.json` — Metadata Analiz Sonucu
- **Skor:** 0.906 (YUKSEK RISK)
- **Bulgu:** 7 adet C2PA binary marker tespit edildi (jumb, c2pa, caBX, c2pa.actions, c2pa.created, c2pa.hash.data, trainedAlgorithmicMedia)
- **AI Kaynağı:** google_ai (C2PA issuer)
- **EXIF:** 0 alan (metadata tamamen boş)
- **Boyut:** 2048x2048 (bilinen AI çıktı boyutu)
- **Evidence floor uygulandı:** C2PA imza güvenilirliği %99 → baz 0.792 + destek 0.114 = 0.906

#### `outputs/test5_PIN-A2.json` — C2PA Analiz Sonucu
- **Skor:** 0.0 (VERI YOK)
- **Bulgu:** c2pa-python kütüphanesi manifest okuyamadı (sertifika tarih hatası)
- **Not:** PIN-A1'in binary taraması ile C2PA marker'lar bulundu ama resmi kütüphane doğrulayamadı

#### `outputs/test5_PIN-A3.json` — ELA Analiz Sonucu
- **Skor:** 0.2 (DUSUK RISK)
- **Format:** PNG (lossless — uniformity güvenilirliği düşük)
- **Uniformity:** uniform (skor=5.25) — zayıf sinyal
- **Anomali:** 0 bölge tespit edildi
- **Dominant sinyal:** uniformity_weak

#### `outputs/test5_PIN-A4.json` — Yüz Tespit Sonucu
- **Skor:** 0.0 (preprocessing PIN)
- **Yüz sayısı:** 1
- **Güvenilirlik:** 0.9117
- **Hizalama:** önden (roll açısı: -4.0°)
- **Kalite:** yüksek çözünürlük, netlik=240.0
- **Çıktı:** test5_face_0.png (224x224)

**Örnek Layer 2 çıktıları (test5.png için):**

#### `outputs/test5_PIN-B1.json` — CLIP Deepfake Tespit Sonucu
- **Skor:** 0.4158 (ORTA RISK)
- **Karar:** REAL
- **Fake olasılığı:** 0.4158
- **Güven:** 0.5842
- **Feature:** 1024 boyutlu vektör (ensemble/LLM katmanları için)
- **Not:** CLIP generalist model, bu görselde kararsız kalmış (orta risk bölgesinde)

#### `outputs/test5_PIN-B2.json` — SigLIP2 Deepfake Tespit Sonucu
- **Skor:** 0.0558 (DUSUK RISK)
- **Karar:** REAL
- **Fake olasılığı:** 0.0558
- **Güven:** 0.9442
- **Feature:** 768 boyutlu vektör
- **Not:** SigLIP2 yüksek çözünürlüklü model, görseli yüksek güvenle REAL olarak sınıflandırmış

#### `outputs/test5_PIN-B3.json` — Frekans Analizi Sonucu
- **Skor:** 0.8264 (YUKSEK RISK)
- **Karar:** FAKE
- **Fake olasılığı:** 0.8264
- **Güven:** 0.8264
- **Feature:** 512 boyutlu vektör
- **Not:** Frekans domain'de yapay izler tespit edilmiş — GAN/diffusion frekans artefaktları görünür

#### `outputs/test5_PIN-B4.json` — Independent Core 3-Sınıf Sonucu
- **Skor:** 1.0 (YUKSEK RISK)
- **Tahmin edilen sınıf:** Deepfake
- **AI olasılığı:** 0.0002
- **Deepfake olasılığı:** 0.9998
- **Real olasılığı:** 0.0000
- **Güven:** 0.9998
- **Not:** Model görseli kesin olarak Deepfake (manipüle edilmiş gerçek içerik) olarak sınıflandırmış

**Örnek Layer 1 çıktıları (test8.png için):**

#### `outputs/test8_PIN-A1.json` — Metadata Analiz Sonucu
- **Skor:** 0.35 (DUSUK RISK)
- **Bulgu:** C2PA yok, AI imzası yok, metadata tamamen boş
- **Boyut:** 1024x1024 (bilinen AI boyutu)

#### `outputs/test8_PIN-A3.json` — ELA Analiz Sonucu
- **Skor:** 0.71 (YUKSEK RISK)
- **Anomali:** 1 hotspot tespit edildi (pozisyon [7,6], 4.4σ sapma, yüksek şiddetli)
- **Dominant sinyal:** hotspot (manipülasyon göstergesi)

#### `outputs/test8_PIN-A4.json` — Yüz Tespit Sonucu
- **Yüz sayısı:** 1
- **Güvenilirlik:** 0.9154
- **Hizalama:** önden (roll açısı: -5.99°)
- **Çıktı:** test8_face_0.png (224x224)

---

## Boş Klasörler (Gelecek Geliştirme İçin)

| Klasör | Planlanan İşlev |
|--------|----------------|
| `layer3_video_temporal/` | Video temporal (zamana bağlı) analiz |
| `layer4_xai/` | Explainable AI — kararların açıklanabilirliği |
| `layer5_llm_reasoning/` | LLM tabanlı muhakeme ve sentez |
| `layer6_ensemble/` | Tüm katmanların birleşik nihai kararı |
| `tests/` | Birim ve entegrasyon testleri |

**Not:** `layer2_detection_core/` artık aktif — 4 PIN modülü (B1-B4) ile çalışmaktadır.

---

## Mimari Özet

```
                              ┌─────────────┐
                              │   main.py   │
                              └──────┬──────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
  ┌──────┴──────┐             ┌──────┴──────┐             ┌──────┴──────┐
  │ config/     │             │   core/     │             │   models/   │
  │ settings.py │             │ base_pin.py │             │ *.pt, *.tf  │
  └─────────────┘             └──────┬──────┘             └─────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                                 │
          ┌─────────┴──────────┐           ┌──────────┴──────────┐
          │  Layer 1 (Preproc) │           │  Layer 2 (Detection) │
          │  layer1_preproc/   │           │  layer2_detection/   │
          │  PIN-A1 .. A4      │           │  PIN-B1 .. B4        │
          └────────────────────┘           └─────────────────────┘

    BasePin (ABC)
        │
        │── Layer 1: Preprocessing (hafif, metadata/sinyal tabanlı)
        │   ├── PinA1Metadata  → EXIF, AI imza, C2PA binary, boyut analizi
        │   ├── PinA2C2pa      → C2PA manifest parse, zincir tarama, dijital kaynak tipi
        │   ├── PinA3Ela       → JPEG ELA, uniformity, hotspot/coldspot tespiti
        │   └── PinA4Face      → MediaPipe yüz tespiti, kırpma, hizalama, kalite
        │
        └── Layer 2: Detection Core (ağır, derin öğrenme modelleri)
            ├── PinB1Clip              → CLIP ViT-L/14 (frozen+LN-tune), 224x224, 1024-dim feature
            ├── PinB2Siglip            → SigLIP2-base-512 (full fine-tune), 512x512, 768-dim feature
            ├── PinB3Freq              → DCT/DWT + CNN (frequency domain), 4-kanal, 512-dim feature
            └── PinB4IndependentCore   → SigLIP2 3-sınıf (AI/Deepfake/Real), 224x224
```

**Skor aralığı:** 0.0 (temiz/gerçek) → 1.0 (kesin sahte/AI üretimi)

**Verdict seviyeleri:**
- `low_risk` (< 0.40): Muhtemelen gerçek fotoğraf
- `medium_risk` (0.40 - 0.70): Belirsiz, ek analiz gerekli
- `high_risk` (≥ 0.70): Yüksek AI üretimi şüphesi
- `no_data`: Yeterli sinyal üretilemedi
- `error`: Analiz sırasında hata oluştu

---

## Teknoloji Yığını

**Temel:**
- **Dil:** Python 3.13
- **Sanal Ortam:** venv

**Layer 1 — Preprocessing:**
- **Görsel İşleme:** Pillow (PIL), OpenCV, NumPy
- **Yüz Tespiti:** MediaPipe (BlazeFace)
- **C2PA Doğrulama:** c2pa-python
- **HEIC Desteği:** pillow-heif

**Layer 2 — Detection Core:**
- **Derin Öğrenme:** PyTorch (torch ≥ 2.0.0)
- **Model Yükleme:** HuggingFace Transformers (≥ 4.40.0) — CLIP, SigLIP2, SiglipForImageClassification
- **Frekans Analizi:** scipy (DCT dönüşümü), PyWavelets / pywt (DWT dönüşümü)
- **GPU Desteği:** CUDA (NVIDIA), MPS (Apple Silicon), CPU fallback — otomatik seçim

**Model Boyutları (toplam ~3.4 GB):**
| Model | PIN | Boyut | Mimari |
|-------|-----|-------|--------|
| `pin_b1_clip_ln_tune_final.pt` | PIN-B1 | ~1.6 GB | CLIP ViT-L/14 (427M param) |
| `pin_b2_siglip2_finetune_final.pt` | PIN-B2 | ~1.4 GB | SigLIP2-base-512 (376M param) |
| `pin_b3_freq_cnn_final.pt` | PIN-B3 | ~19 MB | Custom FreqCNN (4.8M param) |
| `pin_b4_ai_deepfake_real/` | PIN-B4 | ~354 MB | SigLIP2-base-224 (3-class) |
| `blaze_face_short_range.tflite` | PIN-A4 | ~224 KB | BlazeFace (yüz tespiti) |
