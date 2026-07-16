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
    # Bölgesel ELA ortalamasının median'dan kaç robust σ sapma
    # göstermesi gerektiği.
    "hotspot_std_threshold": 3.0,

    # Coldspot için ayrı σ eşiği
    # Coldspot'lar doğal içerik varyasyonuna (cilt, gökyüzü)
    # daha hassas olduğu için hotspot'tan yüksek tutulur.
    "coldspot_std_threshold": 3.5,

    # Coldspot minimum absolute ELA farkı
    # Bir bölgenin coldspot sayılabilmesi için median'dan
    # EN AZ bu kadar ELA birimi düşük olması gerekir.
    # Bu, doğal düşük-ELA bölgelerini (pürüzsüz cilt, gökyüzü)
    # gerçek manipülasyondan ayırır.
    #
    # Doğal varyasyon:  cilt vs arka plan ≈ 10-15 birim fark
    # Gerçek manipülasyon: Q40 patch ≈ 40-50 birim fark
    # Eşik 20: doğal varyasyonu filtreler, manipülasyonu yakalar
    "coldspot_min_absolute_deviation": 20.0,

    # Uniformity eşikleri (AI tespiti için)
    # Bölgesel ortalamaların standart sapması
    # Düşük = uniform (AI üretimi), Yüksek = doğal varyasyon
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
    # Bu değerler C2PA assertion'larında digitalSourceType olarak geçer
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
    # Düşük = daha fazla yüz ama false positive riski
    # Yüksek = daha az yüz ama daha güvenilir
    "min_detection_confidence": 0.5,

    # Yüz kırpma marjini (yüz boyutunun yüzdesi)
    # 0.30 = her yönde %30 ekstra alan
    # Çok az margin → yüz kenarları kesilebilir
    # Çok fazla margin → arka plan gürültüsü artar
    "crop_margin": 0.30,

    # Normalize edilmiş yüz boyutu [genişlik, yükseklik]
    # 224×224: Standart CNN/ViT input boyutu
    # Layer 2 modelleri (CLIP, SigLIP2) bu boyutu bekler
    "normalized_size": [224, 224],

    # Maksimum tespit edilecek yüz sayısı
    # Performans ve bellek yönetimi için sınır
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

# ──────────────────────────────────────────────
# KATMAN 4 — XAI (Açıklanabilirlik Pinleri)
# PIN-D1: Grad-CAM Heatmap | PIN-D2: Anomaly Localization
# ──────────────────────────────────────────────
XAI_CONFIG = {
    # ── PIN-D1: Grad-CAM ──
    # Hangi sınıfın kanıtı görselleştirilsin?
    # "fake" → modelin "sahtelik" kanıtını gösterir (logit index 0)
    "target_class": "fake",
    "fake_logit_index": 0,

    # Heatmap görselleştirme
    "cam_colormap": "jet",       # OpenCV colormap
    "overlay_alpha": 0.45,       # Isı haritası saydamlığı (0-1)

    # Odak bölgesi çıkarımı: normalize cam >= bu eşik olan alanlar
    "focus_threshold": 0.60,
    # Görüntü alanının bu oranından küçük odak bölgeleri gürültü sayılır
    "min_region_area_ratio": 0.001,
    "max_regions": 12,

    # Birleşik (combined) heatmap ağırlıkları
    # Frekans CAM'i uzamsal olarak yaklaşık olduğundan düşük ağırlıklı
    "combine_weights": {"clip": 0.40, "siglip": 0.40, "freq": 0.20},

    # ── PIN-D2: Anomaly Localization (ELA + Grad-CAM füzyonu) ──
    "fusion": {
        # Grad-CAM binary maskesi için quantile eşiği (üst %15)
        "cam_quantile": 0.85,
        # Bir ELA bölgesinin CAM ile "doğrulanmış" sayılması için
        # bölge içi ortalama normalize CAM değeri eşiği
        "ela_cam_confirm_threshold": 0.50,
    },

    # ── PIN-D2 kanıt skorlaması (destekleyici sinyal) ──
    # Skor fake olasılığı DEĞİL, "lokalize manipülasyon kanıtı" gücüdür
    "evidence_scores": {
        "fused_region": 0.80,        # ELA + CAM aynı bölgeyi işaretledi
        "ela_high_only": 0.55,       # Sadece yüksek şiddetli ELA anomalisi
        "ela_low_only": 0.35,        # Sadece düşük/orta ELA anomalisi
        "cam_focus_only": 0.30,      # Sadece güçlü CAM konsantrasyonu
        "none": 0.05,                # Lokalize kanıt yok
    },
    "thresholds": {
        "high_risk": 0.70,
        "medium_risk": 0.40,
    },
}