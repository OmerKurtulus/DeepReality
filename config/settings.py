"""
DeepReality — Global Configuration
Shared settings, thresholds and paths used by every pin in the system.

All tunable parameters (weights, thresholds, model paths, prompts
configuration) live here so that no decision constant is hard-coded
inside pin implementations.
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
# ENVIRONMENT (.env loader)
# ──────────────────────────────────────────────
def _load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """
    Minimal .env reader (no external dependency).

    Parses KEY=VALUE lines, ignores comments and blank lines, and strips
    optional surrounding quotes. Existing environment variables always
    take precedence, so a shell-exported key overrides the file.
    """
    if not path.exists():
        return

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass  # An unreadable .env must never break the pipeline


_load_dotenv()

# ──────────────────────────────────────────────
# PIN-A1: Metadata Analysis Thresholds
# ──────────────────────────────────────────────
METADATA_CONFIG = {
    # Generator metadata signatures — extend as new tools are identified
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

    # Camera EXIF fields — their PRESENCE raises the likelihood of authentic capture
    "camera_fields": [
        "Make", "Model", "LensModel", "LensMake",
        "FocalLength", "FNumber", "ExposureTime",
        "ISOSpeedRatings", "ISO", "Flash",
        "ShutterSpeedValue", "ApertureValue",
        "MeteringMode", "WhiteBalance"
    ],

    # GPS fields — evidence of a physical exposure event
    "gps_fields": [
        "GPSLatitude", "GPSLongitude", "GPSAltitude",
        "GPSLatitudeRef", "GPSLongitudeRef",
        "GPSInfo"
    ],

    # ── C2PA / JUMBF Binary Markers ──
    # Tools such as ChatGPT, DALL-E and Sora embed C2PA metadata in the file.
    # These byte patterns are searched for in the raw container.
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
    # Generators emit characteristic output sizes (multiples of 64, squares).
    # These dimensions are NOT proof, but they contribute an additional signal.
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
        "ai_signature_detected": 0.30,   # Generator signature present in metadata
        "c2pa_detected": 0.25,           # C2PA/JUMBF binary marker bulundu
        "no_camera_data": 0.12,          # Kamera verisi yok
        "no_gps_data": 0.03,             # GPS yok
        "no_datetime": 0.07,             # Tarih bilgisi yok
        "software_suspicious": 0.08,     # Software field looks suspicious
        "metadata_stripped": 0.05,       # Metadata entirely stripped
        "ai_dimensions": 0.05,           # Dimensions typical of generators
        "compression_anomaly": 0.05      # Compression ratio anomaly
    },

    # Verdict thresholds
    "thresholds": {
        "high_risk": 0.70,      # >= 0.70 -> strong suspicion of AI generation
        "medium_risk": 0.40,    # >= 0.40 -> moderate suspicion
        "low_risk": 0.0         # <  0.40 -> weak suspicion / probably authentic
    }
}

# ──────────────────────────────────────────────
# PIN-A3: ELA (Error Level Analysis) Configuration
# ──────────────────────────────────────────────
ELA_CONFIG = {
    # JPEG re-save quality used as the ELA reference
    # Lower = larger visible differences, higher = finer discrimination
    "resave_quality": 90,

    # ELA difference amplification (for visualisation)
    # Pixel differences are small, so they are scaled up to be visible
    "amplification_scale": 20,

    # Regional analysis grid size (N x N)
    "grid_size": 8,

    # Anomaly detection threshold (MAD-based, robust to outliers)
    # How many robust sigma a regional ELA mean must deviate from the
    # median to be flagged.
    "hotspot_std_threshold": 3.0,

    # Separate sigma threshold for coldspots
    # Coldspots are more sensitive to natural content variation (skin, sky),
    # so their threshold is set higher than for hotspots.
    "coldspot_std_threshold": 3.5,

    # Minimum absolute ELA difference for a coldspot
    # A region must sit at least this many ELA units below the median
    # before it can be classified as a coldspot.
    # This separates naturally low-ELA regions (smooth skin, sky) from
    # genuine manipulation.
    #
    # Natural variation:     skin vs background ~ 10-15 units
    # Genuine manipulation:  Q40 patch         ~ 40-50 units
    # A threshold of 20 filters the former while retaining the latter
    "coldspot_min_absolute_deviation": 20.0,

    # Uniformity thresholds (AI-generation signal)
    # Standard deviation of the regional means
    # Low = uniform (typical of generated imagery), high = natural variation
    "uniformity_thresholds": {
        "very_uniform": 3.0,    # < 3.0  -> highly uniform (AI signal)
        "uniform": 6.0,         # < 6.0  -> uniform (possibly AI)
        "moderate": 12.0,       # < 12.0 → orta (belirsiz)
        # > 12.0 -> high variation (authentic or manipulated)
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
    # IPTC digital source types denoting AI generation
    # These appear as digitalSourceType in C2PA assertions
    "ai_digital_source_types": [
        "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
        "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicMedia",
        "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
    ],

    # Known generative tools (appear as C2PA issuer or softwareAgent)
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

    # Non-AI actions (camera capture, scanning)
    "non_ai_actions": [
        "c2pa.captured",       # Captured with a camera
        "c2pa.scanned",        # Scanned
    ],

    # Generative actions
    "ai_creation_actions": [
        "c2pa.created",        # Created (may be AI)
        "c2pa.generated",      # Generated
    ],

    # Editing actions (may indicate AI-assisted editing)
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
    # MediaPipe model selection
    # 0 = short range (<= 2 m, selfie camera)
    # 1 = full range (<= 5 m, general photography)
    "model_selection": 1,

    # Minimum face detection confidence (0.0 - 1.0)
    # Lower = more faces detected, higher false-positive rate
    # Higher = fewer faces, greater reliability
    "min_detection_confidence": 0.5,

    # Face crop margin (as a fraction of face size)
    # 0.30 = 30 percent additional area on every side
    # Too little margin -> facial boundaries are clipped
    # Too much margin   -> background noise increases
    "crop_margin": 0.30,

    # Normalised face size [width, height]
    # 224×224: Standart CNN/ViT input boyutu
    # The Layer 2 models (CLIP, SigLIP2) expect this resolution
    "normalized_size": [224, 224],

    # Maximum number of faces to detect
    # Bounded for runtime and memory predictability
    "max_faces": 10,
}

# ──────────────────────────────────────────────
# PIN-B1: CLIP ViT-L/14 Deepfake Detection
# ──────────────────────────────────────────────
CLIP_CONFIG = {
    # HuggingFace identifier used to instantiate the architecture
    "clip_model_name": "openai/clip-vit-large-patch14",

    # Path to the trained weights
    "model_path": str(PROJECT_ROOT / "models" / "pin_b1_clip_ln_tune_final.pt"),

    # Number of classes
    "num_labels": 2,

    # Label map
    "label_map": {0: "fake", 1: "real"},

    # Verdict thresholds (0.0 = real, 1.0 = fake)
    "thresholds": {
        "high_risk": 0.70,
        "medium_risk": 0.40,
    },
}

# ──────────────────────────────────────────────
# PIN-B2: SigLIP2-base-512 Deepfake Detection
# ──────────────────────────────────────────────
SIGLIP_CONFIG = {
    # HuggingFace identifier used to instantiate the architecture
    "model_name": "google/siglip2-base-patch16-512",

    # Path to the trained weights
    "model_path": str(PROJECT_ROOT / "models" / "pin_b2_siglip2_finetune_final.pt"),

    # Number of classes
    "num_labels": 2,

    # Label map
    "label_map": {0: "fake", 1: "real"},

    # Verdict thresholds (0.0 = real, 1.0 = fake)
    "thresholds": {
        "high_risk": 0.70,
        "medium_risk": 0.40,
    },
}

# ──────────────────────────────────────────────
# PIN-B3: Frequency Analysis (DCT/DWT + CNN)
# ──────────────────────────────────────────────
FREQ_CONFIG = {
    # Path to the trained weights
    "model_path": str(PROJECT_ROOT / "models" / "pin_b3_freq_cnn_final.pt"),

    # Number of classes
    "num_labels": 2,

    # Frequency transform parameters
    "freq_image_size": 224,           # Frequency map resolution
    "dwt_wavelet": "haar",            # DWT wavelet ailesi
    "num_channels": 4,                # DCT + DWT-LH + DWT-HL + DWT-HH

    # Label map
    "label_map": {0: "fake", 1: "real"},

    # Verdict thresholds (0.0 = real, 1.0 = fake)
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
    # Contains: config.json, preprocessor_config.json, model.safetensors
    "model_dir": str(PROJECT_ROOT / "models" / "pin_b4_ai_deepfake_real"),

    # Kaynak bilgisi
    "source": "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2",
    "base_model": "google/siglip2-base-patch16-224",

    # Class count and label map
    "num_labels": 3,
    "label_map": {0: "AI", 1: "Deepfake", 2: "Real"},

    # Verdict thresholds (fake_score = ai_prob + deepfake_prob)
    # 0.0 = certainly authentic, 1.0 = certainly synthetic
    "thresholds": {
        "high_risk": 0.70,
        "medium_risk": 0.40,
    },
}

# ──────────────────────────────────────────────
# LAYER 4 — XAI (EXPLAINABILITY PINS)
# PIN-D1: Grad-CAM Heatmap | PIN-D2: Anomaly Localization
# ──────────────────────────────────────────────
XAI_CONFIG = {
    # ── PIN-D1: Grad-CAM ──
    # Which class should the evidence be visualised for?
    # "fake" -> shows the model's evidence for synthesis (logit index 0)
    "target_class": "fake",
    "fake_logit_index": 0,

    # Heatmap rendering
    "cam_colormap": "jet",       # OpenCV colormap
    "overlay_alpha": 0.45,       # Heatmap opacity (0-1)

    # Focus region extraction: areas where normalised CAM >= this threshold
    "focus_threshold": 0.60,
    # Focus regions below this fraction of the frame are treated as noise
    "min_region_area_ratio": 0.001,
    "max_regions": 12,

    # Combined heatmap weights
    # The frequency CAM is only spatially approximate, hence a lower weight
    "combine_weights": {"clip": 0.40, "siglip": 0.40, "freq": 0.20},

    # ── PIN-D2: Anomaly Localisation (ELA + Grad-CAM fusion) ──
    "fusion": {
        # Quantile threshold for the Grad-CAM binary mask (top 15 percent)
        "cam_quantile": 0.85,
        # Mean normalised CAM activation required inside an ELA region
        # before that region counts as corroborated
        "ela_cam_confirm_threshold": 0.50,
    },

    # ── PIN-D2 evidence scoring (supporting signal) ──
    # This score is NOT a fake probability; it is the strength of
    "evidence_scores": {
        "fused_region": 0.80,        # ELA and Grad-CAM marked the same region
        "ela_high_only": 0.55,       # High-severity ELA anomaly only
        "ela_low_only": 0.35,        # Low or moderate ELA anomaly only
        "cam_focus_only": 0.30,      # Strong CAM concentration only
        "none": 0.05,                # No localised evidence
    },
    "thresholds": {
        "high_risk": 0.70,
        "medium_risk": 0.40,
    },
}
# ──────────────────────────────────────────────
# LAYER 5 — LLM REASONING ENGINE
# PIN-E1: synthesises every upstream pin into a final adjudication
# ──────────────────────────────────────────────
LLM_CONFIG = {
    # ── Provider (OpenAI-compatible chat completions API) ──
    # OpenRouter is the default gateway; any compatible endpoint works.
    "api_base": os.environ.get(
        "DEEPREALITY_LLM_API_BASE",
        "https://openrouter.ai/api/v1",
    ),
    "api_key_env": "OPENROUTER_API_KEY",
    "model": os.environ.get(
        "DEEPREALITY_LLM_MODEL",
        "anthropic/claude-sonnet-4.5",
    ),

    # ── Request parameters ──
    # Temperature is kept low: forensic adjudication must be reproducible.
    "temperature": 0.15,
    # Reasoning models bill their internal deliberation against this
    # budget before emitting a single visible character. At 1600 a model
    # such as claude-sonnet-5 exhausts the allowance while still thinking
    # and returns finish_reason="length" with empty content, which the
    # client can only report as an empty response. The ceiling is set
    # well above the ~1500 tokens the report itself needs so that both
    # reasoning and non-reasoning models complete.
    "max_tokens": 6000,
    "timeout_seconds": 180,
    "max_retries": 2,
    "retry_backoff_seconds": 2.0,

    # Optional attribution headers honoured by OpenRouter
    "referer": "https://github.com/OmerKurtulus/DeepReality",
    "app_title": "DeepReality",

    # ── Output ──
    # Natural-language report language (ISO 639-1). Turkish is the
    # project default; the reasoning process itself is language agnostic.
    "output_language": "tr",

    # ── Evidence packaging ──
    # High-dimensional embeddings and static model cards are stripped
    # from the digest; they carry no adjudication value and would
    # otherwise dominate the token budget.
    "digest": {
        "max_ela_regions": 6,
        "max_focus_regions": 4,
        "max_marked_regions": 6,
        "float_precision": 3,
    },

    # ── Verdict thresholds applied to the returned fake probability ──
    "thresholds": {
        "high_risk": 0.70,
        "medium_risk": 0.40,
    },

    # Persist the exact prompt payload alongside the result for auditability
    "save_prompt_transcript": True,
}

# ──────────────────────────────────────────────
# LAYER 6 — ENSEMBLE FUSION
# PIN-F1: stacked meta-learner over the pin score vector
# ──────────────────────────────────────────────
ENSEMBLE_CONFIG = {
    # Trained artefacts. Absent until the meta-learner has been fitted;
    # the pin reports a transparent baseline in the meantime.
    "model_path": str(PROJECT_ROOT / "models" / "pin_f1_xgboost.json"),
    "metadata_path": str(PROJECT_ROOT / "models" / "pin_f1_metadata.json"),

    # Binary decision boundary applied to the calibrated probability.
    "decision_threshold": 0.50,

    # Verdict bands, consistent with the other pins.
    "thresholds": {
        "high_risk": 0.70,
        "medium_risk": 0.40,
    },

    # Fallback weights used only while the model is untrained. These
    # reflect the design rationale of the detection core: the frozen
    # backbone generalises best, the frequency model contributes a
    # disjoint domain, and the three-class core votes independently.
    "baseline_weights": {"b1": 0.35, "b2": 0.25, "b3": 0.20, "b4": 0.20},

    # Number of contributing features named in the output.
    "report_top_features": 6,

    # ── Training defaults, consumed by the Colab notebook ──
    "training": {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "reg_lambda": 1.5,
        "early_stopping_rounds": 40,
        "cv_folds": 5,
        "random_state": 42,
        # Shallow trees and strong regularisation are deliberate: the
        # design matrix has roughly fifty columns and a realistic corpus
        # supplies only thousands of rows, so an unconstrained booster
        # memorises the training split instead of learning the fusion.
    },
}
