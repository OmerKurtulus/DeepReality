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
