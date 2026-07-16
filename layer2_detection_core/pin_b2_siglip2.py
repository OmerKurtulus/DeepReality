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