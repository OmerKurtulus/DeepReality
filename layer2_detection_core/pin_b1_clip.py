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
