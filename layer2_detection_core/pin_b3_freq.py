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