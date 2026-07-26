"""
DeepReality — PIN-D1: Grad-CAM Heatmap (XAI)
============================================

Recovers the spatial support of each Layer 2 model's decision: which
regions of the image the detectors (PIN-B1 CLIP, PIN-B2 SigLIP2,
PIN-B3 frequency CNN) relied upon when assigning their "fake" score.

Method
------
Grad-CAM (Selvaraju et al., ICCV 2017). The target-class logit is
differentiated with respect to the activations of a chosen layer; the
resulting channel gradients are used as importance weights, and the
weighted activation sum is rectified to retain only evidence that
supports the class. The implementation uses PyTorch hooks directly
rather than an external XAI package, which keeps the pin compatible
with current transformers releases and removes a dependency.

For the ViT backbones (B1, B2) token activations are reshaped onto the
patch grid (B1: 16x16, B2: 32x32) and the CLS token is discarded where
present. For the convolutional frequency model (B3) standard Grad-CAM
applies, but because that model operates in the frequency domain its
map is only spatially approximate and is therefore down-weighted in
the combined heatmap.

Dependencies: PIN-B1, PIN-B2, PIN-B3. The already-loaded model
instances are reused, so no additional memory is required; this is why
the pin is scheduled after the detection core rather than beside it.

Input:  Image path plus the upstream detection results.
Output: Standard pin envelope containing
    - heatmaps:        per-model overlay paths plus the combined map
    - focus_regions:   attention regions per model (bbox, activation)
    - model_agreement: spatial IoU between the CLIP and SigLIP maps
    - cam_cache:       raw CAM matrices held on the instance for PIN-D2
                       (never serialised to JSON)

Score: explainability pins produce no risk score. The envelope carries
0.0 with the verdict "informational".
"""

import numpy as np
import torch
from math import isqrt
from pathlib import Path
from PIL import Image

from core.base_pin import BasePin
from config.settings import XAI_CONFIG, OUTPUTS_DIR

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

import cv2


# ---------------------------------------------------------------------------
# Grad-CAM core (hook-based, no external dependency)
# ---------------------------------------------------------------------------

class _ActivationCapture:
    """
    Captures a target layer's activation with its autograd graph intact.

    Gradients are obtained through torch.autograd.grad(logit,
    activation) rather than a full backward() pass, so back-propagation
    stops at the target layer. Since that layer is the final encoder
    block, this is roughly twenty times faster than differentiating the
    whole 24-block ViT, and it never touches the .grad fields of the
    model parameters.
    """

    def __init__(self, target_layer: torch.nn.Module):
        self.activation = None
        self._handle = target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        self.activation = out  # No detach: autograd.grad requires the graph

    def remove(self):
        self._handle.remove()


def _targeted_gradients(scalar: torch.Tensor,
                        activation: torch.Tensor) -> torch.Tensor:
    """Gradient of a scalar with respect to an activation, along that path only."""
    return torch.autograd.grad(scalar, activation, retain_graph=False)[0]


def _find_vit_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Locate the Grad-CAM target layer of a ViT backbone: the layer_norm1
    module of the final encoder block, which is the standard choice for
    transformer architectures.
    """
    candidates = [
        (name, module) for name, module in model.named_modules()
        if name.endswith(".layer_norm1") and ".layers." in name
    ]
    if not candidates:
        raise RuntimeError(
            "No ViT target layer found (encoder.layers.*.layer_norm1)"
        )
    # Final encoder block: order by layer index
    def layer_index(item):
        parts = item[0].split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                return int(parts[i + 1])
        return -1
    candidates.sort(key=layer_index)
    return candidates[-1][1]


def _token_cam(activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
    """
    Build a Grad-CAM map from ViT token activations of shape (1, T, C).
    A CLS token is detected and discarded automatically when present.

    Returns: (grid, grid) float32, normalised to [0, 1].
    """
    acts = activations.float()
    grads = gradients.float()

    num_tokens = acts.shape[1]
    g = isqrt(num_tokens)
    if g * g == num_tokens:
        pass  # No CLS token (SigLIP)
    elif isqrt(num_tokens - 1) ** 2 == num_tokens - 1:
        acts, grads = acts[:, 1:, :], grads[:, 1:, :]  # Drop CLS (CLIP)
        g = isqrt(num_tokens - 1)
    else:
        raise RuntimeError(
            f"Token count does not map onto a square grid: {num_tokens}"
        )

    weights = grads.mean(dim=1)                          # (1, C)
    cam = torch.einsum("btc,bc->bt", acts, weights)      # (1, T)
    cam = torch.relu(cam).reshape(g, g).cpu().numpy()
    return _normalize_cam(cam)


def _conv_cam(activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
    """
    Build a standard Grad-CAM map from convolutional feature maps of
    shape (1, C, H, W).

    Returns: (H, W) float32, normalised to [0, 1].
    """
    acts = activations.float()
    grads = gradients.float()
    weights = grads.mean(dim=(2, 3), keepdim=True)       # (1, C, 1, 1)
    cam = torch.relu((weights * acts).sum(dim=1))         # (1, H, W)
    return _normalize_cam(cam[0].cpu().numpy())


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam -= cam.min()
    max_val = cam.max()
    if max_val > 0:
        cam /= max_val
    return cam


# ---------------------------------------------------------------------------
# Pin implementation
# ---------------------------------------------------------------------------

class PinD1GradCam(BasePin):
    """
    PIN-D1: renders the decision focus of the Layer 2 detectors as
    Grad-CAM heatmaps.
    """

    def __init__(self):
        super().__init__(
            pin_id="PIN-D1",
            pin_name="Grad-CAM Heatmap (XAI)",
            layer=4
        )
        # Raw CAM matrices consumed by PIN-D2, refreshed per image
        self.cam_cache: dict[str, np.ndarray] = {}

    # ── Per-model CAM computation ───────────────────────────────────

    def _cam_clip(self, image: Image.Image) -> np.ndarray:
        """Grad-CAM for PIN-B1, CLIP ViT-L/14 (16x16 patch grid)."""
        from layer2_detection_core.pin_b1_clip import _load_model, _get_device

        model, processor = _load_model()
        device = _get_device()

        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device).requires_grad_(True)

        capture = _ActivationCapture(_find_vit_target_layer(model))
        try:
            logits, _ = model(pixel_values)
            fake_logit = logits[0, XAI_CONFIG["fake_logit_index"]]
            grads = _targeted_gradients(fake_logit, capture.activation)
            cam = _token_cam(capture.activation.detach(), grads)
        finally:
            capture.remove()
        return cam

    def _cam_siglip(self, image: Image.Image) -> np.ndarray:
        """Grad-CAM for PIN-B2, SigLIP2-512 (32x32 patch grid)."""
        from layer2_detection_core.pin_b2_siglip2 import _load_model, _get_device

        model, processor = _load_model()
        device = _get_device()

        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device).requires_grad_(True)

        capture = _ActivationCapture(_find_vit_target_layer(model))
        try:
            logits, _ = model(pixel_values)
            fake_logit = logits[0, XAI_CONFIG["fake_logit_index"]]
            grads = _targeted_gradients(fake_logit, capture.activation)
            cam = _token_cam(capture.activation.detach(), grads)
        finally:
            capture.remove()
        return cam

    def _cam_freq(self, image: Image.Image) -> np.ndarray:
        """
        Grad-CAM for PIN-B3, the frequency CNN (7x7 feature map).

        Note: because this model operates on a DCT/DWT representation,
        the resulting map corresponds only approximately to image
        coordinates — the wavelet channels retain partial spatial
        structure while the DCT channel does not.
        """
        from layer2_detection_core.pin_b3_freq import (
            _load_model, _get_device, image_to_frequency_map
        )
        from config.settings import FREQ_CONFIG

        model = _load_model()
        device = _get_device()

        freq_map = image_to_frequency_map(
            image,
            target_size=FREQ_CONFIG["freq_image_size"],
            wavelet=FREQ_CONFIG["dwt_wavelet"]
        )
        freq_tensor = (
            torch.from_numpy(freq_map).unsqueeze(0).to(device).requires_grad_(True)
        )

        capture = _ActivationCapture(model.features[-1])
        try:
            logits, _ = model(freq_tensor)
            fake_logit = logits[0, XAI_CONFIG["fake_logit_index"]]
            grads = _targeted_gradients(fake_logit, capture.activation)
            cam = _conv_cam(capture.activation.detach(), grads)
        finally:
            capture.remove()
        return cam

    # ── Rendering and region extraction ─────────────────────────────

    @staticmethod
    def _upscale_cam(cam: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

    def _save_overlay(self, image_bgr: np.ndarray, cam_full: np.ndarray,
                      file_stem: str, tag: str) -> str:
        """Render the CAM as a colour overlay on the original image."""
        alpha = XAI_CONFIG["overlay_alpha"]
        heat = cv2.applyColorMap(
            (np.clip(cam_full, 0, 1) * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(heat, alpha, image_bgr, 1 - alpha, 0)
        out_path = OUTPUTS_DIR / f"{file_stem}_XAI_D1_{tag}.png"
        cv2.imwrite(str(out_path), overlay)
        return str(out_path)

    def _extract_focus_regions(self, cam_full: np.ndarray) -> list[dict]:
        """
        Extract connected components of the normalised CAM above the
        configured focus threshold.

        Returns: [{bbox: {x, y, w, h}, area_ratio, mean_activation, peak}]
        """
        h, w = cam_full.shape
        threshold = XAI_CONFIG["focus_threshold"]
        min_area = XAI_CONFIG["min_region_area_ratio"] * h * w

        mask = (cam_full >= threshold).astype(np.uint8)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

        regions = []
        for i in range(1, num):  # 0 = arka plan
            x, y, bw, bh, area = stats[i]
            if area < min_area:
                continue
            component = cam_full[labels == i]
            regions.append({
                "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
                "area_ratio": round(float(area) / (h * w), 4),
                "mean_activation": round(float(component.mean()), 4),
                "peak_activation": round(float(component.max()), 4),
                "center": {
                    "x": int(centroids[i][0]), "y": int(centroids[i][1])
                },
            })

        regions.sort(key=lambda r: r["mean_activation"] * r["area_ratio"],
                     reverse=True)
        return regions[: XAI_CONFIG["max_regions"]]

    @staticmethod
    def _spatial_agreement(cam_a: np.ndarray, cam_b: np.ndarray,
                           threshold: float) -> float:
        """Intersection-over-union between two thresholded CAM masks."""
        mask_a = cam_a >= threshold
        mask_b = cam_b >= threshold
        union = np.logical_or(mask_a, mask_b).sum()
        if union == 0:
            return 0.0
        return float(np.logical_and(mask_a, mask_b).sum() / union)

    # ── Ana analiz ──────────────────────────────────────────────────

    def analyze(self, file_path: str) -> dict:
        image = Image.open(file_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        file_stem = Path(file_path).stem

        # Upstream scores, used for annotation only
        b1 = self.context.get("PIN-B1", {}).get("results", {})
        b2 = self.context.get("PIN-B2", {}).get("results", {})
        b3 = self.context.get("PIN-B3", {}).get("results", {})

        # ── Compute Grad-CAM for every available model ──
        cams: dict[str, np.ndarray] = {}
        cam_errors: dict[str, str] = {}
        for tag, fn in [("clip", self._cam_clip),
                        ("siglip", self._cam_siglip),
                        ("freq", self._cam_freq)]:
            try:
                cams[tag] = fn(image)
            except Exception as e:
                cam_errors[tag] = str(e)
                self.errors.append(f"{tag} Grad-CAM hatası: {e}")

        if not cams:
            self.cam_cache = {}
            return {
                "results": {"heatmaps": {}, "focus_regions": {},
                            "cam_errors": cam_errors},
                "score": 0.0,
                "verdict": "error",
                "details": "Hiçbir model için Grad-CAM üretilemedi.",
            }

        # ── Upsample to native resolution and write overlays ──
        cams_full = {
            tag: self._upscale_cam(cam, width, height)
            for tag, cam in cams.items()
        }

        heatmap_paths = {}
        focus_regions = {}
        for tag, cam_full in cams_full.items():
            heatmap_paths[tag] = self._save_overlay(
                image_bgr, cam_full, file_stem, tag
            )
            focus_regions[tag] = self._extract_focus_regions(cam_full)

        # ── Combined CAM ──
        weights = XAI_CONFIG["combine_weights"]
        total_weight = sum(weights[t] for t in cams_full)
        combined = sum(
            cams_full[t] * (weights[t] / total_weight) for t in cams_full
        )
        combined = _normalize_cam(combined)
        heatmap_paths["combined"] = self._save_overlay(
            image_bgr, combined, file_stem, "combined"
        )
        focus_regions["combined"] = self._extract_focus_regions(combined)

        # ── Cross-model spatial agreement ──
        agreement = None
        if "clip" in cams_full and "siglip" in cams_full:
            agreement = round(self._spatial_agreement(
                cams_full["clip"], cams_full["siglip"],
                XAI_CONFIG["focus_threshold"]
            ), 4)

        # Retain raw CAMs for PIN-D2 (excluded from the JSON envelope)
        self.cam_cache = dict(cams_full)
        self.cam_cache["combined"] = combined

        # ── Result ──
        n_focus = len(focus_regions.get("combined", []))
        details_parts = [
            f"{len(cams)} model için Grad-CAM üretildi "
            f"({', '.join(cams.keys())})."
        ]
        if n_focus:
            details_parts.append(f"Birleşik haritada {n_focus} odak bölgesi.")
        if agreement is not None:
            details_parts.append(f"CLIP-SigLIP uzamsal uyumu (IoU): {agreement:.2f}.")
        details_parts.append(
            "XAI katmanı bilgilendirme amaçlıdır; risk skoru üretmez."
        )

        results = {
            "target_class": XAI_CONFIG["target_class"],
            "heatmaps": heatmap_paths,
            "focus_regions": focus_regions,
            "model_agreement_iou": agreement,
            "source_scores": {
                "clip_prob": b1.get("clip_prob"),
                "siglip_prob": b2.get("siglip_prob"),
                "freq_prob": b3.get("freq_prob"),
            },
            "cam_errors": cam_errors,
            "notes": (
                "freq haritası frekans domain'inden türetildiği için "
                "uzamsal olarak yaklaşıktır; birleşik haritada "
                f"%{int(XAI_CONFIG['combine_weights']['freq']*100)} "
                "ağırlıkla kullanılır."
            ),
        }

        return {
            "results": results,
            "score": 0.0,
            "verdict": "informational",
            "details": " ".join(details_parts),
        }
