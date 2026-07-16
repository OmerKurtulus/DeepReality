"""
DeepReality — PIN-D1: Grad-CAM Heatmap (XAI)
════════════════════════════════════════════

Katman 2 modellerinin (PIN-B1 CLIP, PIN-B2 SigLIP2, PIN-B3 FreqCNN)
"sahtelik" kararını verirken görüntünün HANGİ bölgelerine baktığını
Grad-CAM tekniğiyle ısı haritası olarak görselleştirir.

Teknik:
    Grad-CAM (Selvaraju et al., ICCV 2017) — hedef sınıf logitinin
    (fake) seçilen katman aktivasyonlarına göre gradyanları alınır,
    kanal bazında ağırlıklandırılıp ReLU'dan geçirilerek uzamsal
    önem haritası üretilir. Harici pakete ihtiyaç duymadan PyTorch
    hook'ları ile implemente edilmiştir (transformers 5.x uyumlu).

    ViT modelleri (B1/B2) için token aktivasyonları patch ızgarasına
    (B1: 16x16, B2: 32x32) yeniden şekillendirilir; CLS token'ı varsa
    (CLIP) atılır. CNN (B3) için klasik Grad-CAM uygulanır — B3
    frekans domain'inde çalıştığından haritası uzamsal olarak
    YAKLAŞIKTIR ve birleşik haritada düşük ağırlıkla kullanılır.

Bağımlılık: PIN-B1, PIN-B2, PIN-B3 (aynı model instance'ları paylaşılır
— ek bellek maliyeti yoktur; bu yüzden pipeline'da bu pinlerden sonra
çalışır. Diğer tüm pinlerle paraleldir.)

Girdi:  Görsel dosya yolu + context (B1/B2/B3 sonuçları)
Çıktı:  Standart PIN JSON +
    - heatmaps:       model başına overlay PNG yolları + combined
    - focus_regions:  model başına odak bölgeleri (bbox, aktivasyon)
    - model_agreement: CLIP ve SigLIP odaklarının uzamsal uyumu (IoU)
    - cam_cache:      (JSON'a yazılmaz) PIN-D2'nin kullanması için
                      ham CAM matrisleri instance üzerinde tutulur

Skor: XAI pinleri risk skoru üretmez (bilgilendirme katmanı) → 0.0,
verdict "informational".
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
# Grad-CAM çekirdeği (hook tabanlı, paket bağımsız)
# ---------------------------------------------------------------------------

class _ActivationCapture:
    """
    Hedef katmanın aktivasyonunu (graf bağlantısı KOPMADAN) yakalar.

    Gradyan, tam backward() yerine torch.autograd.grad(logit, aktivasyon)
    ile hesaplanır — böylece gradyan yalnızca hedef katmana kadar geri
    yayılır. Hedef katman son encoder bloğu olduğundan bu, 24 bloklu
    ViT'in tamamında backward yapmaya kıyasla ~20x daha hızlıdır ve
    model parametrelerinin .grad alanlarına hiç dokunmaz.
    """

    def __init__(self, target_layer: torch.nn.Module):
        self.activation = None
        self._handle = target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        self.activation = out  # detach YOK — autograd.grad için graf gerekli

    def remove(self):
        self._handle.remove()


def _targeted_gradients(scalar: torch.Tensor,
                        activation: torch.Tensor) -> torch.Tensor:
    """scalar'ın activation'a göre gradyanı (yalnızca gereken yol)."""
    return torch.autograd.grad(scalar, activation, retain_graph=False)[0]


def _find_vit_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    ViT tabanlı modelde Grad-CAM hedef katmanını bulur:
    son encoder bloğunun layer_norm1 katmanı (pytorch-grad-cam'in
    ViT için önerdiği standart hedef).
    """
    candidates = [
        (name, module) for name, module in model.named_modules()
        if name.endswith(".layer_norm1") and ".layers." in name
    ]
    if not candidates:
        raise RuntimeError(
            "ViT hedef katmanı bulunamadı (encoder.layers.*.layer_norm1)"
        )
    # Son encoder bloğu: layer index'ine göre sırala
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
    ViT token aktivasyonlarından (1, T, C) Grad-CAM haritası üretir.
    CLS token'ı varsa otomatik tespit edilip atılır.
    Dönüş: (grid, grid) float32, [0, 1] normalize.
    """
    acts = activations.float()
    grads = gradients.float()

    num_tokens = acts.shape[1]
    g = isqrt(num_tokens)
    if g * g == num_tokens:
        pass  # CLS yok (SigLIP)
    elif isqrt(num_tokens - 1) ** 2 == num_tokens - 1:
        acts, grads = acts[:, 1:, :], grads[:, 1:, :]  # CLS'yi at (CLIP)
        g = isqrt(num_tokens - 1)
    else:
        raise RuntimeError(f"Token sayısı kare ızgaraya oturmuyor: {num_tokens}")

    weights = grads.mean(dim=1)                          # (1, C)
    cam = torch.einsum("btc,bc->bt", acts, weights)      # (1, T)
    cam = torch.relu(cam).reshape(g, g).cpu().numpy()
    return _normalize_cam(cam)


def _conv_cam(activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
    """
    CNN feature map'lerinden (1, C, H, W) klasik Grad-CAM haritası üretir.
    Dönüş: (H, W) float32, [0, 1] normalize.
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
# PIN sınıfı
# ---------------------------------------------------------------------------

class PinD1GradCam(BasePin):
    """
    PIN-D1: Grad-CAM Heatmap — Katman 2 modellerinin karar odağını
    ısı haritası olarak görselleştirir.
    """

    def __init__(self):
        super().__init__(
            pin_id="PIN-D1",
            pin_name="Grad-CAM Heatmap (XAI)",
            layer=4
        )
        # PIN-D2'nin kullanacağı ham CAM matrisleri (görsel başına yenilenir)
        self.cam_cache: dict[str, np.ndarray] = {}

    # ── Model bazlı CAM hesaplayıcılar ──────────────────────────────

    def _cam_clip(self, image: Image.Image) -> np.ndarray:
        """PIN-B1 CLIP ViT-L/14 için Grad-CAM (16x16 patch ızgarası)."""
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
        """PIN-B2 SigLIP2-512 için Grad-CAM (32x32 patch ızgarası)."""
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
        PIN-B3 FreqCNN için Grad-CAM (7x7 feature map).
        DİKKAT: Frekans domain'inde çalıştığından uzamsal karşılığı
        yaklaşıktır (DWT kanalları kısmen uzamsal, DCT kanalı değil).
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

    # ── Görselleştirme ve bölge çıkarımı ────────────────────────────

    @staticmethod
    def _upscale_cam(cam: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

    def _save_overlay(self, image_bgr: np.ndarray, cam_full: np.ndarray,
                      file_stem: str, tag: str) -> str:
        """CAM'i renkli overlay olarak orijinal görsel üzerine kaydeder."""
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
        Normalize CAM'de focus_threshold üstü bağlı bileşenleri bulur.
        Dönüş: [{bbox: {x, y, w, h}, area_ratio, mean_activation, peak}]
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
        """İki CAM'in eşik üstü maskeleri arasında IoU hesaplar."""
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

        # Üst pin skorları (etiketleme için; yoksa None)
        b1 = self.context.get("PIN-B1", {}).get("results", {})
        b2 = self.context.get("PIN-B2", {}).get("results", {})
        b3 = self.context.get("PIN-B3", {}).get("results", {})

        # ── Her model için Grad-CAM hesapla ──
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

        # ── Orijinal çözünürlüğe büyüt + overlay kaydet ──
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

        # ── Birleşik (combined) CAM ──
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

        # ── Modeller arası uzamsal uyum ──
        agreement = None
        if "clip" in cams_full and "siglip" in cams_full:
            agreement = round(self._spatial_agreement(
                cams_full["clip"], cams_full["siglip"],
                XAI_CONFIG["focus_threshold"]
            ), 4)

        # PIN-D2 için ham CAM'leri sakla (JSON'a yazılmaz)
        self.cam_cache = dict(cams_full)
        self.cam_cache["combined"] = combined

        # ── Sonuç ──
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
