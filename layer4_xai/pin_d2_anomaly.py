"""
DeepReality — PIN-D2: Anomaly Localization (XAI)
════════════════════════════════════════════════

İki BAĞIMSIZ kanıt kaynağını birleştirerek manipüle edilmiş bölgeleri
tek bir haritada işaretler:

    1. PIN-A3 ELA anomali bölgeleri (sıkıştırma-fiziği tabanlı kanıt)
    2. PIN-D1 Grad-CAM birleşik haritası (model-karar tabanlı kanıt)

Füzyon mantığı:
    - Bir ELA bölgesinin içindeki ortalama Grad-CAM aktivasyonu
      eşiği aşıyorsa bölge "DOĞRULANMIŞ" (fused) sayılır — iki
      bağımsız yöntem aynı bölgeyi işaret ediyor demektir. Bu,
      tek kaynaklı işaretlerden çok daha güçlü bir kanıttır.
    - Sadece ELA veya sadece CAM tarafından işaretlenen bölgeler de
      kaynak etiketiyle raporlanır.

Bağımlılık: PIN-A3 (ELA bölgeleri), PIN-D1 (ham CAM matrisi,
instance üzerindeki cam_cache ile paylaşılır), PIN-B3 (frekans skoru).

Çıktı: Standart PIN JSON +
    - marked_regions: [{bbox, source: ela|gradcam|fused, strength, ...}]
    - annotated_image: işaretli overlay PNG yolu
    - skor: "lokalize manipülasyon kanıtı" gücü (fake olasılığı DEĞİL —
      ensemble katmanında destekleyici sinyal olarak kullanılır)

Görsel işaretleme:
    KIRMIZI  = ELA hotspot   | MAVİ   = ELA coldspot
    SARI     = Grad-CAM odağı | TURUNCU (kalın) = doğrulanmış füzyon
"""

import numpy as np
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


# BGR renkleri
_COLOR_ELA_HOT = (0, 0, 255)      # kırmızı
_COLOR_ELA_COLD = (255, 80, 0)    # mavi
_COLOR_CAM = (0, 220, 255)        # sarı
_COLOR_FUSED = (0, 140, 255)      # turuncu


class PinD2AnomalyLocalization(BasePin):
    """
    PIN-D2: Anomaly Localization — ELA ve Grad-CAM kanıtlarını
    birleştirip manipülasyon bölgelerini tek haritada işaretler.
    """

    def __init__(self):
        super().__init__(
            pin_id="PIN-D2",
            pin_name="Anomaly Localization (XAI)",
            layer=4
        )

    # ── Yardımcılar ─────────────────────────────────────────────────

    @staticmethod
    def _ela_regions_from_context(a3_results: dict) -> list[dict]:
        """PIN-A3 manipulation_regions → standart bbox listesi."""
        regions = []
        for reg in a3_results.get("manipulation_regions", []):
            pixel_range = reg.get("pixel_range", {})
            y_range = pixel_range.get("y")
            x_range = pixel_range.get("x")
            if not y_range or not x_range:
                continue
            y1, y2 = int(y_range[0]), int(y_range[1])
            x1, x2 = int(x_range[0]), int(x_range[1])
            regions.append({
                "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                "type": reg.get("type", "hotspot"),
                "severity": reg.get("severity", "low"),
                "deviation": reg.get("deviation"),
            })
        return regions

    @staticmethod
    def _merge_adjacent_boxes(regions: list[dict]) -> list[dict]:
        """
        Komşu/bitişik ELA grid hücrelerini tek bölgede birleştirir
        (8x8 grid'de büyük bir manipülasyon birden çok hücreye yayılır).
        Aynı tipteki kesişen/bitişik kutular birleştirilir.
        """
        merged: list[dict] = []
        for reg in regions:
            b = reg["bbox"]
            placed = False
            for m in merged:
                if m["type"] != reg["type"]:
                    continue
                mb = m["bbox"]
                # Bitişiklik kontrolü (1 piksel tolerans)
                if (b["x"] <= mb["x"] + mb["w"] + 1 and
                        mb["x"] <= b["x"] + b["w"] + 1 and
                        b["y"] <= mb["y"] + mb["h"] + 1 and
                        mb["y"] <= b["y"] + b["h"] + 1):
                    x1 = min(mb["x"], b["x"])
                    y1 = min(mb["y"], b["y"])
                    x2 = max(mb["x"] + mb["w"], b["x"] + b["w"])
                    y2 = max(mb["y"] + mb["h"], b["y"] + b["h"])
                    m["bbox"] = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
                    m["cell_count"] += 1
                    m["severity"] = max(
                        m["severity"], reg["severity"],
                        key=["low", "medium", "high"].index
                    )
                    placed = True
                    break
            if not placed:
                merged.append({
                    "bbox": dict(reg["bbox"]),
                    "type": reg["type"],
                    "severity": reg["severity"],
                    "cell_count": 1,
                })
        return merged

    @staticmethod
    def _cam_regions(cam: np.ndarray, quantile: float,
                     min_area_ratio: float, max_regions: int) -> list[dict]:
        """Birleşik CAM'in üst quantile maskesinden bölge çıkarır."""
        h, w = cam.shape
        threshold = float(np.quantile(cam, quantile))
        # Tamamen düz CAM'lerde (threshold≈0) sahte bölge üretme
        if threshold <= 1e-6:
            return []
        mask = (cam >= threshold).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        regions = []
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < min_area_ratio * h * w:
                continue
            component = cam[labels == i]
            regions.append({
                "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
                "mean_activation": round(float(component.mean()), 4),
                "area_ratio": round(float(area) / (h * w), 4),
            })
        regions.sort(key=lambda r: r["mean_activation"], reverse=True)
        return regions[:max_regions]

    @staticmethod
    def _mean_cam_in_bbox(cam: np.ndarray, bbox: dict) -> float:
        h, w = cam.shape
        x1 = max(0, bbox["x"])
        y1 = max(0, bbox["y"])
        x2 = min(w, bbox["x"] + bbox["w"])
        y2 = min(h, bbox["y"] + bbox["h"])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return float(cam[y1:y2, x1:x2].mean())

    @staticmethod
    def _draw_region(canvas: np.ndarray, bbox: dict, color: tuple,
                     thickness: int, label: str | None = None):
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        if label:
            font_scale = max(0.5, canvas.shape[1] / 2000)
            cv2.putText(
                canvas, label, (x + 4, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA
            )

    # ── Ana analiz ──────────────────────────────────────────────────

    def analyze(self, file_path: str) -> dict:
        image = Image.open(file_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        file_stem = Path(file_path).stem

        fusion_cfg = XAI_CONFIG["fusion"]
        evidence = XAI_CONFIG["evidence_scores"]

        # ── 1. Kanıt kaynaklarını topla ──
        a3_results = self.context.get("PIN-A3", {}).get("results", {})
        ela_raw = self._ela_regions_from_context(a3_results)
        ela_regions = self._merge_adjacent_boxes(ela_raw)

        # PIN-D1 instance'ından ham CAM matrisi (orijinal çözünürlükte)
        combined_cam = None
        d1_pin = self.context.get("_pins", {}).get("PIN-D1")
        if d1_pin is not None and getattr(d1_pin, "cam_cache", None):
            combined_cam = d1_pin.cam_cache.get("combined")
        if combined_cam is not None and combined_cam.shape != (height, width):
            combined_cam = cv2.resize(
                combined_cam, (width, height), interpolation=cv2.INTER_CUBIC
            )

        cam_regions = []
        if combined_cam is not None:
            cam_regions = self._cam_regions(
                combined_cam,
                quantile=fusion_cfg["cam_quantile"],
                min_area_ratio=XAI_CONFIG["min_region_area_ratio"],
                max_regions=XAI_CONFIG["max_regions"],
            )

        # ── 2. Füzyon: ELA bölgesi + CAM doğrulaması ──
        marked_regions = []
        fused_count = 0
        confirm_thr = fusion_cfg["ela_cam_confirm_threshold"]

        for reg in ela_regions:
            cam_support = (
                self._mean_cam_in_bbox(combined_cam, reg["bbox"])
                if combined_cam is not None else 0.0
            )
            is_fused = cam_support >= confirm_thr
            if is_fused:
                fused_count += 1
            marked_regions.append({
                "bbox": reg["bbox"],
                "source": "fused" if is_fused else "ela",
                "ela_type": reg["type"],
                "severity": reg["severity"],
                "cam_support": round(cam_support, 4),
                "cell_count": reg["cell_count"],
                "description": (
                    f"ELA {reg['type']} ({reg['severity']}) — "
                    + ("Grad-CAM ile DOĞRULANDI" if is_fused
                       else "model odağıyla örtüşmüyor")
                ),
            })

        # CAM'in işaretlediği ama ELA'nın işaretlemediği bölgeler.
        # ÖNEMLİ KAPI: Grad-CAM normalize edildiği için her görselde
        # "en sıcak" bölgeler vardır — modeller fake demiyorsa bu
        # bölgeler anomali değil, sadece modelin baktığı yerdir.
        # Bu yüzden CAM-only bölgeler ancak en az bir Katman 2 modeli
        # medium_risk üstü skor verdiyse işaretlenir.
        d1_scores = (
            self.context.get("PIN-D1", {}).get("results", {})
            .get("source_scores", {})
        )
        b3_prob = (
            self.context.get("PIN-B3", {}).get("results", {}).get("freq_prob")
        )
        model_probs = [
            p for p in (
                d1_scores.get("clip_prob"),
                d1_scores.get("siglip_prob"),
                d1_scores.get("freq_prob"),
                b3_prob,
            ) if p is not None
        ]
        max_model_prob = max(model_probs) if model_probs else 0.0
        models_suspicious = (
            max_model_prob >= XAI_CONFIG["thresholds"]["medium_risk"]
        )

        if models_suspicious:
            for creg in cam_regions:
                overlaps_ela = any(
                    self._bbox_iou(creg["bbox"], m["bbox"]) > 0.20
                    for m in marked_regions
                )
                if overlaps_ela:
                    continue
                marked_regions.append({
                    "bbox": creg["bbox"],
                    "source": "gradcam",
                    "mean_activation": creg["mean_activation"],
                    "area_ratio": creg["area_ratio"],
                    "description": (
                        f"Model odak bölgesi (max model skoru "
                        f"{max_model_prob:.2f}, ELA desteği yok)"
                    ),
                })

        # ── 3. Kanıt skoru (fake olasılığı DEĞİL) ──
        ela_high = any(
            r["severity"] == "high" for r in ela_regions
        )
        # Güçlü CAM konsantrasyonu: modeller şüpheli diyor VE
        # odak küçük bir alanda yoğunlaşmış
        cam_concentrated = models_suspicious and any(
            r["mean_activation"] >= 0.70 and r["area_ratio"] <= 0.15
            for r in cam_regions
        )

        if fused_count > 0:
            score = evidence["fused_region"]
            basis = f"{fused_count} bölge iki bağımsız yöntemle doğrulandı"
        elif ela_high:
            score = evidence["ela_high_only"]
            basis = "yüksek şiddetli ELA anomalisi (CAM desteği yok)"
        elif ela_regions:
            score = evidence["ela_low_only"]
            basis = "düşük/orta ELA anomalisi (CAM desteği yok)"
        elif cam_concentrated:
            score = evidence["cam_focus_only"]
            basis = "yoğun model odağı (ELA desteği yok)"
        else:
            score = evidence["none"]
            basis = "lokalize manipülasyon kanıtı yok"

        thresholds = XAI_CONFIG["thresholds"]
        if score >= thresholds["high_risk"]:
            verdict = "high_risk"
        elif score >= thresholds["medium_risk"]:
            verdict = "medium_risk"
        else:
            verdict = "low_risk"

        # ── 4. İşaretli görseli üret ──
        thickness = max(2, width // 500)
        for m in marked_regions:
            if m["source"] == "fused":
                self._draw_region(canvas, m["bbox"], _COLOR_FUSED,
                                  thickness * 2, "FUZYON")
            elif m["source"] == "ela":
                color = (_COLOR_ELA_HOT if m.get("ela_type") == "hotspot"
                         else _COLOR_ELA_COLD)
                self._draw_region(canvas, m["bbox"], color, thickness,
                                  f"ELA {m.get('ela_type', '')}")
            else:
                self._draw_region(canvas, m["bbox"], _COLOR_CAM, thickness,
                                  "CAM")

        annotated_path = None
        if marked_regions:
            annotated_path = str(
                OUTPUTS_DIR / f"{file_stem}_XAI_D2_anomaly.png"
            )
            cv2.imwrite(annotated_path, canvas)

        # ── 5. Sonuç ──
        results = {
            "marked_regions": marked_regions,
            "region_counts": {
                "fused": fused_count,
                "ela_only": sum(1 for m in marked_regions
                                if m["source"] == "ela"),
                "gradcam_only": sum(1 for m in marked_regions
                                    if m["source"] == "gradcam"),
                "total": len(marked_regions),
            },
            "annotated_image": annotated_path,
            "evidence_basis": basis,
            "sources_available": {
                "ela": bool(a3_results),
                "gradcam": combined_cam is not None,
            },
            "max_model_prob": round(max_model_prob, 4),
            "models_suspicious": models_suspicious,
            "score_note": (
                "Bu skor fake olasılığı değil, lokalize manipülasyon "
                "kanıtının gücüdür (ensemble için destekleyici sinyal)."
            ),
        }

        details = (
            f"Anomali lokalizasyonu: {len(marked_regions)} bölge işaretlendi "
            f"({fused_count} füzyon-doğrulamalı). Kanıt: {basis}."
        )

        return {
            "results": results,
            "score": score,
            "verdict": verdict,
            "details": details,
        }

    @staticmethod
    def _bbox_iou(a: dict, b: dict) -> float:
        ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
        bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = a["w"] * a["h"] + b["w"] * b["h"] - inter
        return inter / union if union > 0 else 0.0
