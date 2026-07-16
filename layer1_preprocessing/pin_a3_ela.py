"""
DeepReality — PIN-A3: ELA (Error Level Analysis)
═════════════════════════════════════════════════

İşlev:
    Görselin JPEG sıkıştırma seviyesi farklarını analiz ederek
    manipülasyon bölgelerini ve AI üretim izlerini tespit eder.

Algoritma:
    1. Orijinal görseli bilinen bir JPEG kalitesinde (Q=90) yeniden kaydet
    2. Orijinal ile yeniden kaydedilmiş versiyonu piksel piksel karşılaştır
    3. Fark haritasını (ELA map) analiz et:
       a. Global istatistikler (ortalama, standart sapma, max)
       b. Bölgesel analiz (8×8 grid) — her bölgenin ELA ortalaması
       c. Uniformity skoru — bölgesel ortalamaların standart sapması
       d. Hotspot tespiti — anormal yüksek ELA bölgeleri
    4. ELA heatmap görselini kaydet

Yorumlama:
    - Gerçek fotoğraf:  Doğal ELA varyasyonu, orta uniformity
    - AI üretim:        Çok uniform ELA (tüm piksel aynı anda üretildi)
    - Manipüle edilmiş: Düzenlenen bölgelerde belirgin hotspot'lar

Teknoloji:
    OpenCV, PIL (Pillow), NumPy

Çıktı:
    ela_heatmap       : str    — ELA heatmap dosya yolu
    manipulation_regions: list — Tespit edilen anormal bölgeler
    uniformity_score  : float  — 0 (çok uniform/AI) → yüksek (doğal)
    global_stats      : dict   — ELA istatistikleri
    grid_analysis     : dict   — Bölgesel analiz sonuçları
    score             : float  — 0.0 (temiz) → 1.0 (manipüle/AI)

Yazar: DeepReality Ekibi
Tarih: 2026-02-17
"""

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# HEIC/HEIF desteği (iPhone fotoğrafları)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

from core.base_pin import BasePin
from config.settings import ELA_CONFIG, OUTPUTS_DIR


class PinA3Ela(BasePin):
    """
    PIN-A3: Error Level Analysis

    JPEG sıkıştırma farkları ile manipülasyon ve AI üretim tespiti.
    """

    def __init__(self):
        super().__init__(
            pin_id="PIN-A3",
            pin_name="ELA (Error Level Analysis)",
            layer=1
        )
        self.resave_quality = ELA_CONFIG["resave_quality"]
        self.amp_scale = ELA_CONFIG["amplification_scale"]
        self.grid_size = ELA_CONFIG["grid_size"]
        self.hotspot_std = ELA_CONFIG["hotspot_std_threshold"]
        self.coldspot_std = ELA_CONFIG.get("coldspot_std_threshold", 3.5)
        self.coldspot_min_abs = ELA_CONFIG.get(
            "coldspot_min_absolute_deviation", 20.0
        )
        self.uniformity_thresholds = ELA_CONFIG["uniformity_thresholds"]
        self.save_heatmap = ELA_CONFIG["save_heatmap"]

    # ════════════════════════════════════════════════════════════════
    # ANA ANALİZ
    # ════════════════════════════════════════════════════════════════

    def analyze(self, file_path: str) -> dict:
        """
        ELA analiz pipeline'ı.

        Adımlar:
            1. Görseli yükle + format tespiti
            2. JPEG Q=90 ile yeniden kaydet (bellekte)
            3. Piksel farkı hesapla → ELA map
            4. Global istatistikler
            5. Bölgesel grid analizi (8×8)
            6. Uniformity skoru
            7. Hotspot tespiti
            8. ELA heatmap kaydet
            9. Format-bilinçli skorlama
        """

        # ── 1. Görseli yükle + format tespiti ──
        original = self._load_image(file_path)
        if original is None:
            return {
                "results": self._build_empty_results("Gorsel yuklenemedi"),
                "score": 0.0,
                "verdict": "error",
                "details": "Gorsel yuklenemedi veya desteklenmeyen format."
            }

        # Format tespiti — ELA güvenilirliği için kritik
        source_format = self._detect_source_format(file_path)

        # ── 2-3. ELA map hesapla ──
        ela_map = self._compute_ela_map(original)
        if ela_map is None:
            return {
                "results": self._build_empty_results("ELA hesaplanamadi"),
                "score": 0.0,
                "verdict": "error",
                "details": "ELA hesaplama hatasi — JPEG donusumu basarisiz."
            }

        # ── 4. Global istatistikler ──
        global_stats = self._compute_global_stats(ela_map)

        # ── 5. Bölgesel grid analizi ──
        grid_analysis = self._compute_grid_analysis(ela_map)

        # ── 6. Uniformity skoru ──
        uniformity = self._compute_uniformity(grid_analysis)

        # ── 7. Hotspot tespiti ──
        hotspots = self._detect_hotspots(grid_analysis, global_stats)

        # ── 8. ELA heatmap kaydet ──
        heatmap_path = None
        if self.save_heatmap:
            heatmap_path = self._save_ela_heatmap(
                ela_map, Path(file_path).stem
            )

        # ── 9. Format-bilinçli skorlama ──
        score, score_breakdown = self._calculate_score(
            global_stats=global_stats,
            uniformity=uniformity,
            hotspots=hotspots,
            source_format=source_format
        )

        # Sinyal güvenilirliği kontrolü — yetersizse verdict=no_data
        is_reliable = score_breakdown.get("signal_reliability", {}).get(
            "is_reliable", True
        )
        if not is_reliable:
            verdict = "no_data"
        else:
            verdict = self._determine_verdict(score)

        details = self._generate_details(
            score, verdict, global_stats, uniformity, hotspots,
            source_format
        )

        return {
            "results": {
                "ela_heatmap": str(heatmap_path) if heatmap_path else None,
                "source_format": source_format,
                "global_stats": global_stats,
                "grid_analysis": {
                    "grid_size": self.grid_size,
                    "region_count": len(grid_analysis),
                    "region_means_summary": {
                        "min": round(min(r["mean"] for r in grid_analysis), 2),
                        "max": round(max(r["mean"] for r in grid_analysis), 2),
                        "median": round(
                            float(np.median([r["mean"] for r in grid_analysis])), 2
                        ),
                    }
                },
                "uniformity": uniformity,
                "manipulation_regions": hotspots,
                "image_dimensions": {
                    "width": original.shape[1],
                    "height": original.shape[0]
                },
                "score_breakdown": score_breakdown,
            },
            "score": score,
            "verdict": verdict,
            "details": details
        }

    # ════════════════════════════════════════════════════════════════
    # ELA HESAPLAMA
    # ════════════════════════════════════════════════════════════════

    def _load_image(self, file_path: str) -> np.ndarray | None:
        """Görseli OpenCV formatında yükler (BGR)."""
        try:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is None:
                # OpenCV bazen Unicode path'lerde başarısız olur
                # PIL ile dene
                pil_img = Image.open(file_path).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            self.errors.append(f"Gorsel yukleme hatasi: {e}")
            return None

    def _detect_source_format(self, file_path: str) -> dict:
        """
        Görselin kaynak formatını ve sıkıştırma türünü tespit eder.

        ELA için kritik olan JPEG vs PNG değil, LOSSY vs LOSSLESS ayrımıdır:

        LOSSY formatlar (sıkıştırma geçmişi VAR):
            - JPEG (.jpg/.jpeg) — DCT tabanlı kayıplı sıkıştırma
            - HEIC/HEIF (.heic/.heif) — HEVC tabanlı kayıplı sıkıştırma
            - WebP lossy (.webp) — VP8 tabanlı kayıplı sıkıştırma
            → ELA uniformity sinyali GÜVENİLİR
            → Farklı alanlar farklı sıkıştırılmış → doğal varyasyon

        LOSSLESS formatlar (sıkıştırma geçmişi YOK):
            - PNG (.png) — kayıpsız
            - BMP (.bmp) — sıkıştırmasız
            - TIFF (.tiff/.tif) — genelde kayıpsız
            - GIF (.gif) — palette-based
            → ELA uniformity sinyali GÜVENİLMEZ
            → Hiç lossy sıkıştırma geçmişi yok
            → Gerçek foto ve AI görseli benzer uniform ELA verir

        NOT: iPhone fotoğrafları HEIC formatındadır. HEIC kayıplı (lossy)
        sıkıştırma kullanır, bu yüzden ELA doğru çalışır. Ancak HEIC→PNG
        dönüştürülürse sıkıştırma geçmişi silinir ve ELA güvenilmez olur.
        """
        ext = Path(file_path).suffix.lower()

        # Lossy formatlar — sıkıştırma geçmişi var
        lossy_formats = {
            ".jpg", ".jpeg", ".jpe", ".jfif",  # JPEG
            ".heic", ".heif",                    # HEIC (iPhone)
            ".webp",                             # WebP (genelde lossy)
        }

        # Lossless formatlar — sıkıştırma geçmişi yok
        lossless_formats = {
            ".png",          # PNG
            ".bmp",          # BMP
            ".tiff", ".tif", # TIFF
            ".gif",          # GIF
        }

        is_lossy = ext in lossy_formats

        if is_lossy:
            return {
                "file_extension": ext,
                "compression_type": "lossy",
                "has_compression_history": True,
                "uniformity_confidence": "high",
                "format_note": (
                    f"{ext.upper()} formati — kayipli sikistirma gecmisi mevcut. "
                    f"ELA uniformity sinyali guvenilir."
                )
            }
        else:
            return {
                "file_extension": ext,
                "compression_type": "lossless",
                "has_compression_history": False,
                "uniformity_confidence": "low",
                "format_note": (
                    f"{ext.upper()} formati — kayipli sikistirma gecmisi yok. "
                    f"ELA uniformity sinyali DUSUK guvenilirlikte. "
                    f"Hotspot tespiti hala gecerli."
                )
            }

    def _compute_ela_map(self, original: np.ndarray) -> np.ndarray | None:
        """
        ELA map hesaplar.

        Algoritma:
            1. Orijinali JPEG Q=90 olarak bellekte yeniden kaydet
            2. Yeniden kaydedilmiş versiyonu yükle
            3. Piksel piksel mutlak fark hesapla
            4. Farkı amplifikasyon çarpanı ile büyüt

        Returns:
            ELA map (grayscale, 0-255) veya None (hata durumunda)
        """
        try:
            # PIL Image'a dönüştür (JPEG kaydetme için)
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(original_rgb)

            # Bellekte JPEG olarak yeniden kaydet
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=self.resave_quality)
            buffer.seek(0)

            # Yeniden kaydedilmiş versiyonu yükle
            resaved = Image.open(buffer)
            resaved_np = np.array(resaved).astype(np.float64)
            original_float = original_rgb.astype(np.float64)

            # Piksel farkı hesapla (her kanal için)
            diff = np.abs(original_float - resaved_np)

            # Kanalları birleştir (ortalama) → grayscale ELA
            ela_gray = np.mean(diff, axis=2)

            # Amplifikasyon — küçük farkları görünür yap
            ela_amplified = ela_gray * self.amp_scale

            # 0-255 arasına sınırla
            ela_map = np.clip(ela_amplified, 0, 255).astype(np.uint8)

            return ela_map

        except Exception as e:
            self.errors.append(f"ELA hesaplama hatasi: {e}")
            return None

    # ════════════════════════════════════════════════════════════════
    # İSTATİSTİKSEL ANALİZ
    # ════════════════════════════════════════════════════════════════

    def _compute_global_stats(self, ela_map: np.ndarray) -> dict:
        """
        ELA map'in global istatistiklerini hesaplar.

        Önemli metrikler:
            - mean: Ortalama ELA değeri (düşük=çok sıkıştırılmış, yüksek=yeni)
            - std: Standart sapma (düşük=uniform, yüksek=çeşitli)
            - max: Maksimum ELA piksel değeri
            - skewness: Çarpıklık (pozitif=sağa çarpık, negatif=sola çarpık)
            - energy: Toplam enerji (kareler toplamı / piksel sayısı)
        """
        flat = ela_map.flatten().astype(np.float64)

        mean = float(np.mean(flat))
        std = float(np.std(flat))
        max_val = float(np.max(flat))
        min_val = float(np.min(flat))
        median = float(np.median(flat))

        # Çarpıklık (skewness) — dağılımın şekli
        if std > 0:
            skewness = float(np.mean(((flat - mean) / std) ** 3))
        else:
            skewness = 0.0

        # Enerji — toplam ELA yoğunluğu
        energy = float(np.mean(flat ** 2))

        # Yüzdelikler (percentiles)
        p25 = float(np.percentile(flat, 25))
        p75 = float(np.percentile(flat, 75))
        p95 = float(np.percentile(flat, 95))
        p99 = float(np.percentile(flat, 99))

        return {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "median": round(median, 2),
            "skewness": round(skewness, 2),
            "energy": round(energy, 2),
            "percentile_25": round(p25, 2),
            "percentile_75": round(p75, 2),
            "percentile_95": round(p95, 2),
            "percentile_99": round(p99, 2),
        }

    def _compute_grid_analysis(self, ela_map: np.ndarray) -> list[dict]:
        """
        ELA map'i NxN grid'e bölerek bölgesel analiz yapar.

        Her bölge için:
            - mean: Ortalama ELA değeri
            - std: Standart sapma
            - max: Maksimum değer
            - position: (row, col) grid konumu

        Bu analiz manipüle edilmiş bölgelerin tespitinde kritik:
        Düzenlenmiş bölgeler farklı ELA seviyesi gösterir.
        """
        h, w = ela_map.shape
        grid_h = h // self.grid_size
        grid_w = w // self.grid_size

        regions = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y_start = row * grid_h
                y_end = (row + 1) * grid_h if row < self.grid_size - 1 else h
                x_start = col * grid_w
                x_end = (col + 1) * grid_w if col < self.grid_size - 1 else w

                region = ela_map[y_start:y_end, x_start:x_end]
                region_float = region.astype(np.float64)

                regions.append({
                    "position": (row, col),
                    "mean": round(float(np.mean(region_float)), 2),
                    "std": round(float(np.std(region_float)), 2),
                    "max": round(float(np.max(region_float)), 2),
                    "pixel_range": {
                        "y": (y_start, y_end),
                        "x": (x_start, x_end)
                    }
                })

        return regions

    def _compute_uniformity(self, grid_analysis: list[dict]) -> dict:
        """
        Bölgesel ELA değerlerinin ne kadar uniform olduğunu hesaplar.

        Uniformity skoru = bölgesel ortalamaların standart sapması

        AI üretimi görseller çok uniform ELA gösterir çünkü:
        - Tüm pikseller aynı anda, aynı model tarafından üretilir
        - Doğal sıkıştırma varyasyonu yoktur
        - Her bölgenin ELA seviyesi birbirine çok yakındır

        Gerçek fotoğraflar doğal varyasyon gösterir:
        - Farklı doku alanları (gökyüzü, ağaç, bina) farklı ELA verir
        - Kenarlar ve düz alanlar farklı sıkıştırılır

        Manipüle edilmiş görseller yüksek varyasyon gösterir:
        - Düzenlenen bölgeler belirgin şekilde farklı ELA verir
        """
        region_means = [r["mean"] for r in grid_analysis]
        region_stds = [r["std"] for r in grid_analysis]

        uniformity_score = float(np.std(region_means))
        mean_of_means = float(np.mean(region_means))
        std_of_stds = float(np.std(region_stds))

        # Coefficient of variation (CV) — normalize edilmiş varyasyon
        cv = (uniformity_score / mean_of_means * 100) if mean_of_means > 0 else 0.0

        # Kategori belirle
        thresholds = self.uniformity_thresholds
        if uniformity_score < thresholds["very_uniform"]:
            category = "very_uniform"
            description = "Cok uniform — AI uretimi sinyali"
        elif uniformity_score < thresholds["uniform"]:
            category = "uniform"
            description = "Uniform — olasi AI uretimi"
        elif uniformity_score < thresholds["moderate"]:
            category = "moderate"
            description = "Orta duzeyde varyasyon — belirsiz"
        else:
            category = "varied"
            description = "Yuksek varyasyon — gercek foto veya manipulasyon"

        return {
            "uniformity_score": round(uniformity_score, 2),
            "coefficient_of_variation": round(cv, 2),
            "mean_of_region_means": round(mean_of_means, 2),
            "std_of_region_stds": round(std_of_stds, 2),
            "category": category,
            "description": description,
        }

    def _detect_hotspots(self, grid_analysis: list[dict],
                         global_stats: dict) -> list[dict]:
        """
        ELA map'te anormal bölgeleri tespit eder — İKİ YÖNLÜ.

        HOTSPOT (yüksek ELA):
            Yeni eklenen / taze içerik → henüz sıkıştırılmamış
            → JPEG re-save'de büyük fark üretir → yüksek ELA

        COLDSPOT (düşük ELA):
            Ağır sıkıştırılmış / farklı kaliteden yapıştırılmış içerik
            → Detayını zaten kaybetmiş → re-save'de az fark üretir
            → Çevresinden belirgin şekilde düşük ELA

        TEKNİK: MAD (Median Absolute Deviation) tabanlı robust tespit.
        MAD outlier'lara karşı dayanıklıdır.

            median = bölgesel ortalamaların medyanı
            MAD = median(|her_bölge - median|)
            robust_std = MAD × 1.4826
            hotspot:  bölge_mean > median + k_hot × robust_std
            coldspot: bölge_mean < median - k_cold × robust_std
                      VE (median - bölge_mean) ≥ min_absolute_deviation

        COLDSPOT İÇİN EK KONTROL:
            Sadece istatistiksel sapma (σ) yetmez.
            Uniform görsellerde MAD çok küçük olunca doğal
            varyasyon (pürüzsüz cilt, gökyüzü) coldspot olarak
            tetiklenebilir. Minimum absolute fark kontrolü ile
            doğal düşük-ELA bölgeleri filtrelenir:
              - Cilt vs arka plan: ~10-15 birim fark → DOĞAL
              - Gerçek manipülasyon: ~40-50 birim fark → COLDSPOT
        """
        region_means = np.array([r["mean"] for r in grid_analysis])

        # Robust istatistikler
        median_val = float(np.median(region_means))
        mad = float(np.median(np.abs(region_means - median_val)))
        mad = max(mad, 0.5)  # Minimum MAD
        robust_std = mad * 1.4826

        # Eşikler — hotspot ve coldspot için AYRI
        high_threshold = median_val + (self.hotspot_std * robust_std)
        low_threshold = median_val - (self.coldspot_std * robust_std)

        anomalies = []
        for region in grid_analysis:
            mean_val = region["mean"]

            if mean_val > high_threshold:
                # HOTSPOT — taze / eklenen içerik
                deviation = (mean_val - median_val) / robust_std
                anomalies.append({
                    "type": "hotspot",
                    "position": region["position"],
                    "mean_ela": round(mean_val, 2),
                    "deviation": round(deviation, 2),
                    "pixel_range": region["pixel_range"],
                    "severity": (
                        "high" if deviation > 4.0 else
                        "medium" if deviation > 2.5 else
                        "low"
                    ),
                    "description": (
                        f"Yuksek ELA — olasi taze ekleme veya "
                        f"farkli sikistirma ({deviation:.1f}σ sapma)"
                    )
                })

            elif mean_val < low_threshold:
                # COLDSPOT adayı — ek kontrol gerekli
                absolute_diff = median_val - mean_val

                # Minimum absolute fark kontrolü:
                # Doğal düşük-ELA bölgeleri (cilt, gökyüzü)
                # genelde median'dan 10-15 birim düşük.
                # Gerçek manipülasyon 40+ birim fark üretir.
                if absolute_diff < self.coldspot_min_abs:
                    continue  # Doğal varyasyon — atla

                deviation = absolute_diff / robust_std
                anomalies.append({
                    "type": "coldspot",
                    "position": region["position"],
                    "mean_ela": round(mean_val, 2),
                    "deviation": round(deviation, 2),
                    "absolute_diff": round(absolute_diff, 2),
                    "pixel_range": region["pixel_range"],
                    "severity": (
                        "high" if deviation > 4.0 else
                        "medium" if deviation > 2.5 else
                        "low"
                    ),
                    "description": (
                        f"Dusuk ELA — olasi agir sikistirilmis "
                        f"icerik yapistirmasi ({deviation:.1f}σ, "
                        f"{absolute_diff:.0f} birim sapma)"
                    )
                })

        return anomalies

    # ════════════════════════════════════════════════════════════════
    # HEATMAP KAYDETME
    # ════════════════════════════════════════════════════════════════

    def _save_ela_heatmap(self, ela_map: np.ndarray,
                          file_stem: str) -> Path | None:
        """
        ELA map'i renkli heatmap olarak kaydeder.

        Colormap: JET (mavi=düşük ELA, kırmızı=yüksek ELA)
        Çıktı: outputs/{dosya_adı}_ELA_heatmap.png
        """
        try:
            # OpenCV colormap uygula
            heatmap = cv2.applyColorMap(ela_map, cv2.COLORMAP_JET)

            # Kaydet
            output_path = OUTPUTS_DIR / f"{file_stem}_ELA_heatmap.png"
            cv2.imwrite(str(output_path), heatmap)

            return output_path
        except Exception as e:
            self.errors.append(f"Heatmap kaydetme hatasi: {e}")
            return None

    # ════════════════════════════════════════════════════════════════
    # SKORLAMA
    # ════════════════════════════════════════════════════════════════

    def _calculate_score(self, global_stats: dict,
                         uniformity: dict,
                         hotspots: list[dict],
                         source_format: dict = None) -> tuple[float, dict]:
        """
        ELA analizinden skor hesaplar.

        ÖNEMLİ TASARIM KARARI:
        ═══════════════════════
        ELA uniformity tek başına AI/gerçek AYIRT EDEMEZ.

        Neden?
        - iPhone hesaplamalı fotoğrafçılık → gerçek foto uniform ELA verir
        - Modern AI (GPT-4o, Imagen 3) → gerçekçi doku, doğal ELA varyasyonu
        - Hem PNG hem lossy formatlarda aynı sorun var

        Bu yüzden:
        1. UNIFORMITY → ZAYIF destekleyici sinyal (max 0.25)
           Tek başına karar verici DEĞİL. Layer 6'da diğer PIN'lerle birleşir.
        2. HOTSPOT → GÜÇLÜ manipülasyon sinyali (max 0.85)
           ELA'nın asıl gücü budur. Format bağımsız çalışır.

        FORMAT CEZASI YOK — çünkü hem AI hem gerçek foto
        her formatta olabilir. Bunu bilemeyiz.
        """
        breakdown = {}
        global_mean = global_stats["mean"]
        global_std = global_stats["std"]

        # Format bilgisi (sadece raporlama için, skor etkilemez)
        comp_type = source_format.get("compression_type", "unknown") if source_format else "unknown"

        # ═══════════════════════════════════════════════
        # ADIM 0: SİNYAL GÜVENİLİRLİĞİ KONTROLÜ
        # ═══════════════════════════════════════════════

        signal_reliable = True
        reliability_reason = "ELA sinyali yeterli"

        if global_mean < 1.0 and global_std < 1.0:
            signal_reliable = False
            reliability_reason = (
                f"ELA sinyali yetersiz (mean={global_mean:.1f}, "
                f"std={global_std:.1f}). Gorsel zaten ayni JPEG "
                f"kalitesinde veya ELA anlamlı fark uretemiyor."
            )
        elif global_std < 0.5 and global_mean > 0:
            signal_reliable = False
            reliability_reason = (
                f"Yapay uniformity (mean={global_mean:.1f}, "
                f"std={global_std:.1f}). Tum pikseller ayni farkta — "
                f"ELA anlamlı bilgi uretemiyor."
            )

        breakdown["signal_reliability"] = {
            "is_reliable": signal_reliable,
            "reason": reliability_reason,
            "global_mean": global_mean,
            "global_std": global_std,
        }

        if not signal_reliable:
            breakdown["_total"] = {
                "uniformity_signal": 0.0,
                "hotspot_signal": 0.0,
                "reliability_penalty": True,
                "final_score": 0.0,
                "dominant_signal": "insufficient_signal"
            }
            return 0.0, breakdown

        # ═══════════════════════════════════════════════
        # ADIM 1: UNIFORMITY SİNYALİ
        # Zayıf destekleyici sinyal — max 0.25
        # Tek başına karar verici DEĞİL
        # ═══════════════════════════════════════════════
        u_score = uniformity["uniformity_score"]
        u_cat = uniformity["category"]
        cv = uniformity["coefficient_of_variation"]

        if u_cat == "very_uniform":
            uniformity_signal = 0.25
        elif u_cat == "uniform":
            uniformity_signal = 0.20
        elif u_cat == "moderate":
            uniformity_signal = 0.10
        else:  # varied
            uniformity_signal = 0.05

        breakdown["uniformity_signal"] = {
            "value": round(uniformity_signal, 4),
            "uniformity_score": u_score,
            "category": u_cat,
            "cv": cv,
            "compression_type": comp_type,
            "note": (
                "Uniformity zayif destekleyici sinyaldir. "
                "Tek basina AI/gercek ayirimi yapamiyor. "
                "Gercek foto da (ozellikle iPhone) uniform olabilir."
            ),
            "description": (
                f"ELA uniformity: {u_cat} "
                f"(skor={u_score:.1f}, CV={cv:.1f}%)"
            )
        }

        # ═══════════════════════════════════════════════
        # ADIM 2: ANOMALİ SİNYALİ (manipülasyon tespiti)
        # Hotspot (yüksek ELA) + Coldspot (düşük ELA)
        # ELA'nın asıl gücü — format bağımsız
        # ═══════════════════════════════════════════════
        anomaly_count = len(hotspots)  # hotspots artık tüm anomalileri içerir
        total_regions = self.grid_size * self.grid_size
        anomaly_ratio = anomaly_count / total_regions if total_regions > 0 else 0

        hotspot_count = sum(1 for h in hotspots if h.get("type") == "hotspot")
        coldspot_count = sum(1 for h in hotspots if h.get("type") == "coldspot")

        if anomaly_count == 0:
            hotspot_signal = 0.0
        elif anomaly_ratio < 0.05:
            hotspot_signal = 0.50  # Az — lokalize manipülasyon
        elif anomaly_ratio < 0.15:
            hotspot_signal = 0.70  # Orta — belirgin manipülasyon
        elif anomaly_ratio < 0.30:
            hotspot_signal = 0.60  # Çok — genel düzenleme
        else:
            hotspot_signal = 0.40  # Yaygın — farklı sıkıştırma

        high_severity = sum(1 for h in hotspots if h.get("severity") == "high")
        if high_severity > 0:
            hotspot_signal += 0.15

        hotspot_signal = min(hotspot_signal, 0.85)

        breakdown["hotspot_signal"] = {
            "value": round(hotspot_signal, 4),
            "anomaly_count": anomaly_count,
            "hotspot_count": hotspot_count,
            "coldspot_count": coldspot_count,
            "total_regions": total_regions,
            "anomaly_ratio": round(anomaly_ratio, 4),
            "high_severity_count": high_severity,
            "note": (
                "Anomali tespiti ELA'nin en guclu sinyalidir. "
                "Hotspot: taze ekleme. Coldspot: agir sikistirilmis yapistirma. "
                "Her iki yon de manipulasyon gostergesidir."
            ),
            "description": (
                f"{anomaly_count} anomali ({hotspot_count} hotspot + "
                f"{coldspot_count} coldspot) / {total_regions} bolge "
                f"(oran: {anomaly_ratio:.1%})"
            )
        }

        # ═══════════════════════════════════════════════
        # NİHAİ SKOR
        # Hotspot varsa → hotspot dominant
        # Hotspot yoksa → sadece zayıf uniformity
        # ═══════════════════════════════════════════════
        if hotspot_signal > 0:
            # Manipülasyon tespit edildi → hotspot + uniformity bonus
            final_score = hotspot_signal + (uniformity_signal * 0.3)
        else:
            # Manipülasyon yok → sadece zayıf uniformity sinyali
            final_score = uniformity_signal

        final_score = round(max(0.0, min(1.0, final_score)), 4)

        breakdown["_total"] = {
            "uniformity_signal": round(uniformity_signal, 4),
            "hotspot_signal": round(hotspot_signal, 4),
            "reliability_penalty": False,
            "final_score": final_score,
            "dominant_signal": (
                "hotspot" if hotspot_signal > 0
                else "uniformity_weak"
            )
        }

        return final_score, breakdown

    def _determine_verdict(self, score: float) -> str:
        """Skora göre verdict belirler."""
        if score >= 0.70:
            return "high_risk"
        elif score >= 0.40:
            return "medium_risk"
        else:
            return "low_risk"

    # ════════════════════════════════════════════════════════════════
    # RAPORLAMA
    # ════════════════════════════════════════════════════════════════

    def _generate_details(self, score: float, verdict: str,
                          global_stats: dict, uniformity: dict,
                          hotspots: list[dict],
                          source_format: dict = None) -> str:
        """Analiz sonucunun Türkçe açıklamasını üretir."""
        parts = []

        fmt_ext = source_format.get("file_extension", "?") if source_format else "?"
        comp_type = source_format.get("compression_type", "?") if source_format else "?"

        parts.append(
            f"ELA Analizi tamamlandi (format: {fmt_ext.upper()}, "
            f"sikistirma: {comp_type}). "
            f"Global ELA: ortalama={global_stats['mean']:.1f}, "
            f"std={global_stats['std']:.1f}, max={global_stats['max']:.1f}."
        )

        # Sinyal güvenilirliği
        g_mean = global_stats["mean"]
        g_std = global_stats["std"]
        if g_mean < 1.0 and g_std < 1.0:
            parts.append(
                "SINYAL YETERSIZ: ELA anlamlı fark uretemiyor."
            )
            parts.append(f"PIN-A3 Sonuc: VERI YOK (Skor: {score:.2f})")
            return " | ".join(parts)
        if g_std < 0.5 and g_mean > 0:
            parts.append(
                "SINYAL YETERSIZ: Tum pikseller ayni farkta."
            )
            parts.append(f"PIN-A3 Sonuc: VERI YOK (Skor: {score:.2f})")
            return " | ".join(parts)

        # Uniformity
        u_cat = uniformity["category"]
        u_score = uniformity["uniformity_score"]
        if u_cat in ("very_uniform", "uniform"):
            parts.append(
                f"UNIFORMITY: {u_cat} ELA dagilimi (skor={u_score:.1f}). "
                f"Zayif destekleyici sinyal — tek basina AI/gercek "
                f"ayirimi yapamiyor (gercek fotolar da uniform olabilir)."
            )
        elif u_cat == "moderate":
            parts.append(
                f"UNIFORMITY: Orta duzeyde ELA varyasyonu (skor={u_score:.1f})."
            )
        else:
            parts.append(
                f"UNIFORMITY: Yuksek ELA varyasyonu (skor={u_score:.1f})."
            )

        # Anomaliler (hotspot + coldspot)
        if hotspots:
            hot_count = sum(1 for h in hotspots if h.get("type") == "hotspot")
            cold_count = sum(1 for h in hotspots if h.get("type") == "coldspot")
            high = sum(1 for h in hotspots if h["severity"] == "high")
            parts.append(
                f"ANOMALI: {len(hotspots)} bolge tespit edildi "
                f"({hot_count} hotspot + {cold_count} coldspot, "
                f"{high} yuksek siddetli). Manipulasyon gostergesi."
            )
        else:
            parts.append("ANOMALI: Anormal bolge tespit edilmedi.")

        verdict_tr = {
            "high_risk": "YUKSEK RISK",
            "medium_risk": "ORTA RISK",
            "low_risk": "DUSUK RISK"
        }
        parts.append(
            f"PIN-A3 Sonuc: {verdict_tr.get(verdict, verdict)} "
            f"(Skor: {score:.2f})"
        )

        return " | ".join(parts)

    def _build_empty_results(self, reason: str) -> dict:
        """Hata durumunda dönen boş sonuç."""
        return {
            "ela_heatmap": None,
            "global_stats": None,
            "grid_analysis": None,
            "uniformity": None,
            "manipulation_regions": [],
            "image_dimensions": None,
            "score_breakdown": {},
            "error_reason": reason
        }