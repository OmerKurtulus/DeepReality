"""
DeepReality — PIN-A3: Error Level Analysis (ELA)
================================================

Function:
    Detects manipulated regions and generative traces by analysing
    differences in JPEG compression level across the image.

Algorithm:
    1. Re-save the original at a known JPEG quality (Q=90)
    2. Compare the original with the re-saved copy pixel by pixel
    3. Analyse the resulting difference map (the ELA map):
       a. Global statistics (mean, standard deviation, maximum)
       b. Regional analysis on an 8x8 grid — per-region ELA mean
       c. Uniformity score — standard deviation of the regional means
       d. Anomaly detection — regions with abnormal ELA level
    4. Write the ELA heatmap image

Interpretation:
    - Authentic photograph: natural ELA variation, moderate uniformity
    - Generated imagery:    highly uniform ELA (every pixel was produced
                            in a single synthesis pass)
    - Manipulated content:  pronounced hotspots over the edited regions

Technology:
    OpenCV, Pillow, NumPy

Output:
    ela_heatmap          : str   — path of the ELA heatmap image
    manipulation_regions : list  — detected anomalous regions
    uniformity_score     : float — 0 (highly uniform) to high (natural)
    global_stats         : dict  — ELA statistics
    grid_analysis        : dict  — regional analysis results
    score                : float — 0.0 (clean) to 1.0 (manipulated)

Author: Omer Faruk Kurtulus
"""

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# HEIC/HEIF support for iPhone photographs
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
    PIN-A3: Error Level Analysis.

    Detects manipulation and generative traces through differences in
    JPEG compression level.
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
    # MAIN ANALYSIS
    # ════════════════════════════════════════════════════════════════

    def analyze(self, file_path: str) -> dict:
        """
        ELA analysis pipeline.

        Steps:
            1. Load the image and identify its source format
            2. Re-save it in memory at JPEG Q=90
            3. Compute the pixel difference -> ELA map
            4. Global statistics
            5. Regional grid analysis (8x8)
            6. Uniformity score
            7. Anomaly detection
            8. Write the ELA heatmap
            9. Format-aware scoring
        """

        # ── 1. Load the image and identify its format ──
        original = self._load_image(file_path)
        if original is None:
            return {
                "results": self._build_empty_results("Gorsel yuklenemedi"),
                "score": 0.0,
                "verdict": "error",
                "details": "Gorsel yuklenemedi veya desteklenmeyen format."
            }

        # Format identification is critical to ELA reliability
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

        # ── 5. Regional grid analysis ──
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

        # ── 9. Format-aware scoring ──
        score, score_breakdown = self._calculate_score(
            global_stats=global_stats,
            uniformity=uniformity,
            hotspots=hotspots,
            source_format=source_format
        )

        # Signal reliability check: insufficient signal yields verdict=no_data
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
        """Load the image in OpenCV (BGR) form."""
        try:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is None:
                # OpenCV occasionally fails on Unicode paths
                # PIL ile dene
                pil_img = Image.open(file_path).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            self.errors.append(f"Gorsel yukleme hatasi: {e}")
            return None

    def _detect_source_format(self, file_path: str) -> dict:
        """
        Identify the source format and its compression class.

        What matters for ELA is not JPEG versus PNG but LOSSY versus
        LOSSLESS:

        LOSSY formats (a compression history EXISTS):
            - JPEG (.jpg/.jpeg)      — DCT-based lossy compression
            - HEIC/HEIF              — HEVC-based lossy compression
            - WebP lossy (.webp)     — VP8-based lossy compression
            -> the ELA uniformity signal is RELIABLE
            -> different areas were compressed differently, producing
               natural variation

        LOSSLESS formats (NO compression history):
            - PNG (.png)             — lossless
            - BMP (.bmp)             — uncompressed
            - TIFF (.tiff/.tif)      — usually lossless
            - GIF (.gif)             — palette based
            -> the ELA uniformity signal is UNRELIABLE
            -> with no lossy history, authentic and generated images
               both yield similarly uniform ELA

        Note: iPhone photographs are stored as HEIC, which is lossy, so
        ELA behaves correctly on them. Converting HEIC to PNG, however,
        erases the compression history and renders ELA unreliable.
        """
        ext = Path(file_path).suffix.lower()

        # Lossy formats — compression history present
        lossy_formats = {
            ".jpg", ".jpeg", ".jpe", ".jfif",  # JPEG
            ".heic", ".heif",                    # HEIC (iPhone)
            ".webp",                             # WebP (genelde lossy)
        }

        # Lossless formats — no compression history
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
        Compute the ELA map.

        Algorithm:
            1. Re-save the original in memory as JPEG Q=90
            2. Load the re-saved copy
            3. Compute the absolute per-pixel difference
            4. Scale the difference by the amplification factor

        Returns:
            The ELA map (grayscale, 0-255), or None on failure.
        """
        try:
            # Convert to a PIL image for JPEG encoding
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(original_rgb)

            # Re-encode as JPEG in memory
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=self.resave_quality)
            buffer.seek(0)

            # Load the re-encoded copy
            resaved = Image.open(buffer)
            resaved_np = np.array(resaved).astype(np.float64)
            original_float = original_rgb.astype(np.float64)

            # Per-channel pixel difference
            diff = np.abs(original_float - resaved_np)

            # Collapse the channels to a grayscale ELA map
            ela_gray = np.mean(diff, axis=2)

            # Amplify so that small differences become visible
            ela_amplified = ela_gray * self.amp_scale

            # Clamp to the 0-255 range
            ela_map = np.clip(ela_amplified, 0, 255).astype(np.uint8)

            return ela_map

        except Exception as e:
            self.errors.append(f"ELA hesaplama hatasi: {e}")
            return None

    # ════════════════════════════════════════════════════════════════
    # STATISTICAL ANALYSIS
    # ════════════════════════════════════════════════════════════════

    def _compute_global_stats(self, ela_map: np.ndarray) -> dict:
        """
        Compute global statistics over the ELA map.

        Key metrics:
            - mean:     average ELA level (low = heavily compressed,
                        high = recently written)
            - std:      standard deviation (low = uniform, high = varied)
            - max:      maximum ELA pixel value
            - skewness: distribution asymmetry
            - energy:   total energy (sum of squares per pixel)
        """
        flat = ela_map.flatten().astype(np.float64)

        mean = float(np.mean(flat))
        std = float(np.std(flat))
        max_val = float(np.max(flat))
        min_val = float(np.min(flat))
        median = float(np.median(flat))

        # Skewness — shape of the distribution
        if std > 0:
            skewness = float(np.mean(((flat - mean) / std) ** 3))
        else:
            skewness = 0.0

        # Energy — overall ELA intensity
        energy = float(np.mean(flat ** 2))

        # Percentiles
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
        Partition the ELA map into an N x N grid and analyse each cell.

        Per region:
            - mean:     average ELA level
            - std:      standard deviation
            - max:      maximum value
            - position: (row, col) within the grid

        This regional view is what makes localisation possible: an
        edited area exhibits a different ELA level from its
        surroundings, which is invisible in the global statistics.
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
        Measure how uniform the regional ELA values are.

        Uniformity score = standard deviation of the regional means.

        Generated images tend towards highly uniform ELA because:
        - every pixel is produced at once by the same model
        - there is no natural compression variation
        - the ELA level of each region closely matches the others

        Authentic photographs exhibit natural variation:
        - different textures (sky, foliage, masonry) yield different ELA
        - edges and flat areas compress differently

        Manipulated images exhibit high variation:
        - edited regions produce a markedly different ELA level
        """
        region_means = [r["mean"] for r in grid_analysis]
        region_stds = [r["std"] for r in grid_analysis]

        uniformity_score = float(np.std(region_means))
        mean_of_means = float(np.mean(region_means))
        std_of_stds = float(np.std(region_stds))

        # Coefficient of variation — normalised dispersion
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
        Detect anomalous regions in the ELA map, in BOTH directions.

        HOTSPOT (elevated ELA):
            Newly inserted content that has not yet been compressed.
            The JPEG re-save produces a large difference, hence high ELA.

        COLDSPOT (suppressed ELA):
            Heavily compressed content pasted in from a lower-quality
            source. Having already lost its detail, it changes little on
            re-save and therefore sits well below its surroundings.

        Method: robust detection based on the Median Absolute Deviation,
        which is resistant to the outliers the anomalies themselves
        introduce:

            median     = median of the regional means
            MAD        = median(|region - median|)
            robust_std = MAD * 1.4826
            hotspot:   region_mean > median + k_hot  * robust_std
            coldspot:  region_mean < median - k_cold * robust_std
                       AND (median - region_mean) >= min_absolute_deviation

        Additional constraint for coldspots:
            A statistical deviation alone is insufficient. On uniform
            images the MAD becomes very small, so natural variation
            (smooth skin, open sky) can cross the sigma threshold. The
            minimum absolute difference check filters those out:
              - skin versus background: ~10-15 units -> NATURAL
              - genuine manipulation:   ~40-50 units -> COLDSPOT
        """
        region_means = np.array([r["mean"] for r in grid_analysis])

        # Robust istatistikler
        median_val = float(np.median(region_means))
        mad = float(np.median(np.abs(region_means - median_val)))
        mad = max(mad, 0.5)  # Minimum MAD
        robust_std = mad * 1.4826

        # Separate thresholds for hotspots and coldspots
        high_threshold = median_val + (self.hotspot_std * robust_std)
        low_threshold = median_val - (self.coldspot_std * robust_std)

        anomalies = []
        for region in grid_analysis:
            mean_val = region["mean"]

            if mean_val > high_threshold:
                # HOTSPOT — freshly inserted content
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
                # Coldspot candidate — requires the additional check
                absolute_diff = median_val - mean_val

                # Minimum absolute difference check:
                # naturally low-ELA regions (skin, sky) usually sit only
                # 10-15 units below the median, whereas genuine
                # manipulation produces a gap of 40 units or more.
                if absolute_diff < self.coldspot_min_abs:
                    continue  # Natural variation — skip

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
        Write the ELA map as a colour heatmap.

        Colormap: JET (blue = low ELA, red = high ELA)
        Output:   outputs/{stem}_ELA_heatmap.png
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
        Derive the pin score from the ELA analysis.

        KEY DESIGN DECISION
        -------------------
        ELA uniformity ALONE cannot separate generated from authentic
        imagery.

        Why:
        - iPhone computational photography makes authentic photographs
          produce uniform ELA
        - modern generators (GPT-4o, Imagen 3) synthesise realistic
          texture with natural ELA variation
        - the problem is present in both lossy and lossless formats

        Therefore:
        1. UNIFORMITY -> WEAK supporting signal (capped at 0.25).
           Never decisive on its own; it is combined with the other
           pins at the adjudication and ensemble stages.
        2. ANOMALIES  -> STRONG manipulation signal (capped at 0.85).
           This is the real strength of ELA and it is format agnostic.

        NO FORMAT PENALTY is applied, because both authentic and
        generated images occur in every format; the format alone
        carries no information about provenance.
        """
        breakdown = {}
        global_mean = global_stats["mean"]
        global_std = global_stats["std"]

        # Format information, reported but not scored
        comp_type = source_format.get("compression_type", "unknown") if source_format else "unknown"

        # ═══════════════════════════════════════════════
        # STEP 0: SIGNAL RELIABILITY CHECK
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
        # STEP 1: UNIFORMITY SIGNAL
        # Weak supporting signal — capped at 0.25
        # NOT decisive on its own
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
        # STEP 2: ANOMALY SIGNAL (manipulation detection)
        # Hotspots (elevated ELA) and coldspots (suppressed ELA)
        # The real strength of ELA — independent of format
        # ═══════════════════════════════════════════════
        anomaly_count = len(hotspots)  # hotspots now holds every anomaly type
        total_regions = self.grid_size * self.grid_size
        anomaly_ratio = anomaly_count / total_regions if total_regions > 0 else 0

        hotspot_count = sum(1 for h in hotspots if h.get("type") == "hotspot")
        coldspot_count = sum(1 for h in hotspots if h.get("type") == "coldspot")

        if anomaly_count == 0:
            hotspot_signal = 0.0
        elif anomaly_ratio < 0.05:
            hotspot_signal = 0.50  # Few — localised manipulation
        elif anomaly_ratio < 0.15:
            hotspot_signal = 0.70  # Moderate — pronounced manipulation
        elif anomaly_ratio < 0.30:
            hotspot_signal = 0.60  # Many — wholesale editing
        else:
            hotspot_signal = 0.40  # Widespread — differing compression

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
        # FINAL SCORE
        # Hotspot varsa → hotspot dominant
        # No anomalies -> weak uniformity signal only
        # ═══════════════════════════════════════════════
        if hotspot_signal > 0:
            # Manipulation detected -> anomaly signal plus uniformity bonus
            final_score = hotspot_signal + (uniformity_signal * 0.3)
        else:
            # No manipulation -> weak uniformity signal only
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
        """Map the numeric score onto a verdict band."""
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
        """Produce the natural-language explanation of the analysis."""
        parts = []

        fmt_ext = source_format.get("file_extension", "?") if source_format else "?"
        comp_type = source_format.get("compression_type", "?") if source_format else "?"

        parts.append(
            f"ELA Analizi tamamlandi (format: {fmt_ext.upper()}, "
            f"sikistirma: {comp_type}). "
            f"Global ELA: ortalama={global_stats['mean']:.1f}, "
            f"std={global_stats['std']:.1f}, max={global_stats['max']:.1f}."
        )

        # Signal reliability
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
        """Empty result returned when the analysis cannot proceed."""
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