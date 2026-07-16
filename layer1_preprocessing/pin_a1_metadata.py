"""
DeepReality — PIN-A1: EXIF / Metadata Analizi
═══════════════════════════════════════════════

İşlev:
    Görselin metadata bilgilerini çıkarır ve analiz eder.
    AI üretim araçlarının bıraktığı metadata izlerini tespit eder.
    Kamera, GPS, tarih bilgisi varlığını kontrol ederek
    görselin gerçek mi yoksa yapay mı üretildiğine dair sinyal üretir.

Teknoloji:
    Pillow (PIL), struct (binary EXIF parsing)

Çıktı:
    metadata_score (0.0-1.0), source_tool, extracted_metadata, signals

Mantık:
    - AI aracı imzası bulunursa → yüksek skor (yapay üretim şüphesi)
    - Kamera verisi yoksa → orta sinyal
    - Metadata tamamen boşsa → şüpheli (kasıtlı silinmiş olabilir)
    - Kamera + GPS + tarih varsa → düşük skor (muhtemelen gerçek fotoğraf)
"""

import struct
import sys
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Projenin kök dizinini Python path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.base_pin import BasePin
from config.settings import METADATA_CONFIG


class PinA1Metadata(BasePin):
    """
    PIN-A1: EXIF/Metadata Analizi

    Görselin tüm metadata katmanlarını tarar:
    1. Standart EXIF verileri (kamera, lens, ayarlar)
    2. GPS koordinatları
    3. Yazılım / araç bilgisi
    4. AI üretim aracı imzaları (SD, MJ, DALL-E, Firefly, vb.)
    5. XMP / IPTC embedded text içinde AI pattern arama
    6. Metadata bütünlük analizi (tamamen silinmiş mi?)
    """

    def __init__(self):
        super().__init__(
            pin_id="PIN-A1",
            pin_name="EXIF/Metadata Analysis",
            layer=1
        )
        self.ai_signatures = METADATA_CONFIG["ai_tool_signatures"]
        self.camera_fields = METADATA_CONFIG["camera_fields"]
        self.gps_fields = METADATA_CONFIG["gps_fields"]
        self.weights = METADATA_CONFIG["weights"]
        self.thresholds = METADATA_CONFIG["thresholds"]
        self.c2pa_markers = METADATA_CONFIG["c2pa_binary_markers"]
        self.c2pa_issuers = METADATA_CONFIG["c2pa_issuers"]
        self.ai_dimensions = METADATA_CONFIG["ai_typical_dimensions"]

    def analyze(self, file_path: str) -> dict:
        """
        Ana analiz metodu. Görselin metadata'sını çıkarır ve değerlendirir.
        """
        file_path = Path(file_path)

        # ── 1. Metadata Extraction ──────────────────────────────────
        exif_data = self._extract_exif(file_path)
        basic_info = self._extract_basic_info(file_path)
        raw_text_chunks = self._extract_raw_text(file_path)

        # ── 2. Signal Analysis ──────────────────────────────────────
        ai_detection = self._detect_ai_signatures(exif_data, raw_text_chunks)
        c2pa_detection = self._detect_c2pa_binary(file_path)
        camera_analysis = self._analyze_camera_data(exif_data)
        gps_analysis = self._analyze_gps_data(exif_data)
        datetime_analysis = self._analyze_datetime(exif_data)
        software_analysis = self._analyze_software_field(exif_data)
        metadata_completeness = self._analyze_metadata_completeness(exif_data)
        dimension_analysis = self._analyze_dimensions(basic_info)
        compression_analysis = self._analyze_compression_ratio(basic_info)

        # ── 3. Score Calculation ────────────────────────────────────
        score, signal_breakdown = self._calculate_score(
            ai_detection=ai_detection,
            c2pa_detection=c2pa_detection,
            camera_analysis=camera_analysis,
            gps_analysis=gps_analysis,
            datetime_analysis=datetime_analysis,
            software_analysis=software_analysis,
            metadata_completeness=metadata_completeness,
            dimension_analysis=dimension_analysis,
            compression_analysis=compression_analysis
        )

        # ── 4. Verdict ─────────────────────────────────────────────
        verdict = self._determine_verdict(score)

        # ── 5. Türkçe Açıklama ─────────────────────────────────────
        details = self._generate_details(
            score, verdict, ai_detection, c2pa_detection, camera_analysis,
            gps_analysis, software_analysis, metadata_completeness,
            dimension_analysis, compression_analysis
        )

        # ── 6. Sonuç Paketi ────────────────────────────────────────
        # source_tool: C2PA issuer varsa onu kullan, yoksa AI signature
        source_tool = (
            c2pa_detection.get("issuer_tool") or
            ai_detection.get("detected_tool") or
            None
        )

        results = {
            "basic_info": basic_info,
            "exif_fields_found": len(exif_data),
            "exif_data": self._sanitize_exif_for_json(exif_data),
            "signals": {
                "ai_detection": ai_detection,
                "c2pa_detection": c2pa_detection,
                "camera_data": camera_analysis,
                "gps_data": gps_analysis,
                "datetime": datetime_analysis,
                "software": software_analysis,
                "metadata_completeness": metadata_completeness,
                "dimension_analysis": dimension_analysis,
                "compression_analysis": compression_analysis
            },
            "score_breakdown": signal_breakdown,
            "source_tool": source_tool,
            "metadata_score": round(score, 4)
        }

        return {
            "results": results,
            "score": score,
            "verdict": verdict,
            "details": details
        }

    # ════════════════════════════════════════════════════════════════
    # EXTRACTION METHODS
    # ════════════════════════════════════════════════════════════════

    def _extract_basic_info(self, file_path: Path) -> dict:
        """Temel görsel bilgilerini çıkarır (format, boyut, mod)."""
        try:
            with Image.open(file_path) as img:
                return {
                    "format": img.format,
                    "mode": img.mode,
                    "width": img.size[0],
                    "height": img.size[1],
                    "file_size_bytes": file_path.stat().st_size,
                    "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
                }
        except Exception as e:
            self.errors.append(f"Temel bilgi çıkarma hatası: {e}")
            return {}

    def _extract_exif(self, file_path: Path) -> dict:
        """
        Pillow ile EXIF verilerini çıkarır.
        GPS ve IFD alt etiketlerini de çözümler.
        """
        exif_data = {}
        try:
            with Image.open(file_path) as img:
                raw_exif = img.getexif()
                if not raw_exif:
                    return exif_data

                # Ana EXIF etiketlerini çözümle
                for tag_id, value in raw_exif.items():
                    tag_name = TAGS.get(tag_id, f"Unknown_{tag_id}")
                    exif_data[tag_name] = self._safe_value(value)

                # IFD (Sub-EXIF) bloklarını çözümle
                for ifd_id in raw_exif.get_ifd(0x8769) or {}:
                    tag_name = TAGS.get(ifd_id, f"ExifIFD_{ifd_id}")
                    value = raw_exif.get_ifd(0x8769).get(ifd_id)
                    if value is not None:
                        exif_data[tag_name] = self._safe_value(value)

                # GPS IFD bloğunu çözümle
                gps_ifd = raw_exif.get_ifd(0x8825)
                if gps_ifd:
                    for gps_tag_id, value in gps_ifd.items():
                        gps_tag_name = GPSTAGS.get(gps_tag_id, f"GPS_{gps_tag_id}")
                        exif_data[gps_tag_name] = self._safe_value(value)

                # Pillow info dict (PNG tEXt chunks, JPEG COM segments vb.)
                if hasattr(img, "info") and img.info:
                    for key, value in img.info.items():
                        if isinstance(key, str) and key.lower() not in ["exif", "icc_profile"]:
                            exif_data[f"info_{key}"] = self._safe_value(value)

        except Exception as e:
            self.errors.append(f"EXIF çıkarma hatası: {e}")

        return exif_data

    def _extract_raw_text(self, file_path: Path) -> list[str]:
        """
        Dosyanın binary içeriğinden okunabilir metin parçalarını çıkarır.
        XMP, IPTC, tEXt chunk'ları ve gömülü prompt'ları yakalar.
        PNG tEXt, iTXt chunk'ları ve JPEG COM/APP segmentlerini okur.
        """
        text_chunks = []
        try:
            with open(file_path, "rb") as f:
                raw = f.read()

            # ── XMP Metadata ──
            xmp_start = raw.find(b"<x:xmpmeta")
            if xmp_start == -1:
                xmp_start = raw.find(b"<rdf:RDF")
            if xmp_start != -1:
                xmp_end = raw.find(b"</x:xmpmeta>", xmp_start)
                if xmp_end == -1:
                    xmp_end = raw.find(b"</rdf:RDF>", xmp_start)
                if xmp_end != -1:
                    xmp_text = raw[xmp_start:xmp_end + 20].decode("utf-8", errors="ignore")
                    text_chunks.append(xmp_text)

            # ── PNG tEXt / iTXt Chunks ──
            if raw[:4] == b"\x89PNG":
                offset = 8  # PNG signature length
                while offset < len(raw) - 8:
                    try:
                        chunk_len = struct.unpack(">I", raw[offset:offset + 4])[0]
                        chunk_type = raw[offset + 4:offset + 8].decode("ascii", errors="ignore")
                        chunk_data = raw[offset + 8:offset + 8 + chunk_len]

                        if chunk_type in ("tEXt", "iTXt", "zTXt"):
                            decoded = chunk_data.decode("utf-8", errors="ignore")
                            text_chunks.append(decoded)

                        offset += 12 + chunk_len  # 4(len) + 4(type) + data + 4(crc)
                    except (struct.error, IndexError):
                        break

            # ── JPEG COM (Comment) Segment ──
            if raw[:2] == b"\xff\xd8":
                idx = 2
                while idx < len(raw) - 4:
                    if raw[idx] != 0xFF:
                        break
                    marker = raw[idx + 1]
                    if marker == 0xFE:  # COM segment
                        seg_len = struct.unpack(">H", raw[idx + 2:idx + 4])[0]
                        comment = raw[idx + 4:idx + 2 + seg_len].decode("utf-8", errors="ignore")
                        text_chunks.append(comment)
                    if marker in (0xDA, 0xD9):  # SOS veya EOI → dur
                        break
                    try:
                        seg_len = struct.unpack(">H", raw[idx + 2:idx + 4])[0]
                        idx += 2 + seg_len
                    except (struct.error, IndexError):
                        break

        except Exception as e:
            self.errors.append(f"Raw text çıkarma hatası: {e}")

        return text_chunks

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS METHODS
    # ════════════════════════════════════════════════════════════════

    def _detect_ai_signatures(self, exif_data: dict, raw_text_chunks: list[str]) -> dict:
        """
        Metadata ve raw text içinde AI üretim aracı imzalarını arar.
        Birden fazla araç bulunursa en yüksek güvenilirlikli olanı seçer.
        """
        detections = []

        # Tüm aranacak metinleri birleştir (küçük harfe çevir)
        searchable_texts = []

        # EXIF alanlarından
        for key, value in exif_data.items():
            if isinstance(value, str):
                searchable_texts.append(value.lower())

        # Raw text chunk'larından
        for chunk in raw_text_chunks:
            if isinstance(chunk, str):
                searchable_texts.append(chunk.lower())

        combined_text = " ||| ".join(searchable_texts)

        # Her AI aracının imzalarını kontrol et
        for tool_name, signatures in self.ai_signatures.items():
            confidence = 0.0
            matched_patterns = []

            # Software patterns
            for pattern in signatures.get("software_patterns", []):
                if pattern.lower() in combined_text:
                    confidence = max(confidence, 0.95)
                    matched_patterns.append(f"software: '{pattern}'")

            # Parameter fields (Stable Diffusion özel)
            for field in signatures.get("parameter_fields", []):
                # EXIF key olarak kontrol
                if field.lower() in [k.lower() for k in exif_data.keys()]:
                    confidence = max(confidence, 0.98)
                    matched_patterns.append(f"param_field: '{field}'")
                # info_ prefix ile (PNG tEXt)
                if f"info_{field}".lower() in [k.lower() for k in exif_data.keys()]:
                    confidence = max(confidence, 0.98)
                    matched_patterns.append(f"info_field: '{field}'")

            # Comment patterns
            for pattern in signatures.get("comment_patterns", []):
                if pattern.lower() in combined_text:
                    confidence = max(confidence, 0.85)
                    matched_patterns.append(f"comment: '{pattern}'")

            # Description patterns
            for pattern in signatures.get("description_patterns", []):
                if pattern.lower() in combined_text:
                    confidence = max(confidence, 0.80)
                    matched_patterns.append(f"description: '{pattern}'")

            # XMP patterns
            for pattern in signatures.get("xmp_patterns", []):
                if pattern.lower() in combined_text:
                    confidence = max(confidence, 0.90)
                    matched_patterns.append(f"xmp: '{pattern}'")

            if confidence > 0:
                detections.append({
                    "tool": tool_name,
                    "confidence": round(confidence, 2),
                    "matched_patterns": matched_patterns
                })

        # Sonuçları güvenilirliğe göre sırala
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        if detections:
            best = detections[0]
            return {
                "ai_detected": True,
                "detected_tool": best["tool"],
                "confidence": best["confidence"],
                "matched_patterns": best["matched_patterns"],
                "all_detections": detections
            }
        else:
            return {
                "ai_detected": False,
                "detected_tool": None,
                "confidence": 0.0,
                "matched_patterns": [],
                "all_detections": []
            }

    def _analyze_camera_data(self, exif_data: dict) -> dict:
        """Kamera bilgisi varlığını analiz eder."""
        found_fields = []
        for field in self.camera_fields:
            if field in exif_data and exif_data[field]:
                found_fields.append(field)

        has_camera = len(found_fields) > 0
        camera_richness = len(found_fields) / len(self.camera_fields)

        camera_info = {}
        if has_camera:
            camera_info = {
                "make": exif_data.get("Make", ""),
                "model": exif_data.get("Model", ""),
                "lens": exif_data.get("LensModel", ""),
                "focal_length": str(exif_data.get("FocalLength", "")),
                "f_number": str(exif_data.get("FNumber", "")),
                "iso": str(exif_data.get("ISOSpeedRatings", exif_data.get("ISO", ""))),
            }

        return {
            "has_camera_data": has_camera,
            "fields_found": found_fields,
            "field_count": len(found_fields),
            "richness": round(camera_richness, 2),
            "camera_info": camera_info
        }

    def _analyze_gps_data(self, exif_data: dict) -> dict:
        """GPS koordinat bilgisi varlığını kontrol eder."""
        found_gps = []
        for field in self.gps_fields:
            if field in exif_data and exif_data[field]:
                found_gps.append(field)

        has_gps = len(found_gps) >= 2  # En az lat+lon olmalı

        coords = {}
        if has_gps:
            lat = exif_data.get("GPSLatitude")
            lon = exif_data.get("GPSLongitude")
            lat_ref = exif_data.get("GPSLatitudeRef", "N")
            lon_ref = exif_data.get("GPSLongitudeRef", "E")
            if lat and lon:
                coords = {
                    "latitude": str(lat),
                    "latitude_ref": str(lat_ref),
                    "longitude": str(lon),
                    "longitude_ref": str(lon_ref)
                }

        return {
            "has_gps": has_gps,
            "fields_found": found_gps,
            "coordinates": coords
        }

    def _analyze_datetime(self, exif_data: dict) -> dict:
        """Tarih/zaman bilgisi varlığını kontrol eder."""
        datetime_fields = [
            "DateTime", "DateTimeOriginal", "DateTimeDigitized",
            "CreateDate", "ModifyDate"
        ]
        found = {}
        for field in datetime_fields:
            if field in exif_data and exif_data[field]:
                found[field] = str(exif_data[field])

        return {
            "has_datetime": len(found) > 0,
            "fields_found": found
        }

    def _analyze_software_field(self, exif_data: dict) -> dict:
        """
        Software/ProcessingSoftware alanını analiz eder.
        Fotoğraf düzenleme vs AI üretim ayrımı yapar.
        """
        software = exif_data.get("Software", "") or ""
        processing = exif_data.get("ProcessingSoftware", "") or ""
        creator_tool = exif_data.get("info_CreatorTool", "") or ""
        combined = f"{software} {processing} {creator_tool}".strip().lower()

        # Bilinen fotoğraf düzenleme yazılımları (şüpheli DEĞİL)
        known_editors = [
            "adobe photoshop", "lightroom", "capture one",
            "gimp", "affinity photo", "darktable", "rawtherapee", "picasa",
            "snapseed", "vsco", "instagram", "samsung", "apple",
            "google photos", "huawei", "xiaomi", "oppo", "vivo"
        ]

        is_known_editor = any(editor in combined for editor in known_editors)

        # AI içerik düzenleme özellikleri (şüpheli)
        ai_edit_features = [
            "generative fill", "neural filters", "ai enhance",
            "ai upscale", "super resolution", "content-aware"
        ]
        has_ai_editing = any(feat in combined for feat in ai_edit_features)

        return {
            "software_value": combined if combined else None,
            "is_known_editor": is_known_editor,
            "has_ai_editing_features": has_ai_editing,
            "is_empty": len(combined) == 0
        }

    def _analyze_metadata_completeness(self, exif_data: dict) -> dict:
        """
        Metadata'nın ne kadar dolu olduğunu analiz eder.
        Tamamen boş metadata şüphelidir (kasıtlı silinmiş olabilir).
        """
        total_fields = len(exif_data)

        # info_ prefix'li alanları say (PNG tEXt, JPEG COM vb.)
        info_fields = sum(1 for k in exif_data if k.startswith("info_"))

        # Standart EXIF alanları
        standard_fields = total_fields - info_fields

        if total_fields == 0:
            completeness = "empty"
        elif standard_fields <= 3:
            completeness = "minimal"
        elif standard_fields <= 10:
            completeness = "partial"
        else:
            completeness = "rich"

        return {
            "total_fields": total_fields,
            "standard_exif_fields": standard_fields,
            "info_fields": info_fields,
            "completeness": completeness,
            "is_stripped": total_fields == 0
        }

    def _detect_c2pa_binary(self, file_path: Path) -> dict:
        """
        Dosyanın binary içeriğinde C2PA/JUMBF marker'larını arar.
        ChatGPT, DALL-E, Sora gibi araçlar bu marker'ları gömer.
        Metadata tamamen silinmiş görünen dosyalarda bile
        C2PA binary izleri kalabilir.
        """
        found_markers = []
        issuer_tool = None
        confidence = 0.0

        try:
            with open(file_path, "rb") as f:
                raw = f.read()

            # C2PA marker'larını ara
            for marker in self.c2pa_markers:
                idx = raw.find(marker)
                if idx != -1:
                    found_markers.append({
                        "marker": marker.decode("utf-8", errors="ignore"),
                        "offset": idx
                    })

            # C2PA issuer'ını tespit et (OpenAI, Adobe, vb.)
            for issuer_bytes, tool_name in self.c2pa_issuers.items():
                if issuer_bytes in raw:
                    issuer_tool = tool_name
                    break

            # Güvenilirlik hesapla
            if found_markers:
                # Temel marker bulundu
                confidence = min(0.60 + len(found_markers) * 0.08, 0.98)
                # Issuer da tespit edildiyse güvenilirlik artır
                if issuer_tool:
                    confidence = min(confidence + 0.15, 0.99)

        except Exception as e:
            self.errors.append(f"C2PA binary tarama hatasi: {e}")

        return {
            "c2pa_detected": len(found_markers) > 0,
            "marker_count": len(found_markers),
            "found_markers": found_markers,
            "issuer_tool": issuer_tool,
            "confidence": round(confidence, 2)
        }

    def _analyze_dimensions(self, basic_info: dict) -> dict:
        """
        Görsel boyutlarının AI üretim araçlarının tipik çıktı boyutlarıyla
        eşleşip eşleşmediğini kontrol eder.
        AI araçları genelde 64'ün katı, kare veya belirli oranlar üretir.
        """
        width = basic_info.get("width", 0)
        height = basic_info.get("height", 0)

        if width == 0 or height == 0:
            return {"is_ai_dimension": False, "details": "Boyut bilgisi yok"}

        # Tam eşleşme kontrolü
        exact_match = (width, height) in self.ai_dimensions

        # 64'ün katı mı? (AI modelleri 64 veya 8'in katlarında çalışır)
        divisible_by_64 = (width % 64 == 0) and (height % 64 == 0)

        # Kare mi?
        is_square = width == height

        # 2'nin kuvveti mi? (256, 512, 1024, 2048)
        def is_power_of_2(n):
            return n > 0 and (n & (n - 1)) == 0

        both_power_of_2 = is_power_of_2(width) and is_power_of_2(height)

        # Sinyal gücü
        signal = 0.0
        reasons = []

        if exact_match:
            signal = 0.8
            reasons.append(f"Bilinen AI cikti boyutu: {width}x{height}")
        elif both_power_of_2 and is_square:
            signal = 0.6
            reasons.append(f"Kare ve 2'nin kuvveti: {width}x{height}")
        elif divisible_by_64 and is_square:
            signal = 0.4
            reasons.append(f"Kare ve 64'un kati: {width}x{height}")
        elif divisible_by_64:
            signal = 0.2
            reasons.append(f"64'un kati: {width}x{height}")

        return {
            "is_ai_dimension": signal > 0.3,
            "exact_match": exact_match,
            "divisible_by_64": divisible_by_64,
            "is_square": is_square,
            "both_power_of_2": both_power_of_2,
            "signal": round(signal, 2),
            "reasons": reasons,
            "dimensions": f"{width}x{height}"
        }

    def _analyze_compression_ratio(self, basic_info: dict) -> dict:
        """
        Dosya boyutu ile piksel sayısı arasındaki oranı analiz eder.
        AI görseller genelde gerçek fotoğraflardan farklı sıkıştırma
        karakteristikleri gösterir:
        - Gerçek fotoğraf JPEG: genelde 2-8 bytes/piksel
        - AI üretimi PNG: genelde daha yüksek (sıkıştırılmamış)
        - AI üretimi JPEG: genelde daha düşük (fazla smooth alanlar)
        """
        width = basic_info.get("width", 0)
        height = basic_info.get("height", 0)
        file_size = basic_info.get("file_size_bytes", 0)
        fmt = basic_info.get("format", "").upper()

        if width == 0 or height == 0 or file_size == 0:
            return {"anomaly_detected": False, "details": "Yetersiz bilgi"}

        total_pixels = width * height
        bytes_per_pixel = file_size / total_pixels

        # Format bazlı beklenen aralıklar
        anomaly = False
        details = ""

        if fmt == "JPEG":
            # Gerçek fotoğraf JPEG: ~1.5-8 bpp
            # AI JPEG: genelde <1.0 bpp (çok smooth) veya >10 bpp (kalite 100)
            if bytes_per_pixel < 0.5:
                anomaly = True
                details = f"Asiri dusuk sikistirma orani ({bytes_per_pixel:.2f} B/px) — yapay uretim sinyali"
            elif bytes_per_pixel > 12.0:
                anomaly = True
                details = f"Asiri yuksek sikistirma orani ({bytes_per_pixel:.2f} B/px) — olagan disi"
        elif fmt == "PNG":
            # PNG sıkıştırma: AI görseller genelde 0.5-3.0 bpp
            # Gerçek fotoğraf PNG: genelde >3.0 bpp
            if bytes_per_pixel < 0.3:
                anomaly = True
                details = f"Cok dusuk PNG boyutu ({bytes_per_pixel:.2f} B/px) — olagan disi"

        if not details:
            details = f"Normal sikistirma orani ({bytes_per_pixel:.2f} B/px)"

        return {
            "anomaly_detected": anomaly,
            "bytes_per_pixel": round(bytes_per_pixel, 3),
            "total_pixels": total_pixels,
            "format": fmt,
            "details": details
        }

    # ════════════════════════════════════════════════════════════════
    # SCORING
    # ════════════════════════════════════════════════════════════════

    def _calculate_score(self, ai_detection: dict, c2pa_detection: dict,
                         camera_analysis: dict, gps_analysis: dict,
                         datetime_analysis: dict, software_analysis: dict,
                         metadata_completeness: dict, dimension_analysis: dict,
                         compression_analysis: dict) -> tuple[float, dict]:
        """
        İki katmanlı skorlama sistemi:

        KATMAN 1 — KESİN KANIT KURALLARI (Evidence Floor)
        Kriptografik veya yapısal kesin kanıt varsa minimum skor garanti edilir.
        Bu kurallar ağırlıklı toplamı OVERRIDE eder.

        Kurallar:
        - C2PA + bilinen AI issuer (OpenAI, Google, Adobe) → min 0.85
        - C2PA marker bulundu (issuer bilinmiyor)           → min 0.70
        - AI metadata imzası ≥ %90 güvenilirlik (SD params) → min 0.75
        - AI metadata imzası ≥ %80 güvenilirlik              → min 0.65

        KATMAN 2 — AĞIRLIKLI TOPLAM (Weighted Sum)
        Kesin kanıt yoksa veya floor'un altında kalırsa,
        tüm sinyallerin ağırlıklı toplamı kullanılır.

        Skor: 0.0 (temiz/gerçek) → 1.0 (kesin yapay üretim)
        """
        breakdown = {}

        # ── Katman 2: Ağırlıklı toplam (her zaman hesaplanır) ──

        # 1. AI imzası tespit edildi mi?
        if ai_detection["ai_detected"]:
            ai_score = ai_detection["confidence"]
            breakdown["ai_signature"] = {
                "weight": self.weights["ai_signature_detected"],
                "signal": ai_score,
                "contribution": round(ai_score * self.weights["ai_signature_detected"], 4)
            }
        else:
            breakdown["ai_signature"] = {
                "weight": self.weights["ai_signature_detected"],
                "signal": 0.0,
                "contribution": 0.0
            }

        # 2. C2PA/JUMBF binary marker bulundu mu?
        if c2pa_detection["c2pa_detected"]:
            c2pa_score = c2pa_detection["confidence"]
            breakdown["c2pa_detected"] = {
                "weight": self.weights["c2pa_detected"],
                "signal": c2pa_score,
                "contribution": round(c2pa_score * self.weights["c2pa_detected"], 4)
            }
        else:
            breakdown["c2pa_detected"] = {
                "weight": self.weights["c2pa_detected"],
                "signal": 0.0,
                "contribution": 0.0
            }

        # 3. Kamera verisi yok mu?
        if not camera_analysis["has_camera_data"]:
            cam_signal = 1.0
        else:
            cam_signal = max(0.0, 1.0 - camera_analysis["richness"] * 1.5)
        breakdown["no_camera_data"] = {
            "weight": self.weights["no_camera_data"],
            "signal": round(cam_signal, 2),
            "contribution": round(cam_signal * self.weights["no_camera_data"], 4)
        }

        # 4. GPS verisi yok mu?
        gps_signal = 0.0 if gps_analysis["has_gps"] else 1.0
        breakdown["no_gps_data"] = {
            "weight": self.weights["no_gps_data"],
            "signal": gps_signal,
            "contribution": round(gps_signal * self.weights["no_gps_data"], 4)
        }

        # 5. Tarih bilgisi yok mu?
        dt_signal = 0.0 if datetime_analysis["has_datetime"] else 1.0
        breakdown["no_datetime"] = {
            "weight": self.weights["no_datetime"],
            "signal": dt_signal,
            "contribution": round(dt_signal * self.weights["no_datetime"], 4)
        }

        # 6. Yazılım alanı şüpheli mi?
        sw_signal = 0.0
        if software_analysis["has_ai_editing_features"]:
            sw_signal = 0.6
        elif software_analysis["is_empty"] and not camera_analysis["has_camera_data"]:
            sw_signal = 0.5
        breakdown["software_suspicious"] = {
            "weight": self.weights["software_suspicious"],
            "signal": round(sw_signal, 2),
            "contribution": round(sw_signal * self.weights["software_suspicious"], 4)
        }

        # 7. Metadata tamamen silinmiş mi?
        strip_signal = 1.0 if metadata_completeness["is_stripped"] else 0.0
        if metadata_completeness["completeness"] == "minimal":
            strip_signal = 0.5
        breakdown["metadata_stripped"] = {
            "weight": self.weights["metadata_stripped"],
            "signal": strip_signal,
            "contribution": round(strip_signal * self.weights["metadata_stripped"], 4)
        }

        # 8. AI tipik boyutları
        dim_signal = dimension_analysis.get("signal", 0.0)
        breakdown["ai_dimensions"] = {
            "weight": self.weights["ai_dimensions"],
            "signal": dim_signal,
            "contribution": round(dim_signal * self.weights["ai_dimensions"], 4)
        }

        # 9. Sıkıştırma anomalisi
        comp_signal = 1.0 if compression_analysis["anomaly_detected"] else 0.0
        breakdown["compression_anomaly"] = {
            "weight": self.weights["compression_anomaly"],
            "signal": comp_signal,
            "contribution": round(comp_signal * self.weights["compression_anomaly"], 4)
        }

        # Ağırlıklı toplam
        weighted_sum = sum(item["contribution"] for item in breakdown.values())
        weighted_sum = max(0.0, min(1.0, weighted_sum))

        # ── Katman 1: Kesin Kanıt Kuralları (Evidence Floor) ──
        evidence_floor = 0.0
        evidence_rule = None

        # ── Katman 1: Dinamik Kesin Kanıt Skorlaması ──
        #
        # Kesin kanıt bulunduğunda, skor kanıtın kendi güvenilirliğinden
        # türer ve destekleyici sinyallerle artırılır.
        #
        # Formül:
        #   evidence_floor = kanıt_güvenilirliği × baz_çarpan
        #                  + destekleyici sinyallerden bonus
        #                  → min(toplam, 0.98)
        #
        # Destekleyici sinyaller (kesin kanıtı güçlendirir):
        #   - AI issuer tespit edildi    → +0.05
        #   - AI boyut eşleşmesi         → dim_signal × 0.03
        #   - Metadata tamamen boş       → +0.02
        #   - Kamera verisi yok          → +0.02

        evidence_floor = 0.0
        evidence_rule = None
        evidence_components = []

        # Destekleyici sinyalleri topla (kesin kanıt varsa kullanılır)
        support_bonus = 0.0
        if c2pa_detection.get("issuer_tool") is not None:
            support_bonus += 0.05
            evidence_components.append(f"AI kaynak ({c2pa_detection['issuer_tool']}): +0.05")
        if dimension_analysis.get("signal", 0) > 0.3:
            dim_bonus = round(dimension_analysis["signal"] * 0.03, 4)
            support_bonus += dim_bonus
            evidence_components.append(f"AI boyut ({dimension_analysis.get('dimensions','')}): +{dim_bonus}")
        if metadata_completeness["is_stripped"]:
            support_bonus += 0.02
            evidence_components.append("Metadata bos: +0.02")
        if not camera_analysis["has_camera_data"]:
            support_bonus += 0.02
            evidence_components.append("Kamera yok: +0.02")

        # Kural 1: C2PA kriptografik imza bulundu
        if c2pa_detection["c2pa_detected"]:
            base = c2pa_detection["confidence"] * 0.80
            evidence_floor = min(base + support_bonus, 0.98)
            evidence_rule = (
                f"C2PA imza (guvenilirlik: %{int(c2pa_detection['confidence']*100)}) "
                f"→ baz: {base:.3f} + destek: {support_bonus:.3f} = {evidence_floor:.3f}"
            )

        # Kural 2: AI metadata imzası (SD params, tool markers vb.)
        elif ai_detection["ai_detected"] and ai_detection["confidence"] >= 0.80:
            base = ai_detection["confidence"] * 0.75
            evidence_floor = min(base + support_bonus, 0.98)
            evidence_rule = (
                f"AI imza ({ai_detection['detected_tool']}, "
                f"guvenilirlik: %{int(ai_detection['confidence']*100)}) "
                f"→ baz: {base:.3f} + destek: {support_bonus:.3f} = {evidence_floor:.3f}"
            )

        # Nihai skor: floor ve weighted_sum'un büyüğü
        final_score = max(weighted_sum, evidence_floor)
        final_score = round(max(0.0, min(1.0, final_score)), 4)

        # Breakdown'a evidence bilgisini ekle
        breakdown["_scoring_method"] = {
            "weighted_sum": round(weighted_sum, 4),
            "evidence_floor": round(evidence_floor, 4),
            "evidence_rule": evidence_rule,
            "evidence_components": evidence_components,
            "final_score": final_score,
            "floor_applied": evidence_floor > weighted_sum
        }

        return final_score, breakdown

    def _determine_verdict(self, score: float) -> str:
        """Skora göre verdict belirler."""
        if score >= self.thresholds["high_risk"]:
            return "high_risk"
        elif score >= self.thresholds["medium_risk"]:
            return "medium_risk"
        else:
            return "low_risk"

    # ════════════════════════════════════════════════════════════════
    # REPORTING
    # ════════════════════════════════════════════════════════════════

    def _generate_details(self, score: float, verdict: str,
                          ai_detection: dict, c2pa_detection: dict,
                          camera_analysis: dict, gps_analysis: dict,
                          software_analysis: dict,
                          metadata_completeness: dict,
                          dimension_analysis: dict,
                          compression_analysis: dict) -> str:
        """Analiz sonucunun Türkçe açıklamasını üretir."""
        parts = []

        # C2PA tespiti (en güçlü sinyal)
        if c2pa_detection["c2pa_detected"]:
            issuer = c2pa_detection.get("issuer_tool", "bilinmeyen")
            count = c2pa_detection["marker_count"]
            conf = int(c2pa_detection["confidence"] * 100)
            parts.append(
                f"C2PA/JUMBF DIJITAL IMZA TESPIT EDILDI: {count} marker bulundu, "
                f"kaynak: {issuer}, guvenilirlik: %{conf}. "
                f"Bu dosya bir AI araci tarafindan uretildigini gosteren "
                f"kriptografik imza icermektedir."
            )
        else:
            parts.append("C2PA/JUMBF dijital imza bulunamadi.")

        # AI aracı tespiti
        if ai_detection["ai_detected"]:
            tool = ai_detection["detected_tool"].replace("_", " ").title()
            conf = int(ai_detection["confidence"] * 100)
            patterns = ", ".join(ai_detection["matched_patterns"][:3])
            parts.append(
                f"METADATA AI IMZASI TESPIT EDILDI: {tool} araci ile uretilmis "
                f"olma olasiligi %{conf}. Eslesme: [{patterns}]."
            )
        else:
            parts.append("Metadata icerisinde bilinen AI uretim araci imzasi bulunamadi.")

        # Kamera verisi
        if camera_analysis["has_camera_data"]:
            cam = camera_analysis["camera_info"]
            make = cam.get("make", "")
            model = cam.get("model", "")
            parts.append(
                f"Kamera verisi mevcut: {make} {model}. "
                f"{camera_analysis['field_count']} kamera alani bulundu — "
                f"gercek fotograf sinyali."
            )
        else:
            parts.append(
                "Kamera bilgisi (Make, Model, Lens, ISO vb.) bulunamadi — "
                "yapay uretim veya metadata silinmis olabilir."
            )

        # GPS
        if gps_analysis["has_gps"]:
            parts.append("GPS koordinat bilgisi mevcut — gercek fotograf sinyali.")
        else:
            parts.append("GPS bilgisi bulunamadi.")

        # Boyut analizi
        if dimension_analysis.get("is_ai_dimension"):
            reasons = ", ".join(dimension_analysis.get("reasons", []))
            parts.append(f"BOYUT UYARISI: {reasons}.")

        # Sıkıştırma anomalisi
        if compression_analysis.get("anomaly_detected"):
            parts.append(f"SIKISTIRMA ANOMALISI: {compression_analysis['details']}.")

        # Metadata bütünlüğü
        comp = metadata_completeness["completeness"]
        total = metadata_completeness["total_fields"]
        if comp == "empty":
            parts.append(
                "UYARI: Metadata tamamen bos — kasitli silinmis olabilir."
            )
        elif comp == "minimal":
            parts.append(f"Metadata minimum seviyede ({total} alan).")
        elif comp == "rich":
            parts.append(f"Metadata zengin ({total} alan) — gercek fotograf sinyali.")

        # Nihai özet
        verdict_tr = {
            "high_risk": "YUKSEK RISK — Yapay uretim suphesi yuksek",
            "medium_risk": "ORTA RISK — Belirsiz, ek analiz gerekli",
            "low_risk": "DUSUK RISK — Muhtemelen gercek fotograf"
        }
        parts.append(f"\nPIN-A1 Sonuc: {verdict_tr.get(verdict, verdict)} (Skor: {score:.2f})")

        return " ".join(parts)

    # ════════════════════════════════════════════════════════════════
    # UTILITIES
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _safe_value(value) -> str | int | float | list | None:
        """EXIF değerini JSON-serializable formata çevirir."""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="ignore").strip()
            except Exception:
                return f"<bytes:{len(value)}>"
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif isinstance(value, tuple):
            return [PinA1Metadata._safe_value(v) for v in value]
        elif isinstance(value, dict):
            return {str(k): PinA1Metadata._safe_value(v) for k, v in value.items()}
        else:
            return str(value)

    @staticmethod
    def _sanitize_exif_for_json(exif_data: dict) -> dict:
        """EXIF verisini JSON'a güvenli yazmak için temizler."""
        sanitized = {}
        for key, value in exif_data.items():
            try:
                json_key = str(key)
                json_value = PinA1Metadata._safe_value(value)
                sanitized[json_key] = json_value
            except Exception:
                sanitized[str(key)] = "<unserializable>"
        return sanitized


# ════════════════════════════════════════════════════════════════
# CLI KULLANIMI
# ════════════════════════════════════════════════════════════════

def main():
    """Komut satırından doğrudan çalıştırma."""
    import json as json_module

    if len(sys.argv) < 2:
        print("Kullanim: python pin_a1_metadata.py <gorsel_yolu>")
        print("Ornek:    python pin_a1_metadata.py /path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"\n{'='*60}")
    print(f"  DeepReality PIN-A1: EXIF/Metadata Analizi")
    print(f"  Dosya: {image_path}")
    print(f"{'='*60}\n")

    pin = PinA1Metadata()
    result = pin.run(image_path)

    # Özet çıktı
    print(f"  Durum:    {result['status']}")
    print(f"  Skor:     {result['score']:.4f}")
    print(f"  Verdict:  {result['verdict']}")
    print(f"  AI Araci: {result['results'].get('source_tool', 'Tespit edilmedi')}")
    print(f"\n  Detay:\n  {result['details']}")

    # JSON dosya yolu
    file_stem = Path(image_path).stem
    output_file = Path("outputs") / f"{file_stem}_PIN-A1.json"
    print(f"\n  JSON cikti: {output_file}")
    print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    main()