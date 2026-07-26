"""
DeepReality — PIN-A1: EXIF / Metadata Analysis
==============================================

Function:
    Extracts and analyses the metadata layers of an image, detecting
    the traces generative tools leave behind and assessing whether
    camera, GPS and timestamp information is present.

    The governing principle is evidential asymmetry: the PRESENCE of
    coherent capture telemetry is strong evidence of authentic capture,
    whereas its ABSENCE is weak evidence of anything, since virtually
    every social platform strips metadata on upload.

Technology:
    Pillow (PIL), struct (binary EXIF parsing)

Output:
    metadata_score (0.0-1.0), source_tool, extracted_metadata, signals

Reasoning:
    - generator signature found      -> high score (synthesis suspected)
    - no camera data                 -> moderate signal
    - metadata entirely absent       -> mildly suspicious (possibly stripped)
    - camera + GPS + timestamp       -> low score (probably authentic)

Author: Omer Faruk Kurtulus
"""

import struct
import sys
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.base_pin import BasePin
from config.settings import METADATA_CONFIG


class PinA1Metadata(BasePin):
    """
    PIN-A1: EXIF / metadata analysis.

    Scans every metadata layer of the image:
    1. Standard EXIF data (camera, lens, exposure settings)
    2. GPS coordinates
    3. Software and tool fields
    4. Generator signatures (Stable Diffusion, Midjourney, DALL-E,
       Firefly and others)
    5. AI patterns embedded in XMP / IPTC text
    6. Metadata completeness analysis (was it stripped?)
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
        Main analysis entry point: extract and evaluate the image
        metadata.
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

        # ── 5. Natural-language explanation ─────────────────────────
        details = self._generate_details(
            score, verdict, ai_detection, c2pa_detection, camera_analysis,
            gps_analysis, software_analysis, metadata_completeness,
            dimension_analysis, compression_analysis
        )

        # ── 6. Result package ──────────────────────────────────────
        # source_tool: prefer the C2PA issuer, else the AI signature
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
        """Extract basic image properties (format, dimensions, mode)."""
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
        Extract EXIF data with Pillow, resolving the GPS and IFD
        sub-tag blocks as well.
        """
        exif_data = {}
        try:
            with Image.open(file_path) as img:
                raw_exif = img.getexif()
                if not raw_exif:
                    return exif_data

                # Resolve the primary EXIF tags
                for tag_id, value in raw_exif.items():
                    tag_name = TAGS.get(tag_id, f"Unknown_{tag_id}")
                    exif_data[tag_name] = self._safe_value(value)

                # Resolve the IFD (sub-EXIF) blocks
                for ifd_id in raw_exif.get_ifd(0x8769) or {}:
                    tag_name = TAGS.get(ifd_id, f"ExifIFD_{ifd_id}")
                    value = raw_exif.get_ifd(0x8769).get(ifd_id)
                    if value is not None:
                        exif_data[tag_name] = self._safe_value(value)

                # Resolve the GPS IFD block
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
        Extract readable text fragments from the raw container.

        Captures XMP, IPTC, tEXt chunks and embedded prompts by reading
        PNG tEXt/iTXt chunks and JPEG COM/APP segments. Generators
        frequently record their parameters here even when the standard
        EXIF block is empty.
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
        Search the metadata and raw text for generator signatures.

        When several tools match, the one with the highest confidence is
        selected.
        """
        detections = []

        # Concatenate every searchable text, lowercased
        searchable_texts = []

        # From the EXIF fields
        for key, value in exif_data.items():
            if isinstance(value, str):
                searchable_texts.append(value.lower())

        # From the raw text chunks
        for chunk in raw_text_chunks:
            if isinstance(chunk, str):
                searchable_texts.append(chunk.lower())

        combined_text = " ||| ".join(searchable_texts)

        # Test the signatures of each generative tool
        for tool_name, signatures in self.ai_signatures.items():
            confidence = 0.0
            matched_patterns = []

            # Software patterns
            for pattern in signatures.get("software_patterns", []):
                if pattern.lower() in combined_text:
                    confidence = max(confidence, 0.95)
                    matched_patterns.append(f"software: '{pattern}'")

            # Parameter fields (specific to Stable Diffusion)
            for field in signatures.get("parameter_fields", []):
                # Check as an EXIF key
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

        # Order the matches by confidence
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
        """Analyse the presence of camera telemetry."""
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
        """Check for GPS coordinate data."""
        found_gps = []
        for field in self.gps_fields:
            if field in exif_data and exif_data[field]:
                found_gps.append(field)

        has_gps = len(found_gps) >= 2  # Latitude and longitude are both required

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
        """Check for date and time information."""
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
        Analyse the Software / ProcessingSoftware fields, separating
        conventional photo editing from generative authoring.
        """
        software = exif_data.get("Software", "") or ""
        processing = exif_data.get("ProcessingSoftware", "") or ""
        creator_tool = exif_data.get("info_CreatorTool", "") or ""
        combined = f"{software} {processing} {creator_tool}".strip().lower()

        # Known photo-editing software (NOT suspicious on its own)
        known_editors = [
            "adobe photoshop", "lightroom", "capture one",
            "gimp", "affinity photo", "darktable", "rawtherapee", "picasa",
            "snapseed", "vsco", "instagram", "samsung", "apple",
            "google photos", "huawei", "xiaomi", "oppo", "vivo"
        ]

        is_known_editor = any(editor in combined for editor in known_editors)

        # Generative editing features (suspicious)
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
        Assess how complete the metadata is.

        Entirely empty metadata is mildly suspicious, since it may have
        been deliberately stripped — though it is equally consistent
        with a routine platform upload, so this signal is weighted low.
        """
        total_fields = len(exif_data)

        # Count info_-prefixed fields (PNG tEXt, JPEG COM and similar)
        info_fields = sum(1 for k in exif_data if k.startswith("info_"))

        # Standard EXIF fields
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
        Search the raw container for C2PA/JUMBF markers.

        Tools such as ChatGPT, DALL-E and Sora embed these markers.
        Because they live in the container rather than the EXIF block,
        they frequently survive in files whose metadata otherwise
        appears to have been stripped.

        This is a heuristic scan. PIN-A2 performs the authoritative
        cryptographic parse; findings here corroborate rather than
        supersede it.
        """
        found_markers = []
        issuer_tool = None
        confidence = 0.0

        try:
            with open(file_path, "rb") as f:
                raw = f.read()

            # Search for the C2PA markers
            for marker in self.c2pa_markers:
                idx = raw.find(marker)
                if idx != -1:
                    found_markers.append({
                        "marker": marker.decode("utf-8", errors="ignore"),
                        "offset": idx
                    })

            # Identify the C2PA issuer (OpenAI, Adobe and others)
            for issuer_bytes, tool_name in self.c2pa_issuers.items():
                if issuer_bytes in raw:
                    issuer_tool = tool_name
                    break

            # Derive the confidence
            if found_markers:
                # Temel marker bulundu
                confidence = min(0.60 + len(found_markers) * 0.08, 0.98)
                # Raise the confidence when the issuer is also identified
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
        Test whether the image dimensions match the characteristic
        output sizes of generative tools, which typically emit squares
        or multiples of 64.

        This is a weak corroborating signal, never decisive on its own:
        authentic images are routinely cropped to these dimensions too.
        """
        width = basic_info.get("width", 0)
        height = basic_info.get("height", 0)

        if width == 0 or height == 0:
            return {"is_ai_dimension": False, "details": "Boyut bilgisi yok"}

        # Exact-match test
        exact_match = (width, height) in self.ai_dimensions

        # Multiple of 64? (generative models operate on 64- or 8-pixel steps)
        divisible_by_64 = (width % 64 == 0) and (height % 64 == 0)

        # Kare mi?
        is_square = width == height

        # 2'nin kuvveti mi? (256, 512, 1024, 2048)
        def is_power_of_2(n):
            return n > 0 and (n & (n - 1)) == 0

        both_power_of_2 = is_power_of_2(width) and is_power_of_2(height)

        # Signal strength
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
        Analyse the ratio between file size and pixel count.

        Generated images typically exhibit different compression
        characteristics from photographs:
        - photographic JPEG: usually 2-8 bytes per pixel
        - generated PNG:     usually higher (little to compress away)
        - generated JPEG:    usually lower (large smooth areas)
        """
        width = basic_info.get("width", 0)
        height = basic_info.get("height", 0)
        file_size = basic_info.get("file_size_bytes", 0)
        fmt = basic_info.get("format", "").upper()

        if width == 0 or height == 0 or file_size == 0:
            return {"anomaly_detected": False, "details": "Yetersiz bilgi"}

        total_pixels = width * height
        bytes_per_pixel = file_size / total_pixels

        # Expected ranges by format
        anomaly = False
        details = ""

        if fmt == "JPEG":
            # Photographic JPEG: ~1.5-8 bpp
            # Generated JPEG: usually <1.0 bpp (very smooth) or >10 bpp (quality 100)
            if bytes_per_pixel < 0.5:
                anomaly = True
                details = f"Asiri dusuk sikistirma orani ({bytes_per_pixel:.2f} B/px) — yapay uretim sinyali"
            elif bytes_per_pixel > 12.0:
                anomaly = True
                details = f"Asiri yuksek sikistirma orani ({bytes_per_pixel:.2f} B/px) — olagan disi"
        elif fmt == "PNG":
            # PNG compression: generated images usually 0.5-3.0 bpp
            # Photographic PNG: usually >3.0 bpp
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
        Two-tier scoring system.

        TIER 1 — DECISIVE EVIDENCE RULES (evidence floor)
        When cryptographic or structural proof is present, a minimum
        score is guaranteed. These rules OVERRIDE the weighted sum,
        because a producer's own declaration of synthesis cannot be
        outvoted by an accumulation of weak heuristics.

        Rules:
        - C2PA + known AI issuer (OpenAI, Google, Adobe) -> min 0.85
        - C2PA marker found, issuer unknown              -> min 0.70
        - AI metadata signature >= 90% confidence         -> min 0.75
        - AI metadata signature >= 80% confidence         -> min 0.65

        TIER 2 — WEIGHTED SUM
        Where no decisive evidence exists, or where it falls below the
        floor, the weighted sum of all signals is used instead.

        Score: 0.0 (clean / authentic) to 1.0 (certain synthesis)
        """
        breakdown = {}

        # ── Tier 2: weighted sum (always computed) ──

        # 1. Was a generator signature detected?
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

        # 6. Is the software field suspicious?
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

        # 7. Was the metadata entirely stripped?
        strip_signal = 1.0 if metadata_completeness["is_stripped"] else 0.0
        if metadata_completeness["completeness"] == "minimal":
            strip_signal = 0.5
        breakdown["metadata_stripped"] = {
            "weight": self.weights["metadata_stripped"],
            "signal": strip_signal,
            "contribution": round(strip_signal * self.weights["metadata_stripped"], 4)
        }

        # 8. Dimensions typical of generators
        dim_signal = dimension_analysis.get("signal", 0.0)
        breakdown["ai_dimensions"] = {
            "weight": self.weights["ai_dimensions"],
            "signal": dim_signal,
            "contribution": round(dim_signal * self.weights["ai_dimensions"], 4)
        }

        # 9. Compression anomaly
        comp_signal = 1.0 if compression_analysis["anomaly_detected"] else 0.0
        breakdown["compression_anomaly"] = {
            "weight": self.weights["compression_anomaly"],
            "signal": comp_signal,
            "contribution": round(comp_signal * self.weights["compression_anomaly"], 4)
        }

        # Weighted sum
        weighted_sum = sum(item["contribution"] for item in breakdown.values())
        weighted_sum = max(0.0, min(1.0, weighted_sum))

        # ── Tier 1: decisive evidence rules (evidence floor) ──
        evidence_floor = 0.0
        evidence_rule = None

        # ── Tier 1: graduated decisive-evidence scoring ──
        #
        # When decisive evidence is present the score derives from the
        # confidence of that evidence, raised by corroborating signals.
        #
        # Formula:
        #   evidence_floor = evidence_confidence * base_multiplier
        #                  + destekleyici sinyallerden bonus
        #                  → min(toplam, 0.98)
        #
        # Corroborating signals (which strengthen decisive evidence):
        #   - AI issuer tespit edildi    → +0.05
        #   - generator-typical dimensions -> dim_signal * 0.03
        #   - metadata entirely absent     -> +0.02
        #   - Kamera verisi yok          → +0.02

        evidence_floor = 0.0
        evidence_rule = None
        evidence_components = []

        # Accumulate corroborating signals, applied only with decisive evidence
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

        # Rule 2: AI metadata signature (SD parameters, tool markers)
        elif ai_detection["ai_detected"] and ai_detection["confidence"] >= 0.80:
            base = ai_detection["confidence"] * 0.75
            evidence_floor = min(base + support_bonus, 0.98)
            evidence_rule = (
                f"AI imza ({ai_detection['detected_tool']}, "
                f"guvenilirlik: %{int(ai_detection['confidence']*100)}) "
                f"→ baz: {base:.3f} + destek: {support_bonus:.3f} = {evidence_floor:.3f}"
            )

        # Final score: the greater of the floor and the weighted sum
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
        """Map the numeric score onto a verdict band."""
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
        """Produce the natural-language explanation of the analysis."""
        parts = []

        # C2PA detection (the strongest signal)
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

        # Generator detection
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

        # Dimension analysis
        if dimension_analysis.get("is_ai_dimension"):
            reasons = ", ".join(dimension_analysis.get("reasons", []))
            parts.append(f"BOYUT UYARISI: {reasons}.")

        # Compression anomaly
        if compression_analysis.get("anomaly_detected"):
            parts.append(f"SIKISTIRMA ANOMALISI: {compression_analysis['details']}.")

        # Metadata completeness
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

        # Closing summary
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
        """Convert an EXIF value into a JSON-serialisable form."""
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
        """Sanitise EXIF data so it can be written to JSON safely."""
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
    """Direct command-line execution."""
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

    # Summary output
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