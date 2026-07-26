"""
DeepReality — Evidence Digest Builder
=====================================

Transforms the raw output of every upstream pin into a compact,
information-dense digest suitable for submission to a language model.

Design rationale
----------------
The unabridged pin output for a single image exceeds 2,300 embedding
values (CLIP 1024-d, SigLIP2 768-d, frequency CNN 512-d) plus static
model cards, per-region pixel ranges and facial landmark coordinates.
Serialised naively this approaches 40,000 tokens, of which essentially
none carries adjudication value: an embedding is meaningful to a
downstream classifier, not to a reasoning model.

The digest therefore applies three reductions:

1. **Elimination** - embeddings, model cards, raw manifests and
   landmark coordinates are dropped entirely. Static model
   characteristics are described once in the system prompt instead of
   being repeated per request.
2. **Aggregation** - region lists are summarised by count, severity
   distribution and the strongest representatives rather than
   enumerated exhaustively.
3. **Normalisation** - every pin is reduced to the same shape
   (score, verdict, salient findings) so the reasoning model can apply
   a uniform interpretation protocol.

The result is a digest of roughly 700-900 tokens that preserves every
decision-relevant fact while remaining economical enough for
high-volume operation.
"""

from pathlib import Path

from config.settings import LLM_CONFIG


# Pin identifiers grouped by the layer they belong to
_LAYER1_PINS = ("PIN-A1", "PIN-A2", "PIN-A3", "PIN-A4")
_LAYER2_PINS = ("PIN-B1", "PIN-B2", "PIN-B3", "PIN-B4")
_LAYER4_PINS = ("PIN-D1", "PIN-D2")


def _round(value, digits: int | None = None):
    """Round a float for compact serialisation; pass through anything else."""
    if digits is None:
        digits = LLM_CONFIG["digest"]["float_precision"]
    if isinstance(value, float):
        return round(value, digits)
    return value


def _pin_status(results_by_pin: dict, pin_id: str) -> tuple[dict, dict | None]:
    """
    Return (pin_output, pin_results) for a pin, or (output, None) when the
    pin failed or never ran. Callers must tolerate a missing pin: the
    architecture guarantees degradation, not completeness.
    """
    output = results_by_pin.get(pin_id)
    if not output or output.get("status") != "success":
        return output or {}, None
    return output, output.get("results", {})


# ---------------------------------------------------------------------------
# Layer 1 — provenance and preprocessing
# ---------------------------------------------------------------------------

def _digest_a1(results_by_pin: dict) -> dict:
    """PIN-A1: EXIF/metadata provenance signals."""
    output, results = _pin_status(results_by_pin, "PIN-A1")
    if results is None:
        return {"available": False}

    signals = results.get("signals", {})
    ai_detection = signals.get("ai_detection", {})
    c2pa_binary = signals.get("c2pa_detection", {})
    camera = signals.get("camera_data", {})
    gps = signals.get("gps_data", {})
    datetime_info = signals.get("datetime", {})
    completeness = signals.get("metadata_completeness", {})
    dimensions = signals.get("dimension_analysis", {})

    digest = {
        "available": True,
        "score": _round(output.get("score")),
        "verdict": output.get("verdict"),
        "exif_field_count": results.get("exif_fields_found", 0),
        "metadata_stripped": completeness.get("is_stripped"),
        "ai_tool_signature": ai_detection.get("detected_tool"),
        "camera_metadata_present": camera.get("has_camera_data", False),
        "gps_present": gps.get("has_gps", False),
        "capture_datetime_present": datetime_info.get("has_datetime", False),
    }

    # Camera identity is decisive supporting evidence for authentic capture,
    # so it is reported verbatim rather than as a boolean.
    if camera.get("has_camera_data"):
        info = camera.get("camera_info", {})
        digest["camera"] = {
            key: info.get(key)
            for key in ("make", "model", "lens", "iso", "f_number")
            if info.get(key)
        }
        digest["camera_field_count"] = camera.get("field_count", 0)

    # Heuristic C2PA byte-marker scan. PIN-A2 performs the authoritative
    # cryptographic parse; this only corroborates.
    if c2pa_binary.get("c2pa_detected"):
        digest["c2pa_binary_markers"] = {
            "issuer_hint": c2pa_binary.get("issuer_tool"),
            "marker_count": c2pa_binary.get("marker_count"),
            "markers": [
                m.get("marker") for m in c2pa_binary.get("found_markers", [])
            ][:8],
        }

    if dimensions.get("is_ai_dimension"):
        digest["ai_typical_dimensions"] = dimensions.get("dimensions")

    return digest


def _digest_a2(results_by_pin: dict) -> dict:
    """PIN-A2: cryptographically verified C2PA provenance."""
    output, results = _pin_status(results_by_pin, "PIN-A2")
    if results is None:
        return {"available": False}

    if not results.get("has_c2pa"):
        return {"available": True, "has_c2pa": False}

    creator = results.get("creator", {})
    tool = results.get("tool", {})
    source_type = results.get("digital_source_type", {})
    validation = results.get("validation", {})
    actions = results.get("actions", {})

    return {
        "available": True,
        "has_c2pa": True,
        "score": _round(output.get("score")),
        "verdict": output.get("verdict"),
        "signature_valid": validation.get("is_valid"),
        "issuer": creator.get("issuer"),
        "issuer_is_known_ai": creator.get("is_known_ai_issuer"),
        "claim_generator": (
            tool.get("claim_generator_parsed") or tool.get("claim_generator")
        ),
        "software_agent": tool.get("software_agent"),
        "tool_is_known_ai": tool.get("is_known_ai_tool"),
        "digital_source_type": source_type.get("source_type"),
        "source_is_ai": source_type.get("is_ai_source"),
        "source_category": source_type.get("source_category"),
        "signature_time": results.get("timestamp", {}).get("signature_time"),
        "actions": actions.get("actions_found", [])[:6],
    }


def _digest_a3(results_by_pin: dict) -> dict:
    """PIN-A3: Error Level Analysis — localised recompression anomalies."""
    output, results = _pin_status(results_by_pin, "PIN-A3")
    if results is None:
        return {"available": False}

    limit = LLM_CONFIG["digest"]["max_ela_regions"]
    regions = results.get("manipulation_regions", [])
    source_format = results.get("source_format", {})
    global_stats = results.get("global_stats", {})
    uniformity = results.get("uniformity", {})

    severity_counts: dict[str, int] = {}
    for region in regions:
        key = f"{region.get('type', 'unknown')}_{region.get('severity', 'low')}"
        severity_counts[key] = severity_counts.get(key, 0) + 1

    # Strongest anomalies only: deviation magnitude carries the signal,
    # the full 8x8 grid does not.
    strongest = sorted(
        regions, key=lambda r: r.get("deviation", 0), reverse=True
    )[:limit]

    return {
        "available": True,
        "score": _round(output.get("score")),
        "verdict": output.get("verdict"),
        "source_format": source_format.get("file_extension"),
        "compression_type": source_format.get("compression_type"),
        "ela_mean": _round(global_stats.get("mean")),
        "ela_std": _round(global_stats.get("std")),
        "uniformity_category": uniformity.get("category"),
        "anomaly_count": len(regions),
        "anomaly_breakdown": severity_counts,
        "strongest_anomalies": [
            {
                "type": r.get("type"),
                "severity": r.get("severity"),
                "deviation_sigma": _round(r.get("deviation")),
                "grid_position": r.get("position"),
            }
            for r in strongest
        ],
    }


def _digest_a4(results_by_pin: dict) -> dict:
    """PIN-A4: face detection — establishes whether deepfake is plausible."""
    output, results = _pin_status(results_by_pin, "PIN-A4")
    if results is None:
        return {"available": False}

    faces = results.get("faces", [])
    face_summaries = []
    for face in faces[:4]:
        quality = face.get("quality", {})
        alignment = face.get("alignment", {})
        face_summaries.append({
            "confidence": _round(face.get("bounding_box", {}).get("confidence")),
            "resolution": quality.get("resolution_category"),
            "sharpness": _round(quality.get("sharpness")),
            "area_ratio": _round(quality.get("face_area_ratio")),
            "frontal": alignment.get("is_frontal"),
        })

    return {
        "available": True,
        "face_count": results.get("face_count", 0),
        "faces": face_summaries,
    }


# ---------------------------------------------------------------------------
# Layer 2 — detection core
# ---------------------------------------------------------------------------

def _digest_layer2(results_by_pin: dict) -> dict:
    """
    PIN-B1..B4: the four detection models.

    Embeddings are discarded; only probability, verdict and confidence
    survive. PIN-B4 additionally contributes its three-class breakdown,
    which is the only signal in the system able to separate wholly
    synthetic imagery from manipulation of authentic content.
    """
    digest: dict = {}

    binary_pins = {
        "PIN-B1": ("clip_prob", "clip_confidence", "clip_verdict"),
        "PIN-B2": ("siglip_prob", "siglip_confidence", "siglip_verdict"),
        "PIN-B3": ("freq_prob", "freq_confidence", "freq_verdict"),
    }

    for pin_id, (prob_key, conf_key, verdict_key) in binary_pins.items():
        output, results = _pin_status(results_by_pin, pin_id)
        if results is None:
            digest[pin_id] = {
                "available": False,
                "error": (output.get("details") or "")[:120] or None,
            }
            continue
        digest[pin_id] = {
            "available": True,
            "fake_probability": _round(results.get(prob_key)),
            "confidence": _round(results.get(conf_key)),
            "verdict": results.get(verdict_key),
            "risk_level": output.get("verdict"),
        }

    output, results = _pin_status(results_by_pin, "PIN-B4")
    if results is None:
        digest["PIN-B4"] = {
            "available": False,
            "error": (output.get("details") or "")[:120] or None,
        }
    else:
        digest["PIN-B4"] = {
            "available": True,
            "predicted_class": results.get("predicted_class"),
            "p_ai_generated": _round(results.get("ai_prob")),
            "p_deepfake": _round(results.get("deepfake_prob")),
            "p_real": _round(results.get("real_prob")),
            "combined_fake_probability": _round(results.get("fake_score")),
            "risk_level": output.get("verdict"),
        }

    # Consensus statistics spare the reasoning model from arithmetic and
    # make disagreement immediately visible.
    probabilities = [
        entry.get("fake_probability")
        for entry in (digest.get(p, {}) for p in ("PIN-B1", "PIN-B2", "PIN-B3"))
        if entry.get("fake_probability") is not None
    ]
    b4_combined = digest.get("PIN-B4", {}).get("combined_fake_probability")
    if b4_combined is not None:
        probabilities.append(b4_combined)

    if probabilities:
        digest["consensus"] = {
            "model_count": len(probabilities),
            "mean_fake_probability": _round(sum(probabilities) / len(probabilities)),
            "min": _round(min(probabilities)),
            "max": _round(max(probabilities)),
            "spread": _round(max(probabilities) - min(probabilities)),
            "models_flagging_fake": sum(1 for p in probabilities if p >= 0.5),
        }

    return digest


# ---------------------------------------------------------------------------
# Layer 4 — explainability
# ---------------------------------------------------------------------------

def _digest_layer4(results_by_pin: dict) -> dict:
    """
    PIN-D1/D2: where the models looked and where anomalies localise.

    Spatial focus is the primary instrument for diagnosing false
    positives: attention concentrated away from the semantic subject
    while the detectors report high fake probability is a signature of
    distribution shift rather than genuine manipulation.
    """
    digest: dict = {}
    focus_limit = LLM_CONFIG["digest"]["max_focus_regions"]
    marked_limit = LLM_CONFIG["digest"]["max_marked_regions"]

    output, results = _pin_status(results_by_pin, "PIN-D1")
    if results is None:
        digest["PIN-D1"] = {"available": False}
    else:
        combined_focus = results.get("focus_regions", {}).get("combined", [])
        digest["PIN-D1"] = {
            "available": True,
            "models_analysed": [
                k for k in results.get("heatmaps", {}) if k != "combined"
            ],
            "clip_siglip_spatial_agreement_iou": results.get("model_agreement_iou"),
            "focus_region_count": len(combined_focus),
            "primary_focus_regions": [
                {
                    "bbox": region.get("bbox"),
                    "mean_activation": region.get("mean_activation"),
                    "area_ratio": region.get("area_ratio"),
                }
                for region in combined_focus[:focus_limit]
            ],
        }

    output, results = _pin_status(results_by_pin, "PIN-D2")
    if results is None:
        digest["PIN-D2"] = {"available": False}
    else:
        counts = results.get("region_counts", {})
        marked = results.get("marked_regions", [])
        fused = [r for r in marked if r.get("source") == "fused"]
        ela_only = [r for r in marked if r.get("source") == "ela"]
        digest["PIN-D2"] = {
            "available": True,
            "localisation_evidence_score": _round(output.get("score")),
            "region_counts": counts,
            "evidence_basis": results.get("evidence_basis"),
            "corroborated_regions": [
                {
                    "bbox": r.get("bbox"),
                    "ela_type": r.get("ela_type"),
                    "severity": r.get("severity"),
                    "cam_support": r.get("cam_support"),
                }
                for r in (fused + ela_only)[:marked_limit]
            ],
        }

    return digest


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_evidence_digest(file_path: str, results_by_pin: dict) -> dict:
    """
    Assemble the complete evidence digest for one image.

    Args:
        file_path: Path of the analysed image.
        results_by_pin: Mapping of pin_id -> standard pin output.

    Returns:
        A nested dictionary mirroring the layer structure of the system,
        containing every adjudication-relevant fact and nothing else.
    """
    path = Path(file_path)

    a1_output = results_by_pin.get("PIN-A1", {})
    basic_info = a1_output.get("results", {}).get("basic_info", {})

    subject: dict = {
        "filename": path.name,
        "sha256": (a1_output.get("input_hash") or "")[:16],
    }
    if basic_info:
        subject.update({
            "format": basic_info.get("format"),
            "dimensions": f"{basic_info.get('width')}x{basic_info.get('height')}",
            "file_size_mb": basic_info.get("file_size_mb"),
        })

    failed = [
        pin_id for pin_id, output in results_by_pin.items()
        if output.get("status") != "success"
    ]

    return {
        "subject": subject,
        "layer1_provenance": {
            "PIN-A1_metadata": _digest_a1(results_by_pin),
            "PIN-A2_c2pa": _digest_a2(results_by_pin),
            "PIN-A3_ela": _digest_a3(results_by_pin),
            "PIN-A4_face": _digest_a4(results_by_pin),
        },
        "layer2_detection": _digest_layer2(results_by_pin),
        "layer4_explainability": _digest_layer4(results_by_pin),
        "pipeline_integrity": {
            "pins_executed": len(results_by_pin),
            "pins_failed": failed,
        },
    }
