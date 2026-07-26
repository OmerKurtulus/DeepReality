"""
DeepReality — Layer 6 Feature Extraction
========================================

Converts the pin envelopes of a single image into the fixed-length,
strictly ordered feature vector consumed by PIN-F1.

This module is the single source of truth for the feature contract.
Training and inference import the same function, which eliminates the
most common defect in stacked ensembles: a meta-learner scored against
a feature vector assembled differently from the one it was trained on.
`FEATURE_NAMES` is versioned through `FEATURE_SCHEMA_VERSION`, and the
trained artefact records the version it was built against so a mismatch
is detected rather than silently mis-scored.

Missing evidence is encoded as NaN rather than zero. Zero is a
measurement ("this detector reported 0.0"); NaN is an absence ("this
detector did not run"). XGBoost learns a default traversal direction
for NaN at every split, so the distinction is preserved through the
model instead of being flattened at the input.

Scope
-----
The vector deliberately spans Tier 3 and Tier 4 evidence — the learned
detectors, their consensus statistics, and the localisation and
attention signals. Tier 1 and Tier 2 evidence (cryptographic
provenance, capture telemetry) is *included* for completeness but is
not what this model is expected to learn from, for two reasons:

1. Provenance is adjudicated by the ordered rule calculus of Layer 5
   (R1-R3), not statistically. A signed producer declaration is not a
   quantity to be regressed against; it is a constraint.
2. Public deepfake corpora are re-encoded during packaging, which
   strips EXIF and C2PA. Provenance features are therefore near-constant
   across any realistic training set and carry no gradient.

The consequence is stated explicitly in the consensus logic: when
Layer 5 resolves a case through a provenance rule, disagreement from
PIN-F1 is expected and is not treated as a warning.
"""

import math

FEATURE_SCHEMA_VERSION = "1.0.0"


def _num(value, default=float("nan")) -> float:
    """Coerce to float, mapping None and non-numerics to NaN."""
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _flag(value) -> float:
    """Coerce a boolean-ish value to 1.0 / 0.0, or NaN when absent."""
    if value is None:
        return float("nan")
    return 1.0 if bool(value) else 0.0


def _ok(envelope) -> dict | None:
    """Return a pin's results payload, or None when it did not succeed."""
    if not envelope or envelope.get("status") != "success":
        return None
    return envelope.get("results", {})


# Ordered feature contract. The order of this tuple IS the column order
# of the design matrix; never reorder it without a version bump.
FEATURE_NAMES = (
    # ── Tier 1/2 — provenance and capture telemetry ──
    "a1_score",
    "a1_ai_signature",
    "a1_c2pa_marker",
    "a1_c2pa_source_type_marker",
    "a1_has_camera",
    "a1_has_gps",
    "a1_has_datetime",
    "a1_metadata_stripped",
    "a1_exif_field_count",
    "a1_ai_dimensions",
    "a2_has_c2pa",
    "a2_source_is_ai",
    "a2_signature_valid",
    "a2_known_ai_issuer",
    "a2_score",
    # ── Tier 4 — compression forensics ──
    "a3_score",
    "a3_is_lossy",
    "a3_anomaly_count",
    "a3_max_deviation",
    "a3_high_severity_count",
    "a3_ela_mean",
    "a3_ela_std",
    # ── Context — subject composition ──
    "a4_face_count",
    "a4_max_face_confidence",
    "a4_max_face_area_ratio",
    "a4_max_sharpness",
    # ── Tier 3 — learned detectors ──
    "b1_prob",
    "b1_confidence",
    "b2_prob",
    "b2_confidence",
    "b3_prob",
    "b3_confidence",
    "b4_ai_prob",
    "b4_deepfake_prob",
    "b4_real_prob",
    "b4_fake_score",
    # ── Tier 3 — consensus statistics ──
    "consensus_mean",
    "consensus_min",
    "consensus_max",
    "consensus_spread",
    "consensus_n_flagging",
    "consensus_n_available",
    # ── Tier 4 — attention and localisation ──
    "d1_clip_siglip_iou",
    "d1_focus_region_count",
    "d1_max_focus_activation",
    "d1_max_focus_area_ratio",
    "d2_evidence_score",
    "d2_fused_count",
    "d2_ela_only_count",
    "d2_gradcam_only_count",
    # ── Engineered interactions ──
    # These encode the two conflict signatures the architecture is
    # designed around. XGBoost can in principle discover them, but only
    # with far more data than a realistic corpus provides; supplying
    # them directly is the difference between the model needing to learn
    # the domain and the model being told it.
    "x_telemetry_detector_conflict",
    "x_attention_off_subject",
    "x_spatial_frequency_agreement",
    "x_provenance_ai_strength",
)

N_FEATURES = len(FEATURE_NAMES)


def extract_features(results_by_pin: dict) -> dict:
    """
    Build the PIN-F1 feature mapping for one image.

    Args:
        results_by_pin: Mapping of pin_id -> standard pin envelope.

    Returns:
        {feature_name: float}, complete and in FEATURE_NAMES order.
        Absent evidence is NaN.
    """
    a1 = _ok(results_by_pin.get("PIN-A1"))
    a2 = _ok(results_by_pin.get("PIN-A2"))
    a3 = _ok(results_by_pin.get("PIN-A3"))
    a4 = _ok(results_by_pin.get("PIN-A4"))
    b1 = _ok(results_by_pin.get("PIN-B1"))
    b2 = _ok(results_by_pin.get("PIN-B2"))
    b3 = _ok(results_by_pin.get("PIN-B3"))
    b4 = _ok(results_by_pin.get("PIN-B4"))
    d1 = _ok(results_by_pin.get("PIN-D1"))
    d2 = _ok(results_by_pin.get("PIN-D2"))

    f: dict[str, float] = {name: float("nan") for name in FEATURE_NAMES}

    # ── PIN-A1 ──
    if a1 is not None:
        signals = a1.get("signals", {})
        ai_det = signals.get("ai_detection", {})
        c2pa_bin = signals.get("c2pa_detection", {})
        camera = signals.get("camera_data", {})
        gps = signals.get("gps_data", {})
        dt = signals.get("datetime", {})
        completeness = signals.get("metadata_completeness", {})
        dims = signals.get("dimension_analysis", {})

        markers = [str(m.get("marker", "")) for m in c2pa_bin.get("found_markers", [])]
        has_source_type = any(
            token in marker
            for marker in markers
            for token in ("trainedAlgorithmicMedia", "algorithmicMedia")
        )

        f["a1_score"] = _num(results_by_pin["PIN-A1"].get("score"))
        f["a1_ai_signature"] = _flag(ai_det.get("ai_detected"))
        f["a1_c2pa_marker"] = _flag(c2pa_bin.get("c2pa_detected"))
        f["a1_c2pa_source_type_marker"] = 1.0 if has_source_type else 0.0
        f["a1_has_camera"] = _flag(camera.get("has_camera_data"))
        f["a1_has_gps"] = _flag(gps.get("has_gps"))
        f["a1_has_datetime"] = _flag(dt.get("has_datetime"))
        f["a1_metadata_stripped"] = _flag(completeness.get("is_stripped"))
        f["a1_exif_field_count"] = _num(a1.get("exif_fields_found"), 0.0)
        f["a1_ai_dimensions"] = _flag(dims.get("is_ai_dimension"))

    # ── PIN-A2 ──
    if a2 is not None:
        f["a2_has_c2pa"] = _flag(a2.get("has_c2pa"))
        f["a2_score"] = _num(results_by_pin["PIN-A2"].get("score"))
        if a2.get("has_c2pa"):
            f["a2_source_is_ai"] = _flag(
                a2.get("digital_source_type", {}).get("is_ai_source")
            )
            f["a2_signature_valid"] = _flag(
                a2.get("validation", {}).get("is_valid")
            )
            f["a2_known_ai_issuer"] = _flag(
                a2.get("creator", {}).get("is_known_ai_issuer")
            )
        else:
            f["a2_source_is_ai"] = 0.0
            f["a2_signature_valid"] = 0.0
            f["a2_known_ai_issuer"] = 0.0

    # ── PIN-A3 ──
    if a3 is not None:
        regions = a3.get("manipulation_regions", [])
        deviations = [_num(r.get("deviation"), 0.0) for r in regions]
        f["a3_score"] = _num(results_by_pin["PIN-A3"].get("score"))
        f["a3_is_lossy"] = _flag(
            a3.get("source_format", {}).get("compression_type") == "lossy"
        )
        f["a3_anomaly_count"] = float(len(regions))
        f["a3_max_deviation"] = max(deviations) if deviations else 0.0
        f["a3_high_severity_count"] = float(
            sum(1 for r in regions if r.get("severity") == "high")
        )
        f["a3_ela_mean"] = _num(a3.get("global_stats", {}).get("mean"))
        f["a3_ela_std"] = _num(a3.get("global_stats", {}).get("std"))

    # ── PIN-A4 ──
    if a4 is not None:
        faces = a4.get("faces", [])
        f["a4_face_count"] = _num(a4.get("face_count"), 0.0)
        if faces:
            f["a4_max_face_confidence"] = max(
                _num(x.get("bounding_box", {}).get("confidence"), 0.0) for x in faces
            )
            f["a4_max_face_area_ratio"] = max(
                _num(x.get("quality", {}).get("face_area_ratio"), 0.0) for x in faces
            )
            f["a4_max_sharpness"] = max(
                _num(x.get("quality", {}).get("sharpness"), 0.0) for x in faces
            )
        else:
            f["a4_max_face_confidence"] = 0.0
            f["a4_max_face_area_ratio"] = 0.0
            f["a4_max_sharpness"] = 0.0

    # ── PIN-B1..B4 ──
    if b1 is not None:
        f["b1_prob"] = _num(b1.get("clip_prob"))
        f["b1_confidence"] = _num(b1.get("clip_confidence"))
    if b2 is not None:
        f["b2_prob"] = _num(b2.get("siglip_prob"))
        f["b2_confidence"] = _num(b2.get("siglip_confidence"))
    if b3 is not None:
        f["b3_prob"] = _num(b3.get("freq_prob"))
        f["b3_confidence"] = _num(b3.get("freq_confidence"))
    if b4 is not None:
        f["b4_ai_prob"] = _num(b4.get("ai_prob"))
        f["b4_deepfake_prob"] = _num(b4.get("deepfake_prob"))
        f["b4_real_prob"] = _num(b4.get("real_prob"))
        f["b4_fake_score"] = _num(b4.get("fake_score"))

    # ── Consensus statistics ──
    probs = [
        p for p in (
            f["b1_prob"], f["b2_prob"], f["b3_prob"], f["b4_fake_score"]
        ) if not math.isnan(p)
    ]
    if probs:
        f["consensus_mean"] = sum(probs) / len(probs)
        f["consensus_min"] = min(probs)
        f["consensus_max"] = max(probs)
        f["consensus_spread"] = max(probs) - min(probs)
        f["consensus_n_flagging"] = float(sum(1 for p in probs if p >= 0.5))
        f["consensus_n_available"] = float(len(probs))
    else:
        f["consensus_n_available"] = 0.0

    # ── PIN-D1 ──
    if d1 is not None:
        combined = d1.get("focus_regions", {}).get("combined", [])
        f["d1_clip_siglip_iou"] = _num(d1.get("model_agreement_iou"))
        f["d1_focus_region_count"] = float(len(combined))
        if combined:
            f["d1_max_focus_activation"] = max(
                _num(r.get("mean_activation"), 0.0) for r in combined
            )
            f["d1_max_focus_area_ratio"] = max(
                _num(r.get("area_ratio"), 0.0) for r in combined
            )
        else:
            f["d1_max_focus_activation"] = 0.0
            f["d1_max_focus_area_ratio"] = 0.0

    # ── PIN-D2 ──
    if d2 is not None:
        counts = d2.get("region_counts", {})
        f["d2_evidence_score"] = _num(results_by_pin["PIN-D2"].get("score"))
        f["d2_fused_count"] = _num(counts.get("fused"), 0.0)
        f["d2_ela_only_count"] = _num(counts.get("ela_only"), 0.0)
        f["d2_gradcam_only_count"] = _num(counts.get("gradcam_only"), 0.0)

    # ── Engineered interactions ──

    # The computational-photography false-positive signature: coherent
    # capture telemetry present while the detectors report synthesis.
    if not math.isnan(f["a1_has_camera"]) and not math.isnan(f["consensus_mean"]):
        f["x_telemetry_detector_conflict"] = (
            f["a1_has_camera"] * f["consensus_mean"]
        )

    # Attention concentrated away from the subject while detectors are
    # confident: high consensus, but no corroborated localisation and a
    # diffuse or off-subject focus.
    if not math.isnan(f["consensus_mean"]) and not math.isnan(f["d2_fused_count"]):
        f["x_attention_off_subject"] = f["consensus_mean"] * (
            1.0 if f["d2_fused_count"] == 0 else 0.0
        )

    # Agreement between the spatial and frequency paradigms. Because the
    # domains are disjoint, concordance here is stronger corroboration
    # than agreement among the spatial models alone.
    spatial = [p for p in (f["b1_prob"], f["b2_prob"]) if not math.isnan(p)]
    if spatial and not math.isnan(f["b3_prob"]):
        f["x_spatial_frequency_agreement"] = 1.0 - abs(
            (sum(spatial) / len(spatial)) - f["b3_prob"]
        )

    # Strongest documentary AI indicator available, on a single axis.
    provenance = [
        v for v in (
            f["a1_ai_signature"], f["a1_c2pa_source_type_marker"],
            f["a2_source_is_ai"],
        ) if not math.isnan(v)
    ]
    if provenance:
        f["x_provenance_ai_strength"] = max(provenance)

    return f


def features_to_vector(feature_map: dict) -> list[float]:
    """Flatten a feature mapping into FEATURE_NAMES order."""
    return [feature_map[name] for name in FEATURE_NAMES]
