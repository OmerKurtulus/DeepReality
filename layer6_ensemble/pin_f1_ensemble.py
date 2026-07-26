"""
DeepReality — PIN-F1: XGBoost Meta-Learner (Ensemble Fusion)
============================================================

Stacked generalisation over the pin score vector. The base learners are
the pins themselves; the meta-learner is a gradient-boosted tree
ensemble trained on their outputs against ground-truth labels, then
probability-calibrated.

Why trees rather than a weighted average
----------------------------------------
The optimal combination is not linear, because the value of a
detector's output is *conditional* on the evidence surrounding it. A
high PIN-B2 score means something different when capture telemetry is
present than when it is absent; PIN-B1's dissent carries more weight
when the subject resembles a photograph than when it does not; PIN-B3
agreeing with the spatial models is stronger corroboration than the
spatial models agreeing with each other, because the domains are
disjoint. Gradient-boosted trees represent such interactions natively.
A fixed weighted average cannot express them at all — it assigns each
detector one influence for every possible input.

Relationship to Layer 5
-----------------------
PIN-F1 and PIN-E1 adjudicate the same evidence by deliberately
different means: E1 reasons symbolically under an ordered evidence
hierarchy, F1 fits statistically to labelled outcomes. They are
redundant by design.

E1's verdict enters this pin only for comparison. It is never a model
input — feeding it in would collapse the cross-check into an imitation
of the language model and destroy the independence the comparison
depends on.

Interpreting divergence requires care, and the pin does it explicitly.
F1 is trained on corpora that are re-encoded during packaging, which
strips EXIF and C2PA; provenance features are therefore near-constant
in training and F1 learns essentially nothing from them. When E1
resolves a case through a provenance rule (R1, R2 or R3), F1 is
reasoning without the evidence that determined the verdict, so
disagreement is *expected* and is not a warning. Only when E1 decided
on statistical grounds (R4, R5) does divergence indicate a genuine
conflict worth surfacing.

Degradation
-----------
When no trained artefact is present the pin reports
`model_status: "untrained"` and emits a transparent weighted-average
baseline, clearly labelled, rather than a silent zero. The pipeline
continues; the consensus section simply records that no learned fusion
was available.
"""

import json
from pathlib import Path

from core.base_pin import BasePin
from config.settings import ENSEMBLE_CONFIG
from layer6_ensemble.feature_extractor import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    extract_features,
)

# Pins whose evidence forms the feature vector. PIN-E1 is deliberately
# excluded: it is compared against, never consumed.
FEATURE_PINS = (
    "PIN-A1", "PIN-A2", "PIN-A3", "PIN-A4",
    "PIN-B1", "PIN-B2", "PIN-B3", "PIN-B4",
    "PIN-D1", "PIN-D2",
)

# Rules under which Layer 5 decided on documentary rather than
# statistical grounds, and where F1 therefore lacks the deciding evidence.
PROVENANCE_RULES = frozenset({"R1", "R2", "R3"})

# Verdicts that assert synthesis, used to align E1's categorical output
# with F1's scalar for comparison.
SYNTHETIC_VERDICTS = frozenset({"AI_GENERATED", "DEEPFAKE"})

_model_cache = None
_metadata_cache = None


def _load_model():
    """Load the booster and its metadata once, caching globally."""
    global _model_cache, _metadata_cache
    if _model_cache is not None:
        return _model_cache, _metadata_cache

    model_path = Path(ENSEMBLE_CONFIG["model_path"])
    meta_path = Path(ENSEMBLE_CONFIG["metadata_path"])
    if not model_path.exists():
        return None, None

    # The saved JSON is evaluated directly rather than through the xgboost
    # runtime; see layer6_ensemble.booster_eval for why that library cannot
    # share a process with PyTorch on macOS. Equivalence with xgboost is
    # asserted by tests/test_booster_eval.py.
    from layer6_ensemble.booster_eval import NativeBooster

    booster = NativeBooster(model_path)

    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    _model_cache, _metadata_cache = booster, metadata
    return booster, metadata


def _apply_calibration(raw: float, metadata: dict) -> float:
    """
    Map the raw booster output onto a calibrated probability.

    Platt scaling is used rather than isotonic regression: with the
    sample sizes realistic for this stage, isotonic overfits the
    calibration set and produces a step function that is worse
    calibrated out of sample than the sigmoid it replaces.
    """
    calibration = metadata.get("calibration") or {}
    if calibration.get("method") != "platt":
        return raw

    import math

    a = float(calibration.get("a", 1.0))
    b = float(calibration.get("b", 0.0))
    # Guard the exponent so an extreme fit cannot overflow
    z = max(-60.0, min(60.0, a * raw + b))
    return 1.0 / (1.0 + math.exp(-z))


class PinF1Ensemble(BasePin):
    """
    PIN-F1: calibrated statistical fusion over the pin score vector,
    cross-validated against the Layer 5 adjudication.
    """

    def __init__(self):
        super().__init__(
            pin_id="PIN-F1",
            pin_name="XGBoost Meta-Learner (Ensemble Fusion)",
            layer=6,
        )

    # ── Untrained baseline ──────────────────────────────────────────

    def _weighted_baseline(self, features: dict) -> tuple[float, str]:
        """
        Transparent fallback used before a model has been trained.

        A fixed weighted average over the available detectors. It is
        stated plainly as a baseline rather than presented as a learned
        fusion, because an untrained stage that reports a confident
        number is worse than one that reports none.
        """
        import math

        weights = ENSEMBLE_CONFIG["baseline_weights"]
        pairs = [
            (features.get("b1_prob"), weights["b1"]),
            (features.get("b2_prob"), weights["b2"]),
            (features.get("b3_prob"), weights["b3"]),
            (features.get("b4_fake_score"), weights["b4"]),
        ]
        usable = [(p, w) for p, w in pairs if p is not None and not math.isnan(p)]
        if not usable:
            return float("nan"), "no detector output available"

        total_weight = sum(w for _, w in usable)
        score = sum(p * w for p, w in usable) / total_weight
        return score, f"weighted average over {len(usable)} detectors"

    # ── Consensus with Layer 5 ──────────────────────────────────────

    def _build_consensus(self, f1_score: float, e1_envelope: dict | None) -> dict:
        """
        Compare the statistical fusion against the Layer 5 adjudication.

        Divergence is reported with its interpretation attached, because
        the same numeric gap means different things depending on which
        rule Layer 5 applied.
        """
        import math

        if not e1_envelope or e1_envelope.get("status") != "success":
            return {
                "available": False,
                "note": "Layer 5 adjudication unavailable; no cross-check performed",
            }

        e1 = e1_envelope.get("results", {})
        e1_verdict = e1.get("final_verdict")
        e1_prob = e1.get("fake_probability")
        applied_rule = e1.get("applied_rule")

        consensus = {
            "available": True,
            "e1_verdict": e1_verdict,
            "e1_fake_probability": e1_prob,
            "e1_applied_rule": applied_rule,
            "f1_fake_probability": (
                None if math.isnan(f1_score) else round(f1_score, 4)
            ),
        }

        if math.isnan(f1_score) or e1_prob is None:
            consensus["agreement"] = "indeterminate"
            consensus["interpretation"] = (
                "One adjudicator produced no score; no comparison is possible."
            )
            return consensus

        # Categorical agreement, on the synthesis question only
        e1_says_synthetic = e1_verdict in SYNTHETIC_VERDICTS
        f1_says_synthetic = f1_score >= ENSEMBLE_CONFIG["decision_threshold"]
        divergence = abs(float(e1_prob) - f1_score)

        consensus["probability_divergence"] = round(divergence, 4)
        consensus["categorical_agreement"] = (e1_says_synthetic == f1_says_synthetic)

        provenance_driven = applied_rule in PROVENANCE_RULES

        if consensus["categorical_agreement"]:
            consensus["agreement"] = "concordant"
            consensus["interpretation"] = (
                "Symbolic and statistical adjudication concur; the finding "
                "is supported by two independently derived methods."
            )
        elif provenance_driven:
            consensus["agreement"] = "expected_divergence"
            consensus["interpretation"] = (
                f"Layer 5 resolved this case through {applied_rule}, a "
                f"documentary rule. PIN-F1 does not observe provenance "
                f"evidence in a form it can learn from, so its dissent is "
                f"expected and does not weaken the verdict."
            )
        else:
            consensus["agreement"] = "conflict"
            consensus["interpretation"] = (
                f"Layer 5 decided on statistical grounds ({applied_rule}) and "
                f"PIN-F1 reaches the opposite conclusion from the same "
                f"evidence. This case warrants human review."
            )

        consensus["review_recommended"] = consensus["agreement"] == "conflict"
        return consensus

    # ── Main analysis ───────────────────────────────────────────────

    def analyze(self, file_path: str) -> dict:
        import math

        results_by_pin = {
            pin_id: self.context[pin_id]
            for pin_id in FEATURE_PINS
            if pin_id in self.context
        }

        if not results_by_pin:
            return {
                "results": {
                    "model_status": "no_input",
                    "error": "No upstream pin results were provided",
                },
                "score": 0.0,
                "verdict": "error",
                "details": (
                    "PIN-F1 upstream pin ciktisi almadi — pipeline "
                    "bagimliliklari hatali yapilandirilmis."
                ),
            }

        features = extract_features(results_by_pin)
        booster, metadata = _load_model()

        schema_warning = None
        if booster is None:
            score, basis = self._weighted_baseline(features)
            model_status = "untrained"
            model_info = {
                "note": (
                    "No trained artefact present. Reporting a transparent "
                    "weighted baseline; train PIN-F1 to enable learned fusion."
                ),
                "basis": basis,
            }
        else:
            trained_version = metadata.get("feature_schema_version")
            if trained_version and trained_version != FEATURE_SCHEMA_VERSION:
                schema_warning = (
                    f"Feature schema mismatch: artefact was trained against "
                    f"{trained_version}, runtime provides "
                    f"{FEATURE_SCHEMA_VERSION}. Scores are not trustworthy "
                    f"until the model is retrained."
                )
                self.errors.append(schema_warning)

            # The artefact records the exact columns it was fitted on, which
            # need not be the full contract: the deployed model is restricted
            # to Tier 3 and Tier 4 evidence because provenance is adjudicated
            # by the Layer 5 rule calculus rather than learned. Scoring
            # against the full vector would misalign every column.
            model_features = metadata.get("feature_names") or list(FEATURE_NAMES)
            missing = [f for f in model_features if f not in features]
            if missing:
                raise KeyError(
                    f"Artefact expects features absent from this runtime: "
                    f"{missing[:5]}"
                )

            raw = booster.predict([features[name] for name in model_features])
            score = _apply_calibration(raw, metadata)
            model_status = "trained"
            model_info = {
                "raw_score": round(raw, 6),
                "calibration": (metadata.get("calibration") or {}).get(
                    "method", "none"
                ),
                "trained_at": metadata.get("trained_at"),
                "training_samples": metadata.get("training_samples"),
                "training_dataset": metadata.get("training_dataset"),
                "holdout_metrics": metadata.get("holdout_metrics"),
                "feature_schema_version": trained_version,
            }

        # Feature attribution: which signals moved this particular decision
        top_features = None
        if booster is not None and metadata.get("feature_importance"):
            importance = metadata["feature_importance"]
            scored = metadata.get("feature_names") or list(FEATURE_NAMES)
            present = {
                name: features[name]
                for name in scored
                if name in features and not math.isnan(features[name])
            }
            ranked = sorted(
                present.items(),
                key=lambda kv: importance.get(kv[0], 0.0),
                reverse=True,
            )[: ENSEMBLE_CONFIG["report_top_features"]]
            top_features = [
                {
                    "feature": name,
                    "value": round(value, 4),
                    "gain": round(importance.get(name, 0.0), 4),
                }
                for name, value in ranked
            ]

        consensus = self._build_consensus(score, self.context.get("PIN-E1"))

        if math.isnan(score):
            final_score, risk = 0.0, "no_data"
        else:
            final_score = score
            thresholds = ENSEMBLE_CONFIG["thresholds"]
            if score >= thresholds["high_risk"]:
                risk = "high_risk"
            elif score >= thresholds["medium_risk"]:
                risk = "medium_risk"
            else:
                risk = "low_risk"

        results = {
            "model_status": model_status,
            "fake_probability": (
                None if math.isnan(score) else round(score, 4)
            ),
            "decision": (
                None if math.isnan(score)
                else ("SYNTHETIC" if score >= ENSEMBLE_CONFIG["decision_threshold"]
                      else "AUTHENTIC")
            ),
            "model_info": model_info,
            "top_features": top_features,
            "consensus": consensus,
            "feature_vector": {
                name: (None if math.isnan(v) else round(v, 6))
                for name, v in features.items()
            },
            "features_available": sum(
                1 for v in features.values() if not math.isnan(v)
            ),
            "features_total": len(FEATURE_NAMES),
            "schema_warning": schema_warning,
        }

        if math.isnan(score):
            details = "Ensemble skoru uretilemedi — detektor ciktisi yok."
        else:
            details = (
                f"Ensemble fusion: {results['decision']} "
                f"(olasilik: {score:.4f}, model: {model_status}, "
                f"uzlasma: {consensus.get('agreement', '-')})"
            )

        return {
            "results": results,
            "score": final_score,
            "verdict": risk,
            "details": details,
        }
