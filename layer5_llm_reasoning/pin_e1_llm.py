"""
DeepReality — PIN-E1: LLM Reasoning Engine
==========================================

The adjudication stage of the architecture. Every upstream pin produces
an independent signal within a deliberately narrow competence; PIN-E1
is the only component that sees all of them simultaneously and is
therefore the only one able to weigh documentary provenance against
statistical inference, recognise when a detector is operating outside
its validated domain, and state a conclusion in natural language.

Dependencies: all Layer 1, Layer 2 and Layer 4 pins. PIN-E1 is by
construction the terminal node of the pipeline dependency graph.

Pipeline
--------
1. Compress the full evidence set into a token-efficient digest
   (`evidence_builder`), discarding embeddings and static model cards
   that carry no adjudication value.
2. Submit the digest under the forensic reasoning protocol
   (`prompts.SYSTEM_PROMPT`) to the configured model.
3. Validate and normalise the returned verdict.
4. Emit the standard pin envelope, with `score` carrying the adjudicated
   fake probability.

Degradation
-----------
The pin never aborts the pipeline. When no credential is configured or
the provider is unreachable, a deterministic rule-based adjudication
implementing the same evidence hierarchy is returned instead, clearly
marked as such via `reasoning_mode`. A forensic system that fails
closed is more dangerous than one that fails transparently.
"""

import json

from core.base_pin import BasePin
from config.settings import LLM_CONFIG, OUTPUTS_DIR
from layer5_llm_reasoning.evidence_builder import build_evidence_digest
from layer5_llm_reasoning.llm_client import (
    LLMError,
    LLMNotConfiguredError,
    complete,
    is_configured,
)
from layer5_llm_reasoning.prompts import build_system_prompt, build_user_prompt


VALID_VERDICTS = (
    "AUTHENTIC",
    "AI_GENERATED",
    "DEEPFAKE",
    "SUSPICIOUS",
    "INCONCLUSIVE",
)

# Pins whose output is synthesised by this stage
UPSTREAM_PINS = (
    "PIN-A1", "PIN-A2", "PIN-A3", "PIN-A4",
    "PIN-B1", "PIN-B2", "PIN-B3", "PIN-B4",
    "PIN-D1", "PIN-D2",
)


def _clamp(value, low: float = 0.0, high: float = 1.0, default: float = 0.5) -> float:
    """Coerce a model-supplied number into the valid probability range."""
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def _as_list(value) -> list:
    """Normalise a field that should be a list but may arrive as a scalar."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


class PinE1LLMReasoning(BasePin):
    """
    PIN-E1: synthesises the complete evidence set into a final verdict,
    a calibrated confidence and a natural-language forensic report.
    """

    def __init__(self):
        super().__init__(
            pin_id="PIN-E1",
            pin_name="LLM Reasoning Engine",
            layer=5,
        )

    # ── Response validation ─────────────────────────────────────────

    def _normalise_response(self, payload: dict) -> dict:
        """
        Validate and repair the model's JSON verdict.

        Language models occasionally return a synonym for a verdict
        label or a probability outside range. Rejecting the whole
        adjudication over a recoverable formatting deviation would be
        counterproductive, so known deviations are repaired and
        recorded.
        """
        if not isinstance(payload, dict):
            raise LLMError(
                f"Adjudication payload was {type(payload).__name__}, "
                f"expected an object"
            )

        verdict = str(payload.get("verdict", "")).strip().upper().replace(" ", "_")

        aliases = {
            "REAL": "AUTHENTIC",
            "GENUINE": "AUTHENTIC",
            "AI": "AI_GENERATED",
            "AI-GENERATED": "AI_GENERATED",
            "SYNTHETIC": "AI_GENERATED",
            "FAKE": "DEEPFAKE",
            "MANIPULATED": "DEEPFAKE",
            "UNCERTAIN": "INCONCLUSIVE",
        }
        verdict = aliases.get(verdict, verdict)

        if verdict not in VALID_VERDICTS:
            self.errors.append(
                f"Unrecognised verdict '{payload.get('verdict')}'; "
                f"defaulted to INCONCLUSIVE"
            )
            verdict = "INCONCLUSIVE"

        fake_probability = _clamp(payload.get("fake_probability"))
        confidence = _clamp(payload.get("confidence"), default=0.5)

        report = str(payload.get("report", "")).strip()
        if not report:
            self.errors.append("Model returned an empty narrative report")

        return {
            "verdict": verdict,
            "fake_probability": round(fake_probability, 4),
            "confidence": round(confidence, 4),
            "decisive_evidence": _as_list(payload.get("decisive_evidence")),
            "contradicting_evidence": _as_list(payload.get("contradicting_evidence")),
            "applied_rule": payload.get("applied_rule"),
            "failure_mode_flags": _as_list(payload.get("failure_mode_flags")),
            "manipulation_regions": payload.get("manipulation_regions"),
            "report": report,
            "recommendation": str(payload.get("recommendation", "")).strip(),
        }

    # ── Deterministic fallback ──────────────────────────────────────

    def _rule_based_adjudication(self, digest: dict) -> dict:
        """
        Deterministic adjudication used when the reasoning model is
        unavailable.

        Implements a reduced form of the same evidence hierarchy the
        language model applies: decisive provenance first, capture
        telemetry second, detector consensus last. It produces a usable
        verdict without any external service, at the cost of the
        narrative explanation and the nuanced conflict handling that
        motivate Layer 5 in the first place.
        """
        layer1 = digest.get("layer1_provenance", {})
        layer2 = digest.get("layer2_detection", {})
        a1 = layer1.get("PIN-A1_metadata", {})
        a2 = layer1.get("PIN-A2_c2pa", {})
        consensus = layer2.get("consensus", {})

        mean_fake = consensus.get("mean_fake_probability", 0.5)
        spread = consensus.get("spread", 0.0)

        # Tier 1 — decisive documentary provenance
        if a2.get("has_c2pa") and a2.get("source_is_ai"):
            return {
                "verdict": "AI_GENERATED",
                "fake_probability": 0.95,
                "confidence": 0.95,
                "decisive_evidence": [
                    f"PIN-A2: C2PA manifest declares "
                    f"digital_source_type={a2.get('digital_source_type')} "
                    f"(issuer={a2.get('issuer')})"
                ],
                "applied_rule": "R1",
            }

        if a1.get("ai_tool_signature"):
            return {
                "verdict": "AI_GENERATED",
                "fake_probability": 0.93,
                "confidence": 0.92,
                "decisive_evidence": [
                    f"PIN-A1: generator signature detected "
                    f"({a1.get('ai_tool_signature')})"
                ],
                "applied_rule": "R1",
            }

        # Tier 1 (partial) — the IPTC term `trainedAlgorithmicMedia` is
        # embedded in the container but PIN-A2 could not validate the
        # manifest cryptographically. The term itself is an explicit
        # producer declaration of synthesis, so it is treated as decisive;
        # confidence is reduced to reflect the unverified signature.
        markers = a1.get("c2pa_binary_markers", {})
        marker_list = markers.get("markers", [])
        if any("trainedAlgorithmicMedia" in str(m) for m in marker_list):
            issuer_hint = markers.get("issuer_hint")
            return {
                "verdict": "AI_GENERATED",
                "fake_probability": 0.88,
                "confidence": 0.82,
                "decisive_evidence": [
                    "PIN-A1: IPTC digital source type "
                    "'trainedAlgorithmicMedia' present in the C2PA "
                    "container — an explicit producer declaration of "
                    "synthesis"
                    + (f" (issuer hint: {issuer_hint})" if issuer_hint else ""),
                    "PIN-A2: manifest could not be validated "
                    "cryptographically, so confidence is reduced",
                ],
                "applied_rule": "R1",
            }

        # Tier 2 — capture telemetry contradicting detector consensus
        has_telemetry = (
            a1.get("camera_metadata_present")
            and a1.get("capture_datetime_present")
        )
        if has_telemetry and mean_fake >= 0.5:
            camera = a1.get("camera", {})
            return {
                "verdict": "SUSPICIOUS",
                "fake_probability": 0.45,
                "confidence": 0.55,
                "decisive_evidence": [
                    f"PIN-A1: coherent capture telemetry present "
                    f"({camera.get('make')} {camera.get('model')})",
                    f"Layer 2 consensus contradicts telemetry "
                    f"(mean fake probability {mean_fake})",
                ],
                "applied_rule": "R3",
                "failure_mode_flags": ["smartphone_computational_photography"],
            }

        if has_telemetry:
            return {
                "verdict": "AUTHENTIC",
                "fake_probability": round(mean_fake, 4),
                "confidence": 0.80,
                "decisive_evidence": [
                    "PIN-A1: coherent capture telemetry present",
                    f"Layer 2 consensus agrees (mean {mean_fake})",
                ],
                "applied_rule": "R2",
            }

        # Tier 3 — detector consensus
        if spread > 0.40:
            verdict, probability, confidence, rule = (
                "SUSPICIOUS", mean_fake, 0.50, "R5"
            )
        elif mean_fake >= 0.70:
            b4 = layer2.get("PIN-B4", {})
            is_deepfake = (
                b4.get("available")
                and (b4.get("p_deepfake") or 0) > (b4.get("p_ai_generated") or 0)
            )
            verdict = "DEEPFAKE" if is_deepfake else "AI_GENERATED"
            probability, confidence, rule = mean_fake, 0.75, "R4"
        elif mean_fake <= 0.30:
            verdict, probability, confidence, rule = (
                "AUTHENTIC", mean_fake, 0.70, "R4"
            )
        else:
            verdict, probability, confidence, rule = (
                "SUSPICIOUS", mean_fake, 0.55, "R4"
            )

        return {
            "verdict": verdict,
            "fake_probability": round(probability, 4),
            "confidence": confidence,
            "decisive_evidence": [
                f"Layer 2 consensus: mean fake probability {mean_fake} "
                f"across {consensus.get('model_count', 0)} detectors "
                f"(spread {spread})"
            ],
            "applied_rule": rule,
        }

    def _build_fallback_result(self, digest: dict, reason: str) -> dict:
        """Wrap the deterministic adjudication in the standard shape."""
        adjudication = self._rule_based_adjudication(digest)

        return {
            "verdict": adjudication["verdict"],
            "fake_probability": adjudication["fake_probability"],
            "confidence": adjudication["confidence"],
            "decisive_evidence": adjudication.get("decisive_evidence", []),
            "contradicting_evidence": [],
            "applied_rule": adjudication.get("applied_rule"),
            "failure_mode_flags": adjudication.get("failure_mode_flags", []),
            "manipulation_regions": None,
            "report": (
                f"[Kural tabanlı yedek muhakeme — LLM devre dışı: {reason}] "
                f"Sistem, kanıt hiyerarşisini deterministik olarak "
                f"uygulayarak '{adjudication['verdict']}' kararına ulaştı. "
                f"Doğal dil gerekçelendirmesi ve nüanslı çelişki çözümü "
                f"için Katman 5'in dil modeli yapılandırılmalıdır."
            ),
            "recommendation": (
                "Tam muhakeme raporu için .env dosyasına geçerli bir "
                "API anahtarı ekleyip analizi tekrarlayın."
            ),
        }

    # ── Persistence ─────────────────────────────────────────────────

    def _save_transcript(self, file_stem: str, system_prompt: str,
                         user_prompt: str, response: dict | None) -> str | None:
        """
        Persist the exact prompt payload and raw response.

        Forensic conclusions must be auditable: a verdict that cannot be
        traced back to the evidence and instructions that produced it
        has no evidentiary standing.
        """
        if not LLM_CONFIG["save_prompt_transcript"]:
            return None

        transcript_path = OUTPUTS_DIR / f"{file_stem}_PIN-E1_transcript.json"
        try:
            transcript_path.write_text(
                json.dumps(
                    {
                        "model": LLM_CONFIG["model"],
                        "temperature": LLM_CONFIG["temperature"],
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "response": response,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return str(transcript_path)
        except OSError as exc:
            self.errors.append(f"Transcript could not be written: {exc}")
            return None

    # ── Main analysis ───────────────────────────────────────────────

    def analyze(self, file_path: str) -> dict:
        from pathlib import Path

        file_stem = Path(file_path).stem

        # Collect upstream results supplied by the orchestrator
        results_by_pin = {
            pin_id: self.context[pin_id]
            for pin_id in UPSTREAM_PINS
            if pin_id in self.context
        }

        if not results_by_pin:
            return {
                "results": {
                    "reasoning_mode": "unavailable",
                    "error": "No upstream pin results were provided",
                },
                "score": 0.0,
                "verdict": "error",
                "details": (
                    "PIN-E1 upstream pin çıktısı almadı — "
                    "pipeline bağımlılıkları hatalı yapılandırılmış."
                ),
            }

        digest = build_evidence_digest(file_path, results_by_pin)
        digest_json = json.dumps(digest, ensure_ascii=False, separators=(",", ":"))

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(digest_json)

        reasoning_mode = "llm"
        usage: dict = {}
        model_name = LLM_CONFIG["model"]
        latency = None
        transcript_path = None

        if not is_configured():
            adjudication = self._build_fallback_result(
                digest, "API anahtarı yapılandırılmamış"
            )
            reasoning_mode = "rule_based_fallback"
        else:
            try:
                response = complete(system_prompt, user_prompt)
                usage = response.get("usage", {})
                model_name = response.get("model", model_name)
                latency = response.get("latency_seconds")
                # Persist before validation: a response that fails to
                # normalise is precisely the one worth inspecting.
                transcript_path = self._save_transcript(
                    file_stem, system_prompt, user_prompt, response["content"]
                )
                adjudication = self._normalise_response(response["content"])
            except LLMNotConfiguredError as exc:
                self.errors.append(str(exc))
                adjudication = self._build_fallback_result(digest, str(exc))
                reasoning_mode = "rule_based_fallback"
            except LLMError as exc:
                self.errors.append(str(exc))
                adjudication = self._build_fallback_result(digest, str(exc))
                reasoning_mode = "rule_based_fallback"
            except Exception as exc:  # noqa: BLE001 — must never fail closed
                self.errors.append(f"Unexpected adjudication failure: {exc}")
                adjudication = self._build_fallback_result(digest, str(exc))
                reasoning_mode = "rule_based_fallback"

        score = adjudication["fake_probability"]
        thresholds = LLM_CONFIG["thresholds"]
        if score >= thresholds["high_risk"]:
            risk_level = "high_risk"
        elif score >= thresholds["medium_risk"]:
            risk_level = "medium_risk"
        else:
            risk_level = "low_risk"

        results = {
            "reasoning_mode": reasoning_mode,
            "final_verdict": adjudication["verdict"],
            "fake_probability": adjudication["fake_probability"],
            "confidence": adjudication["confidence"],
            "applied_rule": adjudication.get("applied_rule"),
            "decisive_evidence": adjudication.get("decisive_evidence", []),
            "contradicting_evidence": adjudication.get("contradicting_evidence", []),
            "failure_mode_flags": adjudication.get("failure_mode_flags", []),
            "manipulation_regions": adjudication.get("manipulation_regions"),
            "report": adjudication["report"],
            "recommendation": adjudication.get("recommendation", ""),
            "evidence_digest": digest,
            "llm_metadata": {
                "model": model_name,
                "output_language": LLM_CONFIG["output_language"],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "latency_seconds": latency,
                "transcript": transcript_path,
            },
            "pins_synthesised": sorted(results_by_pin.keys()),
        }

        details = (
            f"Nihai karar: {adjudication['verdict']} "
            f"(sahtelik olasılığı: {adjudication['fake_probability']:.4f}, "
            f"güven: {adjudication['confidence']:.4f}, "
            f"kural: {adjudication.get('applied_rule') or '-'})"
        )

        return {
            "results": results,
            "score": score,
            "verdict": risk_level,
            "details": details,
        }
