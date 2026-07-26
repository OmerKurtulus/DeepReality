# DeepReality — Architecture Reference

Technical companion to the [README](README.md). This document describes the internal contracts, the execution model and the responsibilities of each module. It is intended for readers extending the system rather than operating it.

---

## 1. Execution model

### 1.1 The pin contract

Every analysis module inherits `BasePin` (`core/base_pin.py`) and implements a single method:

```python
def analyze(self, file_path: str) -> dict:
    return {"results": {...}, "score": 0.0, "verdict": "low_risk", "details": "..."}
```

`BasePin.run()` wraps that return value in the standard envelope, computes the SHA-256 content hash, captures exceptions, and persists the result to `outputs/{stem}_{PIN-ID}.json`. A pin that raises does not propagate the exception: it emits an envelope with `status: "error"`, which the orchestrator treats as a completed dependency so that downstream pins may still proceed with reduced evidence.

The uniform envelope is what allows Layer 5 to apply one interpretation protocol to twelve heterogeneous instruments:

| Field | Meaning |
|---|---|
| `pin_id`, `pin_name`, `layer` | Module identity |
| `input_file`, `input_hash` | Analysed file and its SHA-256 digest |
| `status` | `success` or `error` |
| `score` | 0.0 = authentic, 1.0 = certainly synthetic |
| `verdict` | `low_risk` / `medium_risk` / `high_risk`, or a pin-specific band |
| `results` | Pin-specific findings |
| `details` | Human-readable explanation |
| `errors` | Accumulated error strings |

**Score semantics are uniform with two documented exceptions.** PIN-D1 emits no risk score (0.0, verdict `informational`), and PIN-D2's score expresses the strength of *localised manipulation evidence* rather than a fake probability. Both exceptions are declared in the envelope and honoured by the adjudication stage; neither is averaged with detector probabilities.

### 1.2 Dependency graph and concurrency

`core/pipeline.py` implements `PinPipeline`, a DAG-driven executor.

```python
pipeline = PinPipeline(max_workers=8)
pipeline.add_pin(pin_a1)                                   # independent
pipeline.add_pin(pin_d1, depends_on=["PIN-B1", "PIN-B2"])  # dependent
run = pipeline.run(image_path, on_pin_complete=callback)
```

`_validate()` rejects missing dependencies and cycles before execution begins. The scheduler then loops: every pin whose dependencies are satisfied is dispatched immediately, and the loop blocks on `FIRST_COMPLETED` so a newly satisfied pin starts without waiting for unrelated work.

The current graph:

See [docs/dependency-graph.svg](docs/dependency-graph.svg) for the rendered
graph. In text form:

```
PIN-A1  PIN-A2  PIN-A3  PIN-A4  PIN-B1  PIN-B2  PIN-B3  PIN-B4     (concurrent)
                                   |       |       |
                                   +-------+-------+
                                           v
                                        PIN-D1
                                           |
                    PIN-A3, PIN-B3 --------+
                                           v
                                        PIN-D2
                                           |
                            all upstream --+
                                           v
                                        PIN-E1
                                           |
                     all upstream + E1 ----+
                                           v
                                        PIN-F1  ->  consensus
```

**Threads rather than processes.** PyTorch inference, NumPy/OpenCV kernels and file I/O all release the GIL, so threads achieve real parallelism on this workload. More importantly, the loaded models — several gigabytes in aggregate — are shared from memory rather than duplicated per process.

`PipelineRun` reports `total_time` (wall clock), `sequential_time` (summed pin runtime) and `speedup`. Measured speed-up on the test corpus is approximately 2×; the ceiling is set by the three heavyweight detectors contending for the same accelerator.

### 1.3 Dependency context

Dependent pins receive a `context` dictionary:

```python
{
    "PIN-B1": {...complete PIN-B1 envelope...},
    "_pins":  {"PIN-B1": <PinB1Clip instance>}
}
```

The `_pins` entry passes live object references, which is how PIN-D2 obtains PIN-D1's raw NumPy activation maps without serialising them through JSON, and how PIN-D1 reuses the already-loaded detector models instead of allocating its own.

### 1.4 Thread-safety note

`main._prewarm_imports()` resolves every `transformers` symbol on the main thread before the concurrent stage begins. The package resolves submodules lazily, and that first resolution is not thread-safe: two pins importing simultaneously can observe a partially initialised module and raise a spurious `ImportError`. This was observed in practice before the warm-up was added.

---

## 2. Module reference

### 2.1 Layer 1 — Preprocessing

| Module | Responsibility |
|---|---|
| `pin_a1_metadata.py` | EXIF, XMP, PNG tEXt and JPEG COM extraction; generator-signature matching; heuristic C2PA byte scan; two-tier scoring (evidence floor over weighted sum) |
| `pin_a2_c2pa.py` | C2PA manifest parsing and cryptographic validation via `c2pa-python`; full manifest-chain traversal with field enrichment from parent manifests |
| `pin_a3_ela.py` | JPEG re-save at Q=90, per-pixel difference, 8×8 regional grid, MAD-based hotspot/coldspot detection, heatmap rendering |
| `pin_a4_face.py` | MediaPipe BlazeFace detection, eye-axis alignment, 224×224 normalisation, quality metrics |

**PIN-A1 scoring** is deliberately two-tier. Decisive evidence — a C2PA marker with a known AI issuer, or a high-confidence generator signature — establishes a score floor that overrides the weighted sum, because a producer's own declaration cannot be outvoted by an accumulation of weak heuristics. Absent such evidence, the weighted sum of nine signals applies.

**PIN-A2 chain traversal** exists because some providers place the informative claim in a parent manifest. OpenAI, for example, emits `c2pa.created` + `GPT-4o` + `trainedAlgorithmicMedia` in manifest 1 and only `c2pa.opened` in the active manifest. Reading the active manifest alone would miss the decisive evidence entirely.

**PIN-A3 constraints** are documented in the source: ELA presupposes a lossy compression history, so its uniformity signal is unreliable on PNG and lossless WebP. Uniformity is capped as a weak supporting signal (max 0.25); anomaly detection, which is format-independent, carries the strong signal (max 0.85).

### 2.2 Layer 2 — Detection Core

| Module | Architecture | Input | Trainable / total |
|---|---|---|---|
| `pin_b1_clip.py` | CLIP ViT-L/14, frozen + LayerNorm tuning | 224×224 | 365 K / 427 M |
| `pin_b2_siglip2.py` | SigLIP2-base-patch16-512, full fine-tune | 512×512 | 93.7 M / 376 M |
| `pin_b3_freq.py` | Custom five-block CNN on DCT/DWT maps | 224×224 × 4 ch | 4.8 M |
| `pin_b4_IndependentCore.py` | SigLIP2 three-class classifier | 224×224 | pretrained |

Each module caches its model in a module-level global, so repeated instantiation is free and Layer 4 can reuse the loaded weights.

**PIN-B3's frequency transform** (`image_to_frequency_map`) produces four channels: the DCT log-magnitude spectrum plus the LH, HL and HH wavelet subbands. This representation is statistically independent of the spatial detectors, which is why agreement between B3 and B1/B2 constitutes genuine corroboration rather than a repeated measurement.

**PIN-B4** is the only multi-class detector and supplies the AI-generated versus deepfake distinction that the binary detectors cannot express. Its ensemble-compatible score is `ai_prob + deepfake_prob`.

### 2.3 Layer 4 — Explainability

**`pin_d1_gradcam.py`** implements Grad-CAM without an external XAI dependency. The target layer is located by name (`encoder.layers.*.layer_norm1`, final block). Gradients are obtained through `torch.autograd.grad(logit, activation)` rather than a full `backward()`, so back-propagation terminates at the target layer — roughly twenty times cheaper than differentiating the entire 24-block ViT, and it never touches parameter `.grad` fields.

ViT token activations are reshaped onto the patch grid, with automatic CLS-token detection: if the token count is a perfect square the sequence has no CLS (SigLIP), and if `count - 1` is a perfect square the leading token is discarded (CLIP).

Raw CAM matrices are retained on the instance in `cam_cache` for PIN-D2 and are deliberately excluded from the JSON envelope.

**`pin_d2_anomaly.py`** merges adjacent ELA grid cells, then measures mean CAM activation inside each merged region. Regions above the confirmation threshold are marked `fused`.

One non-obvious gate is important: because the CAM is normalised, *every* image has a hottest region. When no detector reports synthesis, those regions indicate only where the model looked, not that anything is anomalous. CAM-only regions are therefore marked only when at least one Layer 2 model exceeds the medium-risk threshold. Without this gate the pin annotates clean images with meaningless boxes.

### 2.4 Layer 5 — Adjudication

| Module | Responsibility |
|---|---|
| `evidence_builder.py` | Compresses every pin envelope into a token-efficient digest |
| `prompts.py` | Encodes the forensic reasoning protocol and evidence hierarchy |
| `llm_client.py` | OpenAI-compatible client with retry and JSON recovery |
| `pin_e1_llm.py` | PIN-E1: orchestration, response validation, deterministic fallback |

**Evidence compression** applies three reductions — elimination (embeddings, model cards, landmark coordinates), aggregation (region lists to counts and extrema) and normalisation (uniform shape per pin). Measured reduction is 96–97 %, roughly 17,000 tokens to 620, with no loss of decision-relevant content.

**The reasoning protocol** (`prompts.SYSTEM_PROMPT`, ~3,900 tokens) specifies:

1. An instrument reference — the physical or statistical basis of each pin, and its reliability characteristics.
2. A four-tier evidence hierarchy: decisive provenance, capture telemetry, learned detectors, localisation and attention.
3. Documented failure modes, including the smartphone computational-photography false positive and correlated consensus among B1–B3.
4. A deterministic conflict-resolution procedure (rules R1–R6) applied in order.
5. A five-value verdict taxonomy with consistency constraints between verdict, probability and confidence.

Static instrument characteristics live in the system prompt rather than the per-image payload, which keeps the digest small while preserving the interpretive context needed to weight each detector correctly.

**Response handling.** `_normalise_response()` repairs recoverable deviations — verdict synonyms, out-of-range probabilities — rather than discarding an otherwise valid adjudication, and records the repair. `llm_client.extract_json_object()` recovers JSON from fenced or prose-wrapped responses.

**Deterministic fallback.** When no credential is configured or the provider is unreachable, `_rule_based_adjudication()` applies a reduced form of the same hierarchy and returns a usable verdict marked `reasoning_mode: "rule_based_fallback"`. The system remains functional offline, at the cost of the narrative justification.

**Auditability.** With `save_prompt_transcript` enabled, the exact prompt payload and raw response are written to `outputs/{stem}_PIN-E1_transcript.json`. A forensic conclusion that cannot be traced to the evidence and instructions that produced it has no evidentiary standing.

### 2.5 Layer 6 — Ensemble Fusion

| Module | Responsibility |
|---|---|
| `feature_extractor.py` | The 54-column feature contract, imported by both training and inference |
| `booster_eval.py` | Evaluates the saved XGBoost JSON without the xgboost runtime |
| `pin_f1_ensemble.py` | PIN-F1: scoring, calibration, feature attribution, E1 consensus |

**One contract, two consumers.** Training and inference import the same
`extract_features()`, which removes the most common defect in stacked
ensembles — a meta-learner scored against a vector assembled differently
from the one it was fitted on. The contract is versioned, and the trained
artefact records the version it was built against so a mismatch warns
rather than mis-scores silently.

**Missing evidence is NaN, not zero.** Zero is a measurement; NaN is an
absence. Gradient-boosted trees learn a default traversal direction for
NaN at every split, so the distinction survives into the model instead of
being flattened at the input. This is what lets the ensemble degrade
coherently when a pin fails.

**No xgboost at inference.** The library's macOS wheels link against
Homebrew's OpenMP runtime while PyTorch bundles its own; loading both into
one process and entering a parallel region segfaults the interpreter.
Since every pin imports torch, the ensemble stage would crash on every
prediction. `booster_eval.NativeBooster` traverses the saved JSON
directly — a few thousand comparisons for a single row — which removes the
conflict and the dependency together. Equivalence with xgboost is asserted
by `tests/test_booster_eval.py`, currently matching to 2.9e-07.

**Scope.** The deployed model uses 33 of the 54 columns. Provenance and
face-composition features are withheld because the rule calculus of Layer 5
already adjudicates provenance, and because both groups separate the
training corpora at 0.95+ AUC, which makes them shortcut vectors for a
statistical stage.

**Consensus, not a flag.** Divergence from PIN-E1 is classified rather
than merely detected: dissent is *expected* when Layer 5 decided on
documentary grounds (R1–R3), because F1 does not observe provenance, and
constitutes a genuine *conflict* only under statistical rules (R4–R5).
Collapsing the two would raise a warning on every C2PA-signed image.

---

## 3. Configuration

All tunable parameters are centralised in `config/settings.py`; no decision constant is hard-coded inside a pin.

| Block | Governs |
|---|---|
| `METADATA_CONFIG` | Generator signatures, camera/GPS field lists, C2PA markers, scoring weights |
| `ELA_CONFIG` | Re-save quality, amplification, grid size, MAD thresholds, uniformity bands |
| `C2PA_CONFIG` | IPTC source types, known AI issuers and software agents, action taxonomy |
| `FACE_CONFIG` | Model selection, confidence floor, crop margin, normalised size |
| `CLIP_CONFIG`, `SIGLIP_CONFIG`, `FREQ_CONFIG`, `INDEPENDENT_CORE_CONFIG` | Model paths, label maps, verdict thresholds |
| `XAI_CONFIG` | Target class, colormap, focus thresholds, fusion parameters, evidence scores |
| `LLM_CONFIG` | Provider endpoint, model, temperature, digest limits, output language |

`_load_dotenv()` reads `.env` at import time without an external dependency. Existing environment variables take precedence, so a shell-exported key overrides the file.

---

## 4. Extending the system

To add a pin:

1. Subclass `BasePin`, implement `analyze()`, return the standard shape.
2. Add its configuration block to `config/settings.py`.
3. Register it in `main.build_pipeline()` with its `depends_on` list.
4. Add a summary renderer and an entry in `PIN_DISPLAY_ORDER`.
5. If the pin should inform adjudication, extend `evidence_builder.py` and describe the instrument in `prompts.SYSTEM_PROMPT` — a pin the reasoning stage cannot interpret contributes nothing to the verdict.

The one remaining architectural stage is Layer 3 — video temporal analysis: frame consistency, lip-audio synchronisation and biological signal.

---

## 5. Language conventions

Source comments, docstrings and developer-facing error messages are written in English. Strings that reach the end user — the `details` field, terminal summaries and the Layer 5 narrative report — are Turkish, since the natural-language forensic report is a deliverable of the system rather than an implementation detail. The report language is configurable through `LLM_CONFIG["output_language"]`.
