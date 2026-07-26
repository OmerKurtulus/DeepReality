# DeepReality

**A multi-layer forensic architecture for detecting AI-generated and manipulated imagery.**

DeepReality combines several mutually independent analytical paradigms — documentary provenance, compression physics, spatial deep learning, frequency-domain analysis and language-model reasoning — within a single modular pipeline. It is designed to address the central weakness of contemporary detection tools: the *generalisation gap*, whereby systems built on one model architecture recognise only the manipulation techniques present in their training distribution and degrade sharply on unseen generators.

![DeepReality PIN Architecture](docs/architecture.svg)

---

## Overview

Detection systems that rely on a single learned classifier inherit that classifier's blind spots. A model trained on face-swap deepfakes does not necessarily recognise diffusion-generated imagery; a model trained on one generator's artefacts may not transfer to the next. Because generative techniques evolve faster than detection corpora can be assembled, a monolithic detector is structurally destined to lag behind.

DeepReality takes the opposite approach. Eleven independent analysis modules — *pins* — each examine the image from a different epistemic position, and a reasoning stage adjudicates between them under an explicit evidence hierarchy. Where one pin is blind, another is not; where two disagree, the disagreement is itself treated as information rather than averaged away.

The system is additionally built on the principle that **documentary evidence outranks statistical inference**. A cryptographically signed C2PA manifest declaring an asset to be `trainedAlgorithmicMedia` is an assertion by its producer about how it was made. A neural detector's output is an inference about how it appears. When the two conflict, the assertion prevails — and encoding that ordering explicitly is what prevents the system from averaging incommensurable classes of evidence.

## PIN Architecture

The organising metaphor is the pin array of a processor: each pin carries one signal, independently, and the substrate integrates them into a single decision. Two properties follow.

**Concurrency.** Pins are executed by a dependency-graph orchestrator (`core/pipeline.py`). Pins with no mutual dependency run in parallel; a dependent pin starts the moment its own inputs are satisfied, without waiting for unrelated work. The eight Layer 1 and Layer 2 pins are fully independent and therefore execute concurrently; only the explainability and adjudication stages, which consume upstream output, are ordered.

**Extensibility and graceful degradation.** Adding, replacing or removing a pin does not disturb the remainder of the system, which allows rapid adaptation as generative techniques change. A pin that fails does not halt the pipeline: its absence is propagated through the evidence context, the adjudication stage reasons from what remains, and the resulting confidence is reduced accordingly. A partial answer with a stated limitation is more useful than no answer at all.

### Layers and pins

| Layer | Pin | Function | Technology |
|---|---|---|---|
| **1 — Preprocessing** | PIN-A1 | EXIF/metadata analysis; generator signatures and capture telemetry | Pillow, binary parsing |
| | PIN-A2 | C2PA Content Credentials — signed provenance verification | c2pa-python |
| | PIN-A3 | Error Level Analysis — localised recompression anomalies | OpenCV, Pillow |
| | PIN-A4 | Face detection, alignment and normalisation | MediaPipe (BlazeFace) |
| **2 — Detection Core** | PIN-B1 | CLIP ViT-L/14, frozen backbone with LayerNorm tuning | PyTorch, Transformers |
| | PIN-B2 | SigLIP2-base-512, full fine-tune — micro-anomaly resolution | PyTorch, Transformers |
| | PIN-B3 | DCT/DWT frequency transform with a purpose-built CNN | SciPy, PyWavelets |
| | PIN-B4 | Independent Core — three-class AI / Deepfake / Real taxonomy | SigLIP2 |
| **4 — Explainability** | PIN-D1 | Grad-CAM heatmaps recovering each detector's spatial support | PyTorch (hook-based) |
| | PIN-D2 | Anomaly localisation by ELA and Grad-CAM evidence fusion | OpenCV |
| **5 — Adjudication** | PIN-E1 | LLM reasoning engine — final verdict and forensic report | OpenAI-compatible API |

Layer 3 (video temporal analysis) and Layer 6 (an XGBoost ensemble meta-learner) are defined in the architecture and are not part of the present implementation.

### Detection core: rationale for paradigm diversity

The four Layer 2 detectors were selected to fail differently:

- **PIN-B1 (CLIP, frozen + LayerNorm tuning).** Approximately 365 K of 427 M parameters are trained, leaving the pretrained visual representation substantially intact. Because the backbone was never permitted to overfit the deepfake corpus, this detector degrades most gracefully on generators absent from training, and its dissent from the fine-tuned models therefore carries diagnostic weight.
- **PIN-B2 (SigLIP2, full fine-tune).** Operating at 512×512, the highest input resolution in the system, it resolves artefacts below the sampling limit of the other detectors. It achieves the highest in-distribution accuracy and is correspondingly the most sensitive to distribution shift.
- **PIN-B3 (frequency CNN).** Classifies a DCT/DWT representation rather than pixels, responding to upsampling periodicities and spectral signatures of GAN and diffusion pipelines that are invisible in the spatial domain. Because its domain is disjoint from that of B1 and B2, agreement between them constitutes genuine corroboration rather than a repeated measurement.
- **PIN-B4 (three-class Independent Core).** Supplies the taxonomic distinction the binary detectors cannot express — whether an image was synthesised outright or is authentic content that was subsequently altered.

### Explainability

Commercial detection tools are predominantly opaque: a score is returned without any account of its basis. Layer 4 converts model output into inspectable evidence.

**PIN-D1** applies Grad-CAM (Selvaraju et al., 2017) to the transformer backbones by differentiating the target-class logit with respect to the final encoder block's activations. The implementation uses PyTorch hooks and `torch.autograd.grad` directly rather than an external XAI package; because back-propagation terminates at the target layer, computation is roughly twenty times cheaper than a full backward pass. Cross-model spatial agreement is reported as an IoU, which distinguishes a shared reproducible cue from two detectors firing for unrelated reasons.

**PIN-D2** intersects the ELA anomaly regions of PIN-A3 with the combined attention map of PIN-D1. Regions where compression physics and model attention independently converge are marked as corroborated — the strongest localised manipulation evidence the system produces.

### Layer 5: the reasoning engine

The adjudication stage compresses the complete evidence set into a compact digest and submits it to a language model operating under a forensic reasoning protocol.

**Evidence compression.** Unabridged pin output for a single image exceeds 2,300 embedding values (CLIP 1024-d, SigLIP2 768-d, frequency CNN 512-d) alongside static model cards, per-region pixel ranges and facial landmark coordinates — of which essentially none carries adjudication value. The digest eliminates embeddings and model metadata, aggregates region lists into counts and representative extrema, and normalises every pin to a uniform shape. Measured across the test corpus this yields a **96–97 % token reduction** (approximately 17,000 tokens to 620) with no loss of decision-relevant information.

**Reasoning protocol.** The system prompt is not a general instruction to analyse data. It encodes the physical and statistical basis of every instrument, a four-tier evidence hierarchy, the documented failure modes of each detector, and a deterministic conflict-resolution procedure. Two principles are stated explicitly because both are routinely violated by naive implementations:

1. *Provenance dominates statistics.* Tier 1 documentary evidence constrains lower tiers and is not outvoted by them.
2. *Evidence is asymmetric.* The presence of coherent camera telemetry is strong evidence of authentic capture; its absence is weak evidence of anything, since virtually every social platform strips metadata on upload.

**Degradation.** Where no credential is configured or the provider is unreachable, a deterministic rule-based adjudication implementing the same hierarchy is returned instead, explicitly labelled through the `reasoning_mode` field. The system therefore remains functional without external dependencies, at the cost of the narrative justification and the nuanced conflict handling that motivate the stage.

---

## Installation

```bash
git clone https://github.com/<user>/DeepReality.git
cd DeepReality

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r Requirements.txt
```

Trained weights (~3.4 GB in total) are distributed separately from this repository. See [models/README.md](models/README.md) for download links and placement.

## Configuration

Layer 5 requires an API key for an OpenAI-compatible chat-completions endpoint. Copy the template and supply the credential:

```bash
cp .env.example .env
```

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Defaults — provider endpoint, model identifier, temperature, report language — are defined in `config/settings.py` under `LLM_CONFIG` and may be overridden through environment variables. The reasoning model is configurable; `anthropic/claude-sonnet-4.5` is the default. Layers 1, 2 and 4 operate entirely offline and require no credential.

## Usage

```bash
# Place the images to be analysed in input/
# Supported formats: jpg, png, webp, bmp, tiff, gif, heic/heif

python3 main.py
```

Each image is processed by the full pin set. Progress is reported per pin as it completes, followed by layer summaries and the adjudicated verdict. The compute device is selected automatically (CUDA → Apple MPS → CPU).

### Outputs

Written to `outputs/` for each analysed image:

| Artefact | Contents |
|---|---|
| `{image}_PIN-XX.json` | Standard envelope per pin — score, verdict, detailed findings (11 files) |
| `{image}_PIN-E1.json` | Final verdict, confidence, decisive evidence, narrative report |
| `{image}_PIN-E1_transcript.json` | Exact prompt payload and raw model response, for audit |
| `{image}_ELA_heatmap.png` | Error Level Analysis heatmap |
| `{image}_face_N.png` | Detected and normalised face crops |
| `{image}_XAI_D1_{model}.png` | Grad-CAM overlays (clip, siglip, freq, combined) |
| `{image}_XAI_D2_anomaly.png` | Annotated anomaly map with fused regions marked |

Every pin emits the same envelope, which is what allows the adjudication stage to apply a uniform interpretation protocol:

```json
{
    "pin_id": "PIN-B1",
    "layer": 2,
    "input_hash": "sha256...",
    "score": 0.0594,
    "verdict": "low_risk",
    "results": { "...pin-specific findings..." },
    "details": "Human-readable explanation"
}
```

## Project structure

```
DeepReality/
├── main.py                      # Entry point and pipeline construction
├── config/settings.py           # All thresholds, weights and paths
├── core/
│   ├── base_pin.py              # Standard pin contract (BasePin)
│   └── pipeline.py              # Dependency-graph parallel orchestrator
├── layer1_preprocessing/        # PIN-A1 … PIN-A4
├── layer2_detection_core/       # PIN-B1 … PIN-B4
├── layer3_video_temporal/       # (architecture placeholder)
├── layer4_xai/                  # PIN-D1, PIN-D2
├── layer5_llm_reasoning/
│   ├── evidence_builder.py      # Token-efficient evidence digest
│   ├── prompts.py               # Forensic reasoning protocol
│   ├── llm_client.py            # Provider client
│   └── pin_e1_llm.py            # PIN-E1
├── layer6_ensemble/             # (architecture placeholder)
├── models/                      # Trained weights — see models/README.md
├── input/                       # Images to analyse
└── outputs/                     # Analysis results
```

No decision constant is hard-coded inside a pin: thresholds, weights, model paths and prompt configuration are centralised in `config/settings.py`.

## Known limitations

Stated explicitly, since a detection system whose failure modes are undocumented cannot be evaluated.

- **Computational photography false positives.** Modern smartphones apply multi-frame fusion, denoising and skin smoothing, producing a low-noise texture statistically similar to generative output. The fine-tuned detectors over-report on authentic phone photographs as a result. Layer 5 recognises this pattern — authentic capture telemetry combined with high detector scores and background-concentrated attention — and downgrades the verdict accordingly, but the underlying detector bias remains.
- **Correlated consensus.** PIN-B1 through PIN-B3 share a training corpus (OpenDeepfake-Preview, ~20 K images). Their agreement is therefore partially correlated and does not constitute four independent confirmations; confidence derived from detector consensus alone is capped accordingly.
- **ELA scope.** Error Level Analysis presupposes a lossy compression history. On PNG, lossless WebP and converted sources its premise is weakened and its findings are treated as supporting evidence only. ELA localises *editing*, not *synthesis*: a wholly generated image is internally consistent and typically produces no anomalies at all.
- **Metadata is forgeable.** Capture telemetry is strong but rebuttable evidence. A determined adversary can fabricate EXIF fields, which is precisely why the architecture does not rest on any single evidence class.
- **Frequency-domain explainability.** The PIN-B3 Grad-CAM map is derived from a DCT/DWT representation and its correspondence to image coordinates is approximate; it is down-weighted in the combined attention map for that reason.

## Author

**Ömer Faruk Kurtuluş**

## References

1. Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
2. Zhai, X. et al. (2023). *Sigmoid Loss for Language Image Pre-Training.* ICCV 2023. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
3. Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
4. Rössler, A. et al. (2019). *FaceForensics++: Learning to Detect Manipulated Facial Images.* ICCV 2019. [arXiv:1901.08971](https://arxiv.org/abs/1901.08971)
5. Yan, Z. et al. (2024). *DF40: Toward Next-Generation Deepfake Detection.* [arXiv:2406.13156](https://arxiv.org/abs/2406.13156)
6. Coalition for Content Provenance and Authenticity (2024). *C2PA Technical Specification.* [c2pa.org](https://c2pa.org)
