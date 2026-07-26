"""
DeepReality — Command-Line Entry Point
======================================

Usage:
    1. Place images in the input/ directory
    2. python3 main.py
    3. Results are written to outputs/ and summarised in the terminal

Execution model (PIN Architecture — parallel):
    Independent pins execute concurrently. Only pins that consume the
    output of another pin are ordered, and the orchestrator derives that
    ordering automatically from the declared dependency graph.

    +-- CONCURRENT --------------------------------------------+
    | PIN-A1: EXIF / metadata        -> {stem}_PIN-A1.json      |
    | PIN-A2: C2PA provenance        -> {stem}_PIN-A2.json      |
    | PIN-A3: Error Level Analysis   -> {stem}_PIN-A3.json + png|
    | PIN-A4: Face detection         -> {stem}_PIN-A4.json + png|
    | PIN-B1: CLIP ViT-L/14          -> {stem}_PIN-B1.json      |
    | PIN-B2: SigLIP2-base-512       -> {stem}_PIN-B2.json      |
    | PIN-B3: Frequency (DCT/DWT)    -> {stem}_PIN-B3.json      |
    | PIN-B4: Independent Core       -> {stem}_PIN-B4.json      |
    +-----------------------------------------------------------+
                  | after B1 + B2 + B3
    PIN-D1: Grad-CAM heatmaps          -> {stem}_XAI_D1_*.png
                  | after A3 + B3 + D1
    PIN-D2: Anomaly localisation       -> {stem}_XAI_D2_anomaly.png
                  | after every upstream pin
    PIN-E1: LLM reasoning engine       -> {stem}_PIN-E1.json
                  | after every upstream pin plus E1
    PIN-F1: Ensemble fusion            -> {stem}_PIN-F1.json

Author: Omer Faruk Kurtulus
"""

import sys
import os
from pathlib import Path

# Suppress MediaPipe / TFLite console noise before those libraries load
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

# Ensure the project root is importable regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Register HEIC/HEIF support for iPhone photographs
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # Optional dependency; other formats remain unaffected

# Layer 1 — Preprocessing
from layer1_preprocessing.pin_a1_metadata import PinA1Metadata
from layer1_preprocessing.pin_a2_c2pa import PinA2C2pa
from layer1_preprocessing.pin_a3_ela import PinA3Ela
from layer1_preprocessing.pin_a4_face import PinA4Face

# Layer 2 — Detection Core
from layer2_detection_core.pin_b1_clip import PinB1Clip
from layer2_detection_core.pin_b2_siglip2 import PinB2Siglip
from layer2_detection_core.pin_b3_freq import PinB3Freq
from layer2_detection_core.pin_b4_IndependentCore import PinB4IndependentCore

# Layer 4 — Explainability
from layer4_xai.pin_d1_gradcam import PinD1GradCam
from layer4_xai.pin_d2_anomaly import PinD2AnomalyLocalization

# Layer 5 — LLM Reasoning Engine
from layer5_llm_reasoning.pin_e1_llm import PinE1LLMReasoning, UPSTREAM_PINS

# Layer 6 — Ensemble Fusion
from layer6_ensemble.pin_f1_ensemble import PinF1Ensemble, FEATURE_PINS

# Parallel pin orchestrator
from core.pipeline import PinPipeline

SUPPORTED_FORMATS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tiff", ".tif", ".gif",
    ".heic", ".heif",  # iPhone photographs
}

# Shared risk-level labels for terminal output
RISK_LABELS = {
    "high_risk":   "YUKSEK RISK",
    "medium_risk": "ORTA RISK",
    "low_risk":    "DUSUK RISK",
    "no_data":     "VERI YOK",
    "error":       "HATA",
}


def find_images(input_dir: Path) -> list[Path]:
    """Return every supported image file in the input directory."""
    images = []
    for file in sorted(input_dir.iterdir()):
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            images.append(file)
    return images


def print_pin_a1_summary(result: dict):
    """Render the PIN-A1 metadata result."""
    score = result["score"]
    verdict = result["verdict"]
    source_tool = result["results"].get("source_tool", None)
    exif_count = result["results"].get("exif_fields_found", 0)

    breakdown = result["results"].get("score_breakdown", {})
    method = breakdown.get("_scoring_method", {})
    floor_applied = method.get("floor_applied", False)
    evidence_rule = method.get("evidence_rule")

    print(f"    Skor:     {score:.4f} → {RISK_LABELS.get(verdict, verdict)}")
    print(f"    AI Araci: {source_tool if source_tool else 'Tespit edilmedi'}")
    print(f"    EXIF:     {exif_count} alan bulundu")
    if floor_applied and evidence_rule:
        print(f"    Kanit:    {evidence_rule}")


def print_pin_a2_summary(result: dict):
    """Render the PIN-A2 C2PA provenance result."""
    score = result["score"]
    verdict = result["verdict"]
    has_c2pa = result["results"].get("has_c2pa", False)

    print(f"    Skor:     {score:.4f} → {RISK_LABELS.get(verdict, verdict)}")

    if has_c2pa:
        creator = result["results"].get("creator", {})
        tool = result["results"].get("tool", {})
        timestamp = result["results"].get("timestamp", {})
        validation = result["results"].get("validation", {})
        source_type = result["results"].get("digital_source_type", {})

        issuer = creator.get("issuer", "Bilinmiyor")
        tool_name = (
            tool.get("claim_generator_parsed")
            or tool.get("claim_generator")
            or "Bilinmiyor"
        )
        sw_agent = tool.get("software_agent")
        sig_time = timestamp.get("signature_time", "Yok")
        is_valid = "GECERLI" if validation.get("is_valid") else "HATALI"

        print(f"    Issuer:   {issuer}")
        print(f"    Arac:     {tool_name}")
        if sw_agent:
            print(f"    Agent:    {sw_agent}")
        print(f"    Tarih:    {sig_time}")
        print(f"    Imza:     {is_valid}")

        if source_type.get("is_ai_source"):
            print(f"    Kaynak:   {source_type['source_category'].upper()}")

        if tool.get("is_known_ai_tool"):
            print(f"    AI Araci: {tool['matched_tool']}")
    else:
        print(f"    C2PA:     Bulunamadi (bu PIN sinyal uretemiyor)")


def print_pin_a3_summary(result: dict):
    """Render the PIN-A3 Error Level Analysis result."""
    score = result["score"]
    verdict = result["verdict"]

    print(f"    Skor:     {score:.4f} → {RISK_LABELS.get(verdict, verdict)}")

    if verdict == "no_data":
        print(f"    ELA:      Sinyal yetersiz — sonuc guvenilir degil")
        return

    src_fmt = result["results"].get("source_format", {})
    fmt_ext = src_fmt.get("file_extension", "?")
    comp_type = src_fmt.get("compression_type", "?")
    print(f"    Format:   {fmt_ext.upper()} ({comp_type})")

    uniformity = result["results"].get("uniformity", {})
    u_cat = uniformity.get("category", "?")
    u_score = uniformity.get("uniformity_score", 0)
    u_display = {
        "very_uniform": "COK UNIFORM",
        "uniform": "UNIFORM",
        "moderate": "ORTA",
        "varied": "CESITLI",
    }
    print(f"    Uniform:  {u_display.get(u_cat, u_cat)} ({u_score:.1f}) [zayif sinyal]")

    gs = result["results"].get("global_stats", {})
    print(
        f"    ELA Ort:  {gs.get('mean', 0):.1f} | "
        f"Std: {gs.get('std', 0):.1f} | Max: {gs.get('max', 0):.1f}"
    )

    hotspots = result["results"].get("manipulation_regions", [])
    if hotspots:
        hot = sum(1 for h in hotspots if h.get("type") == "hotspot")
        cold = sum(1 for h in hotspots if h.get("type") == "coldspot")
        high = sum(1 for h in hotspots if h.get("severity") == "high")
        print(
            f"    Anomali:  {len(hotspots)} bolge "
            f"({hot} hotspot + {cold} coldspot, {high} yuksek) [guclu sinyal]"
        )
    else:
        print(f"    Anomali:  Yok")

    bd = result["results"].get("score_breakdown", {}).get("_total", {})
    if bd.get("dominant_signal") == "hotspot":
        print(f"    Karar:    Manipulasyon tespit edildi")
    else:
        print(f"    Karar:    Manipulasyon izi yok — ELA destekleyici sinyal")

    heatmap = result["results"].get("ela_heatmap")
    if heatmap:
        print(f"    Heatmap:  {Path(heatmap).name}")


def print_pin_a4_summary(result: dict):
    """Render the PIN-A4 face detection result."""
    score = result["score"]
    verdict = result["verdict"]

    print(f"    Skor:     {score:.4f} → {RISK_LABELS.get(verdict, verdict)}")

    face_count = result["results"].get("face_count", 0)
    has_faces = result["results"].get("has_faces", False)

    if verdict == "error":
        error = result["results"].get("error", "Bilinmeyen hata")
        print(f"    Hata:     {error}")
        return

    if not has_faces:
        print(f"    Yuz:      Tespit edilemedi")
        return

    print(f"    Yuz:      {face_count} yuz tespit edildi")

    for face in result["results"].get("faces", []):
        fid = face["face_id"]
        bbox = face["bounding_box"]
        conf = bbox["confidence"]
        quality = face.get("quality", {})
        res_cat = quality.get("resolution_category", "?")
        sharpness = quality.get("sharpness", 0)
        alignment = face.get("alignment", {})
        frontal = alignment.get("is_frontal", None)
        roll = alignment.get("roll_angle", None)
        crop_path = face.get("crop_path", "")
        crop_name = Path(crop_path).name if crop_path else "?"

        frontal_str = "onden" if frontal else "yan" if frontal is not None else "?"
        roll_str = f"  aci={roll:.1f}°" if roll is not None else ""

        print(
            f"    #{fid}:       guven={conf:.2f} | "
            f"cozunurluk={res_cat} | "
            f"netlik={sharpness:.0f} | "
            f"{frontal_str}{roll_str} → {crop_name}"
        )


def _print_binary_detector(result: dict, prob_key: str,
                           conf_key: str, verdict_key: str):
    """Shared renderer for the binary Layer 2 detectors (B1, B2, B3)."""
    score = result["score"]
    verdict = result["verdict"]
    results = result["results"]

    print(f"    Skor:     {score:.4f} → {RISK_LABELS.get(verdict, verdict)}")
    print(f"    Karar:    {results.get(verdict_key, '?')}")
    print(f"    Fake:     {results.get(prob_key, 0):.4f}")
    print(f"    Guven:    {results.get(conf_key, 0):.4f}")


def print_pin_b1_summary(result: dict):
    """Render the PIN-B1 CLIP result."""
    _print_binary_detector(result, "clip_prob", "clip_confidence", "clip_verdict")


def print_pin_b2_summary(result: dict):
    """Render the PIN-B2 SigLIP2 result."""
    _print_binary_detector(result, "siglip_prob", "siglip_confidence",
                           "siglip_verdict")


def print_pin_b3_summary(result: dict):
    """Render the PIN-B3 frequency analysis result."""
    _print_binary_detector(result, "freq_prob", "freq_confidence", "freq_verdict")


def print_pin_b4_summary(result: dict):
    """Render the PIN-B4 three-class Independent Core result."""
    score = result["score"]
    verdict = result["verdict"]
    results = result["results"]

    print(f"    Skor:     {score:.4f} → {RISK_LABELS.get(verdict, verdict)}")
    print(f"    Karar:    {results.get('predicted_class', '?')}")
    print(f"    AI:       {results.get('ai_prob', 0):.4f}")
    print(f"    Deepfake: {results.get('deepfake_prob', 0):.4f}")
    print(f"    Real:     {results.get('real_prob', 0):.4f}")
    print(f"    Guven:    {results.get('confidence', 0):.4f}")


def print_pin_d1_summary(result: dict):
    """Render the PIN-D1 Grad-CAM result."""
    verdict = result["verdict"]
    r = result["results"]

    if verdict == "error":
        print(f"    Hata:     {result['details']}")
        return

    heatmaps = r.get("heatmaps", {})
    focus = r.get("focus_regions", {})
    agreement = r.get("model_agreement_iou")
    cam_errors = r.get("cam_errors", {})

    model_tags = [t for t in ("clip", "siglip", "freq") if t in heatmaps]
    print(f"    Modeller: {', '.join(model_tags) if model_tags else 'yok'}")
    print(f"    Odak:     {len(focus.get('combined', []))} bolge (birlesik harita)")

    if agreement is not None:
        print(f"    Uyum:     CLIP-SigLIP IoU = {agreement:.2f}")

    if heatmaps.get("combined"):
        print(f"    Heatmap:  {Path(heatmaps['combined']).name}")

    for tag, err in cam_errors.items():
        print(f"    Uyari:    {tag} CAM uretilemedi: {err[:60]}")


def print_pin_d2_summary(result: dict):
    """Render the PIN-D2 anomaly localisation result."""
    score = result["score"]
    verdict = result["verdict"]
    r = result["results"]

    evidence_labels = {
        "high_risk":   "GUCLU KANIT",
        "medium_risk": "ORTA KANIT",
        "low_risk":    "ZAYIF KANIT",
        "error":       "HATA",
    }

    print(f"    Kanit:    {score:.4f} → {evidence_labels.get(verdict, verdict)}")

    if verdict == "error":
        print(f"    Hata:     {result['details']}")
        return

    counts = r.get("region_counts", {})
    print(
        f"    Bolgeler: {counts.get('total', 0)} isaretli "
        f"({counts.get('fused', 0)} fuzyon + "
        f"{counts.get('ela_only', 0)} ELA + "
        f"{counts.get('gradcam_only', 0)} CAM)"
    )
    print(f"    Temel:    {r.get('evidence_basis', '?')}")

    annotated = r.get("annotated_image")
    if annotated:
        print(f"    Gorsel:   {Path(annotated).name}")


def print_pin_e1_summary(result: dict):
    """Render the PIN-E1 final adjudication."""
    r = result["results"]

    if result["verdict"] == "error":
        print(f"    Hata:     {result['details']}")
        return

    mode = r.get("reasoning_mode", "?")
    mode_label = {
        "llm": "LLM muhakemesi",
        "rule_based_fallback": "Kural tabanli yedek (LLM devre disi)",
    }.get(mode, mode)

    print(f"    KARAR:    {r.get('final_verdict', '?')}")
    print(f"    Olasilik: {r.get('fake_probability', 0):.4f}")
    print(f"    Guven:    {r.get('confidence', 0):.4f}")
    print(f"    Kural:    {r.get('applied_rule') or '-'}")
    print(f"    Mod:      {mode_label}")

    for flag in r.get("failure_mode_flags", []):
        print(f"    Uyari:    Olasi hata modu — {flag}")

    evidence = r.get("decisive_evidence", [])
    if evidence:
        print(f"    Kanit:")
        for item in evidence[:3]:
            text = item if len(item) <= 88 else item[:85] + "..."
            print(f"              - {text}")

    metadata = r.get("llm_metadata", {})
    if metadata.get("total_tokens"):
        print(
            f"    Token:    {metadata['total_tokens']} "
            f"({metadata.get('model', '?')})"
        )

    report = (r.get("report") or "").strip()
    if report:
        print()
        print("    ── Rapor ──")
        for paragraph in report.split("\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            # Wrap to terminal width without breaking words
            line = "    "
            for word in paragraph.split():
                if len(line) + len(word) + 1 > 76:
                    print(line)
                    line = "    " + word
                else:
                    line = f"{line} {word}" if line.strip() else line + word
            if line.strip():
                print(line)
            print()

    recommendation = (r.get("recommendation") or "").strip()
    if recommendation:
        print(f"    Oneri:    {recommendation}")


def print_pin_f1_summary(result: dict):
    """Render the PIN-F1 ensemble fusion result."""
    r = result["results"]

    if result["verdict"] == "error":
        print(f"    Hata:     {result['details']}")
        return

    status_label = {
        "trained": "Egitilmis model",
        "untrained": "Egitilmemis — agirlikli taban cizgi",
        "no_input": "Girdi yok",
    }.get(r.get("model_status"), r.get("model_status"))

    probability = r.get("fake_probability")
    if probability is None:
        print(f"    Skor:     uretilemedi")
    else:
        print(f"    Skor:     {probability:.4f} → {r.get('decision')}")
    print(f"    Model:    {status_label}")
    print(
        f"    Oznitelik: {r.get('features_available')}/{r.get('features_total')} mevcut"
    )

    info = r.get("model_info", {})
    metrics = info.get("holdout_metrics") or {}
    if metrics:
        # Report only the metrics the artefact actually recorded; the set
        # differs between a held-out split and a cross-validated fit.
        shown = [f"{k.upper().replace('_', '-')} {v}"
                 for k, v in metrics.items()
                 if k in ("roc_auc", "accuracy", "f1", "ece")]
        if shown:
            print(f"    Kalite:   {' | '.join(shown)}")

    for feature in (r.get("top_features") or [])[:3]:
        print(
            f"    Katki:    {feature['feature']} = {feature['value']} "
            f"(gain {feature['gain']})"
        )

    consensus = r.get("consensus", {})
    if consensus.get("available"):
        agreement_label = {
            "concordant": "UYUMLU",
            "expected_divergence": "BEKLENEN AYRISMA",
            "conflict": "CELISKI — inceleme onerilir",
            "indeterminate": "BELIRSIZ",
        }.get(consensus.get("agreement"), consensus.get("agreement"))
        print(f"    Uzlasma:  E1={consensus.get('e1_verdict')} "
              f"({consensus.get('e1_applied_rule')}) vs F1 → {agreement_label}")
        divergence = consensus.get("probability_divergence")
        if divergence is not None:
            print(f"    Fark:     {divergence:.4f}")
        interpretation = consensus.get("interpretation", "")
        if interpretation:
            line = "    "
            for word in interpretation.split():
                if len(line) + len(word) + 1 > 76:
                    print(line)
                    line = "    " + word
                else:
                    line = f"{line} {word}" if line.strip() else line + word
            if line.strip():
                print(line)

    if r.get("schema_warning"):
        print(f"    UYARI:    {r['schema_warning']}")


# Rendered in this fixed order once the parallel run completes, so that
# concurrent execution never affects the readability of the report.
PIN_DISPLAY_ORDER = [
    ("PIN-A1", "PIN-A1 (Metadata)",             print_pin_a1_summary),
    ("PIN-A2", "PIN-A2 (C2PA Provenance)",      print_pin_a2_summary),
    ("PIN-A3", "PIN-A3 (ELA)",                  print_pin_a3_summary),
    ("PIN-A4", "PIN-A4 (Yuz Tespiti)",          print_pin_a4_summary),
    ("PIN-B1", "PIN-B1 (CLIP Detection)",       print_pin_b1_summary),
    ("PIN-B2", "PIN-B2 (SigLIP2 Detection)",    print_pin_b2_summary),
    ("PIN-B3", "PIN-B3 (Frekans Analizi)",      print_pin_b3_summary),
    ("PIN-B4", "PIN-B4 (Independent Core)",     print_pin_b4_summary),
    ("PIN-D1", "PIN-D1 (Grad-CAM XAI)",         print_pin_d1_summary),
    ("PIN-D2", "PIN-D2 (Anomali Lokalizasyon)", print_pin_d2_summary),
    ("PIN-E1", "PIN-E1 (LLM Muhakeme Motoru)",  print_pin_e1_summary),
    ("PIN-F1", "PIN-F1 (Ensemble Fusion)",      print_pin_f1_summary),
]


def _prewarm_imports():
    """
    Resolve every transformers symbol on the main thread before the
    concurrent stage begins.

    The transformers package resolves submodules lazily. That first
    resolution is not thread-safe: when two pins execute
    `from transformers import X` simultaneously on separate threads, the
    import machinery can observe a partially initialised module and
    raise a spurious ImportError. Warming the cache here reduces every
    later import to a dictionary lookup, which is safe.
    """
    from transformers import (          # noqa: F401
        CLIPModel, CLIPProcessor,                          # PIN-B1
        AutoModel, AutoProcessor,                          # PIN-B2
        AutoImageProcessor, SiglipForImageClassification,  # PIN-B4
    )


def build_pipeline() -> PinPipeline:
    """
    Construct the PIN Architecture dependency graph.

    The eight pins of Layers 1 and 2 are mutually independent and
    therefore execute concurrently. Only the reasoning stages declare
    dependencies:

        PIN-D1 <- B1, B2, B3       (shares the loaded model instances)
        PIN-D2 <- A3, B3, D1       (ELA regions + Grad-CAM activation map)
        PIN-E1 <- every upstream pin (terminal adjudication node)
    """
    _prewarm_imports()

    pipeline = PinPipeline(max_workers=8)

    # Layer 1 — independent
    pipeline.add_pin(PinA1Metadata())
    pipeline.add_pin(PinA2C2pa())
    pipeline.add_pin(PinA3Ela())
    pipeline.add_pin(PinA4Face())

    # Layer 2 — independent
    pipeline.add_pin(PinB1Clip())
    pipeline.add_pin(PinB2Siglip())
    pipeline.add_pin(PinB3Freq())
    pipeline.add_pin(PinB4IndependentCore())

    # Layer 4 — explainability (dependent)
    pipeline.add_pin(PinD1GradCam(),
                     depends_on=["PIN-B1", "PIN-B2", "PIN-B3"])
    pipeline.add_pin(PinD2AnomalyLocalization(),
                     depends_on=["PIN-A3", "PIN-B3", "PIN-D1"])

    # Layer 5 — adjudication
    pipeline.add_pin(PinE1LLMReasoning(), depends_on=list(UPSTREAM_PINS))

    # Layer 6 — ensemble fusion (terminal node). It depends on PIN-E1 only
    # so that the consensus comparison can be made; E1's verdict is never
    # a model input.
    pipeline.add_pin(
        PinF1Ensemble(), depends_on=list(FEATURE_PINS) + ["PIN-E1"]
    )

    return pipeline


def main():
    input_dir = PROJECT_ROOT / "input"
    output_dir = PROJECT_ROOT / "outputs"

    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    images = find_images(input_dir)

    if not images:
        print()
        print("=" * 65)
        print("  DeepReality — Analiz Sistemi")
        print("=" * 65)
        print()
        print("  input/ klasoru bos!")
        print()
        print("  Yapman gereken:")
        print("    1. input/ klasorune gorsel at (jpg, png, webp)")
        print("    2. Tekrar calistir: python3 main.py")
        print()
        print("=" * 65)
        return

    print()
    print("=" * 65)
    print("  DeepReality — Analiz Sistemi (Paralel PIN Architecture)")
    print(f"  {len(images)} gorsel bulundu")
    print("  Layer 1: PIN-A1 (Metadata) + PIN-A2 (C2PA) + PIN-A3 (ELA) + PIN-A4 (Yuz)")
    print("  Layer 2: PIN-B1 (CLIP) + PIN-B2 (SigLIP2) + PIN-B3 (Frekans) + PIN-B4 (Core)")
    print("  Layer 4: PIN-D1 (Grad-CAM) + PIN-D2 (Anomali Lokalizasyon)")
    print("  Layer 5: PIN-E1 (LLM Muhakeme Motoru — nihai karar)")
    print("  Layer 6: PIN-F1 (Ensemble Fusion — istatistiksel capraz dogrulama)")
    print("  Bagimsiz pinler PARALEL calisir; bagimli pinler otomatik siralanir")
    print("=" * 65)

    pipeline = build_pipeline()

    for i, image_path in enumerate(images, 1):
        print()
        print(f"  [{i}/{len(images)}] {image_path.name}")
        print("-" * 65)

        def on_pin_complete(pin_id, result, duration):
            """Live progress line, emitted the moment each pin finishes."""
            status = "OK " if result["status"] == "success" else "HATA"
            print(f"    [{status}] {pin_id:<7} {duration:6.2f}s")

        run = pipeline.run(str(image_path), on_pin_complete=on_pin_complete)

        for pin_id, title, print_fn in PIN_DISPLAY_ORDER:
            if pin_id not in run.results:
                continue
            print()
            print(f"  {title}:")
            print_fn(run.results[pin_id])

        print()
        print(
            f"  Sure: {run.total_time:.1f}s paralel "
            f"(sirali olsaydi {run.sequential_time:.1f}s — "
            f"{run.speedup:.1f}x hizlanma)"
        )

    print()
    print("=" * 65)
    print(f"  Tamamlandi!")
    print(f"  JSON ciktilar:      outputs/ klasoru (gorsel basina 11 JSON)")
    print(f"  ELA heatmap'ler:    outputs/*_ELA_heatmap.png")
    print(f"  Yuz kirpmalari:     outputs/*_face_*.png")
    print(f"  Grad-CAM (XAI):     outputs/*_XAI_D1_*.png")
    print(f"  Anomali haritasi:   outputs/*_XAI_D2_anomaly.png")
    print(f"  Nihai karar:        outputs/*_PIN-E1.json")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
