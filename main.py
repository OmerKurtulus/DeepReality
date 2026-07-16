"""
DeepReality — Ana Çalıştırıcı
═════════════════════════════

Kullanım:
    1. input/ klasörüne görsellerini at
    2. python3 main.py
    3. Sonuçlar outputs/ klasörüne yazılır + terminalde özet gösterilir

Çalışma modeli (PIN Architecture — PARALEL):
    Bağımsız pinler aynı anda çalışır; sadece başka pinin çıktısını
    bekleyen pinler (XAI katmanı) bağımlılık grafiğine göre sıralanır.

    ┌── PARALEL ────────────────────────────────────────────────┐
    │ PIN-A1: EXIF/Metadata       → {dosya}_PIN-A1.json          │
    │ PIN-A2: C2PA Provenance      → {dosya}_PIN-A2.json         │
    │ PIN-A3: ELA                  → {dosya}_PIN-A3.json + png   │
    │ PIN-A4: Yüz Tespiti          → {dosya}_PIN-A4.json + png   │
    │ PIN-B1: CLIP ViT-L/14        → {dosya}_PIN-B1.json         │
    │ PIN-B2: SigLIP2-base-512     → {dosya}_PIN-B2.json         │
    │ PIN-B3: Frekans (DCT/DWT)    → {dosya}_PIN-B3.json         │
    │ PIN-B4: Independent Core     → {dosya}_PIN-B4.json         │
    └────────────────────────────────────────────────────────────┘
                  ↓ (B1+B2+B3 bitince)
    PIN-D1: Grad-CAM Heatmap (XAI)     → {dosya}_XAI_D1_*.png
                  ↓ (A3+D1 bitince)
    PIN-D2: Anomaly Localization (XAI) → {dosya}_XAI_D2_anomaly.png
"""

import sys
import json
import os
from pathlib import Path

# MediaPipe / TFLite uyarı mesajlarını bastır
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

# Proje kök dizinini ayarla
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# HEIC/HEIF desteğini kaydet (iPhone fotoğrafları)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # pillow-heif kurulu değilse sessizce devam et

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

# Layer 4 — XAI (Açıklanabilirlik)
from layer4_xai.pin_d1_gradcam import PinD1GradCam
from layer4_xai.pin_d2_anomaly import PinD2AnomalyLocalization

# Paralel PIN orkestratörü
from core.pipeline import PinPipeline

# Desteklenen görsel formatları
SUPPORTED_FORMATS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tiff", ".tif", ".gif",
    ".heic", ".heif",  # iPhone fotoğrafları
}


def find_images(input_dir: Path) -> list[Path]:
    """input/ klasöründeki tüm görselleri bulur."""
    images = []
    for file in sorted(input_dir.iterdir()):
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            images.append(file)
    return images


def print_pin_a1_summary(result: dict):
    """PIN-A1 sonucunu terminalde gösterir."""
    score = result["score"]
    verdict = result["verdict"]
    source_tool = result["results"].get("source_tool", None)
    exif_count = result["results"].get("exif_fields_found", 0)

    # Scoring method bilgisi
    breakdown = result["results"].get("score_breakdown", {})
    method = breakdown.get("_scoring_method", {})
    floor_applied = method.get("floor_applied", False)
    evidence_rule = method.get("evidence_rule")

    verdict_display = {
        "high_risk":   "YUKSEK RISK",
        "medium_risk": "ORTA RISK",
        "low_risk":    "DUSUK RISK",
        "error":       "HATA"
    }

    print(f"    Skor:     {score:.4f} → {verdict_display.get(verdict, verdict)}")
    print(f"    AI Araci: {source_tool if source_tool else 'Tespit edilmedi'}")
    print(f"    EXIF:     {exif_count} alan bulundu")
    if floor_applied and evidence_rule:
        print(f"    Kanit:    {evidence_rule}")


def print_pin_a2_summary(result: dict):
    """PIN-A2 sonucunu terminalde gösterir."""
    score = result["score"]
    verdict = result["verdict"]
    has_c2pa = result["results"].get("has_c2pa", False)

    verdict_display = {
        "high_risk":   "YUKSEK RISK",
        "medium_risk": "ORTA RISK",
        "low_risk":    "DUSUK RISK",
        "no_data":     "VERI YOK",
        "error":       "HATA"
    }

    print(f"    Skor:     {score:.4f} → {verdict_display.get(verdict, verdict)}")

    if has_c2pa:
        creator = result["results"].get("creator", {})
        tool = result["results"].get("tool", {})
        timestamp = result["results"].get("timestamp", {})
        validation = result["results"].get("validation", {})
        source_type = result["results"].get("digital_source_type", {})

        issuer = creator.get("issuer", "Bilinmiyor")
        tool_name = tool.get("claim_generator_parsed") or tool.get("claim_generator") or "Bilinmiyor"
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
    """PIN-A3 sonucunu terminalde gösterir."""
    score = result["score"]
    verdict = result["verdict"]

    verdict_display = {
        "high_risk":   "YUKSEK RISK",
        "medium_risk": "ORTA RISK",
        "low_risk":    "DUSUK RISK",
        "no_data":     "VERI YOK",
        "error":       "HATA"
    }

    print(f"    Skor:     {score:.4f} → {verdict_display.get(verdict, verdict)}")

    if verdict == "no_data":
        print(f"    ELA:      Sinyal yetersiz — sonuc guvenilir degil")
        return

    # Format bilgisi
    src_fmt = result["results"].get("source_format", {})
    fmt_ext = src_fmt.get("file_extension", "?")
    comp_type = src_fmt.get("compression_type", "?")
    print(f"    Format:   {fmt_ext.upper()} ({comp_type})")

    # Uniformity
    uniformity = result["results"].get("uniformity", {})
    u_cat = uniformity.get("category", "?")
    u_score = uniformity.get("uniformity_score", 0)
    u_display = {
        "very_uniform": "COK UNIFORM",
        "uniform": "UNIFORM",
        "moderate": "ORTA",
        "varied": "CESITLI"
    }
    print(f"    Uniform:  {u_display.get(u_cat, u_cat)} ({u_score:.1f}) [zayif sinyal]")

    # Global stats
    gs = result["results"].get("global_stats", {})
    print(f"    ELA Ort:  {gs.get('mean', 0):.1f} | Std: {gs.get('std', 0):.1f} | Max: {gs.get('max', 0):.1f}")

    # Anomaliler (hotspot + coldspot)
    hotspots = result["results"].get("manipulation_regions", [])
    if hotspots:
        hot = sum(1 for h in hotspots if h.get("type") == "hotspot")
        cold = sum(1 for h in hotspots if h.get("type") == "coldspot")
        high = sum(1 for h in hotspots if h.get("severity") == "high")
        print(f"    Anomali:  {len(hotspots)} bolge ({hot} hotspot + {cold} coldspot, {high} yuksek) [guclu sinyal]")
    else:
        print(f"    Anomali:  Yok")

    # Dominant signal
    bd = result["results"].get("score_breakdown", {}).get("_total", {})
    dom = bd.get("dominant_signal", "?")
    if dom == "hotspot":
        print(f"    Karar:    Manipulasyon tespit edildi")
    else:
        print(f"    Karar:    Manipulasyon izi yok — ELA destekleyici sinyal")

    # Heatmap
    heatmap = result["results"].get("ela_heatmap")
    if heatmap:
        print(f"    Heatmap:  {Path(heatmap).name}")


def print_pin_a4_summary(result: dict):
    """PIN-A4 sonucunu terminalde gösterir."""
    score = result["score"]
    verdict = result["verdict"]

    verdict_display = {
        "high_risk":   "YUKSEK RISK",
        "medium_risk": "ORTA RISK",
        "low_risk":    "DUSUK RISK",
        "no_data":     "VERI YOK",
        "error":       "HATA"
    }

    print(f"    Skor:     {score:.4f} → {verdict_display.get(verdict, verdict)}")

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

    faces = result["results"].get("faces", [])
    for face in faces:
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


def print_pin_b1_summary(result: dict):
    """PIN-B1 sonucunu terminalde gösterir."""
    score = result["score"]
    verdict = result["verdict"]

    verdict_display = {
        "high_risk":   "YUKSEK RISK",
        "medium_risk": "ORTA RISK",
        "low_risk":    "DUSUK RISK",
        "error":       "HATA"
    }

    print(f"    Skor:     {score:.4f} → {verdict_display.get(verdict, verdict)}")

    clip_verdict = result["results"].get("clip_verdict", "?")
    clip_prob = result["results"].get("clip_prob", 0)
    clip_conf = result["results"].get("clip_confidence", 0)

    print(f"    Karar:    {clip_verdict}")
    print(f"    Fake:     {clip_prob:.4f}")
    print(f"    Guven:    {clip_conf:.4f}")


def print_pin_b2_summary(result: dict):
    """PIN-B2 sonucunu terminalde gösterir."""
    score = result["score"]
    verdict = result["verdict"]

    verdict_display = {
        "high_risk":   "YUKSEK RISK",
        "medium_risk": "ORTA RISK",
        "low_risk":    "DUSUK RISK",
        "error":       "HATA"
    }

    print(f"    Skor:     {score:.4f} → {verdict_display.get(verdict, verdict)}")

    siglip_verdict = result["results"].get("siglip_verdict", "?")
    siglip_prob = result["results"].get("siglip_prob", 0)
    siglip_conf = result["results"].get("siglip_confidence", 0)

    print(f"    Karar:    {siglip_verdict}")
    print(f"    Fake:     {siglip_prob:.4f}")
    print(f"    Guven:    {siglip_conf:.4f}")


def print_pin_b3_summary(result: dict):
    """PIN-B3 sonucunu terminalde gösterir."""
    score = result["score"]
    verdict = result["verdict"]

    verdict_display = {
        "high_risk":   "YUKSEK RISK",
        "medium_risk": "ORTA RISK",
        "low_risk":    "DUSUK RISK",
        "error":       "HATA"
    }

    print(f"    Skor:     {score:.4f} → {verdict_display.get(verdict, verdict)}")

    freq_verdict = result["results"].get("freq_verdict", "?")
    freq_prob = result["results"].get("freq_prob", 0)
    freq_conf = result["results"].get("freq_confidence", 0)

    print(f"    Karar:    {freq_verdict}")
    print(f"    Fake:     {freq_prob:.4f}")
    print(f"    Guven:    {freq_conf:.4f}")


def print_pin_b4_summary(result: dict):
    """PIN-B4 sonucunu terminalde gösterir."""
    score = result["score"]
    verdict = result["verdict"]

    verdict_display = {
        "high_risk":   "YUKSEK RISK",
        "medium_risk": "ORTA RISK",
        "low_risk":    "DUSUK RISK",
        "error":       "HATA"
    }

    print(f"    Skor:     {score:.4f} → {verdict_display.get(verdict, verdict)}")

    predicted_class = result["results"].get("predicted_class", "?")
    ai_prob = result["results"].get("ai_prob", 0)
    deepfake_prob = result["results"].get("deepfake_prob", 0)
    real_prob = result["results"].get("real_prob", 0)
    confidence = result["results"].get("confidence", 0)

    print(f"    Karar:    {predicted_class}")
    print(f"    AI:       {ai_prob:.4f}")
    print(f"    Deepfake: {deepfake_prob:.4f}")
    print(f"    Real:     {real_prob:.4f}")
    print(f"    Guven:    {confidence:.4f}")


def print_pin_d1_summary(result: dict):
    """PIN-D1 (Grad-CAM) sonucunu terminalde gösterir."""
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

    combined_focus = focus.get("combined", [])
    print(f"    Odak:     {len(combined_focus)} bolge (birlesik harita)")
    if agreement is not None:
        print(f"    Uyum:     CLIP-SigLIP IoU = {agreement:.2f}")

    if heatmaps.get("combined"):
        print(f"    Heatmap:  {Path(heatmaps['combined']).name}")

    for tag, err in cam_errors.items():
        print(f"    Uyari:    {tag} CAM uretilemedi: {err[:60]}")


def print_pin_d2_summary(result: dict):
    """PIN-D2 (Anomaly Localization) sonucunu terminalde gösterir."""
    score = result["score"]
    verdict = result["verdict"]
    r = result["results"]

    verdict_display = {
        "high_risk":   "GUCLU KANIT",
        "medium_risk": "ORTA KANIT",
        "low_risk":    "ZAYIF KANIT",
        "error":       "HATA"
    }

    print(f"    Kanit:    {score:.4f} → {verdict_display.get(verdict, verdict)}")

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


# Terminal özet fonksiyonları — pipeline bittiğinde bu sırayla yazdırılır
PIN_DISPLAY_ORDER = [
    ("PIN-A1", "PIN-A1 (Metadata)",           print_pin_a1_summary),
    ("PIN-A2", "PIN-A2 (C2PA Provenance)",    print_pin_a2_summary),
    ("PIN-A3", "PIN-A3 (ELA)",                print_pin_a3_summary),
    ("PIN-A4", "PIN-A4 (Yuz Tespiti)",        print_pin_a4_summary),
    ("PIN-B1", "PIN-B1 (CLIP Detection)",     print_pin_b1_summary),
    ("PIN-B2", "PIN-B2 (SigLIP2 Detection)",  print_pin_b2_summary),
    ("PIN-B3", "PIN-B3 (Frekans Analizi)",    print_pin_b3_summary),
    ("PIN-B4", "PIN-B4 (Independent Core)",   print_pin_b4_summary),
    ("PIN-D1", "PIN-D1 (Grad-CAM XAI)",       print_pin_d1_summary),
    ("PIN-D2", "PIN-D2 (Anomali Lokalizasyon)", print_pin_d2_summary),
]


def _prewarm_imports():
    """
    transformers lazy-import kullanır ve İLK import thread-safe DEĞİLDİR:
    iki pin paralel thread'lerde aynı anda ilk kez 'from transformers
    import X' yaparsa import yarış durumuna düşüp sahte ImportError
    üretebilir. Bu yüzden pinlerin kullandığı tüm semboller paralel
    çalışma başlamadan ÖNCE ana thread'de bir kez çözülür — sonraki
    importlar basit sözlük erişimi olur (thread-safe).
    """
    from transformers import (          # noqa: F401
        CLIPModel, CLIPProcessor,                       # PIN-B1
        AutoModel, AutoProcessor,                       # PIN-B2
        AutoImageProcessor, SiglipForImageClassification,  # PIN-B4
    )


def build_pipeline() -> PinPipeline:
    """
    PIN Architecture bağımlılık grafiğini kurar.

    Katman 1 + Katman 2'nin 8 pini tamamen bağımsızdır → PARALEL.
    XAI pinleri model kararlarını görselleştirdiği için Katman 2
    çıktısını bekler:
        PIN-D1 ← PIN-B1, PIN-B2, PIN-B3  (model instance + skor paylaşımı)
        PIN-D2 ← PIN-A3 (ELA bölgeleri), PIN-D1 (CAM), PIN-B3 (frekans)
    """
    _prewarm_imports()

    pipeline = PinPipeline(max_workers=8)

    # Katman 1 — bağımsız
    pipeline.add_pin(PinA1Metadata())
    pipeline.add_pin(PinA2C2pa())
    pipeline.add_pin(PinA3Ela())
    pipeline.add_pin(PinA4Face())

    # Katman 2 — bağımsız
    pipeline.add_pin(PinB1Clip())
    pipeline.add_pin(PinB2Siglip())
    pipeline.add_pin(PinB3Freq())
    pipeline.add_pin(PinB4IndependentCore())

    # Katman 4 — XAI (bağımlı)
    pipeline.add_pin(PinD1GradCam(),
                     depends_on=["PIN-B1", "PIN-B2", "PIN-B3"])
    pipeline.add_pin(PinD2AnomalyLocalization(),
                     depends_on=["PIN-A3", "PIN-B3", "PIN-D1"])

    return pipeline


def main():
    input_dir = PROJECT_ROOT / "input"
    output_dir = PROJECT_ROOT / "outputs"

    # Klasörleri oluştur
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Görselleri bul
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
    print("  Layer 2: PIN-B1 (CLIP) + PIN-B2 (SigLIP2) + PIN-B3 (Frekans) + PIN-B4 (Independent Core)")
    print("  Layer 4: PIN-D1 (Grad-CAM) + PIN-D2 (Anomali Lokalizasyon)")
    print("  Bagimsiz pinler PARALEL calisir; XAI pinleri model ciktisini bekler")
    print("=" * 65)

    pipeline = build_pipeline()

    for i, image_path in enumerate(images, 1):
        print()
        print(f"  [{i}/{len(images)}] {image_path.name}")
        print("-" * 65)

        # Canlı ilerleme: her pin bittiği anda tek satır
        def on_pin_complete(pin_id, result, duration):
            status = "OK " if result["status"] == "success" else "HATA"
            print(f"    [{status}] {pin_id:<7} {duration:6.2f}s")

        run = pipeline.run(str(image_path), on_pin_complete=on_pin_complete)

        # Özetler (sabit sırada, pin bazında)
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
    print(f"  JSON ciktilar: outputs/ klasoru")
    print(f"  Her gorsel icin: *_PIN-A1 ... *_PIN-B4 + *_PIN-D1 + *_PIN-D2")
    print(f"  ELA heatmap'ler:    outputs/*_ELA_heatmap.png")
    print(f"  Yuz kirpmalari:     outputs/*_face_*.png")
    print(f"  Grad-CAM (XAI):     outputs/*_XAI_D1_*.png")
    print(f"  Anomali haritasi:   outputs/*_XAI_D2_anomaly.png")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()