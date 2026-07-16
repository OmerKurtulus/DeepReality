"""
DeepReality — PIN-A4: Yüz Tespiti & Kırpma (Face Detection & Cropping)
═══════════════════════════════════════════════════════════════════════

KATMAN:     Layer 1 — Ön İşlem (Pre-Processing)
PIN ID:     PIN-A4
TEKNOLOJİ:  MediaPipe Face Detection (BlazeFace)

AMAÇ:
    Görseldeki yüzleri tespit et, kırp, hizala ve normalize et.
    Sonraki katmanlar (özellikle Layer 2 deepfake detection) için
    hazır yüz görselleri üret.

    Bu PIN risk skoru ÜRETMEZ (score = 0.0).
    Preprocessing PIN'i olarak veri hazırlar, karar vermez.

API UYUMLULUĞU:
    Yeni MediaPipe (≥0.10.8): mp.tasks.vision.FaceDetector
    Eski MediaPipe (<0.10.8): mp.solutions.face_detection
    Otomatik seçim — her iki API desteklenir.

ÇIKTI:
    - face_count:   Tespit edilen yüz sayısı
    - faces[]:      Her yüz için bbox, landmarks, quality, alignment
    - face_crops[]: Normalize edilmiş yüz görsel dosya yolları
"""

import cv2
import numpy as np
from pathlib import Path
import os
import logging
import warnings
import urllib.request

# MediaPipe / TFLite / protobuf uyarı mesajlarını bastır
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
logging.getLogger("mediapipe").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# ── MediaPipe API tespiti ──
TASKS_API_AVAILABLE = False
SOLUTIONS_API_AVAILABLE = False

try:
    import mediapipe as mp

    # Yeni API: mp.tasks.vision.FaceDetector
    try:
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision as mp_tasks_vision
        TASKS_API_AVAILABLE = True
    except (ImportError, AttributeError):
        pass

    # Eski API: mp.solutions.face_detection
    if not TASKS_API_AVAILABLE:
        try:
            _test = mp.solutions.face_detection
            SOLUTIONS_API_AVAILABLE = True
        except AttributeError:
            pass

    MEDIAPIPE_AVAILABLE = TASKS_API_AVAILABLE or SOLUTIONS_API_AVAILABLE
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from core.base_pin import BasePin
from config.settings import OUTPUTS_DIR, PROJECT_ROOT

try:
    from config.settings import FACE_CONFIG
except ImportError:
    FACE_CONFIG = {}

# ── Model dosyası (Tasks API için) ──
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "blaze_face_short_range.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)


class PinA4Face(BasePin):
    """
    PIN-A4: Yüz Tespiti & Kırpma

    MediaPipe Face Detection (BlazeFace) ile yüzleri tespit eder,
    kırpar, hizalar ve normalize eder.

    İki API desteklenir:
        - Yeni: mp.tasks.vision.FaceDetector (model dosyası gerekir)
        - Eski: mp.solutions.face_detection (dahili model)

    Her yüz için:
        - Bounding box (konum + güvenilirlik)
        - 6 landmark (gözler, burun, ağız, kulaklar)
        - Hizalama bilgisi (roll açısı, önden mi?)
        - Kalite metrikleri (netlik, parlaklık, çözünürlük)
        - Normalize edilmiş yüz görseli (224×224 PNG)
    """

    # Landmark isimleri (her iki API'de aynı sıra)
    KP_NAMES = [
        "right_eye", "left_eye", "nose_tip",
        "mouth_center", "right_ear", "left_ear"
    ]

    def __init__(self):
        super().__init__(
            pin_id="PIN-A4",
            pin_name="Yuz Tespiti & Kirpma (Face Detection & Cropping)",
            layer=1
        )

        cfg = FACE_CONFIG

        self.model_selection = cfg.get("model_selection", 1)
        self.min_confidence = cfg.get("min_detection_confidence", 0.5)
        self.crop_margin = cfg.get("crop_margin", 0.30)
        self.normalized_size = tuple(cfg.get("normalized_size", [224, 224]))
        self.max_faces = cfg.get("max_faces", 10)

        # Hangi API kullanılacak
        self._api_mode = None
        if TASKS_API_AVAILABLE:
            self._api_mode = "tasks"
        elif SOLUTIONS_API_AVAILABLE:
            self._api_mode = "solutions"

    # ═══════════════════════════════════════════════════════════
    # ANA ANALİZ
    # ═══════════════════════════════════════════════════════════

    def analyze(self, file_path: str) -> dict:
        """Görseldeki yüzleri tespit et, kırp, hizala, normalize et."""
        file_path = Path(file_path)

        # ── MediaPipe kontrolü ──
        if not MEDIAPIPE_AVAILABLE:
            self.errors.append(
                "mediapipe kurulu degil. Kurulum: pip install mediapipe"
            )
            return {
                "results": {
                    "face_count": 0, "has_faces": False,
                    "detector": "unavailable",
                    "error": "mediapipe kurulu degil"
                },
                "score": 0.0,
                "verdict": "error",
                "details": (
                    "PIN-A4 calistirilmadi: mediapipe kurulu degil. "
                    "Kurulum: pip install mediapipe"
                )
            }

        # ── Tasks API model dosyası kontrolü ──
        if self._api_mode == "tasks":
            model_path = self._ensure_model()
            if model_path is None:
                self.errors.append(
                    f"Model dosyasi bulunamadi: {MODEL_FILENAME}"
                )
                return {
                    "results": {
                        "face_count": 0, "has_faces": False,
                        "detector": "unavailable",
                        "error": (
                            f"Model dosyasi bulunamadi. "
                            f"models/ klasorune {MODEL_FILENAME} indirin."
                        )
                    },
                    "score": 0.0,
                    "verdict": "error",
                    "details": (
                        f"PIN-A4 calistirilmadi: {MODEL_FILENAME} bulunamadi. "
                        f"Indirme komutu: python3 -c "
                        f"\"import urllib.request; "
                        f"urllib.request.urlretrieve("
                        f"'{MODEL_URL}', "
                        f"'models/{MODEL_FILENAME}')\""
                    )
                }

        # ── Görsel yükleme ──
        image_bgr = self._load_image(str(file_path))

        if image_bgr is None:
            self.errors.append(f"Gorsel okunamadi: {file_path.name}")
            return {
                "results": {
                    "face_count": 0, "has_faces": False,
                    "error": "Gorsel okunamadi"
                },
                "score": 0.0,
                "verdict": "error",
                "details": f"Gorsel okunamadi: {file_path.name}"
            }

        h, w = image_bgr.shape[:2]

        # ── Yüz tespiti (API'ye göre) ──
        if self._api_mode == "tasks":
            raw_detections = self._detect_faces_tasks(image_bgr)
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            raw_detections = self._detect_faces_solutions(image_rgb)

        # ── Her yüzü işle ──
        faces = []
        face_crop_paths = []

        for i, det in enumerate(raw_detections[:self.max_faces]):
            face_data = self._process_face(
                image_bgr=image_bgr,
                detection=det,
                face_idx=i,
                file_stem=file_path.stem,
                img_shape=(h, w)
            )

            if face_data is not None:
                faces.append(face_data)
                crop_path = face_data.get("crop_path")
                if crop_path:
                    face_crop_paths.append(crop_path)

        face_count = len(faces)

        # ── Sonuçlar ──
        results = {
            "face_count": face_count,
            "has_faces": face_count > 0,
            "detector": f"mediapipe_{self._api_mode}_api",
            "detector_config": {
                "model_selection": self.model_selection,
                "min_confidence": self.min_confidence,
                "crop_margin": self.crop_margin,
                "normalized_size": list(self.normalized_size),
            },
            "image_dimensions": {"width": w, "height": h},
            "faces": faces,
            "face_crops": face_crop_paths,
        }

        if face_count == 0:
            verdict = "no_data"
            details = (
                f"Yuz Tespiti tamamlandi ({w}x{h}). "
                f"Hicbir yuz tespit edilemedi. "
                f"Deepfake analizi icin yuz verisi yok."
            )
        else:
            verdict = "low_risk"
            crop_names = [Path(p).name for p in face_crop_paths]
            details = (
                f"Yuz Tespiti tamamlandi ({w}x{h}). "
                f"{face_count} yuz tespit edildi, kirpildi ve "
                f"{self.normalized_size[0]}x{self.normalized_size[1]} "
                f"boyutuna normalize edildi. "
                f"Ciktilar: {', '.join(crop_names)}. "
                f"Layer 2 deepfake analizi icin hazir."
            )

        return {
            "results": results,
            "score": 0.0,
            "verdict": verdict,
            "details": details
        }

    # ═══════════════════════════════════════════════════════════
    # MODEL YÖNETİMİ (Tasks API)
    # ═══════════════════════════════════════════════════════════

    def _ensure_model(self) -> Path | None:
        """
        Model dosyasını kontrol eder, yoksa otomatik indirir.

        Arama sırası:
            1. models/blaze_face_short_range.tflite
            2. Proje kök dizini/blaze_face_short_range.tflite
            3. Yoksa → otomatik indir
        """
        MODELS_DIR.mkdir(exist_ok=True)

        # 1. models/ klasöründe
        model_path = MODELS_DIR / MODEL_FILENAME
        if model_path.exists():
            return model_path

        # 2. Proje kök dizininde
        root_model = PROJECT_ROOT / MODEL_FILENAME
        if root_model.exists():
            return root_model

        # 3. Otomatik indir
        try:
            print(f"    [PIN-A4] Model indiriliyor: {MODEL_FILENAME}...")
            urllib.request.urlretrieve(MODEL_URL, str(model_path))
            if model_path.exists() and model_path.stat().st_size > 0:
                print(f"    [PIN-A4] Model indirildi: {model_path}")
                return model_path
        except Exception as e:
            self.errors.append(f"Model indirilemedi: {e}")

        return None

    # ═══════════════════════════════════════════════════════════
    # GÖRSEL YÜKLEME
    # ═══════════════════════════════════════════════════════════

    def _load_image(self, file_path: str) -> np.ndarray | None:
        """Görseli yükler. HEIC/HEIF desteği PIL üzerinden."""
        image = cv2.imread(file_path)
        if image is not None:
            return image

        try:
            from PIL import Image
            pil_img = Image.open(file_path).convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════
    # YÜZ TESPİTİ — TASKS API (Yeni MediaPipe ≥0.10.8)
    # ═══════════════════════════════════════════════════════════

    def _detect_faces_tasks(self, image_bgr: np.ndarray) -> list[dict]:
        """
        Yeni MediaPipe Tasks API ile yüz tespiti.
        Model: blaze_face_short_range.tflite gerektirir.
        """
        model_path = self._ensure_model()
        if model_path is None:
            return []

        h, w = image_bgr.shape[:2]

        try:
            base_options = mp_tasks_python.BaseOptions(
                model_asset_path=str(model_path)
            )
            options = mp_tasks_vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=self.min_confidence
            )

            with mp_tasks_vision.FaceDetector.create_from_options(
                options
            ) as detector:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=image_rgb
                )
                result = detector.detect(mp_image)

            detections = []
            if result.detections:
                for det in result.detections:
                    bb = det.bounding_box
                    confidence = det.categories[0].score

                    if confidence < self.min_confidence:
                        continue

                    keypoints = {}
                    if det.keypoints:
                        for kp_idx, kp in enumerate(det.keypoints):
                            if kp_idx < len(self.KP_NAMES):
                                keypoints[self.KP_NAMES[kp_idx]] = [
                                    round(float(kp.x * w), 1),
                                    round(float(kp.y * h), 1)
                                ]

                    detections.append({
                        "confidence": float(confidence),
                        "bbox": {
                            "x": int(bb.origin_x),
                            "y": int(bb.origin_y),
                            "width": int(bb.width),
                            "height": int(bb.height)
                        },
                        "keypoints": keypoints
                    })

            detections.sort(key=lambda d: d["confidence"], reverse=True)
            return detections

        except Exception as e:
            self.errors.append(f"Tasks API hatasi: {e}")
            return []

    # ═══════════════════════════════════════════════════════════
    # YÜZ TESPİTİ — SOLUTIONS API (Eski MediaPipe <0.10.8)
    # ═══════════════════════════════════════════════════════════

    def _detect_faces_solutions(self, image_rgb: np.ndarray) -> list[dict]:
        """
        Eski MediaPipe Solutions API ile yüz tespiti.
        Dahili model kullanır (indirme gerekmez).
        """
        h, w = image_rgb.shape[:2]
        detections = []

        try:
            with mp.solutions.face_detection.FaceDetection(
                model_selection=self.model_selection,
                min_detection_confidence=self.min_confidence
            ) as face_det:

                results = face_det.process(image_rgb)

                if results.detections:
                    for det in results.detections:
                        bbox_rel = det.location_data.relative_bounding_box
                        confidence = float(det.score[0])

                        bx = max(0, int(bbox_rel.xmin * w))
                        by = max(0, int(bbox_rel.ymin * h))
                        bw = min(int(bbox_rel.width * w), w - bx)
                        bh = min(int(bbox_rel.height * h), h - by)

                        keypoints = {}
                        kps = det.location_data.relative_keypoints
                        for kp_idx, kp_name in enumerate(self.KP_NAMES):
                            if kp_idx < len(kps):
                                kp = kps[kp_idx]
                                keypoints[kp_name] = [
                                    round(float(kp.x * w), 1),
                                    round(float(kp.y * h), 1)
                                ]

                        detections.append({
                            "confidence": confidence,
                            "bbox": {
                                "x": bx, "y": by,
                                "width": bw, "height": bh
                            },
                            "keypoints": keypoints
                        })

            detections.sort(key=lambda d: d["confidence"], reverse=True)

        except Exception as e:
            self.errors.append(f"Solutions API hatasi: {e}")

        return detections

    # ═══════════════════════════════════════════════════════════
    # TEK YÜZ İŞLEME (API-bağımsız)
    # ═══════════════════════════════════════════════════════════

    def _process_face(self, image_bgr: np.ndarray,
                      detection: dict, face_idx: int,
                      file_stem: str,
                      img_shape: tuple) -> dict | None:
        """
        Tek bir yüzü işler:
            1. Margin ile kırp
            2. Göz hizalaması yap
            3. 224×224 normalize et
            4. Kalite değerlendir
            5. PNG olarak kaydet
        """
        h, w = img_shape
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        landmarks = detection.get("keypoints", {})

        bx, by = bbox["x"], bbox["y"]
        bw, bh = bbox["width"], bbox["height"]

        if bw < 20 or bh < 20:
            return None

        # ── Kırpma ──
        face_crop = self._crop_with_margin(
            image_bgr, bx, by, bw, bh,
            self.crop_margin, (h, w)
        )
        if face_crop is None or face_crop.size == 0:
            return None

        # ── Hizalama ──
        aligned_crop = None
        alignment_info = {}

        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")

        if left_eye and right_eye:
            alignment_info = self._compute_alignment(left_eye, right_eye)
            aligned_crop = self._align_and_crop(
                image_bgr, left_eye, right_eye,
                bx, by, bw, bh, self.crop_margin, (h, w)
            )

        final_crop = aligned_crop if aligned_crop is not None else face_crop

        # ── Normalizasyon ──
        normalized = cv2.resize(
            final_crop, self.normalized_size,
            interpolation=cv2.INTER_AREA
        )

        # ── Kaydet ──
        crop_filename = f"{file_stem}_face_{face_idx}.png"
        crop_path = str(OUTPUTS_DIR / crop_filename)
        cv2.imwrite(crop_path, normalized)

        # ── Kalite ──
        quality = self._assess_quality(face_crop, bw, bh, (h, w))

        return {
            "face_id": face_idx,
            "bounding_box": {
                "x": bx, "y": by,
                "width": bw, "height": bh,
                "confidence": round(confidence, 4)
            },
            "landmarks": landmarks,
            "alignment": alignment_info,
            "quality": quality,
            "crop_path": crop_path,
            "crop_size": list(self.normalized_size)
        }

    # ═══════════════════════════════════════════════════════════
    # KIRPMA & HİZALAMA
    # ═══════════════════════════════════════════════════════════

    def _crop_with_margin(self, image: np.ndarray,
                          bx: int, by: int,
                          bw: int, bh: int,
                          margin: float,
                          img_shape: tuple) -> np.ndarray | None:
        """Yüzü margin ile kırpar."""
        h, w = img_shape
        mx, my = int(bw * margin), int(bh * margin)
        x1 = max(0, bx - mx)
        y1 = max(0, by - my)
        x2 = min(w, bx + bw + mx)
        y2 = min(h, by + bh + my)
        crop = image[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _compute_alignment(self, left_eye: list,
                           right_eye: list) -> dict:
        """Göz pozisyonlarından hizalama bilgisi hesaplar."""
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        roll_angle = float(np.degrees(np.arctan2(dy, dx)))
        inter_eye_dist = float(np.sqrt(dx ** 2 + dy ** 2))
        return {
            "roll_angle": round(roll_angle, 2),
            "inter_eye_distance": round(inter_eye_dist, 1),
            "is_frontal": abs(roll_angle) < 15.0
        }

    def _align_and_crop(self, image: np.ndarray,
                        left_eye: list, right_eye: list,
                        bx: int, by: int,
                        bw: int, bh: int,
                        margin: float,
                        img_shape: tuple) -> np.ndarray | None:
        """Gözlere göre yüzü hizalar ve kırpar."""
        h, w = img_shape
        eye_cx = (left_eye[0] + right_eye[0]) / 2.0
        eye_cy = (left_eye[1] + right_eye[1]) / 2.0
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        angle = float(np.degrees(np.arctan2(dy, dx)))

        if abs(angle) < 0.5:
            return None

        rotation_matrix = cv2.getRotationMatrix2D(
            (eye_cx, eye_cy), angle, 1.0
        )
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        return self._crop_with_margin(
            rotated, bx, by, bw, bh, margin, (h, w)
        )

    # ═══════════════════════════════════════════════════════════
    # KALİTE DEĞERLENDİRME
    # ═══════════════════════════════════════════════════════════

    def _assess_quality(self, face_crop: np.ndarray,
                        face_w: int, face_h: int,
                        img_shape: tuple) -> dict:
        """Yüz kalite metrikleri hesaplar."""
        gray = (cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                if len(face_crop.shape) == 3 else face_crop)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())
        blur_score = max(0.0, min(1.0, 1.0 - (sharpness / 500.0)))

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        img_h, img_w = img_shape
        face_area = face_w * face_h
        image_area = img_h * img_w
        face_ratio = face_area / image_area if image_area > 0 else 0.0

        face_pixels = face_w * face_h
        if face_pixels >= 150 * 150:
            resolution = "high"
        elif face_pixels >= 80 * 80:
            resolution = "medium"
        else:
            resolution = "low"

        return {
            "sharpness": round(sharpness, 1),
            "blur_score": round(blur_score, 3),
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "face_area_ratio": round(face_ratio, 4),
            "face_pixel_count": face_pixels,
            "resolution_category": resolution
        }