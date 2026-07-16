"""
DeepReality — Base Pin Class
Tüm PIN modüllerinin miras alacağı temel sınıf.
Standart JSON çıktı formatı ve ortak yardımcı metodlar burada tanımlanır.
"""

import json
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from config.settings import OUTPUTS_DIR, OUTPUT_SCHEMA_VERSION


class BasePin(ABC):
    """
    Her PIN bu sınıftan türer.
    Standart çıktı formatı:
    {
        "schema_version": "1.0.0",
        "pin_id": "PIN-A1",
        "pin_name": "EXIF/Metadata Analysis",
        "layer": 1,
        "timestamp": "2026-02-16T...",
        "input_file": "image.jpg",
        "input_hash": "sha256...",
        "status": "success" | "error",
        "results": { ... },          # Pin'e özel sonuçlar
        "score": 0.0 - 1.0,          # 0 = temiz, 1 = kesin sahte
        "verdict": "low_risk" | "medium_risk" | "high_risk",
        "details": "Türkçe açıklama",
        "errors": []
    }
    """

    def __init__(self, pin_id: str, pin_name: str, layer: int):
        self.pin_id = pin_id
        self.pin_name = pin_name
        self.layer = layer
        self.errors: list[str] = []
        # Bağımlı pinler için üst pin sonuçları (orkestratör doldurur):
        #   {"PIN-A3": {...tam sonuç...}, "_pins": {"PIN-A3": <pin instance>}}
        self.context: dict = {}

    @abstractmethod
    def analyze(self, file_path: str) -> dict:
        """
        Her pin bu metodu kendi analiz mantığıyla doldurur.
        Returns: {"results": {...}, "score": float, "details": str}
        """
        pass

    def run(self, file_path: str, context: dict | None = None) -> dict:
        """
        Ana çalıştırıcı. analyze() metodunu çağırır,
        standart JSON formatına sarar, dosyaya kaydeder.

        context: Bu pinin bağımlı olduğu pinlerin sonuçları
                 (paralel orkestratör tarafından geçirilir).
        """
        self.errors = []
        self.context = context or {}
        file_path = Path(file_path)

        # Dosya kontrolü
        if not file_path.exists():
            return self._build_output(
                file_path=str(file_path),
                file_hash="",
                status="error",
                results={},
                score=0.0,
                verdict="error",
                details=f"Dosya bulunamadı: {file_path}"
            )

        # Dosya hash'i hesapla (tekrarlı analizleri önlemek ve takip için)
        file_hash = self._compute_hash(file_path)

        try:
            analysis = self.analyze(str(file_path))
            output = self._build_output(
                file_path=str(file_path),
                file_hash=file_hash,
                status="success",
                results=analysis.get("results", {}),
                score=analysis.get("score", 0.0),
                verdict=analysis.get("verdict", "low_risk"),
                details=analysis.get("details", "")
            )
        except Exception as e:
            self.errors.append(str(e))
            output = self._build_output(
                file_path=str(file_path),
                file_hash=file_hash,
                status="error",
                results={},
                score=0.0,
                verdict="error",
                details=f"Analiz hatası: {str(e)}"
            )

        # JSON dosyasına kaydet
        self._save_output(output, file_path.stem)
        return output

    def _build_output(self, file_path: str, file_hash: str,
                      status: str, results: dict, score: float,
                      verdict: str, details: str) -> dict:
        """Standart JSON çıktı formatını oluşturur."""
        return {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "pin_id": self.pin_id,
            "pin_name": self.pin_name,
            "layer": self.layer,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_file": str(file_path),
            "input_hash": file_hash,
            "status": status,
            "results": results,
            "score": round(score, 4),
            "verdict": verdict,
            "details": details,
            "errors": self.errors
        }

    def _save_output(self, output: dict, file_stem: str) -> Path:
        """JSON çıktısını outputs/ klasörüne kaydeder."""
        filename = f"{file_stem}_{self.pin_id}.json"
        output_path = OUTPUTS_DIR / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        return output_path

    @staticmethod
    def _compute_hash(file_path: Path) -> str:
        """Dosyanın SHA-256 hash'ini hesaplar."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()