"""
DeepReality — Base Pin Class

Abstract base class inherited by every pin in the system. It defines
the standard JSON output envelope, the dependency-context mechanism
and the shared utility methods, so that each pin implements only its
own analysis logic.
"""

import json
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from config.settings import OUTPUTS_DIR, OUTPUT_SCHEMA_VERSION


class BasePin(ABC):
    """
    Base class for every analysis pin.

    Standard output envelope:
    {
        "schema_version": "1.0.0",
        "pin_id": "PIN-A1",
        "pin_name": "EXIF/Metadata Analysis",
        "layer": 1,
        "timestamp": "2026-02-16T...",
        "input_file": "image.jpg",
        "input_hash": "sha256...",
        "status": "success" | "error",
        "results": { ... },          # Pin-specific findings
        "score": 0.0 - 1.0,          # 0 = authentic, 1 = certainly fake
        "verdict": "low_risk" | "medium_risk" | "high_risk",
        "details": "Human-readable explanation",
        "errors": []
    }
    """

    def __init__(self, pin_id: str, pin_name: str, layer: int):
        self.pin_id = pin_id
        self.pin_name = pin_name
        self.layer = layer
        self.errors: list[str] = []
        # Upstream results for dependent pins, populated by the orchestrator:
        #   {"PIN-A3": {...full result...}, "_pins": {"PIN-A3": <pin instance>}}
        self.context: dict = {}

    @abstractmethod
    def analyze(self, file_path: str) -> dict:
        """
        Implemented by each pin with its own analysis logic.

        Returns: {"results": {...}, "score": float, "details": str}
        """
        pass

    def run(self, file_path: str, context: dict | None = None) -> dict:
        """
        Execute the pin: invoke analyze(), wrap the outcome in the
        standard envelope and persist it to disk.

        Args:
            file_path: Image to analyse.
            context:   Results of the pins this pin depends on, supplied
                       by the parallel orchestrator.
        """
        self.errors = []
        self.context = context or {}
        file_path = Path(file_path)

        # Reject a missing input before any work is attempted
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

        # Content hash: deduplicates repeated analyses and provides a
        # stable identifier for downstream auditing
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

        # Persist the envelope as JSON
        self._save_output(output, file_path.stem)
        return output

    def _build_output(self, file_path: str, file_hash: str,
                      status: str, results: dict, score: float,
                      verdict: str, details: str) -> dict:
        """Assemble the standard JSON output envelope."""
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
        """Write the JSON envelope to the outputs directory."""
        filename = f"{file_stem}_{self.pin_id}.json"
        output_path = OUTPUTS_DIR / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        return output_path

    @staticmethod
    def _compute_hash(file_path: Path) -> str:
        """Compute the SHA-256 digest of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()