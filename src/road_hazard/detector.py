"""YOLO-backed road-hazard detection, isolated from the rest of SafeDrive."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO


SUPPORTED_HAZARDS = {"pothole", "crack", "manhole"}


@dataclass(frozen=True)
class HazardDetection:
    """A normalized result returned by the road-hazard module."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


class RoadHazardDetector:
    """Runs a teammate-supplied YOLO model for potholes, cracks, and manholes."""

    def __init__(self, weights_path: str | Path, confidence: float = 0.35) -> None:
        self.weights_path = Path(weights_path)
        self.confidence = confidence
        self.model: YOLO | None = None

    def load(self) -> None:
        """Load weights lazily, allowing the rest of the app to be imported safely."""
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Road-hazard weights not found: {self.weights_path}. "
                "Place the trained best.pt file in models/road_hazard/."
            )
        self.model = YOLO(str(self.weights_path))

    def detect(self, frame: Any) -> list[HazardDetection]:
        """Return only the three hazard categories supported by this project."""
        if self.model is None:
            self.load()

        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        detections: list[HazardDetection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = str(names[class_id]).lower()
                if label not in SUPPORTED_HAZARDS:
                    continue
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                detections.append(
                    HazardDetection(label, float(box.conf[0].item()), (x1, y1, x2, y2))
                )
        return detections

    # TODO(road-hazard-team): add model-specific preprocessing or tracking here,
    # while preserving the detect(frame) -> list[HazardDetection] contract.
