"""YOLO-backed road-hazard detection, isolated from the rest of SafeDrive."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO


SUPPORTED_HAZARDS = {"pothole", "crack", "manhole"}

# The shipped checkpoint was trained AND validated at 960 px. Ultralytics defaults
# predict() to 640, which silently degrades accuracy against the reported metrics
# (val mAP50 0.5506 at 960 vs 0.5285 at 640). Keep this aligned with the weights.
MODEL_IMAGE_SIZE = 960


@dataclass(frozen=True)
class HazardDetection:
    """A normalized result returned by the road-hazard module."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    # Frame-relative (x1, y1, x2, y2) in 0-1. Added so the warning module can test
    # "is this hazard in the forward driving region?" without knowing the frame
    # resolution - evaluate() receives detections only, never the frame itself.
    # Optional with a default, so any existing construction of this class still works.
    bbox_norm: tuple[float, float, float, float] | None = None

    @property
    def centre_norm(self) -> tuple[float, float] | None:
        """Frame-relative centre point, or None if normalized coords are absent."""
        if self.bbox_norm is None:
            return None
        x1, y1, x2, y2 = self.bbox_norm
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class RoadHazardDetector:
    """Runs a teammate-supplied YOLO model for potholes, cracks, and manholes."""

    def __init__(
        self,
        weights_path: str | Path,
        confidence: float = 0.35,
        imgsz: int = MODEL_IMAGE_SIZE,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.confidence = confidence
        self.imgsz = imgsz
        self.model: YOLO | None = None

    def load(self) -> None:
        """Load weights lazily, allowing the rest of the app to be imported safely."""
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Road-hazard weights not found: {self.weights_path}. "
                "Place the trained best.pt file in models/road_hazard/."
            )
        self.model = YOLO(str(self.weights_path))

        # Fail loudly on a checkpoint that does not match the integration contract,
        # rather than silently returning zero detections at runtime.
        names = {str(name).lower() for name in self.model.names.values()}
        if not names & SUPPORTED_HAZARDS:
            raise ValueError(
                f"{self.weights_path} exposes classes {sorted(names)}, none of which are "
                f"{sorted(SUPPORTED_HAZARDS)}. Wrong checkpoint?"
            )

    def detect(self, frame: Any) -> list[HazardDetection]:
        """Return only the three hazard categories supported by this project."""
        if self.model is None:
            self.load()

        results = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            verbose=False,
        )
        detections: list[HazardDetection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = str(names[class_id]).lower()
                if label not in SUPPORTED_HAZARDS:
                    continue
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                nx1, ny1, nx2, ny2 = (float(value) for value in box.xyxyn[0].tolist())
                detections.append(
                    HazardDetection(
                        label,
                        float(box.conf[0].item()),
                        (x1, y1, x2, y2),
                        (nx1, ny1, nx2, ny2),
                    )
                )
        return detections
