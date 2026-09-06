"""YOLO traffic-object detection with explainable forward-risk assessment."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ultralytics import YOLO


SUPPORTED_OBJECTS = {"vehicle", "person"}

# Image-relative trapezoid: begin below the horizon and follow the ego vehicle's
# likely lane. The previous broad zone included roadside scenery in narrow-road
# dashcam footage, making the demo overlay misleading.
# Points are ordered top-left, top-right, bottom-right, bottom-left.
# The near-road corridor begins below the horizon/vanishing area.  It is a
# qualitative visual/risk heuristic, deliberately not a lane detector.
FORWARD_REGION = ((0.43, 0.58), (0.57, 0.58), (0.80, 0.98), (0.20, 0.98))


@dataclass(frozen=True)
class CollisionRisk:
    """Normalized risk result consumed by :class:`WarningManager`."""

    object_type: str
    risk_level: str
    confidence: float
    bbox: tuple[int, int, int, int] | None = None


@dataclass
class _Track:
    """Small amount of temporal state used for transparent risk rules."""

    object_type: str
    bbox: tuple[int, int, int, int]
    area_history: deque[float] = field(default_factory=lambda: deque(maxlen=4))
    missed_frames: int = 0


class CollisionDetector:
    """Detect vehicles/people and estimate qualitative forward collision risk.

    Risk is based only on image evidence. It is not a distance, time-to-collision,
    or collision-probability estimate. Detections are associated frame-to-frame by
    class and bounding-box IoU so that apparent-size growth can be measured.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        confidence: float = 0.40,
        *,
        image_size: int = 512,
        iou_match_threshold: float = 0.30,
        max_missed_frames: int = 4,
        forward_region: tuple[tuple[float, float], ...] = FORWARD_REGION,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= iou_match_threshold <= 1.0:
            raise ValueError("iou_match_threshold must be between 0 and 1")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative")
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.confidence = confidence
        self.image_size = image_size
        self.iou_match_threshold = iou_match_threshold
        self.max_missed_frames = max_missed_frames
        self.forward_region = forward_region
        self.model: YOLO | None = None
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 0

    def load(self) -> None:
        """Load the fine-tuned checkpoint lazily."""
        if self.weights_path is None or not self.weights_path.is_file():
            expected = self.weights_path or Path("models/collision/best.pt")
            raise FileNotFoundError(
                f"Collision weights not found: {expected}. "
                "Place the trained best.pt file in models/collision/."
            )
        self.model = YOLO(str(self.weights_path))

    def reset(self) -> None:
        """Discard temporal state before processing an unrelated video."""
        self._tracks.clear()
        self._next_track_id = 0

    def set_forward_region(self, forward_region: tuple[tuple[float, float], ...]) -> None:
        """Set a manually calibrated image-relative forward corridor."""
        if len(forward_region) != 4:
            raise ValueError("forward_region must contain four trapezoid points")
        if any(not (0.0 <= coordinate <= 1.0) for point in forward_region for coordinate in point):
            raise ValueError("forward_region coordinates must be normalized between 0 and 1")
        self.forward_region = forward_region

    def detect(self, frame: Any) -> list[CollisionRisk]:
        """Return a LOW/MEDIUM/HIGH risk for each detected vehicle or person."""
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("frame must be an image-like array with height and width")
        frame_height, frame_width = int(frame.shape[0]), int(frame.shape[1])
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError("frame height and width must be positive")
        if self.model is None:
            self.load()

        raw_detections: list[tuple[str, float, tuple[int, int, int, int]]] = []
        for result in self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        ):
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = str(names[class_id]).strip().lower()
                if label not in SUPPORTED_OBJECTS:
                    continue
                coordinates = tuple(int(round(value)) for value in box.xyxy[0].tolist())
                bbox = self._clip_box(coordinates, frame_width, frame_height)
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                raw_detections.append((label, float(box.conf[0].item()), bbox))

        assignments = self._associate(raw_detections)
        risks: list[CollisionRisk] = []
        matched_track_ids: set[int] = set()
        frame_area = float(frame_width * frame_height)

        for detection_index, (label, score, bbox) in enumerate(raw_detections):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = self._new_track(label, bbox)
            track = self._tracks[track_id]
            track.bbox = bbox
            track.missed_frames = 0
            normalized_area = self._box_area(bbox) / frame_area
            track.area_history.append(normalized_area)
            matched_track_ids.add(track_id)
            risk_level = self._risk_level(track, frame_width, frame_height)
            risks.append(CollisionRisk(label, risk_level, score, bbox))

        self._expire_unmatched_tracks(matched_track_ids)
        return risks

    def _associate(
        self,
        detections: list[tuple[str, float, tuple[int, int, int, int]]],
    ) -> dict[int, int]:
        """Greedily match same-class boxes to existing tracks by descending IoU."""
        candidates: list[tuple[float, int, int]] = []
        for detection_index, (label, _, bbox) in enumerate(detections):
            for track_id, track in self._tracks.items():
                if track.object_type != label:
                    continue
                overlap = self._iou(bbox, track.bbox)
                if overlap >= self.iou_match_threshold:
                    candidates.append((overlap, detection_index, track_id))

        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, detection_index, track_id in sorted(candidates, reverse=True):
            if detection_index not in assignments and track_id not in used_tracks:
                assignments[detection_index] = track_id
                used_tracks.add(track_id)
        return assignments

    def _new_track(self, label: str, bbox: tuple[int, int, int, int]) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        self._tracks[track_id] = _Track(label, bbox)
        return track_id

    def _expire_unmatched_tracks(self, matched_track_ids: set[int]) -> None:
        for track_id in list(self._tracks):
            if track_id not in matched_track_ids:
                self._tracks[track_id].missed_frames += 1
                if self._tracks[track_id].missed_frames > self.max_missed_frames:
                    del self._tracks[track_id]

    def _risk_level(self, track: _Track, width: int, height: int) -> str:
        bbox = track.bbox
        centre_x = (bbox[0] + bbox[2]) / (2.0 * width)
        bottom_y = bbox[3] / height
        in_forward_region = self._inside_forward_region(centre_x, bottom_y, self.forward_region)
        current_area = track.area_history[-1]

        # Outside-lane detections and very small targets remain LOW regardless of
        # noisy single-frame size changes.
        if not in_forward_region or current_area < 0.010:
            return "low"
        if len(track.area_history) < 3:
            return "low"

        previous, current = track.area_history[-2], track.area_history[-1]
        before_previous = track.area_history[-3]
        last_growth = current / max(previous, 1e-9) - 1.0
        prior_growth = previous / max(before_previous, 1e-9) - 1.0
        centred = abs(centre_x - 0.5) <= 0.15

        # HIGH requires a sizeable, centred target and rapid growth on two
        # consecutive frame transitions. MEDIUM accepts milder sustained growth.
        if centred and current_area >= 0.020 and min(prior_growth, last_growth) >= 0.12:
            return "high"
        if current_area >= 0.012 and (prior_growth + last_growth) / 2.0 >= 0.06:
            return "medium"
        return "low"

    @staticmethod
    def _inside_forward_region(
        x: float, y: float, forward_region: tuple[tuple[float, float], ...] = FORWARD_REGION
    ) -> bool:
        """Return whether a point is inside the chosen four-corner polygon.

        This supports a manually calibrated, non-symmetric road corridor rather
        than assuming its left and right edges share the same height.
        """
        inside = False
        for index, (start_x, start_y) in enumerate(forward_region):
            end_x, end_y = forward_region[index - 1]
            crosses_scanline = (start_y > y) != (end_y > y)
            if crosses_scanline:
                intersection_x = (end_x - start_x) * (y - start_y) / (end_y - start_y) + start_x
                if x < intersection_x:
                    inside = not inside
        return inside

    @staticmethod
    def _clip_box(
        bbox: tuple[int, int, int, int], width: int, height: int
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        return (
            min(max(x1, 0), width),
            min(max(y1, 0), height),
            min(max(x2, 0), width),
            min(max(y2, 0), height),
        )

    @staticmethod
    def _box_area(bbox: tuple[int, int, int, int]) -> int:
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    @staticmethod
    def _iou(
        first: tuple[int, int, int, int], second: tuple[int, int, int, int]
    ) -> float:
        intersection = (
            max(0, min(first[2], second[2]) - max(first[0], second[0]))
            * max(0, min(first[3], second[3]) - max(first[1], second[1]))
        )
        union = CollisionDetector._box_area(first) + CollisionDetector._box_area(second) - intersection
        return intersection / union if union else 0.0
