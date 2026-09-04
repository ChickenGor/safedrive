"""A stable extension point for future vehicle/person collision assessment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CollisionRisk:
    """Normalized risk result that the warning manager consumes."""

    object_type: str  # Expected future values: "vehicle" or "person".
    risk_level: str   # Expected future values: "low", "medium", or "high".
    confidence: float
    bbox: tuple[int, int, int, int] | None = None


class CollisionDetector:
    """Collision interface only; no advanced collision logic is implemented yet."""

    def __init__(self, weights_path: str | None = None, confidence: float = 0.40) -> None:
        self.weights_path = weights_path
        self.confidence = confidence

    def detect(self, frame: Any) -> list[CollisionRisk]:
        """Return collision risks for one frame (currently an empty placeholder)."""
        _ = frame
        # TODO(collision-team): load a vehicle/person detector and estimate forward
        # collision risk. Keep this method's return type unchanged for integration.
        return []
