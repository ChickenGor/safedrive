"""Decision policy that combines independent perception modules."""
from __future__ import annotations

from dataclasses import dataclass

from collision.detector import CollisionRisk
from road_hazard.detector import HazardDetection


@dataclass(frozen=True)
class Warning:
    priority: int
    category: str
    message: str


class WarningManager:
    """Prioritize forward collision risk before road-hazard notifications."""

    # Normalized trapezoid used for road hazards. It is narrow near the horizon
    # and wider near the vehicle, approximating the ego vehicle's drivable path.
    FORWARD_REGION = ((0.35, 0.55), (0.65, 0.55), (0.90, 1.00), (0.10, 1.00))

    def evaluate(
        self, road_results: list[HazardDetection], collision_results: list[CollisionRisk]
    ) -> Warning | None:
        """Return the single warning the driver should see right now.

        This is the integration contract for the warning teammate. It deliberately
        suppresses lower-priority road warnings when a forward collision warning is
        active, avoiding competing messages in the final interface.
        """
        warnings = self.create_warnings(collision_results, road_results)
        return warnings[0] if warnings else None

    def create_warnings(
        self, collision_risks: list[CollisionRisk], hazards: list[HazardDetection]
    ) -> list[Warning]:
        """Create supported warnings in deterministic priority order.

        Unknown labels and low collision risks are intentionally ignored. A
        pothole is actionable only when its normalized centre is in the forward
        driving region; crack and manhole warnings follow the assignment policy.
        """
        warnings: list[Warning] = []
        for risk in collision_risks:
            level = str(risk.risk_level).strip().lower()
            if level in {"high", "medium"}:
                warnings.append(
                    Warning(
                        1 if level == "high" else 2,
                        "collision",
                        f"{level.upper()} FORWARD COLLISION RISK: {risk.object_type}",
                    )
                )

        for hazard in hazards:
            label = str(hazard.label).strip().lower()
            if label == "pothole":
                if not self._is_forward_pothole(hazard):
                    continue
                warnings.append(Warning(3, "road_hazard", "ROAD HAZARD: pothole"))
            elif label in {"crack", "manhole"}:
                warnings.append(Warning(4, "road_hazard", f"ROAD HAZARD: {label}"))

        # Python's sort is stable, so equal-priority detections retain detector
        # order while the policy remains High > Medium > Pothole > Crack/Manhole.
        return sorted(warnings, key=lambda warning: warning.priority)

    @classmethod
    def _is_forward_pothole(cls, hazard: HazardDetection) -> bool:
        """Return whether a pothole centre lies inside the forward trapezoid."""
        centre = hazard.centre_norm
        if centre is None:
            return False

        centre_x, centre_y = centre
        (top_left_x, top_y), (top_right_x, _), (bottom_right_x, bottom_y), _ = (
            cls.FORWARD_REGION
        )
        if not top_y <= centre_y <= bottom_y:
            return False

        progress = (centre_y - top_y) / (bottom_y - top_y)
        left_x = top_left_x + progress * (1.0 - bottom_right_x - top_left_x)
        right_x = top_right_x + progress * (bottom_right_x - top_right_x)
        return left_x <= centre_x <= right_x
