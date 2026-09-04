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
        warnings: list[Warning] = []
        for risk in collision_risks:
            level = risk.risk_level.lower()
            if level in {"high", "medium"}:
                warnings.append(
                    Warning(
                        1 if level == "high" else 2,
                        "collision",
                        f"{level.upper()} FORWARD COLLISION RISK: {risk.object_type}",
                    )
                )

        # Road hazards remain visible, but collision warnings always sort first.
        for hazard in hazards:
            priority = 3 if hazard.label == "pothole" else 4
            warnings.append(Warning(priority, "road_hazard", f"ROAD HAZARD: {hazard.label}"))
        return sorted(warnings, key=lambda warning: warning.priority)
