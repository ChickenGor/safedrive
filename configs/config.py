"""Central configuration for SafeDrive.

Keep model paths and confidence thresholds here so each feature module can be
replaced without changing the application entry point.
"""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# TODO(road-hazard-team): replace this with the exported best.pt checkpoint.
ROAD_HAZARD_WEIGHTS = MODELS_DIR / "road_hazard" / "best.pt"
ROAD_HAZARD_CONFIDENCE = 0.35

# Reserved for a future vehicle/person collision-risk model.
# TODO(collision-team): set this when the collision model is available.
COLLISION_WEIGHTS = MODELS_DIR / "collision" / "best.pt"
COLLISION_CONFIDENCE = 0.40

# Video-only road-hazard validation. A real hazard on the road should move
# downwards and/or grow as the vehicle approaches it. This avoids presenting a
# single, stationary windshield/dashboard reflection as a driver warning.
# These safeguards deliberately do not alter still-image inference, where a
# temporal check is impossible.
ROAD_HAZARD_VIDEO_CONFIRMATION_FRAMES = 4
ROAD_HAZARD_MIN_CENTRE_Y_CHANGE = 0.02
ROAD_HAZARD_MIN_AREA_GROWTH = 1.10
ROAD_HAZARD_DASHBOARD_EXCLUSION_Y = 0.92
