# SafeDrive teammate handoff

This repository is the shared application foundation. Do not copy raw datasets, model checkpoints, virtual environments, or generated output videos into it.

## Module contracts

| Owner | File | Required contract |
| --- | --- | --- |
| Member 2 | `src/road_hazard/detector.py` | `detect(frame) -> list[HazardDetection]` with `pothole`, `crack`, and `manhole` labels |
| Member 3 | `src/collision/detector.py` | `detect(frame) -> list[CollisionRisk]` with `vehicle`/`person` and `low`/`medium`/`high` risk levels |
| Member 4 | `src/warning/warning_manager.py` | `evaluate(road_results, collision_results) -> Warning | None` |
| Member 1 | `src/main.py` | Calls all three modules and handles image/video input and output |

## Required handoff from each teammate

1. Their edited module files.
2. Any changed dependency listed in `requirements.txt`.
3. A short note describing model classes, input assumptions, and tested command.
4. For model owners: a link to weights and their metrics. Do not upload weights to this repository.

## Local model locations

- Road hazard: `models/road_hazard/best.pt`
- Collision: `models/collision/best.pt`

Run the shared app from the repository root:

```powershell
python -m src.main path\to\input.mp4
```

Member 3 should use BDD100K only as an external training resource. It has traffic-object labels, not pothole/crack/manhole labels. Member 4 should avoid reporting unvalidated distances, time-to-collision, or collision probabilities.
