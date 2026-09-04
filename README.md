# SafeDrive

SafeDrive is a modular PyTorch/Ultralytics YOLO application for detecting road hazards and presenting driving warnings. The current collision package is an integration-ready framework only; it intentionally does not perform collision-risk inference yet.

## Layout

- `src/road_hazard/`: teammate-owned YOLO inference for `pothole`, `crack`, and `manhole`.
- `src/collision/`: stable future interface for vehicle/person collision risks.
- `src/warning/`: combines results and displays only the highest-priority active warning: high collision, medium collision, pothole, then other hazards.
- `configs/config.py`: model locations and thresholds.
- `models/`: local weight files (not committed or submitted when too large).
- `outputs/`: annotated images and videos.

## Setup

```powershell
cd SafeDrive
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put the road-hazard team's Ultralytics checkpoint at `models/road_hazard/best.pt`. Its class names must be exactly `pothole`, `crack`, and/or `manhole` (case-insensitive).

## Run

```powershell
python -m src.main path\to\road.jpg
python -m src.main path\to\dashcam.mp4
python -m src.main path\to\road.jpg --output outputs\annotated.jpg
```

Outputs are saved under `outputs/` unless `--output` is specified. The application accepts common image formats; all other input files are opened as videos.

## Team integration contracts

The road-hazard module must preserve `RoadHazardDetector.detect(frame) -> list[HazardDetection]`. The collision team should implement `CollisionDetector.detect(frame) -> list[CollisionRisk]`. The warning team should preserve `WarningManager.evaluate(road_results, collision_results) -> Warning | None`.

Search for `TODO(` to find the intended insertion points. No advanced collision decision logic has been added.

## Assignment note

If this is used for UCCD3094, declare any AI-assisted code generation in the team report as required by the assignment brief, and document your actual model training, data preparation, and evaluation work separately.
