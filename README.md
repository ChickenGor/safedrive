# SafeDrive

SafeDrive is an application-based deep learning project for dashcam road safety.
It combines two YOLO-based perception modules with a warning policy that always
prioritizes forward collision risk above road hazards.

## Project scope

- Road-hazard detection: **pothole**, **crack**, and **manhole**.
- Collision-risk detection: **vehicle** and **person** detections with qualitative
  `low`, `medium`, or `high` forward-risk levels.
- Warning policy: `high collision` > `medium collision` > `forward pothole` >
  `crack/manhole`.
- Image and video inference with annotated output.

SafeDrive is a warning-support prototype. Its collision-risk levels are qualitative;
they do not estimate physical distance, time-to-collision, or guarantee avoidance.

## System design

```text
Dashcam image/video
        |
        +--> RoadHazardDetector (YOLO: pothole / crack / manhole)
        |
        +--> CollisionDetector (YOLO: vehicle / person + temporal risk heuristic)
                         |
                         v
                 WarningManager
                         |
                         v
                 Highest-priority warning + annotated output
```

The modules communicate through small dataclasses rather than importing each
other's model internals. A teammate can replace one detector without changing the
warning policy or CLI entry point.

## Repository layout

```text
configs/                 Confidence thresholds and local model paths
src/road_hazard/         Road-hazard YOLO detector
src/collision/           Vehicle/person collision-risk detector
src/warning/             Warning-priority policy
src/main.py              Image/video command-line entry point
tools/                   Reproducible dataset-preparation and training utilities
models/                  Local checkpoint locations (not committed)
outputs/                 Generated annotations (not committed)
```

## Setup

```powershell
git clone https://github.com/ChickenGor/safedrive.git
cd safedrive
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the downloaded checkpoints at:

```text
models/road_hazard/best.pt
models/collision/best.pt
```

The road-hazard checkpoint must expose `Pothole`, `Crack`, and `Manhole`
(case-insensitive). The collision checkpoint must expose `vehicle` and/or `person`.

## Run

```powershell
# One image
python -m src.main "path\to\road.jpg" --output "outputs\annotated.jpg"

# One video
python -m src.main "path\to\dashcam.mp4" --output "outputs\annotated.mp4"
```

For video only, SafeDrive applies a conservative road-hazard presentation filter:
it ignores the lower dashboard strip and requires a matched hazard to persist and
show forward perspective motion before displaying it. This reduces stationary
windshield/dashboard-reflection false positives; it is not a replacement for more
diverse training data.

## Shared project assets

Datasets, checkpoints, training results, and full demo videos are intentionally
excluded from Git because of size limits. They are shared in the team's Google
Drive folder: [SafeDrive Shared Drive](https://drive.google.com/drive/folders/1PmmMRCAnpH9NFXq4-nIflNSoZDC2Y6fF).

Use the following folders in that Drive:

- `Road_Hazard_Dataset` - baseline road-hazard data and documentation.
- `BDD100K_Collision_Dataset` - selected vehicle/person detection subset.
- `Model_Weights` - road-hazard and collision `best.pt` checkpoints.
- `Results_and_Metrics` - training curves, test metrics, and confusion matrices.
- `Test_Images_Videos` - input examples and final demo video.

Before final submission, confirm that the Drive sharing permission is set so the
lecturer can open the link.

## Reproducibility and experiments

The retained road-hazard baseline used YOLO11n with COCO pretraining, 60 epochs,
and 960 px training/inference. Its held-out test split contained 369 images.

| Road-hazard baseline test metric | Result |
| --- | ---: |
| Precision | 0.570 |
| Recall | 0.494 |
| mAP@50 | 0.516 |
| mAP@50-95 | 0.218 |

`tools/prepare_archive5_augmentation.py` reproduces the Archive (5) data
augmentation experiment. It remaps pothole to `Pothole` and all three crack
subtypes to `Crack`, preserving original validation/test splits and manhole data.
The quick 10-epoch, 640 px candidate reached test mAP@50 `0.377`, below the
baseline; therefore the baseline checkpoint was retained. This exploratory result
should be reported honestly rather than presented as an improvement.

## Team module contracts

| Module | Contract |
| --- | --- |
| Road hazards | `RoadHazardDetector.detect(frame) -> list[HazardDetection]` |
| Collision | `CollisionDetector.detect(frame) -> list[CollisionRisk]` |
| Warnings | `WarningManager.evaluate(road_results, collision_results) -> Warning | None` |
| Application | `main.py` constructs the modules and processes images/videos |

## Submission checklist

- Do not include raw datasets, `.pt` checkpoints, `.venv`, `runs/`, or output
  videos in the assignment ZIP.
- Include source code, this README, the final report PDF, and the required links.
- Cite all datasets, pretrained models, and third-party code in the report.
- Describe known limitations, including glare/reflection false positives.
- Include an accurate Generative AI use declaration in the final report.

## Generative AI use note

During development, Codex/ChatGPT was used as a coding aid for project scaffolding,
integration support, and documentation refinement. Each team must revise this note
to accurately reflect its own actual use and include the required AI-use declaration
and relevant prompt/transcript links in the final report.
