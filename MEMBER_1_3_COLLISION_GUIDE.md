# Member 1 + Member 3: collision-module guide

## Goal

Build a defensible forward-collision module that detects traffic objects and returns `low`, `medium`, or `high` risk. Do not claim exact distance, time-to-collision, or collision probability unless those values are genuinely calculated and validated.

## Member 1: BDD100K dataset preparation

1. Work from `BDD100K_Collision_Dataset/bdd100k` and `bdd100k_labels_release` in the shared Drive package.
2. Use the BDD100K train/validation detection JSON files. Do not use `bdd100k_seg` unless the team later decides to build a segmentation-based driving region.
3. Select the relevant object categories:
   - `car`, `truck`, `bus`, `train`, `motor`, `bike` → `vehicle`
   - `person`, `rider` → `person`
4. Convert selected BDD100K bounding-box annotations into YOLO label files: one `.txt` file per image, with normalized `class x_center y_center width height` values.
5. Organize the converted subset with this layout:

   ```text
   collision_dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   ├── labels/
   │   ├── train/
   │   └── val/
   └── data.yaml
   ```

6. Use this class mapping in `data.yaml`:

   ```yaml
   names:
     0: vehicle
     1: person
   ```

7. Validate the conversion by drawing labels on at least 20 random images. Check that every image has its matching label file and normalized values remain between 0 and 1.
8. Give Member 3: the processed dataset path/Drive link, `data.yaml`, class mapping, sample visualizations, conversion script, and any known limitations.

## Member 3: detector training and risk logic

1. Clone/pull the SafeDrive GitHub repository and install dependencies.
2. Download/use Member 1's prepared `collision_dataset` and confirm the two classes are `vehicle` and `person`.
3. Train a YOLO baseline (for example YOLO11n) and save the exact command, model choice, epochs, image size, and confidence threshold.
4. Evaluate the detector with precision, recall, mAP50, mAP50-95, confusion matrix, and inference speed.
5. Define the forward driving region. Start with a simple centre-lower rectangle or trapezoid and document the coordinates as image-relative proportions.
6. For objects inside that region, track the same object across nearby video frames where possible. Use transparent rules such as:
   - `low`: object is outside the forward region or small/stable
   - `medium`: object is in the forward region and becoming noticeably larger
   - `high`: object is centred in the forward region and rapidly becoming larger across consecutive frames
7. Tune the thresholds with short test videos. Record both correct alerts and false-warning cases.
8. Implement only `src/collision/detector.py`. Preserve this contract:

   ```python
   CollisionDetector.detect(frame) -> list[CollisionRisk]
   ```

9. Return to Member 1: updated detector code, model-weight link, test command, metrics, forward-region diagram/rules, and a short list of limitations.

## Integration test: Member 1 + Member 3

1. Place the final collision checkpoint locally at `SafeDrive/models/collision/best.pt`.
2. Update the collision module's configuration only if needed; do not change the return contract.
3. Run `python -m src.main <test-video.mp4>`.
4. Confirm collision results become `CollisionRisk` values and that `WarningManager` gives high/medium collision warnings priority above road hazards.
5. Save a short demo clip and latency/FPS result in `Results_and_Metrics`.

## Definition of done

- Member 1: valid, documented two-class YOLO dataset.
- Member 3: tested detector plus reproducible, explainable LOW/MEDIUM/HIGH logic.
- Both: integrated video demo, metrics, and known failure cases recorded.
