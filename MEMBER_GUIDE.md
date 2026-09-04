# SafeDrive step-by-step teammate guide

## Everyone: first-time setup

1. Receive or clone the `SafeDrive` folder only. Do not copy datasets, weights, `.venv`, or output videos into it.
2. Open a terminal inside `SafeDrive`.
3. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

4. Read `TEAM_SETUP.md` before changing code.
5. Change only your assigned module unless you first agree on an interface change with Member 1.
6. Before handing work back, test your module and provide: changed files, a short run command, model metrics (if applicable), and a link/path to weights. Do not add large datasets or checkpoints to the repository.

## Member 1: dataset, framework, and final integration

1. Keep the road-hazard dataset and BDD100K outside `SafeDrive`.
2. Maintain `configs/config.py`, `src/main.py`, `requirements.txt`, and this guide.
3. Give Member 2 the road-hazard training dataset and tell them the model classes must be `pothole`, `crack`, and `manhole`.
4. Give Member 3 the BDD100K path/link. Explain it is for traffic objects such as cars and people, not road hazards.
5. Receive each teammate's changed module and test their contracts together using `python -m src.main <image-or-video>`.
6. Place received local checkpoints at `models/road_hazard/best.pt` and `models/collision/best.pt` for integration testing only.
7. Test one image and one video; keep generated results under `outputs/`.
8. Record final demo command, module metrics, and limitations for the report/presentation.

## Member 2: road-hazard detection

1. Inspect the supplied YOLO road-hazard dataset, `data.yaml`, labels, and class mapping.
2. Train a YOLO11n baseline for pothole, crack, and manhole.
3. Evaluate precision, recall, mAP50, mAP50-95, per-class AP, confusion matrix, and inference speed.
4. Run the agreed comparison: training without normal/background images versus training with the 446 normal/background images. Report whether false detections changed.
5. Tune only after saving the baseline results (for example epochs, image size, augmentation, or learning rate).
6. Select the best checkpoint and name it `best.pt`.
7. Test it on a few unseen road images/video frames.
8. Update only `src/road_hazard/detector.py` if needed, preserving:

   ```python
   RoadHazardDetector.detect(frame) -> list[HazardDetection]
   ```

9. Hand Member 1: edited detector file, metrics, training command/settings, label mapping, and a link to `best.pt`.

## Member 3: forward-collision detection and risk

1. Use an appropriate BDD100K traffic-object subset for vehicle/person detection. Do not use it as a road-hazard dataset.
2. Convert annotations to YOLO format if your selected model needs YOLO labels.
3. Train or fine-tune a traffic-object detector for at least `vehicle` and `person`-type targets.
4. Define the forward driving region explicitly (for example, a centre-lower trapezoid/rectangle) and document it.
5. For objects in that region, use explainable evidence such as bounding-box size and change across frames to assign `low`, `medium`, or `high` risk.
6. Avoid claiming distance, time-to-collision, or collision probability unless you calculate and validate it.
7. Test on short videos: verify irrelevant side objects do not create unnecessary high-risk alerts.
8. Implement only `src/collision/detector.py`, preserving:

   ```python
   CollisionDetector.detect(frame) -> list[CollisionRisk]
   ```

9. Hand Member 1: edited detector file, forward-region/risk rules, metrics or qualitative test results, tested video examples, and a link to weights.

## Member 4: warning system and experimental analysis

1. Review result formats from Members 2 and 3 before editing warning code.
2. Implement the priority policy in `src/warning/warning_manager.py`:
   1. High forward collision risk
   2. Medium forward collision risk
   3. Pothole in a relevant driving region
   4. Crack or manhole
   5. No warning
3. Show one highest-priority warning at a time; suppress lower-priority messages when a collision warning is active.
4. Preserve this integration method:

   ```python
   WarningManager.evaluate(road_results, collision_results) -> Warning | None
   ```

5. Test combinations of mocked outputs: no detections, one hazard, high collision plus hazard, and medium collision plus hazard.
6. Collect integration-analysis evidence: latency/FPS, false-warning cases, and a comparison or ablation where possible.
7. Hand Member 1: edited warning file, priority rules, test cases/results, and analysis tables/figures or their underlying numbers.

## Final handoff checklist

- No changed interface without approval from Member 1.
- No dataset or model files committed or sent inside the project folder.
- Clear instructions for reproducing the member's test.
- Model owners provide metrics and a weight-file link.
- Member 1 performs the final image/video run after merging all modules.
