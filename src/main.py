"""CLI entry point for SafeDrive image and video inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from configs.config import (  # noqa: E402
    COLLISION_CONFIDENCE,
    COLLISION_WEIGHTS,
    OUTPUTS_DIR,
    ROAD_HAZARD_CONFIDENCE,
    ROAD_HAZARD_WEIGHTS,
)
from collision.detector import CollisionDetector  # noqa: E402
from road_hazard.detector import RoadHazardDetector  # noqa: E402
from warning.warning_manager import WarningManager  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def annotate_frame(frame, hazards, warning):
    """Draw module outputs without coupling detector internals to OpenCV."""
    for hazard in hazards:
        x1, y1, x2, y2 = hazard.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            frame, f"{hazard.label} {hazard.confidence:.2f}", (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2,
        )
    if warning is not None:
        color = (0, 0, 255) if warning.category == "collision" else (0, 165, 255)
        cv2.putText(
            frame, warning.message, (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )
    return frame


def build_pipeline():
    """Construct independent modules in one place for easy teammate replacement."""
    return (
        RoadHazardDetector(ROAD_HAZARD_WEIGHTS, ROAD_HAZARD_CONFIDENCE),
        CollisionDetector(str(COLLISION_WEIGHTS), COLLISION_CONFIDENCE),
        WarningManager(),
    )


def process_frame(frame, road_hazard_detector, collision_detector, warning_manager):
    hazards = road_hazard_detector.detect(frame)
    collision_risks = collision_detector.detect(frame)
    warning = warning_manager.evaluate(hazards, collision_risks)
    return annotate_frame(frame.copy(), hazards, warning)


def process_image(source: Path, destination: Path, pipeline) -> None:
    frame = cv2.imread(str(source))
    if frame is None:
        raise ValueError(f"Could not read image: {source}")
    output = process_frame(frame, *pipeline)
    if not cv2.imwrite(str(destination), output):
        raise RuntimeError(f"Could not write output image: {destination}")


def process_video(source: Path, destination: Path, pipeline) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(process_frame(frame, *pipeline))
    finally:
        capture.release()
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SafeDrive on one image or video.")
    parser.add_argument("source", type=Path, help="Path to an input image or video")
    parser.add_argument("--output", type=Path, help="Optional output file path")
    args = parser.parse_args()

    if not args.source.is_file():
        parser.error(f"Input file does not exist: {args.source}")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    is_image = args.source.suffix.lower() in IMAGE_SUFFIXES
    destination = args.output or OUTPUTS_DIR / f"{args.source.stem}_safedrive{args.source.suffix if is_image else '.mp4'}"
    destination.parent.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline()
    if is_image:
        process_image(args.source, destination, pipeline)
    else:
        process_video(args.source, destination, pipeline)
    print(f"Saved SafeDrive output to: {destination}")


if __name__ == "__main__":
    main()
