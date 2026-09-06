"""CLI entry point for SafeDrive image and video inference."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import wave

import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from configs.config import (  # noqa: E402
    COLLISION_CONFIDENCE,
    COLLISION_WEIGHTS,
    OUTPUTS_DIR,
    ROAD_HAZARD_DASHBOARD_EXCLUSION_Y,
    ROAD_HAZARD_CONFIDENCE,
    ROAD_HAZARD_MIN_AREA_GROWTH,
    ROAD_HAZARD_MIN_CENTRE_Y_CHANGE,
    ROAD_HAZARD_VIDEO_CONFIRMATION_FRAMES,
    ROAD_HAZARD_WEIGHTS,
)
from collision.detector import CollisionDetector, CollisionRisk, FORWARD_REGION  # noqa: E402
from road_hazard.detector import HazardDetection, RoadHazardDetector  # noqa: E402
from warning.warning_manager import WarningManager  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _write_warning_audio(
    destination: Path, duration_seconds: float, alerts: list[tuple[float, str]]
) -> None:
    """Create an alert-only audio track: slow yellow beeps, fast red beeps."""
    sample_rate = 44_100
    samples = np.zeros(max(1, int(np.ceil(duration_seconds * sample_rate))), dtype=np.int16)
    for alert_time, severity in alerts:
        # HIGH forward-collision warnings are red and fast. MEDIUM collision
        # and road-hazard warnings are yellow and deliberately slower.
        frequency, offsets = (880, (0.0, 0.16, 0.32)) if severity == "red" else (660, (0.0, 0.35, 0.70))
        tone_length = int(0.11 * sample_rate)
        tone = (np.sin(2 * np.pi * frequency * np.arange(tone_length) / sample_rate) * 9_000).astype(np.int16)
        for offset in offsets:
            start = int((alert_time + offset) * sample_rate)
            end = min(start + tone_length, len(samples))
            if start < len(samples):
                samples[start:end] = np.maximum(samples[start:end], tone[: end - start])
    with wave.open(str(destination), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def _make_browser_playable_video(
    intermediate: Path, destination: Path, duration_seconds: float, alerts: list[tuple[float, str]]
) -> None:
    """Encode OpenCV's temporary MP4 as H.264 for Chrome/Gradio playback.

    OpenCV commonly writes ``mp4v`` on Windows. It is readable by OpenCV but is
    not reliably supported by browsers, so the web demo would show
    "Video not playable". The bundled FFmpeg supplied by imageio-ffmpeg writes
    a standards-compatible H.264/yuv420p MP4. Original audio is deliberately
    omitted; the output contains only generated beep alerts for confirmed warnings.
    """
    warning_audio = intermediate.with_suffix(".warning.wav")
    _write_warning_audio(warning_audio, duration_seconds, alerts)
    command = [
        get_ffmpeg_exe(), "-y", "-i", str(intermediate), "-i", str(warning_audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        details = completed.stderr.strip().splitlines()[-1] if completed.stderr else "unknown FFmpeg error"
        raise RuntimeError(f"Could not create browser-playable MP4: {details}")
    warning_audio.unlink(missing_ok=True)


@dataclass
class _HazardTrack:
    """Small video-only track used to validate a hazard before displaying it."""

    label: str
    first_detection: HazardDetection
    last_detection: HazardDetection
    observations: int = 1
    last_seen_frame: int = 0
    confirmed: bool = False


class VideoHazardFilter:
    """Suppress transient or stationary reflection-like hazard detections.

    This is a conservative presentation safety layer, not a replacement for
    retraining. A matching detection must be seen repeatedly and exhibit the
    expected perspective change of a fixed object approached by the vehicle.
    """

    def __init__(
        self,
        confirmation_frames: int = ROAD_HAZARD_VIDEO_CONFIRMATION_FRAMES,
        min_centre_y_change: float = ROAD_HAZARD_MIN_CENTRE_Y_CHANGE,
        min_area_growth: float = ROAD_HAZARD_MIN_AREA_GROWTH,
        dashboard_exclusion_y: float = ROAD_HAZARD_DASHBOARD_EXCLUSION_Y,
        sensitive_pothole_mode: bool = False,
    ) -> None:
        self.confirmation_frames = confirmation_frames
        self.min_centre_y_change = min_centre_y_change
        self.min_area_growth = min_area_growth
        self.dashboard_exclusion_y = dashboard_exclusion_y
        self.sensitive_pothole_mode = sensitive_pothole_mode
        self.frame_number = 0
        self.tracks: list[_HazardTrack] = []

    @staticmethod
    def _iou(first: HazardDetection, second: HazardDetection) -> float:
        """Calculate box overlap without making the detector depend on OpenCV."""
        ax1, ay1, ax2, ay2 = first.bbox
        bx1, by1, bx2, by2 = second.bbox
        overlap_width = max(0, min(ax2, bx2) - max(ax1, bx1))
        overlap_height = max(0, min(ay2, by2) - max(ay1, by1))
        overlap = overlap_width * overlap_height
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - overlap
        return overlap / union if union else 0.0

    @staticmethod
    def _area(detection: HazardDetection) -> float:
        x1, y1, x2, y2 = detection.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _has_forward_motion(self, track: _HazardTrack) -> bool:
        """Check the perspective change expected from an approaching road object."""
        first_centre = track.first_detection.centre_norm
        current_centre = track.last_detection.centre_norm
        if first_centre is None or current_centre is None:
            return False
        centre_moved_down = current_centre[1] - first_centre[1] >= self.min_centre_y_change
        initial_area = self._area(track.first_detection)
        area_grew = initial_area > 0 and self._area(track.last_detection) / initial_area >= self.min_area_growth
        return centre_moved_down or area_grew

    def filter(self, detections: list[HazardDetection]) -> list[HazardDetection]:
        """Return only confirmed, non-dashboard detections for one video frame."""
        self.frame_number += 1
        self.tracks = [track for track in self.tracks if self.frame_number - track.last_seen_frame <= 2]
        confirmed: list[HazardDetection] = []

        for detection in detections:
            # The lower edge of a dashcam frame commonly contains dashboard
            # reflections rather than usable road surface.
            if detection.bbox_norm is not None and detection.bbox_norm[3] >= self.dashboard_exclusion_y:
                continue

            # Optional presentation/testing mode: a low-confidence pothole near
            # the ego-lane centre may be visible for only a few frames. Keep the
            # strict temporal filter as the default to avoid reflection alerts.
            if self.sensitive_pothole_mode and detection.label == "pothole" and detection.bbox_norm is not None:
                centre_x, centre_y = detection.centre_norm
                if 0.35 <= centre_x <= 0.65 and 0.55 <= centre_y <= 0.90:
                    confirmed.append(detection)
                    continue

            candidates = [
                track for track in self.tracks
                if track.label == detection.label and self._iou(track.last_detection, detection) >= 0.30
            ]
            if candidates:
                track = max(candidates, key=lambda item: self._iou(item.last_detection, detection))
                track.last_detection = detection
                track.last_seen_frame = self.frame_number
                track.observations += 1
            else:
                track = _HazardTrack(detection.label, detection, detection, last_seen_frame=self.frame_number)
                self.tracks.append(track)

            if (
                not track.confirmed
                and track.observations >= self.confirmation_frames
                and self._has_forward_motion(track)
            ):
                track.confirmed = True
            if track.confirmed:
                confirmed.append(detection)
        return confirmed


def _draw_forward_region(frame, forward_region=FORWARD_REGION) -> None:
    """Draw the image-space corridor used by the collision-risk heuristic."""
    height, width = frame.shape[:2]
    points = [
        (round(x * width), round(y * height))
        for x, y in forward_region
    ]
    cv2.polylines(frame, [np.array(points, dtype=np.int32)], True, (255, 255, 0), 2)
    cv2.putText(
        frame, "FORWARD RISK ZONE", points[0], cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 0), 2,
    )


def _draw_collision_risks(frame, collision_risks: list[CollisionRisk]) -> None:
    """Show detections even at LOW risk; only MEDIUM/HIGH create warnings."""
    colors = {"low": (0, 200, 0), "medium": (0, 165, 255), "high": (0, 0, 255)}
    for index, risk in enumerate(collision_risks):
        if risk.bbox is None:
            continue
        x1, y1, x2, y2 = risk.bbox
        level = risk.risk_level.lower()
        color = colors.get(level, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{risk.object_type} {risk.confidence:.2f} | {level.upper()}"
        label_y = max(y1 - 8 - (index % 3) * 18, 22)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, label_y - text_height - baseline - 3), (x1 + text_width + 4, label_y + 3), color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)


def annotate_frame(frame, hazards, collision_risks, warning, show_forward_zone: bool = False, forward_region=FORWARD_REGION):
    """Draw module outputs without coupling detector internals to OpenCV."""
    if show_forward_zone:
        _draw_forward_region(frame, forward_region)
    _draw_collision_risks(frame, collision_risks)
    for hazard in hazards:
        x1, y1, x2, y2 = hazard.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            frame, f"{hazard.label} {hazard.confidence:.2f}", (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2,
        )
    if warning is not None:
        color = (0, 0, 255) if warning.priority == 1 else (0, 165, 255)
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


def process_frame_with_warning(
    frame,
    road_hazard_detector,
    collision_detector,
    warning_manager,
    hazard_filter=None,
    show_forward_zone: bool = False,
):
    """Run all modules for one frame and retain the selected warning for callers."""
    hazards = road_hazard_detector.detect(frame)
    if hazard_filter is not None:
        hazards = hazard_filter.filter(hazards)
    collision_risks = collision_detector.detect(frame)
    warning = warning_manager.evaluate(hazards, collision_risks)
    return annotate_frame(frame.copy(), hazards, collision_risks, warning, show_forward_zone), warning


def process_frame(
    frame,
    road_hazard_detector,
    collision_detector,
    warning_manager,
    hazard_filter=None,
    show_forward_zone: bool = False,
):
    """Run all modules for one frame and return its annotated image."""
    annotated, _ = process_frame_with_warning(
        frame,
        road_hazard_detector,
        collision_detector,
        warning_manager,
        hazard_filter=hazard_filter,
        show_forward_zone=show_forward_zone,
    )
    return annotated


def process_image(source: Path, destination: Path, pipeline, show_forward_zone: bool = False) -> str:
    frame = cv2.imread(str(source))
    if frame is None:
        raise ValueError(f"Could not read image: {source}")
    # Browser requests reuse one pipeline, so restore the documented default
    # after a manually calibrated video has been analysed.
    if hasattr(pipeline[1], "set_forward_region"):
        pipeline[1].set_forward_region(FORWARD_REGION)
    output, warning = process_frame_with_warning(frame, *pipeline, show_forward_zone=show_forward_zone)
    if not cv2.imwrite(str(destination), output):
        raise RuntimeError(f"Could not write output image: {destination}")
    return warning.message if warning is not None else ""


def process_video(
    source: Path,
    destination: Path,
    pipeline,
    frame_stride: int = 1,
    show_forward_zone: bool = False,
    forward_region=None,
    sensitive_pothole_mode: bool = False,
) -> str:
    """Process a video, optionally reusing analysis between frames for a fast demo.

    ``frame_stride=1`` is the accurate default: both detectors run on every frame.
    Higher values are intended only for an interactive presentation. The video
    remains visually smooth because original skipped frames are retained, but
    their boxes/warning state are reused until the next analysis frame.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    intermediate = destination.with_name(f"{destination.stem}_intermediate.mp4")
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video writer: {intermediate}")
    # A video is a fresh scene; do not carry collision tracks from a prior image
    # or an earlier browser upload into its temporal risk assessment.
    road_hazard_detector, collision_detector, warning_manager = pipeline
    active_forward_region = forward_region or FORWARD_REGION
    if hasattr(collision_detector, "set_forward_region"):
        collision_detector.set_forward_region(active_forward_region)
    if hasattr(collision_detector, "reset"):
        collision_detector.reset()
    hazard_filter = VideoHazardFilter(sensitive_pothole_mode=sensitive_pothole_mode)
    frame_number = 0
    latest_hazards = []
    latest_collision_risks = []
    latest_warning = None
    highest_warning = None
    alerts: list[tuple[float, str]] = []
    last_alert_frame = -1_000_000
    last_alert_message = ""
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % frame_stride == 0:
                hazards = road_hazard_detector.detect(frame, confidence=0.25 if sensitive_pothole_mode else None)
                hazards = hazard_filter.filter(hazards)
                collision_risks = collision_detector.detect(frame)
                warning = warning_manager.evaluate(hazards, collision_risks)
                latest_hazards = hazards
                latest_collision_risks = collision_risks
                latest_warning = warning
                if warning is not None and (highest_warning is None or warning.priority < highest_warning.priority):
                    highest_warning = warning
                # Keep beeps meaningful: a red collision risk may repeat after
                # two seconds, but a yellow warning beeps only when it begins or
                # changes category (for example, from crack to pothole).
                if warning is not None and (
                    warning.message != last_alert_message
                    or (warning.priority == 1 and frame_number - last_alert_frame >= int(2 * fps))
                ):
                    severity = "red" if warning.priority == 1 else "yellow"
                    alerts.append((frame_number / fps, severity))
                    last_alert_frame = frame_number
                    last_alert_message = warning.message
            # Preserve every original video frame. In fast mode, the overlay is
            # simply held between detector updates instead of freezing the scene.
            writer.write(
                annotate_frame(
                    frame.copy(), latest_hazards, latest_collision_risks,
                    latest_warning, show_forward_zone, active_forward_region,
                )
            )
            frame_number += 1
    finally:
        capture.release()
        writer.release()
    _make_browser_playable_video(intermediate, destination, frame_number / fps, alerts)
    intermediate.unlink(missing_ok=True)
    return highest_warning.message if highest_warning is not None else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SafeDrive on one image or video.")
    parser.add_argument("source", type=Path, help="Path to an input image or video")
    parser.add_argument("--output", type=Path, help="Optional output file path")
    parser.add_argument("--show-forward-zone", action="store_true", help="Draw the collision-risk corridor for explanation/demo")
    args = parser.parse_args()

    if not args.source.is_file():
        parser.error(f"Input file does not exist: {args.source}")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    is_image = args.source.suffix.lower() in IMAGE_SUFFIXES
    destination = args.output or OUTPUTS_DIR / f"{args.source.stem}_safedrive{args.source.suffix if is_image else '.mp4'}"
    destination.parent.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline()
    if is_image:
        process_image(args.source, destination, pipeline, args.show_forward_zone)
    else:
        process_video(args.source, destination, pipeline, show_forward_zone=args.show_forward_zone)
    print(f"Saved SafeDrive output to: {destination}")


if __name__ == "__main__":
    main()
