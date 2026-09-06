"""Local browser interface for demonstrating the SafeDrive ADAS prototype.

The app intentionally keeps all inference on the laptop. A phone on the same
Wi-Fi network can open the local URL, upload a dashcam image/video, and receive
an annotated result without exposing model weights or a public service.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from uuid import uuid4

import cv2
import gradio as gr
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import build_pipeline, process_image, process_video  # noqa: E402


WEB_OUTPUTS = PROJECT_ROOT / "outputs" / "web_app"
_pipeline = None
def _get_pipeline():
    """Create model modules once, so browser requests reuse loaded checkpoints."""
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


def _require_upload(upload: str | None, kind: str) -> Path:
    if not upload:
        raise gr.Error(f"Upload a {kind} first.")
    source = Path(upload)
    if not source.is_file():
        raise gr.Error("The uploaded file is no longer available. Please upload it again.")
    return source


def _manual_forward_region(
    mode: str,
    top_y: float,
    top_half_width: float,
    bottom_half_width: float,
    top_left_x: float,
    top_left_y: float,
    top_right_x: float,
    top_right_y: float,
    bottom_right_x: float,
    bottom_right_y: float,
    bottom_left_x: float,
    bottom_left_y: float,
) -> tuple[tuple[float, float], ...]:
    """Build either a stable symmetric or a fully manual four-corner trapezoid."""
    if mode == "Advanced four-corner":
        return (
            (top_left_x, top_left_y),
            (top_right_x, top_right_y),
            (bottom_right_x, bottom_right_y),
            (bottom_left_x, bottom_left_y),
        )
    return (
        (0.5 - top_half_width, top_y),
        (0.5 + top_half_width, top_y),
        (0.5 + bottom_half_width, 0.98),
        (0.5 - bottom_half_width, 0.98),
    )


def analyse_image(upload: str | None, show_forward_zone: bool) -> tuple[str, str, str]:
    """Run the existing image pipeline and return a browser-displayable result."""
    source = _require_upload(upload, "road image")
    WEB_OUTPUTS.mkdir(parents=True, exist_ok=True)
    destination = WEB_OUTPUTS / f"image_{uuid4().hex}.jpg"
    warning = process_image(source, destination, _get_pipeline(), show_forward_zone)
    status = "Analysis complete. Review the warning shown in the image."
    return str(destination), status, warning


def analyse_video(
    upload: str | None,
    frame_stride: int,
    show_forward_zone: bool,
    zone_mode: str,
    top_y: float,
    top_half_width: float,
    bottom_half_width: float,
    top_left_x: float,
    top_left_y: float,
    top_right_x: float,
    top_right_y: float,
    bottom_right_x: float,
    bottom_right_y: float,
    bottom_left_x: float,
    bottom_left_y: float,
) -> tuple[str, str, str]:
    """Run the existing sequential video pipeline and return its annotated video."""
    source = _require_upload(upload, "dashcam video")
    WEB_OUTPUTS.mkdir(parents=True, exist_ok=True)
    destination = WEB_OUTPUTS / f"video_{uuid4().hex}.mp4"
    warning = process_video(
        source,
        destination,
        _get_pipeline(),
        frame_stride=int(frame_stride),
        show_forward_zone=show_forward_zone,
        forward_region=_manual_forward_region(
            zone_mode, top_y, top_half_width, bottom_half_width,
            top_left_x, top_left_y, top_right_x, top_right_y,
            bottom_right_x, bottom_right_y, bottom_left_x, bottom_left_y,
        ),
    )
    mode = "standard" if int(frame_stride) == 1 else f"fast demo (every {int(frame_stride)}th frame analysed)"
    return str(destination), f"Analysis complete using {mode} mode.", warning


def preview_forward_region(
    upload: str | None,
    zone_mode: str,
    top_y: float,
    top_half_width: float,
    bottom_half_width: float,
    top_left_x: float,
    top_left_y: float,
    top_right_x: float,
    top_right_y: float,
    bottom_right_x: float,
    bottom_right_y: float,
    bottom_left_x: float,
    bottom_left_y: float,
):
    """Draw the selected manual corridor on the first frame without inference."""
    if not upload or not Path(upload).is_file():
        return None
    capture = cv2.VideoCapture(str(upload))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    region = _manual_forward_region(
        zone_mode, top_y, top_half_width, bottom_half_width,
        top_left_x, top_left_y, top_right_x, top_right_y,
        bottom_right_x, bottom_right_y, bottom_left_x, bottom_left_y,
    )
    height, width = frame.shape[:2]
    points = np.array([(round(x * width), round(y * height)) for x, y in region], dtype=np.int32)
    cv2.polylines(frame, [points], True, (255, 255, 0), 3)
    cv2.putText(frame, "CALIBRATED FORWARD ZONE", tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_demo() -> gr.Blocks:
    """Build a compact, phone-friendly interface around the proven CLI pipeline."""
    with gr.Blocks(
        title="SafeDrive ADAS Demo",
    ) as demo:
        gr.Markdown(
            "# SafeDrive ADAS Demo\n"
            "Upload a dashcam image or short video. Processing happens on this laptop; "
            "the displayed warning prioritizes forward collision risk over road hazards."
        )
        gr.Markdown(
            "**Prototype notice:** warnings are decision-support only. They do not estimate "
            "physical distance or guarantee collision avoidance.", elem_classes=["notice"]
        )
        with gr.Tab("Analyse image"):
            with gr.Row():
                image_input = gr.Image(label="Dashcam image", type="filepath")
                image_output = gr.Image(label="SafeDrive result", type="filepath")
            image_button = gr.Button("Analyse image", variant="primary")
            image_status = gr.Textbox(label="Status", interactive=False)
            image_zone = gr.Checkbox(label="Show forward-risk zone (presentation overlay)", value=False)
            image_warning = gr.Textbox(label="Highest confirmed warning", interactive=False)
            image_button.click(
                analyse_image,
                inputs=[image_input, image_zone],
                outputs=[image_output, image_status, image_warning],
            )

        with gr.Tab("Analyse video"):
            with gr.Row():
                # Do not ask Gradio to transcode an upload before analysis.
                # The pipeline itself returns a browser-compatible H.264 MP4.
                video_input = gr.Video(label="Dashcam video")
                video_output = gr.Video(label="SafeDrive result")
            frame_stride = gr.Slider(
                minimum=1,
                maximum=30,
                value=4,
                step=1,
                label="Analysis interval (frames)",
                info="1 = analyse every frame (most reliable); 3–6 = recommended fast demo; 30 = rough preview only.",
            )
            video_zone = gr.Checkbox(label="Show forward-risk zone (presentation overlay)", value=False)
            with gr.Accordion("Manually adjust forward-risk zone for this video", open=False):
                gr.Markdown("Adjust the preview first, then analyse. This calibration changes the forward-risk rule for this upload; it is not automatic lane detection.")
                zone_mode = gr.Radio(
                    ["Symmetric guide", "Advanced four-corner"],
                    value="Symmetric guide", label="Calibration mode",
                )
                zone_top_y = gr.Slider(0.35, 0.75, value=0.58, step=0.01, label="Top edge height")
                zone_top_width = gr.Slider(0.05, 0.30, value=0.07, step=0.01, label="Top half-width")
                zone_bottom_width = gr.Slider(0.20, 0.48, value=0.30, step=0.01, label="Bottom half-width")
                with gr.Row():
                    zone_top_left_x = gr.Slider(0.0, 0.50, value=0.43, step=0.01, label="Top-left X")
                    zone_top_left_y = gr.Slider(0.20, 0.85, value=0.58, step=0.01, label="Top-left Y")
                    zone_top_right_x = gr.Slider(0.50, 1.0, value=0.57, step=0.01, label="Top-right X")
                    zone_top_right_y = gr.Slider(0.20, 0.85, value=0.58, step=0.01, label="Top-right Y")
                with gr.Row():
                    zone_bottom_right_x = gr.Slider(0.50, 1.0, value=0.80, step=0.01, label="Bottom-right X")
                    zone_bottom_right_y = gr.Slider(0.60, 1.0, value=0.98, step=0.01, label="Bottom-right Y")
                    zone_bottom_left_x = gr.Slider(0.0, 0.50, value=0.20, step=0.01, label="Bottom-left X")
                    zone_bottom_left_y = gr.Slider(0.60, 1.0, value=0.98, step=0.01, label="Bottom-left Y")
                zone_preview = gr.Image(label="Live zone preview (first video frame)", type="numpy", interactive=False)
            video_button = gr.Button("Analyse video", variant="primary")
            video_status = gr.Textbox(label="Status", interactive=False)
            video_warning = gr.Textbox(label="Highest confirmed warning", interactive=False)
            video_button.click(
                analyse_video,
                inputs=[
                    video_input, frame_stride, video_zone, zone_mode,
                    zone_top_y, zone_top_width, zone_bottom_width,
                    zone_top_left_x, zone_top_left_y, zone_top_right_x, zone_top_right_y,
                    zone_bottom_right_x, zone_bottom_right_y, zone_bottom_left_x, zone_bottom_left_y,
                ],
                outputs=[video_output, video_status, video_warning],
            )
            preview_inputs = [
                video_input, zone_mode, zone_top_y, zone_top_width, zone_bottom_width,
                zone_top_left_x, zone_top_left_y, zone_top_right_x, zone_top_right_y,
                zone_bottom_right_x, zone_bottom_right_y, zone_bottom_left_x, zone_bottom_left_y,
            ]
            for control in preview_inputs:
                control.change(preview_forward_region, inputs=preview_inputs, outputs=zone_preview)

        gr.Markdown(
            "For the live presentation, use a short video clip. Long videos process frame by frame "
            "and may take several minutes."
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local SafeDrive ADAS browser demo.")
    parser.add_argument("--host", default="0.0.0.0", help="Use 0.0.0.0 for phone access on local Wi-Fi")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create an optional temporary Gradio public link")
    args = parser.parse_args()

    WEB_OUTPUTS.mkdir(parents=True, exist_ok=True)
    build_demo().queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(WEB_OUTPUTS)],
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=".gradio-container {max-width: 1000px !important;} .notice {font-size: 0.95rem;}",
    )


if __name__ == "__main__":
    main()
