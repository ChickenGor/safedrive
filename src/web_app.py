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

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import build_pipeline, process_image, process_video  # noqa: E402


WEB_OUTPUTS = PROJECT_ROOT / "outputs" / "web_app"
_pipeline = None
SPEAK_WARNING_JS = """
(warning, enabled) => {
    if (!enabled || !warning || !("speechSynthesis" in window)) return [];
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(warning);
    utterance.rate = 0.92;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
    return [];
}
"""


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


def analyse_image(upload: str | None, show_forward_zone: bool) -> tuple[str, str, str]:
    """Run the existing image pipeline and return a browser-displayable result."""
    source = _require_upload(upload, "road image")
    WEB_OUTPUTS.mkdir(parents=True, exist_ok=True)
    destination = WEB_OUTPUTS / f"image_{uuid4().hex}.jpg"
    warning = process_image(source, destination, _get_pipeline(), show_forward_zone)
    status = "Analysis complete. Review the warning shown in the image."
    return str(destination), status, warning


def analyse_video(upload: str | None, frame_stride: int, show_forward_zone: bool) -> tuple[str, str, str]:
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
    )
    mode = "standard" if int(frame_stride) == 1 else f"fast demo (every {int(frame_stride)}th frame analysed)"
    return str(destination), f"Analysis complete using {mode} mode.", warning


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
            image_voice = gr.Checkbox(label="Speak confirmed warning after analysis", value=True)
            image_warning = gr.Textbox(label="Highest confirmed warning", interactive=False)
            image_event = image_button.click(
                analyse_image,
                inputs=[image_input, image_zone],
                outputs=[image_output, image_status, image_warning],
            )
            image_event.then(fn=None, inputs=[image_warning, image_voice], js=SPEAK_WARNING_JS)

        with gr.Tab("Analyse video"):
            with gr.Row():
                video_input = gr.Video(label="Dashcam video", format="mp4")
                video_output = gr.Video(label="SafeDrive result", format="mp4")
            frame_stride = gr.Slider(
                minimum=1,
                maximum=6,
                value=4,
                step=1,
                label="Demo speed",
                info="1 = analyse every frame (slowest); 4 = fast demo mode (recommended).",
            )
            video_zone = gr.Checkbox(label="Show forward-risk zone (presentation overlay)", value=False)
            video_voice = gr.Checkbox(label="Speak highest confirmed warning after analysis", value=True)
            video_button = gr.Button("Analyse video", variant="primary")
            video_status = gr.Textbox(label="Status", interactive=False)
            video_warning = gr.Textbox(label="Highest confirmed warning", interactive=False)
            video_event = video_button.click(
                analyse_video,
                inputs=[video_input, frame_stride, video_zone],
                outputs=[video_output, video_status, video_warning],
            )
            video_event.then(fn=None, inputs=[video_warning, video_voice], js=SPEAK_WARNING_JS)

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
