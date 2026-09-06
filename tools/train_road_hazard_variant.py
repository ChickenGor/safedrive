"""Train the road-hazard hard-class variant with conservative road-image augmentation."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("road_hazard_experiments"))
    parser.add_argument("--name", default="yolo11n_hardclass_aug")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4, help="Safe default for an RTX 4050 at 960 px")
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu when no NVIDIA GPU is available")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Data-loader workers. Keep 0 on Windows/Python 3.14 to avoid cache serialization errors.",
    )
    args = parser.parse_args()

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        # RAM caching is unreliable with Windows spawn-based worker processes on
        # Python 3.14. Disk reads are slower but make this project reproducible.
        cache=False,
        seed=0,
        deterministic=True,
        cos_lr=True,
        # Keep geometry conservative for thin cracks; vary road appearance instead.
        hsv_h=0.015,
        hsv_s=0.55,
        hsv_v=0.40,
        translate=0.05,
        scale=0.20,
        fliplr=0.50,
        mosaic=0.20,
        mixup=0.0,
        project=str(args.project),
        name=args.name,
    )


if __name__ == "__main__":
    main()
