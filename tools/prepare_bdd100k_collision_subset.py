"""Create a reproducible two-class YOLO subset from BDD100K detection labels.

The source BDD100K files are never modified. Vehicle-like categories are merged
into class 0 (vehicle), while person and rider become class 1 (person).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


VEHICLE_CLASSES = {"car", "truck", "bus", "train", "motor", "bike"}
PERSON_CLASSES = {"person", "rider"}
CLASS_IDS = {**{name: 0 for name in VEHICLE_CLASSES}, **{name: 1 for name in PERSON_CLASSES}}


def find_images(images_root: Path, split: str) -> dict[str, Path]:
    """Index a BDD100K split, including the archive's nested trainA folders."""
    split_root = images_root / split
    return {path.name: path for path in split_root.rglob("*.jpg")}


def yolo_lines(record: dict, image_width: int = 1280, image_height: int = 720) -> list[str]:
    """Convert relevant BDD100K box2d annotations to normalized YOLO lines."""
    lines: list[str] = []
    for label in record.get("labels", []):
        class_id = CLASS_IDS.get(label.get("category", ""))
        box = label.get("box2d")
        if class_id is None or not box:
            continue
        x1, y1 = max(0.0, box["x1"]), max(0.0, box["y1"])
        x2, y2 = min(float(image_width), box["x2"]), min(float(image_height), box["y2"])
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        x_center, y_center = (x1 + x2) / 2 / image_width, (y1 + y2) / 2 / image_height
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width / image_width:.6f} {height / image_height:.6f}")
    return lines


def prepare_split(records: list[dict], image_index: dict[str, Path], destination: Path, split: str, count: int, rng: random.Random) -> None:
    candidates = [record for record in records if record["name"] in image_index and yolo_lines(record)]
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} usable images available; requested {count}.")
    for record in rng.sample(candidates, count):
        source = image_index[record["name"]]
        lines = yolo_lines(record)
        shutil.copy2(source, destination / "images" / split / source.name)
        (destination / "labels" / split / f"{source.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a BDD100K vehicle/person YOLO subset.")
    parser.add_argument("--images-root", type=Path, required=True, help=".../bdd100k/images/100k")
    parser.add_argument("--labels-root", type=Path, required=True, help=".../bdd100k/labels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=12000)
    parser.add_argument("--val-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output must be empty: {args.output}")
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for split, count in (("train", args.train_count), ("val", args.val_count)):
        label_file = args.labels_root / f"bdd100k_labels_images_{split}.json"
        records = json.loads(label_file.read_text(encoding="utf-8"))
        image_index = find_images(args.images_root, split)
        prepare_split(records, image_index, args.output, split, count, rng)

    (args.output / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: vehicle\n  1: person\n",
        encoding="utf-8",
    )
    print(f"Created subset at {args.output}")


if __name__ == "__main__":
    main()
