"""Create a training-only road-hazard variant without altering the baseline data.

Pothole and crack images receive one extra copy in the training split. Validation
and test data are copied unchanged, so results remain directly comparable with the
existing baseline. Run this once after any current dataset/training job finishes.
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path


HARD_CLASSES = {0, 1}  # 0=Pothole, 1=Crack; 2=Manhole is already the strongest class.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def classes_in_label(label_path: Path) -> set[int]:
    return {
        int(line.split()[0])
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def copy_pair(image_path: Path, label_path: Path, destination: Path, suffix: str = "") -> None:
    image_destination = destination / "images" / f"{image_path.stem}{suffix}{image_path.suffix.lower()}"
    label_destination = destination / "labels" / f"{image_path.stem}{suffix}.txt"
    shutil.copy2(image_path, image_destination)
    shutil.copy2(label_path, label_destination)


def copy_split(source: Path, output: Path, split: str, duplicate_hard_examples: bool) -> Counter:
    destination = output / split
    (destination / "images").mkdir(parents=True, exist_ok=True)
    (destination / "labels").mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for image_path in (source / "images" / split).iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = source / "labels" / split / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        image_classes = classes_in_label(label_path)
        copy_pair(image_path, label_path, destination)
        counts["images"] += 1
        counts.update(image_classes)
        if duplicate_hard_examples and image_classes & HARD_CLASSES:
            copy_pair(image_path, label_path, destination, "_hardcopy")
            counts["extra_hard_images"] += 1
            counts.update(image_classes)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a hard-class oversampling variant for SafeDrive.")
    parser.add_argument("--source", type=Path, required=True, help="Existing SafeDrive_Dataset folder")
    parser.add_argument("--output", type=Path, required=True, help="New, empty variant folder")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output folder must be empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    train = copy_split(args.source, args.output, "train", duplicate_hard_examples=True)
    val = copy_split(args.source, args.output, "val", duplicate_hard_examples=False)
    test = copy_split(args.source, args.output, "test", duplicate_hard_examples=False)

    (args.output / "data.yaml").write_text(
        f"path: {args.output.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        "names:\n  0: Pothole\n  1: Crack\n  2: Manhole\n",
        encoding="utf-8",
    )
    print("Created road-hazard hard-class variant")
    for split, counts in (("train", train), ("val", val), ("test", test)):
        print(f"{split}: {dict(counts)}")


if __name__ == "__main__":
    main()
