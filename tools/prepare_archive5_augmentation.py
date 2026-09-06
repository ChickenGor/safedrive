"""Create a safe Archive (5) training augmentation for SafeDrive road hazards.

The original SafeDrive validation and test splits are copied unchanged. This
makes an experiment trained with the augmented training split comparable to the
baseline model. Archive (5) contains four road-damage categories that are
remapped to SafeDrive's three output classes:

    Archive (5) 0 pothole                 -> SafeDrive 0 pothole
    Archive (5) 1 alligator cracking      -> SafeDrive 1 crack
    Archive (5) 2 lateral cracking        -> SafeDrive 1 crack
    Archive (5) 3 longitudinal cracking   -> SafeDrive 1 crack

Archive (5) has no manhole annotations, so existing SafeDrive examples preserve
that class. The script never edits either input folder or any existing model.
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ARCHIVE5_TO_SAFEDRIVE = {0: 0, 1: 1, 2: 1, 3: 1}


def image_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def copy_baseline_split(source: Path, output: Path, split: str) -> Counter:
    """Copy an unmodified SafeDrive split and report its label instances."""
    images_source = source / "images" / split
    labels_source = source / "labels" / split
    images_output = output / "images" / split
    labels_output = output / "labels" / split
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()

    for image_path in image_files(images_source):
        label_path = labels_source / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing baseline label: {label_path}")
        shutil.copy2(image_path, images_output / image_path.name)
        shutil.copy2(label_path, labels_output / label_path.name)
        counts["images"] += 1
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[f"class_{int(line.split()[0])}"] += 1
    return counts


def remap_archive5_label(label_path: Path) -> tuple[str, Counter]:
    """Remap one YOLO label file while retaining its normalized coordinates."""
    output_lines: list[str] = []
    counts: Counter = Counter()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Unexpected YOLO label at {label_path}:{line_number}: {line!r}")
        source_class = int(fields[0])
        if source_class not in ARCHIVE5_TO_SAFEDRIVE:
            raise ValueError(f"Unsupported Archive (5) class {source_class} at {label_path}:{line_number}")
        target_class = ARCHIVE5_TO_SAFEDRIVE[source_class]
        output_lines.append(" ".join([str(target_class), *fields[1:]]))
        counts[f"class_{target_class}"] += 1
    return "\n".join(output_lines) + ("\n" if output_lines else ""), counts


def add_archive5_training_images(source: Path, output: Path) -> Counter:
    """Add Archive (5) examples to training only, with collision-proof names."""
    images_output = output / "images" / "train"
    labels_output = output / "labels" / "train"
    counts: Counter = Counter()
    for image_path in image_files(source):
        label_path = source / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing Archive (5) label: {label_path}")
        label_text, label_counts = remap_archive5_label(label_path)
        destination_stem = f"archive5_{image_path.stem}"
        shutil.copy2(image_path, images_output / f"{destination_stem}{image_path.suffix.lower()}")
        (labels_output / f"{destination_stem}.txt").write_text(label_text, encoding="utf-8")
        counts["images"] += 1
        counts.update(label_counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment SafeDrive road-hazard training data with Archive (5).")
    parser.add_argument("--baseline", type=Path, required=True, help="Existing SafeDrive_Dataset directory")
    parser.add_argument("--archive5", type=Path, required=True, help="Extracted Archive (5) directory")
    parser.add_argument("--output", type=Path, required=True, help="New, empty dataset directory")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output}")
    for required in (args.baseline / "images" / "train", args.baseline / "labels" / "train", args.archive5):
        if not required.is_dir():
            raise FileNotFoundError(f"Required input folder not found: {required}")
    args.output.mkdir(parents=True, exist_ok=True)

    baseline_counts = {
        split: copy_baseline_split(args.baseline, args.output, split)
        for split in ("train", "val", "test")
    }
    archive_counts = add_archive5_training_images(args.archive5, args.output)

    (args.output / "data.yaml").write_text(
        f"path: {args.output.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        "names:\n  0: Pothole\n  1: Crack\n  2: Manhole\n",
        encoding="utf-8",
    )
    (args.output / "ARCHIVE5_MAPPING.md").write_text(
        "# Archive (5) mapping\n\n"
        "Archive (5) was added to the training split only. Validation and test are unchanged from the baseline.\n\n"
        "- 0 pothole -> 0 Pothole\n"
        "- 1 alligator cracking -> 1 Crack\n"
        "- 2 lateral cracking -> 1 Crack\n"
        "- 3 longitudinal cracking -> 1 Crack\n",
        encoding="utf-8",
    )
    print("Created Archive (5) augmented SafeDrive dataset")
    for split, counts in baseline_counts.items():
        print(f"baseline {split}: {dict(counts)}")
    print(f"Archive (5) added to train: {dict(archive_counts)}")


if __name__ == "__main__":
    main()
