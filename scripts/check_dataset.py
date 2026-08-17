"""
scripts/check_dataset.py
========================
Deep check all images in the dataset/ folder.

Usage:
    python scripts/check_dataset.py
    python scripts/check_dataset.py --dataset dataset
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_dataset(dataset_dir: str):
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"ERROR: Dataset folder not found: {dataset_path.resolve()}")
        sys.exit(1)

    bad_images = []
    total = 0
    class_summary = {}

    print(f"Deep checking dataset: {dataset_path.resolve()}\n")

    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue

        count = 0
        bad_count = 0
        print(f"Checking {class_dir.name}...")

        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            total += 1
            count += 1

            try:
                with Image.open(path) as img:
                    img.load()
                    img = img.convert("RGB")
                    img.load()
            except Exception as e:
                bad_images.append((path, repr(e)))
                bad_count += 1
                print(f"  ❌ BAD: {path.name} — {e}")

        class_summary[class_dir.name] = {"total": count, "bad": bad_count}

    print("\n" + "=" * 60)
    print(f"Total checked : {total}")
    print(f"Bad images    : {len(bad_images)}")

    print("\n--- Per-class summary ---")
    for cls, info in class_summary.items():
        status = "✅" if info["bad"] == 0 else "❌"
        print(f"  {status} {cls:10s}: {info['total']} images, {info['bad']} bad")

    if bad_images:
        print("\nLIST OF BAD IMAGES:")
        for path, error in bad_images:
            print(f"  {path}")
            print(f"    {error}")
        sys.exit(1)
    else:
        print("\n✅ All images loaded and converted to RGB successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep check dataset images")
    parser.add_argument("--dataset", default="dataset", help="Dataset root directory")
    args = parser.parse_args()
    check_dataset(args.dataset)
