"""
helpers.py
==========
Shared utility functions for the ASL Alphabet CNN project.

Provides:
- Image preprocessing for EfficientNetB0
- Label I/O
- Dataset discovery
- Prediction smoothing via majority voting
"""

import json
import os
from typing import List

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


def load_labels(path: str) -> List[str]:
    """Load label names from a JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_labels(labels: List[str], path: str) -> None:
    """Save label names to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)


def discover_class_names(dataset_root: str) -> List[str]:
    """Discover class names from subdirectory names under dataset_root."""
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")
    return sorted(
        name for name in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, name))
        and len(name) == 1 and name.isalpha()
    )


def preprocess_frame(image, img_size: int = 224) -> np.ndarray:
    """
    Preprocess a BGR webcam frame for EfficientNetB0 prediction.

    Args:
        image: BGR image from OpenCV (numpy array)
        img_size: Target size (default 224 for EfficientNetB0)

    Returns:
        Batch-ready numpy array of shape (1, img_size, img_size, 3), float32 [0,1].
    """
    if image is None:
        raise ValueError("Image is empty")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def smoothing_vote(history, default: str = "?") -> str:
    """
    Apply majority voting over recent predictions.

    Args:
        history: Iterable of recent prediction labels.
        default: Default label if history is empty.

    Returns:
        The most frequent label in history.
    """
    if not history:
        return default

    counts = {}
    for label in history:
        if label is None:
            continue
        counts[label] = counts.get(label, 0) + 1

    if not counts:
        return default

    return max(counts.items(), key=lambda item: item[1])[0]
