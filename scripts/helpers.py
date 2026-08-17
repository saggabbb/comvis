"""
scripts/helpers.py
==================
Shared utility functions for the ASL Alphabet CNN project.

Provides:
- Image preprocessing for EfficientNetB0 inference
- Label I/O utilities
- Dataset class discovery (supports del/nothing/space)
- Prediction smoothing via majority voting

IMPORTANT — Preprocessing Note:
    EfficientNetB0 in SignVision_EfficientNetB0_final.keras already
    contains an internal Rescaling layer (scale = 1/255).

    Therefore, input to the model should be:
        float32 in range [0, 255]   ← DO NOT pre-divide by 255

    This is consistent with backend/predictor.py inference pipeline.
"""

import json
import os
from typing import List

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

# Special class names that are valid but not single alpha chars
SPECIAL_CLASS_NAMES = {"del", "nothing", "space"}


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
    """
    Discover class names from subdirectory names under dataset_root.

    Supports:
    - Single uppercase letter folders: A, B, C ... Z
    - Special folders: del, nothing, space

    Returns sorted list of class names.
    """
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    names = []
    for name in os.listdir(dataset_root):
        if not os.path.isdir(os.path.join(dataset_root, name)):
            continue
        # Accept single uppercase alpha letters
        if len(name) == 1 and name.isalpha():
            names.append(name.upper())
        # Accept special class names (case-insensitive match)
        elif name.lower() in SPECIAL_CLASS_NAMES:
            names.append(name.lower())

    # Sort: uppercase letters first (A-Z), then special classes (del, nothing, space)
    alpha = sorted(n for n in names if len(n) == 1 and n.isalpha())
    special = sorted(n for n in names if n.lower() in SPECIAL_CLASS_NAMES)
    return alpha + special


def preprocess_frame(image, img_size: int = 224) -> np.ndarray:
    """
    Preprocess a BGR image for EfficientNetB0 prediction.

    Args:
        image: BGR image from OpenCV (numpy array)
        img_size: Target size (default 224 for EfficientNetB0)

    Returns:
        Batch-ready numpy array of shape (1, img_size, img_size, 3),
        dtype float32, range [0, 255].

    NOTE: Do NOT divide by 255 here. EfficientNetB0 in the .keras model
    has an internal Rescaling layer that handles normalization.
    """
    if image is None:
        raise ValueError("Image is empty")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    # Cast to float32 — range stays [0, 255], model handles /255 internally
    as_float = resized.astype(np.float32)
    return np.expand_dims(as_float, axis=0)


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
