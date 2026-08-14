"""
backend/predictor.py
=====================
Predictor interface for ASL alphabet classification.

DummyPredictor: Used when model.keras is not available (demo mode).
EfficientNetPredictor: Used when a trained model exists.

The factory function get_predictor() automatically selects the right one.
"""

import os
import sys
import time
import json
import random
import string

import numpy as np
import cv2

# Add parent directory to path for config imports
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.config import MODEL_PATH, LABELS_PATH, IMG_SIZE


class DummyPredictor:
    """
    Demo predictor that cycles through A-Z with random confidence.
    Used when no trained model is available.
    """

    def __init__(self):
        self.letters = list(string.ascii_uppercase)
        self.idx = 0

    def predict(self, frame_bytes: bytes) -> dict:
        start = time.perf_counter()

        letter = self.letters[self.idx]
        self.idx = (self.idx + 1) % len(self.letters)

        confidence = round(random.uniform(0.70, 0.99), 4)
        elapsed = (time.perf_counter() - start) * 1000

        return {
            "letter": letter,
            "confidence": confidence,
            "inference_ms": round(elapsed, 2),
        }


class EfficientNetPredictor:
    """
    Real predictor using a trained EfficientNetB0 Keras model.
    Decodes JPEG bytes, resizes, normalizes, and runs inference.
    """

    def __init__(self):
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

        self.model = tf.keras.models.load_model(MODEL_PATH)

        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)

        if isinstance(labels, dict):
            self.labels_map = {int(k): v for k, v in labels.items()}
        else:
            self.labels_map = {i: v for i, v in enumerate(labels)}

    def predict(self, frame_bytes: bytes) -> dict:
        start = time.perf_counter()

        # Decode JPEG bytes
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"letter": "?", "confidence": 0.0, "inference_ms": 0.0}

        # BGR -> RGB, resize, normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # Inference
        preds = self.model.predict(img, verbose=0)
        confidence = float(np.max(preds))
        label_idx = int(np.argmax(preds))
        letter = self.labels_map.get(label_idx, "?")

        elapsed = (time.perf_counter() - start) * 1000

        return {
            "letter": letter,
            "confidence": round(confidence, 4),
            "inference_ms": round(elapsed, 2),
        }


def get_predictor():
    """Factory: return EfficientNetPredictor if model exists, else DummyPredictor."""
    if os.path.exists(MODEL_PATH):
        print(f"[Predictor] Loading EfficientNetPredictor from {MODEL_PATH}")
        try:
            return EfficientNetPredictor()
        except Exception as e:
            print(f"[Predictor] Failed to load model: {e}")
            print("[Predictor] Falling back to DummyPredictor")
            return DummyPredictor()
    else:
        print("[Predictor] model.keras not found - using DummyPredictor (demo mode)")
        return DummyPredictor()
