"""
backend/predictor.py
====================

Predictor interface for ASL alphabet classification.

- DummyPredictor:
  Digunakan sebagai mode demo jika model tidak ditemukan.

- EfficientNetPredictor:
  Digunakan jika model EfficientNetB0 tersedia.

Model:
    model/SignVision_EfficientNetB0_final.keras

Labels:
    model/class_names.json
"""

import os
import sys
import time
import json
import random
import string

import numpy as np
import cv2


# ============================================================
# PATH PROJECT
# ============================================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ============================================================
# CONFIG
# ============================================================

from backend.config import (
    MODEL_PATH,
    LABELS_PATH,
    IMG_SIZE,
    ROI_MARGIN_RATIO,
    STATIC_ROI,
    DEBUG_ROI,
)
from backend.hand_detector import HandDetector


# ============================================================
# DUMMY PREDICTOR
# ============================================================

class DummyPredictor:
    """
    Predictor demo.

    Digunakan ketika model EfficientNetB0 tidak tersedia.
    Predictor akan menghasilkan huruf A-Z secara bergantian
    dengan confidence acak.
    """

    def __init__(self):
        self.letters = list(string.ascii_uppercase)
        self.idx = 0

    def predict(self, frame_bytes: bytes) -> dict:

        start = time.perf_counter()

        # Ambil huruf berdasarkan index
        letter = self.letters[self.idx]

        # Pindah ke huruf berikutnya
        self.idx = (self.idx + 1) % len(self.letters)

        # Confidence dummy
        confidence = round(
            random.uniform(0.70, 0.99),
            4
        )

        # Hitung waktu inference
        elapsed = (
            time.perf_counter() - start
        ) * 1000

        return {
            "letter": letter,
            "confidence": confidence,
            "inference_ms": round(elapsed, 2),
            "hand_detected": True,
            "roi": [0, 0, 320, 320],
            "is_hand_roi": True,
        }



# ============================================================
# EFFICIENTNET PREDICTOR
# ============================================================

class EfficientNetPredictor:
    """
    Predictor menggunakan model EfficientNetB0
    hasil training SignVision.

    Proses:
        JPEG bytes
            ↓
        Decode image
            ↓
        BGR → RGB
            ↓
        Resize 224x224
            ↓
        Float32
            ↓
        Tambahkan batch dimension
            ↓
        EfficientNetB0
            ↓
        Probabilitas 29 kelas
            ↓
        Ambil kelas dengan probabilitas terbesar
    """

    def __init__(self):

        # Import TensorFlow hanya ketika
        # EfficientNetPredictor benar-benar digunakan
        import tensorflow as tf

        # Mengurangi log TensorFlow
        tf.get_logger().setLevel("ERROR")

        # ====================================================
        # CEK FILE MODEL
        # ====================================================

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        # ====================================================
        # CEK FILE LABEL
        # ====================================================

        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(
                f"Labels not found: {LABELS_PATH}"
            )

        print(
            f"[Predictor] Loading model: {MODEL_PATH}"
        )

        print(
            f"[Predictor] Loading labels: {LABELS_PATH}"
        )

        # ====================================================
        # LOAD MODEL
        # ====================================================

        self.model = tf.keras.models.load_model(
            MODEL_PATH
        )

        print("[Predictor] Model loaded successfully.")

        # ====================================================
        # LOAD CLASS NAMES
        # ====================================================

        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            labels = json.load(f)

        # ====================================================
        # MEMBUAT LABEL MAP
        # ====================================================

        if isinstance(labels, dict):

            self.labels_map = {
                int(k): v
                for k, v in labels.items()
            }

        else:

            self.labels_map = {
                i: v
                for i, v in enumerate(labels)
            }

        # ====================================================
        # VALIDASI JUMLAH KELAS
        # ====================================================

        if len(self.labels_map) != 29:

            raise ValueError(
                f"Expected 29 classes, "
                f"but found {len(self.labels_map)} classes."
            )

        print(
            f"[Predictor] Classes loaded: "
            f"{len(self.labels_map)}"
        )

        print(
            f"[Predictor] Class names: "
            f"{list(self.labels_map.values())}"
        )


        # ====================================================
        # INIT HAND DETECTOR (MEDIAPIPE AUTO ROI)
        # ====================================================

        self.hand_detector = HandDetector(
            margin_ratio=ROI_MARGIN_RATIO,
            static_roi_config=STATIC_ROI,
        )

        if self.hand_detector.is_available:
            print("[Predictor] MediaPipe Hand ROI Detector ready.")
        else:
            print("[Predictor] MediaPipe unavailable - running with static ROI fallback.")


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, frame_bytes: bytes) -> dict:

        start = time.perf_counter()

        # ====================================================
        # 1. DECODE JPEG BYTES
        # ====================================================

        np_arr = np.frombuffer(
            frame_bytes,
            np.uint8
        )

        img = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        # Jika gambar gagal dibaca
        if img is None:

            return {
                "letter": "?",
                "confidence": 0.0,
                "inference_ms": 0.0,
                "hand_detected": False,
                "roi": None,
                "is_hand_roi": False,
            }

        # ====================================================
        # 2. BGR → RGB
        # ====================================================

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # ====================================================
        # 3. MEDIAPIPE HAND DETECTION & AUTO ROI
        # ====================================================

        roi, hand_detected, is_hand_roi = self.hand_detector.compute_roi(
            rgb
        )
        roi_x, roi_y, roi_w, roi_h = roi

        if DEBUG_ROI:
            if hand_detected:
                print(
                    f"[ROI] Hand: YES | ROI: ({roi_x}, {roi_y}, {roi_x + roi_w}, {roi_y + roi_h})"
                )
            else:
                print(
                    f"[ROI] Hand: NO | Using fallback ROI: ({roi_x}, {roi_y}, {roi_x + roi_w}, {roi_y + roi_h})"
                )

        # ====================================================
        # 4. CROP ROI TANGAN
        # ====================================================

        cropped = rgb[
            roi_y : roi_y + roi_h,
            roi_x : roi_x + roi_w
        ]

        if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
            cropped = rgb

        # ====================================================
        # 5. RESIZE 224x224
        # ====================================================

        resized = cv2.resize(
            cropped,
            (IMG_SIZE, IMG_SIZE)
        )

        # ====================================================
        # 6. CONVERT TO FLOAT32
        # ====================================================
        #
        # TIDAK menggunakan:
        #
        # resized = resized / 255.0
        #
        # karena model EfficientNetB0 yang digunakan
        # menangani preprocessing inputnya sendiri.
        #

        input_img = resized.astype(
            np.float32
        )

        # ====================================================
        # 7. TAMBAHKAN BATCH DIMENSION (1, 224, 224, 3)
        # ====================================================

        input_batch = np.expand_dims(
            input_img,
            axis=0
        )

        # ====================================================
        # 8. MODEL INFERENCE
        # ====================================================

        preds = self.model.predict(
            input_batch,
            verbose=0
        )

        # ====================================================
        # 9. AMBIL CONFIDENCE TERTINGGI & INDEX KELAS
        # ====================================================

        confidence = float(
            np.max(preds)
        )

        label_idx = int(
            np.argmax(preds)
        )

        # ====================================================
        # 10. UBAH INDEX MENJADI LABEL
        # ====================================================

        letter = self.labels_map.get(
            label_idx,
            "?"
        )

        # ====================================================
        # 11. HITUNG WAKTU INFERENCE
        # ====================================================

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        # ====================================================
        # 12. RETURN HASIL
        # ====================================================

        return {
            "letter": letter,
            "confidence": round(
                confidence,
                4
            ),
            "inference_ms": round(
                elapsed,
                2
            ),
            "hand_detected": hand_detected,
            "roi": [roi_x, roi_y, roi_w, roi_h],
            "is_hand_roi": is_hand_roi,
        }



# ============================================================
# PREDICTOR FACTORY
# ============================================================

def get_predictor():
    """
    Memilih predictor secara otomatis.

    Jika model tersedia:
        → EfficientNetPredictor

    Jika model tidak tersedia:
        → DummyPredictor
    """

    # ========================================================
    # CEK MODEL
    # ========================================================

    if os.path.exists(MODEL_PATH):

        print(
            f"[Predictor] Model ditemukan:"
        )

        print(
            f"[Predictor] {MODEL_PATH}"
        )

        try:

            predictor = EfficientNetPredictor()

            print(
                "[Predictor] "
                "EfficientNetPredictor aktif."
            )

            return predictor

        except Exception as e:

            print(
                f"[Predictor] "
                f"Failed to load model: {e}"
            )

            print(
                "[Predictor] "
                "Falling back to DummyPredictor."
            )

            return DummyPredictor()

    # ========================================================
    # MODEL TIDAK DITEMUKAN
    # ========================================================

    else:

        print(
            "[Predictor] Model tidak ditemukan."
        )

        print(
            "[Predictor] "
            "Menggunakan DummyPredictor (demo mode)."
        )

        return DummyPredictor()