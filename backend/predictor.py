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

from backend.config import MODEL_PATH, LABELS_PATH, IMG_SIZE


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
            }

        # ====================================================
        # 2. BGR → RGB
        # ====================================================

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # ====================================================
        # 3. RESIZE
        # ====================================================

        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        )

        # ====================================================
        # 4. CONVERT TO FLOAT32
        # ====================================================
        #
        # TIDAK menggunakan:
        #
        # img = img / 255.0
        #
        # karena model EfficientNetB0 yang digunakan
        # menangani preprocessing inputnya sendiri.
        #

        img = img.astype(
            np.float32
        )

        # ====================================================
        # 5. TAMBAHKAN BATCH DIMENSION
        # ====================================================

        img = np.expand_dims(
            img,
            axis=0
        )

        # Bentuk input:
        #
        # (1, 224, 224, 3)

        # ====================================================
        # 6. MODEL INFERENCE
        # ====================================================

        preds = self.model.predict(
            img,
            verbose=0
        )

        # ====================================================
        # 7. AMBIL CONFIDENCE TERTINGGI
        # ====================================================

        confidence = float(
            np.max(preds)
        )

        # ====================================================
        # 8. AMBIL INDEX KELAS
        # ====================================================

        label_idx = int(
            np.argmax(preds)
        )

        # ====================================================
        # 9. UBAH INDEX MENJADI LABEL
        # ====================================================

        letter = self.labels_map.get(
            label_idx,
            "?"
        )

        # ====================================================
        # 10. HITUNG WAKTU INFERENCE
        # ====================================================

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        # ====================================================
        # 11. RETURN HASIL
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