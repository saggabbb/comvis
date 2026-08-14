"""
backend/config.py
=================

Configuration constants for the SignVision ASL
web application.
"""

import os


# ============================================================
# BASE DIRECTORIES
# ============================================================

# Lokasi folder backend/
BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Lokasi root project/
ROOT_DIR = os.path.abspath(
    os.path.join(BACKEND_DIR, "..")
)


# ============================================================
# SERVER CONFIGURATION
# ============================================================

HOST = "0.0.0.0"

PORT = 8000

WS_PREDICT_PATH = "/ws/predict"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Model hasil final training + fine-tuning
MODEL_PATH = os.path.join(
    ROOT_DIR,
    "model",
    "SignVision_EfficientNetB0_final.keras"
)

# Mapping index output model → nama kelas
LABELS_PATH = os.path.join(
    ROOT_DIR,
    "model",
    "class_names.json"
)


# ============================================================
# PREDICTION CONFIGURATION
# ============================================================

# Ukuran input model
IMG_SIZE = 224

# Minimum confidence prediction
CONFIDENCE_THRESHOLD = 0.60

# Maximum prediction FPS
MAX_FPS = 8


# ============================================================
# SIGN LANGUAGE SETTINGS
# ============================================================

# Trigger word yang digunakan aplikasi
TRIGGER_WORD = "HENGKY"