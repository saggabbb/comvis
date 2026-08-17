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

# Minimum confidence for prediction (below this = UNCERTAIN)
CONFIDENCE_THRESHOLD = 0.60

# Maximum prediction FPS
MAX_FPS = 15


# ============================================================
# TEMPORAL SMOOTHING CONFIGURATION
# ============================================================

# Ukuran sliding window buffer per-koneksi
BUFFER_SIZE = 10

# Jumlah minimum frame dalam buffer sebelum menghasilkan stable prediction
MIN_BUFFER_SIZE = 3

# Minimum fraction of buffer frames that must agree on same class
# before prediction is marked stable (e.g. 0.60 = 60% of frames)
STABILITY_RATIO = 0.60

# Flag pengaktifan logging di terminal
DEBUG_LOGGING = True


# ============================================================
# SIGN LANGUAGE SETTINGS
# ============================================================

# Trigger word yang digunakan aplikasi
TRIGGER_WORD = "HENGKY"


# ============================================================
# ROI & HAND DETECTION CONFIGURATION
# ============================================================

# Margin bounding box tangan (25% — increased from 15% for better finger coverage)
ROI_MARGIN_RATIO = 0.25

# Static ROI fallback (ketika tangan tidak terdeteksi)
STATIC_ROI = {
    "x": 0.30,
    "y": 0.10,
    "w": 0.40,
    "h": 0.75,
}

# Flag pengaktifan logging ROI di terminal
DEBUG_ROI = True
