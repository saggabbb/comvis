"""
backend/config.py
==================
Configuration constants for the ASL web application.
"""

import os

# Base directories
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))

# Server
HOST = "0.0.0.0"
PORT = 8000
WS_PREDICT_PATH = "/ws/predict"

# Model paths
MODEL_PATH = os.path.join(ROOT_DIR, "model", "model.keras")
LABELS_PATH = os.path.join(ROOT_DIR, "model", "labels.json")

# Prediction
TRIGGER_WORD = "HENGKY"
CONFIDENCE_THRESHOLD = 0.60
MAX_FPS = 8
IMG_SIZE = 224
