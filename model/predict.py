"""
model/predict.py
=================
Prediction wrapper for the EfficientNetB0-based ASL image classifier.

Loads SignVision_EfficientNetB0_final.keras and class_names.json, provides a simple
predict() interface for both webcam frames and image files.

IMPORTANT — Preprocessing:
    EfficientNetB0 has an internal Rescaling layer (scale=1/255).
    Input must be float32 in range [0, 255]. Do NOT pre-normalize.
    Use scripts.helpers.preprocess_frame which handles this correctly.
"""

import os
import sys

import numpy as np
import tensorflow as tf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.helpers import load_labels, preprocess_frame


class ImageClassifier:
    """EfficientNetB0-based ASL alphabet classifier."""

    def __init__(self, model_path=None, labels_path=None, img_size=224):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Use the correct final model filename
        self.model_path = model_path or os.path.join(
            base_dir, "SignVision_EfficientNetB0_final.keras"
        )
        self.labels_path = labels_path or os.path.join(base_dir, "class_names.json")
        self.img_size = img_size

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Place SignVision_EfficientNetB0_final.keras in model/ folder.\n"
                "Or run: python model/train_images.py"
            )
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(
                f"Labels not found: {self.labels_path}\n"
                "Run: python model/train_images.py"
            )

        tf.get_logger().setLevel("ERROR")
        self.model = tf.keras.models.load_model(self.model_path)
        self.labels = load_labels(self.labels_path)

    def predict(self, image):
        """
        Predict ASL letter from a BGR image (OpenCV format).

        Args:
            image: BGR numpy array from webcam or cv2.imread

        Returns:
            (letter, confidence, probabilities) tuple
        """
        # preprocess_frame returns float32 [0, 255] — model handles /255 internally
        batch = preprocess_frame(image, img_size=self.img_size)
        probabilities = self.model.predict(batch, verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        return self.labels[best_index], float(probabilities[best_index]), probabilities
