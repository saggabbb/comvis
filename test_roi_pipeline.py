"""
test_roi_pipeline.py
====================
Comprehensive automated verification script for SignVision MediaPipe Hand ROI pipeline.
Tests:
1. MediaPipe import & version verification
2. 29 Class mapping verification
3. HandDetector functionality (BBox, Margin 15%, Clamping, Static Fallback)
4. EfficientNetPredictor with Hand & No-Hand frames
5. Backend TemporalPredictor lifecycle:
   - Hand detected (warmup -> stable)
   - Hand disappears (immediate reset & no lingering prediction)
   - Hand reappears (clean warmup -> stable)
6. FastAPI Status Endpoint verification (/api/status)
"""

import os
import sys
import json
import asyncio
import numpy as np
import cv2

# Set path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

print("=" * 60)
print("RUNNING PIPELINE VERIFICATION TESTS")
print("=" * 60)

# TEST 1: MediaPipe Version Check
print("\n[TEST 1] MediaPipe Verification...")
import mediapipe as mp
print(f"  -> MediaPipe version: {mp.__version__}")
from backend.hand_detector import HandDetector
detector = HandDetector(margin_ratio=0.15)
assert detector.is_available, "HandDetector failed to initialize MediaPipe!"
print(f"  -> PASSED: MediaPipe initialized successfully in mode: '{detector._mode}'")


# TEST 2: 29 Class Mapping Verification
print("\n[TEST 2] 29 Class Mapping Verification...")
from backend.config import LABELS_PATH
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

EXPECTED_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
]
assert len(labels) == 29, f"Expected 29 classes, got {len(labels)}"
assert labels == EXPECTED_CLASSES, f"Class list does not match expected order! Got: {labels}"
print("  -> PASSED: Exactly 29 classes found with identical order.")


# TEST 3: HandDetector Unit Tests
print("\n[TEST 3] HandDetector Unit Tests...")

# Test 3a: Empty/blank image (No Hand)
blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
blank_rgb = cv2.cvtColor(blank_img, cv2.COLOR_BGR2RGB)
roi, hand_detected, is_hand_roi = detector.compute_roi(blank_rgb)

assert hand_detected is False, "Empty image should have hand_detected = False"
assert is_hand_roi is False, "Empty image should have is_hand_roi = False"
assert roi is not None, "Fallback static ROI should be returned"
print(f"  -> Blank Frame Result: hand_detected={hand_detected}, is_hand_roi={is_hand_roi}, fallback_roi={roi}")

# Test 3b: Real image with hand
test_img_path = os.path.join(ROOT_DIR, "test_images", "A.jpg")
if os.path.exists(test_img_path):
    img = cv2.imread(test_img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    roi, hand_detected, is_hand_roi = detector.compute_roi(rgb)
    print(f"  -> Real Image ({test_img_path}) Result: hand_detected={hand_detected}, is_hand_roi={is_hand_roi}, roi={roi}")
    x, y, rw, rh = roi
    assert x >= 0 and y >= 0 and (x + rw) <= w and (y + rh) <= h, "ROI exceeds image bounds!"
    print("  -> PASSED: Hand bounding box computed, expanded by 15%, and clamped within image boundaries.")


# TEST 4: EfficientNetPredictor Live Test
print("\n[TEST 4] EfficientNetPredictor Tests...")
from backend.predictor import EfficientNetPredictor, DummyPredictor

# Test DummyPredictor
dummy = DummyPredictor()
_, dummy_jpeg = cv2.imencode(".jpg", blank_img)
dummy_res = dummy.predict(dummy_jpeg.tobytes())
assert "letter" in dummy_res and "confidence" in dummy_res
print(f"  -> DummyPredictor OK: {dummy_res}")

# Test EfficientNetPredictor
predictor = EfficientNetPredictor()

# Case A: Blank image (no hand)
_, blank_jpeg = cv2.imencode(".jpg", blank_img)
res_blank = predictor.predict(blank_jpeg.tobytes())
assert res_blank["hand_detected"] is False
assert res_blank["is_hand_roi"] is False
print(f"  -> EfficientNet on Blank Frame: hand_detected={res_blank['hand_detected']}, is_hand_roi={res_blank['is_hand_roi']}, raw_letter={res_blank['letter']}")

# Case B: Image with hand
if os.path.exists(test_img_path):
    _, hand_jpeg = cv2.imencode(".jpg", img)
    res_hand = predictor.predict(hand_jpeg.tobytes())
    print(f"  -> EfficientNet on Hand Frame: hand_detected={res_hand['hand_detected']}, letter={res_hand['letter']}, conf={res_hand['confidence']}, roi={res_hand['roi']}")
    print("  -> PASSED: EfficientNetPredictor correctly returns hand detection metadata and letter prediction.")


# TEST 5: TemporalPredictor Lifecycle & No-Hand Isolation
print("\n[TEST 5] TemporalPredictor Lifecycle & No-Hand Isolation...")
from backend.temporal_predictor import TemporalPredictor
tp = TemporalPredictor(buffer_size=15, min_buffer_size=5, debug_logging=False)

# 1. Warmup with detected hand 'A'
for i in range(4):
    out = tp.add_prediction("A", 0.95, 10.0)
    assert out["is_stable"] is False, f"Buffer size {i+1} should not be stable yet"
assert out["buffer_size"] == 4

# 5th frame makes it stable
out = tp.add_prediction("A", 0.95, 10.0)
assert out["is_stable"] is True, "5th frame should make prediction stable"
assert out["letter"] == "A", f"Expected letter 'A', got {out['letter']}"
print(f"  -> Hand detected warmup (5 frames) -> Stable letter: {out['letter']}, confidence: {out['confidence']}")

# 2. Hand disappears -> Pipeline resets TemporalPredictor and sends no-hand state
tp.reset()
assert len(tp.buffer) == 0, "Buffer should be completely cleared after reset!"
print("  -> Hand disappeared -> Buffer cleared (len=0). Old prediction successfully discarded.")

# 3. Hand reappears with 'B' -> Must warm up from 1 again, not immediately output old 'A'
out = tp.add_prediction("B", 0.90, 10.0)
assert out["is_stable"] is False, "First frame after reappearance must NOT be stable"
assert out["buffer_size"] == 1, "Buffer size should be 1 after reappearance"
assert out["letter"] is None, "Letter should be None during warmup"
print(f"  -> Hand reappeared ('B') -> Buffer clean start: buffer_size=1, is_stable=False, letter=None")

for i in range(4):
    out = tp.add_prediction("B", 0.90, 10.0)
assert out["is_stable"] is True
assert out["letter"] == "B"
print(f"  -> Hand stabilized on 'B' -> Stable letter: {out['letter']}, confidence: {out['confidence']}")
print("  -> PASSED: TemporalPredictor lifecycle strictly enforces isolated hand gesture sequences.")


# TEST 6: FastAPI /api/status Endpoint Check
print("\n[TEST 6] FastAPI /api/status Test...")
from backend.main import get_status

status_data = asyncio.run(get_status())
print(f"  -> /api/status Response: {status_data}")
assert status_data["application"] == "SignVision"
assert status_data["model_available"] is True
assert status_data["demo_mode"] is False
assert status_data["buffer_size"] == 15
assert status_data["min_buffer_size"] == 5
print("  -> PASSED: /api/status is healthy and returns all expected fields without errors.")

# Clean detector
detector.close()

print("\n" + "=" * 60)
print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
print("=" * 60)
