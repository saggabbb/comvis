"""
realtime.py
============
Professional real-time ASL alphabet detection using webcam + EfficientNetB0 CNN.

Architecture (MAINTAINED):
    Webcam -> EfficientNetB0 -> Prediction -> Sentence Builder -> Special Mode

MediaPipe is used ONLY for VISUALIZATION:
    - hand detection
    - 21-landmark skeleton drawing
    - landmark connections
    - hand bounding box for Auto ROI

MediaPipe is NEVER used for:
    - classification
    - feature extraction
    - decision making
    - letter determination

If MediaPipe fails to detect a hand, the EfficientNet classifier keeps running
using the static ROI (fallback).

Features:
    - Auto ROI (hand bbox + 15% margin) with static ROI fallback
    - EfficientNetB0 image classification
    - Prediction smoothing (deque + majority voting)
    - Confidence threshold (< 60% = UNKNOWN)
    - Confidence progress bar
    - Hand status (Hand Detected / No Hand)
    - Inference time + FPS counters
    - Sentence builder (SPACE/BACKSPACE/ENTER/ESC)
    - Special mode: cinematic sequence (freeze -> fade -> secret -> countdown -> website)

Usage:
    python realtime.py
"""

import os
import sys
import time
from collections import deque

import cv2
import numpy as np

# MediaPipe init. Imported explicitly so it can be guarded/failed gracefully.
try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    mp = None
    _MEDIAPIPE_AVAILABLE = False

# UTF-8 safe console output on Windows (for emoji like 🟢/🔴/✨)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from helpers import smoothing_vote
from model.predict import ImageClassifier

# ── Configuration ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_WINDOW = 10
WEBCAM_INDEX = 0

# Auto ROI margin (as fraction of hand bounding box size)
ROI_MARGIN_RATIO = 0.15
# Static ROI (fallback when no hand detected) as fractions of frame
STATIC_ROI = {"x": 0.30, "y": 0.10, "w": 0.40, "h": 0.75}

# ── Key codes ──────────────────────────────────────────────────
KEY_ESC = 27
KEY_SPACE = 32
KEY_ENTER = 13
KEY_BACKSPACE = 8

# ── Colors (BGR) ───────────────────────────────────────────────
COLOR_GREEN = (40, 200, 40)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 230, 255)
COLOR_RED = (60, 60, 255)
COLOR_CYAN = (255, 220, 100)
COLOR_DARK_BG = (30, 30, 30)
COLOR_PANEL_BG = (20, 20, 20)
COLOR_SENTENCE_BG = (40, 40, 40)
COLOR_FPS = (180, 180, 240)
COLOR_UNKNOWN = (100, 100, 255)
COLOR_MAGENTA = (200, 50, 200)
COLOR_ORANGE = (0, 165, 255)
COLOR_GOLD = (0, 215, 255)
COLOR_LANDMARK = (0, 255, 0)          # MediaPipe default green for landmarks
COLOR_CONNECTION = (255, 255, 255)     # MediaPipe default white for connections
COLOR_HAND_BOX = (255, 0, 0)           # Blue for hand bounding box (BGR)

# ── MediaPipe hand connections (standard topology) ─────────────
HAND_CONNECTIONS = frozenset({
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
})


# ═══════════════════════════════════════════════════════════════
# MediaPipe Hand visualizer
# ═══════════════════════════════════════════════════════════════
class HandVisualizer:
    """
    Wraps MediaPipe Hands for VISUALIZATION ONLY.

    Responsibilities:
        - detect hand(s)
        - provide normalized 21 landmarks
        - provide a pixel-space bounding box (with optional margin)

    This class NEVER returns classification or feature vectors.
    """

    def __init__(self):
        self._mp_hands = None
        self._hands = None
        self._mp_drawing = None
        self._mp_drawing_styles = None
        self._setup()

    def _setup(self):
        """Initialize MediaPipe Hands if available."""
        if not _MEDIAPIPE_AVAILABLE:
            print("  [MediaPipe] Not available - running with static ROI only.")
            return
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @property
    def available(self):
        return self._hands is not None

    def process(self, rgb_frame):
        """Return MediaPipe results (raw object) or None."""
        if not self.available:
            return None
        try:
            return self._hands.process(rgb_frame)
        except Exception:
            return None

    def draw(self, frame, results):
        """Draw hand skeleton + connections onto the frame."""
        if not self.available or results is None or not results.multi_hand_landmarks:
            return
        for landmarks in results.multi_hand_landmarks:
            # Connections (white) with MediaPipe default style
            self._mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_drawing_styles.get_default_hand_landmarks_style(),
                self._mp_drawing_styles.get_default_hand_connections_style(),
            )

    def get_bbox(self, results, frame_w, frame_h, margin_ratio=ROI_MARGIN_RATIO):
        """
        Compute a pixel-space bounding box around the detected hand.

        Args:
            results: MediaPipe results from process()
            frame_w, frame_h: frame dimensions
            margin_ratio: extra margin added around the hand (fraction of box size)

        Returns:
            (x, y, w, h) integer ROI, or None if no hand detected.
        """
        if not self.available or results is None or not results.multi_hand_landmarks:
            return None

        landmarks = results.multi_hand_landmarks[0].landmark
        xs = [lm.x * frame_w for lm in landmarks]
        ys = [lm.y * frame_h for lm in landmarks]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        box_w = max_x - min_x
        box_h = max_y - min_y
        margin_x = box_w * margin_ratio
        margin_y = box_h * margin_ratio

        x = int(min_x - margin_x)
        y = int(min_y - margin_y)
        w = int(box_w + 2 * margin_x)
        h = int(box_h + 2 * margin_y)

        # Clamp to frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)

        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)


# ═══════════════════════════════════════════════════════════════
# Drawing utilities
# ═══════════════════════════════════════════════════════════════
def _draw_rounded_rect(frame, p1, p2, radius, color, thickness=-1, alpha=0.0):
    """Draw a rectangle, optionally filled with transparency."""
    x1, y1 = p1
    x2, y2 = p2
    if alpha > 0.0 and thickness == -1:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def _put_text(frame, text, org, scale, color, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Helper to draw text with a subtle shadow for readability."""
    x, y = org
    cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1,
                cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_roi_box(frame, roi):
    """Draw the ROI rectangle where the hand/letter is being classified."""
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_GREEN, 2)
    _put_text(frame, "Place hand here", (x, y - 10), 0.5, COLOR_GREEN, thickness=1)


def draw_hand_landmarks(frame, visualizer, results, hand_roi):
    """
    Draw the MediaPipe skeleton + connections + hand bounding box.

    This is purely visual. No classification happens here.
    """
    if not visualizer.available or results is None:
        return

    # Skeleton + connections
    visualizer.draw(frame, results)

    # Hand bounding box (blue) - the Auto ROI from MediaPipe
    if hand_roi is not None:
        x, y, w, h = hand_roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_HAND_BOX, 2)
        _put_text(frame, "Hand ROI", (x, y - 8), 0.45, COLOR_HAND_BOX, thickness=1)


def draw_progress_bar(frame, value, x, y, w, h, label="Confidence"):
    """
    Draw a modern confidence progress bar with percentage.

    Args:
        value: float in [0, 1]
    """
    value = max(0.0, min(1.0, value))

    # Bar background
    _draw_rounded_rect(frame, (x, y), (x + w, y + h), 4, (40, 40, 40), -1)

    fill_w = int(w * value)
    if fill_w > 0:
        # Color transitions green -> yellow -> red based on confidence
        if value >= CONFIDENCE_THRESHOLD:
            bar_color = COLOR_GREEN
        elif value >= 0.4:
            bar_color = COLOR_YELLOW
        else:
            bar_color = COLOR_RED
        _draw_rounded_rect(frame, (x, y), (x + fill_w, y + h), 4, bar_color, -1)

    # Label + percentage
    _put_text(frame, label, (x, y - 8), 0.5, COLOR_WHITE, thickness=1)
    pct = f"{value * 100:.0f}%"
    txt_w = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    _put_text(frame, pct, (x + w - txt_w, y + h + 16), 0.5, bar_color if fill_w > 0 else COLOR_WHITE,
              thickness=1)


def draw_status(frame, hand_detected, x, y):
    """
    Draw 🟢 Hand Detected / 🔴 No Hand status banner.
    """
    if hand_detected:
        text = "HAND DETECTED"
        color = COLOR_GREEN
        icon = "●"
    else:
        text = "NO HAND"
        color = COLOR_RED
        icon = "●"

    # Small pill background
    txt_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
    pill_w = txt_w + 40
    _draw_rounded_rect(frame, (x, y - 24), (x + pill_w, y + 8), 8, COLOR_PANEL_BG, -1, alpha=0.7)

    _put_text(frame, icon, (x + 8, y), 0.6, color, thickness=2)
    _put_text(frame, text, (x + 30, y), 0.55, color, thickness=2)


def draw_info_panel(frame, pred, confidence, fps, inference_ms, hand_detected):
    """Draw the modern top-left info panel (transparent overlay)."""
    h, w = frame.shape[:2]
    panel_w = 360
    panel_h = 150
    x0, y0 = 10, 10

    # Transparent panel background
    _draw_rounded_rect(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), 12, COLOR_PANEL_BG, -1,
                       alpha=0.65)

    # Accent line
    cv2.line(frame, (x0, y0), (x0 + panel_w, y0), COLOR_CYAN, 3)

    # Prediction row
    pred_color = COLOR_UNKNOWN if pred == "UNKNOWN" else COLOR_GREEN
    _put_text(frame, "PREDICTION", (x0 + 14, y0 + 26), 0.45, (150, 150, 150), thickness=1)
    _put_text(frame, pred, (x0 + 150, y0 + 30), 0.7, pred_color, thickness=2)

    # Confidence progress bar
    draw_progress_bar(frame, confidence if confidence else 0.0, x0 + 14, y0 + 44, panel_w - 90, 12)

    # Status hand
    draw_status(frame, hand_detected, x0 + 14, y0 + 92)

    # Right column: FPS + inference time
    _put_text(frame, f"FPS  {fps:.1f}", (x0 + 14, y0 + 122), 0.5, COLOR_FPS, thickness=1)
    _put_text(frame, f"INF  {inference_ms} ms", (x0 + 150, y0 + 122), 0.5, COLOR_FPS, thickness=1)


def draw_sentence(frame, sentence, current_letter):
    """Draw the sentence builder bar at the bottom."""
    h, w = frame.shape[:2]
    bar_height = 90
    bar_y = h - bar_height

    # Transparent bar
    _draw_rounded_rect(frame, (0, bar_y), (w, h), 0, COLOR_SENTENCE_BG, -1, alpha=0.75)
    cv2.line(frame, (0, bar_y), (w, bar_y), COLOR_CYAN, 2)

    _put_text(frame, "SENTENCE BUILDER", (10, bar_y + 24), 0.55, COLOR_CYAN, thickness=1)

    # Controls hint on the right
    controls = "SPACE:add | BS:del | ENTER:done | ESC:quit"
    ctrl_w = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
    _put_text(frame, controls, (max(10, w - ctrl_w - 10), bar_y + 24), 0.4, (150, 150, 150),
              thickness=1)

    # Sentence text with cursor
    display_text = (sentence + "_") if sentence else "Press SPACE to start building..."
    disp_color = COLOR_WHITE if sentence else (160, 160, 160)
    _put_text(frame, display_text, (10, bar_y + 62), 0.9, disp_color, thickness=2)

    # Current letter indicator
    if current_letter and current_letter != "UNKNOWN":
        _put_text(frame, f"Current: [{current_letter}]", (max(10, w - 210), bar_y + 62), 0.65,
                  COLOR_YELLOW, thickness=2)


# ═══════════════════════════════════════════════════════════════
# Auto ROI
# ═══════════════════════════════════════════════════════════════
def get_static_roi(frame_w, frame_h):
    """Return the static (fallback) ROI based on frame dimensions."""
    x = int(frame_w * STATIC_ROI["x"])
    y = int(frame_h * STATIC_ROI["y"])
    w = int(frame_w * STATIC_ROI["w"])
    h = int(frame_h * STATIC_ROI["h"])
    return (x, y, w, h)


def get_hand_roi(visualizer, results, frame_w, frame_h):
    """
    Auto ROI: hand bounding box from MediaPipe + 15% margin.

    Returns None if no hand detected (caller falls back to static ROI).
    """
    if not visualizer.available or results is None:
        return None
    return visualizer.get_bbox(results, frame_w, frame_h, ROI_MARGIN_RATIO)


def compute_active_roi(visualizer, results, frame_w, frame_h):
    """
    Decide which ROI to use for classification.

    Returns (roi, is_hand_roi) where is_hand_roi is True when the hand-based
    Auto ROI was used (helpful for UI status).
    """
    hand_roi = get_hand_roi(visualizer, results, frame_w, frame_h)
    if hand_roi is not None:
        return hand_roi, True
    return get_static_roi(frame_w, frame_h), False


def _show_secret_unlocked(frame, window_name):
    """Show the ✨ SECRET UNLOCKED ✨ reveal animation."""
    h, w = frame.shape[:2]
    for i in range(20):
        display = np.zeros_like(frame)
        # Pulsing glow border
        alpha = 0.4 + 0.3 * (i % 2)
        border_color = COLOR_GOLD if i % 2 == 0 else COLOR_MAGENTA
        cv2.rectangle(display, (8, 8), (w - 8, h - 8), border_color, 4)
        cv2.rectangle(display, (16, 16), (w - 16, h - 16), border_color, 2)

        # Center reveal box
        overlay = display.copy()
        cv2.rectangle(overlay, (w // 5, h // 3), (4 * w // 5, 2 * h // 3), COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)

        _put_text(display, "✨ SECRET UNLOCKED ✨", (w // 5 + 40, h // 2 - 15), 1.2, COLOR_GOLD, 3)
        _put_text(display, "Preparing experience...", (w // 5 + 80, h // 2 + 30), 0.7, COLOR_WHITE,
                  2)

        cv2.imshow(window_name, display)
        cv2.waitKey(60)


def _loading_animation(frame, window_name):
    """Spinner-style loading animation."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    for i in range(24):
        angle = i * 15
        display = frame.copy()
        # Rotating arc
        cv2.ellipse(display, (cx, cy), (45, 45), 0, angle, angle + 120, COLOR_CYAN, 6, cv2.LINE_AA)
        _put_text(display, "LOADING", (cx - 40, cy - 70), 0.8, COLOR_WHITE, 2)
        cv2.imshow(window_name, display)
        cv2.waitKey(50)


def _countdown(frame, window_name):
    """Countdown 3 -> 2 -> 1."""
    h, w = frame.shape[:2]
    for number in (3, 2, 1):
        for _ in range(20):
            display = frame.copy()
            _put_text(display, str(number), (w // 2 - 40, h // 2 + 40), 3.0, COLOR_GOLD, 6)
            _put_text(display, "Opening in...", (w // 2 - 90, h // 2 - 60), 0.8, COLOR_WHITE, 2)
            cv2.imshow(window_name, display)
            cv2.waitKey(25)


def draw_special_animation(frame, window_name):
    """
    Full special mode cinematic sequence:
        freeze -> fade to black -> ✨ SECRET UNLOCKED ✨ -> loading -> countdown 3-2-1

    Returns when done (website opens afterwards).
    """
    _freeze_and_fade(frame, window_name)
    _show_secret_unlocked(frame, window_name)
    _loading_animation(frame, window_name)
    _countdown(frame, window_name)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    WINDOW_NAME = "ASL Alphabet - EfficientNetB0 Pro"

    print("=" * 60)
    print("ASL Alphabet - Real-time Detection (EfficientNetB0 Pro)")
    print("=" * 60)
    print("Controls:")
    print("  SPACE     -> Save current letter to sentence")
    print("  BACKSPACE -> Delete last letter")
    print("  ENTER     -> Complete sentence (print to terminal)")
    print("  ESC       -> Exit")
    print("=" * 60)

    # Load classifier (EfficientNetB0 - MAIN classifier)
    print("\nLoading model...", end=" ", flush=True)
    try:
        classifier = ImageClassifier()
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        print("Run: python model/train_images.py")
        sys.exit(1)

    # MediaPipe hand visualizer (VISUALIZATION ONLY)
    print("Initializing MediaPipe Hands...", end=" ", flush=True)
    visualizer = HandVisualizer()
    if visualizer.available:
        print("OK")
    else:
        print("SKIPPED (static ROI only)")

    # Open webcam
    print("Opening webcam...", end=" ", flush=True)
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("FAILED - Check camera connection")
        sys.exit(1)
    print("OK")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State
    history = deque(maxlen=SMOOTHING_WINDOW)
    sentence = ""
    completed_sentences = []
    prev_time = 0

    print("\nStarting detection...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # ── MediaPipe hand detection (visualization only) ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = visualizer.process(rgb)

        # ── Auto ROI: hand bbox (+15% margin) or static fallback ──
        roi, hand_roi_active = compute_active_roi(visualizer, results, w, h)
        roi_x, roi_y, roi_w, roi_h = roi

        # ── Classify with EfficientNetB0 ──
        cropped = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        infer_start = time.perf_counter()
        try:
            letter, confidence, _ = classifier.predict(cropped)
        except Exception:
            letter, confidence = None, 0.0
        inference_ms = int((time.perf_counter() - infer_start) * 1000)

        # ── Confidence threshold + smoothing ──
        if letter is None or confidence < CONFIDENCE_THRESHOLD:
            display_letter = "UNKNOWN"
            is_unknown = True
            history.append(None)
        else:
            is_unknown = False
            history.append(letter)
            smoothed = smoothing_vote(list(history), default=letter)
            display_letter = smoothed

        current_letter = display_letter if not is_unknown else "UNKNOWN"

        # ── FPS ──
        now = time.time()
        fps = 1.0 / (now - prev_time) if prev_time > 0 else 0.0
        prev_time = now

        # ── Draw everything ──
        draw_roi_box(frame, roi)
        draw_hand_landmarks(frame, visualizer, results, roi if hand_roi_active else None)
        draw_info_panel(frame, current_letter, confidence, fps, inference_ms, hand_roi_active)
        draw_sentence(frame, sentence, current_letter if not is_unknown else None)

        cv2.imshow(WINDOW_NAME, frame)

        # ── Keyboard ──
        key = cv2.waitKey(1) & 0xFF

        if key == KEY_ESC:
            print("\nExiting...")
            break

        elif key == KEY_SPACE:
            if not is_unknown and current_letter != "UNKNOWN":
                sentence += current_letter
                print(f"  Added '{current_letter}' -> Sentence: \"{sentence}\"")

        elif key == KEY_BACKSPACE:
            if sentence:
                removed = sentence[-1]
                sentence = sentence[:-1]
                print(f"  Deleted '{removed}' -> Sentence: \"{sentence}\"")

        elif key == KEY_ENTER:
            if sentence:    
                completed_sentences.append(sentence)
                print(f"\n  >> Completed: \"{sentence}\"\n")
                sentence = ""

    cap.release()
    cv2.destroyAllWindows()

    if completed_sentences:
        print("\n" + "=" * 60)
        print("Completed Sentences:")
        for i, s in enumerate(completed_sentences, 1):
            print(f"  {i}. {s}")

    print("\nDone.")


if __name__ == "__main__":
    main()
