"""
backend/hand_detector.py
========================

Isolated MediaPipe Hand ROI Detector for SignVision ASL Recognition.

Responsibilities:
- Detect hands in an RGB frame using MediaPipe (supports both MediaPipe 1.0+ Tasks API and legacy Solutions API)
- Calculate pixel bounding box (min/max X and Y) across all hand landmarks
- Apply 15% expansion margin (ROI_MARGIN_RATIO = 0.15)
- Clamp bounding box to frame boundaries
- Support multiple hands by calculating a unified bounding box
- Provide static ROI fallback when no hand is detected
- Handle missing MediaPipe gracefully with informative error logging

NOTE: MediaPipe is used ONLY for hand ROI localization.
Classification is strictly performed by EfficientNetB0.
"""

import os
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger("SignVision.HandDetector")

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    _MEDIAPIPE_AVAILABLE = False
    logger.error("[MediaPipe] NOT AVAILABLE. Please install mediapipe: pip install mediapipe")


class HandDetector:
    """
    Isolated MediaPipe Hand Detector for Backend Inference.
    Supports both MediaPipe 1.0+ Tasks API and legacy solutions.hands API.
    """

    def __init__(
        self,
        margin_ratio: float = 0.15,
        static_roi_config: Optional[dict] = None,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
        max_num_hands: int = 2,
        task_model_path: Optional[str] = None,
    ):
        self.margin_ratio = margin_ratio
        self.static_roi_config = static_roi_config or {
            "x": 0.30,
            "y": 0.10,
            "w": 0.40,
            "h": 0.75,
        }
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_hands = max_num_hands

        # Resolve task model path for MediaPipe 1.0+
        if task_model_path is None:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.abspath(os.path.join(backend_dir, ".."))
            task_model_path = os.path.join(root_dir, "model", "hand_landmarker.task")
        self.task_model_path = task_model_path

        self._mode = None  # "tasks" or "solutions"
        self._detector = None
        self._setup()

    def _setup(self):
        """Initialize MediaPipe Hand Landmarker."""
        if not _MEDIAPIPE_AVAILABLE or mp is None:
            logger.warning("[MediaPipe] NOT AVAILABLE - Running with static ROI fallback only.")
            return

        # 1. Try MediaPipe 1.0+ Tasks API
        try:
            if os.path.exists(self.task_model_path):
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=self.task_model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=self.max_num_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_hand_presence_confidence=self.min_tracking_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self._detector = vision.HandLandmarker.create_from_options(options)
                self._mode = "tasks"
                logger.info("[MediaPipe] Initialized via MediaPipe Tasks API.")
                return
        except Exception as e:
            logger.warning(f"[MediaPipe] Tasks API initialization failed: {e}")

        # 2. Try legacy solutions.hands API
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self._detector = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    model_complexity=1,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self._mode = "solutions"
                logger.info("[MediaPipe] Initialized via MediaPipe Solutions API.")
                return
        except Exception as e:
            logger.warning(f"[MediaPipe] Solutions API initialization failed: {e}")

        logger.warning("[MediaPipe] Detector could not be initialized - running with static ROI fallback only.")

    @property
    def is_available(self) -> bool:
        """Check if MediaPipe Hand detector is ready."""
        return self._detector is not None

    def _extract_landmarks_tasks(self, rgb_frame) -> List[Tuple[float, float]]:
        """Extract landmarks using Tasks API."""
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self._detector.detect(mp_image)
            points = []
            if result and result.hand_landmarks:
                frame_h, frame_w = rgb_frame.shape[:2]
                for hand in result.hand_landmarks:
                    for lm in hand:
                        points.append((lm.x * frame_w, lm.y * frame_h))
            return points
        except Exception as e:
            logger.debug(f"[MediaPipe Tasks] Detection error: {e}")
            return []

    def _extract_landmarks_solutions(self, rgb_frame) -> List[Tuple[float, float]]:
        """Extract landmarks using Solutions API."""
        try:
            results = self._detector.process(rgb_frame)
            points = []
            if results and results.multi_hand_landmarks:
                frame_h, frame_w = rgb_frame.shape[:2]
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        points.append((lm.x * frame_w, lm.y * frame_h))
            return points
        except Exception as e:
            logger.debug(f"[MediaPipe Solutions] Detection error: {e}")
            return []

    def get_hand_landmarks(self, rgb_frame) -> List[Tuple[float, float]]:
        """Get pixel landmark coordinates for all detected hands."""
        if not self.is_available or rgb_frame is None:
            return []
        if self._mode == "tasks":
            return self._extract_landmarks_tasks(rgb_frame)
        elif self._mode == "solutions":
            return self._extract_landmarks_solutions(rgb_frame)
        return []

    def compute_roi(
        self,
        rgb_frame,
    ) -> Tuple[Tuple[int, int, int, int], bool, bool]:
        """
        Detect hand and compute active ROI with 15% margin and boundary clamping.

        Returns:
            (roi, hand_detected, is_hand_roi)
            - roi: (x, y, w, h) tuple
            - hand_detected: True if MediaPipe detected a hand, False otherwise
            - is_hand_roi: True if ROI is from MediaPipe, False if static fallback
        """
        frame_h, frame_w = rgb_frame.shape[:2]

        landmarks = self.get_hand_landmarks(rgb_frame)
        if landmarks:
            xs = [p[0] for p in landmarks]
            ys = [p[1] for p in landmarks]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            box_w = max_x - min_x
            box_h = max_y - min_y

            margin_x = box_w * self.margin_ratio
            margin_y = box_h * self.margin_ratio

            x1 = int(min_x - margin_x)
            y1 = int(min_y - margin_y)
            x2 = int(max_x + margin_x)
            y2 = int(max_y + margin_y)

            # Clamp to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_w, x2)
            y2 = min(frame_h, y2)

            w = x2 - x1
            h = y2 - y1

            if w > 0 and h > 0:
                return (x1, y1, w, h), True, True

        # Fallback to static center ROI
        static_roi = self.get_static_roi(frame_w, frame_h)
        return static_roi, False, False

    def get_static_roi(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """
        Compute static fallback ROI based on frame dimensions.
        """
        x = int(frame_w * self.static_roi_config.get("x", 0.30))
        y = int(frame_h * self.static_roi_config.get("y", 0.10))
        w = int(frame_w * self.static_roi_config.get("w", 0.40))
        h = int(frame_h * self.static_roi_config.get("h", 0.75))

        # Clamp
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))

        return (x, y, w, h)

    def close(self):
        """Clean up MediaPipe resources."""
        if self._detector is not None:
            try:
                if hasattr(self._detector, "close"):
                    self._detector.close()
            except Exception:
                pass
            self._detector = None
