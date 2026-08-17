"""
backend/temporal_predictor.py
==============================

Kelas pembantu untuk Temporal Smoothing & Prediction Buffer di Backend.

Fungsi:
1. Menyimpan sliding window hasil per-frame prediction (max_len = BUFFER_SIZE).
2. Melakukan weighted voting berdasarkan confidence setiap frame:
       weighted_score[class] = sum(confidence per frame untuk class tersebut)
       stable_class = class dengan weighted_score terbesar
       stable_confidence = weighted_score[stable_class] / total_confidence_buffer
3. Menjamin minimum buffer size (MIN_BUFFER_SIZE) sebelum menghasilkan stable prediction.
4. Menjamin minimum stability ratio (STABILITY_RATIO): minimal X% frame dalam buffer
   harus prediksi kelas yang sama sebelum dianggap stable.
5. Menandai prediksi sebagai UNCERTAIN jika confidence di bawah threshold.
6. Menyediakan method reset() untuk membersihkan state per koneksi WebSocket.
"""

from collections import deque
import logging

# Set log format
logging.basicConfig(level=logging.INFO, format="%(message)s")


class TemporalPredictor:
    """
    Menangani temporal smoothing dengan sliding window dan weighted voting.
    Instance dari kelas ini dibuat PER KONEKSI WebSocket agar aman dari
    penyampuran data antar-client (Session Isolated & Thread Safe).
    """

    def __init__(
        self,
        buffer_size: int = 15,
        min_buffer_size: int = 5,
        confidence_threshold: float = 0.60,
        stability_ratio: float = 0.60,
        debug_logging: bool = True,
    ):
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.confidence_threshold = confidence_threshold
        self.stability_ratio = stability_ratio
        self.debug_logging = debug_logging
        self.buffer = deque(maxlen=self.buffer_size)

    def add_prediction(
        self,
        raw_letter: str,
        raw_confidence: float,
        inference_ms: float = 0.0,
    ) -> dict:
        """
        Menambahkan hasil raw prediction frame baru ke sliding window buffer,
        lalu mengembalikan dictionary hasil stable prediction.

        Returns dict with keys:
            letter         - Stable predicted class (or None if warming up)
            confidence     - Weighted confidence of stable class
            inference_ms   - Inference time in milliseconds
            is_stable      - True if prediction is stable
            is_uncertain   - True if confidence below threshold (even if stable)
            buffer_size    - Current buffer size
            raw_letter     - Raw per-frame prediction
            raw_confidence - Raw per-frame confidence
        """
        # Masukkan frame baru ke buffer
        self.buffer.append({
            "letter": raw_letter,
            "confidence": float(raw_confidence),
            "inference_ms": float(inference_ms),
        })

        current_size = len(self.buffer)

        # ----------------------------------------------------
        # CEK MINIMUM BUFFER SIZE (Warming up)
        # ----------------------------------------------------
        if current_size < self.min_buffer_size:
            if self.debug_logging:
                logging.info(
                    f"[Prediction] Raw: {raw_letter} ({raw_confidence:.2f}) "
                    f"| Buffer: {current_size}/{self.buffer_size} (Warming up...)"
                )
            return {
                "letter": None,
                "confidence": 0.0,
                "inference_ms": round(inference_ms, 2),
                "is_stable": False,
                "is_uncertain": False,
                "buffer_size": current_size,
                "raw_letter": raw_letter,
                "raw_confidence": round(raw_confidence, 4),
            }

        # ----------------------------------------------------
        # WEIGHTED VOTING CALCULATION
        # ----------------------------------------------------
        weighted_scores = {}
        vote_counts = {}  # Frame count per class for stability ratio
        total_confidence = 0.0

        for item in self.buffer:
            cls = item["letter"]
            conf = item["confidence"]
            weighted_scores[cls] = weighted_scores.get(cls, 0.0) + conf
            vote_counts[cls] = vote_counts.get(cls, 0) + 1
            total_confidence += conf

        # Cari kelas dengan total weighted score terbesar
        stable_letter = max(weighted_scores.items(), key=lambda x: x[1])[0]
        stable_score = weighted_scores[stable_letter]

        # Stable confidence: weighted_score / total_confidence
        stable_confidence = (
            stable_score / total_confidence if total_confidence > 0 else 0.0
        )
        stable_confidence = min(max(stable_confidence, 0.0), 1.0)

        # ----------------------------------------------------
        # STABILITY RATIO CHECK
        # Frame dominance: what fraction of buffer voted for stable_letter?
        # ----------------------------------------------------
        frame_dominance = vote_counts[stable_letter] / current_size
        is_stable = frame_dominance >= self.stability_ratio

        # ----------------------------------------------------
        # CONFIDENCE THRESHOLD CHECK
        # Even if stable, flag as uncertain if confidence is too low
        # ----------------------------------------------------
        is_uncertain = stable_confidence < self.confidence_threshold

        if self.debug_logging:
            logging.info(
                f"[Prediction] Raw: {raw_letter} ({raw_confidence:.2f}) "
                f"| Buffer: {current_size}/{self.buffer_size} "
                f"| Stable: {stable_letter} ({stable_confidence:.2f}) "
                f"| Dominance: {frame_dominance:.0%} "
                f"| {'STABLE' if is_stable else 'UNSTABLE'} "
                f"| {'UNCERTAIN' if is_uncertain else 'OK'}"
            )

        return {
            "letter": stable_letter if is_stable else None,
            "confidence": round(stable_confidence, 4),
            "inference_ms": round(inference_ms, 2),
            "is_stable": is_stable,
            "is_uncertain": is_uncertain,
            "buffer_size": current_size,
            "raw_letter": raw_letter,
            "raw_confidence": round(raw_confidence, 4),
        }

    def reset(self):
        """Mengosongkan buffer (misal saat client disconnect/reconnect atau no-hand)."""
        self.buffer.clear()
