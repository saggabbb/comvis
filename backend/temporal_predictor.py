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
4. Menyediakan method reset() untuk membersihkan state per koneksi WebSocket.
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

    def __init__(self, buffer_size: int = 15, min_buffer_size: int = 5, debug_logging: bool = True):
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.debug_logging = debug_logging
        self.buffer = deque(maxlen=self.buffer_size)

    def add_prediction(self, raw_letter: str, raw_confidence: float, inference_ms: float = 0.0) -> dict:
        """
        Menambahkan hasil raw prediction frame baru ke sliding window buffer,
        lalu mengembalikan dictionary hasil stable prediction.
        """
        # Masukkan frame baru ke buffer (sliding window otomatis mengeluarkan item tertua jika len > maxlen)
        self.buffer.append({
            "letter": raw_letter,
            "confidence": float(raw_confidence),
            "inference_ms": float(inference_ms)
        })

        current_size = len(self.buffer)

        # ----------------------------------------------------
        # CEK MINIMUM BUFFER SIZE
        # ----------------------------------------------------
        if current_size < self.min_buffer_size:
            if self.debug_logging:
                logging.info(
                    f"[Prediction] Raw: {raw_letter} ({raw_confidence:.2f}) | Buffer: {current_size}/{self.buffer_size} (Warming up...)"
                )
            return {
                "letter": None,
                "confidence": 0.0,
                "inference_ms": round(inference_ms, 2),
                "is_stable": False,
                "buffer_size": current_size,
                "raw_letter": raw_letter,
                "raw_confidence": round(raw_confidence, 4)
            }

        # ----------------------------------------------------
        # WEIGHTED VOTING CALCULATION
        # ----------------------------------------------------
        weighted_scores = {}
        total_confidence = 0.0

        for item in self.buffer:
            cls = item["letter"]
            conf = item["confidence"]

            weighted_scores[cls] = weighted_scores.get(cls, 0.0) + conf
            total_confidence += conf

        # Cari kelas dengan total weighted score terbesar
        stable_letter = max(weighted_scores.items(), key=lambda x: x[1])[0]
        stable_score = weighted_scores[stable_letter]

        # Stable confidence dihitung secara rasional: weighted_score / total_confidence
        stable_confidence = stable_score / total_confidence if total_confidence > 0 else 0.0
        stable_confidence = min(max(stable_confidence, 0.0), 1.0)  # Bound to [0.0, 1.0]

        if self.debug_logging:
            logging.info(
                f"[Prediction] Raw: {raw_letter} ({raw_confidence:.2f}) | Buffer: {current_size}/{self.buffer_size} | Stable: {stable_letter} ({stable_confidence:.2f})"
            )

        return {
            "letter": stable_letter,
            "confidence": round(stable_confidence, 4),
            "inference_ms": round(inference_ms, 2),
            "is_stable": True,
            "buffer_size": current_size,
            "raw_letter": raw_letter,
            "raw_confidence": round(raw_confidence, 4)
        }

    def reset(self):
        """Mengosongkan buffer (misal saat client disconnect/reconnect)."""
        self.buffer.clear()
