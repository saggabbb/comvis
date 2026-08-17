"""
backend/main.py
================

FastAPI application for SignVision ASL Alphabet Recognition.

Endpoints:
    GET  /              -> Landing page
    GET  /studio        -> Studio page
    GET  /secret        -> Secret page
    GET  /api/status    -> System status
    WS   /ws/predict    -> Real-time prediction WebSocket

Usage:
    python backend/main.py

Then open:
    http://localhost:8000
"""


# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import time

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import uvicorn


# ============================================================
# PROJECT PATH
# ============================================================

BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ROOT_DIR = os.path.abspath(
    os.path.join(BACKEND_DIR, "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ============================================================
# CONFIG
# ============================================================

from backend.config import (
    HOST,
    PORT,
    WS_PREDICT_PATH,
    MAX_FPS,
    MODEL_PATH,
    TRIGGER_WORD,
    BUFFER_SIZE,
    MIN_BUFFER_SIZE,
    DEBUG_LOGGING,
)

from backend.temporal_predictor import TemporalPredictor


# ============================================================
# PREDICTOR
# ============================================================

from backend.predictor import (
    get_predictor,
    DummyPredictor,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SignVision ASL Recognition",
    description="ASL Alphabet Recognition using EfficientNetB0",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# WEB DIRECTORY
# ============================================================

WEB_DIR = os.path.join(
    ROOT_DIR,
    "web"
)

os.makedirs(
    WEB_DIR,
    exist_ok=True
)


# ============================================================
# STATIC FILES
# ============================================================

# /static
app.mount(
    "/static",
    StaticFiles(directory=WEB_DIR),
    name="static"
)


# /css
CSS_DIR = os.path.join(
    WEB_DIR,
    "css"
)

if os.path.isdir(CSS_DIR):

    app.mount(
        "/css",
        StaticFiles(directory=CSS_DIR),
        name="css"
    )


# /js
JS_DIR = os.path.join(
    WEB_DIR,
    "js"
)

if os.path.isdir(JS_DIR):

    app.mount(
        "/js",
        StaticFiles(directory=JS_DIR),
        name="js"
    )


# ============================================================
# LOAD PREDICTOR
# ============================================================

print("=" * 60)
print("INITIALIZING SIGNVISION")
print("=" * 60)

predictor = get_predictor()

print(
    f"[Server] Predictor: "
    f"{predictor.__class__.__name__}"
)

print("=" * 60)


# ============================================================
# LANDING PAGE
# ============================================================

@app.get("/")
@app.get("/index.html")
async def serve_index():
    """
    Serve landing page.
    """

    index_path = os.path.join(
        WEB_DIR,
        "index.html"
    )

    if not os.path.exists(index_path):

        return {
            "message": "SignVision backend is running.",
            "error": "index.html not found"
        }

    return FileResponse(
        index_path
    )


# ============================================================
# STUDIO PAGE
# ============================================================

@app.get("/studio")
@app.get("/studio.html")
async def serve_studio():
    """
    Serve studio page.
    """

    studio_path = os.path.join(
        WEB_DIR,
        "studio.html"
    )

    if not os.path.exists(studio_path):

        return {
            "message": "Studio page not found."
        }

    return FileResponse(
        studio_path
    )


# ============================================================
# SECRET PAGE
# ============================================================

@app.get("/secret")
@app.get("/secret.html")
async def serve_secret():
    """
    Serve secret page.
    """

    secret_path = os.path.join(
        WEB_DIR,
        "secret.html"
    )

    if not os.path.exists(secret_path):

        return {
            "message": "Secret page not found."
        }

    return FileResponse(
        secret_path
    )


# ============================================================
# API STATUS
# ============================================================

@app.get("/api/status")
async def get_status():
    """
    Return current system status.
    """

    return {
        "application": "SignVision",
        "predictor_type": predictor.__class__.__name__,
        "model_available": os.path.exists(
            MODEL_PATH
        ),
        "demo_mode": isinstance(
            predictor,
            DummyPredictor
        ),
        "trigger_word": TRIGGER_WORD,
        "max_fps": MAX_FPS,
        "buffer_size": BUFFER_SIZE,
        "min_buffer_size": MIN_BUFFER_SIZE,
    }


# ============================================================
# WEBSOCKET PREDICTION
# ============================================================

@app.websocket(WS_PREDICT_PATH)
async def websocket_predict(
    websocket: WebSocket
):
    """
    Real-time prediction WebSocket dengan Backend Temporal Smoothing.

    Client:
        Sends binary JPEG frames.

    Server:
        1. Model inference per-frame (raw prediction)
        2. Masukkan ke TemporalPredictor (sliding window buffer)
        3. Calculate weighted vote & stable prediction
        4. Return JSON stable prediction ke client.

    Example response:

    {
        "letter": "A",
        "confidence": 0.8942,
        "inference_ms": 15.2,
        "is_stable": true,
        "buffer_size": 15,
        "raw_letter": "A",
        "raw_confidence": 0.91
    }
    """

    await websocket.accept()

    print(
        "[WS] Client connected"
    )

    # Instansiasi TemporalPredictor per koneksi WebSocket (Session & Thread Isolated)
    temporal_smoother = TemporalPredictor(
        buffer_size=BUFFER_SIZE,
        min_buffer_size=MIN_BUFFER_SIZE,
        debug_logging=DEBUG_LOGGING,
    )

    # ========================================================
    # FPS LIMIT
    # ========================================================

    min_interval = 1.0 / MAX_FPS

    last_time = 0.0


    # ========================================================
    # RECEIVE FRAMES
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # Receive JPEG bytes
            # ------------------------------------------------

            frame_bytes = (
                await websocket.receive_bytes()
            )

            # ------------------------------------------------
            # Rate limiting
            # ------------------------------------------------

            now = time.time()

            if (
                now - last_time
                < min_interval
            ):
                continue

            last_time = now


            # ------------------------------------------------
            # 1. Per-frame Inference (Raw)
            # ------------------------------------------------

            raw_result = predictor.predict(
                frame_bytes
            )

            raw_letter = raw_result.get("letter")
            raw_confidence = raw_result.get("confidence", 0.0)
            inference_ms = raw_result.get("inference_ms", 0.0)
            hand_detected = raw_result.get("hand_detected", True)
            is_hand_roi = raw_result.get("is_hand_roi", True)
            roi = raw_result.get("roi")


            # ------------------------------------------------
            # 2. Hand State Handling & Temporal Smoothing
            # ------------------------------------------------
            # Jika tangan TIDAK terdeteksi oleh MediaPipe:
            # - Reset temporal buffer agar prediction lama tidak tertahan
            # - Jangan masukkan raw prediction ke TemporalPredictor
            # - Kirimkan state no-hand yang jelas ke WebSocket client
            if not hand_detected:
                temporal_smoother.reset()
                no_hand_response = {
                    "letter": None,
                    "confidence": 0.0,
                    "inference_ms": inference_ms,
                    "is_stable": False,
                    "buffer_size": 0,
                    "raw_letter": None,
                    "raw_confidence": 0.0,
                    "hand_detected": False,
                    "is_hand_roi": False,
                    "roi": roi,
                }
                await websocket.send_json(
                    no_hand_response
                )
                continue


            # ------------------------------------------------
            # 3. Temporal Smoothing & Weighted Voting (Backend)
            # ------------------------------------------------
            # Hanya diproses ketika ROI benar-benar berasal dari tangan

            stable_result = temporal_smoother.add_prediction(
                raw_letter=raw_letter,
                raw_confidence=raw_confidence,
                inference_ms=inference_ms,
            )

            stable_result["hand_detected"] = True
            stable_result["is_hand_roi"] = is_hand_roi
            stable_result["roi"] = roi


            # ------------------------------------------------
            # 4. Send Stable Prediction
            # ------------------------------------------------

            await websocket.send_json(
                stable_result
            )



    # ========================================================
    # CLIENT DISCONNECTED
    # ========================================================

    except WebSocketDisconnect:

        print(
            "[WS] Client disconnected"
        )


    # ========================================================
    # OTHER ERROR
    # ========================================================

    except Exception as e:

        print(
            f"[WS] Error: {e}"
        )

        try:

            await websocket.close()

        except Exception:

            pass

    finally:

        # Reset buffer saat koneksi terputus
        temporal_smoother.reset()


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SIGNVISION ASL RECOGNITION SERVER")
    print("=" * 60)

    print(
        f"Server : http://localhost:{PORT}"
    )

    print(
        f"WebSocket : ws://localhost:{PORT}"
        f"{WS_PREDICT_PATH}"
    )

    print(
        f"Model : {MODEL_PATH}"
    )

    print("=" * 60)

    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )