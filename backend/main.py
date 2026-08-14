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
)


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
    }


# ============================================================
# WEBSOCKET PREDICTION
# ============================================================

@app.websocket(WS_PREDICT_PATH)
async def websocket_predict(
    websocket: WebSocket
):
    """
    Real-time prediction WebSocket.

    Client:
        Sends binary JPEG frames.

    Server:
        Returns JSON prediction.

    Example response:

    {
        "letter": "A",
        "confidence": 0.9997,
        "inference_ms": 140.2
    }
    """

    await websocket.accept()

    print(
        "[WS] Client connected"
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
            # Prediction
            # ------------------------------------------------

            result = predictor.predict(
                frame_bytes
            )


            # ------------------------------------------------
            # Send prediction
            # ------------------------------------------------

            await websocket.send_json(
                result
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