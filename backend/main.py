"""
backend/main.py
================
FastAPI application for ASL Alphabet Recognition.

Endpoints:
    GET  /          -> Landing page
    GET  /studio    -> Studio page (camera + prediction)
    GET  /secret    -> Secret page (trigger word easter egg)
    GET  /api/status -> System status JSON
    WS   /ws/predict -> Real-time prediction WebSocket

Usage:
    python backend/main.py
    -> Open http://localhost:8000
"""

import os
import sys
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Add parent directory to path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.config import HOST, PORT, WS_PREDICT_PATH, MAX_FPS, MODEL_PATH, TRIGGER_WORD
from backend.predictor import get_predictor

app = FastAPI(title="ASL Alphabet Recognition")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
WEB_DIR = os.path.join(ROOT_DIR, "web")
os.makedirs(WEB_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.mount("/css", StaticFiles(directory=os.path.join(WEB_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(WEB_DIR, "js")), name="js")

# Load predictor
predictor = get_predictor()


@app.get("/")
@app.get("/index.html")
async def serve_index():
    """Serve the landing page."""
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/studio")
@app.get("/studio.html")
async def serve_studio():
    """Serve the studio page."""
    return FileResponse(os.path.join(WEB_DIR, "studio.html"))


@app.get("/secret")
@app.get("/secret.html")
async def serve_secret():
    """Serve the secret page."""
    return FileResponse(os.path.join(WEB_DIR, "secret.html"))


@app.get("/api/status")
async def get_status():
    """Return system status."""
    return {
        "predictor_type": predictor.__class__.__name__,
        "model_available": os.path.exists(MODEL_PATH),
        "demo_mode": isinstance(predictor, DummyPredictor),
        "trigger_word": TRIGGER_WORD,
    }


@app.websocket(WS_PREDICT_PATH)
async def websocket_predict(websocket: WebSocket):
    """
    WebSocket endpoint for real-time prediction.
    Receives binary JPEG frames, returns JSON predictions.
    Rate-limited to MAX_FPS.
    """
    await websocket.accept()
    print("[WS] Client connected")

    min_interval = 1.0 / MAX_FPS
    last_time = 0.0

    try:
        while True:
            frame_bytes = await websocket.receive_bytes()

            now = time.time()
            if now - last_time < min_interval:
                continue

            last_time = now
            result = predictor.predict(frame_bytes)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    print("=" * 50)
    print("ASL Alphabet Recognition Server")
    print(f"Open: http://localhost:{PORT}")
    print("=" * 50)
    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )
