# SignVision — ASL Studio

Aplikasi web real-time untuk pengenalan American Sign Language (ASL Alphabet A-Z) berbasis **FastAPI + WebSocket + HTML5 Browser Webcam** dengan antarmuka developer workstation yang clean dan modern.

---

## 🎨 Architecture & UI

- **Theme**: Clean Developer Workstation (Dark Mode)
- **Tech Stack**: 
  - Frontend: Vanilla JS, MediaPipe Hands (Client-side tracking)
  - Backend: FastAPI, WebSockets
  - Model: EfficientNetB0 (TensorFlow/Keras)
- **Pipeline**:
  1. Frontend captures 8 FPS webcam stream via WebSocket.
  2. Backend performs ROI Extraction via MediaPipe (25% margin padding).
  3. Preprocessed ROI (float32 [0-255]) is passed to EfficientNetB0 (Internal Rescaling).
  4. TemporalPredictor smooths predictions over a sliding window buffer.
  5. Result is broadcast back to frontend.

---

## 📁 Project Structure

```
comvis/
├── backend/
│   ├── main.py                  # FastAPI server & WebSocket /ws/predict
│   ├── predictor.py             # EfficientNetPredictor & DummyPredictor wrapper
│   ├── hand_detector.py         # MediaPipe ROI extractor
│   ├── temporal_predictor.py    # Sliding window & weighted voting
│   └── config.py                # Core configuration (confidence, margins, FPS)
├── model/
│   ├── SignVision_EfficientNetB0_final.keras  # Pretrained model
│   ├── class_names.json         # Label mapping (A-Z + del, nothing, space)
│   └── train_images.py          # Training script (Transfer Learning)
├── web/
│   ├── index.html               # Landing page
│   ├── studio.html              # Main Studio App
│   ├── secret.html              # Secret unlocked page
│   ├── css/
│       ├── style.css            # Landing page styles
│       └── studio.css           # Studio styles
│   └── js/
│       ├── camera.js            # HTML5 camera module
│       ├── websocket.js         # WebSocket auto-reconnect module
│       ├── sentence.js          # Sentence builder module
│       └── studio.js            # Studio orchestrator & UI logic
├── evaluation/
│   └── evaluate_model.py        # Pipeline for offline model evaluation
├── scripts/
│   ├── check_dataset.py         # Integrity checker for training images
│   └── helpers.py               # Shared utility functions
├── tests/                       # Automated test scripts
├── dataset/                     # Training images (Not in repo)
├── test_images/                 # Smoke test images (A.jpg - F.jpg)
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python backend/main.py
```
*Note: The server will start on `http://localhost:8000`.*

### 3. Open the Studio

Open `http://localhost:8000/studio` in Chrome, Edge, or Safari.
- Ensure camera permissions are granted.
- **Demo Mode**: If the model is not trained yet (`SignVision_EfficientNetB0_final.keras` is missing), the backend automatically falls back to `DummyPredictor` for frontend testing.

---

## 🧠 Model Training

If you want to train the model from scratch:
1. Place your dataset in the `dataset/` folder, with each class in its own subfolder.
2. Run the training script:
```bash
python model/train_images.py --dataset dataset --epochs 30 --batch 32
```
*Note: The EfficientNetB0 model contains an internal `Rescaling` layer. The input images must remain in `float32 [0-255]` range. The `helpers.py` preprocess function ensures this consistency.*

---

## 📊 Evaluation

To evaluate the model's performance on the dataset:
```bash
# Full evaluation
python evaluation/evaluate_model.py --mode full --dataset dataset

# Quick smoke test
python evaluation/evaluate_model.py --mode smoke
```
Metrics (Accuracy, Precision, Recall, F1) and the Confusion Matrix will be saved in the `evaluation/` directory.
