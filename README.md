# ASL Alphabet Recognition — Web Application Architecture

Aplikasi web real-time untuk pengenalan Bahasa Isyarat (ASL Alphabet A-Z) berbasis **FastAPI + WebSocket + HTML5 Browser Webcam** dengan antarmuka futuristik minimalis.

---

## 🎨 UI & Aesthetics

- **Theme**: Dark Apple-like minimalism & AI Laboratory Aesthetic.
- **Color Palette**:
  - Background: `#08080C`
  - Surface: `#111116`
  - Primary: `#8B5CF6`
  - Secondary: `#EC4899`
  - Text: `#F5F5F5`
  - Muted: `#71717A`

---

## 📁 Data & File Structure

```
jst/
├── model/
│   ├── model.keras             # (Opsional) EfficientNet model weights
│   ├── labels.json              # (Opsional) Label mapping (A-Z)
│   ├── train_images.py          # Script training EfficientNetB0
│   └── predict.py               # ImageClassifier wrapper
├── backend/
│   ├── main.py                  # Server FastAPI & WebSocket /ws/predict
│   ├── predictor.py             # DummyPredictor & EfficientNetPredictor
│   └── config.py                # Konfigurasi aplikasi & trigger word
├── web/
│   ├── index.html               # Landing page (ASL. SIGN. RECOGNIZE. CONNECT.)
│   ├── studio.html              # Studio page (Live video & AI Prediction)
│   ├── secret.html              # Secret unlocked page (Easter Egg)
│   ├── css/
│   │   ├── style.css            # Styling landing page
│   │   ├── studio.css           # Styling studio page & components
│   │   └── secret.css           # Styling secret page
│   └── js/
│       ├── camera.js            # Module HTML5 navigator.mediaDevices camera
│       ├── websocket.js         # Module Client WebSocket dengan auto-reconnect
│       ├── sentence.js          # Module Sentence Builder & Trigger Detector
│       └── studio.js            # Orchestrator UI & prediction loop (8 FPS)
├── requirements.txt
└── README.md
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Install Dependensi

```bash
pip install -r requirements.txt
```

### 2. Jalankan Server FastAPI

Jalankan perintah berikut dari root project (`jst/`):

```bash
python backend/main.py
```

Atau menggunakan uvicorn langsung:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 3. Buka Aplikasi di Browser

Buka URL berikut di Google Chrome, Edge, atau Safari:

```text
http://localhost:8000
```

- Halaman **Landing Page** (`/`): Klik **`[ START CAMERA ]`** untuk masuk ke Studio.
- Halaman **Studio** (`/studio`): Browser akan meminta izin kamera. Setelah diizinkan, video feed webcam akan langsung tampil di `<video>` element.
- **Demo Mode**: Karena `model/model.keras` belum dilatih, server secara otomatis menggunakan `DummyPredictor` (mengirim siklus huruf A-Z dengan confidence 70-99%).
- **Sentence Builder**: Gunakan tombol atau shortcut keyboard:
  - `SPACE`: Tambah huruf terdeteksi saat ini ke kalimat.
  - `BACKSPACE`: Hapus huruf terakhir.
  - `ENTER`: Selesaikan kalimat & cek trigger word.
- **Special Mode**: Ketik huruf hingga membentuk kata **`HENGKY`**, lalu tekan `ENTER` (atau tambah huruf `Y`). Aplikasi akan membuka halaman rahasia `/secret` (*"✨ SECRET UNLOCKED ✨"*).

---

## 🧠 Model Fallback System

Backend menggunakan arsitektur predictor decoupled:
1. Jika `model/model.keras` **belum ada**: Server menggunakan `DummyPredictor`.
2. Jika `model/model.keras` **sudah dilatih**: Server otomatis beralih ke `EfficientNetPredictor` tanpa perlu mengubah kode frontend.
