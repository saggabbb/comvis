# ASL Alphabet Recognition — EfficientNetB0

Aplikasi Python untuk mendeteksi dan mengklasifikasikan bahasa isyarat alfabet (A-Z) secara real-time menggunakan webcam dan CNN berbasis EfficientNetB0 Transfer Learning.

## Arsitektur

```
Dataset Images (200x200 RGB)
    |
Image Preprocessing (resize 224x224, normalize)
    |
Data Augmentation (flip, rotation, zoom, brightness)
    |
EfficientNetB0 (ImageNet pretrained, frozen)
    |
GlobalAveragePooling2D
    |
Dropout(0.3)
    |
Dense(26, softmax)
    |
Real-time Webcam Prediction + Sentence Builder
```

## Stack Teknologi

| Library | Fungsi |
|---------|--------|
| TensorFlow / Keras | Model CNN + Transfer Learning |
| EfficientNetB0 | Backbone (ImageNet weights) |
| OpenCV | Webcam capture + visualisasi |
| NumPy | Manipulasi array |
| Scikit-Learn | Train/test split + evaluasi |
| Matplotlib | Plot training history |
| Pillow | Loading gambar dataset |

## Struktur Proyek

```
project/
├── dataset/
│   ├── A/ ... Z/              # 800 gambar per huruf
├── model/
│   ├── train_images.py        # Script training EfficientNetB0
│   ├── predict.py             # Wrapper prediksi
│   ├── model.keras            # Model terlatih
│   ├── labels.json            # Nama label kelas
│   ├── training_history.png   # Grafik accuracy + loss
│   ├── classification_report.txt
│   └── confusion_matrix.csv
├── realtime.py                # Deteksi real-time + sentence builder
├── helpers.py                 # Fungsi utilitas
├── requirements.txt
└── README.md
```

## Cara Menggunakan

### 1. Install Dependensi

```bash
pip install -r requirements.txt
```

### 2. Siapkan Dataset

Pastikan folder `dataset/` berisi subfolder A-Z, masing-masing berisi gambar tangan yang menunjukkan huruf tersebut.

```
dataset/
├── A/   (800 gambar)
├── B/   (800 gambar)
├── ...
└── Z/   (800 gambar)
```

Untuk mengganti dataset:
- Hapus isi folder A-Z lama
- Isi dengan gambar baru
- Pastikan semua gambar berformat JPG/PNG
- Jalankan ulang training

### 3. Training Model

```bash
python model/train_images.py --dataset dataset --epochs 30 --batch 32
```

**Output:**
- `model/model.keras` — Model terlatih
- `model/labels.json` — Label kelas A-Z
- `model/training_history.png` — Grafik accuracy dan loss
- `model/classification_report.txt` — Laporan per kelas
- `model/confusion_matrix.csv` — Confusion matrix

**Parameter opsional:**
- `--epochs N` — Jumlah epoch maksimum (default: 30)
- `--batch N` — Batch size (default: 32)
- `--img-size N` — Ukuran gambar (default: 224)

### 4. Real-time Detection

```bash
python realtime.py
```

### Tampilan Real-time
- **Bounding Box (ROI)** — Area hijau untuk meletakkan tangan
- **Prediction** — Huruf yang dikenali
- **Confidence** — Persentase keyakinan model
- **FPS** — Frame per detik
- **Sentence Builder** — Bar pembentuk kalimat

### Keyboard Controls

| Tombol    | Fungsi |
|-----------|--------|
| SPACE     | Simpan huruf ke kalimat |
| BACKSPACE | Hapus huruf terakhir |
| ENTER     | Selesaikan kalimat |
| ESC       | Keluar |

### Special Mode

Ketika sentence builder menghasilkan kata **"CLEO"**, otomatis:
1. Animasi visual di layar
2. Loading indicator
3. Membuka website di browser default

### Contoh Penggunaan

```
H → E → L → L → O  →  HELLO
(tekan SPACE untuk setiap huruf, ENTER untuk menyelesaikan)
```

## Catatan Teknis

- Model menggunakan **Transfer Learning** dari EfficientNetB0 pretrained pada ImageNet
- Backbone EfficientNetB0 di-**freeze** (tidak dilatih ulang)
- Hanya classification head (GAP + Dense) yang dilatih
- Input gambar di-resize ke **224×224** dan dinormalisasi [0, 1]
- Data augmentation diterapkan **hanya saat training**
- Confidence threshold: di bawah 60% ditampilkan sebagai "UNKNOWN"
- Prediction smoothing: majority voting dari 10 frame terakhir
