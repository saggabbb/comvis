"""
STEP 25
Test Prediksi Kelas A-F

Struktur folder gambar yang diharapkan:

test_images/
├── A.jpg
├── B.jpg
├── C.jpg
├── D.jpg
├── E.jpg
└── F.jpg
"""

import os
import sys
import cv2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from backend.predictor import EfficientNetPredictor


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = "test_images"

TEST_CLASSES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F"
]


# ============================================================
# LOAD PREDICTOR
# ============================================================

print("=" * 70)
print("STEP 25 - TEST PREDIKSI KELAS A-F")
print("=" * 70)

predictor = EfficientNetPredictor()


# ============================================================
# HASIL
# ============================================================

results = []


# ============================================================
# TEST SETIAP KELAS
# ============================================================

for actual_label in TEST_CLASSES:

    image_path = os.path.join(
        TEST_DIR,
        f"{actual_label}.jpg"
    )

    print("\n" + "-" * 70)
    print(f"Testing: {actual_label}")
    print(f"File   : {image_path}")

    # --------------------------------------------------------
    # Cek file
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        print("ERROR: File tidak ditemukan.")

        results.append({
            "actual": actual_label,
            "predicted": "?",
            "confidence": 0.0,
            "inference_ms": 0.0
        })

        continue

    # --------------------------------------------------------
    # Baca gambar
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:

        print("ERROR: Gambar gagal dibaca.")

        results.append({
            "actual": actual_label,
            "predicted": "?",
            "confidence": 0.0,
            "inference_ms": 0.0
        })

        continue

    print(
        "Ukuran gambar:",
        image.shape
    )

    # --------------------------------------------------------
    # Encode menjadi JPEG
    # --------------------------------------------------------

    success, encoded_image = cv2.imencode(
        ".jpg",
        image
    )

    if not success:

        print("ERROR: Gagal encode gambar.")

        continue

    frame_bytes = encoded_image.tobytes()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    result = predictor.predict(
        frame_bytes
    )

    predicted_label = result["letter"]
    confidence = result["confidence"]
    inference_ms = result["inference_ms"]

    # --------------------------------------------------------
    # Cek benar / salah
    # --------------------------------------------------------

    correct = (
        actual_label == predicted_label
    )

    # --------------------------------------------------------
    # Print hasil
    # --------------------------------------------------------

    print("Actual      :", actual_label)
    print("Predicted   :", predicted_label)
    print(
        "Confidence  :",
        f"{confidence * 100:.2f}%"
    )
    print(
        "Inference   :",
        f"{inference_ms:.2f} ms"
    )

    if correct:
        print("Status      : ✅ BENAR")
    else:
        print("Status      : ❌ SALAH")

    # --------------------------------------------------------
    # Simpan hasil
    # --------------------------------------------------------

    results.append({
        "actual": actual_label,
        "predicted": predicted_label,
        "confidence": confidence,
        "inference_ms": inference_ms,
        "correct": correct
    })


# ============================================================
# RINGKASAN
# ============================================================

print("\n")
print("=" * 70)
print("RINGKASAN HASIL")
print("=" * 70)

correct_count = 0
tested_count = 0

for result in results:

    actual = result["actual"]
    predicted = result["predicted"]
    confidence = result["confidence"]

    if predicted != "?":
        tested_count += 1

        if actual == predicted:
            correct_count += 1

    print(
        f"{actual} -> "
        f"{predicted} | "
        f"{confidence * 100:.2f}%"
    )


# ============================================================
# ACCURACY
# ============================================================

if tested_count > 0:

    accuracy = (
        correct_count /
        tested_count
    ) * 100

    print("\n" + "-" * 70)

    print(
        f"Correct : {correct_count}/{tested_count}"
    )

    print(
        f"Accuracy A-F : {accuracy:.2f}%"
    )

else:

    print(
        "\nTidak ada gambar yang berhasil dites."
    )


print("=" * 70)
print("TEST SELESAI")
print("=" * 70)