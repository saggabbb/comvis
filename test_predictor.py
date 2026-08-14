"""
STEP 23
Test EfficientNetPredictor

Tujuan:
Memastikan model final dan class_names dapat
dimuat oleh backend sebelum digunakan oleh kamera.
"""

from backend.predictor import EfficientNetPredictor


print("=" * 60)
print("STEP 23 - TEST EFFICIENTNET PREDICTOR")
print("=" * 60)


try:

    # Membuat predictor
    predictor = EfficientNetPredictor()

    print("\n[OK] EfficientNetPredictor berhasil dibuat.")

    print("\nModel berhasil dimuat.")
    print("Jumlah kelas:", len(predictor.labels_map))

    print("\nClass mapping:")

    for index, label in predictor.labels_map.items():
        print(f"{index:2d} -> {label}")

    print("\n" + "=" * 60)
    print("TEST BERHASIL")
    print("=" * 60)


except Exception as e:

    print("\n" + "=" * 60)
    print("TEST GAGAL")
    print("=" * 60)

    print("\nError:")
    print(e)

    raise