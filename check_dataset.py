from pathlib import Path
from PIL import Image

DATASET_DIR = Path("dataset")

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

bad_images = []
total = 0

print("Deep checking dataset...\n")

for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    print(f"Checking {class_dir.name}...")

    for path in sorted(class_dir.iterdir()):
        if path.suffix.lower() not in extensions:
            continue

        total += 1

        try:
            with Image.open(path) as img:
                img.load()              # paksa membaca seluruh pixel
                img = img.convert("RGB")
                img.load()              # load hasil convert

        except Exception as e:
            bad_images.append((path, repr(e)))
            print(f"\n❌ BAD IMAGE:")
            print(f"   {path}")
            print(f"   {e}\n")

print("\n" + "=" * 60)
print(f"Total checked : {total}")
print(f"Bad images    : {len(bad_images)}")

if bad_images:
    print("\nLIST BAD IMAGES:")
    for path, error in bad_images:
        print(f"{path}")
        print(f"  {error}")
else:
    print("\n✅ Semua gambar berhasil di-load dan convert RGB.")