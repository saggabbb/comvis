"""
model/train_images.py
======================
Train an EfficientNetB0-based ASL alphabet image classifier using Transfer Learning.

Architecture:
    Input(224x224x3)  [float32, range 0–255 — model handles /255 internally]
    → Data Augmentation (RandomRotation, RandomZoom, RandomTranslation, RandomBrightness)
    → EfficientNetB0 (ImageNet pretrained, frozen backbone)
    → GlobalAveragePooling2D
    → Dropout(0.3)
    → Dense(29, softmax)  [A-Z + del + nothing + space]

Preprocessing Note:
    EfficientNetB0 in this model already contains an internal Rescaling layer
    (scale = 1/255). Input images must be float32 in range [0, 255].
    DO NOT divide by 255 before passing to the model.

Callbacks:
    - EarlyStopping (patience=5, restore_best_weights=True)
    - ReduceLROnPlateau (factor=0.5, patience=3)
    - ModelCheckpoint (save_best_only=True, monitor=val_accuracy)

Outputs:
    model/SignVision_EfficientNetB0_final.keras
    model/class_names.json
    model/training_history.png
    model/classification_report.txt
    model/confusion_matrix.csv

Usage:
    python model/train_images.py --dataset dataset --epochs 30 --batch 32
"""

import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.helpers import discover_class_names, save_labels

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = Path(__file__).resolve().parent
IMG_SIZE = 224


def collect_valid_image_paths(dataset_root: str):
    """Scan dataset folders and collect valid image paths with labels."""
    class_names = discover_class_names(dataset_root)
    image_paths = []
    labels = []

    print(f"\nScanning dataset: {dataset_root}")
    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_root, class_name)
        count = 0
        for file_name in sorted(os.listdir(class_path)):
            file_path = os.path.join(class_path, file_name)
            if not os.path.isfile(file_path):
                continue
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                continue
            image_paths.append(file_path)
            labels.append(label_idx)
            count += 1
        print(f"  {class_name}: {count} valid images")

    if not image_paths:
        raise RuntimeError(f"No valid image files found under {dataset_root}")

    return image_paths, labels, class_names


def load_images_to_array(image_paths, img_size):
    """Load and resize images into a numpy array."""
    images = []
    total = len(image_paths)
    for i, path in enumerate(image_paths):
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize((img_size, img_size), Image.BILINEAR)
            # NOTE: Do NOT divide by 255 here.
            # EfficientNetB0 has an internal Rescaling layer (scale=1/255).
            # Input must be float32 in range [0, 255].
            images.append(np.asarray(img, dtype=np.float32))
        if (i + 1) % 2000 == 0 or (i + 1) == total:
            print(f"  Loaded {i+1}/{total} images", flush=True)
    return np.stack(images, axis=0)


def build_efficientnet_model(input_shape, num_classes):
    """
    Build EfficientNetB0 transfer learning model.

    Architecture matches spec exactly:
        Input → Augmentation → EfficientNetB0(frozen) → GAP → Dropout → Dense(softmax)
    """
    inputs = keras.Input(shape=input_shape)

    # Data augmentation layers (only active during training)
    # NOTE: No RandomFlip(horizontal) — horizontal flip can confuse ASL handedness
    x = layers.RandomRotation(0.08)(inputs)         # ±~29 degrees
    x = layers.RandomZoom(0.10)(x)                  # ±10% zoom
    x = layers.RandomTranslation(0.05, 0.05)(x)     # ±5% shift
    # RandomBrightness value_range must match input range [0, 255]
    x = layers.RandomBrightness(factor=0.1, value_range=(0, 255))(x)

    # EfficientNetB0 backbone with ImageNet weights
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = False  # Freeze backbone

    # EfficientNetB0 has its own preprocessing built-in
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, out_dir):
    """Save training history plots."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history.get("accuracy", []), label="Train", linewidth=2)
    axes[0].plot(history.history.get("val_accuracy", []), label="Validation", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history.get("loss", []), label="Train", linewidth=2)
    axes[1].plot(history.history.get("val_loss", []), label="Validation", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    path = os.path.join(out_dir, "training_history.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training history to {path}")


def save_classification_artifacts(y_true, y_pred, class_names, output_dir):
    """Save classification report and confusion matrix."""
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(output_dir, "confusion_matrix.csv")
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(class_names))
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + list(row))
    print(f"Saved confusion matrix to {cm_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train EfficientNetB0 ASL classifier")
    parser.add_argument("--dataset", default="dataset", help="Dataset root with A-Z subfolders")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    args = parser.parse_args()

    print("=" * 60)
    print("ASL Alphabet - EfficientNetB0 Transfer Learning Trainer")
    print("=" * 60)

    # Collect dataset
    image_paths, labels, class_names = collect_valid_image_paths(args.dataset)
    num_classes = len(class_names)
    print(f"\nTotal: {len(image_paths)} images, {num_classes} classes")

    # Stratified split
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"Train: {len(train_paths)}, Validation: {len(val_paths)}")

    # Load images
    print("\nLoading training images...")
    train_images = load_images_to_array(train_paths, args.img_size)
    print("Loading validation images...")
    val_images = load_images_to_array(val_paths, args.img_size)

    train_labels = np.array(train_labels, dtype=np.int32)
    val_labels = np.array(val_labels, dtype=np.int32)

    # Build datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.shuffle(buffer_size=min(10000, len(train_images))).batch(args.batch).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.batch(args.batch).prefetch(tf.data.AUTOTUNE)

    # Build model
    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)
    input_shape = (args.img_size, args.img_size, 3)
    model = build_efficientnet_model(input_shape, num_classes)
    model.summary()

    # Callbacks
    model_path = str(MODEL_DIR / "SignVision_EfficientNetB0_final.keras")
    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=cbs, verbose=2
    )

    # Save model & labels
    model.save(model_path)
    print(f"\nSaved model to {model_path}")

    labels_path = str(MODEL_DIR / "class_names.json")
    save_labels(class_names, labels_path)
    print(f"Saved class names to {labels_path}")

    # Training history plot
    plot_history(history, str(MODEL_DIR))

    # Evaluation
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    probabilities = model.predict(val_images, verbose=0)
    val_pred = np.argmax(probabilities, axis=1)
    save_classification_artifacts(val_labels, val_pred, class_names, str(MODEL_DIR))

    accuracy = np.mean(val_pred == val_labels)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
