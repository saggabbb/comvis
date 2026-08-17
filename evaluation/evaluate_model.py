"""
evaluation/evaluate_model.py
============================
Evaluation pipeline for the ASL Alphabet Image Classifier.

Modes:
  --mode smoke : Fast check on test_images/ (A-F only)
  --mode full  : Comprehensive check on full dataset (e.g. dataset/)

Outputs generated in evaluation/:
  - evaluation_results.json
  - classification_report.txt
  - confusion_matrix.png
"""

import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Ensure root directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.predictor import EfficientNetPredictor
import cv2

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EVAL_DIR)

def load_smoke_images(smoke_dir: str):
    """Load images from test_images/ (flat structure: A.jpg, B.jpg)"""
    images = []
    labels = []
    
    if not os.path.exists(smoke_dir):
        print(f"Directory not found: {smoke_dir}")
        return images, labels

    for filename in sorted(os.listdir(smoke_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # label is just the filename without extension
            label = os.path.splitext(filename)[0].upper()
            filepath = os.path.join(smoke_dir, filename)
            images.append(filepath)
            labels.append(label)
            
    return images, labels

def load_full_dataset(dataset_dir: str):
    """Load images from a standard dataset format (subfolders for each class)"""
    images = []
    labels = []
    
    if not os.path.exists(dataset_dir):
        print(f"Directory not found: {dataset_dir}")
        return images, labels
        
    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        label = class_name.upper() if len(class_name) == 1 else class_name.lower()
        
        for filename in sorted(os.listdir(class_path)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                filepath = os.path.join(class_path, filename)
                images.append(filepath)
                labels.append(label)
                
    return images, labels

def main():
    parser = argparse.ArgumentParser(description="Evaluate ASL Model")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke", help="Evaluation mode")
    parser.add_argument("--dataset", default=os.path.join(ROOT_DIR, "dataset"), help="Path to full dataset (for full mode)")
    parser.add_argument("--smoke_dir", default=os.path.join(ROOT_DIR, "test_images"), help="Path to smoke test images")
    args = parser.parse_args()

    print("=" * 60)
    print(f"STARTING EVALUATION - MODE: {args.mode.upper()}")
    print("=" * 60)

    # 1. Load Data
    if args.mode == "smoke":
        image_paths, true_labels = load_smoke_images(args.smoke_dir)
        print(f"Loaded {len(image_paths)} images from {args.smoke_dir}")
    else:
        image_paths, true_labels = load_full_dataset(args.dataset)
        print(f"Loaded {len(image_paths)} images from {args.dataset}")

    if not image_paths:
        print("No images found. Exiting.")
        sys.exit(1)

    # 2. Initialize Predictor
    try:
        predictor = EfficientNetPredictor()
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # 3. Run Inference
    predicted_labels = []
    confidences = []
    inference_times = []
    
    print("\nRunning inference...")
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read {path}")
            predicted_labels.append("?")
            confidences.append(0.0)
            continue
            
        # Convert to jpeg bytes to simulate websocket payload
        success, encoded = cv2.imencode(".jpg", img)
        if not success:
            predicted_labels.append("?")
            confidences.append(0.0)
            continue
            
        t0 = time.time()
        result = predictor.predict(encoded.tobytes())
        t1 = time.time()
        
        predicted_labels.append(result["letter"] if result["letter"] else "?")
        confidences.append(result["confidence"])
        inference_times.append((t1 - t0) * 1000)  # ms
        
        if (idx + 1) % 100 == 0 or (idx + 1) == len(image_paths):
            print(f"Processed {idx + 1}/{len(image_paths)}")

    # 4. Filter out unreadable images
    valid_indices = [i for i, p in enumerate(predicted_labels) if p != "?"]
    y_true = [true_labels[i] for i in valid_indices]
    y_pred = [predicted_labels[i] for i in valid_indices]
    valid_confidences = [confidences[i] for i in valid_indices]
    
    if not y_true:
        print("No valid predictions made. Exiting.")
        sys.exit(1)

    # 5. Calculate Metrics
    classes = sorted(list(set(y_true) | set(y_pred)))
    
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    avg_conf = np.mean(valid_confidences)
    avg_inf_time = np.mean(inference_times)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy         : {acc:.4f}")
    print(f"Precision (Macro): {prec_macro:.4f}")
    print(f"Recall (Macro)   : {rec_macro:.4f}")
    print(f"F1 Score (Macro) : {f1_macro:.4f}")
    print(f"Avg Confidence   : {avg_conf:.4f}")
    print(f"Avg Inference    : {avg_inf_time:.2f} ms")

    # 6. Generate Reports
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    # Classification Report
    report_text = classification_report(y_true, y_pred, labels=classes, zero_division=0)
    with open(os.path.join(EVAL_DIR, "classification_report.txt"), "w") as f:
        f.write(report_text)
    print("\nSaved classification_report.txt")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"))
    plt.close()
    print("Saved confusion_matrix.png")
    
    # JSON Results
    results = {
        "mode": args.mode,
        "total_samples": len(y_true),
        "metrics": {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro
        },
        "performance": {
            "avg_confidence": avg_conf,
            "avg_inference_ms": avg_inf_time
        }
    }
    with open(os.path.join(EVAL_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print("Saved evaluation_results.json")

if __name__ == "__main__":
    main()
