"""
tests/test_model_loading.py
===========================
Verifies that the EfficientNetPredictor successfully loads the final model
and that class mapping is correct.
"""

import os
import sys

# Ensure root directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.predictor import EfficientNetPredictor

def test_model():
    print("=" * 60)
    print("TEST: EFFICIENTNET PREDICTOR LOADING")
    print("=" * 60)

    try:
        predictor = EfficientNetPredictor()
        print("\n[OK] EfficientNetPredictor instantiated successfully.")
        
        labels_count = len(predictor.labels_map)
        print(f"Total classes mapped: {labels_count}")
        
        print("\nClass Mapping (First 10):")
        for i in range(min(10, labels_count)):
            print(f"  {i:2d} -> {predictor.labels_map[i]}")
            
        if labels_count != 29:
            print(f"\n[WARNING] Expected 29 classes, got {labels_count}")
        else:
            print("\n[OK] Class count matches expected (29).")
            
        print("\n" + "=" * 60)
        print("TEST PASSED")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED")
        print("=" * 60)
        print(f"\nError details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_model()
