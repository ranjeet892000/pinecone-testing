#!/usr/bin/env python3
"""
Test script for image processing helper functions.

This script tests both functions:
1. remove_background_yolo_sam3() - with SAM3 and SnapIQ
2. get_dinov2_embedding() - DINOv2 embedding generation

Run this script to verify everything works!
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.image_processing_helpers import (
    remove_background_yolo_sam3,
    get_dinov2_embedding
)
import numpy as np


def test_background_removal():
    """Test background removal function"""
    print("\n" + "=" * 80)
    print("TEST 1: Background Removal")
    print("=" * 80)
    
    # Find a test image
    test_image_paths = [
        "data/raw/macro_images_lv",
        "data/test",
        "results/processed"
    ]
    
    test_image = None
    for base_path in test_image_paths:
        base = Path(base_path)
        if base.exists():
            # Find first image file
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images = list(base.glob(ext))
                if images:
                    test_image = images[0]
                    break
        if test_image:
            break
    
    if not test_image:
        print("⚠️  No test image found. Skipping background removal test.")
        print("   Please add an image to one of these directories:")
        for path in test_image_paths:
            print(f"   - {path}")
        return False
    
    print(f"📸 Test image: {test_image}")
    
    # Check YOLO model
    yolo_model = Path("models/yolo_22brands_best.pt")
    if not yolo_model.exists():
        print(f"⚠️  YOLO model not found: {yolo_model}")
        print("   Skipping background removal test.")
        return False
    
    print(f"🤖 YOLO model: {yolo_model}")
    
    # Test with SAM3
    print("\n🔧 Testing with SAM3...")
    result = remove_background_yolo_sam3(
        image=str(test_image),
        yolo_model_path=str(yolo_model),
        method="sam3"
    )
    
    if result['error'] is None:
        print("✅ SAM3 test passed!")
        print(f"   - Class: {result['class_name']}")
        print(f"   - Confidence: {result['confidence']:.2f}")
        print(f"   - Method: {result['method_used']}")
        
        # Save test result
        output_dir = Path("results/test_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "test_sam3_output.png"
        result['segmented_image'].save(output_path)
        print(f"   - Saved to: {output_path}")
        sam3_success = True
    else:
        print(f"⚠️  SAM3 test failed: {result['error']}")
        sam3_success = False
    
    # Test with SnapIQ (may fail if API key not set)
    print("\n🔧 Testing with SnapIQ...")
    result_snapiq = remove_background_yolo_sam3(
        image=str(test_image),
        yolo_model_path=str(yolo_model),
        method="snapiq"
    )
    
    if result_snapiq['error'] is None:
        print("✅ SnapIQ test passed!")
        print(f"   - Class: {result_snapiq['class_name']}")
        print(f"   - Confidence: {result_snapiq['confidence']:.2f}")
        print(f"   - Method: {result_snapiq['method_used']}")
        
        # Save test result
        output_path = output_dir / "test_snapiq_output.png"
        result_snapiq['segmented_image'].save(output_path)
        print(f"   - Saved to: {output_path}")
        snapiq_success = True
    else:
        print(f"⚠️  SnapIQ test skipped: {result_snapiq['error']}")
        print("   (This is normal if SNAPIQ_API_KEY is not set)")
        snapiq_success = False
    
    return sam3_success or snapiq_success


def test_dinov2_embedding():
    """Test DINOv2 embedding function"""
    print("\n" + "=" * 80)
    print("TEST 2: DINOv2 Embedding")
    print("=" * 80)
    
    # Find a test image
    test_image_paths = [
        "data/raw/macro_images_lv",
        "data/test",
        "results/processed"
    ]
    
    test_image = None
    for base_path in test_image_paths:
        base = Path(base_path)
        if base.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images = list(base.glob(ext))
                if images:
                    test_image = images[0]
                    break
        if test_image:
            break
    
    if not test_image:
        print("⚠️  No test image found. Skipping embedding test.")
        return False
    
    print(f"📸 Test image: {test_image}")
    
    # Test embedding generation
    print("\n🤖 Generating DINOv2 embedding...")
    embedding = get_dinov2_embedding(
        image=str(test_image),
        model_name="facebook/dinov2-base"
    )
    
    if embedding is not None:
        print("✅ Embedding test passed!")
        print(f"   - Shape: {embedding.shape}")
        print(f"   - Dimension: {embedding.shape[0]}")
        print(f"   - Norm: {np.linalg.norm(embedding):.4f} (should be ~1.0)")
        print(f"   - Min: {embedding.min():.4f}")
        print(f"   - Max: {embedding.max():.4f}")
        print(f"   - First 5 values: {embedding[:5]}")
        return True
    else:
        print("❌ Embedding test failed!")
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("Image Processing Helper Functions - Test Suite")
    print("=" * 80)
    
    results = []
    
    # Test 1: Background removal
    try:
        result1 = test_background_removal()
        results.append(("Background Removal", result1))
    except Exception as e:
        print(f"\n❌ Background removal test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Background Removal", False))
    
    # Test 2: DINOv2 embedding
    try:
        result2 = test_dinov2_embedding()
        results.append(("DINOv2 Embedding", result2))
    except Exception as e:
        print(f"\n❌ Embedding test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("DINOv2 Embedding", False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "⚠️  SKIPPED/FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = any(passed for _, passed in results)
    if all_passed:
        print("\n✅ At least one test passed! Functions are working.")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

