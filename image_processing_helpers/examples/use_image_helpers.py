#!/usr/bin/env python3
"""
Simple Examples: Using Image Processing Helper Functions

This script demonstrates how to use the three main helper functions:
1. remove_background_with_sam3() - Remove background using YOLO + SAM3 (with optional text prompt)
2. remove_background_with_snapiq() - Remove background using YOLO + SnapIQ
3. get_dinov2_embedding() - Get DINOv2 embedding using Vectify for Pinecone queries

Run this script to see how the functions work!
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.image_processing_helpers import (
    remove_background_with_sam3,
    remove_background_with_snapiq,
    get_dinov2_embedding
)
import numpy as np


def example_sam3_background_removal():
    """Example: Remove background using SAM3 with optional text prompt"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Background Removal with YOLO + SAM3")
    print("=" * 80)
    
    # Configuration
    image_path = "data/raw/macro_images_lv/your_image.jpg"  # Change this to your image
    yolo_model_path = "models/yolo_22brands_best.pt"  # Default YOLO model
    
    # Check if files exist
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        print("   Please update 'image_path' to point to a real image file.")
        return
    
    if not Path(yolo_model_path).exists():
        print(f"⚠️  YOLO model not found: {yolo_model_path}")
        print("   Please update 'yolo_model_path' to point to your YOLO model.")
        return
    
    # Example 1a: With text prompt
    print(f"\n📸 Processing image: {image_path}")
    print(f"🤖 Using YOLO model: {yolo_model_path}")
    print("🔧 Method: SAM3 with text prompt 'logo'")
    
    result = remove_background_with_sam3(
        image=image_path,
        yolo_model_path=yolo_model_path,
        yolo_confidence=0.25,
        text_prompt="logo"  # Optional: helps SAM3 focus on specific component
    )
    
    if result['error'] is None:
        print("✅ Success!")
        print(f"   - Detected class: {result['class_name']}")
        print(f"   - Confidence: {result['confidence']:.2f}")
        print(f"   - Bounding box: {result['bbox']}")
        
        # Save result
        output_path = "results/example_sam3_with_prompt.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result['segmented_image'].save(output_path)
        print(f"   - Saved to: {output_path}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Example 1b: Without text prompt (uses YOLO class name)
    print(f"\n🔧 Method: SAM3 without text prompt (uses YOLO class name)")
    result2 = remove_background_with_sam3(
        image=image_path,
        yolo_model_path=yolo_model_path,
        yolo_confidence=0.25
        # No text_prompt - will use YOLO class name automatically
    )
    
    if result2['error'] is None:
        print("✅ Success!")
        print(f"   - Detected class: {result2['class_name']}")
        print(f"   - Confidence: {result2['confidence']:.2f}")
        
        output_path = "results/example_sam3_no_prompt.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result2['segmented_image'].save(output_path)
        print(f"   - Saved to: {output_path}")
    else:
        print(f"❌ Error: {result2['error']}")


def example_snapiq_background_removal():
    """Example: Remove background using SnapIQ"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Background Removal with YOLO + SnapIQ")
    print("=" * 80)
    
    # Configuration
    image_path = "data/raw/macro_images_lv/your_image.jpg"  # Change this to your image
    yolo_model_path = "models/yolo_22brands_best.pt"
    
    # Check if files exist
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        print("   Please update 'image_path' to point to a real image file.")
        return
    
    if not Path(yolo_model_path).exists():
        print(f"⚠️  YOLO model not found: {yolo_model_path}")
        print("   Please update 'yolo_model_path' to point to your YOLO model.")
        return
    
    # Check for rembg availability
    try:
        from rembg import remove
    except ImportError:
        print("⚠️  rembg not installed.")
        print("   Install with: pip install rembg")
        return
    
    print(f"\n📸 Processing image: {image_path}")
    print(f"🤖 Using YOLO model: {yolo_model_path}")
    print("🔧 Method: SnapIQ")
    
    result = remove_background_with_snapiq(
        image=image_path,
        yolo_model_path=yolo_model_path,
        yolo_confidence=0.25,
        snapiq_model_path=None  # Optional: path to custom SnapIQ model
    )
    
    if result['error'] is None:
        print("✅ Success!")
        print(f"   - Detected class: {result['class_name']}")
        print(f"   - Confidence: {result['confidence']:.2f}")
        
        # Save result
        output_path = "results/example_snapiq.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result['segmented_image'].save(output_path)
        print(f"   - Saved to: {output_path}")
    else:
        print(f"❌ Error: {result['error']}")


def example_dinov2_embedding():
    """Example: Get DINOv2 embedding for Pinecone"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: DINOv2 Embedding for Pinecone")
    print("=" * 80)
    
    # Configuration
    image_path = "data/raw/macro_images_lv/your_image.jpg"  # Change this to your image
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        print("   Please update 'image_path' to point to a real image file.")
        return
    
    print(f"\n📸 Processing image: {image_path}")
    print("🤖 Generating DINOv2 embedding with Vectify...")
    
    embedding = get_dinov2_embedding(
        image=image_path,
        model_name="facebook/dinov2-base"
    )
    
    if embedding is not None:
        print("✅ Success!")
        print(f"   - Embedding shape: {embedding.shape}")
        print(f"   - Embedding dimension: {embedding.shape[0]}")
        print(f"   - Embedding norm: {np.linalg.norm(embedding):.4f} (should be ~1.0 if normalized)")
        print(f"   - First 5 values: {embedding[:5]}")
        print(f"   - Min value: {embedding.min():.4f}")
        print(f"   - Max value: {embedding.max():.4f}")
        
        # Example: Use with Pinecone
        print("\n💡 To use with Pinecone:")
        print("   ```python")
        print("   import pinecone")
        print("   index = pinecone.Index('your-index-name')")
        print("   results = index.query(")
        print("       vectors=[embedding.tolist()],")
        print("       top_k=10")
        print("   )")
        print("   ```")
    else:
        print("❌ Failed to generate embedding")
        print("   Make sure Vectify is installed: pip install vectify")


def example_combined_workflow():
    """Example: Complete workflow - remove background, then get embedding"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Combined Workflow")
    print("=" * 80)
    print("Step 1: Remove background → Step 2: Get embedding → Step 3: Query Pinecone")
    
    # Configuration
    image_path = "data/raw/macro_images_lv/your_image.jpg"
    yolo_model_path = "models/yolo_22brands_best.pt"
    
    # Check if files exist
    if not Path(image_path).exists() or not Path(yolo_model_path).exists():
        print("⚠️  Please update image_path and yolo_model_path to point to real files.")
        return
    
    # Step 1: Remove background with SAM3
    print("\n📸 Step 1: Removing background with SAM3...")
    result = remove_background_with_sam3(
        image=image_path,
        yolo_model_path=yolo_model_path,
        text_prompt="logo"
    )
    
    if result['error'] is not None:
        print(f"❌ Step 1 failed: {result['error']}")
        return
    
    print(f"✅ Background removed! Detected: {result['class_name']}")
    
    # Step 2: Get embedding from segmented image
    print("\n🤖 Step 2: Generating embedding from segmented image...")
    embedding = get_dinov2_embedding(
        image=result['segmented_image'],  # Use the segmented image directly!
        model_name="facebook/dinov2-base"
    )
    
    if embedding is None:
        print("❌ Step 2 failed: Could not generate embedding")
        return
    
    print(f"✅ Embedding generated! Shape: {embedding.shape}")
    
    # Step 3: Ready for Pinecone query
    print("\n🔍 Step 3: Ready to query Pinecone!")
    print("   You can now use this embedding to search for similar images:")
    print(f"   ```python")
    print(f"   results = index.query(vectors=[{embedding.tolist()[:3]}...], top_k=10)")
    print(f"   ```")
    
    # Save intermediate results
    output_path = "results/example_workflow_segmented.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result['segmented_image'].save(output_path)
    print(f"\n💾 Saved segmented image to: {output_path}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Image Processing Helper Functions - Examples")
    print("=" * 80)
    print("\nThis script demonstrates four examples:")
    print("1. Background removal with SAM3 (with and without text prompt)")
    print("2. Background removal with SnapIQ (local model using rembg)")
    print("3. DINOv2 embedding generation")
    print("4. Combined workflow (remove background → get embedding)")
    print("\n⚠️  Note: Update the file paths in this script before running!")
    
    # Uncomment the example you want to run:
    
    # example_sam3_background_removal()
    # example_snapiq_background_removal()
    # example_dinov2_embedding()
    # example_combined_workflow()
    
    print("\n" + "=" * 80)
    print("To run examples, uncomment the function calls above!")
    print("=" * 80)
