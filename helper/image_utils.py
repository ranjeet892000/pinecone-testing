"""
Image processing utilities for downloading, background removal, and embedding creation.
"""

import boto3
import os
from image_processing_helpers.src.utils.image_processing_helpers import (
    remove_background_with_snapiq, 
    get_dinov2_embedding
)


def initialize_s3_client():
    """
    Initialize and return S3 client.
    
    Returns:
        boto3 S3 client
    """
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )


def download_image(s3_client, s3_key, local_path="downloaded_image.jpg"):
    """
    Download an image from S3.
    
    Args:
        s3_client: boto3 S3 client
        s3_key: S3 key of the image to download
        local_path: Local path to save the downloaded image
    
    Returns:
        Path to downloaded image, or None if error
    """
    try:
        s3_client.download_file("entrupy-app-db", s3_key, local_path)
        # print("Downloaded!")
        return local_path
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        return None


def remove_background(image_path, output_path="downloaded_image_no_bg.png", yolo_model_path="yolo_22brands_best.pt"):
    """
    Remove background from downloaded image using SnapIQ (rembg).
    
    Args:
        image_path: Path to the downloaded image
        output_path: Path to save the background-removed image
        yolo_model_path: Path to YOLO model file
    
    Returns:
        Path to background-removed image, or None if error
    """
    # print(f"\n🔄 Removing background from {image_path} using SnapIQ...")
    
    # Call the background removal function
    result = remove_background_with_snapiq(
        image=image_path,
        yolo_model_path=yolo_model_path,
        yolo_confidence=0.25,  # Confidence threshold
        snapiq_model_path=None  # Optional: path to custom SnapIQ model, None uses default rembg model
    )
    
    # Check if successful
    if result['error'] is None:
        # Save the background-removed image
        result['segmented_image'].save(output_path)
        # print(f"✅ Background removed successfully!")
        # print(f"   - Saved to: {output_path}")
        # print(f"   - Detected class: {result['class_name']}")
        # print(f"   - Confidence: {result['confidence']:.2f}")
        return output_path
    else:
        print(f"❌ Error removing background: {result['error']}")
        return None


def create_embedding(image_path, model_name="facebook/dinov2-base"):
    """
    Create DINOv2 embedding for an image.
    
    Args:
        image_path: Path to the image file
        model_name: DINOv2 model name (default: "facebook/dinov2-base")
    
    Returns:
        Numpy array containing the embedding, or None if error
    """
    # print(f"\n🔄 Creating DINOv2 embedding for {image_path}...")
    
    embedding = get_dinov2_embedding(
        image=image_path,
        model_name=model_name,
        normalize=True  # Normalize for cosine similarity
    )
    
    if embedding is not None:
        # print(f"✅ Embedding created successfully!")
        # print(f"   - Embedding shape: {embedding.shape}")
        # print(f"   - Embedding dimension: {len(embedding)}")
        return embedding
    else:
        print(f"❌ Error creating embedding")
        return None

