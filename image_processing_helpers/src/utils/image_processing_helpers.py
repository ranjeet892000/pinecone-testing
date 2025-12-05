#!/usr/bin/env python3
"""
Image Processing Helper Functions

Simple, easy-to-use functions for:
1. Background removal using YOLO + SAM3 (with optional text prompt)
2. Background removal using YOLO + SnapIQ
3. Generating DINOv2 embeddings using Vectify library

These functions are designed to be shared with colleagues who are new to the codebase.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np
from PIL import Image
import torch
import cv2


# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import required modules
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not available. Install with: pip install ultralytics")

# from src.processing.sam3_segmenter import SAM3Segmenter, SAM3_AVAILABLE, SAM3_SUBPROCESS_AVAILABLE

# Import EmbeddingExtractor from local module
# try:
#     from embed_extractor import EmbeddingExtractor
#     VECTIFY_AVAILABLE = True  # Keep same variable name for compatibility

# except ImportError:
#     VECTIFY_AVAILABLE = False
#     print("⚠️  EmbeddingExtractor not available. Make sure embed_extractor.py is in the same directory.")


# Global variables for model caching
_yolo_model_cache = {}
_sam3_segmenter_cache = None
_dinov2_extractor_cache = None


def apply_exif_orientation(image: Image.Image) -> Image.Image:
    """
    Apply EXIF orientation to image if present.
    This fixes rotation issues when images have EXIF orientation data.
    
    Args:
        image: PIL Image object
    
    Returns:
        PIL Image with correct orientation
    """
    try:
        # Check if image has EXIF data
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = image._getexif()
            orientation = exif.get(274)  # EXIF orientation tag
            
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)  # Rotate 270 = -90 (clockwise)
            elif orientation == 8:
                image = image.rotate(90, expand=True)  # Rotate 90 = -270 (counterclockwise)
        elif hasattr(image, 'getexif'):
            # For newer PIL versions
            try:
                exif = image.getexif()
                if exif is not None:
                    orientation = exif.get(274)  # EXIF orientation tag
                    
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)  # Rotate 270 = -90 (clockwise)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)  # Rotate 90 = -270 (counterclockwise)
            except Exception:
                pass
    except Exception:
        # If EXIF handling fails, return image as-is
        pass
    
    return image


def remove_background_with_sam3(
    image: Union[Image.Image, str, Path],
    yolo_model_path: str = "models/yolo_22brands_best.pt",
    yolo_confidence: float = 0.25,
    text_prompt: Optional[str] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Remove background from an image using YOLO detection + SAM3 segmentation.
    
    This function:
    1. Uses YOLO to detect components in the image
    2. Uses SAM3 with optional text prompt to create a precise mask
    3. Removes the background, returning a transparent PNG
    
    Args:
        image: PIL Image object, or path to image file (str or Path)
        yolo_model_path: Path to YOLO model file (.pt file). Default: "models/yolo_22brands_best.pt"
        yolo_confidence: Confidence threshold for YOLO detections (0.0-1.0)
        text_prompt: Optional text description for SAM3 (e.g., "logo", "hardware", "zipper").
                    If None, uses the YOLO detection class name automatically.
        device: Device to use ("cuda", "cpu", "mps"). If None, auto-detects.
    
    Returns:
        Dictionary with:
        - 'segmented_image': PIL Image with transparent background (RGBA)
        - 'bbox': Bounding box [x1, y1, x2, y2] of the detected component
        - 'class_name': YOLO class name (e.g., "logo", "hardware")
        - 'confidence': YOLO detection confidence score
        - 'error': Error message if something went wrong, None otherwise
    
    Example:
        >>> from PIL import Image
        >>> # With text prompt
        >>> result = remove_background_with_sam3(
        ...     image="path/to/image.jpg",
        ...     text_prompt="logo"
        ... )
        >>> # Without text prompt (uses YOLO class name)
        >>> result = remove_background_with_sam3(
        ...     image="path/to/image.jpg"
        ... )
        >>> if result['error'] is None:
        ...     result['segmented_image'].save("output.png")
    """
    # Initialize device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Load image
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            return {
                'segmented_image': None,
                'bbox': None,
                'class_name': None,
                'confidence': None,
                'error': f"Image file not found: {image_path}"
            }
        image = Image.open(image_path).convert('RGB')
    elif not isinstance(image, Image.Image):
        return {
            'segmented_image': None,
            'bbox': None,
            'class_name': None,
            'confidence': None,
            'error': f"Invalid image type: {type(image)}. Expected PIL Image, str, or Path."
        }
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Check YOLO availability
    if not YOLO_AVAILABLE:
        return {
            'segmented_image': None,
            'bbox': None,
            'class_name': None,
            'confidence': None,
            'error': "YOLO not available. Install with: pip install ultralytics"
        }
    
    # Load YOLO model
    yolo_key = (yolo_model_path, device)
    if yolo_key not in _yolo_model_cache:
        if not Path(yolo_model_path).exists():
            return {
                'segmented_image': None,
                'bbox': None,
                'class_name': None,
                'confidence': None,
                'error': f"YOLO model not found: {yolo_model_path}"
            }
        _yolo_model_cache[yolo_key] = YOLO(yolo_model_path)
        _yolo_model_cache[yolo_key].to(device)
    
    yolo_model = _yolo_model_cache[yolo_key]
    
    # Run YOLO detection
    try:
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        results = yolo_model(img_bgr, conf=yolo_confidence, verbose=False)
        
        if not results or len(results[0].boxes) == 0:
            return {
                'segmented_image': None,
                'bbox': None,
                'class_name': None,
                'confidence': None,
                'error': "No components detected by YOLO. Try lowering yolo_confidence."
            }
        
        # Get best detection
        boxes = results[0].boxes
        best_box = boxes[0]
        bbox = best_box.xyxy[0].cpu().numpy()
        conf = float(best_box.conf[0].cpu().numpy())
        cls_id = int(best_box.cls[0].cpu().numpy())
        class_name = yolo_model.names[cls_id]
        
        # Check SAM3 availability
        if not SAM3_AVAILABLE and not SAM3_SUBPROCESS_AVAILABLE:
            return {
                'segmented_image': None,
                'bbox': None,
                'class_name': None,
                'confidence': None,
                'error': "SAM3 not available. See docs/sam3_setup.md for installation instructions."
            }
        
        # Initialize SAM3
        global _sam3_segmenter_cache
        if _sam3_segmenter_cache is None:
            _sam3_segmenter_cache = SAM3Segmenter(device=device)
        
        sam3_segmenter = _sam3_segmenter_cache
        
        # Use text prompt if provided, otherwise use YOLO class name
        sam3_text_prompt = text_prompt if text_prompt else class_name
        
        # Segment with SAM3
        x1, y1, x2, y2 = map(int, bbox)
        bbox_tuple = (x1, y1, x2, y2)
        
        segments = sam3_segmenter.segment_image_with_text(
            image=image,
            text_prompt=sam3_text_prompt,
            bbox_constraint=bbox_tuple,
            min_segment_size=100
        )
        
        if not segments:
            return {
                'segmented_image': None,
                'bbox': None,
                'class_name': None,
                'confidence': None,
                'error': f"SAM3 failed to segment. Try adjusting text_prompt or yolo_confidence."
            }
        
        # Get best segment
        best_segment = max(segments, key=lambda x: x['score'])
        mask = best_segment['mask']
        
        # Remove background
        segmented_image = sam3_segmenter.remove_background(
            image=image,
            mask=mask,
            return_transparent=True
        )
        
        return {
            'segmented_image': segmented_image,
            'bbox': bbox.tolist(),
            'class_name': class_name,
            'confidence': conf,
            'error': None
        }
    
    except Exception as e:
        return {
            'segmented_image': None,
            'bbox': None,
            'class_name': None,
            'confidence': None,
            'error': f"Error during processing: {str(e)}"
        }


# def remove_background_with_snapiq(
#     image: Union[Image.Image, str, Path],
#     yolo_model_path: str = "models/yolo_22brands_best.pt",
#     snapiq_model_path: Optional[str] = None,
#     yolo_confidence: float = 0.25,
#     device: Optional[str] = None
# ) -> Dict[str, Any]:
#     """
#     Remove background from an image using YOLO detection + SnapIQ local model.
    
#     This function:
#     1. Uses YOLO to detect components in the image
#     2. Crops to the detected bounding box
#     3. Uses SnapIQ local model (rembg) to remove background
#     4. Returns a transparent PNG
    
#     Args:
#         image: PIL Image object, or path to image file (str or Path)
#         yolo_model_path: Path to YOLO model file (.pt file). Default: "models/yolo_22brands_best.pt"
#         snapiq_model_path: Path to SnapIQ model file (optional, uses default rembg model if None)
#         yolo_confidence: Confidence threshold for YOLO detections (0.0-1.0)
#         device: Device to use ("cuda", "cpu", "mps"). If None, auto-detects.
    
#     Returns:
#         Dictionary with:
#         - 'segmented_image': PIL Image with transparent background (RGBA)
#         - 'bbox': Bounding box [x1, y1, x2, y2] of the detected component
#         - 'class_name': YOLO class name (e.g., "logo", "hardware")
#         - 'confidence': YOLO detection confidence score
#         - 'error': Error message if something went wrong, None otherwise
    
#     Note:
#         Requires rembg library: pip install rembg
    
#     Example:
#         >>> from PIL import Image
#         >>> result = remove_background_with_snapiq(
#         ...     image="path/to/image.jpg",
#         ...     snapiq_model_path="models/snapiq_model.onnx"  # Optional
#         ... )
#         >>> if result['error'] is None:
#         ...     result['segmented_image'].save("output.png")
#     """
#     # Try to import rembg (SnapIQ uses rembg library)
#     try:
#         from rembg import remove
#         REMBG_AVAILABLE = True
#     except ImportError:
#         REMBG_AVAILABLE = False
#         return {
#             'segmented_image': None,
#             'bbox': None,
#             'class_name': None,
#             'confidence': None,
#             'error': "rembg not available. Install with: pip install rembg"
#         }
    
#     # Initialize device
#     if device is None:
#         if torch.cuda.is_available():
#             device = "cuda"
#         elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#             device = "mps"
#         else:
#             device = "cpu"
    
#     # Load image
#     if isinstance(image, (str, Path)):
#         image_path = Path(image)
#         if not image_path.exists():
#             return {
#                 'segmented_image': None,
#                 'bbox': None,
#                 'class_name': None,
#                 'confidence': None,
#                 'error': f"Image file not found: {image_path}"
#             }
#         image = Image.open(image_path).convert('RGB')
#     elif not isinstance(image, Image.Image):
#         return {
#             'segmented_image': None,
#             'bbox': None,
#             'class_name': None,
#             'confidence': None,
#             'error': f"Invalid image type: {type(image)}. Expected PIL Image, str, or Path."
#         }
    
#     if image.mode != 'RGB':
#         image = image.convert('RGB')
    
#     # Check YOLO availability
#     if not YOLO_AVAILABLE:
#         return {
#             'segmented_image': None,
#             'bbox': None,
#             'class_name': None,
#             'confidence': None,
#             'error': "YOLO not available. Install with: pip install ultralytics"
#         }
    
#     # Load YOLO model
#     yolo_key = (yolo_model_path, device)
#     if yolo_key not in _yolo_model_cache:
#         if not Path(yolo_model_path).exists():
#             return {
#                 'segmented_image': None,
#                 'bbox': None,
#                 'class_name': None,
#                 'confidence': None,
#                 'error': f"YOLO model not found: {yolo_model_path}"
#             }
#         _yolo_model_cache[yolo_key] = YOLO(yolo_model_path)
#         _yolo_model_cache[yolo_key].to(device)
    
#     yolo_model = _yolo_model_cache[yolo_key]
    
#     # Run YOLO detection
#     try:
#         img_array = np.array(image)
#         img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
#         results = yolo_model(img_bgr, conf=yolo_confidence, verbose=False)
        
#         if not results or len(results[0].boxes) == 0:
#             return {
#                 'segmented_image': None,
#                 'bbox': None,
#                 'class_name': None,
#                 'confidence': None,
#                 'error': "No components detected by YOLO. Try lowering yolo_confidence."
#             }
        
#         # Get best detection
#         boxes = results[0].boxes
#         best_box = boxes[0]
#         bbox = best_box.xyxy[0].cpu().numpy()
#         conf = float(best_box.conf[0].cpu().numpy())
#         cls_id = int(best_box.cls[0].cpu().numpy())
#         class_name = yolo_model.names[cls_id]
        
#         # Crop image to bounding box
#         x1, y1, x2, y2 = map(int, bbox)
#         cropped_image = image.crop((x1, y1, x2, y2))
        
#         # Use rembg to remove background
#         # Convert PIL to bytes for rembg
#         from io import BytesIO
#         img_buffer = BytesIO()
#         cropped_image.save(img_buffer, format='PNG')
#         img_bytes = img_buffer.getvalue()
        
#         # Remove background using rembg
#         # If custom model path provided, use it; otherwise use default
#         if snapiq_model_path and Path(snapiq_model_path).exists():
#             # Use custom model (if rembg supports it)
#             output_bytes = remove(img_bytes, model_name=snapiq_model_path)
#         else:
#             # Use default rembg model
#             output_bytes = remove(img_bytes)
        
#         # Convert result back to PIL Image
#         result_image = Image.open(BytesIO(output_bytes))
        
#         # Ensure RGBA format
#         if result_image.mode != 'RGBA':
#             result_image = result_image.convert('RGBA')
        
#         return {
#             'segmented_image': result_image,
#             'bbox': bbox.tolist(),
#             'class_name': class_name,
#             'confidence': conf,
#             'error': None
#         }
    
#     except Exception as e:
#         return {
#             'segmented_image': None,
#             'bbox': None,
#             'class_name': None,
#             'confidence': None,
#             'error': f"Error processing with SnapIQ: {str(e)}"
#         }


def get_dinov2_embedding(
    image: Union[Image.Image, str, Path],
    model_name: str = "facebook/dinov2-base",
    device: Optional[str] = None,
    normalize: bool = True
) -> Optional[np.ndarray]:
    """
    Generate DINOv2 embedding for an image using EmbeddingExtractor.
    
    This function uses the local EmbeddingExtractor to extract embeddings from images
    for use with Pinecone vector queries.
    
    Args:
        image: PIL Image object, or path to image file (str or Path)
        model_name: DINOv2 model name. Options:
                   - "facebook/dinov2-base" (768 dim) - recommended
                   - "google/vit-base-patch16-224-in21k" (alternative)
        device: Device to use ("cuda", "cpu", "mps"). If None, auto-detects.
        normalize: If True, normalize the embedding vector (recommended for cosine similarity)
    
    Returns:
        Numpy array of shape (embedding_dim,) containing the image embedding.
        Returns None if an error occurs.
    
    Example:
        >>> from PIL import Image
        >>> embedding = get_dinov2_embedding(
        ...     image="path/to/image.jpg",
        ...     model_name="facebook/dinov2-base"
        ... )
        >>> if embedding is not None:
        ...     print(f"Embedding shape: {embedding.shape}")
        ...     # Use embedding to query Pinecone
        ...     # index.query(vectors=[embedding.tolist()], top_k=10)
    """
    # if not VECTIFY_AVAILABLE:
    #     print("❌ Vectify not available. Install with: pip install vectify")
    #     return None
    
    # Initialize device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Load image
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            print(f"❌ Image file not found: {image_path}")
            return None
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
    elif not isinstance(image, Image.Image):
        print(f"❌ Invalid image type: {type(image)}. Expected PIL Image, str, or Path.")
        return None
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if image.size[0] == 0 or image.size[1] == 0:
        print("❌ Invalid image: empty or zero size")
        return None
    
    # Initialize EmbeddingExtractor
    global _dinov2_extractor_cache
    
    try:
        from .embed_extractor import EmbeddingExtractor
        
        if _dinov2_extractor_cache is None:
            print(f"📦 Loading DINOv2 model: {model_name}")
            _dinov2_extractor_cache = EmbeddingExtractor(
                model=model_name,
                device=device,
                return_type='numpy'
            )
            print("✅ DINOv2 model loaded!")
        
        # Extract embedding
        embedding = _dinov2_extractor_cache.extract(image)
        
        # Ensure 1D numpy array
        if isinstance(embedding, np.ndarray):
            if embedding.ndim > 1:
                embedding = embedding.flatten()
        else:
            embedding = np.array(embedding).flatten()
        
        # Validate embedding
        if embedding.size == 0:
            print("❌ Invalid embedding: empty array")
            return None
        
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            print("❌ Invalid embedding: contains NaN or Inf values")
            return None
        
        if np.all(embedding == 0):
            print("❌ Invalid embedding: all zeros")
            return None
        
        # Normalize if requested
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm == 0 or np.isnan(norm) or np.isinf(norm):
                print(f"❌ Invalid embedding: zero or invalid norm (norm={norm})")
                return None
            embedding = embedding / norm
            
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                print("❌ Invalid embedding after normalization: contains NaN or Inf values")
                return None
        
        return embedding
    
    except Exception as e:
        print(f"❌ Error generating DINOv2 embedding: {e}")
        import traceback
        traceback.print_exc()
        return None


def remove_background_with_snapiq(
    image: Union[Image.Image, str, Path],
    yolo_model_path: str = "models/yolo_22brands_best.pt",
    snapiq_model_path: Optional[str] = None,
    yolo_confidence: float = 0.25,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Remove background from an image using YOLO detection + SnapIQ local model.
    
    This function:
    1. Uses YOLO to detect components in the image
    2. Crops to the detected bounding box
    3. Uses SnapIQ local model (rembg) to remove background
    4. Replaces the background with white color
    
    Args:
        image: PIL Image object, or path to image file (str or Path)
        yolo_model_path: Path to YOLO model file (.pt file). Default: "models/yolo_22brands_best.pt"
        snapiq_model_path: Path to SnapIQ model file (optional, uses default rembg model if None)
        yolo_confidence: Confidence threshold for YOLO detections (0.0-1.0)
        device: Device to use ("cuda", "cpu", "mps"). If None, auto-detects.
    
    Returns:
        Dictionary with:
        - 'segmented_image': PIL Image with white background (RGB)
        - 'bbox': Bounding box [x1, y1, x2, y2] of the detected component
        - 'class_name': YOLO class name (e.g., "logo", "hardware")
        - 'confidence': YOLO detection confidence score
        - 'error': Error message if something went wrong, None otherwise
    
    Note:
        Requires rembg library: pip install rembg
    
    Example:
        >>> from PIL import Image
        >>> result = remove_background_with_snapiq(
        ...     image="path/to/image.jpg",
        ...     snapiq_model_path="models/snapiq_model.onnx"  # Optional
        ... )
        >>> if result['error'] is None:
        ...     result['segmented_image'].save("output.png")
    """
    # Try to import rembg (SnapIQ uses rembg library)
    try:
        from rembg import remove
        REMBG_AVAILABLE = True
    except ImportError:
        REMBG_AVAILABLE = False
        return {
            'segmented_image': None,
            'bbox': None,
            'class_name': None,
            'confidence': None,
            'error': "rembg not available. Install with: pip install rembg"
        }
    
    # Initialize device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Load image
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            return {
                'segmented_image': None,
                'bbox': None,
                'class_name': None,
                'confidence': None,
                'error': f"Image file not found: {image_path}"
            }
        image = Image.open(image_path)
        # Apply EXIF orientation to fix rotation issues
        image = apply_exif_orientation(image)
        image = image.convert('RGB')
    elif not isinstance(image, Image.Image):
        return {
            'segmented_image': None,
            'bbox': None,
            'class_name': None,
            'confidence': None,
            'error': f"Invalid image type: {type(image)}. Expected PIL Image, str, or Path."
        }
    else:
        # Apply EXIF orientation if image is already a PIL Image
        image = apply_exif_orientation(image)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Check YOLO availability
    if not YOLO_AVAILABLE:
        return {
            'segmented_image': None,
            'bbox': None,
            'class_name': None,
            'confidence': None,
            'error': "YOLO not available. Install with: pip install ultralytics"
        }
    
    # Load YOLO model
    yolo_key = (yolo_model_path, device)
    if yolo_key not in _yolo_model_cache:
        if not Path(yolo_model_path).exists():
            return {
                'segmented_image': None,
                'bbox': None,
                'class_name': None,
                'confidence': None,
                'error': f"YOLO model not found: {yolo_model_path}"
            }
        _yolo_model_cache[yolo_key] = YOLO(yolo_model_path)
        _yolo_model_cache[yolo_key].to(device)
    
    yolo_model = _yolo_model_cache[yolo_key]
    
    # Initialize timing variables
    start_time = time.time()
    yolo_time = 0
    rembg_time = 0
    bg_time = 0
    
    # Run YOLO detection
    try:
        yolo_start = time.time()
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        results = yolo_model(img_bgr, conf=yolo_confidence, verbose=False)
        yolo_time = time.time() - yolo_start
        
        if not results or len(results[0].boxes) == 0:
            return {
                'segmented_image': None,
                'bbox': None,
                'class_name': None,
                'confidence': None,
                'error': "No components detected by YOLO. Try lowering yolo_confidence."
            }
        
        # Get best detection
        boxes = results[0].boxes
        best_box = boxes[0]
        bbox = best_box.xyxy[0].cpu().numpy()
        conf = float(best_box.conf[0].cpu().numpy())
        cls_id = int(best_box.cls[0].cpu().numpy())
        class_name = yolo_model.names[cls_id]
        
        # Crop image to bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cropped_image = image.crop((x1, y1, x2, y2))
        
        # Use rembg to remove background
        # Convert PIL to bytes for rembg
        from io import BytesIO
        img_buffer = BytesIO()
        cropped_image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        # Remove background using rembg
        rembg_start = time.time()
        # If custom model path provided, use it; otherwise use default
        if snapiq_model_path and Path(snapiq_model_path).exists():
            # Use custom model (if rembg supports it)
            output_bytes = remove(img_bytes, model_name=snapiq_model_path)
        else:
            # Use default rembg model
            output_bytes = remove(img_bytes)
        rembg_time = time.time() - rembg_start
        
        # Convert result back to PIL Image
        result_image = Image.open(BytesIO(output_bytes))
        
        # Ensure RGBA format
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        # Create white background instead of transparent
        bg_start = time.time()
        white_background = Image.new('RGB', result_image.size, (255, 255, 255))
        
        # Composite the foreground (with alpha channel) onto white background
        if result_image.mode == 'RGBA':
            # Paste the image onto white background using alpha channel as mask
            white_background.paste(result_image, mask=result_image.split()[3])  # Use alpha channel as mask
            result_image = white_background
        else:
            result_image = result_image.convert('RGB')
        bg_time = time.time() - bg_start
        
        # Fix rotation: Rotate 90 degrees clockwise to counter anticlockwise rotation
        # This fixes the issue where rembg output is rotated anticlockwise by 90 degrees
        # result_image = result_image.rotate(-90, expand=True)  # -90 = 90 degrees clockwise
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Print timing information
        print(f"\n⏱️  Segmentation Speed Metrics:")
        print(f"   - YOLO Detection: {yolo_time:.3f}s")
        print(f"   - Background Removal (rembg): {rembg_time:.3f}s")
        print(f"   - White Background Composition: {bg_time:.3f}s")
        print(f"   - Total Time: {total_time:.3f}s")
        
        return {
            'segmented_image': result_image,
            'bbox': bbox.tolist(),
            'class_name': class_name,
            'confidence': conf,
            'error': None,
            'timing': {
                'yolo_detection': yolo_time,
                'background_removal': rembg_time,
                'background_composition': bg_time,
                'total_time': total_time
            }
        }
    
    except Exception as e:
        return {
            'segmented_image': None,
            'bbox': None,
            'class_name': None,
            'confidence': None,
            'error': f"Error processing with SnapIQ: {str(e)}"
        }




# Example usage
if __name__ == "__main__":
    """
    Example usage of the helper functions.
    """
    print("=" * 80)
    print("Image Processing Helper Functions - Examples")
    print("=" * 80)
    
    print("\n1. Remove Background with SAM3 (with text prompt):")
    print("-" * 80)
    print("""
    from src.utils.image_processing_helpers import remove_background_with_sam3
    
    result = remove_background_with_sam3(
        image="path/to/image.jpg",
        text_prompt="logo"  # Optional: helps SAM3 focus
    )
    
    if result['error'] is None:
        result['segmented_image'].save("output_sam3.png")
        print(f"Detected: {result['class_name']}")
    """)
    
    print("\n2. Remove Background with SnapIQ:")
    print("-" * 80)
    print("""
    import os
    from src.utils.image_processing_helpers import remove_background_with_snapiq
    
    os.environ['SNAPIQ_API_KEY'] = 'your-api-key'
    
    result = remove_background_with_snapiq(
        image="path/to/image.jpg"
    )
    
    if result['error'] is None:
        result['segmented_image'].save("output_snapiq.png")
    """)
    
    print("\n3. Get DINOv2 Embedding:")
    print("-" * 80)
    print("""
    from src.utils.image_processing_helpers import get_dinov2_embedding
    
    embedding = get_dinov2_embedding(
        image="path/to/image.jpg",
        model_name="facebook/dinov2-base"
    )
    
    if embedding is not None:
        print(f"Embedding shape: {embedding.shape}")
        # Use with Pinecone
        # index.query(vectors=[embedding.tolist()], top_k=10)
    """)
    
    print("\n" + "=" * 80)
