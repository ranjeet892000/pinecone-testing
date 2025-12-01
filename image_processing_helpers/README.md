# Image Processing Helper Functions

Simple, easy-to-use functions for image processing tasks.

## Quick Start

```python
from src.utils.image_processing_helpers import (
    remove_background_yolo_sam3,
    get_dinov2_embedding
)
```

## Function 1: Remove Background (YOLO + SAM3 or SnapIQ)

Removes background from images using YOLO detection + SAM3 or SnapIQ segmentation.

### Basic Usage

```python
from src.utils.image_processing_helpers import remove_background_yolo_sam3

# Using SAM3 (default)
result = remove_background_yolo_sam3(
    image="path/to/image.jpg",
    yolo_model_path="models/yolo_22brands_best.pt",  # Default, can be omitted
    method="sam3"  # or "snapiq"
)

# Using SnapIQ
result = remove_background_yolo_sam3(
    image="path/to/image.jpg",
    method="snapiq"
)

if result['error'] is None:
    result['segmented_image'].save("output.png")
    print(f"Detected: {result['class_name']}")
    print(f"Method used: {result['method_used']}")
```

### Parameters

- `image`: PIL Image, or path to image file (str/Path)
- `yolo_model_path`: Path to YOLO model (.pt file), default: "models/yolo_22brands_best.pt"
- `yolo_confidence`: Confidence threshold (0.0-1.0), default: 0.25
- `text_prompt`: Optional text description for SAM3 (e.g., "logo", "hardware")
- `select_best_detection`: If True, only process best detection (default: True)
- `device`: Device to use ("cuda", "cpu", "mps"), or None for auto-detect
- `method`: Background removal method - "sam3" (default) or "snapiq"

### Returns

Dictionary with:
- `segmented_image`: PIL Image with transparent background (RGBA)
- `bbox`: Bounding box [x1, y1, x2, y2]
- `class_name`: YOLO class name
- `confidence`: YOLO confidence score
- `method_used`: Method used ("sam3" or "snapiq")
- `error`: Error message (None if successful)

## Function 2: Get DINOv2 Embedding

Generates DINOv2 embedding for Pinecone vector queries.

### Basic Usage

```python
from src.utils.image_processing_helpers import get_dinov2_embedding

embedding = get_dinov2_embedding(
    image="path/to/image.jpg",
    model_name="facebook/dinov2-base"
)

if embedding is not None:
    # Use with Pinecone
    import pinecone
    index = pinecone.Index('your-index-name')
    results = index.query(
        vectors=[embedding.tolist()],
        top_k=10
    )
```

### Parameters

- `image`: PIL Image, or path to image file (str/Path)
- `model_name`: DINOv2 model name (default: "facebook/dinov2-base")
- `device`: Device to use ("cuda", "cpu", "mps"), or None for auto-detect
- `normalize`: Normalize embedding vector (default: True, recommended)

### Returns

- Numpy array of shape `(embedding_dim,)` containing the normalized embedding
- Returns `None` if an error occurs

## Requirements

- `ultralytics` - for YOLO
- `transformers` - for DINOv2
- `torch` - PyTorch
- `PIL` / `Pillow` - Image processing
- `numpy` - Array operations
- `requests` - For SnapIQ API (if using SnapIQ)
- `opencv-python` - For image processing

For SAM3, see `docs/sam3_setup.md` for installation instructions.

For SnapIQ, set the `SNAPIQ_API_KEY` environment variable.

## Testing

Run the test script to verify everything works:

```bash
python test_image_helpers.py
```

## Examples

See `examples/use_image_helpers.py` for complete working examples.

## Notes

- The default YOLO model is `models/yolo_22brands_best.pt`
- Models are cached globally for efficiency
- Both SAM3 and SnapIQ are supported for background removal
- Embeddings are normalized by default for cosine similarity

