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

**Note:** SAM3 supports text-based prompts for more precise segmentation!

### Basic Usage

```python
from PIL import Image
from src.utils.image_processing_helpers import remove_background_yolo_sam3

# Process an image
result = remove_background_yolo_sam3(
    image="path/to/image.jpg",  # or PIL Image object
    yolo_model_path="models/yolo_22brands_best.pt",
    text_prompt="logo"  # optional: helps SAM3 focus
)

# Check result
if result['error'] is None:
    # Save the segmented image (transparent PNG)
    result['segmented_image'].save("output.png")
    print(f"Detected: {result['class_name']}")
    print(f"Confidence: {result['confidence']}")
else:
    print(f"Error: {result['error']}")
```

### Parameters

- `image`: PIL Image, or path to image file (str/Path)
- `yolo_model_path`: Path to YOLO model (.pt file)
- `yolo_confidence`: Confidence threshold (0.0-1.0), default: 0.25
- `text_prompt`: Optional text description for SAM3 (e.g., "logo", "hardware")
- `select_best_detection`: If True, only process best detection (default: True)
- `device`: Device to use ("cuda", "cpu", "mps"), or None for auto-detect

### Returns

Dictionary with:
- `segmented_image`: PIL Image with transparent background (RGBA)
- `bbox`: Bounding box [x1, y1, x2, y2]
- `class_name`: YOLO class name
- `confidence`: YOLO confidence score
- `error`: Error message (None if successful)

## Function 2: Get DINOv2 Embedding

Generates DINOv2 embedding for Pinecone vector queries using the **Vectify library**.

**Note:** This function uses the Vectify library (preferred) and falls back to transformers if Vectify is not available.

### Basic Usage

```python
from src.utils.image_processing_helpers import get_dinov2_embedding
import numpy as np

# Get embedding
embedding = get_dinov2_embedding(
    image="path/to/image.jpg",  # or PIL Image object
    model_name="facebook/dinov2-base"
)

# Use with Pinecone
if embedding is not None:
    print(f"Embedding shape: {embedding.shape}")
    
    # Query Pinecone
    import pinecone
    index = pinecone.Index('your-index-name')
    results = index.query(
        vectors=[embedding.tolist()],
        top_k=10
    )
```

### Parameters

- `image`: PIL Image, or path to image file (str/Path)
- `model_name`: DINOv2 model name:
  - `"facebook/dinov2-small"` (384 dim)
  - `"facebook/dinov2-base"` (768 dim) - **recommended**
  - `"facebook/dinov2-large"` (1024 dim)
  - `"facebook/dinov2-giant"` (1536 dim)
- `device`: Device to use ("cuda", "cpu", "mps"), or None for auto-detect
- `normalize`: Normalize embedding vector (default: True, recommended for cosine similarity)

### Returns

- Numpy array of shape `(embedding_dim,)` containing the normalized embedding
- Returns `None` if an error occurs

## Complete Workflow Example

```python
from src.utils.image_processing_helpers import (
    remove_background_yolo_sam3,
    get_dinov2_embedding
)
import pinecone

# Step 1: Remove background
result = remove_background_yolo_sam3(
    image="input.jpg",
    yolo_model_path="models/yolo_22brands_best.pt",
    text_prompt="logo"
)

if result['error'] is not None:
    print(f"Error: {result['error']}")
    exit(1)

# Step 2: Get embedding from segmented image
embedding = get_dinov2_embedding(
    image=result['segmented_image'],  # Use segmented image directly!
    model_name="facebook/dinov2-base"
)

if embedding is None:
    print("Failed to generate embedding")
    exit(1)

# Step 3: Query Pinecone
pc = pinecone.Pinecone(api_key="your-api-key")
index = pc.Index("your-index-name")

search_results = index.query(
    vectors=[embedding.tolist()],
    top_k=10
)

print(f"Found {len(search_results.matches)} similar images")
```

## Requirements

- `ultralytics` - for YOLO
- `vectify` - for DINOv2 embeddings (preferred) - `pip install vectify`
- `transformers` - fallback for DINOv2 if Vectify not available
- `torch` - PyTorch
- `PIL` / `Pillow` - Image processing
- `numpy` - Array operations
- `requests` - For SnapIQ API (if using SnapIQ)

For SAM3, see `docs/sam3_setup.md` for installation instructions.

## Tips

1. **YOLO Confidence**: Lower values (0.1-0.25) detect more components but may include false positives. Higher values (0.5-0.7) are more conservative.

2. **Text Prompts**: Using a text prompt (e.g., "logo") helps SAM3 focus on the right component. If None, it uses the YOLO class name.

3. **Model Caching**: Models are cached globally, so loading multiple images is efficient.

4. **Device Selection**: The functions auto-detect the best device (CUDA > MPS > CPU). You can override with the `device` parameter.

5. **Embedding Normalization**: Always use normalized embeddings (`normalize=True`) for cosine similarity in Pinecone.

## Troubleshooting

**"YOLO not available"**
- Install: `pip install ultralytics`

**"SAM3 not available"**
- See `docs/sam3_setup.md` for setup instructions

**"No components detected"**
- Try lowering `yolo_confidence` (e.g., 0.15)
- Check that your image contains detectable components

**"Failed to generate embedding"**
- Ensure image is valid and not corrupted
- Check that DINOv2 model can be downloaded (internet connection)

## See Also

- `examples/use_image_helpers.py` - Complete working examples
- `src/utils/image_processing_helpers.py` - Full source code with detailed docstrings

