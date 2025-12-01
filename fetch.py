import os
from pinecone import Pinecone
import boto3
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Import background removal and embedding functions
from image_processing_helpers.src.utils.image_processing_helpers import remove_background_with_snapiq, get_dinov2_embedding

id="026448e3-7b17-402c-bac4-afc43409041c.macro.camera.0"
# id="5f36b55c-f2c5-4f45-b2fc-2ab2d4e1a485.macro.camera.0"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "pranav-test-prod"

# print(pc.list_indexes())
pinecone_index = pc.Index(index_name)

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

def fetch_image_url(id):
    response=pinecone_index.fetch(ids=[id], namespace="")
    vector = response.vectors[id]

    # Access the metadata
    metadata = vector.metadata

    # Get the s3_key
    s3_key = metadata.get("s3_key")
    return s3_key

def download_image(s3_key):
    s3.download_file("entrupy-app-db", s3_key, "downloaded_image.jpg")
    print("Downloaded!")

def remove_background(image_path, output_path="downloaded_image_no_bg.png"):
    """
    Remove background from downloaded image using SnapIQ (rembg).
    
    Args:
        image_path: Path to the downloaded image
        output_path: Path to save the background-removed image
    """
    print(f"\n🔄 Removing background from {image_path} using SnapIQ...")
    
    # Call the background removal function
    result = remove_background_with_snapiq(
        image=image_path,
        yolo_model_path="yolo_22brands_best.pt",  # Path to YOLO model
        yolo_confidence=0.25,  # Confidence threshold
        snapiq_model_path=None  # Optional: path to custom SnapIQ model, None uses default rembg model
    )
    
    # Check if successful
    if result['error'] is None:
        # Save the background-removed image
        result['segmented_image'].save(output_path)
        print(f"✅ Background removed successfully!")
        print(f"   - Saved to: {output_path}")
        print(f"   - Detected class: {result['class_name']}")
        print(f"   - Confidence: {result['confidence']:.2f}")
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
    print(f"\n🔄 Creating DINOv2 embedding for {image_path}...")
    
    embedding = get_dinov2_embedding(
        image=image_path,
        model_name=model_name,
        normalize=True  # Normalize for cosine similarity
    )
    
    if embedding is not None:
        print(f"✅ Embedding created successfully!")
        print(f"   - Embedding shape: {embedding.shape}")
        print(f"   - Embedding dimension: {len(embedding)}")
        return embedding
    else:
        print(f"❌ Error creating embedding")
        return None

def find_similar_images(embedding, top_k=10, include_metadata=True, filter_dict=None):
    """
    Find similar images in Pinecone index using the embedding.
    
    Args:
        embedding: Numpy array containing the image embedding
        top_k: Number of similar images to return (default: 10)
        include_metadata: Whether to include metadata in results (default: True)
        filter_dict: Optional filter dictionary for metadata filtering
    
    Returns:
        Query results from Pinecone
    """
    print(f"\n🔄 Searching for similar images in Pinecone...")
    
    # Convert numpy array to list for Pinecone
    embedding_list = embedding.tolist()
    
    # Build query parameters
    query_params = {
        "vector": embedding_list,
        "top_k": top_k,
        "include_metadata": include_metadata
    }
    
    # Add filter if provided
    if filter_dict:
        query_params["filter"] = filter_dict
        print(f"   - Using filter: {filter_dict}")
    
    # Perform similarity search
    try:
        results = pinecone_index.query(**query_params)
        print(f"✅ Found {len(results.matches)} similar images!")
        
        # Print results
        for i, match in enumerate(results.matches, 1):
            print(f"\n   {i}. ID: {match.id}")
            print(f"      - Score: {match.score:.4f}")
            if match.metadata:
                print(f"      - Metadata: {match.metadata}")
        
        return results
    except Exception as e:
        print(f"❌ Error querying Pinecone: {e}")
        return None

s3_key = fetch_image_url(id)
download_image(s3_key)

# Remove background from downloaded image
bg_removed_path = remove_background("downloaded_image.jpg", "downloaded_image_no_bg.png")

# Create embedding from background-removed image
if bg_removed_path:
    embedding = create_embedding(bg_removed_path, model_name="facebook/dinov2-base")
    
    # Find similar images in Pinecone
    if embedding is not None:
        # Optional: Add filters if needed
        # filter_dict = {
        #     "primary_component": "camera",
        #     "function_id": "authentication"
        # }
        filter_dict = None
        
        similar_images = find_similar_images(
            embedding=embedding,
            top_k=10,
            include_metadata=True,
            filter_dict=filter_dict
        )

# stats = pinecone_index.describe_index_stats()
# query_params = {
#     "id": id,
#     "top_k": 10,
#     "include_metadata": True
# }
# component_filter, region_id_filter, overlay_key_filter,function_id = None, None, None,None
# # Build filter dictionary
# filter_dict = {}
# if component_filter:
#     filter_dict["primary_component"] = component_filter
# if region_id_filter:
#     filter_dict["region_id"] = region_id_filter
# if overlay_key_filter:
#     filter_dict["overlay_key"] = overlay_key_filter
# filter_dict["function_id"] = "authentication"

# # if filter_dict:
# #     query_params["filter"] = filter_dict

# # Perform similarity search
# results = pinecone_index.query(**query_params)
# print(results)






