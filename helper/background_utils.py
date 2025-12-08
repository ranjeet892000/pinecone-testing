"""
Functions for finding closest matches with and without background removal.
"""

from .image_utils import download_image, remove_background, create_embedding, initialize_s3_client
from .pinecone_utils import find_similar_images, query_by_id


def closest_match_with_removing_background(pinecone_index, s3_client, id, s3_key, brand_id):
    """
    Find closest matches with background removal.
    Returns a list of result IDs.
    
    Args:
        pinecone_index: Pinecone index object
        s3_client: boto3 S3 client
        id: ID of the query item
        s3_key: S3 key of the image to process
        brand_id: Brand ID for filtering
    
    Returns:
        List of result IDs
    """
    result_ids = []
    
    # Download image
    download_image(s3_client, s3_key, "downloaded_image.jpg")

    # Remove background from downloaded image
    bg_removed_path = remove_background("downloaded_image.jpg", "downloaded_image_no_bg.png")

    # Create embedding from background-removed image
    if bg_removed_path:
        embedding = create_embedding(bg_removed_path, model_name="facebook/dinov2-base")
        
        # Find similar images in Pinecone
        if embedding is not None:
            filter_dict = {
                "region_id": "camera",
                "brand_id": brand_id
            }
            
            similar_images = find_similar_images(
                pinecone_index=pinecone_index,
                embedding=embedding,
                top_k=10,
                include_metadata=True,
                filter_dict=filter_dict
            )
            
            if similar_images:
                for i, match in enumerate(similar_images, 1):
                    # print(f"\n   {i}. ID: {match.id}")
                    # print(f"      - Score: {match.score:.4f}")
                    # if match.metadata:
                    #     print(f"      - Metadata: {match.metadata}")
                    result_ids.append(match.id)
    
    return result_ids


def closest_match_without_removing_background(pinecone_index, id, region_id=None, session_ids_filter=None,brand=None):
    """
    Find closest matches without background removal.
    Returns a list of dictionaries with 'id' and 'score' keys.
    
    Args:
        pinecone_index: Pinecone index object
        id: ID of the query item
        region_id: Region ID for filtering
        session_ids_filter: Optional list of session UUIDs to filter by, or a single session UUID string.
                           If a list with multiple values, uses Pinecone's $in operator.
    
    Returns:
        List of dictionaries with 'id' and 'score' keys
    """
    results = []
    
    # Build filter dictionary
    filter_dict = {
        "function_id": "authentication",
        "region_id": region_id,
        "device_type": "camera"
    }
    
    # Add brand_id filter if provided
    if brand:
        filter_dict["brand_id"] = brand
    
    # Add session_uuid filter if provided
    if session_ids_filter:
        filter_dict["session_uuid"] = {"$in": session_ids_filter}


        # if isinstance(session_ids_filter, list):
        #     # If it's a list, use $in operator for multiple values
        #     if len(session_ids_filter) == 1:
        #         # Single value in list, use it directly
        #         filter_dict["session_uuid"] = session_ids_filter[0]
        #     else:
        #         # Multiple values, use $in operator
        #         filter_dict["session_uuid"] = {"$in": session_ids_filter}
        # else:
        #     # Single string value
        #     filter_dict["session_uuid"] = session_ids_filter
    
    # Perform similarity search by ID
    matches = query_by_id(
        pinecone_index=pinecone_index,
        id=id,
        top_k=10,
        include_metadata=True,
        filter_dict=filter_dict
    )
    
    if matches:
        for i, match in enumerate(matches, 1):
            # print(f"\n   {i}. ID: {match.id}")
            # print(f"      - Score: {match.score:.4f}")
            # if match.metadata:
            #     print(f"      - Metadata: {match.metadata}")
            if match.score > 0.7:
                results.append({
                    "id": match.id,
                    "score": match.score
                })
    
    return results

