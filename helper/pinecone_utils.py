"""
Pinecone utility functions for similarity search and querying.
"""


def find_similar_images(pinecone_index, embedding, top_k=10, include_metadata=True, filter_dict=None):
    """
    Find similar images in Pinecone index using the embedding.
    
    Args:
        pinecone_index: Pinecone index object
        embedding: Numpy array containing the image embedding
        top_k: Number of similar images to return (default: 10)
        include_metadata: Whether to include metadata in results (default: True)
        filter_dict: Optional filter dictionary for metadata filtering
    
    Returns:
        List of match objects from Pinecone, or None if error
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
        
        return results.matches
    except Exception as e:
        print(f"❌ Error querying Pinecone: {e}")
        return None


def query_by_id(pinecone_index, id, top_k=10, include_metadata=True, filter_dict=None):
    """
    Query Pinecone index by ID to find similar items.
    
    Args:
        pinecone_index: Pinecone index object
        id: ID of the vector to query
        top_k: Number of similar images to return (default: 10)
        include_metadata: Whether to include metadata in results (default: True)
        filter_dict: Optional filter dictionary for metadata filtering
    
    Returns:
        List of match objects from Pinecone, or None if error
    """
    query_params = {
        "id": id,
        "top_k": top_k,
        "include_metadata": include_metadata
    }
    
    # Add filter if provided
    if filter_dict:
        query_params["filter"] = filter_dict
    
    # Perform similarity search
    try:
        results = pinecone_index.query(**query_params)
        if results and results.matches:
            return results.matches
        return []
    except Exception as e:
        print(f"❌ Error querying Pinecone: {e}")
        return None

