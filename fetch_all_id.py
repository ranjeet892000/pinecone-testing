import os
from pinecone import Pinecone
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

# Initialize Pinecone
def init_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc

# Initialize S3 client
def get_s3_client():
    """Initialize and return S3 client"""
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not aws_access_key or not aws_secret_key:
        return None

    try:
        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
    except Exception as e:
        print(f"Error initializing S3 client: {e}")
        return None

def generate_presigned_url(bucket: str, key: str, expiration: int = 3600) -> Optional[str]:
    """Generate a presigned URL for an S3 object"""
    try:
        s3_client = get_s3_client()
        if s3_client is None:
            return None
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        return url
    except ClientError:
        return None
    except Exception:
        return None

def get_image_url_from_match(match) -> Optional[str]:
    """Extract S3 info and generate presigned URL for a match"""
    if not match:
        return None

    match_dict = match_to_dict(match)
    if not match_dict:
        return None

    metadata = match_dict.get('metadata', {})
    if not isinstance(metadata, dict):
        return None

    bucket = metadata.get('s3_bucket')
    key = metadata.get('s3_key')

    if bucket and key:
        return generate_presigned_url(bucket, key)
    return None

def fetch_all_id(brand_id: str, session_id: str, region_ids: List[str], top_k: int = 50, index=None) -> Dict:
    """
    Fetch matches for all regions.
    """
    if index is None:
        pc = init_pinecone()
        index = pc.Index("luxury-v2")

    matches = {}
    for region_id in region_ids:
        query_params = {
            "id": f"{session_id}.{region_id}.0",
            "top_k": top_k,
            "include_metadata": True,
            "filter": {
                "region_id": region_id.split(".")[1],
                "brand_id": brand_id
            }
        }
        results = index.query(**query_params)
        matches[region_id] = results['matches']
    return matches

def fetch_common_session_id(matches: Dict) -> List[str]:
    """Extract unique session IDs from matches"""
    common_session_ids = set()
    for region_id, region_matches in matches.items():
        for match in region_matches:
            if not match:
                continue
            match_dict = match_to_dict(match)
            if match_dict and 'id' in match_dict:
                try:
                    match_id = match_dict['id']
                    if isinstance(match_id, str):
                        common_session_ids.add(match_id.split(".")[0])
                except (AttributeError, KeyError, IndexError, TypeError):
                    pass
    return list(common_session_ids)

def match_to_dict(match) -> Optional[Dict]:
    """Convert ScoredVector object to dictionary for easier handling"""
    if isinstance(match, dict):
        return match

    if not match:
        return None

    try:
        match_dict = {}

        # Get id
        if hasattr(match, 'id'):
            match_dict['id'] = match.id
        elif hasattr(match, '__getitem__'):
            try:
                match_dict['id'] = match['id']
            except (KeyError, TypeError):
                pass

        # Get score
        if hasattr(match, 'score'):
            match_dict['score'] = match.score
        elif hasattr(match, '__getitem__'):
            try:
                match_dict['score'] = match.get('score') if hasattr(match, 'get') else match['score']
            except (KeyError, TypeError, AttributeError):
                match_dict['score'] = None

        # Get metadata
        if hasattr(match, 'metadata'):
            match_dict['metadata'] = match.metadata or {}
        elif hasattr(match, '__getitem__'):
            try:
                match_dict['metadata'] = match.get('metadata', {}) if hasattr(match, 'get') else match.get('metadata', {})
            except (KeyError, TypeError, AttributeError):
                match_dict['metadata'] = {}

        # Get values if available
        if hasattr(match, 'values'):
            match_dict['values'] = match.values

        return match_dict if match_dict else None
    except Exception:
        try:
            if hasattr(match, '__dict__'):
                return match.__dict__
            elif hasattr(match, '__iter__') and not isinstance(match, str):
                return dict(match)
        except Exception:
            pass
        return None

def rerank_session_ids_with_common_score(matches: Dict, query_session_id: Optional[str] = None) -> List[Dict]:
    """
    Combine scores from all regions for each session_uuid and return sorted by combined score.
    """
    session_scores = {}

    for region_id, region_matches in matches.items():
        for match in region_matches:
            if not match:
                continue

            match_dict = match_to_dict(match)
            if not match_dict or 'id' not in match_dict:
                continue

            try:
                match_id = match_dict['id']
                if not isinstance(match_id, str):
                    continue
                session_uuid = match_id.split(".")[0]
            except (AttributeError, KeyError, IndexError, TypeError):
                continue

            score = match_dict.get('score', 0.0)
            metadata = match_dict.get('metadata', {})

            if session_uuid not in session_scores:
                session_scores[session_uuid] = {
                    'scores': [],
                    'region_scores': {},
                    'metadata': metadata
                }

            session_scores[session_uuid]['scores'].append(score)
            session_scores[session_uuid]['region_scores'][region_id] = score

    ranked_sessions = []
    for session_uuid, data in session_scores.items():
        combined_score = sum(data['scores'])
        ranked_sessions.append({
            'session_uuid': session_uuid,
            'combined_score': combined_score,
            'average_score': combined_score / len(data['scores']),
            'region_scores': data['region_scores'],
            'num_regions': len(data['scores']),
            'metadata': data['metadata']
        })

    ranked_sessions.sort(key=lambda x: x['combined_score'], reverse=True)
    return ranked_sessions

def search_similar_images(
    session_id: str,
    region_ids: List[str],
    brand_id: str = "louis_vuitton",
    top_k: int = 10,
    generate_urls: bool = True
) -> Dict[str, Any]:
    """
    Main function to search for similar images based on session ID and region IDs.

    Args:
        session_id: Session UUID to search for
        region_ids: List of region IDs to search in (e.g., ["macro.camera", "micro.inner_logo"])
        brand_id: Brand identifier (default: "louis_vuitton")
        top_k: Number of top results to return per region (default: 10)
        generate_urls: Whether to generate presigned URLs for images (default: True)

    Returns:
        Dictionary containing:
            - matches: Raw matches by region
            - ranked_sessions: Sessions ranked by combined score
            - common_sessions: List of common session IDs across regions
            - statistics: Summary statistics
            - processed_results: Processed results with image URLs (if generate_urls=True)
    """

    if not region_ids:
        raise ValueError("At least one region ID is required")

    # Initialize Pinecone and fetch results
    pc = init_pinecone()
    pinecone_index = pc.Index("luxury-v2")

    # Fetch matches for all regions
    matches = fetch_all_id(brand_id, session_id, region_ids, top_k, pinecone_index)

    # Rank sessions by combined score
    ranked_sessions = rerank_session_ids_with_common_score(matches, query_session_id=session_id)

    # Get common session IDs
    common_sessions = fetch_common_session_id(matches)

    # Calculate statistics
    total_matches = sum(len(region_matches) for region_matches in matches.values())

    statistics = {
        'total_regions': len(region_ids),
        'unique_sessions': len(ranked_sessions),
        'total_matches': total_matches,
        'best_combined_score': ranked_sessions[0]['combined_score'] if ranked_sessions else 0,
        'common_sessions_count': len(common_sessions)
    }

    # Process results with image URLs if requested
    processed_results = {}
    if generate_urls:
        for region_id, region_matches in matches.items():
            processed_results[region_id] = []
            for match in region_matches:
                match_dict = match_to_dict(match)
                if match_dict:
                    result = {
                        'id': match_dict.get('id'),
                        'score': match_dict.get('score', 0.0),
                        'metadata': match_dict.get('metadata', {}),
                        'session_uuid': match_dict['id'].split('.')[0] if match_dict.get('id') and isinstance(match_dict['id'], str) else None,
                        'image_url': get_image_url_from_match(match)
                    }
                    processed_results[region_id].append(result)

    return {
        'matches': matches,
        'ranked_sessions': ranked_sessions,
        'common_sessions': common_sessions,
        'statistics': statistics,
        'processed_results': processed_results if generate_urls else None,
        'query_params': {
            'session_id': session_id,
            'region_ids': region_ids,
            'brand_id': brand_id,
            'top_k': top_k
        }
    }

# Example usage
if __name__ == "__main__":
    # Example parameters
    session_id = "36033bf0-f0da-438c-b0b4-8bd46fc36306"
    region_ids = ["macro.camera", "micro.inner_logo", "macro.inner_logo"]

    # Call the function
    results = search_similar_images(
        session_id=session_id,
        region_ids=region_ids,
        brand_id="louis_vuitton",
        top_k=10,
        generate_urls=True
    )

    # Print results
    print(f"Statistics: {results['statistics']}")
    print(f"\nTop 5 ranked sessions:")
    for i, session in enumerate(results['ranked_sessions'][:5], 1):
        print(f"{i}. Session: {session['session_uuid']}, Combined Score: {session['combined_score']:.4f}")

    print(f"\nCommon sessions count: {len(results['common_sessions'])}")

    if results['processed_results']:
        for region_id, region_results in results['processed_results'].items():
            print(f"\nRegion {region_id}: {len(region_results)} results")
            if region_results:
                top_result = region_results[0]
                print(f"  Top match - Session: {top_result['session_uuid']}, Score: {top_result['score']:.4f}")