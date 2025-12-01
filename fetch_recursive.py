import os
from pinecone import Pinecone
from typing import List, Dict, Any, Optional, Set
import boto3
from botocore.exceptions import ClientError
from collections import deque

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
            region_name="us-east-1"
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

class RecursiveSearchNode:
    """Node structure for recursive search results"""
    def __init__(self, session_id: str, level: int, score: float = 0.0, parent_id: Optional[str] = None):
        self.session_id = session_id
        self.level = level
        self.score = score
        self.parent_id = parent_id
        self.combined_score = score
        self.children = []
        self.metadata = {}
        self.region_scores = {}
        self.matches_by_region = {}
        self.image_url = None

def search_similar_images_recursive(
    session_id: str,
    region_ids: List[str],
    brand_id: str = "louis_vuitton",
    max_depth: int = 2,
    top_k_per_level: int = 3,
    top_k_fetch: int = 50,
    generate_urls: bool = True,
    visited_sessions: Optional[Set[str]] = None,
    current_level: int = 0,
    parent_score: float = 1.0,
    index = None
) -> RecursiveSearchNode:
    """
    Recursively search for similar images using DFS approach.

    Args:
        session_id: Session UUID to search for
        region_ids: List of region IDs to search in
        brand_id: Brand identifier
        max_depth: Maximum depth of recursion (n levels)
        top_k_per_level: Number of top matches to explore at each level (default: 3)
        top_k_fetch: Number of results to fetch from Pinecone (default: 50)
        generate_urls: Whether to generate presigned URLs for images
        visited_sessions: Set of already visited session IDs to avoid cycles
        current_level: Current depth level (for internal use)
        parent_score: Parent node's score (for internal use)
        index: Pinecone index (for internal use, will be created if None)

    Returns:
        RecursiveSearchNode: Root node of the search tree
    """

    # Initialize visited sessions set if not provided
    if visited_sessions is None:
        visited_sessions = set()

    # Initialize Pinecone index if not provided
    if index is None:
        pc = init_pinecone()
        index = pc.Index("luxury-v2")

    # Create node for current session
    node = RecursiveSearchNode(session_id, current_level, parent_score)

    # Mark this session as visited
    visited_sessions.add(session_id)

    # Base case: reached maximum depth
    if current_level >= max_depth:
        return node

    try:
        # Fetch matches for all regions
        matches = fetch_all_id(brand_id, session_id, region_ids, top_k_fetch, index)
        node.matches_by_region = matches

        # Rank sessions by combined score
        ranked_sessions = rerank_session_ids_with_common_score(matches, query_session_id=session_id)

        # Store metadata and scores for current node
        if ranked_sessions:
            # Find current session in ranked results (if it exists)
            for session_data in ranked_sessions:
                if session_data['session_uuid'] == session_id:
                    node.metadata = session_data.get('metadata', {})
                    node.region_scores = session_data.get('region_scores', {})
                    break

        # Generate image URL for current node if requested
        if generate_urls and matches:
            # Get first match from first region for image URL
            for region_matches in matches.values():
                if region_matches:
                    for match in region_matches:
                        match_dict = match_to_dict(match)
                        if match_dict and match_dict.get('id', '').startswith(session_id):
                            node.image_url = get_image_url_from_match(match)
                            if not node.metadata:
                                node.metadata = match_dict.get('metadata', {})
                            break
                    if node.image_url:
                        break

        # Get top K sessions for recursive search (excluding already visited)
        top_sessions = []
        for session_data in ranked_sessions:
            if session_data['session_uuid'] not in visited_sessions:
                top_sessions.append(session_data)
                if len(top_sessions) >= top_k_per_level:
                    break

        # Recursively search for each top session
        for i, session_data in enumerate(top_sessions):
            child_session_id = session_data['session_uuid']
            child_score = session_data['combined_score']

            print(f"{'  ' * current_level}Level {current_level}: Processing child {i+1}/{len(top_sessions)} - Session: {child_session_id[:12]}... (score: {child_score:.4f})")

            # Recursive call for child node
            child_node = search_similar_images_recursive(
                session_id=child_session_id,
                region_ids=region_ids,
                brand_id=brand_id,
                max_depth=max_depth,
                top_k_per_level=top_k_per_level,
                top_k_fetch=top_k_fetch,
                generate_urls=generate_urls,
                visited_sessions=visited_sessions,
                current_level=current_level + 1,
                parent_score=child_score,
                index=index
            )

            # Store additional data from ranked session
            child_node.combined_score = child_score
            child_node.region_scores = session_data.get('region_scores', {})
            child_node.metadata = session_data.get('metadata', {})
            child_node.parent_id = session_id

            node.children.append(child_node)

    except Exception as e:
        print(f"Error at level {current_level} for session {session_id}: {e}")

    return node

def flatten_tree_to_results(root: RecursiveSearchNode, include_tree_structure: bool = True) -> Dict[str, Any]:
    """
    Convert the recursive tree structure to a flat results dictionary.

    Args:
        root: Root node of the search tree
        include_tree_structure: Whether to include the tree structure in results

    Returns:
        Dictionary with flattened results and statistics
    """
    all_sessions = []
    tree_structure = []

    def traverse(node: RecursiveSearchNode, path: List[str] = []):
        """DFS traversal to flatten the tree"""
        current_path = path + [node.session_id]

        session_info = {
            'session_id': node.session_id,
            'level': node.level,
            'score': node.score,
            'combined_score': node.combined_score,
            'parent_id': node.parent_id,
            'path': ' -> '.join(current_path),
            'metadata': node.metadata,
            'region_scores': node.region_scores,
            'image_url': node.image_url,
            'num_children': len(node.children)
        }

        all_sessions.append(session_info)

        if include_tree_structure:
            tree_node = {
                'session_id': node.session_id,
                'level': node.level,
                'score': node.combined_score,
                'children': []
            }

            for child in node.children:
                child_tree = traverse(child, current_path)
                tree_node['children'].append(child_tree)

            return tree_node
        else:
            for child in node.children:
                traverse(child, current_path)

    if include_tree_structure:
        tree_structure = traverse(root, [])
    else:
        traverse(root, [])

    # Calculate statistics
    levels_count = {}
    for session in all_sessions:
        level = session['level']
        if level not in levels_count:
            levels_count[level] = 0
        levels_count[level] += 1

    statistics = {
        'total_sessions': len(all_sessions),
        'max_depth_reached': max((s['level'] for s in all_sessions), default=0),
        'sessions_per_level': levels_count,
        'total_with_children': sum(1 for s in all_sessions if s['num_children'] > 0)
    }

    return {
        'all_sessions': all_sessions,
        'tree_structure': tree_structure if include_tree_structure else None,
        'statistics': statistics,
        'root_session_id': root.session_id
    }

def search_similar_images(
    session_id: str,
    region_ids: List[str],
    brand_id: str = "louis_vuitton",
    top_k: int = 10,
    generate_urls: bool = True,
    recursive_search: bool = False,
    max_depth: int = 2,
    top_k_per_level: int = 3
) -> Dict[str, Any]:
    """
    Main function to search for similar images with optional recursive search.

    Args:
        session_id: Session UUID to search for
        region_ids: List of region IDs to search in
        brand_id: Brand identifier
        top_k: Number of top results to return per region
        generate_urls: Whether to generate presigned URLs for images
        recursive_search: Whether to perform recursive DFS search
        max_depth: Maximum depth for recursive search (if enabled)
        top_k_per_level: Number of top matches to explore at each level (if recursive)

    Returns:
        Dictionary containing search results
    """

    if not region_ids:
        raise ValueError("At least one region ID is required")

    if recursive_search:
        print(f"Starting recursive DFS search with max_depth={max_depth}, top_k_per_level={top_k_per_level}")

        # Perform recursive search
        root_node = search_similar_images_recursive(
            session_id=session_id,
            region_ids=region_ids,
            brand_id=brand_id,
            max_depth=max_depth,
            top_k_per_level=top_k_per_level,
            top_k_fetch=top_k,
            generate_urls=generate_urls
        )

        # Flatten tree to results
        results = flatten_tree_to_results(root_node, include_tree_structure=True)

        # Add query parameters
        results['query_params'] = {
            'session_id': session_id,
            'region_ids': region_ids,
            'brand_id': brand_id,
            'top_k': top_k,
            'recursive': True,
            'max_depth': max_depth,
            'top_k_per_level': top_k_per_level
        }

        return results

    else:
        # Original non-recursive search
        pc = init_pinecone()
        pinecone_index = pc.Index("luxury-v2")

        matches = fetch_all_id(brand_id, session_id, region_ids, top_k, pinecone_index)
        ranked_sessions = rerank_session_ids_with_common_score(matches, query_session_id=session_id)
        common_sessions = fetch_common_session_id(matches)

        total_matches = sum(len(region_matches) for region_matches in matches.values())

        statistics = {
            'total_regions': len(region_ids),
            'unique_sessions': len(ranked_sessions),
            'total_matches': total_matches,
            'best_combined_score': ranked_sessions[0]['combined_score'] if ranked_sessions else 0,
            'common_sessions_count': len(common_sessions)
        }

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
                'top_k': top_k,
                'recursive': False
            }
        }

def print_tree_structure(tree_node: Dict, indent: int = 0):
    """Helper function to print tree structure"""
    print(f"{'  ' * indent}└─ Session: {tree_node['session_id'][:12]}... (Level: {tree_node['level']}, Score: {tree_node['score']:.4f})")
    for child in tree_node.get('children', []):
        print_tree_structure(child, indent + 1)

# Example usage
if __name__ == "__main__":
    # Example parameters
    session_id = "36033bf0-f0da-438c-b0b4-8bd46fc36306"
    region_ids = ["macro.camera", "micro.inner_logo", "macro.inner_logo"]

    # Example 1: Non-recursive search (original functionality)
    print("=" * 80)
    print("STANDARD SEARCH")
    print("=" * 80)

    results = search_similar_images(
        session_id=session_id,
        region_ids=region_ids,
        brand_id="louis_vuitton",
        top_k=10,
        generate_urls=True,
        recursive_search=False
    )

    print(f"Statistics: {results['statistics']}")
    print(f"\nTop 3 ranked sessions:")
    for i, session in enumerate(results.get('ranked_sessions', [])[:3], 1):
        print(f"{i}. Session: {session['session_uuid']}, Combined Score: {session['combined_score']:.4f}")

    # Example 2: Recursive DFS search
    print("\n" + "=" * 80)
    print("RECURSIVE DFS SEARCH")
    print("=" * 80)

    recursive_results = search_similar_images(
        session_id=session_id,
        region_ids=region_ids,
        brand_id="louis_vuitton",
        top_k=50,  # Fetch more results for better ranking
        generate_urls=True,
        recursive_search=True,
        max_depth=3,  # Search 3 levels deep
        top_k_per_level=3  # Top 3 matches at each level
    )

    print(f"\nStatistics: {recursive_results['statistics']}")
    print(f"Total sessions explored: {recursive_results['statistics']['total_sessions']}")
    print(f"Sessions per level: {recursive_results['statistics']['sessions_per_level']}")

    # Print tree structure
    print("\nSearch Tree Structure:")
    if recursive_results.get('tree_structure'):
        print_tree_structure(recursive_results['tree_structure'])

    # Print paths for all discovered sessions
    print("\nAll discovered session paths:")
    for session in recursive_results['all_sessions'][:10]:  # Show first 10
        print(f"  {session['path']}")
        print(f"    Score: {session['combined_score']:.4f}, Level: {session['level']}")