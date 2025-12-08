
import os
import json
import csv
import unicodedata
from pinecone import Pinecone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Import utility modules
from helper.image_utils import initialize_s3_client
from helper.background_utils import (
    closest_match_with_removing_background,
    closest_match_without_removing_background
)
from extract_metadata import process_jsonl_file

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "luxury-v2"
pinecone_index = pc.Index(index_name)

# Initialize S3 client
s3_client = initialize_s3_client()



def extract_session_ids(id_list):
    """
    Extract session IDs from a list of IDs or ID dictionaries.
    
    Args:
        id_list: List of IDs (strings) or dictionaries with 'id' key
    
    Returns:
        List of unique session IDs (first part before the first dot)
    """
    ids = []
    for item in id_list:
        if isinstance(item, dict):
            # New format: dictionary with 'id' key
            id_str = item.get('id', '')
        else:
            # Old format: string
            id_str = item
        
        if id_str and '.' in id_str:
            session_id = id_str.split('.')[0]
            ids.append(session_id)
    
    return ids[:5]


def process_100_rows():
    """
    Get 100 rows from metadata.jsonl (with region_id='camera') and call both 
    closest_match_with_removing_background and closest_match_without_removing_background for each.
    Returns a JSON structure mapping each row ID to arrays of result IDs from both methods.
    """
    print("📂 Loading 100 records from metadata.jsonl (region_id='camera')...")
    records = process_jsonl_file('metadata.jsonl', max_records=100, region_id_filter='camera')
    print(f"✅ Loaded {len(records)} records\n")
    
    results_dict = {}
    
    for idx, record in enumerate(records, 1):
        print(f"Processing record {idx}/{len(records)}")
        
        # Construct ID from available data
        # Format: {session_uuid}.macro.{region_id}.0
        session_uuid = record.get('session_uuid', '')
        region_id = record.get('region_id', '')
        constructed_id = f"{session_uuid}.macro.camera.0"
        
        s3_key = record.get('s3_key')
        brand_id = record.get('brand_id')
        
        if s3_key:
            try:
                # Call both functions
                print(f"\n🔍 Running closest_match_with_removing_background...")
                results_with_removing_bg = closest_match_with_removing_background(
                    pinecone_index=pinecone_index,
                    s3_client=s3_client,
                    id=constructed_id,
                    s3_key=s3_key,
                    brand_id=brand_id
                )
                
                print(f"\n🔍 Running closest_match_without_removing_background...")
                results_without_removing_bg = closest_match_without_removing_background(
                    pinecone_index=pinecone_index,
                    id=constructed_id
                )
                
                # Store results with separate sections for with_background and without_background
                results_dict[constructed_id] = {
                    "with_background": results_with_removing_bg,
                    "without_background": results_without_removing_bg
                }
                
                # print(f"\n✅ Record {idx} processed: {len(results_with_removing_bg)} results (with bg) + {len(results_without_removing_bg)} results (without bg)")
                
            except Exception as e:
                print(f"❌ Error processing record {idx}: {e}")
                # Store empty lists for failed records
                results_dict[constructed_id] = {
                    "with_background": [],
                    "without_background": []
                }
                continue
        else:
            print(f"⚠️  Skipping record {idx}: No s3_key found")
            results_dict[constructed_id] = {
                "with_background": [],
                "without_background": []
            }
    
    print(f"\n✅ Finished processing {len(records)} records")
    
    
    # Also save to file
    output_file = "results.json"
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n💾 Results saved to {output_file}")
    
    return results_dict

# Global list to collect all IDs
all_collected_ids = []

# Cache for query results to avoid redundant Pinecone queries
_query_cache = {}


def _get_cached_query(cache_key):
    """Get cached query result if available."""
    return _query_cache.get(cache_key)


def _set_cached_query(cache_key, result):
    """Store query result in cache."""
    _query_cache[cache_key] = result


def process_recursively(parent_session_id, visited=None, depth=0, max_depth=3, use_cache=True, brand=None):
    """
    Recursively process a parent session ID:
    1. Get camera results and extract session IDs
    2. Get inner_logo results filtered by camera session IDs
    3. Take top 3 inner_logo IDs, add to global list, and recurse for each
    
    Args:
        parent_session_id: Parent session UUID (e.g., "005aebd9-c260-487b-99de-63eab6aae207")
        visited: Set of visited session IDs to avoid cycles
        depth: Current recursion depth
        max_depth: Maximum recursion depth
    """
    global all_collected_ids
    
    if visited is None:
        visited = set()
    
    # Avoid cycles and respect max depth
    if parent_session_id in visited or depth > max_depth:
        return
    
    visited.add(parent_session_id)
    print(f"\n🔍 Processing (depth={depth}): {parent_session_id}")
    
    try:
        # Step 1: Get camera results (with caching)
        camera_id = f"{parent_session_id}.macro.camera.0"
        cache_key_camera = f"camera_{camera_id}_{brand or 'default'}"
        
        if use_cache and cache_key_camera in _query_cache:
            results_camera = _query_cache[cache_key_camera]
        else:
            results_camera = closest_match_without_removing_background(
                pinecone_index=pinecone_index,
                id=camera_id,
                region_id="camera",
                brand=brand
            )
            if use_cache:
                _set_cached_query(cache_key_camera, results_camera)
        
        # Extract session IDs from camera results
        session_ids_camera = extract_session_ids(results_camera)

        # Step 2: Get inner_logo results (filtered by camera session IDs, with caching)
        inner_logo_id = f"{parent_session_id}.macro.inner_logo.0"
        # Create cache key that includes the filter to ensure correctness
        filter_str = str(sorted(session_ids_camera)) if session_ids_camera else "none"
        cache_key_inner_logo = f"inner_logo_{inner_logo_id}_{filter_str}_{brand or 'default'}"
        
        if use_cache and cache_key_inner_logo in _query_cache:
            results_inner_logo = _query_cache[cache_key_inner_logo]
        else:
            results_inner_logo = closest_match_without_removing_background(
                pinecone_index=pinecone_index,
                id=inner_logo_id,
                region_id="inner_logo",
                session_ids_filter=session_ids_camera if session_ids_camera else None,
                brand=brand
            )
            if use_cache:
                _set_cached_query(cache_key_inner_logo, results_inner_logo)
        
        # Step 3: Take top 4 IDs and add to global list (with scores)
        top_4_inner_logo = results_inner_logo[:4]
        all_collected_ids.extend(top_4_inner_logo)
        
        # Step 4: Recursively process each of the top 4 IDs sequentially
        for inner_logo_result in top_4_inner_logo:
            # print(f"Processing inner logo result: {inner_logo_result}")
            if inner_logo_result['id'].split('.')[0] == parent_session_id:
                # print(f"Skipping inner logo result: {inner_logo_result} because it is the same as the parent session ID")
                continue
            # Extract ID from dictionary
            inner_logo_id_full = inner_logo_result.get('id', '') if isinstance(inner_logo_result, dict) else inner_logo_result
            child_session_id = inner_logo_id_full.split('.')[0] if inner_logo_id_full else ''
            if child_session_id:
                process_recursively(
                    parent_session_id=child_session_id,
                    visited=visited,
                    depth=depth + 1,
                    max_depth=max_depth,
                    use_cache=use_cache,
                    brand=brand
                )
        
    except Exception as e:
        print(f"❌ Error processing {parent_session_id}: {e}")
        import traceback
        traceback.print_exc()


# def get_louis_vuitton_session_uuids(csv_file='all_products.csv'):
#     """
#     Read CSV file and extract unique Session UUIDs where Brand_x = "Louis Vuitton".
    
#     Args:
#         csv_file: Path to the CSV file
    
#     Returns:
#         List of unique Session UUIDs
#     """
#     session_uuids = set()
    
#     try:
#         with open(csv_file, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 brand = row.get('Brand_x', '').strip()
#                 session_uuid = row.get('Session UUID', '').strip()
                
#                 if brand == 'Louis Vuitton' and session_uuid:
#                     session_uuids.add(session_uuid)
        
#         return list(session_uuids)
#     except Exception as e:
#         print(f"❌ Error reading CSV file: {e}")
#         return []

def normalize_brand_name(brand_name):
    """
    Normalize brand name by removing accents and special characters.
    For example, "Céline" becomes "Celine".
    
    Args:
        brand_name: Brand name from CSV (e.g., "Céline", "Louis Vuitton")
    
    Returns:
        Normalized brand name (e.g., "Celine", "Louis Vuitton")
    """
    if not brand_name:
        return brand_name
    
    # Normalize Unicode characters (decompose accents)
    # NFD = Normalization Form Decomposed
    normalized = unicodedata.normalize('NFD', brand_name)
    
    # Remove combining characters (accents, diacritics)
    # Only keep base characters
    normalized = ''.join(
        char for char in normalized 
        if unicodedata.category(char) != 'Mn'  # Mn = Mark, nonspacing (accents)
    )
    
    return normalized


def normalize_brand_name_to_id(brand_name):
    """
    Convert brand name from CSV to brand_id format used in Pinecone.
    First normalizes the brand name (removes accents), then converts to ID format.
    
    Args:
        brand_name: Brand name from CSV (e.g., "Céline", "Louis Vuitton")
    
    Returns:
        Brand ID format (e.g., "celine", "louis_vuitton")
    """
    # First normalize the brand name (remove accents)
    normalized_name = normalize_brand_name(brand_name)
    
    # Convert to lowercase and replace spaces with underscores
    return normalized_name.lower().replace(' ', '_')


def get_all_brand_session_uuids(csv_file='all_products.csv', exclude_brands=None):
    """
    Read CSV file and extract Session UUIDs along with their brands.
    
    Args:
        csv_file: Path to the CSV file
        exclude_brands: Set of brand names to exclude (e.g., already processed brands)
    
    Returns:
        List of dictionaries with 'session_uuid' and 'brand' keys
        Example: [{"session_uuid": "uuid1", "brand": "Louis Vuitton"}, ...]
    """
    if exclude_brands is None:
        exclude_brands = set()
    
    session_brand_pairs = []
    seen = set()  # Track (session_uuid, brand) pairs to avoid duplicates
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                brand_raw = row.get('Brand_x', '').strip()
                session_uuid = row.get('Session UUID', '').strip()
                
                if not brand_raw or not session_uuid:
                    continue
                
                # Normalize brand name (e.g., "Céline" -> "Celine")
                brand = normalize_brand_name(brand_raw)
                
                # Skip excluded brands (check normalized name)
                if brand in exclude_brands:
                    continue
                    
                pair_key = (session_uuid, brand)
                if pair_key not in seen:
                    seen.add(pair_key)
                    session_brand_pairs.append({
                        "session_uuid": session_uuid,
                        "brand": brand  # Store normalized brand name
                    })
        
        return session_brand_pairs
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return []

def process_all_products(csv_file='all_products.csv', max_depth=7, use_cache=True, output_file=None):
    """
    Process all brands and their Session UUIDs from CSV file recursively.
    Results are saved incrementally to prevent data loss.
    
    Args:
        csv_file: Path to the CSV file
        max_depth: Maximum recursion depth
        use_cache: Whether to use query result caching (default: True)
        output_file: Path to output JSON file (default: 'all_products_result_cluster.json')
    
    Returns:
        Dictionary mapping brands to their results
    """
    global all_collected_ids, _query_cache
    
    if output_file is None:
        output_file = "all_products_result_cluster_2.json"
    
    # Initialize output file structure
    output_data = {
        "brands": {},
        "total_brands_processed": 0,
        "total_session_uuids_processed": 0
    }
    
    # Try to load existing results if file exists (for resuming)
    already_processed_uuids = set()
    already_processed_brands = set()
    try:
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, dict):
                    if "brands" in existing_data:
                        output_data["brands"] = existing_data.get("brands", {})
                        # Collect all already processed session UUIDs and brands
                        # Normalize brand names for comparison
                        for brand_name, brand_data in output_data["brands"].items():
                            if "results" in brand_data:
                                brand_uuids = set(brand_data["results"].keys())
                                already_processed_uuids.update(brand_uuids)
                            # Normalize brand name for comparison
                            normalized_brand = normalize_brand_name(brand_name)
                            already_processed_brands.add(normalized_brand)
                        output_data["total_brands_processed"] = len(output_data["brands"])
                        output_data["total_session_uuids_processed"] = len(already_processed_uuids)
                        # print(f"📂 Loaded existing results from {output_file}")
                        # print(f"   - {len(output_data['brands'])} brands already processed")
                        # print(f"   - {len(already_processed_uuids)} session UUIDs already processed")
                        if already_processed_brands:
                            print(f"   - Fully processed brands (will skip): {', '.join(sorted(already_processed_brands))}")
                        print(f"   Will skip already processed brands and session UUIDs\n")
                    elif "results" in existing_data:
                        # Legacy format: single brand
                        legacy_brand_raw = existing_data.get("brand", "Louis Vuitton")
                        # Normalize legacy brand name
                        legacy_brand = normalize_brand_name(legacy_brand_raw)
                        legacy_uuids = set(existing_data.get("results", {}).keys())
                        already_processed_uuids.update(legacy_uuids)
                        output_data["brands"][legacy_brand] = {
                            "brand_id": normalize_brand_name_to_id(legacy_brand),
                            "results": existing_data.get("results", {}),
                            "total_session_uuids_processed": len(legacy_uuids)
                        }
                        already_processed_brands.add(legacy_brand)
                        output_data["total_brands_processed"] = 1
                        output_data["total_session_uuids_processed"] = len(legacy_uuids)
                        # print(f"📂 Loaded legacy format results from {output_file}")
                        # print(f"   - {legacy_brand}: {len(legacy_uuids)} session UUIDs")
                        # print(f"   Will skip already processed brands and session UUIDs\n")
    except Exception as e:
        print(f"⚠️  Could not load existing file: {e}")
        print(f"   Starting fresh\n")
    
    # Get all Session UUIDs with their brands (excluding already processed brands)
    print(f"📂 Reading CSV file: {csv_file}")
    session_brand_pairs = get_all_brand_session_uuids(csv_file, exclude_brands=already_processed_brands)
    
    # Filter out already processed session UUIDs
    session_brand_pairs = [
        pair for pair in session_brand_pairs 
        if pair["session_uuid"] not in already_processed_uuids
    ]
    
    # Group by brand for summary
    brand_counts = {}
    for pair in session_brand_pairs:
        brand = pair["brand"]
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
    
    total_sessions = len(session_brand_pairs)
    print(f"✅ Found {total_sessions} session UUIDs across {len(brand_counts)} brands to process")
    for brand, count in brand_counts.items():
        print(f"   - {brand}: {count} session UUIDs")
    print(f"⚙️  Settings: cache={use_cache}, max_depth={max_depth}")
    print(f"💾 Output file: {output_file}\n")
    
    # Process each session UUID with its brand
    for idx, pair in enumerate(session_brand_pairs, 1):
        session_uuid = pair['session_uuid']
        brand = pair['brand']
        brand_id = normalize_brand_name_to_id(brand)
        
        print(f"\n{'='*80}")
        print(f"Processing Session UUID {idx}/{len(session_brand_pairs)}: {session_uuid} ({brand})")
        print(f"{'='*80}")
        
        try:
            # Reset global list for each session UUID
            all_collected_ids = []
            
            # Process recursively with caching and brand_id
            process_recursively(
                parent_session_id=session_uuid,
                visited=set(),
                depth=0,
                max_depth=max_depth,
                use_cache=use_cache,
                brand=brand_id  # Pass normalized brand_id
            )
            
            # Extract unique IDs (by 'id' key) while preserving scores
            seen_ids = set()
            unique_ids_with_scores = []
            for item in all_collected_ids:
                item_id = item.get('id', '') if isinstance(item, dict) else item
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    unique_ids_with_scores.append(item)
            
            # Store results for this session UUID
            session_result = {
                "unique_ids_with_scores": unique_ids_with_scores,
                "unique_count": len(unique_ids_with_scores),
                "total_collected": len(all_collected_ids)
            }
            
            # Initialize brand data if not exists
            if brand not in output_data["brands"]:
                output_data["brands"][brand] = {
                    "brand_id": brand_id,
                    "results": {},
                    "total_session_uuids_processed": 0
                }
            
            # Update brand data
            output_data["brands"][brand]["results"][session_uuid] = session_result
            output_data["brands"][brand]["total_session_uuids_processed"] = len(output_data["brands"][brand]["results"])
            output_data["total_brands_processed"] = len(output_data["brands"])
            output_data["total_session_uuids_processed"] = sum(
                b["total_session_uuids_processed"] for b in output_data["brands"].values()
            )
            
            # Save incrementally to file
            try:
                # Write to temporary file first, then rename (atomic operation)
                temp_file = output_file + ".tmp"
                with open(temp_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                
                # Atomic rename (works on Unix/Linux/Mac, Windows may need different approach)
                if os.name == 'nt':  # Windows
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    os.rename(temp_file, output_file)
                else:  # Unix/Linux/Mac
                    os.replace(temp_file, output_file)
                
                brand_count = output_data["brands"][brand]["total_session_uuids_processed"]
                # print(f" Completed {session_uuid} ({brand}): {len(unique_ids_with_scores)} unique IDs collected")
                print(f" Saved to {output_file} ({output_data['total_session_uuids_processed']}/{total_sessions + len(already_processed_uuids)} total session UUIDs, {brand}: {brand_count})")
            except Exception as e:
                print(f" Error saving to file: {e}")
                print(f" Completed {session_uuid}: {len(unique_ids_with_scores)} unique IDs collected (not saved)")
        
        except Exception as e:
            print(f" Error processing {session_uuid}: {e}")
            import traceback
            traceback.print_exc()
            # Save what we have so far even if this session failed
            try:
                output_data["total_brands_processed"] = len(output_data["brands"])
                output_data["total_session_uuids_processed"] = sum(
                    b["total_session_uuids_processed"] for b in output_data["brands"].values()
                )
                temp_file = output_file + ".tmp"
                with open(temp_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                if os.name == 'nt':
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    os.rename(temp_file, output_file)
                else:
                    os.replace(temp_file, output_file)
                print(f" Saved partial results to {output_file}")
            except Exception as save_error:
                print(f" Error saving partial results: {save_error}")
            continue
    
    return output_data["brands"]


if __name__ == '__main__':
    # Process all products from CSV with caching (all brands dynamically)
    # Results are saved incrementally to prevent data loss
    print("Processing all products from CSV with caching (all brands)")
    output_file = "all_products_result_cluster_2.json"
    
    results = process_all_products(
        csv_file='all_products.csv',
        max_depth=15,
        use_cache=True,      # Enable caching for faster repeated queries
        output_file=output_file  # Results saved incrementally
    )
    
    # Final summary
    print(f"\n{'='*80}")
    print(f" Final Summary:")
    print(f"{'='*80}")
    print(f"   Total Brands processed: {len(results)}")
    total_sessions = sum(b["total_session_uuids_processed"] for b in results.values())
    print(f"   Total Session UUIDs processed: {total_sessions}")
    total_unique_ids = sum(
        sum(r['unique_count'] for r in brand_data["results"].values())
        for brand_data in results.values()
    )
    print(f"   Total unique IDs across all sessions: {total_unique_ids}")
    print(f"\n All results saved to {output_file}")
    



