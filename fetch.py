
import os
import json
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


def process_recursively(parent_session_id, visited=None, depth=0, max_depth=3):
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
        print(f"⏭️  Skipping {parent_session_id} (visited={parent_session_id in visited}, depth={depth})")
        return
    
    visited.add(parent_session_id)
    # print(f"\n🔍 Processing (depth={depth}): {parent_session_id}")
    
    try:
        # Step 1: Get camera results
        camera_id = f"{parent_session_id}.macro.camera.0"
        results_camera = closest_match_without_removing_background(
            pinecone_index=pinecone_index,
            id=camera_id,
            region_id="camera"
        )
        # print(f"\n🔍 Result camera (depth={depth}): {results_camera}")
        
        # Extract session IDs from camera results
        session_ids_camera = extract_session_ids(results_camera)
        # print(f"\n🔍 camera session ids (depth={depth}): {results_camera}")


        # Step 2: Get inner_logo results (filtered by camera session IDs)
        inner_logo_id = f"{parent_session_id}.macro.inner_logo.0"
        results_inner_logo = closest_match_without_removing_background(
            pinecone_index=pinecone_index,
            id=inner_logo_id,
            region_id="inner_logo",
            session_ids_filter=session_ids_camera if session_ids_camera else None
        )

        # print(f"\n🔍 Result inner logo (depth={depth}): {results_inner_logo}")
        
        
        # Step 3: Take top 4 IDs and add to global list (with scores)
        top_4_inner_logo = results_inner_logo[:4]
        # print(f"\n🔍 Top 4 inner logo (depth={depth}): {top_4_inner_logo}")

        all_collected_ids.extend(top_4_inner_logo)
        # print(f"   Added {len(top_4_inner_logo)} IDs to global list (total: {len(all_collected_ids)})")
        
        # Step 4: Recursively process each of the top 4 IDs
        for inner_logo_result in top_4_inner_logo:
            # Extract ID from dictionary
            inner_logo_id_full = inner_logo_result.get('id', '') if isinstance(inner_logo_result, dict) else inner_logo_result
            child_session_id = inner_logo_id_full.split('.')[0] if inner_logo_id_full else ''
            if child_session_id:
                process_recursively(
                    parent_session_id=child_session_id,
                    visited=visited,
                    depth=depth + 1,
                    max_depth=max_depth
                )
        
    except Exception as e:
        print(f"❌ Error processing {parent_session_id}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Reset global list
    all_collected_ids = []
    
    # Start recursive processing with a parent session ID
    parent_session_id = "005aebd9-c260-487b-99de-63eab6aae207"
    
    print(f" Starting recursive processing for: {parent_session_id}")
    
    process_recursively(
        parent_session_id=parent_session_id,
        visited=set(),
        depth=0,
        max_depth=7
    )
    
    # Print and save results
    # Extract unique IDs (by 'id' key) while preserving scores
    seen_ids = set()
    unique_ids_with_scores = []
    for item in all_collected_ids:
        item_id = item.get('id', '') if isinstance(item, dict) else item
        if item_id and item_id not in seen_ids:
            seen_ids.add(item_id)
            unique_ids_with_scores.append(item)
    
    # Extract just the ID strings for unique_ids field
    unique_ids = [item.get('id', '') if isinstance(item, dict) else item for item in unique_ids_with_scores]
    
    # print(f"\n{'='*80}")
    # print(f"📊 Final Results:")
    # print(f"{'='*80}")
    # print(f"   Total IDs collected: {len(all_collected_ids)}")
    # print(f"   Unique IDs: {len(unique_ids)}")
    
    # Save to JSON file
    output_file = "results_cluster.json"
    with open(output_file, 'w') as f:
        json.dump({
            "parent_session_id": parent_session_id,
            "unique_ids_with_scores": unique_ids_with_scores,  # List of unique dicts with 'id' and 'score'
            "unique_count": len(unique_ids_with_scores),
        }, f, indent=2)


    print(f"\n Results saved to {output_file}")
    
    # Print all IDs with scores
    # print(f"\n   All collected IDs (with scores):")
    # for idx, item in enumerate(all_collected_ids, 1):
    #     if isinstance(item, dict):
    #         print(f"      {idx}. {item.get('id', '')} (score: {item.get('score', 0):.4f})")
    #     else:
    #         print(f"      {idx}. {item}")

    # print(f"\n   Unique IDs (with scores):")
    # for idx, item in enumerate(unique_ids_with_scores, 1):
    #     if isinstance(item, dict):
    #         print(f"      {idx}. {item.get('id', '')} (score: {item.get('score', 0):.4f})")
    #     else:
    #         print(f"      {idx}. {item}")




