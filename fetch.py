import os
import json
import csv
import sys
import asyncio
import unicodedata
from typing import Any, Dict, List, Optional, Set, Iterable
from pathlib import Path

from pinecone import PineconeAsyncio
from dotenv import load_dotenv
from pydantic import BaseModel
import pandas as pd
from tqdm import tqdm

# Increase CSV field size limit to handle large fields (e.g., long session_ids lists)
csv.field_size_limit(sys.maxsize)

load_dotenv()


# ============================================================
# RATE LIMITING
# ============================================================

class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last = None  # Initialize lazily
        self.lock = asyncio.Lock()

    async def consume(self, amount=1):
        async with self.lock:
            loop = asyncio.get_event_loop()
            now = loop.time()

            if self.last is None:
                self.last = now

            elapsed = now - self.last
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last = now

            if self.tokens < amount:
                wait_time = (amount - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= amount


global_bucket = TokenBucket(rate=50, capacity=50)


# ============================================================
# CONFIG MODELS
# ============================================================

class StaticPineconeConfig(BaseModel):
    index_name: str = "uniform-style"
    top_k: int = 10
    score_threshold: float = 0.7
    max_depth: int = 15
    output_file: str = "Saint_Laurent_result_cluster.json"
    use_cache: bool = True
    max_retries: int = 5
    backoff_base: float = 0.5
    max_results_per_session: int = 70


class SimilarityClusterConfig(BaseModel):
    function_id: str = "authentication"
    brand_id: Optional[str] = None
    device_type: str = "camera"
    level_1_region_id: str = "macro.camera.0"
    level_2_region_id: str = "macro.inner_logo.0"
    pinecone: StaticPineconeConfig = StaticPineconeConfig()


# ============================================================
# CACHING
# ============================================================

_query_cache: Dict[str, List[Dict[str, Any]]] = {}


def _set_cached_query(cache_key: str, result: List[Dict[str, Any]]) -> None:
    """Store query result in cache."""
    _query_cache[cache_key] = result


# ============================================================
# ASYNC QUERY WITH RETRIES + RATE LIMITS
# ============================================================

async def query_with_retry(index, params, cfg: StaticPineconeConfig):
    retries = 0

    while True:
        await global_bucket.consume(1)

        try:
            return await index.query(**params)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                wait = cfg.backoff_base * (2 ** retries)
                await asyncio.sleep(wait)
                retries += 1
                if retries > cfg.max_retries:
                    return None
            else:
                return None


# ============================================================
# HELPERS
# ============================================================

def build_filter(function_id: str, brand_id: Optional[str], device_type: str, region_id: Optional[str] = None, session_filter: Optional[Iterable[str]] = None):
    f = {
        "function_id": function_id,
        "device_type": device_type,
    }
    if brand_id:
        f["brand_id"] = brand_id
    if region_id:
        f["region_id"] = region_id
    if session_filter:
        f["session_uuid"] = {"$in": list(session_filter)}
    return f


def extract_session(id_str: str) -> Optional[str]:
    """Extract session ID from full ID string."""
    return id_str.split(".", 1)[0] if "." in id_str else None


def merge_two_levels(cam_results, inner_results):
    """
    Merge camera results and inner logo results, including camera_score.
    Similar to fetch_async.py's merge_two_levels function.
    
    Args:
        cam_results: List of camera match results with 'id' and 'score'
        inner_results: List of inner logo match results with 'id' and 'score'
    
    Returns:
        List of merged results with 'id', 'camera_score', and 'inner_logo_score'
    """
    cam_map = {extract_session(c["id"]): c["score"] for c in cam_results}
    merged = []
    for inner in inner_results:
        sid = extract_session(inner["id"])
        merged.append({
            "id": inner["id"],
            "camera_score": cam_map.get(sid),
            "inner_logo_score": inner["score"]
        })
    return merged


def normalize_brand_name(brand_name: Optional[str]) -> str:
    """
    Normalize brand name by removing accents and special characters.
    For example, "Céline" becomes "Celine".
    
    Args:
        brand_name: Brand name from CSV (e.g., "Céline", "Louis Vuitton")
    
    Returns:
        Normalized brand name (e.g., "Celine", "Louis Vuitton")
    """
    if not brand_name:
        return ""
    
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


def normalize_brand_name_to_id(brand_name: Optional[str]) -> str:
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
    if normalized_name.lower() == "saint laurent":
        return "yves_saint_laurent"
    # Convert to lowercase and replace spaces with underscores
    return normalized_name.lower().replace(' ', '_')


def get_all_brand_session_uuids(
    csv_file: str, 
    exclude_brands: Optional[Set[str]] = None
) -> List[Dict[str, str]]:
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
        return []


# ============================================================
# CORE MATCH PROCESSOR
# ============================================================

async def closest_match(
    index,
    id: str,
    config: SimilarityClusterConfig,
    region_id: Optional[str],
    session_filter: Optional[Iterable[str]],
    region_name: str,
    top_k:int
):
    """
    Async function to find closest matches using Pinecone query.
    Similar to fetch_async.py's closest_match function.
    """
    params = dict(
        id=id,
        top_k=top_k,
        include_metadata=True,
        filter=build_filter(
            config.function_id,
            config.brand_id,
            config.device_type,
            region_id,
            session_filter
        )
    )

    resp = await query_with_retry(index, params, config.pinecone)
    if resp is None:
        return []

    return [
        {"id": m.id, "score": m.score, "region": region_name}
        for m in resp.matches
        if m.score >= config.pinecone.score_threshold
    ]


# ============================================================
# CORE RECURSIVE PROCESSOR
# ============================================================

async def process_recursively(
    index,
    parent_session_id: str,
    config: SimilarityClusterConfig,
    visited: Set[str],
    depth: int,
    all_collected_ids: List[Dict[str, Any]],
    seen_ids: Set[str],
):
    """
    Recursively process a parent session ID:
    1. Get camera results and extract session IDs
    2. Get inner_logo results filtered by camera session IDs
    3. Take top 4 inner_logo IDs, add to global list, and recurse for each
    
    Args:
        index: Pinecone async index
        parent_session_id: Parent session UUID
        config: SimilarityClusterConfig
        visited: Set of visited session IDs to avoid cycles
        depth: Current recursion depth
        all_collected_ids: List to collect all IDs and scores
        seen_ids: Set of unique IDs already collected (to track limit)
    """
    if parent_session_id in visited or depth > config.pinecone.max_depth:
        return []
    
    # Check if we've reached the maximum results limit
    if len(seen_ids) >= config.pinecone.max_results_per_session:
        return []

    visited.add(parent_session_id)
    
    collected = []

    try:
        # LEVEL 1: CAMERA REGION
        camera_id = f"{parent_session_id}.{config.level_1_region_id}"
        cache_key_camera = f"camera_{camera_id}_{config.brand_id or 'default'}"
        
        if config.pinecone.use_cache and cache_key_camera in _query_cache:
            results_camera = _query_cache[cache_key_camera]
        else:
            results_camera = await closest_match(
                index=index,
                id=camera_id,
                config=config,
                region_id=config.level_1_region_id.split(".")[1],
                session_filter=None,
                region_name="camera",
                top_k=config.pinecone.top_k,
            )
            if config.pinecone.use_cache:
                _set_cached_query(cache_key_camera, results_camera)

        allowed = [extract_session(c["id"]) for c in results_camera if extract_session(c["id"])]

        # LEVEL 2: INNER REGION
        inner_id = f"{parent_session_id}.{config.level_2_region_id}"
        filter_str = str(sorted(allowed)) if allowed else "none"
        cache_key_inner_logo = f"inner_logo_{inner_id}_{filter_str}_{config.brand_id or 'default'}"
        
        if config.pinecone.use_cache and cache_key_inner_logo in _query_cache:
            results_inner_logo = _query_cache[cache_key_inner_logo]
        else:
            results_inner_logo = await closest_match(
                index=index,
                id=inner_id,
                config=config,
                region_id=config.level_2_region_id.split(".")[1],
                session_filter=allowed,
                region_name="inner_logo",
                top_k=4,
                )
            if config.pinecone.use_cache:
                _set_cached_query(cache_key_inner_logo, results_inner_logo)

        inner_above = [i for i in results_inner_logo if i["score"] >= config.pinecone.score_threshold]
        top_4_inner_logo = inner_above[:4]
        
        # Merge camera and inner_logo results to include camera_score
        merged_results = merge_two_levels(results_camera, top_4_inner_logo)
        
        # Add to collected list, but only up to the limit
        remaining_slots = config.pinecone.max_results_per_session - len(seen_ids)
        if remaining_slots > 0:
            for item in merged_results:
                item_id = item.get('id', '')
                if item_id and item_id not in seen_ids:
                    if len(seen_ids) >= config.pinecone.max_results_per_session:
                        break
                    seen_ids.add(item_id)
                    collected.append(item)
                    all_collected_ids.append(item)
        else:
            pass

        # RECURSE INTO CHILD SESSIONS (only if we haven't reached the limit)
        tasks = []
        if len(seen_ids) < config.pinecone.max_results_per_session:
            for r in top_4_inner_logo:
                child = extract_session(r["id"])
                if child and child != parent_session_id:
                    tasks.append(
                        process_recursively(index, child, config, visited, depth + 1, all_collected_ids, seen_ids)
                    )

        if tasks:
            nested = await asyncio.gather(*tasks)
            for item in nested:
                collected.extend(item)

    except Exception as e:
        pass

    return collected


# ============================================================
# FILE I/O
# ============================================================

def load_existing_results(output_file: str) -> tuple[Dict[str, Any], Set[str]]:
    """
    Load existing results from output file if it exists.
    
    Args:
        output_file: Path to the output JSON file
    
    Returns:
        Tuple of (output_data dict, set of already processed UUIDs)
    """
    output_data = {
        "brands": {},
        "total_brands_processed": 0,
        "total_session_uuids_processed": 0
    }
    
    already_processed_uuids = set()
    
    try:
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, dict):
                    if "brands" in existing_data:
                        output_data["brands"] = existing_data.get("brands", {})
                        # Collect all already processed session UUIDs
                        for brand_name, brand_data in output_data["brands"].items():
                            if "results" in brand_data:
                                brand_uuids = set(brand_data["results"].keys())
                                already_processed_uuids.update(brand_uuids)
                        output_data["total_brands_processed"] = len(output_data["brands"])
                        output_data["total_session_uuids_processed"] = len(already_processed_uuids)
                    elif "results" in existing_data:
                        # Legacy format: single brand
                        legacy_brand_raw = existing_data.get("brand", "Louis Vuitton")
                        legacy_brand = normalize_brand_name(legacy_brand_raw)
                        legacy_uuids = set(existing_data.get("results", {}).keys())
                        already_processed_uuids.update(legacy_uuids)
                        output_data["brands"][legacy_brand] = {
                            "brand_id": normalize_brand_name_to_id(legacy_brand),
                            "results": existing_data.get("results", {}),
                            "total_session_uuids_processed": len(legacy_uuids)
                        }
                        output_data["total_brands_processed"] = 1
                        output_data["total_session_uuids_processed"] = len(legacy_uuids)
    except Exception as e:
        pass
    
    return output_data, already_processed_uuids


def save_results_atomic(output_data: Dict[str, Any], output_file: str) -> bool:
    """
    Save results to file using atomic write (temp file + rename).
    
    Args:
        output_data: Data dictionary to save
        output_file: Path to output file
    
    Returns:
        True if successful, False otherwise
    """
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
        
        return True
    except Exception as e:
        return False


# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

async def process_one_session(
    index,
    session_uuid: str,
    brand: str,
    brand_id: str,
    config: SimilarityClusterConfig,
    best_match_style: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a single session UUID recursively.
    
    Args:
        index: Pinecone async index
        session_uuid: Session UUID to process
        brand: Brand name (normalized)
        brand_id: Brand ID for Pinecone filtering
        config: SimilarityClusterConfig
        best_match_style: Optional best match style from CSV
    
    Returns:
        Dictionary with session results
    """
    visited = set()
    all_collected_ids = []
    seen_ids = set()  # Track unique IDs to enforce limit
    
    # Update config with brand_id
    config.brand_id = brand_id
    
    items = await process_recursively(index, session_uuid, config, visited, 0, all_collected_ids, seen_ids)
    
    # Extract unique IDs (by 'id' key) while preserving scores
    # Note: seen_ids already tracks uniqueness, so we can use all_collected_ids directly
    # but we'll still deduplicate to be safe
    unique_ids_with_scores = []
    seen_in_final = set()
    for item in all_collected_ids:
        item_id = item.get('id', '') if isinstance(item, dict) else item
        if item_id and item_id not in seen_in_final:
            seen_in_final.add(item_id)
            unique_ids_with_scores.append(item)
    
    result = {
        "unique_ids_with_scores": unique_ids_with_scores,
        "unique_count": len(unique_ids_with_scores),
        "total_collected": len(all_collected_ids)
    }
    
    # Add best_match_style if provided
    if best_match_style:
        result["best_match_style"] = best_match_style
    
    return result


async def process_all_products(
    csv_file: str,
    cfg_static: StaticPineconeConfig,
) -> Dict[str, Any]:
    """
    Process all brands and their Session UUIDs from CSV file recursively.
    Results are saved incrementally to prevent data loss.
    Similar to fetch_async.py - reads CSV directly with pandas.
    
    Args:
        csv_file: Path to the CSV file
        cfg_static: StaticPineconeConfig with all settings (max_depth, use_cache, output_file, etc.)
    
    Returns:
        Dictionary mapping brands to their results
    """
    
    # Load existing results
    output_data, already_processed_uuids = load_existing_results(cfg_static.output_file)
    
    # Read CSV directly with pandas (like fetch_async.py)
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        return {}
    
    # Filter out already processed session UUIDs
    if 'Session UUID' in df.columns:
        df = df[~df['Session UUID'].isin(already_processed_uuids)]
    
    # Group by brand for summary
    brand_counts = {}
    if 'Brand_x' in df.columns:
        for brand_raw in df['Brand_x'].dropna().unique():
            brand = normalize_brand_name(brand_raw)
            count = len(df[df['Brand_x'] == brand_raw])
            brand_counts[brand] = count
    
    total_sessions = len(df)
    
    # Initialize Pinecone async client
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY")
    
    async with PineconeAsyncio(api_key=api_key) as pc:
        # Get index metadata once
        meta = await pc.describe_index(cfg_static.index_name)
        host = meta.host
        
        # Create index connection once and reuse
        index = pc.IndexAsyncio(host=host)
        
        try:
            # Process each row directly (like fetch_async.py) with progress tracking
            pbar = tqdm(total=len(df), desc="Processing sessions", unit="session")
            for idx, (_, row) in enumerate(df.iterrows(), 1):
                session_uuid = row.get('Session UUID', '').strip()
                brand_raw = row.get('Brand_x', '').strip()
                best_match_style = row.get('best_match_style', '').strip() if 'best_match_style' in row else None
                
                if not session_uuid or not brand_raw:
                    pbar.update(1)
                    continue
                
                # Normalize brand name
                brand = normalize_brand_name(brand_raw)
                brand_id = normalize_brand_name_to_id(brand)
                
                try:
                    config = SimilarityClusterConfig(
                        function_id="authentication",
                        brand_id=brand_id,
                        pinecone=cfg_static,
                    )
                    
                    session_result = await process_one_session(
                        index=index,
                        session_uuid=session_uuid,
                        brand=brand,
                        brand_id=brand_id,
                        config=config,
                        best_match_style=best_match_style if best_match_style else None,
                    )
                    
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
                    save_results_atomic(output_data, cfg_static.output_file)
                    pbar.update(1)
                
                except Exception as e:
                    # Save what we have so far even if this session failed
                    output_data["total_brands_processed"] = len(output_data["brands"])
                    output_data["total_session_uuids_processed"] = sum(
                        b["total_session_uuids_processed"] for b in output_data["brands"].values()
                    )
                    save_results_atomic(output_data, cfg_static.output_file)
                    pbar.update(1)
                    continue
            pbar.close()
        
        finally:
            # Explicitly close the index connection
            await index.close()
    
    return output_data["brands"]


# ============================================================
# ENTRYPOINT
# ============================================================

def run_all():
    """Main entry point that runs the async processing."""
    # Create config once with desired settings (consistent with fetch_async.py pattern)
    cfg_static = StaticPineconeConfig()
    
    results = asyncio.run(process_all_products(
        csv_file='styles_sessions_brands_Saint_Laurent.csv',
        cfg_static=cfg_static  # Pass the config object instead of individual parameters
    ))
    
    return results


if __name__ == '__main__':
    run_all()
