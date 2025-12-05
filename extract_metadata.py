#!/usr/bin/env python3
"""
Script to extract brand_id, region_id, s3_key, and session_uuid from metadata.jsonl file.
Each row in the JSONL file can contain multiple regions, and each region can have multiple items with s3_keys.
This script returns up to 100 output records (combinations of brand_id, region_id, s3_key, and session_uuid).
By default, it filters for region_id='camera'.

This module can be imported to use the extraction functions:
    from extract_metadata import process_jsonl_file, extract_metadata_from_row
"""

import json
import sys
from pathlib import Path


def extract_metadata_from_row(row_data, region_id_filter=None):
    """
    Extract brand_id, region_id, s3_key, and session_uuid from a single row.
    Returns a list of dictionaries, each containing brand_id, region_id, s3_key, and session_uuid.
    
    Args:
        row_data: Dictionary containing the row data from JSONL
        region_id_filter: Optional filter to only include records with this region_id (default: None)
    """
    results = []
    brand_id = row_data.get('brand_id')
    session_uuid = row_data.get('session_uuid')
    
    if not brand_id:
        return results
    
    # Extract from regions.macro and regions.micro
    regions = row_data.get('regions', {})
    
    # Process macro regions
    macro_regions = regions.get('macro', {})
    for region_name, region_data in macro_regions.items():
        region_id = region_data.get('region_id')
        
        # Filter by region_id if filter is specified
        if region_id_filter is not None and region_id != region_id_filter:
            continue
        
        items = region_data.get('items', [])
        
        for item in items:
            s3_key = item.get('s3_key')
            if s3_key:
                results.append({
                    'brand_id': brand_id,
                    'region_id': region_id,
                    's3_key': s3_key,
                    'session_uuid': session_uuid
                })
    
    # Process micro regions
    # micro_regions = regions.get('micro', {})
    # for region_name, region_data in micro_regions.items():
    #     region_id = region_data.get('region_id')
    #     items = region_data.get('items', [])
        
    #     for item in items:
    #         s3_key = item.get('s3_key')
    #         if s3_key:
    #             results.append({
    #                 'brand_id': brand_id,
    #                 'region_id': region_id,
    #                 's3_key': s3_key,
    #                 'annotators': annotators
    #             })
    
    return results


def process_jsonl_file(file_path, max_records=100, region_id_filter=None):
    """
    Process the JSONL file and extract metadata from each row.
    Returns a list of extracted records, limited to max_records.
    
    Args:
        file_path: Path to the JSONL file
        max_records: Maximum number of output records to return (default: 100)
        region_id_filter: Optional filter to only include records with this region_id (default: None)
    """
    all_results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Stop if we've reached the maximum number of records
            if len(all_results) >= max_records:
                break
            
            try:
                row_data = json.loads(line)
                row_results = extract_metadata_from_row(row_data, region_id_filter=region_id_filter)
                
                # Add records up to the limit
                remaining = max_records - len(all_results)
                if remaining > 0:
                    all_results.extend(row_results[:remaining])
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}", file=sys.stderr)
                continue
    
    return all_results


def main():
    """Main function to run the script."""
    file_path = Path('metadata.jsonl')
    
    if not file_path.exists():
        print(f"Error: File {file_path} not found", file=sys.stderr)
        sys.exit(1)
    
    region_id_filter = "camera"
    print(f"Processing {file_path} (max 100 records, region_id='{region_id_filter}')...", file=sys.stderr)
    results = process_jsonl_file(file_path, max_records=100, region_id_filter=region_id_filter)
    
    # Output results as JSON (one JSON object per line)
    try:
        for result in results:
            print(json.dumps(result))
    except BrokenPipeError:
        # Handle case where output is piped to a command that closes early (e.g., head)
        pass
    
    print(f"\nTotal records extracted: {len(results)}", file=sys.stderr)


if __name__ == '__main__':
    main()

