#!/usr/bin/env python3
"""
Script to update results.json file by adding 'common_items_in_top_5' field to each entry.
This field counts how many items are common in the top 5 elements of both 
'with_removing_background' and 'without_removing_background' arrays.
"""

import json
from pathlib import Path


def calculate_common_items_in_top_5(with_bg_list, without_bg_list):
    """
    Calculate the number of common items in the top 5 of both lists.
    
    Args:
        with_bg_list: List of IDs from with_removing_background
        without_bg_list: List of IDs from without_removing_background
    
    Returns:
        Number of common items in top 5
    """
    # Get top 5 from each list
    top_5_with_bg = set(with_bg_list[:5]) if with_bg_list else set()
    top_5_without_bg = set(without_bg_list[:5]) if without_bg_list else set()
    
    # Find intersection (common items)
    common_items = top_5_with_bg & top_5_without_bg
    
    return len(common_items)


def update_results_json(file_path='results.json'):
    """
    Update results.json file by adding 'common_items_in_top_5' field to each entry.
    
    Args:
        file_path: Path to the results.json file (default: 'results.json')
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ Error: File {file_path} not found")
        return
    
    print(f"📂 Reading {file_path}...")
    
    # Read the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results_dict = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    print(f"✅ Loaded {len(results_dict)} entries")
    
    # Update each entry
    updated_count = 0
    for entry_id, entry_data in results_dict.items():
        if isinstance(entry_data, dict):
            with_bg = entry_data.get('with_removing_background', [])
            without_bg = entry_data.get('without_removing_background', [])
            
            # Calculate common items in top 5
            common_count = calculate_common_items_in_top_5(with_bg, without_bg)
            
            # Add or update the field
            entry_data['common_items_in_top_5'] = common_count
            updated_count += 1
    
    print(f"✅ Updated {updated_count} entries")
    
    # Write back to file
    print(f"💾 Writing updated data to {file_path}...")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)
        print(f"✅ Successfully updated {file_path}")
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        return
    
    # Print summary statistics
    print(f"\n📊 Summary:")
    print(f"   - Total entries: {len(results_dict)}")
    print(f"   - Updated entries: {updated_count}")
    
    # Calculate statistics about common items
    common_counts = [entry.get('common_items_in_top_5', 0) 
                     for entry in results_dict.values() 
                     if isinstance(entry, dict)]
    
    if common_counts:
        print(f"   - Average common items in top 5: {sum(common_counts) / len(common_counts):.2f}")
        print(f"   - Max common items in top 5: {max(common_counts)}")
        print(f"   - Min common items in top 5: {min(common_counts)}")
        print(f"   - Entries with 0 common items: {sum(1 for c in common_counts if c == 0)}")
        print(f"   - Entries with 5 common items: {sum(1 for c in common_counts if c == 5)}")


if __name__ == '__main__':
    import sys
    
    # Allow custom file path as command line argument
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'results.json'
    
    update_results_json(file_path)

