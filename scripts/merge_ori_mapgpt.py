#!/usr/bin/env python3
"""
Merge MapGPT navigation outputs with original R2R data.

This script:
1. Reads MapGPT/nav_30/all_nav_outputs.json and data/R2R_MapGPT_72_scenes_processed.json
2. Filters R2R_MapGPT_72_scenes_processed.json to keep only instr_id that exist in all_nav_outputs.json
3. Updates the path field in R2R_MapGPT_72_scenes_processed.json based on vp values from all_nav_outputs.json
"""

import json
import os
import sys
from pathlib import Path


def load_json(file_path):
    """Load JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        sys.exit(1)


def save_json(data, file_path):
    """Save data to JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)


def merge_nav_outputs():
    """Main function to merge navigation outputs."""
    # File paths
    script_dir = Path(__file__).parent.parent
    nav_outputs_file = script_dir / "MapGPT" / "nav_30" / ".all_nav_outputs.json"
    nav_outputs_file72 = script_dir / "MapGPT" / "nav_30" / "all_nav_outputs.json"
    r2r_file = script_dir / "data" / "R2R_MapGPT_72_scenes_processed.json"
    output_file = script_dir / "data" / "R2R_MapGPT_30_scenes_processed_merged_1_2.json"

    print(f"Loading navigation outputs from: {nav_outputs_file}")
    all_nav_outputs = load_json(nav_outputs_file)

    print(f"Loading R2R data from: {r2r_file}")
    r2r_data = load_json(r2r_file)

    # Get set of instr_ids that exist in all_nav_outputs
    nav_instr_ids = set(all_nav_outputs.keys())
    print(f"Found {len(nav_instr_ids)} instr_ids in navigation outputs")

    # Filter and update R2R data
    filtered_r2r_data = []
    updated_count = 0
    skipped_count = 0

    for item in r2r_data:
        instr_id = item.get("instr_id")

        if instr_id not in nav_instr_ids:
            skipped_count += 1
            continue

        # Get navigation data for this instr_id
        nav_data = all_nav_outputs[instr_id]

        # Convert string keys to integers if needed
        if isinstance(nav_data, dict):
            converted_nav_data = {}
            for key, value in nav_data.items():
                try:
                    converted_nav_data[int(key)] = value
                except (ValueError, TypeError):
                    converted_nav_data[key] = value
            nav_data = converted_nav_data

        # Build path from vp values in chronological order (0, 1, 2, ..., t-1)
        # The path is constructed from vp values at each time step
        new_path = []
        time_steps = sorted([k for k in nav_data.keys() if isinstance(k, int)])

        for t in time_steps:
            vp = nav_data[t].get("vp")
            if vp:
                new_path.append(vp)

        # Update the item
        item["path"] = new_path
        filtered_r2r_data.append(item)
        updated_count += 1

        if updated_count % 10 == 0:
            print(f"Processed {updated_count} items...")

    print(f"\nSummary:")
    print(f"  Total items in R2R data: {len(r2r_data)}")
    print(f"  Items with matching instr_id: {updated_count}")
    print(f"  Items skipped (no match): {skipped_count}")

    # Save merged data
    print(f"\nSaving merged data to: {output_file}")
    save_json(filtered_r2r_data, output_file)
    print("Done!")

    return output_file


if __name__ == "__main__":
    merge_nav_outputs()
