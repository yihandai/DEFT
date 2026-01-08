#!/usr/bin/env python3
"""
Merge MapGPT navigation outputs with original R2R data (version 2).

This script:
1. Reads MapGPT/nav_30/.all_nav_outputs.json (30 trajectories) to get instr_ids
2. Filters MapGPT/nav_30/all_nav_outputs.json (216 trajectories) to keep only the 30 matching instr_ids
3. Saves the filtered nav_outputs_file72 as all_nav_outputs_new.json
4. Updates the path field in R2R_MapGPT_72_scenes_processed.json based on vp values from filtered nav_outputs_file72
5. Outputs only the 30 matching trajectories to the output file
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


def merge_nav_outputs2():
    """Main function to merge navigation outputs (version 2)."""
    # File paths
    script_dir = Path(__file__).parent.parent
    nav_outputs_file = script_dir / "MapGPT" / "nav_30" / ".all_nav_outputs.json"
    nav_outputs_file72 = script_dir / "MapGPT" / "nav_30" / "all_nav_outputs.json"
    nav_outputs_file_new = script_dir / "MapGPT" / "nav_30" / "all_nav_outputs_new.json"
    r2r_file = script_dir / "data" / "R2R_MapGPT_72_scenes_processed.json"
    output_file = script_dir / "data" / "R2R_MapGPT_30_scenes_processed_merged_2.json"

    # Step 1: Load nav_outputs_file (30 trajectories) to get instr_ids for filtering
    print(f"Loading navigation outputs (30) from: {nav_outputs_file}")
    nav_outputs_30 = load_json(nav_outputs_file)
    nav_instr_ids = set(nav_outputs_30.keys())
    print(f"Found {len(nav_instr_ids)} instr_ids in navigation outputs (30)")

    # Step 2: Load nav_outputs_file72 (216 trajectories)
    print(f"Loading navigation outputs (216) from: {nav_outputs_file72}")
    all_nav_outputs72 = load_json(nav_outputs_file72)
    print(f"Found {len(all_nav_outputs72)} trajectories in nav_outputs_file72")

    # Step 3: Filter nav_outputs_file72 to keep only the 30 matching instr_ids
    filtered_nav_outputs72 = {}
    for instr_id in nav_instr_ids:
        if instr_id in all_nav_outputs72:
            filtered_nav_outputs72[instr_id] = all_nav_outputs72[instr_id]
        else:
            print(f"Warning: instr_id {instr_id} not found in nav_outputs_file72")

    print(f"Filtered nav_outputs_file72 to {len(filtered_nav_outputs72)} trajectories")

    # Save filtered nav_outputs_file72 as nav_outputs_file_new
    print(f"\nSaving filtered navigation outputs to: {nav_outputs_file_new}")
    save_json(filtered_nav_outputs72, nav_outputs_file_new)
    print(f"Saved {len(filtered_nav_outputs72)} trajectories to nav_outputs_file_new")

    # Step 4: Load R2R data
    print(f"Loading R2R data from: {r2r_file}")
    r2r_data = load_json(r2r_file)

    # Step 5: Update R2R data with paths from filtered nav_outputs_file72
    # Only keep the 30 matching trajectories in the output
    filtered_r2r_data = []
    updated_count = 0
    skipped_count = 0

    for item in r2r_data:
        instr_id = item.get("instr_id")

        if instr_id not in filtered_nav_outputs72:
            skipped_count += 1
            continue

        # Get navigation data for this instr_id from filtered nav_outputs_file72
        nav_data = filtered_nav_outputs72[instr_id]

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

        # Update the item with path from filtered nav_outputs_file72
        item["path"] = new_path
        filtered_r2r_data.append(item)
        updated_count += 1

        if updated_count % 10 == 0:
            print(f"Processed {updated_count} items...")

    print(f"\nSummary:")
    print(f"  Total items in R2R data: {len(r2r_data)}")
    print(
        f"  Items updated with paths from filtered nav_outputs_file72: {updated_count}"
    )
    print(f"  Items skipped (no match): {skipped_count}")
    print(f"  Output file will contain {len(filtered_r2r_data)} trajectories")

    # Save merged data (only the 30 matching trajectories)
    print(f"\nSaving merged data to: {output_file}")
    save_json(filtered_r2r_data, output_file)
    print("Done!")

    return output_file


if __name__ == "__main__":
    merge_nav_outputs2()
