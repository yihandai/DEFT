#!/usr/bin/env python3
"""
Filter val72_navgpt2.json to keep only trajectories with instr_id
that appear in NavGPT_30_scenes_processed_merged.json
"""

import json
import os
import sys
from pathlib import Path


def load_json(file_path):
    """Load JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path):
    """Save data to JSON file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} entries to {file_path}")


def extract_instr_ids(data):
    """Extract all instr_id from JSON data."""
    instr_ids = set()
    for item in data:
        if "instr_id" in item:
            instr_ids.add(item["instr_id"])
    return instr_ids


def filter_by_instr_ids(data, instr_ids):
    """Filter data to keep only items with instr_id in instr_ids set."""
    filtered = []
    for item in data:
        if "instr_id" in item and item["instr_id"] in instr_ids:
            filtered.append(item)
    return filtered


def main():
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define file paths
    source_file = (
        project_root
        / "NavGPT"
        / "datasets"
        / "R2R"
        / "annotations"
        / "NavGPT_30_scenes_processed_merged.json"
    )
    target_file = (
        project_root
        / "NavGPT_2"
        / "datasets"
        / "R2R"
        / "annotations"
        / "val72_navgpt2.json"
    )
    output_file = (
        project_root
        / "NavGPT_2"
        / "datasets"
        / "R2R"
        / "annotations"
        / "NavGPT2_30_scenes_processed_merged.json"
    )  # Overwrite the target file

    # Alternative paths if the above don't exist
    if not source_file.exists():
        # Try alternative path in data directory
        alt_source = (
            project_root / "data" / "R2R_NavGPT_30_scenes_processed_merged.json"
        )
        if alt_source.exists():
            source_file = alt_source
            print(f"Using alternative source path: {source_file}")
        else:
            print(f"Error: Source file not found at {source_file} or {alt_source}")
            sys.exit(1)

    if not target_file.exists():
        print(f"Error: Target file not found at {target_file}")
        print("Please ensure the file exists before running this script.")
        sys.exit(1)

    print(f"Loading source file: {source_file}")
    source_data = load_json(source_file)

    print(f"Loading target file: {target_file}")
    target_data = load_json(target_file)

    # Extract instr_ids from source file
    print("Extracting instr_ids from source file...")
    source_instr_ids = extract_instr_ids(source_data)
    print(f"Found {len(source_instr_ids)} unique instr_ids in source file")

    # Filter target data
    print("Filtering target data...")
    filtered_data = filter_by_instr_ids(target_data, source_instr_ids)
    print(
        f"Filtered {len(filtered_data)} entries from {len(target_data)} total entries"
    )

    # Check for missing instr_ids
    target_instr_ids = extract_instr_ids(target_data)
    missing_ids = source_instr_ids - target_instr_ids
    if missing_ids:
        print(
            f"Warning: {len(missing_ids)} instr_ids from source file not found in target file"
        )
        print(f"First 10 missing IDs: {list(missing_ids)[:10]}")

    # Save filtered data
    print(f"Saving filtered data to {output_file}...")
    save_json(filtered_data, output_file)

    print("Done!")


if __name__ == "__main__":
    main()
