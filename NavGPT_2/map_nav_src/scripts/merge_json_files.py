#!/usr/bin/env python3
"""
Merge two JSON files:
- file1: R2R_train_enc.json (with instructions array and instr_encodings array)
- file2: r2r_subset_instr_level_10percent.json (with single instruction and instr_id)

Output: R2R_train_enc_10perc.json (same format as file2, but with instr_encoding)
"""

import json
import os
import sys


def merge_json_files(file1_path, file2_path, output_path):
    """
    Merge two JSON files according to the requirements.

    Args:
        file1_path: Path to R2R_train_enc.json
        file2_path: Path to r2r_subset_instr_level_10percent.json
        output_path: Path to output file R2R_train_enc_10perc.json
    """
    # Load file1
    print(f"Loading file1: {file1_path}")
    with open(file1_path, "r", encoding="utf-8") as f:
        file1_data = json.load(f)
    print(f"Loaded {len(file1_data)} items from file1")

    # Load file2
    print(f"Loading file2: {file2_path}")
    with open(file2_path, "r", encoding="utf-8") as f:
        file2_data = json.load(f)
    print(f"Loaded {len(file2_data)} items from file2")

    # Extract instr_ids from file2
    file2_instr_ids = set()
    for item in file2_data:
        if "instr_id" in item:
            file2_instr_ids.add(item["instr_id"])
    print(f"Found {len(file2_instr_ids)} unique instr_ids in file2")

    # Convert file1 to file2 format
    # Each item in file1 has multiple instructions, we need to create one object per instruction
    converted_data = []
    for item in file1_data:
        path_id = item["path_id"]
        instructions = item.get("instructions", [])
        instr_encodings = item.get("instr_encodings", [])

        # Ensure we have the same number of instructions and encodings
        num_instructions = len(instructions)
        num_encodings = len(instr_encodings)

        if num_instructions != num_encodings:
            print(
                f"Warning: path_id {path_id} has {num_instructions} instructions but {num_encodings} encodings"
            )
            # Use the minimum to avoid index errors
            num_items = min(num_instructions, num_encodings)
        else:
            num_items = num_instructions

        # Create one object per instruction
        for idx in range(num_items):
            instr_id = f"{path_id}_{idx}"

            # Only include if this instr_id is in file2
            if instr_id in file2_instr_ids:
                new_item = {
                    "distance": item.get("distance"),
                    "scan": item.get("scan"),
                    "path": item.get("path", []),
                    "path_id": path_id,
                    "heading": item.get("heading"),
                    "instr_id": instr_id,
                    "instruction": instructions[idx],
                    "instr_encoding": instr_encodings[idx],
                }
                converted_data.append(new_item)

    print(f"Converted {len(converted_data)} items that match file2 instr_ids")

    # Save output
    print(f"Saving to: {output_path}")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved {len(converted_data)} items to {output_path}")

    # Print some statistics
    print("\nStatistics:")
    print(f"  Total items in file1: {len(file1_data)}")
    print(f"  Total items in file2: {len(file2_data)}")
    print(f"  Unique instr_ids in file2: {len(file2_instr_ids)}")
    print(f"  Matched items in output: {len(converted_data)}")

    # Check for missing instr_ids
    matched_instr_ids = {item["instr_id"] for item in converted_data}
    missing_instr_ids = file2_instr_ids - matched_instr_ids
    if missing_instr_ids:
        print(
            f"\nWarning: {len(missing_instr_ids)} instr_ids from file2 were not found in file1"
        )
        print(f"  Examples: {list(missing_instr_ids)[:5]}")


if __name__ == "__main__":
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    navgpt2_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # File paths
    file1_path = os.path.join(
        navgpt2_root, "datasets", "R2R", "annotations", "R2R_train_enc.json"
    )
    file2_path = os.path.join(
        navgpt2_root,
        "..",
        "NavGPT",
        "datasets",
        "R2R",
        "annotations",
        "r2r_subset_instr_level_10percent.json",
    )
    output_path = os.path.join(
        navgpt2_root, "datasets", "R2R", "annotations", "R2R_train_enc_10perc.json"
    )

    # Allow command line arguments to override
    if len(sys.argv) >= 2:
        file1_path = sys.argv[1]
    if len(sys.argv) >= 3:
        file2_path = sys.argv[2]
    if len(sys.argv) >= 4:
        output_path = sys.argv[3]

    # Check if files exist
    if not os.path.exists(file1_path):
        print(f"Error: file1 not found: {file1_path}")
        sys.exit(1)

    if not os.path.exists(file2_path):
        print(f"Error: file2 not found: {file2_path}")
        sys.exit(1)

    # Run merge
    merge_json_files(file1_path, file2_path, output_path)
