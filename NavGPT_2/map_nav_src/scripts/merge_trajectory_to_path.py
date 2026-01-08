#!/usr/bin/env python3
"""
Merge two JSON files:
- file1: r2r_subset_instr_level_20percent.json (with path field)
- file2: submit_r2r_subset_instr_level_20percent.json (with trajectory field)

Replace file1's path with file2's trajectory (converted to path format).
When a time step has multiple viewpoints, take the last one (-1).
"""

import json
import os
import sys


def trajectory_to_path(trajectory):
    """
    Convert trajectory (nested list) to path (flat list).
    For each time step, take the last viewpoint if multiple viewpoints exist.

    Args:
        trajectory: List of lists, each inner list contains viewpoint(s) for that time step

    Returns:
        path: Flat list of viewpoints
    """
    path = []
    for time_step in trajectory:
        if isinstance(time_step, list) and len(time_step) > 0:
            # Take the last viewpoint in this time step
            path.append(time_step[-1])
        elif isinstance(time_step, str):
            # If it's a single string (shouldn't happen based on format, but handle it)
            path.append(time_step)
    return path


def merge_trajectory_to_path(file1_path, file2_path, output_path):
    """
    Merge two JSON files by replacing file1's path with file2's trajectory.

    Args:
        file1_path: Path to file1 (with path field)
        file2_path: Path to file2 (with trajectory field)
        output_path: Path to output file
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

    # Create a mapping from instr_id to trajectory in file2
    file2_trajectories = {}
    for item in file2_data:
        if "instr_id" in item and "trajectory" in item:
            file2_trajectories[item["instr_id"]] = item["trajectory"]
    print(f"Found {len(file2_trajectories)} items with trajectory in file2")

    # Replace path in file1 with trajectory from file2
    updated_count = 0
    missing_count = 0
    for item in file1_data:
        instr_id = item.get("instr_id")
        if instr_id in file2_trajectories:
            # Convert trajectory to path
            trajectory = file2_trajectories[instr_id]
            path = trajectory_to_path(trajectory)
            # Replace path
            item["path"] = path
            updated_count += 1
        else:
            missing_count += 1
            if missing_count <= 5:  # Print first 5 missing items
                print(
                    f"Warning: instr_id {instr_id} not found in file2, keeping original path"
                )

    print(f"\nUpdated {updated_count} items with new paths from file2")
    if missing_count > 0:
        print(
            f"Warning: {missing_count} items in file1 were not found in file2 (kept original paths)"
        )

    # Save output
    print(f"\nSaving to: {output_path}")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(file1_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved {len(file1_data)} items to {output_path}")

    # Print some statistics
    print("\nStatistics:")
    print(f"  Total items in file1: {len(file1_data)}")
    print(f"  Total items in file2: {len(file2_data)}")
    print(f"  Items with trajectory in file2: {len(file2_trajectories)}")
    print(f"  Updated items: {updated_count}")
    print(f"  Items not found in file2: {missing_count}")


if __name__ == "__main__":
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    navgpt2_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    project_root = os.path.abspath(os.path.join(navgpt2_root, ".."))

    # File paths
    file1_path = os.path.join(
        # project_root,
        navgpt2_root,
        # "NavGPT",
        "datasets",
        "R2R",
        "annotations",
        "r2r_subset_instr_level_val72.json",
    )
    file2_path = os.path.join(
        navgpt2_root,
        "datasets",
        "R2R",
        "exprs_map",
        "finetune",
        "NavGPT2-XL-seed.0-bs2",
        "preds",
        "submit_r2r_subset_instr_level_val72.json",
    )
    output_path = os.path.join(
        os.path.dirname(file1_path),
        "r2r_subset_instr_level_val72_with_navgpt2_path.json",
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
    merge_trajectory_to_path(file1_path, file2_path, output_path)
