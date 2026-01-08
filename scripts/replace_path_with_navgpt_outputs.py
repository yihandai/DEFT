#!/usr/bin/env python3
"""
Script to replace path waypoints in R2R_MapGPT_30_scenes_processed_merged.json
with NavGPT outputs from all_nav_outputs.json.

The first output (t=0) corresponds to the second waypoint in the path (index 1),
the second output (t=1) corresponds to the third waypoint (index 2), etc.
The first waypoint (index 0) remains unchanged.
"""

import json
import os
import sys


def replace_paths_with_navgpt_outputs(
    nav_outputs_file,
    input_paths_file,
    output_file,
):
    """
    Replace path waypoints with NavGPT outputs.
    
    Args:
        nav_outputs_file: Path to all_nav_outputs.json
        input_paths_file: Path to R2R_MapGPT_30_scenes_processed_merged.json
        output_file: Path to output file R2R_NavGPT_30_scenes_processed_merged.json
    """
    # Load NavGPT outputs
    print(f"Loading NavGPT outputs from {nav_outputs_file}...")
    with open(nav_outputs_file, "r") as f:
        nav_outputs = json.load(f)
    
    # Load input paths
    print(f"Loading input paths from {input_paths_file}...")
    with open(input_paths_file, "r") as f:
        input_paths = json.load(f)
    
    # Process each path entry
    updated_paths = []
    skipped = []
    
    for entry in input_paths:
        instr_id = entry["instr_id"]
        
        # Check if we have NavGPT outputs for this instruction
        if instr_id not in nav_outputs:
            print(f"Warning: No NavGPT outputs found for {instr_id}, keeping original path")
            updated_paths.append(entry)
            skipped.append(instr_id)
            continue
        
        # Get the original path
        original_path = entry["path"].copy()
        nav_data = nav_outputs[instr_id]
        
        # Build new path: keep first waypoint, replace rest with NavGPT outputs
        new_path = [original_path[0]]  # Keep first waypoint (index 0)
        
        # Get all time steps, sorted by time
        time_steps = sorted([int(k) for k in nav_data.keys() if k.isdigit()])
        
        # Replace waypoints starting from index 1 with NavGPT outputs
        # t=0 -> index 1, t=1 -> index 2, etc.
        for t in time_steps:
            t_str = str(t)
            if t_str not in nav_data:
                continue
            
            viewpoint_id = nav_data[t_str].get("viewpoint_id")
            
            # Skip if viewpoint_id is None or empty (stop action)
            if not viewpoint_id:
                print(f"Warning: Empty viewpoint_id at t={t} for {instr_id}, stopping path")
                break
            
            # Calculate target index: t=0 -> index 1, t=1 -> index 2, etc.
            target_index = t + 1
            
            # Extend path to target_index if necessary (use original path values if available)
            while len(new_path) <= target_index:
                if len(new_path) < len(original_path):
                    # Use original path value if available
                    new_path.append(original_path[len(new_path)])
                else:
                    # Path is longer than original, add placeholder (will be replaced below)
                    new_path.append(None)
            
            # Replace the waypoint at target_index
            new_path[target_index] = viewpoint_id
        
        # Create updated entry
        updated_entry = entry.copy()
        updated_entry["path"] = new_path
        updated_paths.append(updated_entry)
        
        print(f"Processed {instr_id}: {len(original_path)} -> {len(new_path)} waypoints")
    
    # Save updated paths
    print(f"\nSaving updated paths to {output_file}...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(updated_paths, f, indent=2)
    
    print(f"\nDone! Processed {len(updated_paths)} entries")
    if skipped:
        print(f"Skipped {len(skipped)} entries (no NavGPT outputs): {', '.join(skipped)}")
    
    return updated_paths


def main():
    # Default file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    nav_outputs_file = os.path.join(script_dir, "all_nav_outputs.json")
    input_paths_file = os.path.join(project_root, "data", "R2R_MapGPT_30_scenes_processed_merged.json")
    output_file = os.path.join(project_root, "data", "R2R_NavGPT_30_scenes_processed_merged.json")
    
    # Allow command line arguments to override defaults
    if len(sys.argv) >= 2:
        nav_outputs_file = sys.argv[1]
    if len(sys.argv) >= 3:
        input_paths_file = sys.argv[2]
    if len(sys.argv) >= 4:
        output_file = sys.argv[3]
    
    # Check if files exist
    if not os.path.exists(nav_outputs_file):
        print(f"Error: NavGPT outputs file not found: {nav_outputs_file}")
        sys.exit(1)
    
    if not os.path.exists(input_paths_file):
        print(f"Error: Input paths file not found: {input_paths_file}")
        sys.exit(1)
    
    # Run replacement
    replace_paths_with_navgpt_outputs(
        nav_outputs_file,
        input_paths_file,
        output_file,
    )


if __name__ == "__main__":
    main()

