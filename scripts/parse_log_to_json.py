#!/usr/bin/env python3
"""
Parse log file and save results in JSON format matching the structure:
results[instr_id]["mask"][str(mask_perc)][mode][t] = consistency_score
"""

import re
import json
import os
import sys


def parse_log_file(log_file_path):
    """
    Parse the log file and extract:
    - instr_id: instruction ID (e.g., "3965_2")
    - t: time step (e.g., 0, 1, 2, ...)
    - mode: "ins" for Insertion, "del" for Deletion
    - mask_perc: mask percentage (0.25, 0.5, 0.75)
    - consistency_score: sample value (0 or 1)
    """
    results = {}

    # Patterns to match different line types
    instr_id_pattern = re.compile(r"^(\d+_\d+)$")  # Matches "3965_2"
    time_pattern = re.compile(r"^t:\s+(\d+)$")  # Matches "t:  0"
    insertion_pattern = re.compile(
        r"Insertion sample:\s+([01])\.\s+Over-all:.*?for mask percentage:\s+(0\.25|0\.5|0\.75)"
    )
    deletion_pattern = re.compile(
        r"Deletion sample:\s+([01])\.\s+Over-all:.*?for mask percentage:\s+(0\.25|0\.5|0\.75)"
    )

    current_instr_id = None
    current_t = None

    try:
        with open(log_file_path, "r") as f:
            for line in f:
                line = line.strip()

                # Check for instruction ID
                instr_match = instr_id_pattern.match(line)
                if instr_match:
                    current_instr_id = instr_match.group(1)
                    if current_instr_id not in results:
                        results[current_instr_id] = {
                            "mask": {
                                "0.25": {"ins": {}, "del": {}},
                                "0.5": {"ins": {}, "del": {}},
                                "0.75": {"ins": {}, "del": {}},
                            }
                        }
                    current_t = None  # Reset time step when new instr_id appears
                    continue

                # Check for time step
                time_match = time_pattern.match(line)
                if time_match:
                    current_t = time_match.group(1)
                    continue

                # Check for Insertion pattern
                insertion_match = insertion_pattern.search(line)
                if insertion_match and current_instr_id and current_t is not None:
                    sample_value = int(insertion_match.group(1))
                    mask_perc = insertion_match.group(2)

                    if current_instr_id in results:
                        results[current_instr_id]["mask"][mask_perc]["ins"][
                            current_t
                        ] = sample_value
                    continue

                # Check for Deletion pattern
                deletion_match = deletion_pattern.search(line)
                if deletion_match and current_instr_id and current_t is not None:
                    sample_value = int(deletion_match.group(1))
                    mask_perc = deletion_match.group(2)

                    if current_instr_id in results:
                        results[current_instr_id]["mask"][mask_perc]["del"][
                            current_t
                        ] = sample_value
                    continue

    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    return results


def save_results(results, output_file):
    """Save results to JSON file."""
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False


def print_summary(results):
    """Print a summary of the parsed results."""
    if results is None:
        return

    print("\n" + "=" * 70)
    print("Parsing Summary")
    print("=" * 70)

    total_instr_ids = len(results)
    print(f"Total instruction IDs: {total_instr_ids}")

    # Count entries per mask percentage and mode
    mask_counts = {
        "0.25": {"ins": 0, "del": 0},
        "0.5": {"ins": 0, "del": 0},
        "0.75": {"ins": 0, "del": 0},
    }

    for instr_id, data in results.items():
        for mask_perc in ["0.25", "0.5", "0.75"]:
            if mask_perc in data["mask"]:
                mask_counts[mask_perc]["ins"] += len(data["mask"][mask_perc]["ins"])
                mask_counts[mask_perc]["del"] += len(data["mask"][mask_perc]["del"])

    print("\nEntries per mask percentage:")
    for mask_perc in ["0.25", "0.5", "0.75"]:
        print(f"  {mask_perc}:")
        print(f"    Insertion (ins): {mask_counts[mask_perc]['ins']}")
        print(f"    Deletion (del): {mask_counts[mask_perc]['del']}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    import os

    # Default log file path (in project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Default paths
    log_file = os.path.join(
        # project_root, "logs/mapgpt_feature/random_v3_2025_12_15_phase23_r.log"
        project_root,
        "logs/mapgpt_feature/ensemble_v3_2025_12_15_phase3.log",
    )
    output_file = os.path.join(
        project_root, "scripts", "ensemble_v3_2025_12_15_phase3.json"
    )

    # Allow command line arguments
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    print(f"Parsing log file: {log_file}")
    print(f"Output file: {output_file}")
    print()

    results = parse_log_file(log_file)

    if results:
        print_summary(results)
        save_results(results, output_file)
    else:
        print("Failed to parse log file.")
