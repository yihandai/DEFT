#!/usr/bin/env python3
"""
Script to analyze random_v3_2025_12_15_phase23_r.log
Counts occurrences and '1' values for Insertion/Deletion at different mask percentages.
"""

import re
from collections import defaultdict


def analyze_log(log_file_path):
    """
    Analyze the log file and count:
    1. Total occurrences of each pattern
    2. Number of '1' values for each pattern
    """

    # Initialize counters
    # Structure: {pattern: {'total': count, 'ones': count}}
    counters = {
        "Insertion_0.25": {"total": 0, "ones": 0},
        "Insertion_0.5": {"total": 0, "ones": 0},
        "Insertion_0.75": {"total": 0, "ones": 0},
        "Deletion_0.25": {"total": 0, "ones": 0},
        "Deletion_0.5": {"total": 0, "ones": 0},
        "Deletion_0.75": {"total": 0, "ones": 0},
    }

    # Pattern to match: "Insertion sample: X. Over-all: Y for mask percentage: Z"
    # or "Deletion sample: X. Over-all: Y for mask percentage: Z"
    pattern = re.compile(
        r"(Insertion|Deletion)\s+sample:\s+([01])\.\s+Over-all:.*?for mask percentage:\s+(0\.25|0\.5|0\.75)"
    )

    try:
        with open(log_file_path, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    operation = match.group(1)  # 'Insertion' or 'Deletion'
                    sample_value = int(match.group(2))  # 0 or 1
                    mask_percentage = match.group(3)  # '0.25', '0.5', or '0.75'

                    # Create key for this pattern
                    key = f"{operation}_{mask_percentage}"

                    if key in counters:
                        counters[key]["total"] += 1
                        if sample_value == 1:
                            counters[key]["ones"] += 1

    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    return counters


def print_results(counters):
    """Print the analysis results in a formatted way."""
    if counters is None:
        return

    print("=" * 70)
    print("Log File Analysis Results")
    print("=" * 70)
    print()

    # Define display order
    display_order = [
        "Insertion_0.25",
        "Insertion_0.5",
        "Insertion_0.75",
        "Deletion_0.25",
        "Deletion_0.5",
        "Deletion_0.75",
    ]

    # Print header
    print(
        f"{'Pattern':<30} {'Total Count':<15} {'Ones Count':<15} {'Ones Percentage':<15}"
    )
    print("-" * 70)

    # Print results
    for key in display_order:
        if key in counters:
            total = counters[key]["total"]
            ones = counters[key]["ones"]
            percentage = (ones / total * 100) if total > 0 else 0

            # Format the key for display
            display_key = key.replace("_", " ").title()

            print(f"{display_key:<30} {total:<15} {ones:<15} {percentage:>6.2f}%")

    print()
    print("=" * 70)

    # Summary
    total_insertion = sum(
        counters[f"Insertion_{p}"]["total"] for p in ["0.25", "0.5", "0.75"]
    )
    total_deletion = sum(
        counters[f"Deletion_{p}"]["total"] for p in ["0.25", "0.5", "0.75"]
    )
    total_ones_insertion = sum(
        counters[f"Insertion_{p}"]["ones"] for p in ["0.25", "0.5", "0.75"]
    )
    total_ones_deletion = sum(
        counters[f"Deletion_{p}"]["ones"] for p in ["0.25", "0.5", "0.75"]
    )

    print("Summary:")
    print(f"  Total Insertion samples: {total_insertion}")
    print(f"  Total Insertion ones: {total_ones_insertion}")
    print(f"  Total Deletion samples: {total_deletion}")
    print(f"  Total Deletion ones: {total_ones_deletion}")
    print("=" * 70)


if __name__ == "__main__":
    import os
    import sys

    # Get the script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Default log file path (in project root)
    log_file = os.path.join(
        project_root, "./scripts/navgpt_feature_smdl_2025_12_27.log"
    )

    # Allow command line argument for custom log file
    if len(sys.argv) > 1:
        log_file = sys.argv[1]

    print(f"Analyzing log file: {log_file}")
    print()

    counters = analyze_log(log_file)
    print_results(counters)
