#!/usr/bin/env python3

"""Script to check if extract_features_24vp.py has completed successfully.
Compares the number of records in the output TSV file with the expected number of viewpoints.
"""

import os
import json
import csv
import sys

# Increase CSV field size limit to handle large base64-encoded features
csv.field_size_limit(sys.maxsize)

# Configuration (should match extract_features_24vp.py)
GRAPHS = "connectivity/"
OUTFILE = "img_features/ResNet-152-places365_24vp.tsv"
TSV_FIELDNAMES = ["scanId", "viewpointId", "image_w", "image_h", "vfov", "features"]


def load_viewpointids():
    """Load all viewpoint IDs that should be processed (same logic as extract_features_24vp.py)"""
    viewpointIds = []
    scans_file = GRAPHS + "scans.txt"

    if not os.path.exists(scans_file):
        print(f"Error: {scans_file} not found")
        sys.exit(1)

    with open(scans_file) as f:
        scans = [scan.strip() for scan in f.readlines()]

    for scan in scans:
        connectivity_file = GRAPHS + scan + "_connectivity.json"
        if not os.path.exists(connectivity_file):
            print(f"Warning: {connectivity_file} not found, skipping scan {scan}")
            continue

        with open(connectivity_file) as j:
            data = json.load(j)
            for item in data:
                if item["included"]:
                    viewpointIds.append((scan, item["image_id"]))
    # print(viewpointIds)
    return viewpointIds


def count_tsv_records(tsv_file):
    """Count the number of records in the TSV file"""
    if not os.path.exists(tsv_file):
        return 0, set()

    count = 0
    processed_viewpoints = set()

    try:
        with open(tsv_file, "r") as f:
            reader = csv.DictReader(f, delimiter="\t", fieldnames=TSV_FIELDNAMES)
            for row in reader:
                count += 1
                scanId = row.get("scanId", "")
                viewpointId = row.get("viewpointId", "")
                if scanId and viewpointId:
                    processed_viewpoints.add((scanId, viewpointId))
    except Exception as e:
        print(f"Error reading TSV file: {e}")
        return 0, set()

    return count, processed_viewpoints


def check_progress():
    """Check extraction progress"""
    print("=" * 60)
    print("Checking extraction progress for extract_features_24vp.py")
    print("=" * 60)
    print()

    # Count records in TSV file first
    print(f"Checking output file: {OUTFILE}")
    if not os.path.exists(OUTFILE):
        print(f"  File does not exist!")
        print()
        print("Status: NOT STARTED")
        return

    file_size = os.path.getsize(OUTFILE)
    print(f"  File size: {file_size / (1024**3):.2f} GB")

    processed_count, processed_set = count_tsv_records(OUTFILE)
    print(f"  Records found: {processed_count}")
    print()

    # Try to load expected viewpoints
    print("Loading expected viewpoints from connectivity files...")
    try:
        expected_viewpoints = load_viewpointids()
        expected_count = len(expected_viewpoints)
        expected_set = set(expected_viewpoints)
        print(f"Expected viewpoints: {expected_count}")
        print()

        # Compare
        print("=" * 60)
        if processed_count == expected_count:
            print("✓ Status: COMPLETED")
            print(f"  All {expected_count} viewpoints have been processed.")
        elif processed_count == 0:
            print("✗ Status: NOT STARTED or FILE EMPTY")
        elif processed_count < expected_count:
            print("⚠ Status: IN PROGRESS")
            missing_count = expected_count - processed_count
            progress_pct = (processed_count / expected_count) * 100
            print(
                f"  Processed: {processed_count} / {expected_count} ({progress_pct:.2f}%)"
            )
            print(f"  Remaining: {missing_count} viewpoints")

            # Find missing viewpoints
            missing_viewpoints = expected_set - processed_set
            if missing_viewpoints:
                print()
                print("Missing viewpoints (first 10):")
                for i, (scan, vp) in enumerate(list(missing_viewpoints)[:10]):
                    print(f"  {i+1}. {scan} / {vp}")
                if len(missing_viewpoints) > 10:
                    print(f"  ... and {len(missing_viewpoints) - 10} more")
        else:
            print("⚠ Status: UNEXPECTED")
            print(
                f"  Found {processed_count} records, but only {expected_count} expected."
            )
            print("  File may contain duplicate entries or be corrupted.")

    except Exception as e:
        print(f"  Warning: Could not load expected viewpoints: {e}")
        print("  (Connectivity files may not be available)")
        print()
        print("=" * 60)
        print("Status: UNKNOWN (cannot compare with expected count)")
        print(f"  Found {processed_count} records in TSV file")
        print()
        print("To determine completion:")
        print("  1. Check if the file size is reasonable (should be ~several GB)")
        print("  2. Compare with the original ResNet-152-places365.tsv file size")
        print(
            "  3. Or run the extraction script again - it will show 'Loaded X viewpoints'"
        )
        print("     and you can compare X with the record count above")

    print("=" * 60)


if __name__ == "__main__":
    check_progress()
