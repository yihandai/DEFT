#!/usr/bin/env python3

"""Script to remove header row from TSV file if it exists."""

import os
import sys
import csv
import shutil
import tempfile

TSV_FIELDNAMES = ["scanId", "viewpointId", "image_w", "image_h", "vfov", "features"]
OUTFILE = "img_features/ResNet-152-places365_24vp.tsv"


def remove_header(tsv_file):
    """Remove header row from TSV file if it exists."""
    if not os.path.exists(tsv_file):
        print(f"Error: File {tsv_file} does not exist!")
        return False

    # Check file size
    file_size = os.path.getsize(tsv_file)
    print(f"File size: {file_size / (1024**3):.2f} GB")

    # Read first line to check if it's a header
    with open(tsv_file, "r") as f:
        first_line = f.readline().strip()
        first_line_fields = first_line.split("\t")

    # Check if first line is header
    is_header = (
        len(first_line_fields) == len(TSV_FIELDNAMES)
        and first_line_fields[0] == TSV_FIELDNAMES[0]
        and first_line_fields[1] == TSV_FIELDNAMES[1]
    )

    if not is_header:
        print("No header found in file. File already has no header.")
        return True

    print("Header found. Removing header row...")
    print(f"Header: {first_line[:100]}...")

    # Create temporary file
    temp_file = tsv_file + ".tmp"
    
    try:
        # Copy file without first line
        # Use binary mode for better performance on large files
        with open(tsv_file, "rb") as infile, open(temp_file, "wb") as outfile:
            # Skip first line (header) - read until first newline
            first_line_bytes = infile.readline()
            # Copy remaining content
            shutil.copyfileobj(infile, outfile)
        
        # Replace original file with temp file
        shutil.move(temp_file, tsv_file)
        
        # Verify by checking first line of new file
        with open(tsv_file, "r") as f:
            new_first_line = f.readline().strip()
            new_first_fields = new_first_line.split("\t")
            # Check if it's NOT a header (should be data)
            is_still_header = (
                len(new_first_fields) == len(TSV_FIELDNAMES)
                and new_first_fields[0] == TSV_FIELDNAMES[0]
                and new_first_fields[1] == TSV_FIELDNAMES[1]
            )
            
            if is_still_header:
                print("Warning: First line is still a header. Something went wrong.")
                return False
            else:
                print("✓ Header successfully removed!")
                print(f"New first line (data): {new_first_line[:100]}...")
                return True
                
    except Exception as e:
        print(f"Error removing header: {e}")
        import traceback
        traceback.print_exc()
        # Clean up temp file if it exists
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Removing header from TSV file")
    print("=" * 60)
    print()
    
    if len(sys.argv) > 1:
        tsv_file = sys.argv[1]
    else:
        tsv_file = OUTFILE
    
    print(f"Target file: {tsv_file}")
    print()
    
    success = remove_header(tsv_file)
    
    if success:
        print()
        print("=" * 60)
        print("✓ Done!")
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print("✗ Failed!")
        print("=" * 60)
        sys.exit(1)

