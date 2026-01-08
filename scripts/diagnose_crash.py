#!/usr/bin/env python3

"""Diagnostic script to check if extract_features_24vp.py caused server crash.
Analyzes disk usage, file sizes, and potential issues."""

import os
import sys
import math

# Configuration
OUTFILE = "img_features/ResNet-152-places365_24vp.tsv"
FEATURE_SIZE = 2048
VIEWPOINT_SIZE = 24  # 3 heights * 8 horizontal views


def estimate_file_size(num_viewpoints):
    """Estimate TSV file size for given number of viewpoints"""
    # Each viewpoint has:
    # - scanId: ~20 bytes
    # - viewpointId: ~40 bytes  
    # - image_w, image_h, vfov: ~20 bytes
    # - features: base64 encoded (24 * 2048 * 4 bytes) / 3 * 4/3 ≈ 262144 bytes
    # Total per line: ~262200 bytes ≈ 256 KB
    
    bytes_per_viewpoint = 262200
    total_bytes = num_viewpoints * bytes_per_viewpoint
    return total_bytes


def check_disk_usage():
    """Check disk usage and file sizes"""
    print("=" * 70)
    print("Diagnosing potential server crash from extract_features_24vp.py")
    print("=" * 70)
    print()
    
    # Check output file
    print("1. Checking output file:")
    if os.path.exists(OUTFILE):
        file_size = os.path.getsize(OUTFILE)
        file_size_gb = file_size / (1024**3)
        print(f"   ✓ File exists: {OUTFILE}")
        print(f"   File size: {file_size_gb:.2f} GB ({file_size / (1024**2):.1f} MB)")
        
        # Estimate how many viewpoints processed
        estimated_viewpoints = file_size / 262200
        print(f"   Estimated viewpoints processed: ~{int(estimated_viewpoints)}")
    else:
        print(f"   ✗ File not found: {OUTFILE}")
        file_size = 0
    
    print()
    
    # Check img_features directory
    print("2. Checking img_features directory:")
    img_features_dir = os.path.dirname(OUTFILE)
    if os.path.exists(img_features_dir):
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(img_features_dir):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    size = os.path.getsize(filepath)
                    total_size += size
                    file_count += 1
                except:
                    pass
        
        print(f"   Total files: {file_count}")
        print(f"   Total size: {total_size / (1024**3):.2f} GB")
        
        # Check for large files
        print()
        print("   Large files (>100MB):")
        large_files = []
        for root, dirs, files in os.walk(img_features_dir):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    size = os.path.getsize(filepath)
                    if size > 100 * 1024 * 1024:  # > 100MB
                        large_files.append((filepath, size))
                except:
                    pass
        
        large_files.sort(key=lambda x: x[1], reverse=True)
        for filepath, size in large_files[:10]:
            print(f"     {os.path.basename(filepath)}: {size / (1024**2):.1f} MB")
    else:
        print(f"   ✗ Directory not found: {img_features_dir}")
    
    print()
    
    # Memory analysis
    print("3. Memory requirements:")
    print("   According to code comments:")
    print("   - BATCH_SIZE = 4 requires ~11GB GPU memory")
    print("   - Each viewpoint processes 24 images (640x480)")
    print("   - Feature extraction uses ResNet-152 model")
    print()
    print("   Potential issues:")
    print("   ⚠ If GPU memory insufficient, may cause OOM (Out of Memory)")
    print("   ⚠ OOM can cause process to be killed by system")
    print("   ⚠ This would explain crash even if disk not full")
    
    print()
    
    # Disk space analysis
    print("4. Disk space analysis:")
    print("   Estimated file sizes:")
    
    # Typical R2R dataset has ~10,000-15,000 viewpoints
    for num_vp in [5000, 10000, 15000, 20000]:
        est_size = estimate_file_size(num_vp)
        print(f"   - {num_vp} viewpoints: ~{est_size / (1024**3):.2f} GB")
    
    print()
    print("   If you have ~10,000 viewpoints:")
    print("   - Expected file size: ~2.5 GB")
    print("   - With 6GB remaining, should be fine")
    print("   - BUT: If writing fails/corrupts, may create large temp files")
    
    print()
    
    # Recommendations
    print("5. Recommendations:")
    print("   ✓ Check system logs for OOM kills:")
    print("     $ dmesg | grep -i 'killed process'")
    print("     $ journalctl -k | grep -i oom")
    print()
    print("   ✓ Check if process is still running:")
    print("     $ ps aux | grep extract_features_24vp")
    print()
    print("   ✓ Check disk space:")
    print("     $ df -h")
    print()
    print("   ✓ Check for temporary files:")
    print("     $ find /tmp -name '*extract*' -o -name '*ResNet*' 2>/dev/null")
    print()
    print("   ✓ If file exists but incomplete:")
    print("     - Run: python scripts/check_extraction_progress.py")
    print("     - Resume extraction by modifying script to skip processed viewpoints")
    
    print()
    print("=" * 70)
    print("Conclusion:")
    print("=" * 70)
    
    if file_size > 0:
        if file_size < 1024**3:  # < 1GB
            print("⚠ File is relatively small - extraction likely incomplete")
            print("  Server crash was probably NOT due to disk space")
            print("  More likely: Memory (OOM) or other system issue")
        elif file_size < 5 * 1024**3:  # < 5GB
            print("✓ File size is reasonable")
            print("  Server crash unlikely due to disk space (6GB remaining)")
            print("  More likely causes:")
            print("   1. Out of Memory (OOM) - check dmesg/journalctl")
            print("   2. GPU memory exhaustion")
            print("   3. System update/reboot")
            print("   4. Other processes consuming resources")
        else:
            print("⚠ File is very large (>5GB)")
            print("  May have contributed to disk pressure")
            print("  But with 6GB remaining, shouldn't cause crash")
    else:
        print("✗ No output file found")
        print("  Program may not have started or crashed immediately")
        print("  Check logs and system messages")
    
    print()


if __name__ == "__main__":
    check_disk_usage()

