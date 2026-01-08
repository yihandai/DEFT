#!/usr/bin/env python3
"""
Parse inference_most.log to extract most frequent inference action at each time step t.
"""

import re
import json

def parse_most_frequent(log_path):
    """
    Parse the log file and extract most frequent inference action at each time step t.
    
    Returns a list of most frequent values, one for each statistics block.
    """
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Pattern to match "most frequent: value" (case insensitive)
    most_frequent_pattern = re.compile(r"most frequent:\s*([^\s(]+)", re.IGNORECASE)
    
    result = []
    in_statistics = False
    current_block = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if we're entering a statistics block
        if "Statistics: Most frequent inference action at each time step t" in line:
            in_statistics = True
            current_block = []
            i += 1
            continue
        
        # Check if we're leaving the statistics block
        if in_statistics and line.startswith("new rollout"):
            # Save the current block
            if current_block:
                result.append(current_block)
            in_statistics = False
            current_block = []
            i += 1
            continue
        
        # If we're in statistics block, look for "most frequent:" lines
        # Format: "  Most frequent: value (appeared ...)" or "  most frequent: value (appeared ...)"
        if in_statistics and ("most frequent:" in line.lower() or "Most frequent:" in line):
            match = most_frequent_pattern.search(line)
            if match:
                value = match.group(1).strip()
                # Convert "none" to None
                if value.lower() == 'none':
                    value = None
                current_block.append(value)
        
        i += 1
    
    # Don't forget the last block if file ends without "new rollout"
    if in_statistics and current_block:
        result.append(current_block)
    
    return result

def main():
    log_path = "/Users/ian/Project/VLN/Recurrent-VLN-BERT/scripts/inference_most.log"
    output_path = "/Users/ian/Project/VLN/Recurrent-VLN-BERT/scripts/inference_most_parsed.json"
    
    print("Parsing log file...")
    result = parse_most_frequent(log_path)
    
    print(f"Found {len(result)} statistics blocks")
    for i, block in enumerate(result):
        print(f"  Block {i}: {len(block)} time steps")
    
    # Flatten the result into a single list (all most frequent values in order)
    flattened = []
    for block in result:
        flattened.extend(block)
    
    print(f"\nTotal most frequent values: {len(flattened)}")
    print(f"None values: {sum(1 for x in flattened if x is None)}")
    
    # Save as JSON
    print(f"\nWriting results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(flattened, f, indent=2, ensure_ascii=False)
    
    print("Done!")
    
    # Also print first few values for verification
    print(f"\nFirst 10 values: {flattened[:10]}")

if __name__ == "__main__":
    main()

