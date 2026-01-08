#!/usr/bin/env python3
"""
Parse inference log file to extract prediction values organized by group, time step, and percentage.
"""

import re
import json
from collections import defaultdict

def find_group_structure(lines):
    """
    Find the structure of the first group by scanning for operations.
    A group starts from the first operation and ends when we see the same first operation again.
    Returns a list of (op_type, percentage) tuples representing the group structure.
    """
    prediction_pattern = re.compile(r"gt:.*?prediction:\s*\[(.*?)\]")
    operation_pattern = re.compile(r"(Insertion|Deletion).*?percentage:\s*(0\.25|0\.5|0\.75)")
    
    structure = []
    last_prediction = None
    first_operation = None
    found_first = False
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for prediction line
        pred_match = prediction_pattern.search(line)
        if pred_match:
            pred_value = pred_match.group(1).strip()
            if pred_value == 'None':
                pred_value = None
            else:
                pred_value = pred_value.strip("'\"")
            last_prediction = pred_value
            i += 1
            continue
        
        # Check for Insertion/Deletion line
        op_match = operation_pattern.search(line)
        if op_match and last_prediction is not None:
            op_type = op_match.group(1).lower()
            percentage = op_match.group(2)
            
            op_key = "ins" if op_type == "insertion" else "del"
            pct_key = percentage
            op_tuple = (op_key, pct_key)
            
            if not found_first:
                # This is the first operation - mark it as the group start
                first_operation = op_tuple
                found_first = True
                structure.append(op_tuple)
            else:
                # Add to current group structure only if it's different from the last one
                # This way we only record operation type changes, not each time step
                if not structure or op_tuple != structure[-1]:
                    structure.append(op_tuple)
                
                # Check if we've seen the first operation again (new group starting)
                # Only break if we've collected enough unique operations
                # A complete group should have at least 6 operations: ins0.25, del0.25, ins0.5, del0.5, ins0.75, del0.75
                if op_tuple == first_operation and len(structure) >= 6:
                    # We've completed one full group cycle
                    # Remove the last element as it's the start of the next group
                    structure.pop()
                    break
            
            last_prediction = None
            i += 1
            continue
        
        i += 1
    
    return structure, first_operation

def parse_log_file(log_path):
    """
    Parse the log file and extract predictions organized by:
    - operation type (ins/del)
    - percentage (0.25, 0.5, 0.75)
    
    All predictions for the same operation type and percentage are collected into a single list.
    
    Structure:
    {
      "ins": {
        "0.25": ["prediction1", "prediction2", ...],
        "0.5": ["prediction1", "prediction2", ...],
        "0.75": ["prediction1", "prediction2", ...]
      },
      "del": {
        "0.25": ["prediction1", "prediction2", ...],
        "0.5": ["prediction1", "prediction2", ...],
        "0.75": ["prediction1", "prediction2", ...]
      }
    }
    """
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Pattern to match prediction lines
    prediction_pattern = re.compile(r"gt:.*?prediction:\s*\[(.*?)\]")
    
    # Pattern to match Insertion/Deletion lines with percentage
    operation_pattern = re.compile(r"(Insertion|Deletion).*?percentage:\s*(0\.25|0\.5|0\.75)")
    
    # result[op_type][percentage] = [list of predictions]
    result = defaultdict(lambda: defaultdict(list))
    
    # Use a sentinel object to track if prediction has been set (allows None values)
    _SENTINEL = object()
    
    last_prediction = _SENTINEL  # Use a sentinel object to track if prediction has been set
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for prediction line
        pred_match = prediction_pattern.search(line)
        if pred_match:
            pred_value = pred_match.group(1).strip()
            if pred_value == 'None':
                pred_value = None
            else:
                pred_value = pred_value.strip("'\"")
            last_prediction = pred_value
            i += 1
            continue
        
        # Check for Insertion/Deletion line
        op_match = operation_pattern.search(line)
        if op_match and last_prediction is not _SENTINEL:
            op_type = op_match.group(1).lower()
            percentage = op_match.group(2)
            
            op_key = "ins" if op_type == "insertion" else "del"
            pct_key = percentage
            
            # Simply append the prediction to the corresponding list
            result[op_key][pct_key].append(last_prediction)
            last_prediction = _SENTINEL
        
        i += 1
    
    # Convert defaultdict to regular dict for JSON serialization
    def convert_to_dict(d):
        if isinstance(d, defaultdict):
            return {str(k): convert_to_dict(v) for k, v in d.items()}
        return d
    
    return convert_to_dict(result)

def main():
    log_path = "/Users/ian/Project/VLN/Recurrent-VLN-BERT/scripts/inference_4.log"
    output_path = "/Users/ian/Project/VLN/Recurrent-VLN-BERT/scripts/inference_4_parsed.json"
    
    print("Parsing log file...")
    result = parse_log_file(log_path)
    
    print("Extracted predictions:")
    for op_type in ["ins", "del"]:
        if op_type in result:
            print(f"  {op_type}:")
            for pct in ["0.25", "0.5", "0.75"]:
                if pct in result[op_type]:
                    print(f"    {pct}: {len(result[op_type][pct])} predictions")
    
    print(f"\nWriting results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("Done!")

if __name__ == "__main__":
    main()

