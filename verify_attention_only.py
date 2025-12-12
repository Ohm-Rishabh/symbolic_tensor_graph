#!/usr/bin/env python3
"""
Script to list operations in .et files.

Usage:
    python verify_attention_only.py <path_to_et_files>
    
Example:
    python verify_attention_only.py generated_attn/
    python verify_attention_only.py generated_attn/moe_attn_8exp_4ep.0.et
"""

import os
import sys
import re
import glob
from pathlib import Path
from collections import Counter


def extract_strings_from_binary(file_path: str) -> str:
    """Extract readable strings from binary .et file"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Try to decode as UTF-8, ignoring errors
        try:
            content_str = content.decode('utf-8', errors='ignore')
        except:
            content_str = str(content)
        
        return content_str
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def extract_operations(et_file: str):
    """Extract and print operations from .et file"""
    content = extract_strings_from_binary(et_file)
    
    # Extract node names - try multiple patterns
    node_names = []
    
    # Pattern 1: name="..." or name='...'
    pattern1 = r'name["\']?\s*[:=]\s*["\']?([^"\']+)["\']?'
    matches = re.findall(pattern1, content, re.IGNORECASE)
    node_names.extend(matches)
    
    # Pattern 2: Look for common operation patterns
    # Match strings that look like operation names (contain dots, underscores, etc.)
    pattern2 = r'[a-zA-Z_][a-zA-Z0-9_.]*\.[a-zA-Z0-9_.]+'
    matches = re.findall(pattern2, content)
    node_names.extend(matches)
    
    # Pattern 3: Look for COMP_NODE or COLL_COMM_NODE followed by names
    pattern3 = r'(?:COMP_NODE|COLL_COMM_NODE|COMP|COMM)[^a-zA-Z]*([a-zA-Z_][a-zA-Z0-9_.]*)'
    matches = re.findall(pattern3, content, re.IGNORECASE)
    node_names.extend(matches)
    
    # Get unique operations and count them
    unique_ops = sorted(set(node_names))
    op_counts = Counter(node_names)
    
    return unique_ops, op_counts


def find_et_files(path: str):
    """Find all .et files in the given path"""
    path_obj = Path(path)
    
    if path_obj.is_file() and path.endswith('.et'):
        return [path]
    elif path_obj.is_dir():
        et_files = list(path_obj.glob('*.et'))
        return [str(f) for f in sorted(et_files)]
    else:
        et_files = glob.glob(path)
        return [f for f in et_files if f.endswith('.et')]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Please provide path to .et file(s)")
        print("Example: python verify_attention_only.py generated_attn/")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Find all .et files
    et_files = find_et_files(input_path)
    
    if not et_files:
        print(f"Error: No .et files found in: {input_path}")
        sys.exit(1)
    
    # Take the first .et file
    first_file = et_files[0]
    print(f"Analyzing: {first_file}\n")
    
    # Extract operations
    unique_ops, op_counts = extract_operations(first_file)
    
    # Print results
    print("=" * 80)
    print(f"Operations found in {os.path.basename(first_file)}")
    print("=" * 80)
    print(f"\nTotal unique operations: {len(unique_ops)}\n")
    
    # Group by prefix (e.g., transformer.0.mha, transformer.0.ffn_res, etc.)
    grouped = {}
    for op in unique_ops:
        # Extract prefix (e.g., "transformer.0.mha" from "transformer.0.mha.attn_kernel.qkv")
        parts = op.split('.')
        if len(parts) >= 2:
            prefix = '.'.join(parts[:3]) if len(parts) >= 3 else '.'.join(parts[:2])
        else:
            prefix = op
        if prefix not in grouped:
            grouped[prefix] = []
        grouped[prefix].append(op)
    
    # Print grouped operations
    print("Operations grouped by prefix:\n")
    for prefix in sorted(grouped.keys()):
        ops = sorted(grouped[prefix])
        print(f"{prefix}:")
        for op in ops:
            count = op_counts[op]
            print(f"  - {op} (appears {count} time(s))")
        print()
    
    # Also print all unique operations
    print("\n" + "=" * 80)
    print("All unique operations (sorted):")
    print("=" * 80)
    for op in unique_ops:
        count = op_counts[op]
        print(f"  {op} ({count})")


if __name__ == "__main__":
    main()

