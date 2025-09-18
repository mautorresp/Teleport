#!/usr/bin/env python3
"""
Debug deductive composition: Check why patterns aren't being detected.
"""

import sys
sys.path.append('/Users/Admin/Teleport')

from teleport.generators import (
    deduce_prefix_CONST, deduce_suffix_CONST,
    deduce_prefix_STEP, deduce_suffix_STEP
)

def debug_pattern_detection():
    """Debug why deductive composition isn't working"""
    
    # Create simple test data
    test_data = bytes([0x00] * 100 + [1, 4, 7, 10, 13] + [0xAA] * 50)
    print(f"Test data length: {len(test_data)}")
    print(f"Structure: 100 nulls + [1,4,7,10,13] + 50 AAs")
    
    # Test prefix detection
    print(f"\n--- Prefix Detection ---")
    const_ok, const_params, const_len = deduce_prefix_CONST(test_data)
    print(f"CONST prefix: ok={const_ok} params={const_params} len={const_len}")
    
    step_ok, step_params, step_len = deduce_prefix_STEP(test_data)
    print(f"STEP prefix: ok={step_ok} params={step_params} len={step_len}")
    
    # Test suffix detection
    print(f"\n--- Suffix Detection ---")
    const_suf_ok, const_suf_params, const_suf_len = deduce_suffix_CONST(test_data)
    print(f"CONST suffix: ok={const_suf_ok} params={const_suf_params} len={const_suf_len}")
    
    step_suf_ok, step_suf_params, step_suf_len = deduce_suffix_STEP(test_data)
    print(f"STEP suffix: ok={step_suf_ok} params={step_suf_params} len={step_suf_len}")
    
    # Manual analysis
    print(f"\n--- Manual Analysis ---")
    print(f"First 10 bytes: {list(test_data[:10])}")
    print(f"Last 10 bytes: {list(test_data[-10:])}")
    print(f"Middle section: {list(test_data[100:105])}")
    
    # Test the actual factoring logic
    print(f"\n--- Factoring Logic ---")
    
    # Best prefix
    best_prefix = (0, (), 0)
    if const_ok and const_len > best_prefix[2]:
        best_prefix = (2, const_params, const_len)  # OP_CONST = 2
    if step_ok and step_len > best_prefix[2]:
        best_prefix = (3, step_params, step_len)   # OP_STEP = 3
    
    print(f"Best prefix: {best_prefix}")
    
    # Best suffix (non-overlapping)
    max_suffix_start = len(test_data) - best_prefix[2] if best_prefix[2] > 0 else len(test_data)
    print(f"Max suffix start position: {max_suffix_start}")
    
    best_suffix = (0, (), 0)
    if const_suf_ok and const_suf_len <= max_suffix_start and const_suf_len > best_suffix[2]:
        best_suffix = (2, const_suf_params, const_suf_len)
    if step_suf_ok and step_suf_len <= max_suffix_start and step_suf_len > best_suffix[2]:
        best_suffix = (3, step_suf_params, step_suf_len)
    
    print(f"Best suffix: {best_suffix}")
    
    # Calculate segments
    A_len = best_prefix[2] if best_prefix[2] > 0 else 0
    B_len = best_suffix[2] if best_suffix[2] > 0 else 0
    M_start = A_len
    M_end = len(test_data) - B_len
    
    print(f"A_len: {A_len}, B_len: {B_len}")
    print(f"Middle: [{M_start}:{M_end}] = {M_end - M_start} bytes")
    
    if M_start < M_end:
        middle_data = test_data[M_start:M_end]
        print(f"Middle data: {list(middle_data)}")
    else:
        print("No middle section (overlap detected)")

if __name__ == "__main__":
    debug_pattern_detection()
