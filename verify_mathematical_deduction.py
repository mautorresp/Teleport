#!/usr/bin/env python3
"""
Verify that CLF mathematical deduction properly characterizes data
instead of computationally expanding it.
"""

import sys
import os
sys.path.insert(0, '.')

from teleport.dgg import deduce_dynamic

def test_mathematical_characterization():
    """Test that mathematical deduction provides characterization, not expansion."""
    
    print("=== CLF Mathematical Deduction Verification ===")
    print()
    
    # Test 1: Small JPEG file
    test_file = 'test_artifacts/pic1.jpg'
    if os.path.exists(test_file):
        with open(test_file, 'rb') as f:
            data = f.read()
        
        print(f"Testing {test_file}: {len(data)} bytes")
        result = deduce_dynamic(data)
        print(f"Mathematical characterization: OP={result[0]}, params={len(result[1])}, desc={result[2]}")
        
        # Verify this is mathematical characterization, not byte expansion
        if len(result[1]) < len(data):
            print("✅ Proper mathematical characterization (params < input size)")
        else:
            print("❌ Computational expansion detected (params >= input size)")
        print()
    
    # Test 2: Text file
    test_files = [f for f in os.listdir('.') if f.endswith('.py') and os.path.getsize(f) > 1000][:2]
    
    for test_file in test_files:
        with open(test_file, 'rb') as f:
            data = f.read()
        
        print(f"Testing {test_file}: {len(data)} bytes")
        result = deduce_dynamic(data)
        print(f"Mathematical characterization: OP={result[0]}, params={len(result[1])}, desc={result[2][:100]}...")
        
        if len(result[1]) < len(data):
            print("✅ Proper mathematical characterization")
        else:
            print("❌ Computational expansion detected")
        print()
    
    # Test 3: Binary patterns to verify different deduction types
    test_patterns = [
        ([5, 5, 5, 5, 5], "Constant pattern"),
        ([1, 2, 3, 4, 5], "Step pattern"),
        ([42] * 100, "Large constant pattern"),
        (list(range(50)), "Incremental pattern")
    ]
    
    for pattern, desc in test_patterns:
        print(f"Testing {desc}: {len(pattern)} elements")
        result = deduce_dynamic(pattern)
        print(f"Mathematical rule: OP={result[0]}, params={len(result[1])}, desc={result[2]}")
        
        if len(result[1]) <= 10:  # Mathematical rules should be compact
            print("✅ Proper mathematical rule application")
        else:
            print("❌ Non-mathematical approach detected")
        print()
    
    print("=== Mathematical Deduction Verification Complete ===")

if __name__ == "__main__":
    test_mathematical_characterization()
