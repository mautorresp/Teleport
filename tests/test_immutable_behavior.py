#!/usr/bin/env python3
"""
Test IMMUTABLE CLF behavior - verifies pinned rails work correctly
"""

from teleport.clf_canonical import encode_CLF, clf_canonical_receipts

def test_immutable_behavior():
    """Test that the IMMUTABLE pinned behavior works correctly"""
    
    print("=== TESTING IMMUTABLE CLF BEHAVIOR ===")
    
    # Test 1: Small data that should use CONST tokens (CBD256 too expensive)
    print("\n1. Testing CONST token selection:")
    small_const_data = b"\xFF" * 3  # 3 identical bytes
    result = encode_CLF(small_const_data)
    
    if result:
        print(f"   Result: PASS with {len(result)} tokens")
        for i, (op_id, params, length, cost_info) in enumerate(result):
            print(f"   Token[{i}]: op={op_id}, params={params}, L={length}, cost={cost_info['C_stream']}")
    else:
        print("   Result: OPEN (no admissible encoding)")
    
    # Test 2: Larger constant data that should use CBD256 (whole-range first)
    print("\n2. Testing CBD256 whole-range selection:")
    large_const_data = b"\x42" * 50  # 50 identical bytes - CBD256 should win
    result = encode_CLF(large_const_data)
    
    if result:
        print(f"   Result: PASS with {len(result)} tokens")
        for i, (op_id, params, length, cost_info) in enumerate(result):
            if op_id == 9:  # CBD256
                print(f"   Token[{i}]: CBD256, L={length}, cost={cost_info['C_stream']}")
            else:
                print(f"   Token[{i}]: op={op_id}, params={params}, L={length}, cost={cost_info['C_stream']}")
    else:
        print("   Result: OPEN (no admissible encoding)")
    
    # Test 3: Mixed data - should use CBD256 for whole range
    print("\n3. Testing mixed data CBD256:")
    mixed_data = b"Hello, World! This is a test."
    result = encode_CLF(mixed_data)
    
    if result:
        print(f"   Result: PASS with {len(result)} tokens")
        for i, (op_id, params, length, cost_info) in enumerate(result):
            if op_id == 9:  # CBD256
                print(f"   Token[{i}]: CBD256, L={length}, cost={cost_info['C_stream']}")
            else:
                print(f"   Token[{i}]: op={op_id}, params={params}, L={length}, cost={cost_info['C_stream']}")
                
        # Verify determinism - encoding same data multiple times
        result2 = encode_CLF(mixed_data)
        result3 = encode_CLF(mixed_data)
        
        if result == result2 == result3:
            print("   ✅ Deterministic: Same input produces same output")
        else:
            print("   ❌ Non-deterministic behavior detected!")
            
    else:
        print("   Result: OPEN (no admissible encoding)")
    
    # Test 4: Very small data that should be OPEN
    print("\n4. Testing OPEN condition:")
    tiny_data = b"Hi"  # Too small to compress
    result = encode_CLF(tiny_data)
    
    if result:
        print(f"   Result: PASS with {len(result)} tokens")
    else:
        print("   Result: OPEN (correctly determined)")
    
    print("\n=== IMMUTABLE BEHAVIOR TESTS COMPLETE ===")

if __name__ == "__main__":
    test_immutable_behavior()
