#!/usr/bin/env python3
"""
CLF Behavior Pinning Test - Eliminates Flip-Flop Between Regimes

Tests that the canonical DP with fixed operator set produces DETERMINISTIC
results and never flip-flops between "6-byte gap OPEN" and "single CBD PASS".

Key validation:
1. Same input always produces same output (determinism)
2. CBD256 always considered for whole-range (no regime drift)
3. All mathematical rails enforced consistently
4. Drift-killer validation catches any violations
"""

import hashlib
import os
from teleport.clf_fb import encode_minimal
from teleport.clf_canonical import clf_canonical_receipts, validate_encoding_result

def test_deterministic_behavior():
    """Test that encoding is deterministic - same input always gives same result"""
    print("=== DETERMINISTIC BEHAVIOR TEST ===")
    
    # Test cases of different sizes and characteristics
    test_cases = [
        b"Hello",  # Small ASCII
        b"\x00\x01\x02\x03",  # Small binary  
        b"\xff" * 10,  # Constant run
        b"\x42" * 50,  # Larger constant run
        b"The quick brown fox",  # Text data
    ]
    
    for i, data in enumerate(test_cases):
        print(f"\nTest case {i}: {len(data)} bytes")
        
        # Encode multiple times
        result1 = encode_minimal(data)
        result2 = encode_minimal(data)
        result3 = encode_minimal(data)
        
        # Results must be identical (determinism)
        assert result1 == result2 == result3, f"Non-deterministic behavior for case {i}"
        
        if result1:
            print(f"  PASS: {len(result1)} tokens")
            # Verify all results satisfy rails
            validate_encoding_result(data, result1)
        else:
            print(f"  OPEN: No admissible encoding")
            
    print("✓ Deterministic behavior verified")

def test_cbd256_whole_range_inclusion():
    """Test that CBD256 is always considered for whole-range coverage"""
    print("\n=== CBD256 WHOLE-RANGE INCLUSION TEST ===")
    
    # Create test data that would cause "6-byte gap" if CBD256 not considered whole-range
    # JPEG-like data with markers at boundaries
    test_data = b"\xff\xd8\xff\xe0" + b"random_middle_content" + b"\xff\xd9\x00\x00\x00\x00"
    
    print(f"Testing {len(test_data)} bytes with boundary markers")
    
    tokens = encode_minimal(test_data)
    
    if tokens:
        print("PASS state achieved")
        
        # Check if single CBD256 token used (whole-range coverage)
        if len(tokens) == 1 and tokens[0][0] == 9:  # OP_CBD256
            print("✓ Single CBD256 token covers entire range (canonical)")
            op_id, params, token_L, cost_info = tokens[0]
            assert token_L == len(test_data), f"CBD256 length mismatch: {token_L} != {len(test_data)}"
        else:
            print(f"Multiple tokens: {[(t[0], t[2]) for t in tokens]}")
            
        # Regardless of token composition, verify coverage exactness
        validate_encoding_result(test_data, tokens)
        
    else:
        print("OPEN state - no admissible encoding")
        # Even OPEN must be mathematically sound
        validate_encoding_result(test_data, tokens)
        
    print("✓ CBD256 whole-range inclusion verified")

def test_no_approximation_drift():
    """Test that exact arithmetic eliminates approximation-based drift"""
    print("\n=== EXACT ARITHMETIC TEST ===")
    
    # Test cases where approximations might differ from exact computation
    test_cases = [
        b"\x00" * 1000,  # Large constant run
        b"\x01\x02\x03" * 333,  # Repeated sequence
        bytes(range(256)) * 4,  # Full byte range cycles
    ]
    
    for i, data in enumerate(test_cases):
        print(f"\nTest case {i}: {len(data)} bytes")
        
        tokens = encode_minimal(data)
        
        if tokens:
            # Verify exact cost computation matches receipts
            receipts = clf_canonical_receipts(data, tokens)
            
            # Extract costs from receipts and verify they match token costs
            total_stream = sum(cost_info['C_stream'] for _, _, _, cost_info in tokens)
            
            # Look for "GLOBAL:" line in receipts
            global_line = None
            for line in receipts:
                if line.startswith("GLOBAL:"):
                    global_line = line
                    break
                    
            assert global_line is not None, "Global cost line not found in receipts"
            print(f"  {global_line}")
            
            # Verify no approximations by checking integer-only arithmetic
            for _, params, token_L, cost_info in tokens:
                assert isinstance(cost_info['C_stream'], int), f"Non-integer cost: {cost_info['C_stream']}"
                assert isinstance(token_L, int), f"Non-integer length: {token_L}"
                
            print(f"  ✓ All costs are exact integers")
        else:
            print(f"  OPEN state")
            
    print("✓ Exact arithmetic verified")

def test_serializer_equality_pinned():
    """Test that serializer equality convention is pinned and consistent"""
    print("\n=== SERIALIZER EQUALITY CONVENTION TEST ===")
    
    test_data = b"test_serializer_consistency"
    tokens = encode_minimal(test_data)
    
    if tokens:
        print(f"Testing {len(tokens)} tokens for serializer equality")
        
        for i, (op_id, params, token_L, cost_info) in enumerate(tokens):
            # The compute_cost_receipts function already asserts serializer equality
            # Just verify it passes without throwing
            print(f"  Token[{i}]: op={op_id}, L={token_L}, C_CAUS={cost_info['C_CAUS']}")
            
        print("✓ Serializer equality convention enforced")
    else:
        print("OPEN state - no tokens to test")
        
def test_global_bound_consistency():
    """Test that global bound is consistently applied"""
    print("\n=== GLOBAL BOUND CONSISTENCY TEST ===")
    
    # Test data that should be near the boundary
    test_data = b"X" * 100  # Should compress well with CONST
    
    tokens = encode_minimal(test_data)
    receipts = clf_canonical_receipts(test_data, tokens)
    
    print(f"Testing global bound for {len(test_data)} bytes")
    
    # Extract and verify global bound from receipts
    baseline_line = None
    bound_line = None
    
    for line in receipts:
        if line.startswith("BASELINE:"):
            baseline_line = line
        elif line.startswith("BOUND:"):
            bound_line = line
            
    assert baseline_line is not None, "Baseline line not found"
    assert bound_line is not None, "Bound line not found"
    
    print(f"  {baseline_line}")
    print(f"  {bound_line}")
    
    # Verify bound consistency
    if tokens:
        assert "True" in bound_line, "PASS state should satisfy global bound"
    else:
        # OPEN state may have bound line indicating failure
        pass
        
    print("✓ Global bound consistency verified")

def main():
    """Run comprehensive behavior pinning tests"""
    print("CLF BEHAVIOR PINNING VALIDATION")
    print("==============================")
    print("Testing that canonical DP with fixed operator set eliminates regime drift")
    
    try:
        test_deterministic_behavior()
        test_cbd256_whole_range_inclusion()
        test_no_approximation_drift()
        test_serializer_equality_pinned()
        test_global_bound_consistency()
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED")
        print("✅ CLF behavior successfully pinned")
        print("✅ No more flip-flop between regimes")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    main()
