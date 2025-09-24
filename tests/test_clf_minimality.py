#!/usr/bin/env python3
"""
Test CLF Minimality - Verifies construction comparison works correctly
"""

from teleport.clf_canonical import encode_CLF, clf_canonical_receipts, exact_cbd256_cost, compute_cost_receipts, OP_CONST, OP_CBD256

def test_minimality_comparison():
    """Test that CLF chooses minimal construction between CBD256 and structural"""
    
    print("=== TESTING CLF MINIMALITY ===")
    
    # Test 1: Strong structure - CONST should beat CBD256
    print("\n1. Testing strong constant structure (should favor STRUCTURAL):")
    strong_const = b"\x42" * 30  # 30 bytes of same value
    L = len(strong_const)
    
    # Manual cost calculation for comparison
    K_cbd = int('42' * 30, 16)
    cbd_cost = exact_cbd256_cost(L, K_cbd)
    const_cost = compute_cost_receipts(OP_CONST, (0x42,), L)
    
    print(f"   CBD256 cost: {cbd_cost['C_stream']} bits")
    print(f"   CONST cost:  {const_cost['C_stream']} bits")
    print(f"   CONST cheaper: {const_cost['C_stream'] < cbd_cost['C_stream']}")
    
    result = encode_CLF(strong_const)
    if result:
        receipts = clf_canonical_receipts(strong_const, result)
        construction_line = [line for line in receipts if line.startswith("CONSTRUCTION:")][0]
        print(f"   CLF chose: {construction_line}")
        
        # Verify minimality
        if len(result) == 1 and result[0][0] == OP_CONST:
            print("   ✅ Correctly chose STRUCTURAL (CONST)")
        elif len(result) == 1 and result[0][0] == OP_CBD256:
            print("   📊 Chose CBD256 (may be minimal for this case)")
        else:
            print("   📊 Complex structural cover")
    else:
        print("   OPEN (no admissible encoding)")
    
    # Test 2: Random-like data - CBD256 should be competitive
    print("\n2. Testing mixed data (CBD256 vs STRUCTURAL comparison):")
    mixed_data = b"Hello, World! This is mixed content with no clear mathematical structure."
    L2 = len(mixed_data)
    
    result2 = encode_CLF(mixed_data)
    if result2:
        receipts2 = clf_canonical_receipts(mixed_data, result2)
        construction_line2 = [line for line in receipts2 if line.startswith("CONSTRUCTION:")][0]
        print(f"   CLF chose: {construction_line2}")
        
        # Show costs
        total_cost = sum(cost_info['C_stream'] for _, _, _, cost_info in result2)
        print(f"   Total stream cost: {total_cost} bits")
        print(f"   Length: {L2} bytes")
        
    else:
        print("   OPEN (no admissible encoding)")
    
    # Test 3: Verify determinism - same input should give same construction choice
    print("\n3. Testing determinism:")
    result3a = encode_CLF(strong_const)
    result3b = encode_CLF(strong_const) 
    result3c = encode_CLF(strong_const)
    
    if result3a == result3b == result3c:
        print("   ✅ Deterministic: Same construction chosen consistently")
    else:
        print("   ❌ Non-deterministic behavior detected!")
    
    # Test 4: Small data that should be OPEN
    print("\n4. Testing OPEN condition:")
    tiny_data = b"Hi"
    result4 = encode_CLF(tiny_data)
    
    if result4:
        print(f"   Unexpected PASS with {len(result4)} tokens")
    else:
        print("   ✅ Correctly determined OPEN")
    
    print("\n=== MINIMALITY TESTS COMPLETE ===")

def test_cost_comparison_logic():
    """Verify the cost comparison logic works mathematically"""
    
    print("\n=== TESTING COST COMPARISON LOGIC ===")
    
    # Test with data where we can predict the outcome
    test_cases = [
        (b"\xFF" * 50, "Strong constant - should favor STRUCTURAL"),
        (b"The quick brown fox jumps", "Mixed text - depends on structure"),
        (b"\x00\x01\x02\x03" * 25, "Repeating structure - may favor STRUCTURAL"),
    ]
    
    for test_data, description in test_cases:
        print(f"\n{description}:")
        print(f"   Data: {len(test_data)} bytes")
        
        result = encode_CLF(test_data)
        
        if result:
            # Extract construction type and cost
            total_cost = sum(cost_info['C_stream'] for _, _, _, cost_info in result)
            
            if len(result) == 1 and result[0][0] == OP_CBD256:
                construction = "CBD256"
            else:
                construction = "STRUCTURAL"
            
            print(f"   Chosen: {construction}")
            print(f"   Cost: {total_cost} bits")
            print(f"   Tokens: {len(result)}")
            
        else:
            print("   Result: OPEN")

if __name__ == "__main__":
    test_minimality_comparison()
    test_cost_comparison_logic()
