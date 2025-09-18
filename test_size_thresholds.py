#!/usr/bin/env python3
"""
Test large constant data to verify CBD256 whole-range behavior
"""

from teleport.clf_canonical import encode_CLF, clf_canonical_receipts, header_bits, exact_cbd256_cost

def test_large_constant():
    """Test large constant data that should definitely pass global bound"""
    
    print("=== TESTING LARGE CONSTANT DATA ===")
    
    # Large constant run - should compress very well with CBD256
    L = 200  # 200 bytes of same value
    data = b"\x00" * L
    
    print(f"Input: {L} bytes of constant value 0x00")
    
    # Manual calculation to verify
    K = 0  # All zeros -> K = 0
    cbd_cost = exact_cbd256_cost(L, K)
    H_L = header_bits(L)
    total_cost = H_L + cbd_cost['C_stream']
    baseline = 10 * L
    
    print(f"Expected CBD256 cost: {cbd_cost['C_stream']} bits")
    print(f"Header cost: {H_L} bits") 
    print(f"Total cost: {total_cost} bits")
    print(f"Baseline: {baseline} bits")
    print(f"Should pass: {total_cost < baseline}")
    
    result = encode_CLF(data)
    
    if result:
        print(f"Result: PASS with {len(result)} tokens")
        receipts = clf_canonical_receipts(data, result)
        for line in receipts:
            print(f"  {line}")
    else:
        print("Result: OPEN")
        
    print()
    
    # Test with a different constant value
    data2 = b"\xFF" * L
    print(f"Input: {L} bytes of constant value 0xFF")
    
    # K for 0xFF repeated L times
    K2 = int('FF' * L, 16)  # This will be a large number
    cbd_cost2 = exact_cbd256_cost(L, K2)
    total_cost2 = H_L + cbd_cost2['C_stream']
    
    print(f"Expected CBD256 cost: {cbd_cost2['C_stream']} bits")
    print(f"Total cost: {total_cost2} bits") 
    print(f"Should pass: {total_cost2 < baseline}")
    
    result2 = encode_CLF(data2)
    
    if result2:
        print(f"Result: PASS with {len(result2)} tokens")
    else:
        print("Result: OPEN")

def test_mixed_data_sizes():
    """Test different sizes to find the threshold"""
    
    print("\n=== TESTING SIZE THRESHOLDS ===")
    
    for L in [100, 150, 200, 250, 300]:
        data = b"A" * L
        K = int('41' * L, 16)  # 'A' = 0x41 repeated
        
        cbd_cost = exact_cbd256_cost(L, K)
        H_L = header_bits(L)
        total_cost = H_L + cbd_cost['C_stream']
        baseline = 10 * L
        
        result = encode_CLF(data)
        
        status = "PASS" if result else "OPEN"
        predicted = "PASS" if total_cost < baseline else "OPEN"
        
        print(f"L={L}: {total_cost} < {baseline} = {total_cost < baseline}, predicted={predicted}, actual={status}")

if __name__ == "__main__":
    test_large_constant()
    test_mixed_data_sizes()
