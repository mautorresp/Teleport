#!/usr/bin/env python3
"""
Debug CLF serialization vs cost calculation
"""

from clf_bitexact import serialize_caus
from teleport.generators import deduce_all, compute_caus_cost

def debug_serialization():
    # Test data
    test_data = bytes([0xAB, 0xCD, 0xEF, 0x12])
    print(f"Test data: {test_data.hex().upper()}")
    
    # Get deduction result
    result = deduce_all(test_data)
    op_id, params, reason = result
    print(f"Deduced: op_id={op_id}, params={params}")
    
    # Calculate cost
    calculated_cost = compute_caus_cost(op_id, params, len(test_data))
    print(f"Calculated cost: {calculated_cost} bits")
    
    # Serialize 
    caus_seed = serialize_caus(op_id, params, len(test_data))
    actual_bits = len(caus_seed) * 8
    print(f"Actual serialized: {len(caus_seed)} bytes = {actual_bits} bits")
    print(f"Hex: {caus_seed.hex().upper()}")
    
    # Break down the serialization
    print(f"\nBit breakdown:")
    print(f"Difference: {actual_bits - calculated_cost} bits")
    
    # Check if it's padding
    if (actual_bits - calculated_cost) <= 7:
        print("Likely padding to byte boundary - this is normal")
        
        # Calculate unpadded bits
        expected_bytes = (calculated_cost + 7) // 8
        expected_bits = expected_bytes * 8
        print(f"Expected padded size: {expected_bytes} bytes = {expected_bits} bits")
        
        if expected_bits == actual_bits:
            print("✅ Serialization matches cost + padding")
            return True
        else:
            print("❌ Serialization doesn't match expected")
            return False
    else:
        print("❌ Difference too large for padding")
        return False

if __name__ == "__main__":
    debug_serialization()
