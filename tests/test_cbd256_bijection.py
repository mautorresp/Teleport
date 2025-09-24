#!/usr/bin/env python3
"""
CBD256 Bijection Test - Verify the mathematical bijection property.
"""

import sys
sys.path.insert(0, '.')

from teleport.dgg import deduce_dynamic, compute_cost_receipts
from teleport.generators import verify_generator, OP_CBD
import hashlib

def test_cbd256_bijection():
    """
    Test CBD256 bijective properties on a non-pattern file.
    """
    
    print("=" * 70)
    print("CBD256 BIJECTION TEST")
    print("=" * 70)
    print()
    
    # Create a file that won't match any pattern (non-constant, non-step, etc.)
    test_data = bytes([10, 50, 200, 15, 99])  # 5 bytes: irregular pattern
    L = len(test_data)
    
    print(f"Test file: {list(test_data)} ({L} bytes)")
    print(f"SHA256: {hashlib.sha256(test_data).hexdigest()}")
    print()
    
    # 1. Deduction
    print("=== DEDUCTION ===")
    op_id, params, reason = deduce_dynamic(test_data)
    print(f"Result: op_id={op_id}, params={params}")
    print(f"Reason: {reason}")
    
    if op_id == OP_CBD:
        K = params[0]
        print(f"CBD256: K = {K}")
        
        # Verify bijection: compute expected K
        expected_K = 0
        for i in range(L):
            expected_K += test_data[i] * (256 ** (L - 1 - i))
        print(f"Expected K = {expected_K}")
        print(f"Bijection check: K == expected_K = {K == expected_K}")
    print()
    
    # 2. Cost computation
    print("=== COST VERIFICATION ===")
    try:
        receipts = compute_cost_receipts(op_id, params, L)
        print(receipts)
    except Exception as e:
        print(f"Cost computation failed: {e}")
        return False
    print()
    
    # 3. Expansion verification
    print("=== EXPANSION VERIFICATION ===") 
    expand_equal = verify_generator(op_id, params, test_data)
    print(f"expand_equal: {expand_equal}")
    
    if expand_equal:
        print("✅ BIJECTION VERIFIED: E(D(S)) = S")
    else:
        print("❌ BIJECTION FAILED")
        return False
    
    print()
    print("CBD256 BIJECTION TEST: PASSED")
    return True

if __name__ == "__main__":
    success = test_cbd256_bijection()
    sys.exit(0 if success else 1)
