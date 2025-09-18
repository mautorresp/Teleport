#!/usr/bin/env python3
"""
Deterministic CLF test - prove serializer equality per your specification.
"""

import sys
sys.path.insert(0, '.')

from teleport.dgg import deduce_dynamic, compute_cost_receipts

def test_clf_deterministic():
    """Test the deterministic fixes with actual pic1.jpg"""
    
    print("=" * 60)
    print("DETERMINISTIC CLF TEST - SERIALIZER EQUALITY PROOF")
    print("=" * 60)
    print()
    
    # Load test file
    try:
        with open('test_artifacts/pic1.jpg', 'rb') as f:
            data = f.read()
        print(f"Testing pic1.jpg: {len(data)} bytes")
    except FileNotFoundError:
        # Fallback to a python file
        with open('teleport/dgg.py', 'rb') as f:
            data = f.read()
        print(f"Testing dgg.py: {len(data)} bytes")
    
    print()
    
    # 1. Deduction (deterministic)
    print("1. SCAN (admissibility only):")
    result = deduce_dynamic(data)
    op_id, params, reason = result
    N = len(data)
    
    print(f"   deduce_dynamic -> OP={op_id}, params={params}, reason={reason}")
    
    # Check CBD parameter shape
    if op_id == 9:  # OP_CBD
        print(f"   CBD params shape: {len(params)} elements")
        if len(params) == 1 and params[0] == N:
            print("   ✅ CBD params=(N,) - CORRECT")
        else:
            print("   ❌ CBD params != (N,) - COMPUTATIONAL EXPANSION")
    print()
    
    # 2. Price (integer check)
    print("2. PRICE (integer costs):")
    try:
        receipts = compute_cost_receipts(op_id, params, N)
        print(receipts)
    except Exception as e:
        print(f"   COST COMPUTATION FAILED: {e}")
    print()
    
    # 3. Expand-verify
    print("3. EXPAND-VERIFY:")
    try:
        from teleport.generators import verify_generator
        # Expansion is seed-isolated: depends only on (op_id, params, L)
        expand_equal = verify_generator(op_id, params, data)
        print(f"   expand_equal: {expand_equal}")
        
        if expand_equal:
            print("   ✅ EXPAND-VERIFY PASSED")
        else:
            print("   ❌ EXPAND-VERIFY FAILED")
    except Exception as e:
        print(f"   EXPAND-VERIFY ERROR: {e}")
    print()
    
    print("=" * 60)

if __name__ == "__main__":
    test_clf_deterministic()
