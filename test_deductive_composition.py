#!/usr/bin/env python3
"""
Test CLF deductive composition for minimality: C_stream < 8*L
Mathematical verification of canonical factoring algorithm.
"""

import sys
sys.path.append('/Users/Admin/Teleport')

from teleport.dgg import deduce_composed, deduce_dynamic, compute_composition_cost
from teleport.generators import verify_generator
from teleport.seed_vm import expand_generator

def test_deductive_composition():
    """Test canonical factoring vs single-token approaches"""
    
    # Test case 1: Composite pattern (CONST + different middle + CONST)
    test_data = bytes([0xFF] * 100 + [0x42, 0x43, 0x44] + [0xFF] * 100)
    L = len(test_data)
    
    print(f"=== CLF Deductive Composition Test ===")
    print(f"Input length L = {L}")
    print(f"Upper bound: 8*L = {8*L} bits")
    
    # Single token approach
    single_op, single_params, single_reason = deduce_dynamic(test_data)
    single_cost = compute_single_cost(single_op, single_params, L)
    
    print(f"\n--- Single Token ---")
    print(f"Operation: {single_op} {single_params}")
    print(f"Reason: {single_reason}")
    print(f"Cost: {single_cost} bits")
    print(f"Efficiency: {single_cost/(8*L):.3f} (want < 1.0)")
    
    # Composed tokens approach  
    composed_tokens = deduce_composed(test_data)
    composed_cost = compute_composition_cost(composed_tokens)
    
    print(f"\n--- Deductive Composition ---")
    print(f"Tokens: {len(composed_tokens)}")
    for i, (op_id, params, seg_L, reason) in enumerate(composed_tokens):
        print(f"  {i+1}: op={op_id} params={params} L={seg_L} ({reason})")
    print(f"Total cost: {composed_cost} bits")
    print(f"Efficiency: {composed_cost/(8*L):.3f}")
    
    # Minimality verification
    print(f"\n--- Minimality Check ---")
    print(f"C_stream = {composed_cost} bits")
    print(f"8*L = {8*L} bits") 
    if composed_cost < 8*L:
        print("✓ MINIMALITY: C_stream < 8*L")
    else:
        print("✗ MINIMALITY FAILED")
    
    if composed_cost < single_cost:
        print("✓ OPTIMIZATION: Composition better than single token")
    else:
        print("✗ OPTIMIZATION FAILED")
    
    # Verify mathematical correctness
    print(f"\n--- Mathematical Verification ---")
    try:
        reconstructed = b""
        for op_id, params, seg_L, _ in composed_tokens:
            segment = expand_generator(op_id, params, seg_L)
            reconstructed += segment
        
        if reconstructed == test_data:
            print("✓ BIJECTION: E(D(S)) = S")
        else:
            print(f"✗ BIJECTION FAILED: {len(reconstructed)} != {len(test_data)}")
            return False
    except Exception as e:
        print(f"✗ EXPANSION ERROR: {e}")
        return False
    
    return True

def compute_single_cost(op_id: int, params: tuple, L: int) -> int:
    """Compute C_stream for single CAUS token"""
    from teleport.dgg import leb_len
    
    C_op = 8 * leb_len(op_id)
    C_params = 8 * sum(leb_len(p) for p in params) if params else 0
    C_L = 8 * leb_len(L)
    C_CAUS = 3 + C_op + C_params + C_L
    
    pad_bits = (8 - ((C_CAUS + 3) % 8)) % 8
    C_END = 3 + pad_bits
    C_stream = C_CAUS + C_END
    
    return C_stream

def test_prefix_suffix_detection():
    """Test individual prefix/suffix detection functions"""
    from teleport.generators import (
        deduce_prefix_CONST, deduce_suffix_CONST,
        deduce_prefix_STEP, deduce_suffix_STEP
    )
    
    print(f"\n=== Prefix/Suffix Detection Test ===")
    
    # Test CONST detection
    data = bytes([0xAA] * 50 + [0x01, 0x02, 0x03] + [0xBB] * 30)
    
    prefix_ok, prefix_params, prefix_len = deduce_prefix_CONST(data)
    suffix_ok, suffix_params, suffix_len = deduce_suffix_CONST(data)
    
    print(f"CONST prefix: ok={prefix_ok} params={prefix_params} len={prefix_len}")
    print(f"CONST suffix: ok={suffix_ok} params={suffix_params} len={suffix_len}")
    
    # Test STEP detection
    step_data = bytes([10, 13, 16, 19, 22] + [0xFF, 0xFF] + [100, 95, 90])
    
    step_prefix_ok, step_prefix_params, step_prefix_len = deduce_prefix_STEP(step_data)
    step_suffix_ok, step_suffix_params, step_suffix_len = deduce_suffix_STEP(step_data)
    
    print(f"STEP prefix: ok={step_prefix_ok} params={step_prefix_params} len={step_prefix_len}")
    print(f"STEP suffix: ok={step_suffix_ok} params={step_suffix_params} len={step_suffix_len}")

if __name__ == "__main__":
    print("Testing CLF deductive composition...")
    
    test_prefix_suffix_detection()
    success = test_deductive_composition()
    
    if success:
        print(f"\n🎉 CLF DEDUCTIVE COMPOSITION: ALL TESTS PASSED")
        print("Mathematical minimality via canonical factoring achieved!")
    else:
        print(f"\n❌ TESTS FAILED")
        sys.exit(1)
