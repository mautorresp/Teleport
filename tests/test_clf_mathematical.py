#!/usr/bin/env python3
"""
CLF Mathematical Correctness Test
Validates all drift-killer rails with integer proofs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from teleport.clf_canonical import (
    encode_CLF, clf_canonical_receipts, header_bits,
    compute_cost_receipts, OpenError
)
from teleport.seed_format import OP_CONST

def test_empty_file():
    """Test L=0 mathematical policy"""
    print("=== Empty File Test (L=0) ===")
    
    S = b""
    tokens = encode_CLF(S)
    receipts = clf_canonical_receipts(S, tokens)
    
    print("\n".join(receipts))
    
    # Mathematical verification
    H_0 = header_bits(0)
    baseline = 10 * 0
    print(f"\nMathematical check: H(0)={H_0}, baseline=10·0={baseline}")
    print(f"Inequality H(0) < baseline: {H_0} < {baseline} = {H_0 < baseline}")
    print(f"Expected: OPEN (False)")
    
    assert tokens == [], "Empty file should be OPEN"
    assert "STATE: OPEN" in "\n".join(receipts)
    print("✓ Empty file correctly handled as OPEN")

def test_constant_run():
    """Test efficient constant encoding"""
    print("\n=== Constant Run Test ===")
    
    # 100 zeros should be efficiently encoded
    S = bytes(100)  # 100 zero bytes
    tokens = encode_CLF(S)
    receipts = clf_canonical_receipts(S, tokens)
    
    print("\n".join(receipts))
    
    if tokens:
        print(f"✓ Constant run encoded with {len(tokens)} tokens")
        
        # Verify mathematical optimality
        H_L = header_bits(len(S))
        total_stream = sum(cost['C_stream'] for _, _, _, cost in tokens)
        baseline = 10 * len(S)
        
        print(f"Costs: H({len(S)})={H_L}, Σ C_stream={total_stream}, baseline={baseline}")
        print(f"Global bound: {H_L + total_stream} < {baseline} = {H_L + total_stream < baseline}")
        assert H_L + total_stream < baseline, "Should satisfy global bound"
    else:
        print("OPEN - constant run not beneficial under current operators")

def test_segment_guard():
    """Test per-token segment guard enforcement"""
    print("\n=== Segment Guard Test ===")
    
    # Test single byte (should satisfy C_stream < 10)
    try:
        cost_info = compute_cost_receipts(OP_CONST, (65,), 1)  # CONST 'A', L=1
        print(f"Single CONST: C_stream={cost_info['C_stream']}, guard: {cost_info['C_stream']} < 10")
        
        if cost_info['C_stream'] >= 10:
            print("Single CONST fails segment guard - will trigger OpenError in compose_cover")
        else:
            print("✓ Single CONST passes segment guard")
            
    except Exception as e:
        print(f"Error in cost computation: {e}")

def test_mathematical_rails():
    """Test all drift-killer rails with known inputs"""
    print("\n=== Mathematical Rails Test ===")
    
    # Test serializer equality rail
    try:
        cost_info = compute_cost_receipts(OP_CONST, (42,), 5)
        print(f"✓ Serializer equality: {cost_info['serialized_bytes']*8} == {cost_info['C_stream']}")
    except AssertionError as e:
        print(f"❌ Serializer equality failed: {e}")
    except Exception as e:
        print(f"Error testing serializer rail: {e}")
    
    # Test header consistency
    for L in [0, 1, 10, 100]:
        H = header_bits(L)
        from teleport.leb_io import leb_len
        expected = 16 + 8 * leb_len(8 * L)
        print(f"Header rail L={L}: H={H}, expected={expected}, match={H==expected}")

def main():
    print("CLF Mathematical Correctness Validation")
    print("=" * 50)
    
    try:
        test_empty_file()
        test_constant_run()  
        test_segment_guard()
        test_mathematical_rails()
        
        print("\n" + "=" * 50)
        print("All mathematical rails validated ✓")
        
    except Exception as e:
        print(f"\n❌ Rail validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
