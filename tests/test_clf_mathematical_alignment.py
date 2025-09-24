# test_clf_mathematical_alignment.py
"""
Test the mathematical alignment implementation
Verify Construction B multi-distance MATCH and superadditivity restoration
"""

import sys
sys.path.insert(0, '/Users/Admin/Teleport')

from teleport.clf_encoder import encode_CLF, test_canonical_decision_equation
from teleport.clf_canonical_math import CBD_BIJECTION_PROOF
from teleport.clf_receipts import assert_receipt_mathematical_consistency

def test_construction_b_repetition():
    """Test Construction B on clear repetition pattern"""
    print("=== CONSTRUCTION B REPETITION TEST ===")
    
    # Critical test case: b"ABCD" * 200
    data = b"ABCD" * 200  # 800 bytes, 4-byte pattern repeated 200 times
    print(f"Input: {data[:20]}... (800 bytes total)")
    
    # Test canonical decision equation
    result = test_canonical_decision_equation(data)
    print(f"\nCanonical Decision Equation Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Get full encoding with receipts
    tokens, receipt = encode_CLF(data, emit_receipts=True)
    
    print(f"\nReceipt:")
    print(receipt)
    
    # Analyze tokens
    if tokens:
        print(f"\nToken Analysis:")
        print(f"  Total tokens: {len(tokens)}")
        
        token_counts = {}
        for token in tokens:
            token_type = token.type
            token_counts[token_type] = token_counts.get(token_type, 0) + 1
        
        for token_type, count in token_counts.items():
            print(f"  {token_type}: {count}")
        
        # Check for MATCH tokens (should exist for repetitive pattern)
        match_tokens = [t for t in tokens if t.type == "MATCH"]
        if match_tokens:
            print(f"  SUCCESS: Found {len(match_tokens)} MATCH tokens!")
        else:
            print(f"  ISSUE: No MATCH tokens found despite repetitive pattern")
    
    return result, tokens, receipt

def test_cbd_bijection():
    """Test CBD bijection with SHA256 proof"""
    print("\n=== CBD BIJECTION TEST ===")
    
    test_data = b"Hello, World!"
    proof = CBD_BIJECTION_PROOF(test_data)
    
    print(f"Input: {test_data}")
    print(f"CBD bijection proof:")
    for key, value in proof.items():
        print(f"  {key}: {value}")
    
    if proof["BIJECTION_VALID"]:
        print("✓ CBD bijection verified")
    else:
        print("❌ CBD bijection failed")
    
    return proof

def test_mathematical_consistency():
    """Test mathematical consistency across different inputs"""
    print("\n=== MATHEMATICAL CONSISTENCY TEST ===")
    
    test_cases = [
        b"A",                    # Single byte
        b"AB",                   # Two bytes  
        b"ABC",                  # Three bytes
        b"AAAA",                 # Repetition
        b"ABCDABCD",            # Pattern
        b"Hello, World!",        # Mixed content
        b"\x00" * 100,          # Pathological case
    ]
    
    for i, data in enumerate(test_cases):
        print(f"\nTest case {i+1}: {data[:20]}{'...' if len(data) > 20 else ''}")
        
        try:
            result = test_canonical_decision_equation(data)
            tokens, receipt = encode_CLF(data, emit_receipts=True)
            
            # Verify receipt mathematical consistency
            assert_receipt_mathematical_consistency(receipt)
            print("  ✓ Mathematical consistency verified")
            
            # Key metrics
            print(f"  L={result['L']}, H={result['H']}, C_A={result['C_A']}, C_B={result['C_B']}")
            print(f"  Decision: {result['chosen']}, State: {result['state']}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    print("CLF Mathematical Alignment Test Suite")
    print("=====================================")
    
    # Test 1: Construction B repetition (critical case)
    result, tokens, receipt = test_construction_b_repetition()
    
    # Test 2: CBD bijection  
    bijection_proof = test_cbd_bijection()
    
    # Test 3: Mathematical consistency
    test_mathematical_consistency()
    
    print("\n=== SUMMARY ===")
    
    # Check critical issues
    construction_b_working = any(t.type == "MATCH" for t in tokens) if tokens else False
    bijection_working = bijection_proof["BIJECTION_VALID"]
    
    print(f"Multi-distance MATCH working: {construction_b_working}")
    print(f"CBD bijection working: {bijection_working}")
    print(f"Superadditivity: B({result['C_B_total']}) vs A({result['C_A_total']}) = {'✓' if result['C_B_total'] <= result['C_A_total'] else '❌'}")
    
    if construction_b_working and bijection_working:
        print("✓ Mathematical alignment successful!")
    else:
        print("❌ Issues remain - see details above")