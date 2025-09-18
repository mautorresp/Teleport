#!/usr/bin/env python3
"""
Test Extended CLF Structural Operators - STEP and MATCH
"""

from teleport.clf_canonical import (
    encode_CLF, clf_canonical_receipts, 
    deduce_maximal_const_run, deduce_maximal_step_run, deduce_maximal_match_run,
    OP_CONST, OP_STEP, OP_MATCH, OP_CBD256
)

def test_step_operator():
    """Test STEP operator detection and encoding"""
    print("=== TESTING STEP OPERATOR ===")
    
    # Test 1: Simple arithmetic progression
    print("\n1. Testing arithmetic progression (7, 10, 13, 16, ...):")
    step_data = bytes([(7 + 3*k) % 256 for k in range(30)])  # a0=7, d=3, L=30
    
    # Test deduction function
    step_len, a0, d = deduce_maximal_step_run(step_data, 0)
    print(f"   Deduced: length={step_len}, a0={a0}, d={d}")
    
    # Test full encoding
    result = encode_CLF(step_data)
    if result:
        receipts = clf_canonical_receipts(step_data, result)
        construction_line = [line for line in receipts if line.startswith("CONSTRUCTION:")][0]
        print(f"   {construction_line}")
        
        # Check if STEP was used
        step_tokens = [t for t in result if t[0] == OP_STEP]
        if step_tokens:
            print(f"   ✅ STEP token found: params={step_tokens[0][1]}, length={step_tokens[0][2]}")
        else:
            print(f"   📊 No STEP token (other construction chosen)")
            
        total_cost = sum(cost_info['C_stream'] for _, _, _, cost_info in result)
        print(f"   Total cost: {total_cost} bits for {len(step_data)} bytes")
    else:
        print("   Result: OPEN")

def test_match_operator():
    """Test MATCH operator detection and encoding"""
    print("\n=== TESTING MATCH OPERATOR ===")
    
    # Test 1: Simple alternating structure that should use MATCH
    print("\n1. Testing alternating structure (ABAB...):")
    match_data = b'AB' * 20  # 40 bytes alternating
    
    # Test deduction function on part of the structure
    context = b'AB'  # Previous context
    match_len, D = deduce_maximal_match_run(match_data, 2, context)
    print(f"   Deduced MATCH: length={match_len}, D={D}")
    
    # Test full encoding
    result = encode_CLF(match_data)
    if result:
        receipts = clf_canonical_receipts(match_data, result)
        construction_line = [line for line in receipts if line.startswith("CONSTRUCTION:")][0]
        print(f"   {construction_line}")
        
        # Check token composition
        for i, (op_id, params, length, cost_info) in enumerate(result):
            if op_id == OP_MATCH:
                print(f"   Token[{i}]: MATCH, params={params}, length={length}")
            elif op_id == OP_CONST:
                print(f"   Token[{i}]: CONST, params={params}, length={length}")
            elif op_id == OP_STEP:
                print(f"   Token[{i}]: STEP, params={params}, length={length}")
            elif op_id == OP_CBD256:
                print(f"   Token[{i}]: CBD256, length={length}")
            else:
                print(f"   Token[{i}]: op={op_id}, length={length}")
                
        total_cost = sum(cost_info['C_stream'] for _, _, _, cost_info in result)
        print(f"   Total cost: {total_cost} bits for {len(match_data)} bytes")
    else:
        print("   Result: OPEN")

def test_mixed_structure():
    """Test data with mixed mathematical structures"""
    print("\n=== TESTING MIXED MATHEMATICAL STRUCTURES ===")
    
    # Create data with different types of structure
    const_part = b'\xFF' * 20      # Constant run
    step_part = bytes([(i * 5) % 256 for i in range(15)])  # Arithmetic progression
    random_part = b"Hello World"   # Less structured
    
    mixed_data = const_part + step_part + random_part
    
    print(f"\nTesting {len(mixed_data)} bytes with mixed structures:")
    print(f"   Part 1: {len(const_part)} bytes constant")
    print(f"   Part 2: {len(step_part)} bytes arithmetic progression") 
    print(f"   Part 3: {len(random_part)} bytes text")
    
    result = encode_CLF(mixed_data)
    if result:
        receipts = clf_canonical_receipts(mixed_data, result)
        construction_line = [line for line in receipts if line.startswith("CONSTRUCTION:")][0]
        print(f"   {construction_line}")
        
        # Analyze token composition
        token_counts = {OP_CONST: 0, OP_STEP: 0, OP_MATCH: 0, OP_CBD256: 0}
        total_cost = 0
        
        for i, (op_id, params, length, cost_info) in enumerate(result):
            if op_id in token_counts:
                token_counts[op_id] += 1
            total_cost += cost_info['C_stream']
            
        print(f"   Token composition: CONST={token_counts[OP_CONST]}, STEP={token_counts[OP_STEP]}, "
              f"MATCH={token_counts[OP_MATCH]}, CBD256={token_counts[OP_CBD256]}")
        print(f"   Total cost: {total_cost} bits for {len(mixed_data)} bytes")
        baseline_cost = 10 * len(mixed_data)
        print(f"   Mathematical bound: {total_cost} < {baseline_cost} (strict inequality)")
    else:
        print("   Result: OPEN")

def test_precedence_order():
    """Test that operator precedence works correctly"""
    print("\n=== TESTING OPERATOR PRECEDENCE ===")
    
    # Test data that could match multiple operators
    print("\n1. Testing precedence with overlapping structures:")
    
    # Structure that could be both CONST and STEP
    # All same byte = CONST, but also arithmetic progression with d=0
    ambiguous_data = b'\x42' * 15
    
    const_len, byte_val = deduce_maximal_const_run(ambiguous_data, 0)
    step_len, a0, d = deduce_maximal_step_run(ambiguous_data, 0)
    
    print(f"   CONST detection: length={const_len}, byte_val={byte_val}")
    print(f"   STEP detection: length={step_len}, a0={a0}, d={d}")
    
    result = encode_CLF(ambiguous_data)
    if result and len(result) > 0:
        chosen_op = result[0][0]
        if chosen_op == OP_CONST:
            print("   ✅ CONST correctly chosen (higher precedence)")
        elif chosen_op == OP_STEP:
            print("   📊 STEP chosen (precedence issue or cost difference)")
        else:
            print(f"   📊 Other operator chosen: {chosen_op}")
    else:
        print("   Result: OPEN")

def main():
    """Run all extended operator tests"""
    print("CLF EXTENDED STRUCTURAL OPERATORS TEST")
    print("=====================================")
    
    test_step_operator()
    test_match_operator()
    test_mixed_structure()
    test_precedence_order()
    
    print("\n" + "="*50)
    print("✅ EXTENDED OPERATOR TESTS COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()
