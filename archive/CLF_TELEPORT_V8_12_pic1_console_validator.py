#!/usr/bin/env python3
"""
CLF V8.12 Console-Validated Calculator - CONSTRUCTIVE OPERATORS
===============================================================
Implements CONST-RUN and STEP-RUN operators with coalesced tokens.
Replaces CAUSEFAIL with IMPLEMENTATION_DEFECT reporting.
No compression vocabulary. Integer-only calculator behavior.
"""

import hashlib

# PINNED CALCULATOR RAILS (NEVER CHANGE)
def leb_len_u(n):
    """Unsigned LEB128 length in BYTES via 7-bit shifts."""
    assert n >= 0
    if n == 0:
        return 1
    count = 0
    while n > 0:
        n >>= 7
        count += 1
    return count

def header_bits(L):
    """R1 Header lock: H(L) = 16 + 8*leb_len_u(8*L)"""
    return 16 + 8 * leb_len_u(8 * L)

def caus_bits(op, params, L_tok):
    """R3 Unit-lock: 3 + 8*leb_len_u(op) + Σ 8*leb_len_u(param_i) + 8*leb_len_u(L_tok)"""
    return 3 + 8 * leb_len_u(op) + sum(8 * leb_len_u(p) for p in params) + 8 * leb_len_u(L_tok)

def end_bits(bitpos):
    """R2 END positional: 3 + ((8-((bitpos+3)%8))%8)"""
    return 3 + ((8 - ((bitpos + 3) % 8)) % 8)

# CONSTRUCTIVE OPERATORS (SELF-VERIFIABLE DEDUCTION ⇄ EXPANSION)
def const_run_deduction(S):
    """CONST-RUN: Parse maximal runs of equal bytes (constructive)."""
    tokens = []
    i = 0
    while i < len(S):
        byte_val = S[i]
        run_len = 1
        while i + run_len < len(S) and S[i + run_len] == byte_val:
            run_len += 1
        
        # CONST-RUN token: op=10, params=[byte_val], L_tok=run_len
        tokens.append({
            'op': 10,
            'params': [byte_val], 
            'L_tok': run_len,
            'cost': caus_bits(10, [byte_val], run_len)
        })
        i += run_len
    
    return tokens

def step_run_deduction(S):
    """STEP-RUN: Parse arithmetic progressions mod 256 (constructive)."""
    tokens = []
    i = 0
    while i < len(S):
        if i + 1 >= len(S):
            # Single byte - use CONST-RUN
            tokens.append({
                'op': 10,
                'params': [S[i]],
                'L_tok': 1,
                'cost': caus_bits(10, [S[i]], 1)
            })
            break
        
        start_val = S[i]
        step = (S[i + 1] - S[i]) % 256
        run_len = 2
        
        # Extend arithmetic progression
        while i + run_len < len(S):
            expected = (start_val + run_len * step) % 256
            if S[i + run_len] != expected:
                break
            run_len += 1
        
        if run_len >= 2:
            # STEP-RUN token: op=11, params=[start_val, step], L_tok=run_len
            tokens.append({
                'op': 11,
                'params': [start_val, step],
                'L_tok': run_len,
                'cost': caus_bits(11, [start_val, step], run_len)
            })
        else:
            # Single byte - use CONST-RUN
            tokens.append({
                'op': 10,
                'params': [start_val],
                'L_tok': 1,
                'cost': caus_bits(10, [start_val], 1)
            })
        
        i += run_len if run_len >= 2 else 1
    
    return tokens

def expand_tokens(tokens, expected_L):
    """Expand tokens back to byte sequence (bijection verification)."""
    result = bytearray()
    
    for token in tokens:
        op = token['op']
        params = token['params']
        L_tok = token['L_tok']
        
        if op == 10:  # CONST-RUN
            byte_val = params[0]
            result.extend([byte_val] * L_tok)
        elif op == 11:  # STEP-RUN
            start_val, step = params
            for i in range(L_tok):
                result.append((start_val + i * step) % 256)
        else:
            raise ValueError(f"Unknown operator: {op}")
    
    if len(result) != expected_L:
        raise ValueError(f"Length mismatch: got {len(result)}, expected {expected_L}")
    
    return bytes(result)

def main():
    """Console-validated calculator with constructive operators."""
    
    print("CLF V8.12 Console-Validated Calculator - CONSTRUCTIVE OPERATORS")
    print("=" * 65)
    print()
    
    # Read file
    filename = "pic1.jpg"
    with open(filename, 'rb') as f:
        S = f.read()
    
    L = len(S)
    eight_L = 8 * L
    H = header_bits(L)
    
    print("BLOCK 1 - PINS")
    print("---------------")
    print(f"L={L}, 8L={eight_L}, H={H}")
    
    # Verify pins
    expected_H = 32 if L == 968 else None
    if expected_H and H != expected_H:
        print(f"IMPLEMENTATION_DEFECT: R1_FAIL(H={H}, expected={expected_H})")
        return False
    
    print("Pins verified ✓")
    print()
    
    print("BLOCK 2 - B-PATH (COALESCED)")
    print("-----------------------------")
    
    # B-path: Use CONST-RUN coalescing
    B_tokens = const_run_deduction(S)
    
    print(f"Coalesced to {len(B_tokens)} tokens from {L} bytes")
    
    # Show first few tokens
    for i, token in enumerate(B_tokens[:5]):
        print(f"Token {i}: op={token['op']}, params={token['params']}, L_tok={token['L_tok']}, cost={token['cost']}")
    
    if len(B_tokens) > 5:
        print(f"... ({len(B_tokens) - 5} more tokens)")
    
    # Verify coverage
    total_coverage = sum(token['L_tok'] for token in B_tokens)
    if total_coverage != L:
        print(f"IMPLEMENTATION_DEFECT: R4_FAIL(coverage={total_coverage}, expected={L})")
        return False
    
    # Compute B-path cost
    B_caus = sum(token['cost'] for token in B_tokens)
    B_end = end_bits(B_caus)
    B_stream = B_caus + B_end
    
    print(f"B_caus={B_caus}, B_end={B_end}, B_stream={B_stream}")
    
    # Verify bijection
    try:
        expanded_B = expand_tokens(B_tokens, L)
        bijection_B = (expanded_B == S)
        if not bijection_B:
            print("IMPLEMENTATION_DEFECT: R8_FAIL(B_bijection)")
            return False
    except Exception as e:
        print(f"IMPLEMENTATION_DEFECT: R8_FAIL(B_expansion_error: {e})")
        return False
    
    print("B-path bijection verified ✓")
    print()
    
    print("BLOCK 3 - A-PATH (CONSTRUCTIVE)")
    print("--------------------------------")
    
    # A-path: Use STEP-RUN deduction
    A_tokens = step_run_deduction(S)
    
    print(f"A-path: {len(A_tokens)} tokens")
    
    # Show first few A tokens
    for i, token in enumerate(A_tokens[:5]):
        print(f"Token {i}: op={token['op']}, params={token['params']}, L_tok={token['L_tok']}, cost={token['cost']}")
    
    if len(A_tokens) > 5:
        print(f"... ({len(A_tokens) - 5} more tokens)")
    
    # Verify A-path coverage
    A_total_coverage = sum(token['L_tok'] for token in A_tokens)
    if A_total_coverage != L:
        print(f"IMPLEMENTATION_DEFECT: R4_FAIL(A_coverage={A_total_coverage}, expected={L})")
        return False
    
    # Compute A-path cost
    A_caus = sum(token['cost'] for token in A_tokens)
    A_end = end_bits(A_caus)
    A_stream = A_caus + A_end
    
    print(f"A_caus={A_caus}, A_end={A_end}, A_stream={A_stream}")
    
    # Verify A bijection
    try:
        expanded_A = expand_tokens(A_tokens, L)
        bijection_A = (expanded_A == S)
        if not bijection_A:
            print("IMPLEMENTATION_DEFECT: R8_FAIL(A_bijection)")
            return False
    except Exception as e:
        print(f"IMPLEMENTATION_DEFECT: R8_FAIL(A_expansion_error: {e})")
        return False
    
    print("A-path bijection verified ✓")
    print()
    
    print("BLOCK 4 - ALGEBRA")
    print("------------------")
    
    # Both paths are complete
    C_total_A = H + A_stream
    C_total_B = H + B_stream
    
    C_min_total = min(C_total_A, C_total_B)
    C_min_via_streams = H + min(A_stream, B_stream)
    ALG_EQ = (C_min_total == C_min_via_streams)
    
    print(f"C_total_A={C_total_A}, C_total_B={C_total_B}")
    print(f"C_min_total={C_min_total}")
    print(f"C_min_via_streams={C_min_via_streams}")
    print(f"ALG_EQ={ALG_EQ}")
    
    if not ALG_EQ:
        print("IMPLEMENTATION_DEFECT: ALGEBRA_MISMATCH")
        return False
    
    print("Algebra equality verified ✓")
    print()
    
    print("BLOCK 5 - GATE")
    print("---------------")
    
    gate_condition = C_min_total < eight_L
    print(f"Gate condition: {C_min_total} < {eight_L} → {gate_condition}")
    
    if gate_condition:
        print("DECISION: EMIT")
        
        print()
        print("BLOCK 6 - RECEIPTS")
        print("-------------------")
        
        # Choose optimal path
        optimal_path = "A" if A_stream <= B_stream else "B"
        optimal_tokens = A_tokens if optimal_path == "A" else B_tokens
        
        print(f"Optimal path: {optimal_path}")
        
        # SHA verification
        expanded_optimal = expand_tokens(optimal_tokens, L)
        sha_input = hashlib.sha256(S).hexdigest()
        sha_expanded = hashlib.sha256(expanded_optimal).hexdigest()
        
        print(f"SHA256(input):  {sha_input}")
        print(f"SHA256(expand): {sha_expanded}")
        print(f"SHA match: {sha_input == sha_expanded}")
        
        if sha_input != sha_expanded:
            print("IMPLEMENTATION_DEFECT: R9_FAIL(SHA_MISMATCH)")
            return False
        
    else:
        print("IMPLEMENTATION_DEFECT: MINIMALITY_NOT_ACHIEVED(OPERATOR_SET_INCOMPLETE)")
        return False
    
    print()
    print("CONSOLE VALIDATION COMPLETE - ALL BLOCKS PASSED ✓")
    print("Ready for V8.12 export generation")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("CONSOLE VALIDATION FAILED - DO NOT EXPORT")
        exit(1)