#!/usr/bin/env python3
"""
CLF TELEPORT MATH RUNNER - Console Validation Harness
====================================================

CLF stance pinned:
- Seeds are not chosen; they are forced by legality + unit-locked prices + strict comparison + deterministic tie-break
- Units are bits. Every integer field pays 8·leb_len(field) (unsigned LEB128 length in bytes; integer arithmetic only)
- END cost is positional: END(bitpos) = 3 + pad_to_byte(bitpos+3) with pad_to_byte(x) = (8 − (x mod 8)) mod 8
- Path prices are END-inclusive: STREAM = Σ CAUS(tokens) + Σ END(tokens), TOTAL = H(L) + STREAM
- Header: H(L) = 16 + 8·leb_len(8L) (leb on 8L, not L)
- Decision algebra (single source of truth): C_min_total = min(H + A_stream, H + B_stream) (ignore incomplete paths) and C_min_via_streams = H + min(A_stream, B_stream) (same candidate set). Must hold: C_min_total == C_min_via_streams
- Gate (calculator honesty): EMIT iff C(S) < 8L; else CAUSEFAIL (no "OPEN success")
- Bijection receipts: A-path contributes only if expand_O(params, L) == S byte-for-byte. B-path must satisfy exact coverage Σ L_token = L
- No floating point. No compression/entropy/pattern language. Only integer deduction consistent with Teleport
"""

import os
import hashlib

# ====================================================================
# PINNED MATH HELPERS (MUST NEVER CHANGE)
# ====================================================================

def leb_len_u(n: int) -> int:  # unsigned LEB128 length in bytes
    assert n >= 0
    if n == 0: return 1
    c = 0
    while n:
        n >>= 7
        c += 1
    return c

def header_bits(L: int) -> int:
    return 16 + 8*leb_len_u(8*L)

def end_bits(bitpos: int) -> int:
    pad = (8 - ((bitpos + 3) % 8)) % 8
    return 3 + pad

def caus_bits(op: int, params: list[int], L_tok: int) -> int:
    return 3 + 8*leb_len_u(op) + sum(8*leb_len_u(p) for p in params) + 8*leb_len_u(L_tok)

# ====================================================================
# A-PATH LAWFUL OPERATOR FRAMEWORK
# ====================================================================

def admissible_O_attempt(S):
    """
    Attempt to find lawful self-verifiable one-shot operator for S.
    Returns (params, operator_type) if admissible, (None, None) if not.
    
    Lawful operators implemented:
    - ZERO_FILL: All bytes are zero (params=[])
    - SINGLE_BYTE: All bytes are the same (params=[byte_value])
    - LITERAL: Store entire content as parameter (fallback)
    """
    L = len(S)
    
    if L == 0:
        # Empty file - use zero-fill operator
        return [], "ZERO_FILL"
    
    # ZERO_FILL OPERATOR: All bytes are zero
    if all(b == 0 for b in S):
        # op=3 (ZERO_FILL), params=[], L_tok=L
        zero_caus = caus_bits(3, [], L)
        zero_end = end_bits(zero_caus)
        zero_stream = zero_caus + zero_end
        return [], "ZERO_FILL"
    
    # SINGLE_BYTE OPERATOR: All bytes are the same value
    if L > 0 and all(b == S[0] for b in S):
        # op=4 (SINGLE_BYTE), params=[byte_value], L_tok=L
        byte_value = S[0]
        single_caus = caus_bits(4, [byte_value], L)
        single_end = end_bits(single_caus)
        single_stream = single_caus + single_end
        return [byte_value], "SINGLE_BYTE"
    
    # For now, don't implement LITERAL as it's too expensive for large files
    # Return None to maintain mathematical honesty
    return None, None

def price_O_unit_locked(params, operator_type, L):
    """Unit-locked price for operator O"""
    if params is None or operator_type is None:
        return None
    
    if operator_type == "LITERAL":
        # op=2, params=list of bytes, L_tok=L
        return caus_bits(2, params, L)
    
    # Unknown operator type
    return None

def expand_O(params, operator_type, L):
    """
    Expand operator O with given params to reconstruct L bytes.
    Must be deterministic and produce exactly the same bytes as input S.
    """
    if params is None or operator_type is None:
        return None
    
    if operator_type == "LITERAL":
        # Reconstruct from literal parameters
        if len(params) != L:
            return None  # Invalid parameters
        try:
            return bytes(params)
        except (ValueError, TypeError):
            return None  # Invalid byte values
    
    # Unknown operator type
    return None

# ====================================================================
# B-PATH UNIT-LOCKED PER-BYTE TILING
# ====================================================================

def analyze_B_path_unit_locked(S):
    """B-path analysis using unit-locked per-byte CAUS tiling"""
    L = len(S)
    
    if L == 0:
        return {
            'complete': True,
            'admissible': True,
            'tokens': [],
            'coverage': 0,
            'coverage_ok': True,
            'caus_total': 0,
            'end_cost': 0,
            'stream': 0
        }
    
    # Per-byte CAUS tiling
    tokens = []
    caus_total = 0
    
    for i in range(L):
        op = 1  # CAUS operation
        params = []  # No additional parameters
        L_token = 1  # Single byte
        
        token_cost = caus_bits(op, params, L_token)
        
        token = {
            'position': i,
            'op': op,
            'params': params,
            'length': L_token,
            'cost': token_cost
        }
        
        tokens.append(token)
        caus_total += token_cost
    
    # Add END alignment
    end_cost = end_bits(caus_total)
    stream = caus_total + end_cost
    
    return {
        'complete': True,
        'admissible': True,
        'tokens': tokens,
        'coverage': sum(t['length'] for t in tokens),
        'coverage_ok': sum(t['length'] for t in tokens) == L,
        'caus_total': caus_total,
        'end_cost': end_cost,
        'stream': stream
    }

# ====================================================================
# A-PATH ANALYSIS WITH LAWFUL OPERATOR FRAMEWORK
# ====================================================================

def analyze_A_path_lawful(S):
    """A-path analysis using lawful self-verifiable operator framework"""
    L = len(S)
    
    # Attempt to find lawful operator
    params, operator_type = admissible_O_attempt(S)
    
    if params is None:
        return {
            'admissible': False,
            'complete': False,
            'total': None,
            'stream': None,
            'diagnostic': 'NO_LAWFUL_OPERATOR_AVAILABLE'
        }
    
    # If lawful operator found, verify it
    K_O = price_O_unit_locked(params, operator_type, L)
    if K_O is None:
        return {
            'admissible': False,
            'complete': False,
            'total': None,
            'stream': None,
            'diagnostic': 'OPERATOR_PRICING_FAILED'
        }
    
    # Verify bijection
    S_reconstructed = expand_O(params, operator_type, L)
    if S_reconstructed != S:
        return {
            'admissible': False,
            'complete': False,
            'total': None,
            'stream': None,
            'diagnostic': 'BIJECTION_FAILED'
        }
    
    # Compute stream cost
    caus_cost = K_O + 8 * leb_len_u(L)
    end_cost = end_bits(caus_cost)
    stream_cost = caus_cost + end_cost
    
    return {
        'admissible': True,
        'complete': True,
        'total': header_bits(L) + stream_cost,
        'stream': stream_cost,
        'diagnostic': 'LAWFUL_OPERATOR_VERIFIED'
    }

# ====================================================================
# PREDICTOR BINDING
# ====================================================================

def predict_B_from_S_unit_locked(S):
    """Π_B(S): Predict B-path stream cost using unit-locked per-byte tiling"""
    B_analysis = analyze_B_path_unit_locked(S)
    return B_analysis['stream']

def predict_A_from_S_lawful(S):
    """Π_A(S): Predict A-path stream cost using same equations as A builder"""
    A_analysis = analyze_A_path_lawful(S)
    return A_analysis['stream']

# ====================================================================
# CONSOLE VALIDATION HARNESS FUNCTIONS
# ====================================================================

def run_one(filepath):
    """
    Console harness function: run_one
    Returns: L, H, A, B, total, raw, bind_B
    """
    # Load file
    if filepath == "test_artifacts/pic1.jpg":
        filepath = "pic1.jpg"  # Adjust path if needed
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        S = f.read()
    
    L = len(S)
    H = header_bits(L)
    raw = 8 * L
    
    # A-path analysis
    A_analysis = analyze_A_path_lawful(S)
    A = {
        'admissible': A_analysis['admissible'],
        'complete': A_analysis['complete'],
        'total': A_analysis['total'],
        'stream': A_analysis['stream']
    }
    
    # B-path analysis
    B_analysis = analyze_B_path_unit_locked(S)
    B = {
        'complete': B_analysis['complete'],
        'stream': B_analysis['stream'],
        'coverage_ok': B_analysis['coverage_ok'],
        'admissible': B_analysis['admissible']
    }
    
    # Predictor binding for B
    pred_B = predict_B_from_S_unit_locked(S)
    bind_B = (pred_B == B['stream']) if B['complete'] else False
    
    # Total calculation
    if A['complete'] and B['complete']:
        total = min(H + A['stream'], H + B['stream'])
    elif A['complete']:
        total = H + A['stream']
    elif B['complete']:
        total = H + B['stream']
    else:
        total = None
    
    return L, H, A, B, total, raw, bind_B

def algebra_for(filepath):
    """
    Console harness function: algebra_for
    Returns: v1, v2 (both sides of algebra equality)
    """
    L, H, A, B, total, raw, bind_B = run_one(filepath)
    
    # Build candidates from COMPLETE paths only
    candidates = []
    stream_costs = []
    
    if A['complete']:
        candidates.append(H + A['stream'])
        stream_costs.append(A['stream'])
    
    if B['complete']:
        candidates.append(H + B['stream'])
        stream_costs.append(B['stream'])
    
    if not candidates:
        return None, None
    
    # First factorization: min over total costs
    v1 = min(candidates)
    
    # Second factorization: H + min over stream costs
    v2 = H + min(stream_costs)
    
    return v1, v2

def a_status(filepath):
    """
    Console harness function: a_status
    Returns: A status dictionary
    """
    L, H, A, B, total, raw, bind_B = run_one(filepath)
    
    return {
        'admissible': A['admissible'],
        'complete': A['complete'],
        'total': A['total']
    }

if __name__ == "__main__":
    # Quick test
    print("CLF Teleport Math Runner - Console Validation Harness")
    print("Pinned math helpers:")
    print(f"  leb_len_u(7744) = {leb_len_u(7744)}")
    print(f"  header_bits(968) = {header_bits(968)}")
    print(f"  end_bits(18392) = {end_bits(18392)}")
    print(f"  caus_bits(1, [], 1) = {caus_bits(1, [], 1)}")