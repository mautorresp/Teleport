"""
CLF Canonical Encoder - Global Dynamic Programming

Implements exact dynamic programming to find globally optimal token sequences
under CLF cost model with deterministic tie-breaking.
"""

from typing import List, Tuple, Optional
from functools import lru_cache
from teleport.clf_int import leb, pad_to_byte
from teleport.seed_format import OP_LIT, OP_MATCH, OP_CONST, OP_STEP

def _lcp(a: bytes, i: int, b: bytes, j: int, n: int) -> int:
    """Longest common prefix of a[i:] and b[j:], up to n bytes, integer-only"""
    k = 0
    limit = n
    while k < limit and i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
        k += 1
    return k

def _max_const_run(S: bytes, p: int) -> int:
    """Find maximal constant run starting at position p"""
    if p >= len(S):
        return 0
    
    byte_val = S[p]
    L = 1
    while p + L < len(S) and S[p + L] == byte_val:
        L += 1
    return L

def _deduce_step(S: bytes, p: int) -> Tuple[int, int, int]:
    """
    Deduce arithmetic step: returns (L, start, stride) where L is maximal domain.
    Returns (0, 0, 0) if no valid step pattern.
    """
    if p + 1 >= len(S):
        return (0, 0, 0)
    
    start = S[p]
    stride = (S[p + 1] - start) & 255  # Modular arithmetic in Z_256
    L = 2
    
    while p + L < len(S) and S[p + L] == ((start + L * stride) & 255):
        L += 1
    
    return (L, start, stride)

def enumerate_tokens_at(S: bytes, p: int) -> List[Tuple[str, Tuple, int, int]]:
    """
    Returns all admissible tokens from position p.
    Each element: (kind, params_tuple, L, bit_cost)
    """
    N = len(S)
    out = []

    # 1) LIT candidates: all 1..min(10, N-p)
    max_lit = min(10, N - p)
    for L in range(1, max_lit + 1):
        out.append(("LIT", (L,), L, 10 * L))

    # 2) MATCH candidates: exhaustive enumeration
    for j in range(p):
        D = p - j  # distance back
        if D <= 0:
            continue
        L = _lcp(S, j, S, p, N - p)
        if L >= 3:
            c = 2 + 8 * leb(D) + 8 * leb(L)
            out.append(("MATCH", (D,), L, c))

    # 3) CAUS.CONST: all-equal run
    L_const = _max_const_run(S, p)
    if L_const >= 1:
        c = 3 + 8 * leb(OP_CONST) + 8 * leb(S[p]) + 8 * leb(L_const)
        out.append(("CAUS.CONST", (S[p],), L_const, c))

    # 4) CAUS.STEP: arithmetic sequence
    L_step, start, stride = _deduce_step(S, p)
    if L_step >= 1:
        c = 3 + 8 * leb(OP_STEP) + 8 * leb(start) + 8 * leb(stride) + 8 * leb(L_step)
        out.append(("CAUS.STEP", (start, stride), L_step, c))

    return out

def _token_rank(kind: str) -> int:
    """Return canonical ordering rank for tie-breaking"""
    if kind == "LIT":
        return 0
    elif kind == "MATCH":
        return 1
    elif kind == "CAUS.CONST":
        return 2
    elif kind == "CAUS.STEP":
        return 3
    else:
        return 999

def _canonical_key(kind: str, params: Tuple, L: int) -> Tuple[int, Tuple]:
    """
    Return canonical key for deterministic tie-breaking.
    Orders by: tag rank, then parameters, then length.
    """
    rank = _token_rank(kind)
    return (rank, params, L)

def solve_dp(S: bytes) -> Tuple[List[Tuple], int]:
    """
    Iterative DP solver: returns (choices, total_cost).
    Solves backwards from end to avoid recursion limits.
    """
    N = len(S)
    
    # DP arrays: dp[p] = (cost_from_p, choice_at_p)
    dp = [None] * (N + 1)
    dp[N] = (0, None)  # Base case: end of string
    
    # Fill DP table backwards
    for p in range(N - 1, -1, -1):
        best = None  # (cost, canonical_key, choice)
        
        # Enumerate all admissible tokens at p
        candidates = enumerate_tokens_at(S, p)
        
        for (kind, params, L, c_bits) in candidates:
            if p + L > N:
                continue  # Invalid length
                
            # Cost from suffix
            cost_suffix = dp[p + L][0]
            total_cost = c_bits + cost_suffix
            
            # Canonical tie-breaking key
            canonical_key = _canonical_key(kind, params, L)
            
            candidate = (total_cost, canonical_key, (kind, params, L))
            
            if best is None or candidate < best:
                best = candidate
        
        if best is None:
            raise RuntimeError(f"No admissible tokens at position {p}")
        
        dp[p] = (best[0], best[2])  # (cost, choice)
    
    # Reconstruct solution
    choices = []
    p = 0
    while p < N:
        choice = dp[p][1]
        if choice is None:
            break
        choices.append(choice)
        _, _, L = choice
        p += L
    
    total_cost = dp[0][0] if dp[0] else 0
    return choices, total_cost

def token_cost_bits(kind: str, params: Tuple, L: int) -> int:
    """Calculate exact bit cost for a token"""
    if kind == "LIT":
        return 10 * L
    elif kind == "MATCH":
        D = params[0]
        return 2 + 8 * leb(D) + 8 * leb(L)
    elif kind == "CAUS.CONST":
        b = params[0]
        return 3 + 8 * leb(OP_CONST) + 8 * leb(b) + 8 * leb(L)
    elif kind == "CAUS.STEP":
        start, stride = params
        return 3 + 8 * leb(OP_STEP) + 8 * leb(start) + 8 * leb(stride) + 8 * leb(L)
    else:
        raise ValueError(f"Unknown token kind: {kind}")

def canonize_bytes_dp(S: bytes, print_receipts: bool = False) -> Tuple[List[Tuple], int]:
    """
    Global DP canonicalization with receipts.
    Returns: (choices, total_bit_cost)
    """
    # Solve using iterative DP
    choices, token_cost_total = solve_dp(S)
    
    # Print receipts for first 3 non-LIT choices
    if print_receipts:
        receipt_count = 0
        p = 0
        for choice in choices:
            kind, params, L = choice
            token_cost = token_cost_bits(kind, params, L)
            
            if receipt_count < 3 and kind != "LIT":
                if L <= 10:
                    C_lit = 10 * L
                    strict_ineq = 1 if token_cost < C_lit else 0
                    print(f"p={p} chosen={kind}{params},L={L} C_token={token_cost} C_LIT({L})={C_lit} strict_ineq={strict_ineq}")
                else:
                    print(f"p={p} chosen={kind}{params},L={L} C_token={token_cost} C_LIT({L})=inadmissible(L>10) forced_selection=1")
                receipt_count += 1
            
            p += L

    # Add END cost
    C_end = 3 + pad_to_byte(token_cost_total + 3)
    total_bits = token_cost_total + C_end
    
    if print_receipts:
        print(f"C_END= {C_end}")
    
    return choices, total_bits

def canonize_dp(S: bytes, print_receipts: bool = False) -> bytes:
    """
    Main DP canonicalization function.
    Returns canonical minimal seed bytes.
    """
    from teleport.seed_format import emit_LIT, emit_MATCH, emit_CAUS
    from teleport.encoder import emit_END
    
    choices, total_bits = canonize_bytes_dp(S, print_receipts)
    
    # Build seed from choices - need to track position for LIT
    seed_parts = []
    p = 0
    
    for kind, params, L in choices:
        if kind == "LIT":
            L = params[0]
            block = S[p:p + L]
            seed_parts.append(emit_LIT(block))
        elif kind == "MATCH":
            D = params[0]
            seed_parts.append(emit_MATCH(D, L))
        elif kind == "CAUS.CONST":
            b = params[0]
            seed_parts.append(emit_CAUS(OP_CONST, [b], L))
        elif kind == "CAUS.STEP":
            start, stride = params
            seed_parts.append(emit_CAUS(OP_STEP, [start, stride], L))
        
        p += L
    
    # Add END token
    seed_parts.append(emit_END(total_bits - 3 - pad_to_byte(total_bits)))
    
    return b''.join(seed_parts)
