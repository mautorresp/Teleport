#!/usr/bin/env python3
"""
Canonical DP implementation for CLF minimality.
Implements exact DP: F(x) = min over all tokens covering [x-ℓ, x) of (F(x-ℓ) + C_stream(t))
"""

from typing import List, Tuple, Optional
from teleport.clf_int import assert_boundary_types
from teleport.clf_canonical import OP_CONST, OP_CBD256, compute_cost_receipts, expand_with_context


def generate_all_candidates(segment: bytes, pos: int, max_len: int) -> List[Tuple[int, tuple, int]]:
    """Generate all possible tokens starting at pos with length <= max_len."""
    candidates = []
    L = len(segment)
    
    if pos >= L:
        return candidates
    
    # CONST tokens of various lengths
    byte_val = segment[pos]
    run_end = pos + 1
    while run_end < L and run_end <= pos + max_len and segment[run_end] == byte_val:
        run_end += 1
    
    # Add CONST candidates of length 1 to (run_end - pos)
    for length in range(1, min(run_end - pos + 1, max_len + 1)):
        candidates.append((OP_CONST, (byte_val,), length))
    
    # CBD256 token covering [pos:pos+length] for each possible length
    for length in range(1, min(L - pos + 1, max_len + 1)):
        candidates.append((OP_CBD256, None, length))  # params computed later
    
    return candidates


def compute_exact_token_cost(segment: bytes, pos: int, op_id: int, length: int, context: bytes) -> Optional[Tuple[int, tuple, dict]]:
    """
    Compute exact cost for token covering segment[pos:pos+length].
    Returns (op_id, params, cost_info) or None if invalid.
    """
    if pos + length > len(segment):
        return None
    
    token_segment = segment[pos:pos + length]
    
    # Compute parameters
    if op_id == OP_CONST:
        if length == 0:
            return None
        # Verify all bytes are the same
        byte_val = token_segment[0]
        if not all(b == byte_val for b in token_segment):
            return None
        params = (byte_val,)
    elif op_id == OP_CBD256:
        # Compute K = Σ segment[i] * 256^(length-1-i)
        K = 0
        for byte_val in token_segment:
            K = (K << 8) | byte_val
        params = (K,)
    else:
        return None
    
    try:
        # Compute cost and verify segment guard
        cost_info = compute_cost_receipts(op_id, params, length)
        if cost_info['C_stream'] >= 10 * length:
            return None  # Fails segment guard
        
        # Verify seed-only expansion matches
        expanded = expand_with_context(op_id, params, length, context)
        if expanded != token_segment:
            return None  # Expansion mismatch
        
        return (op_id, params, cost_info)
    
    except Exception:
        return None  # Invalid token


def canonical_dp_minimize(segment: bytes) -> List[Tuple[int, tuple, int, dict]]:
    """
    Canonical DP minimization: F(x) = min over tokens ending at x of (F(x-ℓ) + C_stream).
    Returns optimal token sequence or raises OpenError if no valid tiling exists.
    """
    L = len(segment)
    if L == 0:
        return []
    
    # DP state: F[x] = minimum cost to cover [0:x], parent[x] = (token_info, prev_pos)
    # PURE INTEGER ARITHMETIC - no floating point allowed
    MAX_COST = 100 * L  # Integer upper bound
    F = [MAX_COST] * (L + 1)
    parent = [None] * (L + 1)
    F[0] = 0
    
    print(f"DEBUG: DP minimization for L={L}")
    
    for x in range(1, L + 1):
        if x % 1000 == 0:
            print(f"DEBUG: DP progress {x}/{L}")
        
        # Try all possible tokens ending at position x
        for start_pos in range(x):
            token_len = x - start_pos
            
            # Skip if previous position is unreachable - integer check
            if F[start_pos] >= MAX_COST:
                continue
            
            # Reconstruct context [0:start_pos] for seed-only expansion
            context = b""
            if start_pos > 0:
                # Build context by following parent chain
                contexts = []
                pos = start_pos
                while pos > 0:
                    if parent[pos] is None:
                        break
                    token_info, prev_pos = parent[pos]
                    contexts.append((prev_pos, token_info))
                    pos = prev_pos
                
                if pos != 0:
                    continue  # Broken chain
                
                # Apply tokens in order to build context
                contexts.reverse()
                for ctx_pos, (op_id, params, _, _) in contexts:
                    ctx_len = len(params) if op_id == OP_CONST else 0  # Simplified
                    # This is a simplified context building - in practice need exact reconstruction
                    pass
                
                # For now, use direct segment reconstruction (this is the key correctness requirement)
                context = segment[:start_pos]
            
            # Generate candidate tokens of this length
            candidates = generate_all_candidates(segment, start_pos, token_len)
            
            for op_id, _, cand_len in candidates:
                if cand_len != token_len:
                    continue
                
                result = compute_exact_token_cost(segment, start_pos, op_id, token_len, context)
                if result is None:
                    continue
                
                op_id, params, cost_info = result
                total_cost = F[start_pos] + cost_info['C_stream']
                
                if total_cost < F[x]:
                    F[x] = total_cost
                    parent[x] = ((op_id, params, token_len, cost_info), start_pos)
    
    # Check if solution exists - pure integer check
    if F[L] >= MAX_COST:
        from teleport.clf_canonical import OpenError
        raise OpenError("No valid DP solution found")
    
    # Reconstruct optimal path
    tokens = []
    pos = L
    while pos > 0:
        if parent[pos] is None:
            from teleport.clf_canonical import OpenError
            raise OpenError("DP parent chain broken")
        
        token_info, prev_pos = parent[pos]
        tokens.append(token_info)
        pos = prev_pos
    
    tokens.reverse()
    
    print(f"DEBUG: DP found optimal solution with {len(tokens)} tokens, cost={F[L]}")
    return tokens
