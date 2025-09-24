"""
CLF Builder Separation System
============================

Section C: Builder separation (stop B=A aliasing for good)

Implement separate pure builders:
- build_A_exact(S) -> (C_A_stream, token_A): whole-range CBD pricing
- build_B_structural(S) -> (B_COMPLETE, C_B_stream, tokens_B, struct_counts): 
  deterministic tiler over fixed op set

Decision consumes only the outputs, never re-computes inside.
"""

from typing import Tuple, List, Dict, Optional
from teleport.clf_integer_guards import runtime_integer_guard, integer_sum
from teleport.clf_leb_lock import leb_len, compute_leb_cost_bits

# Fixed operator constants
OP_CONST = 1
OP_STEP = 2  
OP_MATCH = 3
OP_CBD = 9

def build_A_exact(S: bytes) -> Tuple[int, List]:
    """
    Pure A construction: whole-range CBD exact pricing.
    INVARIANT C.1: Must be whole-range CBD pricing with the same leb lock.
    Returns (C_A_stream, tokens_A)
    """
    L = runtime_integer_guard(len(S), "input length")
    if L == 0:
        return 0, []
    
    # A construction: single CBD token covering entire range
    # Cost components for CBD:
    # - op_id: 8 * leb_len(CBD_op_id) 
    # - param (LEB7 seed): 8 * leb_len(|leb7_param|) + 8 * |leb7_param|
    # - length: 8 * leb_len(L)
    # - END padding
    
    # CLF CAUSAL SEED REQUIREMENT: Mathematical derivation of integer seed K
    # Every byte string admits causal deduction - if A cannot derive K, this is builder incompleteness
    # Never pack S into LEB7 groups - derive K by mathematical structure analysis
    
    # Check if S has obvious causal structure that permits whole-range CBD
    if len(set(S)) == 1:
        # CONST case: seed = (byte_value, count) 
        byte_val = S[0]
        K_seed = byte_val  # Minimal seed for constant data
        
        # Estimate cost: CBD(K_seed, L) 
        C_op = 8 * runtime_integer_guard(1, "CBD op")  # 1 byte for OP_CBD
        C_param = 8 * runtime_integer_guard(1, "seed param")  # 1 byte for constant value
        C_length = 8 * runtime_integer_guard(3, "length param")  # ~3 bytes for L (conservative)
        C_END = runtime_integer_guard(8, "CBD END")
        
        C_A_stream = runtime_integer_guard(C_op + C_param + C_length + C_END, "A CONST stream")
        
        # Only emit if strictly minimal
        RAW_BITS = runtime_integer_guard(8 * L, "raw bits")
        if C_A_stream >= RAW_BITS:
            return None, []  # Not minimal, force B path
            
        # Token for CONST case with causal seed provenance
        token_A = [('CBD_CONST', K_seed, L, {
            'C_stream': C_A_stream, 
            'construction_method': 'CAUSAL-SEED-CONST',
            'seed_origin': 'DERIVED_FROM_A_EXACT',
            'seed_value': K_seed,
            'C_END': C_END
        }, 0)]
        
        return C_A_stream, token_A
            
    else:
        # A builder mathematical derivation incomplete - defer to B structural tiling
        # This is builder incompleteness, not a property of the string
        return None, []

def build_B_structural(S: bytes) -> Tuple[bool, int, List, Dict]:
    """
    Pure B construction: deterministic structural tiler.
    INVARIANT C.2: Deterministic tiler over fixed op set {CONST, STEP, MATCH, ...}.
    Returns (B_COMPLETE, C_B_stream, tokens_B, struct_counts)
    """
    L = runtime_integer_guard(len(S), "input length")
    if L == 0:
        return True, 0, [], {}
    
    tokens_B = []
    pos = 0
    struct_counts = {'CONST': 0, 'STEP': 0, 'MATCH': 0, 'CBD_TILE': 0}
    
    # INVARIANT C.3: Coverage exactness: Σ token lengths = L
    total_covered = 0
    
    try:
        while pos < L:
            current_pos = pos
            token_created = False
            
            # Strategy 1: CONST detection (homogeneous runs)
            if pos < L:
                run_length = _detect_const_run(S, pos)
                if run_length >= 1:  # Accept any CONST run
                    byte_val = S[pos]
                    # CONST cost: op + param + length + END
                    C_op = compute_leb_cost_bits(OP_CONST) 
                    C_param = compute_leb_cost_bits(byte_val)
                    C_length = compute_leb_cost_bits(run_length)
                    C_CAUS = runtime_integer_guard(C_op + C_param + C_length, "CONST CAUS")
                    pad_bits = runtime_integer_guard((8 - ((C_CAUS + 3) % 8)) % 8, "CONST padding")
                    C_END = runtime_integer_guard(3 + pad_bits, "CONST END")
                    C_stream = runtime_integer_guard(C_CAUS + C_END, "CONST total")
                    
                    tokens_B.append(('CONST', (byte_val,), run_length, {'C_stream': C_stream}, pos))
                    pos += run_length
                    total_covered += run_length
                    struct_counts['CONST'] += 1
                    token_created = True
            
            # Strategy 2: STEP detection (arithmetic progressions)
            if not token_created and pos + 2 < L:
                step_length = _detect_step_run(S, pos)
                if step_length >= 3:  # Minimum viable STEP
                    a0 = S[pos]
                    d = (S[pos + 1] - S[pos]) % 256
                    # STEP cost calculation
                    C_op = compute_leb_cost_bits(OP_STEP)
                    C_param_a0 = compute_leb_cost_bits(a0)
                    C_param_d = compute_leb_cost_bits(d)
                    C_length = compute_leb_cost_bits(step_length)
                    C_CAUS = runtime_integer_guard(C_op + C_param_a0 + C_param_d + C_length, "STEP CAUS")
                    pad_bits = runtime_integer_guard((8 - ((C_CAUS + 3) % 8)) % 8, "STEP padding")
                    C_END = runtime_integer_guard(3 + pad_bits, "STEP END")
                    C_stream = runtime_integer_guard(C_CAUS + C_END, "STEP total")
                    
                    tokens_B.append(('STEP', (a0, d), step_length, {'C_stream': C_stream}, pos))
                    pos += step_length
                    total_covered += step_length
                    struct_counts['STEP'] += 1
                    token_created = True
            
            # Strategy 3: MATCH detection (careful window bounds)
            if not token_created and pos >= 1:
                match_result = _detect_match_run(S, pos)
                if match_result and match_result[0] >= 3:  # Minimum viable MATCH
                    match_length, D = match_result
                    # INVARIANT C.7: Window legality check BEFORE pricing
                    if _is_match_legal(pos, D, match_length, L):
                        C_op = compute_leb_cost_bits(OP_MATCH) 
                        C_param_D = compute_leb_cost_bits(D)
                        C_length = compute_leb_cost_bits(match_length)
                        C_CAUS = runtime_integer_guard(C_op + C_param_D + C_length, "MATCH CAUS")
                        pad_bits = runtime_integer_guard((8 - ((C_CAUS + 3) % 8)) % 8, "MATCH padding")
                        C_END = runtime_integer_guard(3 + pad_bits, "MATCH END")
                        C_stream = runtime_integer_guard(C_CAUS + C_END, "MATCH total")
                        
                        tokens_B.append(('MATCH', (D, match_length), match_length, {'C_stream': C_stream}, pos))
                        pos += match_length
                        total_covered += match_length
                        struct_counts['MATCH'] += 1
                        token_created = True
            
            # Strategy 4: CBD tile (fallback for remaining bytes)
            if not token_created:
                # Small CBD tile for remaining bytes
                remaining = L - pos
                tile_size = min(64, remaining)  # Small tiles
                
                # CBD tile cost (conservative)
                C_op = compute_leb_cost_bits(OP_CBD)
                C_param_len = compute_leb_cost_bits(tile_size + 16)  # Conservative seed size
                C_param_data = runtime_integer_guard(8 * (tile_size + 16), "CBD tile data")
                C_length = compute_leb_cost_bits(tile_size)
                C_CAUS = runtime_integer_guard(C_op + C_param_len + C_param_data + C_length, "CBD tile CAUS")
                pad_bits = runtime_integer_guard((8 - ((C_CAUS + 3) % 8)) % 8, "CBD tile padding")
                C_END = runtime_integer_guard(3 + pad_bits, "CBD tile END")
                C_stream = runtime_integer_guard(C_CAUS + C_END, "CBD tile total")
                
                tokens_B.append(('CBD_TILE', tile_size + 16, tile_size, {'C_stream': C_stream}, pos))
                pos += tile_size
                total_covered += tile_size
                struct_counts['CBD_TILE'] += 1
                token_created = True
            
            # Safety: prevent infinite loops
            if pos == current_pos:
                # Force progress with single byte
                byte_val = S[pos]
                C_op = compute_leb_cost_bits(OP_CONST)
                C_param = compute_leb_cost_bits(byte_val)
                C_length = compute_leb_cost_bits(1)
                C_CAUS = runtime_integer_guard(C_op + C_param + C_length, "fallback CONST CAUS")
                pad_bits = runtime_integer_guard((8 - ((C_CAUS + 3) % 8)) % 8, "fallback padding")
                C_END = runtime_integer_guard(3 + pad_bits, "fallback END")
                C_stream = runtime_integer_guard(C_CAUS + C_END, "fallback total")
                
                tokens_B.append(('CONST', (byte_val,), 1, {'C_stream': C_stream}, pos))
                pos += 1
                total_covered += 1
                struct_counts['CONST'] += 1
        
        # INVARIANT C.3: Coverage exactness verification
        if total_covered != L:
            return False, 0, [], struct_counts  # B_COMPLETE = False
        
        # MATHEMATICAL REQUIREMENT: Convert tiling cover to causal seed K
        from teleport.clf_cover_to_seed import U_B_cover_to_seed, estimate_seed_cost
        
        try:
            # Derive causal seed from structural tiling
            K_seed = U_B_cover_to_seed(tokens_B, L)
            
            # Estimate cost of seed-based encoding CBD(K_seed, L)
            seed_cost = estimate_seed_cost(K_seed, L)
            
            # Use seed cost if better than structural tiling
            structural_cost = integer_sum(token[3]['C_stream'] for token in tokens_B)
            
            if seed_cost < structural_cost:
                # Use causal seed encoding
                C_B_stream = seed_cost
                # Replace tokens with single CBD seed token
                tokens_B = [('CBD_SEED', K_seed, L, {
                    'C_stream': seed_cost,
                    'seed_origin': 'DERIVED_FROM_B',
                    'structural_basis': struct_counts,
                    'mathematical_reduction': True
                }, 0)]
            else:
                # Use structural tiling
                C_B_stream = structural_cost
        
        except Exception as e:
            # Seed derivation failed - use structural tiling
            C_B_stream = integer_sum(token[3]['C_stream'] for token in tokens_B)
        
        return True, C_B_stream, tokens_B, struct_counts
        
    except Exception:
        # Any error in B construction -> B_COMPLETE = False
        return False, 0, [], struct_counts

def _detect_const_run(S: bytes, pos: int) -> int:
    """Detect length of homogeneous byte run starting at pos"""
    if pos >= len(S):
        return 0
    
    target = S[pos]
    length = 1
    while pos + length < len(S) and S[pos + length] == target:
        length += 1
    return length

def _detect_step_run(S: bytes, pos: int) -> int:
    """Detect length of arithmetic progression starting at pos"""
    if pos + 2 >= len(S):
        return 0
    
    a0 = S[pos]
    d = (S[pos + 1] - S[pos]) % 256
    length = 2
    
    while pos + length < len(S):
        expected = (a0 + d * length) % 256
        if S[pos + length] != expected:
            break
        length += 1
    
    return length if length >= 3 else 0

def _detect_match_run(S: bytes, pos: int) -> Optional[Tuple[int, int]]:
    """Detect MATCH pattern. Returns (length, distance) or None"""
    if pos < 1:
        return None
    
    # Try different distances
    for D in range(1, min(pos + 1, 256)):  # Reasonable distance limit
        if pos - D < 0:
            continue
            
        # Check how long the match extends
        match_len = 0
        while (pos + match_len < len(S) and 
               pos - D + match_len >= 0 and
               S[pos + match_len] == S[pos - D + match_len]):
            match_len += 1
        
        if match_len >= 3:  # Minimum viable MATCH
            return match_len, D
    
    return None

def _is_match_legal(pos: int, D: int, length: int, total_len: int) -> bool:
    """
    Check MATCH window legality per INVARIANT C.7.
    MATCH(D, L) copies L bytes from position (pos - D)
    """
    # Source window bounds
    src_start = pos - D
    src_end = src_start + length
    
    # Target window bounds  
    tgt_start = pos
    tgt_end = pos + length
    
    # Legality checks
    if src_start < 0:  # Source before start
        return False
    if src_end > pos:  # Source overlaps target start (forward copy only)
        return False
    if tgt_end > total_len:  # Target beyond end
        return False
    if D < 1:  # Invalid distance
        return False
    if length < 3:  # Below minimum viable length
        return False
    
    return True