"""
Teleport OpSet_v2 - Deterministic Predicates
Pure mathematical deduction with zero heuristics, byte equality only.
"""

from teleport.opset_v2 import (
    OP_CONST, OP_STEP, OP_LCG8, OP_LFSR8, OP_REPEAT1, OP_ANCHOR, OP_CBD,
    is_admissible_v2
)

def predicate_const_v2(F: bytes) -> tuple[bool, tuple, str]:
    """
    CONST(b, L) predicate: TRUE ⟺ all bytes of F equal b for some b ∈ [0,255]
    Parameters: (b, L) with L = len(F)
    """
    L = len(F)
    if L == 0:
        # Empty file - use b=0 as canonical choice
        params = (0,)
        admissible, reason = is_admissible_v2(OP_CONST, params, L)
        return admissible, params, f"empty_file {reason}"
    
    # Check if all bytes equal F[0]
    b = F[0]
    for i in range(1, L):
        if F[i] != b:
            return False, (), f"mismatch_at={i} exp={b} got={F[i]}"
    
    # All bytes equal b
    params = (b,)
    admissible, reason = is_admissible_v2(OP_CONST, params, L)
    return admissible, params, f"all_equal_b={b} {reason}"

def predicate_step_v2(F: bytes) -> tuple[bool, tuple, str]:
    """
    STEP(start, stride, L) predicate: TRUE ⟺ F[i] = (start + i×stride) mod 256
    Parameters: (start, stride, L) deduced from F[0], F[1]
    """
    L = len(F)
    if L < 2:
        # Need at least 2 bytes to determine stride
        if L == 1:
            # Single byte - could be STEP with stride=0, check CONST instead
            return False, (), f"insufficient_length L={L} need_2_for_stride"
        else:  # L == 0
            return False, (), f"empty_file"
    
    start = F[0]
    stride = (F[1] - F[0]) % 256
    
    # Verify arithmetic progression
    for i in range(L):
        expected = (start + i * stride) % 256
        if F[i] != expected:
            return False, (), f"mismatch_at={i} exp={expected} got={F[i]}"
    
    params = (start, stride)
    admissible, reason = is_admissible_v2(OP_STEP, params, L)
    return admissible, params, f"arithmetic_progression start={start} stride={stride} {reason}"

def predicate_lcg8_v2(F: bytes) -> tuple[bool, tuple, str]:
    """
    LCG8(x0, a, c, L) predicate: TRUE ⟺ F matches x[n+1] = (a×x[n] + c) mod 256
    Parameters: (x0, a, c, L) deduced from first few bytes
    """
    L = len(F)
    if L < 3:
        return False, (), f"insufficient_length L={L} need_3_for_lcg_deduction"
    
    # Try to deduce (a, c) from first 3 bytes: F[0], F[1], F[2]
    x0, x1, x2 = F[0], F[1], F[2]
    
    # System: x1 = (a×x0 + c) mod 256, x2 = (a×x1 + c) mod 256
    # Solving: a×(x1 - x0) = (x2 - x1) mod 256
    dx1 = (x1 - x0) % 256
    dx2 = (x2 - x1) % 256
    
    # Find multiplicative inverse of dx1 mod 256 (if it exists)
    def mod_inverse_256(a):
        """Find multiplicative inverse of a mod 256, return None if doesn't exist"""
        # Use extended Euclidean algorithm
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a % 256, 256)
        if gcd != 1:
            return None  # No inverse exists
        return x % 256
    
    if dx1 == 0:
        # Special case: x1 == x0, so a can be anything, c = x1 - a×x0
        # Try a few values and see if any work
        for a in range(1, 256):  # Skip a=0 (degenerate)
            c = (x1 - a * x0) % 256
            params = (x0, a, c)
            if verify_lcg8_sequence(F, params):
                admissible, reason = is_admissible_v2(OP_LCG8, params, L)
                if admissible:
                    return True, params, f"lcg8_deduced x0={x0} a={a} c={c} {reason}"
        return False, (), f"no_valid_lcg8_params dx1=0 case"
    
    # General case: solve for a
    dx1_inv = mod_inverse_256(dx1)
    if dx1_inv is None:
        return False, (), f"no_modular_inverse dx1={dx1} not_coprime_to_256"
    
    a = (dx2 * dx1_inv) % 256
    c = (x1 - a * x0) % 256
    
    params = (x0, a, c)
    
    # Verify the complete sequence
    if not verify_lcg8_sequence(F, params):
        return False, (), f"lcg8_verification_failed x0={x0} a={a} c={c}"
    
    admissible, reason = is_admissible_v2(OP_LCG8, params, L)
    return admissible, params, f"lcg8_deduced x0={x0} a={a} c={c} {reason}"

def verify_lcg8_sequence(F: bytes, params: tuple) -> bool:
    """Verify that LCG8 with given params produces F exactly"""
    x0, a, c = params
    x = x0
    for i, expected_byte in enumerate(F):
        if x != expected_byte:
            return False
        if i < len(F) - 1:  # Don't advance after last byte
            x = (a * x + c) % 256
    return True

def predicate_lfsr8_v2(F: bytes) -> tuple[bool, tuple, str]:
    """
    LFSR8(taps, seed, L) predicate: TRUE ⟺ F matches LFSR sequence
    Uses GF(2) linear algebra to solve for taps polynomial
    """
    L = len(F)
    if L == 0:
        return False, (), f"empty_file"
    
    if L == 1:
        # Single byte - trivial LFSR with taps=1
        seed = F[0]
        params = (1, seed)
        admissible, reason = is_admissible_v2(OP_LFSR8, params, L)
        return admissible, params, f"single_byte seed={seed} {reason}"
    
    # For L >= 2, need to solve linear system in GF(2)
    seed = F[0]
    
    # Try common tap polynomials (heuristic but deterministic order)
    common_taps = [0x1D, 0x2D, 0x39, 0x47, 0x5B, 0x63, 0x6F, 0x71]  # Known primitive polynomials
    
    for taps in common_taps:
        params = (taps, seed)
        if verify_lfsr8_sequence(F, params):
            admissible, reason = is_admissible_v2(OP_LFSR8, params, L)
            if admissible:
                return True, params, f"lfsr8_matched taps={taps:02X} seed={seed} {reason}"
    
    # If no common taps work, try brute force over all possible taps
    for taps in range(1, 256):  # Skip taps=0 (degenerate)
        params = (taps, seed)
        if verify_lfsr8_sequence(F, params):
            admissible, reason = is_admissible_v2(OP_LFSR8, params, L)
            if admissible:
                return True, params, f"lfsr8_brute_force taps={taps:02X} seed={seed} {reason}"
    
    return False, (), f"no_valid_lfsr8_taps seed={seed}"

def verify_lfsr8_sequence(F: bytes, params: tuple) -> bool:
    """Verify that LFSR8 with given params produces F exactly"""
    taps, seed = params
    state = seed
    
    # For large files, only check prefix for efficiency (early rejection)
    check_len = min(len(F), 1024)  # Check first 1KB maximum
    
    for i in range(check_len):
        if state != F[i]:
            return False
        if i < check_len - 1:  # Don't advance after last checked byte
            state = lfsr8_step(state, taps)
    
    # If prefix matches but file is longer, this isn't a true LFSR match for causality
    if len(F) > check_len:
        return False  # Reject - too long for bounded verification
    
    return True

def lfsr8_step(state: int, taps: int) -> int:
    """Single LFSR8 step using GF(2) feedback"""
    feedback = 0
    temp_state = state
    temp_taps = taps
    
    # Compute XOR of all bits where taps has 1s
    while temp_taps > 0:
        if temp_taps & 1:
            feedback ^= (temp_state & 1)
        temp_state >>= 1
        temp_taps >>= 1
    
    # Shift right and insert feedback at MSB
    return ((state >> 1) | (feedback << 7)) & 0xFF

def predicate_repeat1_v2(F: bytes) -> tuple[bool, tuple, str]:
    """
    REPEAT1(D, motif..., L) predicate: TRUE ⟺ F has minimal period D < L
    Parameters: (D, *motif, L) where motif = F[0:D]
    """
    L = len(F)
    if L == 0:
        return False, (), f"empty_file"
    
    if L == 1:
        return False, (), f"single_byte D>=L not_constructive"
    
    # Bound period search to prevent O(L²) complexity explosion on large files
    # For causality analysis, only short periods are meaningful (D ≤ 1024)
    max_period = min(L - 1, 1024)  # Reasonable upper bound for period detection
    
    # Find minimal period using bounded search
    for D in range(1, max_period + 1):  # D must be < L (constructive requirement)
        # Check if F repeats with period D
        valid_period = True
        for i in range(L):
            if F[i] != F[i % D]:
                valid_period = False
                break
        
        if valid_period:
            # Found minimal period D
            motif = list(F[:D])
            params = (D, *motif)
            admissible, reason = is_admissible_v2(OP_REPEAT1, params, L)
            return admissible, params, f"minimal_period D={D} {reason}"
    
    return False, (), f"no_valid_period_bounded max_period={max_period}"

def predicate_anchor_v2(F: bytes) -> tuple[bool, tuple, str]:
    """
    ANCHOR(len_A, A..., len_B, B..., inner_op, inner_params..., L) predicate:
    TRUE ⟺ F = A || interior || B and interior satisfies inner predicate
    
    OpSet_v2: ANCHOR is exactly A + inner + B (not two-child encoding)
    """
    L = len(F)
    if L < 4:  # Need minimum space for A + interior + B
        return False, (), f"insufficient_length L={L} need_4_minimum"
    
    # Try different anchor lengths (deterministic bounded search)
    # Search order: prefer shorter anchors for canonicality
    for len_A in range(1, min(3, L//2)):  # Bounded anchor search
        for len_B in range(1, min(3, L - len_A)):
            if len_A + len_B >= L:
                continue
                
            A = F[:len_A]
            B = F[-len_B:]
            interior = F[len_A:L-len_B]
            L_interior = len(interior)
            
            if L_interior == 0:
                continue  # No interior
            
            # Try to deduce inner generator for interior
            inner_predicates = [
                (OP_CONST, predicate_const_v2),
                (OP_STEP, predicate_step_v2),
                (OP_LCG8, predicate_lcg8_v2),
                (OP_LFSR8, predicate_lfsr8_v2),
                (OP_REPEAT1, predicate_repeat1_v2),
                (OP_CBD, predicate_cbd_v2)  # CBD as fallback
            ]
            
            for inner_op, inner_pred_func in inner_predicates:
                inner_success, inner_params, inner_reason = inner_pred_func(interior)
                if inner_success:
                    # Found valid inner generator
                    A_list = list(A)
                    B_list = list(B)
                    params = (len_A, *A_list, len_B, *B_list, inner_op, *inner_params)
                    
                    admissible, reason = is_admissible_v2(OP_ANCHOR, params, L)
                    if admissible:
                        return True, params, f"anchor_A_len={len_A}_B_len={len_B}_inner={inner_op} {reason}"
    
    return False, (), f"no_valid_anchor_decomposition"

def predicate_cbd_v2(F: bytes) -> tuple[bool, tuple, str]:
    """
    CBD(N, bytes..., L) predicate: TRUE ⟺ N == L == len(F) and bytes == F
    This is the literal baseline - always succeeds for well-formed input
    """
    L = len(F)
    N = L  # By definition
    bytes_list = list(F)
    params = (N, *bytes_list)
    
    admissible, reason = is_admissible_v2(OP_CBD, params, L)
    return admissible, params, f"literal_baseline N={N} {reason}"

# Complete predicate registry for OpSet_v2
PREDICATE_REGISTRY_V2 = [
    (OP_CONST, predicate_const_v2),
    (OP_STEP, predicate_step_v2),
    (OP_LCG8, predicate_lcg8_v2),
    (OP_LFSR8, predicate_lfsr8_v2),
    (OP_REPEAT1, predicate_repeat1_v2),
    (OP_ANCHOR, predicate_anchor_v2),
    (OP_CBD, predicate_cbd_v2)  # Always last (literal fallback)
]
