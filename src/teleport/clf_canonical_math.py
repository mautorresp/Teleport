# teleport/clf_canonical_math.py
"""
Single Source of Truth: CLF Mathematical Constants and Decision Equation
TOE-anchored mathematical specification - no duplicates, no alternates allowed.
"""

import hashlib
from teleport.clf_int import leb as leb_len

# ============================================================================
# CANONICAL DECISION EQUATION (PIN THIS FIRST)
# ============================================================================

def C_DECISION_EQUATION(S: bytes) -> tuple:
    """
    The one canonical equation the calculator computes for every byte string S:
    C(S) = H(L) + min(C_CBD(S), C_STRUCT(S))
    
    Returns: (total_cost, chosen_construction, C_A, C_B, H)
    """
    L = len(S)
    H = H_HEADER(L)
    C_A = C_CBD_WHOLE_RANGE(S)  # CBD whole-range construction
    C_B = C_STRUCT_TILING(S)    # Structural tiling construction
    
    # Tie rule: if C_A = C_B => choose CBD (label choice only - costs identical)
    if C_A <= C_B:
        chosen = "CBD"
        total_cost = H + C_A
    else:
        chosen = "STRUCT" 
        total_cost = H + C_B
        
    return (total_cost, chosen, C_A, C_B, H)


# ============================================================================
# HEADER COST (EXACT LEB128 BYTE COUNT)
# ============================================================================

def H_HEADER(L: int) -> int:
    """
    Header cost: H(L) = 16 + 8 * leb_len(8L) bits
    Exact LEB128 byte count for 8L value.
    """
    return 16 + 8 * leb_len(8 * L)


# ============================================================================
# CBD BIJECTION (INTEGER-ONLY ARITHMETIC)
# ============================================================================

def CBD_BIJECTION_FORWARD(S: bytes) -> int:
    """
    CBD256 seed: K = Σ(i=0 to L-1) S[i] * 256^(L-1-i)
    Pure integer arithmetic, no floating point.
    """
    L = len(S)
    K = 0
    for i in range(L):
        K += S[i] * (256 ** (L - 1 - i))
    return K


def CBD_BIJECTION_INVERSE(K: int, L: int) -> bytes:
    """
    Inverse CBD bijection: K -> S by exact div/mod
    Postcondition: K -> 0 after L operations
    """
    result = bytearray(L)
    for i in range(L):
        result[L - 1 - i] = K % 256
        K //= 256
    
    assert K == 0, f"CBD bijection postcondition violated: K={K} != 0"
    return bytes(result)


def CBD_BIJECTION_PROOF(S: bytes) -> dict:
    """
    Generate SHA256 bijection proof for receipts
    """
    K = CBD_BIJECTION_FORWARD(S)
    S_reconstructed = CBD_BIJECTION_INVERSE(K, len(S))
    
    sha_in = hashlib.sha256(S).hexdigest()
    sha_out = hashlib.sha256(S_reconstructed).hexdigest()
    equality = (S == S_reconstructed)
    
    return {
        "K": K,
        "SHA256_IN": sha_in,
        "SHA256_OUT": sha_out,
        "EQUALITY": equality,
        "BIJECTION_VALID": equality
    }


# ============================================================================
# COST COMPUTATION STUBS (TO BE IMPLEMENTED)
# ============================================================================

def C_CBD_WHOLE_RANGE(S: bytes) -> int:
    """
    Whole-range CBD construction cost (C_A)
    Single CBD token covering entire input
    """
    # TODO: Implement exact CBD cost calculation
    # For now, use placeholder based on length
    return len(S) * 8  # Placeholder - replace with actual CBD cost


def C_STRUCT_TILING(S: bytes) -> int:
    """
    Structural tiling construction cost (C_B)
    CONST->STEP->MATCH precedence with deterministic maximal tiling
    """
    # TODO: Implement structural tiling with fixed precedence
    # For now, use placeholder
    return len(S) * 8  # Placeholder - replace with actual tiling cost


# ============================================================================
# BASELINE GATES (PINNED)
# ============================================================================

# Pinned baseline - choose ONE and use consistently everywhere
BASELINE_RAW_8L = True   # If True: use 8*L as baseline
BASELINE_VIRTUAL_10L = False  # If True: use 10*L as baseline (auxiliary only)

def GATE_ADMISSIBLE(total_cost: int, L: int) -> bool:
    """
    Admissibility gate based on pinned baseline
    """
    if BASELINE_RAW_8L:
        return total_cost < 8 * L
    elif BASELINE_VIRTUAL_10L:
        return total_cost < 10 * L
    else:
        raise ValueError("No baseline pinned - must choose exactly one")


def COMPUTE_RATIOS(total_cost: int, L: int) -> dict:
    """
    Compute all ratios for receipt reporting
    """
    raw_bits = 8 * L
    virtual_bits = 10 * L
    
    return {
        "RAW_BITS": raw_bits,
        "VIRTUAL_BITS": virtual_bits,
        "RATIO_RAW": total_cost / raw_bits if raw_bits > 0 else float('inf'),
        "RATIO_10L": total_cost / virtual_bits if virtual_bits > 0 else float('inf'),
        "ADMISSIBLE": GATE_ADMISSIBLE(total_cost, L)
    }


# ============================================================================
# IMMUTABLE INVARIANTS (ENFORCEMENT FUNCTIONS)
# ============================================================================

FIXED_OPERATOR_SET = {"CONST", "STEP", "MATCH", "CBD"}

def ASSERT_INTEGER_ONLY(value):
    """Enforce integer-only arithmetic invariant"""
    if not isinstance(value, (int, int)):
        raise ValueError(f"Non-integer arithmetic detected: {value} (type: {type(value)})")


def ASSERT_SERIALIZER_IDENTITY(seed_bytes: bytes, C_stream: int):
    """Enforce serializer identity: 8 * |seed| = C_stream"""
    expected = 8 * len(seed_bytes)
    if expected != C_stream:
        raise ValueError(f"Serializer identity violated: 8*|seed|={expected} != C_stream={C_stream}")


def ASSERT_COVERAGE_COMPLETE(tokens: list, L: int):
    """Enforce coverage: Σ token lengths = L"""
    total_length = sum(token.length for token in tokens)
    if total_length != L:
        raise ValueError(f"Coverage incomplete: Σ lengths={total_length} != L={L}")


# ============================================================================
# MATHEMATICAL CONSTANTS (PINNED)
# ============================================================================

# These values are mathematically pinned - do not modify
COMPLEXITY_ALPHA = 32    # Base complexity constant  
COMPLEXITY_BETA = 1      # Linear complexity coefficient
WINDOW_SIZE = 32         # MATCH detection window
ALLOWED_DISTANCES = (1, 2, 4, 8, 16, 32, 64, 128, 256)  # Multi-distance MATCH

# Tie rule (deterministic)
TIE_RULE_CHOOSE_CBD = True  # If C_A = C_B, choose CBD construction