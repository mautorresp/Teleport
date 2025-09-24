# teleport/clf_encoder.py
"""
CLF Universal Calculator: Single Decision Equation Implementation
C(S) = H(L) + min(C_CBD(S), C_STRUCT(S))
No modes, no aliasing - pure mathematical computation.
"""

import hashlib
from typing import Tuple, List
from teleport.clf_canonical_math import (
    C_DECISION_EQUATION, H_HEADER, GATE_ADMISSIBLE,
    ASSERT_INTEGER_ONLY, ASSERT_COVERAGE_COMPLETE
)
from teleport.clf_builders import build_A, build_B, decide_min, CLFToken
from teleport.clf_receipts import generate_mandatory_receipt


# ============================================================================
# UNIVERSAL CLF CALCULATOR (SINGLE ENTRY POINT)
# ============================================================================

def encode_CLF(S: bytes, emit_receipts: bool = True) -> Tuple[List[CLFToken], str]:
    """
    Universal CLF calculator implementing the canonical decision equation:
    C(S) = H(L) + min(C_CBD(S), C_STRUCT(S))
    
    No modes, no shortcuts - both constructions computed independently.
    
    Args:
        S: Input byte sequence
        emit_receipts: Whether to generate mandatory receipts
    
    Returns:
        (chosen_tokens, receipt_string)
    """
    
    # Input validation
    if not isinstance(S, bytes):
        raise TypeError(f"Input must be bytes, got {type(S)}")
    
    L = len(S)
    
    # Handle empty input with complete receipt
    if L == 0:
        H = H_HEADER(0)  # H(0) = 16 + 8*leb_len(0) = 16 + 8*1 = 24 bits
        empty_receipt = f"""IDENTITY:
  L = 0 bytes
  RAW_BITS = 8·L = 0 bits
  SHA256_IN  = {hashlib.sha256(S).hexdigest().upper()}
  SHA256_OUT = {hashlib.sha256(S).hexdigest().upper()}
  EQUALITY   = True

HEADER:
  leb_len(8·L) = leb_len(0) = 1
  H(L) = 16 + 8·1 = 24 bits

A (CBD whole-range):
  tokensA = 0
  C_A_stream = 0
  C_A_total  = H + C_A_stream = 24 + 0 = 24

B (structural tiling):
  tokensB = 0
  C_B_stream = 0
  C_B_total  = H + C_B_stream = 24 + 0 = 24

DECISION:
  min(C_A_total, C_B_total) = min(24, 24) = 24
  C(S) = 24 vs 8·L = 0
  24 ≥ 0 → OPEN (expansion)
  argmin(A,B) = TIE (both 24)

STATE = OPEN"""
        return ([], empty_receipt)
    
    # Compute header cost
    H = H_HEADER(L)
    ASSERT_INTEGER_ONLY(H)
    
    # ========================================================================
    # INDEPENDENT CONSTRUCTION COMPUTATION (NO ALIASING)
    # ========================================================================
    
    # Construction A: Whole-range CBD
    tokens_A, info_A = build_A(S)
    ASSERT_COVERAGE_COMPLETE(tokens_A, L)
    
    # Construction B: Structural tiling  
    tokens_B, info_B = build_B(S)
    ASSERT_COVERAGE_COMPLETE(tokens_B, L)
    
    # Verify both constructions computed independently
    if info_A is info_B:
        raise ValueError("Construction aliasing detected: A and B share same info object")
    
    # ========================================================================
    # MINIMAL DECISION (CANONICAL EQUATION)
    # ========================================================================
    
    chosen_label, chosen_tokens, chosen_info, C_A_total, C_B_total = decide_min(
        tokens_A, info_A, tokens_B, info_B, H
    )
    
    # Verify integer arithmetic
    ASSERT_INTEGER_ONLY(C_A_total)
    ASSERT_INTEGER_ONLY(C_B_total)
    
    # ========================================================================
    # ADMISSIBILITY GATE & STATE DECISION
    # ========================================================================
    
    chosen_total_cost = C_A_total if chosen_label == "CBD" else C_B_total
    is_admissible = GATE_ADMISSIBLE(chosen_total_cost, L)
    
    # State decision: EMIT if admissible, OPEN otherwise
    emit_decision = is_admissible
    
    # ========================================================================
    # MANDATORY RECEIPTS
    # ========================================================================
    
    receipt_string = ""
    if emit_receipts:
        try:
            receipt_string = generate_mandatory_receipt(
                S=S,
                chosen_label=chosen_label,
                chosen_tokens=chosen_tokens,
                chosen_info=chosen_info,
                C_A_total=C_A_total,
                C_B_total=C_B_total,
                info_A=info_A,
                info_B=info_B,
                H=H,
                emit_decision=emit_decision
            )
        except Exception as e:
            # Receipt generation failure is a hard error
            raise ValueError(f"Mandatory receipt generation failed: {e}")
    
    # ========================================================================  
    # RETURN CHOSEN CONSTRUCTION
    # ========================================================================
    
    if emit_decision:
        return (chosen_tokens, receipt_string)
    else:
        # OPEN state: return empty token list but with full receipt
        return ([], receipt_string)


# ============================================================================
# CONVENIENCE FUNCTIONS (DELEGATING TO MAIN CALCULATOR)
# ============================================================================

def encode_minimal(S: bytes) -> List[CLFToken]:
    """
    Convenience function returning only tokens (for backward compatibility)
    Delegates to main calculator
    """
    tokens, _ = encode_CLF(S, emit_receipts=False)
    return tokens


def encode_with_receipts(S: bytes) -> str:
    """
    Convenience function returning only receipts (for testing/verification)
    """
    _, receipt = encode_CLF(S, emit_receipts=True)
    return receipt


def verify_CLF_determinism(S: bytes, runs: int = 3) -> bool:
    """
    Verify deterministic behavior across multiple runs
    All runs must produce identical tokens and receipts
    """
    results = []
    
    for _ in range(runs):
        tokens, receipt = encode_CLF(S, emit_receipts=True)
        results.append((tokens, receipt))
    
    # Check all results are identical
    first_tokens, first_receipt = results[0]
    
    for tokens, receipt in results[1:]:
        # Token comparison (deep equality)
        if len(tokens) != len(first_tokens):
            return False
        
        for t1, t2 in zip(tokens, first_tokens):
            if (t1.type != t2.type or 
                t1.length != t2.length or
                t1.position != t2.position):
                return False
        
        # Receipt comparison (exact string match)
        if receipt != first_receipt:
            return False
    
    return True


# ============================================================================
# MATHEMATICAL VALIDATION FUNCTIONS
# ============================================================================

def validate_construction_independence(S: bytes) -> dict:
    """
    Validate that constructions A and B are computed independently
    Returns diagnostic information
    """
    H = H_HEADER(len(S))
    
    # Build constructions independently
    tokens_A, info_A = build_A(S)
    tokens_B, info_B = build_B(S)
    
    # Compute totals
    C_A_total = H + info_A["C_stream"]
    C_B_total = H + info_B["C_stream"]
    
    return {
        "A_tokens": len(tokens_A),
        "B_tokens": len(tokens_B),
        "A_total": C_A_total,
        "B_total": C_B_total,
        "independent": info_A is not info_B,
        "both_computed": info_A["C_stream"] > 0 and info_B["C_stream"] > 0,
        "decision": "CBD" if C_A_total <= C_B_total else "STRUCT"
    }


def validate_serializer_identity_all_tokens(tokens: List[CLFToken]) -> bool:
    """
    Validate serializer identity for all tokens: 8 * |seed| = C_stream
    """
    try:
        for token in tokens:
            token.validate_serializer_identity()
        return True
    except Exception:
        return False


def validate_complexity_envelope(tokens: List[CLFToken], L: int) -> bool:
    """
    Validate complexity envelope: total_operations <= α + β*L
    Using pinned constants α=32, β=1
    """
    from teleport.clf_canonical_math import COMPLEXITY_ALPHA, COMPLEXITY_BETA
    
    total_ops = len(tokens)
    max_allowed = COMPLEXITY_ALPHA + COMPLEXITY_BETA * L
    
    return total_ops <= max_allowed


# ============================================================================
# TESTING ENTRY POINTS
# ============================================================================

def test_canonical_decision_equation(S: bytes) -> dict:
    """
    Test the canonical decision equation with full diagnostics
    Returns all intermediate values for verification
    """
    L = len(S)
    H = H_HEADER(L)
    
    # Independent constructions
    tokens_A, info_A = build_A(S)
    tokens_B, info_B = build_B(S)
    
    # Costs
    C_A = info_A["C_stream"]
    C_B = info_B["C_stream"]
    C_A_total = H + C_A
    C_B_total = H + C_B
    
    # Decision
    min_cost = min(C_A_total, C_B_total)
    chosen = "CBD" if C_A_total <= C_B_total else "STRUCT"
    
    # Gate
    admissible = GATE_ADMISSIBLE(min_cost, L)
    state = "EMIT" if admissible else "OPEN"
    
    return {
        "L": L,
        "H": H,
        "C_A": C_A,
        "C_B": C_B,
        "C_A_total": C_A_total,
        "C_B_total": C_B_total,
        "min_cost": min_cost,
        "chosen": chosen,
        "admissible": admissible,
        "state": state,
        "decision_equation": f"C(S) = H(L) + min(C_CBD(S), C_STRUCT(S)) = {H} + min({C_A}, {C_B}) = {min_cost}"
    }