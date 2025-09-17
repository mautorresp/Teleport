"""
CLF Causal Deduction - Pure Mathematical Predicates

This module implements deterministic causal deduction (not heuristic detection).
Each generator has a mathematical predicate that must be TRUE for all positions
it claims to generate. No statistics, no guessing - only proven mathematical facts.
"""

from typing import Optional, Tuple, List


# Opcode definitions for CLF 
OP_LIT = 0
OP_MATCH = 1  
OP_CAUS = 2
OP_END = 3

# CAUS operation IDs
CAUS_OP_CONST = 0  # Constant byte pattern: ∀i: S[i] == b
CAUS_OP_STEP = 1   # Arithmetic progression: ∀i: S[i] == (a + i*d) mod 256


def deduct_caus_global(S: bytes, print_receipts: bool = False) -> Tuple[Optional[Tuple[int, Tuple]], List[str]]:
    """
    Deduction-first global CAUS evaluation.
    
    Evaluates all mathematical predicates and returns first successful global CAUS
    description along with detailed receipts for all predicates.
    
    Args:
        S: Complete input bytes (full file)
        print_receipts: Whether to print receipts to console
        
    Returns:
        (global_caus_or_none, receipt_strings)
        where global_caus = (op_id, params) if any predicate TRUE, else None
    """
    from .predicates import evaluate_all_predicates
    
    global_caus, receipts = evaluate_all_predicates(S)
    
    if print_receipts:
        print(f"bytes= {len(S)}")
        for receipt in receipts:
            print(receipt)
    
    return global_caus, receipts


def receipts_caus(S: bytes) -> list[str]:
    """
    Generate deterministic predicate receipts.
    
    Returns TRUE/FALSE for each predicate with mathematical proof.
    Uses the same predicate suite as deduct_caus_global.
    """
    from .predicates import evaluate_all_predicates
    
    global_caus, receipts = evaluate_all_predicates(S)
    return receipts


# Legacy function for compatibility
def deduct_caus(S: bytes) -> Optional[Tuple[int, Tuple[int, ...]]]:
    """Legacy wrapper - use deduct_caus_global for new code"""
    global_caus, _ = deduct_caus_global(S)
    return global_caus


def _is_constant_pattern(S: bytes) -> bool:
    """
    Check if all bytes are identical (CONST pattern predicate).
    
    Predicate: ∀i ∈ [0, len(S)): S[i] == S[0]
    """
    if len(S) <= 1:
        return True
        
    first_byte = S[0]
    for i in range(1, len(S)):
        if S[i] != first_byte:
            return False
    return True


def _deduce_step_pattern(S: bytes) -> Optional[Tuple[int, int]]:
    """
    Check if bytes follow arithmetic progression (STEP pattern predicate).
    
    Predicate: ∀i ∈ [0, len(S)): S[i] == (a + i*d) mod 256
    
    Returns (a, d) if pattern holds, None otherwise.
    """
    if len(S) <= 1:
        # Single byte can be considered step with d=0, but CONST is better
        return None
        
    if len(S) == 2:
        # Two bytes define unique arithmetic progression
        a = S[0]
        d = (S[1] - S[0]) % 256
        return (a, d)
    
    # For 3+ bytes, verify the pattern holds exactly
    a = S[0]
    d = (S[1] - S[0]) % 256
    
    # Verify predicate: ∀i: S[i] == (a + i*d) mod 256
    for i in range(len(S)):
        expected = (a + i * d) % 256
        if S[i] != expected:
            return None
            
    return (a, d)


def compute_caus_cost(op_id: int, params: Tuple[int, ...], N: int) -> int:
    """
    Compute exact CAUS cost in bits according to CLF formula.
    
    For basic predicates (P_CONST, P_STEP, P_REPEAT1):
    C_CAUS = 3 + 8*leb(op_id) + 8*∑leb(param_i) + 8*leb(N)
    
    For P_ANCHOR_WINDOW (op_id=2), includes anchor bytes and interior data:
    C_CAUS = 3 + 8*leb(2) + 8*leb(len_A) + 8*len_A + 8*leb(len_B) + 8*len_B + 8*leb(k) + 8*leb(interior_len) + 8*interior_len + 8*leb(N)
    """
    from teleport.leb_io import leb128_emit_single
    
    # 3 bits for CAUS tag (111 in binary)
    cost = 3
    
    # 8 * leb(op_id) 
    cost += 8 * len(leb128_emit_single(op_id))
    
    if op_id == 2:  # P_ANCHOR_WINDOW - special encoding
        # params: (len_A, a0, a1, len_B, b0, b1, seg_marker, seg_len, interior_len, s, e)
        len_a, a0, a1, len_b, b0, b1, seg_marker, seg_len, interior_len, s, e = params
        
        # Encode anchor A: length + bytes
        cost += 8 * len(leb128_emit_single(len_a))
        cost += 8 * len_a  # The actual anchor bytes
        
        # Encode anchor B: length + bytes  
        cost += 8 * len(leb128_emit_single(len_b))
        cost += 8 * len_b  # The actual anchor bytes
        
        # Encode segment parameters
        cost += 8 * len(leb128_emit_single(seg_marker))
        cost += 8 * len(leb128_emit_single(seg_len))
        
        # Encode interior data: length + bytes
        cost += 8 * len(leb128_emit_single(interior_len))
        cost += 8 * interior_len  # The interior entropy bytes
        
        # Encode N (total length)
        cost += 8 * len(leb128_emit_single(N))
        
    else:  # Standard predicates: P_CONST, P_STEP, P_REPEAT1
        # 8 * ∑leb(param_i)
        for param in params:
            cost += 8 * len(leb128_emit_single(param))
            
        # 8 * leb(N)
        cost += 8 * len(leb128_emit_single(N))
    
    return cost


def compute_caus_seed_bytes(op_id: int, params: Tuple[int, ...], N: int) -> int:
    """
    Compute expected seed bytes for CAUS token.
    
    seed_bytes = ceil(C_CAUS/8) + 3  (header + exact CAUS bitstream)
    """
    c_caus = compute_caus_cost(op_id, params, N)
    return (c_caus + 7) // 8 + 3
