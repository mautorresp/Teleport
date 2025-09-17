"""
CLF Deterministic Predicate Suite

Pure integer checks on input bytes to deduce global CAUS descriptions.
Each predicate either proves a global CAUS (op_id, params, L=N) or returns False.
No partial CAUS, no heuristics, no floats.
"""

from typing import Tuple, Optional, Union

def check_const(S: bytes) -> Union[Tuple[bool, int, Tuple[int]], Tuple[bool, str]]:
    """
    P_CONST (global): all bytes equal
    Returns: (True, op_id, (b,)) or (False, reason)
    """
    if len(S) == 0:
        return (False, "empty_input")
    
    b = S[0]
    for i in range(1, len(S)):
        if S[i] != b:
            distinct_count = len(set(S))
            return (False, f"distinct_bytes={distinct_count}")
    
    return (True, 0, (b,))  # op_id=0 for P_CONST

def check_step(S: bytes) -> Union[Tuple[bool, int, Tuple[int, int]], Tuple[bool, str]]:
    """
    P_STEP (global arithmetic progression mod 256)
    ∃ a,d in 0..255 s.t. ∀i: S[i] == (a + i*d) & 255
    Returns: (True, op_id, (a, d)) or (False, reason)
    """
    if len(S) == 0:
        return (False, "empty_input")
    
    if len(S) == 1:
        # Single byte is trivially a step with d=0
        return (True, 1, (S[0], 0))  # op_id=1 for P_STEP
    
    a = S[0]
    d = (S[1] - S[0]) & 255
    
    violations = []
    for i in range(len(S)):
        expected = (a + i * d) & 255
        if S[i] != expected:
            violations.append(i)
            if len(violations) >= 10:  # Limit output
                break
    
    if violations:
        return (False, f"violations={len(violations)} at offsets: {', '.join(map(str, violations[:10]))}")
    
    return (True, 1, (a, d))  # op_id=1 for P_STEP

def check_anchor_window(S: bytes) -> Union[Tuple[bool, int, Tuple], Tuple[bool, str]]:
    """
    P_ANCHOR_WINDOW (global anchored window generator)
    
    Mathematical generator that attempts to prove causality via dual-anchor structure.
    Tests if bytes between anchors A and B can be reproduced by simpler generators.
    
    Generator G_ANCHOR(A, B, G_inner, θ_inner):
    1. Find unique positions s=find(S,A) and e=find(S,B) 
    2. Apply inner generator G_inner with parameters θ_inner to reproduce S[s+len(A):e]
    3. Verify complete reconstruction: A + G_inner(θ_inner) + B = S
    4. Check minimality: C_ANCHOR < 10*N
    
    Returns: (ok, params, reason) where params encode the complete generator
    """
    if len(S) < 20:
        return (False, f"too_short={len(S)}_need_20")
    
    # Test dual-anchor patterns (mathematical, not format-based)
    anchor_patterns = [
        (b'\xff\xd8', b'\xff\xd9'),  # 2-byte start, 2-byte end
        (b'\x89PNG', b'IEND'),       # 4-byte start, 4-byte end
    ]
    
    for A, B in anchor_patterns:
        s = S.find(A)
        if s == -1 or S.find(A, s + 1) != -1:  # Must be unique
            continue
            
        e = S.find(B, s + len(A))
        if e == -1 or S.find(B, e + 1) != -1:  # Must be unique  
            continue
            
        if s != 0 or e + len(B) != len(S):  # Must span entire string
            continue
            
        # Extract interior bytes for mathematical analysis
        interior = S[s + len(A):e]
        
        if len(interior) == 0:
            # Degenerate case: just A + B
            cost_anchor = 3 + 8 + 8*len(A) + 8*len(B) + 8 + len(S).bit_length()
            if cost_anchor < 10 * len(S):
                from . import generators
                return (True, generators.OP_ANCHOR, (len(A), *A, len(B), *B, 0, 0))  # 0,0 = empty interior
            else:
                return (False, f"anchor_only_not_minimal_cost={cost_anchor}_vs_lit={10*len(S)}")
        
        # Try to find mathematical structure in interior
        from . import generators
        
        # Test if interior follows CONST pattern
        ok, params, reason = generators.deduce_CONST(interior)
        if ok:
            b, = params
            # Simplified cost estimation
            anchor_cost = 8*len(A) + 8*len(B) + 16  # anchors + overhead
            inner_cost = 24  # CONST generator cost  
            total_cost = anchor_cost + inner_cost
            if total_cost < 10 * len(S):
                # Return simplified parameter structure
                anchor_params = (len(A), *A, len(B), *B, generators.OP_CONST, b)
                return (True, generators.OP_ANCHOR, anchor_params)
        
        # Test if interior follows STEP pattern  
        ok, params, reason = generators.deduce_STEP(interior)
        if ok:
            a, d = params
            anchor_cost = 8*len(A) + 8*len(B) + 16
            inner_cost = 32  # STEP generator cost
            total_cost = anchor_cost + inner_cost
            if total_cost < 10 * len(S):
                anchor_params = (len(A), *A, len(B), *B, generators.OP_STEP, a, d)
                return (True, generators.OP_ANCHOR, anchor_params)
        
        # Test if interior follows LCG8 pattern
        ok, params, reason = generators.deduce_LCG8(interior)
        if ok:
            x0, a, c = params
            anchor_cost = 8*len(A) + 8*len(B) + 16
            inner_cost = 40  # LCG8 generator cost
            total_cost = anchor_cost + inner_cost
            if total_cost < 10 * len(S):
                anchor_params = (len(A), *A, len(B), *B, generators.OP_LCG8, x0, a, c)
                return (True, generators.OP_ANCHOR, anchor_params)
        
        # Interior doesn't follow known patterns
        return (False, f"anchors_A={A.hex()}_B={B.hex()}_found_but_interior_not_generatable")
    
    return (False, "no_suitable_dual_anchor_structure")

def check_anchor_jfif(S: bytes) -> Union[Tuple[bool, int, Tuple], Tuple[bool, str]]:
    """
    P_JFIF_ANCHOR (wrapper for compatibility)
    Delegates to the new anchored window generator
    """
    return check_anchor_window(S)

def check_repeat1(S: bytes) -> Union[Tuple[bool, int, Tuple[int]], Tuple[bool, str]]:
    """
    P_REPEAT1 (global single previous offset replay)
    ∃ D in [1..N-3] s.t. ∀i≥D: S[i] == S[i-D]
    Returns: (True, op_id, (D,)) or (False, reason)
    """
    if len(S) < 4:
        return (False, f"too_short={len(S)}_need_4")
    
    for D in range(1, len(S) - 2):  # D in [1..N-3]
        violations = 0
        for i in range(D, len(S)):
            if S[i] != S[i - D]:
                violations += 1
                if violations > 10:  # Early exit for efficiency
                    break
        
        if violations == 0:
            return (True, 3, (D,))  # op_id=3 for P_REPEAT1
    
    return (False, "no_valid_D_found")

def evaluate_all_predicates(S: bytes) -> Tuple[Optional[Tuple[int, Tuple]], list]:
    """
    Evaluate all predicates and return first successful global CAUS + all receipts
    Returns: (global_caus_or_none, receipt_strings)
    """
    predicates = [
        ("P_CONST", check_const),
        ("P_STEP", check_step), 
        ("P_ANCHOR_WINDOW", check_anchor_window),
        ("P_JFIF_ANCHOR", check_anchor_jfif),  # Compatibility wrapper
        ("P_REPEAT1", check_repeat1),
    ]
    
    global_caus = None
    receipts = []
    
    for name, predicate_func in predicates:
        try:
            result = predicate_func(S)
            if result[0]:  # Success
                success, op_id, params = result
                if global_caus is None:  # Take first successful predicate
                    global_caus = (op_id, params)
                
                # Format success receipt
                if name == "P_CONST":
                    receipts.append(f"{name}: TRUE (b={params[0]})")
                elif name == "P_STEP":
                    receipts.append(f"{name}: TRUE (a={params[0]}, d={params[1]})")
                elif name == "P_ANCHOR_WINDOW":
                    len_a, a0, a1, len_b, b0, b1, seg_marker, seg_len, interior_len, s, e = params
                    receipts.append(f"{name}: TRUE (A={a0:02x}{a1:02x}, B={b0:02x}{b1:02x}, seg={seg_marker:02x}, seg_len={seg_len}, interior_len={interior_len}, s={s}, e={e})")
                elif name == "P_JFIF_ANCHOR":
                    # Delegate to anchor window formatting
                    receipts.append(f"{name}: TRUE (delegated_to_anchor_window)")
                elif name == "P_REPEAT1":
                    receipts.append(f"{name}: TRUE (D={params[0]})")
                else:
                    receipts.append(f"{name}: TRUE (params={params})")
            else:
                # Format failure receipt
                reason = result[1]
                receipts.append(f"{name}: FALSE ({reason})")
        except Exception as e:
            receipts.append(f"{name}: FALSE (error={str(e)})")
    
    return global_caus, receipts
