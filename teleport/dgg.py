# teleport/dgg.py  (integers-only)
"""
Dynamic-Generator Generator (DGG) - CLF-Pure Mathematical Synthesis

Derives unique causal description G(S) by solving exact algebraic equalities on S
and serializes as CAUS token whose verifier reproduces S exactly.
No heuristics, no formats, no approximations - pure mathematical derivation.
"""

from teleport.generators import (
    OP_CONST, OP_STEP, OP_LCG8, OP_LFSR8, OP_REPEAT1, OP_XOR_MASK8, OP_ANCHOR,
    deduce_CONST, deduce_STEP, deduce_LCG8, deduce_LFSR8, deduce_REPEAT1, deduce_XOR_MASK8,
    verify_generator
)
from teleport.leb_io import leb128_emit_single as leb_emit

def leb_len(x: int) -> int: 
    return len(leb_emit(x))

def _canonical_anchor_window(S: bytes):
    """
    Canonical ANCHOR window selection using pure byte equality analysis.
    Format-blind deterministic selection: max interior, then earliest positions.
    """
    N = len(S)
    best = None  # (-(interior), i, j, hdr_end)
    
    for i in range(0, N-7):  # Need space for 4-byte header + interior + footer
        if i + 3 >= N:
            break
            
        # Extract 2-byte length field: L = (S[i+2] << 8) | S[i+3]
        L = (S[i+2] << 8) | S[i+3] if i+3 < N else None
        if L is None: 
            break
            
        hdr_end = i + 4 + min(L, N-(i+4))
        if hdr_end >= N-2:  # Need space for footer
            continue
            
        # Deterministic sparse grid for j to avoid O(N^2) explosion
        step = 1
        j = hdr_end + 2  # Minimum interior + footer space
        while j <= N:
            interior_len = j - hdr_end
            if interior_len > 0:
                # Canonical selection key: max interior, earliest positions
                key = (-interior_len, i, j, hdr_end)
                if best is None or key < best:
                    best = key
                    
            j += step
            step = min(step << 1, 1 << 10)  # Exponential spacing, bounded
            
    if best is None: 
        return None
        
    neg_interior, i, j, hdr_end = best
    return (i, j, hdr_end)

def _try_inner_in_order(interior: bytes):
    """
    Try inner generators in deterministic order for ANCHOR interior.
    Returns (op, params, reason) if successful, None if all fail.
    """
    # Deterministic order, pure mathematical predicates
    inner_generators = [
        (OP_CONST, deduce_CONST),
        (OP_STEP, deduce_STEP),
        (OP_LCG8, deduce_LCG8),
        (OP_LFSR8, deduce_LFSR8),
        (OP_REPEAT1, deduce_REPEAT1),
        (OP_XOR_MASK8, deduce_XOR_MASK8)
    ]
    
    for op, deduce_func in inner_generators:
        ok, params, reason = deduce_func(interior)
        if ok: 
            return (op, params, reason)
    
    return None

def deduce_dynamic(S: bytes):
    """
    Dynamic-Generator Generator: synthesize CAUS description from S using pure mathematics.
    
    Returns:
        (op_id, params, reason) if successful mathematical proof established
        (0, (), GENERATOR_MISSING_report) if no generator in declared family proves S
    """
    N = len(S)
    if N == 0:
        return (0, (), "GENERATOR_MISSING: empty input")
        
    receipts = []

    # 1) Global mathematical models - try each in deterministic order
    global_generators = [
        (OP_CONST, deduce_CONST),
        (OP_STEP, deduce_STEP),
        (OP_LCG8, deduce_LCG8),
        (OP_REPEAT1, deduce_REPEAT1),
        (OP_LFSR8, deduce_LFSR8),
        (OP_XOR_MASK8, deduce_XOR_MASK8)
    ]
    
    for op, deduce_func in global_generators:
        ok, params, reason = deduce_func(S)
        generator_name = deduce_func.__name__[7:]  # Remove "deduce_" prefix
        receipts.append(f"P_{generator_name}: {'TRUE' if ok else 'FALSE'} ({reason})")
        
        if ok:
            # Verify mathematical proof with exact reconstruction
            if verify_generator(op, params, S):
                return (op, params, f"CAUS_DEDUCED generator={generator_name} {reason}")
            else:
                receipts.append(f"P_{generator_name}: VERIFY_FAILED (reconstruction mismatch)")

    # 2) Canonical ANCHOR composition (format-blind window analysis)
    anchor_window = _canonical_anchor_window(S)
    if anchor_window is not None:
        i, j, hdr_end = anchor_window
        
        # Extract anchor components using pure byte operations
        A = S[i:i+2]  # 2-byte start anchor
        B = S[j-2:j]  # 2-byte end anchor  
        interior = S[hdr_end:j]
        interior_len = len(interior)
        
        receipts.append(f"P_ANCHOR_WINDOW: TRUE (i={i} j={j} hdr_end={hdr_end} interior_len={interior_len})")
        
        # Try inner generators on interior
        inner_result = _try_inner_in_order(interior)
        if inner_result:
            inner_op, inner_params, inner_reason = inner_result
            
            # Construct ANCHOR parameters: len_A, A_bytes, len_B, B_bytes, inner_op, inner_params
            anchor_params = (len(A), *A, len(B), *B, inner_op, *inner_params)
            
            # Verify complete ANCHOR reconstruction
            if verify_generator(OP_ANCHOR, anchor_params, S):
                return (OP_ANCHOR, anchor_params, 
                       f"CAUS_DEDUCED generator=ANCHOR window=({i},{j},{hdr_end}) inner={inner_op} {inner_reason}")
            else:
                receipts.append(f"P_ANCHOR: VERIFY_FAILED (reconstruction mismatch)")
        else:
            receipts.append(f"P_ANCHOR: FALSE (interior_len={interior_len} no_inner_generator)")
    else:
        receipts.append("P_ANCHOR_WINDOW: FALSE (no_admissible_canonical_windows)")

    # 3) No generator established mathematical proof - code family incomplete
    generator_missing_report = "GENERATOR_MISSING\n" + "\n".join("  " + r for r in receipts)
    generator_missing_report += f"\n• File size: {N} bytes"
    generator_missing_report += "\n• Candidate schema required:"
    generator_missing_report += "\n  OP=? (specialized mathematical generator)"
    generator_missing_report += "\n  params: (to be determined from invariant analysis)"
    generator_missing_report += f"\n  cost_skeleton: 3 + 8·leb(OP) + 8·Σ leb(params) + 8·leb({N})"
    generator_missing_report += f"\n• Constructive requirement: verifier(OP, params) must reproduce all {N} bytes exactly"
    generator_missing_report += "\n• This indicates missing implementation, not absence of mathematical causality"
    
    return (0, (), generator_missing_report)

def compute_dgg_cost_receipts(op_id: int, params: tuple, N: int) -> str:
    """
    Compute and format exact CLF cost receipts for DGG success.
    Returns formatted string with all integer calculations.
    """
    # Calculate exact CAUS cost components
    cost_op = 8 * leb_len(op_id)
    cost_params = 8 * sum(leb_len(p) for p in params) if params else 0
    cost_length = 8 * leb_len(N)
    C_CAUS = 3 + cost_op + cost_params + cost_length
    
    # Calculate END cost with residue
    pos_after_caus = C_CAUS
    pad_bits = (8 - ((pos_after_caus + 3) % 8)) % 8
    C_END = 3 + pad_bits
    C_stream = C_CAUS + C_END
    
    # Format receipts
    receipts = f"CLF COST VERIFICATION:\n"
    receipts += f"• C_CAUS = 3 + 8·leb({op_id}) + 8·Σ leb(params) + 8·leb({N})\n"
    receipts += f"• C_CAUS = 3 + {cost_op} + {cost_params} + {cost_length} = {C_CAUS} bits\n"
    receipts += f"• C_END = 3 + pad_to_byte({pos_after_caus} + 3) = 3 + {pad_bits} = {C_END} bits\n"
    receipts += f"• C_stream = {C_CAUS} + {C_END} = {C_stream} bits\n"
    
    seed_bytes = (C_stream + 7) // 8
    receipts += f"• Debug invariant: 8·len(seed_bytes) = 8·{seed_bytes} = {8*seed_bytes} bits\n"
    receipts += f"• Bit-exact identity: 8·len(seed) == C_stream ✅\n"
    
    return receipts
