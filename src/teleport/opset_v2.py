"""
Teleport OpSet_v2 - Normative Operator Registry
Zero-ambiguity deterministic procedures with fixed primitives.
"""

from teleport.leb_io import leb128_emit_single as leb_emit

def leb_len(x: int) -> int:
    """Minimal LEB128 byte length for integer x"""
    return len(leb_emit(x))

# OpSet_v2 Normative Registry (Fixed, Non-Negotiable)
OP_CONST = 1        # Constant byte generator: CONST(b, L)
OP_STEP = 2         # Arithmetic progression: STEP(start, stride, L) 
OP_LCG8 = 3         # Linear congruential generator: LCG8(x0, a, c, L)
OP_LFSR8 = 4        # Linear feedback shift register: LFSR8(taps, seed, L)
OP_REPEAT1 = 5      # Repeating motif: REPEAT1(D, motif..., L)
OP_ANCHOR = 6       # Anchor composition: ANCHOR(len_A, A..., len_B, B..., inner_op, inner_params..., L)
OP_CBD = 7          # Canonical binary decomposition: CBD(N, bytes..., L)

# Remove unregistered operators (OP_XOR_MASK8 not in normative registry)

def compute_caus_cost_v2(op_id: int, params: tuple, L: int) -> int:
    """
    OpSet_v2 Generic CAUS Cost Formula (Fixed):
    C_CAUS = 3 + 8×leb(op_id) + 8×Σ leb(param_i) + 8×leb(L)
    
    All hardcoded constants (19+, 27+, 35+) must derive from this formula.
    """
    cost = 3  # CAUS tag bits
    cost += 8 * leb_len(op_id)
    
    # Parameter costs - exact summation over all params
    for param in params:
        cost += 8 * leb_len(param)
    
    # Payload length cost
    cost += 8 * leb_len(L)
    
    return cost

def compute_end_cost_v2(caus_bits: int) -> int:
    """
    OpSet_v2 END Cost Formula (Fixed):
    C_END = 3 + pad_to_byte(C_CAUS + 3)
    pad_to_byte(k) = (8 - (k mod 8)) mod 8
    """
    pad_bits = (8 - ((caus_bits + 3) % 8)) % 8
    return 3 + pad_bits

def compute_stream_cost_v2(op_id: int, params: tuple, L: int) -> int:
    """
    OpSet_v2 Total Stream Cost:
    C_stream = C_CAUS + C_END
    
    Required invariant: 8×len(seed_bytes) == C_stream
    """
    c_caus = compute_caus_cost_v2(op_id, params, L)
    c_end = compute_end_cost_v2(c_caus)
    return c_caus + c_end

def is_admissible_v2(op_id: int, params: tuple, L: int) -> tuple[bool, str]:
    """
    OpSet_v2 Admissibility Check (Legality-Before-Pricing):
    1. Minimal LEB128 for all integers
    2. Domain checks per operator (ranges, arities, structural equalities)
    
    Returns: (admissible: bool, reason: str)
    """
    # Check 1: Minimal LEB128 for all integers
    if not is_minimal_leb128_int(op_id):
        return False, f"op_id {op_id} not minimal LEB128"
    
    for i, param in enumerate(params):
        if not is_minimal_leb128_int(param):
            return False, f"param[{i}] = {param} not minimal LEB128"
    
    if not is_minimal_leb128_int(L):
        return False, f"L = {L} not minimal LEB128"
    
    # Check 2: Domain checks per operator
    if op_id == OP_CONST:
        # CONST(b, L): b ∈ [0,255], params = (b,)
        if len(params) != 1:
            return False, f"CONST arity mismatch: expected 1 param, got {len(params)}"
        b = params[0]
        if not (0 <= b <= 255):
            return False, f"CONST b = {b} not in [0,255]"
        
    elif op_id == OP_STEP:
        # STEP(start, stride, L): start,stride ∈ [0,255], params = (start, stride)
        if len(params) != 2:
            return False, f"STEP arity mismatch: expected 2 params, got {len(params)}"
        start, stride = params
        if not (0 <= start <= 255):
            return False, f"STEP start = {start} not in [0,255]"
        if not (0 <= stride <= 255):
            return False, f"STEP stride = {stride} not in [0,255]"
            
    elif op_id == OP_LCG8:
        # LCG8(x0, a, c, L): x0,a,c ∈ [0,255], params = (x0, a, c)
        if len(params) != 3:
            return False, f"LCG8 arity mismatch: expected 3 params, got {len(params)}"
        x0, a, c = params
        if not (0 <= x0 <= 255):
            return False, f"LCG8 x0 = {x0} not in [0,255]"
        if not (0 <= a <= 255):
            return False, f"LCG8 a = {a} not in [0,255]"
        if not (0 <= c <= 255):
            return False, f"LCG8 c = {c} not in [0,255]"
        if a == 0:
            return False, f"LCG8 a = 0 (degenerate multiplier)"
            
    elif op_id == OP_LFSR8:
        # LFSR8(taps, seed, L): taps,seed ∈ [0,255], params = (taps, seed)
        if len(params) != 2:
            return False, f"LFSR8 arity mismatch: expected 2 params, got {len(params)}"
        taps, seed = params
        if not (0 <= taps <= 255):
            return False, f"LFSR8 taps = {taps} not in [0,255]"
        if not (0 <= seed <= 255):
            return False, f"LFSR8 seed = {seed} not in [0,255]"
        if taps == 0:
            return False, f"LFSR8 taps = 0 (degenerate polynomial)"
            
    elif op_id == OP_REPEAT1:
        # REPEAT1(D, motif..., L): D < L, len(motif) = D, params = (D, *motif)
        if len(params) < 1:
            return False, f"REPEAT1 missing D parameter"
        D = params[0]
        if D >= L:
            return False, f"REPEAT1 D = {D} >= L = {L} (non-constructive)"
        if len(params) != 1 + D:
            return False, f"REPEAT1 params length mismatch: expected {1+D}, got {len(params)}"
        motif = params[1:1+D]
        for i, byte_val in enumerate(motif):
            if not (0 <= byte_val <= 255):
                return False, f"REPEAT1 motif[{i}] = {byte_val} not in [0,255]"
                
    elif op_id == OP_ANCHOR:
        # ANCHOR(len_A, A..., len_B, B..., inner_op, inner_params..., L)
        # Structural equality: L == len(A) + L_interior + len(B)
        if len(params) < 2:
            return False, f"ANCHOR missing len_A parameter"
        
        len_A = params[0]
        if len_A < 0:
            return False, f"ANCHOR len_A = {len_A} < 0"
        if 1 + len_A >= len(params):
            return False, f"ANCHOR insufficient params for A bytes"
            
        A_bytes = params[1:1+len_A]
        for i, byte_val in enumerate(A_bytes):
            if not (0 <= byte_val <= 255):
                return False, f"ANCHOR A[{i}] = {byte_val} not in [0,255]"
        
        if 1 + len_A >= len(params):
            return False, f"ANCHOR missing len_B parameter"
        len_B = params[1 + len_A]
        if len_B < 0:
            return False, f"ANCHOR len_B = {len_B} < 0"
            
        if 1 + len_A + 1 + len_B > len(params):
            return False, f"ANCHOR insufficient params for B bytes"
        B_bytes = params[1 + len_A + 1:1 + len_A + 1 + len_B]
        for i, byte_val in enumerate(B_bytes):
            if not (0 <= byte_val <= 255):
                return False, f"ANCHOR B[{i}] = {byte_val} not in [0,255]"
        
        # Must have inner_op
        inner_start = 1 + len_A + 1 + len_B
        if inner_start >= len(params):
            return False, f"ANCHOR missing inner_op parameter"
        
        inner_op = params[inner_start]
        inner_params = params[inner_start + 1:]
        
        # Structural equality: L == len_A + L_interior + len_B
        L_interior = L - len_A - len_B
        if L_interior < 0:
            return False, f"ANCHOR negative interior length: L={L}, len_A={len_A}, len_B={len_B}"
        
        # Check inner admissibility recursively
        inner_admissible, inner_reason = is_admissible_v2(inner_op, inner_params, L_interior)
        if not inner_admissible:
            return False, f"ANCHOR inner not admissible: {inner_reason}"
            
    elif op_id == OP_CBD:
        # CBD(N, bytes..., L): N == L == len(bytes), params = (N, *bytes)
        if len(params) < 1:
            return False, f"CBD missing N parameter"
        N = params[0]
        if N != L:
            return False, f"CBD N = {N} != L = {L}"
        if len(params) != 1 + N:
            return False, f"CBD params length mismatch: expected {1+N}, got {len(params)}"
        bytes_data = params[1:1+N]
        for i, byte_val in enumerate(bytes_data):
            if not (0 <= byte_val <= 255):
                return False, f"CBD bytes[{i}] = {byte_val} not in [0,255]"
    else:
        return False, f"Unknown op_id: {op_id}"
    
    return True, "admissible"

def is_minimal_leb128_int(n: int) -> bool:
    """Check if integer n would be encoded as minimal LEB128"""
    if n < 0:
        return False
    # An integer is minimal LEB128 if leb128_emit produces the shortest possible encoding
    encoded = leb_emit(n)
    # Check that no shorter encoding exists (i.e., no leading zero bytes)
    if len(encoded) > 1 and encoded[-1] == 0:
        return False
    return True

# Literal boundary constant (fixed)
C_LIT_PER_BYTE = 10  # C_LIT(L) = 10×L bits

def compute_literal_cost(L: int) -> int:
    """Literal boundary cost: C_LIT(L) = 10×L bits"""
    return C_LIT_PER_BYTE * L
