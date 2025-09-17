# teleport/generators.py
# All functions: integers-only, bytes-only, no floats.
# Complete mathematical generators with exact deduction and verification.

from teleport.leb_io import leb128_emit_single

# Each generator G returns (ok, params, reason). If ok==1, also returns a verifier.
# Cost model (bits) for CAUS tokens: C_CAUS = 3 + 8*leb(op) + 8*Σ leb(param_i) + 8*leb(N)

OP_CONST = 2   # byte b repeated
OP_STEP  = 3   # a + i*d mod 256
OP_LCG8  = 4   # x_{i+1} = (a*x_i + c) mod 256, emit x_i
OP_LFSR8 = 5   # 8-bit LFSR with taps mask T, seed s (bit-stream packed to bytes)
OP_ANCHOR = 6  # dual-anchor A + G_inner(θ) + B

def leb(x: int) -> int:
    """Return minimal LEB128 byte length for integer x"""
    return len(leb128_emit_single(x))

def deduce_CONST(S: bytes):
    """
    Generator G_CONST(b): S[i] = b for all i
    Returns: (ok, params, reason)
    """
    if not S: 
        return (0, (), "empty")
    
    b0 = S[0]
    for i, b in enumerate(S):
        if b != b0:
            return (0, (), f"mismatch_at={i} exp={b0} got={b}")
    
    return (1, (b0,), "")

def deduce_STEP(S: bytes):
    """
    Generator G_STEP(a,d): S[i] = (a + i*d) mod 256
    Returns: (ok, params, reason) 
    """
    if len(S) < 2: 
        return (0, (), "need_len≥2")
    
    a = S[0]
    d = (S[1] - S[0]) & 0xFF
    
    for i, b in enumerate(S):
        expected = (a + i*d) & 0xFF
        if b != expected:
            return (0, (), f"mismatch_at={i} exp={expected} got={b}")
    
    return (1, (a, d), "")

def deduce_LCG8(S: bytes):
    """
    Generator G_LCG8(x0,a,c): x_{i+1} = (a*x_i + c) mod 256, emit x_i
    Returns: (ok, params, reason)
    """
    if len(S) < 3: 
        return (0, (), "need_len≥3")
    
    x0, x1, x2 = S[0], S[1], S[2]
    
    # Solve (a*x0 + c) % 256 = x1 and (a*x1 + c) % 256 = x2
    # Subtract: a*(x1 - x0) ≡ (x2 - x1) (mod 256)
    dx = (x1 - x0) & 0xFF
    dy = (x2 - x1) & 0xFF
    
    # Explicit degenerate case check
    if dx == 0 and dy != 0:
        return (0, (), "lcg_inconsistent_dx0_dy!=0")
    
    # Try all a ∈ [0..255] that satisfy a*dx ≡ dy (mod 256)
    sols = []
    for a in range(256):
        if ((a * dx) & 0xFF) == dy:
            c = (x1 - (a * x0)) & 0xFF
            sols.append((a, c))
    
    # Verify candidates; pick lexicographically smallest (canonical)
    for a, c in sorted(sols):
        x = x0
        ok = True
        for i, b in enumerate(S):
            if x != b:
                ok = False
                break
            x = (a * x + c) & 0xFF
        if ok: 
            return (1, (x0, a, c), "")
    
    return (0, (), f"no_solution_count={len(sols)}")

def _lfsr_step(word: int, taps: int) -> int:
    """
    8-bit LFSR step: XOR tap parity on LSB, shift right, insert feedback in MSB
    """
    # Compute parity bit of (word & taps)
    v = word & taps
    fb = 0
    while v:
        fb ^= (v & 1)
        v >>= 1
    
    # Shift right and insert feedback in MSB
    out = ((word >> 1) | ((fb & 1) << 7)) & 0xFF
    return out

def deduce_ANCHOR(S: bytes, max_A: int = 64, max_B: int = 64):
    """
    Deduce OP_ANCHOR: S = A || M || B with non-empty interior M,
    where A,B are raw anchors (lengths ≤ max_A/max_B), and M is reproduced
    by an inner known generator (CONST/STEP/LCG8/LFSR8).
    Returns (ok, params, reason) with canonical smallest (|A|,|B|), then lexicographic A,B, then inner op/params.
    """
    N = len(S)
    if N == 0: 
        return (0, (), "empty")
    
    best = None

    # Enumerate anchor lengths deterministically, favoring smaller anchors
    for len_A in range(0, min(max_A, N-1)+1):
        for len_B in range(0, min(max_B, N-1-len_A)+1):
            if len_A + len_B >= N: 
                continue
            A = S[:len_A]
            B = S[N-len_B:] if len_B else b""
            M = S[len_A:N-len_B]
            
            # Require non-empty interior for proper anchor structure
            if len(M) == 0:
                continue
            
            # Try inner generators in fixed order (canonical rank)
            for op_id, deduce in (
                (OP_CONST, deduce_CONST),
                (OP_STEP,  deduce_STEP),
                (OP_LCG8,  deduce_LCG8),
                (OP_LFSR8, deduce_LFSR8),
            ):
                ok, inner_params, _r = deduce(M)
                if not ok:
                    continue
                # Canonical tuple encoding for params:
                params = (len_A, *A, len_B, *B, op_id, *inner_params)
                cand = (len_A, len_B, A, B, op_id, inner_params, params)
                if best is None or cand < best:
                    best = cand

    if best is None:
        return (0, (), "no_anchor_innerG")
    # return only the parameter vector layout used by compute_caus_cost/verify_generator
    return (1, best[-1], "")

def deduce_LFSR8(S: bytes):
    """
    Deduce 8-bit right-shift LFSR with taps mask T (1..255), seed s = S[0]:
      x_{i+1} = ((x_i >> 1) | (parity(T & x_i) << 7))
    Return (1,(taps, seed),"") iff S is exactly generated by some taps∈[1..255].
    """

    N = len(S)
    if N == 0:
        return (0, (), "empty")

    # Degenerate N=1 is admissible: any taps generate the single state
    if N == 1:
        return (1, (1, S[0]), "")

    seed = S[0]

    # A) Enforce shift constraints for all transitions
    for i in range(N-1):
        xi = S[i]
        xj = S[i+1]
        # next lower 7 bits must equal previous upper 7 bits:
        if (xj & 0x7F) != (xi >> 1):
            return (0, (), f"shift_mismatch_at={i}: next_low7={xj & 0x7F} prev_hi7={xi >> 1}")

    # B) Build MSB equations over GF(2):
    # For each i:  bit7(x_{i+1}) = parity(taps & x_i)
    # => sum_j (t_j * (x_i >> j) & 1) == (x_{i+1} >> 7) & 1  (mod 2)
    eqs = []
    used = set()
    for i in range(N-1):
        xi = S[i]
        if xi in used:
            # duplicate lhs row adds no rank; still okay, we just keep first 8 independent rows below
            continue
        used.add(xi)
        row = [((xi >> bit) & 1) for bit in range(8)]
        rhs = (S[i+1] >> 7) & 1
        eqs.append(row + [rhs])
        if len(eqs) == 8:  # enough to solve 8 unknown taps bits
            break

    taps = None
    if len(eqs) == 8:
        cand = _solve_gf2_system(eqs)  # returns integer taps or None
        if cand is not None and cand != 0:
            # Verify full replay
            x = seed
            ok = True
            for k, expect in enumerate(S):
                if x != expect:
                    ok = False
                    break
                if k + 1 < N:
                    # one step
                    v = x & cand
                    fb = 0
                    while v:
                        fb ^= (v & 1)
                        v >>= 1
                    x = ((x >> 1) | ((fb & 1) << 7)) & 0xFF
            if ok:
                taps = cand

    # C) Brute-force fallback to guarantee completeness
    if taps is None:
        best = None
        for t in range(1, 256):  # deterministic enumeration
            x = seed
            ok = True
            for k, expect in enumerate(S):
                if x != expect:
                    ok = False
                    break
                if k + 1 < N:
                    v = x & t
                    fb = 0
                    while v:
                        fb ^= (v & 1)
                        v >>= 1
                    x = ((x >> 1) | ((fb & 1) << 7)) & 0xFF
            if ok:
                best = t if best is None or t < best else best
        if best is not None:
            taps = best

    if taps is None:
        return (0, (), "no_lfsr_params")
    return (1, (taps, seed), "")

def _solve_gf2_system(equations):
    """
    Solve system of linear equations over GF(2) using Gaussian elimination.
    equations: list of [coeff0, coeff1, ..., coeff7, result]
    Returns: taps value (0-255) if unique solution exists, None otherwise
    """
    if len(equations) != 8:
        return None
    
    # Copy equations to avoid mutation
    matrix = [row[:] for row in equations]
    
    # Gaussian elimination over GF(2)
    for pivot_col in range(8):
        # Find pivot row
        pivot_row = None
        for row in range(pivot_col, 8):
            if row < len(matrix) and matrix[row][pivot_col] == 1:
                pivot_row = row
                break
        
        if pivot_row is None:
            return None  # No unique solution
        
        # Swap rows if needed
        if pivot_row != pivot_col:
            matrix[pivot_col], matrix[pivot_row] = matrix[pivot_row], matrix[pivot_col]
        
        # Eliminate other rows
        for row in range(8):
            if row != pivot_col and row < len(matrix) and matrix[row][pivot_col] == 1:
                for col in range(9):  # 8 coeffs + 1 result
                    matrix[row][col] ^= matrix[pivot_col][col]  # XOR for GF(2)
    
    # Back substitution
    taps = 0
    for i in range(8):
        if i < len(matrix):
            if matrix[i][8] == 1:  # result bit
                taps |= (1 << i)
    
    return taps if taps > 0 else None

def compute_caus_cost(op_id: int, params: tuple, N: int) -> int:
    """
    Exact CAUS cost computation in bits:
    C_CAUS = 3 + 8*leb(op_id) + 8*Σ leb(param_i) + 8*leb(N)
    
    For OP_ANCHOR: includes anchor bytes and inner generator parameters
    """
    cost = 3  # CAUS tag bits
    cost += 8 * leb(op_id)
    
    if op_id == OP_ANCHOR:
        # Simplified encoding for anchor generators  
        # params: (len_A, *A_bytes, len_B, *B_bytes, inner_op, *inner_params)
        if len(params) < 2:
            return cost + 8 * leb(N)  # Fallback
            
        i = 0
        len_A = params[i]; i += 1
        cost += 8 * leb(len_A)
        
        # A bytes (raw encoding)
        for j in range(len_A):
            if i < len(params):
                cost += 8  # Each anchor byte
                i += 1
        
        if i < len(params):
            len_B = params[i]; i += 1  
            cost += 8 * leb(len_B)
            
            # B bytes (raw encoding)
            for j in range(len_B):
                if i < len(params):
                    cost += 8  # Each anchor byte
                    i += 1
                    
            # Inner generator operation and parameters
            while i < len(params):
                cost += 8 * leb(params[i])
                i += 1
    else:
        # Standard generators
        for param in params:
            cost += 8 * leb(param)
    
    cost += 8 * leb(N)
    return cost

def verify_generator(op_id: int, params: tuple, S: bytes) -> bool:
    """
    Verify that generator with given parameters reproduces S exactly
    """
    if op_id == OP_CONST:
        b, = params
        return all(byte == b for byte in S)
    
    elif op_id == OP_STEP:
        a, d = params
        for i, byte in enumerate(S):
            if byte != ((a + i * d) & 0xFF):
                return False
        return True
    
    elif op_id == OP_LCG8:
        x0, a, c = params
        x = x0
        for i, byte in enumerate(S):
            if x != byte:
                return False
            x = (a * x + c) & 0xFF
        return True
    
    elif op_id == OP_LFSR8:
        taps, seed = params
        x = seed
        for i, byte in enumerate(S):
            if x != byte:
                return False
            if i < len(S) - 1:  # Don't advance after last byte
                x = _lfsr_step(x, taps)
        return True
    
    elif op_id == OP_ANCHOR:
        # Verify anchor generator: A + G_inner(θ) + B = S
        if len(params) < 2:
            return False
            
        i = 0
        len_A = params[i]; i += 1
        
        if i + len_A >= len(params):
            return False
            
        A = bytes(params[i:i+len_A]); i += len_A
        
        if i >= len(params):
            return False
            
        len_B = params[i]; i += 1
        
        if i + len_B > len(params):
            return False
            
        B = bytes(params[i:i+len_B]); i += len_B
        
        # Check anchors match
        if not S.startswith(A) or not S.endswith(B):
            return False
            
        # Extract interior and verify with inner generator
        interior = S[len(A):len(S)-len(B)]
        
        if len(interior) == 0:
            return i == len(params)  # No inner generator needed
            
        if i >= len(params):
            return False  # Missing inner generator
            
        inner_op = params[i]; i += 1
        inner_params = params[i:] if i < len(params) else ()
        
        return verify_generator(inner_op, inner_params, interior)
    
    return False
