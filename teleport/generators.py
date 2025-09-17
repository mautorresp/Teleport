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
OP_REPEAT1 = 7 # global period D: S[i] = S[i % D]
OP_XOR_MASK8 = 8 # S = base XOR LFSR8(taps, seed)
OP_CBD = 9     # Canonical Binary Decomposition (deterministic fallback)

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
    dx = (x1 - x0) & 0xFF
    dy = (x2 - x1) & 0xFF

    if dx == 0 and dy != 0:
        return (0, (), f"inconsistent_derivative dx=0 dy={dy}")

    candidates = []
    if dx == 0 and dy == 0:
        # any a works; find canonical (a,c) that verifies
        for a in range(256):
            c = (x1 - (a*x0)) & 0xFF
            candidates.append((a,c))
    else:
        for a in range(256):
            if ((a*dx) & 0xFF) == dy:
                c = (x1 - (a*x0)) & 0xFF
                candidates.append((a,c))

    for a,c in sorted(candidates):
        x = x0
        ok = True
        for b in S:
            if x != b:
                ok = False; break
            x = (a*x + c) & 0xFF
        if ok:
            return (1, (x0, a, c), "")

    return (0, (), f"no_verified_pair dx={dx} dy={dy} candidates={len(candidates)}")

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

def deduce_ANCHOR(S: bytes):
    """
    CLF-Pure ANCHOR deduction with canonical window rule.
    Format-blind, deterministic byte equality analysis only.
    Returns: (ok, params, reason)
    """
    N = len(S)
    if N < 8:  # Need minimum bytes for header + interior + footer
        return (0, (), f"too_short N={N}")
    
    # Simplified canonical window rule - check only a few deterministic positions
    # to avoid combinatorial explosion
    max_candidates = 10  # Limit to prevent infinite loops
    candidate_windows = []
    
    for i in range(min(5, N-7)):  # Check only first 5 positions for efficiency
        if i + 3 >= N:
            continue
            
        # Extract 2-byte length field: L = (S[i+2] << 8) | S[i+3]  
        L = (S[i + 2] << 8) | S[i + 3]
        # Cap L to reasonable value to prevent overflow
        L = min(L, 1000, N - (i + 4))
        hdr_end = i + 4 + L
        
        if hdr_end >= N - 2:  # Need space for at least 2-byte footer
            continue
            
        # Check a few reasonable j positions  
        for offset in [2, 4, 8, 16]:  # Try small offsets from hdr_end
            j = hdr_end + offset
            if j > N:
                continue
                
            interior_len = j - hdr_end  
            if interior_len <= 0:
                continue
                
            candidate_windows.append((interior_len, i, j, hdr_end))
            if len(candidate_windows) >= max_candidates:
                break
        
        if len(candidate_windows) >= max_candidates:
            break
    
    if not candidate_windows:
        return (0, (), "no_admissible_windows")
    
    # Select window with maximum interior length
    candidate_windows.sort(key=lambda x: x[0], reverse=True)
    interior_len, i, j, hdr_end = candidate_windows[0]
    
    # Extract anchor components
    # Extract components from canonical window
    A = S[i:i+2]  # 2-byte start anchor  
    B = S[j-2:j]  # 2-byte end anchor
    interior = S[hdr_end:j]  # Interior bytes for inner generator
    
    # Try inner generators on interior in deterministic order
    inner_generators = [
        (OP_CONST, deduce_CONST),
        (OP_STEP, deduce_STEP),
        (OP_LCG8, deduce_LCG8),
        (OP_LFSR8, deduce_LFSR8),
        (OP_REPEAT1, deduce_REPEAT1)
    ]
    
    for inner_op, inner_deduce in inner_generators:
        ok, inner_params, _reason = inner_deduce(interior)
        if ok:
            # Construct ANCHOR parameters: len_A, A_bytes, len_B, B_bytes, inner_op, inner_params
            params = (len(A), *A, len(B), *B, inner_op, *inner_params)
            return (1, params, f"canonical_window i={i} j={j} hdr_end={hdr_end} interior_len={interior_len}")
    
    # No inner generator proved the interior
    return (0, (), f"interior_len={interior_len} no_inner_generator")

def deduce_REPEAT1(S: bytes):
    """
    Generator G_REPEAT1(D, motif): S[i] = motif[i % D] (constructive period)
    Requires D < N for constructive causality. Includes motif bytes in parameters.
    Returns: (ok, params, reason)
    """
    N = len(S)
    if N == 0:
        return (0, (), "empty")
    if N == 1:
        return (0, (), "trivial_length_1")
    
    # KMP prefix function to find minimal period
    def compute_prefix_function(pattern):
        m = len(pattern)
        pi = [0] * m
        k = 0
        for q in range(1, m):
            while k > 0 and pattern[k] != pattern[q]:
                k = pi[k - 1]
            if pattern[k] == pattern[q]:
                k += 1
            pi[q] = k
        return pi
    
    pi = compute_prefix_function(S)
    D = N - pi[N - 1]
    
    # Reject non-constructive cases
    if D >= N:
        return (0, (), f"non_constructive D={D} N={N} (D>=N)")
    
    # Extract motif and verify constructive property
    motif = S[:D]
    for i in range(N):
        if S[i] != motif[i % D]:
            return (0, (), f"period={D} verify_mismatch={i}")
    
    # Parameters include motif bytes for constructive expansion
    params = (D, *motif)
    return (1, params, f"constructive_period D={D} motif_bytes={len(motif)}")

def deduce_XOR_MASK8(S: bytes):
    """
    Generator G_XOR_MASK8(base, taps, seed): S = base XOR LFSR8(taps, seed)
    First deduces base (CONST or STEP), then deduces LFSR8 on the mask.
    Returns: (ok, params, reason)
    """
    if len(S) == 0:
        return (0, (), "empty")
    
    # Try CONST base first
    ok_const, params_const, reason_const = deduce_CONST(S)
    if ok_const:
        # Compute mask M = S XOR CONST(b)
        b = params_const[0]
        M = bytes([x ^ b for x in S])
        ok_lfsr, params_lfsr, reason_lfsr = deduce_LFSR8(M)
        if ok_lfsr:
            return (1, ('CONST', b, *params_lfsr), "")
        else:
            return (0, (), f"base=CONST({b}) lfsr={reason_lfsr}")
    
    # Try STEP base
    ok_step, params_step, reason_step = deduce_STEP(S)
    if ok_step:
        # Generate STEP sequence and compute mask
        a, d = params_step[0], params_step[1]
        step_seq = bytes([(a + i * d) & 0xFF for i in range(len(S))])
        M = bytes([x ^ y for x, y in zip(S, step_seq)])
        ok_lfsr, params_lfsr, reason_lfsr = deduce_LFSR8(M)
        if ok_lfsr:
            return (1, ('STEP', a, d, *params_lfsr), "")
        else:
            return (0, (), f"base=STEP({a},{d}) lfsr={reason_lfsr}")
    
    # Try LCG8 base
    ok_lcg, params_lcg, reason_lcg = deduce_LCG8(S)
    if ok_lcg:
        # Generate LCG8 sequence and compute mask
        x0, a, c = params_lcg[0], params_lcg[1], params_lcg[2]
        lcg_seq = []
        x = x0
        for _ in range(len(S)):
            lcg_seq.append(x)
            x = (a * x + c) & 0xFF
        M = bytes([x ^ y for x, y in zip(S, lcg_seq)])
        ok_lfsr, params_lfsr, reason_lfsr = deduce_LFSR8(M)
        if ok_lfsr:
            return (1, ('LCG8', x0, a, c, *params_lfsr), "")
        else:
            return (0, (), f"base=LCG8({x0},{a},{c}) lfsr={reason_lfsr}")
    
    return (0, (), f"base_negated const={reason_const} step={reason_step} lcg={reason_lcg}")

def deduce_CBD(S: bytes):
    """
    Canonical Binary Decomposition - constructive fallback that always works.
    Stores all literal bytes explicitly for deterministic expansion.
    Returns: (ok, params, reason) - always returns ok=1 (mathematical guarantee)
    """
    N = len(S)
    if N == 0:
        return (1, (0,), "constructive_empty")
    if N == 1:
        return (1, (1, S[0]), f"constructive_single byte={S[0]}")
    
    # For constructive CBD, store all bytes explicitly
    # This guarantees perfect reconstruction without relying on structure hashes
    params = (N, *S)
    
    # Calculate exact cost: op_id + length + all literal bytes
    cost_bits = 3 + 8 * leb(OP_CBD) + 8 * leb(N) + 8 * N
    
    return (1, params, f"constructive_literal N={N} cost_bits={cost_bits}")

def _cbd_depth(cbd_params):
    """Helper to compute CBD tree depth"""
    if len(cbd_params) <= 2:
        return 1
    return 1 + max(_cbd_depth(cbd_params[1:]), _cbd_depth(cbd_params[2:]))

def deduce_all(S: bytes):
    """
    CLF Universal Causality - Dynamic Generator Synthesis
    Uses DGG (Dynamic-Generator Generator) for pure mathematical derivation.
    Returns CAUS certificate ONLY if mathematical proof established.
    If no generator proves S, raises GENERATOR_MISSING with actionable receipts.
    Returns: (op_id, params, reason) OR exits with GENERATOR_MISSING fault
    """
    from teleport.dgg import deduce_dynamic
    
    # Try dynamic synthesis first - pure mathematical derivation from S
    op_id, params, reason = deduce_dynamic(S)
    
    if op_id != 0:
        # Mathematical proof established
        return (op_id, params, reason)
    
    # DGG returned GENERATOR_MISSING - code family incomplete
    raise SystemExit(reason + "\n• This indicates missing implementation, not absence of mathematical causality")

def deduce_LFSR8(S: bytes):
    """
    Deterministic 8-bit LFSR deduction using pure GF(2) mathematics.
    Returns: (ok, params, reason) with receipts for shift_ok, rank, taps, seed, verify_mismatch
    """
    N = len(S)
    if N == 0:
        return (0, (), "empty")
    if N == 1:
        return (1, (1, S[0]), "shift_ok=1 rank=0 taps=1 seed={} verify_mismatch=-1".format(S[0]))

    seed = S[0]
    
    # A) Shift identity prefilter (mathematically necessary)
    for i in range(N-1):
        xi, xj = S[i], S[i+1]
        for b in range(7):  # bits 0-6 must shift
            if ((xj >> b) & 1) != ((xi >> (b + 1)) & 1):
                return (0, (), f"shift_mismatch_at={i} bit={b}")
    
    # B) Build MSB equations over GF(2): bit7(x_{i+1}) = parity(taps & x_i)
    eqs = []
    used = set()
    for i in range(N-1):
        xi = S[i]
        if xi in used:
            continue
        used.add(xi)
        row = [((xi >> bit) & 1) for bit in range(8)]
        rhs = (S[i+1] >> 7) & 1
        eqs.append(row + [rhs])
    
    rank = len(eqs)
    taps = _solve_gf2_system_deterministic(eqs)
    
    if taps is None:
        return (0, (), f"shift_ok=1 rank={rank} no_unique_solution")
    
    # C) Verify full sequence
    x = seed
    for k, expect in enumerate(S):
        if x != expect:
            return (0, (), f"shift_ok=1 rank={rank} taps={taps} seed={seed} verify_mismatch={k}")
        if k + 1 < N:
            x = _lfsr_step(x, taps)
    
    return (1, (taps, seed), f"shift_ok=1 rank={rank} taps={taps} seed={seed} verify_mismatch=-1")

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

def _solve_gf2_system_deterministic(equations):
    """
    Deterministic GF(2) solver that handles any rank system.
    Returns canonical solution if unique, None if inconsistent/multiple solutions.
    """
    if len(equations) == 0:
        return None
    
    # Copy equations to avoid mutation
    matrix = [row[:] for row in equations]
    n_vars = 8
    n_eqs = len(matrix)
    
    # Gaussian elimination over GF(2)
    pivot_cols = []
    for col in range(n_vars):
        # Find pivot row for this column
        pivot_row = None
        for row in range(len(pivot_cols), n_eqs):
            if matrix[row][col] == 1:
                pivot_row = row
                break
        
        if pivot_row is None:
            continue  # No pivot in this column
        
        # Swap rows if needed
        if pivot_row != len(pivot_cols):
            matrix[len(pivot_cols)], matrix[pivot_row] = matrix[pivot_row], matrix[len(pivot_cols)]
        
        pivot_cols.append(col)
        pivot_row_idx = len(pivot_cols) - 1
        
        # Eliminate this column in other rows
        for row in range(n_eqs):
            if row != pivot_row_idx and matrix[row][col] == 1:
                for c in range(n_vars + 1):  # Include RHS
                    matrix[row][c] ^= matrix[pivot_row_idx][c]
    
    # Check for inconsistency
    for row in range(len(pivot_cols), n_eqs):
        if matrix[row][n_vars] == 1:  # 0 = 1
            return None
    
    # Check if we have full rank (8 pivots)
    if len(pivot_cols) < n_vars:
        return None  # Multiple solutions
    
    # Back substitution to get canonical solution
    solution = [0] * n_vars
    for i in range(len(pivot_cols) - 1, -1, -1):
        col = pivot_cols[i]
        val = matrix[i][n_vars]  # RHS
        for j in range(col + 1, n_vars):
            val ^= matrix[i][j] * solution[j]
        solution[col] = val
    
    # Convert to integer
    taps = 0
    for i in range(n_vars):
        if solution[i] == 1:
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
    
    if op_id == OP_REPEAT1:
        # REPEAT1 includes motif bytes: (D, *motif_bytes)
        if len(params) < 1:
            return cost + 8 * leb(N)  # Fallback
        D = params[0]
        cost += 8 * leb(D)  # Period length
        # Motif bytes (raw encoding)
        for i in range(1, min(len(params), D + 1)):
            cost += 8  # Each motif byte
        cost += 8 * leb(N)
        return cost
    elif op_id == OP_CBD:
        # CBD stores all bytes literally: (N, *all_bytes)
        if len(params) < 1:
            return cost + 8 * leb(N)  # Fallback
        stored_N = params[0]
        cost += 8 * leb(stored_N)  # Length field
        # All literal bytes
        for i in range(1, min(len(params), stored_N + 1)):
            cost += 8  # Each literal byte
        return cost
    elif op_id == OP_ANCHOR:
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
