# === CLF CALCULATOR-GRADE VALIDATOR — FIXED MAXIMALITY, ELIMINATES DRIFT ===
# CLF STANCE (PINNED):
# Units: bits
# leb_len_u(n): 7-bit shifts; leb_len_u(0)=1
# H(L) = 16 + 8*leb_len_u(8*L)
# C_CAUS(op, params[], L_tok) = 3 + 8*leb_len_u(op) + Σ 8*leb_len_u(param_i) + 8*leb_len_u(L_tok)
# C_END(bitpos) = 3 + ((8 - ((bitpos + 3) % 8)) % 8)
# Roles: A = one whole-range token or N/A. B = multi-token tiling. No exceptions.
# Decision: C_min_total = H + min(complete_streams); EMIT iff C_min_total < 8*L
# Deduction ⇄ Expansion are the same math inverted

import hashlib
import os
import subprocess
import sys

BUILD_ID = "CLF_FIXED_MAXIMAL_20250923"

# ---------- PINS (must not change) ----------
def leb_len_u(n:int)->int:
    assert n>=0
    if n==0: return 1
    c=0
    while n>0:
        n >>= 7
        c += 1
    return c

def header_bits(L:int)->int: return 16 + 8*leb_len_u(8*L)

def end_bits(bitpos:int)->int: return 3 + ((8 - ((bitpos+3)%8))%8)

def caus_bits(op:int, params:list[int], L_tok:int)->int:
    return 3 + 8*leb_len_u(op) + sum(8*leb_len_u(p) for p in params) + 8*leb_len_u(L_tok)

# ---------- FIXED STEP-RUN ADMISSIBILITY (STRICT MAXIMALITY) ----------
def verify_step_run_admissibility(S:bytes, off:int, L:int, start:int, step:int, total_L:int):
    """
    STEP-RUN token for segment S[off:off+L) with params (start, step) is lawful iff:
    (i) seed match: start == S[off]
    (ii) step deduced locally: If L≥2: step == (S[off+1] - S[off]) % 256
    (iii) progression holds: For all i in [0,L): S[off+i] == (start + i*step) % 256
    (iv) strict maximality: Left-maximal AND Right-maximal
    """
    print(f"STEP-RUN ADMISSIBILITY: off={off} L={L} start={start} step={step}")
    
    # Clause (i): seed match
    clause_i = (start == S[off])
    print(f"Clause (i): start({start}) == S[{off}]({S[off]}) → {clause_i}")
    if not clause_i:
        return False
    
    # Clause (ii): step deduced locally
    if L >= 2:
        expected_step = (S[off+1] - S[off]) % 256
        clause_ii = (step == expected_step)
        print(f"Clause (ii): L>=2, step({step}) == (S[{off+1}]({S[off+1]}) - S[{off}]({S[off]}))%256 = {expected_step} → {clause_ii}")
        if not clause_ii:
            return False
    else:
        # L=1: define step=0 (but maximality will usually reject this)
        clause_ii = (step == 0)
        print(f"Clause (ii): L=1, step({step}) == 0 → {clause_ii}")
        if not clause_ii:
            return False
    
    # Clause (iii): progression holds
    clause_iii = True
    for i in range(L):
        expected = (start + i*step) % 256
        actual = S[off+i]
        if actual != expected:
            clause_iii = False
            print(f"Clause (iii): FAILED at i={i}: S[{off+i}]({actual}) != (start + i*step)%256 = {expected}")
            return False
    print(f"Clause (iii): All {L} positions match arithmetic progression → {clause_iii}")
    
    # Clause (iv): strict maximality
    left_maximal = (off == 0) or (S[off-1] != (start - step) % 256)
    right_maximal = (off + L == total_L) or (S[off+L] != (start + L*step) % 256)
    clause_iv = left_maximal and right_maximal
    
    if off > 0:
        prev_expected = (start - step) % 256
        print(f"Left maximality: off>0, S[{off-1}]({S[off-1]}) != {prev_expected} → {left_maximal}")
    else:
        print(f"Left maximality: off==0 → {left_maximal}")
    
    if off + L < total_L:
        next_expected = (start + L*step) % 256
        print(f"Right maximality: S[{off+L}]({S[off+L]}) != {next_expected} → {right_maximal}")
    else:
        print(f"Right maximality: off+L==total_L → {right_maximal}")
    
    print(f"Clause (iv): Maximal (left={left_maximal} AND right={right_maximal}) → {clause_iv}")
    
    all_clauses = clause_i and clause_ii and clause_iii and clause_iv
    print(f"STEP-RUN ADMISSIBILITY: {all_clauses}")
    return all_clauses

# ---------- GREEDY-MAXIMAL DETERMINISTIC SPLITTER ----------
def get_deductor_signature():
    """Return SHA256 of deduct_B function source + build ID"""
    source_lines = []
    with open(__file__, 'r') as f:
        in_function = False
        for line in f:
            if 'def deduct_B(' in line:
                in_function = True
            if in_function:
                source_lines.append(line)
                if line.strip().startswith('return toks') and 'deduct_B' in ''.join(source_lines):
                    break
    
    source_text = ''.join(source_lines) + BUILD_ID
    return hashlib.sha256(source_text.encode()).hexdigest()

def deduct_B(S:bytes):
    """
    FIXED GREEDY-MAXIMAL DETERMINISTIC SPLITTER
    Scan left→right with fixed precedence:
    1. Try maximal STEP-RUN (with strict maximality verification)
    2. Else try maximal CONST-RUN (STEP-RUN with step=0)
    3. Else 1-byte CONST fallback
    NEVER start a token where left-maximality would fail
    """
    L = len(S)
    i = 0
    toks = []
    
    while i < L:
        # Try maximal STEP-RUN from position i
        found_step_run = False
        
        if i + 1 < L:  # Need at least 2 bytes for non-trivial STEP-RUN
            start = S[i]
            step = (S[i+1] - S[i]) % 256
            
            # Extend maximally to the right
            j = i + 2
            while j < L and S[j] == ((start + (j-i)*step) % 256):
                j += 1
            
            step_len = j - i
            
            # Verify strict maximality (all 4 clauses)
            if verify_step_run_admissibility(S, i, step_len, start, step, L):
                toks.append((11, [start, step], step_len))
                i = j
                found_step_run = True
        
        if not found_step_run:
            # Try maximal CONST-RUN (STEP-RUN with step=0)
            start = S[i]
            step = 0
            
            # Extend maximally to the right
            j = i + 1
            while j < L and S[j] == start:
                j += 1
            
            const_len = j - i
            
            # Verify strict maximality for CONST-RUN
            if verify_step_run_admissibility(S, i, const_len, start, step, L):
                toks.append((10, [start], const_len))
                i = j
            else:
                # Fallback: 1-byte CONST (should be rare if maximality is correct)
                print(f"FALLBACK: 1-byte CONST at position {i}")
                toks.append((10, [S[i]], 1))
                i += 1
    
    return toks

# ---------- FALSIFIABILITY FOR LONG TOKENS ----------
def verify_long_token_bytes(S:bytes, off:int, L:int, start:int, step:int, token_id:str):
    """Print falsifiable evidence for every token L_tok >= 32"""
    print(f"TOKEN {token_id}: off={off} L={L} start={start} step={step}")
    
    if L >= 32:
        print(f"LONG TOKEN VERIFICATION: {token_id}")
        # First 32 bytes as integers
        window = S[off:off+min(32, L)]
        print(f"First 32 bytes: {list(window)}")
        distinct_count = len(set(window))
        print(f"Distinct values in first 32 bytes: {distinct_count}")
        
        # Verify clauses for this segment
        print(f"FALSIFIABILITY CHECK:")
        all_good = verify_step_run_admissibility(S, off, L, start, step, len(S))
        if not all_good:
            print(f"ABORT: TOKEN {token_id} FAILS ADMISSIBILITY")
            sys.exit(1)
    else:
        print(f"Short token (L<32): basic verification")
        # Still verify admissibility for short tokens
        if not verify_step_run_admissibility(S, off, L, start, step, len(S)):
            print(f"ABORT: TOKEN {token_id} FAILS ADMISSIBILITY") 
            sys.exit(1)

# ---------- LOCALITY GUARDS ----------
def expand_token_with_locality_guard(op, params, L_tok):
    """Expansion using ONLY token fields - no RAW readback"""
    if op == 10:  # CONST-RUN
        assert len(params) == 1, f"CONST-RUN requires 1 param, got {len(params)}"
        return bytes([params[0]]) * L_tok
    
    if op == 11:  # STEP-RUN
        assert len(params) == 2, f"STEP-RUN requires 2 params, got {len(params)}"
        start, step = params
        return bytes(((start + i*step) % 256) for i in range(L_tok))
    
    raise ValueError(f"Unknown operator: {op}")

# ---------- DETERMINISM GUARDS ----------
def verify_tokenization_determinism(S:bytes, file_path:str):
    """Same S must yield same tokens across runs"""
    print("TOKENIZATION DETERMINISM CHECK")
    
    # Run 1: Same process
    tokens1 = deduct_B(S)
    tokens1_str = str(tokens1)
    tokens1_sha = hashlib.sha256(tokens1_str.encode()).hexdigest()
    print(f"Run 1 (same process): tokens_sha={tokens1_sha}")
    
    # Run 2: Same process, second call
    tokens2 = deduct_B(S)
    tokens2_str = str(tokens2)
    tokens2_sha = hashlib.sha256(tokens2_str.encode()).hexdigest()
    print(f"Run 2 (same process): tokens_sha={tokens2_sha}")
    
    if tokens1_sha != tokens2_sha:
        print("ABORT: NON-DETERMINISTIC TOKENIZATION")
        sys.exit(1)
    
    print("DETERMINISM CHECK: PASS")
    return tokens1_sha, tokens1

# ---------- DEDUCTION ⇄ EXPANSION IDENTITY ----------
def verify_deduction_expansion_identity(S:bytes, tokens:list):
    """Verify deduct(S) → expand(tokens) → S and re-deduct consistency"""
    print("DEDUCTION ⇄ EXPANSION IDENTITY CHECK")
    
    # Expand tokens back to bytes
    reconstructed = b''
    for op, params, L_tok in tokens:
        segment = expand_token_with_locality_guard(op, params, L_tok)
        reconstructed += segment
    
    # Verify S == reconstructed
    s_sha = hashlib.sha256(S).hexdigest()
    recon_sha = hashlib.sha256(reconstructed).hexdigest()
    print(f"Original SHA: {s_sha}")
    print(f"Reconstructed SHA: {recon_sha}")
    
    if s_sha != recon_sha:
        print("ABORT: DEDUCTION→EXPANSION IDENTITY FAILED")
        sys.exit(1)
    
    # Re-deduct reconstructed bytes
    tokens_rededuced = deduct_B(reconstructed)
    tokens_rededuced_str = str(tokens_rededuced)
    rededuced_sha = hashlib.sha256(tokens_rededuced_str.encode()).hexdigest()
    
    original_tokens_str = str(tokens)
    original_sha = hashlib.sha256(original_tokens_str.encode()).hexdigest()
    
    print(f"Original tokens SHA: {original_sha}")
    print(f"Re-deduced tokens SHA: {rededuced_sha}")
    
    if original_sha != rededuced_sha:
        print("ABORT: RE-DEDUCTION INCONSISTENCY")
        sys.exit(1)
    
    print("DEDUCTION ⇄ EXPANSION IDENTITY: PASS")

# ---------- MAIN VALIDATOR ----------
def validate_pic1_calculator_grade():
    """Validate pic1.jpg with fixed maximality and all guards"""
    file_path = "pic1.jpg"
    
    print(f"CLF CALCULATOR-GRADE VALIDATOR - FIXED MAXIMALITY")
    print(f"BUILD_ID: {BUILD_ID}")
    print(f"DEDUCTOR_SIGNATURE: {get_deductor_signature()}")
    
    if not os.path.exists(file_path):
        print(f"ABORT: {file_path} not found")
        sys.exit(1)
    
    S = open(file_path, "rb").read()
    L = len(S)
    RAW = 8 * L
    H = header_bits(L)
    
    print(f"INPUT: {file_path} L={L} bytes, RAW={RAW} bits, H={H} bits")
    
    # Determinism check
    tokens_sha, tokens = verify_tokenization_determinism(S, file_path)
    
    print(f"\nB-PATH ANALYSIS (FIXED MAXIMALITY):")
    print(f"Token count: {len(tokens)}")
    
    # Verify each token with falsifiability
    off = 0
    B_caus = 0
    
    # Print first ten tokens and all long tokens
    for idx, (op, params, Lt) in enumerate(tokens):
        if idx < 10 or Lt >= 32:
            if op == 11:  # STEP-RUN
                start, step = params
            else:  # CONST-RUN
                start, step = params[0], 0
            verify_long_token_bytes(S, off, Lt, start, step, f"B_{idx}")
        
        # Verify expansion
        seg = S[off:off+Lt]
        exp = expand_token_with_locality_guard(op, params, Lt)
        if seg != exp:
            print(f"ABORT: EXPANSION MISMATCH at token B_{idx}")
            sys.exit(1)
        
        # Cost calculation
        c = caus_bits(op, params, Lt)
        B_caus += c
        off += Lt
    
    # Coverage check
    if off != L:
        print(f"ABORT: COVERAGE MISMATCH {off} != {L}")
        sys.exit(1)
    
    print(f"\nTOTALS:")
    print(f"Σ L_tok = {off} (coverage verified)")
    
    B_end = end_bits(B_caus)
    B_stream = B_caus + B_end
    
    print(f"B_caus = {B_caus}")
    print(f"B_end = {B_end}")
    print(f"B_stream = {B_stream}")
    
    # A-path (whole-range detection)
    A_stream = None
    if L == 0:
        A_stream = 0
        print("A: ZERO whole-range")
    elif all(b == S[0] for b in S):
        A_caus = caus_bits(10, [S[0]], L)
        A_stream = A_caus + end_bits(A_caus)
        print(f"A: CONST whole-range, A_stream={A_stream}")
    elif L >= 2:
        start, step = S[0], (S[1] - S[0]) % 256
        if all(S[i] == ((start + i*step) % 256) for i in range(L)):
            A_caus = caus_bits(11, [start, step], L)
            A_stream = A_caus + end_bits(A_caus)
            print(f"A: STEP whole-range, A_stream={A_stream}")
        else:
            print("A: N/A (no single whole-range operator)")
    else:
        print("A: N/A (L=1, no STEP possible)")
    
    # Decision algebra
    candidates = []
    if A_stream is not None:
        candidates.append(H + A_stream)
    candidates.append(H + B_stream)
    
    C_min_total = min(candidates)
    print(f"H = {H}")
    print(f"C_min_total = {C_min_total}")
    print(f"RAW = {RAW}")
    
    emit_decision = C_min_total < RAW
    print(f"EMIT GATE: {C_min_total} < {RAW} → {emit_decision}")
    
    # Identity verification
    verify_deduction_expansion_identity(S, tokens)
    
    print(f"\nFINAL RESULTS:")
    print(f"Tokens: {len(tokens)}")
    print(f"C_total: {C_min_total}")
    print(f"EMIT: {emit_decision}")
    print(f"Tokens SHA: {tokens_sha}")

if __name__ == "__main__":
    validate_pic1_calculator_grade()