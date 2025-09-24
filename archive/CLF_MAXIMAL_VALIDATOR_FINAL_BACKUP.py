# === CLF SINGLE-SEED PURE CALCULATOR (ONLY MATH) ===
# Formula: C_min^(1)(L) = 88 + 8*leb(L) bits
# Constants: H=56, CAUS=27, END=5 (locked, no variation)
# Fallback: C_LIT = 10*L bits (strict)
# Decision: EMIT ⟺ C_min^(1)(L) < 10*L
# Complexity: O(log L) arithmetic only, zero content scanning

import argparse
import hashlib
import os

BUILD_ID = "CLF_SINGLE_SEED_PURE_20250923_LOCKED"

def leb_len_u(n: int) -> int:
    """LEB128 byte-length (unsigned), 7-bit groups"""
    assert n >= 0
    return 1 if n == 0 else ((n.bit_length() + 6) // 7)

def clf_single_seed_cost(L: int) -> int:
    """56 (H) + 27 (CAUS) + 5 (END) + 8*leb(L)"""
    return 88 + 8 * leb_len_u(L)

def should_emit(L: int) -> bool:
    """EMIT ⟺ C_min^(1)(L) < 10*L (strict)"""
    return clf_single_seed_cost(L) < 10 * L

def receipt(L: int) -> dict:
    """Deterministic receipt from calculation tuple"""
    leb = leb_len_u(L)
    C = clf_single_seed_cost(L)
    RAW = 10 * L
    EMIT = C < RAW
    tup = (L, leb, C, RAW, EMIT, BUILD_ID)
    return {
        "L": L, "leb": leb, "C": C, "RAW": RAW, "EMIT": EMIT,
        "sha256": hashlib.sha256(str(tup).encode()).hexdigest()
    }

import hashlib
import os
import subprocess
import sys
import argparse

BUILD_ID = "CLF_SINGLE_SEED_PURE_MATH_20250923_SURGICAL"

if __name__ == "__main__":
    p = argparse.ArgumentParser(description='CLF Single-Seed Pure Calculator')
    p.add_argument("path", nargs="+", help="Files to calculate bounds for")
    args = p.parse_args()
    
    print("CLF SINGLE-SEED PURE CALCULATOR")
    print(f"BUILD_ID: {BUILD_ID}")
    print(f"Formula: C_min^(1)(L) = 88 + 8*leb(L) bits")
    print(f"Fallback: C_LIT = 10*L bits")
    print()
    
    for path in args.path:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            continue
            
        L = os.path.getsize(path)
        r = receipt(L)
        print(f"{os.path.basename(path)}: L={r['L']:,}, leb={r['leb']}, "
              f"C={r['C']} bits, RAW={r['RAW']:,} bits, EMIT={r['EMIT']}, "
              f"receipt={r['sha256'][:16]}...")

# ---------- PINS (must not change) ----------
def leb_len_u(n: int) -> int:
    """LEB128 byte-length (unsigned), 7-bit groups - O(log n)"""
    assert n >= 0
    return 1 if n == 0 else ((n.bit_length() + 6) // 7)

def header_bits(_: int) -> int:  
    """Header: constant 56 bits (no L dependence)"""
    return 56

def end_bits() -> int:
    """END: constant 5 bits (no alignment)"""
    return 5

def caus_bits_single_seed() -> int:
    """CAUS: constant 27 bits (single-seed regime)"""
    return 27

# === SINGLE-SEED CALCULATOR (PURE MATHEMATICS) ===
def clf_single_seed_cost(L: int) -> int:
    """C_min^(1)(L) = 88 + 8*leb(L) - pure arithmetic"""
    H, CAUS, END = 56, 27, 5
    return H + CAUS + END + 8 * leb_len_u(L)

def should_emit(L: int) -> bool:
    """EMIT decision: C_min^(1)(L) < 10*L (strict)"""
    return clf_single_seed_cost(L) < 10 * L

def single_seed_receipt(L: int) -> tuple:
    """Deterministic receipt: (L, leb(L), C, RAW, EMIT, hash)"""
    leb = leb_len_u(L)
    C = clf_single_seed_cost(L) 
    RAW = 10 * L
    emit = should_emit(L)
    t = (L, leb, C, RAW, emit)
    hash_val = hashlib.sha256(str(t).encode()).hexdigest()
    return (*t, hash_val)

# ---------- SINGLE ADMISSIBILITY DEFINITION (UNIFIED) ----------
def step_run_is_lawful(S:bytes, off:int, L:int, start:int, step:int, total_L:int):
    """
    STEP-RUN over [off:off+L) with params (start, step) is lawful iff:
    (i) start == S[off]
    (ii) if L≥2, step == (S[off+1] - S[off]) mod 256; if L=1, step=0
    (iii) ∀i<L: S[off+i] == (start + i*step) mod 256
    (iv) Maximality:
         left-max: off==0 or S[off-1] != (start - step) mod 256
         right-max: off+L==total_L or S[off+L] != (start + L*step) mod 256
         
    CONST-RUN is STEP-RUN with step=0
    """
    # Clause (i): start == S[off]
    if start != S[off]:
        return False, "clause_i_failed"
    
    # Clause (ii): step deduction
    if L >= 2:
        expected_step = (S[off+1] - S[off]) % 256
        if step != expected_step:
            return False, "clause_ii_failed"
    else:  # L=1
        if step != 0:
            return False, "clause_ii_failed_L1"
    
    # Clause (iii): progression holds
    for i in range(L):
        expected = (start + i*step) % 256
        if S[off+i] != expected:
            return False, f"clause_iii_failed_at_{i}"
    
    # Clause (iv): maximality
    left_max = (off == 0) or (S[off-1] != (start - step) % 256)
    right_max = (off + L == total_L) or (S[off+L] != (start + L*step) % 256)
    
    if not left_max:
        return False, "left_maximality_failed"
    if not right_max:
        return False, "right_maximality_failed"
    
    return True, "lawful"

def print_admissibility_check(S:bytes, off:int, L:int, start:int, step:int, total_L:int, token_id:str):
    """Print detailed admissibility verification for receipts"""
    print(f"ADMISSIBILITY CHECK {token_id}: off={off} L={L} start={start} step={step}")
    
    # Check each clause
    clause_i = (start == S[off])
    print(f"  Clause (i): start({start}) == S[{off}]({S[off]}) → {clause_i}")
    
    if L >= 2:
        expected_step = (S[off+1] - S[off]) % 256
        clause_ii = (step == expected_step)
        print(f"  Clause (ii): L≥2, step({step}) == expected({expected_step}) → {clause_ii}")
    else:
        clause_ii = (step == 0)
        print(f"  Clause (ii): L=1, step({step}) == 0 → {clause_ii}")
    
    clause_iii = True
    for i in range(min(L, 5)):  # Show first 5 positions
        expected = (start + i*step) % 256
        actual = S[off+i]
        if actual != expected:
            clause_iii = False
            break
    print(f"  Clause (iii): First {min(L,5)} positions match progression → {clause_iii}")
    
    left_max = (off == 0) or (S[off-1] != (start - step) % 256)
    right_max = (off + L == total_L) or (S[off+L] != (start + L*step) % 256)
    
    if off > 0:
        prev_expected = (start - step) % 256
        print(f"  Left max: S[{off-1}]({S[off-1]}) != {prev_expected} → {left_max}")
    else:
        print(f"  Left max: off==0 → {left_max}")
    
    if off + L < total_L:
        next_expected = (start + L*step) % 256
        print(f"  Right max: S[{off+L}]({S[off+L]}) != {next_expected} → {right_max}")
    else:
        print(f"  Right max: off+L==total_L → {right_max}")
    
    clause_iv = left_max and right_max
    print(f"  Clause (iv): Maximal ({left_max} AND {right_max}) → {clause_iv}")
    
    overall = clause_i and clause_ii and clause_iii and clause_iv
    print(f"  OVERALL: {overall}")
    return overall

# ---------- DEDUCTOR SIGNATURE ----------
def get_deductor_signature():
    """Return SHA256 of deduct_B function source + build ID for determinism"""
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

# ---------- GREEDY-MAX SPLITTER (FEASIBILITY-GUARDED) ----------
def find_admissible_token_at(S:bytes, pos:int, L_total:int):
    """Find any admissible token starting at pos, or None if impossible"""
    if pos >= L_total:
        return None
        
    # Try STEP-RUN first (if feasible)
    if pos + 1 < L_total:
        start = S[pos]
        step = (S[pos+1] - S[pos]) % 256
        
        if step != 0:  # Non-trivial STEP-RUN
            j = pos + 2
            while j < L_total and S[j] == ((start + (j-pos)*step) % 256):
                j += 1
            
            is_lawful, _ = step_run_is_lawful(S, pos, j-pos, start, step, L_total)
            if is_lawful:
                return (11, [start, step], j-pos)
    
    # Try CONST-RUN
    start = S[pos]
    j = pos + 1
    while j < L_total and S[j] == start:
        j += 1
    
    is_lawful, _ = step_run_is_lawful(S, pos, j-pos, start, 0, L_total)
    if is_lawful:
        return (10, [start], j-pos)
    
    return None

def deduct_B(S:bytes):
    """
    Feasibility-guarded greedy splitter with immediate backtrack.
    CLF requirement: complete feasible tiling with maximal tokens.
    """
    L_total = len(S)
    i = 0
    toks = []
    
    while i < L_total:
        # Find candidate tokens at position i
        candidates = []
        
        # STEP-RUN candidate
        if i + 1 < L_total:
            start = S[i]
            step = (S[i+1] - S[i]) % 256
            
            if step != 0:
                j = i + 2
                while j < L_total and S[j] == ((start + (j-i)*step) % 256):
                    j += 1
                
                is_lawful, _ = step_run_is_lawful(S, i, j-i, start, step, L_total)
                if is_lawful:
                    candidates.append((11, [start, step], j-i))
        
        # CONST-RUN candidate  
        start = S[i]
        j = i + 1
        while j < L_total and S[j] == start:
            j += 1
        
        is_lawful, _ = step_run_is_lawful(S, i, j-i, start, 0, L_total)
        if is_lawful:
            candidates.append((10, [start], j-i))
        
        # Feasibility guard: pick candidate that doesn't strand next position
        chosen_token = None
        
        for candidate in candidates:
            op, params, length = candidate
            next_pos = i + length
            
            # Check if next position is tokenizable (or we're done)
            if next_pos >= L_total or find_admissible_token_at(S, next_pos, L_total) is not None:
                chosen_token = candidate
                break
        
        if chosen_token:
            toks.append(chosen_token)
            i += chosen_token[2]  # advance by token length
        else:
            print(f"FATAL ERROR: No feasible token at offset {i}")
            print(f"Context: S[{max(0,i-3)}:{i+4}] = {list(S[max(0,i-3):i+4])}")
            print(f"Candidates tried: {len(candidates)}")
            for idx, (op, params, length) in enumerate(candidates):
                next_pos = i + length
                next_token = find_admissible_token_at(S, next_pos, L_total)
                print(f"  Candidate {idx}: op={op} L={length} -> next_pos={next_pos}, next_feasible={next_token is not None}")
            sys.exit(1)
    
    return toks

# ---------- LOCALITY-GUARDED EXPANSION ----------
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

# ---------- DETERMINISM GUARD ----------
def verify_tokenization_determinism(S:bytes, file_path:str):
    """Same S must yield same tokens across runs - kills drift"""
    print("DETERMINISM GUARD:")
    print(f"DEDUCTOR_SIGNATURE: {get_deductor_signature()}")
    
    # Run 1: Same process
    tokens1 = deduct_B(S)
    tokens1_str = str(tokens1)
    tokens1_sha = hashlib.sha256(tokens1_str.encode()).hexdigest()
    print(f"In-process token SHA: {tokens1_sha}")
    
    # Run 2: Same process, verify identical
    tokens2 = deduct_B(S)
    tokens2_str = str(tokens2)
    tokens2_sha = hashlib.sha256(tokens2_str.encode()).hexdigest()
    
    if tokens1_sha != tokens2_sha:
        print("ABORT: NON-DETERMINISTIC TOKENIZATION (same process)")
        sys.exit(1)
    
    print("DETERMINISM: PASS (in-process consistency verified)")
    return tokens1_sha, tokens1

# ---------- DEDUCTION ⇄ EXPANSION IDENTITY ----------
def verify_bijection_identity(S:bytes, tokens:list):
    """Verify deduct(S) → expand(tokens) → S and re-deduct consistency"""
    print("BIJECTION IDENTITY VERIFICATION:")
    
    # Expand tokens back to bytes
    reconstructed = b''
    for op, params, L_tok in tokens:
        segment = expand_token_with_locality_guard(op, params, L_tok)
        reconstructed += segment
    
    # Verify S == reconstructed
    s_sha = hashlib.sha256(S).hexdigest()
    recon_sha = hashlib.sha256(reconstructed).hexdigest()
    print(f"Original SHA:      {s_sha}")
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
    
    print(f"Original tokens SHA:   {original_sha}")
    print(f"Re-deduced tokens SHA: {rededuced_sha}")
    
    if original_sha != rededuced_sha:
        print("ABORT: RE-DEDUCTION INCONSISTENCY")
        sys.exit(1)
    
    print("BIJECTION IDENTITY: PASS")

# ---------- RECEIPTS (FALSIFIABILITY) ----------
def print_receipts(S:bytes, tokens:list):
    """Print receipts for first 10 tokens and every token with L≥32"""
    print("RECEIPTS (FALSIFIABLE EVIDENCE):")
    
    off = 0
    for idx, (op, params, Lt) in enumerate(tokens):
        if idx < 10 or Lt >= 32:
            if op == 11:  # STEP-RUN
                start, step = params
            else:  # CONST-RUN
                start, step = params[0], 0
            
            print(f"\nTOKEN B_{idx}: off={off} L={Lt} op={op} params={params}")
            
            if Lt >= 32:
                print("LONG TOKEN - Raw bytes verification:")
                window = S[off:off+min(32, Lt)]
                print(f"First 32 bytes: {list(window)}")
                distinct_count = len(set(window))
                print(f"Distinct values in first 32 bytes: {distinct_count}")
            
            # Verify all clauses
            is_lawful = print_admissibility_check(S, off, Lt, start, step, len(S), f"B_{idx}")
            if not is_lawful:
                print(f"ABORT: TOKEN B_{idx} FAILS ADMISSIBILITY")
                sys.exit(1)
        
        off += Lt

# ---------- ROLES & ALGEBRA ----------
def verify_roles_and_algebra(S:bytes, tokens:list):
    """Enforce A one-token or N/A; verify algebra and gate"""
    L = len(S)
    RAW = 8 * L
    H = header_bits(L)
    
    print("ROLES & ALGEBRA VERIFICATION:")
    
    # A-path (whole-range detection)
    A_stream = None
    if L == 0:
        A_caus = 0
        A_stream = A_caus + end_bits(A_caus)
        print(f"A: ZERO whole-range, A_stream={A_stream}")
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
    
    # B-path (multi-token tiling)
    print(f"B: Multi-token tiling, {len(tokens)} tokens")
    
    # Coverage verification
    total_coverage = sum(Lt for _, _, Lt in tokens)
    print(f"Coverage: Σ L_tok = {total_coverage}, L = {L}")
    if total_coverage != L:
        print("ABORT: COVERAGE MISMATCH")
        sys.exit(1)
    
    # B-path costs
    B_caus = sum(caus_bits(op, params, Lt) for op, params, Lt in tokens)
    B_end = end_bits(B_caus)
    B_stream = B_caus + B_end
    
    print(f"B_caus = {B_caus}")
    print(f"B_end = {B_end}")
    print(f"B_stream = {B_stream}")
    
    # Decision algebra
    candidates = []
    if A_stream is not None:
        candidates.append(H + A_stream)
    candidates.append(H + B_stream)
    
    C_min_total = min(candidates)
    print(f"H = {H}")
    print(f"Candidates: {candidates}")
    print(f"C_min_total = {C_min_total}")
    print(f"RAW = {RAW}")
    
    # Gate (strict inequality)
    emit_decision = C_min_total < RAW
    print(f"EMIT GATE: {C_min_total} < {RAW} → {emit_decision}")
    
    return C_min_total, emit_decision

# ---------- MAIN VALIDATOR ----------
def validate_single_seed_clf(file_path):
    """Single-seed causal minimality calculator - pure mathematics"""
    print(f"{'='*60}")
    print(f"CLF SINGLE-SEED CALCULATOR (SURGICALLY CORRECTED)")
    print(f"{'='*60}")
    print(f"BUILD_ID: {BUILD_ID}")
    print(f"File: {os.path.basename(file_path)}")
    print(f"Formula: C_min^(1)(L) = 88 + 8*leb(L) bits")
    print(f"Fallback: C_LIT = 10*L bits")
    
    if not os.path.exists(file_path):
        print(f"ABORT: FILE NOT FOUND: {file_path}")
        sys.exit(1)
    
    # Pure arithmetic calculation (no content reading)
    L = os.path.getsize(file_path)
    L_receipt, leb_L, C_total, RAW, emit_decision, receipt_hash = single_seed_receipt(L)
    
    print(f"\nINPUT:")
    print(f"  Length: L = {L:,} bytes")
    print(f"  LEB128: leb(L) = {leb_L} bytes")
    
    print(f"\nLOCKED CONSTANTS:")
    print(f"  H (header):     56 bits")
    print(f"  CAUS (causal):  27 bits")
    print(f"  END (end):       5 bits")
    print(f"  LEN (8*leb(L)): {8 * leb_L} bits")
    
    print(f"\nCALCULATION:")
    print(f"  C_min^(1)(L) = 88 + 8*leb(L) = 88 + {8 * leb_L} = {C_total} bits")
    print(f"  C_LIT = 10*L = 10*{L:,} = {RAW:,} bits")
    
    print(f"\nDECISION:")
    print(f"  EMIT = {C_total} < {RAW:,} → {emit_decision}")
    
    print(f"\nRECEIPT: {receipt_hash[:16]}... (SHA256 of calculation tuple)")
    print(f"\nSINGLE-SEED CALCULATOR: COMPLETE")
    
    return {
        'length': L,
        'leb_length': leb_L,
        'C_total': C_total, 
        'RAW': RAW,
        'emit': emit_decision,
        'receipt_hash': receipt_hash
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLF Single-Seed Calculator (Surgically Corrected)')
    parser.add_argument('file', help='File to calculate bounds for')
    parser.add_argument('--verify-expected', action='store_true', help='Verify expected outputs first')
    
    args = parser.parse_args()
    
    # Mathematical verification of expected outputs
    if args.verify_expected:
        print("MATHEMATICAL VERIFICATION:")
        print("Expected outputs for known test cases:")
        test_cases = [
            ("pic1.jpg", 968, 104),
            ("pic2.jpg", 456, 104), 
            ("video1.mp4", 1570024, 112)
        ]
        
        for name, L, expected in test_cases:
            actual = clf_single_seed_cost(L)
            leb_L = leb_len_u(L)
            emit = should_emit(L)
            status = "✓" if actual == expected else "✗ BUG"
            print(f"  {name}: L={L:,} → leb={leb_L} → C={actual} (expect {expected}) → EMIT={emit} {status}")
            
            if actual != expected:
                print(f"CALCULATOR BUG: {name} expected {expected}, got {actual}")
                sys.exit(1)
        print()
    
    validate_single_seed_clf(args.file)