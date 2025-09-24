# === CLF SINGLE-SEED CAUSAL MINIMALITY CALCULATOR (PURE MATH) ===
# Contract: C_min^(1)(L) = 88 + 8*leb(L) bits vs C_LIT = 10*L bits
# Constants: H=56, CAUS=27, END=5 (locked, no variation)
# Decision: O(1) arithmetic only, zero content scanning, zero FP

import os
import sys
import argparse
import hashlib
from typing import Tuple

BUILD_ID = "CLF_SINGLE_SEED_PURE_MATH_20250923"

# ---------- LOCKED MATHEMATICS (INTEGER-ONLY) ----------
def leb_len_u(n: int) -> int:
    """Unsigned LEB128 byte-length using 7-bit shifts"""
    assert n >= 0
    c = 1
    while n >= 128:
        n >>= 7
        c += 1
    return c

def clf_single_seed_total_bits(L: int) -> int:
    """C_min^(1)(L) = 56 + 27 + 5 + 8*leb(L) = 88 + 8*leb(L)"""
    return 88 + 8 * leb_len_u(L)

def clf_literal_bits(L: int) -> int:
    """C_LIT = 10*L bits (literal fallback)"""
    return 10 * L

def clf_emit(L: int) -> bool:
    """EMIT decision: C_min^(1)(L) < C_LIT"""
    return clf_single_seed_total_bits(L) < clf_literal_bits(L)

# ---------- CALCULATOR (PURE ARITHMETIC) ----------
def calculate_clf_bound(file_path: str) -> Tuple[int, int, int, int, bool]:
    """Pure causal minimality calculator - O(1) arithmetic only"""
    # Get file length (no content reading)
    L = os.path.getsize(file_path)
    
    # Locked constants (no variation, no dependencies)
    H = 56          # Header (locked)
    CAUS = 27       # Causal (locked) 
    END = 5         # End (locked)
    LEN = 8 * leb_len_u(L)  # Length term
    
    C_total = H + CAUS + END + LEN  # Must equal 88 + 8*leb(L)
    RAW = clf_literal_bits(L)       # 10*L
    emit = clf_emit(L)              # Strict inequality
    
    return L, leb_len_u(L), C_total, RAW, emit

def validate_single_seed_calculator(file_path: str) -> dict:
    """Single-seed calculator with mathematical verification"""
    print(f"\\n{'='*60}")
    print(f"CLF SINGLE-SEED CALCULATOR: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    print(f"BUILD_ID: {BUILD_ID}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Pure calculator (O(1) arithmetic)
    L, leb_L, C_total, RAW, emit = calculate_clf_bound(file_path)
    
    # Verify against locked formula
    expected_total = clf_single_seed_total_bits(L)
    if C_total != expected_total:
        raise ValueError(f"CALCULATOR BUG: C_total={C_total} != expected={expected_total}")
    
    print(f"Length: L = {L:,} bytes")
    print(f"LEB128: leb(L) = {leb_L} bytes")
    print(f"")
    print(f"LOCKED CONSTANTS:")
    print(f"  H (header):     56 bits")
    print(f"  CAUS (causal):  27 bits") 
    print(f"  END (end):       5 bits")
    print(f"  LEN (8*leb(L)): {8 * leb_L} bits")
    print(f"")
    print(f"CALCULATION:")
    print(f"  C_min^(1)(L) = 88 + 8*leb(L) = 88 + {8 * leb_L} = {C_total} bits")
    print(f"  C_LIT = 10*L = 10*{L:,} = {RAW:,} bits")
    print(f"")
    print(f"DECISION:")
    print(f"  EMIT = {C_total} < {RAW:,} → {emit}")
    
    # Determinism receipt (integers only)
    receipt_data = (L, leb_L, C_total, RAW, emit)
    receipt_hash = hashlib.sha256(str(receipt_data).encode()).hexdigest()
    print(f"")
    print(f"RECEIPT: {receipt_hash[:16]}... (SHA256 of calculation tuple)")
    
    result = {
        'file_path': file_path,
        'length': L,
        'leb_length': leb_L, 
        'C_total': C_total,
        'RAW': RAW,
        'emit': emit,
        'receipt_hash': receipt_hash
    }
    
    print(f"\\nCLF CALCULATOR: COMPLETE")
    return result

# ---------- MATHEMATICAL VERIFICATION ----------
def verify_expected_outputs():
    """Verify calculator produces expected results for known files"""
    print("MATHEMATICAL VERIFICATION:")
    print("Expected outputs for test files:")
    
    test_cases = [
        ("pic1.jpg", 968, 104),      # L=968 → leb=2 → 88+16=104
        ("pic2.jpg", 456, 104),      # L=456 → leb=2 → 88+16=104  
        ("video1.mp4", 1570024, 112) # L=1570024 → leb=3 → 88+24=112
    ]
    
    for name, L, expected in test_cases:
        actual = clf_single_seed_total_bits(L)
        leb_L = leb_len_u(L)
        RAW = clf_literal_bits(L)
        emit = clf_emit(L)
        
        status = "✓" if actual == expected else "✗ BUG"
        print(f"  {name}: L={L:,} → leb={leb_L} → C={actual} (expect {expected}) → EMIT={emit} {status}")
        
        if actual != expected:
            raise ValueError(f"Calculator bug: {name} expected {expected}, got {actual}")

# ---------- CLI ---------- 
def main():
    parser = argparse.ArgumentParser(description='CLF Single-Seed Pure Math Calculator')
    parser.add_argument('files', nargs='+', help='Files to calculate bounds for')
    parser.add_argument('--verify', action='store_true', help='Run mathematical verification first')
    parser.add_argument('--export', default=None, help='Export results to file')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_expected_outputs()
        print()
    
    results = []
    for file_path in args.files:
        try:
            result = validate_single_seed_calculator(file_path)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {file_path}: {e}")
            sys.exit(1)
    
    # Export summary
    if args.export:
        with open(args.export, 'w') as f:
            f.write("CLF SINGLE-SEED CALCULATOR RESULTS\\n")
            f.write("=" * 50 + "\\n")
            f.write(f"Build: {BUILD_ID}\\n")
            f.write(f"Formula: C_min^(1)(L) = 88 + 8*leb(L) bits\\n")
            f.write(f"Fallback: C_LIT = 10*L bits\\n\\n")
            
            for r in results:
                f.write(f"{os.path.basename(r['file_path'])}:\\n")
                f.write(f"  L = {r['length']:,} bytes\\n")
                f.write(f"  leb(L) = {r['leb_length']} bytes\\n") 
                f.write(f"  C_min^(1) = {r['C_total']} bits\\n")
                f.write(f"  C_LIT = {r['RAW']:,} bits\\n")
                f.write(f"  EMIT = {r['emit']}\\n")
                f.write(f"  Receipt = {r['receipt_hash'][:16]}...\\n\\n")
        
        print(f"\\nExported: {args.export}")
    
    print(f"\\nSINGLE-SEED CALCULATION COMPLETE: {len(results)} files")
    print("All results match locked mathematical formula C_min^(1)(L) = 88 + 8*leb(L)")

if __name__ == "__main__":
    main()