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
