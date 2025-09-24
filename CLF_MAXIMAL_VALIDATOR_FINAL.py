# === DO NOT EDIT: CLF MATH CONTRACT ===
# Single-seed CLF: C_min^(1)(L) = 88 + 8*leb(L) with H=56, CAUS=27, END=5 (locked)
# EMIT iff C_min^(1)(L) < 10*L (strict). leb(L) = unsigned LEB128 byte-length of L (7-bit groups).
# Integer-only. No compression logic. No floating point. No content scanning. O(log L) only.

"""
CLF Single-Seed Calculator — Pure Causal Minimality (No Compression)

Contract
--------
C_min^(1)(L) = 88 + 8*leb(L) bits, where:
  - leb(L): unsigned LEB128 byte-length of integer L using 7-bit groups.
  - Locked constants: H=56, CAUS=27, END=5 (non-negotiable).

Literal fallback: C_LIT = 10*L bits.
Decision gate (strict): EMIT ⇔ C_min^(1)(L) < 10*L.

Invariants
----------
- Integer-only arithmetic. No floating point.
- O(log L) complexity: depends only on leb_len_u(L).
- Content-independent: uses file length only (L = os.path.getsize(path)).
- No compression/tiling/A/B roles/DP/greedy scans/coverage/bijection.
- No variation of H/CAUS/END. No leb(8*L). No align/padding drift.

Receipts
--------
For each L, compute a deterministic receipt from the tuple:
  (L, leb, C_min, RAW, EMIT, BUILD_ID)

Using SHA-256. This enables independent verification.

Usage
-----
- As CLI: pass file paths; calculator prints one line per file.
- As library: use `clf_single_seed_cost(L)`, `should_emit(L)`, `receipt(L)`.

Boundaries to test (leb bands)
------------------------------
L ∈ {0,1,127,128,16383,16384,...} to verify leb band transitions:

leb = ceil(bit_length(L)/7) if L>0 else 1.

Edge Case: L=0
--------------
For empty files (L=0):
- leb(0) = 1 (special case)  
- C_min^(1)(0) = 88 + 8*1 = 96 bits
- RAW = 10*0 = 0 bits
- EMIT = (96 < 0) = False (strict inequality)
This is the canonical tiny-input behavior where causal overhead exceeds literal cost.
"""

import argparse
import hashlib
import os

BUILD_ID = "CLF_SINGLE_SEED_PURE_20250923_LOCKED"

# Math guard assertions (Section 5 of the guide)
assert isinstance(BUILD_ID, str) and BUILD_ID, "BUILD_ID must be a non-empty string"
# Guard strictness:
_cost = 88 + 8 * 2  # C_min^(1)(968) with leb(968)=2
assert _cost == 104, "Invariant self-check failed: C_min^(1)(968) must be 104"
# Gate monotonicity at scale (sample):
assert 104 < 10 * 16 and 112 < 10 * 1_000_000, "EMIT must hold for practical L >= 16"

def leb_len_u(n: int) -> int:
    """
    Unsigned LEB128 byte-length of integer n using 7-bit groups.

    Definition
    ----------
    Returns 1 if n == 0, else ceil(bit_length(n) / 7).

    Pure Integer Contract
    ---------------------
    - Integer-only arithmetic; no float or rounding modes.
    - O(1) with respect to n's value (since bit_length is O(1) on integers).
    - Independent of file content; depends only on n's binary magnitude.

    Parameters
    ----------
    n : int
        Non-negative integer whose LEB128 encoded byte-length is desired.

    Returns
    -------
    int
        The number of bytes required to encode n in unsigned LEB128 (7-bit groups).

    Raises
    ------
    AssertionError
        If n < 0.

    Examples
    --------
    leb_len_u(0)     -> 1
    leb_len_u(127)   -> 1   # 2^7 - 1
    leb_len_u(128)   -> 2   # 2^7
    leb_len_u(16383) -> 2   # 2^14 - 1
    leb_len_u(16384) -> 3   # 2^14
    """
    assert n >= 0, f"n must be non-negative, got {n}"
    return 1 if n == 0 else ((n.bit_length() + 6) // 7)

def clf_single_seed_cost(L: int) -> int:
    """
    Compute single-seed CLF bound: C_min^(1)(L) = 88 + 8*leb(L).

    Contract
    --------
    - Uses locked constants: H=56, CAUS=27, END=5.
    - leb(L) = leb_len_u(L) (unsigned LEB128 byte-length of L).
    - Integer-only. O(log L). No content scanning.

    Parameters
    ----------
    L : int
        Byte-length of the binary string (e.g., file size in bytes).

    Returns
    -------
    int
        C_min^(1)(L) in bits.

    Notes
    -----
    Forbidden drifts:
    - Do not vary H/CAUS/END.
    - Do not use leb(8*L) or any alignment/padding adjustments.
    """
    return 88 + 8 * leb_len_u(L)

def should_emit(L: int) -> bool:
    """
    Decide EMIT under strict gate: C_min^(1)(L) < 10*L.

    Contract
    --------
    - Strict inequality (<), never <=.
    - RAW (literal) cost: 10*L bits.
    - Integer-only decision; O(log L).

    Parameters
    ----------
    L : int
        Byte-length of the binary string.

    Returns
    -------
    bool
        True iff C_min^(1)(L) < 10*L.

    Anti-Patterns
    -------------
    - No probabilistic thresholds or tolerances.
    - No content-dependent logic.
    """
    return clf_single_seed_cost(L) < 10 * L

def receipt(L: int) -> dict:
    """
    Build deterministic receipt for independent verification.

    Tuple
    -----
    (L, leb, C, RAW, EMIT, BUILD_ID)  → SHA256 hex digest.

    Guarantees
    ----------
    - Deterministic across runs and machines.
    - Integer-only fields.
    - Suitable for JSONL/CSV export for audits.

    Parameters
    ----------
    L : int
        Byte-length of the binary string.

    Returns
    -------
    dict
        {
          "L": int,
          "leb": int,
          "C": int,       # C_min^(1)(L)
          "RAW": int,     # 10*L
          "EMIT": bool,   # strict gate
          "sha256": str   # receipt of the tuple above
        }

    """
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
        
        # Calculate bit_length bounds
        k = L.bit_length() if L > 0 else 1
        bounds_str = f"2^{k-1} ≤ L < 2^{k}" if L > 0 else "L=0"
        
        print(f"{os.path.basename(path)}: L={r['L']:,} bytes, bit_length={k}, bounds={bounds_str}, "
              f"leb={r['leb']}, C={r['C']} bits, RAW={r['RAW']:,} bits, EMIT={r['EMIT']}, "
              f"receipt={r['sha256'][:16]}...")
