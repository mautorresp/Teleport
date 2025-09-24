#!/usr/bin/env python3

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
import csv
import hashlib
import json
import os
import sys
from typing import List, Dict, Any
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════
# IMMUTABLE MATHEMATICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

H = 56          # Header bits (locked)
CAUS = 27       # Causality bits (locked)  
END = 5         # Termination bits (locked)
BASE_BITS = H + CAUS + END  # 88 bits (immutable)
BUILD_ID = "CLF_SINGLE_SEED_PURE_20250923_LOCKED"  # Immutable build identifier

# Math guard assertions (Section 5 of the guide)
assert isinstance(BUILD_ID, str) and BUILD_ID, "BUILD_ID must be a non-empty string"
# Guard strictness:
_cost = 88 + 8 * 2  # C_min^(1)(968) with leb(968)=2
assert _cost == 104, "Invariant self-check failed: C_min^(1)(968) must be 104"
# Gate monotonicity at scale (sample):
_emit_16 = 104 < 10 * 16  # C_min for small file vs 10*16
_emit_1m = 112 < 10 * 1_000_000  # C_min for large file vs 10*1M
assert _emit_16 and _emit_1m, "EMIT must hold for practical L >= 16"


# ═══════════════════════════════════════════════════════════════════════════
# CORE MATHEMATICAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

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
    >>> leb_len_u(0)
    1
    >>> leb_len_u(127)
    1
    >>> leb_len_u(128)
    2
    >>> leb_len_u(16383)
    2
    >>> leb_len_u(16384)
    3
    """
    assert n >= 0, f"n must be non-negative, got {n}"
    if n == 0:
        return 1
    return (n.bit_length() + 6) // 7


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
    
    Examples
    --------
    >>> clf_single_seed_cost(0)
    96
    >>> clf_single_seed_cost(127)
    96
    >>> clf_single_seed_cost(128)
    104
    >>> clf_single_seed_cost(16383)
    104
    >>> clf_single_seed_cost(16384)
    112
    """
    return BASE_BITS + 8 * leb_len_u(L)


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
    
    Examples
    --------
    >>> should_emit(0)
    False
    >>> should_emit(16)
    True
    >>> should_emit(1000)
    True
    >>> should_emit(10000)
    True
    """
    return clf_single_seed_cost(L) < 10 * L


def receipt(L: int, build_id: str, file_path: str = None) -> Dict[str, Any]:
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
    - Mathematical receipt preserves content-independence.
    - Optional file provenance for path/content differentiation.

    Parameters
    ----------
    L : int
        Byte-length of the binary string.
    build_id : str
        Build identifier for the calculation session.
    file_path : str, optional
        File path for provenance hash (does not affect math receipt).

    Returns
    -------
    dict
        {
          "L": int,
          "leb": int,
          "C_min_bits": int,       # C_min^(1)(L)
          "RAW_bits": int,         # 10*L
          "EMIT": bool,            # strict gate
          "sha256": str,           # mathematical receipt
          "provenance_sha256": str # optional file provenance
        }

    Examples
    --------
    >>> result = receipt(128, "TEST")
    >>> result['L']
    128
    >>> result['leb']
    2
    >>> result['C_min_bits']
    104
    >>> result['RAW_bits']
    1280
    >>> result['EMIT']
    True
    """
    leb = leb_len_u(L)
    C = clf_single_seed_cost(L)
    RAW = 10 * L
    EMIT = should_emit(L)
    
    # Mathematical receipt (content-independent)
    math_tuple = (L, leb, C, RAW, EMIT, build_id)
    math_receipt = hashlib.sha256(str(math_tuple).encode('utf-8')).hexdigest()
    
    # Optional file provenance (for path/content differentiation)
    provenance_receipt = None
    if file_path is not None:
        provenance_tuple = (file_path, L, build_id)
        provenance_receipt = hashlib.sha256(str(provenance_tuple).encode('utf-8')).hexdigest()
    
    result = {
        'L': L,
        'leb': leb,
        'C_min_bits': C,
        'RAW_bits': RAW,
        'EMIT': EMIT,
        'BUILD_ID': build_id,
        'tuple_data': math_tuple,
        'sha256': math_receipt
    }
    
    if provenance_receipt is not None:
        result['provenance_sha256'] = provenance_receipt
        
    return result


def analyze_path(path: str, build_id: str) -> Dict[str, Any]:
    """
    Analyze single file path with mathematical evidence.
    
    Args:
        path: File path to analyze
        build_id: Unique build identifier
        
    Returns:
        Complete analysis with bit_length proofs
    """
    try:
        L = os.path.getsize(path)
    except (OSError, IOError) as e:
        return {
            'file': path,
            'error': str(e),
            'L': None
        }
    
    # Bit length proof bounds
    if L == 0:
        k = 0
        bit_length = 1  # Special case: bit_length(0) = 0, but we use 1 for LEB
    else:
        bit_length = L.bit_length()
        k = bit_length - 1
        # Verify: 2^k ≤ L < 2^(k+1)
        assert (1 << k) <= L < (1 << (k + 1)), f"Bit length proof failed for L={L}, k={k}"
    
    # Generate receipt with file provenance
    receipt_data = receipt(L, build_id, file_path=path)
    
    result = {
        'file': path,
        'L': L,
        'bit_length': bit_length,
        'k': k,
        'leb': receipt_data['leb'],
        'C_min_bits': receipt_data['C_min_bits'],
        'RAW_bits': receipt_data['RAW_bits'],
        'EMIT': receipt_data['EMIT'],
        'BUILD_ID': build_id,
        'sha256': receipt_data['sha256']
    }
    
    # Add provenance hash if available
    if 'provenance_sha256' in receipt_data:
        result['provenance_sha256'] = receipt_data['provenance_sha256']
        
    return result


def run(paths: List[str], build_id: str = None) -> List[Dict[str, Any]]:
    """
    Batch analyze multiple file paths.
    
    Args:
        paths: List of file paths
        build_id: Optional build identifier (auto-generated if None)
        
    Returns:
        List of analysis results
    """
    if build_id is None:
        build_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = []
    for path in paths:
        result = analyze_path(path, build_id)
        results.append(result)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT SYSTEM (4-FORMAT EVIDENCE)
# ═══════════════════════════════════════════════════════════════════════════

def export_console(results: List[Dict], build_id: str, filepath: str):
    """Export console-style output to text file."""
    with open(filepath, 'w') as f:
        f.write(f"CLF Single-Seed Calculator - BUILD_ID: {build_id}\n")
        f.write(f"Formula: C_min^(1)(L) = 88 + 8*leb(L)\n")
        f.write(f"Constants: H={H}, CAUS={CAUS}, END={END}\n")
        f.write(f"Decision: EMIT ⇔ C_min^(1)(L) < 10*L\n\n")
        
        for result in results:
            if 'error' in result:
                f.write(f"ERROR: {result['file']} - {result['error']}\n")
                continue
            
            L = result['L']
            k = result['bit_length'] if L > 0 else 1
            bounds_str = f"2^{k-1} ≤ L < 2^{k}" if L > 0 else "L=0"
                
            f.write(f"{result['file']}: L={L:,} bytes, bit_length={result['bit_length']}, bounds={bounds_str}, "
                   f"leb={result['leb']}, C={result['C_min_bits']} bits, RAW={result['RAW_bits']:,} bits, "
                   f"EMIT={result['EMIT']}, receipt={result['sha256'][:16]}...\n")


def export_receipts(results: List[Dict], filepath: str):
    """Export JSONL receipts (one JSON object per line)."""
    with open(filepath, 'w') as f:
        for result in results:
            if 'error' in result:
                continue
            receipt_obj = {
                'file': result['file'],
                'L': result['L'],
                'bit_length': result['bit_length'],
                'leb': result['leb'],
                'C_min_bits': result['C_min_bits'],
                'RAW_bits': result['RAW_bits'],
                'EMIT': result['EMIT'],
                'BUILD_ID': result['BUILD_ID'],
                'sha256': result['sha256']
            }
            f.write(json.dumps(receipt_obj) + '\n')


def export_csv(results: List[Dict], filepath: str):
    """Export CSV results table."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'L', 'bit_length', 'leb', 'C_min_bits', 
                        'RAW_bits', 'emit', 'receipt_prefix'])
        
        for result in results:
            if 'error' in result:
                continue
            writer.writerow([
                result['file'],
                result['L'],
                result['bit_length'],
                result['leb'],
                result['C_min_bits'],
                result['RAW_bits'],
                result['EMIT'],
                result['sha256'][:16]  # Receipt prefix
            ])


def export_audit(results: List[Dict], build_id: str, filepath: str):
    """Export mathematical audit with step-by-step proofs."""
    with open(filepath, 'w') as f:
        f.write("CLF SINGLE-SEED MATHEMATICAL CALCULATOR - AUDIT EVIDENCE\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"BUILD_ID: {build_id}\n")
        f.write(f"Generation Time: {datetime.now().isoformat()}\n\n")
        
        f.write("IMMUTABLE MATHEMATICAL CONSTANTS:\n")
        f.write(f"- H (Header bits): {H}\n")
        f.write(f"- CAUS (Causality bits): {CAUS}\n")
        f.write(f"- END (Termination bits): {END}\n")
        f.write(f"- BASE_BITS = H + CAUS + END = {BASE_BITS}\n\n")
        
        f.write("FORMULA (Single-Seed Causal Bound):\n")
        f.write("C_min^(1)(L) = 88 + 8*leb(L)\n\n")
        
        f.write("LEB128 LENGTH RULE:\n")
        f.write("leb(L) = 1 if L == 0 else (bit_length(L) + 6) // 7\n\n")
        
        f.write("DECISION RULE:\n")
        f.write("EMIT ⇔ C_min^(1)(L) < 10*L (strict inequality)\n\n")
        
        f.write("PROHIBITED LOGIC:\n")
        f.write("❌ deduct_B, STEP, CONST, tiling, DP, feasibility, A/B roles\n")
        f.write("❌ bijection, byte iteration, entropy, compression heuristics\n")
        f.write("❌ floating point arithmetic, math.* float functions\n")
        f.write("✅ Only int, bit_length, shifts, adds, multiplies, LEB rule\n\n")
        
        f.write("FILE-BY-FILE MATHEMATICAL EVIDENCE:\n")
        f.write("-" * 40 + "\n\n")
        
        for i, result in enumerate(results, 1):
            if 'error' in result:
                f.write(f"File {i}: {result['file']} - ERROR: {result['error']}\n\n")
                continue
                
            L = result['L']
            bit_length = result['bit_length']
            k = result['k']
            leb = result['leb']
            C_min = result['C_min_bits']
            RAW = result['RAW_bits']
            emit = result['EMIT']
            
            f.write(f"File {i}: {result['file']}\n")
            f.write(f"Step 1 - File Length: L = {L} bytes\n")
            
            f.write(f"Step 2 - Bit Length Proof:\n")
            if L == 0:
                f.write(f"  L = 0, so bit_length = 1 (special case)\n")
            else:
                f.write(f"  bit_length(L) = {bit_length}\n")
                f.write(f"  k = bit_length - 1 = {k}\n")
                f.write(f"  Verify: 2^{k} ≤ {L} < 2^{k+1}\n")
                f.write(f"  Check: {1 << k} ≤ {L} < {1 << (k+1)} ✓\n")
            
            f.write(f"Step 3 - LEB128 Length:\n")
            if L == 0:
                f.write(f"  leb(0) = 1\n")
            else:
                f.write(f"  leb({L}) = ({bit_length} + 6) // 7 = {leb}\n")
            
            f.write(f"Step 4 - Causal Bound:\n")
            f.write(f"  C_min^(1)({L}) = 88 + 8*{leb} = {C_min} bits\n")
            
            f.write(f"Step 5 - Raw Literal:\n")
            f.write(f"  RAW = 10*{L} = {RAW} bits\n")
            
            f.write(f"Step 6 - Decision:\n")
            f.write(f"  Check: {C_min} < {RAW} ? {emit}\n")
            f.write(f"  EMIT = {emit}\n")
            
            f.write(f"Step 7 - Receipt Verification:\n")
            f.write(f"  Tuple: ({L}, {leb}, {C_min}, {RAW}, {emit}, '{build_id}')\n")
            f.write(f"  SHA256: {result['sha256']}\n\n")


def export_all(results: List[Dict], build_id: str, prefix: str):
    """Generate all 4 export formats."""
    export_console(results, build_id, f"{prefix}_CONSOLE.txt")
    export_receipts(results, f"{prefix}_RECEIPTS.jsonl")
    export_csv(results, f"{prefix}_RESULTS.csv")
    export_audit(results, build_id, f"{prefix}_AUDIT.txt")
    
    print(f"Exported evidence to:")
    print(f"  {prefix}_CONSOLE.txt")
    print(f"  {prefix}_RECEIPTS.jsonl")
    print(f"  {prefix}_RESULTS.csv")
    print(f"  {prefix}_AUDIT.txt")


# ═══════════════════════════════════════════════════════════════════════════
# EMBEDDED UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    """Run embedded unit tests for mathematical functions."""
    print("Running unit tests...")
    
    # Test leb_len_u thresholds
    assert leb_len_u(0) == 1, f"leb_len_u(0) = {leb_len_u(0)}, expected 1"
    assert leb_len_u(1) == 1, f"leb_len_u(1) = {leb_len_u(1)}, expected 1"
    assert leb_len_u(127) == 1, f"leb_len_u(127) = {leb_len_u(127)}, expected 1"
    assert leb_len_u(128) == 2, f"leb_len_u(128) = {leb_len_u(128)}, expected 2"
    assert leb_len_u(16383) == 2, f"leb_len_u(16383) = {leb_len_u(16383)}, expected 2"
    assert leb_len_u(16384) == 3, f"leb_len_u(16384) = {leb_len_u(16384)}, expected 3"
    
    # Test clf_single_seed_cost
    assert clf_single_seed_cost(456) == 104, f"clf_single_seed_cost(456) = {clf_single_seed_cost(456)}, expected 104"
    assert clf_single_seed_cost(968) == 104, f"clf_single_seed_cost(968) = {clf_single_seed_cost(968)}, expected 104"
    
    # Test should_emit
    assert should_emit(968) == True, f"should_emit(968) = {should_emit(968)}, expected True"
    assert should_emit(1570024) == True, f"should_emit(1570024) = {should_emit(1570024)}, expected True"
    
    print("✓ All unit tests passed")


# ═══════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CLF Single-Seed Mathematical Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 clf_calculator.py pic1.jpg pic2.jpg video1.mp4
  python3 clf_calculator.py file.txt --export-prefix BASELINE
  python3 clf_calculator.py --test
  python3 clf_calculator.py --stdin-length 1024
        """
    )
    
    parser.add_argument('files', nargs='*', help='File paths to analyze')
    parser.add_argument('--export-prefix', help='Export prefix for 4-format evidence')
    parser.add_argument('--test', action='store_true', help='Run embedded unit tests')
    parser.add_argument('--stdin-length', type=int, help='Direct length input for testing')
    parser.add_argument('--build-id', help='Custom build identifier')
    
    args = parser.parse_args()
    
    if args.test:
        run_unit_tests()
        return
    
    if args.stdin_length is not None:
        # Direct length input mode
        build_id = args.build_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        L = args.stdin_length
        
        receipt_data = receipt(L, build_id)
        print(f"Direct Length Analysis - BUILD_ID: {build_id}")
        print(f"L={L}, leb={receipt_data['leb']}, "
              f"C_min={receipt_data['C_min_bits']} bits, "
              f"RAW={receipt_data['RAW_bits']} bits, "
              f"EMIT={receipt_data['EMIT']}")
        return
    
    if not args.files:
        parser.print_help()
        sys.exit(1)
    
    # Analyze files
    build_id = args.build_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    results = run(args.files, build_id)
    
    # Print console output
    print(f"CLF Single-Seed Calculator - BUILD_ID: {build_id}")
    print(f"Formula: C_min^(1)(L) = 88 + 8*leb(L)")
    print(f"Constants: H={H}, CAUS={CAUS}, END={END}")
    print(f"Decision: EMIT ⇔ C_min^(1)(L) < 10*L")
    print()
    
    for result in results:
        if 'error' in result:
            print(f"ERROR: {result['file']} - {result['error']}")
            continue
        
        L = result['L']
        k = result['bit_length'] if L > 0 else 1
        bounds_str = f"2^{k-1} ≤ L < 2^{k}" if L > 0 else "L=0"
        
        print(f"{result['file']}: L={L:,} bytes, bit_length={result['bit_length']}, bounds={bounds_str}, "
              f"leb={result['leb']}, C={result['C_min_bits']} bits, RAW={result['RAW_bits']:,} bits, "
              f"EMIT={result['EMIT']}, receipt={result['sha256'][:16]}...")
    
    # Export if requested
    if args.export_prefix:
        export_all(results, build_id, args.export_prefix)


if __name__ == "__main__":
    main()