#!/usr/bin/env python3
"""
CLF Single-Seed Mathematical Calculator

Pure O(log L) causal minimality calculator with immutable mathematical rails.
Formula: C_min^(1)(L) = 88 + 8*leb(L)
Constants: H=56, CAUS=27, END=5 (locked)
Decision: EMIT ⇔ C_min^(1)(L) < 10*L (strict inequality)

No compression, tiling, DP, byte scanning, or floating point arithmetic.
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


# ═══════════════════════════════════════════════════════════════════════════
# CORE MATHEMATICAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def leb_len_u(n: int) -> int:
    """
    Compute unsigned LEB128 byte length for integer n.
    
    Pure integer formula:
    - leb(0) = 1
    - leb(n) = (bit_length(n) + 6) // 7 for n > 0
    
    Args:
        n: Non-negative integer
        
    Returns:
        LEB128 byte length (always >= 1)
    """
    if n == 0:
        return 1
    return (n.bit_length() + 6) // 7


def clf_cost(L: int) -> int:
    """
    Compute single-seed causal minimality bound.
    
    Formula: C_min^(1)(L) = 88 + 8*leb(L)
    
    Args:
        L: File length in bytes
        
    Returns:
        Minimum causal bound in bits
    """
    return BASE_BITS + 8 * leb_len_u(L)


def should_emit(L: int) -> bool:
    """
    Determine if causal bound beats raw literal.
    
    Decision rule: EMIT ⇔ C_min^(1)(L) < 10*L (strict inequality)
    
    Args:
        L: File length in bytes
        
    Returns:
        True if causal bound is beneficial
    """
    return clf_cost(L) < 10 * L


def receipt(L: int, build_id: str) -> Dict[str, Any]:
    """
    Generate deterministic calculation receipt.
    
    Args:
        L: File length in bytes
        build_id: Unique build identifier
        
    Returns:
        Dictionary with calculation details and SHA256 verification
    """
    leb = leb_len_u(L)
    C = clf_cost(L)
    RAW = 10 * L
    EMIT = should_emit(L)
    
    # Create tuple for hash verification
    tuple_data = (L, leb, C, RAW, EMIT, build_id)
    tuple_str = str(tuple_data)
    sha256_hash = hashlib.sha256(tuple_str.encode('utf-8')).hexdigest()
    
    return {
        'L': L,
        'leb': leb,
        'C_min_bits': C,
        'RAW_bits': RAW,
        'EMIT': EMIT,
        'BUILD_ID': build_id,
        'tuple_data': tuple_data,
        'sha256': sha256_hash
    }


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
    
    # Generate receipt
    receipt_data = receipt(L, build_id)
    
    return {
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
                
            f.write(f"{result['file']}: L={result['L']}, "
                   f"C_min={result['C_min_bits']} bits, "
                   f"RAW={result['RAW_bits']} bits, "
                   f"EMIT={result['EMIT']}\n")


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
    
    # Test clf_cost
    assert clf_cost(456) == 104, f"clf_cost(456) = {clf_cost(456)}, expected 104"
    assert clf_cost(968) == 104, f"clf_cost(968) = {clf_cost(968)}, expected 104"
    
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
            
        print(f"{result['file']}: L={result['L']}, "
              f"C_min={result['C_min_bits']} bits, "
              f"RAW={result['RAW_bits']} bits, "
              f"EMIT={result['EMIT']}")
    
    # Export if requested
    if args.export_prefix:
        export_all(results, build_id, args.export_prefix)


if __name__ == "__main__":
    main()