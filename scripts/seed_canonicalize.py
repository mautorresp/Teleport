#!/usr/bin/env python3
"""
Seed Canonicalization CLI

Usage: python3 seed_canonicalize.py --in <seed_file> --out <canonical_seed> [--print-receipts]

Takes any seed, expands it, then produces the unique canonical minimal seed T*
that reproduces the same bytes with mathematical determinism.
"""

import sys
import argparse
import hashlib
from pathlib import Path

# Add teleport to path
sys.path.append(str(Path(__file__).parent.parent))

from teleport.encoder_dp import canonize_dp, canonize_bytes_dp
from teleport.seed_vm import expand
from teleport.clf_int import leb, pad_to_byte

def run_self_tests():
    """Run mandatory self-tests and print integer receipts"""
    print("=== SELF-TESTS (Integer Math Only) ===")
    
    # Test 1: MATCH beats LIT
    N = 50
    C_MATCH = 2 + 8*leb(3) + 8*leb(N)
    C_LIT = 10*N
    print(f"proof_MATCH_lt_LIT= {int(C_MATCH < C_LIT)}")
    
    # Test 2: CAUS.CONST frontier
    L = 40
    C_CONST = 3 + 8*1 + 8*1 + 8*leb(L)  # op=1, b=any, L=40
    C_LIT = 10*L
    print(f"proof_CONST_lt_LIT= {int(C_CONST < C_LIT)}")
    
    # Test 3: END pad exactness
    pos = 127
    C_END = 3 + pad_to_byte(pos + 3)
    print(f"C_END= {C_END}")
    
    print()

def main():
    parser = argparse.ArgumentParser(description="Canonical seed minimization")
    parser.add_argument("--in", dest="input_file", required=True,
                       help="Input seed file")
    parser.add_argument("--out", dest="output_file", required=True, 
                       help="Output canonical seed file")
    parser.add_argument("--print-receipts", action="store_true",
                       help="Print detailed console receipts")
    
    args = parser.parse_args()
    
    # Run self-tests first
    if args.print_receipts:
        run_self_tests()
    
    # Load input seed
    try:
        with open(args.input_file, 'rb') as f:
            seed_in = f.read()
    except FileNotFoundError:
        print(f"Error: Input file {args.input_file} not found")
        sys.exit(1)
    
    print("=== CANONICALIZATION PROCESS ===")
    
    # Step 1: Expand input seed to bytes S
    try:
        S = expand(seed_in)
        print(f"# Expand input")
        print(f"bytes= {len(S)}")
        print(f"sha256= {hashlib.sha256(S).hexdigest().upper()}")
        print()
    except Exception as e:
        print(f"Error expanding input seed: {e}")
        sys.exit(1)
    
    # Step 2: Canonical re-encoding
    print("# Canonicalization")
    try:
        # DP canonicalization with global optimality
        seed_min = canonize_dp(S, print_receipts=args.print_receipts)
        
        # Get detailed analysis
        choices, total_bits = canonize_bytes_dp(S, print_receipts=False)
        
        # Step 3: Verify identity
        S_prime = expand(seed_min)
        
        eq_bytes = int(len(S) == len(S_prime))
        eq_sha = int(hashlib.sha256(S).digest() == hashlib.sha256(S_prime).digest())
        
        print(f"bytes'= {len(S_prime)}")
        print(f"sha256'= {hashlib.sha256(S_prime).hexdigest().upper()}")
        print(f"eq_bytes= {eq_bytes}")
        print(f"eq_sha= {eq_sha}")
        print()
        
        # Global cost receipts
        if args.print_receipts:
            C_stream = total_bits
            C_LIT_N = 10 * len(S)
            delta_vs_LIT = C_LIT_N - C_stream
            avg_bits_per_byte = C_stream / len(S) if len(S) > 0 else 0
            
            print(f"# Global receipts")
            print(f"C_stream= {C_stream}")
            print(f"C_LIT({len(S)})= {C_LIT_N}")
            print(f"delta_vs_LIT= {delta_vs_LIT}")
            print(f"avg_bits_per_byte= {avg_bits_per_byte:.2f}")
            print()
        
        # Verify identity
        if eq_bytes != 1 or eq_sha != 1:
            print("ERROR: Canonicalization failed identity check!")
            sys.exit(1)
        
        # Step 4: Write canonical seed
        with open(args.output_file, 'wb') as f:
            f.write(seed_min)
        
        print(f"# Canonical seed written to {args.output_file}")
        print(f"# Input size: {len(seed_in)} bytes")
        print(f"# Output size: {len(seed_min)} bytes") 
        print(f"# Payload size: {len(S)} bytes")
        
        if args.print_receipts:
            print()
            print("=== DETAILED RECEIPTS ===")
            # Additional receipt printing would go here
            # (requires tracking tokens during canonization)
        
    except Exception as e:
        print(f"Error during canonicalization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
