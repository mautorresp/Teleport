#!/usr/bin/env python3
"""
Teleport Canonicalization Tool with Strict Autodetection
Self-selects between 'this is a Teleport seed' and 'this is just raw bytes'.
No heuristics - pure grammar validation per mathematical specification.

Usage: python3 seed_canonicalize.py <input_file> <output_seed>
"""

import sys
import hashlib
from pathlib import Path
from pathlib import Path

# Add teleport to path
sys.path.append(str(Path(__file__).parent.parent))

from teleport.encoder_dp import canonize_bytes_dp, serialize_tokens_to_seed
from teleport.seed_vm import expand
from teleport.seed_validate import validate_teleport_stream
from teleport.spec_constants import TELEPORT_MAGIC_VERSION_BE
from teleport.clf_int import leb

def main():
    if len(sys.argv) < 2:
        print("Usage: python seed_canonicalize.py --in <file> --out <output> [--print-receipts] [--no-require-caus]")
        sys.exit(1)
    
    input_file = None
    output_file = None
    print_receipts = False
    
    # CLF REALIGNMENT: REQUIRE_CAUS = True by default (deduction-first)
    require_caus = "--no-require-caus" not in sys.argv
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--in" and i + 1 < len(sys.argv):
            input_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--out" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--print-receipts":
            print_receipts = True
            i += 1
        elif sys.argv[i] == "--no-require-caus":
            i += 1  # Already handled above
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            sys.exit(1)
    
    # Read input
    if not Path(input_file).exists():
        print(f"ERROR: Input file not found: {input_file}")
        return 1
    
    with open(input_file, "rb") as f:
        input_data = f.read()
    
    print(f"Input size: {len(input_data)} bytes")
    
    # STRICT AUTODETECTION: Grammar validation only
    print("\n=== MODE DECISION (strict) ===")
    is_teleport_seed, validation_note = validate_teleport_stream(input_data, TELEPORT_MAGIC_VERSION_BE)
    print(f"is_seed= {int(is_teleport_seed)} reason= {validation_note}")
    
    if is_teleport_seed:
        print("✓ Input validates as TELEPORT SEED")
        print(f"Validation note: {validation_note}")
        print("Action: Validate and re-canonicalize existing seed")
        
        # Process as existing seed
        try:
            recovered_data = expand(input_data)
            print(f"Seed decodes to: {len(recovered_data)} bytes")
            
            # Re-canonicalize the decoded data
            print("\n--- Re-Canonicalizing Decoded Data ---")
            tokens, total_bits, C_end, _ = canonize_bytes_dp(recovered_data)
            
            if not tokens:
                print("ERROR: Re-canonicalization failed")
                return 1
            
            # Generate new canonical seed using bit-exact format
            from clf_bitexact import serialize_tokens_bit_exact
            canonical_seed = serialize_tokens_bit_exact(tokens, total_bits)
            
            print(f"Re-canonicalized token sequence (cost = {C_end}):")
            for i, (kind, params, L) in enumerate(tokens):
                print(f"  {i+1}. {kind}{params} → {L} bytes")
            
            # Compare seeds
            if canonical_seed == input_data:
                print("✓ Input seed is already canonical")
            else:
                print("→ Generating optimized canonical seed")
            
            # Write output
            with open(output_file, "wb") as f:
                f.write(canonical_seed)
            
            print(f"Canonical seed written: {output_file} ({len(canonical_seed)} bytes)")
            
        except Exception as e:
            print(f"ERROR processing seed: {e}")
            return 1
            
    else:
        print("✓ Input detected as RAW BYTES")  
        print(f"Validation note: {validation_note}")
        print("Action: Generate canonical Teleport seed")
        
        # Process as raw bytes
        print("\n--- DP Canonicalization ---")
        tokens, total_bits, C_end, predicate_receipts = canonize_bytes_dp(input_data, print_receipts=print_receipts)
        
        # CLF CAUS-or-FAIL gate (no silent fallback)
        first_kind, first_params, first_L = tokens[0]
        if require_caus and not (first_kind == "CAUS" and first_L == len(input_data)):
            print("CAUSE_NOT_DEDUCED")
            print("evaluated_predicates=", predicate_receipts)
            sys.exit(2)
        
        if not tokens:
            print("CANONICALIZATION FAILED: No valid tokenization found")
            return 1
        
        # Generate mathematical receipt
        print(f"\n# DP Canonicalization")
        pos = 0
        for i, (kind, params, L) in enumerate(tokens):
            if kind == "LIT":
                # LIT params: (byte_value, run_length)
                b, run_len = params
                c_bits = 10 * L
                print(f"p={pos} chosen=LIT(b={b},L={L}) C_token={c_bits} C_LIT({L})={10*L} strict_ineq=0")
            elif kind == "MATCH":
                D = params[0]
                c_match = 2 + 8 * leb(D) + 8 * leb(L)
                c_lit = 10 * L if L <= 10 else "inadmissible"
                strict_ineq = 1 if L > 10 else (1 if c_match < 10 * L else 0)
                print(f"p={pos} chosen=MATCH(D={D},L={L}) C_token={c_match} C_LIT({L})={c_lit} strict_ineq={strict_ineq}")
            pos += L
        
        print(f"\n# Global receipts")
        token_bits = total_bits - C_end
        print(f"C_tokens= {token_bits}")
        print(f"C_END= {C_end}")
        print(f"C_stream= {total_bits}")
        print(f"C_LIT({len(input_data)})= {10 * len(input_data)}")
        print(f"delta_vs_LIT= {10 * len(input_data) - total_bits}")
        
        # Create canonical seed using bit-exact serialization
        from clf_bitexact import serialize_tokens_bit_exact
        canonical_seed = serialize_tokens_bit_exact(tokens, total_bits)
        
        # Write seed file
        with open(output_file, "wb") as f:
            f.write(canonical_seed)
        
        print(f"Seed written: {output_file} ({len(canonical_seed)} bytes)")
        
        # Check for CAUS-specific receipts (drastic minimality)
        if tokens and len(tokens) > 0:
            first_token = tokens[0]
            kind, params, L = first_token
            if kind == "CAUS" and L == len(input_data):
                # CAUS covers entire input - drastic minimality achieved!
                op_id = params[0] 
                op_params = params[1:]
                
                from teleport.caus_deduction import compute_caus_cost, compute_caus_seed_bytes
                c_caus = compute_caus_cost(op_id, op_params, len(input_data))
                expected_seed_bytes = compute_caus_seed_bytes(op_id, op_params, len(input_data))
                
                print(f"\n🎯 CLF DRASTIC MINIMALITY ACHIEVED:")
                print(f"CAUS op={op_id}, params={op_params}, N={len(input_data)}")
                print(f"C_CAUS = {c_caus} bits")
                print(f"Expected seed_bytes = {expected_seed_bytes}")
                print(f"Actual seed_bytes = {len(canonical_seed)}")
                print(f"Compression ratio N/seed = {len(input_data)}/{len(canonical_seed)} = {len(input_data)/len(canonical_seed):.2f}")
                print(f"Drastic gap: {len(input_data) - len(canonical_seed)} bytes saved")
        
        # Verify fundamental CLF invariant
        bits_on_disk = len(canonical_seed) * 8
        print(f"\n🔑 FUNDAMENTAL CLF INVARIANT:")
        print(f"8 × len(seed) = {bits_on_disk}")  
        print(f"C_stream      = {total_bits}")
        print(f"MATCHES:      {bits_on_disk == total_bits}")
    
    # Final validation
    print("\n--- Round-Trip Validation ---")
    try:
        with open(output_file, "rb") as f:
            final_seed = f.read()
        
        from clf_bitexact import expand_bit_exact
        recovered_data = expand_bit_exact(final_seed)
        
        if is_teleport_seed:
            # Compare with decoded original
            original_decoded = expand(input_data)
            if recovered_data == original_decoded:
                print("✓ Round-trip PASSED (seed→data→seed)")
            else:
                print("✗ Round-trip FAILED")
                return 1
        else:
            # Compare with original raw bytes
            if recovered_data == input_data:
                print("✓ Round-trip PASSED (bytes→seed→bytes)")
            else:
                print("✗ Round-trip FAILED")
                return 1
                
    except Exception as e:
        print(f"✗ Round-trip ERROR: {e}")
        return 1
    
    print("\n=== CANONICALIZATION COMPLETE ===")
    return 0

if __name__ == "__main__":
    main()
