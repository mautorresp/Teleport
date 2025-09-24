"""
CLF S-Packing Elimination Test
=============================

Unit test that fails if any seed length scales as Θ(L) for arbitrary S.
Ensures no residual bit-grouping of input data.
"""

import pytest
import os
import hashlib
from teleport.clf_spec_alignment import build_A_exact_aligned, build_B_structural_aligned

def test_no_s_packing_residue():
    """
    Critical test: verify no code path groups bits of S as a 'seed'.
    Any Θ(L) scaling in seed parameters indicates S-packing violation.
    """
    # Generate test strings of increasing length
    test_lengths = [100, 500, 1000, 5000, 10000]
    
    for L in test_lengths:
        # Create arbitrary data (high entropy)
        S = os.urandom(L)
        
        # Test A builder - must NOT produce seeds scaling with L
        C_A_stream, tokens_A = build_A_exact_aligned(S)
        
        if C_A_stream is not None:
            # A found a CAUS representation - verify parameters don't scale with L
            for token in tokens_A:
                if token[0] == 'CAUS':
                    op_id = token[1]
                    params = token[2]
                    
                    # CRITICAL: parameters must be O(1) in size, not O(L)
                    for param in params:
                        param_bits = param.bit_length()
                        
                        # Fail if parameter size grows with L (S-packing indicator)
                        max_param_bits = 64  # Reasonable constant bound
                        assert param_bits <= max_param_bits, \
                            f"S-PACKING DETECTED: param {param} has {param_bits} bits for L={L}, exceeds {max_param_bits}"
                        
                        # Additional check: parameter should not encode length-dependent information
                        if L > 1000 and param_bits > 32:
                            # For large L, parameters shouldn't need many bits unless truly structured
                            assert param < 2**20, \
                                f"SUSPICIOUS PARAMETER: {param} for L={L} suggests length-dependent encoding"
        
        # Test B builder - structural tiling should not produce length-dependent parameters
        B_complete, C_B_stream, tokens_B, struct_counts = build_B_structural_aligned(S)
        
        if B_complete and tokens_B:
            for token in tokens_B:
                if token[0] == 'CAUS':
                    params = token[2] if len(token) > 2 else []
                    for param in params:
                        param_bits = param.bit_length()
                        assert param_bits <= 32, \
                            f"B-BUILDER S-PACKING: param {param} has {param_bits} bits for L={L}"

def test_seed_parameter_bounds():
    """
    Verify all CAUS parameters are bounded integers representing
    mathematical properties, not bit-encodings of input data.
    """
    # Test with structured data that should produce small parameters
    test_cases = [
        (b'\x42' * 100, "CONST case"),  # Should produce param=0x42, not length-dependent
        (bytes(range(50)), "STEP case"),  # Should produce start=0, stride=1
        (b'ABCD' * 25, "Pattern case")   # Should find structural representation
    ]
    
    for S, description in test_cases:
        L = len(S)
        
        # A builder test
        C_A_stream, tokens_A = build_A_exact_aligned(S)
        
        if tokens_A:
            for token in tokens_A:
                if token[0] == 'CAUS':
                    params = token[2]
                    print(f"{description}: A params = {params} for L={L}")
                    
                    # Parameters should be small integers representing structure
                    for param in params:
                        assert isinstance(param, int), f"Non-integer parameter: {param}"
                        assert 0 <= param < 2**20, f"Parameter {param} too large, suggests S-packing"
        
        # B builder test
        B_complete, C_B_stream, tokens_B, struct_counts = build_B_structural_aligned(S)
        
        print(f"{description}: B_complete={B_complete}, tokens={len(tokens_B) if tokens_B else 0}")

def test_no_leb_encoding_of_input_bits():
    """
    Verify no code path computes leb_len(8*L) except in header.
    This would indicate attempting to LEB-encode the raw input size.
    """
    from teleport.clf_leb_lock import leb_len
    
    # Generate various input sizes
    for L in [1, 10, 100, 1000]:
        S = b'A' * L
        
        # Build with both paths
        C_A_stream, tokens_A = build_A_exact_aligned(S)
        B_complete, C_B_stream, tokens_B, struct_counts = build_B_structural_aligned(S)
        
        # Check that no token cost computation uses leb_len(8*L)
        # This is indirect - we verify costs are reasonable and don't scale as leb_len(8*L)
        
        raw_bits = 8 * L
        leb_8L = leb_len(raw_bits)
        
        if C_A_stream is not None:
            # A cost should not be dominated by leb_len(8*L) term
            # For CONST case, cost should be ~constant, not growing with leb_len(8*L)
            if L > 100:  # Large enough to see scaling
                cost_per_bit = C_A_stream / raw_bits
                assert cost_per_bit < 0.1, \
                    f"A cost {C_A_stream} for L={L} suggests leb_len(8*L)={leb_8L} scaling"
        
        if B_complete and C_B_stream is not None:
            # B cost should be reasonable structural tiling, not input-bit scaling
            efficiency = C_B_stream / raw_bits
            assert efficiency < 2.0, \
                f"B cost {C_B_stream} for L={L} exceeds 2x raw bits, suggests improper scaling"

if __name__ == "__main__":
    test_no_s_packing_residue()
    test_seed_parameter_bounds() 
    test_no_leb_encoding_of_input_bits()
    print("✅ All S-packing elimination tests passed")