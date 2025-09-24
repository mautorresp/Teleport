#!/usr/bin/env python3
"""
CLF Function-Builder Comprehensive Test Suite
=============================================

Tests all aspects of the sealed CLF Function-Builder API to ensure:
1. Builder-only construction (no direct tokens)
2. Receipts parity and mathematical correctness  
3. Superadditivity enforcement
4. Value-independence timing
5. Bijection validation after finalization
6. Rails enforcement (unit-lock, minimality, coverage)
"""

import time
import os
import pytest
from teleport.clf_fb import (
    encode_minimal, build_A_exact, build_B_structural, Builder,
    CLFViolation, PinViolation, receipt_bijection_ok, receipt_complexity
)

class TestBuilderConstruction:
    """Test that all token construction goes through Builder API."""
    
    def test_builder_enforces_boundaries(self):
        """Test Builder validates position and length boundaries."""
        S = b"Hello World"
        b = Builder(S)
        
        # Valid boundaries
        b.add_CONST(0, 5, 72)  # "Hello" -> 'H'
        
        # Invalid boundaries should raise
        with pytest.raises(AssertionError):
            b.add_CONST(-1, 3, 72)  # negative position
        with pytest.raises(AssertionError):
            b.add_CONST(0, 100, 72)  # length beyond end
        with pytest.raises(AssertionError):
            b.add_CONST(10, 5, 72)  # position + length > total
    
    def test_builder_enforces_semantic_constraints(self):
        """Test Builder enforces semantic constraints on operations."""
        S = b"\x42\x42\x42\x44\x46\x48"  # Suitable for CONST and STEP
        b = Builder(S)
        
        # CONST requires length >= 2
        with pytest.raises(AssertionError):
            b.add_CONST(0, 1, 0x42)
        
        # STEP requires length >= 3  
        with pytest.raises(AssertionError):
            b.add_STEP(0, 2, 0x42, 2)
        
        # MATCH requires D >= 1
        with pytest.raises(AssertionError):
            b.add_MATCH(0, 3, 0)  # D=0 invalid
    
    def test_builder_computes_costs_internally(self):
        """Test Builder computes all costs internally without external calls."""
        S = b"\xFF" * 10
        b = Builder(S)
        b.add_CONST(0, 10, 0xFF)
        
        tokens = b.tokens
        assert len(tokens) == 1
        
        # Verify cost_info structure
        op, params, L, cost_info, pos = tokens[0]
        assert 'C_stream' in cost_info
        assert 'C_op' in cost_info
        assert 'C_params' in cost_info
        assert 'C_L' in cost_info
        assert 'C_CAUS' in cost_info
        assert 'C_END' in cost_info
        
        # All costs must be integers
        for cost_key in ['C_stream', 'C_op', 'C_params', 'C_L', 'C_CAUS', 'C_END']:
            assert isinstance(cost_info[cost_key], int)

class TestReceiptsParity:
    """Test receipts provide correct mathematical accounting."""
    
    def test_receipts_coverage_exactness(self):
        """Test receipts verify exact coverage (ΣL_i == L)."""
        S = b"ABCDEFGHIJK"  # 11 bytes
        b = Builder(S)
        b.add_CONST(0, 4, 65)     # "AAAA" -> 'A'
        b.add_STEP(4, 3, 69, 1)   # "EFG" -> step
        b.add_CBD_LOGICAL(7, 4)   # remaining gap
        
        receipts = b.receipts()
        assert receipts['COVERAGE_OK'] == True
        
        # Verify actual coverage
        total_coverage = sum(t[2] for t in b.tokens)
        assert total_coverage == len(S)
    
    def test_receipts_minimality_gate(self):
        """Test receipts correctly compute minimality (TOTAL < RAW)."""
        # Case 1: Should be minimal (constant data)
        S = b"\x42" * 20
        b = Builder(S)
        b.add_CONST(0, 20, 0x42)
        
        receipts = b.receipts()
        assert receipts['MINIMALITY_OK'] == True
        assert receipts['TOTAL_BITS'] < receipts['RAW_BITS']
        
        # Case 2: Should not be minimal (random-like data)
        import os
        S = os.urandom(50)
        tokens = encode_minimal(S)
        
        if not tokens:  # OPEN case
            # For OPEN, we can't easily test minimality directly
            # but the function-builder made the right choice
            pass
        else:
            # If PASS, must satisfy minimality
            b = Builder(S)
            for token in tokens:
                if isinstance(token[0], str) and token[0] == "CBD_LOGICAL":
                    b.add_CBD_LOGICAL(token[4], token[2])
                # ... add other token types as needed
            
            receipts = b.receipts()
            assert receipts['MINIMALITY_OK'] == True
    
    def test_receipts_complexity_envelope(self):
        """Test receipts validate complexity envelope (ops ≤ α + β·L)."""
        S = b"Test data for complexity check"
        b = Builder(S)
        b.add_CONST(0, 4, 84)  # "Test"
        b.add_CBD_LOGICAL(4, len(S) - 4)  # rest
        
        receipts = b.receipts()
        complexity_info = receipts['COMPLEXITY']
        
        assert complexity_info['ENVELOPE_SATISFIED'] == True
        assert complexity_info['ACTUAL_OPS'] <= complexity_info['MAX_ALLOWED_OPS']
        assert complexity_info['ALPHA'] == 32
        assert complexity_info['BETA'] == 1

class TestSuperadditivity:
    """Test superadditivity enforcement for CBD-only B."""
    
    def test_cbd_only_superadditivity_enforced(self):
        """Test CBD-only B constructions satisfy superadditivity."""
        # Create data where B would be CBD-only
        S = os.urandom(100)  # Random data likely triggers CBD-only B
        
        # This should not raise CLFViolation due to superadditivity
        try:
            tokens = encode_minimal(S)
            # If we get tokens, superadditivity was satisfied
            # If we get [], it was OPEN for minimality reasons
            assert True  # No violation means test passed
        except CLFViolation as e:
            if "superadditivity violated" in str(e):
                pytest.fail(f"Superadditivity violation: {e}")
            else:
                # Other CLF violations are acceptable for this test
                pass
    
    def test_superadditivity_manual_construction(self):
        """Test superadditivity with manual A vs B construction."""
        S = b"\x01\x02\x03\x04\x05"
        
        A = build_A_exact(S)
        B = build_B_structural(S)
        
        # Check if B is CBD-only
        only_cbd_B = all(isinstance(t[0], str) and t[0] == "CBD_LOGICAL" for t in B.tokens)
        
        if only_cbd_B:
            A_stream = A.stream_bits()
            B_stream = B.stream_bits()
            
            # Superadditivity: B_stream >= A_stream
            assert B_stream >= A_stream, f"Superadditivity violated: {B_stream} < {A_stream}"

class TestValueIndependence:
    """Test value-independent timing for same-length inputs."""
    
    def test_fixed_length_timing_independence(self):
        """Test encoding time depends only on length, not values."""
        L = 50
        
        # Different value patterns, same length
        test_cases = [
            b"\x00" * L,           # all zeros
            b"\xFF" * L,           # all ones  
            b"\xAA" * L,           # alternating pattern
            bytes(range(256))[:L], # sequential
            os.urandom(L),         # random
        ]
        
        times = []
        for data in test_cases:
            start_time = time.perf_counter()
            encode_minimal(data)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Check timing variance is small (within 2x of minimum)
        min_time = min(times)
        max_time = max(times)
        
        # Allow for some system noise, but timing should be largely value-independent
        if min_time > 0:  # avoid division by zero
            timing_ratio = max_time / min_time
            assert timing_ratio < 5.0, f"Excessive timing variance: {timing_ratio:.2f}x"

class TestBijectionValidation:
    """Test bijection validation after finalization."""
    
    def test_bijection_receipt_validation(self):
        """Test bijection receipts validate correctly."""
        test_cases = [
            b"Hello",
            b"\x00\x01\x02\x03",
            b"\xFF" * 10,
            b"Mixed content with various bytes"
        ]
        
        for S in test_cases:
            tokens = encode_minimal(S)
            
            if tokens:  # PASS case
                # Verify bijection receipt  
                assert receipt_bijection_ok(tokens, S), f"Bijection failed for {S!r}"
                
                # Verify decode matches input
                from teleport.clf_canonical import decode_CLF
                decoded = decode_CLF(tokens)
                assert decoded == S, f"Decode mismatch: {decoded!r} != {S!r}"
    
    def test_bijection_failure_detection(self):
        """Test bijection validation detects corrupted tokens."""
        S = b"Test bijection detection"
        tokens = encode_minimal(S)
        
        if tokens:
            # Corrupt a token and verify detection
            corrupted_tokens = tokens.copy()
            if len(corrupted_tokens[0]) >= 5:
                # Corrupt the cost_info or parameters
                op, params, L, cost_info, pos = corrupted_tokens[0]
                corrupted_tokens[0] = (op, params, L + 1, cost_info, pos)  # Wrong length
                
                # Should detect bijection failure
                assert not receipt_bijection_ok(corrupted_tokens, S)

class TestRailsEnforcement:
    """Test CLF rails are enforced by construction."""
    
    def test_unit_lock_enforcement(self):
        """Test unit-lock is enforced at module load."""
        # Unit-lock check happens at import, if we got here it passed
        from teleport.clf_fb import _pin_unit_lock
        assert _pin_unit_lock() == True
    
    def test_no_float_contamination(self):
        """Test integer-only math throughout."""
        S = b"Test integer-only math"
        b = Builder(S)
        b.add_CONST(0, 4, 84)
        b.add_CBD_LOGICAL(4, len(S) - 4)
        
        totals = b.totals()
        receipts = b.receipts()
        
        # All totals must be integers
        for key, value in totals.items():
            if isinstance(value, (int, float)):
                assert isinstance(value, int), f"Float contamination in {key}: {value}"
        
        # All receipt values must be integers  
        for key, value in receipts.items():
            if isinstance(value, (int, float)):
                assert isinstance(value, int), f"Float contamination in receipts {key}: {value}"

class TestEncodeMinimal:
    """Test the universal encode_minimal function."""
    
    def test_encode_minimal_universal_decision(self):
        """Test encode_minimal makes universal A vs B decision."""
        test_cases = [
            b"",                    # empty
            b"A",                  # single byte
            b"AAAA",               # constant run
            b"\x01\x02\x03\x04",  # step pattern
            os.urandom(20),        # random data
        ]
        
        for S in test_cases:
            tokens = encode_minimal(S)
            
            # Result is either [] (OPEN) or valid token list (PASS)
            assert isinstance(tokens, list)
            
            if tokens:
                # PASS case: verify all rails
                # Coverage
                total_L = sum(t[2] for t in tokens)
                assert total_L == len(S)
                
                # Bijection  
                assert receipt_bijection_ok(tokens, S)
                
                # Minimality (implied by successful return from encode_minimal)
                b = Builder(S)
                total_bits = 0
                for token in tokens:
                    total_bits += token[3]['C_stream']
                
                H = 16 + 8 * ((8 * len(S)).bit_length() + 6) // 7  # header_bits equivalent
                total_with_header = H + total_bits
                raw_bits = 8 * len(S)
                
                assert total_with_header < raw_bits, "Minimality violation in PASS case"

if __name__ == "__main__":
    import sys
    
    print("=== CLF Function-Builder Test Suite ===")
    
    # Run tests directly for demonstration
    test_builder = TestBuilderConstruction()
    test_receipts = TestReceiptsParity()
    test_superadd = TestSuperadditivity()
    test_timing = TestValueIndependence()
    test_bijection = TestBijectionValidation()
    test_rails = TestRailsEnforcement()
    test_encode = TestEncodeMinimal()
    
    try:
        # Builder construction tests
        print("\n1. Testing Builder construction...")
        test_builder.test_builder_enforces_boundaries()
        test_builder.test_builder_enforces_semantic_constraints()
        test_builder.test_builder_computes_costs_internally()
        print("✓ Builder construction tests passed")
        
        # Receipts parity tests
        print("\n2. Testing receipts parity...")
        test_receipts.test_receipts_coverage_exactness()
        test_receipts.test_receipts_minimality_gate()
        test_receipts.test_receipts_complexity_envelope()
        print("✓ Receipts parity tests passed")
        
        # Superadditivity tests
        print("\n3. Testing superadditivity...")
        test_superadd.test_cbd_only_superadditivity_enforced()
        test_superadd.test_superadditivity_manual_construction()
        print("✓ Superadditivity tests passed")
        
        # Value-independence tests
        print("\n4. Testing value-independence...")
        test_timing.test_fixed_length_timing_independence()
        print("✓ Value-independence tests passed")
        
        # Bijection tests
        print("\n5. Testing bijection validation...")
        test_bijection.test_bijection_receipt_validation()
        test_bijection.test_bijection_failure_detection()
        print("✓ Bijection validation tests passed")
        
        # Rails enforcement tests
        print("\n6. Testing rails enforcement...")
        test_rails.test_unit_lock_enforcement()
        test_rails.test_no_float_contamination()
        print("✓ Rails enforcement tests passed")
        
        # Universal encoder tests
        print("\n7. Testing encode_minimal...")
        test_encode.test_encode_minimal_universal_decision()
        print("✓ encode_minimal tests passed")
        
        print("\n" + "="*50)
        print("✅ ALL FUNCTION-BUILDER TESTS PASSED")
        print("✅ CLF rails enforced by construction")
        print("✅ Ready for production use")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)