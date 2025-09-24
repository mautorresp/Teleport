#!/usr/bin/env python3
"""
Unit tests for CLF calculator functions.

Tests the mathematical functions directly using pure integer lengths 
(no file reads), as specified in Section 7 of the CLF docstring guide.
"""

import sys
import os
import unittest
import hashlib

# Add parent directory to path to import clf_calculator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clf_calculator import leb_len_u, clf_single_seed_cost, should_emit, receipt


class TestCLFCalculator(unittest.TestCase):
    """Test cases for CLF calculator mathematical functions."""

    def test_leb_len_u_bands(self):
        """Test leb_len_u at LEB128 band boundaries."""
        # Test cases: leb_len_u(L) -> expected_leb
        test_cases = [
            (0, 1),      # Special case: leb(0) = 1
            (1, 1),      # 2^0
            (127, 1),    # 2^7 - 1 (last 1-byte LEB)
            (128, 2),    # 2^7 (first 2-byte LEB)
            (16383, 2),  # 2^14 - 1 (last 2-byte LEB)
            (16384, 3),  # 2^14 (first 3-byte LEB)
        ]
        
        for n, expected_leb in test_cases:
            with self.subTest(n=n):
                actual_leb = leb_len_u(n)
                self.assertEqual(actual_leb, expected_leb, 
                    f"leb_len_u({n}) = {actual_leb}, expected {expected_leb}")

    def test_leb_len_u_negative(self):
        """Test leb_len_u raises assertion for negative input."""
        with self.assertRaises(AssertionError):
            leb_len_u(-1)

    def test_cost_examples(self):
        """Test clf_single_seed_cost at specific examples."""
        # Test cases: L -> expected C_min^(1)(L)
        test_cases = [
            (456, 104),      # leb(456) = 2, so 88 + 8*2 = 104
            (968, 104),      # leb(968) = 2, so 88 + 8*2 = 104
            (1_570_024, 112), # leb(1570024) = 3, so 88 + 8*3 = 112
        ]
        
        for L, expected_cost in test_cases:
            with self.subTest(L=L):
                actual_cost = clf_single_seed_cost(L)
                self.assertEqual(actual_cost, expected_cost,
                    f"clf_single_seed_cost({L}) = {actual_cost}, expected {expected_cost}")

    def test_gate_examples(self):
        """Test should_emit returns True for practical file sizes."""
        # Files that should definitely emit (C_min^(1)(L) < 10*L)
        test_files = [16, 456, 968, 1_570_024]
        
        for L in test_files:
            with self.subTest(L=L):
                result = should_emit(L)
                self.assertTrue(result, 
                    f"should_emit({L}) = {result}, expected True")

    def test_receipt_shape(self):
        """Test receipt returns correct structure and deterministic SHA256."""
        L = 1000
        build_id = "TEST_BUILD_20250923"
        
        # Get receipt
        r = receipt(L, build_id)
        
        # Check required keys present
        required_keys = {"L", "leb", "C_min_bits", "RAW_bits", "EMIT", "sha256"}
        self.assertTrue(required_keys.issubset(r.keys()),
            f"Receipt missing keys: {required_keys - r.keys()}")
        
        # Check SHA256 is 64 hex characters
        sha256_val = r["sha256"]
        self.assertEqual(len(sha256_val), 64, 
            f"SHA256 length = {len(sha256_val)}, expected 64")
        self.assertTrue(all(c in "0123456789abcdef" for c in sha256_val),
            "SHA256 contains non-hex characters")
        
        # Check deterministic: same input yields same receipt
        r2 = receipt(L, build_id)
        self.assertEqual(r["sha256"], r2["sha256"],
            "Receipt should be deterministic")
        
        # Check receipt values are integers/bool as expected
        self.assertIsInstance(r["L"], int)
        self.assertIsInstance(r["leb"], int)
        self.assertIsInstance(r["C_min_bits"], int)
        self.assertIsInstance(r["RAW_bits"], int)
        self.assertIsInstance(r["EMIT"], bool)

    def test_receipt_calculation_consistency(self):
        """Test receipt calculations match individual function calls."""
        L = 11751  # pic2.jpg size
        build_id = "TEST_BUILD"
        
        r = receipt(L, build_id)
        
        # Check calculations match individual functions
        self.assertEqual(r["L"], L)
        self.assertEqual(r["leb"], leb_len_u(L))
        self.assertEqual(r["C_min_bits"], clf_single_seed_cost(L))
        self.assertEqual(r["RAW_bits"], 10 * L)
        self.assertEqual(r["EMIT"], should_emit(L))

    def test_mathematical_invariants(self):
        """Test key mathematical invariants hold."""
        # Test that C_min increases with leb bands
        L_small = 100    # leb = 1, C_min = 96
        L_medium = 1000  # leb = 2, C_min = 104  
        L_large = 20000  # leb = 3, C_min = 112
        
        C_small = clf_single_seed_cost(L_small)
        C_medium = clf_single_seed_cost(L_medium)
        C_large = clf_single_seed_cost(L_large)
        
        self.assertLess(C_small, C_medium, 
            "C_min should increase with leb bands")
        self.assertLess(C_medium, C_large,
            "C_min should increase with leb bands")
        
        # Test that EMIT threshold is strict (<, not <=)
        # Find a theoretical boundary case where C_min = 10*L
        # This is unlikely for real files, but test the strictness
        for L in [1, 2, 3, 4, 5]:
            C = clf_single_seed_cost(L)
            RAW = 10 * L
            if C == RAW:  # If we find exact equality
                self.assertFalse(should_emit(L), 
                    f"should_emit({L}) should be False when C_min == 10*L")
                break


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)