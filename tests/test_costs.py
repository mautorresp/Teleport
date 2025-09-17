"""
Test Teleport Cost Law — Integer-Exact

Tests for the four fundamental Teleport cost functions that define
the pricing rules for literals, matches, end tokens, and causal operations.
"""

import pytest
from teleport.costs import cost_lit, cost_match, cost_end, cost_caus


class TestCostLit:
    """Test literal cost calculation."""
    
    def test_cost_lit(self):
        assert cost_lit(0) == 0
        assert cost_lit(3) == 30
        assert cost_lit(10) == 100
    
    def test_cost_lit_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            cost_lit(-1)


class TestCostMatch:
    """Test match cost calculation."""
    
    def test_cost_match(self):
        # leb(1)=1, leb(3)=1
        assert cost_match(1, 3) == 2 + 8*1 + 8*1  # 18
    
    def test_cost_match_larger_values(self):
        # leb(128)=2, leb(200)=2
        assert cost_match(128, 200) == 2 + 8*2 + 8*2  # 34
    
    def test_cost_match_zero_values(self):
        # leb(0)=1, leb(0)=1
        assert cost_match(0, 0) == 2 + 8*1 + 8*1  # 18
    
    def test_cost_match_negative_d_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            cost_match(-1, 5)
    
    def test_cost_match_negative_l_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            cost_match(5, -1)


class TestCostEnd:
    """Test end token cost calculation."""
    
    def test_cost_end(self):
        # p=0 → pad_to_byte(3)=5 (bits to next byte), so C_END=3+5=8
        assert cost_end(0) == 8
    
    def test_cost_end_aligned(self):
        # p=5 → pad_to_byte(8)=0, so C_END=3+0=3
        assert cost_end(5) == 3
    
    def test_cost_end_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            cost_end(-1)


class TestCostCaus:
    """Test causal operation cost calculation."""
    
    def test_cost_caus_simple(self):
        # op=1 → leb(1)=1, params=[7] → leb(7)=1, L=5 → leb(5)=1
        expected = 3 + 8*1 + 8*(1) + 8*1  # 27
        assert cost_caus(1, [7], 5) == expected
    
    def test_cost_caus_multiple_params(self):
        # op=2 → leb(2)=1, params=[10, 20] → leb(10)=1, leb(20)=1, L=3 → leb(3)=1
        expected = 3 + 8*1 + 8*(1+1) + 8*1  # 35
        assert cost_caus(2, [10, 20], 3) == expected
    
    def test_cost_caus_empty_params(self):
        # op=5 → leb(5)=1, params=[] → sum=0, L=2 → leb(2)=1
        expected = 3 + 8*1 + 8*0 + 8*1  # 19
        assert cost_caus(5, [], 2) == expected
    
    def test_cost_caus_large_values(self):
        # op=300 → leb(300)=2, params=[128] → leb(128)=2, L=500 → leb(500)=2
        expected = 3 + 8*2 + 8*2 + 8*2  # 51
        assert cost_caus(300, [128], 500) == expected
    
    def test_cost_caus_negative_op_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            cost_caus(-1, [5], 3)
    
    def test_cost_caus_negative_l_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            cost_caus(1, [5], -3)
    
    def test_cost_caus_negative_param_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            cost_caus(1, [5, -2], 3)


class TestFloatGuardIntegration:
    """Test that all cost functions properly reject floats."""
    
    def test_cost_lit_rejects_float(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            cost_lit(3.14)
    
    def test_cost_match_rejects_float(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            cost_match(1.5, 2)
        
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            cost_match(1, 2.5)
    
    def test_cost_end_rejects_float(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            cost_end(1.5)
    
    def test_cost_caus_rejects_float(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            cost_caus(1.5, [2], 3)
        
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            cost_caus(1, [2.5], 3)
        
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            cost_caus(1, [2], 3.5)
