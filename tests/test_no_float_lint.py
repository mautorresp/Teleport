"""
Test No-Float Guards and Enforcement

Tests for the boot guards module that enforces integer-only operations
and prevents floating-point contamination.
"""

import pytest
from teleport.guards import (
    no_float_guard, assert_integer_only, safe_int_divide, safe_int_power,
    NoFloatContext, _check_no_float
)


class TestNoFloatGuard:
    """Test the @no_float_guard decorator."""
    
    def test_guard_allows_integers(self):
        @no_float_guard
        def add_ints(a, b):
            return a + b
        
        result = add_ints(5, 10)
        assert result == 15
    
    def test_guard_allows_integer_lists(self):
        @no_float_guard
        def sum_list(values):
            return sum(values)
        
        result = sum_list([1, 2, 3, 4])
        assert result == 10
    
    def test_guard_blocks_float_args(self):
        @no_float_guard
        def add_values(a, b):
            return a + b
        
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            add_values(5, 3.14)
    
    def test_guard_blocks_float_kwargs(self):
        @no_float_guard
        def add_values(a=0, b=0):
            return a + b
        
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            add_values(a=5, b=2.718)
    
    def test_guard_blocks_float_in_list(self):
        @no_float_guard
        def process_list(values):
            return sum(values)
        
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            process_list([1, 2, 3.5, 4])
    
    def test_guard_blocks_float_in_dict(self):
        @no_float_guard
        def process_dict(data):
            return sum(data.values())
        
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            process_dict({"a": 1, "b": 2, "c": 3.14})
    
    def test_guard_blocks_float_return_value(self):
        @no_float_guard
        def bad_function():
            return 3.14159
        
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            bad_function()
    
    def test_guard_allows_nested_structures(self):
        @no_float_guard
        def process_nested(data):
            return data
        
        nested = {
            "values": [1, 2, 3],
            "metadata": {"count": 3, "total": 6}
        }
        result = process_nested(nested)
        assert result == nested


class TestAssertIntegerOnly:
    """Test the assert_integer_only function."""
    
    def test_assert_allows_integers(self):
        assert_integer_only(1, 2, 3)  # Should not raise
    
    def test_assert_allows_integer_collections(self):
        assert_integer_only([1, 2, 3], {"a": 1, "b": 2})  # Should not raise
    
    def test_assert_blocks_float(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            assert_integer_only(1, 2, 3.14)
    
    def test_assert_blocks_float_in_collection(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            assert_integer_only([1, 2.5, 3])


class TestSafeIntDivide:
    """Test safe integer division."""
    
    def test_exact_division(self):
        quotient, remainder = safe_int_divide(15, 3)
        assert quotient == 5
        assert remainder == 0
    
    def test_division_with_remainder(self):
        quotient, remainder = safe_int_divide(17, 5)
        assert quotient == 3
        assert remainder == 2
    
    def test_division_by_zero_raises(self):
        with pytest.raises(ValueError, match="Division by zero"):
            safe_int_divide(10, 0)
    
    def test_non_integer_dividend_raises(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            safe_int_divide(10.5, 2)
    
    def test_non_integer_divisor_raises(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            safe_int_divide(10, 2.0)
    
    def test_negative_division(self):
        quotient, remainder = safe_int_divide(-17, 5)
        assert quotient == -4
        assert remainder == 3


class TestSafeIntPower:
    """Test safe integer exponentiation."""
    
    def test_positive_power(self):
        result = safe_int_power(2, 10)
        assert result == 1024
    
    def test_zero_power(self):
        result = safe_int_power(5, 0)
        assert result == 1
    
    def test_power_of_one(self):
        result = safe_int_power(7, 1)
        assert result == 7
    
    def test_negative_exponent_raises(self):
        with pytest.raises(ValueError, match="Negative exponents not supported"):
            safe_int_power(2, -3)
    
    def test_excessive_exponent_raises(self):
        with pytest.raises(ValueError, match="exceeds max bit-length"):
            safe_int_power(2, 2000000)  # 2M bits > 1M default limit
    
    def test_non_integer_base_raises(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            safe_int_power(2.5, 3)
    
    def test_non_integer_exponent_raises(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            safe_int_power(2, 3.0)


class TestNoFloatContext:
    """Test the NoFloatContext context manager."""
    
    def test_context_manager_usage(self):
        with NoFloatContext() as ctx:
            assert ctx is not None
            # Context should work normally
            result = 2 + 3
            assert result == 5


class TestCheckNoFloat:
    """Test the internal _check_no_float function."""
    
    def test_check_allows_int(self):
        _check_no_float(42)  # Should not raise
    
    def test_check_allows_str(self):
        _check_no_float("hello")  # Should not raise
    
    def test_check_allows_bool(self):
        _check_no_float(True)  # Should not raise
    
    def test_check_blocks_float(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            _check_no_float(3.14)
    
    def test_check_allows_empty_list(self):
        _check_no_float([])  # Should not raise
    
    def test_check_allows_integer_list(self):
        _check_no_float([1, 2, 3])  # Should not raise
    
    def test_check_blocks_float_in_list(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            _check_no_float([1, 2.5, 3])
    
    def test_check_allows_integer_dict(self):
        _check_no_float({"a": 1, "b": 2})  # Should not raise
    
    def test_check_blocks_float_in_dict_value(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            _check_no_float({"a": 1, "b": 2.5})
    
    def test_check_nested_structures(self):
        complex_data = {
            "numbers": [1, 2, 3],
            "metadata": {
                "count": 3,
                "nested": [10, 20, 30]
            }
        }
        _check_no_float(complex_data)  # Should not raise
    
    def test_check_object_attributes(self):
        class TestObj:
            def __init__(self):
                self.value = 42
                self.name = "test"
        
        obj = TestObj()
        _check_no_float(obj)  # Should not raise
    
    def test_check_object_with_float_attribute(self):
        class TestObj:
            def __init__(self):
                self.value = 3.14
        
        obj = TestObj()
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            _check_no_float(obj)
