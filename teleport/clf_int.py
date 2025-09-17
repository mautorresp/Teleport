"""
Integer Helpers - CLF Int Module

Core integer-only utilities for mathematical causality:
- LEB128 encoding/decoding
- Byte padding and alignment
- Bitwise operations and shifts
- Integer sequence manipulation

All operations maintain strict integer-only semantics.
"""

from typing import List, Tuple, Union
from .guards import no_float_guard, assert_integer_only


@no_float_guard
def leb(n: int) -> int:
    """
    Minimal unsigned LEB128 byte-length of non-negative integer n.
    Equation: leb(n)=1 + count of right-shifts by 7 until 0 (n>=0).
    """
    assert_integer_only(n)
    if n < 0:
        raise ValueError("leb requires non-negative integer")
    count = 1
    x = n >> 7
    while x != 0:
        count += 1
        x >>= 7
    return count





@no_float_guard
def pad_to_byte(bit_count: int) -> int:
    """
    Calculate padding needed to align bit count to byte boundary.
    CLF Formula: pad_to_byte(k) = (8 - (k % 8)) % 8
    
    Args:
        bit_count: Current bit count
        
    Returns:
        int: Number of padding bits needed (0-7)
    """
    assert_integer_only(bit_count)
    
    if bit_count < 0:
        raise ValueError("Bit count must be non-negative")
    
    return (8 - (bit_count % 8)) % 8


@no_float_guard
def bits_to_bytes(bit_count: int) -> int:
    """
    Convert bit count to minimum required byte count.
    
    Args:
        bit_count: Number of bits
        
    Returns:
        int: Minimum bytes needed
    """
    assert_integer_only(bit_count)
    
    if bit_count < 0:
        raise ValueError("Bit count must be non-negative")
    
    return (bit_count + 7) // 8


@no_float_guard
def safe_left_shift(value: int, shift: int, max_bits: int = 1 << 20) -> int:
    """
    Left shift with integer-only resource guard.
    Equation: result = value << shift ; require result.bit_length() <= max_bits
    """
    assert_integer_only(value, shift, max_bits)
    if shift < 0:
        raise ValueError("Shift must be non-negative")
    result = value << shift
    if result.bit_length() > max_bits:
        raise OverflowError("Left shift exceeds max bit-length")
    return result


@no_float_guard
def safe_right_shift(value: int, shift: int) -> int:
    """
    Safe right shift operation.
    
    Args:
        value: Value to shift
        shift: Number of positions to shift right
        
    Returns:
        int: Shifted value
    """
    assert_integer_only(value, shift)
    
    if shift < 0:
        raise ValueError("Shift count must be non-negative")
    
    return value >> shift


@no_float_guard
def extract_bits(value: int, start_bit: int, bit_count: int) -> int:
    """
    Extract a range of bits from an integer.
    
    Args:
        value: Source integer
        start_bit: Starting bit position (0-based, from LSB)
        bit_count: Number of bits to extract
        
    Returns:
        int: Extracted bits as integer
    """
    assert_integer_only(value, start_bit, bit_count)
    
    if start_bit < 0 or bit_count < 0:
        raise ValueError("Bit positions and counts must be non-negative")
    
    if bit_count == 0:
        return 0
    
    # Create mask for desired bits
    mask = (1 << bit_count) - 1
    
    # Shift value and apply mask
    return (value >> start_bit) & mask


@no_float_guard
def pack_bits(values: List[int], bit_widths: List[int]) -> int:
    """
    Pack (values[i], bit_widths[i]) LSB-first into a single integer.
    Equation: sum_{i} values[i] << sum_{j<i} bit_widths[j]
    """
    assert_integer_only(*values, *bit_widths)
    if len(values) != len(bit_widths):
        raise ValueError("Mismatched lengths")
    total = 0
    shift = 0
    for val, w in zip(values, bit_widths):
        if w <= 0:
            raise ValueError("Bit widths must be positive")
        if val < 0 or val >= (1 << w):
            raise ValueError(f"value {val} does not fit in {w} bits")
        total |= (val << shift)
        shift += w
    return total


@no_float_guard
def integer_log2(value: int) -> int:
    """
    Calculate floor(log2(value)) using integer operations only.
    
    Args:
        value: Positive integer
        
    Returns:
        int: Floor of log base 2
    """
    assert_integer_only(value)
    
    if value <= 0:
        raise ValueError("Value must be positive for log2")
    
    result = 0
    while value > 1:
        value >>= 1
        result += 1
    
    return result


@no_float_guard
def next_power_of_2(value: int, max_bits: int | None = None) -> int:
    """
    Next power of 2 >= value. Returns 1 for value <= 0.
    Resource guard: if max_bits is provided, result.bit_length() must not exceed it.
    """
    assert_integer_only(value)
    if value <= 0:
        res = 1
    elif (value & (value - 1)) == 0:
        res = value
    else:
        # Fill up to highest bit then add 1 (classic bit twiddling)
        v = value - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        v |= v >> 32
        v |= v >> 64  # harmless in Python; keeps logic symmetrical
        res = v + 1
    if max_bits is not None and res.bit_length() > max_bits:
        raise OverflowError("next_power_of_2 exceeds max bit-length")
    return res
