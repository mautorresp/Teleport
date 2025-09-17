"""
Test Integer Helpers (clf_int module)

Tests for core integer-only utilities    def test_leb_encode_decode_roundtrip(self):
        for v in [0, 1, 127, 128, 255, 256, (1 << 63), (1 << 80) - 1]:
            enc = leb128_encode(v)
            v2, n = leb128_decode(enc, 0)
            assert v2 == v and n == len(enc)

    def test_decode_unterminated(self):
        # Continuation bit set but no more bytes
        with pytest.raises(Exception, match="unterminated|Incomplete"):
            leb128_decode(b"\x80", 0)

    def test_decode_resource_bound(self):
        # 0x80 0x80 0x80 0x01 needs 4 bytes; cap at 2 should fail
        with pytest.raises(Exception, match="resource|bound|Incomplete"):
            leb128_decode(b"\x80\x80\x80\x01", 0, max_bytes=2) bit operations,
padding, shifts, and integer sequence manipulation.
"""

import pytest

# LEB128 lives in leb_io; we alias to the legacy names used in this suite.
from teleport.leb_io import (
    leb128_emit_single as leb128_encode,
    leb128_parse_single as leb128_decode,
    leb128_parse_single_minimal as leb128_read_minimal,
)

# Core integer helpers live in clf_int.
from teleport.clf_int import (
    leb, pad_to_byte,
    safe_left_shift, safe_right_shift,
    extract_bits, pack_bits,
    integer_log2, next_power_of_2,
)


class TestLEBLength:
    """Test LEB length calculation function."""
    
    def test_leb_length(self):
        assert leb(0) == 1
        assert leb(127) == 1
        assert leb(128) == 2
        assert leb((1<<64)-1) == 10  # 64 bits need 10 LEB bytes
    
    def test_leb_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            leb(-1)


class TestLEB128:
    """Test LEB128 encoding and decoding."""
    
    def test_encode_zero(self):
        result = leb128_encode(0)
        assert result == b'\x00'
    
    def test_encode_small_values(self):
        assert leb128_encode(1) == b'\x01'
        assert leb128_encode(127) == b'\x7F'
    
    def test_encode_multi_byte_values(self):
        assert leb128_encode(128) == b'\x80\x01'
        assert leb128_encode(300) == b'\xAC\x02'
        assert leb128_encode(16384) == b'\x80\x80\x01'
    
    def test_encode_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            leb128_encode(-1)
    
    def test_decode_zero(self):
        value, consumed = leb128_decode(b'\x00', 0)
        assert value == 0 and consumed == 1
    
    def test_decode_small_values(self):
        value, consumed = leb128_decode(b'\x01', 0)
        assert value == 1 and consumed == 1
        value, consumed = leb128_decode(b'\x7F', 0)
        assert value == 127 and consumed == 1
    
    def test_decode_multi_byte_values(self):
        value, consumed = leb128_decode(b'\x80\x01', 0)
        assert value == 128 and consumed == 2
        value, consumed = leb128_decode(b'\xAC\x02', 0)
        assert value == 300 and consumed == 2
    
    def test_decode_with_offset(self):
        data = b'\xFF\x80\x01\xFF'
        value, consumed = leb128_decode(data, 1)
        assert value == 128 and consumed == 2
    
    def test_leb_encode_decode_roundtrip(self):
        for v in [0, 1, 127, 128, 255, 256, (1<<63), (1<<80) - 1]:
            b = leb128_encode(v)
            v2, n = leb128_decode(b, 0)
            assert v2 == v
            assert n == len(b)
    
    def test_decode_unterminated(self):
        # Continuation bit set but no more bytes
        with pytest.raises(Exception, match="unterminated|Incomplete"):
            leb128_decode(b"\x80", 0)

    def test_decode_resource_bound(self):
        # 0x80 0x80 0x80 0x01 needs 4 bytes; cap at 2 should fail
        with pytest.raises(Exception, match="resource|bound|Incomplete"):
            leb128_decode(b"\x80\x80\x80\x01", 0, max_bytes=2)


class TestLEBMinimal:
    """Test minimal LEB128 enforcement."""
    
    def test_leb_minimal_enforced(self):
        # Non-minimal form for 0 (should be just 0x00)
        non_min = bytes([0x80, 0x00])
        with pytest.raises(Exception, match="Non-minimal"):
            leb128_read_minimal(non_min, 0)

    def test_leb_minimal_valid(self):
        value, consumed = leb128_read_minimal(b"\x00", 0)
        assert value == 0 and consumed == 1
        value, consumed = leb128_read_minimal(b"\x80\x01", 0)
        assert value == 128 and consumed == 2
class TestBitOperations:
    """Test bit manipulation functions."""
    
    def test_pad_to_byte_aligned(self):
        assert pad_to_byte(0) == 0
        assert pad_to_byte(8) == 0
        assert pad_to_byte(16) == 0
    
    def test_pad_to_byte_unaligned(self):
        assert pad_to_byte(1) == 7
        assert pad_to_byte(3) == 5
        assert pad_to_byte(7) == 1
        assert pad_to_byte(9) == 7
    
    def test_pad_to_byte_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            pad_to_byte(-1)
    
    def test_pad_to_byte_equation(self):
        for k in range(0, 17):
            assert pad_to_byte(k) == ((8 - (k % 8)) % 8)


class TestSafeShifts:
    """Test safe shift operations."""
    
    def test_safe_left_shift_basic(self):
        assert safe_left_shift(1, 3) == 8
        assert safe_left_shift(5, 2) == 20
        assert safe_left_shift(0, 10) == 0
    
    def test_safe_left_shift_overflow_protection(self):
        with pytest.raises(OverflowError, match="max bit-length"):
            safe_left_shift(1 << 19, 10, max_bits=20)  # Would exceed bit limit
    
    def test_safe_left_shift_negative_shift_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            safe_left_shift(5, -1)
    
    def test_safe_right_shift_basic(self):
        assert safe_right_shift(16, 2) == 4
        assert safe_right_shift(100, 3) == 12
        assert safe_right_shift(1, 1) == 0
    
    def test_safe_right_shift_negative_shift_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            safe_right_shift(16, -2)


class TestBitExtraction:
    """Test bit extraction and packing."""
    
    def test_extract_bits_basic(self):
        # Extract bits from 0b11010110 (214)
        value = 214  # 0b11010110
        assert extract_bits(value, 0, 3) == 6  # 0b110
        assert extract_bits(value, 3, 2) == 2  # 0b10
        assert extract_bits(value, 5, 3) == 6  # 0b110
    
    def test_extract_bits_full_width(self):
        value = 255  # 0b11111111
        assert extract_bits(value, 0, 8) == 255
    
    def test_extract_bits_zero_count(self):
        assert extract_bits(123, 5, 0) == 0
    
    def test_extract_bits_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            extract_bits(100, -1, 3)
        
        with pytest.raises(ValueError, match="non-negative"):
            extract_bits(100, 2, -1)
    
    def test_pack_bits_basic(self):
        # Pack [5, 3, 1] with widths [3, 2, 1]
        result = pack_bits([5, 3, 1], [3, 2, 1])
        # LSB-first: 101 + (11<<3) + (1<<5) = 0b111101 = 61
        assert result == 61
    
    def test_pack_bits_single_value(self):
        result = pack_bits([7], [3])
        assert result == 7
    
    def test_pack_bits_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="Mismatched lengths"):
            pack_bits([1, 2], [3])
    
    def test_pack_bits_lsb_first(self):
        # values 1(3b), 2(3b), 3(2b) → 1 + (2<<3) + (3<<6) = 1 + 16 + 192 = 209
        assert pack_bits([1,2,3], [3,3,2]) == 209
    
    def test_pack_bits_value_too_large_raises(self):
        with pytest.raises(ValueError, match="does not fit"):
            pack_bits([8], [3])  # 8 needs 4 bits, only 3 provided
    
    def test_pack_bits_zero_width_raises(self):
        with pytest.raises(ValueError, match="positive"):
            pack_bits([1], [0])


class TestIntegerMath:
    """Test integer-only mathematical functions."""
    
    def test_integer_log2_powers_of_2(self):
        assert integer_log2(1) == 0
        assert integer_log2(2) == 1
        assert integer_log2(4) == 2
        assert integer_log2(8) == 3
        assert integer_log2(1024) == 10
    
    def test_integer_log2_non_powers(self):
        assert integer_log2(3) == 1  # floor(log2(3))
        assert integer_log2(5) == 2  # floor(log2(5))
        assert integer_log2(7) == 2  # floor(log2(7))
        assert integer_log2(15) == 3  # floor(log2(15))
    
    def test_integer_log2_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            integer_log2(0)
    
    def test_integer_log2_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            integer_log2(-5)
    
    def test_next_power_of_2_already_power(self):
        assert next_power_of_2(1) == 1
        assert next_power_of_2(2) == 2
        assert next_power_of_2(4) == 4
        assert next_power_of_2(1024) == 1024
    
    def test_next_power_of_2_not_power(self):
        assert next_power_of_2(3) == 4
        assert next_power_of_2(5) == 8
        assert next_power_of_2(100) == 128
        assert next_power_of_2(1000) == 1024
    
    def test_next_power_of_2_zero_or_negative(self):
        assert next_power_of_2(0) == 1
        assert next_power_of_2(-5) == 1
    
    def test_next_power_of_2_bit_limit(self):
        with pytest.raises(OverflowError, match="max bit-length"):
            next_power_of_2(15, max_bits=3)  # next_power_of_2(15) = 16, needs 5 bits > 3


class TestFloatGuardIntegration:
    """Test that all clf_int functions properly reject floats."""
    
    def test_leb128_encode_rejects_float(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            leb128_encode(3.14)
    
    def test_pad_to_byte_rejects_float(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            pad_to_byte(3.5)
    
    def test_safe_left_shift_rejects_float(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            safe_left_shift(5.0, 2)
        
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            safe_left_shift(5, 2.0)
    
    def test_extract_bits_rejects_float(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            extract_bits(100.0, 2, 3)
    
    def test_pack_bits_rejects_float(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            pack_bits([1.5, 2], [3, 2])
        
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            pack_bits([1, 2], [3.0, 2])
    
    def test_integer_log2_rejects_float(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            integer_log2(8.0)
    
    def test_next_power_of_2_rejects_float(self):
        with pytest.raises(ValueError, match="Non-integer numeric detected"):
            next_power_of_2(7.5)
