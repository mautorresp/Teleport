"""
Minimal LEB128 Parse/Emit

Lightweight implementation of Little Endian Base 128 encoding for
integer serialization with strict integer-only operations.
"""

from typing import Tuple, Iterator, List
from .guards import no_float_guard, assert_integer_only


class LEBError(Exception):
    """Base exception for LEB128 operations."""
    pass


class LEBOverflowError(LEBError):
    """Raised when LEB128 value is too large."""
    pass


class LEBUnderflowError(LEBError):
    """Raised when insufficient data for LEB128 decoding."""
    pass


@no_float_guard
def leb_len(n: int) -> int:
    """Minimal unsigned LEB128 byte-length of non-negative integer n."""
    # Use clf_int.leb for consistency (single source of truth)
    from .clf_int import leb
    return leb(n)


@no_float_guard
def leb128_emit_single(value: int, max_bytes: int | None = None) -> bytes:
    """
    Emit minimal unsigned LEB128.
    Resource guard: if max_bytes is given, fail when encoded length exceeds it.
    """
    assert_integer_only(value, max_bytes if max_bytes is not None else 0)
    if value < 0:
        raise ValueError("LEB128 requires non-negative integers")
    if value == 0:
        if max_bytes is not None and max_bytes < 1:
            raise LEBOverflowError("Exceeds max_bytes")
        return b"\x00"
    out = bytearray()
    while value > 0:
        byte_val = value & 0x7F
        value >>= 7
        if value > 0:
            byte_val |= 0x80
        out.append(byte_val)
        if max_bytes is not None and len(out) > max_bytes:
            raise LEBOverflowError("Exceeds max_bytes")
    return bytes(out)


@no_float_guard
def leb128_parse_single(data: bytes, offset: int = 0, max_bytes: int | None = None) -> Tuple[int, int]:
    """
    Decode unsigned LEB128 starting at offset.
    Returns (value, bytes_consumed). Raises on unterminated or resource breach.
    """
    assert_integer_only(offset, max_bytes if max_bytes is not None else 0)
    if offset < 0 or offset >= len(data):
        raise LEBUnderflowError("No data available at offset")
    value = 0
    shift = 0
    read = 0
    limit = max_bytes if max_bytes is not None else (len(data) - offset)
    while offset + read < len(data) and read < limit:
        byte_val = data[offset + read]
        read += 1
        value |= (byte_val & 0x7F) << shift
        if (byte_val & 0x80) == 0:
            return value, read  # terminated
        shift += 7
    raise LEBUnderflowError("Incomplete/unterminated LEB128 sequence")


@no_float_guard
def leb128_parse_single_minimal(data: bytes, offset: int = 0, max_bytes: int | None = None) -> Tuple[int, int]:
    """
    Decode and enforce minimal unsigned LEB128:
    minimality rule → bytes_used == leb_len(value)
    """
    v, n = leb128_parse_single(data, offset, max_bytes=max_bytes)
    if n != leb_len(v):
        raise LEBOverflowError("Non-minimal LEB128 encoding")
    return v, n


@no_float_guard
def leb128_emit_sequence(values: List[int], max_each: int | None = None, max_total_bytes: int | None = None) -> bytes:
    """
    Emit a sequence of integers as concatenated LEB128.
    """
    assert_integer_only(*values, max_each if max_each is not None else 0, max_total_bytes if max_total_bytes is not None else 0)
    out = bytearray()
    for v in values:
        part = leb128_emit_single(v, max_bytes=max_each)
        out.extend(part)
        if max_total_bytes is not None and len(out) > max_total_bytes:
            raise LEBOverflowError("Sequence exceeds max_total_bytes")
    return bytes(out)


@no_float_guard
def leb128_parse_sequence(
    data: bytes,
    count: int | None = None,
    *,
    require_minimal: bool = False,
    strict_consume_all: bool = False
) -> List[int]:
    """
    Parse a sequence of LEB128 integers.
    - require_minimal: enforce minimal encodings per integer.
    - strict_consume_all: raise if trailing bytes remain (not parsed).
    """
    if count is not None:
        assert_integer_only(count)
    values: List[int] = []
    offset = 0
    parsed = 0
    parse_one = leb128_parse_single_minimal if require_minimal else leb128_parse_single
    while offset < len(data) and (count is None or parsed < count):
        v, n = parse_one(data, offset)
        values.append(v)
        offset += n
        parsed += 1
    if strict_consume_all and offset != len(data):
        raise LEBUnderflowError("Trailing bytes after parsing sequence")
    return values


@no_float_guard
def leb128_stream_emit() -> 'LEBEmitter':
    """
    Create a streaming LEB128 emitter.
    
    Returns:
        LEBEmitter: Streaming emitter instance
    """
    return LEBEmitter()


@no_float_guard
def leb128_stream_parse(data: bytes) -> 'LEBParser':
    """
    Create a streaming LEB128 parser.
    
    Args:
        data: Byte data to parse
        
    Returns:
        LEBParser: Streaming parser instance
    """
    return LEBParser(data)


class LEBEmitter:
    """Streaming LEB128 emitter for incremental encoding."""
    
    def __init__(self):
        self._buffer = bytearray()
    
    @no_float_guard
    def emit(self, value: int) -> 'LEBEmitter':
        """Emit a single value and return self for chaining."""
        assert_integer_only(value)
        
        encoded = leb128_emit_single(value)
        self._buffer.extend(encoded)
        return self
    
    @no_float_guard
    def emit_many(self, values: List[int]) -> 'LEBEmitter':
        """Emit multiple values and return self for chaining."""
        assert_integer_only(*values)
        
        for value in values:
            self.emit(value)
        return self
    
    def get_bytes(self) -> bytes:
        """Get the accumulated encoded bytes."""
        return bytes(self._buffer)
    
    def clear(self) -> 'LEBEmitter':
        """Clear the buffer and return self for chaining."""
        self._buffer.clear()
        return self
    
    def byte_count(self) -> int:
        """Get current buffer size in bytes."""
        return len(self._buffer)


class LEBParser:
    """Streaming LEB128 parser for incremental decoding."""
    
    def __init__(self, data: bytes, *, require_minimal: bool = False):
        self._data = data
        self._offset = 0
        self._require_minimal = require_minimal
    
    @no_float_guard
    def parse(self) -> int:
        """
        Parse the next LEB128 value.
        
        Returns:
            int: Parsed value
            
        Raises:
            LEBUnderflowError: If no more data available
        """
        if self._offset >= len(self._data):
            raise LEBUnderflowError("No more data to parse")
        
        if self._require_minimal:
            value, consumed = leb128_parse_single_minimal(self._data, self._offset)
        else:
            value, consumed = leb128_parse_single(self._data, self._offset)
        self._offset += consumed
        return value
    
    @no_float_guard
    def parse_many(self, count: int) -> List[int]:
        """Parse multiple values."""
        assert_integer_only(count)
        return [self.parse() for _ in range(count)]
    
    def has_more(self) -> bool:
        """Check if more data is available."""
        return self._offset < len(self._data)
    
    def remaining_bytes(self) -> int:
        """Get number of unparsed bytes."""
        return len(self._data) - self._offset
    
    def get_position(self) -> int:
        """Get current parsing position."""
        return self._offset
    
    @no_float_guard
    def seek(self, offset: int) -> None:
        """Set parsing position."""
        assert_integer_only(offset)
        
        if offset < 0 or offset > len(self._data):
            raise ValueError("Invalid seek offset")
        self._offset = offset


@no_float_guard
def leb128_size_exact(max_value: int) -> int:
    """
    Calculate exact maximum LEB128 size for values up to max_value.
    
    Args:
        max_value: Maximum value to encode
        
    Returns:
        int: Maximum bytes needed
    """
    assert_integer_only(max_value)
    
    if max_value < 0:
        raise ValueError("Max value must be non-negative")
    
    if max_value == 0:
        return 1
    
    # Calculate bits needed, then convert to LEB128 bytes
    bits_needed = max_value.bit_length()
    leb_bytes = (bits_needed + 6) // 7  # Ceiling division
    return leb_bytes


@no_float_guard
def leb128_decode_sequence(data: bytes, *, require_minimal: bool = False) -> List[int]:
    """Decode a complete sequence of LEB128 values."""
    parser = LEBParser(data, require_minimal=require_minimal)
    values = []
    
    while parser.has_more():
        values.append(parser.parse())
    
    return values


@no_float_guard 
def leb128_encode_sequence(values: List[int], *, max_bytes: int = 10) -> bytes:
    """Encode a sequence of integers as LEB128."""
    assert_integer_only(*values, max_bytes)
    if max_bytes <= 0:
        raise LEBOverflowError("max_bytes must be positive")
    
    result = []
    for value in values:
        result.append(leb128_emit_single(value, max_bytes=max_bytes))
    return b''.join(result)


@no_float_guard
def leb128_validate(data: bytes, *, require_minimal: bool = False) -> Tuple[bool, str]:
    """
    Validate LEB128 byte sequence.
    
    Args:
        data: Bytes to validate
        require_minimal: If True, require canonical minimal encoding
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not data:
        return True, ""
    
    try:
        # Attempt to parse all values
        leb128_decode_sequence(data, require_minimal=require_minimal)
        return True, ""
    except LEBError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"
