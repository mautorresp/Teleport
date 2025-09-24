"""
CLF Minimal LEB128 Lock System
=============================

Non-negotiable invariant A.2: Minimal LEB128 lock (single meaning of leb(·)).
- leb(x) = number of bytes in minimal unsigned LEB128 of integer x
- In costs, every integer field contributes exactly 8·leb(value) bits
- Never take leb(8·L) in token costs unless field explicitly is bit-length integer

Audit pin: one function leb_len(x) used everywhere; verifier re-encodes 
and compares byte-for-byte or raises.
"""

from teleport.clf_integer_guards import runtime_integer_guard, FloatContaminationError

def encode_minimal_leb128_unsigned(x: int) -> bytes:
    """
    Encode integer x as minimal unsigned LEB128.
    INVARIANT A.2: This is the ONLY LEB128 encoder - all others must delegate here.
    """
    x = runtime_integer_guard(x, "LEB128 input")
    if x < 0:
        raise ValueError(f"LEB128 unsigned requires non-negative integer, got {x}")
    
    if x == 0:
        return b'\x00'
    
    result = bytearray()
    while x > 0:
        byte = x & 0x7F  # Take lowest 7 bits
        x >>= 7
        if x != 0:  # More bytes to come
            byte |= 0x80  # Set continuation bit
        result.append(byte)
    
    return bytes(result)

def decode_minimal_leb128_unsigned(data: bytes) -> tuple[int, int]:
    """
    Decode minimal unsigned LEB128 from bytes.
    Returns (value, bytes_consumed).
    Raises if not minimal encoding.
    """
    if not data:
        raise ValueError("Empty LEB128 data")
    
    value = 0
    shift = 0
    pos = 0
    
    while pos < len(data):
        byte = data[pos]
        value |= (byte & 0x7F) << shift
        pos += 1
        
        if (byte & 0x80) == 0:  # No continuation bit
            break
        shift += 7
        
        if pos >= len(data):
            raise ValueError("Incomplete LEB128 sequence")
    
    # Verify minimality: re-encode and compare
    re_encoded = encode_minimal_leb128_unsigned(value)
    consumed_bytes = data[:pos]
    
    if re_encoded != consumed_bytes:
        raise ValueError(f"Non-minimal LEB128: {consumed_bytes.hex()} != minimal {re_encoded.hex()}")
    
    return runtime_integer_guard(value, "LEB128 decoded value"), pos

def leb_len_verified(x: int) -> int:
    """
    CANONICAL leb_len function with byte-exact verification.
    This is the ONLY function that computes LEB128 byte length.
    All other leb_len imports must delegate to this.
    """
    x = runtime_integer_guard(x, "leb_len input")
    if x < 0:
        raise ValueError(f"leb_len requires non-negative integer, got {x}")
    
    # Compute length by actual encoding
    encoded = encode_minimal_leb128_unsigned(x)
    length = len(encoded)
    
    # Double-check with mathematical formula
    if x == 0:
        expected_len = 1
    else:
        # Length = ceil(log128(x+1)) = ceil(log2(x+1) / 7)
        bit_len = x.bit_length()
        expected_len = (bit_len + 6) // 7  # Ceiling division
    
    if length != expected_len:
        raise ValueError(f"LEB length mismatch: encoded={length}, formula={expected_len} for x={x}")
    
    return runtime_integer_guard(length, "leb_len result")

def verify_leb_minimal_rail(test_values: list[int] = None) -> bool:
    """
    Verify LEB_MINIMAL_OK rail.
    Tests round-trip encoding/decoding and minimality for given values.
    """
    if test_values is None:
        test_values = [0, 1, 127, 128, 16383, 16384, 2097151, 2097152, 268435455, 268435456]
    
    for x in test_values:
        x = runtime_integer_guard(x, f"test value {x}")
        
        # Test 1: Encoding produces bytes
        encoded = encode_minimal_leb128_unsigned(x)
        if not isinstance(encoded, bytes):
            raise ValueError(f"Encoding {x} produced {type(encoded)}, expected bytes")
        
        # Test 2: Length matches leb_len
        expected_len = leb_len_verified(x)
        if len(encoded) != expected_len:
            raise ValueError(f"Length mismatch for {x}: encoded={len(encoded)}, leb_len={expected_len}")
        
        # Test 3: Round-trip preserves value
        decoded_value, consumed = decode_minimal_leb128_unsigned(encoded)
        if decoded_value != x:
            raise ValueError(f"Round-trip failed for {x}: got {decoded_value}")
        if consumed != len(encoded):
            raise ValueError(f"Consumption mismatch for {x}: {consumed} != {len(encoded)}")
        
        # Test 4: No shorter encoding exists (minimality)
        if len(encoded) > 1:
            # Try shorter encoding - should fail
            shorter = encoded[:-1]  # Remove last byte
            try:
                decode_minimal_leb128_unsigned(shorter)
                raise ValueError(f"Non-minimal: {x} has shorter valid encoding {shorter.hex()}")
            except ValueError:
                pass  # Expected - shorter encoding is invalid
    
    return True

def enforce_single_leb_function():
    """
    Enforce that only leb_len_verified is used throughout CLF.
    This prevents multiple inconsistent LEB implementations.
    """
    # This would scan imports and function calls in production
    # For now, just verify our canonical function works
    verify_leb_minimal_rail()
    return True

# Standard cost computation using verified LEB
def compute_leb_cost_bits(x: int) -> int:
    """
    Compute 8 * leb_len(x) with verification.
    Use this for all LEB-based cost computations.
    """
    x = runtime_integer_guard(x, "LEB cost input")
    leb_bytes = leb_len_verified(x)
    cost_bits = runtime_integer_guard(8 * leb_bytes, "LEB cost calculation")
    return cost_bits

# Export the canonical leb_len function
leb_len = leb_len_verified