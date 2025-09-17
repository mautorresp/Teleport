#!/usr/bin/env python3
"""
CLF CBD serialization fix
"""

from teleport.leb_io import leb128_emit_single
from teleport.seed_format import OP_CBD

def emit_CBD(N: int, literal_bytes: list[int]) -> bytes:
    """
    Emit CBD with correct format: OP_CBD + LEB128(N) + N raw bytes
    """
    if len(literal_bytes) != N:
        raise ValueError(f"CBD: expected {N} bytes, got {len(literal_bytes)}")
    
    result = bytes([OP_CBD])  # Tag
    result += leb128_emit_single(N)  # Length as LEB128
    result += bytes(literal_bytes)  # Raw bytes (no LEB128)
    
    return result

def serialize_cbd_caus(op_id: int, params: tuple, L: int) -> bytes:
    """
    Serialize CAUS certificate, handling CBD specially
    """
    if op_id == OP_CBD:
        N = params[0]
        literal_bytes = list(params[1:])
        return emit_CBD(N, literal_bytes)
    else:
        # Use standard emit_CAUS for other operations
        from teleport.seed_format import emit_CAUS
        return emit_CAUS(op_id, list(params), L)

# Test the fix
if __name__ == "__main__":
    # Test CBD serialization
    test_params = (4, 171, 205, 239, 18)
    cbd_seed = serialize_cbd_caus(9, test_params, 4)
    print(f"Fixed CBD seed: {cbd_seed.hex().upper()}")
    
    # Should be: 09 04 AB CD EF 12
    expected = bytes([0x09, 0x04, 0xAB, 0xCD, 0xEF, 0x12])
    print(f"Expected:       {expected.hex().upper()}")
    print(f"Match: {cbd_seed == expected}")
