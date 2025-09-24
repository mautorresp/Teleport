"""
CBD256 Serialization - CLF Mathematical Minimality
PIN-DR: Direct CBD256 serialization without K contamination
"""

from teleport.seed_format import OP_CBD256
from teleport.leb_io import leb128_emit_single

def serialize_cbd_caus(data: bytes) -> bytes:
    """
    Emit CBD with correct format: OP_CBD256 + LEB128(N) + N raw bytes
    """
    if not data:
        return bytes([OP_CBD256]) + leb128_emit_single(0)
    
    N_bytes = leb128_emit_single(len(data))
    result = bytes([OP_CBD256])  # Tag
    result += N_bytes            # Length parameter
    result += data               # Raw bytes
    return result

def parse_cbd_caus(buffer: bytes, offset: int = 0):
    """Parse CBD CAUS from buffer"""
    if offset >= len(buffer):
        return None, offset
        
    op_id = buffer[offset]
    if op_id == OP_CBD256:
        # Parse ULEB128 length
        pos = offset + 1
        length = 0
        shift = 0
        while pos < len(buffer):
            byte = buffer[pos]
            length |= (byte & 0x7F) << shift
            pos += 1
            if (byte & 0x80) == 0:
                break
            shift += 7
        
        # Extract data
        data = buffer[pos:pos + length]
        return (op_id, length, data), pos + length
    
    return None, offset
