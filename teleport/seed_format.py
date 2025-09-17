"""
Seed Format (CLF)

Equations:
- LIT: encoding = [0][ULEB(len)][raw bytes]; expansion O <- O || B
- MATCH: encoding = [1][ULEB(D)][ULEB(L)]; expansion O <- O || O[|O|-D : |O|-D+L]
- CAUS: encoding = [OP_TAG][ULEB(params...)][ULEB(L)]
Costs (used later): C_LIT(len)=10*len; C_MATCH(D,L)=2+8*leb(D)+8*leb(L); C_CAUS(op,params,L)=3+8*leb(op)+8*sum(leb(pi))+8*leb(L)

Invariants:
- Public I/O restricted to integers/bytes (containers thereof).
- ULEB128 operations are called from teleport.leb_io only.
"""

from teleport.leb_io import (
    leb128_emit_single,
    leb128_parse_single_minimal,
)
from teleport.guards import assert_boundary_types

# Operator tags (fixed)
OP_LIT: int = 0
OP_MATCH: int = 1
OP_CONST: int = 2
OP_STEP: int = 3
# Additional CAUS operators
OP_LCG8: int = 4
OP_LFSR8: int = 5
OP_ANCHOR: int = 6
OP_REPEAT1: int = 7
OP_XOR_MASK8: int = 8
OP_CBD: int = 9

def emit_LIT(b: bytes) -> bytes:
    """
    Encode literal block as L single-byte LIT tokens: [OP_LIT][byte] repeated L times
    CLF Grammar: LIT(byte_val) is a single-byte literal token (no length field)
    Integer/byte semantics only; no LEB128 lengths to avoid lone 0xFF issues.
    """
    assert_boundary_types(b)
    # Domain validation: LIT length must be 1 <= L <= 10
    L = len(b)
    if L < 1 or L > 10:
        raise ValueError("LIT length > 10")
    # Emit L copies of single-byte literal tokens (no length varint)
    result = bytearray()
    for byte_val in b:
        result.append(OP_LIT)      # Fixed tag for one-byte literal
        result.append(byte_val)    # The literal byte
    return bytes(result)

def emit_MATCH(D: int, L: int) -> bytes:
    """
    Encode match block: [OP_MATCH][ULEB(D)][ULEB(L)]
    Integer/byte semantics only; ULEB minimality guaranteed by emitter.
    """
    assert_boundary_types(D, L)
    # Domain validation: D and L must be positive
    if D <= 0 or L <= 0:
        raise ValueError("MATCH parameters must be positive")
    # D and L are integers; encode with minimal ULEB
    d_enc = leb128_emit_single(D)
    l_enc = leb128_emit_single(L)
    return bytes([OP_MATCH]) + d_enc + l_enc

def emit_CAUS(op_tag: int, params: list[int], L: int) -> bytes:
    """
    Encode causal operation: [OP_TAG][ULEB(p1)][ULEB(p2)...][ULEB(L)]
    Integer/byte semantics only; ULEB minimality guaranteed by emitter.
    op_tag should be OP_CONST, OP_STEP, etc.
    """
    assert_boundary_types(op_tag, L, *params)
    # Domain validation: op_tag and L must be valid
    if op_tag < OP_CONST or L <= 0:
        raise ValueError("CAUS op_tag must be >= OP_CONST, L must be positive")
    for p in params:
        if p < 0:
            raise ValueError("CAUS params must be non-negative")
    
    # Encode: [op_tag][ULEB(p1)]...[ULEB(L)]
    result = bytes([op_tag])
    for p in params:
        result += leb128_emit_single(p)
    result += leb128_emit_single(L)
    return result

def parse_next(seed: bytes, off: int):
    """
    Parse one operator at offset 'off'.
    Returns (op:int, params:tuple, new_off:int).
    Enforces minimal ULEB and domain constraints.
    """
    assert_boundary_types(seed, off)
    # Validate offset within range to read the tag byte.
    if off < 0 or off >= len(seed):
        raise ValueError("Offset out of range")
    tag = seed[off]
    pos = off + 1

    if tag == OP_LIT:
        # CLF Grammar: LIT is a single-byte literal token (no length field)
        # Read exactly one literal byte
        if pos >= len(seed):
            raise ValueError("LIT missing literal byte")
        literal_byte = seed[pos]
        pos += 1
        # Return single-byte literal as 1-element bytes object
        return (OP_LIT, (bytes([literal_byte]),), pos)

    elif tag == OP_MATCH:
        # Read minimal ULEB for D
        D, consumed_d = leb128_parse_single_minimal(seed, pos)
        pos += consumed_d
        # Read minimal ULEB for L
        L, consumed_l = leb128_parse_single_minimal(seed, pos)
        pos += consumed_l
        # Enforce domain: positive integers
        if D <= 0 or L <= 0:
            raise ValueError("MATCH parameters must be positive")
        return (OP_MATCH, (D, L), pos)

    elif tag >= OP_CONST:
        # CAUS operations: tag encodes the actual operation
        params = []
        
        if tag == OP_CBD:
            # CBD has different structure: N followed by N literal bytes
            N, consumed_n = leb128_parse_single_minimal(seed, pos)
            pos += consumed_n
            if N < 0:
                raise ValueError("CBD length must be non-negative")
            
            params = [N]
            # Read N literal bytes
            for _ in range(N):
                if pos >= len(seed):
                    raise ValueError("CBD: unexpected end of seed")
                params.append(seed[pos])
                pos += 1
            
            return (tag, tuple(params), pos)
        
        # Standard CAUS operations with param_count + L structure
        if tag == OP_CONST:
            param_count = 1  # [b]
        elif tag == OP_STEP:  
            param_count = 2  # [start, stride]
        else:
            raise ValueError(f"Unknown CAUS operation tag: {tag}")
        
        # Read parameters
        for _ in range(param_count):
            param, consumed = leb128_parse_single_minimal(seed, pos)
            params.append(param)
            pos += consumed
        
        # Read length L
        L, consumed_l = leb128_parse_single_minimal(seed, pos)
        pos += consumed_l
        
        # Validation
        if L <= 0:
            raise ValueError("CAUS length must be positive")
        for p in params:
            if p < 0:
                raise ValueError("CAUS parameters must be non-negative")
        
        # Return params + [L] to match VM expectation
        return (tag, tuple(params + [L]), pos)

    else:
        raise ValueError("Unknown operator tag")
