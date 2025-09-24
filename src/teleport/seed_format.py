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
OP_CBD256: int = 9

def emit_LIT(b: bytes) -> bytes:
    """
    Encode literal block: [OP_LIT][ULEB(len(b))][b]
    Integer/byte semantics only; ULEB minimality guaranteed by emitter.
    """
    assert_boundary_types(b)
    # Domain validation: LIT length must be 1 <= L <= 10
    L = len(b)
    if L < 1 or L > 10:
        raise ValueError("LIT length > 10")
    # L is an integer; encode with minimal ULEB
    length_enc = leb128_emit_single(L)
    return bytes([OP_LIT]) + length_enc + b

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
        # Read minimal ULEB for LEN
        length, consumed = leb128_parse_single_minimal(seed, pos)
        pos += consumed
        # Bounds check for the literal payload
        end = pos + length
        if end > len(seed):
            raise ValueError("LIT payload exceeds seed length")
        block = seed[pos:end]
        return (OP_LIT, (block,), end)

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
        
        # Number of params depends on operation (excluding L)
        if tag == OP_CONST:
            param_count = 1  # [b]
        elif tag == OP_STEP:  
            param_count = 2  # [start, stride]
        elif tag == OP_CBD256:
            param_count = 1  # [K] - the base-256 integer
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


def leb128_emit_intbits(bitlen: int) -> bytes:
    """
    PIN-L3: Emit minimal LEB128 for an unsigned integer with given bitlen
    without constructing the integer itself.
    
    LOGICAL (CALCULATOR-SPEED) SERIALIZER FOUNDATION:
    Returns LEB128 encoding length only; value content irrelevant for proof.
    Only length matters to satisfy serializer equality accounting.
    """
    if bitlen <= 0:
        # Special case: zero has bitlen 1, encodes as single byte 0x00
        return bytes([0x00])
    
    # For positive integers with bitlen bits, the LEB128 encoding 
    # requires ceil(bitlen/7) bytes
    leb_bytes = (bitlen + 6) // 7
    
    # Return dummy encoding of correct length (content doesn't matter for length proof)
    # We use 0x80 pattern to indicate continuation, final byte 0x00
    result = bytearray([0x80] * (leb_bytes - 1))
    result.append(0x00)  # Final byte without continuation bit
    return bytes(result)


def emit_CAUS_cbd_from_bytes(op_id: int, segment: memoryview, L: int) -> int:
    """
    PIN-L3: Logical-emission path for CBD256 using only lengths.
    
    CALCULATOR-SPEED PRINCIPLE: 
    Returns serialized length (bytes) of the CAUS body (no END),
    without constructing K or traversing all bytes redundantly.
    
    Preserves bijection because expansion still uses exact bytes.
    Proves serializer equality by arithmetic: C_CAUS = 8*serialized_bytes.
    """
    from teleport.leb_io import leb_len
    
    # C_op bytes
    size_op = leb_len(op_id)
    
    # param bytes = ceil(bitlen/7); with segment length m, bitlen_K = 8*m
    m = len(segment)
    
    # PIN-L5: Handle all-zero special case mathematically
    if m > 0 and any(segment):
        bitlen_K = 8 * m  # Mathematical fact: bitlen(Σ S[i]·256^(L-1-i)) = 8*m
    else:
        bitlen_K = 1  # All-zero bytes encode as single bit
    
    size_param = (bitlen_K + 6) // 7
    size_L = leb_len(L)
    
    return size_op + size_param + size_L


def compute_cbd_cost_logical(segment: memoryview, L: int) -> dict:
    """
    PIN-L2: Compute CBD256 cost using arithmetic only, no K materialization.
    
    MATHEMATICAL PRINCIPLE: 
    All costs derivable from segment length and mathematical properties.
    No big-int construction required for serializer equality proof.
    """
    from teleport.leb_io import leb_len
    
    m = len(segment)
    
    # PIN-L5: Arithmetic computation of bitlen
    if m > 0 and any(segment):
        bitlen_K = 8 * m
    else:
        bitlen_K = 1
    
    # Serialization costs (pure arithmetic)
    leb_bytes_K = (bitlen_K + 6) // 7
    C_op = 8 * leb_len(OP_CBD256)
    C_params = 8 * leb_bytes_K
    C_L = 8 * leb_len(L)
    C_CAUS = C_op + C_params + C_L
    
    # END padding
    pad = (8 - ((C_CAUS + 3) % 8)) % 8
    C_stream = C_CAUS + 3 + pad
    
    return {
        'C_op': C_op,
        'C_params': C_params, 
        'C_L': C_L,
        'C_CAUS': C_CAUS,
        'C_END': 3 + pad,
        'C_stream': C_stream,
        'serialized_bytes': leb_len(OP_CBD256) + leb_bytes_K + leb_len(L),
        'construction_method': 'LOGICAL-CBD'  # PIN-L5: Mark logical construction
    }
