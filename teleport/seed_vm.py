"""
Seed VM (CLF)

Equations (to be implemented in later phases):
- expand(seed): replay LIT and MATCH deterministically into bytes.
- seed_cost(seed): sum costs using teleport.costs.{cost_lit,cost_match}.

Invariants:
- Only integer/byte semantics; no floats.
- LEB128 encode/parse is used via teleport.leb_io indirectly through seed_format.parse_next.
"""

from teleport.guards import assert_boundary_types
from teleport.costs import cost_lit, cost_match, cost_end, cost_caus  # used in Phase 5
from teleport.seed_format import OP_LIT, OP_MATCH, OP_CONST, OP_STEP, parse_next  # parse_next implemented in Phase 2/3

class SeedDomainError(Exception):
    """Raised when seed violates domain constraints (integer-only behavior)."""
    pass

def expand_generator(op_id: int, params: tuple, L: int) -> bytes:
    """
    Expand a single generator (op_id, params) to produce L bytes.
    Used by deductive composition for verification.
    """
    out = bytearray()
    
    if op_id == 1:  # OP_MATCH
        # MATCH cannot be expanded without context - should not be called directly
        raise ValueError("MATCH expansion requires full context (use expand() on full seed)")
    
    elif op_id == OP_CONST:
        # CONST: repeat byte b exactly L times
        if len(params) != 1:
            raise ValueError("CONST requires exactly 1 parameter")
        b = params[0]
        if not (0 <= b <= 255):
            raise ValueError("CONST byte must be 0..255")
        out += bytes([b] * L)
    
    elif op_id == OP_STEP:
        # STEP: arithmetic sequence start + i*stride mod 256  
        if len(params) != 2:
            raise ValueError("STEP requires exactly 2 parameters")
        start, stride = params
        if not (0 <= start <= 255):
            raise ValueError("STEP start must be 0..255")
        if not (0 <= stride <= 255):
            raise ValueError("STEP stride must be 0..255")
        for i in range(L):
            out.append((start + i * stride) & 255)
    
    elif op_id == 9:  # OP_CBD
        # CBD256: Bijective base-256 decoding
        if len(params) != 1:
            raise ValueError("CBD256 requires exactly 1 parameter")
        K = params[0]
        if K < 0:
            raise ValueError("CBD256 K must be non-negative")
        
        # Domain check
        if L > 0 and K >= (256 ** L):
            raise ValueError(f"CBD256: K={K} must be < 256^{L}")
        
        # Base-256 decoding
        if L == 0:
            pass  # Empty
        else:
            temp_K = K
            bytes_to_emit = []
            
            for i in range(L):
                byte_val = temp_K % 256
                bytes_to_emit.append(byte_val)
                temp_K //= 256
            
            # Emit in reverse order (most significant first)
            for i in range(L - 1, -1, -1):
                out.append(bytes_to_emit[i])
    
    else:
        # Add support for other generators as needed
        raise ValueError(f"Unsupported generator operation: {op_id}")
    
    return bytes(out)

def expand(seed: bytes) -> bytes:
    """
    Replay operators into bytes.

    Equations:
      - LIT(block): O <- O || block
      - MATCH(D,L):  O <- O || O[|O|-D : |O|-D+L]
    Domain at replay:
      - 1 <= D <= len(O)
      - L >= 1
      - (len(O) - D + L) <= len(O)  (source must be fully within O)
    """
    assert_boundary_types(seed)

    # local mutable buffer as bytearray for efficient appends and slices
    out = bytearray()
    off = 0
    n = len(seed)

    while off < n:
        op, params, new_off = parse_next(seed, off)

        if op == OP_LIT:
            (block,) = params
            # Domain validation: LIT length must be 1 <= L <= 10
            L = len(block)
            if L < 1 or L > 10:
                raise SeedDomainError("LIT length > 10")
            # block is bytes by construction; append exact bytes
            out += block

        elif op == OP_MATCH:
            D, L = params
            # Replay-time domain checks (integers only)
            cur = len(out)
            if D < 1 or D > cur:
                raise ValueError("MATCH distance out of bounds")
            if L < 1:
                raise ValueError("MATCH length must be positive")
            # CLF MATCH semantics: copy L bytes iteratively from distance D
            # Each copied byte becomes available for subsequent copies
            for _ in range(L):
                out.append(out[-D])

        elif op == OP_CONST:
            # CAUS_CONST: repeat byte b exactly L times
            if len(params) != 2:  # [b, L]
                raise ValueError("CONST requires exactly 2 parameters")
            b, L = params
            if L < 1:
                raise ValueError("CONST length must be positive")
            if not (0 <= b <= 255):
                raise ValueError("CONST byte must be 0..255")
            # Append b exactly L times
            out += bytes([b] * L)

        elif op == OP_STEP:
            # CAUS_STEP: arithmetic sequence start + i*stride mod 256
            if len(params) != 3:  # [start, stride, L]
                raise ValueError("STEP requires exactly 3 parameters")
            start, stride, L = params
            if L < 1:
                raise ValueError("STEP length must be positive")
            if not (0 <= start <= 255):
                raise ValueError("STEP start must be 0..255")
            if not (0 <= stride <= 255):
                raise ValueError("STEP stride must be 0..255")
            # Generate arithmetic sequence
            for i in range(L):
                out.append((start + i * stride) & 255)

        elif op == 9:  # OP_CBD
            # CBD256: Bijective base-256 decoding from single integer K
            # The L parameter is parsed by parse_next for CAUS operations
            if len(params) < 2:
                raise ValueError("CBD256 requires K and L parameters")
            
            K, L = params[0], params[1]
            if K < 0 or L < 0:
                raise ValueError("CBD256 parameters K and L must be non-negative")
                
            # Domain check: K must be < 256^L
            if L > 0 and K >= (256 ** L):
                raise ValueError(f"CBD256: K={K} must be < 256^{L}")
            
            # Base-256 decoding: extract bytes from K
            if L == 0:
                # Empty file - nothing to emit
                pass
            else:
                # Extract L bytes using repeated divmod
                temp_K = K
                bytes_to_emit = []
                
                for i in range(L):
                    byte_val = temp_K % 256
                    bytes_to_emit.append(byte_val)
                    temp_K //= 256
                
                # Emit bytes in reverse order (most significant first)
                for i in range(L - 1, -1, -1):
                    out.append(bytes_to_emit[i])

        else:
            # Handle other CAUS operations or unknown tags
            if op >= OP_CONST:
                raise ValueError(f"Unsupported CAUS operation: {op}")
            else:
                raise ValueError("Unknown operator tag from parser")

        # Monotonic progress
        if new_off <= off:
            raise ValueError("Parser did not advance; seed malformed")
        off = new_off

    # return immutable bytes
    return bytes(out)

def seed_cost(seed: bytes) -> int:
    """
    Compute the cost of a seed.

    Equations:
      - C_LIT(len) = 10 * len
      - C_MATCH(D,L) = 2 + 8*leb(D) + 8*leb(L)
      - Total = sum(op costs) + C_END(len(output))
    """
    assert_boundary_types(seed)

    off = 0
    n = len(seed)
    out_len = 0
    total_cost = 0

    while off < n:
        op, params, new_off = parse_next(seed, off)

        if op == OP_LIT:
            (block,) = params
            L = len(block)
            total_cost += cost_lit(L)
            out_len += L

        elif op == OP_MATCH:
            D, L = params
            total_cost += cost_match(D, L)
            out_len += L

        elif op == OP_CONST:
            b, L = params
            total_cost += cost_caus(0, [b], L)  # OP_CONST maps to op=0
            out_len += L

        elif op == OP_STEP:
            start, stride, L = params
            total_cost += cost_caus(1, [start, stride], L)  # OP_STEP maps to op=1
            out_len += L

        else:
            if op >= OP_CONST:
                raise ValueError(f"Unsupported CAUS operation in cost calculator: {op}")
            else:
                raise ValueError("Unknown operator tag in cost calculator")

        if new_off <= off:
            raise ValueError("Parser did not advance; seed malformed")
        off = new_off

    # Add end cost (END cost uses bit position, not accumulated cost)
    output_bits = out_len * 8
    total_cost += cost_end(output_bits)
    return total_cost
