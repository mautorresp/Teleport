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
            # CAUS_CBD: Canonical Binary Decomposition - literal byte storage
            if len(params) < 1:
                raise ValueError("CBD requires at least 1 parameter (N)")
            N = params[0]
            if N < 0:
                raise ValueError("CBD length must be non-negative")
            if len(params) != N + 1:
                raise ValueError(f"CBD expects {N+1} parameters, got {len(params)}")
            # Extract literal bytes and emit exactly
            literal_bytes = params[1:N+1]
            for b in literal_bytes:
                if not (0 <= b <= 255):
                    raise ValueError(f"CBD byte must be 0..255, got {b}")
                out.append(b)

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
