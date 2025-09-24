"""
Teleport Cost Law — Integer-Exact
Rosetta:
  Equation:
    C_LIT(L)    = 10*L
    C_MATCH(D,L)= 2 + 8*leb(D) + 8*leb(L)
    C_END(p)    = 3 + pad_to_byte(p + 3)
    C_CAUS(op, params, L) = 3 + 8*leb(op) + 8*sum(leb(pi)) + 8*leb(L)
  Explanation:
    Every integer field is priced as 8 * leb(value), where leb(value) is the minimal unsigned-LEB byte count.
    Never use leb(8*L). These formulas are the only pricing rules.
  Instruction:
    Provide four pure integer functions; reject negative inputs; rely on clf_int.leb and clf_int.pad_to_byte.
"""

from typing import List
from .guards import no_float_guard, assert_integer_only
from .clf_int import leb, pad_to_byte


@no_float_guard
def cost_lit(L: int) -> int:
    """
    Cost for literal sequence of length L.
    Equation: C_LIT(L) = 10*L
    """
    assert_integer_only(L)
    if L < 0:
        raise ValueError("L must be non-negative")
    return 10 * L


@no_float_guard
def cost_match(D: int, L: int) -> int:
    """
    Cost for match with distance D and length L.
    Equation: C_MATCH(D,L) = 2 + 8*leb(D) + 8*leb(L)
    """
    assert_integer_only(D, L)
    if D < 0 or L < 0:
        raise ValueError("D and L must be non-negative")
    return 2 + 8 * leb(D) + 8 * leb(L)


@no_float_guard
def cost_end(pos_bits: int) -> int:
    """
    Cost for END token at bit position pos_bits.
    Equation: C_END(p) = 3 + pad_to_byte(p + 3)
    """
    assert_integer_only(pos_bits)
    if pos_bits < 0:
        raise ValueError("pos_bits must be non-negative")
    return 3 + pad_to_byte(pos_bits + 3)


@no_float_guard
def cost_caus(op: int, params: List[int], L: int) -> int:
    """
    Cost for causal operation with opcode op, parameters params, and length L.
    Equation: C_CAUS(op, params, L) = 3 + 8*leb(op) + 8*sum(leb(pi)) + 8*leb(L)
    """
    assert_integer_only(op, L, *params)
    if op < 0 or L < 0:
        raise ValueError("op and L must be non-negative")
    s = 0
    for p in params:
        if p < 0:
            raise ValueError("params must be non-negative")
        s += leb(p)
    return 3 + 8 * leb(op) + 8 * s + 8 * leb(L)
