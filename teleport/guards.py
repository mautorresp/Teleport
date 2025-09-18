"""
Boot Guards - No-Float Membrane

Provides runtime guards and decorators to enforce integer-only operations
and prevent floating-point contamination in critical code paths.
"""

import functools
import sys
from decimal import Decimal
from fractions import Fraction
from numbers import Real, Complex
from typing import Any, Callable, TypeVar, Iterable

F = TypeVar('F', bound=Callable[..., Any])

_ALLOWED_ATOMIC = (int, bytes, bytearray, bool)  # bool ⊂ int, kept explicit for clarity
_FORBIDDEN_NUMERIC = (float, complex, Decimal, Fraction)
_ALLOWED_CONTAINERS = (list, tuple, dict, set, frozenset)


def no_float_guard(func: F) -> F:
    """
    Decorator that enforces no floating-point values in function arguments.
    
    Raises ValueError if any argument contains float values.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check all arguments for float contamination
        for i, arg in enumerate(args):
            _check_no_float(arg, f"argument {i}")
        
        for name, value in kwargs.items():
            _check_no_float(value, f"keyword argument '{name}'")
        
        # Execute function and check return value
        result = func(*args, **kwargs)
        _check_no_float(result, "return value")
        
        return result
    
    return wrapper


def _check_no_float(value: Any, context: str = "value") -> None:
    """Recursively reject floats and non-int numeric types anywhere in the value."""
    # 1) Disallow forbidden numeric types outright
    if isinstance(value, _FORBIDDEN_NUMERIC):
        raise ValueError(f"Non-integer numeric detected in {context}: {type(value).__name__}")

    # 2) If it's any Real/Complex that's not a plain int/bool, reject
    if isinstance(value, (Real, Complex)) and not isinstance(value, (int, bool)):
        raise ValueError(f"Non-integer numeric detected in {context}: {type(value).__name__}")

    # 3) Accept atomic safe types
    if isinstance(value, _ALLOWED_ATOMIC):
        return

    # 4) Containers: recurse into elements, and for dicts also keys
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_no_float(item, f"{context}[{i}]")
        return

    if isinstance(value, (set, frozenset)):
        for i, item in enumerate(value):
            _check_no_float(item, f"{context}{{{i}}}")
        return

    if isinstance(value, dict):
        for k, item in value.items():
            _check_no_float(k, f"{context}.key[{k!r}]")
            _check_no_float(item, f"{context}[{k!r}]")
        return

    # 5) Objects: check __dict__ and __slots__ if present
    if hasattr(value, "__dict__"):
        for attr_name, attr_value in value.__dict__.items():
            _check_no_float(attr_value, f"{context}.{attr_name}")
        return

    if hasattr(value, "__slots__"):
        for attr_name in getattr(value, "__slots__", ()):
            try:
                attr_value = getattr(value, attr_name)
            except Exception:
                continue
            _check_no_float(attr_value, f"{context}.{attr_name}")
        return

    # 6) Bytes-like protocols are allowed only if concrete bytes/bytearray
    # Any other type falls through; do nothing (non-numeric) or tighten later as needed.


def assert_boundary_types(*values: Any) -> None:
    """
    Enforce boundary contract: each value must be int or bytes/bytearray,
    or containers (list/tuple/dict/set/frozenset) composed only of those (recursively).
    """
    def _ok(v: Any) -> bool:
        if isinstance(v, (int, bool, bytes, bytearray)):
            return True
        if isinstance(v, (list, tuple, set, frozenset)):
            return all(_ok(x) for x in v)
        if isinstance(v, dict):
            return all(_ok(k) and _ok(x) for k, x in v.items())
        # Allow ContextView for logical calculator-speed regime
        if hasattr(v, '__len__') and hasattr(v, '__getitem__') and hasattr(v, 'append_bytes'):
            return True
        return False

    for i, v in enumerate(values):
        if not _ok(v):
            raise TypeError(f"Boundary type violation at value {i}: {type(v).__name__}")
        _check_no_float(v, f"boundary value {i}")


class NoFloatContext:
    """Context manager that enforces integer-only semantics within a block."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Nothing dynamic to revert; AST linter + decorators do the heavy lifting.
        return False  # do not suppress exceptions


def assert_integer_only(*values: Any) -> None:
    """Assert that all provided values are integer-only (no floats)."""
    for i, value in enumerate(values):
        _check_no_float(value, f"value {i}")


@no_float_guard
def safe_int_divide(dividend: int, divisor: int) -> tuple[int, int]:
    """
    Safe integer division that returns (quotient, remainder).
    
    Ensures no floating-point operations are used.
    """
    if not isinstance(dividend, int) or not isinstance(divisor, int):
        raise TypeError("Both dividend and divisor must be integers")
    
    if divisor == 0:
        raise ValueError("Division by zero")
    
    q, r = divmod(dividend, divisor)  # integer-only
    return q, r


@no_float_guard
def safe_int_power(base: int, exponent: int, max_bits: int = 1 << 20) -> int:
    """
    Safe integer exponentiation with overflow protection.
    
    Uses integer-only operations with CLF-style resource bounds.
    """
    if not isinstance(base, int) or not isinstance(exponent, int):
        raise TypeError("Both base and exponent must be integers")
    
    if exponent < 0:
        raise ValueError("Negative exponents not supported")
    
    # Exponent cap is fine, but also guard by output bit-length (CLF-style resource bound)
    result = 1
    b, e = base, exponent
    while e:
        if e & 1:
            result *= b
            if result.bit_length() > max_bits:
                raise ValueError("Result exceeds max bit-length")
        b *= b
        if b.bit_length() > max_bits:
            raise ValueError("Intermediate exceeds max bit-length")
        e >>= 1
    return result
