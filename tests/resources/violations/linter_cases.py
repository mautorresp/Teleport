#!/usr/bin/env python3
"""
Test cases for the enhanced no-float linter.
This file should trigger multiple linter errors.
"""

# Test 1: Augmented assignments (should be caught)
x = 10
x /= 2  # Should error: augmented division
y = 5
y **= 3  # Should error: augmented power

# Test 2: pow function variations
a = pow(2, 3)  # Should error: 2-arg pow
b = pow(2, 3, 97)  # Should pass: 3-arg modular pow (integers only)
c = pow(2.0, 3)  # Should error: contains float

# Test 3: Deep dotted names
import package.sub.module
result = package.sub.module.math.sqrt(4)  # Should be caught

# Test 4: int() laundering
clean_int = int(42)  # OK
dirty1 = int(3.14)  # Should error: float constant
dirty2 = int(a / b)  # Should error: division in argument
dirty3 = int(x ** 2)  # Should error: power in argument

# Test 5: Enhanced imports (should all error)
import random
import statistics
import decimal
import fractions
import time
from decimal import Decimal
from fractions import Fraction

# Test 6: Regular operations that should pass
good_add = 5 + 3
good_mult = 4 * 7
good_floor_div = 15 // 3
good_mod = 17 % 5

# Test 7: Float constants (should error)
bad_float = 3.14159
bad_complex = 1 + 2j

# Test 8: Regular division and power (should error)
bad_div = 10 / 3
bad_pow = 2 ** 0.5
