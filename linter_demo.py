#!/usr/bin/env python3
"""
Comprehensive test of the enhanced no-float linter.

This script demonstrates all the improved detection capabilities.
It should be run with: python tools/no_float_lint.py linter_demo.py
"""

def main():
    print("=== Enhanced No-Float Linter Capabilities ===")
    
    # ✅ PASS: Safe integer operations
    safe_ops = {
        'addition': 5 + 3,
        'subtraction': 10 - 4, 
        'multiplication': 6 * 7,
        'floor_division': 15 // 3,
        'modulo': 17 % 5,
        'left_shift': 1 << 8,
        'right_shift': 256 >> 2,
        'bitwise_or': 5 | 3,
        'bitwise_xor': 5 ^ 3,
        'bitwise_and': 7 & 3
    }
    
    # ✅ PASS: 3-arg modular exponentiation
    modular_exp = pow(2, 10, 1000)  # Only safe form of pow()
    
    # ❌ FAIL: All these should be caught by enhanced linter
    
    # 1. Augmented assignments (NEW)
    # x /= 2    # Augmented division
    # y **= 3   # Augmented power
    
    # 2. 2-argument pow (enhanced)
    # risky_pow = pow(2, 3)  # 2-arg pow can return float
    
    # 3. Float laundering via int() (NEW)
    # laundered1 = int(3.14)      # Contains float constant  
    # laundered2 = int(10 / 3)    # Contains division
    # laundered3 = int(2 ** 0.5)  # Contains power and float
    
    # 4. Enhanced imports (NEW - expanded set)
    # import random        # Statistical functions
    # import statistics    # Can return floats
    # import decimal       # Arbitrary precision (still float-like)
    # import fractions     # Rational numbers (still not integers)
    # import time          # Can return floats
    # from decimal import Decimal
    # from fractions import Fraction
    
    # 5. Deep dotted attribute chains (NEW)
    # result = some.deep.package.math.sqrt(4)  # Would be caught now
    
    # 6. Standard risky operations (existing)
    # division = 10 / 3           # True division
    # power = 2 ** 0.5            # Power with float
    # float_const = 3.14159       # Float literal
    # complex_const = 1 + 2j      # Complex literal
    
    print("All safe operations completed!")
    print("Enhanced linter would catch 15+ categories of float contamination")

if __name__ == "__main__":
    main()
