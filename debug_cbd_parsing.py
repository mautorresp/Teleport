#!/usr/bin/env python3
"""
Debug CBD parsing issue
"""

from teleport.seed_format import emit_CAUS, parse_next
from teleport.generators import deduce_all

def debug_cbd_parsing():
    # Test data that should trigger CBD
    test_data = bytes([0xAB, 0xCD, 0xEF, 0x12])
    print(f"Test data: {test_data.hex().upper()}")
    
    # Get CBD deduction
    result = deduce_all(test_data)
    op_id, params, reason = result
    print(f"Deduced: op_id={op_id}, params={params}")
    
    # Create CAUS seed
    caus_seed = emit_CAUS(op_id, list(params), len(test_data))
    print(f"CAUS seed: {caus_seed.hex().upper()}")
    
    # Parse step by step
    print(f"\nParsing step by step:")
    off = 0
    step = 1
    
    while off < len(caus_seed):
        print(f"Step {step}: offset={off}")
        print(f"  Remaining bytes: {caus_seed[off:].hex().upper()}")
        try:
            op, parsed_params, new_off = parse_next(caus_seed, off)
            print(f"  Parsed: op={op}, params={parsed_params}, new_off={new_off}")
            off = new_off
            step += 1
        except Exception as e:
            print(f"  ❌ Parse error: {e}")
            break

if __name__ == "__main__":
    debug_cbd_parsing()
