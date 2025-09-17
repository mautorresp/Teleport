#!/usr/bin/env python3
"""
Seed Linter (CLF)

Validates seed domain constraints:
- LIT length must be 1 <= L <= 10
- Reports all LIT/MATCH tokens with byte offsets
- Exits 3 if any domain violations found
"""

import argparse
import sys
from teleport.seed_format import OP_LIT, OP_MATCH, parse_next

def main():
    parser = argparse.ArgumentParser(description="Lint CLF seed for domain violations")
    parser.add_argument("--seed", required=True, help="Path to seed file")
    args = parser.parse_args()

    with open(args.seed, "rb") as f:
        seed = f.read()

    violation = False
    off = 0
    n = len(seed)

    while off < n:
        try:
            op, params, new_off = parse_next(seed, off)

            if op == OP_LIT:
                (block,) = params
                L = len(block)
                print(f"LIT_off={off + (new_off - off - L)} L={L}")
                if L < 1 or L > 10:
                    print("violation=1")
                    violation = True

            elif op == OP_MATCH:
                D, L = params
                print(f"MATCH_off={off} D={D} L={L}")

            if new_off <= off:
                print("seed_parse_error=1")
                sys.exit(1)
            off = new_off

        except Exception as e:
            print(f"seed_parse_error=1 {type(e).__name__}: {e}")
            sys.exit(1)

    if violation:
        sys.exit(3)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
