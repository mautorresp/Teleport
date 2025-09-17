#!/usr/bin/env python3
"""Debug the DP tokenization on a small input"""

from teleport.encoder_dp import canonize_bytes_dp

def test_small():
    # Test with last few bytes of pic1.jpg
    from pathlib import Path
    data = Path("test_artifacts/pic1.jpg").read_bytes()
    
    # Last 5 bytes
    tail = data[-5:]
    print(f"Testing last 5 bytes: {tail.hex()}")
    print(f"As integers: {list(tail)}")
    
    tokens, total_bits, C_end = canonize_bytes_dp(tail, print_receipts=True)
    
    print(f"\nTokens: {len(tokens)}")
    total_len = sum(L for _, _, L in tokens)
    print(f"Total token length: {total_len}")
    print(f"Input length: {len(tail)}")
    
    if total_len != len(tail):
        print(f"❌ LENGTH MISMATCH: tokens cover {total_len}, input is {len(tail)}")
    else:
        print(f"✓ Length matches")

if __name__ == "__main__":
    test_small()
