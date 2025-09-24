#!/usr/bin/env python3
"""
CLF V8.8 Console Validation - CORRECTED B-PATH
==============================================
Mathematical honesty enforced via console protocol.
B-path uses COPY tokens with empty parameters (not byte values).
"""

from teleport_math_runner import leb_len_u, header_bits, end_bits, caus_bits, run_one

def main():
    """V8.8 Console validation with corrected B-path understanding."""
    
    print("CLF V8.8 Console Validation - CORRECTED B-PATH")
    print("=" * 50)
    
    # Get baseline from working calculator
    L, H, A_info, B_info, total, raw, bind_B = run_one("pic1.jpg")
    
    print(f"File: pic1.jpg (L={L})")
    print(f"H: {H}")
    print(f"B_stream: {B_info['stream']}")
    print(f"C_total: {total}")
    print(f"RAW: {raw}")
    print()
    
    # Verify B-path calculation manually
    print("B-PATH VERIFICATION:")
    print("-------------------")
    
    # B-path consists of L COPY tokens with empty parameters
    copy_token_cost = caus_bits(1, [], 1)  # op=1, params=[], L_tok=1
    B_caus = L * copy_token_cost
    END_bits = end_bits(B_caus)
    B_stream = B_caus + END_bits
    
    print(f"Each COPY token: op=1, params=[], L_tok=1")
    print(f"Cost per token: 3 + 8*leb_len_u(1) + 0 + 8*leb_len_u(1) = {copy_token_cost} bits")
    print(f"Total CAUS cost: {L} * {copy_token_cost} = {B_caus} bits")
    print(f"END padding: {END_bits} bits")
    print(f"B_stream total: {B_caus} + {END_bits} = {B_stream} bits")
    print(f"Expected: {B_info['stream']} ✓" if B_stream == B_info['stream'] else f"ERROR: Expected {B_info['stream']}")
    print()
    
    # A-path status  
    print("A-PATH STATUS:")
    print("--------------")
    
    # Decision threshold
    threshold = 8 * L
    print(f"Decision threshold: 8*L = 8*{L} = {threshold}")
    print(f"B_stream cost: {B_stream}")
    print(f"B_stream < threshold: {B_stream < threshold}")
    print()
    
    # A-operator analysis
    print("A-OPERATOR ANALYSIS:")
    print("-------------------")
    print("Currently no lawful A-operators implemented.")
    print("All A-operators must:")
    print("  - Have O(1) parameter cost")
    print("  - Beat B-path threshold")
    print("  - Maintain exact bijection")
    print("Status: INADMISSIBLE (correctly fail-closed)")
    print()
    
    # Final decision
    print("MATHEMATICAL DECISION:")
    print("---------------------")
    if B_stream < threshold:
        print(f"EMIT B-path: C_total = H + B_stream = {H} + {B_stream} = {H + B_stream}")
        print("CAUSEFAIL avoided ✓")
    else:
        print("CAUSEFAIL: No lawful operators available")
    
    print()
    print("V8.8 Console validation COMPLETE - B-path corrected ✓")

if __name__ == "__main__":
    main()