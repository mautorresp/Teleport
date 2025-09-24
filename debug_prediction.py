"""
CLF Bijection Debug - Find the prediction mismatch source
"""

import math

def leb_len(n: int) -> int:
    """LEB128 byte length"""
    if n == 0:
        return 1
    length = 0
    while n > 0:
        length += 1
        n >>= 7
    return length

def header_bits(L: int) -> int:
    """H(L) = 16 + 8*leb_len(8*L)"""
    return 16 + 8 * leb_len(8 * L)

def pad_to_byte(x: int) -> int:
    """Padding to next byte boundary"""
    return (8 - (x % 8)) % 8

def end_bits(bitpos: int) -> int:
    """END(pos) = 3 + pad_to_byte(pos+3)"""
    return 3 + pad_to_byte(bitpos + 3)

def caus_stream_bits(op: int, params, L: int) -> int:
    """C_CAUS = 3 + 8*leb_len(op) + Σ 8*leb_len(param_i) + 8*leb_len(L)"""
    cost = 3 + 8 * leb_len(op)  
    for param in params:
        cost += 8 * leb_len(param)
    cost += 8 * leb_len(L)
    return cost

def pack_canonical_seed(S: bytes) -> int:
    """Pack bytes into integer"""
    if len(S) == 0:
        return 0
    K = 0
    for i, byte in enumerate(S):
        K |= (byte << (8 * i))
    return K

# Debug test case b'A'
S = b'A'
L = len(S)
print(f"Input: {S!r}, L = {L}")

# Compute seed
K = pack_canonical_seed(S)
print(f"K = {K} (0x{K:x})")
print(f"leb_len(K) = {leb_len(K)}")

# Header
H = header_bits(L)
print(f"H = 16 + 8*leb_len(8*{L}) = 16 + 8*{leb_len(8*L)} = {H}")

# A path costs
OP_CBD = 1
caus_cost = caus_stream_bits(OP_CBD, [K], L)
print(f"CAUS cost = 3 + 8*leb_len({OP_CBD}) + 8*leb_len({K}) + 8*leb_len({L})")
print(f"          = 3 + 8*{leb_len(OP_CBD)} + 8*{leb_len(K)} + 8*{leb_len(L)}")
print(f"          = 3 + {8*leb_len(OP_CBD)} + {8*leb_len(K)} + {8*leb_len(L)}")
print(f"          = {caus_cost}")

end_cost = end_bits(caus_cost)
print(f"END cost = 3 + pad_to_byte({caus_cost}+3)")
print(f"         = 3 + pad_to_byte({caus_cost+3})")
print(f"         = 3 + {pad_to_byte(caus_cost+3)}")
print(f"         = {end_cost}")

A_stream = caus_cost + end_cost
print(f"A_stream = {caus_cost} + {end_cost} = {A_stream}")

# Prediction
CAUS_pred = 3 + 8 * leb_len(OP_CBD) + 8 * leb_len(K) + 8 * leb_len(L)
END_pred = end_bits(CAUS_pred)
A_stream_pred = CAUS_pred + END_pred
print(f"Predicted: CAUS={CAUS_pred}, END={END_pred}, total={A_stream_pred}")

print(f"Match: {A_stream == A_stream_pred}")