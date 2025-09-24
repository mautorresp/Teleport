#!/usr/bin/env python3
"""Debug LEB7 roundtrip issues"""

# Copy just the essential functions to test them in isolation
def _bitlen_base256_mv(mv):
    """Compute bit length of big-endian integer in memoryview without constructing the integer"""
    for i, byte in enumerate(mv):
        if byte != 0:
            # First non-zero byte found at position i
            leading_zeros = 0
            for bit_pos in range(7, -1, -1):
                if (byte >> bit_pos) & 1:
                    break
                leading_zeros += 1
            
            remaining_bytes = len(mv) - i - 1
            return (remaining_bytes * 8) + (8 - leading_zeros)
    
    # All bytes are zero
    return 0

def emit_cbd_param_leb7_from_bytes(mv):
    """Emit LEB128 encoding of K where K is the base-256 integer"""
    # 1) Compute exact bitlen_K directly
    bitlen = _bitlen_base256_mv(mv)
    if bitlen == 0:
        return b'\x00'

    # 2) Produce 7-bit groups MSB→LSB by scanning bits across mv
    groups = []
    acc = 0
    acc_bits = 0
    for byte in mv:                # big-endian → feed MSB-first into acc
        for k in range(7, -1, -1): # bits 7..0
            acc = (acc << 1) | ((byte >> k) & 1)
            acc_bits += 1
            if acc_bits == 7:
                groups.append(acc)
                acc = 0
                acc_bits = 0
    if acc_bits:                   # leftover bits
        groups.append(acc << (7 - acc_bits))

    # Trim to exact groups needed - but keep the SIGNIFICANT groups
    needed = (bitlen + 6) // 7
    if len(groups) > needed:
        # Keep the LAST 'needed' groups (they contain the significant bits)
        groups = groups[-needed:]
    elif len(groups) < needed:
        groups = [0] * (needed - len(groups)) + groups

    # 3) LEB128: set continuation bit for all but last group
    out = bytearray()
    for i, g in enumerate(groups):
        byte7 = g & 0x7F
        if i < len(groups) - 1:
            out.append(0x80 | byte7)
        else:
            out.append(byte7)
    return bytes(out)

def expand_cbd256_from_leb7(leb7_bytes, L):
    """Decode LEB7 back to original bytes"""
    # 1) Extract ALL 7-bit groups
    groups = [(b & 0x7F) for b in leb7_bytes]

    # 2) Stitch back into a MSB-first bitstream
    bitbuf = bytearray()
    acc = 0
    acc_bits = 0

    def _flush_bit(bit):
        nonlocal acc, acc_bits
        acc = (acc << 1) | bit
        acc_bits += 1
        if acc_bits == 8:
            bitbuf.append(acc)
            acc = 0
            acc_bits = 0

    for g in groups:
        # each group holds 7 bits, MSB-first
        for k in range(6, -1, -1):
            _flush_bit((g >> k) & 1)

    # If we ended mid-byte, left-pad that last byte with zeros (MSB side)
    if acc_bits > 0:
        acc <<= (8 - acc_bits)
        bitbuf.append(acc)

    # Keep the first L bytes (drop trailing zero-padded tail if present)
    if len(bitbuf) < L:
        return b"\x00" * (L - len(bitbuf)) + bytes(bitbuf)
    else:
        return bytes(bitbuf[:L])

# Test the specific failing case with detailed debug
if __name__ == "__main__":
    # Focus on the failing case
    val = 0x03  # 00000011
    seg = bytes([val])
    print(f"=== DETAILED DEBUG: {seg.hex()} = {val:08b} ===")
    
    mv = memoryview(seg)
    bitlen = _bitlen_base256_mv(mv)
    print(f"Bit length: {bitlen}")
    
    # Manual encoding trace
    print("\nENCODING:")
    groups = []
    acc = 0
    acc_bits = 0
    
    byte = val  # 0x03 = 00000011
    for k in range(7, -1, -1):  # bits 7..0
        bit = (byte >> k) & 1
        acc = (acc << 1) | bit
        acc_bits += 1
        print(f"  Bit {k}: {bit}, acc: {acc:07b} ({acc}), acc_bits: {acc_bits}")
        if acc_bits == 7:
            groups.append(acc)
            print(f"    -> Group added: {acc:07b} ({acc})")
            acc = 0
            acc_bits = 0
    
    if acc_bits:  # leftover bits
        final_acc = acc << (7 - acc_bits)
        groups.append(final_acc)
        print(f"  Final group: {final_acc:07b} ({final_acc})")
    
    print(f"Raw groups: {groups}")
    
    # Trimming
    needed = (bitlen + 6) // 7
    print(f"Needed groups: {needed}")
    if len(groups) > needed:
        print(f"Trimming from {groups} to {groups[:needed]}")
        groups = groups[:needed]
    
    print(f"Final groups: {groups}")
    
    # LEB128 encoding
    leb_bytes = []
    for i, g in enumerate(groups):
        byte7 = g & 0x7F
        if i < len(groups) - 1:
            leb_bytes.append(0x80 | byte7)
        else:
            leb_bytes.append(byte7)
    
    leb = bytes(leb_bytes)
    print(f"LEB7 result: {leb.hex()}")
    
    # DECODING
    print(f"\nDECODING LEB7 {leb.hex()}:")
    decode_groups = [(b & 0x7F) for b in leb]
    print(f"Decode groups: {decode_groups}")
    
    bitbuf = bytearray()
    acc = 0
    acc_bits = 0
    
    print("Bit reconstruction:")
    for g in decode_groups:
        print(f"  Processing group {g:07b} ({g})")
        for k in range(6, -1, -1):
            bit = (g >> k) & 1
            acc = (acc << 1) | bit
            acc_bits += 1
            print(f"    Bit {k}: {bit}, acc: {acc:08b} ({acc}), acc_bits: {acc_bits}")
            if acc_bits == 8:
                bitbuf.append(acc)
                print(f"      -> Byte added: {acc:08b} ({acc:02x})")
                acc = 0
                acc_bits = 0
    
    if acc_bits > 0:
        acc <<= (8 - acc_bits)
        bitbuf.append(acc)
        print(f"  Final byte: {acc:08b} ({acc:02x})")
    
    print(f"Bitbuf: {bitbuf}")
    result = bytes(bitbuf[:1])  # Keep first L=1 bytes
    print(f"Final result: {result.hex()}")
    print(f"Match: {result == seg}")