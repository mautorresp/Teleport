"""
CLF Bit-Exact Serializer
Emits exactly the bits specified by cost formulas - no extra bytes.
"""

def serialize_tokens_bit_exact(tokens, total_bits):
    """
    Serialize tokens to bit-exact CLF stream.
    
    Rules (matching cost formulas exactly):
    - LIT(b): TAG_LIT (2 bits) + b (8 bits) = 10 bits total
    - MATCH(D,L): TAG_MATCH (2 bits) + ULEB(D) + ULEB(L) = 2 + 8*leb(D) + 8*leb(L) bits
    - END: TAG_END (3 bits) + pad to byte boundary
    
    Tags:
    - LIT = 00 (2 bits)
    - MATCH = 01 (2 bits) 
    - END = 111 (3 bits)
    """
    from teleport.leb_io import leb128_emit_single
    
    # Build bit stream
    bits = []
    
    for kind, params, L in tokens:
        if kind == "LIT":
            b, run_len = params
            # Emit run_len copies of [00][bbbbbbbb]
            for _ in range(run_len):
                bits.extend([0, 0])  # TAG_LIT = 00
                # Add 8 bits for byte value
                for i in range(7, -1, -1):
                    bits.append((b >> i) & 1)
                    
        elif kind == "MATCH":
            D = params[0]
            # TAG_MATCH = 01 (2 bits)
            bits.extend([0, 1])
            
            # ULEB(D) - exactly 8*leb(D) bits  
            d_bytes = leb128_emit_single(D)
            for byte_val in d_bytes:
                for i in range(7, -1, -1):
                    bits.append((byte_val >> i) & 1)
            
            # ULEB(L) - exactly 8*leb(L) bits
            l_bytes = leb128_emit_single(L)
            for byte_val in l_bytes:
                for i in range(7, -1, -1):
                    bits.append((byte_val >> i) & 1)
                    
        elif kind == "CAUS":
            # TAG_CAUS = 10 (2 bits) - matching OP_CAUS=2
            bits.extend([1, 0])
            
            # ULEB(op_id) - first parameter is operation ID
            op_id = params[0]
            op_bytes = leb128_emit_single(op_id)
            for byte_val in op_bytes:
                for i in range(7, -1, -1):
                    bits.append((byte_val >> i) & 1)
            
            # ULEB(param_i) for each parameter after op_id
            for param in params[1:]:
                param_bytes = leb128_emit_single(param)
                for byte_val in param_bytes:
                    for i in range(7, -1, -1):
                        bits.append((byte_val >> i) & 1)
                        
            # ULEB(L) - length of data covered
            l_bytes = leb128_emit_single(L)
            for byte_val in l_bytes:
                for i in range(7, -1, -1):
                    bits.append((byte_val >> i) & 1)
                    
        else:
            raise AssertionError(f"Unsupported token: {kind}")
    
    # END tag
    bits.extend([1, 1, 1])  # TAG_END = 111
    
    # Pad to byte boundary
    while len(bits) % 8 != 0:
        bits.append(0)
    
    # Convert to bytes
    seed = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bits):
                byte_val |= bits[i + j] << (7 - j)
        seed.append(byte_val)
    
    return bytes(seed)

def expand_bit_exact(seed: bytes) -> bytes:
    """
    Expand bit-exact CLF stream.
    Mirrors the serialization rules exactly.
    """
    from teleport.leb_io import leb128_parse_single_minimal
    
    # Convert to bit stream
    bits = []
    for byte_val in seed:
        for i in range(7, -1, -1):
            bits.append((byte_val >> i) & 1)
    
    output = bytearray()
    pos = 0
    
    while pos < len(bits):
        # Read tag
        if pos + 2 <= len(bits) and bits[pos:pos+2] == [0, 0]:
            # LIT: read next 8 bits
            pos += 2
            if pos + 8 > len(bits):
                break
            byte_val = 0
            for i in range(8):
                byte_val |= bits[pos + i] << (7 - i)
            output.append(byte_val)
            pos += 8
            
        elif pos + 2 <= len(bits) and bits[pos:pos+2] == [0, 1]:
            # MATCH: read ULEB(D) and ULEB(L) 
            pos += 2
            
            # Read ULEB(D) bit by bit
            D = 0
            shift = 0
            while pos < len(bits):
                # Read next 8 bits as a byte
                if pos + 8 > len(bits):
                    break
                byte_val = 0
                for i in range(8):
                    byte_val |= bits[pos + i] << (7 - i)
                pos += 8
                
                # Process LEB128 byte
                D |= (byte_val & 0x7F) << shift
                shift += 7
                if (byte_val & 0x80) == 0:
                    break
            
            # Read ULEB(L) bit by bit  
            L = 0
            shift = 0
            while pos < len(bits):
                # Read next 8 bits as a byte
                if pos + 8 > len(bits):
                    break
                byte_val = 0
                for i in range(8):
                    byte_val |= bits[pos + i] << (7 - i)
                pos += 8
                
                # Process LEB128 byte
                L |= (byte_val & 0x7F) << shift
                shift += 7
                if (byte_val & 0x80) == 0:
                    break
            
            # Copy match
            if D > len(output) or L < 1:
                raise ValueError(f"Invalid MATCH: D={D}, L={L}, output_len={len(output)}")
            
            start = len(output) - D
            for i in range(L):
                output.append(output[start + (i % D)])
        elif pos + 2 <= len(bits) and bits[pos:pos+2] == [1, 0]:
            # CAUS: read op_id, parameters, and L
            pos += 2
            
            # Read ULEB(op_id)
            op_id = 0
            shift = 0
            while pos < len(bits):
                if pos + 8 > len(bits):
                    break
                byte_val = 0
                for i in range(8):
                    byte_val |= bits[pos + i] << (7 - i)
                pos += 8
                
                op_id |= (byte_val & 0x7F) << shift
                shift += 7
                if (byte_val & 0x80) == 0:
                    break
            
            # For now, implement CONST and STEP generators
            if op_id == 0:  # CONST
                # Read parameter: constant byte value
                b = 0
                shift = 0
                while pos < len(bits):
                    if pos + 8 > len(bits):
                        break
                    byte_val = 0
                    for i in range(8):
                        byte_val |= bits[pos + i] << (7 - i)
                    pos += 8
                    
                    b |= (byte_val & 0x7F) << shift
                    shift += 7
                    if (byte_val & 0x80) == 0:
                        break
                
                # Read L
                L = 0
                shift = 0
                while pos < len(bits):
                    if pos + 8 > len(bits):
                        break
                    byte_val = 0
                    for i in range(8):
                        byte_val |= bits[pos + i] << (7 - i)
                    pos += 8
                    
                    L |= (byte_val & 0x7F) << shift
                    shift += 7
                    if (byte_val & 0x80) == 0:
                        break
                
                # Generate CONST pattern
                for _ in range(L):
                    output.append(b)
                    
            elif op_id == 1:  # STEP
                # Read a parameter
                a = 0
                shift = 0
                while pos < len(bits):
                    if pos + 8 > len(bits):
                        break
                    byte_val = 0
                    for i in range(8):
                        byte_val |= bits[pos + i] << (7 - i)
                    pos += 8
                    
                    a |= (byte_val & 0x7F) << shift
                    shift += 7
                    if (byte_val & 0x80) == 0:
                        break
                        
                # Read d parameter  
                d = 0
                shift = 0
                while pos < len(bits):
                    if pos + 8 > len(bits):
                        break
                    byte_val = 0
                    for i in range(8):
                        byte_val |= bits[pos + i] << (7 - i)
                    pos += 8
                    
                    d |= (byte_val & 0x7F) << shift
                    shift += 7
                    if (byte_val & 0x80) == 0:
                        break
                
                # Read L
                L = 0
                shift = 0
                while pos < len(bits):
                    if pos + 8 > len(bits):
                        break
                    byte_val = 0
                    for i in range(8):
                        byte_val |= bits[pos + i] << (7 - i)
                    pos += 8
                    
                    L |= (byte_val & 0x7F) << shift
                    shift += 7
                    if (byte_val & 0x80) == 0:
                        break
                
                # Generate STEP pattern: S[i] = (a + i*d) mod 256
                for i in range(L):
                    val = (a + i * d) % 256
                    output.append(val)
                
        elif pos + 3 <= len(bits) and bits[pos:pos+3] == [1, 1, 1]:
            # END: stop
            break
            
        else:
            # Unknown tag or EOF
            break
    
    return bytes(output)

if __name__ == "__main__":
    # Test bit-exact serialization
    print("Testing bit-exact CLF serialization...")
    
    # Simple test: "AAA" should use MATCH
    from teleport.encoder_dp import canonize_bytes_dp
    
    data = b"AAA"
    tokens, total_bits, C_end = canonize_bytes_dp(data, print_receipts=True)
    
    print(f"\nMathematical costs:")
    print(f"C_stream (calculated): {total_bits}")
    
    # Serialize bit-exact
    seed = serialize_tokens_bit_exact(tokens, total_bits)
    print(f"Seed size: {len(seed)} bytes = {len(seed) * 8} bits")
    
    # Verify the fundamental invariant
    bits_on_disk = len(seed) * 8
    print(f"\n🔑 FUNDAMENTAL INVARIANT:")
    print(f"8 × len(seed) = {bits_on_disk}")
    print(f"C_stream      = {total_bits}")
    print(f"MATCHES:      {bits_on_disk == total_bits}")
    
    # Test expansion
    try:
        expanded = expand_bit_exact(seed)
        print(f"\nRound-trip: {data} → {len(seed)}B seed → {expanded}")
        print(f"SUCCESS: {expanded == data}")
    except Exception as e:
        print(f"Expand error: {e}")
