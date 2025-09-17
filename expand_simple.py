#!/usr/bin/env python3
"""
Simple byte-level CLF expander for testing
Uses only opcodes 0=LIT, 1=MATCH, 3=END
"""

def expand_simple(seed: bytes) -> bytes:
    """
    Simple expander for byte-level CLF format:
    - Magic/Version (2 bytes) 
    - OutputLengthBits (LEB128)
    - Opcodes: 0=LIT, 1=MATCH, 3=END
    """
    from teleport.leb_io import leb128_parse_single_minimal
    
    pos = 0
    
    # Skip magic/version (2 bytes)
    pos += 2
    
    # Skip OutputLengthBits 
    _, leb_len = leb128_parse_single_minimal(seed, pos)
    pos += leb_len
    
    # Expand opcodes
    output = bytearray()
    
    while pos < len(seed):
        opcode = seed[pos]
        pos += 1
        
        if opcode == 0:  # LIT
            # Read single literal byte
            if pos >= len(seed):
                raise ValueError("Truncated LIT")
            output.append(seed[pos])
            pos += 1
            
        elif opcode == 1:  # MATCH
            # Read D and L 
            D, dlen = leb128_parse_single_minimal(seed, pos)
            pos += dlen
            L, llen = leb128_parse_single_minimal(seed, pos)
            pos += llen
            
            # Copy from history
            if D > len(output) or L < 1:
                raise ValueError(f"Invalid MATCH: D={D}, L={L}, output_len={len(output)}")
            
            start = len(output) - D
            for i in range(L):
                output.append(output[start + (i % D)])
                
        elif opcode == 3:  # END
            break
            
        else:
            raise ValueError(f"Unknown opcode: {opcode}")
    
    return bytes(output)

if __name__ == "__main__":
    # Test with the canonical seed
    import sys
    from pathlib import Path
    
    seed_path = "test_artifacts/pic1_canonical.bin"
    if Path(seed_path).exists():
        seed = Path(seed_path).read_bytes()
        try:
            result = expand_simple(seed)
            print(f"Expanded {len(seed)} bytes → {len(result)} bytes")
            
            # Compare with original
            orig = Path("test_artifacts/pic1.jpg").read_bytes()
            if result == orig:
                print("✓ Round-trip SUCCESS")
            else:
                print(f"✗ Round-trip FAILED: {len(orig)} vs {len(result)} bytes")
                if len(result) != len(orig):
                    print(f"Length mismatch: expected {len(orig)}, got {len(result)}")
                else:
                    # Find first difference
                    for i, (a, b) in enumerate(zip(orig, result)):
                        if a != b:
                            print(f"First difference at byte {i}: {a} vs {b}")
                            break
        except Exception as e:
            print(f"Expand error: {e}")
    else:
        print(f"Seed file {seed_path} not found")
