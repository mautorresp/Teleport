#!/usr/bin/env python3
"""
Create a realistic test file to demonstrate CLF deductive composition.
This file will have segments with genuine causal structure.
"""

def create_realistic_test_file():
    """Create a file with mixed causal and complex segments"""
    
    # JPEG-like header (complex causality - needs CBD256)
    jpeg_header = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
        0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00, 0x48,
        0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43
    ])
    
    # Quantization table (structured but complex - CBD256)
    quant_table = bytes([
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    ])
    
    # Padding section (simple causality - CONST)
    padding = bytes([0x00] * 500)
    
    # Color ramp (simple causality - STEP)  
    color_ramp = bytes([(i * 2) % 256 for i in range(128)])
    
    # More padding (simple causality - CONST)
    more_padding = bytes([0xFF] * 200)
    
    # Arithmetic sequence (simple causality - STEP)
    arith_seq = bytes([(10 + i * 5) % 256 for i in range(50)])
    
    # JPEG footer (complex causality - CBD256)
    jpeg_footer = bytes([0xFF, 0xD9])
    
    # Combine all segments
    full_data = (jpeg_header + quant_table + padding + color_ramp + 
                 more_padding + arith_seq + jpeg_footer)
    
    return full_data

if __name__ == "__main__":
    test_data = create_realistic_test_file()
    
    with open('/Users/Admin/Teleport/realistic_test.dat', 'wb') as f:
        f.write(test_data)
    
    print(f"Created realistic test file: {len(test_data)} bytes")
    print("Segments:")
    print("- JPEG header: 24 bytes (complex causality)")  
    print("- Quantization table: 64 bytes (complex causality)")
    print("- Null padding: 500 bytes (CONST causality)")
    print("- Color ramp: 128 bytes (STEP causality)")
    print("- More padding: 200 bytes (CONST causality)")
    print("- Arithmetic sequence: 50 bytes (STEP causality)")
    print("- JPEG footer: 2 bytes (complex causality)")
