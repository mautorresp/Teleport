"""
Strict Teleport Seed Validation
Validates byte streams against exact Teleport grammar.
Pure mathematical grammar validation only.
"""

from typing import Tuple
from teleport.leb_io import leb128_parse_single_minimal
from teleport.clf_int import pad_to_byte

class SeedValidationError(Exception):
    pass

def _read_u16_be(b: bytes, off: int) -> Tuple[int, int]:
    """Read 16-bit big-endian integer"""
    if off + 2 > len(b):
        raise SeedValidationError("truncated: magic+version")
    return (b[off] << 8) | b[off + 1], off + 2

def _bits_eof_ok(total_bits: int, end_pos_bits: int) -> bool:
    """Check if END position matches exact EOF"""
    return end_pos_bits == total_bits

def validate_teleport_stream(b: bytes, magic_be: int) -> Tuple[bool, str]:
    """
    Returns (ok, note). ok==True IFF the entire byte array is a valid Teleport stream
    per spec: header(valid) + tokens(valid) + END + exact zero-pad + immediate EOF.
    On failure, note is the first precise reason.
    """
    try:
        # 1) Header: Magic+Version (16 bits, big-endian)
        mv, off = _read_u16_be(b, 0)
        if mv != magic_be:
            return (False, f"bad magic/version: 0x{mv:04x} != 0x{magic_be:04x}")
        
        # 2) OutputLengthBits as minimal ULEB128u
        try:
            val, leb_len = leb128_parse_single_minimal(b, off)
        except:
            return (False, "invalid LEB length")
        
        if leb_len <= 0:
            return (False, "invalid LEB length")
        
        off += leb_len
        if val == 0 or (val % 8) != 0:
            return (False, f"OutputLengthBits invalid: {val} (must be 8*N, N>=1)")
        
        N = val // 8
        
        # Header length in bits
        H_bits = 16 + 8 * leb_len
        
        # Header must be byte-aligned (it always is: H_bits % 8 == 0)
        if (H_bits % 8) != 0:
            return (False, f"header not byte-aligned: {H_bits} bits")

        # 3) Token parse (exact grammar)
        bitpos = H_bits  # count in bits
        p_out = 0        # emitted bytes count (must equal N at END)
        
        def _read_bits(nbits: int) -> int:
            nonlocal bitpos
            # For tag fields (1-2 bits), read from current bit position
            byte_ix = bitpos // 8
            bit_in_byte = bitpos % 8
            
            if byte_ix >= len(b):
                raise SeedValidationError("unexpected EOF while reading bits")
            
            # Read bits within current byte
            if bit_in_byte + nbits <= 8:
                cur = b[byte_ix]
                mask = ((1 << nbits) - 1) << (8 - bit_in_byte - nbits)
                val = (cur & mask) >> (8 - bit_in_byte - nbits)
                bitpos += nbits
                return val
            else:
                # Cross-byte bit reads not supported for tag fields
                raise SeedValidationError("cross-byte bit read not allowed for tag fields")

        def _align_to_next_byte_after_END_and_must_EOF():
            nonlocal bitpos
            # END: positioned just after 3 bits of END (tag '11' + disc '0')
            pad = pad_to_byte(bitpos)
            
            # Check pad bits are all zero
            if pad > 0:
                byte_ix = bitpos // 8
                bit_in_byte = bitpos % 8
                
                if byte_ix >= len(b):
                    raise SeedValidationError("EOF inside END padding")
                
                # Check remaining bits in current byte are zero
                mask = (1 << pad) - 1
                val = b[byte_ix] & mask
                if val != 0:
                    raise SeedValidationError("non-zero pad bits after END")
                
                bitpos += pad
            
            # After padding, must be exactly at EOF
            return _bits_eof_ok(len(b) * 8, bitpos)

        # Token parsing loop
        while True:
            # Read 2-bit tag
            tag = _read_bits(2)
            
            if tag == 0b00:
                # LIT: then one data byte
                if bitpos % 8 != 0:
                    raise SeedValidationError("LIT not byte-aligned")
                
                byte_ix = bitpos // 8
                if byte_ix >= len(b):
                    raise SeedValidationError("truncated LIT data")
                
                p_out += 1
                bitpos += 8
                
                if p_out > N:
                    return (False, f"LIT overrun: p_out={p_out} > N={N}")
            
            elif tag == 0b01:
                # MATCH(D,L): both minimal LEB128u
                if bitpos % 8 != 0:
                    raise SeedValidationError("MATCH not byte-aligned at fields")
                
                # Parse D
                try:
                    D, dlen = leb128_parse_single_minimal(b, bitpos // 8)
                except:
                    return (False, "invalid leb(D)")
                
                if dlen <= 0:
                    return (False, "invalid leb(D)")
                
                bitpos += 8 * dlen
                
                # Parse L
                try:
                    L, llen = leb128_parse_single_minimal(b, bitpos // 8)
                except:
                    return (False, "invalid leb(L)")
                
                if llen <= 0:
                    return (False, "invalid leb(L)")
                
                bitpos += 8 * llen
                
                # Legality checks
                if D < 1 or L < 3:
                    return (False, f"MATCH domain: D={D}, L={L}")
                
                if p_out - D < 0:
                    return (False, f"MATCH window crosses origin: p={p_out}, D={D}")
                
                if (p_out - D) + L > p_out:
                    return (False, f"MATCH peeks into future: p={p_out}, D={D}, L={L}")
                
                p_out += L
                
                if p_out > N:
                    return (False, f"MATCH overrun: p_out={p_out} > N={N}")
            
            elif tag == 0b10:
                return (False, "encountered INVALID tag 10")
            
            else:  # tag == 0b11
                disc = _read_bits(1)
                
                if disc == 0:
                    # END
                    if p_out != N:
                        return (False, f"END before reaching N bytes: p_out={p_out}, N={N}")
                    
                    # Pad to next byte and check EOF
                    ok = _align_to_next_byte_after_END_and_must_EOF()
                    if not ok:
                        return (False, "trailing bits after END+pad")
                    
                    return (True, "valid seed")
                
                else:
                    # CAUS: conservatively reject in autodetect mode
                    return (False, "CAUS present but validator lacks OpSet parser; wire OpSet decode to validate fully")
    
    except SeedValidationError as e:
        return (False, f"validation error: {e}")
    except Exception as e:
        return (False, f"parse error: {e}")
