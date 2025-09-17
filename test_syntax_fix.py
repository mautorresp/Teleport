#!/usr/bin/env python3
"""Test the corrected serializer and scanner functions."""

def _scan_seed_varints_strict(seed: bytes) -> None:
    """Walk seed by grammar and validate all LEB128 fields for minimality."""
    from teleport.leb_io import leb128_parse_single_minimal
    from teleport.seed_format import OP_LIT, OP_MATCH, OP_CONST, OP_STEP
    from teleport.spec_constants import TELEPORT_MAGIC_VERSION_BE
    
    i = 0
    # Check header exactly matches TELEPORT_MAGIC_VERSION_BE
    magic_bytes = TELEPORT_MAGIC_VERSION_BE.to_bytes(2, 'big')
    assert len(seed) >= 2, "Seed too short for magic/version"
    assert seed[:2] == magic_bytes, f"Bad magic/version: expected {magic_bytes.hex()}, got {seed[:2].hex()}"
    i = 2
    
    # Parse token stream
    while i < len(seed):
        tag = seed[i]
        i += 1
        
        if tag == OP_LIT:
            # CLF Grammar: LIT is [tag][single_byte] - no varints
            assert i < len(seed), f"Truncated LIT payload at {i-1}"
            i += 1  # Skip the literal byte
        elif tag == OP_MATCH:
            # Parse D and L with strict minimal LEB128
            D, dlen = leb128_parse_single_minimal(seed, i)
            i += dlen
            L, llen = leb128_parse_single_minimal(seed, i)
            i += llen
        elif tag == OP_CONST:
            # Parse b and L parameters
            b, blen = leb128_parse_single_minimal(seed, i)
            i += blen
            L, llen = leb128_parse_single_minimal(seed, i)
            i += llen
        elif tag == 0x03:  # END tag
            # END terminates stream - no parameters to parse
            break
        elif tag == OP_STEP:
            # Parse start, stride, L parameters
            start, slen = leb128_parse_single_minimal(seed, i)
            i += slen
            stride, stride_len = leb128_parse_single_minimal(seed, i)
            i += stride_len
            L, llen = leb128_parse_single_minimal(seed, i)
            i += llen
        else:
            # Unknown tag - this indicates stream desync
            raise AssertionError(f"Unknown tag {tag} at offset {i-1}")

def test_seed_sanity():
    """Test synthetic seed creation to verify no rogue tags"""
    from teleport.encoder_dp import serialize_tokens_to_seed
    from teleport.spec_constants import TELEPORT_MAGIC_VERSION_BE
    
    # seed = [header][LIT 'A'][MATCH D=255,L=255]
    tokens = [('LITRUN',(0x41,1),1), ('MATCH',(255,),255)]
    try:
        seed = serialize_tokens_to_seed(tokens, N=256, magic_be=TELEPORT_MAGIC_VERSION_BE)
        _scan_seed_varints_strict(seed)
        print('SEED_OK=1')
        print(f'Seed: {seed.hex()}')
        print(f'Length: {len(seed)} bytes')
        
        # Verify structure manually
        print(f'Magic: {seed[:2].hex()}')
        pos = 2
        print(f'LIT tag: {seed[pos]:02x}, literal: {seed[pos+1]:02x}')
        pos += 2  
        print(f'MATCH tag: {seed[pos]:02x}')
        print(f'D (255): {seed[pos+1:pos+3].hex()} = {255}')
        print(f'L (255): {seed[pos+3:pos+5].hex()} = {255}')
        
    except Exception as e:
        print(f'SEED_ERROR: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_seed_sanity()
