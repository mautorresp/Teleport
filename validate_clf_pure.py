#!/usr/bin/env python3
"""
CLF-Pure Validation: Stream grammar post-check
Validates only public opcodes (0,1,3) and minimal LEB128
"""

from pathlib import Path
from teleport.seed_vm import expand
from teleport.seed_validate import validate_teleport_stream
from teleport.spec_constants import TELEPORT_MAGIC_VERSION_BE
from teleport.encoder_dp import _scan_seed_varints_strict
import hashlib

def main():
    seed_path = "test_artifacts/pic1_canonical.bin"
    orig_path = "test_artifacts/pic1.jpg"
    
    if not Path(seed_path).exists():
        print(f"ERROR: {seed_path} not found")
        return 1
    
    seed = Path(seed_path).read_bytes()

    # 1. Stream validation
    ok, note = validate_teleport_stream(seed, TELEPORT_MAGIC_VERSION_BE)
    print("VALIDATE=", int(ok), "note=", note)

    # 2. Varint scan 
    try:
        _scan_seed_varints_strict(seed)
        print("VARINT_SCAN= PASS")
    except Exception as e:
        print("VARINT_SCAN= FAIL:", str(e))
        return 1

    # 3. Round-trip test
    S1 = Path(orig_path).read_bytes()
    try:
        S2 = expand(seed)
        eq = (S1 == S2)
        print("eq_bytes=", int(eq))
        print("sha256_orig=", hashlib.sha256(S1).hexdigest())
        print("sha256_exp =", hashlib.sha256(S2).hexdigest())
    except Exception as e:
        print("eq_bytes= 0")
        print("expand_error=", str(e))
        S2 = b""

    # 4. Opcodes audit: only 0,1,3 allowed 
    allowed = {0,1,3}
    pos = 2  # skip 2-byte magic/version (0x0003)
    
    # Skip OutputLengthBits LEB128 field
    from teleport.leb_io import leb128_parse_single_minimal
    try:
        output_len_bits, leb_len = leb128_parse_single_minimal(seed, pos)
        pos += leb_len
    except Exception:
        print("SERIALIZE_OPCODES= BAD_HEADER")
        return 1
    
    bad = None
    
    while pos < len(seed):
        tag = seed[pos]
        pos += 1
        if tag not in allowed:
            bad = tag
            break
        if tag == 0:   # LIT
            # single byte payload - no varints
            if pos >= len(seed):
                bad = "truncated_LIT"
                break
            pos += 1
        elif tag == 1: # MATCH
            # D and L varints
            from teleport.leb_io import leb128_parse_single_minimal
            try:
                D, n1 = leb128_parse_single_minimal(seed, pos)
                pos += n1
                L, n2 = leb128_parse_single_minimal(seed, pos) 
                pos += n2
            except Exception:
                bad = "bad_MATCH_varint"
                break
        elif tag == 3: # END
            break
    
    print("SERIALIZE_OPCODES=", "OK" if bad is None else f"BAD_TAG={bad}")

if __name__ == "__main__":
    main()
