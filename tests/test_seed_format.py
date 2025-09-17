import pytest
import binascii

from teleport.seed_format import (
    OP_LIT, OP_MATCH,
    emit_LIT, emit_MATCH, parse_next
)

# --- LIT ---

def test_emit_LIT_hex_ABC():
    seed = emit_LIT(b"ABC")
    assert binascii.hexlify(seed).decode() == "0003414243"

def test_parse_next_LIT_roundtrip():
    s = emit_LIT(b"ABC")
    op, params, new_off = parse_next(s, 0)
    assert op == OP_LIT
    (blk,) = params
    assert blk == b"ABC"
    assert new_off == len(s)

def test_parse_next_LIT_non_minimal_length_rejected():
    # OP_LIT + non-minimal ULEB for 0 (0x80,0x00)
    bad = bytes([OP_LIT, 0x80, 0x00])
    with pytest.raises(Exception):
        parse_next(bad, 0)

# --- MATCH ---

def test_emit_MATCH_hex_3_3():
    s = emit_MATCH(3, 3)
    assert binascii.hexlify(s).decode() == "010303"

def test_parse_next_MATCH_roundtrip():
    s = emit_MATCH(3, 3)
    op, params, new_off = parse_next(s, 0)
    assert op == OP_MATCH
    D, L = params
    assert (D, L) == (3, 3)
    assert new_off == len(s)

def test_emit_MATCH_positive_domain():
    with pytest.raises(ValueError):
        emit_MATCH(0, 3)
    with pytest.raises(ValueError):
        emit_MATCH(3, 0)

def test_parse_next_MATCH_non_minimal_fields_rejected():
    # Non-minimal D (0x80,0x00), followed by minimal L=0x03
    badD = bytes([OP_MATCH, 0x80, 0x00, 0x03])
    with pytest.raises(Exception):
        parse_next(badD, 0)
    # Minimal D=0x03, non-minimal L=0x80,0x00
    badL = bytes([OP_MATCH, 0x03, 0x80, 0x00])
    with pytest.raises(Exception):
        parse_next(badL, 0)
