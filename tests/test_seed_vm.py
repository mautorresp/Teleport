import pytest
import binascii

from teleport.seed_format import emit_LIT, emit_MATCH
from teleport.seed_vm import expand, seed_cost

# --- Expansion (deterministic replay) ---

def test_expand_literal_then_match_ABCABC():
    s = emit_LIT(b"ABC") + emit_MATCH(3, 3)
    out = expand(s)
    assert out == b"ABCABC"
    assert out.hex() == "414243414243"

def test_expand_literal_then_two_matches_ABCABCABC():
    s = emit_LIT(b"ABC") + emit_MATCH(3, 3) + emit_MATCH(6, 3)
    out = expand(s)
    assert out == b"ABCABCABC"
    assert out.hex() == "414243414243414243"

def test_expand_invalid_distance_rejected():
    s = emit_LIT(b"ABC") + emit_MATCH(4, 1)  # D > len("ABC") == 3
    with pytest.raises(ValueError):
        expand(s)

def test_expand_source_beyond_output_rejected():
    s = emit_LIT(b"ABC") + emit_MATCH(2, 3)  # src_end exceeds current out
    with pytest.raises(ValueError):
        expand(s)

def test_expand_empty_seed():
    assert expand(b"") == b""

# --- Costs (bit-accurate) ---

def test_seed_cost_literal_AB():
    s = emit_LIT(b"AB")  # L=2 -> C_LIT(2) = 20; output=16 bits; C_END(16) = 3+pad_to_byte(19) = 3+5 = 8; total = 28
    assert seed_cost(s) == 28

def test_seed_cost_literal_plus_match_ABC_plus_333():
    # LIT "ABC" -> 30 bits; MATCH(3,3) -> 2 + 8*1 + 8*1 = 18; p=48
    # C_END(48) = 3 + pad_to_byte(51) = 3 + 5 = 8; total = 30 + 18 + 8 = 56
    s = emit_LIT(b"ABC") + emit_MATCH(3, 3)
    assert seed_cost(s) == 56

def test_seed_cost_empty_seed():
    # p=0 -> C_END(0) = 3 + pad_to_byte(3) = 8
    assert seed_cost(b"") == 8
