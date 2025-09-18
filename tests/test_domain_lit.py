"""
Domain LIT Tests (CLF)

Test cases for LIT domain constraints:
- emit_LIT must reject L > 10
- expand must reject seeds with LIT L > 10
- Causal encoder produces valid seeds
"""

import pytest
from teleport.seed_format import emit_LIT
from teleport.seed_vm import expand, SeedDomainError
import pathlib
import subprocess
import sys

class TestDomainLIT:
    def test_emit_LIT_rejects_length_11(self):
        """emit_LIT(b'A'*11) -> ValueError"""
        with pytest.raises(ValueError, match="LIT length > 10"):
            emit_LIT(b'A' * 11)

    def test_emit_LIT_rejects_empty(self):
        """emit_LIT(b'') -> ValueError"""
        with pytest.raises(ValueError, match="LIT length > 10"):
            emit_LIT(b'')

    def test_emit_LIT_accepts_valid_lengths(self):
        """emit_LIT accepts 1 <= L <= 10"""
        for L in range(1, 11):
            seed = emit_LIT(b'X' * L)
            assert len(seed) >= L + 2  # tag + ULEB + payload

    def test_expand_rejects_domain_violation(self):
        """Hand-encoded LIT(L=11) -> expand() raises SeedDomainError"""
        # Construct illegal seed: [00][ULEB(11)=0x0B][payload]
        bad_seed = bytes([0x00, 0x0B]) + b'X' * 11
        with pytest.raises(SeedDomainError, match="LIT length > 10"):
            expand(bad_seed)

    def test_causal_encoder_produces_valid_seed(self):
        """encode_causal.py on 1KB payload -> valid seed with eq_sha=1"""
        # Create test payload
        payload = b'ABCDEFGHIJ' * 100  # 1KB with structures for MATCH operators
        payload_path = pathlib.Path("test_artifacts/test_payload_1kb.bin")
        payload_path.parent.mkdir(exist_ok=True)
        payload_path.write_bytes(payload)

        # Run causal encoder
        result = subprocess.run([
            sys.executable, "scripts/encode_causal.py", 
            "--payload", str(payload_path)
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        
        # Verify seed was created
        seed_path = pathlib.Path("test_artifacts/seed_causal.bin")
        assert seed_path.exists()
        
        # Verify seed passes linting
        lint_result = subprocess.run([
            sys.executable, "scripts/seed_lint.py",
            "--seed", str(seed_path)
        ], capture_output=True)
        assert lint_result.returncode == 0
        
        # Verify expansion produces identity
        seed_data = seed_path.read_bytes()
        expanded = expand(seed_data)
        assert expanded == payload
