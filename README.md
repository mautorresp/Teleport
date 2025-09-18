# Tele## Features

- **🧮 Mathematical Generators**: Complete family G = {CONST, STEP, LCG8, LFSR8, ANCHOR} with deterministic deduction
- **🎯 Causality Proofs**: Either proves causality with exact parameters or provides formal refutation
- **🔢 Integer-Only Math**: Pure integer arithmetic with no floating-point contamination
- **📊 Exact Cost Model**: C_CAUS = 3 + 8×leb(op) + 8×Σleb(param_i) + 8×leb(N) 
- **⚡ CLF Compliance**: CAUS-or-FAIL enforcement with quantified mathematical witnesses
- **🚫 No Heuristics**: Only deterministic deduction and formal verification
- **✨ OpSet_v2**: Zero-ambiguity deterministic procedures with normative operator registryCLF Mathematical Causality Analysis System**

Teleport is a mathematical causality analysis system that determines whether byte sequences can be reproduced by deterministic generators. It uses pure integer arithmetic to either prove causality with exact parameters or provide formal refutation with quantified witnesses. No compression, no heuristics - only mathematical deduction.

## Features

- **� Mathematical Generators**: Complete family G = {CONST, STEP, LCG8, LFSR8, ANCHOR} with deterministic deduction
- **🎯 Causality Proofs**: Either proves causality with exact parameters or provides formal refutation
- **🔢 Integer-Only Math**: Pure integer arithmetic with no floating-point contamination
- **📊 Exact Cost Model**: C_CAUS = 3 + 8×leb(op) + 8×Σleb(param_i) + 8×leb(N) 
- **⚡ CLF Compliance**: CAUS-or-FAIL enforcement with quantified mathematical witnesses
- **🚫 No Heuristics**: Only deterministic deduction and formal verification

## Quick Start

### OpSet_v2 CLI Tools (Real-File Testing)

```bash
# Complete analysis of any file
./teleport_cli test-all photo.jpg

# Individual operations
./teleport_cli scan video.mp4          # Pattern detection  
./teleport_cli price document.pdf      # Cost calculation (integer bits)
./teleport_cli expand-verify data.bin  # Round-trip verification
./teleport_cli canonical file.txt      # Cost formula verification
```

### Installation

```bash
pip install teleport
```

### Basic Usage

```python
from teleport.caus_deduction_complete import formal_caus_test
from teleport.generators import deduce_CONST, verify_generator, OP_CONST

# Test causality of byte sequence
with open('data.bin', 'rb') as f:
    data = f.read()

# Complete mathematical analysis
result = formal_caus_test(data)
if result['causality_proven']:
    print(f"✅ Causality proven: {result['generator_name']}")
    print(f"Parameters: {result['params']}")
    print(f"Cost: {result['C_CAUS']} bits < {result['C_LIT']} bits")
else:
    print(f"❌ Formal refutation: {result['refutation_witnesses']}")

# Manual generator deduction
ok, params, reason = deduce_CONST(data)
if ok:
    verified = verify_generator(OP_CONST, params, data)
    print(f"CONST generator: verified={verified}")
else:
    print(f"CONST failed: {reason}")
```

## Core Modules

### `teleport.generators` - Mathematical Generator Family

Complete deterministic generator deduction:

```python
from teleport.generators import (
    deduce_CONST, deduce_STEP, deduce_LCG8, deduce_LFSR8, deduce_ANCHOR,
    verify_generator, compute_caus_cost
)

# Test if sequence follows CONST pattern
S = bytes([0x42] * 100)  # 100 repeated bytes
ok, params, reason = deduce_CONST(S)
if ok:
    byte_value, = params
    print(f"CONST generator: byte={byte_value}")
    
    # Verify mathematical correctness
    verified = verify_generator(OP_CONST, params, S)
    cost = compute_caus_cost(OP_CONST, params, len(S))
    print(f"Verified: {verified}, Cost: {cost} bits")
else:
    print(f"Not constant: {reason}")
```

### `teleport.caus_deduction_complete` - Formal Causality Analysis

Complete mathematical proof system:

```python
from teleport.caus_deduction_complete import (
    try_deduce_caus, formal_caus_test, generate_bytes
)

# Exhaustive generator evaluation
best, receipts = try_deduce_caus(data)
if best:
    gen_name, op_id, params, cost = best
    print(f"Causality proven: {gen_name} with cost {cost} bits")
else:
    print("Formal refutation: No generator in family can reproduce sequence")

# Complete analysis with exit codes
result = formal_caus_test(data)
# Returns: causality_proven, generator_name, params, costs, receipts

# Generate bytes from proven parameters
if result['causality_proven']:
    reconstructed = generate_bytes(result['op_id'], result['params'], len(data))
    assert reconstructed == data  # Mathematical verification
```

### `teleport.predicates` - Specialized Pattern Analysis

Mathematical pattern detection for complex structures:

```python
from teleport.predicates import check_anchor_window, check_repeat1

# Test dual-anchor structure (format-blind)
S = b'\\xff\\xd8' + bytes([0x42] * 1000) + b'\\xff\\xd9'
ok, op_id, params = check_anchor_window(S)
if ok:
    print(f"Anchor structure found with interior generator")
else:
    print("No mathematical anchor structure")
```

### `teleport.leb_io` - LEB128 I/O

Minimal LEB128 implementation with streaming support:

```python
from teleport.leb_io import (
    leb128_emit_single, leb128_parse_single,
    leb128_stream_emit, leb128_stream_parse
)

# Single value operations
encoded = leb128_emit_single(12345)
value, consumed = leb128_parse_single(encoded)

# Streaming operations
emitter = leb128_stream_emit()
emitter.emit(100).emit(200).emit(300)
data = emitter.get_bytes()

parser = leb128_stream_parse(data)
while parser.has_more():
    value = parser.parse()
    print(f"Parsed: {value}")
```

## No-Float Linting

Teleport includes a static analysis tool to detect floating-point contamination:

```bash
# Lint a single file
python -m tools.no_float_lint my_file.py

# Lint a directory recursively
python -m tools.no_float_lint src/ --recursive

# Strict mode with additional checks
python -m tools.no_float_lint src/ --strict
```

The linter detects:
- Float literals (`3.14`, `2.0`)
- Risky operators (`/` division, `**` power)
- Risky functions (`float()`, `math.sqrt()`)
- Risky imports (`import math`, `import numpy`)
- Float type annotations

## Development Setup

### Prerequisites

- Python 3.8+
- pip

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/mautorresp/Teleport.git
cd Teleport

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=teleport --cov-report=html

# Run specific test categories
pytest -m "not slow"              # Skip slow tests
pytest tests/test_guards.py       # Run specific test file
```

### Code Quality

```bash
# Format code
black teleport tests

# Lint code
flake8 teleport tests

# Type checking
mypy teleport

# No-float linting
python tools/no_float_lint.py teleport/ --recursive --strict

# Run all checks (via pre-commit)
pre-commit run --all-files
```

## Project Structure

```
Teleport/
├── teleport/                    # Main package
│   ├── __init__.py             # Package exports
│   ├── generators.py           # Mathematical generator family
│   ├── caus_deduction_complete.py  # Formal causality system
│   ├── predicates.py           # Specialized pattern analysis
│   ├── guards.py               # No-float membrane
│   ├── costs.py                # Exact cost computation
│   ├── leb_io.py               # LEB128 encoding
│   └── seed_vm.py              # VM for reconstructions
├── scripts/                    # Analysis tools
│   ├── caus_proof_complete.py  # Complete mathematical analysis
│   └── seed_verify.py          # Verification utilities
├── tests/                      # Test suite
│   ├── test_generators.py      # Generator tests
│   └── test_no_float.py        # Float contamination tests
├── test_artifacts/             # Evidence files
│   └── evidence*.txt           # Mathematical analysis results
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Design Principles

### Mathematical Causality Analysis

Teleport determines causality through pure mathematical deduction:

- Either proves causality with exact generator parameters
- Or provides formal refutation with quantified witnesses  
- No heuristics, no approximations, no guessing
- Complete deterministic evaluation of generator family

### CLF Compliance (CAUS-or-FAIL)

Every analysis provides either:

1. **Positive Proof**: Generator with exact parameters and costs (C_CAUS < 10×N)
2. **Formal Refutation**: Mathematical witnesses proving no causality exists
3. **Deterministic Results**: Same input always produces same mathematical outcome
4. **Exact Costs**: Integer-only cost model with minimal LEB128 encoding

### Generator Completeness

Complete mathematical evaluation over finite generator family:

- **CONST**: Repeated byte patterns
- **STEP**: Arithmetic progressions modulo 256
- **LCG8**: Linear congruential generators  
- **LFSR8**: Linear feedback shift registers
- **ANCHOR**: Dual-anchor with inner generator structure

## Use Cases

### Forensic Analysis

Determine if byte sequences contain mathematical structure:

```python
from teleport.caus_deduction_complete import formal_caus_test

# Analyze unknown binary data
with open('suspicious.bin', 'rb') as f:
    data = f.read()

result = formal_caus_test(data)
if result['causality_proven']:
    print(f"Mathematical structure detected: {result['generator_name']}")
    print(f"Compression ratio: {result['compression_ratio']:.1f}:1")
else:
    print("No mathematical causality found - likely natural/encrypted data")
```

### Reverse Engineering

Extract generator parameters from observed sequences:

```python
from teleport.generators import deduce_LCG8, verify_generator

# Analyze potential LCG sequence
ok, params, reason = deduce_LCG8(observed_bytes)
if ok:
    x0, a, c = params
    print(f"LCG parameters: seed={x0}, multiplier={a}, increment={c}")
    
    # Predict future values
    next_values = generate_lcg_sequence(x0, a, c, start_offset=len(observed_bytes))
    print(f"Predicted next bytes: {next_values[:10]}")
```

### Cryptographic Analysis

Mathematical evaluation of pseudo-random sequences:

```python
from teleport.generators import deduce_LFSR8

# Test if sequence follows LFSR pattern
key_stream = bytes([...])  # Observed keystream
ok, params, reason = deduce_LFSR8(key_stream)
if ok:
    taps, seed = params
    print(f"⚠️  LFSR weakness detected: taps=0x{taps:02x}, seed=0x{seed:02x}")
else:
    print(f"✅ No LFSR structure found: {reason}")
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Guidelines

- All mathematical deduction must be deterministic and reproducible
- Maintain CLF compliance: CAUS-or-FAIL with no middle ground
- Add mathematical receipts for new generator types
- Provide formal refutation witnesses for failure cases
- Update generator family completeness proofs

### Testing Float Rejection

When adding new functions, always test that they properly reject floats:

```python
def test_new_function_rejects_floats():
    with pytest.raises(ValueError, match="Float detected"):
        new_function(3.14)  # Should raise ValueError
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

---

**Teleport** - Mathematical causality analysis with zero compromise! �
