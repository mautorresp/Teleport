# Teleport

**Integer-Only Compression and Processing Library**

Teleport is a strict integer-only library for compression, encoding, and data processing that enforces no-float constraints through guards and linting. It provides mathematically precise operations without floating-point contamination.

## Features

- **🚫 No-Float Membrane**: Runtime guards and decorators to prevent floating-point contamination
- **🔢 Integer-Only Operations**: LEB128 encoding, bit manipulation, and compression algorithms
- **📊 Exact Cost Formulas**: Precise integer-only cost calculations for compression schemes
- **🔍 Static Analysis**: AST linter to detect and reject floating-point operations
- **⚡ Minimal Dependencies**: Pure Python with no external dependencies for core functionality

## Quick Start

### Installation

```bash
pip install teleport
```

### Basic Usage

```python
from teleport import no_float_guard, leb128_encode, huffman_cost_bits

# Protect functions from float contamination
@no_float_guard
def process_data(values):
    return sum(values)

# Integer-only LEB128 encoding
data = leb128_encode(12345)
print(f"Encoded: {data.hex()}")  # Encoded: b9606

# Exact compression cost calculation
frequencies = [10, 5, 3, 2]
code_lengths = [1, 2, 3, 4]
cost = huffman_cost_bits(frequencies, code_lengths)
print(f"Huffman cost: {cost} bits")  # Huffman cost: 37 bits
```

## Core Modules

### `teleport.guards` - No-Float Membrane

Runtime protection against floating-point contamination:

```python
from teleport.guards import no_float_guard, assert_integer_only

@no_float_guard
def safe_calculation(a, b):
    return a * b + 1

# This will raise ValueError due to float contamination
try:
    safe_calculation(5, 3.14)  # ❌ Float detected!
except ValueError as e:
    print(f"Caught: {e}")

# This works fine
result = safe_calculation(5, 3)  # ✅ Integer-only
```

### `teleport.clf_int` - Integer Helpers

Core integer-only utilities:

```python
from teleport.clf_int import (
    leb128_encode, leb128_decode, 
    pack_bits, extract_bits,
    next_power_of_2, integer_log2
)

# LEB128 encoding/decoding
encoded = leb128_encode(300)
decoded, consumed = leb128_decode(encoded)

# Bit manipulation
packed = pack_bits([5, 3, 1], [3, 2, 1])  # Pack values with bit widths
bits = extract_bits(packed, 0, 3)         # Extract 3 bits from position 0

# Integer math
next_pow = next_power_of_2(100)  # 128
log_val = integer_log2(256)      # 8
```

### `teleport.costs` - Exact Cost Formulas

Precise integer-only cost calculations:

```python
from teleport.costs import (
    huffman_cost_bits, entropy_cost_estimate,
    lz77_cost, compression_ratio_scaled
)

# Huffman coding cost
frequencies = [50, 30, 15, 5]
code_lengths = [1, 2, 3, 4]
cost = huffman_cost_bits(frequencies, code_lengths)

# Entropy estimation (integer approximation)
estimated = entropy_cost_estimate(frequencies)

# Compression ratio (scaled integer)
ratio = compression_ratio_scaled(1000, 600, scale=1000)  # 1666 = 1.666x
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
git clone https://github.com/teleport-project/teleport.git
cd teleport

# Install in development mode with all dependencies
pip install -e ".[dev]"

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
├── teleport/              # Main package
│   ├── __init__.py       # Package exports
│   ├── guards.py         # No-float membrane
│   ├── clf_int.py        # Integer helpers
│   ├── costs.py          # Cost formulas
│   └── leb_io.py         # LEB128 I/O
├── tests/                # Test suite
│   ├── test_guards.py    # Guard tests
│   ├── test_clf_int.py   # Integer helper tests
│   └── test_costs.py     # Cost formula tests
├── tools/                # Development tools
│   └── no_float_lint.py  # Float contamination linter
├── pyproject.toml        # Project configuration
├── .pre-commit-config.yaml  # Pre-commit hooks
└── README.md             # This file
```

## Design Principles

### Integer-Only Operations

Teleport maintains strict integer semantics throughout:

- All operations use integer arithmetic
- No implicit float conversion
- Explicit overflow/underflow handling
- Deterministic results across platforms

### No-Float Membrane

The no-float membrane provides multiple layers of protection:

1. **Runtime Guards**: Decorators check function arguments and return values
2. **Static Analysis**: AST linter catches float operations at development time
3. **Type Annotations**: Clear integer-only type hints
4. **Testing**: Comprehensive test suite validates all operations

### Mathematical Precision

Cost formulas and calculations maintain mathematical exactness:

- Integer-only entropy approximations
- Exact bit counting for compression schemes
- Scaled arithmetic for ratios and percentages
- No floating-point rounding errors

## Use Cases

### Compression Libraries

Build compression algorithms with guaranteed integer semantics:

```python
from teleport import huffman_cost_bits, leb128_encode

# Calculate exact compression costs
def analyze_compression(data):
    frequencies = count_symbols(data)
    code_lengths = calculate_huffman_lengths(frequencies)
    cost = huffman_cost_bits(frequencies, code_lengths)
    return cost
```

### Embedded Systems

Integer-only operations for resource-constrained environments:

```python
from teleport.clf_int import pack_bits, extract_bits

# Pack sensor data into minimal bit representation
def pack_sensor_data(temperature, humidity, pressure):
    # Temperature: 0-127 (7 bits), Humidity: 0-100 (7 bits), Pressure: 0-1023 (10 bits)
    return pack_bits([temperature, humidity, pressure], [7, 7, 10])
```

### Financial Systems

Exact arithmetic for financial calculations:

```python
from teleport.guards import no_float_guard

@no_float_guard
def calculate_interest(principal_cents, rate_basis_points, periods):
    # All monetary values in cents, rates in basis points
    # No floating-point rounding errors
    return (principal_cents * rate_basis_points * periods) // 10000
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Guidelines

- All code must pass the no-float linter
- Maintain 100% integer-only operations
- Add comprehensive tests for new features
- Follow existing code style and patterns
- Update documentation for user-facing changes

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

**Teleport** - Pure integer operations for a floating-point-free future! 🚀
