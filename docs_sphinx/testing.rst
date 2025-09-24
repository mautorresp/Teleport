Testing Guide
=============

This guide covers comprehensive testing strategies for the Teleport CLF Calculator, including unit tests, validation procedures, and regression testing.

Built-in Testing
----------------

The CLF Calculator includes several layers of built-in testing:

**Self-Test Mode**

Run the embedded unit tests::

    python clf_calculator.py --self-test

This executes a comprehensive test suite covering:

- LEB128 band boundaries (127→128, 16383→16384)
- Mathematical edge cases
- Cost calculation validation  
- Decision gate verification
- Receipt generation and hashing

**Mathematical Validation Script**

The ``pic2_gate.sh`` script provides comprehensive mathematical validation::

    ./tools/pic2_gate.sh

This script:

- Validates known test cases (pic1.jpg, pic2.jpg, video3.mp4)
- Cross-checks between multiple calculator implementations
- Verifies mathematical invariants
- Detects any drift in core calculations

Unit Test Suite
---------------

**Running Tests**

Execute the formal test suite::

    python -m pytest tests/test_clf_calculator.py -v

**Test Coverage**

The unit test suite includes:

.. code-block:: python

    def test_leb128_band_1_boundary():
        """Test LEB128 band 1 boundary at L=127->128."""
        # Band 1: L ∈ [1, 127] → leb=1 → C=96
        assert leb128_byte_length(127) == 1
        assert clf_single_seed_cost(127) == 96
        
        # Band 2: L ∈ [128, 16383] → leb=2 → C=104  
        assert leb128_byte_length(128) == 2
        assert clf_single_seed_cost(128) == 104

    def test_cost_calculation_examples():
        """Test cost calculation for known examples."""
        test_cases = [
            (1, 1, 96),        # Smallest possible file
            (127, 1, 96),      # Band 1 maximum
            (128, 2, 104),     # Band 2 minimum
            (16383, 2, 104),   # Band 2 maximum
            (16384, 3, 112),   # Band 3 minimum
        ]
        
        for L, expected_leb, expected_cost in test_cases:
            assert leb128_byte_length(L) == expected_leb
            assert clf_single_seed_cost(L) == expected_cost

    def test_emit_gate_logic():
        """Test emission decision gate."""
        # All practical file sizes should emit
        test_sizes = [1, 42, 127, 128, 1000, 16383, 16384, 100000, 2097151]
        
        for L in test_sizes:
            cost = clf_single_seed_cost(L)
            raw = 10 * L
            assert should_emit(L) == (cost < raw)
            assert should_emit(L) == True  # All should emit for practical sizes

**Adding Custom Tests**

Create custom test functions following the established pattern:

.. code-block:: python

    def test_custom_validation():
        """Custom validation test example."""
        # Test specific requirements
        L = 11751  # Known test case
        
        # Verify expected values
        assert leb128_byte_length(L) == 2
        assert clf_single_seed_cost(L) == 104
        assert should_emit(L) == True
        
        # Verify receipt generation
        r = receipt(L, "TEST_BUILD")
        assert r['L'] == L
        assert r['leb_bytes'] == 2
        assert r['cost_bits'] == 104
        assert r['raw_bits'] == 117510
        assert r['emit'] == True
        assert len(r['receipt_hash']) == 64  # SHA-256 hex length

Validation Procedures
=====================

**Mathematical Property Validation**

.. code-block:: python

    def validate_mathematical_properties():
        """Comprehensive mathematical validation."""
        
        # Test 1: Monotonicity
        print("Testing cost monotonicity...")
        test_sequence = [1, 50, 127, 128, 1000, 16383, 16384, 100000]
        prev_cost = 0
        
        for L in test_sequence:
            cost = clf_single_seed_cost(L)
            assert cost >= prev_cost, f"Non-monotonic at L={L}"
            prev_cost = cost
        print("PASS: Monotonicity verified")
        
        # Test 2: Band consistency
        print("Testing LEB band consistency...")
        for band in range(1, 5):
            start = 128 ** (band - 1) if band > 1 else 1
            end = min(128 ** band - 1, 10**6)
            
            # Sample points in band
            sample_points = [start, start + 1, (start + end) // 2, end - 1, end]
            costs = [clf_single_seed_cost(L) for L in sample_points if L <= end]
            
            # All costs in band should be identical
            expected_cost = 88 + 8 * band
            assert all(c == expected_cost for c in costs), f"Band {band} inconsistent"
        print("PASS: Band consistency verified")
        
        # Test 3: Formula accuracy
        print("Testing formula accuracy...")
        for L in [1, 127, 128, 16383, 16384, 2097151]:
            leb_bytes = leb128_byte_length(L)
            expected_cost = 88 + 8 * leb_bytes
            actual_cost = clf_single_seed_cost(L)
            assert actual_cost == expected_cost, f"Formula error at L={L}"
        print("PASS: Formula accuracy verified")

**Regression Testing**

Create regression tests for known values::

    # Known test cases from development
    REGRESSION_CASES = [
        # (L, expected_leb, expected_cost, expected_emit, description)
        (63379, 3, 112, True, "pic1.jpg baseline"),
        (11751, 2, 104, True, "pic2.jpg baseline"),  
        (9840497, 4, 120, True, "video3.mp4 baseline"),
    ]
    
    def test_regression_cases():
        """Test against known regression cases."""
        for L, exp_leb, exp_cost, exp_emit, desc in REGRESSION_CASES:
            # Test all components
            assert leb128_byte_length(L) == exp_leb, f"LEB failure: {desc}"
            assert clf_single_seed_cost(L) == exp_cost, f"Cost failure: {desc}" 
            assert should_emit(L) == exp_emit, f"Emit failure: {desc}"
            
            # Test receipt generation
            r = receipt(L, "REGRESSION_TEST")
            assert r['L'] == L
            assert r['leb_bytes'] == exp_leb
            assert r['cost_bits'] == exp_cost
            assert r['emit'] == exp_emit

Performance Testing
-------------------

**Computational Complexity Verification**

.. code-block:: python

    import time
    
    def test_performance_scaling():
        """Verify O(log L) computational complexity."""
        test_sizes = [
            ("Small", [10**i for i in range(1, 4)]),      # 10-1000
            ("Medium", [10**i for i in range(4, 7)]),     # 10K-1M  
            ("Large", [10**i for i in range(7, 10)]),     # 10M-1B
        ]
        
        for category, sizes in test_sizes:
            times = []
            
            for L in sizes:
                start = time.time()
                for _ in range(1000):  # Multiple iterations for accuracy
                    clf_single_seed_cost(L)
                elapsed = time.time() - start
                times.append(elapsed)
            
            # Verify performance doesn't degrade significantly
            max_time = max(times)
            min_time = min(times)
            ratio = max_time / min_time if min_time > 0 else 1
            
            print(f"{category}: {min_time*1000:.3f}ms - {max_time*1000:.3f}ms "
                  f"(ratio: {ratio:.1f}×)")
            
            # Should scale logarithmically, not linearly
            assert ratio < 5.0, f"Performance degradation in {category} category"

**Memory Usage Testing**

.. code-block:: python

    import sys
    
    def test_memory_efficiency():
        """Verify constant memory usage."""
        # Test with various input sizes
        test_cases = [1, 1000, 1000000, 1000000000]
        
        baseline_size = sys.getsizeof(clf_single_seed_cost(1))
        
        for L in test_cases:
            result_size = sys.getsizeof(clf_single_seed_cost(L))
            # Result should be constant size integer
            assert result_size == baseline_size, f"Memory usage varies with input size at L={L}"

Error Handling Tests
====================

**Input Validation Testing**

.. code-block:: python

    def test_input_validation():
        """Test error handling for invalid inputs."""
        
        # Test invalid file sizes
        invalid_inputs = [-1, 0, -100, 3.14, "123", None, []]
        
        for invalid_input in invalid_inputs:
            with pytest.raises((ValueError, TypeError)):
                clf_single_seed_cost(invalid_input)
            
            with pytest.raises((ValueError, TypeError)):
                should_emit(invalid_input)
                
            with pytest.raises((ValueError, TypeError)):
                receipt(invalid_input, "TEST")

**File Processing Error Handling**

.. code-block:: python

    def test_file_processing_errors():
        """Test file processing error handling."""
        
        # Test non-existent file
        result = process_file("nonexistent_file.txt", "TEST")
        assert result is None
        
        # Test directory instead of file
        result = process_file("/tmp", "TEST") 
        assert result is None
        
        # Test empty filename
        result = process_file("", "TEST")
        assert result is None

Continuous Integration Testing
===============================

**Automated Test Execution**

Create a comprehensive test script::

    #!/bin/bash
    # comprehensive_test.sh
    
    echo "Starting comprehensive CLF Calculator testing..."
    
    # 1. Unit tests
    echo "Running unit tests..."
    python -m pytest tests/test_clf_calculator.py -v
    if [ $? -ne 0 ]; then
        echo "FAIL: Unit tests failed"
        exit 1
    fi
    
    # 2. Self-tests
    echo "Running self-tests..."
    python clf_calculator.py --self-test
    if [ $? -ne 0 ]; then
        echo "FAIL: Self-tests failed" 
        exit 1
    fi
    
    # 3. Mathematical validation
    echo "Running mathematical validation..."
    if [ -f "tools/pic2_gate.sh" ]; then
        ./tools/pic2_gate.sh
        if [ $? -ne 0 ]; then
            echo "FAIL: Mathematical validation failed"
            exit 1
        fi
    fi
    
    # 4. CLI interface testing
    echo "Testing CLI interfaces..."
    python clf_calculator.py --stdin-length 11751 > /dev/null
    if [ $? -ne 0 ]; then
        echo "FAIL: CLI test failed"
        exit 1
    fi
    
    echo "SUCCESS: All tests passed successfully!"

**Test Data Management**

Maintain consistent test data::

    # test_data/README.md
    Test Data Files:
    - pic1.jpg: 63,379 bytes (LEB band 3, C=112, EMIT=True)
    - pic2.jpg: 11,751 bytes (LEB band 2, C=104, EMIT=True) 
    - video3.mp4: 9,840,497 bytes (LEB band 4, C=120, EMIT=True)
    
    Expected Results:
    - All files should result in EMIT=True
    - Cost calculations must match exactly
    - Receipt hashes must be consistent

Test Documentation
==================

**Test Case Documentation**

Document each test case with clear expectations:

.. code-block:: python

    class TestCLFCalculator:
        """
        Comprehensive test suite for CLF Calculator.
        
        Test Categories:
        1. Mathematical correctness (formula, boundaries, edge cases)
        2. Performance characteristics (time, memory, scaling)
        3. Error handling (invalid inputs, file errors)
        4. Integration (CLI, exports, receipts)
        """
        
        def test_mathematical_correctness(self):
            """
            Tests mathematical correctness of core calculations.
            
            Validates:
            - Formula: C = 88 + 8*leb(L)
            - LEB128 byte-length calculation
            - Decision gate: EMIT iff C < 10*L
            - Boundary conditions between LEB bands
            """
            pass

**Coverage Requirements**

Maintain test coverage standards:

- **Formula accuracy**: 100% coverage of all mathematical operations
- **Boundary conditions**: All LEB band transitions tested
- **Error cases**: All error paths validated
- **Integration points**: CLI, file I/O, export generation
- **Performance**: Complexity and memory usage verified

Best Practices
--------------

**Test Development Guidelines**:

1. **Pure Functions**: Test mathematical functions in isolation
2. **Deterministic**: All tests must produce identical results on repeated runs
3. **Comprehensive**: Cover normal cases, edge cases, and error conditions
4. **Fast Execution**: Unit tests should complete in seconds, not minutes
5. **Clear Assertions**: Each assertion should test exactly one property
6. **Documentation**: Each test should clearly state what it validates

**Continuous Validation**:

1. Run tests before any code changes
2. Validate mathematical properties after modifications
3. Use pic2_gate.sh for regression detection
4. Maintain test data consistency across environments
5. Document any changes to expected test outcomes

This testing framework ensures the CLF Calculator remains mathematically accurate, performant, and reliable across all use cases and modifications.