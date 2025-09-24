Contributing Guide
==================

Welcome to the Teleport CLF Calculator project! This guide provides everything you need to know about contributing to this mathematical analysis tool.

Project Overview
----------------

The Teleport CLF Calculator implements a pure mathematical system for causal minimality analysis using the formula:

.. math::
   C_{min}^{(1)}(L) = 88 + 8 \cdot \text{leb}(L)

**Core Principles:**

- **Mathematical Purity**: Integer-only arithmetic, no floating-point operations
- **Theoretical Grounding**: Based on causal minimality principles
- **Performance**: O(log L) computational complexity
- **Reliability**: Comprehensive testing and validation frameworks

**Critical Constraints:**

- Mathematical constants are **locked**: H=56, CAUS=27, END=5  
- Core formula is **immutable**: C_min^(1)(L) = 88 + 8*leb(L)
- Decision rule is **fixed**: EMIT iff C_min^(1)(L) < 10*L (strict)

Getting Started
---------------

**Prerequisites**

- Python 3.7 or higher
- Git for version control
- Understanding of integer arithmetic and algorithmic complexity

**Development Setup**

.. code-block:: bash

    # Clone the repository
    git clone <repository-url>
    cd Teleport
    
    # Set up Python environment (optional but recommended)
    python -m venv clf_env
    source clf_env/bin/activate  # On Windows: clf_env\Scripts\activate
    
    # Install development dependencies
    pip install -r requirements-dev.txt  # If available
    
    # Verify installation
    python clf_calculator.py --self-test
    ./tools/pic2_gate.sh  # Mathematical validation

**Project Structure**

.. code-block:: text

    Teleport/
    ├── src/teleport/           # Main package code
    │   ├── clf_calculator.py   # Core calculator implementation
    │   └── ...
    ├── tests/                  # Unit tests and test data
    │   ├── test_clf_calculator.py
    │   └── test_data/
    ├── tools/                  # Utility scripts
    │   ├── pic2_gate.sh       # Mathematical validation
    │   └── ...
    ├── docs_sphinx/           # Sphinx documentation
    ├── archive/               # Development history
    └── CLF_MAXIMAL_VALIDATOR_FINAL.py  # Lightweight CLI tool

Types of Contributions
======================

**1. Bug Reports**

Before reporting a bug:

- Run mathematical validation: ``./tools/pic2_gate.sh``
- Execute self-tests: ``python clf_calculator.py --self-test``
- Check existing issues in the issue tracker

**Bug Report Template:**

.. code-block:: text

    **Bug Description**
    Clear description of the unexpected behavior
    
    **Steps to Reproduce**
    1. Command or code that triggers the bug
    2. Input parameters used
    3. Expected vs. actual results
    
    **Environment**
    - Python version: 
    - Operating system:
    - File sizes involved:
    
    **Mathematical Validation**
    - pic2_gate.sh result: [PASS/FAIL]
    - Self-test result: [PASS/FAIL]
    
    **Additional Context**
    Any relevant logs, error messages, or context

**2. Feature Requests**

Feature requests should align with project goals:

**Acceptable Features:**
- New export formats
- Additional CLI options  
- Performance improvements
- Enhanced testing capabilities
- Documentation improvements
- Integration helpers

**Unacceptable Features:**
- Changes to core mathematical formula
- Modifications to locked constants
- Floating-point arithmetic introduction
- Non-deterministic behavior
- File content analysis (length-only principle)

**3. Documentation Improvements**

Documentation contributions are highly valued:

- API documentation enhancements
- Usage examples and tutorials
- Mathematical explanation improvements  
- Error message clarifications
- Code comments for complex logic

**4. Code Contributions**

Code contributions must preserve mathematical integrity.

Development Guidelines
======================

**Mathematical Integrity**

**CRITICAL: DO NOT MODIFY**

The following elements are mathematically locked:

.. code-block:: python

    # LOCKED CONSTANTS - DO NOT CHANGE
    H = 56      # Header bits
    CAUS = 27   # Causal encoding bits  
    END = 5     # Termination bits
    
    # LOCKED FORMULA - DO NOT CHANGE
    def clf_single_seed_cost(L):
        return 88 + 8 * leb128_byte_length(L)
    
    # LOCKED DECISION RULE - DO NOT CHANGE  
    def should_emit(L):
        return clf_single_seed_cost(L) < 10 * L  # Strict inequality

**Required Practices**

1. **Integer-Only Arithmetic**: Never introduce floating-point operations
2. **Input Validation**: All functions must validate input types and ranges
3. **Mathematical Guards**: Include runtime assertions for mathematical properties
4. **Complexity Preservation**: Maintain O(log L) complexity where applicable
5. **Deterministic Results**: Identical inputs must produce identical outputs

**Code Style**

Follow established patterns:

.. code-block:: python

    def new_utility_function(L):
        """
        Brief description of function purpose.
        
        Args:
            L (int): File size in bytes, must be positive integer
            
        Returns:
            int/bool/str: Description of return value
            
        Relationship to Core CLF:
            Explain how this function relates to core CLF calculations.
            
        Examples:
            >>> new_utility_function(11751)
            expected_result
            
        Raises:
            ValueError: If input validation fails
        """
        # Math guards (DO NOT REMOVE - prevent drift)
        if not isinstance(L, int) or L <= 0:
            raise ValueError("L must be positive integer")
        
        # Implementation using core CLF functions
        result = some_calculation_using_existing_functions(L)
        
        # Post-calculation verification if needed
        assert isinstance(result, expected_type), f"Unexpected result type: {type(result)}"
        
        return result

**Testing Requirements**

All contributions must include comprehensive tests:

.. code-block:: python

    def test_new_function():
        """Test new utility function with comprehensive coverage."""
        
        # Test normal cases
        assert new_utility_function(127) == expected_result_127
        assert new_utility_function(128) == expected_result_128
        
        # Test boundary conditions  
        assert new_utility_function(1) == expected_result_1
        assert new_utility_function(16383) == expected_result_16383
        assert new_utility_function(16384) == expected_result_16384
        
        # Test error conditions
        with pytest.raises(ValueError):
            new_utility_function(0)
        with pytest.raises(ValueError):
            new_utility_function(-1)
        with pytest.raises(TypeError):
            new_utility_function("123")

**Performance Considerations**

- **Complexity Analysis**: Document time complexity of new algorithms
- **Memory Efficiency**: Avoid unnecessary memory allocation
- **Scalability**: Test with large input values (up to 10^9 bytes)
- **Benchmarking**: Include performance tests for significant changes

Development Workflow
====================

**1. Issue Discussion**

- Create or comment on relevant issues
- Discuss approach with maintainers
- Confirm mathematical compatibility

**2. Development**

.. code-block:: bash

    # Create feature branch
    git checkout -b feature/your-feature-name
    
    # Make changes following guidelines
    # Add comprehensive tests
    # Update documentation
    
    # Validate changes
    python -m pytest tests/ -v
    python clf_calculator.py --self-test
    ./tools/pic2_gate.sh

**3. Testing**

Before submitting changes:

.. code-block:: bash

    # Run full test suite
    python -m pytest tests/ -v --cov=clf_calculator
    
    # Validate mathematical properties
    ./tools/pic2_gate.sh
    
    # Test CLI interfaces
    python clf_calculator.py --stdin-length 11751
    python CLF_MAXIMAL_VALIDATOR_FINAL.py test_data/pic2.jpg
    
    # Performance validation
    python -c "
    import time
    from clf_calculator import clf_single_seed_cost
    start = time.time()
    for i in range(10000):
        clf_single_seed_cost(i + 1)
    print(f'Performance: {time.time() - start:.3f}s for 10k calculations')
    "

**4. Documentation**

Update relevant documentation:

- Add docstrings following established patterns
- Update API documentation if needed
- Add examples to docs_sphinx/examples.rst
- Update changelog with significant changes

**5. Pull Request**

**Pull Request Template:**

.. code-block:: text

    **Description**
    Brief description of changes and motivation
    
    **Type of Change**
    - [ ] Bug fix (non-breaking change that fixes an issue)
    - [ ] New feature (non-breaking change that adds functionality)
    - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
    - [ ] Documentation update
    
    **Mathematical Validation**
    - [ ] pic2_gate.sh passes
    - [ ] Self-tests pass  
    - [ ] No changes to locked constants (H=56, CAUS=27, END=5)
    - [ ] No changes to core formula
    - [ ] Integer-only arithmetic preserved
    
    **Testing**
    - [ ] Unit tests pass
    - [ ] New tests added for new functionality
    - [ ] Performance validated
    - [ ] Edge cases covered
    
    **Documentation**
    - [ ] Code is documented with appropriate docstrings
    - [ ] API documentation updated if needed
    - [ ] Examples added if appropriate

Code Review Process
-------------------

**Review Criteria**

1. **Mathematical Correctness**: Preserves CLF mathematical integrity
2. **Performance**: Maintains O(log L) complexity where applicable  
3. **Testing**: Comprehensive test coverage with edge cases
4. **Documentation**: Clear docstrings and API documentation
5. **Code Quality**: Follows established patterns and style

**Review Focus Areas**

- **Input Validation**: Proper type checking and range validation
- **Error Handling**: Appropriate exception handling and error messages
- **Mathematical Guards**: Runtime assertions for mathematical properties
- **Integration**: Compatibility with existing CLI and API interfaces
- **Performance**: No performance regressions introduced

**Approval Process**

- All tests must pass (unit tests, self-tests, pic2_gate.sh)
- At least one maintainer approval required
- Mathematical validation by core team for formula-adjacent changes
- Performance validation for optimization changes

Common Contribution Scenarios
-----------------------------

**Adding New Export Format**

.. code-block:: python

    def export_xml(receipts, prefix):
        """
        Export CLF analysis results to XML format.
        
        Args:
            receipts (list): List of receipt dictionaries
            prefix (str): File prefix for export
            
        Purpose:
            Provides XML export capability complementing existing JSONL/CSV exports.
            Maintains identical data structure with different serialization format.
        """
        # Implementation here
        pass

**Adding CLI Option**

.. code-block:: python

    def main():
        """..."""
        parser = argparse.ArgumentParser(description="CLF Calculator")
        # Existing options...
        parser.add_argument('--verbose', action='store_true', 
                          help='Enable verbose output with calculation details')

**Performance Optimization**

.. code-block:: python

    def optimized_leb128_byte_length(L):
        """
        Optimized LEB128 byte-length calculation with lookup table.
        
        Maintains identical mathematical behavior with improved performance
        for frequently-used file size ranges.
        """
        # Optimization implementation maintaining mathematical equivalence

**Utility Function Addition**

.. code-block:: python

    def leb_band_info(L):
        """
        Determine LEB128 band information for given file size.
        
        Returns band number, range boundaries, and cost information
        for educational and debugging purposes.
        """
        # Implementation providing band analysis

Recognition
-----------

Contributors are recognized in:

- Changelog with contribution details
- Code comments for significant contributions  
- Documentation acknowledgments
- Project README contributor list

**Contribution Categories:**

- **Core Contributors**: Major feature development and mathematical validation
- **Documentation Contributors**: Comprehensive documentation improvements
- **Testing Contributors**: Test suite enhancements and validation tools
- **Community Contributors**: Bug reports, feature suggestions, and usage feedback

Getting Help
------------

**Development Questions:**

- Review existing documentation thoroughly
- Check API reference for function specifications
- Run mathematical validation tools
- Create detailed issue with reproduction steps

**Mathematical Questions:**

- Review mathematical_foundation.rst for theoretical background
- Check test cases for boundary behavior examples
- Validate with pic2_gate.sh for consistency checks
- Consult academic literature on causal minimality principles

**Technical Support:**

- Ensure environment meets prerequisites
- Run comprehensive test suite
- Check for conflicting dependencies
- Validate system compatibility

This contributing guide ensures all contributions maintain the mathematical rigor, performance characteristics, and reliability that define the Teleport CLF Calculator project.