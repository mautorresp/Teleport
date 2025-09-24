Docstring Guide Reference
=========================

This document provides the complete docstring guide that was applied throughout the Teleport CLF Calculator codebase.

Overview
--------

The docstring system implements a comprehensive contract-based documentation approach designed to prevent mathematical drift and ensure long-term code integrity.

Key Principles
--------------

**DO NOT EDIT Comment Blocks**

All files contain critical ``DO NOT EDIT`` comment blocks that preserve the mathematical contract:

.. code-block:: python

    """
    DO NOT EDIT: This module implements the single-seed CLF causal minimality calculator.
    
    Contract: C_min^(1)(L) = 88 + 8*leb(L) with H=56, CAUS=27, END=5 (locked)
    Decision: EMIT iff C_min^(1)(L) < 10*L (strict). leb = unsigned LEB128 byte-length.
    Invariants: Integer-only. No compression logic. No floating point. O(log L) complexity.
    """

**Module Docstring Structure**

Every module follows the Contract/Invariants/Receipts pattern:

.. code-block:: python

    """
    Single-seed CLF causal minimality calculator for integer-only file size analysis.
    
    Contract:
        Formula: C_min^(1)(L) = 88 + 8*leb(L) bits
        Constants: H=56, CAUS=27, END=5 (locked, DO NOT MODIFY)
        Decision: EMIT iff C_min^(1)(L) < 10*L (strict inequality)
        
    Invariants:
        - Integer-only arithmetic (no floating-point operations)
        - O(log L) computational complexity
        - No file content analysis (length-only)
        - Deterministic results for identical inputs
        
    Receipts:
        All calculations generate SHA-256 receipts for verification.
        Drift detection through comprehensive runtime assertions.
    """

Function Docstring Format
-------------------------

**Core Mathematical Functions**

Functions implementing core mathematics use detailed specifications:

.. code-block:: python

    def clf_single_seed_cost(L):
        """
        Calculate single-seed CLF causal minimality cost: C_min^(1)(L) = 88 + 8*leb(L).
        
        Args:
            L (int): File size in bytes, must be positive integer
            
        Returns:
            int: Causal minimality cost in bits
            
        Mathematical Specification:
            C_min^(1)(L) = 88 + 8 * leb128_byte_length(L)
            where 88 = H(56) + CAUS(27) + END(5) are locked constants
            
        Complexity: O(log L)
        
        Examples:
            >>> clf_single_seed_cost(127)    # Band 1: leb=1
            96
            >>> clf_single_seed_cost(128)    # Band 2: leb=2  
            104
            >>> clf_single_seed_cost(16384)  # Band 3: leb=3
            112
            
        Raises:
            ValueError: If L <= 0 or L not integer
        """

**Utility Functions**

Utility functions include purpose and relationship to core functions:

.. code-block:: python

    def bit_length_info(L):
        """
        Generate bit-length bounds information for display purposes.
        
        Args:
            L (int): File size in bytes
            
        Returns:
            str: Formatted bounds string like "bounds=2^13 ≤ L < 2^14"
            
        Purpose:
            Provides human-readable context for file sizes in output formatting.
            Complements core CLF calculations with magnitude information.
            
        Note:
            This is a display utility only. Core CLF calculations do not depend
            on bit_length operations.
        """

**Guard Assertions**

All mathematical functions include runtime guards:

.. code-block:: python

    def clf_single_seed_cost(L):
        """..."""
        # Math guards (DO NOT REMOVE - prevent drift)
        if not isinstance(L, int) or L <= 0:
            raise ValueError("L must be positive integer")
        
        # Core calculation with locked constants
        leb_bytes = leb128_byte_length(L)
        cost = 88 + 8 * leb_bytes  # H(56) + CAUS(27) + END(5) + 8*leb
        
        # Post-calculation verification
        assert cost >= 88, f"Cost {cost} below minimum"
        assert isinstance(cost, int), f"Non-integer cost: {cost}"
        
        return cost

Class Documentation
-------------------

**Test Classes**

Test classes document their coverage scope:

.. code-block:: python

    class TestCLFCalculator:
        """
        Comprehensive unit tests for CLF causal minimality calculator.
        
        Test Coverage:
            - Mathematical correctness (formula accuracy, boundary conditions)
            - LEB128 encoding (band transitions, byte-length calculation)
            - Decision gate logic (emit conditions, efficiency calculations)
            - Error handling (invalid inputs, type validation)
            - Receipt generation (hashing, verification data)
            
        Test Philosophy:
            Pure integer testing only. No file I/O operations in unit tests.
            Focus on mathematical properties and edge cases.
        """

CLI Documentation
-----------------

**Main Function Documentation**

Command-line interfaces include comprehensive usage information:

.. code-block:: python

    def main():
        """
        Command-line interface for CLF causal minimality analysis.
        
        Usage:
            python clf_calculator.py [files...] [options]
            
        Options:
            --stdin-length LENGTH    Analyze specific length directly
            --export-prefix PREFIX   Generate JSONL, CSV, and audit exports
            --self-test             Run embedded validation tests
            
        Output Format:
            filename: L=bytes, bit_length=N, bounds=2^M ≤ L < 2^N,
            leb=K, C=bits, RAW=10*L_bits, EMIT=decision, receipt=hash...
            
        Export Files:
            {prefix}_clf_analysis.jsonl - JSON Lines format
            {prefix}_clf_analysis.csv   - CSV format  
            {prefix}_clf_audit.txt      - Human-readable audit
            
        Examples:
            python clf_calculator.py test.jpg
            python clf_calculator.py *.mp4 --export-prefix BATCH
            python clf_calculator.py --stdin-length 11751 --self-test
        """

Documentation Standards
-----------------------

**Consistency Requirements**

1. **Mathematical Notation**: Always use C_min^(1)(L) notation
2. **Constant References**: Always reference H=56, CAUS=27, END=5
3. **Decision Rule**: Always state "EMIT iff C_min^(1)(L) < 10*L (strict)"
4. **Complexity**: Always specify O(log L) where applicable
5. **Type Specifications**: Always specify integer-only arithmetic

**Prohibited Modifications**

The following elements are locked and must not be modified:

- ``DO NOT EDIT`` comment blocks
- Mathematical constants (H=56, CAUS=27, END=5)
- Core formula specification
- Decision rule specification
- Math guard assertions
- Receipt generation logic

**Update Procedures**

When adding new functionality:

1. **Preserve Contracts**: Never modify existing mathematical contracts
2. **Add Guards**: Include appropriate math guards for new functions  
3. **Document Relationships**: Explain how new functions relate to core CLF logic
4. **Maintain Formatting**: Follow established docstring structure
5. **Validate Integration**: Ensure new code doesn't affect core calculations

Implementation Examples
-----------------------

**Correct Docstring Addition**

When adding a new utility function:

.. code-block:: python

    def efficiency_ratio(L):
        """
        Calculate CLF efficiency ratio: 10*L / C_min^(1)(L).
        
        Args:
            L (int): File size in bytes, must be positive integer
            
        Returns:
            float: Efficiency ratio (>1.0 indicates EMIT=True)
            
        Relationship to Core CLF:
            Uses clf_single_seed_cost(L) for C_min^(1)(L) calculation.
            Provides efficiency context for emission decisions.
            
        Note:
            Result interpretation: ratio > 1.0 means CLF cost is lower than
            raw cost, corresponding to EMIT=True decision.
            
        Examples:
            >>> efficiency_ratio(11751)  # pic2.jpg
            1130.3846...
        """
        # Math guards (DO NOT REMOVE)
        if not isinstance(L, int) or L <= 0:
            raise ValueError("L must be positive integer")
        
        cost = clf_single_seed_cost(L)  # Core CLF calculation
        raw = 10 * L
        ratio = raw / cost
        
        # Verification
        assert ratio > 0, f"Invalid ratio: {ratio}"
        
        return ratio

**Incorrect Modifications (PROHIBITED)**

.. code-block:: python

    # PROHIBITED: Modifying constants
    H = 60  # Changed from 56 - BREAKS CONTRACT
    
    # PROHIBITED: Changing formula  
    def clf_single_seed_cost(L):
        return 90 + 7 * leb128_byte_length(L)  # Modified formula - BREAKS CONTRACT
    
    # PROHIBITED: Removing guards
    def clf_single_seed_cost(L):
        # Removed input validation - BREAKS SAFETY
        return 88 + 8 * leb128_byte_length(L)
    
    # PROHIBITED: Adding floating-point
    def clf_single_seed_cost(L):
        return 88.0 + 8.0 * leb128_byte_length(L)  # Float arithmetic - BREAKS INVARIANTS

Quality Assurance
-----------------

**Documentation Validation**

Before committing documentation changes:

1. **Contract Verification**: Confirm all mathematical contracts remain intact
2. **Guard Preservation**: Verify all math guards are present and unchanged
3. **Constant Validation**: Check that H=56, CAUS=27, END=5 remain locked
4. **Formula Consistency**: Ensure C_min^(1)(L) = 88 + 8*leb(L) is preserved
5. **Test Execution**: Run full test suite to verify no regressions

**Maintenance Procedures**

For ongoing maintenance:

1. **Periodic Audits**: Regular review of all docstrings for consistency
2. **Drift Detection**: Use pic2_gate.sh for mathematical drift detection
3. **Documentation Updates**: Keep examples and usage patterns current
4. **Integration Checks**: Verify documentation matches actual behavior

This docstring guide ensures the Teleport CLF Calculator codebase remains mathematically consistent, well-documented, and resistant to accidental modifications that could compromise its theoretical foundations.