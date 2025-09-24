API Reference
=============

This section provides detailed API documentation for all CLF Calculator modules and functions.

Core Functions
--------------

.. currentmodule:: clf_calculator

.. autofunction:: clf_single_seed_cost
   :no-index:

.. autofunction:: leb_len_u
   :no-index:

.. autofunction:: should_emit
   :no-index:

.. autofunction:: receipt
   :no-index:

Utility Functions
-----------------

.. autofunction:: main
   :no-index:



Constants
---------

Mathematical constants used throughout the system:

.. py:data:: H
   :value: 56
   
   Header information bits (locked constant)

.. py:data:: CAUS  
   :value: 27
   
   Causal relationship encoding bits (locked constant)

.. py:data:: END
   :value: 5
   
   Termination marker bits (locked constant)

.. py:data:: END
   :no-index:
   :value: 5
   
   Termination encoding bits (locked constant)

Type Definitions
----------------

The API uses standard Python types with the following conventions:

.. py:class:: Receipt
   
   Dictionary containing complete calculation results:
   
   .. py:attribute:: L
      :type: int
      
      File size in bytes
   
   .. py:attribute:: leb_bytes
      :type: int
      
      LEB128 byte-length of L
   
   .. py:attribute:: cost_bits
      :type: int
      
      C_min^(1)(L) cost in bits
   
   .. py:attribute:: raw_bits
      :type: int
      
      10*L raw bit cost
   
   .. py:attribute:: emit
      :type: bool
      
      True if cost_bits < raw_bits (strict)
   
   .. py:attribute:: build_id
      :type: str
      
      Build identifier for verification
   
   .. py:attribute:: receipt_hash
      :type: str
      
      SHA-256 hash of calculation parameters

Error Handling
--------------

The API raises standard Python exceptions:

.. py:exception:: ValueError
   
   Raised for invalid input parameters:
   
   - Non-positive file sizes (L ≤ 0)
   - Non-integer length values
   - Invalid file paths

.. py:exception:: FileNotFoundError
   
   Raised when specified files cannot be accessed

.. py:exception:: PermissionError
   
   Raised when file read permissions are insufficient

Usage Patterns
--------------

**Basic Usage**::

    from clf_calculator import clf_single_seed_cost, should_emit
    
    L = 11751
    cost = clf_single_seed_cost(L)  # Returns 104
    emit = should_emit(L)           # Returns True

**Detailed Analysis**::

    from clf_calculator import receipt
    
    r = receipt(L, "ANALYSIS_001")
    print(f"Size: {r['L']} bytes")
    print(f"LEB length: {r['leb_bytes']}")  
    print(f"Cost: {r['cost_bits']} bits")
    print(f"Raw: {r['raw_bits']} bits")
    print(f"Emit: {r['emit']}")
    print(f"Hash: {r['receipt_hash']}")

**File Processing**::

    from clf_calculator import process_file
    
    result = process_file("test_data/pic2.jpg", "BUILD_001")
    if result:
        print(f"File processed: {result}")
    else:
        print("File processing failed")

**Batch Processing**::

    import glob
    from clf_calculator import process_file, generate_exports
    
    files = glob.glob("data/*.jpg")
    receipts = []
    
    for file_path in files:
        result = process_file(file_path, "BATCH_001")
        if result:
            receipts.append(result)
    
    # Export results
    generate_exports(receipts, "BATCH_ANALYSIS")

CLI Reference
-------------

**clf_calculator.py**

Command-line interface for the main calculator::

    python clf_calculator.py [files...] [options]

**Positional Arguments**:

- ``files``: One or more file paths to analyze

**Options**:

- ``--stdin-length LENGTH``: Analyze specific length directly
- ``--export-prefix PREFIX``: Generate exports with given prefix
- ``--self-test``: Run embedded unit tests
- ``--help``: Show help message

**Examples**::

    # Single file
    python clf_calculator.py test.jpg
    
    # Multiple files  
    python clf_calculator.py file1.jpg file2.mp4
    
    # Direct length
    python clf_calculator.py --stdin-length 11751
    
    # With exports
    python clf_calculator.py data/*.jpg --export-prefix ANALYSIS

**CLF_MAXIMAL_VALIDATOR_FINAL.py**

Lightweight CLI validator::

    python CLF_MAXIMAL_VALIDATOR_FINAL.py [files...]

**Examples**::

    # Single file validation
    python CLF_MAXIMAL_VALIDATOR_FINAL.py test.jpg
    
    # Multiple files
    python CLF_MAXIMAL_VALIDATOR_FINAL.py data/*.jpg

Export Formats
--------------

The calculator generates exports in multiple formats:

**JSONL Format** (``{prefix}_clf_analysis.jsonl``)::

    {"L": 11751, "leb_bytes": 2, "cost_bits": 104, "raw_bits": 117510, "emit": true, "build_id": "BUILD_001", "receipt_hash": "a99a8a35..."}

**CSV Format** (``{prefix}_clf_analysis.csv``)::

    L,leb_bytes,cost_bits,raw_bits,emit,build_id,receipt_hash
    11751,2,104,117510,True,BUILD_001,a99a8a35...

**Audit Format** (``{prefix}_clf_audit.txt``)::

    CLF Analysis Audit Report
    Generated: 2025-09-23T23:30:00Z
    Build ID: BUILD_001
    
    File: test_data/pic2.jpg
    L=11,751 bytes, leb=2, C=104 bits, RAW=117,510 bits, EMIT=True
    Receipt: a99a8a35...

Implementation Notes
====================

**Performance Characteristics**:

- All functions operate in O(log L) time
- Memory usage is O(1) for individual calculations
- Batch operations scale linearly with file count

**Thread Safety**:

- All functions are pure (no side effects)
- Safe for concurrent use across multiple threads
- No shared mutable state

**Numerical Stability**:

- Integer-only arithmetic prevents floating-point errors
- All calculations exact within Python's integer precision
- No rounding or approximation errors

**Validation Built-in**:

- Runtime assertions verify mathematical invariants
- Input validation prevents invalid calculations
- Receipt hashing enables calculation verification

See Also
--------

- :doc:`mathematical_foundation` for theoretical background
- :doc:`quickstart` for usage examples  
- :doc:`examples` for advanced patterns
- :doc:`testing` for validation approaches