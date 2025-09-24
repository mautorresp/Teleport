Causal Minimality Contract
==========================

This document formalizes the mathematical invariants that MUST be preserved in all implementations of the Teleport CLF Calculator.

Mathematical Contract
======================

**Canonical Formula**

.. math::
   C_{\min}^{(1)}(L) = 88 + 8 \cdot \mathrm{leb}(L)

**Locked Constants**

- **H = 56**: Header information bits
- **CAUS = 27**: Causal relationship encoding bits  
- **END = 5**: Termination marker bits
- **Total overhead = 88**: H + CAUS + END (immutable)

**Decision Rule**

.. math::
   \text{EMIT} \Leftrightarrow C_{\min}^{(1)}(L) < 10 \cdot L \text{ (strict inequality)}

**LEB128 Definition**

The function leb(L) returns the number of bytes required to encode L in unsigned LEB128 format using 7-bit data groups:

.. math::
   \mathrm{leb}(L) = \begin{cases}
   1 & \text{if } 0 \leq L \leq 127 \\
   2 & \text{if } 128 \leq L \leq 16383 \\
   3 & \text{if } 16384 \leq L \leq 2097151 \\
   k & \text{if } 128^{k-1} \leq L < 128^k
   \end{cases}

**Edge Case: Empty Files (L=0)**

For zero-length files, the canonical mathematical behavior is:

- leb(0) = 1 (special case for zero-length encoding)
- C_min^(1)(0) = 88 + 8*1 = 96 bits
- RAW(0) = 10*0 = 0 bits  
- EMIT = (96 < 0) = False (causal overhead exceeds literal cost)

This demonstrates that causal overhead dominates for tiny inputs, maintaining strict mathematical consistency.

Core Function Specifications
=============================

**The Four Golden Functions**

1. ``leb_len_u(n: int) -> int``
   
   Returns the unsigned LEB128 byte-length of n.
   
   Examples:
   
   - leb_len_u(0) = 1
   - leb_len_u(127) = 1  
   - leb_len_u(128) = 2

2. ``clf_single_seed_cost(L: int) -> int``
   
   Returns C_min^(1)(L) = 88 + 8*leb(L).
   
   Examples:
   
   - clf_single_seed_cost(127) = 96
   - clf_single_seed_cost(128) = 104
   - clf_single_seed_cost(16384) = 112

3. ``should_emit(L: int) -> bool``
   
   Returns True iff C_min^(1)(L) < 10*L (strict).
   
   Examples:
   
   - should_emit(16) = True (96 < 160)
   - should_emit(456) = True (104 < 4560)

4. ``receipt(L: int, build_id: str) -> dict``
   
   Returns complete calculation with verification hash.

Invariant Constraints
======================

**Type Invariants**

- All calculations use integer arithmetic only
- No floating-point operations permitted anywhere
- All results must be exact integers

**Computational Invariants**

- O(log L) complexity for all core functions
- No file content analysis (length-only)
- Deterministic results for identical inputs
- No compression logic or content scanning

**Mathematical Invariants**

- Constants H=56, CAUS=27, END=5 are locked forever
- Formula C_min^(1)(L) = 88 + 8*leb(L) is immutable
- Decision rule EMIT iff C < 10*L (strict) is immutable
- LEB128 definition using 7-bit groups is canonical

Anti-Drift Checklist
--------------------

**PROHIBITED Operations (Will Break Mathematical Contract)**

No compression logic:
   - No tiling algorithms
   - No dynamic programming or backtracking
   - No content scanning beyond length
   - Calculator must remain O(log L) arithmetic only

No floating point anywhere:
   - All calculations must use integer-only arithmetic
   - No rounding, approximation, or tolerance checks
   - Results must be mathematically exact

No per-tile END costs:
   - Single END=5 cost only
   - No END cost multiplication or distribution

No length term modifications:
   - Length term is strictly 8 * leb(L)
   - No leb(8*L) or other transformations
   - No alignment padding or bit manipulation

No header/END cost drift:
   - H=56, CAUS=27, END=5 are mathematically locked
   - No parameter tuning or optimization of constants
   - No A/B testing of different constant values

No content dependence:
   - Decisions depend only on L (file length in bytes)
   - No analysis of actual file contents or headers
   - No format-specific optimizations

No tolerance margins:
   - All equations yield exact integers
   - No "within epsilon" or approximate comparisons
   - Strict inequality C < 10*L must be preserved

No role-based modifications:
   - No A/B roles, sender/receiver distinctions
   - No feasibility guards or bijection proofs in core calculator
   - Receipt generation is deterministic from (L, build_id)

Verification Protocol
======================

**Mathematical Validation**

All implementations must pass these exact test cases:

.. code-block:: text

   L=127    -> leb=1 -> C=96   -> EMIT=True  (96 < 1270)
   L=128    -> leb=2 -> C=104  -> EMIT=True  (104 < 1280)
   L=16383  -> leb=2 -> C=104  -> EMIT=True  (104 < 163830)
   L=16384  -> leb=3 -> C=112  -> EMIT=True  (112 < 163840)

**Known Regression Cases**

These historical cases must continue to pass:

.. code-block:: text

   pic1.jpg:    L=63,379    -> C=112 -> EMIT=True
   pic2.jpg:    L=11,751    -> C=104 -> EMIT=True
   video3.mp4:  L=9,840,497 -> C=120 -> EMIT=True

**CI Quality Gates**

1. **Link Check**: ``sphinx-build -b linkcheck docs/ docs/_build/linkcheck -W``
2. **Warnings as Errors**: ``sphinx-build -b html docs/ docs/_build/html -W``
3. **Doctest Validation**: ``sphinx-build -b doctest docs/ docs/_build/doctest -W``

Implementation Notes
====================

**For Developers**

- Cross-reference this contract in all function docstrings
- Include anti-drift warnings in code comments
- Use runtime assertions to prevent constant modification
- Generate receipts with SHA-256 hashing for verification

**For Code Reviews**

- Verify no prohibited operations from the anti-drift checklist
- Confirm all mathematical invariants are preserved
- Test against known regression cases
- Validate O(log L) complexity is maintained

**For Documentation**

- Link to this contract from all API documentation
- Include examples demonstrating correct mathematical behavior
- Document any new functions in terms of their relationship to core CLF logic

This contract ensures the mathematical integrity of the CLF Calculator across all implementations and modifications.